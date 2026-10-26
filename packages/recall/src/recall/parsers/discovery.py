"""Bounded canonical discovery and configured-root provenance shared by all adapters."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR, S_ISLNK, S_ISREG
from typing import Protocol

from recall.core.transcripts import (
    plain_transcript_path,
    transcript_file,
    transcript_identity,
    transcript_name_matches,
)

logger = logging.getLogger(__name__)

ROOT_LIMIT = 1_024
DIRECTORY_LIMIT = 100_000
FILE_LINK_LIMIT = 100_000
DEPTH_LIMIT = 256
DIRECTORY_ENTRY_LIMIT = 65_536
SCAN_ENTRY_LIMIT = 2_000_000
DIAGNOSTIC_LIMIT = 32
DETAIL_LIMIT = 256
DELEGATED_WATCH_LIMIT = 1_024
_SCOPE_CACHE_LIMIT = 5
_SCOPE_MAX_AGE_SECONDS = 30.0
_CANCELLED: ContextVar[Callable[[], bool]] = ContextVar(
    "discovery_cancelled", default=lambda: False
)


@contextmanager
def discovery_cancellation(cancelled: Callable[[], bool]) -> Iterator[None]:
    token = _CANCELLED.set(cancelled)
    try:
        yield
    finally:
        _CANCELLED.reset(token)


_SCOPE_LOCK = threading.Lock()
_SCOPE_CACHE: OrderedDict[tuple[str, str, tuple[Path, ...]], tuple[float, DiscoveryScope]] = (
    OrderedDict()
)


class DiscoveryFacts(Protocol):
    @property
    def source(self) -> object: ...

    @property
    def file_pattern(self) -> str: ...


class DiscoveryParser(DiscoveryFacts, Protocol):
    @property
    def roots(self) -> tuple[Path, ...] | None: ...

    def default_roots(self) -> list[Path]: ...


@dataclass(frozen=True)
class DiscoveredFile:
    path: Path
    root: Path


@dataclass(frozen=True)
class Directory:
    path: Path
    root: Path
    identity: tuple[int, int, int, int]
    parent: Path | None
    entrance: Path


@dataclass(frozen=True)
class Link:
    path: Path
    target: Path
    root: Path
    identity: tuple[int, int, int, int]


def _identity(status: os.stat_result) -> tuple[int, int, int, int]:
    return status.st_dev, status.st_ino, status.st_ctime_ns, status.st_mtime_ns


class DiscoveryScope:
    """One source scan owns directory membership, limits, and traversal failures.

    Ordinary files are streamed, never retained. Directory ownership and bounded
    file-link exceptions establish canonical membership before captures are emitted.
    A failed scope can admit observed content but cannot prove absence.
    """

    def __init__(
        self,
        source: str,
        pattern: str,
        roots: tuple[Path, ...],
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        assert "/" not in pattern, "discovery patterns match encountered filenames"
        self.source = source
        self.pattern = pattern
        self.roots = roots
        self.directories: dict[Path, Directory] = {}
        self.discovered_counts: dict[str, int] = {}
        self.file_links: dict[Path, Path] = {}
        self.links: list[Link] = []
        self.root_spellings: list[tuple[Path, Path]] = []
        # Coverage needs a durable key and its own diagnostic even when no
        # canonical traversal root exists. Bounded by ROOT_LIMIT.
        self.unresolved_roots: dict[Path, str] = {}
        self.failures: list[str] = []
        self.failure_count = 0
        self.examined_entries = 0
        self._cancelled = cancelled or _CANCELLED.get()
        self._exhausted = False
        self.source_wide = False

    @property
    def complete(self) -> bool:
        return self.failure_count == 0

    def fail(self, root: Path, detail: str | OSError | RuntimeError) -> str:
        self.failure_count += 1
        reason = str(detail)
        if isinstance(detail, OSError) and detail.strerror is not None:
            reason = f"{type(detail).__name__}: {detail.strerror}; path={detail.filename}"
        prefix = f"source={self.source} "
        reason_budget = DETAIL_LIMIT - len(prefix) - len(" root=") - 24
        diagnostic = f"{prefix}{reason[:reason_budget]} root={root}"[:DETAIL_LIMIT]
        if len(self.failures) < DIAGNOSTIC_LIMIT:
            self.failures.append(diagnostic)
            logger.warning("incomplete discovery: %s", diagnostic)
        return diagnostic

    def _limit(self, root: Path, name: str) -> None:
        self.fail(root, f"limit={name}")
        self._exhausted = True

    def _stopped(self, root: Path) -> bool:
        if self._cancelled() and not self._exhausted:
            self._limit(root, "cancelled")
        return self._exhausted

    @classmethod
    def build(
        cls,
        parser: DiscoveryFacts,
        *,
        roots: tuple[Path, ...] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> DiscoveryScope:
        if roots is not None:
            raw_roots = roots
        else:
            from typing import cast

            configured_parser = cast(DiscoveryParser, parser)
            configured = configured_parser.roots
            raw_roots = configured_parser.default_roots() if configured is None else configured
        source = str(getattr(parser.source, "value", parser.source))
        # Bound before canonicalization, including duplicate spellings.
        scope = cls(source, parser.file_pattern, (), cancelled=cancelled)
        scope.source_wide = roots is None
        if len(raw_roots) > ROOT_LIMIT:
            scope._limit(raw_roots[0], "configured_roots")
            raw_roots = raw_roots[:ROOT_LIMIT]
        resolved: set[Path] = set()
        for root in raw_roots:
            try:
                canonical = root.expanduser().resolve()
                resolved.add(canonical)
                scope.root_spellings.append((root, canonical))
            except (OSError, RuntimeError) as error:
                scope.unresolved_roots[root.absolute()] = scope.fail(root, error)
        scope.roots = tuple(sorted(resolved))
        for root in scope.roots:
            spelling = next(raw for raw, canonical in scope.root_spellings if canonical == root)
            scope._visit(root, root, 0, top=True, entrance=spelling)
        if scope.source_wide:
            key = (source, parser.file_pattern, tuple(raw_roots))
            with _SCOPE_LOCK:
                _SCOPE_CACHE[key] = (time.monotonic(), scope)
                _SCOPE_CACHE.move_to_end(key)
                while len(_SCOPE_CACHE) > _SCOPE_CACHE_LIMIT:
                    _SCOPE_CACHE.popitem(last=False)
        return scope

    def _entries(self, path: Path, root: Path) -> list[os.DirEntry[str]] | None:
        entries: list[os.DirEntry[str]] = []
        try:
            with os.scandir(path) as iterator:
                for entry in iterator:
                    if self._stopped(root):
                        break
                    self.examined_entries += 1
                    if self.examined_entries > SCAN_ENTRY_LIMIT:
                        self._limit(root, "examined_entries")
                        break
                    if len(entries) == DIRECTORY_ENTRY_LIMIT:
                        self._limit(root, "directory_entries")
                        break
                    entries.append(entry)
        except OSError as error:
            self.fail(root, error)
            return None
        return sorted(entries, key=lambda entry: entry.name)

    def _visit(
        self,
        path: Path,
        root: Path,
        depth: int,
        *,
        top: bool = False,
        parent: Path | None = None,
        entrance: Path | None = None,
    ) -> None:
        if self._stopped(root) or path in self.directories:
            return
        if depth > DEPTH_LIMIT:
            self._limit(root, "traversal_depth")
            return
        if len(self.directories) == DIRECTORY_LIMIT:
            self._limit(root, "canonical_directories")
            return
        try:
            status = path.stat()
        except FileNotFoundError as error:
            if not top:
                self.fail(root, error)
            return
        except OSError as error:
            self.fail(root, error)
            return
        if not S_ISDIR(status.st_mode):
            self.fail(root, f"not a directory: {path}")
            return
        entries = self._entries(path, root)
        if entries is None:
            return
        self.directories[path] = Directory(path, root, _identity(status), parent, entrance or path)
        for entry in entries:
            if self._stopped(root):
                return
            child = path / entry.name
            try:
                status = entry.stat(follow_symlinks=False)
                if S_ISLNK(status.st_mode):
                    selected = (
                        transcript_file(child)
                        if transcript_name_matches(entry.name, self.pattern)
                        else child
                    )
                    target = Path(os.path.realpath(selected, strict=True))
                    target_status = target.stat()
                    if S_ISDIR(target_status.st_mode):
                        if target not in self.directories:
                            self.links.append(Link(child, target, root, _identity(status)))
                        self._visit(target, root, depth + 1, parent=path, entrance=child)
                    elif transcript_name_matches(entry.name, self.pattern):
                        canonical = transcript_identity(child)
                        if canonical in self.file_links:
                            continue
                        if len(self.file_links) == FILE_LINK_LIMIT:
                            self._limit(root, "file_link_targets")
                            return
                        if S_ISREG(target_status.st_mode):
                            self.links.append(Link(child, target, root, _identity(status)))
                            self.file_links.setdefault(canonical, root)
                elif S_ISDIR(status.st_mode):
                    self._visit(child, root, depth + 1, parent=path)
            except (OSError, RuntimeError) as error:
                self.fail(root, error)

    def owner(self, path: Path) -> Path | None:
        """Owner of a canonical file, without treating a file link as a subtree."""
        directory = self.directories.get(path.parent)
        direct = (
            directory.root
            if directory is not None and transcript_name_matches(path.name, self.pattern)
            else None
        )
        linked = self.file_links.get(path)
        owners = [root for root in (direct, linked) if root is not None]
        return min(owners) if owners else None

    def locate(self, path: Path) -> DiscoveredFile:
        canonical = transcript_identity(path)
        owner = self.owner(canonical)
        if owner is None:
            raise FileNotFoundError(f"outside configured discovery scope: {path}")
        self._validate_route(canonical, owner)
        return DiscoveredFile(canonical, owner)

    def _validate_route(self, path: Path, owner: Path) -> None:
        directory = self.directories.get(path.parent)
        if (
            directory is None
            or directory.root != owner
            or not transcript_name_matches(path.name, self.pattern)
        ):
            link = next(
                (
                    link
                    for link in self.links
                    if plain_transcript_path(link.target) == path and link.root == owner
                ),
                None,
            )
            if link is None or transcript_identity(link.path) != path:
                raise FileNotFoundError(f"discovery route changed: {path}")
            directory = self.directories.get(link.path.parent)
        for _ in range(DEPTH_LIMIT + 1):
            if directory is None:
                return
            status = directory.path.stat()
            if (status.st_dev, status.st_ino) != directory.identity[
                :2
            ] or directory.entrance.resolve(strict=True) != directory.path:
                raise FileNotFoundError(f"discovery route changed: {directory.path}")
            directory = (
                self.directories.get(directory.parent) if directory.parent is not None else None
            )
        raise OSError("discovery route exceeds traversal_depth")

    def iter_files(self, root: Path | None = None) -> Iterator[DiscoveredFile]:
        for directory in self.directories.values():
            if root is not None and directory.root != root:
                continue
            entries = self._entries(directory.path, directory.root)
            if entries is None:
                continue
            for entry in entries:
                if self._stopped(directory.root):
                    return
                if not transcript_name_matches(entry.name, self.pattern):
                    continue
                try:
                    status = entry.stat(follow_symlinks=False)
                except OSError as error:
                    self.fail(directory.root, error)
                    continue
                if S_ISREG(status.st_mode):
                    physical = Path(entry.path)
                    try:
                        selected = transcript_file(physical)
                        if selected != physical:
                            selected.stat()
                            continue
                    except OSError as error:
                        reason = error.strerror or str(error)
                        self.fail(
                            directory.root,
                            f"stat failed for: {type(error).__name__}: {reason}; path={physical}",
                        )
                        continue
                    path = plain_transcript_path(physical)
                    if self.owner(path) == directory.root:
                        yield DiscoveredFile(path, directory.root)
        for path, linked_root in sorted(self.file_links.items()):
            owner = self.owner(path)
            assert owner is not None
            # A matching ordinary file is emitted by its parent directory.
            directory = self.directories.get(path.parent)
            if (
                directory is not None
                and directory.root == owner
                and transcript_name_matches(path.name, self.pattern)
            ):
                continue
            if (root is None or owner == root) and not self._stopped(linked_root):
                yield DiscoveredFile(path, owner)

    def revalidate(self) -> bool:
        for spelling, canonical in self.root_spellings:
            try:
                if spelling.resolve() != canonical:
                    self.fail(canonical, f"configured root changed during scan: {spelling}")
            except (OSError, RuntimeError) as error:
                self.fail(canonical, error)
        for directory in self.directories.values():
            if self._stopped(directory.root):
                return False
            try:
                if _identity(directory.path.stat()) != directory.identity:
                    self.fail(directory.root, f"directory changed during scan: {directory.path}")
            except OSError as error:
                self.fail(directory.root, error)
        for link in self.links:
            if self._stopped(link.root):
                return False
            try:
                if (
                    _identity(link.path.lstat()) != link.identity
                    or Path(os.path.realpath(transcript_file(link.path), strict=True))
                    != link.target
                ):
                    self.fail(link.root, f"link changed during scan: {link.path}")
            except (OSError, RuntimeError) as error:
                self.fail(link.root, error)
        return self.complete

    def structural_roots(self) -> tuple[Path, ...]:
        """Minimal canonical directory coverage, bounded independently by the watcher."""
        candidates = set(self.roots) | {
            link.target for link in self.links if link.target in self.directories
        }
        return minimal_roots(candidates)


def minimal_roots(paths: set[Path]) -> tuple[Path, ...]:
    return tuple(
        sorted(path for path in paths if not any(parent in paths for parent in path.parents))
    )


def current_scope(parser: DiscoveryParser, *, path: Path | None = None) -> DiscoveryScope:
    """Reuse bounded topology between inventories; revalidate the selected route on capture.

    A new directory or changed route triggers a fresh scope. Notifications remain
    hints; every periodic inventory independently rebuilds complete membership.
    """
    roots = parser.default_roots() if parser.roots is None else parser.roots
    source = str(getattr(parser.source, "value", parser.source))
    key = (source, parser.file_pattern, tuple(roots))
    with _SCOPE_LOCK:
        cached = _SCOPE_CACHE.get(key)
    if cached is not None and time.monotonic() - cached[0] < _SCOPE_MAX_AGE_SECONDS:
        scope = cached[1]
        if path is None:
            return scope
        try:
            scope.locate(path)
        except (OSError, RuntimeError):
            pass
        else:
            return scope
    return DiscoveryScope.build(parser)
