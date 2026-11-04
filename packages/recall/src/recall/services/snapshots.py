from __future__ import annotations

import errno
import logging
import os
import stat
import time
from collections import deque
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from recall.core.config import AppConfig

logger = logging.getLogger(__name__)

DEFAULT_GC_DAYS = 7
SECONDS_PER_DAY = 86_400
RECALL_SIDECAR_INTRODUCED_AT = datetime(2026, 5, 28, tzinfo=UTC)
# Recovery artifacts, including incomplete ones, require explicit retirement.
_RECOVERY_BACKUP_PREFIXES = (
    "index-migration-",
    "storage-migration-",
    "recreate-",
    "keyword-rebuild-",
)
_DEPTH_MAX = 128
_VISITS_MAX = 100_000
_DESCRIPTORS_MAX = 132
_LINKS_MAX = 40
_PATH_BYTES_MAX = 4096
_COMPONENTS_MAX = 256


def _link_components(text: str) -> tuple[str, ...]:
    """Keep empty and dot components: both require a preceding directory."""
    return tuple(text.split("/"))


@dataclass(frozen=True)
class SnapshotGcResult:
    removed_paths: tuple[str, ...] = ()
    kept_paths: tuple[str, ...] = ()
    failed_paths: tuple[str, ...] = ()
    partial_paths: tuple[str, ...] = ()
    total_bytes_freed: int = 0
    dry_run: bool = False
    snapshots_dir_missing: bool = False


@dataclass(frozen=True)
class SnapshotEntry:
    paths: tuple[str, ...]
    total_bytes: int
    age_seconds: float
    retained: bool
    partial: bool


@dataclass(frozen=True)
class SnapshotInventory:
    snapshots_dir: str
    snapshots_dir_missing: bool = False
    entries: tuple[SnapshotEntry, ...] = ()
    entry_count: int = 0
    total_bytes: int = 0
    unreadable_paths: tuple[str, ...] = ()


class _ScanFailure(OSError):
    """An unsupported operation or exhausted traversal envelope."""


class _VisitLimit(_ScanFailure):
    def __init__(self, enumerated: list[str]) -> None:
        super().__init__("snapshot visit budget exhausted")
        self.enumerated = tuple(enumerated)


@dataclass(frozen=True)
class _Removal:
    components: tuple[str, ...]
    directory: bool


@dataclass(frozen=True)
class _Snapshot:
    names: tuple[str, ...]
    mtime: float
    size: int
    removal: tuple[_Removal, ...]
    retained: bool = False
    partial: bool = False


@dataclass(frozen=True)
class _Scan:
    entries: tuple[_Snapshot, ...] = ()
    failed: tuple[str, ...] = ()
    exhausted: bool = False


class _SnapshotTree:
    """One operation's authority over a pinned, no-follow snapshot tree.

    Display paths never authorize effects. All descendant effects use a held
    directory and a single enumerated component. Only root binding resolves
    the operator-selected data root's trusted aliases.
    """

    def __init__(self, selected: Path) -> None:
        self.display = selected / "snapshots"
        self.selected = Path.cwd() / selected
        self.prefixes: tuple[tuple[str, ...], ...] = ()
        self.stack = ExitStack()
        self.root: int | None = None
        self.snapshots: int | None = None
        self.missing = False
        self.failed = False
        self.descriptors = 0
        self.visits = 0

    def __enter__(self) -> _SnapshotTree:
        try:
            self._bind()
        except OSError as err:
            self.failed = True
            logger.warning("snapshots could not open %s: %s", self.display, err)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *_args: object) -> None:
        self.stack.close()
        assert self.descriptors == 0

    def _bind(self) -> None:
        supported = (
            all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW"))
            and all(
                fn in os.supports_dir_fd
                for fn in (os.open, os.stat, os.readlink, os.unlink, os.rmdir)
            )
            and os.stat in os.supports_follow_symlinks
            and os.scandir in os.supports_fd
        )
        if not supported:
            raise _ScanFailure("snapshot scanning requires no-follow descriptor operations")
        # Path preserves link/.. in the configured spelling. Never normalize it
        # lexically: the kernel resolves .. against the physical link target.
        try:
            canonical = self.selected.resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError):
            self.missing = True
            return
        except RuntimeError as err:
            raise _ScanFailure("selected snapshot data root cannot be resolved") from err
        self.prefixes = tuple(
            tuple(
                part for part in _link_components(str(root / "snapshots")) if part not in {"", "."}
            )
            for root in (self.selected, canonical)
        )
        try:
            self.root = self.stack.enter_context(self._open_directory(canonical))
        except FileNotFoundError:
            self.missing = True
            return
        except NotADirectoryError:
            self.missing = True
            logger.warning("snapshots refusing non-directory data root %s", self.selected)
            return
        try:
            info = os.stat("snapshots", dir_fd=self.root, follow_symlinks=False)
        except FileNotFoundError:
            self.missing = True
            return
        if not stat.S_ISDIR(info.st_mode):
            logger.warning("snapshots refusing linked or non-directory root %s", self.display)
            return
        self.snapshots = self.stack.enter_context(self._open_directory("snapshots", self.root))

    @contextmanager
    def _open_directory(self, name: str | Path, parent: int | None = None) -> Iterator[int]:
        if self.descriptors >= _DESCRIPTORS_MAX:
            raise _ScanFailure("snapshot descriptor budget exhausted")
        descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        self.descriptors += 1
        try:
            yield descriptor
        finally:
            os.close(descriptor)
            self.descriptors -= 1

    def path(self, name: str) -> str:
        return str(self.display / name)

    def _names(self, directory: int) -> list[str]:
        # scandir(fd) temporarily duplicates the directory descriptor.
        if self.descriptors >= _DESCRIPTORS_MAX:
            raise _ScanFailure("snapshot enumeration descriptor budget exhausted")
        names: list[str] = []
        with os.scandir(directory) as entries:
            for entry in entries:
                if self.visits >= _VISITS_MAX:
                    raise _VisitLimit(names)
                self.visits += 1
                names.append(entry.name)
        return sorted(names)

    @contextmanager
    def _directory(self, components: tuple[str, ...]) -> Iterator[int]:
        assert self.snapshots is not None
        if len(components) > _DEPTH_MAX:
            raise _ScanFailure("snapshot depth budget exhausted")
        with ExitStack() as stack:
            parent = self.snapshots
            for name in components:
                parent = stack.enter_context(self._open_directory(name, parent))
            yield parent

    def _absolute_suffix(self, components: tuple[str, ...]) -> tuple[str, ...] | None:
        # Prefix admission is lexical; presence uses the untouched suffix.
        # In particular, a trailing slash or dot must survive this mapping.
        for prefix in self.prefixes:
            matched = 0
            for index, part in enumerate(components):
                if part in {"", "."}:
                    continue
                if part != prefix[matched]:
                    break
                matched += 1
                if matched == len(prefix):
                    return components[index + 1 :]
        return None

    def local_target(self, components: tuple[str, ...]) -> Literal["present", "missing", "refused"]:
        """Interpret link text locally, never asking an escaped target about itself."""
        assert self.snapshots is not None
        pending = deque(components)
        physical: list[str] = []
        links = 0
        missing = False
        # Keep only the physical parent chain. Absolute expansions release it
        # before acquiring another chain, sharing the operation's fd budget.
        with ExitStack() as stack:
            directories = [self.snapshots]
            scopes: list[ExitStack] = []
            while pending:
                part = pending.popleft()
                if part in {"", "."}:
                    continue
                if part == "..":
                    if not physical:
                        return "refused"
                    physical.pop()
                    if not missing:
                        scopes.pop().close()
                        directories.pop()
                    continue
                physical.append(part)
                if missing:
                    continue
                parent = directories[-1]
                try:
                    info = os.stat(part, dir_fd=parent, follow_symlinks=False)
                except (FileNotFoundError, NotADirectoryError):
                    missing = True
                    continue
                if stat.S_ISLNK(info.st_mode):
                    physical.pop()
                    links += 1
                    if links > _LINKS_MAX:
                        raise _ScanFailure("snapshot link expansion budget exhausted or loop")
                    target = os.readlink(part, dir_fd=parent)
                    parts = _link_components(target)
                    expanded = (
                        (*parts, *pending)
                        if target.startswith("/")
                        else (*physical, *parts, *pending)
                    )
                    if (
                        len(os.fsencode("/".join(expanded))) > _PATH_BYTES_MAX
                        or len(expanded) > _COMPONENTS_MAX
                    ):
                        raise _ScanFailure("snapshot expanded pathname budget exhausted")
                    if target.startswith("/"):
                        suffix = self._absolute_suffix(parts)
                        if suffix is None:
                            return "refused"
                        for scope in reversed(scopes):
                            scope.close()
                        scopes.clear()
                        directories = [self.snapshots]
                        physical.clear()
                        parts = suffix
                    pending.extendleft(reversed(parts))
                elif pending:
                    if not stat.S_ISDIR(info.st_mode):
                        missing = True
                        continue
                    if len(physical) > _DEPTH_MAX:
                        raise _ScanFailure("snapshot link depth budget exhausted")
                    scope = ExitStack()
                    stack.enter_context(scope)
                    try:
                        child = scope.enter_context(self._open_directory(part, parent))
                    except OSError as err:
                        # Darwin returns ENOTDIR for O_NOFOLLOW on a link.
                        # Classify locally; never retry with following flags.
                        if err.errno not in {errno.ELOOP, errno.ENOTDIR, errno.ENOENT}:
                            raise
                        current = self._stat_or_missing(parent, part)
                        if current is None:
                            missing = True
                        elif stat.S_ISLNK(current.st_mode):
                            physical.pop()
                            pending.appendleft(part)
                        elif not stat.S_ISDIR(current.st_mode):
                            missing = True
                        else:
                            raise
                        scope.close()
                        continue
                    scopes.append(scope)
                    directories.append(child)
            return "missing" if missing else "present"

    @staticmethod
    def _stat_or_missing(parent: int, name: str) -> os.stat_result | None:
        try:
            return os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return None

    def preflight(self, name: str, info: os.stat_result) -> tuple[int, tuple[_Removal, ...]]:
        """Bounded iterative post-order, shared by sizing and removal."""
        assert self.snapshots is not None
        if not stat.S_ISDIR(info.st_mode):
            return info.st_size, (_Removal((name,), False),)
        total = 0
        removal: list[_Removal] = []
        with ExitStack() as stack:
            scope = ExitStack()
            stack.enter_context(scope)
            descriptor = scope.enter_context(self._open_directory(name, self.snapshots))
            frames: list[tuple[tuple[str, ...], int, Iterator[str], ExitStack]] = [
                ((name,), descriptor, iter(self._names(descriptor)), scope)
            ]
            while frames:
                components, parent, names, scope = frames[-1]
                child = next(names, None)
                if child is None:
                    removal.append(_Removal(components, True))
                    scope.close()
                    frames.pop()
                    continue
                child_info = os.stat(child, dir_fd=parent, follow_symlinks=False)
                relative = (*components, child)
                if stat.S_ISDIR(child_info.st_mode):
                    if len(relative) > _DEPTH_MAX:
                        raise _ScanFailure("snapshot depth budget exhausted")
                    child_scope = ExitStack()
                    stack.enter_context(child_scope)
                    descriptor = child_scope.enter_context(self._open_directory(child, parent))
                    frames.append(
                        (relative, descriptor, iter(self._names(descriptor)), child_scope)
                    )
                else:
                    # Symlinks contribute their own length; their targets are
                    # neither sized nor included in the removal plan.
                    total += child_info.st_size
                    removal.append(_Removal(relative, False))
        return total, tuple(removal)

    def scan(self) -> _Scan:
        assert self.snapshots is not None
        try:
            names = self._names(self.snapshots)
        except OSError as err:
            logger.warning("snapshots could not enumerate %s: %s", self.display, err)
            enumerated = (
                tuple(self.path(name) for name in sorted(err.enumerated))
                if isinstance(err, _VisitLimit)
                else ()
            )
            return _Scan(
                failed=(*enumerated, str(self.display)), exhausted=isinstance(err, _VisitLimit)
            )
        entries: list[_Snapshot] = []
        failed: list[str] = []
        siblings: dict[str, dict[str, _Snapshot]] = {}
        exhausted = False
        for index, name in enumerate(names):
            try:
                entry = self._inspect(name)
            except _VisitLimit as err:
                logger.warning("snapshots stopped traversal at %s: %s", self.path(name), err)
                failed.extend(self.path(item) for item in names[index:])
                exhausted = True
                break
            except OSError as err:
                logger.warning("snapshots could not inspect %s: %s", self.path(name), err)
                failed.append(self.path(name))
                continue
            if entry is None:
                continue
            kind = _sibling_snapshot_kind(name)
            if kind is not None and not entry.retained and not entry.removal[-1].directory:
                basename, half = kind
                siblings.setdefault(basename, {})[half] = entry
            else:
                entries.append(entry)
        for basename in sorted(siblings):
            group = siblings[basename]
            halves = [group[kind] for kind in ("duckdb", "sidecar") if kind in group]
            mtime = max(item.mtime for item in halves)
            partial = len(halves) != 2 and not (
                "duckdb" in group and _is_pre_sidecar_snapshot(mtime)
            )
            entries.append(
                _Snapshot(
                    names=tuple(name for item in halves for name in item.names),
                    mtime=mtime,
                    size=sum(item.size for item in halves),
                    removal=tuple(step for item in halves for step in item.removal),
                    partial=partial,
                )
            )
        return _Scan(tuple(entries), tuple(failed), exhausted)

    def _inspect(self, name: str) -> _Snapshot | None:
        assert self.snapshots is not None
        info = os.stat(name, dir_fd=self.snapshots, follow_symlinks=False)
        directory = stat.S_ISDIR(info.st_mode)
        retained = False
        partial = False
        if stat.S_ISLNK(info.st_mode):
            retained = self.local_target((name,)) == "refused"
            if retained:
                logger.warning(
                    "snapshots retaining %s (link leaves snapshots dir)", self.path(name)
                )
        elif directory:
            retained = name.startswith(_RECOVERY_BACKUP_PREFIXES)
            if not retained:
                duckdb = self.local_target((name, "recall.duckdb"))
                sidecar = self.local_target((name, "recall.fts.sqlite"))
                retained = "refused" in (duckdb, sidecar)
                if retained:
                    logger.warning("snapshots retaining %s (unsafe payload link)", self.path(name))
                else:
                    has_duckdb, has_sidecar = duckdb == "present", sidecar == "present"
                    partial = (has_duckdb != has_sidecar) and not (
                        has_duckdb and _is_pre_sidecar_snapshot(info.st_mtime)
                    )
        if not directory and not retained and name.endswith(".duckdb.wal"):
            return None
        size, removal = self.preflight(name, info)
        return _Snapshot((name,), info.st_mtime, size, removal, retained, partial)

    def remove(self, entry: _Snapshot) -> None:
        for step in entry.removal:
            with self._directory(step.components[:-1]) as parent:
                name = step.components[-1]
                if step.directory:
                    os.rmdir(name, dir_fd=parent)
                else:
                    os.unlink(name, dir_fd=parent)
        assert self.snapshots is not None
        for name in entry.names:
            if self._stat_or_missing(self.snapshots, name) is not None:
                raise _ScanFailure(f"snapshot still present after removal: {self.path(name)}")


def list_snapshots(config: AppConfig) -> SnapshotInventory:
    """Report confined snapshot sizes, age, partial pairs and retained artifacts."""
    with _SnapshotTree(config.data_dir) as tree:
        if tree.snapshots is None:
            return SnapshotInventory(
                snapshots_dir=str(tree.display),
                snapshots_dir_missing=tree.missing,
                unreadable_paths=(str(tree.display),) if tree.failed else (),
            )
        scan = tree.scan()
        now = time.time()
        entries = [
            SnapshotEntry(
                paths=tuple(tree.path(name) for name in entry.names),
                total_bytes=entry.size,
                age_seconds=max(0.0, now - entry.mtime),
                retained=entry.retained,
                partial=entry.partial,
            )
            for entry in scan.entries
        ]
        entries.sort(key=lambda entry: (-entry.age_seconds, entry.paths))
        return SnapshotInventory(
            snapshots_dir=str(tree.display),
            entries=tuple(entries),
            entry_count=len(entries),
            total_bytes=sum(entry.total_bytes for entry in entries),
            unreadable_paths=scan.failed,
        )


def gc_snapshots(
    config: AppConfig,
    *,
    days: int = DEFAULT_GC_DAYS,
    dry_run: bool = False,
) -> SnapshotGcResult:
    """Preflight the whole operation before pruning stale, confined snapshots."""
    with _SnapshotTree(config.data_dir) as tree:
        if tree.snapshots is None:
            return SnapshotGcResult(
                dry_run=dry_run,
                snapshots_dir_missing=tree.missing,
                failed_paths=(str(tree.display),) if tree.failed else (),
            )
        scan = tree.scan()
        cutoff = time.time() - days * SECONDS_PER_DAY
        removed: list[str] = []
        kept: list[str] = []
        failed = list(scan.failed)
        partial: list[str] = []
        freed = 0
        for entry in sorted(scan.entries, key=lambda item: item.names):
            paths = tuple(tree.path(name) for name in entry.names)
            if entry.retained:
                kept.extend(paths)
            elif entry.partial:
                partial.extend(paths)
                logger.warning("snapshots gc refusing partial snapshot paths=%s", paths)
            elif scan.exhausted or entry.mtime >= cutoff:
                kept.extend(paths)
            else:
                if not dry_run:
                    try:
                        tree.remove(entry)
                    except OSError as err:
                        logger.warning("snapshots gc failed to remove %s: %s", paths, err)
                        failed.extend(paths)
                        continue
                removed.extend(paths)
                freed += entry.size
        return SnapshotGcResult(
            removed_paths=tuple(removed),
            kept_paths=tuple(kept),
            failed_paths=tuple(failed),
            partial_paths=tuple(partial),
            total_bytes_freed=freed,
            dry_run=dry_run,
        )


def _sibling_snapshot_kind(name: str) -> tuple[str, str] | None:
    if name.endswith(".duckdb"):
        return name.removesuffix(".duckdb"), "duckdb"
    if name.endswith(".fts.sqlite"):
        return name.removesuffix(".fts.sqlite"), "sidecar"
    return None


def _is_pre_sidecar_snapshot(snapshot_mtime: float) -> bool:
    return snapshot_mtime < RECALL_SIDECAR_INTRODUCED_AT.timestamp()
