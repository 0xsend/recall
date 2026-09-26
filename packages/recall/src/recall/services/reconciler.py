"""Bounded capture and fair selection for durable source reconciliation."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISLNK
from typing import Protocol

from recall.db.source_files import (
    PresentSource,
    SourceCatalog,
    SourceFile,
    SourceSignature,
    observation_identity,
    source_key,
)

logger = logging.getLogger(__name__)


class InventoryParser(Protocol):
    """The parser facts needed by inventory, intentionally smaller than parsing."""

    @property
    def source(self) -> object: ...

    @property
    def file_pattern(self) -> str: ...

    def sidecar_paths(self, path: Path) -> list[Path]: ...


class InventoryRootParser(Protocol):
    @property
    def source(self) -> object: ...

    @property
    def roots(self) -> tuple[Path, ...] | None: ...

    def default_roots(self) -> list[Path]: ...


READY_LIMIT = 256
PREPARATION_LIMIT = 2
NOTIFICATION_LIMIT = 4096
SCAN_FAILURE_LIMIT = 32
COALESCE_SECONDS = 10.0
# A copy-then-delete move inside one root never loses more than half of it: each
# vanished original left a copy behind.
_LOST_FRACTION_MAX_DENOMINATOR = 2
QUIET_SECONDS = 5.0


@dataclass(frozen=True)
class CapturedSource:
    source: str
    root_path: str
    source_path: str
    signature: SourceSignature


@dataclass(frozen=True)
class InventoryBatch:
    """A bounded, captured unit that can cross into the catalog writer."""

    files: tuple[CapturedSource, ...]


@dataclass(frozen=True)
class InventoryResult:
    source: str
    root_path: str
    started_at: float
    finished_at: float
    discovered_count: int
    failures: tuple[str, ...]
    failure_count: int
    # Discovered paths reached through a symlink. Their resolved identity can
    # repeat another discovered path, so they make the discovered count an
    # unreliable measure of distinct catalog membership.
    aliased_count: int = 0

    @property
    def complete(self) -> bool:
        return self.failure_count == 0


@dataclass(frozen=True)
class InventoryRootScope:
    """Configured root state for coordinator status before any walk is attempted."""

    source: str
    root_path: Path | None
    configuration: str  # default, configured, or disabled
    available: bool


def inventory_root_scopes(parser: InventoryRootParser) -> tuple[InventoryRootScope, ...]:
    """Preserve default, disabled, and configured-missing root distinctions."""
    source = str(getattr(parser.source, "value", parser.source))
    if parser.roots == ():
        return (InventoryRootScope(source, None, "disabled", False),)
    configuration = "default" if parser.roots is None else "configured"
    roots = parser.default_roots() if parser.roots is None else list(parser.roots)
    scopes: list[InventoryRootScope] = []
    for root in roots:
        try:
            available = root.is_dir()
        except OSError:
            # Status remains usable; the subsequent scan records the actual IO
            # failure without erasing prior root membership.
            available = False
        scopes.append(
            InventoryRootScope(source, root.resolve(strict=False), configuration, available)
        )
    return tuple(scopes)


def begin_inventory_scan(
    catalog: SourceCatalog, source: str, root_path: str, started_at: float
) -> int:
    """Persist scan intent before capture; callers may commit this independently."""
    return catalog.start_scan(source, root_path, started_at)


def unobserved(batch: InventoryBatch, present: Mapping[str, PresentSource]) -> InventoryBatch:
    """Keep the captured sources whose observation would change the catalog.

    `present` is `SourceCatalog.present_states` for the batch's root. A source
    it lacks is new, missing, or catalogued under another root, and is kept; the
    writer's own comparison stays authoritative for everything kept.
    """
    return InventoryBatch(tuple(item for item in batch.files if not is_observed(item, present)))


def is_observed(item: CapturedSource, present: Mapping[str, PresentSource]) -> bool:
    """Whether the root's catalog already holds exactly this observation."""
    known = present.get(item.source_path)
    return known is not None and known.identity == observation_identity(item.signature)


def persist_inventory_batch(catalog: SourceCatalog, batch: InventoryBatch) -> tuple[str, ...]:
    """Persist one bounded capture batch; return existing paths whose content changed."""
    if not batch.files:
        return ()
    return catalog.observe_batch(
        [(item.source, item.root_path, item.source_path, item.signature) for item in batch.files]
    )


VanishedHandler = Callable[[str, list[str]], Collection[str]]
"""Receives a source and a page of its catalogued paths a complete walk no longer
found; returns the paths to leave present so the next complete walk offers them
again."""


def finish_inventory_scan(
    catalog: SourceCatalog, result: InventoryResult, *, on_vanished: VanishedHandler | None = None
) -> None:
    """Finalize coverage, marking absence only after a successful complete scan."""
    if result.complete:
        mark_vanished_sources(catalog, result, on_vanished=on_vanished)
    catalog.record_scan(
        result.source,
        result.root_path,
        started_at=result.started_at,
        finished_at=result.finished_at,
        discovered_count=result.discovered_count,
        failures=list(result.failures),
        failure_count=result.failure_count,
        complete=result.complete,
    )


def mark_vanished_sources(
    catalog: SourceCatalog, result: InventoryResult, *, on_vanished: VanishedHandler | None = None
) -> None:
    """Mark the root's catalogued sources that a complete walk no longer found.

    Every path a complete walk discovers is present in the catalog when the walk
    finishes, so equal counts prove nothing vanished; that is the common case,
    and it costs one read. Otherwise the root's present sources are checked page
    by page, so no walk retains a whole root.

    `on_vanished` sees each page before it is marked and keeps the paths it
    returns present.  A walk that found nothing, or lost more than half of what
    the root held, looks like an unmounted or emptied volume rather than files
    that moved, so its pages are marked without it.
    """
    assert result.complete
    present_count = catalog.present_count(result.source, result.root_path)
    if result.aliased_count == 0 and present_count == result.discovered_count:
        return
    if on_vanished is not None and _looks_unmounted(result, present_count):
        logger.warning(
            "walk found %d of %d present sources; not treating them as moved source=%s root=%s",
            result.discovered_count,
            present_count,
            result.source,
            result.root_path,
        )
        on_vanished = None
    after_key: str | None = None
    while page := catalog.present_page(result.source, result.root_path, after_key=after_key):
        vanished = {path: key for key, path in page if _vanished(path)}
        if vanished and on_vanished is not None:
            for path in on_vanished(result.source, list(vanished)):
                del vanished[path]
        if vanished:
            catalog.mark_missing(list(vanished.values()))
        after_key = page[-1][0]


def _looks_unmounted(result: InventoryResult, present_count: int) -> bool:
    lost = present_count - result.discovered_count
    return result.discovered_count == 0 or lost * _LOST_FRACTION_MAX_DENOMINATOR > present_count


def _vanished(path: str) -> bool:
    try:
        os.stat(path)
    except FileNotFoundError:
        return True
    except OSError:
        # An unreadable path is not proven absent.
        return False
    return False


def iter_inventory_batches(
    parser: InventoryParser,
    root: Path,
    *,
    clock: Callable[[], float],
    parser_revision: str = "",
    chunk_size: int = 128,
    optional_root: bool = False,
) -> Iterable[InventoryBatch | InventoryResult]:
    """Capture a root without catalog access; retained paths and errors are bounded."""
    if not 1 <= chunk_size <= READY_LIMIT:
        raise ValueError("chunk_size must be between 1 and 256")
    raw_source = parser.source
    source = str(getattr(raw_source, "value", raw_source))
    root_path = str(root.resolve(strict=False))
    started = clock()
    failures: list[str] = []
    failure_count = 0
    discovered_count = 0

    def add_failure(error: OSError | str) -> None:
        nonlocal failure_count
        failure_count += 1
        if len(failures) < SCAN_FAILURE_LIMIT:
            failures.append(str(error))

    try:
        root.stat()
    except OSError as error:
        # Absence of a built-in harness root is a complete empty scope. Explicit
        # roots and access failures retain prior membership and fail closed.
        if not (optional_root and isinstance(error, FileNotFoundError)):
            add_failure(error)
        yield InventoryResult(
            source, root_path, started, clock(), 0, tuple(failures), failure_count
        )
        return

    chunk: list[CapturedSource] = []
    pattern = str(parser.file_pattern)
    assert "/" not in pattern, "inventory patterns match a file name"
    aliased_count = 0

    def on_walk_error(error: OSError) -> None:
        add_failure(error)

    try:
        # Walking the resolved root descends only real directories, so a path
        # resolves elsewhere only when the file itself is a symlink. Resolving
        # every path instead costs a stat per path component.
        for directory, _, names in os.walk(root_path, onerror=on_walk_error):
            for name in sorted(names):
                if not fnmatch.fnmatchcase(name, pattern):
                    continue
                path = os.path.join(directory, name)
                try:
                    status = os.lstat(path)
                    aliased = S_ISLNK(status.st_mode)
                    if aliased:
                        status = os.stat(path)
                    sidecar_mtime, sidecar_signature = capture_sidecars(parser, Path(path))
                except OSError as err:
                    add_failure(f"{path}: {err}")
                    continue
                if aliased:
                    aliased_count += 1
                    path = os.path.realpath(path)
                chunk.append(
                    CapturedSource(
                        source,
                        root_path,
                        path,
                        SourceSignature(
                            status.st_dev,
                            status.st_ino,
                            status.st_ctime_ns,
                            status.st_mtime_ns,
                            status.st_size,
                            sidecar_mtime,
                            parser_revision,
                            sidecar_signature,
                        ),
                    )
                )
                if len(chunk) == chunk_size:
                    discovered_count += len(chunk)
                    yield InventoryBatch(tuple(chunk))
                    chunk.clear()
    except OSError as err:
        add_failure(err)
    if chunk:
        discovered_count += len(chunk)
        yield InventoryBatch(tuple(chunk))
    yield InventoryResult(
        source,
        root_path,
        started,
        clock(),
        discovered_count,
        tuple(failures),
        failure_count,
        aliased_count,
    )


def inventory_root(
    catalog: SourceCatalog,
    parser: InventoryParser,
    root: Path,
    *,
    clock: Callable[[], float],
    parser_revision: str = "",
    chunk_size: int = 128,
) -> int:
    """Stream one root into the catalog; only a complete walk may mark files missing."""
    raw_source = parser.source
    source = str(getattr(raw_source, "value", raw_source))
    root_path = str(root.resolve(strict=False))
    begin_inventory_scan(catalog, source, root_path, clock())
    count = 0
    for event in iter_inventory_batches(
        parser, root, clock=clock, parser_revision=parser_revision, chunk_size=chunk_size
    ):
        if isinstance(event, InventoryBatch):
            persist_inventory_batch(catalog, event)
            count += len(event.files)
            continue
        finish_inventory_scan(catalog, event)
    return count


@dataclass(frozen=True)
class ScheduledSource:
    source: SourceFile
    lane: str


class FairScheduler:
    """Deterministic 4 active : 2 recent : 1 oldest selection."""

    _LANES = ("active", "active", "active", "active", "recent", "recent", "oldest")

    def __init__(self, *, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._sequence = 0

    def select(
        self,
        catalog: SourceCatalog,
        active_paths: set[str],
        *,
        limit: int = READY_LIMIT,
        eligible_roots: tuple[tuple[str, str], ...] | None = None,
        requested_paths: tuple[str, ...] = (),
        active_since_ns: int | None = None,
        excluded_keys: tuple[str, ...] = (),
        can_admit: Callable[[SourceFile], bool] | None = None,
    ) -> list[ScheduledSource]:
        if not 1 <= limit <= READY_LIMIT:
            raise ValueError(f"limit must be between 1 and {READY_LIMIT}")
        active, recent, oldest = catalog.priority_pending(
            active_paths,
            limit=READY_LIMIT,
            eligible_roots=eligible_roots,
            requested_paths=requested_paths,
            active_since_ns=active_since_ns,
            excluded_keys=excluded_keys,
        )
        reserved_oldest = source_key(oldest[0].source, oldest[0].source_path) if oldest else None
        if reserved_oldest in {source_key(item.source, item.source_path) for item in active}:
            reserved_oldest = None
        selected: list[ScheduledSource] = []
        used: set[str] = set()
        for index in range(limit):
            lane = self._LANES[(self._sequence + index) % len(self._LANES)]
            candidates = active if lane == "active" else recent if lane == "recent" else oldest
            candidate = next(
                (
                    item
                    for item in candidates
                    if source_key(item.source, item.source_path) not in used
                    and (
                        lane == "oldest"
                        or source_key(item.source, item.source_path) != reserved_oldest
                    )
                ),
                None,
            )
            if candidate is None:
                fallback = (
                    oldest + active + recent if lane == "oldest" else active + recent + oldest
                )
                candidate = next(
                    (
                        item
                        for item in fallback
                        if source_key(item.source, item.source_path) not in used
                    ),
                    None,
                )
            if candidate is None or (can_admit is not None and not can_admit(candidate)):
                # Capacity refusal delivers no service. Preserve this lane position
                # until a claim releases; skipping it would falsify the 7*N bound.
                break
            used.add(source_key(candidate.source, candidate.source_path))
            selected.append(ScheduledSource(candidate, lane))
        self._sequence += len(selected)
        return selected


class Coalescer:
    """Injected-clock quiet/max debounce without resetting a source's durable age."""

    def __init__(self, *, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._marks: dict[str, tuple[float, float]] = {}

    def mark(self, source_path: str) -> bool:
        if len(self._marks) >= NOTIFICATION_LIMIT and source_path not in self._marks:
            return False
        now = self._clock()
        first, _ = self._marks.get(source_path, (now, now))
        self._marks[source_path] = (first, now)
        return True

    def ready(self, source_path: str) -> bool:
        mark = self._marks.get(source_path)
        if mark is None:
            return True
        first, latest = mark
        now = self._clock()
        return now - first >= COALESCE_SECONDS or now - latest >= QUIET_SECONDS

    def consume(self, source_path: str) -> None:
        self._marks.pop(source_path, None)


def stat_signature(path: Path, *, parser_revision: str = "") -> SourceSignature:
    """Small adapter for notification paths; capture callers handle IO errors."""
    stat = os.stat(path)
    return SourceSignature(
        dev=stat.st_dev,
        inode=stat.st_ino,
        ctime_ns=stat.st_ctime_ns,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
        parser_revision=parser_revision,
    )


def capture_sidecars(parser: InventoryParser, path: Path) -> tuple[int, str]:
    """Fingerprint every declared sidecar, including disappearance and replacement."""
    digest = hashlib.sha256()
    newest = 0
    for sidecar in sorted(parser.sidecar_paths(path)):
        try:
            stat = sidecar.stat()
        except FileNotFoundError:
            identity = (str(sidecar), None)
        else:
            newest = max(newest, stat.st_mtime_ns)
            identity = (
                str(sidecar),
                stat.st_dev,
                stat.st_ino,
                stat.st_ctime_ns,
                stat.st_mtime_ns,
                stat.st_size,
            )
        digest.update(json.dumps(identity).encode())
        digest.update(b"\n")
    return newest, digest.hexdigest()
