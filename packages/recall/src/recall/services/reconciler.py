"""Bounded capture and fair selection for durable source reconciliation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISLNK
from typing import Literal, Protocol

from recall.db.source_files import (
    PresentSource,
    SourceCatalog,
    SourceFile,
    SourceSignature,
    observation_identity,
)
from recall.parsers.discovery import DiscoveryScope

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
    scope: DiscoveryScope | None = None


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
    scope: DiscoveryScope | None = None
    finalize_scope: bool = False

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
        try:
            canonical = root.resolve(strict=False)
        except (OSError, RuntimeError):
            canonical = root.absolute()
            available = False
        scopes.append(InventoryRootScope(source, canonical, configuration, available))
    unique = {scope.root_path: scope for scope in scopes}
    return tuple(unique[path] for path in sorted(unique, key=str))


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
    return InventoryBatch(
        tuple(item for item in batch.files if not is_observed(item, present)), batch.scope
    )


def is_observed(item: CapturedSource, present: Mapping[str, PresentSource]) -> bool:
    """Whether the root's catalog already holds exactly this observation."""
    known = present.get(item.source_path)
    return known is not None and known.identity == observation_identity(item.signature)


def persist_inventory_batch(
    catalog: SourceCatalog,
    batch: InventoryBatch,
    *,
    notification: bool = False,
) -> tuple[str, ...]:
    """Persist one bounded capture batch; return existing paths whose content changed."""
    if not batch.files:
        return ()
    observations = [
        (item.source, item.root_path, item.source_path, item.signature) for item in batch.files
    ]
    if notification:
        observations = catalog.notification_sources(observations)
    if not observations:
        return ()
    return catalog.observe_batch(observations)


VanishedHandler = Callable[[str, list[str]], Collection[str]]
"""Receives a source and a page of its catalogued paths a complete walk no longer
found; returns the paths to leave present so the next complete walk offers them
again."""


def finish_inventory_scan(
    catalog: SourceCatalog, result: InventoryResult, *, on_vanished: VanishedHandler | None = None
) -> None:
    """Finalize coverage, marking absence only after a successful complete scan."""
    if result.complete and (result.scope is None or result.finalize_scope):
        mark_vanished_sources(catalog, result, on_vanished=on_vanished)
    if not result.root_path:
        return
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
    scope = result.scope
    source_wide = scope is not None and scope.source_wide
    root_filter = None if source_wide else result.root_path
    present_count = catalog.present_count(result.source, result.root_path)
    if scope is None and result.aliased_count == 0 and present_count == result.discovered_count:
        return
    after_key: str | None = None
    allowed_moves: dict[str, bool] = {}
    while page := catalog.present_scope_page(result.source, root_filter, after_key=after_key):
        vanished_by_root: dict[str, dict[str, str]] = {}
        detached: list[str] = []
        reassigned: list[tuple[str, str]] = []
        for previous_root in {row[2] for row in page}:
            if scope is not None and previous_root not in scope.discovered_counts:
                continue
            if previous_root not in allowed_moves:
                count = (
                    scope.discovered_counts.get(previous_root, 0)
                    if scope is not None
                    else result.discovered_count
                )
                root_result = InventoryResult(
                    result.source,
                    previous_root,
                    result.started_at,
                    result.finished_at,
                    count,
                    (),
                    0,
                )
                allowed_moves[previous_root] = not _looks_unmounted(
                    root_result, catalog.present_count(result.source, previous_root)
                )
        for key, path, previous_root in page:
            owner = scope.owner(Path(path)) if scope is not None else None
            if _vanished(path, check_resolution=scope is not None and owner is None):
                vanished_by_root.setdefault(previous_root, {})[path] = key
            elif scope is not None and owner is None:
                detached.append(key)
            elif owner is not None and str(owner) != previous_root:
                reassigned.append((key, str(owner)))
        if reassigned:
            catalog.reassign_roots(reassigned)
        if detached:
            catalog.mark_missing(detached)
        for previous_root, vanished in vanished_by_root.items():
            if on_vanished is not None and allowed_moves.get(previous_root, False):
                for path in on_vanished(result.source, list(vanished)):
                    del vanished[path]
            if vanished:
                catalog.mark_missing(list(vanished.values()))
        after_key = page[-1][0]


def _looks_unmounted(result: InventoryResult, present_count: int) -> bool:
    lost = present_count - result.discovered_count
    return result.discovered_count == 0 or lost * _LOST_FRACTION_MAX_DENOMINATOR > present_count


def _vanished(path: str, *, check_resolution: bool = False) -> bool:
    try:
        status = os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError:
        # An unreadable path is not proven absent.
        return False
    return (S_ISLNK(status.st_mode) or check_resolution) and os.path.realpath(path) != path


def iter_inventory_batches(
    parser: InventoryParser,
    root: Path,
    *,
    clock: Callable[[], float],
    chunk_size: int = 128,
    scope: DiscoveryScope | None = None,
    validate: bool = True,
) -> Iterable[InventoryBatch | InventoryResult]:
    """Capture a root without catalog access; retained paths and errors are bounded."""
    if not 1 <= chunk_size <= READY_LIMIT:
        raise ValueError("chunk_size must be between 1 and 256")
    scope = scope or DiscoveryScope.build(parser, roots=(root,))
    source = scope.source
    root = root.resolve()
    started = clock()
    chunk: list[CapturedSource] = []
    discovered_count = 0
    for item in scope.iter_files(root):
        try:
            status = item.path.stat()
            sidecar_mtime, sidecar_signature = capture_sidecars(parser, item.path)
            after = item.path.stat()
            if (status.st_dev, status.st_ino, status.st_ctime_ns, status.st_size) != (
                after.st_dev,
                after.st_ino,
                after.st_ctime_ns,
                after.st_size,
            ):
                raise OSError(f"source changed during capture: {item.path}")
        except OSError as error:
            scope.fail(root, f"{item.path}: {error}")
            continue
        chunk.append(
            CapturedSource(
                source,
                str(item.root),
                str(item.path),
                SourceSignature(
                    status.st_dev,
                    status.st_ino,
                    status.st_ctime_ns,
                    status.st_mtime_ns,
                    status.st_size,
                    sidecar_mtime,
                    sidecar_signature,
                ),
            )
        )
        if len(chunk) == chunk_size:
            discovered_count += len(chunk)
            yield InventoryBatch(tuple(chunk), scope)
            chunk.clear()
    if chunk:
        discovered_count += len(chunk)
        yield InventoryBatch(tuple(chunk), scope)
    scope.discovered_counts[str(root)] = discovered_count
    if validate:
        scope.revalidate()
    yield InventoryResult(
        source,
        str(root),
        started,
        clock(),
        discovered_count,
        tuple(scope.failures),
        scope.failure_count,
        len(scope.links),
        scope,
        validate,
    )


def inventory_root(
    catalog: SourceCatalog,
    parser: InventoryParser,
    root: Path,
    *,
    clock: Callable[[], float],
    chunk_size: int = 128,
) -> int:
    """Stream one root into the catalog; only a complete walk may mark files missing."""
    raw_source = parser.source
    source = str(getattr(raw_source, "value", raw_source))
    root_path = str(root.resolve(strict=False))
    begin_inventory_scan(catalog, source, root_path, clock())
    count = 0
    for event in iter_inventory_batches(parser, root, clock=clock, chunk_size=chunk_size):
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
    is_active: bool


AdmissionRefusal = Literal["wal_pressure", "checkpoint", "preparation"]


@dataclass(frozen=True)
class Admission:
    refusal: AdmissionRefusal | None = None

    @property
    def admitted(self) -> bool:
        return self.refusal is None


@dataclass(frozen=True)
class RefusedSource:
    scheduled: ScheduledSource
    reason: AdmissionRefusal


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
        can_admit: Callable[[ScheduledSource, RefusedSource | None], Admission] | None = None,
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
        reserved_oldest = oldest[0].key if oldest and not oldest[0].is_active else None
        selected: list[ScheduledSource] = []
        used: set[str] = set()
        refused: dict[str, RefusedSource] = {}
        claims: dict[str, SourceFile | None] = {}
        sequence = self._sequence
        for _ in range(limit):
            lane = self._LANES[sequence % len(self._LANES)]
            candidates = active if lane == "active" else recent if lane == "recent" else oldest
            preferred = [
                item for item in candidates if lane == "oldest" or item.key != reserved_oldest
            ]
            fallback = active + recent if lane == "oldest" else active + recent + oldest
            slot_refused = False
            priority: RefusedSource | None = None
            for candidate in preferred + fallback:
                key = candidate.key
                if key in used:
                    continue
                if key not in claims:
                    claims[key] = catalog.claimable(candidate)
                claimed = claims[key]
                if claimed is None:
                    used.add(key)
                    continue
                scheduled = ScheduledSource(claimed, lane, is_active=candidate.is_active)
                refusal = refused.get(key)
                if refusal is None and can_admit is not None:
                    decision = can_admit(scheduled, priority)
                    if decision.refusal is not None:
                        refusal = RefusedSource(scheduled, decision.refusal)
                        refused[key] = refusal
                if refusal is not None:
                    slot_refused = True
                    if priority is None:
                        priority = refusal
                    continue
                used.add(key)
                selected.append(scheduled)
                # Other lanes may borrow a refused slot, but cannot consume its
                # service obligation. Retry it first once capacity returns.
                if not slot_refused:
                    sequence += 1
                break
            else:
                break
        self._sequence = sequence
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


def stat_signature(path: Path) -> SourceSignature:
    """Small adapter for notification paths; capture callers handle IO errors."""
    stat = os.stat(path)
    return SourceSignature(
        dev=stat.st_dev,
        inode=stat.st_ino,
        ctime_ns=stat.st_ctime_ns,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
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
