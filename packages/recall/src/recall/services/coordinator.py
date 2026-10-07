"""Raw reconciliation control shared by poll and watch daemon paths.

The marker deliberately lives beside the daemon socket rather than in the
database: it must be readable before a damaged database is opened and must
survive a daemon restart.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import duckdb

from recall.core.config import AppConfig, SourceConfig
from recall.core.models import NormalizationCheckpoint, ParseResult, Session
from recall.core.types import UNATTRIBUTED_HOST, Source, default_session_host
from recall.db.source_files import SourceCatalog, SourceFile, SourceSignature, source_key
from recall.parsers import SessionParser, all_parsers
from recall.parsers.checkpoint import UnsupportedResumeState
from recall.parsers.revision import parser_revision
from recall.services.moves import (
    MovedDuplicate,
    plan_vanished_moves,
    supersede_moved_duplicate,
)
from recall.services.reconciler import (
    CapturedSource,
    FairScheduler,
    InventoryBatch,
    InventoryResult,
    ScheduledSource,
    begin_inventory_scan,
    capture_sidecars,
    finish_inventory_scan,
    inventory_root_scopes,
    iter_inventory_batches,
    persist_inventory_batch,
    stat_signature,
)
from recall.services.unsupported_diagnostics import diagnostic_payload, unsupported_summary
from recall.services.watcher import _ContextCounters, index_single_session

logger = logging.getLogger(__name__)

# Move proofs one reconciliation walk examines, by count and by the transcript
# bytes they cover.  Each re-reads a file prefix and a proven one rewrites a
# session inside the walk's writer turn, so a mass move is collapsed a bounded
# slice per walk instead of in one long turn.  16 MiB is about 40k messages,
# roughly a second of writer time.
_MOVE_PROOFS_PER_WALK_MAX = 64
_MOVE_PROOF_BYTES_PER_WALK_MAX = 16 * 1024 * 1024

_PAUSE_FILE = "reconciliation-paused"


def pause_path(config: AppConfig) -> Path:
    return config.data_dir / _PAUSE_FILE


def is_paused(config: AppConfig) -> bool:
    return pause_path(config).is_file()


def set_paused(config: AppConfig, paused: bool) -> bool:
    """Persist operator maintenance intent atomically."""
    path = pause_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    if paused:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text("paused\n", encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        path.unlink(missing_ok=True)
    return is_paused(config)


@dataclass(frozen=True)
class ReconciliationSnapshot:
    paused: bool
    runtime_mode: str
    rpc_ready: bool
    catalog_scan_complete: bool = False
    pending: int = 0
    error: str | None = None
    coverage: tuple[dict[str, object], ...] = ()


def reconciliation_status(
    config: AppConfig,
    *,
    conn: duckdb.DuckDBPyConnection,
    limit: int = 100,
    cursor: str | None = None,
) -> dict[str, object]:
    """Return bounded coverage as of durable observations, separate from readiness."""
    from dataclasses import asdict

    catalog = SourceCatalog(conn, clock=time.time)
    page = catalog.status_page(limit=limit, cursor=cursor)
    rows = conn.execute("""
        SELECT source, root_path, COUNT(*),
          COUNT(*) FILTER (WHERE NOT missing AND desired_generation = committed_generation
             AND committed_offset = size AND last_error IS NULL),
          COUNT(*) FILTER (WHERE NOT missing AND (desired_generation > committed_generation
             OR committed_offset < size OR last_error IS NOT NULL)),
          COUNT(*) FILTER (WHERE retry_count > 0),
          COUNT(*) FILTER (WHERE NOT missing AND last_error = 'unsupported'),
          COUNT(*) FILTER (WHERE missing),
          MIN(first_pending_at) FILTER (WHERE NOT missing AND (
            desired_generation > committed_generation OR committed_offset < size
            OR last_error IS NOT NULL))
        FROM source_files GROUP BY source, root_path
    """).fetchall()
    counts = {(row[0], row[1]): row[2:] for row in rows}
    scans = conn.execute("""
        SELECT source, root_path, scan_started_at, scan_finished_at, scan_complete,
               failure_count, CAST(failures AS VARCHAR) FROM reconciliation_roots
    """).fetchall()
    scan_by_root = {(row[0], row[1]): row[2:] for row in scans}
    coverage: list[dict[str, object]] = []
    for parser in all_parsers(config.sources):
        for scope in inventory_root_scopes(parser):
            key = (scope.source, str(scope.root_path))
            scan = scan_by_root.get(key)
            count = counts.get(key, (0, 0, 0, 0, 0, 0, None))
            coverage.append(
                {
                    "source": scope.source,
                    "root_path": str(scope.root_path) if scope.root_path else None,
                    "configuration": scope.configuration,
                    "available": scope.available,
                    "scan_started_at": scan[0] if scan else None,
                    "scan_finished_at": scan[1] if scan else None,
                    "scan_complete": bool(scan[2]) if scan else scope.configuration == "disabled",
                    "failure_count": scan[3] if scan else 0,
                    "errors": scan[4] if scan else None,
                    **dict(
                        zip(
                            ("discovered", "current", "pending", "retry", "unsupported", "missing"),
                            count[:6],
                            strict=True,
                        )
                    ),
                    "oldest_pending_age": max(0, time.time() - count[6])
                    if count[6] is not None
                    else None,
                }
            )
    eligible_roots = {
        (row["source"], row["root_path"]) for row in coverage if row["configuration"] != "disabled"
    }
    pending = sum(int(count[2]) for key, count in counts.items() if key in eligible_roots)
    scan_complete = all(row["scan_complete"] for row in coverage)
    sidecar_pending = conn.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone()
    assert sidecar_pending is not None
    return {
        "coverage": coverage,
        "pending": pending,
        "out_of_scope_discovered": sum(
            int(count[0]) for key, count in counts.items() if key not in eligible_roots
        ),
        "out_of_scope_pending": sum(
            int(count[2]) for key, count in counts.items() if key not in eligible_roots
        ),
        "catalog_scan_complete": scan_complete,
        "raw_indexing_ready": scan_complete and pending == 0,
        "keyword_search_ready": scan_complete and pending == 0 and sidecar_pending[0] == 0,
        "source_page": [
            {**asdict(row), "eligible": (row.source, row.root_path) in eligible_roots}
            for row in page.rows
        ],
        "next_cursor": page.next_cursor,
        "index_only_sessions": index_only_sessions(conn),
        "unsupported_summary": unsupported_summary(conn),
        "index_migration": _index_migration_status(conn),
    }


def index_only_sessions(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Count this host's indexed sessions whose transcript survives nowhere on disk.

    A session is backed when any row sharing its source session identity maps to
    a present catalog file, so a transcript moved to a new path (and indexed there
    again) is not counted as lost. Rows imported for another host are excluded:
    their transcripts were never expected on this disk. See REQ-RECON-028.
    """
    rows = conn.execute(
        """
        WITH local_sessions AS (
            SELECT s.source, COALESCE(s.source_session_id, s.id) AS identity, s.source_path
            FROM sessions s JOIN session_state st ON st.session_id = s.id
            WHERE st.host IS NULL OR st.host IN (?, ?)
        ), backed AS (
            SELECT DISTINCT l.source, l.identity
            FROM local_sessions l JOIN source_files f
              ON f.source = l.source AND f.source_path = l.source_path AND NOT f.missing
        )
        SELECT l.source, COUNT(DISTINCT l.identity)
        FROM local_sessions l
        ANTI JOIN backed b ON b.source = l.source AND b.identity = l.identity
        GROUP BY l.source
        """,
        [default_session_host(), UNATTRIBUTED_HOST],
    ).fetchall()
    return {str(source): int(count) for source, count in sorted(rows)}


def _index_migration_status(conn: duckdb.DuckDBPyConnection) -> dict[str, object]:
    from dataclasses import asdict as dataclass_asdict

    from recall.services.index_migration import migration_status

    status = dataclass_asdict(migration_status(conn))
    # `backup_path` is the job's recorded rollback target, and the record
    # outlives the directory: snapshot GC retains recovery backups, but an
    # operator reclaiming disk does not, and a path read as a live rollback
    # when nothing is there is worse than no path at all. The presence flag is
    # a status reading rather than a field of the job record, so it stays out
    # of `IndexMigrationStatus` — the storage plan digest hashes that dataclass
    # and must not move with the filesystem.
    backup_path = status.get("backup_path")
    status["backup_path_present"] = (
        Path(backup_path).exists() if isinstance(backup_path, str) and backup_path else None
    )
    return status


@dataclass(frozen=True)
class PreparedRawSource:
    item: SourceFile
    parser: SessionParser
    result: ParseResult | None
    error: str | None = None
    observed: tuple[str, str, SourceSignature] | None = None


@dataclass(frozen=True)
class IndexRequestScope:
    full: bool = False
    recreate: bool = False
    since: datetime | None = None
    project: str | None = None
    home_root: Path | None = None
    host: str | None = None
    embed: bool = False
    # The context mode this request pinned, if any. Per-path options live only
    # while the request is in flight, so a request that pinned one must serve
    # every source it observes itself (`REQ-RECON-025`).
    context: str | None = None

    def owns_every_observed_source(self) -> bool:
        """Whether this request's options must ride with each source it observes."""
        return (
            self.full
            or self.recreate
            or any(
                option is not None
                for option in (self.since, self.project, self.home_root, self.host, self.context)
            )
        )


@dataclass
class RawIndexRequest:
    """Per-path options and results owned by a bounded shared-queue request."""

    config: AppConfig
    parser: SessionParser
    full: bool = False
    host: str | None = None
    changed: bool = False
    context: _ContextCounters = field(default_factory=_ContextCounters)


def config_with_home_root(config: AppConfig, home_root: Path | None) -> AppConfig:
    """Resolve default roots before work crosses an executor's context boundary."""
    if home_root is None:
        return config
    from recall.parsers.common import use_home_root

    with use_home_root(home_root):
        sources = {
            parser.source.value: SourceConfig(
                roots=tuple(
                    scope.root_path
                    for scope in inventory_root_scopes(parser)
                    if scope.root_path is not None
                )
            )
            for parser in all_parsers(config.sources)
        }
    return replace(config, sources=sources)


@dataclass(frozen=True)
class PreparedRawCycle:
    parser: SessionParser
    root: Path
    event: InventoryBatch | InventoryResult | None


def prepare_raw_cycle(
    config: AppConfig, *, source: Source | None = None
) -> Iterator[PreparedRawCycle]:
    """Yield one bounded filesystem batch; never retain an entire root."""
    for parser in all_parsers(config.sources):
        if source is not None and parser.source is not source:
            continue
        for scope in inventory_root_scopes(parser):
            if scope.root_path is None:
                continue
            yield PreparedRawCycle(parser, scope.root_path, None)
            for event in iter_inventory_batches(
                parser,
                scope.root_path,
                clock=time.time,
                optional_root=scope.configuration == "default",
            ):
                yield PreparedRawCycle(parser, scope.root_path, event)


def persist_prepared_inventory(
    prepared: PreparedRawCycle, generation: int | None, *, conn: duckdb.DuckDBPyConnection
) -> tuple[int, tuple[str, ...]]:
    """Persist one inventory event; return the scan generation and changed existing paths."""
    catalog = SourceCatalog(conn, clock=time.time)
    if prepared.event is None:
        started = begin_inventory_scan(
            catalog, prepared.parser.source.value, str(prepared.root), time.time()
        )
        return started, ()
    if generation is None:
        raise RuntimeError("inventory batch has no durable scan generation")
    if isinstance(prepared.event, InventoryBatch):
        return generation, persist_inventory_batch(catalog, prepared.event)
    finish_inventory_scan(
        catalog,
        prepared.event,
        on_vanished=_MovedAwaySuperseder(conn, catalog, host=default_session_host()),
    )
    return generation, ()


class _MovedAwaySuperseder:
    """Retire rows whose transcript was copied elsewhere before it vanished (REQ-INDEX-027).

    A copy-then-delete move indexes the copy while the original still exists,
    so the insert-time proof fails; the original's disappearance completes it.
    One walk examines a bounded number of proofs, all in one writer turn; the
    paths it did not reach stay present for the next walk.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, catalog: SourceCatalog, *, host: str):
        self._conn = conn
        self._catalog = catalog
        self._host = host
        self._proofs_left = _MOVE_PROOFS_PER_WALK_MAX
        self._proof_bytes_left = _MOVE_PROOF_BYTES_PER_WALK_MAX

    def __call__(self, source: str, vanished_paths: list[str]) -> frozenset[str]:
        moves = plan_vanished_moves(
            self._conn,
            source=source,
            vanished_paths=vanished_paths,
            host=self._host,
            proofs_max=self._proofs_left,
            proof_bytes_max=self._proof_bytes_left,
        )
        self._proofs_left -= moves.proofs
        self._proof_bytes_left -= moves.proof_bytes
        if moves.plans:
            # Absence is recorded before any attempt, so a supersession that
            # fails, even by aborting the process, is never retried by a walk.
            self._catalog.mark_missing(
                [source_key(source, plan.predecessor_path) for plan in moves.plans]
            )
        for plan in moves.plans:
            self._supersede(plan)
        if moves.deferred_paths:
            logger.info(
                "deferred %d moved-away candidate(s) to the next walk source=%s",
                len(moves.deferred_paths),
                source,
            )
        return moves.deferred_paths

    def _supersede(self, plan: MovedDuplicate) -> None:
        try:
            supersede_moved_duplicate(self._conn, plan, queue_sidecar_deletes=True)
        except duckdb.FatalException:
            raise
        except Exception as err:
            logger.warning(
                "moved-away supersession failed; left for `recall db supersede-moved`"
                " predecessor_id=%s predecessor_path=%s successor_id=%s error=%s: %s",
                plan.predecessor_id,
                plan.predecessor_path,
                plan.successor_id,
                type(err).__name__,
                err,
            )
            return
        logger.info(
            "superseded moved-away session predecessor_id=%s successor_id=%s",
            plan.predecessor_id,
            plan.successor_id,
        )


def capture_path(parser: SessionParser, path: Path) -> tuple[str, str, SourceSignature]:
    from dataclasses import replace

    signature = stat_signature(path)
    sidecar_mtime, sidecar_signature = capture_sidecars(parser, path)
    signature = replace(
        signature, sidecar_mtime_ns=sidecar_mtime, sidecar_signature=sidecar_signature
    )
    roots = [
        scope.root_path for scope in inventory_root_scopes(parser) if scope.root_path is not None
    ]
    resolved = path.resolve()
    root = next((root for root in roots if resolved.is_relative_to(root)), resolved.parent)
    return str(root), str(resolved), signature


def observe_path(
    parser: SessionParser,
    captured: tuple[str, str, SourceSignature],
    *,
    conn: duckdb.DuckDBPyConnection,
) -> SourceFile:
    root, path, signature = captured
    catalog = SourceCatalog(conn, clock=time.time)
    catalog.observe(parser.source.value, root, path, signature)
    item = catalog.get(parser.source.value, path)
    assert item is not None
    return item


@dataclass(frozen=True)
class ActiveObservationTarget:
    source: str
    source_path: str
    signature: SourceSignature


@dataclass(frozen=True)
class ActiveObservationPage:
    targets: tuple[ActiveObservationTarget, ...]
    next_cursor: str | None


def active_observation_page(
    config: AppConfig,
    *,
    conn: duckdb.DuckDBPyConnection,
    cursor: str | None,
    now: float,
) -> ActiveObservationPage:
    """Page known active paths independently of notification subscriptions."""
    roots = tuple(
        (scope.source, str(scope.root_path))
        for parser in all_parsers(config.sources)
        for scope in inventory_root_scopes(parser)
        if scope.root_path is not None
    )
    scope = " OR ".join("(source = ? AND root_path = ?)" for _ in roots) or "FALSE"
    rows = conn.execute(
        f"""SELECT source, source_path, dev, inode, ctime_ns, mtime_ns, size,
                   sidecar_mtime_ns, sidecar_signature
            FROM source_files
            WHERE source_key > COALESCE(?, '') AND mtime_ns >= ? AND NOT missing
              AND ({scope}) ORDER BY source_key LIMIT 129""",
        [
            cursor,
            int((now - config.daemon.live_idle_threshold) * 1e9),
            *(value for root in roots for value in root),
        ],
    ).fetchall()
    targets = tuple(
        ActiveObservationTarget(
            source=str(row[0]),
            source_path=str(row[1]),
            signature=SourceSignature(
                dev=row[2],
                inode=row[3],
                ctime_ns=int(row[4]),
                mtime_ns=int(row[5]),
                size=int(row[6]),
                sidecar_mtime_ns=int(row[7]),
                sidecar_signature=str(row[8]),
            ),
        )
        for row in rows[:128]
    )
    next_cursor = (
        source_key(targets[-1].source, targets[-1].source_path) if len(rows) > 128 else None
    )
    return ActiveObservationPage(targets, next_cursor)


def capture_active_observations(
    config: AppConfig, page: ActiveObservationPage
) -> tuple[InventoryBatch, tuple[str, ...]]:
    """Capture one bounded stat/sidecar batch without holding a database connection."""
    assert len(page.targets) <= 128
    parsers = {parser.source.value: parser for parser in all_parsers(config.sources)}
    files: list[CapturedSource] = []
    failures: list[str] = []
    for target in page.targets:
        try:
            root, resolved, signature = capture_path(
                parsers[target.source], Path(target.source_path)
            )
            files.append(CapturedSource(target.source, root, resolved, signature))
        except OSError as error:
            if len(failures) < 32:
                failures.append(
                    f"{target.source}:{target.source_path}: {type(error).__name__}: {error}"
                )
    return InventoryBatch(tuple(files)), tuple(failures)


def select_raw_sources(
    parser_by_source: dict[str, SessionParser],
    *,
    conn: duckdb.DuckDBPyConnection,
    scheduler: FairScheduler,
    active_paths: set[str],
    requested_paths: tuple[str, ...] = (),
    active_since_ns: int | None = None,
    excluded_keys: tuple[str, ...] = (),
    can_admit: Callable[[SourceFile], bool] | None = None,
) -> tuple[ScheduledSource, ...]:
    catalog = SourceCatalog(conn, clock=time.time)
    eligible_roots = tuple(
        (scope.source, str(scope.root_path))
        for parser in parser_by_source.values()
        for scope in inventory_root_scopes(parser)
        if scope.root_path is not None
    )
    selected = scheduler.select(
        catalog,
        active_paths | set(requested_paths),
        limit=1,
        eligible_roots=eligible_roots,
        requested_paths=requested_paths,
        active_since_ns=active_since_ns,
        excluded_keys=excluded_keys,
        can_admit=can_admit,
    )
    return tuple(selected)


# Prefix verification rereads every byte the checkpoint claims, so it is read in
# the same sized blocks the parser capture uses rather than one allocation.
_PREFIX_HASH_BLOCK_BYTES = 1024 * 1024


def _on_disk_prefix_sha256(path: Path, size: int) -> str | None:
    """Digest exactly the first `size` bytes, or None when the source is shorter."""
    assert size > 0
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as source:
        while remaining:
            block = source.read(min(_PREFIX_HASH_BLOCK_BYTES, remaining))
            if not block:
                return None
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _verified_resume_checkpoint(
    item: SourceFile, parser: SessionParser, path: Path, size: int
) -> NormalizationCheckpoint | None:
    """The stored proof, when every precondition for a suffix parse holds.

    Preparation runs outside the writer with no connection, so the catalog
    values this compares against are the ones the acknowledgement committed in
    the same statement as the envelope (REQ-INDEX-025). A proof that describes
    a different offset or digest than the acknowledged prefix came from an
    older generation and is worth nothing, and only rereading the bytes
    themselves separates an append from a rewrite that happens to have grown.

    Returning None is the full reference parse, never an error: no fallback
    here costs rows, only time.
    """
    checkpoint = NormalizationCheckpoint.decode(item.normalization_checkpoint)
    if checkpoint is None:
        return None
    if checkpoint.parser_revision != parser_revision(type(parser)):
        return None
    if checkpoint.offset <= 0 or checkpoint.offset != item.committed_offset:
        return None
    if checkpoint.prefix_sha256 != item.committed_prefix_sha256:
        return None
    if item.signature is None or item.signature.dev is None or item.signature.inode is None:
        return None
    if (
        checkpoint.source_dev != item.signature.dev
        or checkpoint.source_inode != item.signature.inode
    ):
        return None
    # Equal is not enough: a source pending with no bytes past the acknowledged
    # prefix owes its generation to something the transcript does not carry --
    # an explicit reparse, a sidecar edit, a retry -- and an empty suffix would
    # answer none of them. Shorter is the truncation REQ-INDEX-012 names.
    if size <= checkpoint.offset:
        return None
    if _on_disk_prefix_sha256(path, checkpoint.offset) != checkpoint.prefix_sha256:
        return None
    return checkpoint


def _parse_verified_source(
    parser: SessionParser, path: Path, item: SourceFile, size: int, *, full: bool
) -> ParseResult:
    """Normalize the suffix the stored proof justifies, else the whole source.

    A suffix that diagnoses anything is discarded rather than committed: the
    diagnosis is about records the adapter met while carrying state it was
    handed, and only a parse that saw the whole file can say what the source
    really holds. Publishing the speculative rows first and reconciling later
    would put them in front of a reader (REQ-INDEX-025).
    """
    if full:
        return parser.parse(path, offset=0)
    checkpoint = _verified_resume_checkpoint(item, parser, path, size)
    if checkpoint is None:
        return parser.parse(path, offset=0)
    try:
        suffix = parser.parse(
            path,
            offset=checkpoint.offset,
            message_idx_base=checkpoint.message_idx_base,
            orphan_tool_call_idx_base=checkpoint.orphan_tool_call_idx_base,
            resume_state=checkpoint.adapter_state,
        )
    except UnsupportedResumeState:
        # This build cannot read the state the envelope carries. That is a
        # checkpoint this adapter no longer honors, not a parser defect, so
        # the source takes the reference path instead of failing.
        return parser.parse(path, offset=0)
    # Verification and parsing open the path separately. The prefix captured
    # on the parser's own handle must still be the one we authorized; otherwise
    # a replacement in that interval would merge a new suffix onto old rows.
    if (
        suffix.initial_prefix_sha256 != checkpoint.prefix_sha256
        or suffix.source_dev != checkpoint.source_dev
        or suffix.source_inode != checkpoint.source_inode
        or suffix.diagnostics
    ):
        return parser.parse(path, offset=0)
    return suffix


def prepare_raw_sources(
    selected: tuple[SourceFile, ...],
    parser_by_source: dict[str, SessionParser],
    *,
    full: bool = False,
) -> tuple[PreparedRawSource, ...]:
    """Parse at most two complete source captures outside the writer.

    `full` is the in-flight request's own reparse option (`REQ-RECON-025`). The
    catalog cannot infer it: an explicit reparse of a source that also grew
    raises the same desired generation an ordinary append does, and answering
    it with a suffix would leave the history the caller asked to rebuild.
    """
    ready: list[PreparedRawSource] = []
    for item in selected[:2]:
        parser = parser_by_source[item.source]
        observed = None
        try:
            path = Path(item.source_path)
            before = capture_path(parser, path)
            result = _parse_verified_source(parser, path, item, before[2].size, full=full)
            observed = capture_path(parser, path)
            if before[2].sidecar_signature != observed[2].sidecar_signature:
                raise RuntimeError("source sidecars changed during preparation")
            ready.append(PreparedRawSource(item, parser, result, observed=observed))
        except Exception as err:
            ready.append(PreparedRawSource(item, parser, None, str(err), observed=observed))
    return tuple(ready)


def commit_prepared_raw_sources(
    prepared: tuple[PreparedRawSource, ...],
    config: AppConfig,
    *,
    conn: duckdb.DuckDBPyConnection,
    request: RawIndexRequest | None = None,
) -> int:
    """Commit raw content and catalog acknowledgement in the same transaction."""
    if is_paused(config):
        return 0
    from recall.db.source_files import source_key as catalog_key
    from recall.services.index_migration import (
        mark_captured_complete,
        maybe_complete_migration,
        migration_status,
    )
    from recall.services.indexer import _preserves_indexed_prefix, _require_committable_capture

    catalog = SourceCatalog(conn, clock=time.time)
    committed = 0

    def settle_captured(source: str, source_path: str) -> None:
        if migration_status(conn).phase == "running":
            mark_captured_complete(conn, catalog_key(source, source_path))

    for capture in prepared:
        if is_paused(config):
            break
        item = capture.item
        try:
            current = catalog.get(item.source, item.source_path)
            if current is None or (
                current.committed_generation != item.committed_generation
                or current.content_epoch != item.content_epoch
            ):
                raise RuntimeError("committed source generation changed during preparation")
            if (
                capture.observed is not None
                and current.desired_generation == item.desired_generation
            ):
                # Carry the capture's observation forward only if the catalog has
                # not seen a later change. An older verified generation can commit
                # while later appends remain pending; observations never move back.
                # Prefix validation below still rejects growth plus a prefix edit.
                current = observe_path(capture.parser, capture.observed, conn=conn)
                item = current
            if capture.error is not None or capture.result is None:
                raise RuntimeError(capture.error or "raw preparation produced no capture")
            if current.current:
                continue
            has_history = (
                conn.execute(
                    "SELECT 1 FROM sessions WHERE id = ?", [capture.result.session.id]
                ).fetchone()
                is not None
            )
            _require_committable_capture(
                Path(item.source_path), capture.result, has_indexed_history=has_history, conn=conn
            )
            # A verified suffix appends to the prefix it was proven against, so
            # it can never be the rewrite `_preserves_indexed_prefix` looks for
            # -- and asking would compare stored history against new messages
            # alone and bump the content epoch on every ordinary append.
            semantic_rewrite = (
                capture.result.is_full_parse
                and has_history
                and not _preserves_indexed_prefix(conn, capture.result.session)
            )

            def acknowledge(
                session: Session,
                result: ParseResult,
                *,
                source: SourceFile = item,
                rewrite: bool = semantic_rewrite,
            ) -> None:
                prefix = result.committed_prefix_sha256
                if prefix is None:
                    raise RuntimeError("raw reconciliation requires a captured prefix digest")
                checkpoint = result.normalization_checkpoint
                catalog.bind_session(source.source, source.source_path, session.id)
                if not catalog.acknowledge(
                    source.source,
                    source.source_path,
                    source.desired_generation,
                    result.next_byte_offset,
                    prefix,
                    semantic_rewrite=rewrite,
                    # The proof rides in the acknowledgement that commits the
                    # rows it describes, so no later turn can read one beside
                    # an offset it does not belong to (REQ-INDEX-025).
                    checkpoint=checkpoint.encode() if checkpoint is not None else None,
                    full_parse=result.is_full_parse,
                ):
                    raise RuntimeError("catalog generation changed before acknowledgement")

            committed += int(
                index_single_session(
                    Path(item.source_path),
                    capture.parser,
                    config,
                    conn=conn,
                    lightweight_context=True,
                    on_commit=acknowledge,
                    prepared_result=capture.result,
                    host=request.host if request else None,
                    context_counts=request.context if request else None,
                )
            )
            settle_captured(item.source, item.source_path)
            if capture.result.diagnostics:
                details = diagnostic_payload(capture.result.diagnostics)
                if any(d.kind == "unsupported_record" for d in capture.result.diagnostics):
                    catalog.defer_unsupported(item.source, item.source_path, details)
                else:
                    catalog.defer_until_changed(
                        item.source, item.source_path, "incomplete capture", details
                    )
        except Exception as err:
            details = diagnostic_payload(capture.result.diagnostics) if capture.result else {}
            if capture.result and any(
                d.kind == "unsupported_record" for d in capture.result.diagnostics
            ):
                catalog.defer_unsupported(item.source, item.source_path, details)
                # Keep prior history and the unsupported diagnostic; the captured
                # key is still accounted so versioned migration can finish.
                settle_captured(item.source, item.source_path)
            else:
                catalog.fail(item.source, item.source_path, str(err), details)
    if migration_status(conn).phase == "running":
        maybe_complete_migration(conn)
    return committed
