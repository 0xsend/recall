from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from uuid import uuid4

import duckdb

from recall.core.config import AppConfig
from recall.core.durability import sync_directory
from recall.db import advisory_lock
from recall.db.connection import (
    STORAGE_VERSION,
    _apply_runtime_pragmas,
    storage_version,
)
from recall.db.lease import (
    DbLease,
    DbLifetime,
    StandaloneLifetime,
    owned_connection,
    owned_resource,
)
from recall.db.queries import create_fts_indexes
from recall.db.schema import ensure_schema_on
from recall.services.recovery_admission import RecoveryAdmission, offline_file_lifetime
from recall.services.snapshots import DEFAULT_GC_DAYS, gc_snapshots

logger = logging.getLogger("recall.compaction")

DATA_TABLES = (
    "sessions",
    "session_state",
    "source_files",
    "reconciliation_roots",
    "messages",
    "message_state",
    "tool_calls",
    "tool_results",
    "tool_use_ids",
    "session_stop_markers",
    "live_marks",
    "fts_sidecar_pending",
    "usage_events",
    "usage_log_cursors",
    "message_embeddings",
    "tool_call_embeddings",
    "embedding_cache",
    "index_migration_scope",
)
# Small metadata tables. Dropped and recopied like DATA_TABLES; the leading
# DELETE clears rows ensure_schema seeds into a fresh DB so the copy cannot
# collide. schema_migration_undo carries migration pre-images (REQ-MIG-008)
# and must survive compaction like the migration log beside it.
SINGLETON_TABLES = (
    "runtime_state",
    "schema_version",
    "schema_migrations",
    "schema_migration_undo",
    "index_migration_jobs",
    "daemon_index_repair",
)
SENTINEL_NAME = "recall.compacting"
_compaction_sentinel_counts: dict[Path, int] = {}
_compaction_sentinel_lock = threading.Lock()


@dataclass(frozen=True)
class BloatStats:
    """How far a database file has drifted from the live data it holds.

    ``ratio`` is ``used_bytes / live_bytes``. Free blocks are excluded: a
    checkpoint writes every row group a commit touched into fresh blocks and
    only then frees the old ones, and the next checkpoint reuses them. On a
    freshly compacted file one ordinary commit therefore grows ``file_size``
    by the touched tables' footprint without adding a single dead row, and a
    file-size ratio made every auto-compaction check trigger the next
    compaction. Dead row-versions stay in used blocks while any row of their
    row group survives, so the ratio still tracks real churn.
    """

    file_size: int
    used_bytes: int
    live_bytes: int
    ratio: float
    block_size: int


@dataclass(frozen=True)
class CompactResult:
    before: BloatStats
    after: BloatStats
    tables_copied: dict[str, int]
    elapsed_seconds: float
    replaced: bool
    skipped_reason: str | None


class CompactionError(RuntimeError):
    """Compaction aborted; original DB intact."""


class _CompactionSentinel:
    def __init__(self, path: Path) -> None:
        self._path = path.expanduser().resolve(strict=False)

    def __enter__(self) -> Path:
        # The sentinel means some caller is in the compaction critical section.
        # Daemon auto-fork checks only file existence, so nested users keep the
        # file alive until the outermost caller exits.
        with _compaction_sentinel_lock:
            count = _compaction_sentinel_counts.get(self._path, 0)
            if count == 0:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.touch()
                logger.info("compact sentinel created path=%s", self._path)
            else:
                logger.debug("compact sentinel reused path=%s count=%d", self._path, count + 1)
            _compaction_sentinel_counts[self._path] = count + 1
        return self._path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        with _compaction_sentinel_lock:
            count = _compaction_sentinel_counts[self._path] - 1
            if count > 0:
                _compaction_sentinel_counts[self._path] = count
                logger.debug("compact sentinel retained path=%s count=%d", self._path, count)
                return
            _compaction_sentinel_counts.pop(self._path, None)
            try:
                self._path.unlink(missing_ok=True)
                logger.info("compact sentinel removed path=%s", self._path)
            except OSError:
                logger.exception("failed to remove compact sentinel path=%s", self._path)


def _sentinel_path(config: AppConfig) -> Path:
    return config.data_dir / SENTINEL_NAME


def _compaction_sentinel(config: AppConfig) -> _CompactionSentinel:
    """Create a lifecycle sentinel that blocks daemon auto-fork during compact."""
    return _CompactionSentinel(_sentinel_path(config))


def _clone_file(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` using the filesystem's copy-on-write primitive.

    Raises ``OSError`` when the filesystem cannot clone, so the caller can fall
    back to a byte copy.  On APFS and reflink-capable Linux filesystems a clone
    is O(1) and allocates nothing until the two files diverge, which is the
    whole point here: the pre-compact backup is a full second copy of the
    database, and paying for it in bytes is what pushed peak demand to roughly
    three times the database size and produced ENOSPC on the hosts that most
    needed compacting.
    """
    argv = (
        ["cp", "-c", str(src), str(dst)]
        if sys.platform == "darwin"
        else ["cp", "--reflink=always", str(src), str(dst)]
    )
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise OSError(result.stderr.strip() or f"clone exited {result.returncode}")


def _clone_or_copy(src: Path, dst: Path) -> None:
    """Clone where the filesystem allows it, byte-copy where it does not."""
    try:
        _clone_file(src, dst)
        return
    except OSError as err:
        logger.debug("clone unavailable path=%s falling back to copy: %s", src, err)
    shutil.copy2(src, dst)


def estimate_bloat_ratio(db_path: Path, config: AppConfig) -> BloatStats:
    return estimate_bloat_ratio_owned(db_path, config, lifetime=StandaloneLifetime())


def estimate_bloat_ratio_owned(
    db_path: Path, config: AppConfig, *, lifetime: DbLifetime
) -> BloatStats:
    if not db_path.exists():
        return BloatStats(file_size=0, used_bytes=0, live_bytes=0, ratio=0.0, block_size=0)
    target = replace(config, db_path=db_path)
    admission = RecoveryAdmission(target, lifetime)
    with owned_connection(
        lifetime,
        "bloat",
        lambda: admission.open_preserving(read_only=True),
    ) as lease:
        return estimate_bloat_ratio_on(lease, db_path)


def estimate_bloat_ratio_on(lease: DbLease, db_path: Path) -> BloatStats:
    """Materialize bloat statistics on the already admitted parent connection."""
    block_size, used_blocks = lease.call("bloat_blocks", _database_blocks)
    live_bytes = lease.call("bloat_live_bytes", lambda conn: _estimate_live_bytes(conn, block_size))
    used_bytes = used_blocks * block_size
    return BloatStats(
        file_size=db_path.stat().st_size,
        used_bytes=used_bytes,
        live_bytes=live_bytes,
        ratio=float(used_bytes / live_bytes) if live_bytes else 0.0,
        block_size=block_size,
    )


def compact(config: AppConfig) -> CompactResult:
    """Ordinary offline compaction preserves recovery state and grants no release."""
    with offline_file_lifetime(config) as (lifetime, admission), _compaction_sentinel(config):
        return _compact_locked(config, lifetime=lifetime, admission=admission)


def compact_owned(config: AppConfig, *, lifetime: DbLifetime) -> CompactResult:
    with advisory_lock(config.lock_path), _compaction_sentinel(config):
        return _compact_locked(
            config, lifetime=lifetime, admission=RecoveryAdmission(config, lifetime)
        )


def _compact_locked(
    config: AppConfig, *, lifetime: DbLifetime, admission: RecoveryAdmission
) -> CompactResult:
    start = time.monotonic()
    db_path = config.db_path
    identity = uuid4().hex
    compact_path = _sibling(db_path, f".compact-{identity}")
    backup_path = _sibling(db_path, f".pre-compact-{identity}")
    with owned_connection(
        lifetime,
        "compact_source",
        lambda: admission.open_preserving(read_only=True),
    ) as lease:
        source_storage_version = lease.call("compact_storage_version", storage_version)
        before = estimate_bloat_ratio_on(lease, db_path)
    compact_storage_version = (source_storage_version or STORAGE_VERSION).removesuffix("+")
    replaced = False
    backup_complete = False
    try:
        with owned_connection(
            lifetime,
            "compact_candidate",
            lambda: duckdb.connect(
                str(compact_path),
                config={"storage_compatibility_version": compact_storage_version},
            ),
        ) as lease:
            lease.call("compact_pragmas", lambda conn: _apply_runtime_pragmas(conn, config))
            ensure_schema_on(lease, embed_dim=config.embedding.dimensions)
            tables_copied = lease.call("compact_copy", lambda conn: _copy_database(conn, config))
            lease.call("compact_checkpoint", lambda conn: conn.execute("CHECKPOINT").fetchone())
        _optimize_sidecar_after_checkpoint(config, lifetime=lifetime)
        with owned_connection(
            lifetime,
            "compact_verification",
            lambda: duckdb.connect(str(compact_path), read_only=True),
        ) as lease:
            lease.call("compact_verify_pragmas", lambda conn: _apply_runtime_pragmas(conn, config))
            lease.call(
                "compact_fresh_verification",
                lambda conn: _verify_compact_database(conn, tables_copied),
            )
            after = estimate_bloat_ratio_on(lease, compact_path)

        # Every attempt owns unique paths. A failed predecessor's matched set
        # can never be silently reused or removed by another attempt.
        _clone_or_copy(db_path, backup_path)
        live_wal_path = _wal_path(db_path)
        if live_wal_path.exists():
            _clone_or_copy(live_wal_path, _wal_path(backup_path))
        for path in (backup_path, _wal_path(backup_path)):
            if path.exists():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        sync_directory(backup_path.parent)
        backup_complete = True
        try:
            _replace_database_files(compact_path, db_path)
        except OSError as error:
            raise CompactionError(
                f"database replacement failed; retained backup at {backup_path}"
            ) from error
        replaced = True
        gc_result = gc_snapshots(config, days=DEFAULT_GC_DAYS)
        if gc_result.failed_paths:
            logger.warning("snapshots gc failed after compact: %s", gc_result.failed_paths[:5])
        _remove_database_files(backup_path)
        return CompactResult(
            before=before,
            after=after,
            tables_copied=tables_copied,
            elapsed_seconds=time.monotonic() - start,
            replaced=True,
            skipped_reason=None,
        )
    except BaseException as error:
        lifetime.observe(error, "compaction")
        cleanup_errors: list[BaseException] = []
        if not replaced:
            # A complete source backup remains after failed replacement. An
            # incomplete copy created by this attempt is not a usable backup.
            paths = (compact_path,) if backup_complete else (compact_path, backup_path)
            for path in paths:
                try:
                    _remove_database_files(path)
                except BaseException as cleanup:
                    lifetime.observe(cleanup, "compaction_cleanup")
                    cleanup_errors.append(cleanup)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "compaction and cleanup failed", [error, *cleanup_errors]
            ) from None
        if isinstance(error, Exception) and not isinstance(error, CompactionError):
            raise CompactionError(f"database compaction failed: {error}") from error
        raise


def _copy_database(conn: duckdb.DuckDBPyConnection, config: AppConfig) -> dict[str, int]:
    db_path = config.db_path
    tables_copied: dict[str, int] = {}
    logger.info("attaching source database read-only path=%s", db_path)
    conn.execute(f"ATTACH {_sql_string(db_path)} AS old (READ_ONLY)")

    for table in SINGLETON_TABLES:
        # ensure_schema seeds singleton rows in a fresh DB. Delete first so the
        # database copy cannot collide on primary key placeholders if DuckDB
        # ever supports full-copying into an existing catalog.
        conn.execute(f"DELETE FROM {table}")

    new_db = _current_database(conn)
    # DuckDB's full database copy recreates catalog objects, including
    # indexes and managed schemas. Drop the bootstrap schema after the
    # singleton truncation so COPY runs as DuckDB's documented primitive.
    _drop_bootstrap_tables(conn)
    logger.info("copying database with COPY FROM DATABASE old TO %s", new_db)
    conn.execute(f"COPY FROM DATABASE old TO {_sql_identifier(new_db)}")
    if config.fts.backend == "sqlite_sidecar":
        logger.info(
            "skipping DuckDB FTS shadow schema drop/rebuild because "
            "fts.backend=sqlite_sidecar is active"
        )
    else:
        _drop_fts_shadow_schemas(conn)

    for table in (*DATA_TABLES, *SINGLETON_TABLES):
        row_count = _verify_table_count(conn, table)
        tables_copied[table] = row_count
        logger.info("verified copied table=%s rows=%d", table, row_count)

    conn.execute("DETACH old")
    if config.fts.backend == "duckdb":
        # FTS indexes are derived from message/tool-call rows. Rebuilding them
        # refreshes any copied shadow state and preserves the configured fields.
        logger.info("rebuilding FTS indexes")
        create_fts_indexes(conn, config.fts)
    return tables_copied


def _verify_compact_database(
    conn: duckdb.DuckDBPyConnection, expected_counts: dict[str, int]
) -> None:
    """Verify the private candidate through its freshly opened read-only lease."""
    for table, count in expected_counts.items():
        quoted = _sql_identifier(table)
        if conn.execute(f"SELECT count(*) FROM {quoted}").fetchone() != (count,):
            raise CompactionError(f"compacted table count changed after reopen: {table}")
    tables = conn.execute(
        "SELECT schema_name, table_name FROM duckdb_tables() WHERE NOT internal"
    ).fetchall()
    for schema, table in tables:
        name = f"{_sql_identifier(schema)}.{_sql_identifier(table)}"
        segments = conn.execute("SELECT compression FROM pragma_storage_info(?)", [name]).fetchall()
        if any(compression.lower() == "roaring" for (compression,) in segments):
            raise CompactionError(f"compacted table retains Roaring encoding: {name}")


def _optimize_sidecar_after_checkpoint(config: AppConfig, *, lifetime: DbLifetime) -> None:
    if config.fts.backend != "sqlite_sidecar":
        return

    from recall.db.fts_sidecar import (
        FtsSidecarUnavailableError,
        open_sidecar_for_publication,
        optimize_sidecar,
        sidecar_path,
    )

    try:
        with owned_resource(
            lifetime,
            "compact_sidecar",
            lambda: open_sidecar_for_publication(sidecar_path(config.data_dir)),
        ) as sidecar_conn:
            lifetime.call_db("compact_sidecar:optimize", lambda: optimize_sidecar(sidecar_conn))
    except (FtsSidecarUnavailableError, sqlite3.Error) as err:
        if lifetime.observe(err, "compact_sidecar:optimization"):
            raise
        if isinstance(err, FtsSidecarUnavailableError):
            logger.warning("compact: sidecar unavailable, skipping optimize: %s", err)
        else:
            logger.warning("compact: sidecar optimize failed, continuing: %s", err)


def _verify_table_count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    source_count = _count_rows(conn, f"old.{table}")
    dest_count = _count_rows(conn, table)
    if dest_count != source_count:
        raise CompactionError(
            f"row count mismatch for {table}: old={source_count} new={dest_count}"
        )
    return dest_count


def _current_database(conn: duckdb.DuckDBPyConnection) -> str:
    row = conn.execute("SELECT current_database()").fetchone()
    if row is None or row[0] is None:
        raise CompactionError("could not determine compact database alias")
    return str(row[0])


def _drop_bootstrap_tables(conn: duckdb.DuckDBPyConnection) -> None:
    for table in (*DATA_TABLES, *SINGLETON_TABLES):
        conn.execute(f"DROP TABLE IF EXISTS {_sql_identifier(table)}")


def _drop_fts_shadow_schemas(conn: duckdb.DuckDBPyConnection) -> None:
    rows = conn.execute(
        """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name LIKE 'fts\\_main\\_%' ESCAPE '\\'
        ORDER BY schema_name
        """
    ).fetchall()
    for (schema_name,) in rows:
        # FTS shadow schemas are derived state. Keeping copied schemas would let
        # disabled or field-changed FTS config leak through compaction.
        conn.execute(f"DROP SCHEMA IF EXISTS {_sql_identifier(str(schema_name))} CASCADE")


def _count_rows(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    if row is None:
        raise CompactionError(f"could not count rows in {table}")
    return int(row[0])


def _database_blocks(conn: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    """Return (block_size, used_blocks) for the connection's database."""
    row = conn.execute(
        """
        SELECT block_size, used_blocks
        FROM pragma_database_size()
        WHERE database_name = current_database()
        """
    ).fetchone()
    if row is None or row[0] is None or row[1] is None:
        return 0, 0
    return int(row[0]), int(row[1])


def _estimate_live_bytes(conn: duckdb.DuckDBPyConnection, block_size: int) -> int:
    """Return the bytes the *live* rows need, discounting dead row-versions.

    The previous measure summed every block referenced by a column segment and
    called that "live". After INSERT-OR-REPLACE / UPDATE churn, DuckDB reclaims
    a row group's blocks only once *all* its rows are dead; a row group with a
    single surviving row keeps every block it ever touched referenced. A table
    that is 99% dead versions therefore read as ~100% live — the observed 1.01x
    on a 145 GiB database whose true live data was ~6.5 GiB (95% dead, almost
    all in tool_call_embeddings after ~115 rewrites per row). Compaction copies
    only live rows, so its output tracks live data, not referenced blocks.

    Fix: scale each table's referenced-block bytes by its live/physical row
    ratio. `count(*)` excludes dead versions (live rows); physical row-versions
    come from summed segment counts. A healthy, dense table has live == physical
    so the scale is 1.0 and the ratio stays ~1.0; a churned table collapses to
    its live footprint so the file:live ratio crosses the compaction threshold.
    """
    total = 0.0
    for table in _main_tables(conn):
        blocks = _table_referenced_blocks(conn, table)
        if blocks == 0:
            continue
        live, physical = _table_live_and_physical_rows(conn, table)
        live_fraction = 1.0 if physical <= 0 else min(1.0, live / physical)
        total += blocks * block_size * live_fraction
    return int(total)


def _table_referenced_blocks(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    # DuckDB reports -1 for non-file pseudo blocks; excluding it avoids inflating
    # live bytes by whole blocks and hiding real compaction candidates.
    row = conn.execute(
        f"""
        SELECT COUNT(DISTINCT block_id) FROM (
            SELECT block_id
            FROM pragma_storage_info({_sql_string(f"main.{table}")})
            WHERE block_id IS NOT NULL AND block_id >= 0
            UNION
            SELECT block_id
            FROM (
                SELECT UNNEST(additional_block_ids) AS block_id
                FROM pragma_storage_info({_sql_string(f"main.{table}")})
                WHERE additional_block_ids IS NOT NULL
            )
            WHERE block_id IS NOT NULL AND block_id >= 0
        )
        """
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _table_live_and_physical_rows(conn: duckdb.DuckDBPyConnection, table: str) -> tuple[int, int]:
    """Return (live_rows, physical_row_versions) for a main table.

    `physical_row_versions` is the MIN over columns of each column's summed
    data-segment row counts. A scalar column stores one value per physical row
    (live and dead), while a fixed-size ARRAY column (e.g. FLOAT[384] embeddings)
    stores one value per element; taking the MIN selects a scalar column, giving
    the true physical row count instead of an element-inflated one. VALIDITY
    segments are excluded: DuckDB emits one per column carrying its own
    row-count, so summing them alongside the data segments doubles every count
    and would make a pristine table look 2x bloated (a spurious 0.5 fraction).
    """
    live_row = conn.execute(f'SELECT count(*) FROM main."{table}"').fetchone()
    live = int(live_row[0]) if live_row and live_row[0] is not None else 0
    physical_row = conn.execute(
        f"""
        SELECT MIN(column_rows) FROM (
            SELECT SUM(count) AS column_rows
            FROM pragma_storage_info({_sql_string(f"main.{table}")})
            WHERE segment_type != 'VALIDITY'
            GROUP BY column_name
        )
        """
    ).fetchone()
    physical = int(physical_row[0]) if physical_row and physical_row[0] is not None else 0
    return live, physical


def _main_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
          AND table_name NOT LIKE 'fts_main_%'
        ORDER BY table_name
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def _replace_database_files(replacement_path: Path, db_path: Path) -> None:
    """Replace the live DB with replacement_path, including WAL housekeeping.

    The visible database path flips with one same-directory atomic rename. Any
    existing live WAL is staged aside first so DuckDB can never replay an old
    sidecar against the newly compacted database file.
    """
    replacement_wal_path = _wal_path(replacement_path)
    live_wal_path = _wal_path(db_path)
    staged_live_wal_path: Path | None = None

    if live_wal_path.exists():
        staged_live_wal_path = _wal_path(replacement_path).with_suffix(".staged-source-wal")
        os.replace(live_wal_path, staged_live_wal_path)

    try:
        os.replace(replacement_path, db_path)
    except OSError:
        if staged_live_wal_path is not None and staged_live_wal_path.exists():
            try:
                os.replace(staged_live_wal_path, live_wal_path)
            except OSError:
                logger.exception(
                    "failed to restore staged WAL after replace failure path=%s",
                    live_wal_path,
                )
        raise

    if replacement_wal_path.exists():
        os.replace(replacement_wal_path, live_wal_path)
    if staged_live_wal_path is not None and staged_live_wal_path.exists():
        staged_live_wal_path.unlink()


def _remove_database_files(db_path: Path) -> None:
    _remove_if_exists(db_path)
    _remove_if_exists(_wal_path(db_path))


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _sibling(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.name}{suffix}")


def _wal_path(db_path: Path) -> Path:
    return db_path.with_suffix(db_path.suffix + ".wal")


def _sql_string(value: str | Path) -> str:
    raw = str(value)
    return "'" + raw.replace("'", "''") + "'"


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
