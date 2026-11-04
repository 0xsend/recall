"""Versioned historical index migration, separate from normal reconciliation."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import batched
from pathlib import Path
from typing import Literal, TypedDict

import duckdb

from recall.core.config import AppConfig
from recall.db.connection import (
    STORAGE_VERSION,
    connect_existing,
    connect_storage_transition,
    storage_upgrade_needed,
    storage_version,
)
from recall.db.schema import INDEX_MIGRATION_VERSION
from recall.db.source_files import SourceCatalog
from recall.services.migration_backup import original_backup
from recall.services.recreation import consistent_backup, restore_consistent_backup

Phase = Literal["idle", "backup", "running", "verifying", "failed"]


class _JobState(TypedDict):
    phase: Phase
    backup_path: str | None
    captured_count: int
    completed_count: int
    started_at: datetime | None
    error: str | None
    target_version: int
    storage_target: str | None
    storage_attempt: int


@dataclass(frozen=True)
class IndexMigrationStatus:
    phase: Phase
    applied_version: int
    target_version: int
    captured: int
    completed: int
    remaining: int
    backup_path: str | None
    started_at: datetime | None
    error: str | None
    eligible: bool
    storage_version: str | None
    storage_target: str | None
    storage_attempt: int
    storage_eligible: bool


@dataclass(frozen=True)
class StorageMigrationPlan:
    operation_id: str
    plan_id: str
    db_path: str
    desired_storage_version: str
    migration: IndexMigrationStatus


def plan_storage_migration(
    conn: duckdb.DuckDBPyConnection, config: AppConfig
) -> StorageMigrationPlan:
    """Inspect the singleton format operation and bind its file/job generation."""
    status = migration_status(conn)
    if status.phase != "idle":
        require_original_backup(conn, config)
    path = config.db_path.resolve()
    stat = path.stat()
    operation = hashlib.sha256(f"{path}:{STORAGE_VERSION}".encode()).hexdigest()[:32]
    generation = (
        operation,
        stat.st_dev,
        stat.st_ino,
        status.storage_version,
        status.applied_version,
        status.phase,
        status.backup_path,
        str(status.started_at),
        status.captured,
        status.storage_attempt,
    )
    digest = hashlib.sha256(json.dumps(generation).encode()).hexdigest()
    return StorageMigrationPlan(operation, digest, str(path), STORAGE_VERSION, status)


def applied_version(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute(
        "SELECT index_migration_version FROM runtime_state WHERE singleton"
    ).fetchone()
    if row is None or row[0] is None:
        return 0
    return int(row[0])


def is_eligible(conn: duckdb.DuckDBPyConnection) -> bool:
    job = _job(conn)
    if job["phase"] != "idle" and job["captured_count"] > 0:
        return True
    return applied_version(conn) < INDEX_MIGRATION_VERSION


def migration_status(conn: duckdb.DuckDBPyConnection) -> IndexMigrationStatus:
    job = _job(conn)
    captured = int(job["captured_count"])
    completed = int(job["completed_count"])
    remaining = max(0, captured - completed)
    applied = applied_version(conn)
    current_storage = storage_version(conn)
    return IndexMigrationStatus(
        phase=job["phase"],
        applied_version=applied,
        target_version=INDEX_MIGRATION_VERSION,
        captured=captured,
        completed=completed,
        remaining=remaining,
        backup_path=job["backup_path"],
        started_at=job["started_at"],
        error=job["error"],
        eligible=is_eligible(conn),
        storage_version=current_storage,
        storage_target=job["storage_target"],
        storage_attempt=job["storage_attempt"],
        storage_eligible=storage_upgrade_needed(current_storage),
    )


def captured_keys(conn: duckdb.DuckDBPyConnection) -> tuple[str, ...]:
    rows = conn.execute(
        "SELECT source_key FROM index_migration_scope ORDER BY source_key"
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def begin_migration(conn: duckdb.DuckDBPyConnection, config: AppConfig) -> IndexMigrationStatus:
    from recall.services.coordinator import is_paused

    if is_paused(config):
        raise RuntimeError("reconciliation is paused")
    job = _job(conn)
    if job["phase"] != "idle":
        require_original_backup(conn, config)
        if job["captured_count"] == 0:
            raise RuntimeError("storage maintenance must complete before index migration")
        target = STORAGE_VERSION if storage_upgrade_needed(storage_version(conn)) else None
        conn.execute(
            """UPDATE index_migration_jobs SET phase='running', error=NULL,
                   storage_target=COALESCE(storage_target, ?) WHERE singleton""",
            [target],
        )
        return migration_status(conn)
    if not is_eligible(conn):
        raise RuntimeError("index migration is not eligible")

    backup = _create_backup(conn, config, kind="index-migration")
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM index_migration_scope")
        conn.execute(
            """INSERT INTO index_migration_scope (source_key, source, source_path, completed)
               SELECT source_key, source, source_path, FALSE
               FROM source_files WHERE NOT missing"""
        )
        catalog = SourceCatalog(conn, clock=_now)
        requests = [
            (str(source), str(path))
            for source, path in conn.execute(
                "SELECT source, source_path FROM index_migration_scope ORDER BY source_key"
            ).fetchall()
        ]
        for batch in batched(requests, 256):
            catalog.force_reconcile_batch(batch)
        captured = conn.execute("SELECT COUNT(*) FROM index_migration_scope").fetchone()
        assert captured is not None
        captured_count = int(captured[0])
        if captured_count <= 0:
            raise RuntimeError("captured scope is empty")
        target = STORAGE_VERSION if storage_upgrade_needed(storage_version(conn)) else None
        conn.execute(
            """UPDATE index_migration_jobs
               SET phase = 'running', backup_path = ?, captured_count = ?, completed_count = 0,
                   error = NULL, target_version = ?, started_at = CURRENT_TIMESTAMP,
                   storage_target = ?, storage_attempt = 0
               WHERE singleton""",
            [str(backup), captured_count, INDEX_MIGRATION_VERSION, target],
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return migration_status(conn)


def begin_storage_migration(
    conn: duckdb.DuckDBPyConnection, config: AppConfig
) -> IndexMigrationStatus:
    """Capture explicit storage intent after backup, without authorizing a reparse."""
    job = _job(conn)
    if job["phase"] != "idle":
        require_original_backup(conn, config)
        phase = "running" if job["captured_count"] else "verifying"
        conn.execute(
            """UPDATE index_migration_jobs SET storage_target = ?, phase = ?, error = NULL
               WHERE singleton""",
            [job["storage_target"] or STORAGE_VERSION, phase],
        )
        return migration_status(conn)
    if not storage_upgrade_needed(storage_version(conn)):
        return migration_status(conn)
    backup = _create_backup(conn, config, kind="storage-migration")
    conn.execute(
        """UPDATE index_migration_jobs SET target_version = ?, phase = 'verifying',
               backup_path = ?, captured_count = 0, completed_count = 0,
               started_at = CURRENT_TIMESTAMP, error = NULL,
               storage_target = ?, storage_attempt = 0 WHERE singleton""",
        [applied_version(conn), str(backup), STORAGE_VERSION],
    )
    return migration_status(conn)


def perform_storage_transition(config: AppConfig) -> None:
    """Convert under exclusive file ownership; the caller closes/reopens shared handles.

    A durable attempt transition creates legitimate maintenance WAL even after
    interruption. A requested ATTACH tag alone is not proof of persisted format.
    The owner must reopen normally and call verify_and_complete afterward.
    """
    with closing(connect_existing(config, read_only=True)) as reader:
        job = _job(reader)
        require_original_backup(reader, config)
        if job["phase"] == "idle" or job["storage_target"] != STORAGE_VERSION:
            raise RuntimeError("no active fixed-format storage maintenance intent")
        if not storage_upgrade_needed(storage_version(reader)):
            return
    with closing(connect_storage_transition(config)) as conn:
        conn.execute(
            """UPDATE index_migration_jobs SET storage_attempt = storage_attempt + 1,
                   error = NULL WHERE singleton"""
        )
        conn.execute("CHECKPOINT maintenance")


def mark_captured_complete(conn: duckdb.DuckDBPyConnection, source_key: str) -> None:
    result = conn.execute(
        """UPDATE index_migration_scope SET completed = TRUE
           WHERE source_key = ? AND completed = FALSE
           RETURNING source_key""",
        [source_key],
    ).fetchone()
    if result is None:
        return
    conn.execute(
        """UPDATE index_migration_jobs
           SET completed_count = (
               SELECT COUNT(*) FROM index_migration_scope WHERE completed
           )
           WHERE singleton"""
    )


def verify_and_complete(conn: duckdb.DuckDBPyConnection, *, verified: bool) -> IndexMigrationStatus:
    job = _job(conn)
    if job["phase"] not in {"running", "verifying", "failed", "backup"}:
        return migration_status(conn)
    if not verified:
        conn.execute(
            """UPDATE index_migration_jobs
               SET phase = 'failed', error = 'captured scope was not verified'
               WHERE singleton"""
        )
        return migration_status(conn)
    target = job["storage_target"]
    if target is not None:
        current = storage_version(conn)
        if current is None or storage_upgrade_needed(current, target):
            raise RuntimeError("persisted storage format has not been verified")
    storage_only = target is not None and job["captured_count"] == 0
    remaining = migration_status(conn).remaining
    if remaining > 0 or (int(job["captured_count"]) <= 0 and not storage_only):
        raise RuntimeError("captured scope is not complete")
    if not _captured_catalog_current(conn):
        raise RuntimeError("captured scope is not current")
    conn.execute("BEGIN")
    try:
        if not storage_only:
            conn.execute(
                "UPDATE runtime_state SET index_migration_version = ? WHERE singleton",
                [INDEX_MIGRATION_VERSION],
            )
            conn.execute("DELETE FROM index_migration_scope")
        conn.execute(
            """UPDATE index_migration_jobs
               SET phase = 'idle', error = NULL, completed_count = captured_count
               WHERE singleton"""
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return migration_status(conn)


def maybe_complete_migration(conn: duckdb.DuckDBPyConnection) -> IndexMigrationStatus:
    status = migration_status(conn)
    if status.phase != "running" or status.remaining > 0 or status.captured <= 0:
        return status
    if not _captured_catalog_current(conn):
        return status
    return verify_and_complete(conn, verified=True)


def rollback_migration(config: AppConfig) -> Path:
    conn = connect_existing(config, read_only=True)
    try:
        row = conn.execute(
            "SELECT backup_path FROM index_migration_jobs WHERE singleton"
        ).fetchone()
    finally:
        conn.close()
    if row is None or not row[0]:
        raise RuntimeError("no index-migration backup to restore")
    recorded = str(row[0])
    with original_backup(config, recorded) as inputs:
        restore_consistent_backup(config, inputs)
    return Path(recorded)


def _job(conn: duckdb.DuckDBPyConnection) -> _JobState:
    row = conn.execute(
        """SELECT phase, backup_path, captured_count, completed_count, started_at, error,
                  target_version, storage_target, storage_attempt
           FROM index_migration_jobs WHERE singleton"""
    ).fetchone()
    if row is None:
        conn.execute(
            """INSERT INTO index_migration_jobs (singleton, target_version, phase)
               VALUES (TRUE, 0, 'idle')"""
        )
        row = conn.execute(
            """SELECT phase, backup_path, captured_count, completed_count, started_at, error,
                      target_version, storage_target, storage_attempt
               FROM index_migration_jobs WHERE singleton"""
        ).fetchone()
    assert row is not None
    allowed: set[str] = {"idle", "backup", "running", "verifying", "failed"}
    phase: Phase = row[0] if row[0] in allowed else "idle"
    return {
        "phase": phase,
        "backup_path": str(row[1]) if row[1] else None,
        "captured_count": int(row[2] or 0),
        "completed_count": int(row[3] or 0),
        "started_at": row[4],
        "error": str(row[5]) if row[5] else None,
        "target_version": int(row[6] or 0),
        "storage_target": str(row[7]) if row[7] else None,
        "storage_attempt": int(row[8] or 0),
    }


def _create_backup(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
    *,
    kind: Literal["index-migration", "storage-migration"],
) -> Path:
    # A newly owned backup must record a canonical local reference even when
    # the selected root is relative or linked. Existing records stay metadata.
    root = (Path.cwd() / config.data_dir).resolve(strict=True)
    backup = consistent_backup(conn, replace(config, data_dir=root), kind=kind)
    with original_backup(config, str(backup)):
        pass
    return backup


def require_original_backup(conn: duckdb.DuckDBPyConnection, config: AppConfig) -> None:
    """Requalify the recorded backup while the caller owns migration admission."""
    with original_backup(config, _job(conn)["backup_path"]):
        pass


def _captured_catalog_current(conn: duckdb.DuckDBPyConnection) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*) FROM index_migration_scope scope
        JOIN source_files files ON files.source_key = scope.source_key
        WHERE NOT (
            files.missing
            OR files.last_error IS NOT NULL
            OR (
                files.desired_generation = files.committed_generation
                AND files.committed_offset = files.size
            )
        )
        """
    ).fetchone()
    return row is not None and int(row[0]) == 0


def _now() -> float:
    return time.time()
