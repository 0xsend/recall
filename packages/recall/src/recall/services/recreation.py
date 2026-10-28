"""Consistent pre-recreate backups and transactional schema replacement."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import duckdb

from recall.core.config import AppConfig
from recall.core.schema_version import SchemaTooNewError
from recall.db import sidecar_path
from recall.db.connection import _open_database, advisory_lock, check_database_schema
from recall.db.ownership import offline_ownership
from recall.db.schema import _apply_schema, _set_schema_version, _store_embedding_dimensions
from recall.services.compaction import _clone_or_copy


class OfflineRecreationError(RuntimeError):
    """An offline reset failed; any completed recovery snapshot remains usable."""

    def __init__(self, message: str, backup_path: Path | None) -> None:
        self.backup_path = backup_path
        if backup_path is not None:
            message = f"recreate failed; backup: {backup_path}; {message}"
        super().__init__(message)


def recreate_newer_database(config: AppConfig) -> Path | None:
    """Consume one confirmed offline reset; compatible inputs retain RPC handling."""
    from recall.services.coordinator import ReconciliationPausedError, require_reconciliation

    try:
        check_database_schema(config)
    except SchemaTooNewError:
        pass
    else:
        return None

    backup: Path | None = None

    def reset(conn: duckdb.DuckDBPyConnection) -> None:
        nonlocal backup
        require_reconciliation(config)
        backup = backup_before_recreate(conn, config)
        require_reconciliation(config)
        replace_schema(conn, config)

    try:
        with offline_ownership(config.db_path), advisory_lock(config.lock_path):
            require_reconciliation(config)
            with closing(_open_database(config, on_newer_schema=reset)):
                pass
    except ReconciliationPausedError:
        raise
    except (duckdb.Error, OSError, RuntimeError) as err:
        raise OfflineRecreationError(str(err), backup) from err
    return backup


def consistent_backup(conn: duckdb.DuckDBPyConnection, config: AppConfig, *, kind: str) -> Path:
    """Copy a checkpointed database and consistent sidecar before destructive work.

    The caller owns the writer and keeps the shared handle open. Concurrent
    read cursors may continue: checkpoint plus the writer lock freezes both
    stores, and SQLite's backup API produces its consistent snapshot.
    An incomplete backup is retained for diagnosis and never authorizes mutation.
    """
    # DuckDB 1.5.5 may reject plain CHECKPOINT while a read cursor retains an
    # older transaction. FORCE waits for that cursor; it does not abort it.
    # The storage ownership tests preserve a real reader across this boundary.
    conn.execute("FORCE CHECKPOINT")
    backup = config.data_dir / "snapshots" / f"{kind}-{uuid4().hex}"
    backup.mkdir(parents=True, mode=0o700)
    database = backup / "recall.duckdb"
    _clone_or_copy(config.db_path, database)
    copied = [database]
    sidecar = sidecar_path(config.data_dir)
    if sidecar.is_file():
        destination = backup / "recall.fts.sqlite"
        with (
            closing(sqlite3.connect(sidecar)) as source,
            closing(sqlite3.connect(destination)) as target,
        ):
            source.backup(target)
        copied.append(destination)
    for path in copied:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    manifest = backup / "manifest.json"
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "kind": kind,
                "complete": True,
                "created_at": datetime.now(UTC).isoformat(),
                "files": [path.name for path in copied],
            },
            handle,
        )
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(backup, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return backup


def backup_before_recreate(conn: duckdb.DuckDBPyConnection, config: AppConfig) -> Path:
    """Copy a checkpointed database and consistent sidecar before destructive work."""
    return consistent_backup(conn, config, kind="recreate")


def restore_consistent_backup(config: AppConfig, backup: Path) -> Path:
    """Replace the live database and sidecar from a complete consistent backup."""
    manifest_path = backup / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"backup manifest missing: {backup}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError(f"incomplete backup cannot be restored: {backup}")
    database = backup / "recall.duckdb"
    if not database.is_file():
        raise RuntimeError(f"backup is missing recall.duckdb: {backup}")
    wal = Path(str(config.db_path) + ".wal")
    wal.unlink(missing_ok=True)
    _clone_or_copy(database, config.db_path)
    sidecar_source = backup / "recall.fts.sqlite"
    sidecar = sidecar_path(config.data_dir)
    if sidecar_source.is_file():
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        _clone_or_copy(sidecar_source, sidecar)
    return backup


def replace_schema(conn: duckdb.DuckDBPyConnection, config: AppConfig) -> None:
    """Replace main tables atomically, preserving the backup on every failure."""
    conn.execute("BEGIN")
    try:
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        for (table,) in tables:
            quoted = str(table).replace('"', '""')
            conn.execute(f'DROP TABLE IF EXISTS "{quoted}" CASCADE')
        _apply_schema(conn, config.embedding.dimensions)
        _set_schema_version(conn)
        _store_embedding_dimensions(conn, config.embedding.dimensions)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    sidecar = sidecar_path(config.data_dir)
    if sidecar.is_file():
        with closing(sqlite3.connect(sidecar)) as connection, connection:
            for table in (
                "message_fts",
                "message_fts_rowid",
                "tool_calls_fts",
                "tool_calls_fts_rowid",
            ):
                connection.execute(f"DELETE FROM {table}")
    conn.execute("CHECKPOINT")
