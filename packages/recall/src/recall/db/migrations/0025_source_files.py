"""Migration 0025: durable source inventory and reconciliation scan state."""

from __future__ import annotations

import logging

import duckdb

from recall.db.migrations import Migration, register

logger = logging.getLogger("recall.schema")

MIGRATION_ID = "0025_source_files"

_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS source_files (
        source_key TEXT PRIMARY KEY, source TEXT NOT NULL, source_path TEXT NOT NULL,
        root_path TEXT NOT NULL,
        session_id TEXT, dev BIGINT, inode BIGINT, ctime_ns BIGINT NOT NULL,
        mtime_ns BIGINT NOT NULL, size BIGINT NOT NULL,
        sidecar_mtime_ns BIGINT NOT NULL DEFAULT 0, parser_revision TEXT NOT NULL DEFAULT '',
        sidecar_signature TEXT NOT NULL DEFAULT '',
        desired_generation BIGINT NOT NULL DEFAULT 0,
        committed_generation BIGINT NOT NULL DEFAULT 0,
        committed_offset BIGINT NOT NULL DEFAULT 0, committed_prefix_sha256 TEXT,
        content_epoch BIGINT NOT NULL DEFAULT 0, first_pending_at DOUBLE,
        last_serviced_seq BIGINT NOT NULL DEFAULT 0, retry_count INTEGER NOT NULL DEFAULT 0,
        next_retry_at DOUBLE NOT NULL DEFAULT 0, last_error TEXT, diagnostics JSON,
        missing BOOLEAN NOT NULL DEFAULT FALSE, observed_at DOUBLE NOT NULL,
        inventory_generation BIGINT NOT NULL DEFAULT 0
    )
    """,
    """CREATE INDEX IF NOT EXISTS idx_source_files_pending
       ON source_files(committed_generation, desired_generation, next_retry_at)""",
    "CREATE INDEX IF NOT EXISTS idx_source_files_source_path ON source_files(source, source_path)",
    """
    CREATE TABLE IF NOT EXISTS reconciliation_roots (
        source TEXT NOT NULL, root_path TEXT NOT NULL, scan_started_at DOUBLE,
        scan_finished_at DOUBLE, discovered_count BIGINT NOT NULL DEFAULT 0,
        failure_count BIGINT NOT NULL DEFAULT 0, failures JSON,
        scan_complete BOOLEAN NOT NULL DEFAULT FALSE,
        scan_generation BIGINT NOT NULL DEFAULT 0,
        PRIMARY KEY (source, root_path)
    )
    """,
)


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    if get_schema_version(conn) < 24:
        return False
    try:
        for statement in _STATEMENTS:
            conn.execute(statement)
        set_schema_version_to(conn, 25)
    except Exception:
        logger.exception("Failed to apply %s", MIGRATION_ID)
        return False
    return True


register(Migration(id=MIGRATION_ID, target_version=25, upgrade=_upgrade))
