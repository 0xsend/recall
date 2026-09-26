"""Migration 0027: durable index-migration version and in-flight job state.

Additive only. Existing databases keep applied version 0 so they are eligible
for versioned historical index migration (REQ-RECON-010). Fresh schema.sql
creates the same tables; this upgrade is idempotent.
"""

from __future__ import annotations

import logging

import duckdb

from recall.db.migrations import Migration, register, table_exists

logger = logging.getLogger("recall.schema")

MIGRATION_ID = "0027_index_migration_state"


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < 26:
        logger.warning(
            "%s requires schema_version >= 26; DB is at %d.",
            MIGRATION_ID,
            current,
        )
        return False

    try:
        if table_exists(conn, "runtime_state"):
            conn.execute(
                "ALTER TABLE runtime_state ADD COLUMN IF NOT EXISTS "
                "index_migration_version INTEGER DEFAULT 0"
            )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS index_migration_jobs (
                singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
                target_version INTEGER NOT NULL,
                phase TEXT NOT NULL CHECK (
                    phase IN ('idle', 'backup', 'running', 'verifying', 'failed')
                ),
                backup_path TEXT,
                captured_count BIGINT NOT NULL DEFAULT 0,
                completed_count BIGINT NOT NULL DEFAULT 0,
                started_at TIMESTAMP,
                error TEXT
            )
            """
        )
        conn.execute(
            """INSERT OR IGNORE INTO index_migration_jobs (singleton, target_version, phase)
               VALUES (TRUE, 0, 'idle')"""
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS index_migration_scope (
                source_key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_path TEXT NOT NULL,
                completed BOOLEAN NOT NULL DEFAULT FALSE
            )
            """
        )
        set_schema_version_to(conn, 27)
    except Exception:
        logger.exception("Failed to apply %s", MIGRATION_ID)
        return False
    return True


register(Migration(id=MIGRATION_ID, target_version=27, upgrade=_upgrade))
