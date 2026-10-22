"""Migration 0032: consumed-tail identity for incremental usage-log cursors.

Additive, nullable, and metadata-only: existing cursors retain their offsets
and gain NULL identity. Their next harvest replays once with idempotent event
keys. No row data is mutated, so REQ-MIG-008 needs no pre-image.

Rollback requires restoring the pre-upgrade database and sidecar backup:
v0.39.1 sidecar startup checks the schema strictly even though the primary
connection opens leniently. Downgrades are not automatic.

Reverse statement (for recovery, not an automatic downgrade):

    ALTER TABLE usage_log_cursors DROP COLUMN consumed_tail_sha256;
"""

from __future__ import annotations

import logging

import duckdb

from recall.db.migrations import Migration, register, table_exists

logger = logging.getLogger("recall.schema")
MIGRATION_ID = "0032_usage_cursor_identity"
TARGET_VERSION = 32


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < TARGET_VERSION - 1:
        logger.warning("%s requires schema version 31; found %d", MIGRATION_ID, current)
        return False

    conn.execute("BEGIN TRANSACTION")
    try:
        if table_exists(conn, "usage_log_cursors"):
            conn.execute(
                "ALTER TABLE usage_log_cursors ADD COLUMN IF NOT EXISTS consumed_tail_sha256 TEXT"
            )
        if current < TARGET_VERSION:
            set_schema_version_to(conn, TARGET_VERSION)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        logger.exception("Failed to apply %s", MIGRATION_ID)
        return False
    return True


register(Migration(id=MIGRATION_ID, target_version=TARGET_VERSION, upgrade=_upgrade))
