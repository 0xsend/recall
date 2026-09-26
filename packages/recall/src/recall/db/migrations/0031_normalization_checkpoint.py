"""Migration 0031: durable normalization checkpoint on the source catalog.

Additive only. `source_files.normalization_checkpoint` is nullable with no
default, so every row acknowledged before this upgrade reads NULL -- the one
value meaning "no resume proof exists" -- and the next acknowledgement is what
writes one (REQ-INDEX-025). No existing row or companion state is read, changed
or deleted, so REQ-MIG-008 needs no pre-image.

DuckDB refuses `ALTER TABLE ... DROP COLUMN` while a secondary index is present
but accepts `ADD COLUMN`, so `idx_source_files_source_path` is left in place.

Reverse statement (downgrades are a non-goal; recorded for recovery):

    DROP INDEX IF EXISTS idx_source_files_source_path;
    ALTER TABLE source_files DROP COLUMN normalization_checkpoint;
    CREATE INDEX idx_source_files_source_path ON source_files(source, source_path);
"""

from __future__ import annotations

import logging

import duckdb

from recall.db.migrations import Migration, register, table_exists

logger = logging.getLogger("recall.schema")

MIGRATION_ID = "0031_normalization_checkpoint"
TARGET_VERSION = 31


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < TARGET_VERSION - 1:
        logger.warning(
            "%s requires schema_version >= %d; DB is at %d. "
            "Halting; earlier migrations must complete first.",
            MIGRATION_ID,
            TARGET_VERSION - 1,
            current,
        )
        return False

    if not table_exists(conn, "source_files"):
        logger.info("%s: source_files is absent; nothing to extend", MIGRATION_ID)
        if current < TARGET_VERSION:
            set_schema_version_to(conn, TARGET_VERSION)
        return True

    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(
            "ALTER TABLE source_files ADD COLUMN IF NOT EXISTS normalization_checkpoint TEXT"
        )
        # Replaying an applied migration must not insert a duplicate version row.
        if current < TARGET_VERSION:
            set_schema_version_to(conn, TARGET_VERSION)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        logger.exception("Failed to apply %s", MIGRATION_ID)
        return False
    return True


register(Migration(id=MIGRATION_ID, target_version=TARGET_VERSION, upgrade=_upgrade))
