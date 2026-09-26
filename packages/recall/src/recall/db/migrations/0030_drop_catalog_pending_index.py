"""Migration 0030: retire the unused catalog pending index and generation column.

`idx_source_files_pending` indexed `(committed_generation, desired_generation,
next_retry_at)`, but the pending scan filters on column-to-column comparisons
(`desired_generation > committed_generation OR committed_offset < size`), which
a DuckDB ART index cannot serve. It only cost write amplification on every
catalog update. `source_files.inventory_generation` was created by 0025 and is
neither read nor written by the current reconciliation path.

Shape-only: no logical `source_files` row or companion state is deleted or changed,
so REQ-MIG-008 needs no pre-image. DuckDB refuses `ALTER TABLE ... DROP COLUMN`
while any secondary index is present, so `idx_source_files_source_path` is
dropped and recreated inside the same transaction; a failure rolls the whole
statement set back, restoring both indexes and leaving the version at 29.

Reverse statement (downgrades are a non-goal; recorded for recovery):

    DROP INDEX IF EXISTS idx_source_files_source_path;
    ALTER TABLE source_files ADD COLUMN inventory_generation BIGINT DEFAULT 0;
    ALTER TABLE source_files ALTER COLUMN inventory_generation SET NOT NULL;
    CREATE INDEX idx_source_files_source_path ON source_files(source, source_path);
    CREATE INDEX idx_source_files_pending
        ON source_files(committed_generation, desired_generation, next_retry_at);
"""

from __future__ import annotations

import logging

import duckdb

from recall.db.migrations import Migration, register, table_exists

logger = logging.getLogger("recall.schema")

MIGRATION_ID = "0030_drop_catalog_pending_index"
TARGET_VERSION = 30


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
        logger.info("%s: source_files is absent; nothing to drop", MIGRATION_ID)
        if current < TARGET_VERSION:
            set_schema_version_to(conn, TARGET_VERSION)
        return True

    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute("DROP INDEX IF EXISTS idx_source_files_pending")
        conn.execute("DROP INDEX IF EXISTS idx_source_files_source_path")
        conn.execute("ALTER TABLE source_files DROP COLUMN IF EXISTS inventory_generation")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_files_source_path "
            "ON source_files(source, source_path)"
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
