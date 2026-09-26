"""Add storage intent/attempt metadata without changing index eligibility."""

from __future__ import annotations

import duckdb

from recall.db.migrations import Migration, register


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    if get_schema_version(conn) < 27:
        return False
    conn.execute("ALTER TABLE index_migration_jobs ADD COLUMN IF NOT EXISTS storage_target TEXT")
    conn.execute(
        "ALTER TABLE index_migration_jobs ADD COLUMN IF NOT EXISTS storage_attempt BIGINT DEFAULT 0"
    )
    set_schema_version_to(conn, 28)
    return True


register(Migration(id="0028_storage_maintenance", target_version=28, upgrade=_upgrade))
