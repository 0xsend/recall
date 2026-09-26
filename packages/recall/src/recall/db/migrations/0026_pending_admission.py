"""Migration 0026: retain fair admission independently of completed service."""

from __future__ import annotations

import duckdb

from recall.db.migrations import Migration, register


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    if get_schema_version(conn) < 25:
        return False
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(
            "ALTER TABLE source_files ADD COLUMN IF NOT EXISTS first_pending_seq BIGINT DEFAULT 0"
        )
        # DuckDB requires secondary indexes removed while adding a constraint.
        conn.execute("DROP INDEX IF EXISTS idx_source_files_pending")
        conn.execute("DROP INDEX IF EXISTS idx_source_files_source_path")
        conn.execute("ALTER TABLE source_files ALTER COLUMN first_pending_seq SET NOT NULL")
        conn.execute(
            "CREATE INDEX idx_source_files_pending "
            "ON source_files(committed_generation, desired_generation, next_retry_at)"
        )
        conn.execute(
            "CREATE INDEX idx_source_files_source_path ON source_files(source, source_path)"
        )
        set_schema_version_to(conn, 26)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    return True


register(Migration(id="0026_pending_admission", target_version=26, upgrade=_upgrade))
