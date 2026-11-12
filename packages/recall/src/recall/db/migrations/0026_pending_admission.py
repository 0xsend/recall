"""Migration 0026: retain fair admission independently of completed service."""

from __future__ import annotations

from recall.db.lease import DbLease
from recall.db.migrations import Migration, register


def _upgrade(lease: DbLease) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    if lease.call("0026_pending_admission:_upgrade", lambda conn: get_schema_version(conn)) < 25:
        return False
    lease.call(
        "0026_pending_admission:_upgrade", lambda conn: conn.execute("BEGIN TRANSACTION").fetchone()
    )
    try:
        lease.call(
            "0026_pending_admission:_upgrade",
            lambda conn: conn.execute(
                "ALTER TABLE source_files ADD COLUMN IF "
                "NOT EXISTS first_pending_seq BIGINT DEFAULT 0"
            ).fetchone(),
        )
        # DuckDB requires secondary indexes removed while adding a constraint.
        lease.call(
            "0026_pending_admission:_upgrade",
            lambda conn: conn.execute("DROP INDEX IF EXISTS idx_source_files_pending").fetchone(),
        )
        lease.call(
            "0026_pending_admission:_upgrade",
            lambda conn: conn.execute(
                "DROP INDEX IF EXISTS idx_source_files_source_path"
            ).fetchone(),
        )
        lease.call(
            "0026_pending_admission:_upgrade",
            lambda conn: conn.execute(
                "ALTER TABLE source_files ALTER COLUMN first_pending_seq SET NOT NULL"
            ).fetchone(),
        )
        lease.call(
            "0026_pending_admission:_upgrade",
            lambda conn: conn.execute(
                "CREATE INDEX idx_source_files_pending "
                "ON source_files(committed_generation, desired_generation, next_retry_at)"
            ).fetchone(),
        )
        lease.call(
            "0026_pending_admission:_upgrade",
            lambda conn: conn.execute(
                "CREATE INDEX idx_source_files_source_path ON source_files(source, source_path)"
            ).fetchone(),
        )
        lease.call("0026_pending_admission:_upgrade", lambda conn: set_schema_version_to(conn, 26))
        lease.call(
            "0026_pending_admission:_upgrade", lambda conn: conn.execute("COMMIT").fetchone()
        )
    except Exception:
        lease.cleanup_call(
            "0026_pending_admission:_upgrade", lambda conn: conn.execute("ROLLBACK").fetchone()
        )
        raise

    return True


register(Migration(id="0026_pending_admission", target_version=26, upgrade=_upgrade))
