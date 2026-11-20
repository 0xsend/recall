"""Add storage intent/attempt metadata without changing index eligibility."""

from __future__ import annotations

from recall.db.lease import DbLease
from recall.db.migrations import Migration, register


def _upgrade(lease: DbLease) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    if lease.call("0028_storage_maintenance:_upgrade", lambda conn: get_schema_version(conn)) < 27:
        return False
    lease.call(
        "0028_storage_maintenance:_upgrade",
        lambda conn: conn.execute(
            "ALTER TABLE index_migration_jobs ADD COLUMN IF NOT EXISTS storage_target TEXT"
        ).fetchone(),
    )
    lease.call(
        "0028_storage_maintenance:_upgrade",
        lambda conn: conn.execute(
            "ALTER TABLE index_migration_jobs ADD COLUMN "
            "IF NOT EXISTS storage_attempt BIGINT DEFAULT 0"
        ).fetchone(),
    )
    lease.call("0028_storage_maintenance:_upgrade", lambda conn: set_schema_version_to(conn, 28))
    return True


register(Migration(id="0028_storage_maintenance", target_version=28, upgrade=_upgrade))
