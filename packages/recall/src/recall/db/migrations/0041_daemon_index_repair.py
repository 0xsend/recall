"""Install the recovery state, logical fence and audit in one transaction."""

from __future__ import annotations

from recall.db.lease import DbLease
from recall.db.migrations import AlreadyAudited, Migration, register
from recall.db.repair import FENCE_VERSION, MIGRATION_ID, initialize_repair


def _upgrade(lease: DbLease) -> bool | AlreadyAudited:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = lease.call("0041_daemon_index_repair:_upgrade", lambda conn: get_schema_version(conn))
    # Schema 36 is the oldest layout with all prerequisite migrations.
    if current < 36:
        return False
    lease.call("0041_daemon_index_repair:_upgrade", lambda conn: conn.execute("BEGIN").fetchone())
    try:
        lease.call("0041_daemon_index_repair:_upgrade", lambda conn: initialize_repair(conn))
        if current < FENCE_VERSION:
            lease.call(
                "0041_daemon_index_repair:_upgrade",
                lambda conn: set_schema_version_to(conn, FENCE_VERSION),
            )
        lease.call(
            "0041_daemon_index_repair:_upgrade", lambda conn: conn.execute("COMMIT").fetchone()
        )
    except BaseException:
        lease.cleanup_call(
            "0041_daemon_index_repair:_upgrade", lambda conn: conn.execute("ROLLBACK").fetchone()
        )
        raise
    return AlreadyAudited(MIGRATION_ID, FENCE_VERSION)


register(Migration(MIGRATION_ID, FENCE_VERSION, _upgrade))
