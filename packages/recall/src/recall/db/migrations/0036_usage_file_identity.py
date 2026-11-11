"""Persist usage-log descriptor identity without changing cursor or event rows.

Legacy cursors gain a NULL device/inode pair and replay once on harvest.
DDL, version and migration audit commit together; interruption retries safely.
Rollback restores the paired pre-upgrade database and keyword-sidecar backup.
Downgrades are not automatic.
"""

from __future__ import annotations

from recall.db.lease import DbLease
from recall.db.migrations import Migration, register, table_exists

MIGRATION_ID = "0036_usage_file_identity"
TARGET_VERSION = 36


def _upgrade(lease: DbLease) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = lease.call("0036_usage_file_identity:_upgrade", lambda conn: get_schema_version(conn))
    if current < TARGET_VERSION - 1:
        return False
    lease.call(
        "0036_usage_file_identity:_upgrade",
        lambda conn: conn.execute("BEGIN TRANSACTION").fetchone(),
    )
    try:
        if lease.call(
            "0036_usage_file_identity:_upgrade",
            lambda conn: table_exists(conn, "usage_log_cursors"),
        ):
            lease.call(
                "0036_usage_file_identity:_upgrade",
                lambda conn: conn.execute(
                    "ALTER TABLE usage_log_cursors ADD COLUMN IF NOT EXISTS file_device UBIGINT"
                ).fetchone(),
            )
            lease.call(
                "0036_usage_file_identity:_upgrade",
                lambda conn: conn.execute(
                    "ALTER TABLE usage_log_cursors ADD COLUMN IF NOT EXISTS file_inode UBIGINT"
                ).fetchone(),
            )
        if current < TARGET_VERSION:
            lease.call(
                "0036_usage_file_identity:_upgrade",
                lambda conn: set_schema_version_to(conn, TARGET_VERSION),
            )
        lease.call(
            "0036_usage_file_identity:_upgrade",
            lambda conn: conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (migration_id) VALUES (?)", [MIGRATION_ID]
            ).fetchone(),
        )
        lease.call(
            "0036_usage_file_identity:_upgrade", lambda conn: conn.execute("COMMIT").fetchone()
        )
    except BaseException:
        lease.cleanup_call(
            "0036_usage_file_identity:_upgrade", lambda conn: conn.execute("ROLLBACK").fetchone()
        )
        raise
    return True


register(Migration(id=MIGRATION_ID, target_version=TARGET_VERSION, upgrade=_upgrade))
