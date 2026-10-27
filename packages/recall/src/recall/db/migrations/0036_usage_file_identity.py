"""Persist usage-log descriptor identity without changing cursor or event rows.

Legacy cursors gain a NULL device/inode pair and replay once on harvest.
DDL, version and migration audit commit together; interruption retries safely.
Rollback restores the paired pre-upgrade database and keyword-sidecar backup.
Downgrades are not automatic.
"""

from __future__ import annotations

import duckdb

from recall.db.migrations import Migration, register, table_exists

MIGRATION_ID = "0036_usage_file_identity"
TARGET_VERSION = 36


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < TARGET_VERSION - 1:
        return False
    conn.execute("BEGIN TRANSACTION")
    try:
        if table_exists(conn, "usage_log_cursors"):
            conn.execute(
                "ALTER TABLE usage_log_cursors ADD COLUMN IF NOT EXISTS file_device UBIGINT"
            )
            conn.execute(
                "ALTER TABLE usage_log_cursors ADD COLUMN IF NOT EXISTS file_inode UBIGINT"
            )
        if current < TARGET_VERSION:
            set_schema_version_to(conn, TARGET_VERSION)
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (migration_id) VALUES (?)", [MIGRATION_ID]
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return True


register(Migration(id=MIGRATION_ID, target_version=TARGET_VERSION, upgrade=_upgrade))
