"""Add exact Claude response accounting; historical totals initialize on reparse.

DDL, allocator, schema version and audit commit together. No existing totals
are changed. Rollback restores the paired pre-upgrade backup and old binary;
there is no supported in-place downgrade after scopes have been populated.
"""

from __future__ import annotations

import duckdb

from recall.db.migrations import Migration, register, table_exists

MIGRATION_ID = "0039_claude_response_usage"
TARGET_VERSION = 39


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < 36:
        return False
    conn.execute("BEGIN")
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS claude_usage_scopes (
    session_id TEXT PRIMARY KEY,
    scope_token TEXT NOT NULL,
    format_revision INTEGER NOT NULL,
    committed_offset BIGINT NOT NULL,
    prefix_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS claude_response_usage (
    response_key TEXT PRIMARY KEY,
    scope_token TEXT NOT NULL,
    input_tokens BIGINT,
    cache_creation_input_tokens BIGINT,
    cache_read_input_tokens BIGINT,
    output_tokens BIGINT
);
""")
        if table_exists(conn, "runtime_state"):
            conn.execute(
                "ALTER TABLE runtime_state ADD COLUMN IF NOT EXISTS "
                "claude_usage_incarnation BIGINT DEFAULT 0"
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
