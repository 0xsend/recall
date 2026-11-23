"""Add exact Claude response accounting; historical totals initialize on reparse.

DDL, allocator, schema version and audit commit together inside one observed
lease transaction after verifying the schema-41 recovery fence. No existing
totals are changed. Rollback restores the paired pre-upgrade backup and old
binary; there is no supported in-place downgrade after scopes have been
populated.
"""

from __future__ import annotations

import duckdb

from recall.db.lease import DbLease
from recall.db.migrations import AlreadyAudited, Migration, register, table_exists
from recall.db.repair import FENCE_VERSION, require_schema_fence

MIGRATION_ID = "0042_claude_response_usage"
TARGET_VERSION = 42

_CREATE_CLAUDE_USAGE_TABLES = """CREATE TABLE IF NOT EXISTS claude_usage_scopes (
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
);"""


def _upgrade(lease: DbLease) -> bool | AlreadyAudited:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = lease.call(
        "0042_claude_response_usage:_upgrade", lambda conn: get_schema_version(conn)
    )
    if current < FENCE_VERSION:
        return False
    try:
        lease.call("0042_claude_response_usage:fence", require_schema_fence)
    except (ValueError, duckdb.CatalogException):
        return False
    with lease.transaction("migration:0042") as transaction:
        transaction.call(
            "claude_usage:tables",
            lambda conn: conn.execute(_CREATE_CLAUDE_USAGE_TABLES).fetchone(),
        )
        has_runtime = transaction.call(
            "claude_usage:runtime_check", lambda conn: table_exists(conn, "runtime_state")
        )
        if has_runtime:
            transaction.call(
                "claude_usage:incarnation",
                lambda conn: conn.execute(
                    "ALTER TABLE runtime_state ADD COLUMN IF NOT EXISTS "
                    "claude_usage_incarnation BIGINT DEFAULT 0"
                ).fetchone(),
            )
        if current < TARGET_VERSION:
            transaction.call(
                "claude_usage:version", lambda conn: set_schema_version_to(conn, TARGET_VERSION)
            )
        transaction.call(
            "claude_usage:audit",
            lambda conn: conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (migration_id) VALUES (?)",
                [MIGRATION_ID],
            ).fetchone(),
        )
    return AlreadyAudited(MIGRATION_ID, TARGET_VERSION)


register(Migration(id=MIGRATION_ID, target_version=TARGET_VERSION, upgrade=_upgrade))
