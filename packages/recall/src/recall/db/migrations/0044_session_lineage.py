"""Add subagent lineage to session identity rows (REQ-SCHEMA-007).

Adds four nullable columns to ``sessions`` and backfills Claude Code subagent
transcripts from their stored paths
(``<project>/<sid>/subagents/agent-<id>.jsonl``). Agent type and description are
not in the path; they arrive when each subagent re-indexes for its newly
declared meta sidecar. Codex lineage arrives with each rollout's next full parse.

DDL, backfill, schema version and audit commit together inside one observed
lease transaction after verifying schema 43 and the schema-41 recovery fence.
No transcript is read. Rollback restores the paired pre-upgrade backup and old
binary; an old binary ignores the added columns.
"""

from __future__ import annotations

import duckdb

from recall.db.lease import DbLease
from recall.db.migrations import (
    AlreadyAudited,
    Migration,
    record_migration_audit,
    register,
    table_exists,
)
from recall.db.repair import require_schema_fence

MIGRATION_ID = "0044_session_lineage"
REQUIRED_VERSION = 43
TARGET_VERSION = 44

LINEAGE_COLUMNS = ("root_source_session_id", "agent_id", "agent_type", "agent_description")

# Groups: the parent's session id directory, then the agent id.
_CLAUDE_SUBAGENT_PATH = r"/([^/]+)/subagents/agent-([^/]+)\.jsonl$"


def _add_columns(conn: duckdb.DuckDBPyConnection) -> None:
    for column in LINEAGE_COLUMNS:
        conn.execute(f"ALTER TABLE sessions ADD COLUMN IF NOT EXISTS {column} TEXT")


def _backfill_claude_subagents(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        UPDATE sessions
        SET root_source_session_id = regexp_extract(source_path, ?, 1),
            agent_id = regexp_extract(source_path, ?, 2)
        WHERE source = 'claude_code'
          AND root_source_session_id IS NULL
          AND regexp_matches(source_path, ?)
        """,
        [_CLAUDE_SUBAGENT_PATH] * 3,
    )


def _upgrade(lease: DbLease) -> bool | AlreadyAudited:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = lease.call(f"{MIGRATION_ID}:_upgrade", lambda conn: get_schema_version(conn))
    if current < REQUIRED_VERSION:
        return False
    try:
        lease.call(f"{MIGRATION_ID}:fence", require_schema_fence)
    except (ValueError, duckdb.CatalogException):
        return False

    with lease.transaction("migration:0044") as transaction:
        if transaction.call("lineage:sessions_check", lambda conn: table_exists(conn, "sessions")):
            transaction.call("lineage:columns", _add_columns)
            transaction.call("lineage:backfill", _backfill_claude_subagents)
        tx_version = transaction.call("lineage:tx_version", get_schema_version)
        if tx_version < TARGET_VERSION:
            transaction.call(
                "lineage:version", lambda conn: set_schema_version_to(conn, TARGET_VERSION)
            )
        transaction.call("lineage:audit", lambda conn: record_migration_audit(conn, MIGRATION_ID))
    return AlreadyAudited(MIGRATION_ID, TARGET_VERSION)


register(Migration(id=MIGRATION_ID, target_version=TARGET_VERSION, upgrade=_upgrade))
