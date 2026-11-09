"""Add durable duplicate-transfer authority without changing existing rows.

DDL, version and audit commit atomically; replay is a no-op. Rollback restores
both the pre-upgrade DuckDB and keyword-sidecar backup. Earlier releases refuse
schema 38. No undo rows are needed for this additive migration.
"""

from __future__ import annotations

import duckdb

from recall.db.migrations import Migration, register

MIGRATION_ID = "0038_duplicate_transfers"
TARGET_VERSION = 38

_DDL = """
-- One row per unfinished duplicate transfer (REQ-INDEX-027, REQ-NEW-2). The only
-- authority for a transfer: a session pair is in transfer exactly when a row
-- names it. survivor_offset/survivor_prefix_sha256 are S's committed boundary,
-- frozen at rebind, and the only byte witness that authorizes deleting S's rows.
CREATE TABLE IF NOT EXISTS duplicate_transfers (
    survivor_id TEXT PRIMARY KEY,
    predecessor_id TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    original_path TEXT NOT NULL,
    moved_path TEXT NOT NULL,
    survivor_offset BIGINT NOT NULL,
    survivor_prefix_sha256 TEXT NOT NULL,
    survivor_message_count BIGINT NOT NULL,
    survivor_tool_count BIGINT NOT NULL,
    context_message_bound BIGINT NOT NULL,
    survivor_orphan_count BIGINT,
    phase TEXT NOT NULL,
    predecessor_publish_seq BIGINT NOT NULL DEFAULT 0,
    predecessor_refusal_kind TEXT,
    predecessor_refusal_observation JSON,
    witness_epoch BIGINT,
    coverage_proven_at TIMESTAMP,
    predecessor_content_epoch BIGINT,
    stuck_reason TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source, moved_path),
    CHECK (survivor_id <> predecessor_id),
    CHECK (survivor_offset > 0),
    CHECK (predecessor_publish_seq >= 0),
    CHECK ((predecessor_refusal_kind IS NULL)
           = (predecessor_refusal_observation IS NULL)),
    CHECK (predecessor_refusal_observation IS NULL
           OR length(CAST(predecessor_refusal_observation AS VARCHAR)) <= 1024),
    CHECK (coverage_proven_at IS NULL OR predecessor_refusal_kind IS NULL),
    CHECK (length(survivor_prefix_sha256) = 64),
    CHECK (survivor_message_count >= 0 AND survivor_tool_count >= 0),
    CHECK (context_message_bound >= survivor_message_count),
    CHECK ((coverage_proven_at IS NULL) = (predecessor_content_epoch IS NULL)),
    CHECK ((coverage_proven_at IS NULL) = (survivor_orphan_count IS NULL)),
    CHECK (survivor_orphan_count IS NULL
           OR survivor_orphan_count BETWEEN 0 AND survivor_tool_count),
    CHECK (coverage_proven_at IS NULL OR witness_epoch IS NOT NULL),
    CHECK (coverage_proven_at IS NULL OR stuck_reason IS NULL)
);
"""


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < 36:
        return False
    conn.execute("BEGIN")
    try:
        conn.execute(_DDL)
        if current < TARGET_VERSION:
            set_schema_version_to(conn, TARGET_VERSION)
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(migration_id) VALUES (?)",
            [MIGRATION_ID],
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return True


register(Migration(id=MIGRATION_ID, target_version=TARGET_VERSION, upgrade=_upgrade))
