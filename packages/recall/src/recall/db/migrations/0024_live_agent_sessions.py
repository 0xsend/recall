"""Migration 0024: tool results, harness tool-use ids, stop markers, live marks.

Three additive tables for live agent sessions. Nothing existing is rewritten:
`tool_calls` deliberately gains no column, because `_upsert_tool_calls` issues
an `UPDATE` on any difference between the stored row and the freshly parsed
one — a new column would be NULL on every historical row and non-NULL after a
re-parse, so one full re-parse would rewrite the whole table. A DuckDB UPDATE
is DELETE+INSERT, which is exactly the churn that produced the 140 GiB
`tool_call_embeddings` bloat (REQ-INDEX-017, REQ-LIVE-006).

Historical rows simply carry no mapping and no result, so turn state derives
`unknown` for them — what REQ-LIVE-006 already prescribes.
"""

from __future__ import annotations

import logging

import duckdb

from recall.db.migrations import Migration, register

logger = logging.getLogger("recall.schema")

MIGRATION_ID = "0024_live_agent_sessions"

_TABLES: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS tool_results (
        tool_call_id TEXT PRIMARY KEY,
        result_summary TEXT,
        is_error BOOLEAN,
        completed_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_use_ids (
        tool_call_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        tool_use_id TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_stop_markers (
        session_id TEXT NOT NULL,
        message_idx INTEGER NOT NULL,
        reason TEXT NOT NULL,
        ends_turn BOOLEAN NOT NULL,
        PRIMARY KEY (session_id, message_idx)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS live_marks (
        source TEXT NOT NULL,
        source_session_id TEXT NOT NULL,
        host TEXT NOT NULL,
        pid BIGINT,
        surface_key TEXT,
        marked_at TIMESTAMP,
        PRIMARY KEY (source, source_session_id, host)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tool_use_ids_lookup ON tool_use_ids(session_id, tool_use_id)",
)


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < 23:
        logger.warning(
            "%s requires schema_version >= 23; DB is at %d. "
            "Halting; earlier migrations must complete first.",
            MIGRATION_ID,
            current,
        )
        return False

    try:
        for statement in _TABLES:
            conn.execute(statement)
        set_schema_version_to(conn, 24)
    except Exception:
        logger.exception("Failed to apply %s", MIGRATION_ID)
        return False
    return True


register(
    Migration(
        id=MIGRATION_ID,
        target_version=24,
        upgrade=_upgrade,
    )
)
