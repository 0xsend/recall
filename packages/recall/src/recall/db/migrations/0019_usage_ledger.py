"""Migration 0019: fleet/usage ledger tables and session columns.

Adds:
- usage_events (Grok unified.jsonl harvest rows)
- usage_log_cursors (byte-offset harvest cursors)
- session_state.cached_input_tokens
- session_state.host (default 'local')

See SPEC Fleet Token / Usage Ledger (REQ-USAGE-016).
"""

from __future__ import annotations

import logging

import duckdb

from recall.db.migrations import Migration, register

logger = logging.getLogger("recall.schema")


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < 18:
        logger.warning(
            "0019_usage_ledger requires schema_version >= 18; DB is at %d. "
            "Halting; earlier migrations must complete first.",
            current,
        )
        return False

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_events (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_session_id TEXT NOT NULL,
                session_id TEXT,
                ts TIMESTAMP,
                prompt_tokens INTEGER,
                cached_prompt_tokens INTEGER,
                completion_tokens INTEGER,
                reasoning_tokens INTEGER,
                host TEXT,
                harvested_at TIMESTAMP NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_usage_events_source_sid
            ON usage_events(source, source_session_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_usage_events_session
            ON usage_events(session_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_usage_events_ts
            ON usage_events(ts)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_log_cursors (
                path TEXT PRIMARY KEY,
                byte_offset BIGINT NOT NULL DEFAULT 0,
                file_size BIGINT NOT NULL DEFAULT 0,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )
        # DuckDB cannot ADD COLUMN with NOT NULL / CHECK constraints; use
        # nullable + default, then backfill so existing rows get 'local'.
        _add_session_state_column(conn, "cached_input_tokens", "INTEGER")
        _add_session_state_column(conn, "host", "TEXT DEFAULT 'local'")
        tables = conn.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = 'session_state'
            """
        ).fetchone()
        if tables and tables[0]:
            conn.execute("UPDATE session_state SET host = 'local' WHERE host IS NULL")
    except Exception:
        logger.exception("Failed to apply 0019_usage_ledger")
        return False

    if current < 19:
        set_schema_version_to(conn, 19)
    return True


def _add_session_state_column(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    ddl_type: str,
) -> None:
    tables = conn.execute(
        """
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_name = 'session_state'
        """
    ).fetchone()
    if not tables or not tables[0]:
        # Minimal simulated DBs in cutover tests omit session_state; real
        # installs always have it from schema.sql. Skip column DDL only.
        logger.info(
            "session_state missing; skipping ADD COLUMN %s (usage tables still applied)",
            name,
        )
        return
    existing = {
        str(row[1]) for row in conn.execute("PRAGMA table_info('session_state')").fetchall()
    }
    if name in existing:
        return
    conn.execute(f"ALTER TABLE session_state ADD COLUMN {name} {ddl_type}")


register(
    Migration(
        id="0019_usage_ledger",
        target_version=19,
        upgrade=_upgrade,
    )
)
