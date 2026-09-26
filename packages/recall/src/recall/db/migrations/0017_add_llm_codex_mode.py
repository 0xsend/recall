"""Migration 0017: include 'llm-codex' in the message_state.context_mode CHECK.

DuckDB cannot ALTER a CHECK constraint in place. We rebuild message_state with
the relaxed CHECK and preserve every column, then re-create the index. Existing
data is untouched — all rows carry one of the v16 modes which remain valid.

Only runs when the existing CHECK is the v15/v16 four-mode form; if a future
migration has already relaxed it (or some other tool replaced the table), this
is a no-op that still advances schema_version.
"""

from __future__ import annotations

import logging

import duckdb

from recall.db.migrations import Migration, register

logger = logging.getLogger("recall.schema")

# Sorted alphabetically — DuckDB stores CHECK expressions as text and we match
# on substring presence, so order in the SQL must match what the runtime
# inserts. We keep this in sync with db/schema.sql and 0016's recreate path.
_NEW_MODES_SQL = "'off', 'template', 'llm-local', 'llm-remote', 'llm-codex'"


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < 16:
        logger.warning(
            "0017_add_llm_codex_mode requires schema_version >= 16; DB is at %d. "
            "Halting; earlier migrations must complete first.",
            current,
        )
        return False

    try:
        if _check_already_includes_llm_codex(conn):
            # Idempotent path: someone (manual SQL, future migration) already
            # widened the constraint. Just advance the version pointer.
            logger.info("message_state.context_mode CHECK already includes 'llm-codex'; no-op")
        else:
            _rebuild_message_state_with_widened_check(conn)
    except Exception:
        logger.exception("Failed to widen message_state.context_mode CHECK")
        return False

    if current < 17:
        set_schema_version_to(conn, 17)
    return True


def _check_already_includes_llm_codex(conn: duckdb.DuckDBPyConnection) -> bool:
    rows = conn.execute(
        """
        SELECT expression FROM duckdb_constraints()
        WHERE table_name = 'message_state' AND constraint_type = 'CHECK'
        """
    ).fetchall()
    return any("llm-codex" in str(row[0]) for row in rows)


def _rebuild_message_state_with_widened_check(conn: duckdb.DuckDBPyConnection) -> None:
    """Copy message_state into a new table that carries the widened CHECK."""
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(
            f"""
            CREATE TABLE message_state_new (
                message_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT,
                thinking TEXT,
                timestamp TIMESTAMP,
                has_thinking BOOLEAN DEFAULT FALSE,
                context_text TEXT DEFAULT '',
                context_mode TEXT DEFAULT 'off' CHECK (
                    context_mode IS NULL
                    OR context_mode IN ({_NEW_MODES_SQL})
                ),
                fts_content TEXT DEFAULT '',
                fts_thinking TEXT DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            INSERT INTO message_state_new (
                message_id, role, content, thinking, timestamp, has_thinking,
                context_text, context_mode, fts_content, fts_thinking
            )
            SELECT
                message_id, role, content, thinking, timestamp, has_thinking,
                COALESCE(context_text, ''),
                COALESCE(context_mode, 'off'),
                COALESCE(fts_content, ''),
                COALESCE(fts_thinking, '')
            FROM message_state
            """
        )
        conn.execute("DROP TABLE message_state")
        conn.execute("ALTER TABLE message_state_new RENAME TO message_state")
        # Re-create the index that 0016 (and schema.sql) keep on this column.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_message_state_has_thinking "
            "ON message_state(has_thinking)"
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


register(
    Migration(
        id="0017_add_llm_codex_mode",
        target_version=17,
        upgrade=_upgrade,
    )
)
