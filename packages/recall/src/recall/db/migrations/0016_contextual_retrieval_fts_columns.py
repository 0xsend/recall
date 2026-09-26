"""Migration 0016: add materialized contextual FTS text columns."""

from __future__ import annotations

import logging

import duckdb

from recall.db.migrations import Migration, register

logger = logging.getLogger("recall.schema")

ColumnInfo = tuple[str, bool, str | None]


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    """Apply the v15->v16 additive FTS column change."""
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < 15:
        logger.warning(
            "0016_contextual_retrieval_fts_columns requires schema_version >= 15; "
            "DB is at %d. Halting; earlier migrations must complete first.",
            current,
        )
        return False

    try:
        _ensure_message_state_fts_columns(conn)
        _backfill_message_state_fts_columns(conn)
    except Exception:
        logger.exception("Failed to add Contextual Retrieval FTS columns")
        return False

    if current < 16:
        set_schema_version_to(conn, 16)
    return True


def _ensure_message_state_fts_columns(conn: duckdb.DuckDBPyConnection) -> None:
    columns = _column_info(conn, "message_state")
    for name in ("fts_content", "fts_thinking"):
        info = columns.get(name)
        if info is None:
            conn.execute(f"ALTER TABLE message_state ADD COLUMN {name} TEXT DEFAULT ''")
            continue
        if not _is_text_column(info):
            logger.info(
                "message_state.%s has non-text shape %s; recreating message_state",
                name,
                info,
            )
            _recreate_message_state_with_fts_columns(conn)
            return
        if not _default_contains(info[2], "''"):
            conn.execute(f"ALTER TABLE message_state ALTER COLUMN {name} SET DEFAULT ''")


def _backfill_message_state_fts_columns(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        UPDATE message_state
        SET
            fts_content = COALESCE(context_text, '') || COALESCE(content, ''),
            fts_thinking = COALESCE(context_text, '') || COALESCE(thinking, '')
        """
    )


def _recreate_message_state_with_fts_columns(conn: duckdb.DuckDBPyConnection) -> None:
    columns = _column_names(conn, "message_state")
    context_text_expr = "COALESCE(context_text, '')" if "context_text" in columns else "''"
    context_mode_expr = "COALESCE(context_mode, 'off')" if "context_mode" in columns else "'off'"

    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(
            """
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
                    OR context_mode IN ('off', 'template', 'llm-local', 'llm-remote')
                ),
                fts_content TEXT DEFAULT '',
                fts_thinking TEXT DEFAULT ''
            )
            """
        )
        conn.execute(
            f"""
            INSERT INTO message_state_new (
                message_id, role, content, thinking, timestamp, has_thinking,
                context_text, context_mode, fts_content, fts_thinking
            )
            SELECT
                message_id, role, content, thinking, timestamp, has_thinking,
                {context_text_expr}, {context_mode_expr},
                {context_text_expr} || COALESCE(content, ''),
                {context_text_expr} || COALESCE(thinking, '')
            FROM message_state
            """
        )
        conn.execute("DROP TABLE message_state")
        conn.execute("ALTER TABLE message_state_new RENAME TO message_state")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_message_state_has_thinking "
            "ON message_state(has_thinking)"
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _column_info(conn: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, ColumnInfo]:
    rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    if not rows:
        raise RuntimeError(f"expected table to exist before migration: {table_name}")
    return {
        str(row[1]): (str(row[2]).upper(), bool(row[3]), None if row[4] is None else str(row[4]))
        for row in rows
    }


def _column_names(conn: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    return set(_column_info(conn, table_name))


def _is_text_column(info: ColumnInfo) -> bool:
    return info[0] in {"TEXT", "VARCHAR"}


def _default_contains(default: str | None, expected: str) -> bool:
    return default is not None and expected in default


register(
    Migration(
        id="0016_contextual_retrieval_fts_columns",
        target_version=16,
        upgrade=_upgrade,
    )
)
