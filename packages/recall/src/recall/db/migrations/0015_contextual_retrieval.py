"""Migration 0015: add Contextual Retrieval schema columns.

Contextual Retrieval stores the write-time prefix and mode on message_state,
tracks a per-cache-row context version, and records last-run context counters in
runtime_state. This migration is deliberately schema-only; later phases compute
prefixes, thread them into embeddings and FTS, and update runtime counters.

The migration is idempotent:
- v14 databases receive any missing columns and advance to v15.
- If the columns already exist with the required defaults/constraints, DDL is
  skipped and the version can still advance.
- If a previous run stopped midway, shape checks converge partial columns to the
  schema.sql contract before recording success.
- Migration 0014 advances only to v14, so a failed 0015 leaves the database at
  v14 and the runner can retry this migration on the next ensure_schema call.
- Databases older than v14 are not safe to mutate here because earlier schema
  work may not be replayable. The migration returns False in that state so the
  runner records nothing, halts, and the user gets the existing --recreate path.

DuckDB 1.5.x supports additive columns but cannot alter existing CHECK
constraints in place. Adding the new message_state.context_mode CHECK usually
works through ALTER TABLE; if it does not, or if a partial run left context_mode
without its CHECK, the migration falls back to recreating message_state with
context_text defaulting to '' and context_mode defaulting to 'off'. Existing
embedding_cache rows are preserved and receive context_version 0.
"""

from __future__ import annotations

import logging

import duckdb

from recall.db.migrations import Migration, register

logger = logging.getLogger("recall.schema")

CONTEXT_MODES = ("off", "template", "llm-local", "llm-remote")
ColumnInfo = tuple[str, bool, str | None]


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    """Apply the v14→v15 additive schema change, or no-op when not applicable."""
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < 14:
        logger.warning(
            "0015_contextual_retrieval requires schema_version >= 14; "
            "DB is at %d. Halting; user must run `recall index --recreate --yes`.",
            current,
        )
        return False

    try:
        _add_message_state_context_columns(conn)
        _add_embedding_cache_context_version(conn)
        _add_runtime_state_context_columns(conn)
    except Exception:
        logger.exception("Failed to add Contextual Retrieval schema columns")
        return False

    if current < 15:
        set_schema_version_to(conn, 15)
    return True


def _add_message_state_context_columns(conn: duckdb.DuckDBPyConnection) -> None:
    columns = _column_info(conn, "message_state")
    text_info = columns.get("context_text")
    mode_info = columns.get("context_mode")
    text_ok = text_info is not None and _default_contains(text_info[2], "''")
    mode_ok = (
        mode_info is not None
        and _default_contains(mode_info[2], "'off'")
        and _has_context_mode_check(conn)
    )
    if text_ok and mode_ok:
        return

    # The clean v14 path can stay additive. Any partial state that already has
    # one of the columns but lacks its required default/CHECK is normalized by
    # table recreation below, because DuckDB cannot add CHECKs to existing columns.
    if text_info is None and mode_info is None:
        conn.execute("ALTER TABLE message_state ADD COLUMN context_text TEXT DEFAULT ''")
        try:
            conn.execute(
                """
                ALTER TABLE message_state
                ADD COLUMN context_mode TEXT DEFAULT 'off' CHECK (
                    context_mode IS NULL
                    OR context_mode IN ('off', 'template', 'llm-local', 'llm-remote')
                )
                """
            )
            return
        except Exception:
            logger.info(
                "ALTER TABLE could not add message_state.context_mode CHECK; "
                "recreating message_state"
            )

    _recreate_message_state_with_context_columns(conn)


def _recreate_message_state_with_context_columns(conn: duckdb.DuckDBPyConnection) -> None:
    columns = _column_names(conn, "message_state")
    context_text_expr = "COALESCE(context_text, '')" if "context_text" in columns else "''"
    if "context_mode" in columns:
        allowed_modes = ", ".join(f"'{mode}'" for mode in CONTEXT_MODES)
        context_mode_expr = (
            f"CASE WHEN context_mode IN ({allowed_modes}) THEN context_mode ELSE 'off' END"
        )
    else:
        context_mode_expr = "'off'"

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
                )
            )
            """
        )
        conn.execute(
            f"""
            INSERT INTO message_state_new (
                message_id, role, content, thinking, timestamp, has_thinking,
                context_text, context_mode
            )
            SELECT
                message_id, role, content, thinking, timestamp, has_thinking,
                {context_text_expr}, {context_mode_expr}
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


def _add_embedding_cache_context_version(conn: duckdb.DuckDBPyConnection) -> None:
    columns = _column_info(conn, "embedding_cache")
    info = columns.get("context_version")
    if info is None:
        try:
            conn.execute(
                """
                ALTER TABLE embedding_cache
                ADD COLUMN context_version INTEGER NOT NULL DEFAULT 0
                """
            )
        except duckdb.Error:
            logger.info(
                "ALTER TABLE could not add canonical embedding_cache.context_version; "
                "recreating embedding_cache"
            )
            _recreate_embedding_cache_with_context_version(conn)
        return

    if _is_canonical_context_version(info):
        return

    logger.info(
        "embedding_cache.context_version has non-canonical shape %s; repairing column",
        info,
    )
    try:
        conn.execute("ALTER TABLE embedding_cache DROP COLUMN context_version")
        conn.execute(
            """
            ALTER TABLE embedding_cache
            ADD COLUMN context_version INTEGER NOT NULL DEFAULT 0
            """
        )
    except duckdb.Error:
        logger.info(
            "ALTER TABLE could not repair embedding_cache.context_version in place; "
            "recreating embedding_cache"
        )
        _recreate_embedding_cache_with_context_version(conn)


def _recreate_embedding_cache_with_context_version(conn: duckdb.DuckDBPyConnection) -> None:
    columns = _column_info(conn, "embedding_cache")
    embedding_type = columns["embedding"][0]

    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(
            f"""
            CREATE TABLE embedding_cache_new (
                cache_key TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK (kind IN ('content', 'thinking', 'bash')),
                raw_text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                embedding {embedding_type} NOT NULL,
                normalization_version INTEGER NOT NULL DEFAULT 1,
                context_version INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT INTO embedding_cache_new (
                cache_key, kind, raw_text, normalized_text, embedding,
                normalization_version, context_version, created_at
            )
            SELECT
                cache_key, kind, raw_text, normalized_text, embedding,
                normalization_version, 0, created_at
            FROM embedding_cache
            """
        )
        conn.execute("DROP TABLE embedding_cache")
        conn.execute("ALTER TABLE embedding_cache_new RENAME TO embedding_cache")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding_cache_kind ON embedding_cache(kind)")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _add_runtime_state_context_columns(conn: duckdb.DuckDBPyConnection) -> None:
    existing = _column_names(conn, "runtime_state")
    columns = {
        "last_context_messages": "INTEGER",
        "last_context_mode": "TEXT",
        "last_context_input_tokens": "BIGINT",
        "last_context_output_tokens": "BIGINT",
        "last_context_model": "TEXT",
    }
    for name, sql_type in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE runtime_state ADD COLUMN {name} {sql_type}")


def _column_names(conn: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    return set(_column_info(conn, table_name))


def _column_info(conn: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, ColumnInfo]:
    rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    if not rows:
        raise RuntimeError(f"expected table to exist before migration: {table_name}")
    return {
        str(row[1]): (str(row[2]).upper(), bool(row[3]), None if row[4] is None else str(row[4]))
        for row in rows
    }


def _default_contains(default: str | None, expected: str) -> bool:
    return default is not None and expected in default


def _is_canonical_context_version(info: ColumnInfo) -> bool:
    return info == ("INTEGER", True, "0")


def _has_context_mode_check(conn: duckdb.DuckDBPyConnection) -> bool:
    rows = conn.execute(
        """
        SELECT expression FROM duckdb_constraints()
        WHERE table_name = 'message_state' AND constraint_type = 'CHECK'
        """
    ).fetchall()
    return any(
        "context_mode" in str(row[0]) and all(f"'{mode}'" in str(row[0]) for mode in CONTEXT_MODES)
        for row in rows
    )


register(
    Migration(
        id="0015_contextual_retrieval",
        target_version=15,
        upgrade=_upgrade,
    )
)
