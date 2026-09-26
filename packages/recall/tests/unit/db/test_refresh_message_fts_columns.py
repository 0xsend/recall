from __future__ import annotations

import duckdb
import pytest
from recall.db.queries import refresh_message_fts_columns


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    handle = duckdb.connect(":memory:")
    handle.execute(
        """
        CREATE TABLE message_state (
            message_id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT,
            thinking TEXT,
            timestamp TIMESTAMP,
            has_thinking BOOLEAN DEFAULT FALSE,
            context_text TEXT DEFAULT '',
            context_mode TEXT DEFAULT 'off',
            fts_content TEXT DEFAULT '',
            fts_thinking TEXT DEFAULT ''
        )
        """
    )
    return handle


def _insert(
    conn: duckdb.DuckDBPyConnection,
    *,
    message_id: str,
    content: str | None,
    thinking: str | None,
    context_text: str,
    fts_content: str,
    fts_thinking: str,
) -> None:
    conn.execute(
        """
        INSERT INTO message_state (
            message_id, role, content, thinking, context_text,
            fts_content, fts_thinking
        ) VALUES (?, 'user', ?, ?, ?, ?, ?)
        """,
        [message_id, content, thinking, context_text, fts_content, fts_thinking],
    )


def test_converged_db_touches_zero_rows(conn: duckdb.DuckDBPyConnection) -> None:
    _insert(
        conn,
        message_id="m1",
        content="hello",
        thinking="reasoning",
        context_text="ctx::",
        fts_content="ctx::hello",
        fts_thinking="ctx::reasoning",
    )
    _insert(
        conn,
        message_id="m2",
        content=None,
        thinking=None,
        context_text="",
        fts_content="",
        fts_thinking="",
    )

    updated = refresh_message_fts_columns(conn)

    assert updated == 0


def test_backfills_rows_with_stale_derived_columns(conn: duckdb.DuckDBPyConnection) -> None:
    _insert(
        conn,
        message_id="m1",
        content="hello",
        thinking="reasoning",
        context_text="ctx::",
        fts_content="WRONG",
        fts_thinking="WRONG",
    )
    _insert(
        conn,
        message_id="m2",
        content="ok",
        thinking="thoughts",
        context_text="",
        fts_content="ok",
        fts_thinking="thoughts",
    )

    updated = refresh_message_fts_columns(conn)

    assert updated == 1
    row = conn.execute(
        "SELECT fts_content, fts_thinking FROM message_state WHERE message_id = 'm1'"
    ).fetchone()
    assert row == ("ctx::hello", "ctx::reasoning")
    # Other row stays untouched.
    row = conn.execute(
        "SELECT fts_content, fts_thinking FROM message_state WHERE message_id = 'm2'"
    ).fetchone()
    assert row == ("ok", "thoughts")


def test_treats_null_content_thinking_as_empty(conn: duckdb.DuckDBPyConnection) -> None:
    _insert(
        conn,
        message_id="m1",
        content=None,
        thinking=None,
        context_text="ctx::",
        fts_content="WRONG",
        fts_thinking="WRONG",
    )

    updated = refresh_message_fts_columns(conn)

    assert updated == 1
    row = conn.execute(
        "SELECT fts_content, fts_thinking FROM message_state WHERE message_id = 'm1'"
    ).fetchone()
    assert row == ("ctx::", "ctx::")


def test_idempotent_second_call_is_zero(conn: duckdb.DuckDBPyConnection) -> None:
    _insert(
        conn,
        message_id="m1",
        content="hello",
        thinking="reasoning",
        context_text="ctx::",
        fts_content="WRONG",
        fts_thinking="WRONG",
    )

    first = refresh_message_fts_columns(conn)
    second = refresh_message_fts_columns(conn)

    assert first == 1
    assert second == 0
