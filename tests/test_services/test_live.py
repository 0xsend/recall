"""Tail and cursor reads for live agent sessions (REQ-LIVE-004).

``load_session`` answers "the whole session, optionally the first N messages".
A driver watching a running agent needs the opposite: the last N messages, or
only what landed after the cursor it last saw. These tests pin that contract on
a hand-built database so every idx and every expected message is a literal.
"""

from __future__ import annotations

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.services.sessions import load_session_tail

SESSION_ID = "aabbccddeeff00112233445566778899"


def _make_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    return conn


def _insert_session(conn: duckdb.DuckDBPyConnection, *, message_count: int) -> None:
    conn.execute(
        "INSERT INTO sessions (id, source, source_path, source_session_id)"
        " VALUES (?, 'claude_code', ?, NULL)",
        [SESSION_ID, f"/tmp/{SESSION_ID}.jsonl"],
    )
    conn.execute(
        "INSERT INTO session_state (session_id, file_mtime, file_size,"
        " message_count, tool_count) VALUES (?, 0, 0, ?, 0)",
        [SESSION_ID, message_count],
    )


def _insert_message(conn: duckdb.DuckDBPyConnection, idx: int, *, role: str, content: str) -> str:
    message_id = f"msg{idx:02d}"
    conn.execute(
        "INSERT INTO messages (id, session_id, idx, agent_id) VALUES (?, ?, ?, NULL)",
        [message_id, SESSION_ID, idx],
    )
    conn.execute(
        "INSERT INTO message_state (message_id, role, content) VALUES (?, ?, ?)",
        [message_id, role, content],
    )
    return message_id


@pytest.fixture
def five_message_session() -> duckdb.DuckDBPyConnection:
    """A session whose five messages carry distinguishable content per idx."""
    conn = _make_conn()
    _insert_session(conn, message_count=5)
    for idx, (role, content) in enumerate(
        [
            ("user", "zero"),
            ("assistant", "one"),
            ("user", "two"),
            ("assistant", "three"),
            ("user", "four"),
        ]
    ):
        _insert_message(conn, idx, role=role, content=content)
    return conn


def test_tail_returns_the_last_messages_in_ascending_idx(five_message_session) -> None:
    """REQ-LIVE-004: --tail N is anchored at the end, unlike --message-limit."""
    session = load_session_tail(SESSION_ID, tail=3, conn=five_message_session)

    assert [message.idx for message in session.messages] == [2, 3, 4]
    assert [message.content for message in session.messages] == ["two", "three", "four"]


def test_tail_larger_than_the_session_returns_every_message(five_message_session) -> None:
    session = load_session_tail(SESSION_ID, tail=50, conn=five_message_session)

    assert [message.idx for message in session.messages] == [0, 1, 2, 3, 4]


def test_after_idx_returns_only_later_messages(five_message_session) -> None:
    """REQ-LIVE-004: a cursor read is a delta, not a re-read."""
    session = load_session_tail(SESSION_ID, after_idx=2, conn=five_message_session)

    assert [message.idx for message in session.messages] == [3, 4]
    assert [message.content for message in session.messages] == ["three", "four"]


def test_after_idx_at_the_head_water_mark_returns_an_empty_delta(five_message_session) -> None:
    """REQ-LIVE-004: an empty delta is a successful, empty response."""
    session = load_session_tail(SESSION_ID, after_idx=4, conn=five_message_session)

    assert session.messages == []
    assert session.id == SESSION_ID


def test_after_idx_and_tail_compose_to_the_last_n_of_the_delta(five_message_session) -> None:
    session = load_session_tail(SESSION_ID, after_idx=0, tail=2, conn=five_message_session)

    assert [message.idx for message in session.messages] == [3, 4]


def test_no_window_returns_the_whole_session(five_message_session) -> None:
    session = load_session_tail(SESSION_ID, conn=five_message_session)

    assert [message.idx for message in session.messages] == [0, 1, 2, 3, 4]


def test_tail_attaches_only_the_window_s_tool_calls(five_message_session) -> None:
    """A windowed read stays bounded: tool calls outside the window never load."""
    conn = five_message_session
    for tool_idx, (message_id, tool_name) in enumerate([("msg01", "Read"), ("msg03", "Bash")]):
        conn.execute(
            "INSERT INTO tool_calls (id, session_id, message_id, idx, tool_name)"
            " VALUES (?, ?, ?, ?, ?)",
            [f"tc{tool_idx}", SESSION_ID, message_id, tool_idx, tool_name],
        )

    session = load_session_tail(SESSION_ID, tail=3, tools=True, conn=conn)

    by_idx = {message.idx: message for message in session.messages}
    assert [call.tool_name for call in by_idx[3].tool_calls] == ["Bash"]
    assert by_idx[2].tool_calls == []
    assert by_idx[4].tool_calls == []
    assert session.orphan_tool_calls == []


def test_tail_without_tools_leaves_tool_calls_empty(five_message_session) -> None:
    conn = five_message_session
    conn.execute(
        "INSERT INTO tool_calls (id, session_id, message_id, idx, tool_name)"
        " VALUES ('tc0', ?, 'msg03', 0, 'Bash')",
        [SESSION_ID],
    )

    session = load_session_tail(SESSION_ID, tail=3, conn=conn)

    assert all(message.tool_calls == [] for message in session.messages)


def test_tail_must_be_positive(five_message_session) -> None:
    with pytest.raises(ValueError, match="tail must be positive"):
        load_session_tail(SESSION_ID, tail=0, conn=five_message_session)


def test_unknown_session_raises_not_found() -> None:
    conn = _make_conn()

    with pytest.raises(ValueError, match="session not found"):
        load_session_tail(SESSION_ID, tail=1, conn=conn)
