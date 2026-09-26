"""A stop marker applies only below the same session's surviving message tail."""

from collections.abc import Iterator

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.services.live import fetch_last_stop_marker


@pytest.fixture
def conn() -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect(":memory:") as connection:
        ensure_schema(connection)
        # Mixed-source startup includes a tool-only session with an end marker
        # at -1 and no messages, between two sessions that do have messages.
        connection.execute(
            "INSERT INTO messages (id, session_id, idx) VALUES"
            " ('a0', 'a', 0), ('a1', 'a', 1),"
            " ('z0', 'z', 0), ('z1', 'z', 1), ('z2', 'z', 2), ('z3', 'z', 3)"
        )
        connection.execute(
            "INSERT INTO session_stop_markers VALUES"
            " ('a', 1, 'tool_use', false), ('m', -1, 'task_complete', true),"
            " ('z', 1, 'tool_use', false), ('z', 3, 'end_turn', true)"
        )
        yield connection


def test_a_message_free_session_has_no_applicable_stop_marker(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    with conn.cursor() as reader:
        assert fetch_last_stop_marker(reader, "m") is None


def test_the_latest_marker_belongs_to_the_requested_session(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    marker = fetch_last_stop_marker(conn, "a")

    assert marker is not None
    assert marker.idx == 1
    assert marker.reason == "tool_use"
    assert marker.ends_turn is False


def test_truncation_ignores_markers_above_the_surviving_tail(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    conn.execute("DELETE FROM messages WHERE session_id = 'z' AND idx >= 2")

    marker = fetch_last_stop_marker(conn, "z")

    assert marker is not None
    assert marker.idx == 1
    assert marker.reason == "tool_use"
    assert marker.ends_turn is False


def test_deleting_the_last_message_leaves_no_applicable_marker(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    conn.execute("DELETE FROM messages WHERE session_id = 'z'")

    assert fetch_last_stop_marker(conn, "z") is None
