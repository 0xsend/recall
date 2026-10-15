"""Identity pages and their payloads belong to one caller-owned generation."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.db.schema import ensure_schema
from recall.services import sessions


@pytest.fixture
def population(tmp_path: Path) -> Iterator[tuple[AppConfig, duckdb.DuckDBPyConnection]]:
    config = replace(AppConfig.load(), data_dir=tmp_path, db_path=tmp_path / "recall.duckdb")
    conn = duckdb.connect(str(config.db_path))
    ensure_schema(conn)
    conn.execute("INSERT INTO sessions VALUES ('s', 'codex', '/owned/source', NULL)")
    conn.execute(
        "INSERT INTO session_state (session_id, message_count, file_mtime, file_size)"
        " VALUES ('s', 600, 0, 0)"
    )
    conn.execute(
        "INSERT INTO messages (id, session_id, idx) SELECT 'm' || i, 's', i FROM range(600) r(i)"
    )
    conn.execute("""
        INSERT INTO message_state (message_id, role, content, thinking, context_text, context_mode)
        SELECT 'm' || i, 'assistant', 'old-' || i, 'thought-' || i, 'context-' || i, 'template'
        FROM range(600) r(i)
    """)
    conn.execute("""
        INSERT INTO tool_calls (id, session_id, message_id, idx, tool_name, tool_input)
        SELECT 't' || i, 's', 'm' || i, i, 'Read', '{"generation":"old"}' FROM range(600) r(i)
    """)
    conn.execute("""
        INSERT INTO tool_calls (id, session_id, message_id, idx, tool_name)
        VALUES ('orphan', 's', NULL, 600, 'Bash')
    """)
    yield config, conn
    conn.close()


def test_tail_window_spans_pages_and_preserves_complete_payloads(population) -> None:
    _config, conn = population
    session = sessions.load_session_tail("s", tail=300, after_idx=250, tools=True, conn=conn)

    assert [message.idx for message in session.messages] == list(range(300, 600))
    assert [message.content for message in session.messages] == [
        f"old-{i}" for i in range(300, 600)
    ]
    assert [message.thinking for message in session.messages] == [
        f"thought-{i}" for i in range(300, 600)
    ]
    assert [message.context_text for message in session.messages] == [
        f"context-{i}" for i in range(300, 600)
    ]
    assert all(message.context_mode == "template" for message in session.messages)
    assert [message.tool_calls[0].id for message in session.messages] == [
        f"t{i}" for i in range(300, 600)
    ]
    assert session.orphan_tool_calls == []


@pytest.mark.parametrize(
    "read_offset", [0, 256, 600], ids=["before-payload", "between-pages", "before-tools"]
)
def test_owned_tail_snapshot_survives_a_concurrent_rewrite(
    population, monkeypatch: pytest.MonkeyPatch, read_offset: int
) -> None:
    config, writer = population
    reached = threading.Event()
    committed = threading.Event()
    failures: list[BaseException] = []
    original = sessions._message_rows_by_id
    # The daemon's readers and writer are read-write handles to one instance.
    # DuckDB disallows mixing a read-only open with its existing read-write open.
    monkeypatch.setattr(
        sessions, "connect_readonly", lambda config: duckdb.connect(str(config.db_path))
    )

    def rewrite() -> None:
        try:
            assert reached.wait(10), "reader did not reach rewrite barrier"
            writer.execute("BEGIN")
            writer.execute("UPDATE message_state SET content = 'rewritten'")
            writer.execute('UPDATE tool_calls SET tool_input = \'{"generation":"rewritten"}\'')
            writer.execute("COMMIT")
        except BaseException as err:
            failures.append(err)
        finally:
            committed.set()

    def gated_rows(conn, identifiers):
        def gated_ids():
            for index, identifier in enumerate(identifiers):
                if index == read_offset:
                    reached.set()
                    assert committed.wait(10), "writer did not commit"
                yield identifier
            if read_offset == 600:
                reached.set()
                assert committed.wait(10), "writer did not commit"

        return original(conn, gated_ids())

    monkeypatch.setattr(sessions, "_message_rows_by_id", gated_rows)
    thread = threading.Thread(target=rewrite)
    thread.start()
    try:
        session = sessions.load_session_tail("s", tail=600, tools=True, config=config)
    finally:
        reached.set()
        thread.join(timeout=15)
    assert not thread.is_alive()
    assert not failures
    assert [message.content for message in session.messages] == [f"old-{i}" for i in range(600)]
    assert all(
        message.tool_calls[0].tool_input == {"generation": "old"} for message in session.messages
    )
    assert writer.execute("SELECT content FROM message_state WHERE message_id='m0'").fetchone() == (
        "rewritten",
    )


@pytest.mark.parametrize("missing", [False, True], ids=["success", "failure"])
def test_borrowed_tail_preserves_uncommitted_state_and_caller_rollback(population, missing) -> None:
    _config, conn = population
    conn.execute("BEGIN")
    conn.execute("UPDATE message_state SET content='uncommitted' WHERE message_id='m599'")
    if missing:
        with pytest.raises(ValueError, match="session not found"):
            sessions.load_session_tail("missing", tail=1, conn=conn)
    else:
        session = sessions.load_session_tail("s", tail=1, conn=conn)
        assert session.messages[0].content == "uncommitted"
    assert conn.execute("SELECT content FROM message_state WHERE message_id='m599'").fetchone() == (
        "uncommitted",
    )
    conn.execute("ROLLBACK")
    assert conn.execute("SELECT content FROM message_state WHERE message_id='m599'").fetchone() == (
        "old-599",
    )
