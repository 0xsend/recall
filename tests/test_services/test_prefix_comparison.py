"""Prefix classification compares raw facts with bounded payload allocation."""

import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest
from recall.core.models import Message, Session, TailFacts, ToolCall
from recall.core.types import Role, Source
from recall.db.schema import ensure_schema
from recall.services.indexer import _iter_prefix_messages, _preserves_indexed_prefix, _write_session


def _session(count: int = 1) -> Session:
    return Session(
        id="prefix",
        source=Source.CODEX,
        source_path="/owned/prefix",
        file_mtime=1,
        file_size=100,
        messages=[
            Message(
                id=f"m-{i}",
                session_id="prefix",
                idx=i,
                role=Role.ASSISTANT,
                content="retained fact",
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                tool_calls=[
                    ToolCall(
                        id=f"t-{i}",
                        session_id="prefix",
                        message_id=f"m-{i}",
                        idx=i,
                        tool_name="Bash",
                        tool_input={"b": 2, "a": 1},
                        bash_command="echo retained",
                        bash_base="echo",
                    )
                ],
            )
            for i in range(count)
        ],
    )


def test_prefix_comparison_does_not_materialize_vector_sized_payloads(tmp_path: Path) -> None:
    with duckdb.connect(str(tmp_path / "prefix.duckdb")) as conn:
        ensure_schema(conn)
        session = _session(1000)
        _write_session(conn, session, tail_facts=TailFacts())
        conn.execute("""INSERT INTO message_embeddings
            (message_id, content_embedding, thinking_embedding)
            SELECT id, list_resize([0.01], 384, 0.01), list_resize([0.02], 384, 0.02)
            FROM messages""")
        conn.execute("""INSERT INTO tool_call_embeddings (tool_call_id, bash_embedding)
            SELECT id, list_resize([0.03], 384, 0.03) FROM tool_calls""")
        tracemalloc.start()
        try:
            assert _preserves_indexed_prefix(conn, session)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # This corpus has tiny raw facts but >1M vector elements. Comparison
        # must fit an 8 MiB Python allocation budget regardless of those vectors.
        assert peak < 8 * 1024 * 1024


@pytest.mark.parametrize("change", ["none", "content", "agent", "tool", "remove", "identity"])
def test_prefix_comparison_rejects_changed_old_facts(tmp_path: Path, change: str) -> None:
    with duckdb.connect(str(tmp_path / "facts.duckdb")) as conn:
        ensure_schema(conn)
        session = _session()
        _write_session(conn, session, tail_facts=TailFacts())
        changed = session.model_copy(deep=True)
        if change == "content":
            changed.messages[0].content = "different"
        elif change == "agent":
            changed.messages[0].agent_id = "another-agent"
        elif change == "tool":
            changed.messages[0].tool_calls[0].tool_input = {"a": 3}
        elif change == "remove":
            changed.messages = []
        elif change == "identity":
            changed.messages[0].id = "different"
        else:
            changed.messages[0].tool_calls[0].tool_input = {"a": 1, "b": 2}
        assert _preserves_indexed_prefix(conn, changed) is (change == "none")


def test_prefix_comparison_preserves_missing_companion_semantics(tmp_path: Path) -> None:
    with duckdb.connect(str(tmp_path / "companions.duckdb")) as conn:
        ensure_schema(conn)
        session = _session()
        assert not _preserves_indexed_prefix(conn, session)
        _write_session(conn, session, tail_facts=TailFacts())
        conn.execute("DELETE FROM message_state")
        assert _preserves_indexed_prefix(conn, session)
        conn.execute("DELETE FROM session_state")
        assert not _preserves_indexed_prefix(conn, session)


def test_a_rewrite_leaves_unchanged_vectors_unread_and_in_place(tmp_path: Path) -> None:
    """A full sync of an embedded session reads only the vectors that move.

    Every stored vector of a long session is tens of megabytes as Python
    floats and seconds of reads under the writer; a one-message edit keeps all
    but one of them where they are.
    """
    with duckdb.connect(str(tmp_path / "vectors.duckdb")) as conn:
        ensure_schema(conn)
        session = _session(1000)
        _write_session(conn, session, tail_facts=TailFacts())
        conn.execute("""INSERT INTO message_embeddings
            (message_id, content_embedding, thinking_embedding)
            SELECT id, list_resize([0.01], 384, 0.01), list_resize([0.02], 384, 0.02)
            FROM messages""")
        conn.execute("""INSERT INTO tool_call_embeddings (tool_call_id, bash_embedding)
            SELECT id, list_resize([0.03], 384, 0.03) FROM tool_calls""")
        edited = session.model_copy(deep=True)
        edited.messages[500].content = "edited fact"

        tracemalloc.start()
        try:
            _write_session(conn, edited, tail_facts=TailFacts())
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert peak < 8 * 1024 * 1024
        assert conn.execute(
            """SELECT COUNT(*) FILTER (WHERE content_embedding IS NOT NULL),
                      COUNT(*) FILTER (WHERE thinking_embedding IS NOT NULL)
               FROM message_embeddings"""
        ).fetchone() == (999, 1000)
        assert conn.execute(
            "SELECT content_embedding FROM message_embeddings WHERE message_id = 'm-500'"
        ).fetchone() == (None,)
        assert conn.execute(
            "SELECT COUNT(*) FROM tool_call_embeddings WHERE bash_embedding IS NOT NULL"
        ).fetchone() == (1000,)


def test_prefix_iteration_includes_every_stored_message(tmp_path: Path) -> None:
    with duckdb.connect(str(tmp_path / "long-prefix.duckdb")) as conn:
        ensure_schema(conn)
        session = _session(3000)
        _write_session(conn, session, tail_facts=TailFacts())
        assert [message.id for message in _iter_prefix_messages(conn, session.id)] == [
            f"m-{i}" for i in range(3000)
        ]
