"""Wide payloads must not be retained for every vector candidate before LIMIT."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from recall.core.types import Source
from recall.db.schema import ensure_schema
from recall.services.search import _vector_search_messages, _vector_search_tool_calls


@pytest.mark.parametrize("kind", ["message", "tool_call"])
def test_vector_search_bounds_wide_candidate_payloads(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "vectors.duckdb"
    with duckdb.connect(str(path)) as conn:
        ensure_schema(conn)
        conn.execute("INSERT INTO sessions VALUES ('s', 'codex', '/owned/source', NULL)")
        conn.execute(
            "INSERT INTO messages (id, session_id, idx) "
            "SELECT i::TEXT, 's', i FROM range(8192) r(i)"
        )
        conn.execute("""
            INSERT INTO message_state (message_id, role, content)
            SELECT i::TEXT, 'assistant', repeat(md5(i::TEXT), 512) FROM range(8192) r(i)
        """)
        if kind == "message":
            conn.execute("""
                INSERT INTO message_embeddings (message_id, content_embedding)
                SELECT i::TEXT, list_transform(range(384), j ->
                    CASE WHEN j = 0 THEN 1.0 WHEN j = 1 THEN i::FLOAT ELSE 0.0 END)
                FROM range(8192) r(i)
            """)
        else:
            conn.execute("""
                INSERT INTO tool_calls (id, session_id, message_id, idx, tool_name, bash_command)
                SELECT i::TEXT, 's', i::TEXT, i, 'Bash', repeat(md5(i::TEXT), 512)
                FROM range(8192) r(i)
            """)
            conn.execute("""
                INSERT INTO tool_call_embeddings (tool_call_id, bash_embedding)
                SELECT i::TEXT, list_transform(range(384), j ->
                    CASE WHEN j = 0 THEN 1.0 WHEN j = 1 THEN i::FLOAT ELSE 0.0 END)
                FROM range(8192) r(i)
            """)
        conn.execute("CHECKPOINT")

    with duckdb.connect(str(path), config={"memory_limit": "64MiB", "threads": 2}) as conn:
        query = _vector_search_messages if kind == "message" else _vector_search_tool_calls
        results = query(conn, [1.0] + [0.0] * 383, source=Source.CODEX, session="s", limit=2)
        assert [result.message_id for result in results] == ["0", "1"]
        assert results[0].score == pytest.approx(1.0)
        assert results[1].score == pytest.approx(2**-0.5)
        assert results[0].source_path == "/owned/source"
        expected = conn.execute("SELECT repeat(md5('0'), 512)").fetchone()
        assert expected is not None
        payload = results[0].content if kind == "message" else results[0].bash_command
        assert payload == expected[0]


@pytest.mark.parametrize("kind", ["message", "tool_call"])
def test_vector_search_bounds_vectors_during_identity_joins(tmp_path: Path, kind: str) -> None:
    """Identity filtering must not carry full vectors through every hash join."""
    path = tmp_path / "joined-vectors.duckdb"
    with duckdb.connect(str(path)) as conn:
        ensure_schema(conn)
        conn.execute("INSERT INTO sessions VALUES ('s', 'codex', '/owned/join-source', NULL)")
        conn.execute(
            "INSERT INTO messages (id, session_id, idx) "
            "SELECT i::TEXT, 's', i FROM range(32768) r(i)"
        )
        conn.execute("""
            INSERT INTO message_state (message_id, role, content)
            SELECT i::TEXT, 'assistant', 'payload' FROM range(32768) r(i)
        """)
        if kind == "message":
            conn.execute("""
                INSERT INTO message_embeddings (message_id, content_embedding)
                SELECT i::TEXT, list_transform(range(384), j ->
                    CASE WHEN j = 0 THEN 1.0 WHEN j = 1 THEN i::FLOAT ELSE 0.0 END)
                FROM range(32768) r(i)
            """)
        else:
            conn.execute("""
                INSERT INTO tool_calls (id, session_id, message_id, idx, tool_name, bash_command)
                SELECT i::TEXT, 's', i::TEXT, i, 'Bash', 'payload' FROM range(32768) r(i)
            """)
            conn.execute("""
                INSERT INTO tool_call_embeddings (tool_call_id, bash_embedding)
                SELECT i::TEXT, list_transform(range(384), j ->
                    CASE WHEN j = 0 THEN 1.0 WHEN j = 1 THEN i::FLOAT ELSE 0.0 END)
                FROM range(32768) r(i)
            """)
        conn.execute("CHECKPOINT")

    with duckdb.connect(str(path), config={"memory_limit": "64MiB", "threads": 2}) as conn:
        # A pending payload rewrite and its indexes share this handle with search.
        conn.execute("UPDATE message_state SET content='current' WHERE message_id='0'")
        query = _vector_search_messages if kind == "message" else _vector_search_tool_calls
        results = query(conn, [1.0] + [0.0] * 383, source=None, limit=2)
        assert [result.message_id for result in results] == ["0", "1"]
        assert results[0].score == pytest.approx(1.0)
        assert results[1].score == pytest.approx(2**-0.5)
        assert results[0].source_path == "/owned/join-source"
        assert results[0].content == ("current" if kind == "message" else None)
