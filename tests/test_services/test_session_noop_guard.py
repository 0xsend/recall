"""Regression tests for the session_state no-op re-index guard.

An unchanged re-index must not rewrite the ``session_state`` row: DuckDB
implements UPDATE as delete+insert, so a needless UPDATE is exactly the
update-churn the compaction/bloat work tries to avoid, and the bloat estimator
is blind to it. Two write paths churn the row on every re-index:

* ``_update_session_row`` — its skip-when-unchanged guard was dead (a 19-column
  SELECT compared against a 17-tuple, plus an always-fresh ``indexed_at``).
* ``_set_session_host`` — an unconditional ``UPDATE session_state SET host``.

These tests spy on executed SQL to assert an unchanged re-index issues zero
``UPDATE session_state`` statements, while genuine changes (content, host) still
write.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest
from recall.core.models import Message, Session, TailFacts, ToolCall
from recall.core.types import Role, Source
from recall.db.schema import ensure_schema
from recall.services.indexer import _write_session


class _RecordingConn:
    """Forwarding proxy over a DuckDB connection that records executed SQL."""

    def __init__(self, inner: duckdb.DuckDBPyConnection) -> None:
        self._inner = inner
        self.sql: list[str] = []

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        self.sql.append(sql)
        return self._inner.execute(sql, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def session_state_updates(self) -> list[str]:
        return [s for s in self.sql if "UPDATE session_state" in s]


def _rewrite(conn: _RecordingConn, session: Session, **kwargs: Any) -> None:
    """Call ``_write_session`` through the recording proxy (typed as a conn)."""
    kwargs.setdefault("tail_facts", TailFacts())
    _write_session(cast("duckdb.DuckDBPyConnection", conn), session, **kwargs)


def _col(inner: duckdb.DuckDBPyConnection, column: str) -> Any:
    row = inner.execute(
        f"SELECT {column} FROM session_state WHERE session_id = 'noop-session'"
    ).fetchone()
    assert row is not None
    return row[0]


def _tokens(inner: duckdb.DuckDBPyConnection) -> tuple[Any, ...] | None:
    return inner.execute(
        "SELECT input_tokens, output_tokens, cached_input_tokens "
        "FROM session_state WHERE session_id = 'noop-session'"
    ).fetchone()


def _session(**overrides: Any) -> Session:
    fields: dict[str, Any] = dict(
        id="noop-session",
        source=Source.CODEX,
        source_path="/tmp/noop-session.jsonl",
        file_mtime=1.0,
        file_size=1,
        git_repo="acme/recall",
        git_branch="main",
        # Timezone-aware timestamps: DuckDB stores TIMESTAMP columns naive
        # (aware values down-convert to local wall-clock), so the no-op guard
        # must normalize before comparing or it churns on every real session
        # (non-UTC-host tz footgun — regression coverage for the review reject).
        started_at=datetime(2026, 7, 23, 12, 0, 0, tzinfo=UTC),
        ended_at=datetime(2026, 7, 23, 12, 30, 0, tzinfo=UTC),
        input_tokens=100,
        output_tokens=50,
        messages=[
            Message(
                id="noop-message-1",
                session_id="noop-session",
                idx=0,
                role=Role.ASSISTANT,
                content="hello",
            )
        ],
        message_count=1,
    )
    fields.update(overrides)
    return Session(**fields)


def _fresh_conn() -> tuple[_RecordingConn, duckdb.DuckDBPyConnection]:
    inner = duckdb.connect(":memory:")
    ensure_schema(inner)
    return _RecordingConn(inner), inner


def test_unchanged_reindex_issues_no_session_state_update() -> None:
    conn, inner = _fresh_conn()
    session = _session()

    _rewrite(conn, session)
    before = _col(inner, "indexed_at")

    conn.sql.clear()
    _rewrite(conn, session)  # identical content — must be a no-op

    assert conn.session_state_updates() == [], (
        f"unchanged re-index rewrote session_state: {conn.session_state_updates()}"
    )
    after = _col(inner, "indexed_at")
    assert after == before, "indexed_at advanced on a no-op re-index"


def test_changed_session_still_updates() -> None:
    conn, inner = _fresh_conn()
    _rewrite(conn, _session())

    conn.sql.clear()
    _rewrite(conn, _session(model="gpt-5", message_count=1))

    assert conn.session_state_updates(), "a changed session must still rewrite session_state"
    assert _col(inner, "model") == "gpt-5"


def test_grok_null_token_reindex_preserves_tokens_without_churn() -> None:
    """Grok parses tokens as None; harvest fills them. Re-index must not churn."""
    conn, inner = _fresh_conn()
    _rewrite(conn, _session(input_tokens=None, output_tokens=None))
    # Simulate a later harvest rollup writing token totals onto the row.
    inner.execute(
        "UPDATE session_state SET input_tokens = 4200, output_tokens = 900, "
        "cached_input_tokens = 1900 WHERE session_id = 'noop-session'"
    )

    conn.sql.clear()
    _rewrite(conn, _session(input_tokens=None, output_tokens=None))

    assert conn.session_state_updates() == [], (
        "Grok null-token re-index rewrote session_state (null-preserve should be a no-op)"
    )
    assert _tokens(inner) == (4200, 900, 1900), "harvest-filled tokens were clobbered on re-index"


def test_pi_cached_tokens_persist_and_update() -> None:
    conn, inner = _fresh_conn()
    cached = _session(
        source=Source.PI_AGENT, input_tokens=7160, output_tokens=30, cached_input_tokens=7000
    )
    _rewrite(conn, cached)
    assert _tokens(inner) == (7160, 30, 7000)

    conn.sql.clear()
    _rewrite(conn, cached)
    assert conn.session_state_updates() == [], "unchanged cache totals must not rewrite the row"

    _rewrite(
        conn,
        _session(
            source=Source.PI_AGENT, input_tokens=7160, output_tokens=30, cached_input_tokens=7500
        ),
    )
    assert conn.session_state_updates(), "a cache-only change must rewrite session_state"
    assert _tokens(inner) == (7160, 30, 7500)


def test_host_change_still_reindexes_host() -> None:
    conn, inner = _fresh_conn()
    _rewrite(conn, _session(), host="host-a")

    conn.sql.clear()
    _rewrite(conn, _session(), host="host-b")

    assert any("SET host" in s for s in conn.session_state_updates()), (
        "a real host change must still write session_state.host (REQ-MULTIHOST-002)"
    )
    assert _col(inner, "host") == "host-b"


def _rewrite_one_of_eight_aware_messages() -> None:
    conn, inner = _fresh_conn()
    messages = [
        Message(
            id=f"noop-message-{index}",
            session_id="noop-session",
            idx=index,
            role=Role.ASSISTANT,
            content=f"message {index}",
            timestamp=datetime(2026, 7, 23, 12, index, 0, tzinfo=UTC),
        )
        for index in range(8)
    ]
    _rewrite(conn, _session(messages=messages, message_count=len(messages)))
    stored = inner.execute("SELECT message_id, timestamp FROM message_state").fetchall()

    edited = [message.model_copy() for message in messages]
    edited[3] = edited[3].model_copy(update={"content": "edited"})
    conn.sql.clear()
    _rewrite(conn, _session(messages=edited, message_count=len(edited)))

    updates = [sql for sql in conn.sql if "UPDATE message_state" in sql]
    assert len(updates) == 1, updates
    assert "content = incoming.content" in updates[0]
    assert "timestamp" not in updates[0]
    assert inner.execute("SELECT message_id, timestamp FROM message_state").fetchall() == stored
    assert inner.execute(
        "SELECT content FROM message_state WHERE message_id = 'noop-message-3'"
    ).fetchone() == ("edited",)


@pytest.mark.parametrize("zone", ["UTC", "America/Los_Angeles", "Asia/Kolkata"])
def test_a_reparse_rewrites_only_the_message_whose_content_changed(zone: str) -> None:
    """Parsers stamp aware timestamps; an unchanged message must not be rewritten.

    A full reparse of a rewritten transcript names every message. Comparing a
    stored naive timestamp with the parsed aware one made each of them look
    changed, so a one-message edit updated every row of a long session. DuckDB
    fixes its zone when it loads, so each host zone runs in its own process.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.test_services.test_session_noop_guard import "
            "_rewrite_one_of_eight_aware_messages as run; run()",
        ],
        env={**os.environ, "TZ": zone},
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr


def _embedding_rows(inner: duckdb.DuckDBPyConnection) -> list[tuple[Any, ...]]:
    return inner.execute(
        """SELECT 'message', message_id, rowid, content_embedding, thinking_embedding
           FROM message_embeddings
           UNION ALL
           SELECT 'tool', tool_call_id, rowid, bash_embedding, NULL FROM tool_call_embeddings
           ORDER BY 1, 2"""
    ).fetchall()


def _embedded_session(*, thinking_vector: bool) -> Session:
    return _session(
        messages=[
            Message(
                id="noop-message-1",
                session_id="noop-session",
                idx=0,
                role=Role.ASSISTANT,
                content="hello",
                thinking="considered",
                content_embedding=[0.125] * 384,
                thinking_embedding=[0.25] * 384 if thinking_vector else None,
                tool_calls=[
                    ToolCall(
                        id="noop-tool-1",
                        session_id="noop-session",
                        message_id="noop-message-1",
                        idx=0,
                        tool_name="Bash",
                        bash_command="git status",
                        bash_embedding=[0.5] * 384,
                    )
                ],
            )
        ],
    )


@pytest.mark.parametrize("thinking_vector", [True, False], ids=["pair", "content-only"])
def test_an_unchanged_replay_of_explicit_vectors_rewrites_no_embedding_row(
    thinking_vector: bool,
) -> None:
    """Replaying the vectors a row already stores is a no-op, field pairs included."""
    inner = duckdb.connect(":memory:")
    ensure_schema(inner)
    session = _embedded_session(thinking_vector=thinking_vector)
    _write_session(inner, session, tail_facts=TailFacts())
    before = _embedding_rows(inner)

    _write_session(inner, session.model_copy(deep=True), tail_facts=TailFacts())

    assert _embedding_rows(inner) == before


def test_a_replay_naming_one_vector_keeps_its_stored_sibling_in_place() -> None:
    inner = duckdb.connect(":memory:")
    ensure_schema(inner)
    session = _embedded_session(thinking_vector=True)
    _write_session(inner, session, tail_facts=TailFacts())
    before = _embedding_rows(inner)

    replay = session.model_copy(deep=True)
    replay.messages[0].thinking_embedding = None
    replay.messages[0].tool_calls[0].bash_embedding = None
    _write_session(inner, replay, tail_facts=TailFacts())

    assert _embedding_rows(inner) == before
