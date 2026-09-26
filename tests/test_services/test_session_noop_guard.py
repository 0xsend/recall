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

from datetime import UTC, datetime
from typing import Any, cast

import duckdb
from recall.core.models import Message, Session, TailFacts
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
        "UPDATE session_state SET input_tokens = 4200, output_tokens = 900 "
        "WHERE session_id = 'noop-session'"
    )

    conn.sql.clear()
    _rewrite(conn, _session(input_tokens=None, output_tokens=None))

    assert conn.session_state_updates() == [], (
        "Grok null-token re-index rewrote session_state (null-preserve should be a no-op)"
    )
    tokens = inner.execute(
        "SELECT input_tokens, output_tokens FROM session_state WHERE session_id = 'noop-session'"
    ).fetchone()
    assert tokens == (4200, 900), "harvest-filled tokens were clobbered on re-index"


def test_host_change_still_reindexes_host() -> None:
    conn, inner = _fresh_conn()
    _rewrite(conn, _session(), host="host-a")

    conn.sql.clear()
    _rewrite(conn, _session(), host="host-b")

    assert any("SET host" in s for s in conn.session_state_updates()), (
        "a real host change must still write session_state.host (REQ-MULTIHOST-002)"
    )
    assert _col(inner, "host") == "host-b"
