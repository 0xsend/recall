"""Tests for list_sessions --since filter semantics.

The --since filter should match sessions by their "last active" time,
using COALESCE(ended_at, started_at, indexed_at). A session that started
hours ago but ended recently should appear with a short --since window.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import duckdb
import pytest
from recall.db import ensure_schema
from recall.services.sessions import list_sessions


@pytest.fixture
def db_with_sessions(tmp_path):
    """Create a DB with three sessions spanning different time ranges."""
    db_path = tmp_path / "recall.duckdb"
    conn = duckdb.connect(str(db_path))
    ensure_schema(conn)

    now = datetime.now(UTC)

    # Session A: started 3h ago, ended 10min ago (recently active)
    conn.execute(
        "INSERT INTO sessions (id, source, source_path) VALUES (?, ?, ?)",
        ["sess_a", "claude_code", "/tmp/a.jsonl"],
    )
    conn.execute(
        """INSERT INTO session_state
           (session_id, started_at, ended_at, message_count, tool_count,
            is_complete, file_mtime, file_size)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ["sess_a", now - timedelta(hours=3), now - timedelta(minutes=10), 50, 20, True, 0.0, 100],
    )

    # Session B: started 30min ago, ended 20min ago
    conn.execute(
        "INSERT INTO sessions (id, source, source_path) VALUES (?, ?, ?)",
        ["sess_b", "claude_code", "/tmp/b.jsonl"],
    )
    conn.execute(
        """INSERT INTO session_state
           (session_id, started_at, ended_at, message_count, tool_count,
            is_complete, file_mtime, file_size)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ["sess_b", now - timedelta(minutes=30), now - timedelta(minutes=20), 10, 5, True, 0.0, 100],
    )

    # Session C: started 5h ago, ended 4h ago (old)
    conn.execute(
        "INSERT INTO sessions (id, source, source_path) VALUES (?, ?, ?)",
        ["sess_c", "claude_code", "/tmp/c.jsonl"],
    )
    conn.execute(
        """INSERT INTO session_state
           (session_id, started_at, ended_at, message_count, tool_count,
            is_complete, file_mtime, file_size)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ["sess_c", now - timedelta(hours=5), now - timedelta(hours=4), 5, 2, True, 0.0, 100],
    )

    conn.execute("CHECKPOINT")
    conn.close()
    return db_path, now


def test_since_filters_by_ended_at(db_with_sessions):
    """Session that started 3h ago but ended 10min ago appears with --since 1h."""
    db_path, now = db_with_sessions
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        cutoff = now - timedelta(hours=1)
        results = list_sessions(source=None, since=cutoff, project=None, conn=conn)
        ids = {r.id for r in results}
        assert "sess_a" in ids, "session that ended recently should match --since 1h"
        assert "sess_b" in ids, "session that ended 20min ago should match --since 1h"
        assert "sess_c" not in ids, "session that ended 4h ago should not match --since 1h"
    finally:
        conn.close()


def test_since_excludes_old_sessions(db_with_sessions):
    """Sessions ended outside the --since window are excluded."""
    db_path, now = db_with_sessions
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        cutoff = now - timedelta(minutes=15)
        results = list_sessions(source=None, since=cutoff, project=None, conn=conn)
        ids = {r.id for r in results}
        assert "sess_a" in ids, "session A ended 10min ago, within 15min window"
        assert "sess_b" not in ids, "session B ended 20min ago, outside 15min window"
        assert "sess_c" not in ids, "session C ended 4h ago, outside 15min window"
    finally:
        conn.close()


def test_since_with_no_ended_at_falls_back_to_started_at(tmp_path):
    """When ended_at is NULL, since filter falls back to started_at."""
    db_path = tmp_path / "recall.duckdb"
    conn = duckdb.connect(str(db_path))
    ensure_schema(conn)
    now = datetime.now(UTC)

    # In-progress session, started 30min ago, no ended_at
    conn.execute(
        "INSERT INTO sessions (id, source, source_path) VALUES (?, ?, ?)",
        ["sess_active", "claude_code", "/tmp/active.jsonl"],
    )
    conn.execute(
        """INSERT INTO session_state
           (session_id, started_at, ended_at, message_count, tool_count,
            is_complete, file_mtime, file_size)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ["sess_active", now - timedelta(minutes=30), None, 10, 5, False, 0.0, 100],
    )
    conn.execute("CHECKPOINT")
    conn.close()

    ro_conn = duckdb.connect(str(db_path), read_only=True)
    try:
        cutoff = now - timedelta(hours=1)
        results = list_sessions(source=None, since=cutoff, project=None, conn=ro_conn)
        ids = {r.id for r in results}
        assert "sess_active" in ids, "in-progress session started 30min ago should match --since 1h"
    finally:
        ro_conn.close()


def test_list_sessions_returns_model_and_token_counts(tmp_path):
    """list output includes model and token totals from session_state."""
    db_path = tmp_path / "recall.duckdb"
    conn = duckdb.connect(str(db_path))
    ensure_schema(conn)
    now = datetime.now(UTC)

    conn.execute(
        "INSERT INTO sessions (id, source, source_path) VALUES (?, ?, ?)",
        ["sess_rich", "kimi_code", "/tmp/rich.jsonl"],
    )
    conn.execute(
        """INSERT INTO session_state
           (session_id, started_at, ended_at, model, message_count, tool_count,
            input_tokens, output_tokens, is_complete, file_mtime, file_size)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            "sess_rich",
            now - timedelta(minutes=10),
            now - timedelta(minutes=5),
            "kimi-code/k3",
            42,
            7,
            123456,
            789,
            True,
            0.0,
            100,
        ],
    )
    # Session with unknown model/tokens must surface None, not zeros.
    conn.execute(
        "INSERT INTO sessions (id, source, source_path) VALUES (?, ?, ?)",
        ["sess_bare", "grok", "/tmp/bare.jsonl"],
    )
    conn.execute(
        """INSERT INTO session_state
           (session_id, started_at, message_count, tool_count,
            is_complete, file_mtime, file_size)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ["sess_bare", now - timedelta(minutes=2), 1, 0, True, 0.0, 100],
    )
    conn.execute("CHECKPOINT")
    conn.close()

    ro_conn = duckdb.connect(str(db_path), read_only=True)
    try:
        results = {
            r.id: r for r in list_sessions(source=None, since=None, project=None, conn=ro_conn)
        }
        rich = results["sess_rich"]
        assert rich.model == "kimi-code/k3"
        assert rich.input_tokens == 123456
        assert rich.output_tokens == 789
        bare = results["sess_bare"]
        assert bare.model is None
        assert bare.input_tokens is None
        assert bare.output_tokens is None
    finally:
        ro_conn.close()


def test_list_sessions_includes_host(tmp_path):
    """REQ-HOST-API-001: list returns session_state.host on every summary."""
    db_path = tmp_path / "recall.duckdb"
    conn = duckdb.connect(str(db_path))
    ensure_schema(conn)
    now = datetime.now(UTC)

    conn.execute(
        "INSERT INTO sessions (id, source, source_path) VALUES (?, ?, ?)",
        ["sess_host", "claude_code", "/tmp/host.jsonl"],
    )
    conn.execute(
        """INSERT INTO session_state
           (session_id, started_at, ended_at, message_count, tool_count,
            is_complete, file_mtime, file_size, host)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            "sess_host",
            now - timedelta(minutes=5),
            now - timedelta(minutes=1),
            3,
            1,
            True,
            0.0,
            100,
            "devbox",
        ],
    )
    conn.execute("CHECKPOINT")
    conn.close()

    ro_conn = duckdb.connect(str(db_path), read_only=True)
    try:
        results = list_sessions(source=None, since=None, project=None, conn=ro_conn)
        assert len(results) == 1
        assert results[0].host == "devbox"
    finally:
        ro_conn.close()


def test_list_sessions_host_fallback_when_empty(tmp_path):
    """REQ-HOST-API-004: empty host falls back to a non-empty label."""
    db_path = tmp_path / "recall.duckdb"
    conn = duckdb.connect(str(db_path))
    ensure_schema(conn)
    now = datetime.now(UTC)

    conn.execute(
        "INSERT INTO sessions (id, source, source_path) VALUES (?, ?, ?)",
        ["sess_empty_host", "grok", "/tmp/empty-host.jsonl"],
    )
    # DDL default is 'local'; force empty string to exercise fallback.
    conn.execute(
        """INSERT INTO session_state
           (session_id, started_at, message_count, tool_count,
            is_complete, file_mtime, file_size, host)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ["sess_empty_host", now, 1, 0, True, 0.0, 50, ""],
    )
    conn.execute("CHECKPOINT")
    conn.close()

    ro_conn = duckdb.connect(str(db_path), read_only=True)
    try:
        results = list_sessions(source=None, since=None, project=None, conn=ro_conn)
        assert len(results) == 1
        assert results[0].host  # non-empty
        assert results[0].host == "local"
    finally:
        ro_conn.close()


def test_list_sessions_filters_by_host(tmp_path):
    """REQ-HOST-API-005: --host exact match on session_state.host."""
    db_path = tmp_path / "recall.duckdb"
    conn = duckdb.connect(str(db_path))
    ensure_schema(conn)
    now = datetime.now(UTC)

    for sid, host in [("a", "devbox"), ("b", "buildbox")]:
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) VALUES (?, ?, ?)",
            [sid, "claude_code", f"/tmp/{sid}.jsonl"],
        )
        conn.execute(
            """INSERT INTO session_state
               (session_id, started_at, message_count, tool_count,
                is_complete, file_mtime, file_size, host)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [sid, now, 1, 0, True, 0.0, 10, host],
        )
    conn.execute("CHECKPOINT")
    conn.close()

    ro_conn = duckdb.connect(str(db_path), read_only=True)
    try:
        results = list_sessions(source=None, since=None, project=None, host="devbox", conn=ro_conn)
        assert [r.id for r in results] == ["a"]
        assert results[0].host == "devbox"
    finally:
        ro_conn.close()


def test_list_sessions_returns_freshness_columns(tmp_path):
    """REQ-LIVE-003: file_mtime, file_size and indexed_at ride on every summary."""
    db_path = tmp_path / "recall.duckdb"
    conn = duckdb.connect(str(db_path))
    ensure_schema(conn)

    conn.execute(
        "INSERT INTO sessions (id, source, source_path) VALUES (?, ?, ?)",
        ["sess_fresh", "claude_code", "/tmp/fresh.jsonl"],
    )
    conn.execute(
        """INSERT INTO session_state
           (session_id, started_at, message_count, tool_count,
            is_complete, file_mtime, file_size, indexed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            "sess_fresh",
            datetime(2026, 9, 7, 12, 0, 0),
            4,
            1,
            False,
            1757246400.5,
            10896104,
            datetime(2026, 9, 7, 12, 30, 0),
        ],
    )
    conn.execute("CHECKPOINT")
    conn.close()

    ro_conn = duckdb.connect(str(db_path), read_only=True)
    try:
        results = list_sessions(source=None, since=None, project=None, conn=ro_conn)
        assert len(results) == 1
        row = results[0]
        assert row.file_mtime == 1757246400.5
        assert row.file_size == 10896104
        assert row.indexed_at == datetime(2026, 9, 7, 12, 30, 0)
    finally:
        ro_conn.close()


def _connect_utc(db_path, *, read_only: bool):
    """Open a connection pinned to UTC so epoch/timestamp casts are literal.

    ``session_state.ended_at`` is a naive DuckDB ``TIMESTAMP`` holding local
    wall clock, and ``file_mtime`` is a Unix epoch, so ``last_activity_at``
    depends on the connection's TimeZone. Pinning it keeps expectations exact
    on any host.
    """
    conn = duckdb.connect(str(db_path), read_only=read_only)
    conn.execute("SET TimeZone='UTC'")
    return conn


def _insert_session(conn, session_id, *, ended_at, file_mtime):
    conn.execute(
        "INSERT INTO sessions (id, source, source_path) VALUES (?, ?, ?)",
        [session_id, "claude_code", f"/tmp/{session_id}.jsonl"],
    )
    conn.execute(
        """INSERT INTO session_state
           (session_id, started_at, ended_at, message_count, tool_count,
            is_complete, file_mtime, file_size)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            session_id,
            datetime(2026, 9, 7, 10, 0, 0),
            ended_at,
            3,
            1,
            False,
            file_mtime,
            100,
        ],
    )


def test_last_activity_at_takes_the_newer_of_ended_at_and_file_mtime(tmp_path):
    """REQ-LIVE-002: a live session's mtime outruns its last indexed message."""
    db_path = tmp_path / "recall.duckdb"
    conn = _connect_utc(db_path, read_only=False)
    ensure_schema(conn)

    # epoch 1788782400 == 2026-09-07T12:00:00Z; 1788786000 == 2026-09-07T13:00:00Z
    _insert_session(
        conn, "mtime_ahead", ended_at=datetime(2026, 9, 7, 12, 0, 0), file_mtime=1788786000.0
    )
    _insert_session(
        conn, "ended_ahead", ended_at=datetime(2026, 9, 7, 14, 0, 0), file_mtime=1788782400.0
    )
    _insert_session(conn, "never_ended", ended_at=None, file_mtime=1788782400.0)
    conn.execute("CHECKPOINT")
    conn.close()

    ro_conn = _connect_utc(db_path, read_only=True)
    try:
        rows = {r.id: r for r in list_sessions(source=None, since=None, project=None, conn=ro_conn)}
        assert rows["mtime_ahead"].last_activity_at == datetime(2026, 9, 7, 13, 0, 0)
        assert rows["ended_ahead"].last_activity_at == datetime(2026, 9, 7, 14, 0, 0)
        assert rows["never_ended"].last_activity_at == datetime(2026, 9, 7, 12, 0, 0)
    finally:
        ro_conn.close()
