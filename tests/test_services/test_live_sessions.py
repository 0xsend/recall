"""The daemon live set joined to the index (REQ-LIVE-001).

Liveness is derived, never declared: the in-memory live set is the only thing
that knows a transcript is being written to right now. This layer only has to
get the join right — which watched path is which indexed session, what the
answer is for a path the index has never seen, and what it is when the daemon
is not watching at all.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import duckdb
from recall.db.schema import ensure_schema
from recall.services.live import live_session_rows
from recall.services.live_session_set import LiveMember
from recall.services.rpc_server import RpcServer

INDEXED_PATH = "/home/dev/.claude/projects/proj/indexed.jsonl"
SESSION_ID = "11223344556677889900aabbccddeeff"


def _make_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    return conn


def _insert_indexed_session(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        "INSERT INTO sessions (id, source, source_path, source_session_id)"
        " VALUES (?, 'claude_code', ?, 'live-lane-1')",
        [SESSION_ID, INDEXED_PATH],
    )
    conn.execute(
        "INSERT INTO session_state (session_id, file_mtime, file_size, host, indexed_at)"
        " VALUES (?, 1788782400.0, 10896104, 'laptop', TIMESTAMP '2026-09-07 12:00:00')",
        [SESSION_ID],
    )


def _member(path: str, *, mtime: float, last_event_at: float) -> LiveMember:
    return LiveMember(
        path=Path(path),
        mtime=mtime,
        last_event_at=last_event_at,
        subscription_key=str(Path(path).parent),
    )


def test_a_watched_path_carries_the_indexed_session_it_belongs_to() -> None:
    conn = _make_conn()
    _insert_indexed_session(conn)

    rows = live_session_rows(
        [_member(INDEXED_PATH, mtime=1788786000.0, last_event_at=4210.5)],
        conn=conn,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.path == INDEXED_PATH
    assert row.mtime == 1788786000.0
    assert row.last_event_at == 4210.5
    assert row.session_id == SESSION_ID
    assert row.source == "claude_code"
    assert row.source_session_id == "live-lane-1"
    assert row.host == "laptop"
    assert row.indexed_mtime == 1788782400.0
    assert row.indexed_size == 10896104
    assert row.indexed_at == datetime(2026, 9, 7, 12, 0, 0)


def test_a_watched_path_the_index_has_never_seen_still_appears() -> None:
    """The daemon promotes a transcript on its first write, before any index pass.

    Dropping it here would make a session invisible for exactly the window in
    which it is most obviously live.
    """
    conn = _make_conn()

    rows = live_session_rows(
        [_member("/home/dev/.claude/projects/proj/brand-new.jsonl", mtime=99.0, last_event_at=1.0)],
        conn=conn,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.path == "/home/dev/.claude/projects/proj/brand-new.jsonl"
    assert row.mtime == 99.0
    assert row.session_id is None
    assert row.source is None
    assert row.indexed_mtime is None
    assert row.indexed_at is None


def test_rows_keep_the_live_sets_most_recently_modified_first_order() -> None:
    conn = _make_conn()
    _insert_indexed_session(conn)

    rows = live_session_rows(
        [
            _member(INDEXED_PATH, mtime=1788786000.0, last_event_at=4210.5),
            _member("/home/dev/.claude/projects/proj/older.jsonl", mtime=17.0, last_event_at=8.0),
        ],
        conn=conn,
    )

    assert [row.path for row in rows] == [
        INDEXED_PATH,
        "/home/dev/.claude/projects/proj/older.jsonl",
    ]


def test_live_sessions_rpc_reports_that_it_is_not_watching_rather_than_an_empty_set() -> None:
    """A daemon with no live set answers "not watching", not "nothing is live".

    A daemon in poll mode has no live set, so it cannot say a session is
    inactive — reporting an empty list would hand the caller a fact recall
    does not have.
    """
    server = RpcServer()

    result = asyncio.run(server._handle_live_sessions({}, None))

    assert result.watching is False
    assert result.sessions == ()
    assert result.idle_threshold_seconds > 0
