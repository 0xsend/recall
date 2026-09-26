from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import cast

import duckdb
import pytest
import recall.services.fts_sidecar_reconcile as reconcile_module
from recall.core.models import Message, Session, TailFacts, ToolCall
from recall.core.types import Role, Source
from recall.db.fts_sidecar import (
    open_sidecar,
    upsert_message_fts,
    upsert_message_fts_batch,
    upsert_tool_call_fts,
    upsert_tool_call_fts_batch,
)
from recall.db.schema import ensure_schema
from recall.services.fts_sidecar_reconcile import ReconcileStats, reconcile_sidecar
from recall.services.indexer import _write_session


def test_reconcile_drains_pending_upsert(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _seed_message(duckdb_conn, "msg-1", fts_content="hello indexed world")
        _seed_pending(duckdb_conn, "message", "msg-1", "upsert")

        stats = reconcile_sidecar(duckdb_conn, sidecar_conn, batch_size=2)

        assert _message_ids(sidecar_conn) == ["msg-1"]
        assert _matches(sidecar_conn, "message_fts", "indexed") == ["msg-1"]
        assert _pending_rows(duckdb_conn) == []
        assert stats.pending_drained["message"] == 1
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_reconcile_drains_pending_delete(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        upsert_message_fts(sidecar_conn, "msg-1", "delete target", "")
        _seed_pending(duckdb_conn, "message", "msg-1", "delete")

        stats = reconcile_sidecar(duckdb_conn, sidecar_conn, batch_size=2)

        assert _message_ids(sidecar_conn) == []
        assert _count(sidecar_conn, "message_fts") == 0
        assert _pending_rows(duckdb_conn) == []
        assert stats.pending_drained["message"] == 1
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_reconcile_handles_message_deleted_between_enqueue_and_drain(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _seed_pending(duckdb_conn, "message", "msg-x", "upsert")

        stats = reconcile_sidecar(duckdb_conn, sidecar_conn, batch_size=2)

        assert _message_ids(sidecar_conn) == []
        assert _pending_rows(duckdb_conn) == []
        assert stats.pending_drained["message"] == 1
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_reconcile_backfills_orphans(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _seed_messages(duckdb_conn, 5)
        _seed_tool_calls(duckdb_conn, 3)

        stats = reconcile_sidecar(duckdb_conn, sidecar_conn, batch_size=2)

        assert _message_ids(sidecar_conn) == [f"msg-{index:03d}" for index in range(5)]
        assert _tool_call_ids(sidecar_conn) == [f"tc-{index:03d}" for index in range(3)]
        assert stats.orphans_backfilled["message"] == 5
        assert stats.orphans_backfilled["tool_call"] == 3
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_reconcile_skips_tool_calls_with_null_bash_command(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _seed_tool_calls(duckdb_conn, 5, null_bash_ids={"tc-001", "tc-003"})

        stats = reconcile_sidecar(duckdb_conn, sidecar_conn, batch_size=2)

        assert _tool_call_ids(sidecar_conn) == ["tc-000", "tc-002", "tc-004"]
        assert stats.orphans_backfilled["tool_call"] == 3
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_reconcile_deletes_ghosts(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _seed_message(duckdb_conn, "msg-1", fts_content="live row")
        for message_id in ("msg-1", "msg-2", "msg-3"):
            upsert_message_fts(sidecar_conn, message_id, f"content {message_id}", "")

        stats = reconcile_sidecar(duckdb_conn, sidecar_conn, batch_size=2)

        assert _message_ids(sidecar_conn) == ["msg-1"]
        assert _count(sidecar_conn, "message_fts") == 1
        assert stats.ghosts_deleted["message"] == 2
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_reconcile_no_op_on_healthy_db(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _write_session(
            duckdb_conn,
            _session(tool_calls=[_tool_call("tc-1", "printf healthy")]),
            sidecar_conn=sidecar_conn,
            tail_facts=TailFacts(),
        )

        stats = reconcile_sidecar(duckdb_conn, sidecar_conn, batch_size=2)

        assert stats == ReconcileStats(
            pending_drained=_zero_counts(),
            orphans_backfilled=_zero_counts(),
            ghosts_deleted=_zero_counts(),
            pending_remaining=_zero_counts(),
        )
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


@pytest.mark.parametrize(
    "suffixes, orphans, ghosts",
    [
        (["002"], 1, 0),
        (["002", "003", "ghost"], 0, 1),
        (["002", "ghost"], 1, 1),
    ],
)
def test_membership_drift_after_matching_pages_is_repaired(
    tmp_path: Path, suffixes: list[str], orphans: int, ghosts: int
) -> None:
    with _duckdb_with_schema() as conn:
        sidecar = open_sidecar(tmp_path / "recall.fts.sqlite")
        try:
            _seed_messages(conn, 4)
            for suffix in ["000", "001", *suffixes]:
                content = f"message content {int(suffix)}" if suffix.isdigit() else "obsolete"
                upsert_message_fts(sidecar, f"msg-{suffix}", content, "")
            stats = reconcile_sidecar(conn, sidecar, batch_size=2)
            assert _message_ids(sidecar) == ["msg-000", "msg-001", "msg-002", "msg-003"]
            assert _matches(sidecar, "message_fts", "obsolete") == []
            assert stats.orphans_backfilled == {"message": orphans, "tool_call": 0}
            assert stats.ghosts_deleted == {"message": ghosts, "tool_call": 0}
        finally:
            sidecar.close()


def test_membership_repair_observes_uncommitted_primary_facts(tmp_path: Path) -> None:
    with _duckdb_with_schema() as conn:
        sidecar = open_sidecar(tmp_path / "recall.fts.sqlite")
        try:
            conn.execute("BEGIN")
            _seed_message(conn, "uncommitted", fts_content="visible before commit")
            stats = reconcile_sidecar(conn, sidecar, batch_size=2)
            assert stats.orphans_backfilled == {"message": 1, "tool_call": 0}
            assert _matches(sidecar, "message_fts", "visible") == ["uncommitted"]
            conn.execute("ROLLBACK")
            assert conn.execute("SELECT COUNT(*) FROM message_state").fetchone() == (0,)
            # Sidecar publication is independent; the next pass repairs the rollback ghost.
            stats = reconcile_sidecar(conn, sidecar, batch_size=2)
            assert stats.ghosts_deleted == {"message": 1, "tool_call": 0}
            assert _message_ids(sidecar) == []
        finally:
            sidecar.close()


@pytest.mark.parametrize("kind", ["message", "tool_call"])
def test_healthy_membership_checks_have_linear_scan_work(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], kind: str
) -> None:
    conn = _duckdb_with_schema()
    sidecar = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        if kind == "message":
            _seed_messages(conn, 513)
            rows = conn.execute(
                "SELECT message_id, fts_content, fts_thinking FROM message_state"
            ).fetchall()
            upsert_message_fts_batch(sidecar, rows)
        else:
            _seed_tool_calls(conn, 513)
            rows = conn.execute("SELECT id, bash_command FROM tool_calls").fetchall()
            upsert_tool_call_fts_batch(sidecar, rows)
        capfd.readouterr()
        conn.execute("PRAGMA enable_profiling='json'")
        stats = reconcile_sidecar(conn, sidecar, batch_size=128)
        conn.execute("PRAGMA disable_profiling")
        raw = capfd.readouterr().err.strip()
        profiles = []
        decoder = json.JSONDecoder()
        while raw:
            profile, end = decoder.raw_decode(raw)
            profiles.append(profile)
            raw = raw[end:].lstrip()
        assert profiles, "DuckDB must emit actual scan-work measurements"
        # Returning smaller pages must not cause repeated full-table scans.
        assert sum(profile["cumulative_rows_scanned"] for profile in profiles) <= 2 * 513
        assert stats == ReconcileStats(
            pending_drained=_zero_counts(),
            orphans_backfilled=_zero_counts(),
            ghosts_deleted=_zero_counts(),
            pending_remaining=_zero_counts(),
        )
    finally:
        sidecar.close()
        conn.close()


@pytest.mark.parametrize("kind", ["message", "tool_call"])
def test_reconcile_large_existing_documents_with_bounded_memory(tmp_path: Path, kind: str) -> None:
    database = tmp_path / "recall.duckdb"
    sidecar = open_sidecar(tmp_path / "recall.fts.sqlite")
    conn = duckdb.connect(str(database))
    try:
        ensure_schema(conn)
        # The existing text exceeds the repair connection's memory budget;
        # membership repair must still preserve it and repair the small gaps.
        if kind == "message":
            conn.execute(
                """INSERT INTO message_state (message_id, role, fts_content, fts_thinking)
                   SELECT printf('doc-%05d', i), 'assistant', repeat(md5(i::VARCHAR), 2048), ''
                   FROM range(2048) AS rows(i)"""
            )
            rows = conn.execute("SELECT message_id, fts_content, fts_thinking FROM message_state")
            while batch := rows.fetchmany(256):
                upsert_message_fts_batch(sidecar, batch)
            _seed_message(conn, "missing", fts_content="restored message")
            upsert_message_fts(sidecar, "ghost", "obsolete", "")
        else:
            conn.execute(
                """INSERT INTO tool_calls (id, session_id, idx, tool_name, bash_command)
                   SELECT printf('doc-%05d', i), 'session', i, 'bash',
                          repeat(md5(i::VARCHAR), 2048)
                   FROM range(2048) AS rows(i)"""
            )
            rows = conn.execute("SELECT id, bash_command FROM tool_calls")
            while batch := rows.fetchmany(256):
                upsert_tool_call_fts_batch(sidecar, batch)
            conn.execute(
                """INSERT INTO tool_calls (id, session_id, idx, tool_name, bash_command)
                   VALUES ('missing', 'session', 2048, 'bash', 'echo restored'),
                          ('ghost', 'session', 2049, 'Read', NULL)"""
            )
            upsert_tool_call_fts(sidecar, "ghost", "obsolete")
        conn.close()
        conn = duckdb.connect(str(database), config={"memory_limit": "48MB", "threads": 2})

        stats = reconcile_sidecar(conn, sidecar, batch_size=1024)

        assert stats.orphans_backfilled[kind] == 1
        assert stats.ghosts_deleted[kind] == 1
        assert stats.pending_remaining == {"message": 0, "tool_call": 0}
        table = "message_fts" if kind == "message" else "tool_calls_fts"
        assert _count(sidecar, table) == 2049
        assert _matches(sidecar, table, "restored") == ["missing"]
        assert _matches(sidecar, table, "obsolete") == []
        again = reconcile_sidecar(conn, sidecar, batch_size=1024)
        assert again.orphans_backfilled == {"message": 0, "tool_call": 0}
        assert again.ghosts_deleted == {"message": 0, "tool_call": 0}
    finally:
        sidecar.close()
        conn.close()


def test_reconcile_leaves_pending_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _seed_message(duckdb_conn, "msg-1", fts_content="will fail")
        _seed_pending(duckdb_conn, "message", "msg-1", "upsert")

        def fail_upsert(*_args: object, **_kwargs: object) -> None:
            raise sqlite3.Error("boom")

        monkeypatch.setattr(reconcile_module, "upsert_message_fts", fail_upsert)

        stats = reconcile_sidecar(duckdb_conn, sidecar_conn, batch_size=2)

        assert _pending_rows(duckdb_conn) == [("message", "msg-1", "upsert")]
        assert _message_ids(sidecar_conn) == []
        assert stats.pending_remaining["message"] == 1
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_reconcile_drains_pending_in_keyset_batches(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        batch_size = 2
        row_count = 5
        for index in range(row_count):
            message_id = f"msg-{index:03d}"
            _seed_message(duckdb_conn, message_id, fts_content=f"batched content {index}")
            _seed_pending(duckdb_conn, "message", message_id, "upsert")

        spy_conn = _DuckDBExecuteSpy(duckdb_conn)

        stats = reconcile_sidecar(
            cast(duckdb.DuckDBPyConnection, spy_conn),
            sidecar_conn,
            batch_size=batch_size,
        )

        assert _message_ids(sidecar_conn) == [f"msg-{index:03d}" for index in range(row_count)]
        assert _pending_rows(duckdb_conn) == []
        assert stats.pending_drained == {"message": row_count, "tool_call": 0}
        assert stats.pending_remaining == _zero_counts()
        assert spy_conn.pending_fetch_sizes == [2, 2, 1]
        assert all(" LIMIT " in sql.upper() for sql in spy_conn.pending_fetch_sql)

    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_reconcile_pending_cursor_advances_past_failed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        batch_size = 2
        message_ids = ["msg-ok-1", "msg-fail", "msg-ok-2"]
        for message_id in message_ids:
            _seed_message(duckdb_conn, message_id, fts_content=f"content {message_id}")
            _seed_pending(duckdb_conn, "message", message_id, "upsert")

        attempts: list[str] = []
        original_upsert = reconcile_module.upsert_message_fts

        def fail_one_message(
            conn: sqlite3.Connection,
            message_id: str,
            content: str,
            thinking: str,
            *,
            fields: tuple[str, ...] | None = None,
        ) -> None:
            attempts.append(message_id)
            if message_id == "msg-fail":
                raise sqlite3.Error("boom")
            original_upsert(conn, message_id, content, thinking, fields=fields)

        monkeypatch.setattr(reconcile_module, "upsert_message_fts", fail_one_message)
        spy_conn = _DuckDBExecuteSpy(duckdb_conn)

        stats = reconcile_sidecar(
            cast(duckdb.DuckDBPyConnection, spy_conn),
            sidecar_conn,
            batch_size=batch_size,
        )

        assert attempts == message_ids
        assert _message_ids(sidecar_conn) == ["msg-ok-1", "msg-ok-2"]
        assert _pending_rows(duckdb_conn) == [("message", "msg-fail", "upsert")]
        assert stats.pending_drained == {"message": 2, "tool_call": 0}
        assert stats.pending_remaining == {"message": 1, "tool_call": 0}
        assert spy_conn.pending_fetch_sizes == [2, 1]

    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def _duckdb_with_schema() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    return conn


def _seed_messages(conn: duckdb.DuckDBPyConnection, count: int) -> None:
    for index in range(count):
        _seed_message(conn, f"msg-{index:03d}", fts_content=f"message content {index}")


def _seed_message(
    conn: duckdb.DuckDBPyConnection,
    message_id: str,
    *,
    fts_content: str,
) -> None:
    conn.execute(
        """
        INSERT INTO message_state (
            message_id, role, content, thinking, has_thinking, fts_content, fts_thinking
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            message_id,
            "assistant",
            f"raw {message_id}",
            f"thinking {message_id}",
            True,
            fts_content,
            f"thoughts {message_id}",
        ],
    )


def _seed_tool_calls(
    conn: duckdb.DuckDBPyConnection,
    count: int,
    *,
    null_bash_ids: set[str] | None = None,
) -> None:
    null_bash_ids = null_bash_ids or set()
    for index in range(count):
        tool_call_id = f"tc-{index:03d}"
        bash_command = None if tool_call_id in null_bash_ids else f"echo {index}"
        conn.execute(
            """
            INSERT INTO tool_calls (
                id, session_id, message_id, idx, tool_name, bash_command, is_compound
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                tool_call_id,
                "session-1",
                None,
                index,
                "bash",
                bash_command,
                False,
            ],
        )


def _seed_pending(
    conn: duckdb.DuckDBPyConnection,
    kind: str,
    entity_id: str,
    op: str,
) -> None:
    conn.execute(
        "INSERT INTO fts_sidecar_pending(kind, id, op) VALUES (?, ?, ?)",
        [kind, entity_id, op],
    )


def _session(*, tool_calls: list[ToolCall] | None = None) -> Session:
    message = Message(
        id="msg-1",
        session_id="session-1",
        idx=0,
        role=Role.ASSISTANT,
        content="healthy content",
    )
    message.tool_calls = tool_calls or []
    return Session(
        id="session-1",
        source=Source.CODEX,
        source_path="/tmp/session.jsonl",
        source_session_id="source-session-1",
        file_mtime=1.0,
        file_size=100,
        message_count=1,
        tool_count=len(message.tool_calls),
        messages=[message],
    )


def _tool_call(tool_call_id: str, bash_command: str | None) -> ToolCall:
    return ToolCall(
        id=tool_call_id,
        session_id="session-1",
        message_id="msg-1",
        idx=0,
        tool_name="bash",
        bash_command=bash_command,
    )


def _message_ids(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT message_id FROM message_fts_rowid ORDER BY message_id"
        ).fetchall()
    ]


def _tool_call_ids(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT tool_call_id FROM tool_calls_fts_rowid ORDER BY tool_call_id"
        ).fetchall()
    ]


def _pending_rows(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str]]:
    return [
        (str(row[0]), str(row[1]), str(row[2]))
        for row in conn.execute(
            """
            SELECT kind, id, op
            FROM fts_sidecar_pending
            ORDER BY queued_at, kind, id, op
            """
        ).fetchall()
    ]


def _matches(conn: sqlite3.Connection, table: str, term: str) -> list[str]:
    if table == "message_fts":
        rows = conn.execute(
            """
            SELECT m.message_id
            FROM message_fts f
            JOIN message_fts_rowid m ON m.rowid = f.rowid
            WHERE message_fts MATCH ?
            ORDER BY m.message_id
            """,
            [term],
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT t.tool_call_id
            FROM tool_calls_fts f
            JOIN tool_calls_fts_rowid t ON t.rowid = f.rowid
            WHERE tool_calls_fts MATCH ?
            ORDER BY t.tool_call_id
            """,
            [term],
        ).fetchall()
    return [str(row[0]) for row in rows]


def _count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def _zero_counts() -> dict[str, int]:
    return {"message": 0, "tool_call": 0}


class _DuckDBExecuteSpy:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn
        self.pending_fetch_sql: list[str] = []
        self.pending_fetch_sizes: list[int] = []

    def execute(self, query: str, parameters: object | None = None) -> object:
        if parameters is None:
            result = self._conn.execute(query)
        else:
            result = self._conn.execute(query, parameters)
        if _is_pending_ordered_fetch(query):
            return _FetchAllSpyCursor(result, query, self._record_pending_fetch)
        return result

    def _record_pending_fetch(self, query: str, row_count: int) -> None:
        self.pending_fetch_sql.append(query)
        self.pending_fetch_sizes.append(row_count)

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)


class _FetchAllSpyCursor:
    def __init__(
        self,
        cursor: duckdb.DuckDBPyConnection,
        query: str,
        on_fetchall: Callable[[str, int], None],
    ) -> None:
        self._cursor = cursor
        self._query = query
        self._on_fetchall = on_fetchall

    def fetchall(self) -> list[tuple[object, ...]]:
        rows = self._cursor.fetchall()
        self._on_fetchall(self._query, len(rows))
        return rows

    def fetchone(self) -> object:
        return self._cursor.fetchone()

    def __getattr__(self, name: str) -> object:
        return getattr(self._cursor, name)


def _is_pending_ordered_fetch(query: str) -> bool:
    normalized = " ".join(query.split()).upper()
    return "FROM FTS_SIDECAR_PENDING" in normalized and "ORDER BY QUEUED_AT, ROWID" in normalized
