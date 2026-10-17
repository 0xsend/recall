from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import duckdb
import pytest
import recall.db.fts_sidecar_buffer as buffer_module
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.core.models import Message, Session, TailFacts, ToolCall
from recall.core.types import Role, Source
from recall.db.fts_sidecar import open_sidecar, sidecar_path
from recall.db.fts_sidecar_buffer import SidecarBuffer
from recall.db.schema import ensure_schema
from recall.services.fts_sidecar_reconcile import reconcile_sidecar, repair_pending_batch
from recall.services.indexer import _write_session, index_sessions


def test_committed_write_reaches_sidecar_only_at_flush(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    buffer = SidecarBuffer(_sidecar(tmp_path))
    try:
        _write_session(
            duckdb_conn,
            _session(content="hello indexed world", tool_calls=[_tool_call("tc-1", "echo hi")]),
            keywords=buffer,
            tail_facts=TailFacts(),
        )

        # Committed and durable as intents, not yet searchable.
        assert _pending(duckdb_conn) == [
            ("message", "msg-1", "upsert"),
            ("tool_call", "tc-1", "upsert"),
        ]
        assert _matches(tmp_path, "message_fts", "indexed") == []

        assert buffer.flush(duckdb_conn) == 2

        assert _matches(tmp_path, "message_fts", "indexed") == ["msg-1"]
        assert _matches(tmp_path, "tool_calls_fts", "echo") == ["tc-1"]
        assert _pending(duckdb_conn) == []
    finally:
        duckdb_conn.close()


def test_one_flush_coalesces_many_session_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    buffer = SidecarBuffer(_sidecar(tmp_path))
    commits = _count_sidecar_commits(monkeypatch)
    try:
        for index in range(5):
            _write_session(
                duckdb_conn,
                _session(content=f"chunk{index} text", session_id=f"session-{index}"),
                keywords=buffer,
                tail_facts=TailFacts(),
            )
        buffer.flush(duckdb_conn)

        assert commits == [5]
        assert _matches(tmp_path, "message_fts", "chunk3") == ["session-3-msg-1"]
    finally:
        duckdb_conn.close()


def test_flush_bounds_rows_per_sqlite_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    buffer = SidecarBuffer(_sidecar(tmp_path))
    monkeypatch.setattr(buffer_module, "FTS_SIDECAR_BATCH_SIZE", 2)
    commits = _count_sidecar_commits(monkeypatch)
    try:
        _write_session(
            duckdb_conn,
            _session(
                content="bounded",
                tool_calls=[_tool_call(f"tc-{index}", f"echo {index}") for index in range(4)],
            ),
            keywords=buffer,
            tail_facts=TailFacts(),
        )
        buffer.flush(duckdb_conn)

        # The message, then the four tool calls in pairs: no commit exceeds the bound.
        assert commits == [1, 2, 2]
        assert _matches(tmp_path, "tool_calls_fts", "echo") == ["tc-0", "tc-1", "tc-2", "tc-3"]
    finally:
        duckdb_conn.close()


def test_update_keeps_rowid_and_replaces_text(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    buffer = SidecarBuffer(_sidecar(tmp_path))
    try:
        _write_session(
            duckdb_conn,
            _session(content="initial alpha", tool_calls=[_tool_call("tc-1", "echo first")]),
            keywords=buffer,
            tail_facts=TailFacts(),
        )
        buffer.flush(duckdb_conn)
        original = (_rowid(tmp_path, "message", "msg-1"), _rowid(tmp_path, "tool_call", "tc-1"))

        _write_session(
            duckdb_conn,
            _session(content="updated beta", tool_calls=[_tool_call("tc-1", "printf second")]),
            keywords=buffer,
            tail_facts=TailFacts(),
        )
        buffer.flush(duckdb_conn)

        assert (_rowid(tmp_path, "message", "msg-1"), _rowid(tmp_path, "tool_call", "tc-1")) == (
            original
        )
        assert _matches(tmp_path, "message_fts", "initial") == []
        assert _matches(tmp_path, "message_fts", "updated") == ["msg-1"]
        assert _matches(tmp_path, "tool_calls_fts", "first") == []
        assert _matches(tmp_path, "tool_calls_fts", "second") == ["tc-1"]
    finally:
        duckdb_conn.close()


def test_null_bash_command_writes_no_tool_row(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    buffer = SidecarBuffer(_sidecar(tmp_path))
    try:
        _write_session(
            duckdb_conn,
            _session(content="no shell", tool_calls=[_tool_call("tc-null", None)]),
            keywords=buffer,
            tail_facts=TailFacts(),
        )
        buffer.flush(duckdb_conn)

        assert _rowid(tmp_path, "tool_call", "tc-null") is None
        assert _pending(duckdb_conn) == []
    finally:
        duckdb_conn.close()


def test_removed_rows_leave_the_sidecar(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    buffer = SidecarBuffer(_sidecar(tmp_path))
    try:
        _write_session(
            duckdb_conn,
            _session(content="delete me", tool_calls=[_tool_call("tc-1", "rm target")]),
            keywords=buffer,
            tail_facts=TailFacts(),
        )
        buffer.flush(duckdb_conn)

        _write_session(
            duckdb_conn,
            _session(content="", message_ids=[]),
            keywords=buffer,
            tail_facts=TailFacts(),
        )
        buffer.flush(duckdb_conn)

        assert _rowid(tmp_path, "message", "msg-1") is None
        assert _rowid(tmp_path, "tool_call", "tc-1") is None
        assert _pending(duckdb_conn) == []
    finally:
        duckdb_conn.close()


def test_rewritten_identity_replaces_rows_in_order(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    buffer = SidecarBuffer(_sidecar(tmp_path))
    try:
        _write_session(
            duckdb_conn, _session(content="before rewrite"), keywords=buffer, tail_facts=TailFacts()
        )
        # A changed source identity deletes and reinserts the same ids in one
        # transaction; one flush must apply the delete before the reinsert.
        _write_session(
            duckdb_conn,
            _session(content="after rewrite", source_session_id="source-session-2"),
            keywords=buffer,
            tail_facts=TailFacts(),
        )
        buffer.flush(duckdb_conn)

        assert _matches(tmp_path, "message_fts", "before") == []
        assert _matches(tmp_path, "message_fts", "after") == ["msg-1"]
        assert _pending(duckdb_conn) == []
    finally:
        duckdb_conn.close()


def test_rolled_back_write_publishes_nothing(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    buffer = SidecarBuffer(_sidecar(tmp_path))

    def refuse() -> None:
        raise RuntimeError("acknowledgement refused")

    try:
        with pytest.raises(RuntimeError, match="acknowledgement refused"):
            _write_session(
                duckdb_conn,
                _session(content="never committed"),
                keywords=buffer,
                tail_facts=TailFacts(),
                on_commit=refuse,
            )

        assert buffer.seconds_until_due() is None
        assert buffer.flush(duckdb_conn) == 0
        assert _pending(duckdb_conn) == []
        assert _matches(tmp_path, "message_fts", "committed") == []
    finally:
        duckdb_conn.close()


def test_failed_flush_leaves_intents_that_repair_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    buffer = SidecarBuffer(_sidecar(tmp_path))
    try:
        _write_session(
            duckdb_conn,
            _session(content="survives failure"),
            keywords=buffer,
            tail_facts=TailFacts(),
        )

        def locked(_sidecar: sqlite3.Connection, _rows: object) -> None:
            raise sqlite3.OperationalError("database is locked")

        with monkeypatch.context() as patch:
            patch.setattr(buffer_module, "write_message_fts_rows", locked)
            assert buffer.flush(duckdb_conn) == 0
        assert buffer.last_error == "OperationalError: database is locked"
        assert _pending(duckdb_conn) == [("message", "msg-1", "upsert")]

        with closing(open_sidecar(_sidecar(tmp_path))) as sidecar:
            repair = repair_pending_batch(duckdb_conn, sidecar)
        assert (repair.applied, repair.remaining) == (1, 0)
        assert _matches(tmp_path, "message_fts", "survives") == ["msg-1"]
    finally:
        duckdb_conn.close()


def test_unflushed_changes_lost_in_a_crash_are_reconciled(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    try:
        first = SidecarBuffer(_sidecar(tmp_path))
        _write_session(
            duckdb_conn,
            _session(content="stale words", tool_calls=[_tool_call("tc-1", "echo stale")]),
            keywords=first,
            tail_facts=TailFacts(),
        )
        first.flush(duckdb_conn)

        # Committed in DuckDB, then the process dies before the flush: the
        # update to an existing id, the new id and the removed id are all lost.
        crashed = SidecarBuffer(_sidecar(tmp_path))
        _write_session(
            duckdb_conn,
            _session(
                content="fresh words",
                message_ids=["msg-1", "msg-2"],
                tool_calls=[],
            ),
            keywords=crashed,
            tail_facts=TailFacts(),
        )
        del crashed

        with closing(open_sidecar(_sidecar(tmp_path))) as sidecar:
            stats = reconcile_sidecar(duckdb_conn, sidecar)

        assert stats.pending_remaining == {"message": 0, "tool_call": 0}
        assert _matches(tmp_path, "message_fts", "stale") == []
        assert _matches(tmp_path, "message_fts", "fresh") == ["msg-1", "msg-2"]
        assert _rowid(tmp_path, "tool_call", "tc-1") is None
        assert _pending(duckdb_conn) == []
    finally:
        duckdb_conn.close()


def test_flush_keeps_delete_work_queued_outside_the_buffer(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    buffer = SidecarBuffer(_sidecar(tmp_path))
    try:
        _write_session(
            duckdb_conn, _session(content="moved away"), keywords=buffer, tail_facts=TailFacts()
        )
        duckdb_conn.execute(
            "INSERT INTO fts_sidecar_pending(kind, id, op) VALUES ('message', 'msg-1', 'delete')"
        )

        buffer.flush(duckdb_conn)

        assert _pending(duckdb_conn) == [("message", "msg-1", "delete")]
    finally:
        duckdb_conn.close()


def test_flush_leaves_ids_with_foreign_intents_for_repair(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    buffer = SidecarBuffer(_sidecar(tmp_path))
    try:
        _write_session(
            duckdb_conn, _session(content="plain body"), keywords=buffer, tail_facts=TailFacts()
        )
        # A writer outside the buffer (a moved session's carried context)
        # changes the same row's keyword text and queues its own upsert.
        duckdb_conn.execute(
            "UPDATE message_state SET fts_content = 'carried context body'"
            " WHERE message_id = 'msg-1'"
        )
        duckdb_conn.execute(
            "INSERT INTO fts_sidecar_pending(kind, id, op) VALUES ('message', 'msg-1', 'upsert')"
        )

        buffer.flush(duckdb_conn)

        # The buffered text is older than the foreign change, so its intents stay.
        assert _pending(duckdb_conn) == [("message", "msg-1", "upsert")] * 2
        with closing(open_sidecar(_sidecar(tmp_path))) as sidecar:
            repair_pending_batch(duckdb_conn, sidecar)
        assert _matches(tmp_path, "message_fts", "carried") == ["msg-1"]
        assert _pending(duckdb_conn) == []
    finally:
        duckdb_conn.close()


def test_buffer_flushes_at_its_row_bound_and_reports_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    now = [100.0]
    buffer = SidecarBuffer(_sidecar(tmp_path), clock=lambda: now[0])
    monkeypatch.setattr(buffer_module, "FLUSH_ROWS", 3)
    try:
        _write_session(
            duckdb_conn,
            _session(content="one", session_id="a"),
            keywords=buffer,
            tail_facts=TailFacts(),
        )
        now[0] += 2.0
        assert buffer.seconds_until_due() == buffer_module.FLUSH_AGE_SECONDS - 2.0
        now[0] += buffer_module.FLUSH_AGE_SECONDS
        assert buffer.seconds_until_due() == 0.0

        _write_session(
            duckdb_conn,
            _session(content="two", session_id="b", tool_calls=[_tool_call("b-tc", "ls")]),
            keywords=buffer,
            tail_facts=TailFacts(),
        )

        # Three rows reached the bound: the commit itself flushed.
        assert buffer.seconds_until_due() is None
        assert _matches(tmp_path, "message_fts", "two") == ["b-msg-1"]
        assert _pending(duckdb_conn) == []
    finally:
        duckdb_conn.close()


def test_daemon_buffer_defers_a_full_flush_to_its_own_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    buffer = SidecarBuffer(_sidecar(tmp_path), flush_when_full=False)
    monkeypatch.setattr(buffer_module, "FLUSH_ROWS", 1)
    try:
        _write_session(
            duckdb_conn, _session(content="deferred"), keywords=buffer, tail_facts=TailFacts()
        )

        # Full, so due at once, but the committing write did not flush it.
        assert buffer.seconds_until_due() == 0.0
        assert _matches(tmp_path, "message_fts", "deferred") == []
        assert buffer.flush(duckdb_conn) == 1
        assert _matches(tmp_path, "message_fts", "deferred") == ["msg-1"]
    finally:
        duckdb_conn.close()


def test_bounded_flush_writes_the_oldest_rows_and_keeps_the_rest_due(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    buffer = SidecarBuffer(_sidecar(tmp_path), flush_when_full=False)
    try:
        _write_session(
            duckdb_conn,
            _session(content="firstsession", session_id="a", message_ids=["m1", "m2", "m3"]),
            keywords=buffer,
            tail_facts=TailFacts(),
        )
        _write_session(
            duckdb_conn,
            _session(content="secondsession", session_id="b"),
            keywords=buffer,
            tail_facts=TailFacts(),
        )

        # The bound cuts the first session's change after two rows.
        assert buffer.flush(duckdb_conn, max_rows=2) == 2
        assert _matches(tmp_path, "message_fts", "firstsession") == ["a-m1", "a-m2"]
        assert _matches(tmp_path, "message_fts", "secondsession") == []
        assert _pending(duckdb_conn) == [
            ("message", "a-m3", "upsert"),
            ("message", "b-msg-1", "upsert"),
        ]
        assert buffer.seconds_until_due() == 0.0

        assert buffer.flush(duckdb_conn, max_rows=2) == 2
        assert _matches(tmp_path, "message_fts", "firstsession") == ["a-m1", "a-m2", "a-m3"]
        assert _matches(tmp_path, "message_fts", "secondsession") == ["b-msg-1"]
        assert _pending(duckdb_conn) == []
        assert buffer.seconds_until_due() is None
    finally:
        duckdb_conn.close()


def test_backend_duckdb_skips_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path, backend="duckdb")
    sidecar = sidecar_path(config.data_dir)

    summary = index_sessions(
        source=None,
        config=config,
        full=True,
        recreate=True,
        embed=False,
        verbose=False,
    )

    assert summary.total == 0
    assert not sidecar.exists()


def _count_sidecar_commits(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Keyword rows written by each SQLite transaction the buffer commits."""
    commits: list[int] = []
    rows = [0]

    def trace(statement: str) -> None:
        if statement.lstrip().startswith("INSERT OR REPLACE INTO"):
            rows[0] += 1
        elif statement == "COMMIT":
            commits.append(rows[0])
            rows[0] = 0

    open_original = buffer_module.open_sidecar

    def open_traced(path: Path) -> sqlite3.Connection:
        conn = open_original(path)
        conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr(buffer_module, "open_sidecar", open_traced)
    return commits


def _duckdb_with_schema(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "recall.duckdb"))
    ensure_schema(conn)
    return conn


def _sidecar(tmp_path: Path) -> Path:
    return tmp_path / "recall.fts.sqlite"


def _pending(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str]]:
    return [
        (str(kind), str(entity_id), str(op))
        for kind, entity_id, op in conn.execute(
            "SELECT kind, id, op FROM fts_sidecar_pending ORDER BY kind, id, op"
        ).fetchall()
    ]


def _message(message_id: str, content: str, session_id: str = "session-1") -> Message:
    return Message(
        id=message_id,
        session_id=session_id,
        idx=0,
        role=Role.ASSISTANT,
        content=content,
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


def _session(
    *,
    content: str,
    tool_calls: list[ToolCall] | None = None,
    session_id: str = "session-1",
    message_ids: list[str] | None = None,
    source_session_id: str = "source-session-1",
) -> Session:
    prefix = "" if session_id == "session-1" else f"{session_id}-"
    ids = ["msg-1"] if message_ids is None else message_ids
    messages = [
        _message(f"{prefix}{message_id}", content, session_id).model_copy(update={"idx": index})
        for index, message_id in enumerate(ids)
    ]
    calls = [
        call.model_copy(update={"session_id": session_id, "message_id": messages[0].id})
        for call in tool_calls or []
    ]
    if messages:
        messages[0].tool_calls = calls
    return Session(
        id=session_id,
        source=Source.CODEX,
        source_path=f"/tmp/{session_id}.jsonl",
        source_session_id=source_session_id,
        file_mtime=1.0,
        file_size=100,
        message_count=len(messages),
        tool_count=len(calls),
        messages=messages,
    )


def _app_config(tmp_path: Path, *, backend: str) -> AppConfig:
    data_dir = tmp_path / ".local" / "share" / "recall"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / ".config" / "recall" / "config.toml",
        fts=FtsConfig(fields=(), backend=backend),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )


def _matches(tmp_path: Path, table: str, term: str) -> list[str]:
    mapping, column = (
        ("message_fts_rowid", "message_id")
        if table == "message_fts"
        else ("tool_calls_fts_rowid", "tool_call_id")
    )
    with closing(open_sidecar(_sidecar(tmp_path))) as conn:
        rows = conn.execute(
            f"""
            SELECT m.{column}
            FROM {table} f
            JOIN {mapping} m ON m.rowid = f.rowid
            WHERE {table} MATCH ?
            ORDER BY m.{column}
            """,
            [term],
        ).fetchall()
    return [str(row[0]) for row in rows]


def _rowid(tmp_path: Path, kind: str, entity_id: str) -> int | None:
    table, column = (
        ("message_fts_rowid", "message_id")
        if kind == "message"
        else ("tool_calls_fts_rowid", "tool_call_id")
    )
    with closing(open_sidecar(_sidecar(tmp_path))) as conn:
        row = conn.execute(f"SELECT rowid FROM {table} WHERE {column} = ?", [entity_id]).fetchone()
    return None if row is None else int(row[0])
