from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
import pytest
import recall.services.indexer as indexer_module
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.core.models import Message, Session, TailFacts, ToolCall
from recall.core.types import Role, Source
from recall.db.fts_sidecar import open_sidecar, sidecar_path
from recall.db.queries import delete_session
from recall.db.schema import ensure_schema
from recall.services.indexer import (
    _insert_messages,
    _upsert_tool_call_sidecar_rows,
    _upsert_tool_calls,
    _write_session,
    index_sessions,
)


def test_insert_messages_dual_writes_to_sidecar(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        message = _message("msg-1", content="hello indexed world", thinking="private thought")

        _insert_messages(duckdb_conn, [message], context_text="ctx ", sidecar_conn=sidecar_conn)

        assert duckdb_conn.execute(
            "SELECT fts_content, fts_thinking FROM message_state WHERE message_id = ?",
            [message.id],
        ).fetchone() == ("ctx hello indexed world", "ctx private thought")
        assert _matches(sidecar_conn, "message_fts", "indexed") == [message.id]
        assert _matches(sidecar_conn, "message_fts", "thought") == [message.id]
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_upsert_messages_updates_sidecar_in_place(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _write_session(
            duckdb_conn,
            _session(content="initial alpha"),
            sidecar_conn=sidecar_conn,
            tail_facts=TailFacts(),
        )
        original_rowid = _message_rowid(sidecar_conn, "msg-1")

        _write_session(
            duckdb_conn,
            _session(content="updated beta"),
            sidecar_conn=sidecar_conn,
            tail_facts=TailFacts(),
        )

        assert _message_rowid(sidecar_conn, "msg-1") == original_rowid
        assert _matches(sidecar_conn, "message_fts", "initial") == []
        assert _matches(sidecar_conn, "message_fts", "updated") == ["msg-1"]
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_upsert_tool_calls_dual_writes(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        first = _tool_call("tc-1", bash_command="echo first")
        _upsert_tool_calls(duckdb_conn, {}, {first.id: first}, sidecar_conn=sidecar_conn)
        original_rowid = _tool_call_rowid(sidecar_conn, first.id)

        second = _tool_call("tc-1", bash_command="printf second")
        existing_rows = {
            str(row[0]): tuple(row)
            for row in duckdb_conn.execute(
                """
                SELECT tc.*, tce.bash_embedding
                FROM tool_calls tc
                LEFT JOIN tool_call_embeddings tce ON tce.tool_call_id = tc.id
                """
            ).fetchall()
        }
        _upsert_tool_calls(
            duckdb_conn, existing_rows, {second.id: second}, sidecar_conn=sidecar_conn
        )

        assert _tool_call_rowid(sidecar_conn, second.id) == original_rowid
        assert _matches(sidecar_conn, "tool_calls_fts", "first") == []
        assert _matches(sidecar_conn, "tool_calls_fts", "second") == ["tc-1"]
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_tool_call_with_null_bash_command_skips_sidecar(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        tool_call = _tool_call("tc-null", bash_command=None)

        _upsert_tool_calls(
            duckdb_conn,
            {},
            {tool_call.id: tool_call},
            sidecar_conn=sidecar_conn,
        )

        assert _tool_call_rowid(sidecar_conn, tool_call.id) is None
        assert _count(sidecar_conn, "tool_calls_fts_rowid") == 0
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_delete_session_cascades_to_sidecar(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _write_session(
            duckdb_conn,
            _session(content="delete me", tool_calls=[_tool_call("tc-1", "rm target")]),
            sidecar_conn=sidecar_conn,
            tail_facts=TailFacts(),
        )

        delete_session(duckdb_conn, "session-1", sidecar_conn=sidecar_conn)

        assert _message_rowid(sidecar_conn, "msg-1") is None
        assert _tool_call_rowid(sidecar_conn, "tc-1") is None
        assert _matches(sidecar_conn, "message_fts", "delete") == []
        assert _matches(sidecar_conn, "tool_calls_fts", "target") == []
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_sidecar_write_failure_queues_pending_and_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:

        def fail_upsert(*_args: object, **_kwargs: object) -> None:
            raise sqlite3.Error("boom")

        monkeypatch.setattr(indexer_module, "upsert_message_fts_batch", fail_upsert)

        _insert_messages(
            duckdb_conn, [_message("msg-1", "still commits")], sidecar_conn=sidecar_conn
        )

        assert duckdb_conn.execute(
            "SELECT content FROM message_state WHERE message_id = ?",
            ["msg-1"],
        ).fetchone() == ("still commits",)
        assert duckdb_conn.execute(
            """
            SELECT kind, id, op
            FROM fts_sidecar_pending
            """
        ).fetchall() == [("message", "msg-1", "upsert")]
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_tool_sidecar_delete_failure_queues_null_command_for_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duckdb_conn = _duckdb_with_schema(tmp_path)
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        tool_call = _tool_call("tc-null-failure", bash_command=None)

        def fail_batch(*_args: object, **_kwargs: object) -> None:
            raise sqlite3.Error("boom")

        monkeypatch.setattr(indexer_module, "upsert_tool_call_fts_batch", fail_batch)
        _upsert_tool_call_sidecar_rows(
            duckdb_conn,
            sidecar_conn,
            (call for call in [tool_call]),
        )

        assert duckdb_conn.execute("SELECT kind, id, op FROM fts_sidecar_pending").fetchall() == [
            ("tool_call", tool_call.id, "upsert")
        ]
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_backend_duckdb_skips_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path, backend="duckdb")
    sidecar = sidecar_path(config.data_dir)

    def fail_open_sidecar(_path: Path) -> sqlite3.Connection:
        raise AssertionError("sidecar should not be opened for duckdb backend")

    monkeypatch.setattr("recall.db.open_sidecar", fail_open_sidecar)

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


def _duckdb_with_schema(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "recall.duckdb"))
    ensure_schema(conn)
    return conn


def _message(message_id: str, content: str, thinking: str | None = None) -> Message:
    return Message(
        id=message_id,
        session_id="session-1",
        idx=0,
        role=Role.ASSISTANT,
        content=content,
        thinking=thinking,
        has_thinking=thinking is not None,
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
) -> Session:
    message = _message("msg-1", content)
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


def _message_rowid(conn: sqlite3.Connection, message_id: str) -> int | None:
    row = conn.execute(
        "SELECT rowid FROM message_fts_rowid WHERE message_id = ?",
        [message_id],
    ).fetchone()
    return None if row is None else int(row[0])


def _tool_call_rowid(conn: sqlite3.Connection, tool_call_id: str) -> int | None:
    row = conn.execute(
        "SELECT rowid FROM tool_calls_fts_rowid WHERE tool_call_id = ?",
        [tool_call_id],
    ).fetchone()
    return None if row is None else int(row[0])


def _count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])
