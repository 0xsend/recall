from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import duckdb
import recall.db.fts_sidecar as fts_sidecar_module
from recall.core.models import Message, ToolCall
from recall.core.types import Role
from recall.db.fts_sidecar import open_sidecar, search_messages_fts
from recall.services.indexer import _insert_messages, _upsert_tool_call_sidecar_rows


class _CountingConnection(sqlite3.Connection):
    transaction_entries: int

    def __enter__(self) -> _CountingConnection:
        self.transaction_entries += 1
        return cast("_CountingConnection", super().__enter__())


def _create_message_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            idx INTEGER NOT NULL,
            agent_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE message_state (
            message_id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT,
            thinking TEXT,
            timestamp TIMESTAMP,
            has_thinking BOOLEAN DEFAULT FALSE,
            context_text TEXT DEFAULT '',
            context_mode TEXT DEFAULT 'off',
            fts_content TEXT DEFAULT '',
            fts_thinking TEXT DEFAULT ''
        )
        """
    )
    # insert_messages clears the id it is about to own across all three
    # message tables, so the harness needs the embedding table too.
    conn.execute(
        """
        CREATE TABLE message_embeddings (
            message_id TEXT PRIMARY KEY,
            content_embedding FLOAT[384],
            thinking_embedding FLOAT[384]
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE fts_sidecar_pending (
            kind TEXT NOT NULL,
            id TEXT NOT NULL,
            op TEXT NOT NULL,
            queued_at TIMESTAMP NOT NULL DEFAULT now()
        )
        """
    )


def _message_ids(conn: sqlite3.Connection, query: str, fields: list[str]) -> list[str]:
    return [message_id for message_id, _score in search_messages_fts(conn, query, fields, 10)]


def _open_counting_sidecar(path: Path) -> _CountingConnection:
    setup_conn = open_sidecar(path)
    setup_conn.close()
    conn = sqlite3.connect(path, factory=_CountingConnection)
    counting_conn = conn
    counting_conn.transaction_entries = 0
    return counting_conn


def test_insert_messages_writes_only_configured_sidecar_fields(tmp_path: Path) -> None:
    conn = duckdb.connect(":memory:")
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _create_message_tables(conn)
        message = Message(
            id="msg-index-scope",
            session_id="session-index-scope",
            idx=0,
            role=Role.ASSISTANT,
            content="visiblecontentterm",
            thinking="thinkingonlyterm",
            has_thinking=True,
        )

        _insert_messages(
            conn,
            [message],
            sidecar_conn=sidecar_conn,
            fts_fields=("content",),
        )

        assert _message_ids(sidecar_conn, "thinkingonlyterm", ["content", "thinking"]) == []
        assert _message_ids(sidecar_conn, "visiblecontentterm", ["content"]) == ["msg-index-scope"]
    finally:
        sidecar_conn.close()
        conn.close()


def test_indexer_sidecar_writes_messages_and_tool_calls_in_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    conn = duckdb.connect(":memory:")
    sidecar_conn = _open_counting_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        monkeypatch.setattr(fts_sidecar_module, "FTS_SIDECAR_BATCH_SIZE", 2)
        _create_message_tables(conn)
        messages = [
            Message(
                id=f"msg-index-batch-{idx}",
                session_id="session-index-batch",
                idx=idx,
                role=Role.ASSISTANT,
                content=f"batch content {idx}",
                thinking=f"batch thinking {idx}",
                has_thinking=True,
            )
            for idx in range(5)
        ]
        tool_calls = [
            ToolCall(
                id=f"tool-index-batch-{idx}",
                session_id="session-index-batch",
                message_id=messages[idx].id,
                idx=idx,
                tool_name="Bash",
                bash_command=f"echo batch tool {idx}",
            )
            for idx in range(5)
        ]

        _insert_messages(
            conn,
            messages,
            sidecar_conn=sidecar_conn,
            fts_fields=("content", "thinking"),
        )
        _upsert_tool_call_sidecar_rows(
            conn,
            sidecar_conn,
            tool_calls,
            fts_fields=("bash",),
        )

        assert sidecar_conn.transaction_entries == 6
    finally:
        sidecar_conn.close()
        conn.close()


def test_insert_messages_includes_thinking_when_configured(tmp_path: Path) -> None:
    conn = duckdb.connect(":memory:")
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _create_message_tables(conn)
        message = Message(
            id="msg-index-thinking",
            session_id="session-index-thinking",
            idx=0,
            role=Role.ASSISTANT,
            content="ordinary content",
            thinking="includedthinkingterm",
            has_thinking=True,
        )

        _insert_messages(
            conn,
            [message],
            sidecar_conn=sidecar_conn,
            fts_fields=("content", "thinking"),
        )

        assert _message_ids(sidecar_conn, "includedthinkingterm", ["thinking"]) == [
            "msg-index-thinking"
        ]
    finally:
        sidecar_conn.close()
        conn.close()
