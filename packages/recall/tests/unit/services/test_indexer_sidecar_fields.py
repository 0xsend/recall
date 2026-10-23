from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
from recall.core.models import Message
from recall.core.types import Role
from recall.db.fts_sidecar import open_sidecar, search_messages_fts
from recall.services.indexer import _insert_messages


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
