from __future__ import annotations

from pathlib import Path

import duckdb
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.core.types import DaemonMode, Source
from recall.db.fts_sidecar import (
    open_sidecar,
    search_messages_fts,
    search_tool_calls_fts,
    sidecar_path,
    upsert_message_fts,
    upsert_tool_call_fts,
)
from recall.db.schema import ensure_schema
from recall.services.search import _search_messages_sidecar, _search_tool_calls_sidecar


def _app_config(tmp_path: Path) -> AppConfig:
    data_dir = tmp_path / ".local" / "share" / "recall"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / ".config" / "recall" / "config.toml",
        fts=FtsConfig(fields=("content", "thinking", "bash"), backend="sqlite_sidecar"),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(mode=DaemonMode.POLL),
        cli=CliConfig(),
    )


def _duckdb_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    return conn


def _insert_session(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    source: Source,
) -> None:
    conn.execute(
        "INSERT INTO sessions(id, source, source_path) VALUES (?, ?, ?)",
        [session_id, source.value, f"/sessions/{session_id}.jsonl"],
    )


def _insert_message(
    conn: duckdb.DuckDBPyConnection,
    *,
    message_id: str,
    session_id: str,
    idx: int,
    content: str,
) -> None:
    conn.execute(
        "INSERT INTO messages(id, session_id, idx) VALUES (?, ?, ?)",
        [message_id, session_id, idx],
    )
    conn.execute(
        """
        INSERT INTO message_state(message_id, role, content, thinking, fts_content, fts_thinking)
        VALUES (?, 'assistant', ?, '', ?, '')
        """,
        [message_id, content, content],
    )


def _insert_tool_call(
    conn: duckdb.DuckDBPyConnection,
    *,
    tool_call_id: str,
    message_id: str,
    session_id: str,
    idx: int,
    tool_name: str,
    bash_command: str,
) -> None:
    conn.execute(
        "INSERT INTO messages(id, session_id, idx) VALUES (?, ?, ?)",
        [message_id, session_id, idx],
    )
    conn.execute(
        """
        INSERT INTO message_state(message_id, role, content, thinking, fts_content, fts_thinking)
        VALUES (?, 'assistant', '', '', '', '')
        """,
        [message_id],
    )
    conn.execute(
        """
        INSERT INTO tool_calls(id, session_id, message_id, idx, tool_name, tool_input, bash_command)
        VALUES (?, ?, ?, ?, ?, '{}', ?)
        """,
        [tool_call_id, session_id, message_id, idx, tool_name, bash_command],
    )


def test_message_sidecar_source_filter_is_not_starved_by_global_top_k(tmp_path: Path) -> None:
    config = _app_config(tmp_path)
    duckdb_conn = _duckdb_conn()
    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        _insert_session(duckdb_conn, "session-claude", Source.CLAUDE_CODE)
        _insert_session(duckdb_conn, "session-codex", Source.CODEX)
        for idx in range(5):
            message_id = f"msg-claude-{idx}"
            content = "filterstarve filterstarve filterstarve filterstarve filterstarve"
            _insert_message(
                duckdb_conn,
                message_id=message_id,
                session_id="session-claude",
                idx=idx,
                content=content,
            )
            upsert_message_fts(sidecar_conn, message_id, content, "")
        _insert_message(
            duckdb_conn,
            message_id="msg-codex-target",
            session_id="session-codex",
            idx=0,
            content="filterstarve",
        )
        upsert_message_fts(sidecar_conn, "msg-codex-target", "filterstarve", "")

        assert [
            message_id
            for message_id, _score in search_messages_fts(
                sidecar_conn,
                "filterstarve",
                ["content"],
                limit=3,
            )
        ] == ["msg-claude-0", "msg-claude-1", "msg-claude-2"]

        results = _search_messages_sidecar(
            duckdb_conn,
            "filterstarve",
            Source.CODEX,
            3,
            ["content"],
            None,
            config,
        )

        assert [result.message_id for result in results] == ["msg-codex-target"]
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_tool_call_sidecar_tool_filter_is_not_starved_by_global_top_k(tmp_path: Path) -> None:
    config = _app_config(tmp_path)
    duckdb_conn = _duckdb_conn()
    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        _insert_session(duckdb_conn, "session-write", Source.CLAUDE_CODE)
        _insert_session(duckdb_conn, "session-bash", Source.CODEX)
        for idx in range(5):
            tool_call_id = f"tool-write-{idx}"
            command = "toolstarve toolstarve toolstarve toolstarve toolstarve"
            _insert_tool_call(
                duckdb_conn,
                tool_call_id=tool_call_id,
                message_id=f"msg-write-{idx}",
                session_id="session-write",
                idx=idx,
                tool_name="Write",
                bash_command=command,
            )
            upsert_tool_call_fts(sidecar_conn, tool_call_id, command)
        _insert_tool_call(
            duckdb_conn,
            tool_call_id="tool-bash-target",
            message_id="msg-bash-target",
            session_id="session-bash",
            idx=99,
            tool_name="Bash",
            bash_command="toolstarve",
        )
        upsert_tool_call_fts(sidecar_conn, "tool-bash-target", "toolstarve")

        assert [
            tool_call_id
            for tool_call_id, _score in search_tool_calls_fts(
                sidecar_conn,
                "toolstarve",
                limit=3,
            )
        ] == ["tool-write-0", "tool-write-1", "tool-write-2"]

        results = _search_tool_calls_sidecar(
            duckdb_conn,
            "toolstarve",
            Source.CODEX,
            "Bash",
            3,
            "session-bash",
            config,
        )

        assert [result.tool_call_id for result in results] == ["tool-bash-target"]
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_unfiltered_sidecar_message_search_preserves_top_k_order(tmp_path: Path) -> None:
    config = _app_config(tmp_path)
    duckdb_conn = _duckdb_conn()
    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        _insert_session(duckdb_conn, "session-claude", Source.CLAUDE_CODE)
        _insert_session(duckdb_conn, "session-codex", Source.CODEX)
        rows = [
            ("msg-best", "session-claude", "rankterm rankterm rankterm rankterm rankterm"),
            ("msg-second", "session-codex", "rankterm rankterm"),
            ("msg-third", "session-codex", "rankterm"),
        ]
        for idx, (message_id, session_id, content) in enumerate(rows):
            _insert_message(
                duckdb_conn,
                message_id=message_id,
                session_id=session_id,
                idx=idx,
                content=content,
            )
            upsert_message_fts(sidecar_conn, message_id, content, "")

        expected_ids = [
            message_id
            for message_id, _score in search_messages_fts(
                sidecar_conn,
                "rankterm",
                ["content"],
                limit=2,
            )
        ]

        results = _search_messages_sidecar(
            duckdb_conn,
            "rankterm",
            None,
            2,
            ["content"],
            None,
            config,
        )

        assert [result.message_id for result in results] == expected_ids
    finally:
        sidecar_conn.close()
        duckdb_conn.close()
