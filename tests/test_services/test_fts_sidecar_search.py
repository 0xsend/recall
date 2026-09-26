from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path
from typing import Any

import duckdb
import pytest
from recall.core.config import (
    DEFAULT_EMBED_MODEL,
    KNOWN_MODEL_DIMENSIONS,
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.types import SearchMode, Source
from recall.db.fts_sidecar import (
    open_sidecar,
    search_messages_fts,
    sidecar_path,
    upsert_message_fts,
    upsert_tool_call_fts,
)
from recall.db.schema import ensure_schema
from recall.services.search import search

_TEST_EMBED_DIM = KNOWN_MODEL_DIMENSIONS[DEFAULT_EMBED_MODEL]
search_module: Any = importlib.import_module("recall.services.search")


def test_search_messages_keyword_uses_sidecar(tmp_path: Path) -> None:
    config = _app_config(tmp_path, backend="sqlite_sidecar")
    conn = _duckdb_with_schema(config)
    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        _seed_session(
            conn,
            sidecar_conn,
            session_id="session-1",
            message_id="msg-1",
            content="foo alpha foo",
            thinking="quiet",
        )
        _seed_session(
            conn,
            sidecar_conn,
            session_id="session-2",
            message_id="msg-2",
            content="foo beta",
            thinking="quiet",
        )

        results = search(
            query="foo",
            source=None,
            tool=None,
            mode=SearchMode.KEYWORD,
            config=config,
            conn=conn,
        )

        assert [result.message_id for result in results] == ["msg-1", "msg-2"]
        assert results[0].score >= results[1].score
    finally:
        sidecar_conn.close()
        conn.close()


def test_search_tool_calls_keyword_uses_sidecar(tmp_path: Path) -> None:
    config = _app_config(tmp_path, backend="sqlite_sidecar")
    conn = _duckdb_with_schema(config)
    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        _seed_session(
            conn,
            sidecar_conn,
            session_id="session-1",
            message_id="msg-1",
            content="plain message",
            tool_call_id="tc-1",
            bash_command="git status && git diff",
        )

        results = search(
            query="git",
            source=None,
            tool="bash",
            mode=SearchMode.KEYWORD,
            config=config,
            conn=conn,
        )

        assert [result.tool_call_id for result in results] == ["tc-1"]
        assert results[0].bash_command == "git status && git diff"
    finally:
        sidecar_conn.close()
        conn.close()


def test_search_hybrid_uses_sidecar_bm25(tmp_path: Path) -> None:
    config = _app_config(tmp_path, backend="sqlite_sidecar")
    conn = _duckdb_with_schema(config)
    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        _seed_session(
            conn,
            sidecar_conn,
            session_id="keyword-session",
            message_id="msg-keyword",
            content="foo keyword match",
            content_embedding=_vec(1),
        )
        _seed_session(
            conn,
            sidecar_conn,
            session_id="vector-session",
            message_id="msg-vector",
            content="semantic match",
            content_embedding=_vec(0),
        )
        backend = _FakeBackend()

        first = search(
            query="foo",
            source=None,
            tool=None,
            mode=SearchMode.HYBRID,
            config=config,
            conn=conn,
            embed_backend=backend,
        )
        second = search(
            query="foo",
            source=None,
            tool=None,
            mode=SearchMode.HYBRID,
            config=config,
            conn=conn,
            embed_backend=backend,
        )

        assert {result.message_id for result in first} >= {"msg-keyword", "msg-vector"}
        assert [result.message_id for result in first] == [result.message_id for result in second]
    finally:
        sidecar_conn.close()
        conn.close()


def test_search_keyword_returns_empty_on_no_match(tmp_path: Path) -> None:
    config = _app_config(tmp_path, backend="sqlite_sidecar")
    conn = _duckdb_with_schema(config)
    try:
        results = search(
            query="missing",
            source=None,
            tool=None,
            mode=SearchMode.KEYWORD,
            config=config,
            conn=conn,
        )

        assert results == []
    finally:
        conn.close()


def test_search_respects_source_and_session_filters_under_sidecar(tmp_path: Path) -> None:
    config = _app_config(tmp_path, backend="sqlite_sidecar")
    conn = _duckdb_with_schema(config)
    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        _seed_session(
            conn,
            sidecar_conn,
            session_id="codex-1",
            message_id="msg-codex-1",
            content="shared filter token",
            source=Source.CODEX,
        )
        _seed_session(
            conn,
            sidecar_conn,
            session_id="codex-2",
            message_id="msg-codex-2",
            content="shared filter token",
            source=Source.CODEX,
        )
        _seed_session(
            conn,
            sidecar_conn,
            session_id="grok-1",
            message_id="msg-grok-1",
            content="shared filter token",
            source=Source.GROK,
        )

        results = search(
            query="shared",
            source=Source.CODEX,
            tool=None,
            session="codex-2",
            mode=SearchMode.KEYWORD,
            config=config,
            conn=conn,
        )

        assert [result.message_id for result in results] == ["msg-codex-2"]
    finally:
        sidecar_conn.close()
        conn.close()


def test_search_messages_sidecar_field_filters(tmp_path: Path) -> None:
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        upsert_message_fts(
            sidecar_conn,
            "msg-content",
            fts_content="visible needle",
            fts_thinking="private",
        )
        upsert_message_fts(
            sidecar_conn,
            "msg-thinking",
            fts_content="visible",
            fts_thinking="private needle",
        )

        assert [row[0] for row in search_messages_fts(sidecar_conn, "needle", ["content"], 10)] == [
            "msg-content"
        ]
        thinking_hits = search_messages_fts(sidecar_conn, "needle", ["thinking"], 10)
        assert [row[0] for row in thinking_hits] == ["msg-thinking"]
    finally:
        sidecar_conn.close()


def test_search_messages_sidecar_handles_empty_and_wildcard_queries(tmp_path: Path) -> None:
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        upsert_message_fts(sidecar_conn, "msg-1", fts_content="foobar", fts_thinking="")

        assert search_messages_fts(sidecar_conn, "", ["content"], 10) == []
        assert [row[0] for row in search_messages_fts(sidecar_conn, "foo*", ["content"], 10)] == [
            "msg-1"
        ]
    finally:
        sidecar_conn.close()


def test_search_sidecar_unavailable_error_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _app_config(tmp_path, backend="sqlite_sidecar")
    conn = _duckdb_with_schema(config)

    def fail_open_sidecar(_path: Path) -> sqlite3.Connection:
        raise search_module.FtsSidecarUnavailableError("unsupported runtime")

    monkeypatch.setattr(search_module, "open_sidecar", fail_open_sidecar)
    try:
        with pytest.raises(RuntimeError, match="RECALL_FTS_BACKEND=duckdb"):
            search(
                query="foo",
                source=None,
                tool=None,
                mode=SearchMode.KEYWORD,
                config=config,
                conn=conn,
            )
    finally:
        conn.close()


class _FakeBackend:
    dimensions = _TEST_EMBED_DIM
    model_id = "fake"
    query_prefix = ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_vec(0) for _text in texts]


def _vec(index: int) -> list[float]:
    values = [0.0] * _TEST_EMBED_DIM
    values[index] = 1.0
    return values


def _app_config(tmp_path: Path, *, backend: str) -> AppConfig:
    data_dir = tmp_path / "data"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / "config.toml",
        fts=FtsConfig(backend=backend),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )


def _duckdb_with_schema(config: AppConfig) -> duckdb.DuckDBPyConnection:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(config.db_path))
    ensure_schema(conn)
    return conn


def _seed_session(
    conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    session_id: str,
    message_id: str,
    content: str,
    thinking: str | None = None,
    source: Source = Source.CODEX,
    tool_call_id: str | None = None,
    bash_command: str | None = None,
    content_embedding: list[float] | None = None,
) -> None:
    conn.execute(
        "INSERT INTO sessions (id, source, source_path, source_session_id) VALUES (?, ?, ?, ?)",
        [session_id, source.value, f"/tmp/{session_id}.jsonl", None],
    )
    conn.execute(
        "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
        [message_id, session_id, 0],
    )
    conn.execute(
        """
        INSERT INTO message_state (
            message_id, role, content, thinking, has_thinking, fts_content, fts_thinking
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            message_id,
            "assistant",
            content,
            thinking,
            thinking is not None,
            content,
            thinking or "",
        ],
    )
    upsert_message_fts(sidecar_conn, message_id, content, thinking or "")
    if content_embedding is not None:
        conn.execute(
            """
            INSERT INTO message_embeddings (
                message_id, content_embedding, thinking_embedding
            ) VALUES (?, ?, ?)
            """,
            [message_id, content_embedding, None],
        )
    if tool_call_id is None:
        return
    conn.execute(
        """
        INSERT INTO tool_calls (
            id, session_id, message_id, idx, tool_name, bash_command, is_compound
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [tool_call_id, session_id, message_id, 0, "bash", bash_command, False],
    )
    upsert_tool_call_fts(sidecar_conn, tool_call_id, bash_command)
