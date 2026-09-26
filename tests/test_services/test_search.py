from __future__ import annotations

import dataclasses
import importlib
import math
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
from recall.core.types import SearchMode
from recall.db.fts_sidecar import open_sidecar, sidecar_path, upsert_message_fts
from recall.db.queries import create_fts_indexes
from recall.db.schema import ensure_schema
from recall.services.search import (
    SearchResult,
    _apply_result_quality_adjustments,
    _hybrid_search,
    _resolve_mode,
    _search_messages,
    _vector_search_messages,
)

search_module: Any = importlib.import_module("recall.services.search")

_TEST_EMBED_DIM = KNOWN_MODEL_DIMENSIONS[DEFAULT_EMBED_MODEL]


def _vec(index: int) -> list[float]:
    values = [0.0] * _TEST_EMBED_DIM
    values[index] = 1.0
    return values


def _seed_session(
    conn: duckdb.DuckDBPyConnection, *, message_thinking: list[float] | None = None
) -> None:
    conn.execute(
        """
        INSERT INTO sessions (
            id, source, source_path, source_session_id
        ) VALUES (?, ?, ?, ?)
        """,
        ["session-1", "codex", "/tmp/session.jsonl", None],
    )
    conn.execute(
        """
        INSERT INTO session_state (
            session_id, message_count, tool_count, is_complete, file_mtime, file_size
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        ["session-1", 1, 0, True, 1.0, 1],
    )
    conn.execute(
        """
        INSERT INTO messages (
            id, session_id, idx
        ) VALUES (?, ?, ?)
        """,
        ["msg-1", "session-1", 0],
    )
    conn.execute(
        """
        INSERT INTO message_state (
            message_id, role, content, thinking, has_thinking
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            "msg-1",
            "assistant",
            "content text",
            "thinking text",
            True,
        ],
    )
    conn.execute(
        """
        INSERT INTO message_embeddings (
            message_id, content_embedding, thinking_embedding
        ) VALUES (?, ?, ?)
        """,
        ["msg-1", _vec(0), message_thinking],
    )


def test_vector_search_messages_uses_thinking_embedding_score() -> None:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _seed_session(conn, message_thinking=_vec(1))

    results = _vector_search_messages(conn, _vec(1), source=None, limit=5)

    assert len(results) == 1
    assert results[0].message_id == "msg-1"
    assert math.isclose(results[0].score, 1.0)


def test_search_results_include_host(tmp_path: Path) -> None:
    """REQ-HOST-API-003: search hits carry session_state.host."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / "config.toml",
        fts=FtsConfig(backend="duckdb"),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )
    conn = duckdb.connect(str(config.db_path))
    ensure_schema(conn)
    try:
        conn.execute(
            """
            INSERT INTO sessions (id, source, source_path, source_session_id)
            VALUES (?, ?, ?, ?)
            """,
            ["session-host", "codex", "/tmp/session-host.jsonl", None],
        )
        conn.execute(
            """
            INSERT INTO session_state (
                session_id, message_count, tool_count, is_complete,
                file_mtime, file_size, host
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ["session-host", 1, 0, True, 1.0, 1, "devbox"],
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
            ["msg-host", "session-host", 0],
        )
        conn.execute(
            """
            INSERT INTO message_state (message_id, role, content, thinking, has_thinking)
            VALUES (?, ?, ?, ?, ?)
            """,
            ["msg-host", "assistant", "unique host stamp phrase for search", "", False],
        )
        create_fts_indexes(conn, FtsConfig(backend="duckdb"))

        results = search_module.search(
            query="unique host stamp phrase",
            source=None,
            tool=None,
            mode=SearchMode.KEYWORD,
            config=config,
            conn=conn,
        )

        assert results
        assert all(r.host == "devbox" for r in results)
        assert all(r.host for r in results)
    finally:
        conn.close()


@pytest.mark.parametrize("backend", ["duckdb", "sqlite_sidecar"])
def test_keyword_search_smoke_supports_duckdb_and_sidecar(
    tmp_path: Path,
    backend: str,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / "config.toml",
        fts=FtsConfig(backend=backend),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )
    conn = duckdb.connect(str(config.db_path))
    ensure_schema(conn)
    try:
        _seed_two_sessions(conn)
        if backend == "duckdb":
            create_fts_indexes(conn, FtsConfig(backend="duckdb"))
        else:
            sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
            try:
                upsert_message_fts(
                    sidecar_conn,
                    "msg-alpha",
                    "alpha about git rebase",
                    "",
                )
                upsert_message_fts(
                    sidecar_conn,
                    "msg-beta",
                    "beta about git rebase",
                    "",
                )
            finally:
                sidecar_conn.close()

        results = search_module.search(
            query="git rebase",
            source=None,
            tool=None,
            mode=SearchMode.KEYWORD,
            config=config,
            conn=conn,
        )

        assert results
    finally:
        conn.close()


@pytest.mark.parametrize(
    "sql",
    [
        (
            "INSERT OR REPLACE INTO message_embeddings "
            "(message_id, thinking_embedding) VALUES (?, ?)"
        ),
        ("INSERT INTO tool_call_embeddings (tool_call_id, bash_embedding) VALUES (?, ?)"),
    ],
)
def test_resolve_mode_detects_non_content_embeddings(sql: str) -> None:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _seed_session(conn, message_thinking=None)
    if "message_embeddings" in sql:
        conn.execute(sql, ["msg-1", _vec(1)])
    else:
        # Insert the tool_call first, then the embedding
        conn.execute(
            "INSERT INTO tool_calls "
            "(id, session_id, message_id, idx, tool_name, bash_command, is_compound) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ["tc-1", "session-1", "msg-1", 0, "bash", "git status", False],
        )
        conn.execute(sql, ["tc-1", _vec(2)])

    assert _resolve_mode(conn, SearchMode.AUTO) == SearchMode.HYBRID


def test_hybrid_search_assigns_single_ranker_score() -> None:
    message_result = SearchResult(
        kind="message",
        session_id="session-1",
        source="codex",
        source_path="/tmp/session.jsonl",
        score=10.0,
        message_id="msg-1",
        tool_call_id=None,
        role="assistant",
        content="hello",
        thinking=None,
        timestamp=None,
        tool_name=None,
        bash_command=None,
    )
    tool_result = SearchResult(
        kind="tool_call",
        session_id="session-1",
        source="codex",
        source_path="/tmp/session.jsonl",
        score=9.0,
        message_id="msg-1",
        tool_call_id="tc-1",
        role=None,
        content=None,
        thinking=None,
        timestamp=None,
        tool_name="bash",
        bash_command="git status",
    )

    original_search_all = search_module._search_all
    original_vector_search_all = search_module._vector_search_all
    search_module._search_all = lambda *_args, **_kwargs: [message_result]
    search_module._vector_search_all = lambda *_args, **_kwargs: [tool_result]
    try:
        results = _hybrid_search(
            conn=duckdb.connect(":memory:"),
            query="git",
            query_embedding=_vec(0),
            source=None,
            limit=10,
            fields=("content", "thinking", "bash"),
            k=60,
        )
    finally:
        search_module._search_all = original_search_all
        search_module._vector_search_all = original_vector_search_all

    assert [result.kind for result in results] == ["message", "tool_call"]
    assert math.isclose(results[0].score, 1.0 / 61.0)
    assert math.isclose(results[1].score, 1.0 / 61.0)


def test_apply_result_quality_adjustments_downweights_command_wrappers() -> None:
    wrapper = SearchResult(
        kind="message",
        session_id="session-1",
        source="codex",
        source_path="/tmp/session.jsonl",
        score=10.0,
        message_id="msg-1",
        tool_call_id=None,
        role="system",
        content=(
            "---\nallowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*)\n"
            "description: Create a git commit\n---"
        ),
        thinking=None,
        timestamp=None,
        tool_name=None,
        bash_command=None,
    )
    direct = SearchResult(
        kind="message",
        session_id="session-1",
        source="codex",
        source_path="/tmp/session.jsonl",
        score=10.0,
        message_id="msg-2",
        tool_call_id=None,
        role="assistant",
        content="Lets create a git commit for the changes.",
        thinking=None,
        timestamp=None,
        tool_name=None,
        bash_command=None,
    )

    adjusted = _apply_result_quality_adjustments([wrapper, direct])

    assert [result.message_id for result in adjusted] == ["msg-2", "msg-1"]


def test_hybrid_search_breaks_rrf_ties_with_component_scores() -> None:
    stronger_bm25 = SearchResult(
        kind="message",
        session_id="session-1",
        source="codex",
        source_path="/tmp/session.jsonl",
        score=10.0,
        message_id="msg-b",
        tool_call_id=None,
        role="assistant",
        content="strong keyword match",
        thinking=None,
        timestamp=None,
        tool_name=None,
        bash_command=None,
    )
    stronger_vector = SearchResult(
        kind="message",
        session_id="session-1",
        source="codex",
        source_path="/tmp/session.jsonl",
        score=5.0,
        message_id="msg-a",
        tool_call_id=None,
        role="assistant",
        content="strong semantic match",
        thinking=None,
        timestamp=None,
        tool_name=None,
        bash_command=None,
    )
    vector_ranked = [
        dataclasses.replace(stronger_vector, score=0.9),
        dataclasses.replace(stronger_bm25, score=0.1),
    ]

    original_search_all = search_module._search_all
    original_vector_search_all = search_module._vector_search_all
    search_module._search_all = lambda *_args, **_kwargs: [stronger_bm25, stronger_vector]
    search_module._vector_search_all = lambda *_args, **_kwargs: vector_ranked
    try:
        results = _hybrid_search(
            conn=duckdb.connect(":memory:"),
            query="git",
            query_embedding=_vec(0),
            source=None,
            limit=10,
            fields=("content", "thinking", "bash"),
            k=60,
        )
    finally:
        search_module._search_all = original_search_all
        search_module._vector_search_all = original_vector_search_all

    assert [result.message_id for result in results[:2]] == ["msg-b", "msg-a"]


def _seed_two_sessions(conn: duckdb.DuckDBPyConnection) -> None:
    """Seed two sessions with distinct content for session-filter tests."""
    rows = [
        ("sess-alpha", "msg-alpha", "alpha about git rebase", "tc-alpha", "git rebase main"),
        ("sess-beta", "msg-beta", "beta about git rebase", "tc-beta", "git rebase develop"),
    ]
    for sid, msg_id, content, tc_id, bash_cmd in rows:
        conn.execute(
            "INSERT INTO sessions (id, source, source_path, source_session_id) VALUES (?, ?, ?, ?)",
            [sid, "codex", f"/tmp/{sid}.jsonl", None],
        )
        conn.execute(
            "INSERT INTO session_state "
            "(session_id, message_count, tool_count, is_complete, file_mtime, file_size) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [sid, 1, 1, True, 1.0, 1],
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
            [msg_id, sid, 0],
        )
        conn.execute(
            "INSERT INTO message_state "
            "(message_id, role, content, thinking, has_thinking) "
            "VALUES (?, ?, ?, ?, ?)",
            [msg_id, "assistant", content, None, False],
        )
        conn.execute(
            "INSERT INTO tool_calls "
            "(id, session_id, message_id, idx, tool_name, bash_command, is_compound) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [tc_id, sid, msg_id, 0, "bash", bash_cmd, False],
        )


def _seed_contextual_sessions(
    conn: duckdb.DuckDBPyConnection,
    *,
    context_mode: str,
) -> None:
    rows = [
        ("sess-recall", "msg-recall", "[acme/recall main] "),
        ("sess-other", "msg-other", "[example/other main] "),
    ]
    for sid, msg_id, template_context in rows:
        context_text = template_context if context_mode == "template" else ""
        content = "deterministic shared retrieval content"
        conn.execute(
            "INSERT INTO sessions (id, source, source_path, source_session_id) VALUES (?, ?, ?, ?)",
            [sid, "codex", f"/tmp/{sid}.jsonl", None],
        )
        conn.execute(
            "INSERT INTO session_state "
            "(session_id, git_repo, git_branch, message_count, tool_count, "
            "is_complete, file_mtime, file_size) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                sid,
                "acme/recall" if sid == "sess-recall" else "example/other",
                "main",
                1,
                0,
                True,
                1.0,
                1,
            ],
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
            [msg_id, sid, 0],
        )
        conn.execute(
            """
            INSERT INTO message_state (
                message_id, role, content, thinking, has_thinking,
                context_text, context_mode, fts_content, fts_thinking
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                msg_id,
                "assistant",
                content,
                None,
                False,
                context_text,
                context_mode,
                f"{context_text}{content}",
                "",
            ],
        )


def test_contextual_bm25_ranks_matching_repo_prefix_higher() -> None:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _seed_contextual_sessions(conn, context_mode="template")
    create_fts_indexes(conn, FtsConfig(backend="duckdb"))

    results = _search_messages(
        conn,
        "acme deterministic",
        source=None,
        limit=10,
        fields=["content"],
    )

    assert [result.session_id for result in results] == ["sess-recall", "sess-other"]
    assert results[0].score > results[1].score


def test_off_mode_bm25_ranking_is_identical_for_identical_content() -> None:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _seed_contextual_sessions(conn, context_mode="off")
    create_fts_indexes(conn, FtsConfig(backend="duckdb"))

    results = _search_messages(
        conn,
        "acme deterministic",
        source=None,
        limit=10,
        fields=["content"],
    )

    assert [result.session_id for result in results] == ["sess-recall", "sess-other"]
    assert math.isclose(results[0].score, results[1].score)


def test_search_has_no_context_mode_branching() -> None:
    search_path = (
        Path(__file__).resolve().parents[2] / "packages/recall/src/recall/services/search.py"
    )

    assert "context_mode" not in search_path.read_text(encoding="utf-8")


def test_hybrid_search_uses_raw_query_for_embedding_and_bm25(monkeypatch, tmp_path) -> None:
    class CapturingBackend:
        dimensions = _TEST_EMBED_DIM
        model_id = "capture"
        query_prefix = ""

        def __init__(self) -> None:
            self.inputs: list[str] = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.inputs.extend(texts)
            return [_vec(0) for _text in texts]

    bm25_queries: list[str] = []

    def capture_hybrid(
        conn,
        query,
        query_embedding,
        source,
        limit,
        fields,
        *,
        session=None,
        config=None,
    ):
        del conn, query_embedding, source, limit, fields, session, config
        bm25_queries.append(query)
        return []

    monkeypatch.setattr(search_module, "load_fts_extension", lambda _conn: None)
    monkeypatch.setattr(search_module, "_hybrid_search", capture_hybrid)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    backend = CapturingBackend()
    config = AppConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data/recall.duckdb",
        lock_path=tmp_path / "data/recall.lock",
        config_path=tmp_path / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )

    search_module.search(
        query="foo bar",
        source=None,
        tool=None,
        mode=SearchMode.HYBRID,
        config=config,
        conn=conn,
        embed_backend=backend,
    )

    assert backend.inputs == ["foo bar"]
    assert bm25_queries == ["foo bar"]


def test_keyword_search_messages_filters_by_session() -> None:
    """Session filter restricts keyword message search to a single session."""
    from recall.services.search import _search_messages

    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _seed_two_sessions(conn)

    create_fts_indexes(conn, FtsConfig(backend="duckdb"))

    # Without session filter — both sessions match
    all_results = _search_messages(conn, "git rebase", source=None, limit=10, fields=["content"])
    assert len(all_results) == 2

    # With session filter — only matching session
    filtered = _search_messages(
        conn, "git rebase", source=None, limit=10, fields=["content"], session="sess-alpha"
    )
    assert len(filtered) == 1
    assert filtered[0].session_id == "sess-alpha"
    assert filtered[0].message_id == "msg-alpha"


def test_keyword_search_tool_calls_filters_by_session() -> None:
    """Session filter restricts keyword tool_call search to a single session."""
    from recall.services.search import _search_tool_calls

    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _seed_two_sessions(conn)

    conn.execute("INSTALL fts; LOAD fts")
    conn.execute("PRAGMA create_fts_index('tool_calls', 'id', 'bash_command', overwrite=1)")

    # Without session filter — both match
    all_results = _search_tool_calls(conn, "git rebase", source=None, tool=None, limit=10)
    assert len(all_results) == 2

    # With session filter
    filtered = _search_tool_calls(
        conn, "git rebase", source=None, tool=None, limit=10, session="sess-beta"
    )
    assert len(filtered) == 1
    assert filtered[0].session_id == "sess-beta"
    assert filtered[0].tool_call_id == "tc-beta"


def test_vector_search_messages_filters_by_session() -> None:
    """Session filter restricts vector message search to a single session."""
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _seed_two_sessions(conn)

    # Both sessions get identical embeddings so both would normally match
    embed_sql = (
        "INSERT INTO message_embeddings "
        "(message_id, content_embedding, thinking_embedding) "
        "VALUES (?, ?, ?)"
    )
    conn.execute(embed_sql, ["msg-alpha", _vec(0), None])
    conn.execute(embed_sql, ["msg-beta", _vec(0), None])

    # Without filter — both match
    all_results = _vector_search_messages(conn, _vec(0), source=None, limit=10)
    assert len(all_results) == 2

    # With session filter
    filtered = _vector_search_messages(conn, _vec(0), source=None, limit=10, session="sess-alpha")
    assert len(filtered) == 1
    assert filtered[0].session_id == "sess-alpha"


# ---- REQ-CLI-017: lexical_match labeling ----


def _build_search_config(tmp_path: Path, *, backend: str = "duckdb") -> AppConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
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


def _lexical_result(kind: str, **overrides: Any) -> SearchResult:
    base: dict[str, Any] = dict(
        kind=kind,
        session_id="session-1",
        source="codex",
        source_path="/tmp/session.jsonl",
        score=1.0,
        message_id="msg-1",
        tool_call_id=None if kind == "message" else "tc-1",
        role="assistant" if kind == "message" else None,
        content="hello" if kind == "message" else None,
        thinking=None,
        timestamp=None,
        tool_name=None if kind == "message" else "bash",
        bash_command=None if kind == "message" else "git status",
    )
    base.update(overrides)
    return SearchResult(**base)


def test_keyword_search_marks_results_lexical_without_probing_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _build_search_config(tmp_path)
    conn = duckdb.connect(str(config.db_path))
    ensure_schema(conn)
    try:
        _seed_two_sessions(conn)
        create_fts_indexes(conn, FtsConfig(backend="duckdb"))
        monkeypatch.setattr(
            search_module,
            "_probe_backend",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("keyword search probed an embedding backend")
            ),
        )
        results = search_module.search(
            query="git rebase",
            source=None,
            tool=None,
            mode=SearchMode.KEYWORD,
            config=config,
            conn=conn,
        )
        assert results
        assert all(r.lexical_match for r in results)
    finally:
        conn.close()


def test_tool_filtered_search_marks_results_lexical(tmp_path: Path) -> None:
    config = _build_search_config(tmp_path)
    conn = duckdb.connect(str(config.db_path))
    ensure_schema(conn)
    try:
        _seed_two_sessions(conn)
        create_fts_indexes(conn, FtsConfig(backend="duckdb"))
        results = search_module.search(
            query="git rebase",
            source=None,
            tool="bash",
            config=config,
            conn=conn,
        )
        assert results
        assert all(r.kind == "tool_call" for r in results)
        assert all(r.lexical_match for r in results)
    finally:
        conn.close()


def test_vector_search_marks_results_non_lexical(tmp_path: Path) -> None:
    config = _build_search_config(tmp_path)
    conn = duckdb.connect(str(config.db_path))
    ensure_schema(conn)
    try:
        _seed_session(conn)
        results = search_module.search(
            query="anything",
            source=None,
            tool=None,
            mode=SearchMode.VECTOR,
            config=config,
            conn=conn,
            query_embedding=_vec(0),
        )
        assert results
        assert not any(r.lexical_match for r in results)
    finally:
        conn.close()


def test_hybrid_search_marks_bm25_lexical_and_vector_only_not() -> None:
    """REQ-CLI-017: a BM25-leg row is lexical; a vector-only neighbor is not."""
    message_result = _lexical_result("message", message_id="msg-1", score=10.0)
    tool_result = _lexical_result("tool_call", tool_call_id="tc-1", score=9.0)

    original_search_all = search_module._search_all
    original_vector_search_all = search_module._vector_search_all
    search_module._search_all = lambda *_args, **_kwargs: [message_result]
    search_module._vector_search_all = lambda *_args, **_kwargs: [tool_result]
    try:
        results = _hybrid_search(
            conn=duckdb.connect(":memory:"),
            query="git",
            query_embedding=_vec(0),
            source=None,
            limit=10,
            fields=("content", "thinking", "bash"),
            k=60,
        )
    finally:
        search_module._search_all = original_search_all
        search_module._vector_search_all = original_vector_search_all

    by_kind = {result.kind: result for result in results}
    assert by_kind["message"].lexical_match is True  # matched the BM25 leg
    assert by_kind["tool_call"].lexical_match is False  # vector-only neighbor
