from __future__ import annotations

import hashlib

import duckdb
import pytest
from recall.core.models import Message, Session, ToolCall
from recall.core.types import EmbedKind, Role, Source
from recall.db.schema import ensure_schema
from recall.services.embeddings import (
    _REGISTRY,
    NORMALIZATION_VERSION,
    _prepare_text,
    any_backend_available,
    available_backends,
    context_version_for_mode,
    embed_session,
    normalize_bash_command,
    resolve_backend_name,
)

TEST_EMBED_DIM = 384
TEST_CACHE_NAMESPACE = "tests.backend:test-model"


class FailingOnSecondCallBackend:
    dimensions = TEST_EMBED_DIM
    model_id = "test-model"
    query_prefix = ""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated backend failure")
        return [[float(self.calls)] * self.dimensions for _ in texts]


class StaticBackend:
    dimensions = TEST_EMBED_DIM
    model_id = "test-model"
    query_prefix = ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * self.dimensions for _ in texts]


class RecordingBackend:
    dimensions = TEST_EMBED_DIM
    model_id = "test-model"
    query_prefix = ""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(self.calls))] * self.dimensions for _ in texts]


def _make_session() -> Session:
    tool_call = ToolCall(
        id="tc-1",
        session_id="session-1",
        message_id="msg-1",
        idx=0,
        tool_name="bash",
        bash_command="git status",
    )
    message = Message(
        id="msg-1",
        session_id="session-1",
        idx=0,
        role=Role.ASSISTANT,
        content="content text",
        thinking="thinking text",
        tool_calls=[tool_call],
    )
    return Session(
        id="session-1",
        source=Source.CODEX,
        source_path="/tmp/session.jsonl",
        file_mtime=1.0,
        file_size=1,
        messages=[message],
    )


def test_embed_session_is_atomic_on_backend_failure() -> None:
    session = _make_session()

    with pytest.raises(RuntimeError, match="simulated backend failure"):
        embed_session(session, FailingOnSecondCallBackend(), batch_size=64)

    message = session.messages[0]
    assert message.content_embedding is None
    assert message.thinking_embedding is None
    assert message.tool_calls[0].bash_embedding is None


def test_embed_session_reuses_normalized_bash_embeddings_within_session() -> None:
    backend = RecordingBackend()
    session = Session(
        id="session-1",
        source=Source.CODEX,
        source_path="/tmp/session.jsonl",
        file_mtime=1.0,
        file_size=1,
        messages=[
            Message(
                id="msg-1",
                session_id="session-1",
                idx=0,
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        session_id="session-1",
                        message_id="msg-1",
                        idx=0,
                        tool_name="bash",
                        bash_command="kubectl logs web-6dbc8bdc58-8cghv -c web",
                    ),
                    ToolCall(
                        id="tc-2",
                        session_id="session-1",
                        message_id="msg-1",
                        idx=1,
                        tool_name="bash",
                        bash_command="kubectl logs web-7ddf9ccf77-zx2lm -c web",
                    ),
                ],
            )
        ],
    )

    embed_session(session, backend, batch_size=64)

    assert backend.calls == [["kubectl logs <k8s-name> -c web"]]
    embeddings = [tool_call.bash_embedding for tool_call in session.messages[0].tool_calls]
    assert embeddings[0] == embeddings[1]


def test_normalize_bash_command_only_rewrites_pod_like_names() -> None:
    assert normalize_bash_command("kubectl logs web-6dbc8bdc58-8cghv -c web") == (
        "kubectl logs <k8s-name> -c web"
    )
    assert normalize_bash_command("docker logs recall-server") == "docker logs recall-server"
    assert normalize_bash_command("rg --files src/core-types") == "rg --files src/core-types"
    assert normalize_bash_command("git checkout feature/search-ranking") == (
        "git checkout feature/search-ranking"
    )


def test_embedding_cache_is_scoped_by_model_identity() -> None:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)

    backend1 = RecordingBackend()
    backend1.model_id = "model-1"
    backend2 = RecordingBackend()
    backend2.model_id = "model-2"

    embed_session(_make_session(), backend1, batch_size=64, conn=conn)
    embed_session(_make_session(), backend2, batch_size=64, conn=conn)

    assert len(backend1.calls) == 3
    assert len(backend2.calls) == 3


def test_embed_session_reuses_in_memory_cache_across_sessions() -> None:
    backend = RecordingBackend()
    cache: dict[str, list[float]] = {}

    first = Session(
        id="session-1",
        source=Source.CODEX,
        source_path="/tmp/session-1.jsonl",
        file_mtime=1.0,
        file_size=1,
        messages=[
            Message(
                id="msg-1",
                session_id="session-1",
                idx=0,
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        session_id="session-1",
                        message_id="msg-1",
                        idx=0,
                        tool_name="bash",
                        bash_command="kubectl logs web-6dbc8bdc58-8cghv -c web",
                    )
                ],
            )
        ],
    )
    second = Session(
        id="session-2",
        source=Source.CODEX,
        source_path="/tmp/session-2.jsonl",
        file_mtime=1.0,
        file_size=1,
        messages=[
            Message(
                id="msg-2",
                session_id="session-2",
                idx=0,
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(
                        id="tc-2",
                        session_id="session-2",
                        message_id="msg-2",
                        idx=0,
                        tool_name="bash",
                        bash_command="kubectl logs web-7ddf9ccf77-zx2lm -c web",
                    )
                ],
            )
        ],
    )

    embed_session(first, backend, batch_size=64, resolved_cache=cache)
    embed_session(second, backend, batch_size=64, resolved_cache=cache)

    assert backend.calls == [["kubectl logs <k8s-name> -c web"]]


def test_content_cache_key_without_context_matches_legacy_key() -> None:
    prepared = _prepare_text(EmbedKind.CONTENT, "content text", TEST_CACHE_NAMESPACE)
    expected = hashlib.sha256(
        f"{TEST_CACHE_NAMESPACE}:content:{NORMALIZATION_VERSION}:content text".encode()
    ).hexdigest()

    assert prepared.cache_key == expected


def test_content_cache_key_changes_with_context() -> None:
    empty_context = _prepare_text(EmbedKind.CONTENT, "content text", TEST_CACHE_NAMESPACE)
    contextual = _prepare_text(
        EmbedKind.CONTENT,
        "content text",
        TEST_CACHE_NAMESPACE,
        context_text="[repo branch] ",
    )

    assert contextual.cache_key != empty_context.cache_key


def test_llm_context_modes_use_distinct_cache_versions() -> None:
    assert context_version_for_mode("llm-local") != context_version_for_mode("llm-remote")
    assert context_version_for_mode("off") == 0
    assert context_version_for_mode("template") == 0


def test_bash_cache_key_ignores_context_text() -> None:
    empty_context = _prepare_text(EmbedKind.BASH, "git status", TEST_CACHE_NAMESPACE)
    contextual = _prepare_text(
        EmbedKind.BASH,
        "git status",
        TEST_CACHE_NAMESPACE,
        context_text="[repo branch] ",
    )

    assert contextual.cache_key == empty_context.cache_key


def test_embed_session_prefixes_content_and_thinking_but_not_bash() -> None:
    backend = RecordingBackend()

    embed_session(_make_session(), backend, batch_size=64, context_text="[repo branch] ")

    assert backend.calls == [
        ["[repo branch] content text"],
        ["[repo branch] thinking text"],
        ["git status"],
    ]


def test_embed_session_uses_legacy_cache_version_for_llm_skipped_messages() -> None:
    backend = RecordingBackend()
    session = _make_session()
    session.messages[0].context_text = ""
    session.messages[0].context_mode = "off"

    embed_session(
        session,
        backend,
        batch_size=64,
        context_version=context_version_for_mode("llm-remote"),
    )

    assert backend.calls == [["content text"], ["thinking text"], ["git status"]]


# ---- Backend registry tests ----


def test_resolve_backend_name_auto_picks_highest_priority(monkeypatch) -> None:
    from recall.core.embeddings import BackendDescriptor

    saved = dict(_REGISTRY)
    try:
        _REGISTRY.clear()
        _REGISTRY["alpha"] = BackendDescriptor(
            name="alpha",
            is_available=lambda: True,
            factory=lambda _model: StaticBackend(),
            priority=20,
        )
        _REGISTRY["beta"] = BackendDescriptor(
            name="beta",
            is_available=lambda: True,
            factory=lambda _model: StaticBackend(),
            priority=5,
        )

        assert resolve_backend_name("auto") == "beta"
        assert resolve_backend_name("alpha") == "alpha"
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


def test_resolve_backend_name_auto_raises_when_none_available() -> None:
    from recall.core.embeddings import BackendDescriptor

    saved = dict(_REGISTRY)
    try:
        _REGISTRY.clear()
        _REGISTRY["unavailable"] = BackendDescriptor(
            name="unavailable",
            is_available=lambda: False,
            factory=lambda _model: StaticBackend(),
            priority=10,
        )

        with pytest.raises(ValueError, match="no embedding backend available"):
            resolve_backend_name("auto")
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


def test_resolve_backend_name_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported embedding backend"):
        resolve_backend_name("nonexistent")


def test_available_backends_filters_unavailable() -> None:
    from recall.core.embeddings import BackendDescriptor

    saved = dict(_REGISTRY)
    try:
        _REGISTRY.clear()
        _REGISTRY["yes"] = BackendDescriptor(
            name="yes",
            is_available=lambda: True,
            factory=lambda _model: StaticBackend(),
            priority=10,
        )
        _REGISTRY["no"] = BackendDescriptor(
            name="no",
            is_available=lambda: False,
            factory=lambda _model: StaticBackend(),
            priority=5,
        )

        assert available_backends() == ["yes"]
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


def test_any_backend_available_returns_false_when_empty() -> None:
    saved = dict(_REGISTRY)
    try:
        _REGISTRY.clear()
        assert any_backend_available() is False
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


def test_mlx_is_registered() -> None:
    assert "mlx" in _REGISTRY


def test_auto_resolution_falls_back_to_onnx() -> None:
    from recall.core.embeddings import BackendDescriptor

    saved = dict(_REGISTRY)
    try:
        _REGISTRY.clear()
        _REGISTRY["mlx"] = BackendDescriptor(
            name="mlx",
            is_available=lambda: False,
            factory=lambda _model: StaticBackend(),
            priority=10,
        )
        _REGISTRY["onnx"] = BackendDescriptor(
            name="onnx",
            is_available=lambda: True,
            factory=lambda _model: StaticBackend(),
            priority=20,
        )
        assert resolve_backend_name("auto") == "onnx"
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)
