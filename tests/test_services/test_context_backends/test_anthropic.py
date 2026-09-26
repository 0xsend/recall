from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from recall.core.config import ContextConfig
from recall.core.models import Message, Session
from recall.core.types import Role, Source
from recall.services.context import resolve_message_context
from recall.services.context_backends import ContextResult
from recall.services.context_backends.anthropic import AnthropicRemoteBackend


class FakeMessages:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response
        self.error = error

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.response is None:
            raise AssertionError("fake Anthropic response was not configured")
        return self.response


def _response(
    text: str = "short context",
    *,
    input_tokens: int = 11,
    cache_read_input_tokens: int = 17,
    cache_creation_input_tokens: int = 19,
    output_tokens: int = 5,
) -> object:
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            output_tokens=output_tokens,
        ),
    )


class FakeAnthropicConstructor:
    def __init__(self, client: object) -> None:
        self._client = client
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self._client


def _install_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    messages: FakeMessages,
) -> FakeAnthropicConstructor:
    client = SimpleNamespace(messages=messages)
    ctor = FakeAnthropicConstructor(client)
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=ctor))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    return ctor


def _session(*, content: str = "This is a long enough message for remote context.") -> Session:
    session = Session(
        id="session-1",
        source=Source.CODEX,
        source_path="/tmp/session.jsonl",
        file_mtime=1.0,
        file_size=1,
        git_repo="acme/recall",
        git_branch="main",
    )
    session.messages.extend(
        [
            Message(
                id="message-0",
                session_id=session.id,
                idx=0,
                role=Role.USER,
                content="previous message with useful context",
            ),
            Message(
                id="message-1",
                session_id=session.id,
                idx=1,
                role=Role.ASSISTANT,
                content=content,
            ),
            Message(
                id="message-2",
                session_id=session.id,
                idx=2,
                role=Role.USER,
                content="following message with useful context",
            ),
        ]
    )
    return session


def test_anthropic_min_chars_skip_does_not_call_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = FakeMessages(_response())
    _install_anthropic(monkeypatch, messages)
    session = _session(content="short")
    backend = AnthropicRemoteBackend(ContextConfig(mode="llm-remote", min_chars=50))

    result = backend.generate_prefix(session, session.messages[1])

    assert result == ContextResult(prefix="", mode="off")
    assert messages.calls == []


def test_anthropic_successful_generation_populates_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = FakeMessages(_response())
    _install_anthropic(monkeypatch, messages)
    config = ContextConfig(
        mode="llm-remote",
        model="claude-haiku-test",
        min_chars=5,
        max_tokens=25,
    )
    session = _session()
    backend = AnthropicRemoteBackend(config)

    result = backend.generate_prefix(session, session.messages[1])

    assert result == ContextResult(
        prefix="[short context] ",
        mode="llm-remote",
        input_tokens=47,
        output_tokens=5,
        model="claude-haiku-test",
    )
    call = messages.calls[0]
    assert call["model"] == "claude-haiku-test"
    assert call["max_tokens"] == 25


def test_anthropic_default_model_is_remote_haiku(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = FakeMessages(_response())
    _install_anthropic(monkeypatch, messages)
    session = _session()
    backend = AnthropicRemoteBackend(ContextConfig(mode="llm-remote", min_chars=5))

    result = backend.generate_prefix(session, session.messages[1])

    assert result.model == "claude-haiku-4-5-20251001"
    assert messages.calls[0]["model"] == "claude-haiku-4-5-20251001"


def test_anthropic_prompt_cache_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = FakeMessages(_response())
    _install_anthropic(monkeypatch, messages)
    session = _session()
    backend = AnthropicRemoteBackend(ContextConfig(mode="llm-remote", min_chars=5))

    backend.generate_prefix(session, session.messages[1])

    content = messages.calls[0]["messages"][0]["content"]  # type: ignore[index]
    document_block = content[0]
    chunk_block = content[1]
    assert document_block["text"].startswith("<document>0:user: previous message")
    assert document_block["cache_control"] == {"type": "ephemeral"}
    assert "<chunk>This is a long enough message" in chunk_block["text"]
    assert "cache_control" not in chunk_block


def test_anthropic_reuses_document_block_per_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = FakeMessages(_response())
    _install_anthropic(monkeypatch, messages)
    session = _session()
    backend = AnthropicRemoteBackend(ContextConfig(mode="llm-remote", min_chars=5))

    backend.generate_prefix(session, session.messages[0])
    backend.generate_prefix(session, session.messages[1])

    first_content = messages.calls[0]["messages"][0]["content"]  # type: ignore[index]
    second_content = messages.calls[1]["messages"][0]["content"]  # type: ignore[index]
    assert first_content[0] is second_content[0]


def test_anthropic_falls_back_to_template_on_message_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = FakeMessages(error=RuntimeError("remote failure"))
    _install_anthropic(monkeypatch, messages)
    session = _session()
    backend = AnthropicRemoteBackend(ContextConfig(mode="llm-remote", min_chars=5))

    result = resolve_message_context(
        session,
        session.messages[1],
        ContextConfig(mode="llm-remote", fallback="template", min_chars=5),
        backend,
    )

    assert result == ContextResult(prefix="[acme/recall main] ", mode="template")


def test_anthropic_unavailable_without_api_key_engages_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = FakeMessages(_response())
    client = SimpleNamespace(messages=messages)
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda: client))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    session = _session()
    backend = AnthropicRemoteBackend(ContextConfig(mode="llm-remote", min_chars=5))

    result = resolve_message_context(
        session,
        session.messages[1],
        ContextConfig(mode="llm-remote", fallback="template", min_chars=5),
        backend,
    )

    assert backend.is_available() is False
    assert result == ContextResult(prefix="[acme/recall main] ", mode="template")
    assert messages.calls == []


def test_anthropic_input_tokens_include_cache_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = FakeMessages(
        _response(input_tokens=3, cache_read_input_tokens=101, cache_creation_input_tokens=7)
    )
    _install_anthropic(monkeypatch, messages)
    session = _session()
    backend = AnthropicRemoteBackend(ContextConfig(mode="llm-remote", min_chars=5))

    result = backend.generate_prefix(session, session.messages[1])

    assert result.input_tokens == 111


@pytest.mark.parametrize(
    ("config_overrides", "env_api_key", "expected_kwargs"),
    [
        # Regression: when base_url/timeout are unset, the SDK constructor receives
        # no kwargs so it reads ANTHROPIC_API_KEY from env and uses its default
        # base URL -- the documented "env-var resolution by SDK" path.
        ({}, "test-key-not-real", {}),
        (
            {"base_url": "http://litellm.local:4000"},
            "test-key-not-real",
            {"base_url": "http://litellm.local:4000"},
        ),
        ({"timeout": 30.0}, "test-key-not-real", {"timeout": 30.0}),
        (
            {"base_url": "http://litellm.local:4000", "timeout": 15.0},
            "test-key-not-real",
            {"base_url": "http://litellm.local:4000", "timeout": 15.0},
        ),
        # config.api_key is passed explicitly so launchd-managed daemons (which
        # don't inherit shell env) can still authenticate; the config value
        # alone is sufficient.
        ({"api_key": "sk-from-config"}, None, {"api_key": "sk-from-config"}),
        # Precedence rule: config.api_key wins over env so a user-overridden
        # config isn't silently shadowed by stale env from a parent shell or
        # systemd unit.
        ({"api_key": "sk-from-config"}, "sk-from-env", {"api_key": "sk-from-config"}),
    ],
    ids=["default", "base-url", "timeout", "base-url-and-timeout", "config-key", "config-key-wins"],
)
def test_anthropic_constructor_receives_only_configured_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    config_overrides: dict[str, Any],
    env_api_key: str | None,
    expected_kwargs: dict[str, Any],
) -> None:
    messages = FakeMessages(_response())
    ctor = _install_anthropic(monkeypatch, messages)
    if env_api_key is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", env_api_key)
    session = _session()
    backend = AnthropicRemoteBackend(
        ContextConfig(mode="llm-remote", min_chars=5, **config_overrides)
    )

    backend.generate_prefix(session, session.messages[1])

    assert ctor.calls == [expected_kwargs]


def test_anthropic_available_with_config_api_key_and_no_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: pre-v0.14.1 the backend hard-required ANTHROPIC_API_KEY in env.
    # Now config.api_key alone must satisfy availability so launchd daemons work.
    messages = FakeMessages(_response())
    _install_anthropic(monkeypatch, messages)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = AnthropicRemoteBackend(
        ContextConfig(mode="llm-remote", min_chars=5, api_key="sk-from-config")
    )

    assert backend.is_available() is True


def test_anthropic_instruction_prefix_is_prepended_to_chunk_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Qwen3's `/no_think` directive must reach the model as the very first thing
    # in the chunk message; trailing newlines are preserved so the model parses
    # the marker correctly.
    messages = FakeMessages(_response())
    _install_anthropic(monkeypatch, messages)
    session = _session()
    backend = AnthropicRemoteBackend(
        ContextConfig(mode="llm-remote", min_chars=5, instruction_prefix="/no_think\n\n")
    )

    backend.generate_prefix(session, session.messages[1])

    chunk_block = messages.calls[0]["messages"][0]["content"][1]  # type: ignore[index]
    assert chunk_block["text"].startswith("/no_think\n\nHere is the chunk")


def test_anthropic_instruction_prefix_unset_keeps_legacy_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When instruction_prefix is not set, the chunk prompt is exactly what it
    # was before v0.14.2 — no surprise leading characters for models that don't
    # need a hint.
    messages = FakeMessages(_response())
    _install_anthropic(monkeypatch, messages)
    session = _session()
    backend = AnthropicRemoteBackend(ContextConfig(mode="llm-remote", min_chars=5))

    backend.generate_prefix(session, session.messages[1])

    chunk_block = messages.calls[0]["messages"][0]["content"][1]  # type: ignore[index]
    assert chunk_block["text"].startswith("Here is the chunk")


@pytest.mark.parametrize(
    ("text", "expected_prefix"),
    [
        # Qwen3 with `/no_think` still emits an empty `<think></think>` wrapper
        # that would otherwise leak into stored context, polluting embeddings
        # and BM25.
        ("<think>\n\n</think>\n\nreal context here", "[real context here] "),
        # Larger reasoning models can emit substantial chain-of-thought before
        # the final summary. We keep only the post-think portion.
        (
            "<think>\nLet me think about this carefully. The chunk is about X and Y.\n"
            "</think>\n\nDiscussion of X and Y in the context of debugging.",
            "[Discussion of X and Y in the context of debugging.] ",
        ),
        # Regression: stripping must be a no-op for non-thinking models. Llama,
        # Mistral, Anthropic-direct, etc. never emit `<think>` and their output
        # passes through verbatim.
        ("Plain summary with no think tags.", "[Plain summary with no think tags.] "),
    ],
    ids=["empty-think", "nonempty-think", "no-think"],
)
def test_anthropic_strips_think_block_from_response(
    monkeypatch: pytest.MonkeyPatch, text: str, expected_prefix: str
) -> None:
    messages = FakeMessages(_response(text=text))
    _install_anthropic(monkeypatch, messages)
    session = _session()
    backend = AnthropicRemoteBackend(ContextConfig(mode="llm-remote", min_chars=5))

    result = backend.generate_prefix(session, session.messages[1])

    assert result.prefix == expected_prefix
