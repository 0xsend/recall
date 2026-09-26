from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from recall.core.config import ContextConfig
from recall.core.models import Message, Session
from recall.core.types import Role, Source
from recall.services.context import resolve_message_context
from recall.services.context_backends import ContextResult
from recall.services.context_backends import mlx_local as mlx_local_module
from recall.services.context_backends.mlx_local import MlxLocalBackend


@pytest.fixture(autouse=True)
def _clear_model_cache() -> object:
    """Reset the process-level (model, tokenizer) cache between tests.

    `_MODEL_CACHE` is keyed by model name and shared across the module, so
    without this the first test's tokenizer double leaks into every later test
    that reuses the default model name.
    """
    mlx_local_module._MODEL_CACHE.clear()
    yield
    mlx_local_module._MODEL_CACHE.clear()


class Tokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


class ChatTokenizer(Tokenizer):
    """Tokenizer double that exposes a chat template, like a real instruct model."""

    chat_template = "{{ messages }}"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        # Mirror mlx_lm's usage: a string prompt with role markers, no tokenization.
        assert add_generation_prompt is True
        assert tokenize is False
        content = messages[0]["content"]
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


class RaisingBackend:
    def is_available(self) -> bool:
        return True

    def generate_prefix(self, session: Session, message: Message) -> ContextResult:
        raise RuntimeError(f"cannot contextualize {session.id}/{message.id}")


def _session(*, content: str = "This is a long enough message for local context.") -> Session:
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


def test_mlx_local_min_chars_skip_does_not_call_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    mlx_lm = SimpleNamespace(
        load=lambda _model: (object(), Tokenizer()),
        generate=lambda *_args, **_kwargs: calls.append("generate") or "unused",
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    session = _session(content="short")
    backend = MlxLocalBackend(ContextConfig(mode="llm-local", min_chars=50))

    result = backend.generate_prefix(session, session.messages[1])

    assert result == ContextResult(prefix="", mode="off")
    assert calls == []


def test_mlx_local_successful_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def generate(_model, _tokenizer, prompt: str, *, max_tokens: int) -> str:
        captured["prompt"] = prompt
        captured["max_tokens"] = max_tokens
        return "short context"

    mlx_lm = SimpleNamespace(
        load=lambda model: (f"model:{model}", Tokenizer()),
        generate=generate,
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    config = ContextConfig(mode="llm-local", min_chars=5, max_tokens=25)
    session = _session()
    backend = MlxLocalBackend(config)

    result = backend.generate_prefix(session, session.messages[1])

    assert result.prefix == "[short context] "
    assert result.mode == "llm-local"
    assert result.model == config.model
    assert result.input_tokens > 0
    assert result.output_tokens == 2
    assert captured["max_tokens"] == 25
    assert "<chunk>This is a long enough message" in str(captured["prompt"])


def test_mlx_local_applies_chat_template_with_instruction_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An instruct model must receive a chat-templated prompt, not a raw string.

    Regression guard for the Qwen3-4B leakage: a raw prompt sent to
    `mlx_lm.generate` runs the model in completion mode (it echoes the
    instruction and rambles). The prompt handed to generate must be wrapped in
    the tokenizer chat template, and the configured instruction_prefix must ride
    inside the user turn.
    """
    captured: dict[str, object] = {}

    def generate(_model, _tokenizer, prompt: str, *, max_tokens: int) -> str:
        captured["prompt"] = prompt
        return "clean context"

    mlx_lm = SimpleNamespace(
        load=lambda model: (f"model:{model}", ChatTokenizer()),
        generate=generate,
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    config = ContextConfig(
        mode="llm-local",
        min_chars=5,
        max_tokens=512,
        instruction_prefix="/no_think ",
    )
    session = _session()
    backend = MlxLocalBackend(config)

    result = backend.generate_prefix(session, session.messages[1])

    prompt = str(captured["prompt"])
    assert prompt.startswith("<|im_start|>user\n")  # chat template applied
    assert prompt.rstrip().endswith("<|im_start|>assistant")  # generation prompt added
    assert "/no_think " in prompt  # instruction_prefix wired into the MLX path
    assert "<chunk>This is a long enough message" in prompt  # real instruction inside the turn
    assert result.prefix == "[clean context] "
    assert result.mode == "llm-local"


def test_mlx_local_trim_keeps_balanced_document_with_instruction_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long-document trimming must keep balanced <document> tags with a prefix set.

    Regression: prepending instruction_prefix ahead of `<document>` pushed
    `_trim_prompt` into its plain-text truncation fallback, which dropped the
    closing `</document>` (or the whole document) on long inputs — corrupting the
    exact Qwen3 `/no_think` path under normal trimming.
    """
    captured: dict[str, object] = {}

    def generate(_model, _tokenizer, prompt: str, *, max_tokens: int) -> str:
        captured["prompt"] = prompt
        return "ctx"

    mlx_lm = SimpleNamespace(
        load=lambda model: (f"model:{model}", ChatTokenizer()),
        generate=generate,
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    # A long chunk + document forces _trim_prompt to engage at a small budget.
    session = _session(content="word " * 500)
    config = ContextConfig(
        mode="llm-local",
        min_chars=5,
        max_tokens=50,
        instruction_prefix="/no_think\n\n",
    )
    backend = MlxLocalBackend(config)

    backend.generate_prefix(session, session.messages[1])

    prompt = str(captured["prompt"])
    assert "/no_think" in prompt  # prefix survives
    # Tags stay balanced: the trim must not strip the closing </document>.
    assert prompt.count("<document>") == 1
    assert prompt.count("</document>") == 1


def test_mlx_local_strips_think_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reasoning model's <think> block must not reach the stored prefix."""
    mlx_lm = SimpleNamespace(
        load=lambda model: (f"model:{model}", ChatTokenizer()),
        generate=lambda *_a, **_k: "<think>weighing options</think>actual summary",
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    session = _session()
    backend = MlxLocalBackend(ContextConfig(mode="llm-local", min_chars=5))

    result = backend.generate_prefix(session, session.messages[1])

    assert result.prefix == "[actual summary] "


def test_mlx_local_falls_back_to_raw_prompt_without_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Base models / tokenizers with no chat template keep the raw-prompt path."""
    captured: dict[str, object] = {}

    def generate(_model, _tokenizer, prompt: str, *, max_tokens: int) -> str:
        captured["prompt"] = prompt
        return "ctx"

    mlx_lm = SimpleNamespace(
        load=lambda model: (f"model:{model}", Tokenizer()),  # no apply_chat_template
        generate=generate,
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    session = _session()
    backend = MlxLocalBackend(ContextConfig(mode="llm-local", min_chars=5, max_tokens=512))

    backend.generate_prefix(session, session.messages[1])

    prompt = str(captured["prompt"])
    assert "<|im_start|>" not in prompt  # no chat markers
    assert prompt.startswith("<document>")  # raw instruction prompt, unchanged shape


def test_resolve_message_context_falls_back_to_off() -> None:
    session = _session()
    result = resolve_message_context(
        session,
        session.messages[1],
        ContextConfig(mode="llm-local", fallback="off"),
        RaisingBackend(),
    )

    assert result == ContextResult(prefix="", mode="off")


def test_resolve_message_context_fallback_error_propagates() -> None:
    session = _session()

    with pytest.raises(RuntimeError, match="cannot contextualize"):
        resolve_message_context(
            session,
            session.messages[1],
            ContextConfig(mode="llm-local", fallback="error"),
            RaisingBackend(),
        )
