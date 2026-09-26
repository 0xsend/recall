"""Local MLX context generation backend."""

from __future__ import annotations

import threading
from typing import Any, ClassVar, cast

from recall.core.config import ContextConfig
from recall.core.models import Message, Session
from recall.services.context_backends import ContextResult
from recall.services.context_backends._text import strip_think_blocks

# Process-level cache of loaded (model, tokenizer) pairs keyed by model name.
# In watch mode `_prepare_context_run` builds a fresh MlxLocalBackend per indexed
# session; without this cache each one called `mlx_lm.load()`, re-reading the
# weights AND making a huggingface.co revision call on every session (~0.73s plus
# one network round-trip each — REQ-CTX-021). Weights are immutable for a given
# model name, so a fresh instance can safely share them. The cache lives for the
# process; recall daemons are long-lived and load a single model.
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_cached_model(mlx_lm: Any, model_name: str) -> tuple[Any, Any]:
    """Return the loaded (model, tokenizer) for ``model_name``, loading once.

    Loading happens under the lock so concurrent first-loads of the same model
    collapse to a single `mlx_lm.load()` rather than racing and double-allocating
    GPU memory. Daemon indexing is already serialized, so contention is negligible.
    """
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is None:
            cached = mlx_lm.load(model_name)
            _MODEL_CACHE[model_name] = cached
        return cached


class MlxLocalBackend:
    _load_failure: ClassVar[Exception | None] = None

    def __init__(self, config: ContextConfig) -> None:
        self._config = config
        self._available: bool | None = None
        self._mlx_lm: Any | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            self._ensure_loaded()
        except Exception as err:
            type(self)._load_failure = err
            self._available = False
            return False
        self._available = True
        return True

    def generate_prefix(self, session: Session, message: Message) -> ContextResult:
        if len(message.content or "") < self._config.min_chars:
            return ContextResult(prefix="", mode="off")

        self._ensure_loaded()
        mlx_lm = self._mlx_lm
        if mlx_lm is None:
            raise RuntimeError("mlx_lm backend did not initialize")
        prompt = self._build_prompt(session, message)
        result = cast(
            str,
            mlx_lm.generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=self._config.max_tokens,
            ),
        )
        # Strip any <think> block before trimming whitespace: a Qwen3 "Thinking"
        # variant (or a user-pointed reasoning model) would otherwise bury the
        # summary behind its chain-of-thought. A no-op for the default
        # Instruct-2507 model, which does not emit them.
        text = strip_think_blocks(result).strip()
        prefix = f"[{text}] " if text else ""
        return ContextResult(
            prefix=prefix,
            mode="llm-local" if prefix else "off",
            input_tokens=self._count_tokens(prompt),
            output_tokens=self._count_tokens(result),
            model=self._config.model,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._mlx_lm is not None:
            return
        try:
            import mlx_lm
        except ImportError as err:
            raise RuntimeError("mlx_lm is not installed; install recall[mlx]") from err

        self._mlx_lm = mlx_lm
        self._model, self._tokenizer = _load_cached_model(mlx_lm, self._config.model)

    def _build_prompt(self, session: Session, message: Message) -> str:
        target_index = _message_index(session, message)
        context_messages = _window_messages(
            session.messages,
            target_index=target_index,
            surrounding_messages=self._config.batch_size,
        )
        document = "\n".join(_render_message(item) for item in context_messages)
        chunk = message.content or ""
        prompt = (
            f"<document>{document}</document>\n"
            "Here is the chunk we want to situate within the whole document\n"
            f"<chunk>{chunk}</chunk>\n"
            "Please give a short succinct context to situate this chunk within the overall "
            "document for the purposes of improving search retrieval of the chunk. Answer only "
            "with the succinct context and nothing else."
        )
        # Trim FIRST, on the bare prompt: _trim_prompt only preserves balanced
        # <document></document> tags when the prompt starts with the open tag, so
        # the instruction_prefix must not lead here (a prefix forces the plain-text
        # truncation fallback, which drops the closing tag on long documents).
        # instruction_prefix carries model-specific hints (e.g. Qwen3 `/no_think`),
        # kept identical to the anthropic/codex backends so one config knob steers
        # every backend; prepend it after trimming, then wrap in the chat template
        # — whose fixed role-marker overhead must not eat into the token budget.
        instruction = self._config.instruction_prefix or ""
        return self._apply_chat_template(f"{instruction}{self._trim_prompt(prompt)}")

    def _apply_chat_template(self, prompt: str) -> str:
        """Wrap the instruction in the model's chat template.

        `mlx_lm.generate` does not apply the chat template — handed a raw prompt,
        an *instruct* model runs in completion mode: it echoes/invents formatting
        instructions ("The answer should be 100 words…") and rambles past the
        budget instead of obeying "answer only with the succinct context"
        (observed leakage on Qwen3-4B-Instruct). Framing the prompt as a user
        turn restores instruct-mode behaviour — clean, single-summary output.

        Falls back to the raw prompt for base models or test doubles that expose
        no chat template, preserving the historical completion-mode path there.
        """
        tokenizer = self._tokenizer
        apply = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply) or getattr(tokenizer, "chat_template", None) is None:
            return prompt
        return cast(
            str,
            apply(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

    def _trim_prompt(self, prompt: str) -> str:
        if self._count_tokens(prompt) <= self._config.max_tokens:
            return prompt
        char_budget = max(self._config.max_tokens * 4, 1)
        if len(prompt) <= char_budget:
            return prompt
        chunk_marker = "\nHere is the chunk we want to situate within the whole document\n"
        marker_index = prompt.find(chunk_marker)
        if marker_index == -1:
            return prompt[:char_budget].rstrip()

        document_part = prompt[:marker_index]
        chunk_part = prompt[marker_index:]
        document_budget = max(char_budget - len(chunk_part), 0)
        open_tag = "<document>"
        close_tag = "</document>"
        if document_part.startswith(open_tag) and document_part.endswith(close_tag):
            body = document_part[len(open_tag) : -len(close_tag)]
            body_budget = max(document_budget - len(open_tag) - len(close_tag), 0)
            document_part = f"{open_tag}{body[:body_budget].rstrip()}{close_tag}"
        else:
            document_part = document_part[:document_budget].rstrip()
        return f"{document_part}{chunk_part}"

    def _count_tokens(self, text: str) -> int:
        tokenizer = self._tokenizer
        if tokenizer is None:
            return 0
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            encoded = encode(text)
            return len(encoded)
        return len(text.split())


def _message_index(session: Session, message: Message) -> int:
    for index, candidate in enumerate(session.messages):
        if candidate.id == message.id:
            return index
    return 0


def _window_messages(
    messages: list[Message],
    *,
    target_index: int,
    surrounding_messages: int,
) -> list[Message]:
    window = max(surrounding_messages, 0)
    before = window // 2
    after = window - before
    start = max(target_index - before, 0)
    end = min(target_index + after + 1, len(messages))
    return messages[start:end]


def _render_message(message: Message) -> str:
    content = message.content or message.thinking or ""
    return f"{message.idx}:{message.role.value}: {content}"
