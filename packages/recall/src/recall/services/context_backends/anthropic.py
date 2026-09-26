"""Anthropic prompt-cached context generation backend."""

from __future__ import annotations

import importlib
import os
from typing import Any, ClassVar

from recall.core.config import DEFAULT_CONTEXT_MODEL, ContextConfig
from recall.core.models import Message, Session
from recall.services.context_backends import ContextResult
from recall.services.context_backends._text import strip_think_blocks

DEFAULT_REMOTE_MODEL = "claude-haiku-4-5-20251001"


class AnthropicRemoteBackend:
    _load_failure: ClassVar[Exception | None] = None

    def __init__(self, config: ContextConfig) -> None:
        self._config = config
        self._model = (
            DEFAULT_REMOTE_MODEL if config.model == DEFAULT_CONTEXT_MODEL else config.model
        )
        self._available: bool | None = None
        self._client: Any | None = None
        self._anthropic: Any | None = None
        self._document_blocks_by_session: dict[str, list[dict[str, Any]]] = {}

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
        client = self._client
        if client is None:
            raise RuntimeError("anthropic backend did not initialize")

        response = client.messages.create(
            model=self._model,
            max_tokens=self._config.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": self._build_content_blocks(session, message),
                }
            ],
        )
        text = _extract_response_text(response).strip()
        prefix = f"[{text}] " if text else ""
        usage = getattr(response, "usage", None)
        return ContextResult(
            prefix=prefix,
            mode="llm-remote" if prefix else "off",
            input_tokens=(
                _usage_value(usage, "input_tokens")
                + _usage_value(usage, "cache_read_input_tokens")
                + _usage_value(usage, "cache_creation_input_tokens")
            ),
            output_tokens=_usage_value(usage, "output_tokens"),
            model=self._model,
        )

    def _ensure_loaded(self) -> None:
        if self._client is not None and self._anthropic is not None:
            return
        try:
            anthropic = importlib.import_module("anthropic")
        except ImportError as err:
            raise RuntimeError("anthropic is not installed; install recall[anthropic]") from err

        # Accept the key from either source — env is the historical path, config.toml
        # is the new path for launchd/systemd-managed daemons that don't inherit the
        # invoking shell's env. Check availability before constructing the client so
        # the fallback path (template/off) engages cleanly via is_available().
        if self._config.api_key is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "anthropic api_key is not set "
                "(set ANTHROPIC_API_KEY env var or context.api_key in config.toml)"
            )

        self._anthropic = anthropic
        # Forward only the kwargs the user has set so the SDK falls back to its
        # native defaults (env-var resolution for api_key, anthropic.com base URL,
        # 600s timeout) for anything unspecified. When config.api_key is set we
        # pass it explicitly; otherwise the SDK reads ANTHROPIC_API_KEY itself.
        client_kwargs: dict[str, Any] = {}
        if self._config.api_key is not None:
            client_kwargs["api_key"] = self._config.api_key
        if self._config.base_url is not None:
            client_kwargs["base_url"] = self._config.base_url
        if self._config.timeout is not None:
            client_kwargs["timeout"] = self._config.timeout
        try:
            self._client = anthropic.Anthropic(**client_kwargs)
        except Exception as err:
            raise RuntimeError("anthropic client initialization failed") from err

    def _build_content_blocks(
        self,
        session: Session,
        message: Message,
    ) -> list[dict[str, Any]]:
        document_blocks = self._document_blocks_by_session.get(session.id)
        if document_blocks is None:
            document = "\n".join(_render_message(item) for item in session.messages)
            document_blocks = [
                {
                    "type": "text",
                    "text": f"<document>{document}</document>",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            self._document_blocks_by_session[session.id] = document_blocks

        chunk = message.content or ""
        # instruction_prefix typically carries model-specific hints like '/no_think'
        # for Qwen3 — passed through as-is, never sent when unset to avoid polluting
        # prompts for models that don't recognise it.
        prefix = self._config.instruction_prefix or ""
        chunk_block: dict[str, Any] = {
            "type": "text",
            "text": (
                f"{prefix}"
                "Here is the chunk we want to situate within the whole document\n"
                f"<chunk>{chunk}</chunk>\n"
                "Please give a short succinct context to situate this chunk within the overall "
                "document for the purposes of improving search retrieval of the chunk. Answer only "
                "with the succinct context and nothing else."
            ),
        }
        return [*document_blocks, chunk_block]


def _render_message(message: Message) -> str:
    content = message.content or message.thinking or ""
    return f"{message.idx}:{message.role.value}: {content}"


def _extract_response_text(response: object) -> str:
    content = getattr(response, "content", None)
    if not content:
        return ""
    for block in content:
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if isinstance(text, str):
            return strip_think_blocks(text)
    return ""


def _usage_value(usage: object | None, field: str) -> int:
    if usage is None:
        return 0
    value = getattr(usage, field, 0)
    if value is None:
        return 0
    return int(value)
