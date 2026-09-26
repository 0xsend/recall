"""Context generation backends for contextual retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from recall.core.config import AppConfig, ContextConfig
from recall.core.models import Message, Session


@dataclass(frozen=True)
class ContextResult:
    prefix: str
    mode: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None


class ContextBackend(Protocol):
    def is_available(self) -> bool:
        """Probe whether this backend can serve generation at runtime."""

    def generate_prefix(self, session: Session, message: Message) -> ContextResult:
        """Produce per-message context for embedding and FTS inputs."""


# Modes whose backend is an external/optional dependency that must be present for
# the configured mode to function. "off"/"template" need no backend.
LLM_CONTEXT_MODES = frozenset({"llm-local", "llm-remote", "llm-codex"})

# Per-mode remediation appended to the fatal error. The backend's own
# `_load_failure` carries the precise cause (missing import, missing key, model
# load failure); this maps the mode to the concrete fix the operator must apply.
_MODE_REMEDIATION: dict[str, str] = {
    "llm-local": (
        "install the MLX extra, e.g. "
        '`uv tool install --force --editable --python 3.12 "packages/recall[mlx]"`'
    ),
    "llm-remote": (
        "install the Anthropic extra (recall[anthropic]) and set "
        "[embedding.context].api_key or the ANTHROPIC_API_KEY env var"
    ),
    "llm-codex": (
        "ensure the `codex` CLI is installed and on PATH (or set [embedding.context].executable)"
    ),
}


class ContextBackendUnavailableError(RuntimeError):
    """A configured `llm-*` context backend cannot run on this host.

    Raised — regardless of the `fallback` policy — when an LLM context mode is
    configured but its backend is unsupported here (missing extra, missing
    credential, model load failure). This is a *configuration* error distinct
    from a transient per-message generation failure, which `fallback` governs.
    Silently degrading to template would hide a misconfiguration the operator
    explicitly asked us to make, so it is always fatal (REQ-CTX-020).
    """

    def __init__(self, mode: str, reason: BaseException | None) -> None:
        self.mode = mode
        self.reason = reason
        hint = _MODE_REMEDIATION.get(mode, "install the matching backend extra")
        # The underlying reason already names the missing dependency in most
        # cases; include it so the message is self-contained even when the
        # remediation map lacks a mode-specific entry.
        detail = f": {reason}" if reason is not None else ""
        super().__init__(
            f"context mode {mode!r} is configured but its backend is unavailable on "
            f"this host{detail}. To fix, {hint}, then restart the daemon — or set "
            f"[embedding.context].mode to 'template' or 'off' to run without an LLM."
        )


def ensure_context_backend(config: AppConfig | ContextConfig) -> ContextBackend | None:
    """Build the configured context backend, hard-failing on an unsupported LLM mode.

    Returns the backend (``None`` for off/template). For ``llm-*`` modes this is
    the single chokepoint that converts "configured but unsupported" into a loud,
    actionable failure (``ContextBackendUnavailableError``) instead of a silent
    fallback to template. Callers run it at daemon startup and at the start of
    each index run so the failure surfaces immediately rather than per message.
    """
    context_config = config.embedding.context if isinstance(config, AppConfig) else config
    backend = get_context_backend(context_config)
    if context_config.mode not in LLM_CONTEXT_MODES:
        return backend
    if backend is not None and backend.is_available():
        return backend
    # `is_available()` records why initialisation failed on the backend class;
    # surface it so the operator sees the precise cause, not just the mode.
    reason = getattr(backend, "_load_failure", None)
    raise ContextBackendUnavailableError(context_config.mode, reason)


def get_context_backend(config: AppConfig | ContextConfig) -> ContextBackend | None:
    context_config = config.embedding.context if isinstance(config, AppConfig) else config
    if context_config.mode == "llm-local":
        from recall.services.context_backends.mlx_local import MlxLocalBackend

        return MlxLocalBackend(context_config)
    if context_config.mode == "llm-remote":
        from recall.services.context_backends.anthropic import AnthropicRemoteBackend

        return AnthropicRemoteBackend(context_config)
    if context_config.mode == "llm-codex":
        from recall.services.context_backends.codex_cli import CodexCliBackend

        return CodexCliBackend(context_config)
    return None


__all__ = [
    "LLM_CONTEXT_MODES",
    "AnthropicRemoteBackend",
    "CodexCliBackend",
    "ContextBackend",
    "ContextBackendUnavailableError",
    "ContextResult",
    "MlxLocalBackend",
    "ensure_context_backend",
    "get_context_backend",
]


def __getattr__(name: str) -> object:
    if name == "AnthropicRemoteBackend":
        from recall.services.context_backends.anthropic import AnthropicRemoteBackend

        return AnthropicRemoteBackend
    if name == "MlxLocalBackend":
        from recall.services.context_backends.mlx_local import MlxLocalBackend

        return MlxLocalBackend
    if name == "CodexCliBackend":
        from recall.services.context_backends.codex_cli import CodexCliBackend

        return CodexCliBackend
    raise AttributeError(name)
