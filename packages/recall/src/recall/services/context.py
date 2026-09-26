"""Context prefix rendering for retrieval inputs."""

from __future__ import annotations

import logging
import posixpath
from typing import Protocol, cast

from recall.core.config import ContextConfig
from recall.core.models import Message, Session
from recall.services.context_backends import ContextBackend, ContextResult

logger = logging.getLogger(__name__)


class _BatchedContextBackend(Protocol):
    def plan_batches(self, session: Session, messages: list[Message]) -> list[list[Message]]:
        """Return backend-specific message batches for one session."""

    def generate_prefixes(self, session: Session, messages: list[Message]) -> list[ContextResult]:
        """Generate contexts for one planned batch."""


def render_template_prefix(session: Session) -> tuple[str, str]:
    """Return the static template context prefix for a session."""
    if session.git_repo is None and session.cwd is None and session.git_branch is None:
        return "", "off"

    repo_label = session.git_repo or _cwd_basename(session.cwd) or "local"
    branch_or_head = session.git_branch or "HEAD"
    return f"[{repo_label} {branch_or_head}] ", "template"


def resolve_context(
    session: Session,
    mode: str,
    backend: ContextBackend | None = None,
) -> tuple[str, str]:
    """Return the resolved context text and mode for a session."""
    _ = backend
    if mode == "off":
        return "", "off"
    if mode == "template":
        return render_template_prefix(session)
    if mode in {"llm-local", "llm-remote", "llm-codex"}:
        raise ValueError(f"{mode} context requires resolve_message_context()")
    raise ValueError(f"unsupported context mode: {mode}")


def resolve_message_context(
    session: Session,
    message: Message,
    config: ContextConfig,
    backend: ContextBackend | None,
) -> ContextResult:
    """Return write-time context for one message, including LLM fallback policy."""
    if config.mode == "off":
        return ContextResult(prefix="", mode="off")
    if config.mode == "template":
        prefix, mode = render_template_prefix(session)
        return ContextResult(prefix=prefix, mode=mode)
    if config.mode not in {"llm-local", "llm-remote", "llm-codex"}:
        raise ValueError(f"unsupported context mode: {config.mode}")

    if backend is None or not backend.is_available():
        return _apply_fallback(session, message, config, reason="context backend unavailable")
    try:
        return backend.generate_prefix(session, message)
    except Exception:
        logger.exception(
            "context backend %s failed (session=%s message=%s); falling back to %s",
            config.mode,
            session.id,
            getattr(message, "idx", "?"),
            config.fallback,
        )
        if config.fallback == "error":
            raise
        return _apply_fallback(session, message, config, reason="context generation failed")


def resolve_message_contexts(
    session: Session,
    messages: list[Message],
    config: ContextConfig,
    backend: ContextBackend | None,
) -> list[ContextResult]:
    """Return write-time contexts for messages, using backend batch APIs when present."""

    if not messages:
        return []
    if config.mode == "off":
        return [ContextResult(prefix="", mode="off") for _message in messages]
    if config.mode == "template":
        prefix, mode = render_template_prefix(session)
        return [ContextResult(prefix=prefix, mode=mode) for _message in messages]
    if config.mode not in {"llm-local", "llm-remote", "llm-codex"}:
        raise ValueError(f"unsupported context mode: {config.mode}")

    if backend is None or not backend.is_available():
        return [
            _apply_fallback(session, message, config, reason="context backend unavailable")
            for message in messages
        ]

    batched_backend = _as_batched_backend(backend)
    if batched_backend is None:
        return [resolve_message_context(session, message, config, backend) for message in messages]

    positions_by_id = {message.id: index for index, message in enumerate(messages)}
    resolved: list[ContextResult | None] = [None] * len(messages)
    for batch in batched_backend.plan_batches(session, messages):
        if not batch:
            continue
        try:
            batch_results = batched_backend.generate_prefixes(session, batch)
            if len(batch_results) != len(batch):
                raise RuntimeError(
                    f"context backend returned {len(batch_results)} results "
                    f"for {len(batch)} requested messages"
                )
        except Exception:
            logger.exception(
                "context backend %s failed for batch (session=%s messages=%s); falling back to %s",
                config.mode,
                session.id,
                [getattr(message, "idx", "?") for message in batch],
                config.fallback,
            )
            if config.fallback == "error":
                raise
            batch_results = [
                _apply_fallback(session, message, config, reason="context generation failed")
                for message in batch
            ]
        for message, result in zip(batch, batch_results, strict=True):
            try:
                resolved[positions_by_id[message.id]] = result
            except KeyError as err:
                raise RuntimeError(
                    f"context backend planned unknown message {message.id!r}"
                ) from err

    missing = [
        message.id for message, result in zip(messages, resolved, strict=True) if result is None
    ]
    if missing:
        raise RuntimeError(f"context backend did not resolve messages: {missing}")
    return cast("list[ContextResult]", resolved)


def _as_batched_backend(backend: ContextBackend) -> _BatchedContextBackend | None:
    if callable(getattr(backend, "plan_batches", None)) and callable(
        getattr(backend, "generate_prefixes", None)
    ):
        return cast("_BatchedContextBackend", backend)
    return None


def _apply_fallback(
    session: Session,
    message: Message,
    config: ContextConfig,
    *,
    reason: str,
) -> ContextResult:
    _ = (message, reason)
    if config.fallback == "template":
        prefix, mode = render_template_prefix(session)
        return ContextResult(prefix=prefix, mode=mode)
    if config.fallback == "off":
        return ContextResult(prefix="", mode="off")
    raise RuntimeError("context backend failed and fallback policy is error")


def _cwd_basename(cwd: str | None) -> str:
    if cwd is None:
        return ""
    return posixpath.basename(cwd.rstrip("/"))
