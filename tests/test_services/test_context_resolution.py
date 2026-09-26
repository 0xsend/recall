"""Tests for context resolution fallback behaviour."""

from __future__ import annotations

import logging

from recall.core.config import ContextConfig
from recall.core.models import Message, Session
from recall.core.types import Role, Source
from recall.services.context import resolve_message_context
from recall.services.context_backends import ContextResult


class RaisingBackend:
    def is_available(self) -> bool:
        return True

    def generate_prefix(self, session: Session, message: Message) -> ContextResult:
        _ = (session, message)
        raise RuntimeError("boom")


class RaisingBatchedBackend:
    def is_available(self) -> bool:
        return True

    def generate_prefix(self, session: Session, message: Message) -> ContextResult:
        _ = (session, message)
        raise RuntimeError("single path should not be used")

    def plan_batches(self, session: Session, messages: list[Message]) -> list[list[Message]]:
        _ = session
        return [messages]

    def generate_prefixes(self, session: Session, messages: list[Message]) -> list[ContextResult]:
        _ = (session, messages)
        raise RuntimeError("batch boom")


def test_resolve_message_context_logs_backend_exception_before_template_fallback(
    caplog,
) -> None:
    session = Session(
        id="session-1",
        source=Source.CODEX,
        source_path="/tmp/session.jsonl",
        file_mtime=1.0,
        file_size=1,
        git_repo="recall",
        git_branch="main",
    )
    message = Message(
        id="m1",
        session_id=session.id,
        idx=1,
        role=Role.USER,
        content="This message is long enough to require context generation.",
    )
    config = ContextConfig(mode="llm-codex", fallback="template")
    caplog.set_level(logging.ERROR, logger="recall.services.context")

    result = resolve_message_context(session, message, config, RaisingBackend())

    assert result == ContextResult(prefix="[recall main] ", mode="template")
    records = [
        record for record in caplog.records if "falling back to template" in record.getMessage()
    ]
    assert records
    assert records[0].exc_info is not None
    assert records[0].exc_info[0] is RuntimeError
    assert str(records[0].exc_info[1]) == "boom"


def test_resolve_message_contexts_falls_back_for_every_batch_item() -> None:
    from recall.services.context import resolve_message_contexts

    session = Session(
        id="session-1",
        source=Source.CODEX,
        source_path="/tmp/session.jsonl",
        file_mtime=1.0,
        file_size=1,
        git_repo="recall",
        git_branch="main",
    )
    messages = [
        Message(
            id=f"m{idx}",
            session_id=session.id,
            idx=idx,
            role=Role.USER,
            content=f"This message {idx} is long enough to require context generation.",
        )
        for idx in range(3)
    ]
    session.messages.extend(messages)
    config = ContextConfig(mode="llm-codex", fallback="template")

    results = resolve_message_contexts(session, messages, config, RaisingBatchedBackend())

    assert results == [ContextResult(prefix="[recall main] ", mode="template")] * 3
