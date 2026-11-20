"""Ordinary model fallback cannot convert a terminal graph to successful output."""

import duckdb
import pytest
from recall.core.config import AppConfig, ContextConfig
from recall.core.models import Message, Session
from recall.core.types import Role, Source
from recall.services.context import resolve_message_context, resolve_message_contexts
from recall.services.context_backends import ContextResult
from recall.services.embed_phase import (
    EmbedPhaseState,
    PendingEmbeds,
    PreparedEmbedCycle,
    generate_prepared_embed_cycle,
)


class FatalBackend:
    def is_available(self) -> bool:
        return True

    def generate_prefix(self, session: Session, message: Message) -> ContextResult:
        raise duckdb.FatalException("Failed to delete all rows from index in context model")

    def plan_batches(self, session: Session, messages: list[Message]) -> list[list[Message]]:
        return [messages]

    def generate_prefixes(self, session: Session, messages: list[Message]) -> list[ContextResult]:
        return [self.generate_prefix(session, messages[0])]


@pytest.mark.parametrize("batch", [False, True])
def test_terminal_context_error_propagates_before_template_fallback(batch: bool) -> None:
    session = Session(
        id="session-1",
        source=Source.CODEX,
        source_path="capture.jsonl",
        file_mtime=1.0,
        file_size=1,
        git_repo="recall",
        git_branch="main",
    )
    message = Message(id="m1", session_id=session.id, idx=0, role=Role.USER, content="hello")
    config = ContextConfig(mode="llm-codex", fallback="template")
    with pytest.raises(duckdb.FatalException, match="in context model"):
        if batch:
            resolve_message_contexts(session, [message], config, FatalBackend())
        else:
            resolve_message_context(session, message, config, FatalBackend())


def test_terminal_generation_error_propagates_before_error_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = EmbedPhaseState(_config=AppConfig.load())

    def fatal() -> None:
        raise duckdb.FatalException("Failed to delete all rows from index in embed model")

    monkeypatch.setattr(state, "load_backend", fatal)
    prepared = PreparedEmbedCycle(PendingEmbeds(1, 0, ["s1"]), [])
    with pytest.raises(duckdb.FatalException, match="in embed model"):
        generate_prepared_embed_cycle(AppConfig.load(), state, prepared)


@pytest.mark.parametrize("batch", [False, True])
def test_daemon_context_origin_is_published_before_fallback_policy(batch: bool) -> None:
    import threading

    from recall.services.context import resolve_message_context_on, resolve_message_contexts_on
    from recall.services.owned_jobs import DaemonJobs, ObservedFailure

    published: list[tuple[BaseException, str]] = []
    stopped = threading.Event()
    owner = DaemonJobs(lambda error, site, role: published.append((error, site)), stopped.set)
    session = Session(
        id="s1",
        source=Source.CODEX,
        source_path="capture.jsonl",
        file_mtime=1.0,
        file_size=1,
    )
    message = Message(id="m1", session_id="s1", idx=0, role=Role.USER, content="hello")
    config = ContextConfig(mode="llm-codex", fallback="template")

    def model() -> object:
        if batch:
            return resolve_message_contexts_on(
                session, [message], config, FatalBackend(), observer=owner
            )
        return resolve_message_context_on(session, message, config, FatalBackend(), observer=owner)

    result = owner.inline("model", model)
    assert isinstance(result, ObservedFailure)
    assert stopped.is_set()
    assert len(published) == 1
    assert published[0][0] is result.error
    assert published[0][1] == ("context:generate_prefixes" if batch else "context:generate_prefix")
