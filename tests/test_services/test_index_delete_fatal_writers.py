"""A recognized index-delete fatal stops every session-label writer at its origin.

DuckDB reports a diverged ART index as ``FatalException: Failed to delete all
rows from index``. The injected statement failure below stands in for that
engine error: a real divergence cannot be provoked deterministically, so what
stays unverified here is the engine's own behaviour, not the writers' handling.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import duckdb
import pytest
from recall import _native
from recall.core.models import Message, Session, TailFacts
from recall.core.source_path import SourcePath
from recall.core.types import Role, Source
from recall.db.lease import StandaloneLifetime
from recall.db.schema import ensure_schema
from recall.services.indexer import (
    _incremental_write_session,
    _write_session,
    relabel_sessions,
)

from tests.keyword_turns import without_keyword_index

INDEX_DELETE = "Failed to delete all rows from index"
HOST_STAMP = "UPDATE session_state SET host"
SOURCE_PATH = "/tmp/label-session.jsonl"


class _InjectingConn:
    """Forwarding proxy that fails chosen statements and records the order of events."""

    def __init__(self, inner: duckdb.DuckDBPyConnection, events: list[str]) -> None:
        self._inner = inner
        self.events = events
        self.fail_on: dict[str, BaseException] = {}

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        for fragment, error in self.fail_on.items():
            if fragment in sql:
                self.events.append(f"fail:{fragment}")
                raise error
        self.events.append(
            sql.split()[0].upper() if sql.strip() in {"BEGIN", "ROLLBACK", "COMMIT"} else sql
        )
        return self._inner.execute(sql, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _RecordingLifetime(StandaloneLifetime):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def observe(self, error: BaseException, operation: str) -> bool:
        self._events.append(f"observe:{type(error).__name__}")
        return super().observe(error, operation)


def _session() -> Session:
    return Session(
        id="label-session",
        source=Source.CODEX,
        source_path=SOURCE_PATH,
        file_mtime=1.0,
        file_size=1,
        started_at=datetime(2026, 7, 23, 12, 0, 0, tzinfo=UTC),
        ended_at=datetime(2026, 7, 23, 12, 30, 0, tzinfo=UTC),
        messages=[
            Message(
                id="label-message",
                session_id="label-session",
                idx=0,
                role=Role.ASSISTANT,
                content="hello",
            )
        ],
        message_count=1,
    )


def _seeded() -> tuple[_InjectingConn, duckdb.DuckDBPyConnection, list[str]]:
    inner = _native.duckdb_connect(":memory:")
    ensure_schema(inner)
    _write_session(
        inner, _session(), turn=without_keyword_index(), tail_facts=TailFacts(), host="alpha"
    )
    events: list[str] = []
    return _InjectingConn(inner, events), inner, events


def _stored_host(inner: duckdb.DuckDBPyConnection) -> str:
    row = inner.execute("SELECT host FROM session_state WHERE session_id = 'label-session'")
    return cast("str", row.fetchone()[0])  # ty: ignore[not-subscriptable]


def test_host_stamp_after_a_session_write_surfaces_the_index_delete_fatal() -> None:
    conn, inner, events = _seeded()
    conn.fail_on[HOST_STAMP] = duckdb.FatalException(INDEX_DELETE)

    with pytest.raises(duckdb.FatalException, match=INDEX_DELETE):
        _write_session(
            cast("duckdb.DuckDBPyConnection", conn),
            _session(),
            turn=without_keyword_index(),
            tail_facts=TailFacts(),
            host="beta",
        )

    assert events[-1] == f"fail:{HOST_STAMP}", "the run kept writing after the fatal"
    assert _stored_host(inner) == "alpha"


def test_host_stamp_inside_an_incremental_append_stops_the_transaction() -> None:
    conn, inner, events = _seeded()
    conn.fail_on[HOST_STAMP] = duckdb.FatalException(INDEX_DELETE)

    with pytest.raises(duckdb.FatalException, match=INDEX_DELETE):
        _incremental_write_session(
            cast("duckdb.DuckDBPyConnection", conn),
            _session(),
            turn=without_keyword_index(),
            last_byte_offset=1,
            tail_facts=TailFacts(),
            host="beta",
        )

    after_failure = events[events.index(f"fail:{HOST_STAMP}") + 1 :]
    assert after_failure == ["ROLLBACK"], f"statements ran after the fatal: {after_failure}"
    assert _stored_host(inner) == "alpha"


@pytest.mark.parametrize("rollback_aborts", [False, True])
def test_relabel_observes_the_native_error_before_it_rolls_back(rollback_aborts: bool) -> None:
    conn, inner, events = _seeded()
    conn.fail_on[HOST_STAMP] = duckdb.FatalException(INDEX_DELETE)
    if rollback_aborts:
        conn.fail_on["ROLLBACK"] = duckdb.FatalException("rollback aborted")
    lifetime = _RecordingLifetime(events)
    paths = [SourcePath(Source.CODEX.value, SOURCE_PATH)]

    with (
        lifetime.lease(cast("duckdb.DuckDBPyConnection", conn)) as lease,
        pytest.raises(BaseException) as caught,
    ):
        relabel_sessions(lease, paths, "beta")

    leaves = [caught.value, *getattr(caught.value, "exceptions", ())]
    assert any(INDEX_DELETE in str(error) for error in leaves), repr(caught.value)
    rollback = next(i for i, e in enumerate(events) if e in {"ROLLBACK", "fail:ROLLBACK"})
    assert "observe:FatalException" in events[:rollback], events
    assert _stored_host(inner) == "alpha"


def test_relabel_applies_the_host_to_the_selected_sessions() -> None:
    _, inner, _ = _seeded()
    paths = [SourcePath(Source.CODEX.value, SOURCE_PATH)]

    with StandaloneLifetime().lease(inner) as lease:
        relabel_sessions(lease, paths, "beta")

    assert _stored_host(inner) == "beta"
