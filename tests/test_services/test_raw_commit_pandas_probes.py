"""A raw commit must not re-search for an absent pandas once per row (REQ-RECON-027).

DuckDB 1.5.5 searches for an optional ``pandas`` twice for every bound non-NULL
parameter, and Python caches nothing about a failed import, so every search walks
``sys.path`` again. The raw-commit path binds one parameter per message and per
tool call when it reads persisted rows back and when it deletes obsolete ones, so
one commit of a 2.9k-message session paid ~25k failed imports.

The contract here is that the cost is bounded by the session, not by its size:
quadrupling the rows must not change the probe count. Rendering writer-owned ids
as SQL literals instead of binding them is what buys that, so these tests also
hold the change invisible for an id that cannot be a literal and for one that
would be an injection if the predicate ever interpolated an unvalidated value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import pytest
from recall.core.models import Message, Session, TailFacts
from recall.core.types import Role, Source
from recall.db.schema import ensure_schema
from recall.services.indexer import _write_session

from scripts.benchmark_raw_commit import PandasProbeCounter, build_session

if TYPE_CHECKING:
    from collections.abc import Iterator

SMALL = 64
LARGE = 256
# The per-session scalars a commit legitimately binds: the session identity row,
# its state row, and the host update. Comfortably above the 42 observed today,
# far below the 3,038 a 600-message append cost before this contract existed.
PER_SESSION_PROBE_CEILING = 200
HOSTILE_ID = "x' OR '1'='1"


@pytest.fixture
def counter() -> Iterator[PandasProbeCounter]:
    """Tally probes only after the one-off optional-dependency imports settle.

    ``pyarrow``'s own first import searches for pandas as well. That is a cost
    per process, not per row, so a throwaway commit pays it before any tally.
    """
    with PandasProbeCounter() as probe_counter:
        _commit_probes(1, probe_counter)
        probe_counter.take()
        yield probe_counter


def _commit_probes(messages: int, counter: PandasProbeCounter) -> list[int]:
    """Commit a session, then commit it rewritten, tallying probes per commit."""
    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn)
        tallies: list[int] = []
        for session in (build_session(messages), build_session(messages, edit=" revised")):
            counter.counting = True
            counter.take()
            _write_session(conn, session, tail_facts=TailFacts())
            counter.counting = False
            tallies.append(counter.take())
        return tallies
    finally:
        conn.close()


def _two_message_session(
    first_id: str, content: str, *, session_id: str = "probe-session"
) -> Session:
    """A session whose first message carries a caller-chosen id."""
    return Session(
        id=session_id,
        source=Source.CLAUDE_CODE,
        source_path=f"/tmp/{session_id}.jsonl",
        file_mtime=1.0,
        file_size=1,
        messages=[
            Message(
                id=first_id,
                session_id=session_id,
                idx=0,
                role=Role.USER,
                content=content,
            ),
            Message(
                id=f"{session_id}-second",
                session_id=session_id,
                idx=1,
                role=Role.ASSISTANT,
                content=content,
            ),
        ],
        message_count=2,
    )


def test_raw_commit_probes_do_not_grow_with_the_session(counter: PandasProbeCounter) -> None:
    """REQ-RECON-027: four times the rows must not mean four times the imports."""
    small = _commit_probes(SMALL, counter)
    large = _commit_probes(LARGE, counter)

    assert large == small, (
        f"a {LARGE}-message session probed for pandas {large} times against "
        f"{small} for {SMALL} messages: binding still scales with the session"
    )


def test_a_large_raw_commit_stays_under_the_per_session_probe_ceiling(
    counter: PandasProbeCounter,
) -> None:
    """REQ-RECON-027: what remains is the bounded per-session row binding."""
    insert, rewrite = _commit_probes(LARGE, counter)

    assert insert <= PER_SESSION_PROBE_CEILING, f"insert probed for pandas {insert} times"
    assert rewrite <= PER_SESSION_PROBE_CEILING, f"rewrite probed for pandas {rewrite} times"


def test_an_id_that_reads_as_sql_never_reaches_another_session() -> None:
    """An injection-shaped message id selects its own row and leaves others alone."""
    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn)
        bystander = _two_message_session("d" * 32, "bystander", session_id="bystander-session")
        _write_session(conn, bystander, tail_facts=TailFacts())

        _write_session(conn, _two_message_session(HOSTILE_ID, "before"), tail_facts=TailFacts())
        _write_session(conn, _two_message_session(HOSTILE_ID, "after"), tail_facts=TailFacts())

        assert conn.execute(
            "SELECT content FROM message_state WHERE message_id = ?", [HOSTILE_ID]
        ).fetchall() == [("after",)]
        assert conn.execute(
            "SELECT count(*) FROM message_state WHERE content = 'bystander'"
        ).fetchone() == (2,)
    finally:
        conn.close()


def test_a_rewrite_keeps_every_row_when_an_id_cannot_be_a_literal() -> None:
    """An id outside the validated token set must still commit the same rows."""
    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn)
        _write_session(conn, _two_message_session("quote'id", "before"), tail_facts=TailFacts())
        _write_session(conn, _two_message_session("quote'id", "after"), tail_facts=TailFacts())

        assert conn.execute(
            "SELECT message_id, content FROM message_state ORDER BY message_id"
        ).fetchall() == [("probe-session-second", "after"), ("quote'id", "after")]
    finally:
        conn.close()
