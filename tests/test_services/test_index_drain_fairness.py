"""An operator index request must not be paced by the background embed drain.

While the daemon drained an embed backlog, `recall index` became unusable: the
drain re-arms 0.01s after every draining cycle (`REQ-ADAPT-008`) and takes the
enrichment lock the request needs once per session it enriches, so the request
absorbed a whole embed cycle per changed session. On a host whose cycles
ran ~15s that is minutes of silence per request, and the refusal the operator
finally saw named a pause that was never set.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from lane_harness import FIXTURES, LANE_NOW, build_lane, lane_config
from recall.core.models import Message, Session, TailFacts
from recall.core.rpc_types import RpcError
from recall.core.types import Role, Source
from recall.services.rpc_server import PAUSED_MESSAGE, SHUTTING_DOWN_MESSAGE, RpcServer

# Long enough that a per-session wait is unmistakable against the control run,
# short enough that the whole case stays inside a normal unit-test budget.
EMBED_BATCH_SECONDS = 1.0


class PacedBackend:
    """An embedding backend whose every call costs a known, observable amount of time."""

    dimensions = 384
    model_id = "paced-test-model"
    query_prefix = ""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def embed(self, texts: list[str]) -> list[list[float]]:
        time.sleep(self.delay)
        return [[1.0] * self.dimensions for _ in texts]


def _pending_session(session_id: str) -> Session:
    """A long-idle session with un-embedded content, so the drain always has work."""
    return Session(
        id=session_id,
        source=Source.CODEX,
        source_path=f"/tmp/{session_id}.jsonl",
        file_mtime=1.0,
        file_size=1,
        messages=[
            Message(
                id=f"{session_id}-m{idx}",
                session_id=session_id,
                idx=idx,
                role=Role.ASSISTANT,
                content=f"pending body {idx}",
            )
            for idx in range(3)
        ],
        message_count=3,
    )


def _allow_embedding(monkeypatch: pytest.MonkeyPatch, state: Any, backend: Any) -> None:
    from recall.services.system_state import EmbedPreconditionResult

    monkeypatch.setattr(state, "load_backend", lambda: backend)
    monkeypatch.setattr(
        "recall.services.system_state.check_power",
        lambda: EmbedPreconditionResult(ok=True),
    )
    monkeypatch.setattr(
        "recall.services.system_state.check_load",
        lambda _threshold: EmbedPreconditionResult(ok=True),
    )


async def _drained_by_the_background_loop(server: Any) -> int:
    """Count backlog sessions the timer loop has embedded.

    The drain takes one session per cycle and only the backlog carries the
    `pending-` ids, so this counts background cycles that committed. Read from
    the database rather than from `EmbedPhaseState.loop_iterations`: that
    counter moves on every timer cycle, including the ones that deferred or
    found nothing, so an iteration delta does not measure committed work.
    """
    return await server._run_readonly(
        lambda conn: conn.execute(
            "SELECT COUNT(DISTINCT m.session_id) FROM messages m "
            "JOIN message_embeddings e ON e.message_id = m.id "
            "WHERE m.session_id LIKE 'pending-%'"
        ).fetchone()[0]
    )


def test_index_request_waits_at_most_one_in_flight_embed_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ADAPT-017: four changed sessions cost one embed cycle, not four.

    Red before the fix: the drain committed one cycle per session the request
    enriched (measured 4 overlapping cycles, 12.2s against an 8.3s idle-host
    control at a 1s cycle).
    """
    from recall.services.embed_phase import EmbedPhaseState
    from recall.services.indexer import _write_session

    lane = build_lane(tmp_path, monkeypatch)
    server = lane.server
    server._config = replace(
        server._config,
        daemon=replace(server._config.daemon, embed=True, embed_idle_session=1),
    )
    conn = server._get_conn()
    for index in range(40):
        _write_session(conn, _pending_session(f"pending-{index}"), tail_facts=TailFacts())

    changed = [lane.watched, lane.unwatched]
    for index in range(2):
        extra = lane.watched.parent / f"extra-{index}.jsonl"
        shutil.copy(FIXTURES / "claude_code" / "live_end_turn.jsonl", extra)
        extra.write_text(
            extra.read_text(encoding="utf-8").replace("live-end-turn", f"extra-{index}"),
            encoding="utf-8",
        )
        os.utime(extra, (LANE_NOW.timestamp(), LANE_NOW.timestamp()))
        changed.append(extra)
    for index, path in enumerate(changed):
        lane.append(path, f"changed {index}", uuid=f"fairness-{index}")

    state = EmbedPhaseState(_config=server._config)
    server._embed_state = state
    _allow_embedding(monkeypatch, state, PacedBackend(EMBED_BATCH_SECONDS))

    async def scenario() -> None:
        drain = await server._start_embed_phase()
        assert drain is not None
        try:
            deadline = time.monotonic() + 30
            while await _drained_by_the_background_loop(server) < 1 and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            before = await _drained_by_the_background_loop(server)
            assert before >= 1, "the drain never reached a steady cadence"

            summary = await asyncio.wait_for(
                server._handle_index({"since": "1d", "embed": True}, None), timeout=120
            )
            overlapped = await _drained_by_the_background_loop(server) - before

            assert summary.indexed == len(changed)
            assert overlapped <= 1, (
                f"{overlapped} background embed cycles ran during a "
                f"{summary.indexed}-session index request; at most the one already "
                "in flight may overlap"
            )

            resumed = time.monotonic() + 30
            while (
                await _drained_by_the_background_loop(server) <= before + overlapped
                and time.monotonic() < resumed
            ):
                await asyncio.sleep(0.05)
            assert await _drained_by_the_background_loop(server) > before + overlapped, (
                "the drain never resumed"
            )
        finally:
            server._shutdown_event.set()
            drain.cancel()
            with pytest.raises(asyncio.CancelledError):
                await drain

    asyncio.run(scenario())
    asyncio.run(server.stop())


def test_embed_loop_yield_is_reported_and_bounded(tmp_path: Path) -> None:
    """REQ-ADAPT-017/REQ-ADAPT-012: the wait is inspectable and cannot be forever."""
    from recall.services.embed_phase import EmbedPhaseState

    config = lane_config(tmp_path)
    server = RpcServer(config)
    state = EmbedPhaseState(_config=config)

    async def scenario() -> None:
        assert server._index_turns.in_flight == 0
        await server._yield_to_index_requests(state, timeout=5.0)
        assert state.stage == "", "an idle daemon must not report a yield it did not make"

        server._index_turns.enter()
        try:
            started = time.monotonic()
            await server._yield_to_index_requests(state, timeout=0.1)
            waited = time.monotonic() - started
        finally:
            server._index_turns.leave()

        assert state.stage == "index-request-yield"
        assert 0.1 <= waited < 5.0, "a request that never finishes must not hold the drain"

    asyncio.run(scenario())


def test_paused_and_shutting_down_refusals_carry_distinct_messages(tmp_path: Path) -> None:
    """REQ-ADAPT-018: the operator is told which condition actually holds."""
    from recall.services.coordinator import set_paused

    config = lane_config(tmp_path)
    server = RpcServer(config)

    assert server._require_reconciliation(config) is None

    server._shutdown_event.set()
    with pytest.raises(RpcError) as shutting_down:
        server._require_reconciliation(config)
    assert shutting_down.value.message == SHUTTING_DOWN_MESSAGE
    assert "paused" not in shutting_down.value.message
    assert "retry" in shutting_down.value.message

    assert set_paused(config, True) is True
    with pytest.raises(RpcError) as paused:
        server._require_reconciliation(config)
    assert paused.value.message == PAUSED_MESSAGE
    assert "shutting down" not in paused.value.message
    assert "recall daemon resume" in paused.value.message


def test_scoped_index_talks_while_scanning_even_when_nothing_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RPC-008: a `--since` run that filters out every file still sends frames.

    Red before the fix: frames were emitted only per surviving file, so
    `recall index --since 1d` over a corpus with nothing recent sent its first
    and only frame at the summary -- and the client's 300s idle timeout fired
    first on a daemon that was working.
    """
    lane = build_lane(tmp_path, monkeypatch)
    aged = LANE_NOW.timestamp() - 30 * 86400
    for path in (lane.watched, lane.unwatched):
        os.utime(path, (aged, aged))

    statuses: list[str] = []

    class RecordingClient:
        async def send_progress(self, **kwargs: Any) -> None:
            statuses.append(str(kwargs["status"]))

    summary = asyncio.run(
        lane.server._handle_index(
            {"since": "1d", "embed": False},
            cast(Any, RecordingClient()),
        )
    )
    asyncio.run(lane.server.stop())

    assert summary.total == 0, "the scope must have excluded every transcript"
    assert statuses[-1] == "done"
    assert "scanning" in statuses[:-1], (
        "the request sent nothing until its summary; a client bounded by an "
        "idle timeout cannot tell that apart from a wedged daemon"
    )


def test_the_embed_loop_reports_one_yield_per_run_of_requests(tmp_path: Path) -> None:
    """REQ-ADAPT-017: expected behaviour is announced once, not once per cycle.

    Red before the fix: every embed cycle logged the yield at INFO and its
    bounded resume at WARNING, so one `--full` run wrote 15-30 WARNINGs
    describing a daemon that was working exactly as designed.
    """
    from recall.services.embed_phase import EmbedPhaseState

    config = lane_config(tmp_path)
    server = RpcServer(config)
    state = EmbedPhaseState(_config=config)
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("recall.rpc_server")
    handler = Capture(level=logging.INFO)
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.INFO)

    async def keep_serving() -> None:
        while True:
            await asyncio.sleep(0.005)
            server._note_raw_turn_served()

    async def scenario() -> None:
        server._index_turns.enter()
        serving = asyncio.create_task(keep_serving())
        try:
            for _ in range(4):
                await server._yield_to_index_requests(state, timeout=0.05)
            announced = [record for record in records if "yielding to" in record.getMessage()]
            warnings = [record for record in records if record.levelno >= logging.WARNING]
            assert len(announced) == 1, (
                f"{len(announced)} yield announcements for one run of requests"
            )
            assert warnings == [], (
                "the drain warned about a wait while the scheduler was serving sources: "
                f"{[record.getMessage() for record in warnings]}"
            )

            # A yield that covers no served source at all is still worth one.
            serving.cancel()
            with suppress(asyncio.CancelledError):
                await serving
            records.clear()
            await server._yield_to_index_requests(state, timeout=0.05)
            assert [record for record in records if record.levelno >= logging.WARNING], (
                "a drain that stood down while nothing at all was served said nothing"
            )
        finally:
            serving.cancel()
            with suppress(asyncio.CancelledError):
                await serving
            server._index_turns.leave()

    try:
        asyncio.run(scenario())
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
