"""What bounds an operator index request: its own pass, not the whole backlog.

A plain `recall index` observed every transcript in the corpus and then waited
for the shared scheduler to make each one current. On a host whose catalog
already owed 6,515 sources that is the whole drain -- the operator saw no
output for 19m49s and killed it. The request's own discoveries are its
work; everything that was already pending belongs to reconciliation, and the
summary says so.

The same request must also survive its client: a departure ends only the work
watch mode can resume, and one 300 MiB transcript takes longer to serve than
the client's idle timeout allows between frames.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest
from lane_harness import FIXTURES, LANE_NOW, Lane, build_lane, lane_config
from recall.core.types import Source
from recall.db.source_files import SourceCatalog
from recall.services import runtime_state
from recall.services.rpc_server import ClientConnection, RpcServer

# Worst case, one requested path is served after a full lane rotation
# (4 active : 2 recent : 1 oldest), so a pass may serve at most that many
# backlog sources on its way to its own.
LANE_ROTATION = 7


class _FakeWriter:
    """Just enough `asyncio.StreamWriter` to hang up on a handler."""

    def __init__(self) -> None:
        self._closing = False
        self.checks = 0

    def write(self, data: bytes) -> None:
        if self._closing:
            raise BrokenPipeError("peer closed")

    async def drain(self) -> None:
        if self._closing:
            raise ConnectionResetError("peer closed")

    def is_closing(self) -> bool:
        self.checks += 1
        return self._closing

    def hang_up(self) -> None:
        self._closing = True


def _observe_as_pending(server: RpcServer, path: Path) -> None:
    """Record a transcript the way the daemon's own observation does.

    This is how a source becomes part of the backlog before any request
    arrives: observed, pending, and waiting for a scheduler turn.
    """
    from recall.parsers import all_parsers
    from recall.services.coordinator import capture_path, observe_path
    from recall.services.watcher import _resolve_parser_for_path

    parser = _resolve_parser_for_path(str(path), all_parsers(server._config.sources))
    assert parser is not None
    item = observe_path(parser, capture_path(parser, path), conn=server._get_conn())
    assert not item.current, f"{path} was expected to be pending after observation"


def _install_backlog(lane: Lane, count: int) -> list[Path]:
    """Install `count` never-indexed transcripts and leave them pending."""
    installed: list[Path] = []
    for index in range(count):
        path = lane.watched.parent / f"backlog-{index}.jsonl"
        shutil.copy(FIXTURES / "claude_code" / "live_end_turn.jsonl", path)
        path.write_text(
            path.read_text(encoding="utf-8").replace("live-end-turn", f"backlog-{index}"),
            encoding="utf-8",
        )
        os.utime(path, (LANE_NOW.timestamp(), LANE_NOW.timestamp()))
        _observe_as_pending(lane.server, path)
        installed.append(path)
    return installed


def _is_current(server: RpcServer, path: Path) -> bool:
    item = SourceCatalog(server._get_conn(), clock=time.time).get(
        Source.CLAUDE_CODE.value, str(path)
    )
    return item is not None and item.current


def test_a_plain_index_serves_its_own_change_and_reports_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RECON-025: the pass waits for what it discovered, not for the backlog.

    Red before the fix: the pass waited out the backlog too, indexing 12 of
    these 22 sources before it returned -- on the live host that was 6,515
    sources and 19m49s of silence before the operator killed it.
    """
    lane = build_lane(tmp_path, monkeypatch)
    server = lane.server
    backlog = _install_backlog(lane, 20)
    lane.append(lane.watched, "the change this request found", uuid="own-pass-1")

    summary = asyncio.run(server._handle_index({"embed": False}, None))
    asyncio.run(server.stop())

    assert summary.indexed <= 1 + LANE_ROTATION, (
        f"the pass indexed {summary.indexed} of 22 sources: it drained the pre-existing "
        "backlog instead of leaving it to reconciliation"
    )
    assert summary.indexed >= 1, "the request did not serve the change it discovered itself"
    assert summary.failed == 0, "a source handed to the drain was counted as a failure"
    assert summary.total == 22
    assert summary.backlog_pending >= len(backlog) - LANE_ROTATION
    assert _is_current(server, lane.watched), "the request's own discovery was left unserved"
    assert not all(_is_current(server, path) for path in backlog)


def test_a_full_request_serves_every_source_it_observes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RECON-025: options that ride with a source keep the request waiting.

    `--full`, `--context`, `--root/--host` and the scoped filters are applied
    per path while the request is in flight; handing those sources to the drain
    would silently serve them with the daemon's own options instead.
    """
    lane = build_lane(tmp_path, monkeypatch)
    server = lane.server
    backlog = _install_backlog(lane, 4)

    summary = asyncio.run(server._handle_index({"full": True, "embed": False}, None))
    asyncio.run(server.stop())

    assert summary.indexed == 6, "a --full pass must serve every source it observes"
    assert summary.backlog_pending == 0
    assert all(_is_current(server, path) for path in backlog)


def test_another_workers_claim_is_progress_not_a_stall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RECON-025: the stall bound tracks the scheduler, not this caller's turns.

    Red before the fix: only this caller's own served turn renewed the budget,
    so a background worker preparing a 143-301 MiB rollout -- which refuses
    every turn the request takes meanwhile -- made the request fail with
    "served no source" against a scheduler that was working.
    """
    from recall.services import coordinator

    lane = build_lane(tmp_path, monkeypatch)
    server = lane.server
    monkeypatch.setattr("recall.services.rpc_server._REQUESTED_SERVICE_STALL_SECONDS", 0.2)
    lane.append(lane.watched, "slow to prepare", uuid="claim-held-1")
    _observe_as_pending(server, lane.watched)

    release = threading.Event()
    prepare_raw_sources = coordinator.prepare_raw_sources

    def held_prepare(*args: Any, **kwargs: Any) -> Any:
        assert release.wait(timeout=30), "the held preparation was never released"
        return prepare_raw_sources(*args, **kwargs)

    monkeypatch.setattr(coordinator, "prepare_raw_sources", held_prepare)

    async def scenario() -> None:
        background = asyncio.create_task(server._serve_raw_turn())
        deadline = time.monotonic() + 10
        while not server._raw_claims and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert server._raw_claims, "the background worker never claimed the pending source"

        request = asyncio.create_task(server.index_session_now(lane.watched))
        # Well past the stall bound: the claim in flight is the progress.
        await asyncio.sleep(1.0)
        assert not request.done(), (
            "the request gave up while another worker was preparing the very source "
            f"it waits for: {request.exception() if request.done() else ''}"
        )
        release.set()
        await asyncio.wait_for(background, timeout=30)
        await asyncio.wait_for(request, timeout=30)

    try:
        asyncio.run(scenario())
    finally:
        release.set()
    assert _is_current(server, lane.watched)
    asyncio.run(server.stop())


def test_a_departed_client_leaves_its_run_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RPC-018: a cancelled write records the outcome nobody could be told.

    Red before the fix: cancellation is a `BaseException`, so the handler's
    `except Exception` missed it and the run kept `last_attempted_at` newer
    than every completion with no message to explain it.
    """
    lane = build_lane(tmp_path, monkeypatch)
    server = lane.server
    running = asyncio.Event()

    async def never_finishes(**_kwargs: object) -> None:
        running.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "_reconcile_index_request", never_finishes)

    async def scenario() -> None:
        request = asyncio.create_task(server._handle_index({"embed": False}, None))
        await asyncio.wait_for(running.wait(), timeout=10)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

    asyncio.run(scenario())
    status = runtime_state.load_runtime_status_from_conn(server._get_conn())
    asyncio.run(server.stop())

    assert status.last_failure_at is not None, (
        "the cancelled run recorded an attempt and no outcome; the host reads as "
        "a run that started and never ended"
    )
    assert status.last_failure_message and "departed" in status.last_failure_message


def test_a_recreate_finishes_after_its_client_departs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RPC-018: work watch mode cannot resume outlives its client.

    Red before the fix: the departure watch cancelled every long write, so a
    `--recreate` cancelled after its reset left an emptied database with the
    attempt recorded and no run to explain it.
    """
    lane = build_lane(tmp_path, monkeypatch)
    server = lane.server
    writer = _FakeWriter()
    client = ClientConnection(cast(Any, writer))
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "recall.index",
            "params": {
                "recreate": True,
                "confirmed": True,
                "embed": False,
                "context": "template",
            },
            "id": 11,
        }
    ).encode("utf-8")

    async def scenario() -> None:
        writer.hang_up()
        await asyncio.wait_for(
            server._process_request(request, cast(Any, writer), client), timeout=120
        )

    asyncio.run(scenario())
    conn = server._get_conn()
    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    status = runtime_state.load_runtime_status_from_conn(conn)
    asyncio.run(server.stop())

    assert status.last_successful_at is not None, (
        "the rebuild was cancelled with its client; nothing else rebuilds the database"
    )
    assert status.last_index_summary is not None and status.last_index_summary.indexed == 2
    assert sessions is not None and sessions[0] == 2


def test_one_slow_source_keeps_its_client_attached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RPC-018: silence while one source is served is what times a client out.

    Red before the fix: frames were emitted only per finished file, so a
    transcript that took longer to serve than the client's idle timeout failed
    the command on a daemon that was working.
    """
    from recall.services import coordinator

    lane = build_lane(tmp_path, monkeypatch)
    monkeypatch.setattr("recall.services.rpc_server._PROGRESS_KEEPALIVE_SECONDS", 0.05)
    prepare_raw_sources = coordinator.prepare_raw_sources

    def slow_prepare(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.4)
        return prepare_raw_sources(*args, **kwargs)

    monkeypatch.setattr(coordinator, "prepare_raw_sources", slow_prepare)
    lane.append(lane.watched, "slow to serve", uuid="keepalive-1")
    frames: list[dict[str, Any]] = []

    class RecordingClient:
        async def send_progress(self, **kwargs: Any) -> None:
            frames.append(kwargs)

    summary = asyncio.run(lane.server._handle_index({"embed": False}, cast(Any, RecordingClient())))
    asyncio.run(lane.server.stop())

    assert summary.indexed == 1
    serving = [frame for frame in frames if frame["status"] == "serving"]
    assert serving, "the client heard nothing while one source was being served"
    assert serving[0]["path"] == str(lane.watched)


def test_a_deadline_less_write_polls_for_departure_at_a_coarse_cadence(tmp_path: Path) -> None:
    """REQ-RPC-018: a watch that runs for hours must not wake 20 times a second.

    Red before the fix: the read deadline's 50ms interval was used for writes
    with no deadline at all, so every `recall index` paid ~72,000 wakeups an
    hour to notice a departure that costs nothing to notice half a second late.
    """
    server = RpcServer(lane_config(tmp_path))
    writer = _FakeWriter()

    async def scenario() -> None:
        watch = asyncio.create_task(
            server._wait_for_read_request_end(None, cast(Any, writer), timeout=None)
        )
        await asyncio.sleep(1.0)
        checks = writer.checks
        assert 0 < checks <= 4, f"the departure watch checked the peer {checks} times in 1s"

        writer.hang_up()
        assert await asyncio.wait_for(watch, timeout=5) == "disconnect"

    asyncio.run(scenario())
