"""Captured-generation fresh waits, through the real catalog and scheduler."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from lane_harness import WATCHED_SESSION, Lane, build_lane
from recall.db.source_files import SourceCatalog
from recall.parsers import all_parsers
from recall.services.coordinator import capture_path, observe_path
from recall.services.watcher import _resolve_parser_for_path


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lane:
    return build_lane(tmp_path, monkeypatch)


def test_req_live_013_observes_the_page_before_servicing_it(lane: Lane, monkeypatch) -> None:
    lane.append(lane.watched, "first pending message", uuid="batch-first")
    lane.append(lane.unwatched, "second pending message", uuid="batch-second")
    serve = lane.server._serve_raw_turn

    async def serve_observed_page():
        def pending(conn):
            rows = conn.execute(
                "SELECT source_path FROM source_files "
                "WHERE desired_generation > committed_generation"
            ).fetchall()
            return {row[0] for row in rows}

        observed = await lane.server._run_readonly(pending)
        assert {str(lane.watched), str(lane.unwatched)} <= observed
        monkeypatch.setattr(lane.server, "_serve_raw_turn", serve)
        return await serve()

    monkeypatch.setattr(lane.server, "_serve_raw_turn", serve_observed_page)

    async def scenario():
        try:
            assert await lane.server._refresh_now(
                [str(lane.watched), str(lane.unwatched)], timeout=5
            )
            rows = await lane.server._handle_live({"all": True}, None)
            assert len(rows["sessions"]) == 2
            assert all(row.freshness.current for row in rows["sessions"])
        finally:
            await lane.server.stop()

    asyncio.run(scenario())


def test_req_live_013_later_append_does_not_extend_the_captured_wait(lane: Lane, monkeypatch):
    lane.append(lane.watched, "captured message", uuid="batch-captured")
    serve = lane.server._serve_raw_turn
    parser = _resolve_parser_for_path(str(lane.watched), all_parsers(lane.server._config.sources))
    assert parser is not None
    turns = 0

    async def append_after_commit():
        nonlocal turns
        turns += 1
        assert turns == 1, "fresh request chased a generation appended after its capture"
        worked = await serve()
        assert worked
        lane.append(lane.watched, "later message", uuid="batch-later")
        captured = capture_path(parser, lane.watched)
        await lane.server._writer_call(
            lambda: observe_path(parser, captured, conn=lane.server._get_conn()),
            "later append observation",
        )
        return worked

    monkeypatch.setattr(lane.server, "_serve_raw_turn", append_after_commit)

    async def scenario():
        try:
            assert await lane.server._refresh_now([str(lane.watched)], timeout=5)
            result = await lane.server._handle_show({"session_id": WATCHED_SESSION}, None)
            assert result["message_count"] == 3
            assert result["freshness"].current is False
            assert result["freshness"].lag_bytes > 0
            # The later generation stays eligible for ordinary reconciliation.
            monkeypatch.setattr(lane.server, "_serve_raw_turn", serve)
            await lane.server.index_session_now(lane.watched)
            result = await lane.server._handle_show({"session_id": WATCHED_SESSION}, None)
            assert result["message_count"] == 4
            assert result["freshness"].current is True
            items = await lane.server._run_readonly(
                lambda conn: SourceCatalog(conn, clock=lambda: 0).get(
                    "claude_code", str(lane.watched)
                )
            )
            assert items is not None and items.current
        finally:
            await lane.server.stop()

    asyncio.run(scenario())


def test_req_live_013_capacity_is_atomic_and_cancelled_waiters_share_work(lane: Lane, monkeypatch):
    """An overlapping page joins; a disjoint path cannot consume a 257th slot."""
    from recall.core.rpc_types import APP_LOCKED, RpcError

    entered = asyncio.Event()
    release = asyncio.Event()
    observe = lane.server._observe_requested_sources
    paths = [str(lane.watched.parents[3] / f"unclaimed-{i}.txt") for i in range(256)]

    async def blocked_observation(*args):
        entered.set()
        await release.wait()
        return await observe(*args)

    monkeypatch.setattr(lane.server, "_observe_requested_sources", blocked_observation)

    async def scenario():
        first = asyncio.create_task(lane.server._refresh_now(paths, timeout=5))
        await asyncio.wait_for(entered.wait(), 1)
        joined = asyncio.create_task(lane.server._refresh_now(paths[::-1], timeout=5))
        try:
            with pytest.raises(RpcError) as caught:
                await lane.server._refresh_now([*paths[:255], str(lane.watched)], timeout=5)
            assert caught.value.code == APP_LOCKED
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            release.set()
            assert await asyncio.wait_for(joined, 5)
            assert not lane.server._fresh_jobs
            assert not lane.server._fresh_batches
            assert not lane.server._raw_requests
            # Refused admission did not register or index the neighboring path.
            result = await lane.server._handle_show({"session_id": WATCHED_SESSION}, None)
            assert result["message_count"] == 2
        finally:
            release.set()
            await asyncio.gather(first, joined, return_exceptions=True)
            await lane.server.stop()

    asyncio.run(scenario())
