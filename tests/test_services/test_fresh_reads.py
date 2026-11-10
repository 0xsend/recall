"""`--fresh`: catch up pending bytes before answering (REQ-LIVE-003/010/011).

`live --fresh` is a roster catch-up of already-indexed rows; `show --fresh` is
the one-session write. Both are bounded, a session that blows the budget still
gets an answer, and the answer says so rather than looking current.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from lane_harness import (
    HANG_TIMEOUT_SECONDS,
    WATCHED_SESSION,
    Lane,
    build_lane,
    run_to_completion,
    run_with_deadlines,
)
from recall.core.source_path import SourcePath
from recall.read_budget import RESPONSIVE_READ_SECONDS
from recall.services.rpc_server import RpcServer


@pytest.fixture
def server() -> RpcServer:
    return RpcServer()


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lane:
    return build_lane(tmp_path, monkeypatch)


def _by_path(result: dict) -> dict:
    return {str(session.path): session for session in result["sessions"]}


class TestRefreshNow:
    def test_no_paths_is_not_a_request(self, server: RpcServer) -> None:
        """`--fresh` on an empty listing must not inflate the counter."""
        assert run_with_deadlines(server._refresh_now([], timeout=1.0)) is True
        assert server.live_fresh_requests == 0
        assert server.live_fresh_timeouts == 0

    def test_a_slow_session_gives_up_the_budget_and_still_returns(
        self, server: RpcServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def hanging_index(paths, **kwargs) -> None:
            await asyncio.sleep(30)

        monkeypatch.setattr(server, "_index_sessions_now", hanging_index)

        assert (
            run_with_deadlines(
                server._refresh_now([SourcePath("claude_code", "/slow.jsonl")], timeout=0.05)
            )
            is False
        )
        assert server.live_fresh_requests == 1
        assert server.live_fresh_timeouts == 1

    def test_a_path_no_parser_claims_does_not_deny_another_paths_refresh(self, lane: Lane):
        lane.append(lane.watched, "valid pending message", uuid="valid-pending")
        stray = lane.watched.parents[3] / "unclaimed.txt"
        stray.touch()

        async def scenario():
            try:
                assert await lane.server._refresh_now(
                    [
                        SourcePath.resolve("unclaimed", stray),
                        SourcePath.resolve("claude_code", lane.watched),
                    ],
                    timeout=5,
                )
                result = await lane.server._handle_show({"session_id": WATCHED_SESSION}, None)
                assert result["message_count"] == 3
                assert result["freshness"].current is True
                assert lane.server.live_fresh_timeouts == 0
            finally:
                await lane.server.stop()

        run_to_completion(scenario())

    def test_a_timed_out_index_is_left_running_rather_than_cancelled(
        self, server: RpcServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancelling mid-write would release the write lock under the executor.

        `_await_shared_conn_work` only honours cancellation once the executor
        work is done (REQ-RESIL-021), so cancelling here would block the answer
        instead of bounding it — and abandoning the lock would break the
        single-user connection rule. The task therefore runs on.
        """
        finished = asyncio.Event()
        release = asyncio.Event()

        async def slow_index(paths, **kwargs) -> None:
            await release.wait()
            finished.set()

        monkeypatch.setattr(server, "_index_sessions_now", slow_index)

        async def scenario() -> bool:
            completed = await server._refresh_now(
                [SourcePath("claude_code", "/slow.jsonl")], timeout=0.02
            )
            assert not finished.is_set()
            release.set()
            # This is a hang guard, not the refresh deadline under test.
            await asyncio.wait_for(finished.wait(), HANG_TIMEOUT_SECONDS)
            return finished.is_set() and not completed

        assert run_with_deadlines(scenario()) is True

    def test_a_non_positive_budget_is_rejected(self, server: RpcServer) -> None:
        with pytest.raises(ValueError, match="timeout"):
            run_with_deadlines(
                server._refresh_now([SourcePath("claude_code", "/a.jsonl")], timeout=0.0)
            )


class TestShowFresh:
    """`recall.show` against a real index, with the transcript moving underneath."""

    def test_show_reports_how_far_behind_the_index_is(self, lane: Lane) -> None:
        """REQ-LIVE-002: a read of a session that may still be running says its lag."""
        appended = lane.append(lane.watched, "one more thing", uuid="mid-u2")

        payload = run_with_deadlines(
            lane.server._handle_show({"session_id": WATCHED_SESSION}, None)
        )

        assert payload["freshness"].current is False
        assert payload["freshness"].lag_bytes == appended
        assert payload["message_count"] == 2

    def test_fresh_indexes_the_pending_bytes_before_answering(self, lane: Lane) -> None:
        """REQ-LIVE-003: the point of `--fresh` is to see what was just written."""
        lane.append(lane.watched, "one more thing", uuid="mid-u2")

        payload = run_to_completion(
            lane.server._handle_show({"session_id": WATCHED_SESSION, "fresh": True}, None)
        )

        assert payload["freshness"].current is True
        assert payload["freshness"].lag_bytes == 0
        assert payload["message_count"] == 3
        assert lane.server.live_fresh_requests == 1
        assert lane.server.live_fresh_timeouts == 0

    def test_an_ordinary_show_indexes_nothing(self, lane: Lane) -> None:
        """Reads stay reads: only `--fresh` may take the write lock."""
        lane.append(lane.watched, "one more thing", uuid="mid-u2")

        run_with_deadlines(lane.server._handle_show({"session_id": WATCHED_SESSION}, None))

        assert lane.server.live_fresh_requests == 0
        after = run_with_deadlines(lane.server._handle_show({"session_id": WATCHED_SESSION}, None))
        assert after["message_count"] == 2


class TestLiveFresh:
    """`recall.live --fresh` catches up already-indexed behind rows; it does not first-index."""

    @pytest.mark.parametrize("unavailable", ["deleted", "outside_catalog_scope"])
    def test_retained_unavailable_history_does_not_deny_the_rosters_refresh(
        self, lane: Lane, monkeypatch: pytest.MonkeyPatch, unavailable: str
    ) -> None:
        runtime = lane.watch_runtime([lane.watched, lane.unwatched])
        monkeypatch.setattr(lane.server, "_watch_runtime", runtime)
        lane.append(lane.unwatched, "available neighbor append", uuid="available-neighbor")
        if unavailable == "deleted":
            lane.watched.unlink()
            limitation = "source_unreadable"
        else:
            # A completed inventory owns eligibility independently of readable bytes.
            lane.server._get_conn().execute(
                "UPDATE source_files SET missing = TRUE WHERE source_path = ?",
                [str(lane.watched)],
            )
            lane.append(lane.watched, "detached append", uuid="detached-append")
            limitation = "catalog_source_missing"

        async def scenario():
            try:
                rows = _by_path(await lane.server._handle_live({"fresh": True, "all": True}, None))
                assert not rows[str(lane.watched)].freshness.current
                assert limitation in rows[str(lane.watched)].freshness.limitations
                assert rows[str(lane.unwatched)].freshness.current
                retained = await lane.server._handle_show({"session_id": WATCHED_SESSION}, None)
                assert retained["message_count"] == 2
                neighbor = await lane.server._handle_show({"session_id": "live-end-turn"}, None)
                assert neighbor["message_count"] == 5
                assert "available neighbor append" in str(neighbor["messages"])
                assert lane.server.live_fresh_requests == 1
                assert lane.server.live_fresh_timeouts == 0
                assert not lane.server._fresh_jobs
            finally:
                await lane.server.stop()

        run_to_completion(scenario())

    def test_fresh_catches_up_every_already_indexed_row_in_the_answer(
        self, lane: Lane, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-LIVE-003: the listing answers from bytes that were on disk when it was asked.

        Both already-indexed rows, not just the watched one. Scoping the refresh
        to the live set (as this did until the U18 bug bash) skips a session
        that is being written to right now but has not been promoted yet -- the
        discovery loop runs every 30 s -- which is precisely the already-indexed
        row a `--fresh` caller is chasing.
        """
        monkeypatch.setattr(lane.server, "_watch_runtime", lane.watch_runtime([lane.watched]))
        lane.append(lane.watched, "one more thing", uuid="mid-u2")
        lane.append(lane.unwatched, "and another", uuid="end-u3")

        rows = _by_path(
            run_to_completion(lane.server._handle_live({"fresh": True, "all": True}, None))
        )

        assert rows[str(lane.watched)].freshness.current is True
        assert rows[str(lane.unwatched)].freshness.current is True
        assert lane.server.live_fresh_requests == 1
        assert lane.server.live_fresh_timeouts == 0

    def test_fresh_spends_nothing_when_every_row_is_already_current(
        self, lane: Lane, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing is behind, so there is nothing to index and no write lock to take."""
        monkeypatch.setattr(lane.server, "_watch_runtime", lane.watch_runtime([lane.watched]))

        rows = _by_path(
            run_to_completion(lane.server._handle_live({"fresh": True, "all": True}, None))
        )

        assert all(row.freshness.current for row in rows.values())
        assert lane.server.live_fresh_requests == 0

    def test_an_unwatched_daemon_has_nothing_to_refresh(self, lane: Lane) -> None:
        """No live set means no watched path, so `--fresh` spends nothing."""
        result = run_to_completion(lane.server._handle_live({"fresh": True}, None))

        assert result["watching"] is False
        assert lane.server.live_fresh_requests == 0

    def test_an_ordinary_live_read_leaves_the_lag_alone(
        self, lane: Lane, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reads stay reads: only `--fresh` may take the write lock."""
        monkeypatch.setattr(lane.server, "_watch_runtime", lane.watch_runtime([lane.watched]))
        appended = lane.append(lane.watched, "one more thing", uuid="mid-u2")

        rows = _by_path(run_to_completion(lane.server._handle_live({"all": True}, None)))

        assert rows[str(lane.watched)].freshness.lag_bytes == appended
        assert lane.server.live_fresh_requests == 0

    def test_fresh_does_not_first_index_a_never_indexed_row(
        self, lane: Lane, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First index stays on the coordinator; the roster still answers."""
        extra = lane.watched.parent / "brand-new.jsonl"
        extra.write_text('{"type":"user","sessionId":"brand-new"}\n', encoding="utf-8")
        monkeypatch.setattr(
            lane.server, "_watch_runtime", lane.watch_runtime([lane.watched, extra])
        )
        indexed: list[str] = []
        real_index = lane.server.index_session_now

        async def spy(path: Path) -> None:
            indexed.append(str(path))
            await real_index(path)

        monkeypatch.setattr(lane.server, "index_session_now", spy)

        rows = _by_path(run_to_completion(lane.server._handle_live({"fresh": True}, None)))

        assert str(extra) in rows
        assert rows[str(extra)].freshness.current is False
        assert "not_yet_indexed" in rows[str(extra)].freshness.limitations
        assert str(extra) not in indexed
        assert lane.server.live_fresh_requests == 0

    def test_show_fresh_keeps_the_one_session_write_budget(self, lane: Lane, monkeypatch) -> None:
        """`show --fresh` is the one already-known session write, not the 2s roster bound."""
        captured: list[float] = []

        async def spy(paths, *, timeout: float) -> bool:
            captured.append(timeout)
            return True

        monkeypatch.setattr(lane.server, "_refresh_now", spy)

        run_with_deadlines(
            lane.server._handle_show({"session_id": WATCHED_SESSION, "fresh": True}, None)
        )

        assert captured == [lane.server._config.live.fresh_timeout]
        assert captured[0] > RESPONSIVE_READ_SECONDS


class TestDaemonStatusCounters:
    def test_the_fresh_counters_start_at_zero(self, server: RpcServer) -> None:
        assert server.live_fresh_requests == 0
        assert server.live_fresh_timeouts == 0

    def test_daemon_status_reports_what_fresh_reads_have_cost(self, lane: Lane) -> None:
        """REQ-LIVE-011: an operator tuning `live.fresh_timeout` needs the spend, not a guess."""
        lane.append(lane.watched, "one more thing", uuid="mid-u2")
        run_to_completion(
            lane.server._handle_show({"session_id": WATCHED_SESSION, "fresh": True}, None)
        )

        status = run_to_completion(lane.server._handle_daemon_status({}, None))

        assert status["live_fresh_requests"] == 1
        assert status["live_fresh_timeouts"] == 0


class TestAbandonedRefreshEscalation:
    """A `--fresh` index that outlives its budget still has to report a dead DB.

    The shield is what lets the answer return on deadline, but it also leaves a
    task nothing awaits. A fatal DuckDB invalidation raised there is
    process-terminal (REQ-RESIL-011): dropping it means the daemon serves errors
    forever while `daemon status` reads green. Found by the U12 concurrency
    review.
    """

    def test_a_fatal_db_error_after_the_budget_still_stops_the_daemon(
        self, server: RpcServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import duckdb

        release = asyncio.Event()

        async def fatal_after_the_budget(path, config, request):
            await release.wait()
            raise duckdb.FatalException("database has been invalidated")

        monkeypatch.setattr(server, "_observe_requested_sources", fatal_after_the_budget)

        async def scenario() -> bool:
            completed = await server._refresh_now(
                [SourcePath("claude_code", "/slow.jsonl")], timeout=0.01
            )
            assert not server._shutdown_event.is_set()
            release.set()
            # Wait for the reported fatal error; the limit only catches a hang.
            await asyncio.wait_for(server._shutdown_event.wait(), HANG_TIMEOUT_SECONDS)
            return completed

        assert run_with_deadlines(scenario()) is False
        assert server._shutdown_event.is_set() is True

    def test_an_ordinary_failure_after_the_budget_does_not_stop_the_daemon(
        self, server: RpcServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One unparseable transcript is not a reason to take the daemon down."""
        release = asyncio.Event()

        async def broken_after_the_budget(path, config, request):
            await release.wait()
            raise RuntimeError("one transcript went sideways")

        monkeypatch.setattr(server, "_observe_requested_sources", broken_after_the_budget)

        async def scenario() -> None:
            assert not await server._refresh_now(
                [SourcePath("claude_code", "/slow.jsonl")], timeout=0.01
            )
            batches = tuple(server._fresh_batches)
            assert batches
            release.set()
            # Observe completion before asserting that shutdown was not requested.
            await asyncio.wait_for(asyncio.gather(*batches), HANG_TIMEOUT_SECONDS)

        run_with_deadlines(scenario())

        assert server._shutdown_event.is_set() is False

    def test_a_non_source_refresh_error_still_propagates(
        self, server: RpcServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A torn last line or a bad config must not read as 'no parser'."""

        async def bad_config(paths, **kwargs) -> None:
            raise ValueError("invalid context mode: nonsense")

        monkeypatch.setattr(server, "_index_sessions_now", bad_config)

        with pytest.raises(ValueError, match="invalid context mode"):
            run_to_completion(
                server._refresh_now([SourcePath("claude_code", "/a.jsonl")], timeout=5.0)
            )


class TestStalledSourceFreshness:
    """A source the index refuses past a malformed record is behind, by those bytes."""

    def _write_stalled(self, lane: Lane) -> tuple[Path, int]:
        """A new transcript: complete records, a malformed one, then two valid appends.

        Returns the path and the byte offset where the malformed record starts.
        """
        path = lane.watched.parent / "stalled.jsonl"
        records = lane.watched.read_text(encoding="utf-8").replace(WATCHED_SESSION, "stalled")
        malformed = '{"type":"user","message":{"role":"user","content":"half writ\n'
        path.write_text(records + malformed, encoding="utf-8")
        lane.append(path, "written after the bad line", uuid="after-1")
        lane.append(path, "and once more", uuid="after-2")
        run_with_deadlines(lane.server.index_session_now(path))
        return path, len(records.encode("utf-8"))

    def test_show_counts_every_byte_past_the_last_acknowledged_record(self, lane: Lane) -> None:
        path, acknowledged = self._write_stalled(lane)

        payload = run_with_deadlines(lane.server._handle_show({"session_id": "stalled"}, None))

        freshness = payload["freshness"]
        assert freshness.current is False
        assert freshness.indexed_size == acknowledged
        assert freshness.lag_bytes == path.stat().st_size - acknowledged
        assert freshness.lag_bytes > 0
        # How long those bytes have waited is unknown, which is not "no wait".
        assert freshness.lag_seconds is None

    def test_the_live_roster_reports_the_same_lag(
        self, lane: Lane, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path, acknowledged = self._write_stalled(lane)
        monkeypatch.setattr(lane.server, "_watch_runtime", lane.watch_runtime([path]))

        rows = _by_path(run_to_completion(lane.server._handle_live({"all": True}, None)))

        assert rows[str(path)].freshness.current is False
        assert rows[str(path)].freshness.lag_bytes == path.stat().st_size - acknowledged


def test_moved_source_finishes_a_timed_out_refresh_without_a_traceback(
    lane: Lane, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    entered = asyncio.Event()
    release = asyncio.Event()
    observe = lane.server._observe_requested_sources

    async def delayed_observation(path, config, request):
        entered.set()
        await release.wait()
        return await observe(path, config, request)

    monkeypatch.setattr(lane.server, "_observe_requested_sources", delayed_observation)

    async def scenario() -> None:
        refresh = asyncio.create_task(
            lane.server._refresh_now(
                [SourcePath.resolve("claude_code", lane.watched)], timeout=0.05
            )
        )
        await asyncio.wait_for(entered.wait(), HANG_TIMEOUT_SECONDS)
        lane.watched.rename(lane.watched.with_name("moved.jsonl"))
        assert await refresh is False
        jobs = tuple(lane.server._fresh_jobs.values())
        assert jobs
        release.set()
        with pytest.raises(FileNotFoundError) as caught:
            await asyncio.wait_for(asyncio.gather(*jobs), HANG_TIMEOUT_SECONDS)
        assert caught.value.filename == str(lane.watched)
        await asyncio.sleep(0)  # Deliver the outer refresh task's completion callback.

    try:
        with caplog.at_level(logging.INFO, logger="recall.rpc_server"):
            run_with_deadlines(scenario())
        disappeared = [r for r in caplog.records if "source disappeared" in r.getMessage()]
        assert len(disappeared) == 1
        assert str(lane.watched) in disappeared[0].getMessage()
        assert disappeared[0].levelno == logging.INFO
        assert not any(r.exc_info or r.levelno >= logging.WARNING for r in caplog.records)
        assert not lane.server._shutdown_event.is_set()
    finally:
        run_with_deadlines(lane.server.stop())


def test_refresh_preserves_failures_for_missing_dependencies(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = FileNotFoundError(2, "missing dependency", "/models/dependency")

    async def unavailable_dependency(*_args):
        raise error

    monkeypatch.setattr(lane.server, "_observe_requested_sources", unavailable_dependency)
    try:
        with pytest.raises(FileNotFoundError) as caught:
            run_to_completion(
                lane.server._refresh_now(
                    [SourcePath.resolve("claude_code", lane.watched)], timeout=1
                )
            )
        assert caught.value is error
    finally:
        run_with_deadlines(lane.server.stop())


@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize(
    "unavailable", ["departed", "parent_departed", "permission_denied", "not_a_directory"]
)
def test_show_retains_unreadable_source_history(
    lane: Lane, monkeypatch: pytest.MonkeyPatch, fresh: bool, unavailable: str
) -> None:
    import shutil

    original_stat = Path.stat
    if unavailable == "departed":
        moved = lane.watched.parent / "moved" / lane.watched.name
        moved.parent.mkdir()
        shutil.copyfile(lane.watched, moved)
        lane.watched.unlink()
    elif unavailable in ("parent_departed", "not_a_directory"):
        shutil.rmtree(lane.watched.parent)
        if unavailable == "not_a_directory":
            lane.watched.parent.touch()
    else:

        def denied(path: Path, *args, **kwargs):
            if path == lane.watched:
                raise PermissionError(13, "Permission denied", str(path))
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", denied)

    async def scenario():
        try:
            result = await lane.server._handle_show(
                {"session_id": WATCHED_SESSION, "fresh": fresh, "tools": True}, None
            )
            assert result["message_count"] == 2
            assert "run the test suite" in str(result["messages"])
            assert result["freshness"].current is False
            assert "source_unreadable" in result["freshness"].limitations
        finally:
            await lane.server.stop()

    run_to_completion(scenario())


def test_show_fresh_preserves_an_unrelated_refresh_error(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = OSError(5, "database storage unavailable", "/scratch/database")

    async def broken_observation(*args):
        raise error

    monkeypatch.setattr(lane.server, "_observe_requested_sources", broken_observation)

    async def scenario():
        try:
            with pytest.raises(OSError) as caught:
                await lane.server._handle_show({"session_id": WATCHED_SESSION, "fresh": True}, None)
            assert caught.value is error
        finally:
            await lane.server.stop()

    run_to_completion(scenario())


def test_live_fresh_retains_a_source_that_disappears_during_catch_up(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        lane.server, "_watch_runtime", lane.watch_runtime([lane.watched, lane.unwatched])
    )
    lane.append(lane.watched, "departing append", uuid="departing")
    lane.append(lane.unwatched, "neighbor append", uuid="neighbor")
    observe = lane.server._observe_requested_sources

    async def depart_before_capture(paths, config, request):
        lane.watched.unlink()
        return await observe(paths, config, request)

    monkeypatch.setattr(lane.server, "_observe_requested_sources", depart_before_capture)

    async def scenario():
        try:
            rows = _by_path(await lane.server._handle_live({"fresh": True, "all": True}, None))
            assert rows[str(lane.watched)].freshness.current is False
            assert "source_unreadable" in rows[str(lane.watched)].freshness.limitations
            assert rows[str(lane.unwatched)].freshness.current is True
            neighbor = await lane.server._handle_show({"session_id": "live-end-turn"}, None)
            assert "neighbor append" in str(neighbor["messages"])
        finally:
            await lane.server.stop()

    run_to_completion(scenario())


def test_unreadable_source_does_not_hide_a_neighbors_refresh_error(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.core.source_path import SourcePath

    lane.watched.unlink()
    watched = SourcePath.resolve("claude_code", lane.watched)
    neighbor = SourcePath.resolve("claude_code", lane.unwatched)
    error = OSError(5, "read failure", str(lane.unwatched))
    observe = lane.server._observe_requested_sources

    async def broken_neighbor(paths, config, request):
        results = await observe(paths, config, request)
        return {path: error if path == neighbor else result for path, result in results.items()}

    monkeypatch.setattr(lane.server, "_observe_requested_sources", broken_neighbor)

    async def scenario():
        try:
            with pytest.raises(OSError) as caught:
                await lane.server._refresh_now([watched, neighbor], timeout=5)
            assert caught.value is error
            with pytest.raises(FileNotFoundError):
                await lane.server.index_session_now(lane.watched, source="claude_code")
        finally:
            await lane.server.stop()

    run_to_completion(scenario())
