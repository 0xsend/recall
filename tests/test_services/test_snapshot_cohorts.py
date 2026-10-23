"""Fresh snapshot membership, coherence and cursor ownership on real DuckDB."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from lane_harness import WATCHED_SESSION, Lane, build_lane
from recall.core.rpc_types import RpcError


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lane:
    return build_lane(tmp_path, monkeypatch)


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


async def hold_readers(server):
    for _ in range(8):
        await server._read_slots.acquire()


def release_readers(server):
    for _ in range(8):
        server._read_slots.release()


def test_queued_members_share_but_mid_query_arrival_gets_a_later_snapshot(lane):
    server = lane.server
    conn = server._get_conn()
    conn.execute("CREATE TABLE cohort_value(value INTEGER)")
    conn.execute("INSERT INTO cohort_value VALUES (0)")
    started = threading.Event()
    release = threading.Event()
    executions = []

    def read(cursor):
        value = cursor.execute("SELECT value FROM cohort_value").fetchone()[0]
        executions.append(value)
        if value == 0:
            started.set()
            assert release.wait(5)
        return {"rows": [value]}

    async def scenario():
        await hold_readers(server)
        first = asyncio.create_task(server._cohort_snapshot(("live", "final"), 0, read))
        second = asyncio.create_task(server._cohort_snapshot(("live", "final"), 0, read))
        try:
            await until(lambda: server._snapshot_cohorts._waiters == 2)
            release_readers(server)
            assert await asyncio.to_thread(started.wait, 5)
            conn.execute("UPDATE cohort_value SET value = 1")
            late = await server._cohort_snapshot(("live", "final"), 0, read)
            assert late == {"rows": [1]}
            release.set()
            a, b = await asyncio.gather(first, second)
            assert a == b == {"rows": [0]}
            a["rows"].append(99)
            assert b == {"rows": [0]}
            # Completed results are never reused, even without a mutation.
            assert await server._cohort_snapshot(("live", "final"), 0, read) == late
            assert executions == [0, 1, 1]
        finally:
            release.set()
            await asyncio.gather(first, second, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())


def test_one_departing_waiter_does_not_interrupt_another_and_all_departures_drain(lane):
    server = lane.server
    entered = threading.Event()
    release = threading.Event()

    def read(conn):
        result = conn.execute("SELECT 42").fetchone()[0]
        entered.set()
        assert release.wait(5)
        return result

    async def scenario():
        await hold_readers(server)
        a = asyncio.create_task(server._cohort_snapshot(("live", "final"), 0, read))
        b = asyncio.create_task(server._cohort_snapshot(("live", "final"), 0, read))
        try:
            await until(lambda: server._snapshot_cohorts._waiters == 2)
            release_readers(server)
            assert await asyncio.to_thread(entered.wait, 5)
            a.cancel()
            with pytest.raises(asyncio.CancelledError):
                await a
            assert len(server._read_executions) == 1
            release.set()
            assert await b == 42
            await until(lambda: not server._read_executions)
            entered.clear()
            release.clear()
            c = asyncio.create_task(server._cohort_snapshot(("live", "final"), 0, read))
            assert await asyncio.to_thread(entered.wait, 5)
            c.cancel()
            with pytest.raises(asyncio.CancelledError):
                await c
            assert len(server._read_executions) == 1
            release.set()
            await until(lambda: not server._snapshot_cohorts._owned)
            assert not server._read_executions
            assert server._read_slots._value == 8
        finally:
            release.set()
            await asyncio.gather(a, b, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())


def test_all_queued_departures_remove_work_and_replacement_invalidates_results(lane):
    server = lane.server
    calls = []

    def read(conn):
        calls.append(True)
        return conn.execute("SELECT 17").fetchone()[0]

    async def scenario():
        await hold_readers(server)
        task = asyncio.create_task(server._cohort_snapshot(("live", "probe"), 0, read))
        try:
            await until(lambda: server._snapshot_cohorts._waiters == 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await until(lambda: not server._snapshot_cohorts._owned)
            assert calls == []
            # A queued request belongs to the captured database, not its replacement.
            old = asyncio.create_task(server._cohort_snapshot(("live", "probe"), 0, read))
            await until(lambda: server._snapshot_cohorts._waiters == 1)
            server._database_epoch += 1
            release_readers(server)
            with pytest.raises(RpcError, match="database replaced"):
                await old
            assert calls == []
            assert await server._cohort_snapshot(("live", "probe"), 1, read) == 17
        finally:
            await asyncio.gather(task, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", [1, 50, 256])
def test_live_pages_filters_and_phases_do_not_share_distinct_queries(lane, limit):
    server = lane.server

    async def scenario():
        try:
            page, empty, fresh = await asyncio.gather(
                server._handle_live({"all": True, "limit": limit}, None),
                server._handle_live({"project": "/absent", "limit": limit}, None),
                server._handle_live({"all": True, "limit": limit, "fresh": True}, None),
            )
            assert len(page["sessions"]) == min(limit, 2)
            assert len(fresh["sessions"]) == min(limit, 2)
            assert empty["sessions"] == ()
            assert empty["coverage"] is not None
            assert set(page) == {
                "schema_version",
                "watching",
                "sessions",
                "next_cursor",
                "coverage",
            }
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_live_budget_covers_probe_wait_and_final_snapshot_with_owned_raw_work(lane, monkeypatch):
    server = lane.server
    lane.append(lane.watched, "budgeted append", uuid="cohort-budget")
    entered = asyncio.Event()
    release = asyncio.Event()
    serve = server._serve_raw_turn

    async def blocked_service():
        entered.set()
        await release.wait()
        return await serve()

    monkeypatch.setattr(server, "_serve_raw_turn", blocked_service)

    async def scenario():
        try:
            # The server must reserve final-read time from the entire duration,
            # rather than starting another 2 s wait after its membership probe.
            result = await asyncio.wait_for(
                server._methods["recall.live"](
                    {"all": True, "fresh": True, "remaining_read_ms": 1100}, None
                ),
                1.4,
            )
            assert entered.is_set()
            row = next(
                row for row in result["sessions"] if row.source_session_id == WATCHED_SESSION
            )
            assert not row.freshness.current
            assert row.freshness.lag_bytes > 0
            assert server._fresh_jobs
            release.set()
            await server._index_sessions_now([str(lane.watched)])
            final = await server._handle_show({"session_id": WATCHED_SESSION}, None)
            assert final["message_count"] == 3
            assert final["freshness"].current
        finally:
            release.set()
            await server.stop()

    asyncio.run(scenario())


def test_status_timeout_does_not_cancel_another_waiters_snapshot(lane):
    server = lane.server

    async def scenario():
        await server._status_count_lock.acquire()
        short = asyncio.create_task(
            server._methods["recall.daemon_status"]({"remaining_read_ms": 20}, None)
        )
        long = asyncio.create_task(server._methods["recall.daemon_status"]({}, None))
        try:
            with pytest.raises(RpcError, match="timed out"):
                await asyncio.wait_for(short, 0.5)
            server._status_count_lock.release()
            result = await long
            assert result["daemon_pid"] is not None
            assert result["embed_pending_eligible"] is not None
            assert result["embed_pending_held"] is not None
            assert result["embed_pending_measured_at"] is not None
        finally:
            if server._status_count_lock.locked():
                server._status_count_lock.release()
            await asyncio.gather(short, long, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())


def test_queued_status_reads_share_fresh_counts_and_one_coverage_snapshot(lane, monkeypatch):
    from recall.services import daemon

    server = lane.server
    count = daemon.count_pending_split
    statements = []

    def counted(*args, **kwargs):
        result = count(*args, **kwargs)
        statements.append(result)
        return result

    monkeypatch.setattr(daemon, "count_pending_split", counted)

    async def scenario():
        await server._status_count_lock.acquire()
        tasks = [asyncio.create_task(server._handle_daemon_status({}, None)) for _ in range(20)]
        try:
            await until(lambda: server._snapshot_cohorts._waiters == 20)
            # A mutation before admission to the transaction must be visible
            # to all members, for both the count and coverage portions.
            conn = server._get_conn()
            conn.execute("DELETE FROM message_embeddings")
            conn.execute("DELETE FROM tool_call_embeddings")
            conn.execute("UPDATE session_state SET file_mtime = 0")
            expected = conn.execute(
                "SELECT (SELECT COUNT(*) FROM messages) + (SELECT COUNT(*) FROM tool_calls)"
            ).fetchone()[0]
            assert expected > 0
            server._status_count_lock.release()
            results = await asyncio.gather(*tasks)
            assert len(statements) == 1
            assert {r["embed_pending_eligible"] for r in results} == {expected}
            assert all(r["embed_pending_held"] == 0 for r in results)
            assert len({r["embed_pending_measured_at"] for r in results}) == 1
            assert all(r["reconciliation"] == results[0]["reconciliation"] for r in results)
            results[0]["reconciliation"]["pending"] = 999
            assert results[1]["reconciliation"]["pending"] != 999
            # The next call must execute another fresh count, even if quiet.
            await server._handle_daemon_status({}, None)
            assert len(statements) == 2
        finally:
            if server._status_count_lock.locked():
                server._status_count_lock.release()
            await asyncio.gather(*tasks, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())


def test_cohort_waiters_are_bounded_and_queued_shutdown_releases_capacity(lane):
    server = lane.server

    async def scenario():
        await hold_readers(server)
        tasks = [
            asyncio.create_task(server._cohort_snapshot(("live", "final", i), 0, lambda c: 1))
            for i in range(64)
        ]
        try:
            await until(lambda: server._snapshot_cohorts._waiters == 64)
            with pytest.raises(RpcError, match="capacity"):
                await server._cohort_snapshot(("live", "final", "overflow"), 0, lambda c: 2)
            await server.stop()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert all(isinstance(r, asyncio.CancelledError) for r in results)
            assert not server._snapshot_cohorts._owned
            assert server._snapshot_cohorts._waiters == 0
            release_readers(server)
            assert server._read_slots._value == 8
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())


def test_executing_cohort_rejects_an_epoch_change_before_delivery(lane):
    server = lane.server
    entered = threading.Event()
    release = threading.Event()

    def read(conn):
        value = conn.execute("SELECT 7").fetchone()[0]
        entered.set()
        assert release.wait(5)
        return value

    async def scenario():
        task = asyncio.create_task(server._cohort_snapshot(("live", "final"), 0, read))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            server._database_epoch += 1
            release.set()
            with pytest.raises(RpcError, match="database replaced"):
                await task
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("request_id", [1, 'quoted"雪', None])
def test_live_cohort_wire_reply_preserves_envelope_and_request_identity(lane, request_id):
    from typing import cast

    from recall.core.rpc_types import serialize_rpc_value
    from recall.services.live import LiveCoverage, LivePage
    from recall.services.rpc_server import _LiveResponse

    class Writer:
        def __init__(self):
            self.data = bytearray()

        def write(self, data):
            self.data.extend(data)

        async def drain(self):
            pass

    expected = {
        "schema_version": 2,
        "watching": True,
        "sessions": [],
        "next_cursor": None,
        "coverage": {
            "watched_paths": 0,
            "indexed_paths": 0,
            "filtered_unknown_paths": [],
            "observed_paths": 0,
            "unindexed_paths": 0,
            "unknown_count": 0,
            "catalog_scan_complete": False,
            "complete": False,
        },
    }
    result = _LiveResponse(LivePage((), None), LiveCoverage(0, 0, ()), True)
    assert serialize_rpc_value(result) == expected
    writer = Writer()

    async def scenario():
        try:
            await lane.server._send_result(cast(asyncio.StreamWriter, writer), request_id, result)
            import json

            reply = json.loads(writer.data)
            assert reply == {
                "jsonrpc": "2.0",
                "id": request_id,
                "_config_fp": lane.server._config_fp,
                "result": expected,
            }
            assert writer.data.endswith(b"\n")
        finally:
            await lane.server.stop()

    asyncio.run(scenario())
