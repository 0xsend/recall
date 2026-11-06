"""Public reads across controlled, real database replacement boundaries."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from lane_harness import Lane, build_lane, run_to_completion
from recall import read_budget
from recall.core.rpc_types import RpcError


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lane:
    return build_lane(tmp_path, monkeypatch)


@pytest.mark.parametrize("phase", ["queued", "delivery"])
@pytest.mark.parametrize("method", ["daemon_status", "live", "live_fresh", "search"])
def test_read_across_recreate(lane: Lane, monkeypatch: pytest.MonkeyPatch, method, phase):
    server = lane.server
    entered = asyncio.Event()
    release = asyncio.Event()
    run_readonly = server._run_readonly
    intercepted = False

    async def controlled_read(operation):
        nonlocal intercepted
        if intercepted:
            return await run_readonly(operation)
        intercepted = True
        result = await run_readonly(operation) if phase == "delivery" else None
        entered.set()
        await release.wait()
        if phase == "queued":
            return await run_readonly(operation)
        return result

    monkeypatch.setattr(server, "_run_readonly", controlled_read)

    async def scenario():
        name = "live" if method == "live_fresh" else method
        params = {"all": True, "fresh": method == "live_fresh"}
        if method == "search":
            params = {"query": "tool", "mode": "keyword"}
        task = asyncio.ensure_future(server._methods[f"recall.{name}"](params, None))
        try:
            await entered.wait()
            await server._writer_call(lambda: server._reset_database(server._config), "recreate")
            release.set()
            result = (await asyncio.gather(task, return_exceptions=True))[0]
            assert not isinstance(result, BaseException), repr(result)
            if method == "daemon_status":
                assert result["embed_pending_eligible"] == 0
                assert result["embed_pending_held"] == 0
                assert result["reconciliation"]["pending"] == 0
            elif method.startswith("live"):
                assert result["sessions"] == ()
                assert result["coverage"] is not None
            else:
                assert isinstance(result, list)
                assert result == []
            assert not server._read_executions
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await server.stop()

    run_to_completion(scenario())


def test_fresh_live_restarts_when_recreate_follows_its_probe(lane: Lane, monkeypatch):
    server = lane.server
    lane.append(lane.watched, "pending append", uuid="pending-append")

    async def replace_before_catch_up(*args, **kwargs):
        await server._writer_call(lambda: server._reset_database(server._config), "recreate")

    monkeypatch.setattr(server, "_refresh_now", replace_before_catch_up)

    async def scenario():
        try:
            result = await server._methods["recall.live"]({"all": True, "fresh": True}, None)
            assert result["sessions"] == ()
            assert not server._fresh_jobs
        finally:
            await server.stop()

    run_to_completion(scenario())


def test_repeated_replacement_spends_the_original_read_budget(lane: Lane, monkeypatch):
    server = lane.server
    run_readonly = server._run_readonly

    async def scenario():
        loop = asyncio.get_running_loop()
        started = loop.time()
        clock = started

        async def replace_after_read(operation):
            nonlocal clock
            result = await run_readonly(operation)
            await server._writer_call(lambda: server._reset_database(server._config), "recreate")
            clock += 0.06
            return result

        try:
            with monkeypatch.context() as timers:
                timers.setattr(loop, "time", lambda: clock)
                timers.setattr(read_budget, "time", SimpleNamespace(monotonic=lambda: clock))
                timers.setattr(server, "_run_readonly", replace_after_read)
                task = asyncio.ensure_future(
                    server._methods["recall.daemon_status"]({"remaining_read_ms": 100}, None)
                )
                result = (await asyncio.gather(task, return_exceptions=True))[0]
                assert isinstance(result, RpcError)
                assert result.message == "responsive read timed out"
                assert 0.1 <= clock - started < 0.2
            assert not server._read_executions
        finally:
            await server.stop()

    run_to_completion(scenario())
