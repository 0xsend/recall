"""Public search wait frames and deadline diagnostics (REQ-RPC-022/023)."""

from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any, cast

import duckdb
import pytest
from recall.services.rpc_server import ClientConnection, RpcServer, _PeerStreamReader

rpc = importlib.import_module("recall.services.rpc_server")


class Writer:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.notified = asyncio.Event()
        self.slow = False

    def is_closing(self) -> bool:
        return False

    def write(self, data: bytes) -> None:
        self.frames.append(json.loads(data))
        self.notified.set()

    async def drain(self) -> None:
        if self.slow and self.frames[-1].get("method") == "search_progress":
            await asyncio.Event().wait()


def dispatch(server: RpcServer, writer: Writer, params: dict, request_id: object = "one"):
    stream = cast(asyncio.StreamWriter, writer)
    return server._process_request(
        json.dumps(
            {"jsonrpc": "2.0", "method": "recall.search", "params": params, "id": request_id}
        ).encode(),
        stream,
        ClientConnection(stream),
        reader=_PeerStreamReader(),
    )


def prepared(monkeypatch: pytest.MonkeyPatch, server: RpcServer) -> None:
    async def embed(_operation: object) -> list[float]:
        return [0.25] * 384

    monkeypatch.setattr(server, "_prepare_query_embedding", embed)


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_wait_frames_arrive_before_result_and_release_on_idle(monkeypatch, mode):
    async def scenario():
        server = RpcServer()
        prepared(monkeypatch, server)
        server._background_enrichment_idle.clear()
        writer = Writer()
        task = asyncio.create_task(dispatch(server, writer, {"query": "test", "mode": mode}))
        try:
            await asyncio.wait_for(writer.notified.wait(), 2)
            assert not task.done()
            wait = writer.frames[0]
            assert wait["id"] is None and wait["method"] == "search_progress"
            assert wait["params"] == {
                "request_id": "one",
                "attempt": 1,
                "state": "waiting",
                "waiting_for": "embedding_drain",
                "elapsed_ms": wait["params"]["elapsed_ms"],
                "preference_limit_ms": 5000,
            }
            server._background_enrichment_idle.set()
            await asyncio.wait_for(task, 2)
            assert [f.get("method", "result") for f in writer.frames] == [
                "search_progress",
                "search_progress",
                "result",
            ]
            release = writer.frames[1]["params"]
            assert release["state"] == "released" and release["reason"] == "drain_idle"
            assert release["waiting_for"] is None and release["request_id"] == "one"
            assert release["wait_ms"] >= 0
            assert writer.frames[-1]["result"] == []
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("slow", [False, True])
def test_preference_cap_includes_advisory_backpressure(monkeypatch, slow):
    monkeypatch.setattr(rpc, "_SEARCH_ENRICHMENT_WAIT_SECONDS", 0.02)

    async def scenario():
        server = RpcServer()
        prepared(monkeypatch, server)
        server._background_enrichment_idle.clear()
        writer = Writer()
        writer.slow = slow
        try:
            await asyncio.wait_for(dispatch(server, writer, {"query": "x", "mode": "vector"}), 2)
            assert len(writer.frames) == 3
            waiting, released = [frame["params"] for frame in writer.frames[:2]]
            assert waiting["preference_limit_ms"] == 20
            assert released["reason"] == "preference_elapsed"
            assert 20 <= released["wait_ms"] < 100
            assert writer.frames[-1]["result"] == []
        finally:
            await server.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "params",
    [
        {"mode": "keyword"},
        {"mode": "auto"},
        {"tool": "Bash", "mode": "hybrid"},
        {"mode": "vector"},
    ],
)
def test_idle_keyword_tool_and_auto_fallback_have_no_wait_frames(monkeypatch, params):
    async def scenario():
        server = RpcServer()
        prepared(monkeypatch, server)
        if params.get("mode") != "vector":
            server._background_enrichment_idle.clear()
        writer = Writer()
        try:
            await asyncio.wait_for(dispatch(server, writer, {"query": "x", **params}), 2)
            assert len(writer.frames) == 1 and writer.frames[0]["result"] == []
        finally:
            await server.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "phase",
    [
        "resolve_mode",
        "query_embedding",
        "enrichment_wait",
        "scan_admission",
        "read_queue",
        "scan",
        "recovery",
    ],
)
def test_timeout_names_active_and_largest_phase_and_cleans_up(monkeypatch, phase):
    monkeypatch.setattr(rpc, "_READ_REQUEST_TIMEOUT", 0.1)

    async def scenario():
        server = RpcServer()
        # Initialization is not part of these phase barriers. Otherwise the
        # schema-open cost can legitimately become the largest phase under load.
        await server._run_readonly(lambda conn: conn.execute("SELECT 1").fetchone())
        prepared(monkeypatch, server)
        writer = Writer()
        owner = None
        release = asyncio.Event()

        async def hold(*_args):
            await release.wait()

        if phase == "resolve_mode":
            monkeypatch.setattr(server, "_run_readonly", hold)
        elif phase == "query_embedding":
            monkeypatch.setattr(server, "_prepare_query_embedding", hold)
        elif phase == "enrichment_wait":
            server._background_enrichment_idle.clear()
        elif phase == "scan_admission":
            entered = asyncio.Event()

            async def held_scan():
                entered.set()
                await release.wait()

            owner = asyncio.create_task(server._run_duckdb_scan(held_scan))
            await entered.wait()
        elif phase == "read_queue":
            # Tool mode skips the resolution read so the read slots are the
            # first database boundary it reaches.
            server._read_slots = asyncio.BoundedSemaphore(0)
        elif phase == "scan":

            def heavy(**kwargs):
                return (
                    kwargs["conn"]
                    .execute(
                        "SELECT sum(sin(a.i * b.i)) FROM range(10000000) a(i), range(10000000) b(i)"
                    )
                    .fetchall()
                )

            monkeypatch.setattr(importlib.import_module("recall.services.search"), "search", heavy)
        else:

            def aborted(**_kwargs):
                raise duckdb.TransactionException("Current transaction is aborted; please ROLLBACK")

            monkeypatch.setattr(
                importlib.import_module("recall.services.search"), "search", aborted
            )
            monkeypatch.setattr(server, "_recover_search_shared_conn", hold)
        try:
            params = {"query": "x", "mode": "vector"}
            if phase == "read_queue":
                params["tool"] = "Bash"
            await asyncio.wait_for(dispatch(server, writer, params), 3)
            error = writer.frames[-1]["error"]
            assert error["code"] == -32603
            assert error["message"] == (
                "recall.search timed out after 0.1 seconds "
                f"(phase: {phase}; largest phase: {phase})"
            )
            data = error["data"]
            assert set(data) == {
                "kind",
                "phase",
                "largest_phase",
                "deadline_ms",
                "elapsed_ms",
                "cleanup_ms",
                "phase_ms",
            }
            assert data["kind"] == "search_timeout" and data["deadline_ms"] == 100
            assert data["phase"] == data["largest_phase"] == phase
            assert 100 <= data["elapsed_ms"] < 500
            assert 0 <= data["elapsed_ms"] - sum(data["phase_ms"].values()) <= 7
            assert list(data["phase_ms"]) == [
                "resolve_mode",
                "query_embedding",
                "enrichment_wait",
                "scan_admission",
                "read_queue",
                "scan",
                "recovery",
            ]
            assert all(f.get("params", {}).get("state") != "released" for f in writer.frames)
            assert not server._read_executions
        finally:
            release.set()
            if owner is not None:
                await owner
            await server.stop()

    asyncio.run(scenario())


def test_retry_frames_are_bounded_and_correlated(monkeypatch):
    monkeypatch.setattr(rpc, "_SEARCH_ENRICHMENT_WAIT_SECONDS", 0.01)

    async def scenario():
        server = RpcServer()
        prepared(monkeypatch, server)
        server._background_enrichment_idle.clear()
        calls = 0

        def search(**_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise duckdb.TransactionException("Current transaction is aborted; please ROLLBACK")
            return []

        async def recover(_err):
            return None

        monkeypatch.setattr(importlib.import_module("recall.services.search"), "search", search)
        monkeypatch.setattr(server, "_recover_search_shared_conn", recover)
        writer = Writer()
        try:
            await asyncio.wait_for(
                dispatch(server, writer, {"query": "x", "mode": "vector"}, 42), 2
            )
            frames = [frame["params"] for frame in writer.frames[:-1]]
            assert [(frame["attempt"], frame["state"]) for frame in frames] == [
                (1, "waiting"),
                (1, "released"),
                (2, "waiting"),
                (2, "released"),
            ]
            assert all(frame["request_id"] == 42 for frame in frames)
            assert writer.frames[-1]["result"] == []
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_timeout_snapshot_survives_cleanup_and_concurrent_request(monkeypatch):
    from recall.core.rpc_types import SearchPhase
    from recall.services.search_request import search_request

    monkeypatch.setattr(rpc, "_READ_REQUEST_TIMEOUT", 0.05)

    async def scenario():
        server = RpcServer()
        entered = asyncio.Event()
        cleaned = asyncio.Event()

        async def handler(params, _client):
            timing = search_request.get()
            assert timing is not None
            if params["query"] == "quick":
                timing.enter(SearchPhase.SCAN)
                return ["quick"]
            timing.enter(SearchPhase.QUERY_EMBEDDING)
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                timing.enter(SearchPhase.RECOVERY)
                await asyncio.sleep(0.03)
                cleaned.set()

        server._methods["recall.search"] = handler
        slow, quick = Writer(), Writer()
        task = asyncio.create_task(dispatch(server, slow, {"query": "slow"}, "slow-id"))
        try:
            await entered.wait()
            await dispatch(server, quick, {"query": "quick"}, "quick-id")
            await asyncio.wait_for(task, 1)
            data = slow.frames[0]["error"]["data"]
            assert cleaned.is_set()
            assert data["phase"] == data["largest_phase"] == "query_embedding"
            assert data["phase_ms"]["recovery"] == 0
            assert data["cleanup_ms"] >= 30
            assert data["elapsed_ms"] < 75
            assert quick.frames[0]["result"] == ["quick"]
            assert quick.frames[0]["id"] == "quick-id"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())


def test_timeout_largest_phase_can_differ_from_active_phase(monkeypatch):
    from recall.core.rpc_types import SearchPhase
    from recall.services.search_request import search_request

    monkeypatch.setattr(rpc, "_READ_REQUEST_TIMEOUT", 0.2)

    async def scenario():
        server = RpcServer()

        async def handler(_params, _client):
            timing = search_request.get()
            assert timing is not None
            timing.enter(SearchPhase.QUERY_EMBEDDING)
            await asyncio.sleep(0.13)
            timing.enter(SearchPhase.SCAN_ADMISSION)
            await asyncio.Event().wait()

        server._methods["recall.search"] = handler
        writer = Writer()
        try:
            await asyncio.wait_for(dispatch(server, writer, {"query": "x"}), 1)
            error = writer.frames[0]["error"]
            assert error["message"] == (
                "recall.search timed out after 0.2 seconds "
                "(phase: scan_admission; largest phase: query_embedding)"
            )
            assert error["data"]["phase_ms"]["query_embedding"] >= 130
            assert error["data"]["phase_ms"]["scan_admission"] < 130
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_disconnect_during_preference_leaves_no_notifier_or_claim(monkeypatch):
    async def scenario():
        server = RpcServer()
        prepared(monkeypatch, server)
        server._background_enrichment_idle.clear()
        writer = Writer()
        stream = cast(asyncio.StreamWriter, writer)
        reader = _PeerStreamReader()
        task = asyncio.create_task(
            server._process_request(
                b'{"jsonrpc":"2.0","method":"recall.search","params":{"query":"x","mode":"vector"},"id":1}',
                stream,
                ClientConnection(stream),
                reader=reader,
            )
        )
        try:
            await asyncio.wait_for(writer.notified.wait(), 2)
            reader.disconnected.set()
            await asyncio.wait_for(task, 1)
            assert len(writer.frames) == 1
            assert writer.frames[0]["params"]["state"] == "waiting"
            assert not server._read_executions
            server._background_enrichment_idle.set()
            following = Writer()
            await asyncio.wait_for(dispatch(server, following, {"query": "x", "mode": "vector"}), 2)
            assert following.frames[-1]["result"] == []
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())
