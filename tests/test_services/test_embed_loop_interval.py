"""How long the embed loop waits after each kind of cycle."""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from lane_harness import lane_config
from recall.services.embed_phase import PendingEmbeds
from recall.services.rpc_server import EmbedBatchResult, EmbedStep, RpcServer, next_embed_step

rpc = importlib.import_module("recall.services.rpc_server")


def _config(tmp_path: Path):
    config = lane_config(tmp_path)
    return replace(config, daemon=replace(config.daemon, embed_interval=5, embed_backoff=30))


def test_a_drain_that_left_eligible_work_re_arms_at_once(tmp_path: Path) -> None:
    left = PendingEmbeds(message_count=40, tool_call_count=2, session_ids=["next"])
    assert next_embed_step(3, left, _config(tmp_path)) == EmbedStep(0.01, draining=True)


def test_a_drain_that_emptied_the_roster_waits_the_normal_interval(tmp_path: Path) -> None:
    """The cycle after a full drain finds nothing; running it at once only replaced
    `drained` as the reported outcome within milliseconds of the batch."""
    nothing = PendingEmbeds(message_count=0, tool_call_count=0, session_ids=[])
    assert next_embed_step(3, nothing, _config(tmp_path)) == EmbedStep(5, draining=False)


def test_an_empty_cycle_waits_the_normal_interval(tmp_path: Path) -> None:
    assert next_embed_step(0, None, _config(tmp_path)) == EmbedStep(5, draining=False)


def test_a_failed_or_deferred_cycle_backs_off(tmp_path: Path) -> None:
    left = PendingEmbeds(message_count=40, tool_call_count=2, session_ids=["next"])
    assert next_embed_step(-1, left, _config(tmp_path)) == EmbedStep(30, draining=False)


async def _wait_until(predicate, *, timeout: float = 5) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def _vector_search(server: RpcServer, mode: str = "vector") -> list[dict]:
    import json

    from recall.services.rpc_server import ClientConnection, _PeerStreamReader

    class Writer:
        def __init__(self) -> None:
            self.frames: list[dict] = []

        def is_closing(self) -> bool:
            return False

        def write(self, data: bytes) -> None:
            self.frames.append(json.loads(data))

        async def drain(self) -> None:
            return None

    writer = Writer()
    stream = cast(asyncio.StreamWriter, writer)
    request = {"jsonrpc": "2.0", "method": "recall.search", "id": 1}
    request["params"] = {"query": "x", "mode": mode}
    await asyncio.wait_for(
        server._process_request(
            json.dumps(request).encode(),
            stream,
            ClientConnection(stream),
            reader=_PeerStreamReader(),
        ),
        10,
    )
    return writer.frames


def _loop_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batches) -> RpcServer:
    """A server whose background loop runs `batches` at default intervals, first at once."""
    config = lane_config(tmp_path)
    config = replace(
        config,
        daemon=replace(config.daemon, embed=True, embed_interval=120, embed_idle_session=600),
    )
    server = RpcServer(config)
    monkeypatch.setattr(rpc, "_EMBED_FIRST_WAIT_SECONDS", 0.01)
    monkeypatch.setattr(server, "_load_runtime_config", lambda: config)

    async def embed_query(_operation: object) -> list[float]:
        return [0.25] * 384

    monkeypatch.setattr(server, "_prepare_query_embedding", embed_query)
    planned = iter(batches)

    async def batch(_config, state):
        result = await next(planned)(server)
        state.record_outcome("drained" if result.drained > 0 else "no-eligible-sessions")
        return result

    monkeypatch.setattr(server, "_run_embed_batch", batch)
    return server


async def _forever(_server: RpcServer) -> EmbedBatchResult:
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


@pytest.mark.parametrize(
    "left",
    [
        PendingEmbeds(message_count=0, tool_call_count=0, session_ids=[]),
        # Only a cooled session remains: it holds rows but no roster slot.
        PendingEmbeds(message_count=500, tool_call_count=0, session_ids=[]),
    ],
    ids=["nothing", "only-cooled"],
)
@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_drain_ending_batch_releases_preference(tmp_path, monkeypatch, left, mode) -> None:
    """REQ-RPC-020: the cycle that drains the last eligible session leaves searches unblocked.

    At default settings the next timer cycle is 120 s away; until then a search
    must not wait for a drain that has already ended.
    """

    async def draining(_server: RpcServer) -> EmbedBatchResult:
        return EmbedBatchResult(3, PendingEmbeds(3, 0, ["next"]))

    async def final(_server: RpcServer) -> EmbedBatchResult:
        return EmbedBatchResult(3, left)

    server = _loop_server(tmp_path, monkeypatch, [draining, final, _forever])

    async def scenario() -> None:
        task = await server._start_embed_phase()
        assert task is not None
        state = server._embed_state
        assert state is not None
        try:
            await _wait_until(lambda: state.next_interval == 120)
            assert state.last_outcome == "drained"
            frames = await _vector_search(server, mode)
            assert [frame.get("method", "result") for frame in frames] == ["result"]
        finally:
            server._shutdown_event.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_still_draining_backlog_keeps_bounded_preference(tmp_path, monkeypatch, mode) -> None:
    entered = asyncio.Event()

    async def draining(_server: RpcServer) -> EmbedBatchResult:
        return EmbedBatchResult(25, PendingEmbeds(1_000_000, 0, ["next"]))

    async def held(server: RpcServer) -> EmbedBatchResult:
        entered.set()
        return await _forever(server)

    monkeypatch.setattr(rpc, "_SEARCH_ENRICHMENT_WAIT_SECONDS", 0.05)
    server = _loop_server(tmp_path, monkeypatch, [draining, held])

    async def scenario() -> None:
        task = await server._start_embed_phase()
        assert task is not None
        try:
            await asyncio.wait_for(entered.wait(), 5)
            frames = await _vector_search(server, mode)
            progress = [frame["params"] for frame in frames if frame.get("method")]
            assert [(p["state"], p.get("reason")) for p in progress] == [
                ("waiting", None),
                ("released", "preference_elapsed"),
            ]
            assert progress[1]["wait_ms"] >= 50
            assert frames[-1]["result"] == []
        finally:
            server._shutdown_event.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())
