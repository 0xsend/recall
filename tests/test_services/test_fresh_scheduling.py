"""Fresh callers share the daemon's durable queue (REQ-RECON-002/003/004)."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from recall.core.config import AppConfig
from recall.db.source_files import SourceCatalog
from recall.parsers.codex import CodexParser
from recall.services.coordinator import capture_path, observe_path
from recall.services.rpc_server import RpcServer


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[RpcServer]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "off")
    server = RpcServer(AppConfig.load())
    yield server
    asyncio.run(server.stop())


def transcript(name: str, *, old: bool = False) -> Path:
    path = Path.home() / ".codex/sessions" / f"rollout-{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "session_meta", "payload": {"id": name}}) + "\n")
    append(path, name)
    if old:
        os.utime(path, (1, 1))
    return path


def append(path: Path, text: str) -> None:
    with path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": text,
                    },
                }
            )
            + "\n"
        )


def test_simultaneous_fresh_requests_capture_a_source_once(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = transcript("coalesced")
    entered, release = threading.Event(), threading.Event()
    original = CodexParser.parse
    captures = 0
    count_lock = threading.Lock()

    def blocked(self, *args, **kwargs):
        nonlocal captures
        with count_lock:
            captures += 1
        entered.set()
        assert release.wait(5)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CodexParser, "parse", blocked)

    async def scenario() -> None:
        requests = [asyncio.create_task(server.index_session_now(path)) for _ in range(25)]
        try:
            assert await asyncio.to_thread(entered.wait, 3)
        finally:
            release.set()
            await asyncio.gather(*requests)
        assert captures == 1
        assert server._get_conn().execute("SELECT COUNT(*) FROM messages").fetchone() == (1,)

    asyncio.run(scenario())


def test_continuous_fresh_requests_cannot_starve_history(server: RpcServer) -> None:
    parser = CodexParser()
    history = [transcript(f"old-{i}", old=True) for i in range(7)]
    for path in history:
        observe_path(parser, capture_path(parser, path), conn=server._get_conn())
    active = transcript("active")

    async def scenario() -> None:
        # Every request writes again. Fresh work must still spend reserved
        # service turns on the seven finite historical sources within 7*N.
        for i in range(49):
            append(active, f"active message {i}")
            job = asyncio.create_task(server.index_session_now(active))
            done, _ = await asyncio.wait({job}, timeout=3)
            if not done:
                server.request_shutdown()
                await job
                states = (
                    server._get_conn()
                    .execute(
                        "SELECT source_path, desired_generation, committed_generation, "
                        "committed_offset, size, last_error FROM source_files"
                    )
                    .fetchall()
                )
                pytest.fail(f"fresh request did not yield; catalog={states}")
            await job
        catalog = SourceCatalog(server._get_conn(), clock=time.time)
        for path in history:
            item = catalog.get("codex", str(path))
            assert item is not None
            assert item.committed_generation == item.desired_generation, path.name
            assert item.committed_offset == path.stat().st_size
        assert server._get_conn().execute("SELECT COUNT(*) FROM sessions").fetchone() == (8,)
        assert server._get_conn().execute("SELECT COUNT(*) FROM messages").fetchone() == (57,)

    asyncio.run(scenario())


def test_explicit_refresh_retries_error_after_an_acknowledged_generation(server: RpcServer) -> None:
    path = transcript("retry-ack")

    async def scenario() -> None:
        await server.index_session_now(path)
        catalog = SourceCatalog(server._get_conn(), clock=time.time)
        catalog.fail("codex", str(path), "temporary post-commit failure")
        await server.index_session_now(path)
        item = catalog.get("codex", str(path))
        assert item is not None and item.last_error is None
        assert item.committed_offset == path.stat().st_size
        assert server._get_conn().execute("SELECT COUNT(*) FROM messages").fetchone() == (1,)

    asyncio.run(scenario())


def test_unsupported_committed_prefix_remains_visible_without_an_automatic_retry_spin(
    server: RpcServer,
) -> None:
    path = transcript("unsupported")
    with path.open("a") as handle:
        handle.write('{"type":"future-conversation-record","payload":{"text":"unknown"}}\n')

    async def scenario() -> None:
        await server.index_session_now(path)
        catalog = SourceCatalog(server._get_conn(), clock=lambda: time.time() + 3600)
        item = catalog.get("codex", str(path))
        assert item is not None
        assert item.last_error == "unsupported"
        assert item.first_pending_at is not None
        assert item.committed_offset < path.stat().st_size
        assert catalog.pending() == []
        catalog.retry_now("codex", str(path))
        assert [entry.source_path for entry in catalog.pending()] == [str(path)]

    asyncio.run(scenario())


def test_kind_only_unsupported_park_is_rewritten_with_detail_once_per_start(
    server: RpcServer,
) -> None:
    """REQ-RECON-029: an older build's kind-only park gains its record detail."""
    path = transcript("opaque-park")
    with path.open("a") as handle:
        handle.write('{"type":"future-conversation-record","payload":{"text":"unknown"}}\n')

    async def scenario() -> None:
        await server.index_session_now(path)
        conn = server._get_conn()
        catalog = SourceCatalog(conn, clock=time.time)
        parked = catalog.get("codex", str(path))
        assert parked is not None and parked.last_error == "unsupported"
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        conn.execute(
            "UPDATE source_files SET diagnostics = ? WHERE source_path = ?",
            [json.dumps({"records": ["unsupported_record"]}), str(path)],
        )

        # Scheduling alone never reopens the park.
        assert not await server._serve_raw_turn()
        assert await server._reopen_opaque_unsupported() == 1
        assert await server._serve_raw_turn()

        item = catalog.get("codex", str(path))
        assert item is not None
        assert item.last_error == "unsupported"
        assert not item.current
        assert item.committed_offset == parked.committed_offset
        records = json.loads(item.diagnostics or "{}")["records"]
        assert records[0] == {
            "kind": "unsupported_record",
            "detail": "record: 'future-conversation-record'",
            "byte_offset": parked.committed_offset,
        }
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == messages

        # The rewritten park stays parked, including across a later start.
        assert not await server._serve_raw_turn()
        assert await server._reopen_opaque_unsupported() == 0
        assert not await server._serve_raw_turn()

    asyncio.run(scenario())


def test_append_between_observation_and_capture_does_not_leave_a_stuck_generation(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = transcript("capture-growth")
    original = CodexParser.parse
    appended = False

    def append_before_capture(self, *args, **kwargs):
        nonlocal appended
        if not appended:
            appended = True
            append(path, "arrived before capture opened")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CodexParser, "parse", append_before_capture)

    async def scenario() -> None:
        job = asyncio.create_task(server.index_session_now(path))
        done, _ = await asyncio.wait({job}, timeout=3)
        if not done:
            server.request_shutdown()
        await job
        assert done, "fresh request must yield after captured progress"
        item = SourceCatalog(server._get_conn(), clock=time.time).get("codex", str(path))
        assert item is not None and item.current
        assert item.committed_offset == path.stat().st_size
        assert server._get_conn().execute("SELECT COUNT(*) FROM messages").fetchone() == (2,)

    asyncio.run(scenario())
