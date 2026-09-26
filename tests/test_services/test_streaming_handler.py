"""Serving a streaming method to a client that may vanish (REQ-LIVE-004).

`--follow` holds one connection open and emits notifications until its
deadline. The connection loop awaits its handler inline, so while a stream is
running nothing is reading the socket — the only place the daemon learns the
peer is gone is the next write. That makes `send_notification` the
cancel-on-disconnect point, and the contract it owes is: stop the producer,
and answer nobody.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest
from recall.services.rpc_server import (
    ClientConnection,
    ClientDisconnected,
    RpcServer,
)


class _FakeWriter:
    """Just enough `asyncio.StreamWriter` to record frames and hang up."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self._closing = False

    def write(self, data: bytes) -> None:
        if self._closing:
            raise BrokenPipeError("peer closed")
        self.frames.append(json.loads(data))

    async def drain(self) -> None:
        if self._closing:
            raise ConnectionResetError("peer closed")

    def is_closing(self) -> bool:
        return self._closing

    def hang_up(self) -> None:
        self._closing = True


def test_send_notification_to_a_departed_client_raises_client_disconnected() -> None:
    writer = _FakeWriter()
    client = ClientConnection(cast(asyncio.StreamWriter, writer))
    writer.hang_up()

    with pytest.raises(ClientDisconnected):
        asyncio.run(client.send_notification("live.delta", {"idx": 1}))


def test_send_progress_to_a_departed_client_is_ignored() -> None:
    """An index must not fail because whoever asked for it stopped watching."""
    writer = _FakeWriter()
    client = ClientConnection(cast(asyncio.StreamWriter, writer))
    writer.hang_up()

    asyncio.run(client.send_progress(processed=1, total=2, status="indexing"))

    assert writer.frames == []


def test_a_stream_that_loses_its_client_answers_nobody(tmp_path, monkeypatch) -> None:
    """No result frame, no error frame — and the handler's cleanup still runs."""
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path))
    from recall.core.config import AppConfig

    server = RpcServer(config=AppConfig.load())
    writer = _FakeWriter()
    client = ClientConnection(cast(asyncio.StreamWriter, writer))
    released: list[str] = []

    async def streaming_handler(params: dict[str, Any], connection: ClientConnection | None) -> Any:
        assert connection is not None
        try:
            await connection.send_notification("live.delta", {"idx": 1})
            writer.hang_up()
            await connection.send_notification("live.delta", {"idx": 2})
            return {"event": "closed", "reason": "timeout"}
        finally:
            released.append("subscription")

    server._methods["recall.test_stream"] = streaming_handler
    request = json.dumps(
        {"jsonrpc": "2.0", "method": "recall.test_stream", "params": {}, "id": "1"}
    )

    asyncio.run(
        server._process_request(request.encode(), cast(asyncio.StreamWriter, writer), client)
    )

    assert released == ["subscription"]
    assert writer.frames == [
        {"jsonrpc": "2.0", "method": "live.delta", "params": {"idx": 1}, "id": None}
    ]
