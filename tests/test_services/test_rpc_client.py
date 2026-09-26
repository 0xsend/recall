"""Unit tests for the RPC client."""

from __future__ import annotations

import contextlib
import json
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest
from recall.core.rpc_client import RpcCallError, RpcClient, RpcConnectionError


class TestRpcClientConnect:
    def test_raises_when_socket_missing_and_no_auto_fork(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path))
        from recall.core.config import AppConfig

        config = AppConfig.load()
        client = RpcClient(config=config)

        with pytest.raises(RpcConnectionError) as exc_info:
            client.connect(auto_fork=False)
        assert "daemon not running" in exc_info.value.message

    def test_call_raises_when_not_connected(self) -> None:
        client = RpcClient.__new__(RpcClient)
        client._sock = None

        with pytest.raises(RpcConnectionError) as exc_info:
            client.call("recall.search", {"query": "test"})
        assert "not connected" in exc_info.value.message


class TestRpcClientProtocol:
    def test_sends_jsonrpc_request_and_receives_response(self) -> None:
        """Test the client can send a request and parse a response via a real Unix socket."""
        sock_path = Path(tempfile.mkdtemp()) / "t.sock"

        # Start a simple echo server
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(str(sock_path))
        server_sock.listen(1)

        def handle():
            conn, _ = server_sock.accept()
            data = conn.recv(4096)
            request = json.loads(data.strip())
            response = {
                "jsonrpc": "2.0",
                "result": {"found": True},
                "id": request["id"],
            }
            conn.sendall(json.dumps(response, separators=(",", ":")).encode() + b"\n")
            conn.close()

        thread = threading.Thread(target=handle, daemon=True)
        thread.start()

        # Create client pointing at our test socket
        client = RpcClient.__new__(RpcClient)
        client._socket_path = sock_path
        client._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client._sock.settimeout(5.0)
        client._sock.connect(str(sock_path))

        result = client.call("recall.search", {"query": "test"})
        assert result == {"found": True}

        client.close()
        server_sock.close()
        thread.join(timeout=2)

    def test_handles_error_response(self) -> None:
        sock_path = Path(tempfile.mkdtemp()) / "t.sock"

        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(str(sock_path))
        server_sock.listen(1)

        def handle():
            conn, _ = server_sock.accept()
            data = conn.recv(4096)
            request = json.loads(data.strip())
            response = {
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": "query is required"},
                "id": request["id"],
            }
            conn.sendall(json.dumps(response, separators=(",", ":")).encode() + b"\n")
            conn.close()

        thread = threading.Thread(target=handle, daemon=True)
        thread.start()

        client = RpcClient.__new__(RpcClient)
        client._socket_path = sock_path
        client._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client._sock.settimeout(5.0)
        client._sock.connect(str(sock_path))

        with pytest.raises(RpcCallError) as exc_info:
            client.call("recall.search", {})

        assert exc_info.value.code == -32602
        assert exc_info.value.message == "query is required"

        client.close()
        server_sock.close()
        thread.join(timeout=2)

    def test_handles_progress_notifications(self) -> None:
        sock_path = Path(tempfile.mkdtemp()) / "t.sock"

        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(str(sock_path))
        server_sock.listen(1)

        def handle():
            conn, _ = server_sock.accept()
            data = conn.recv(4096)
            request = json.loads(data.strip())
            # Send progress notification first
            progress = {
                "jsonrpc": "2.0",
                "method": "progress",
                "params": {"processed": 1, "total": 2, "status": "indexed"},
            }
            conn.sendall(json.dumps(progress, separators=(",", ":")).encode() + b"\n")
            # Then send final response
            response = {
                "jsonrpc": "2.0",
                "result": {"indexed": 2},
                "id": request["id"],
            }
            conn.sendall(json.dumps(response, separators=(",", ":")).encode() + b"\n")
            conn.close()

        thread = threading.Thread(target=handle, daemon=True)
        thread.start()

        progress_events: list[dict] = []

        client = RpcClient.__new__(RpcClient)
        client._socket_path = sock_path
        client._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client._sock.settimeout(5.0)
        client._sock.connect(str(sock_path))

        result = client.call(
            "recall.index", {"full": True}, on_progress=lambda p: progress_events.append(p)
        )

        assert result == {"indexed": 2}
        assert len(progress_events) == 1
        assert progress_events[0]["processed"] == 1

        client.close()
        server_sock.close()
        thread.join(timeout=2)


class TestRpcClientContextManager:
    def test_context_manager_closes_on_exit(self) -> None:
        sock_path = Path(tempfile.mkdtemp()) / "t.sock"
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(str(sock_path))
        server_sock.listen(1)

        client = RpcClient.__new__(RpcClient)
        client._socket_path = sock_path
        client._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client._sock.settimeout(1.0)
        client._sock.connect(str(sock_path))

        with client:
            assert client._sock is not None

        assert client._sock is None
        server_sock.close()


class TestRpcClientStreamHygiene:
    """A timed-out or desynced socket must never serve the next request.

    Regression: RpcClient left the socket open after a read timeout, so on a
    reused client the abandoned request's late response stayed queued in the
    stream; with millisecond-timestamp request ids a rapid retry could reuse
    the same id and adopt the stale payload as its own (observed once as a
    mismatched session payload from `recall show`).
    """

    def test_timeout_closes_socket_and_late_response_is_not_replayed(self) -> None:
        import time as time_module

        sock_path = Path(tempfile.mkdtemp()) / "t.sock"
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(str(sock_path))
        server_sock.listen(2)

        def handle() -> None:
            conn, _ = server_sock.accept()
            conn.recv(4096)
            # Withhold the reply past the client timeout, then deliver it
            # late into the first (abandoned) connection.
            time_module.sleep(0.5)
            late = {
                "jsonrpc": "2.0",
                "result": {"payload": "stale-session"},
                "id": 0,
            }
            with contextlib.suppress(OSError):
                conn.sendall(json.dumps(late, separators=(",", ":")).encode() + b"\n")
            # Serve the client's second connection correctly.
            conn2, _ = server_sock.accept()
            data2 = conn2.recv(4096)
            req2 = json.loads(data2.strip())
            resp2 = {"jsonrpc": "2.0", "result": {"payload": "fresh"}, "id": req2["id"]}
            conn2.sendall(json.dumps(resp2, separators=(",", ":")).encode() + b"\n")
            conn2.close()
            conn.close()

        thread = threading.Thread(target=handle, daemon=True)
        thread.start()

        client = RpcClient.__new__(RpcClient)
        client._socket_path = sock_path
        client._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client._sock.settimeout(5.0)
        client._sock.connect(str(sock_path))

        with pytest.raises(RpcConnectionError) as exc_info:
            client.call("recall.show", {"session_id": "x"}, idle_timeout=0.2)
        # The message names the silence it observed, not a total deadline it
        # never had: an operator who read "timeout waiting for response" while
        # the daemon was still indexing had no way to tell those apart.
        assert "idle timeout" in exc_info.value.message
        assert "no frames from the daemon for 0.2s" in exc_info.value.message

        # The poisoned socket must be gone so nothing can read the late frame.
        assert client._sock is None

        client.connect(auto_fork=False)
        result = client.call("recall.show", {"session_id": "y"}, idle_timeout=5.0)
        assert result == {"payload": "fresh"}

        client.close()
        server_sock.close()
        thread.join(timeout=2)

    def test_request_ids_unique_when_clock_frozen(self, monkeypatch) -> None:
        import recall.core.rpc_client as rpc_client_module

        monkeypatch.setattr(rpc_client_module.time, "monotonic", lambda: 1234.5678)

        sock_path = Path(tempfile.mkdtemp()) / "t.sock"
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(str(sock_path))
        server_sock.listen(1)
        seen_ids: list[object] = []

        def handle() -> None:
            conn, _ = server_sock.accept()
            buffer = b""
            for _ in range(2):
                while b"\n" not in buffer:
                    buffer += conn.recv(4096)
                line, buffer = buffer.split(b"\n", 1)
                request = json.loads(line)
                seen_ids.append(request["id"])
                response = {"jsonrpc": "2.0", "result": {}, "id": request["id"]}
                conn.sendall(json.dumps(response, separators=(",", ":")).encode() + b"\n")
            conn.close()

        thread = threading.Thread(target=handle, daemon=True)
        thread.start()

        client = RpcClient.__new__(RpcClient)
        client._socket_path = sock_path
        client._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client._sock.settimeout(5.0)
        client._sock.connect(str(sock_path))

        client.call("recall.list", {}, idle_timeout=5.0)
        client.call("recall.list", {}, idle_timeout=5.0)

        assert len(seen_ids) == 2
        assert seen_ids[0] != seen_ids[1]

        client.close()
        server_sock.close()
        thread.join(timeout=2)

    def test_malformed_frame_raises_connection_error_and_closes(self) -> None:
        sock_path = Path(tempfile.mkdtemp()) / "t.sock"
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(str(sock_path))
        server_sock.listen(1)

        def handle() -> None:
            conn, _ = server_sock.accept()
            conn.recv(4096)
            conn.sendall(b"{this is not json\n")
            conn.close()

        thread = threading.Thread(target=handle, daemon=True)
        thread.start()

        client = RpcClient.__new__(RpcClient)
        client._socket_path = sock_path
        client._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client._sock.settimeout(5.0)
        client._sock.connect(str(sock_path))

        with pytest.raises(RpcConnectionError) as exc_info:
            client.call("recall.list", {}, idle_timeout=5.0)
        assert "malformed" in exc_info.value.message
        assert client._sock is None

        server_sock.close()
        thread.join(timeout=2)

    def test_connection_reset_raises_connection_error_and_closes(self) -> None:
        """REQ-RPC-017: non-timeout read failures (e.g. ECONNRESET) must also
        close the socket and surface RpcConnectionError, not propagate raw."""

        class _ResettingSocket:
            def settimeout(self, value: float | None) -> None:
                return None

            def sendall(self, data: bytes) -> None:
                return None

            def recv(self, size: int) -> bytes:
                raise ConnectionResetError(54, "Connection reset by peer")

            def close(self) -> None:
                return None

        client = RpcClient.__new__(RpcClient)
        client._socket_path = Path("/nonexistent")
        client._sock = cast(socket.socket, _ResettingSocket())

        with pytest.raises(RpcConnectionError) as exc_info:
            client.call("recall.list", {}, idle_timeout=5.0)
        assert "socket error" in exc_info.value.message
        assert client._sock is None


class TestRpcClientNotifications:
    """REQ-LIVE-004: a streaming method needs every notification, not just progress."""

    def _call(
        self,
        *,
        frames: list[dict[str, object]],
        gap: float = 0.0,
        **call_kwargs: Any,
    ) -> object:
        """Answer one request with `frames`, spaced by `gap`, then a final result."""
        sock_path = Path(tempfile.mkdtemp()) / "t.sock"
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(str(sock_path))
        server_sock.listen(1)

        def handle() -> None:
            conn, _ = server_sock.accept()
            request = json.loads(conn.recv(4096).strip())
            # A client that gives up on its idle timeout closes the socket
            # mid-script, which is the case one of these tests is about.
            with contextlib.suppress(OSError):
                for frame in frames:
                    time.sleep(gap)
                    conn.sendall(json.dumps(frame, separators=(",", ":")).encode() + b"\n")
                time.sleep(gap)
                final = {"jsonrpc": "2.0", "result": {"ok": True}, "id": request["id"]}
                conn.sendall(json.dumps(final, separators=(",", ":")).encode() + b"\n")
            conn.close()

        thread = threading.Thread(target=handle, daemon=True)
        thread.start()

        client = RpcClient.__new__(RpcClient)
        client._socket_path = sock_path
        client._local_config_fp = "test"
        client._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client._sock.settimeout(5.0)
        client._sock.connect(str(sock_path))
        try:
            return client.call("recall.show", {}, **call_kwargs)
        finally:
            client.close()
            server_sock.close()
            thread.join(timeout=2)

    def test_on_notification_sees_every_frame_with_its_method(self) -> None:
        seen: list[tuple[str, dict[str, object]]] = []

        result = self._call(
            frames=[
                {"jsonrpc": "2.0", "method": "progress", "params": {"processed": 1}, "id": None},
                {"jsonrpc": "2.0", "method": "live.delta", "params": {"idx": 42}, "id": None},
            ],
            on_notification=lambda method, params: seen.append((method, params)),
        )

        assert result == {"ok": True}
        assert seen == [("progress", {"processed": 1}), ("live.delta", {"idx": 42})]

    def test_on_progress_ignores_a_notification_that_is_not_progress(self) -> None:
        """A `--follow` delta must not be handed to the index progress renderer."""
        seen: list[dict[str, object]] = []

        self._call(
            frames=[
                {"jsonrpc": "2.0", "method": "live.delta", "params": {"idx": 42}, "id": None},
                {"jsonrpc": "2.0", "method": "progress", "params": {"processed": 3}, "id": None},
            ],
            on_progress=seen.append,
        )

        assert seen == [{"processed": 3}]

    def test_idle_timeout_bounds_silence_not_the_whole_call(self) -> None:
        """`--follow` runs longer than any per-frame gap; only a quiet stream is a failure.

        Four 0.15 s gaps take 0.6 s in total, well past the 0.4 s bound, and the
        call still succeeds because no single silence reaches it.
        """
        result = self._call(
            frames=[
                {"jsonrpc": "2.0", "method": "live.delta", "params": {"idx": idx}, "id": None}
                for idx in range(3)
            ],
            gap=0.15,
            idle_timeout=0.4,
        )

        assert result == {"ok": True}

    def test_a_stream_silent_past_the_idle_timeout_fails(self) -> None:
        with pytest.raises(RpcConnectionError) as exc_info:
            self._call(
                frames=[
                    {"jsonrpc": "2.0", "method": "live.delta", "params": {"idx": 0}, "id": None}
                ],
                gap=0.4,
                idle_timeout=0.15,
            )

        assert "timeout" in exc_info.value.message
