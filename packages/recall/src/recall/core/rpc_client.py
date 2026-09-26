"""JSON-RPC 2.0 client over Unix domain socket."""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from recall.core.config import AppConfig, create_private_dir
from recall.core.rpc_types import RpcCallError, RpcConnectionError, config_fingerprint

logger = logging.getLogger("recall.rpc_client")

# Timeouts
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 30.0
WRITE_TIMEOUT = 300.0

# Auto-fork polling
AUTO_FORK_MAX_WAIT = 5.0
AUTO_FORK_POLL_INTERVAL = 0.1

# Compact lifecycle sentinel
COMPACTION_SENTINEL_NAME = "recall.compacting"
STALE_COMPACTION_SENTINEL_AGE = 30 * 60


class RpcClient:
    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig.load()
        self._socket_path = self._config.data_dir / "recall.sock"
        self._sock: socket.socket | None = None
        self._local_config_fp = config_fingerprint(self._config)

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    def connect(self, *, auto_fork: bool = True) -> None:
        if self._sock is not None:
            return

        if not self._socket_path.exists():
            if not auto_fork:
                raise RpcConnectionError(
                    message=f"daemon not running (no socket at {self._socket_path}). "
                    "Start with: recall daemon start"
                )
            self._raise_if_compaction_blocks_auto_fork()
            self._auto_fork_daemon()

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(CONNECT_TIMEOUT)
        try:
            self._sock.connect(str(self._socket_path))
        except ConnectionRefusedError as err:
            self._sock.close()
            self._sock = None
            if not auto_fork:
                raise RpcConnectionError(
                    message="daemon socket exists but connection refused (stale socket). "
                    "Start with: recall daemon start"
                ) from err
            self._raise_if_compaction_blocks_auto_fork()
            logger.info("stale socket detected, removing and re-forking daemon")
            self._socket_path.unlink(missing_ok=True)
            self._auto_fork_daemon()
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.settimeout(CONNECT_TIMEOUT)
            try:
                self._sock.connect(str(self._socket_path))
            except (ConnectionRefusedError, FileNotFoundError, OSError) as err:
                self._sock.close()
                self._sock = None
                raise RpcConnectionError(
                    message=f"cannot connect to daemon after auto-fork retry: {err}"
                ) from err
        except (FileNotFoundError, OSError) as err:
            self._sock.close()
            self._sock = None
            raise RpcConnectionError(message=f"cannot connect to daemon: {err}") from err

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        idle_timeout: float | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        on_notification: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> Any:
        """Send one request and return its result.

        `idle_timeout` bounds *silence*, not the whole call: it is the socket
        timeout, so it is refreshed by every frame the daemon sends. A long
        index or a `--follow` stream therefore runs as long as it keeps
        talking, and only a quiet daemon is a failure. The old name for this
        argument was `timeout`, which promised a total deadline it never had.

        `on_notification` receives every notification frame as
        `(method, params)`. `on_progress` is the narrow adapter over it for
        `recall.index`: it sees `progress` frames only, so a streaming method's
        deltas never reach a progress renderer.
        """
        if self._sock is None:
            raise RpcConnectionError(message="not connected")

        # Unique per request: a timestamp-derived id collides across calls in
        # the same millisecond, letting a stale queued response satisfy the
        # next request's id filter.
        request_id = uuid4().hex
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": request_id,
        }
        line = json.dumps(request, separators=(",", ":")) + "\n"
        try:
            self._sock.sendall(line.encode("utf-8"))
        except OSError as err:
            self.close()
            raise RpcConnectionError(message=f"failed to send {method} request: {err}") from err

        effective_idle_timeout = idle_timeout or (
            WRITE_TIMEOUT if method == "recall.index" else READ_TIMEOUT
        )
        self._sock.settimeout(effective_idle_timeout)

        # Any abandoned or desynced exchange poisons the byte stream: the
        # daemon's late reply (it never cancels an in-flight handler) or a
        # half-read frame would be served to the next call on a reused
        # client. Close the socket on every such exit so the next call gets
        # a fresh connection.
        buffer = b""
        while True:
            try:
                chunk = self._sock.recv(65536)
            except TimeoutError as err:
                self.close()
                raise RpcConnectionError(
                    message=(
                        f"no frames from the daemon for {effective_idle_timeout:g}s "
                        f"while waiting on {method} (idle timeout, not a total deadline). "
                        "The daemon may still be working; check `recall daemon status`."
                    )
                ) from err
            except OSError as err:
                # ECONNRESET and friends leave the exchange unusable the same
                # way a timeout does (REQ-RPC-017): close, never propagate raw.
                self.close()
                raise RpcConnectionError(
                    message=f"socket error while reading {method} response: {err}"
                ) from err
            if not chunk:
                self.close()
                raise RpcConnectionError(message="connection closed by daemon")
            buffer += chunk

            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                try:
                    response = json.loads(line_bytes)
                except json.JSONDecodeError as err:
                    self.close()
                    raise RpcConnectionError(
                        message=f"malformed response frame from daemon: {err}"
                    ) from err

                # Notification frame (id is absent or null)
                if response.get("id") is None and "method" in response:
                    notification_method = str(response["method"])
                    notification_params = response.get("params", {})
                    if on_notification is not None:
                        on_notification(notification_method, notification_params)
                    if on_progress is not None and notification_method == "progress":
                        on_progress(notification_params)
                    continue

                # Final response
                if response.get("id") != request_id:
                    continue

                if "error" in response:
                    error = response["error"]
                    raise RpcCallError(
                        code=error.get("code", -1),
                        message=error.get("message", "unknown error"),
                        data=error.get("data"),
                    )

                # REQ-RPC-010: detect config staleness
                daemon_fp = response.get("_config_fp")
                if daemon_fp and daemon_fp != self._local_config_fp:
                    logger.warning(
                        "daemon embedding config differs from local config "
                        "(daemon=%s, local=%s). Run `recall daemon restart` "
                        "to apply the new configuration.",
                        daemon_fp,
                        self._local_config_fp,
                    )

                return response.get("result")

    def _auto_fork_daemon(self) -> None:
        logger.info("auto-forking daemon")
        recall_bin = _resolve_recall_binary()
        try:
            create_private_dir(self._config.data_dir)
            log_dir = self._config.data_dir / "logs"
            create_private_dir(log_dir)
            stdout_log = log_dir / "daemon.log"
            stderr_log = log_dir / "daemon.err.log"
            with (
                open(stdout_log, "a") as stdout_f,
                open(stderr_log, "a") as stderr_f,
            ):
                subprocess.Popen(
                    [recall_bin, "daemon", "start", "--background"],
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    start_new_session=True,
                )
        except Exception as err:
            raise RpcConnectionError(message=f"failed to auto-fork daemon: {err}") from err

        # Poll for socket with backoff
        waited = 0.0
        interval = AUTO_FORK_POLL_INTERVAL
        while waited < AUTO_FORK_MAX_WAIT:
            time.sleep(interval)
            waited += interval
            if self._socket_path.exists():
                logger.info("daemon socket appeared after %.1fs", waited)
                return
            interval = min(interval * 2, 1.0)

        raise RpcConnectionError(
            message=f"daemon did not start within {AUTO_FORK_MAX_WAIT}s. Try: recall daemon start"
        )

    def _raise_if_compaction_blocks_auto_fork(self) -> None:
        sentinel = self._config.data_dir / COMPACTION_SENTINEL_NAME
        if not sentinel.exists():
            return
        try:
            age = time.time() - sentinel.stat().st_mtime
        except OSError:
            age = 0.0
        if age < STALE_COMPACTION_SENTINEL_AGE:
            raise RpcConnectionError(
                message="recall compact is in progress; rerun this command after compact completes."
            )
        logger.warning(
            "ignoring stale compaction sentinel (age=%.0fs); proceeding with auto-fork",
            age,
        )

    def __enter__(self) -> RpcClient:
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def rpc_call(
    method: str,
    params: dict[str, Any] | None = None,
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    config: AppConfig | None = None,
) -> Any:
    """Convenience: connect, call, close in one shot."""
    client = RpcClient(config=config)
    try:
        client.connect()
        return client.call(method, params, on_progress=on_progress)
    finally:
        client.close()


def _resolve_recall_binary() -> str:
    """Find the recall binary for auto-fork."""
    from recall.core.config import resolve_recall_binary

    return resolve_recall_binary()
