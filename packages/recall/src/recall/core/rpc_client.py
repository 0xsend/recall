"""JSON-RPC 2.0 client over Unix domain socket."""

from __future__ import annotations

import errno
import json
import logging
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from recall.core.config import AppConfig
from recall.core.entrypoint import spawn_detached_recall
from recall.core.rpc_params import ResponsiveReadParams
from recall.core.rpc_types import (
    INVALID_PARAMS,
    RpcAdmissionBusyError,
    RpcCallError,
    RpcConnectionError,
    config_fingerprint,
)
from recall.read_budget import READ_OUTPUT_RESERVE_SECONDS, RESPONSIVE_READ_METHODS, ReadBudget

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


class RpcReadDeadlineError(RpcConnectionError):
    """A responsive request exhausted its local deadline."""


class RpcClient:
    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        before_auto_fork: Callable[[AppConfig], None] | None = None,
    ) -> None:
        self._config = config or AppConfig.load()
        self._before_auto_fork = before_auto_fork
        self._socket_path = self._config.data_dir / "recall.sock"
        self._sock: socket.socket | None = None
        self._local_config_fp = config_fingerprint(self._config)

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    def connect(self, *, auto_fork: bool = True, read_budget: ReadBudget | None = None) -> None:
        if self._sock is not None:
            return

        if not self._socket_path.exists():
            if not auto_fork:
                raise RpcConnectionError(
                    message=f"daemon not running (no socket at {self._socket_path}). "
                    "Start with: recall daemon start"
                )
            self._raise_if_auto_fork_blocked()
            self._auto_fork_daemon(read_budget=read_budget)

        try:
            self._connect_attempt(read_budget)
        except ConnectionRefusedError as err:
            if not auto_fork:
                raise RpcConnectionError(
                    message="daemon socket exists but connection refused (stale socket). "
                    "Start with: recall daemon start"
                ) from err
            self._raise_if_auto_fork_blocked()
            logger.info("stale socket detected, removing and re-forking daemon")
            self._socket_path.unlink(missing_ok=True)
            self._auto_fork_daemon(read_budget=read_budget)
            try:
                self._connect_attempt(read_budget)
            except OSError as err:
                raise self._transport_failure(
                    "cannot connect to daemon after auto-fork retry", err, read_budget
                ) from err
        except OSError as err:
            raise self._transport_failure("cannot connect to daemon", err, read_budget) from err

    def _connect_attempt(self, read_budget: ReadBudget | None) -> None:
        timeout = self._bounded_timeout(CONNECT_TIMEOUT, read_budget)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(str(self._socket_path))
        except OSError as err:
            sock.close()
            if err.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
                raise RpcAdmissionBusyError() from err
            raise
        self._sock = sock

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
        read_budget: ReadBudget | None = None,
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

        `read_budget` carries a process-local deadline. Responsive live/status
        calls send its remaining duration, bounded by any shorter supplied
        hint, and notifications cannot renew it. No clock value goes on wire.
        """
        if self._sock is None:
            raise RpcConnectionError(message="not connected")

        wire_params = dict(params or {})
        if read_budget is not None and method in RESPONSIVE_READ_METHODS:
            remaining_ms = int(read_budget.remaining(reserve=READ_OUTPUT_RESERVE_SECONDS) * 1000)
            if remaining_ms < 1:
                self.close()
                raise RpcReadDeadlineError(
                    message="responsive read deadline expired before sending"
                )
            try:
                hint = ResponsiveReadParams.model_validate(wire_params)
            except ValidationError as err:
                raise RpcCallError(
                    code=INVALID_PARAMS, message="remaining_read_ms must be an integer in 1..2000"
                ) from err
            wire_params["remaining_read_ms"] = min(hint.remaining_read_ms, remaining_ms)

        # Unique per request: a timestamp-derived id collides across calls in
        # the same millisecond, letting a stale queued response satisfy the
        # next request's id filter.
        request_id = uuid4().hex
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": wire_params,
            "id": request_id,
        }
        line = json.dumps(request, separators=(",", ":")) + "\n"
        try:
            if read_budget is not None:
                self._sock.settimeout(self._bounded_timeout(READ_TIMEOUT, read_budget))
            self._sock.sendall(line.encode("utf-8"))
        except OSError as err:
            raise self._transport_failure(
                f"failed to send {method} request", err, read_budget
            ) from err

        effective_idle_timeout = idle_timeout or (
            WRITE_TIMEOUT if method == "recall.index" else READ_TIMEOUT
        )
        self._sock.settimeout(self._bounded_timeout(effective_idle_timeout, read_budget))

        # Any abandoned or desynced exchange poisons the byte stream: the
        # daemon's late reply (it never cancels an in-flight handler) or a
        # half-read frame would be served to the next call on a reused
        # client. Close the socket on every such exit so the next call gets
        # a fresh connection.
        buffer = b""
        while True:
            try:
                if read_budget is not None:
                    self._sock.settimeout(
                        self._bounded_timeout(effective_idle_timeout, read_budget)
                    )
                chunk = self._sock.recv(65536)
            except TimeoutError as err:
                self.close()
                if read_budget is not None:
                    raise RpcReadDeadlineError(message="responsive read deadline expired") from err
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

    def _transport_failure(
        self, message: str, error: OSError, budget: ReadBudget | None
    ) -> RpcConnectionError:
        self.close()
        if isinstance(error, TimeoutError) and budget is not None:
            return RpcReadDeadlineError(message="responsive read deadline expired")
        return RpcConnectionError(message=f"{message}: {error}")

    def _bounded_timeout(self, timeout: float, budget: ReadBudget | None) -> float:
        if budget is None:
            return timeout
        remaining = budget.remaining(reserve=READ_OUTPUT_RESERVE_SECONDS)
        if remaining <= 0:
            self.close()
            raise RpcReadDeadlineError(message="responsive read deadline expired")
        return min(timeout, remaining)

    def _auto_fork_daemon(self, *, read_budget: ReadBudget | None = None) -> None:
        logger.info("auto-forking daemon")
        try:
            started = spawn_detached_recall(
                self._config.data_dir, ["daemon", "start", "--background"]
            )
        except Exception as err:
            raise RpcConnectionError(message=f"failed to auto-fork daemon: {err}") from err

        # Poll for socket with backoff
        waited = 0.0
        interval = AUTO_FORK_POLL_INTERVAL
        while waited < AUTO_FORK_MAX_WAIT:
            time.sleep(self._bounded_timeout(interval, read_budget))
            waited += interval
            if self._socket_path.exists():
                logger.info("daemon socket appeared after %.1fs", waited)
                return
            # A start that already failed will never open the socket.
            exit_status = started.process.poll()
            if exit_status is not None and exit_status != 0:
                raise RpcConnectionError(
                    message=f"daemon did not start: `{started.command}` exited with status "
                    f"{exit_status}; see {started.stderr_log}"
                )
            interval = min(interval * 2, 1.0)

        raise RpcConnectionError(
            message=f"daemon did not start within {AUTO_FORK_MAX_WAIT}s of running "
            f"`{started.command}`; see {started.stderr_log}"
        )

    def _raise_if_auto_fork_blocked(self) -> None:
        from recall.core.keyword_maintenance import MARKER_NAME

        if (self._config.data_dir / MARKER_NAME).exists():
            raise RpcConnectionError(
                message="Keyword maintenance is held; resume or finish its operation first."
            )
        self._raise_if_compaction_blocks_auto_fork()
        if self._before_auto_fork is not None:
            self._before_auto_fork(self._config)

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
