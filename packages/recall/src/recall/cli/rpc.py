"""Thin wrapper bridging RPC client errors to CLI errors."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from typing import Any

from recall.cli.contract import CliError, ErrorCode
from recall.core.rpc_client import (
    RpcCallError,
    RpcClient,
    RpcConnectionError,
)
from recall.core.rpc_types import (
    APP_CONFIRMATION_REQUIRED,
    APP_LOCKED,
    APP_NOT_FOUND,
    INVALID_PARAMS,
    RpcError,
    serialize_rpc_value,
)

# Map JSON-RPC error codes to CLI error codes
_ERROR_CODE_MAP: dict[int, tuple[ErrorCode, int]] = {
    INVALID_PARAMS: (ErrorCode.VALIDATION, 2),
    APP_LOCKED: (ErrorCode.LOCKED, 1),
    APP_CONFIRMATION_REQUIRED: (ErrorCode.CONFIRMATION_REQUIRED, 2),
    APP_NOT_FOUND: (ErrorCode.NOT_FOUND, 1),
}

# When set, rpc_call_or_error dispatches in-process instead of via socket.
# Tests set this to bypass the daemon. Production code never touches it.
_in_process_server: Any = None


def set_in_process_server(server: Any) -> None:
    """Configure in-process RPC dispatch (for testing)."""
    global _in_process_server
    _in_process_server = server


def rpc_call_or_error(
    method: str,
    params: dict[str, Any] | None = None,
    *,
    idle_timeout: float | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    on_notification: Callable[[str, dict[str, Any]], None] | None = None,
    auto_fork: bool = True,
) -> Any:
    """Call an RPC method, raising CliError on failure.

    `idle_timeout` bounds silence between frames rather than the whole call;
    see `RpcClient.call`.
    """
    if _in_process_server is not None:
        return _in_process_call(_in_process_server, method, params or {})

    client = RpcClient()
    try:
        client.connect(auto_fork=auto_fork)
        return client.call(
            method,
            params,
            idle_timeout=idle_timeout,
            on_progress=on_progress,
            on_notification=on_notification,
        )
    except RpcConnectionError as err:
        raise CliError(
            code=ErrorCode.RUNTIME,
            message=err.message,
            exit_code=1,
        ) from err
    except RpcCallError as err:
        error_code, exit_code = _ERROR_CODE_MAP.get(err.code, (ErrorCode.RUNTIME, 1))
        raise CliError(
            code=error_code,
            message=err.message,
            details=err.data if isinstance(err.data, dict) else None,
            exit_code=exit_code,
        ) from err
    finally:
        client.close()


def _in_process_call(server: Any, method: str, params: dict[str, Any]) -> Any:
    """Dispatch directly to RPC server handlers without a socket."""
    real_server = server._real if hasattr(server, "_real") else server
    handler = real_server._methods.get(method)
    if handler is None:
        raise CliError(
            code=ErrorCode.RUNTIME,
            message=f"method not found: {method}",
            exit_code=1,
        )
    try:
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(handler(params, None))
        finally:
            loop.close()
        return serialize_rpc_value(result)
    except RpcError as err:
        error_code, exit_code = _ERROR_CODE_MAP.get(err.code, (ErrorCode.RUNTIME, 1))
        raise CliError(
            code=error_code,
            message=err.message,
            details=err.data if isinstance(err.data, dict) else None,
            exit_code=exit_code,
        ) from err
    except ValueError as err:
        raise CliError(
            code=ErrorCode.VALIDATION,
            message=str(err),
            exit_code=2,
        ) from err


def structured_progress(progress: dict[str, Any]) -> None:
    """Render progress notifications as plain stderr lines for a structured run.

    `REQ-INDEX-007`/`REQ-INDEX-024`: stdout belongs to the one final payload, so
    a `--json` run used to have no signal at all between starting and finishing.
    These lines never touch stdout, carry no cursor control (the destination is
    usually a log or a pipe, not a terminal), and are bounded by the frames the
    daemon sends -- one per captured inventory batch, plus phases and the
    terminal update.
    """
    total = progress.get("total", 0)
    sys.stderr.write(
        f"index: {progress.get('status', 'working')} "
        f"{progress.get('processed', 0)}/{total} | "
        f"indexed {progress.get('indexed', 0)} "
        f"skipped {progress.get('skipped', 0)} "
        f"failed {progress.get('failed', 0)}\n"
    )
    sys.stderr.flush()


def stderr_progress(progress: dict[str, Any]) -> None:
    """Render progress notifications to stderr."""
    if not sys.stderr.isatty():
        return
    processed = progress.get("processed", 0)
    total = progress.get("total", 0)
    indexed = progress.get("indexed", 0)
    skipped = progress.get("skipped", 0)
    failed = progress.get("failed", 0)
    if total == 0:
        return
    message = (
        f"\rIndexing {processed}/{total} | indexed {indexed} skipped {skipped} failed {failed}"
    )
    sys.stderr.write(message)
    sys.stderr.flush()
    if progress.get("status") == "done":
        sys.stderr.write("\n")
        sys.stderr.flush()
