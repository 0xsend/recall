"""Thin wrapper bridging RPC client errors to CLI errors."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from recall.cli.contract import CliError, ErrorCode
from recall.core.config import AppConfig
from recall.core.rpc_client import (
    RpcCallError,
    RpcClient,
    RpcConnectionError,
    RpcReadDeadlineError,
)
from recall.core.rpc_types import (
    APP_CONFIRMATION_REQUIRED,
    APP_LOCKED,
    APP_NOT_FOUND,
    INVALID_PARAMS,
    RpcAdmissionBusyError,
    RpcError,
    serialize_rpc_value,
)
from recall.core.schema_version import SchemaTooNewError
from recall.read_budget import CLI_READ_BUDGET, RESPONSIVE_READ_METHODS, ReadBudget

# Map JSON-RPC error codes to CLI error codes
_ERROR_CODE_MAP: dict[int, tuple[ErrorCode, int]] = {
    INVALID_PARAMS: (ErrorCode.VALIDATION, 2),
    APP_LOCKED: (ErrorCode.LOCKED, 1),
    APP_CONFIRMATION_REQUIRED: (ErrorCode.CONFIRMATION_REQUIRED, 2),
    APP_NOT_FOUND: (ErrorCode.NOT_FOUND, 1),
}


@dataclass(frozen=True)
class DaemonUnreachableError(CliError):
    """No daemon answered: the call never reached a server that could judge it."""


@dataclass(frozen=True)
class DaemonBusyError(CliError):
    """Admission was refused; absence-only fallback is not permitted."""


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

    # Daemon lifecycle orchestration owns the local service check. Import lazily
    # because its commands also use this RPC wrapper.
    from recall.cli.daemon import refuse_service_auto_fork

    effective_params = dict(params or {})
    backup: str | None = None

    def before_auto_fork(config: AppConfig) -> None:
        nonlocal backup
        from recall.cli.daemon import prepare_confirmed_recreation

        snapshot = prepare_confirmed_recreation(config)
        if snapshot is not None:
            backup = str(snapshot)
            # The reset has already consumed the destructive request. Only the
            # full indexing work remains, with its original validated options.
            effective_params.update(recreate=False, full=True)

    confirmed_recreation = (
        method == "recall.index"
        and effective_params.get("recreate") is True
        and effective_params.get("confirmed") is True
    )
    try:
        result = _socket_call(
            method,
            effective_params,
            idle_timeout=idle_timeout,
            on_progress=on_progress,
            on_notification=on_notification,
            auto_fork=auto_fork,
            before_auto_fork=before_auto_fork if confirmed_recreation else refuse_service_auto_fork,
        )
    except CliError as err:
        if backup is None:
            raise
        raise CliError(
            code=err.code,
            message=f"{err.message}; pre-recreate backup: {backup}",
            details={**(err.details or {}), "backup_path": backup},
            exit_code=err.exit_code,
        ) from err
    if backup is not None:
        assert isinstance(result, dict)
        result = {**result, "backup_path": backup}
    return result


def _socket_call(
    method: str,
    params: dict[str, Any],
    *,
    idle_timeout: float | None,
    on_progress: Callable[[dict[str, Any]], None] | None,
    on_notification: Callable[[str, dict[str, Any]], None] | None,
    auto_fork: bool,
    before_auto_fork: Callable[[AppConfig], None],
) -> Any:
    client = RpcClient(before_auto_fork=before_auto_fork)
    budget = None
    if method in RESPONSIVE_READ_METHODS:
        budget = CLI_READ_BUDGET.get() or ReadBudget.start()
    try:
        client.connect(auto_fork=auto_fork, read_budget=budget)
        return client.call(
            method,
            params,
            idle_timeout=idle_timeout,
            on_progress=on_progress,
            on_notification=on_notification,
            read_budget=budget,
        )
    except SchemaTooNewError as err:
        raise CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=1) from err
    except RpcAdmissionBusyError as err:
        raise DaemonBusyError(
            code=ErrorCode.RUNTIME,
            message=err.message,
            details={"kind": "daemon_busy", "phase": "connect", "retryable": True},
            exit_code=1,
        ) from err
    except RpcReadDeadlineError as err:
        raise CliError(code=ErrorCode.RUNTIME, message=err.message, exit_code=1) from err
    except RpcConnectionError as err:
        raise DaemonUnreachableError(
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
