from __future__ import annotations

import logging
import math
import os
import signal
import stat as _stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

from recall.cli.contract import (
    CliError,
    ErrorCode,
    OutputFormat,
    emit_data_with_cta,
    emit_error,
    is_explicit_source,
    parse_fields,
    parse_params,
    render_cta_hints,
    render_dry_run,
    require_confirmation,
    resolve_bool_param,
    resolve_int_param,
    resolve_optional_bool_param,
    resolve_optional_int_param,
    resolve_output_format_early,
    resolve_output_format_from_params,
    resolve_param,
    validate_source,
    validate_structured_fields,
)
from recall.cli.cta import (
    cta_for_daemon,
    cta_for_daemon_install,
    cta_for_daemon_restart,
    cta_for_daemon_start,
    cta_for_daemon_status,
    cta_for_daemon_stop,
    cta_for_daemon_uninstall,
    cta_for_dry_run,
)
from recall.cli.manifest import COMMANDS, output_fields_for
from recall.cli.rpc import rpc_call_or_error
from recall.cli.status_notices import _render_status_notice_from_status
from recall.core.config import AppConfig, create_private_dir
from recall.core.types import parse_scheduler_kind
from recall.services.daemon import (
    _log_paths,
    daemon_status,
    install_scheduler,
    restart_daemon_durable,
    start_daemon_durable,
    stop_daemon_durable,
    stop_daemon_soft,
    uninstall_scheduler,
)

logger = logging.getLogger("recall.cli.daemon")

app = typer.Typer(add_completion=False, invoke_without_command=True)

# Local wall-clock time with an explicit UTC offset, so a line read out of
# `daemon.log` on any host is unambiguous without knowing the host's timezone.
_DAEMON_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DAEMON_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


class _DaemonLogHandler(logging.StreamHandler):
    """The daemon process's own stderr handler.

    A distinct type so repeated startup configuration is idempotent and so a
    test can tell Recall's handler from whatever else shares the root logger.
    """


def _configure_daemon_logging(level: int) -> None:
    """Install the daemon's log handler before anything can emit a record.

    The daemon is a long-lived process whose only diagnostic artifact is
    `daemon.log` / `daemon.err.log`. Without this, nothing configures logging at
    startup: Python's lastResort handler prints bare WARNING+ text with no
    timestamp and no logger name, and the first client index request to reach
    `logging.basicConfig` decides the level for the rest of the process
    lifetime.

    Only `recall` loggers move to `level`; third-party loggers keep their
    default effective level so a debug daemon is not buried under asyncio and
    watchdog records. Records from either still carry the timestamped format,
    because the handler lives on the root logger.
    """
    root = logging.getLogger()
    handler = next(
        (existing for existing in root.handlers if isinstance(existing, _DaemonLogHandler)),
        None,
    )
    if handler is None:
        handler = _DaemonLogHandler(sys.stderr)
        root.addHandler(handler)
    handler.setFormatter(logging.Formatter(_DAEMON_LOG_FORMAT, datefmt=_DAEMON_LOG_DATEFMT))
    logging.getLogger("recall").setLevel(level)


def _fd_is_managed_log(fd: int, path: Path) -> bool:
    """Return True when fd points at the same regular file as `path`.

    Startup log rotation must only steal fd 1/2 when launchd or systemd already
    connected them to Recall's managed log files. A tty, pipe, socket, or
    unrelated file belongs to the caller and must keep its existing stream.
    """
    try:
        st_fd = os.fstat(fd)
        st_path = os.stat(path)
    except OSError:
        return False
    if not _stat.S_ISREG(st_fd.st_mode):
        return False
    if not _stat.S_ISREG(st_path.st_mode):
        return False
    return st_fd.st_ino == st_path.st_ino and st_fd.st_dev == st_path.st_dev


def _rotate_daemon_log_if_needed(
    path: Path,
    max_bytes: int,
    target_fd: int | None,
) -> None:
    """Rotate `path` if oversized and optionally redirect target_fd to the fresh file.

    target_fd is 1 for daemon.log and 2 for daemon.err.log so subsequent writes
    from the running daemon process land in the new active file instead of the
    renamed `.1` inode inherited from launchd/systemd.
    """
    if max_bytes <= 0:
        return
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return
    try:
        os.chmod(path, 0o600)
    except OSError as err:
        logger.warning(
            "log rotation: chmod 0600 on under-threshold %s failed (%s); "
            "file may be world-readable",
            path,
            err,
        )
    if size <= max_bytes:
        return
    rotated = path.with_suffix(path.suffix + ".1")
    # Atomically overwrite the single retained generation.
    try:
        os.replace(path, rotated)
    except OSError as err:
        logger.warning(
            "log rotation: replace %s -> %s failed (%s); leaving original in place",
            path,
            rotated,
            err,
        )
        return
    # Legacy logs may have been created before private permissions were enforced.
    try:
        os.chmod(rotated, 0o600)
    except OSError as err:
        logger.warning(
            "log rotation: chmod 0600 on rotated %s failed (%s); file may be world-readable",
            rotated,
            err,
        )
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    try:
        new_fd = os.open(str(path), flags, 0o600)
    except OSError as err:
        logger.warning(
            "log rotation: open of fresh %s failed (%s); "
            "rolling rename back to keep active log path present",
            path,
            err,
        )
        # Put the renamed file back so the expected active path is not absent.
        try:
            os.replace(rotated, path)
        except OSError as restore_err:
            logger.warning(
                "log rotation: rollback rename %s -> %s failed (%s); "
                "active log path %s is absent until next daemon restart",
                rotated,
                path,
                restore_err,
                path,
            )
        return
    try:
        if target_fd is None:
            return
        try:
            if target_fd == 1:
                sys.stdout.flush()
            elif target_fd == 2:
                sys.stderr.flush()
        except Exception:
            pass
        try:
            os.dup2(new_fd, target_fd)
        except OSError as err:
            logger.warning(
                "log rotation: dup2 of fd %d to %s failed (%s); "
                "running daemon will continue writing to %s",
                target_fd,
                path,
                err,
                rotated,
            )
    finally:
        os.close(new_fd)


@app.callback()
def command(
    ctx: typer.Context,
    once: bool = typer.Option(False, "--once", help="Run a single daemon cycle and exit"),
    mode: str | None = typer.Option(None, "--mode", help="Daemon mode: auto, watch, poll"),
    interval: int | None = typer.Option(None, "--interval", help="Polling interval in seconds"),
    embed: bool | None = typer.Option(
        None, "--embed/--no-embed", help="Enable/disable the adaptive embed phase"
    ),
    source: str | None = typer.Option(
        None, "--source", help="claude-code, codex, pi-agent, grok, kimi-code"
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size", help="Override embedding batch size for this daemon process"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        # Rich renders square brackets as markup, so the TOML table is named in
        # dotted form rather than as `[daemon] log_level`.
        help=(
            "Verbose logging for a foreground daemon started by this command. "
            "With --once the running daemon serves the cycle and keeps its own "
            "level: set daemon.log_level in config.toml or "
            "RECALL_DAEMON_LOG_LEVEL instead"
        ),
    ),
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output (single-run only)"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
    cta: bool = typer.Option(False, "--cta", help="Include call-to-action suggestions"),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(
            params,
            allowed={
                "once",
                "interval",
                "embed",
                "source",
                "batch_size",
                "verbose",
                "mode",
                "fields",
                "format",
                "json",
                "cta",
            },
            types={
                "once": "boolean",
                "interval": "integer",
                "embed": "boolean",
                "source": "string",
                "batch_size": "integer",
                "verbose": "boolean",
                "mode": "string",
                "fields": "string",
                "format": "string",
                "json": "boolean",
                "cta": "boolean",
            },
        )
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        once = resolve_bool_param(ctx, payload, "once", once)
        mode_value = resolve_param(ctx, payload, "mode", mode)
        if mode_value is not None and mode_value not in {"auto", "watch", "poll"}:
            raise ValueError(f"unsupported daemon mode: {mode_value}")
        interval = resolve_optional_int_param(ctx, payload, "interval", interval)
        if interval is not None and interval <= 0:
            raise ValueError("interval must be positive")
        embed = resolve_optional_bool_param(ctx, payload, "embed", embed)
        source = resolve_param(ctx, payload, "source", source)
        validate_source(source)
        batch_size = resolve_optional_int_param(ctx, payload, "batch_size", batch_size)
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        verbose = resolve_bool_param(ctx, payload, "verbose", verbose)
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)

        if not once and output_format != OutputFormat.TEXT:
            # Foreground daemon: check if format was explicitly requested
            # If auto-resolved (no explicit format/json), default to TEXT
            format_was_explicit = (
                is_explicit_source(ctx, "json_output")
                or is_explicit_source(ctx, "format_name")
                or "format" in payload
                or "json" in payload
            )
            if not format_was_explicit:
                output_format = OutputFormat.TEXT

        if once:
            # --once: send a single cycle RPC to the daemon
            rpc_params: dict[str, object] = {"once": True, "verbose": verbose}
            if mode_value:
                rpc_params["mode"] = str(mode_value)
            if interval is not None:
                rpc_params["interval"] = interval
            if embed is not None:
                rpc_params["embed"] = embed
            if source:
                rpc_params["source"] = source
            if batch_size is not None:
                rpc_params["batch_size"] = batch_size
            summary = rpc_call_or_error("recall.daemon_run", rpc_params)
        else:
            # No --once: start the RPC server directly (this IS the daemon)
            if output_format != OutputFormat.TEXT:
                raise CliError(
                    code=ErrorCode.VALIDATION,
                    message="structured output is only supported with --once",
                    exit_code=2,
                )
            # `_start_foreground_server` owns this process's logging from here.
            _start_foreground_server(
                verbose=verbose,
                mode_override=mode_value,
                embed_override=embed,
                source_override=source,
                batch_size_override=batch_size,
            )
            return
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None

    summary_dict = summary if isinstance(summary, dict) else {}
    ctas = cta_for_daemon(summary_dict) if (cta or output_format == OutputFormat.TEXT) else []

    if output_format != OutputFormat.TEXT:
        try:
            emit_data_with_cta(
                summary,
                ctas,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("daemon"),
            )
        except CliError as err:
            emit_error(err, output_format=output_format)
            raise typer.Exit(code=err.exit_code) from None
        return

    if once and summary is not None and isinstance(summary, dict):
        if summary.get("record_status_persisted") is False:
            typer.echo(
                "WARN: cycle ran but status metadata was not persisted; "
                "run `recall daemon status` to inspect runtime_state"
            )
        idx_summary = summary.get("index_summary", {})
        typer.echo(
            f"Daemon cycle indexed {idx_summary.get('indexed', 0)}, "
            f"skipped {idx_summary.get('skipped', 0)}, "
            f"failed {idx_summary.get('failed', 0)}."
        )
    render_cta_hints(ctas)


def _start_foreground_server(
    *,
    verbose: bool = False,
    mode_override: str | None = None,
    embed_override: bool | None = None,
    source_override: str | None = None,
    batch_size_override: int | None = None,
    _exit_before_bind: bool = False,
) -> None:
    """Start the RPC server in the foreground (blocking)."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    from dataclasses import replace as dc_replace

    from recall.core.types import DaemonMode, parse_daemon_mode, parse_source
    from recall.services.rpc_server import start_server_blocking
    from recall.services.watcher import resolve_daemon_mode

    config = AppConfig.load()
    # CLI overrides for daemon config
    daemon_overrides: dict[str, object] = {}
    if embed_override is not None:
        daemon_overrides["embed"] = embed_override
    if source_override is not None:
        daemon_overrides["source"] = parse_source(source_override)
    if daemon_overrides:
        config = dc_replace(config, daemon=dc_replace(config.daemon, **daemon_overrides))
    # CLI --batch-size override for embedding config
    if batch_size_override is not None:
        config = dc_replace(
            config,
            embedding=dc_replace(config.embedding, batch_size=batch_size_override),
        )
    # Before the first record anything here can emit, including the rotation
    # warning below. `-v` overrides the configured level; nothing else may.
    _configure_daemon_logging(
        logging.DEBUG
        if verbose
        else logging.getLevelNamesMapping()[config.daemon.log_level.upper()]
    )
    log_path, err_log_path = _log_paths(config)
    if _fd_is_managed_log(1, log_path):
        _rotate_daemon_log_if_needed(log_path, config.daemon.log_max_bytes, 1)
    if _fd_is_managed_log(2, err_log_path):
        _rotate_daemon_log_if_needed(err_log_path, config.daemon.log_max_bytes, 2)
    if _exit_before_bind:
        return
    # CLI --mode flag overrides config
    effective_daemon_mode = (
        parse_daemon_mode(str(mode_override)) if mode_override else config.daemon.mode
    )
    try:
        resolved_mode = resolve_daemon_mode(effective_daemon_mode)
    except RuntimeError:
        resolved_mode = None
    watch = resolved_mode == DaemonMode.WATCH
    idle_timeout = config.daemon.idle_timeout
    effective_timeout = None if watch else idle_timeout
    from recall.services.self_repair import DaemonStartupRefused

    try:
        start_server_blocking(config=config, idle_timeout=effective_timeout, watch=watch)
    except DaemonStartupRefused as err:
        # REQ-RESIL-016: the same fatal signature recurred after an index
        # rebuild. A non-zero exit keeps the supervisor's restart cheap and the
        # cause named, instead of the silent exit-0 crash loop of the incident.
        logger.error("daemon refused to start: %s", err)
        typer.echo(f"recall daemon refused to start: {err}", err=True)
        raise typer.Exit(code=err.exit_code) from None


def _resolve_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout must be a number")
    timeout = float(value)
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    return timeout


@app.command("start")
def start_command(
    ctx: typer.Context,
    background: bool = typer.Option(
        False,
        "--background",
        help="Legacy direct background process start instead of scheduler start",
    ),
    timeout: float = typer.Option(10.0, "--timeout", help="Seconds to wait for daemon readiness"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
    cta: bool = typer.Option(False, "--cta", help="Include call-to-action suggestions"),
) -> None:
    """Start the installed daemon scheduler."""
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(
            params,
            allowed={"background", "timeout", "verbose", "fields", "format", "json", "cta"},
            types={
                "background": "boolean",
                "timeout": "number",
                "verbose": "boolean",
                "fields": "string",
                "format": "string",
                "json": "boolean",
                "cta": "boolean",
            },
        )
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        background = resolve_bool_param(ctx, payload, "background", background)
        timeout = _resolve_timeout(resolve_param(ctx, payload, "timeout", timeout))
        verbose = resolve_bool_param(ctx, payload, "verbose", verbose)
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)

        config = AppConfig.load()

        if background:
            bg = _start_background(config, verbose)
            # Wait for the daemon subprocess to write its PID file so an
            # immediate "stop" after "start" does not race with NOT_FOUND.
            _wait_for_pid_file(config, timeout=30.0)
            result: dict[str, object] = {
                "scheduler": "pid",
                "started": True,
                "pid": bg.pid,
                "duration_seconds": 0.0,
                "message": f"legacy background start; logs: {bg.log_path}",
            }
        else:
            result = start_daemon_durable(config, timeout=timeout).__dict__
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None

    ctas_list = cta_for_daemon_start() if (cta or output_format == OutputFormat.TEXT) else []

    if output_format != OutputFormat.TEXT:
        try:
            emit_data_with_cta(
                result,
                ctas_list,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("daemon start"),
            )
        except CliError as err:
            emit_error(err, output_format=output_format)
            raise typer.Exit(code=err.exit_code) from None
        if not result.get("started"):
            raise typer.Exit(code=1)
        return

    if result.get("started"):
        pid_text = f" (pid {result['pid']})" if result.get("pid") is not None else ""
        typer.echo(f"Daemon started via {result['scheduler']}{pid_text}.")
        if result.get("message"):
            typer.echo(str(result["message"]))
    else:
        typer.echo(f"Daemon did not start via {result['scheduler']}: {result.get('message')}")
    render_cta_hints(ctas_list)
    if not result.get("started"):
        raise typer.Exit(code=1)


@app.command("stop")
def stop_command(
    ctx: typer.Context,
    soft: bool = typer.Option(
        False,
        "--soft",
        help="Soft stop using the legacy PID-file signal path; schedulers may respawn",
    ),
    timeout: float = typer.Option(10.0, "--timeout", help="Seconds to wait for daemon exit"),
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
    cta: bool = typer.Option(False, "--cta", help="Include call-to-action suggestions"),
) -> None:
    """Stop the running daemon scheduler."""
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(
            params,
            allowed={"soft", "timeout", "fields", "format", "json", "cta"},
            types={
                "soft": "boolean",
                "timeout": "number",
                "fields": "string",
                "format": "string",
                "json": "boolean",
                "cta": "boolean",
            },
        )
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        soft = resolve_bool_param(ctx, payload, "soft", soft)
        timeout = _resolve_timeout(resolve_param(ctx, payload, "timeout", timeout))
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)

        config = AppConfig.load()
        result = (
            stop_daemon_soft(config) if soft else stop_daemon_durable(config, timeout=timeout)
        ).__dict__
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None

    ctas_list = cta_for_daemon_stop() if (cta or output_format == OutputFormat.TEXT) else []

    if output_format != OutputFormat.TEXT:
        try:
            emit_data_with_cta(
                result,
                ctas_list,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("daemon stop"),
            )
        except CliError as err:
            emit_error(err, output_format=output_format)
            raise typer.Exit(code=err.exit_code) from None
        if not result.get("stopped"):
            raise typer.Exit(code=1)
        return

    if result.get("stopped"):
        pid_text = f" (pid {result['pid']})" if result.get("pid") is not None else ""
        typer.echo(f"Daemon stopped via {result['scheduler']}{pid_text}.")
        if result.get("message"):
            typer.echo(str(result["message"]))
    else:
        typer.echo(f"Daemon did not stop via {result['scheduler']}: {result.get('message')}")
    render_cta_hints(ctas_list)
    if not result.get("stopped"):
        raise typer.Exit(code=1)


@app.command("restart")
def restart_command(
    ctx: typer.Context,
    timeout: float = typer.Option(10.0, "--timeout", help="Seconds to wait per lifecycle step"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
    cta: bool = typer.Option(False, "--cta", help="Include call-to-action suggestions"),
) -> None:
    """Restart the installed daemon scheduler."""
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(
            params,
            allowed={"timeout", "verbose", "fields", "format", "json", "cta"},
            types={
                "timeout": "number",
                "verbose": "boolean",
                "fields": "string",
                "format": "string",
                "json": "boolean",
                "cta": "boolean",
            },
        )
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        timeout = _resolve_timeout(resolve_param(ctx, payload, "timeout", timeout))
        verbose = resolve_bool_param(ctx, payload, "verbose", verbose)
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)

        config = AppConfig.load()
        if verbose:
            logging.basicConfig(level=logging.INFO)
        result = restart_daemon_durable(config, timeout=timeout).__dict__
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None

    ctas_list = cta_for_daemon_restart() if (cta or output_format == OutputFormat.TEXT) else []

    if output_format != OutputFormat.TEXT:
        try:
            emit_data_with_cta(
                result,
                ctas_list,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("daemon restart"),
            )
        except CliError as err:
            emit_error(err, output_format=output_format)
            raise typer.Exit(code=err.exit_code) from None
        if not (result.get("stopped") and result.get("started")):
            raise typer.Exit(code=1)
        return

    if result.get("stopped") and result.get("started"):
        pid_text = f" (pid {result['pid']})" if result.get("pid") is not None else ""
        typer.echo(f"Daemon restarted via {result['scheduler']}{pid_text}.")
        if result.get("message"):
            typer.echo(str(result["message"]))
    else:
        typer.echo(f"Daemon did not restart via {result['scheduler']}: {result.get('message')}")
    render_cta_hints(ctas_list)
    if not (result.get("stopped") and result.get("started")):
        raise typer.Exit(code=1)


@dataclass(frozen=True)
class _BackgroundResult:
    pid: int
    log_path: str


def _start_background(config: AppConfig, verbose: bool) -> _BackgroundResult:
    from recall.core.config import resolve_recall_binary

    try:
        recall_bin = resolve_recall_binary()
    except RuntimeError as err:
        raise CliError(
            code=ErrorCode.RUNTIME,
            message="cannot find recall binary for background start",
            exit_code=1,
        ) from err

    create_private_dir(config.data_dir)
    log_dir = config.data_dir / "logs"
    create_private_dir(log_dir)
    stdout_log = log_dir / "daemon.log"
    stderr_log = log_dir / "daemon.err.log"

    cmd = [recall_bin, "daemon"]
    if verbose:
        cmd.append("--verbose")

    with (
        open(stdout_log, "a") as stdout_f,
        open(stderr_log, "a") as stderr_f,
    ):
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=stdout_f,
            stderr=stderr_f,
            start_new_session=True,
        )
    return _BackgroundResult(pid=proc.pid, log_path=str(stdout_log))


def _stop_running_daemon(config: AppConfig) -> int | None:
    """Stop a running daemon and return its PID, or None if not running.

    Returns None when no PID file exists (daemon not running).
    Raises CliError if the PID file is unreadable or the process cannot
    be signaled. ProcessLookupError (already dead) is treated as a
    successful stop — the daemon is gone either way.
    """
    import time

    pid_path = config.data_dir / "recall.pid"

    if not pid_path.exists():
        return None

    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError) as err:
        raise CliError(
            code=ErrorCode.RUNTIME,
            message=f"cannot read PID file: {err}",
            exit_code=1,
        ) from err

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        # Process already gone — still a successful stop
        _cleanup_daemon_files(config)
        return pid
    except PermissionError as err:
        raise CliError(
            code=ErrorCode.RUNTIME,
            message=f"cannot signal daemon: {err}",
            exit_code=1,
        ) from err

    # Wait for process to exit
    for _ in range(50):
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
    _cleanup_daemon_files(config)
    return pid


def _wait_for_pid_file(config: AppConfig, *, timeout: float = 5.0) -> None:
    """Wait for the daemon subprocess to write its PID file.

    The background daemon creates recall.pid once the RPC server is listening.
    Without this wait, a start-then-stop sequence would race.
    Raises CliError if the PID file does not appear within the timeout.
    """
    import time

    pid_path = config.data_dir / "recall.pid"
    waited = 0.0
    while waited < timeout:
        if pid_path.exists():
            return
        time.sleep(0.1)
        waited += 0.1
    raise CliError(
        code=ErrorCode.RUNTIME,
        message=f"daemon did not create PID file within {timeout}s",
        exit_code=1,
    )


def _cleanup_daemon_files(config: AppConfig) -> None:
    socket_path = config.data_dir / "recall.sock"
    pid_path = config.data_dir / "recall.pid"
    socket_path.unlink(missing_ok=True)
    pid_path.unlink(missing_ok=True)


@app.command("install")
def install_command(
    ctx: typer.Context,
    scheduler: str | None = typer.Option(
        None,
        "--scheduler",
        help="Scheduler type: auto, launchd, systemd, cron",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and print the resolved request",
    ),
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
    cta: bool = typer.Option(False, "--cta", help="Include call-to-action suggestions"),
) -> None:
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(
            params,
            allowed={"scheduler", "dry_run", "fields", "format", "json", "cta"},
            types={
                "scheduler": "string",
                "dry_run": "boolean",
                "fields": "string",
                "format": "string",
                "json": "boolean",
                "cta": "boolean",
            },
        )
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        scheduler = resolve_param(ctx, payload, "scheduler", scheduler)
        dry_run = resolve_bool_param(ctx, payload, "dry_run", dry_run)
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)
        requested = parse_scheduler_kind(scheduler) if scheduler else None
        if dry_run:
            render_dry_run(
                command="daemon install",
                request={"scheduler": requested.value if requested is not None else None},
                safety=COMMANDS["daemon install"]["safety"],
                output_format=output_format,
                fields=fields_value,
                ctas=(
                    cta_for_dry_run("daemon install")
                    if (cta or output_format == OutputFormat.TEXT)
                    else None
                ),
                include_cta=cta,
            )
            return
        status = install_scheduler(scheduler=requested)
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None
    except RuntimeError as err:
        emit_error(
            CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=1),
            output_format=output_format,
        )
        raise typer.Exit(code=1) from None
    except Exception as err:
        emit_error(
            CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=1),
            output_format=output_format,
        )
        raise typer.Exit(code=1) from None

    ctas = cta_for_daemon_install() if (cta or output_format == OutputFormat.TEXT) else []

    if output_format != OutputFormat.TEXT:
        try:
            emit_data_with_cta(
                status,
                ctas,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("daemon install"),
            )
        except CliError as err:
            emit_error(err, output_format=output_format)
            raise typer.Exit(code=err.exit_code) from None
        return

    scheduler_name = status.scheduler.value if status.scheduler is not None else "unknown"
    typer.echo(f"Installed recall daemon via {scheduler_name}.")
    typer.echo(f"Command: {status.command}")
    render_cta_hints(ctas)


@app.command("uninstall")
def uninstall_command(
    ctx: typer.Context,
    yes: bool = typer.Option(False, "--yes", help="Confirm scheduler removal"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and print the resolved request",
    ),
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
    cta: bool = typer.Option(False, "--cta", help="Include call-to-action suggestions"),
) -> None:
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(
            params,
            allowed={"yes", "dry_run", "fields", "format", "json", "cta"},
            types={
                "yes": "boolean",
                "dry_run": "boolean",
                "fields": "string",
                "format": "string",
                "json": "boolean",
                "cta": "boolean",
            },
        )
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        yes = resolve_bool_param(ctx, payload, "yes", yes)
        dry_run = resolve_bool_param(ctx, payload, "dry_run", dry_run)
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)
        require_confirmation(
            should_confirm=True,
            confirmed=yes,
            dry_run=dry_run,
            message="`recall daemon uninstall` removes scheduler artifacts.",
        )
        if dry_run:
            render_dry_run(
                command="daemon uninstall",
                request={"yes": yes},
                safety=COMMANDS["daemon uninstall"]["safety"],
                output_format=output_format,
                fields=fields_value,
                ctas=(
                    cta_for_dry_run("daemon uninstall --yes")
                    if (cta or output_format == OutputFormat.TEXT)
                    else None
                ),
                include_cta=cta,
            )
            return
        status = uninstall_scheduler()
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None
    except RuntimeError as err:
        emit_error(
            CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=1),
            output_format=output_format,
        )
        raise typer.Exit(code=1) from None
    except Exception as err:
        emit_error(
            CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=1),
            output_format=output_format,
        )
        raise typer.Exit(code=1) from None

    ctas = cta_for_daemon_uninstall() if (cta or output_format == OutputFormat.TEXT) else []

    if output_format != OutputFormat.TEXT:
        try:
            emit_data_with_cta(
                status,
                ctas,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("daemon uninstall"),
            )
        except CliError as err:
            emit_error(err, output_format=output_format)
            raise typer.Exit(code=err.exit_code) from None
        return

    typer.echo("Removed recall daemon scheduler artifacts.")
    render_cta_hints(ctas)


# Source coverage rows a status read requests when the caller names no page.
# A default read is a health check, not a catalog dump: 100 rows rendered 2,501
# of the 2,719 default output lines. `--limit`/`--cursor` page the
# rest, and the page carries `next_cursor` either way.
DEFAULT_SOURCE_PAGE_LIMIT = 10


@app.command("status")
def status_command(
    ctx: typer.Context,
    limit: int = typer.Option(
        DEFAULT_SOURCE_PAGE_LIMIT,
        "--limit",
        help="Maximum source coverage rows per page (1..256)",
    ),
    cursor: str | None = typer.Option(None, "--cursor", help="Continue source coverage"),
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    cta: bool = typer.Option(False, "--cta", help="Include call-to-action suggestions"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
) -> None:
    """Show daemon scheduler and runtime status.

    The reconciliation block is the indexing backlog: `pending` counts
    catalogued transcript files whose committed index is behind the file on
    disk, and it drains as the daemon commits that work. Read the whole object
    with `--json --limit 1 --fields reconciliation`. See
    docs/reconciliation-operations.md for the field meanings, the expected drain
    and what to do when the count stops moving.
    """
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(
            params,
            allowed={"fields", "format", "json", "cta", "limit", "cursor"},
            types={
                "limit": "integer",
                "cursor": "string",
                "fields": "string",
                "format": "string",
                "json": "boolean",
                "cta": "boolean",
            },
        )
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)
        limit = resolve_int_param(ctx, payload, "limit", limit)
        if not 1 <= limit <= 256:
            raise ValueError("limit must be 1..256")
        cursor = resolve_param(ctx, payload, "cursor", cursor)
        rpc_params: dict[str, object] = {"limit": limit}
        if cursor:
            rpc_params["cursor"] = cursor
        # A caller who named a page gets that page or an error: the local
        # fallback below answers a different one (REQ-LIVE-009). The default
        # page is not such a request, so the daemon-down read still works.
        page_requested = limit != DEFAULT_SOURCE_PAGE_LIMIT or cursor is not None
        # Try RPC first for live embed state, fall back to local
        try:
            status = rpc_call_or_error("recall.daemon_status", rpc_params, auto_fork=False)
        except CliError:
            if page_requested:
                raise
            status = daemon_status()
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None
    except RuntimeError as err:
        emit_error(
            CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=1),
            output_format=output_format,
        )
        raise typer.Exit(code=1) from None
    except Exception as err:
        emit_error(
            CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=1),
            output_format=output_format,
        )
        raise typer.Exit(code=1) from None

    # Normalize to dict — status may be a dataclass (local) or dict (RPC)
    if not isinstance(status, dict):
        from dataclasses import asdict

        status_dict: dict[str, Any] = asdict(status)
    else:
        status_dict = dict(status)

    status_dict.setdefault(
        "reconciliation",
        {
            "rpc_ready": False,
            "catalog_scan_complete": False,
            "live_observation_ready": False,
            "raw_indexing_ready": False,
            "keyword_search_ready": False,
            "enrichment_ready": False,
            "coverage": None,
            "source_page": [],
            "next_cursor": None,
            "error": "runtime coverage unavailable; daemon RPC was not reached",
        },
    )
    ctas_list = (
        cta_for_daemon_status(status_dict) if (cta or output_format == OutputFormat.TEXT) else []
    )

    if output_format != OutputFormat.TEXT:
        # Warnings-only health notice on stderr (REQ-CLI-021/022, REQ-CLI-012) so
        # agents using --json still get version-drift/bloat warnings without
        # polluting the structured payload; reuse the already-fetched status_dict
        # rather than issuing a second daemon_status RPC.
        health_notice = _render_status_notice_from_status(
            status_dict,
            interval_seconds=AppConfig.load().daemon.interval,
            include_informational=False,
        )
        if health_notice:
            typer.echo(health_notice, err=True)
        try:
            emit_data_with_cta(
                status_dict,
                ctas_list,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("daemon status"),
            )
        except CliError as err:
            emit_error(err, output_format=output_format)
            raise typer.Exit(code=err.exit_code) from None
        return

    _print_daemon_status_text(status_dict)
    render_cta_hints(ctas_list)


def _set_reconciliation_pause(
    ctx: typer.Context,
    *,
    paused: bool,
    format_name: str | None,
    fields: str | None,
    json_output: bool,
    params: str | None,
) -> None:
    """Apply the durable maintenance switch through the running daemon."""
    output_format = resolve_output_format_early(
        ctx, json_output=json_output, format_name=format_name, raw_params=params
    )
    try:
        payload = parse_params(
            params,
            allowed={"fields", "format", "json"},
            types={"fields": "string", "format": "string", "json": "boolean"},
        )
        output_format = resolve_output_format_from_params(
            ctx, payload, json_output=json_output, format_name=format_name
        )
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)
        result = rpc_call_or_error(
            "recall.daemon_pause" if paused else "recall.daemon_resume", {}, auto_fork=False
        )
        if not isinstance(result, dict) or result.get("paused") is not paused:
            raise CliError(
                code=ErrorCode.RUNTIME,
                message="daemon did not confirm the requested maintenance state",
            )
        emit_data_with_cta(
            result,
            [],
            output_format=output_format,
            include_cta=False,
            fields=fields_value,
            allowed_fields=output_fields_for("daemon pause"),
        )
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None


@app.command("pause")
def pause_command(
    ctx: typer.Context,
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
) -> None:
    """Pause reconciliation durably; later client auto-starts retain this state."""
    _set_reconciliation_pause(
        ctx,
        paused=True,
        format_name=format_name,
        fields=fields,
        json_output=json_output,
        params=params,
    )


@app.command("migrate-storage")
def migrate_storage_command(
    ctx: typer.Context,
    dry_run: bool = typer.Option(False, "--dry-run", help="Inspect the plan without applying it"),
    plan_id: str | None = typer.Option(None, "--plan-id", help="Require the inspected plan digest"),
    wait: bool = typer.Option(False, "--wait", help="Wait for the accepted operation to finish"),
    timeout: float = typer.Option(5.0, "--timeout", help="Overall wait deadline in seconds"),
    format_name: str | None = typer.Option(None, "--format", help="Output format"),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
) -> None:
    """Inspect or request backed-up fixed-format maintenance in the running daemon."""
    output_format = resolve_output_format_early(
        ctx, json_output=json_output, format_name=format_name, raw_params=params
    )
    try:
        payload = parse_params(
            params,
            allowed={"dry_run", "plan_id", "wait", "timeout", "fields", "format", "json"},
            types={
                "dry_run": "boolean",
                "plan_id": "string",
                "wait": "boolean",
                "timeout": "number",
                "fields": "string",
                "format": "string",
                "json": "boolean",
            },
        )
        output_format = resolve_output_format_from_params(
            ctx, payload, json_output=json_output, format_name=format_name
        )
        dry_run = resolve_bool_param(ctx, payload, "dry_run", dry_run)
        wait = resolve_bool_param(ctx, payload, "wait", wait)
        plan_id = resolve_param(ctx, payload, "plan_id", plan_id)
        timeout = float(resolve_param(ctx, payload, "timeout", timeout))
        if not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError("timeout must be finite and in (0, 3600] seconds")
        if dry_run and wait:
            raise ValueError("--wait requires apply; omit --dry-run")
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)
        deadline = time.monotonic() + timeout
        request: dict[str, Any] = {"dry_run": dry_run}
        if plan_id is not None:
            request["plan_id"] = plan_id
        result = rpc_call_or_error(
            "recall.migrate_storage",
            request,
            auto_fork=False,
            idle_timeout=min(5.0, timeout) if wait else 5.0,
        )
        result = _validate_storage_response(result)
        if wait:
            result = _wait_for_storage_maintenance(result, deadline=deadline)
        emit_data_with_cta(
            result,
            [],
            output_format=output_format,
            include_cta=False,
            fields=fields_value,
            allowed_fields=output_fields_for("daemon migrate-storage"),
        )
        if result.get("status") in {"failed", "timed_out"}:
            raise typer.Exit(code=1)
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None


def _validate_storage_response(result: Any) -> dict[str, Any]:
    states = {"planned", "accepted", "running", "succeeded", "unchanged", "failed"}
    if (
        not isinstance(result, dict)
        or result.get("schema_version") != 1
        or not isinstance(result.get("operation_id"), str)
        or result.get("status") not in states
        or not isinstance(result.get("accepted"), bool)
    ):
        raise CliError(code=ErrorCode.RUNTIME, message="invalid storage maintenance response")
    return result


def _wait_for_storage_maintenance(result: Any, *, deadline: float) -> dict[str, Any]:
    """Poll the singleton operation; timeout stops waiting, not daemon execution."""
    operation_id = result.get("operation_id") if isinstance(result, dict) else None
    if not isinstance(operation_id, str):
        raise CliError(code=ErrorCode.RUNTIME, message="missing storage operation identity")
    while True:
        result = _validate_storage_response(result)
        if result.get("operation_id") != operation_id:
            raise CliError(code=ErrorCode.RUNTIME, message="storage operation identity changed")
        state = result.get("status")
        if state in {"succeeded", "unchanged", "failed"}:
            return result
        if state not in {"accepted", "running"}:
            raise CliError(code=ErrorCode.RUNTIME, message="unexpected storage operation state")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {
                **result,
                "status": "timed_out",
                "error": "wait deadline expired; inspect the operation before retrying",
            }
        time.sleep(min(0.1, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            continue
        result = rpc_call_or_error(
            "recall.migrate_storage",
            {"dry_run": True},
            auto_fork=False,
            idle_timeout=min(5.0, remaining),
        )


@app.command("resume")
def resume_command(
    ctx: typer.Context,
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
) -> None:
    """Resume reconciliation after a durable maintenance pause."""
    _set_reconciliation_pause(
        ctx,
        paused=False,
        format_name=format_name,
        fields=fields,
        json_output=json_output,
        params=params,
    )


def _print_index_health_text(s: dict[str, Any], runtime: dict[str, Any]) -> None:
    """Failure-signature memory and the startup index probe (REQ-RESIL-014/018)."""
    if runtime.get("last_fatal_signature") is not None:
        repeat = runtime.get("fatal_repeat_count") or 0
        # Undated, a signature from a repaired failure reads as a live one.
        at = runtime.get("last_fatal_at")
        when = f" at {at}" if at else " at an unrecorded time"
        typer.echo(f"Last fatal: {runtime['last_fatal_signature']} (x{repeat}){when}")
    if runtime.get("last_index_repair_at") is not None:
        typer.echo(f"Index repair: {runtime['last_index_repair_at']}")
    if runtime.get("needs_index_verification"):
        typer.echo("Index verification: needs verification at next daemon start (disk-full seen)")
    refusal = s.get("startup_refusal")
    if isinstance(refusal, str) and refusal:
        typer.echo(f"Startup refused: {refusal}")
    divergence = s.get("index_divergence")
    if not isinstance(divergence, dict):
        return
    diverged = divergence.get("diverged")
    diverged_count = divergence.get("diverged_count")
    if not isinstance(diverged_count, int):
        diverged_count = len(diverged) if isinstance(diverged, list) else 0
    line = (
        f"Index divergence: {diverged_count} diverged, "
        f"{divergence.get('samples_checked', 0)} key(s) checked, "
        f"{divergence.get('samples_unverifiable', 0)} unverifiable"
    )
    if not divergence.get("complete", True):
        line += " (probe incomplete)"
    if divergence.get("checked_at"):
        line += f" at {divergence['checked_at']}"
    typer.echo(line)


def _format_age(seconds: float) -> str:
    """Render an elapsed span in the coarsest unit that still reads at a glance."""
    span = max(0.0, seconds)
    if span < 60.0:
        return f"{span:.0f}s"
    if span < 3600.0:
        return f"{span / 60.0:.0f}m"
    if span < 86400.0:
        return f"{span / 3600.0:.0f}h"
    return f"{span / 86400.0:.0f}d"


def _format_epoch(value: object) -> str | None:
    """Local ISO-8601 plus age for an epoch-seconds field, or None when unset.

    Runtime timestamps cross the RPC boundary as epoch floats while their
    datetime siblings arrive as ISO-8601 strings; text output states both the
    same way.
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    moment = float(value)
    stamp = datetime.fromtimestamp(moment).isoformat(timespec="seconds")
    return f"{stamp} ({_format_age(time.time() - moment)} ago)"


def _version_drift_text(s: dict[str, Any]) -> str:
    """Drift is unknowable with no daemon version to compare the binary against."""
    if s.get("daemon_version") is None:
        return "n/a"
    return "yes" if s.get("version_drift") else "no"


def _print_status_summary_text(s: dict[str, Any]) -> None:
    """Lead the report with the fields an operator reads first.

    The detail below answers 85 keys; these four lines answer "is it running,
    is it draining, is it healthy" without scrolling. The health advisory
    itself stays on stderr per REQ-CLI-012 — this states the facts it derives
    from.
    """
    daemon_version = s.get("daemon_version")
    # With no daemon version in hand, naming the binary here would read as the
    # version that is running; the summary says it is unknown instead.
    version = (
        str(daemon_version)
        if daemon_version is not None
        else f"unknown (binary {s.get('binary_version') or 'unknown'})"
    )
    mode = s.get("resolved_mode") or s.get("mode") or "unknown"
    if hasattr(mode, "value"):
        mode = mode.value
    scheduler = s.get("scheduler") or "none"
    if hasattr(scheduler, "value"):
        scheduler = scheduler.value
    header = f"Daemon: version={version} mode={mode} scheduler={scheduler}"
    pid = s.get("daemon_pid")
    if pid is not None:
        header += f" pid={pid}"
    typer.echo(header)

    embed_line = f"  embed: pending={s.get('embed_pending', 0)}"
    outcome = s.get("embed_loop_last_outcome")
    if outcome:
        embed_line += f" outcome={outcome}"
    last_batch = _format_epoch(s.get("embed_last_batch_at"))
    embed_line += f" last_batch={last_batch if last_batch else 'never'}"
    reconciliation = s.get("reconciliation")
    reconciliation = reconciliation if isinstance(reconciliation, dict) else {}
    deferred = reconciliation.get("enrichment_deferred")
    if deferred:
        embed_line += f" deferred={deferred}"
    if s.get("embed_last_error"):
        embed_line += f" error={s['embed_last_error']}"
    typer.echo(embed_line)

    pending = reconciliation.get("pending")
    if isinstance(pending, int):
        typer.echo(
            f"  reconciliation: pending={pending} "
            f"catalog_scan_complete={_status_bool(reconciliation.get('catalog_scan_complete'))} "
            f"keyword_search_ready={_status_bool(reconciliation.get('keyword_search_ready'))}"
            f"{_index_only_text(reconciliation.get('index_only_sessions'))}"
        )
    else:
        reason = reconciliation.get("error") or "daemon RPC did not report coverage"
        typer.echo(f"  reconciliation: coverage unavailable ({reason})")

    health_line = f"  health: version_drift={_version_drift_text(s)}"
    bloat = s.get("bloat_ratio")
    threshold = s.get("bloat_ratio_threshold")
    if isinstance(bloat, int | float) and isinstance(threshold, int | float):
        health_line += f" bloat={bloat:.2f}/{threshold:.2f}"
    if s.get("startup_refusal"):
        health_line += " startup_refused=yes"
    typer.echo(health_line)
    typer.echo("")


def _print_daemon_status_text(s: dict[str, Any]) -> None:
    """Render daemon status as human-readable text."""
    _print_status_summary_text(s)
    reconciliation = s.get("reconciliation")
    if isinstance(reconciliation, dict):
        flags = (
            "rpc_ready",
            "catalog_scan_complete",
            "live_observation_ready",
            "raw_indexing_ready",
            "keyword_search_ready",
            "enrichment_ready",
        )
        typer.echo(
            "Reconciliation: "
            + ", ".join(f"{flag}={reconciliation.get(flag, 'unknown')}" for flag in flags)
        )
        if reconciliation.get("error"):
            typer.echo(f"Reconciliation error: {reconciliation['error']}")
        migration = reconciliation.get("index_migration")
        if isinstance(migration, dict) and migration.get("phase") not in (None, "idle"):
            typer.echo(
                "Index migration: "
                f"phase={migration.get('phase')} "
                f"{migration.get('completed', 0)}/{migration.get('captured', 0)} "
                f"applied={migration.get('applied_version')} "
                f"target={migration.get('target_version')}"
            )
        if reconciliation.get("next_cursor"):
            typer.echo(f"More source coverage: --cursor {reconciliation['next_cursor']}")
    configured = s.get("configured_scheduler", "unknown")
    actual = s.get("scheduler") or "none"
    # Enum values may already be strings (from RPC) or enum objects (from local)
    if hasattr(configured, "value"):
        configured = configured.value
    if hasattr(actual, "value"):
        actual = actual.value
    mode = s.get("mode", "unknown")
    if hasattr(mode, "value"):
        mode = mode.value
    resolved_mode = s.get("resolved_mode") or "unknown"
    if hasattr(resolved_mode, "value"):
        resolved_mode = resolved_mode.value
    installed_mode = s.get("installed_mode")
    if hasattr(installed_mode, "value"):
        installed_mode = installed_mode.value

    mode_line = f"Mode: {mode} (resolved: {resolved_mode})"
    if s.get("mode_mismatch_reason") and installed_mode is not None:
        mode_line += f" [installed: {installed_mode}]"
    typer.echo(mode_line)
    binary_version = s.get("binary_version") or "unknown"
    daemon_version = s.get("daemon_version")
    if daemon_version is None:
        version_drift = "n/a"
        daemon_version_text = "unknown"
    else:
        version_drift = "yes" if s.get("version_drift") else "no"
        daemon_version_text = str(daemon_version)
    typer.echo(
        f"Version: binary={binary_version} daemon={daemon_version_text} drift={version_drift}"
    )
    if s.get("version_drift") and daemon_version is not None:
        typer.echo(
            "Binary upgraded since daemon started — run `recall daemon restart` "
            "to load the new version."
        )
    if s.get("runtime_unavailable_reason"):
        # REQ-DAEMON-074: the local fallback found the database held, so the
        # runtime fields below are defaults, not readings.
        pid = s.get("daemon_pid")
        if pid is not None:
            typer.echo(
                f"Daemon: pid {pid} alive, RPC not answering (starting or busy); "
                "runtime fields unavailable until it answers"
            )
        else:
            typer.echo(f"Database: {s['runtime_unavailable_reason']}")
    # A served status already carried the pid into the summary header; only the
    # local fallback above has something to add (that the RPC is not answering).
    typer.echo(f"Configured scheduler: {configured}")
    typer.echo(f"Installed: {'yes' if s.get('installed') else 'no'} ({actual})")
    watched_dirs = s.get("watched_dirs", ())
    if watched_dirs:
        typer.echo(f"Watched dirs: {', '.join(watched_dirs)}")
    typer.echo(f"Debounce: {s.get('debounce', '?')}s (FTS: {s.get('fts_debounce', '?')}s)")
    typer.echo(f"Command: {s.get('command', '?')}")
    typer.echo(f"Installed binary: {s.get('installed_binary_path') or 'unknown'}")
    typer.echo(f"Binary drift: {'yes' if s.get('installed_binary_stale') else 'no'}")
    scheduler_health = s.get("scheduler_health_state") or "unknown"
    scheduler_exit = s.get("scheduler_last_exit_status")
    typer.echo(
        "Scheduler health: "
        f"state={scheduler_health} "
        f"last_exit={scheduler_exit if scheduler_exit is not None else 'n/a'}"
    )
    typer.echo(f"Config path: {s.get('config_path', '?')}")
    artifact_paths = s.get("artifact_paths", ())
    if artifact_paths:
        typer.echo("Artifacts:")
        for path in artifact_paths:
            typer.echo(f"  {path}")

    runtime = s.get("runtime_status", {})
    if runtime is None:
        runtime = {}
    # runtime may be a dataclass or dict
    if not isinstance(runtime, dict):
        from dataclasses import asdict

        runtime = asdict(runtime)
    last_kind = runtime.get("last_run_kind") or "unknown"
    if hasattr(last_kind, "value"):
        last_kind = last_kind.value
    typer.echo(
        "Last run: "
        f"attempted={runtime.get('last_attempted_at') or 'never'} "
        f"successful={runtime.get('last_successful_at') or 'never'} "
        f"kind={last_kind}"
    )
    idx_summary = runtime.get("last_index_summary")
    if idx_summary is not None:
        if not isinstance(idx_summary, dict):
            from dataclasses import asdict

            idx_summary = asdict(idx_summary)
        summary_line = (
            "Index summary: "
            f"total={idx_summary.get('total', 0)} "
            f"changed={idx_summary.get('changed', 0)} "
            f"indexed={idx_summary.get('indexed', 0)} "
            f"skipped={idx_summary.get('skipped', 0)} "
            f"failed={idx_summary.get('failed', 0)}"
        )
        total_secs = idx_summary.get("total_seconds")
        if total_secs is not None:
            summary_line += f" duration={total_secs:.2f}s"
        typer.echo(summary_line)
    if runtime.get("last_failure_message") is not None:
        typer.echo(f"Last failure: {runtime['last_failure_message']}")
    _print_index_health_text(s, runtime)

    embed_enabled = s.get("embed_phase_enabled", False)
    embed_status = "enabled" if embed_enabled else "disabled"
    model_status = "loaded" if s.get("embed_model_loaded") else "not loaded"
    embed_pending = s.get("embed_pending", 0)
    embed_line = f"Embed phase: {embed_status} (model: {model_status}, pending: {embed_pending}"
    # The pending count is served from the last snapshot the phase took, so a
    # deferred cycle freezes it; its age says whether it still describes the
    # backlog (REQ-ADAPT-012).
    pending_at = s.get("embed_pending_at")
    if pending_at:
        embed_line += f", measured {max(0.0, time.time() - pending_at):.0f}s ago"
    embed_line += ")"
    last_batch_at = _format_epoch(s.get("embed_last_batch_at"))
    last_batch_size = s.get("embed_last_batch_size", 0)
    if last_batch_at is not None:
        last_batch_duration = s.get("embed_last_batch_duration", 0.0)
        embed_line += f" last_batch={last_batch_at} items={last_batch_size}"
        if last_batch_duration:
            embed_line += f" duration={last_batch_duration:.1f}s"
    typer.echo(embed_line)

    stage = s.get("embed_loop_stage")
    if stage:
        stage_at = s.get("embed_loop_stage_at")
        # Seconds, not a coarsened age: stall detection reads this field.
        stage_age = f"{max(0.0, time.time() - stage_at):.0f}s" if stage_at else "?"
        last_iteration_at = _format_epoch(s.get("embed_loop_last_iteration_at"))
        typer.echo(
            f"Embed loop: iteration={s.get('embed_loop_iterations', 0)} stage={stage}"
            f" stage_age={stage_age} trigger={s.get('embed_loop_last_trigger') or 'none'}"
            f" last_iteration={last_iteration_at if last_iteration_at else 'never'}"
            f" outcome={s.get('embed_loop_last_outcome')}"
            f" next_interval={s.get('embed_loop_next_interval', 0.0)}s"
        )
    requested_stage = s.get("embed_requested_stage")
    if requested_stage:
        requested_stage_at = s.get("embed_requested_stage_at")
        requested_age = (
            f"{max(0.0, time.time() - requested_stage_at):.0f}s" if requested_stage_at else "?"
        )
        last_requested_at = _format_epoch(s.get("embed_requested_at"))
        typer.echo(
            f"Embed requested: cycles={s.get('embed_requested_cycles', 0)}"
            f" stage={requested_stage} stage_age={requested_age}"
            f" last_cycle={last_requested_at if last_requested_at else 'never'}"
        )
    if s.get("embed_deferred_reason"):
        deferred_line = f"Embed deferred: {s['embed_deferred_reason']}"
        # The reason names a power-dependent threshold, so it is only checkable
        # against the host beside the time it was evaluated (REQ-ADAPT-006).
        deferred_at = _format_epoch(s.get("embed_deferred_at"))
        if deferred_at is not None:
            deferred_line += f" (evaluated {deferred_at})"
        typer.echo(deferred_line)
    if s.get("embed_last_error"):
        typer.echo(f"Embed last error: {s['embed_last_error']}")
    cooldown_sessions = s.get("embed_cooldown_sessions", 0)
    if cooldown_sessions:
        cooldown_until = s.get("embed_cooldown_until")
        until_text = (
            datetime.fromtimestamp(cooldown_until).isoformat(timespec="seconds")
            if cooldown_until
            else "unknown"
        )
        typer.echo(f"Embed cooldown: {cooldown_sessions} session(s) skipped until {until_text}")

    # Watch index metrics — only shown when the daemon has indexed sessions
    watch_total = s.get("watch_total_indexed", 0)
    watch_failed = s.get("watch_total_failed", 0)
    if watch_total > 0 or watch_failed > 0:
        watch_avg = s.get("watch_avg_duration")
        watch_min = s.get("watch_min_duration")
        watch_max = s.get("watch_max_duration")
        watch_line = f"Watch index: {watch_total} indexed, {watch_failed} failed"
        if watch_avg is not None:
            watch_line += f" (avg={watch_avg:.3f}s"
            if watch_min is not None:
                watch_line += f", min={watch_min:.3f}s"
            if watch_max is not None:
                watch_line += f", max={watch_max:.3f}s"
            watch_line += ")"
        last_dur = s.get("watch_last_event_duration")
        if last_dur is not None:
            watch_line += f" last={last_dur:.3f}s"
        typer.echo(watch_line)

    fts_failures = s.get("fts_rebuild_consecutive_failures", 0)
    if fts_failures > 0:
        last_at = s.get("last_fts_rebuild_failure_at") or "?"
        next_retry = s.get("fts_rebuild_next_retry_at") or "?"
        reason = s.get("last_fts_rebuild_failure_reason") or "?"
        typer.echo(
            f"FTS rebuild: {fts_failures} consecutive OOM failure(s); "
            f"last at {last_at}, next retry at {next_retry}"
        )
        typer.echo(f"  reason: {reason}")

    if s.get("fts_sidecar_enabled", False):
        sidecar_last_run_at = s.get("fts_sidecar_last_run_at") or "(not yet run)"
        typer.echo("FTS sidecar: enabled=true")
        typer.echo(
            "  bootstrap: "
            f"messages={s.get('fts_sidecar_bootstrap_messages_processed', 0)} "
            f"done={_status_bool(s.get('fts_sidecar_bootstrap_messages_done', False))} | "
            f"tool_calls={s.get('fts_sidecar_bootstrap_tool_calls_processed', 0)} "
            f"done={_status_bool(s.get('fts_sidecar_bootstrap_tool_calls_done', False))}"
        )
        typer.echo(
            f"  reconcile (last run {sidecar_last_run_at}): "
            f"drained={_sidecar_kind_pair(s, 'fts_sidecar_reconcile_pending_drained')}, "
            f"backfilled={_sidecar_kind_pair(s, 'fts_sidecar_reconcile_orphans_backfilled')}, "
            f"ghosts={_sidecar_kind_pair(s, 'fts_sidecar_reconcile_ghosts_deleted')}, "
            f"pending={_sidecar_kind_pair(s, 'fts_sidecar_reconcile_pending_remaining')}"
        )
        typer.echo(f"  error: {s.get('fts_sidecar_error') or '—'}")
    else:
        typer.echo("FTS sidecar: disabled (backend=duckdb)")

    mode_value = s.get("resolved_mode")
    if hasattr(mode_value, "value"):
        mode_value = mode_value.value
    has_live_data = (
        int(s.get("live_session_count") or 0) > 0 or s.get("discovery_last_run_at") is not None
    )
    if mode_value == "watch" or has_live_data:
        live_session_count = s.get("live_session_count", 0)
        watcher_subscription_count = s.get("watcher_subscription_count", 0)
        typer.echo(f"Live sessions: {live_session_count} (subs={watcher_subscription_count})")
        discovery_last_run_at = s.get("discovery_last_run_at") or "never"
        typer.echo(
            "Discovery: "
            f"interval={s.get('discovery_interval_seconds', 0.0)}s "
            f"last_run={discovery_last_run_at} "
            f"promoted={s.get('discovery_last_promoted', 0)} "
            f"demoted={s.get('discovery_last_demoted', 0)}"
        )
        if s.get("catchup_in_progress") or s.get("catchup_total") or s.get("catchup_done"):
            state = "running" if s.get("catchup_in_progress") else "idle"
            typer.echo(
                "Catch-up: "
                f"{state} done={s.get('catchup_done', 0)} "
                f"total={s.get('catchup_total', 0)}"
            )
        typer.echo(
            "Fresh reads: "
            f"requests={s.get('live_fresh_requests', 0)} "
            f"timeouts={s.get('live_fresh_timeouts', 0)} "
            f"followers={s.get('follow_subscriptions', 0)}"
        )
        live_session_paths = s.get("live_session_paths", ())
        if live_session_count > 0 and live_session_paths:
            typer.echo("Live paths:")
            for path in live_session_paths:
                typer.echo(f"  {path}")

    notice = _render_status_notice_from_status(s, interval_seconds=AppConfig.load().daemon.interval)
    if notice:
        # Human-only advisory belongs on stderr (REQ-CLI-012), keeping stdout for
        # the status report itself and matching search/list/stats.
        typer.echo(f"Notice: {notice}", err=True)


def _index_only_text(counts: object) -> str:
    """Render sessions surviving only in the index; empty when none or unreported."""
    if not isinstance(counts, dict):
        return ""
    total = sum(value for value in counts.values() if isinstance(value, int))
    return f" index_only_sessions={total}" if total else ""


def _status_bool(value: object) -> str:
    return "true" if bool(value) else "false"


def _sidecar_kind_pair(status: dict[str, Any], prefix: str) -> str:
    messages = int(status.get(f"{prefix}_messages") or 0)
    tool_calls = int(status.get(f"{prefix}_tool_calls") or 0)
    return f"messages={messages}, tool_calls={tool_calls}"
