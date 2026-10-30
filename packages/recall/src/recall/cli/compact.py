from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path

import duckdb
import typer

from recall.cli.contract import (
    CliError,
    ErrorCode,
    OutputFormat,
    emit_data_with_cta,
    emit_error,
    parse_fields,
    parse_params,
    require_confirmation,
    resolve_bool_param,
    resolve_output_format_early,
    resolve_output_format_from_params,
    resolve_param,
    validate_structured_fields,
)
from recall.cli.daemon import _start_background, _wait_for_pid_file
from recall.cli.manifest import output_fields_for
from recall.core.config import AppConfig
from recall.core.schema_version import SchemaTooNewError
from recall.core.types import SchedulerKind
from recall.db import RecallLockError, advisory_lock, is_lock_conflict
from recall.db.connection import check_database_schema
from recall.services.compaction import (
    BloatStats,
    CompactionError,
    _compaction_sentinel,
    compact,
    estimate_bloat_ratio,
)
from recall.services.daemon import (
    _remove_cron_block,
    _start_service,
    _stop_service,
    _write_crontab,
    resolve_service_scope,
)
from recall.services.service_records import ServiceCapability

PARAM_TYPES = dict(
    dry_run="boolean",
    no_restart="boolean",
    yes="boolean",
    threshold="number",
    fields="string",
    format="string",
    json="boolean",
    cta="boolean",
)

logger = logging.getLogger("recall.cli.compact")


class _DaemonStopError(RuntimeError):
    """Carry lifecycle state out of a partial daemon stop failure."""

    def __init__(
        self,
        message: str,
        *,
        scheduler: ServiceCapability | None,
        restart_required: bool,
    ) -> None:
        super().__init__(message)
        self.scheduler = scheduler
        self.restart_required = restart_required


def command(
    ctx: typer.Context,
    dry_run: bool = typer.Option(False, "--dry-run", help="Report bloat ratio without rebuilding"),
    no_restart: bool = typer.Option(False, "--no-restart", help="Skip restarting the daemon"),
    yes: bool = typer.Option(False, "--yes", help="Confirm destructive database compaction"),
    threshold: float = typer.Option(1.0, "--threshold", help="Skip below bloat ratio"),
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
        payload = parse_params(params, allowed=set(PARAM_TYPES), types=PARAM_TYPES)
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        dry_run = resolve_bool_param(ctx, payload, "dry_run", dry_run)
        no_restart = resolve_bool_param(ctx, payload, "no_restart", no_restart)
        yes = resolve_bool_param(ctx, payload, "yes", yes)
        threshold = _resolve_threshold(resolve_param(ctx, payload, "threshold", threshold))
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)

        def emit(data: dict[str, object]) -> None:
            _emit_compact_result(data, output_format=output_format, fields=fields_value, cta=cta)

        config = AppConfig.load()
        check_database_schema(config)
        daemon_was_running = False
        scheduler: ServiceCapability | None = None
        daemon_restarted = False
        daemon_restart_attempted = False

        if dry_run:
            try:
                before = estimate_bloat_ratio(config.db_path, config)
            except duckdb.IOException as err:
                if is_lock_conflict(err):
                    raise CliError(
                        code=ErrorCode.RUNTIME,
                        message=(
                            "cannot read database while daemon holds the lock; "
                            "stop the daemon first (`recall daemon stop`) or re-run "
                            "without --dry-run to coordinate automatically"
                        ),
                        exit_code=2,
                    ) from None
                raise CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=2) from err
            except Exception as err:
                raise CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=2) from err
            emit(
                _payload(
                    before=before,
                    daemon_was_running=False,
                    daemon_restarted=False,
                    skipped_reason="dry_run",
                )
            )
            return

        require_confirmation(
            should_confirm=True,
            confirmed=yes,
            dry_run=dry_run,
            message="`recall compact` rebuilds the live database.",
        )

        compaction_error: CliError | None = None
        pending_error: CliError | None = None
        try:
            with _compaction_sentinel(config):
                try:
                    daemon_was_running, scheduler = _stop_daemon(config)
                except _DaemonStopError as err:
                    daemon_was_running = True
                    scheduler = err.scheduler
                    primary_error = CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=2)
                    if err.restart_required and not no_restart:
                        daemon_restart_attempted = True
                        primary_error = _restart_daemon_preserving_primary_error(
                            config, scheduler, primary_error
                        )
                    pending_error = primary_error
                if pending_error is None:
                    with advisory_lock(config.lock_path):
                        try:
                            before = _estimate_bloat_ratio_for_cli(config)
                            if before.ratio < threshold:
                                if daemon_was_running and not no_restart:
                                    daemon_restart_attempted = True
                                    daemon_restarted = _restart_daemon_for_cli(config, scheduler)
                                emit(
                                    _payload(
                                        before=before,
                                        daemon_was_running=daemon_was_running,
                                        daemon_restarted=daemon_restarted,
                                        skipped_reason="below_threshold",
                                    )
                                )
                                return
                            result = compact(config)
                        except CliError as err:
                            primary_error = err
                            if (
                                daemon_was_running
                                and not no_restart
                                and not daemon_restart_attempted
                            ):
                                daemon_restart_attempted = True
                                primary_error = _restart_daemon_preserving_primary_error(
                                    config, scheduler, primary_error
                                )
                            pending_error = primary_error
                        except CompactionError as err:
                            compaction_error = _compaction_cli_error(err)
                            if daemon_was_running and not no_restart:
                                daemon_restart_attempted = True
                                compaction_error = _restart_daemon_preserving_primary_error(
                                    config, scheduler, compaction_error
                                )
                        except RuntimeError as err:
                            primary_error = CliError(
                                code=ErrorCode.RUNTIME,
                                message=str(err),
                                exit_code=1 if isinstance(err, SchemaTooNewError) else 2,
                            )
                            if daemon_was_running and not no_restart:
                                daemon_restart_attempted = True
                                primary_error = _restart_daemon_preserving_primary_error(
                                    config, scheduler, primary_error
                                )
                            pending_error = primary_error

                        if pending_error is None and compaction_error is None:
                            if daemon_was_running and not no_restart:
                                daemon_restart_attempted = True
                                daemon_restarted = _restart_daemon_for_cli(config, scheduler)

                            emit(
                                _payload(
                                    before=before,
                                    result=result,
                                    daemon_was_running=daemon_was_running,
                                    daemon_restarted=daemon_restarted,
                                )
                            )
                            return
        except RecallLockError as err:
            raise CliError(code=ErrorCode.LOCKED, message=str(err), exit_code=1) from None
        except RuntimeError as err:
            raise CliError(
                code=ErrorCode.RUNTIME,
                message=str(err),
                exit_code=1 if isinstance(err, SchemaTooNewError) else 2,
            ) from None
        if pending_error is not None:
            raise pending_error
        if compaction_error is not None:
            raise compaction_error
    except SchemaTooNewError as err:
        emit_error(
            CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=1),
            output_format=output_format,
        )
        raise typer.Exit(code=1) from None
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        error = CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2)
        emit_error(error, output_format=output_format)
        raise typer.Exit(code=2) from None


def _estimate_bloat_ratio_for_cli(config: AppConfig) -> BloatStats:
    try:
        return estimate_bloat_ratio(config.db_path, config)
    except Exception as err:
        raise CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=2) from err


def _resolve_threshold(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("threshold must be a number")
    if not isinstance(value, (int, float, str)):
        raise ValueError("threshold must be a number")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError("threshold must be a number") from err
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return threshold


def _payload(
    *,
    before,
    result=None,
    daemon_was_running: bool = False,
    daemon_restarted: bool = False,
    skipped_reason: str | None = None,
) -> dict[str, object]:
    after = result.after if result is not None else None
    return {
        "before_bytes": before.file_size,
        "before_used_bytes": before.used_bytes,
        "before_live_bytes": before.live_bytes,
        "before_ratio": before.ratio,
        "after_bytes": after.file_size if after is not None else None,
        "after_used_bytes": after.used_bytes if after is not None else None,
        "after_live_bytes": after.live_bytes if after is not None else None,
        "after_ratio": after.ratio if after is not None else None,
        "elapsed_seconds": result.elapsed_seconds if result is not None else None,
        "tables_copied": result.tables_copied if result is not None else {},
        "daemon_was_running": daemon_was_running,
        "daemon_restarted": daemon_restarted,
        "skipped_reason": skipped_reason if result is None else result.skipped_reason,
    }


def _emit_compact_result(
    payload: dict[str, object],
    *,
    output_format: OutputFormat,
    fields: tuple[str, ...] | None,
    cta: bool,
) -> None:
    if output_format == OutputFormat.TEXT:
        _emit_compact_text(payload)
        return

    emit_data_with_cta(
        payload,
        [],
        output_format=output_format,
        include_cta=cta,
        fields=fields,
        allowed_fields=output_fields_for("compact"),
    )


def _emit_compact_text(payload: dict[str, object]) -> None:
    typer.echo(f"Before size: {_format_bytes(payload.get('before_bytes'))}")
    typer.echo(f"Before used: {_format_bytes(payload.get('before_used_bytes'))}")
    typer.echo(f"Before live: {_format_bytes(payload.get('before_live_bytes'))}")
    typer.echo(f"Before ratio: {_format_ratio(payload.get('before_ratio'))}")

    tables_copied = payload.get("tables_copied")
    if isinstance(tables_copied, dict) and tables_copied:
        typer.echo("Tables copied:", err=True)
        for table, rows in tables_copied.items():
            typer.echo(f"  {table}: {rows} rows", err=True)

    skipped_reason = payload.get("skipped_reason")
    if skipped_reason is not None:
        typer.echo(f"Skipped: {skipped_reason}")
        return

    typer.echo(f"After size: {_format_bytes(payload.get('after_bytes'))}")
    typer.echo(f"After used: {_format_bytes(payload.get('after_used_bytes'))}")
    typer.echo(f"After live: {_format_bytes(payload.get('after_live_bytes'))}")
    typer.echo(f"After ratio: {_format_ratio(payload.get('after_ratio'))}")
    typer.echo(f"Elapsed seconds: {_format_seconds(payload.get('elapsed_seconds'))}")
    typer.echo(f"Daemon restart: {_format_daemon_restart(payload)}")


def _format_bytes(value: object) -> str:
    if isinstance(value, int):
        return f"{value} bytes"
    return "unknown"


def _format_ratio(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "unknown"


def _format_seconds(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "unknown"


def _format_daemon_restart(payload: dict[str, object]) -> str:
    if payload.get("daemon_was_running") is not True:
        return "not needed"
    return "restarted" if payload.get("daemon_restarted") is True else "not restarted"


def _stop_daemon(config: AppConfig) -> tuple[bool, ServiceCapability | None]:
    pid = _running_daemon_pid_for_compact(config)
    capability, scheduler_pid = _blocking_scheduler_for_compact(config, pid)
    if pid is None and capability is None:
        return False, None
    restart_required = False
    try:
        if capability is not None:
            record = capability.record
            if record.scheduler == SchedulerKind.LAUNCHD:
                capability.snapshot_artifacts(config.data_dir)
            if record.scheduler == SchedulerKind.CRON:
                assert record.crontab is not None
                _write_crontab(_remove_cron_block(record.crontab))
                restart_required = True
            else:
                restart_required = True
                result = _stop_service(
                    config,
                    capability,
                    timeout=10,
                    started_at=time.monotonic(),
                    stop_unowned=_stop_auto_forked_daemon,
                )
                if not result.stopped:
                    raise RuntimeError(result.message)
        if pid is not None and (
            capability is None or capability.record.scheduler == SchedulerKind.CRON
        ):
            _stop_auto_forked_daemon(pid)
            restart_required = True
        _wait_for_daemon_release(config, original_pid=pid or scheduler_pid)
    except RuntimeError as err:
        raise _DaemonStopError(
            str(err), scheduler=capability, restart_required=restart_required
        ) from err
    return True, capability


def _blocking_scheduler_for_compact(
    config: AppConfig,
    pid: int | None,
) -> tuple[ServiceCapability | None, int | None]:
    scope = resolve_service_scope(config)
    if scope.other_service is not None or scope.service is None:
        return None, None
    capability = scope.authorize(config, "stop")
    return capability, scope.service.pid


def _daemon_socket_path(config: AppConfig) -> Path:
    return config.data_dir / "recall.sock"


def _daemon_pid_path(config: AppConfig) -> Path:
    return config.data_dir / "recall.pid"


def _running_daemon_pid_for_compact(config: AppConfig) -> int | None:
    """Return the PID of a running recall daemon, or None.

    Per REQ-COMPACT-004a, pidfile-with-live-PID is the authoritative signal.
    Socket presence corroborates that finding but is not required: compact
    must stop a daemon that is mid-startup before the socket exists.
    """
    socket_present = _daemon_socket_path(config).exists()
    pid_path = _daemon_pid_path(config)
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if not _process_is_alive(pid):
        return None
    if socket_present:
        logger.debug("compact detected daemon socket for pid=%s", pid)
    return pid


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _stop_auto_forked_daemon(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError as err:
        raise RuntimeError(f"cannot signal daemon pid={pid}: {err}") from err


def _restart_daemon(config: AppConfig, scheduler: ServiceCapability | None) -> bool:
    if scheduler is None:
        _start_background(config, verbose=False)
        _wait_for_pid_file(config, timeout=30.0)
        return True
    if scheduler.record.scheduler == SchedulerKind.CRON:
        assert scheduler.record.crontab is not None
        _write_crontab(scheduler.record.crontab)
        return True
    result = _start_service(config, scheduler, timeout=30, started_at=time.monotonic())
    if not result.started:
        raise RuntimeError(result.message)
    return True


def _restart_daemon_for_cli(config: AppConfig, scheduler: ServiceCapability | None) -> bool:
    try:
        daemon_restarted = _restart_daemon(config, scheduler)
    except RuntimeError as err:
        raise CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=2) from None
    if not daemon_restarted:
        raise CliError(
            code=ErrorCode.RUNTIME,
            message="daemon failed to restart after compact",
            exit_code=2,
        )
    return daemon_restarted


def _restart_daemon_preserving_primary_error(
    config: AppConfig,
    scheduler: ServiceCapability | None,
    primary_error: CliError,
) -> CliError:
    try:
        _restart_daemon_for_cli(config, scheduler)
    except CliError as restart_error:
        return _cli_error_with_restart_failure(primary_error, restart_error)
    return primary_error


def _cli_error_with_restart_failure(primary_error: CliError, restart_error: CliError) -> CliError:
    details = dict(primary_error.details or {})
    details["restart_error"] = restart_error.message
    message = (
        f"{primary_error.message}; additionally, daemon restart failed: {restart_error.message}"
    )
    return CliError(
        code=primary_error.code,
        message=message,
        details=details,
        exit_code=primary_error.exit_code,
    )


def _wait_for_daemon_release(
    config: AppConfig, *, original_pid: int | None, timeout: float = 30.0
) -> None:
    pid_path = _daemon_pid_path(config)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pid_released = original_pid is None or not _process_is_alive(original_pid)
        # The pidfile is the authoritative lifecycle artifact for compact.
        # Socket removal cannot be required because startup may be stopped
        # before the daemon creates one.
        if pid_released and not pid_path.exists():
            return
        # launchd's bootout SIGKILLs the daemon if it doesn't exit within its
        # grace period, so atexit-driven pidfile cleanup never runs. If the PID
        # in the pidfile matches the (now-dead) original_pid, treat it as stale
        # and unlink it. A different PID means a respawn — keep waiting for that
        # process to release the file itself.
        if (
            pid_released
            and original_pid is not None
            and _pidfile_pid_is(pid_path, original_pid)
            and _unlink_stale_pidfile(pid_path)
        ):
            return
        # If the unlink branch fell through (write failure: permission, FS error),
        # keep polling so the caller eventually hits the timeout rather than
        # claiming a clean stop while the authoritative pidfile is still on disk.
        time.sleep(0.2)
    if original_pid is None:
        raise RuntimeError(f"daemon did not release pidfile within {timeout:.0f}s")
    raise RuntimeError(
        f"daemon at pid={original_pid} did not release pidfile or exit within {timeout:.0f}s"
    )


def _pidfile_pid_is(pid_path: Path, expected_pid: int) -> bool:
    try:
        content = pid_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return False
    try:
        return int(content) == expected_pid
    except ValueError:
        return False


def _unlink_stale_pidfile(pid_path: Path) -> bool:
    """Remove a stale pidfile left by a SIGKILLed daemon. Returns True on success."""
    try:
        pid_path.unlink(missing_ok=True)
    except OSError as err:
        logger.warning("compact: failed to unlink stale pidfile %s: %s", pid_path, err)
        return False
    return True


def _compaction_cli_error(err: CompactionError) -> CliError:
    message = str(err)
    exit_code = 3 if "replacement failed" in message else 1
    return CliError(code=ErrorCode.RUNTIME, message=message, exit_code=exit_code)
