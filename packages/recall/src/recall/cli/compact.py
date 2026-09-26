from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
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
from recall.core.types import SchedulerKind
from recall.db import RecallLockError, advisory_lock, is_lock_conflict
from recall.services.compaction import (
    BloatStats,
    CompactionError,
    _compaction_sentinel,
    compact,
    estimate_bloat_ratio,
)
from recall.services.daemon import (
    CRON_BEGIN_MARKER,
    _active_launchd_label,
    _launchd_plist_path,
    _launchd_service_target,
    _read_crontab,
    _remove_cron_block,
    _write_crontab,
    daemon_status,
)

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

_stopped_crontab_for_compact: str | None = None


class _DaemonStopError(RuntimeError):
    """Carry lifecycle state out of a partial daemon stop failure."""

    def __init__(
        self,
        message: str,
        *,
        scheduler: SchedulerKind | None,
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
        daemon_was_running = False
        scheduler: SchedulerKind | None = None
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
                                code=ErrorCode.RUNTIME, message=str(err), exit_code=2
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
            raise CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=2) from None
        if pending_error is not None:
            raise pending_error
        if compaction_error is not None:
            raise compaction_error
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
        "before_live_bytes": before.live_bytes,
        "before_ratio": before.ratio,
        "after_bytes": after.file_size if after is not None else None,
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


def _stop_daemon(config: AppConfig) -> tuple[bool, SchedulerKind | None]:
    pid = _running_daemon_pid_for_compact(config)
    scheduler, scheduler_pid = _blocking_scheduler_for_compact(config, pid)
    if pid is None and scheduler is None:
        return False, None

    _stop_daemon_from_lifecycle(config, scheduler=scheduler, pid=pid, scheduler_pid=scheduler_pid)
    return True, scheduler


def _stop_daemon_from_lifecycle(
    config: AppConfig,
    *,
    scheduler: SchedulerKind | None,
    pid: int | None,
    scheduler_pid: int | None,
) -> None:
    restart_required = False
    try:
        if scheduler == SchedulerKind.LAUNCHD:
            restart_required = True
            cmd = ["launchctl", "bootout", _launchd_service_target(_active_launchd_label())]
            _run_lifecycle_command(cmd, "stop launchd daemon")
            # bootout is async — wait until launchd reports the unit unloaded so
            # _wait_for_daemon_release cannot observe a respawned daemon's fresh
            # pidfile during the boot-out window.
            _wait_for_launchd_unit_unloaded()
            _stop_pid_file_daemon_if_unowned(pid, scheduler_pid)
        elif scheduler == SchedulerKind.SYSTEMD:
            restart_required = True
            _stop_systemd_daemon()
            _stop_pid_file_daemon_if_unowned(pid, scheduler_pid)
        elif scheduler == SchedulerKind.CRON:
            _stop_cron_scheduler()
            restart_required = _stopped_crontab_for_compact is not None
            if pid is not None:
                _stop_auto_forked_daemon(pid)
                restart_required = True
        else:
            if pid is not None:
                _stop_auto_forked_daemon(pid)
                restart_required = True

        _wait_for_daemon_release(config, original_pid=pid or scheduler_pid)
    except RuntimeError as err:
        restart_required = restart_required or (
            scheduler == SchedulerKind.CRON and _stopped_crontab_for_compact is not None
        )
        raise _DaemonStopError(
            str(err), scheduler=scheduler, restart_required=restart_required
        ) from err


def _blocking_scheduler_for_compact(
    config: AppConfig, pid: int | None
) -> tuple[SchedulerKind | None, int | None]:
    scheduler, scheduler_pid = _running_lifecycle_scheduler(config, pid)
    if scheduler is not None:
        return scheduler, scheduler_pid
    if _cron_scheduler_installed():
        return SchedulerKind.CRON, None
    return None, None


def _running_lifecycle_scheduler(
    config: AppConfig, pid: int | None
) -> tuple[SchedulerKind | None, int | None]:
    """Detect whether launchd/systemd owns the live compact-blocking daemon."""
    for scheduler in _lifecycle_scheduler_candidates(config):
        if scheduler == SchedulerKind.LAUNCHD and _launchd_artifact_path().exists():
            scheduler_pid = _launchd_daemon_pid()
            if (
                _scheduler_pid_matches_daemon_pid(scheduler_pid, pid)
                or _launchd_watch_scheduler_installed()
                or _launchd_poll_scheduler_loaded()
            ):
                return SchedulerKind.LAUNCHD, scheduler_pid
        if scheduler == SchedulerKind.SYSTEMD and _systemd_scheduler_installed():
            scheduler_pid = _systemd_daemon_pid()
            if (
                _scheduler_pid_matches_daemon_pid(scheduler_pid, pid)
                or _systemd_watch_scheduler_installed()
                or _systemd_timer_installed()
            ):
                return SchedulerKind.SYSTEMD, scheduler_pid
    return None, None


def _stop_pid_file_daemon_if_unowned(pid: int | None, scheduler_pid: int | None) -> None:
    if pid is None or pid == scheduler_pid:
        return
    _stop_auto_forked_daemon(pid)


def _scheduler_pid_matches_daemon_pid(scheduler_pid: int | None, daemon_pid: int | None) -> bool:
    if scheduler_pid is None:
        return False
    return daemon_pid is None or scheduler_pid == daemon_pid


def _lifecycle_scheduler_candidates(config: AppConfig) -> tuple[SchedulerKind, ...]:
    platform_candidates = _platform_lifecycle_scheduler_candidates()
    requested = config.daemon.scheduler
    if requested not in {SchedulerKind.LAUNCHD, SchedulerKind.SYSTEMD}:
        return platform_candidates

    return (requested, *(candidate for candidate in platform_candidates if candidate != requested))


def _platform_lifecycle_scheduler_candidates() -> tuple[SchedulerKind, ...]:
    if sys.platform == "darwin":
        return (SchedulerKind.LAUNCHD,)
    if sys.platform.startswith("linux"):
        return (SchedulerKind.SYSTEMD,)
    return tuple()


def _launchd_artifact_path() -> Path:
    return _launchd_plist_path(_active_launchd_label())


def _launchd_poll_scheduler_loaded() -> bool:
    if not _launchd_poll_scheduler_installed():
        return False
    label = _active_launchd_label()
    print_result = _run_lifecycle_query(["launchctl", "print", _launchd_service_target(label)])
    if print_result is not None and print_result.returncode == 0:
        return True
    list_result = _run_lifecycle_query(["launchctl", "list", label])
    return list_result is not None and list_result.returncode == 0


def _launchctl_unit_is_loaded() -> bool:
    """True iff launchctl reports the recall service registered to the user domain.

    Independent of plist mode (`KeepAlive`, `RunAtLoad`, or `StartInterval`).
    `_launchd_poll_scheduler_loaded` gates on StartInterval and is unsuitable for
    the KeepAlive/watch-mode plist that auto-fork installs by default.
    """
    label = _active_launchd_label()
    print_result = _run_lifecycle_query(["launchctl", "print", _launchd_service_target(label)])
    if print_result is not None and print_result.returncode == 0:
        return True
    list_result = _run_lifecycle_query(["launchctl", "list", label])
    return list_result is not None and list_result.returncode == 0


def _wait_for_launchd_unit_unloaded(timeout: float = 10.0, interval: float = 0.2) -> None:
    """Poll launchctl until the recall unit reports unloaded, or raise on timeout.

    `launchctl bootout` returns before the unit's process has actually exited; without
    this wait, `_wait_for_daemon_release` can observe a respawned daemon's fresh
    pidfile and time out.
    """
    label = _active_launchd_label()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _launchctl_unit_is_loaded():
            return
        time.sleep(interval)
    raise RuntimeError(
        f"launchctl unit {label} still loaded after {timeout:.0f}s; "
        "bootout did not unregister the unit"
    )


def _launchd_poll_scheduler_installed() -> bool:
    artifact_path = _launchd_artifact_path()
    if not artifact_path.exists():
        return False
    try:
        plist = artifact_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "<key>StartInterval</key>" in plist


def _launchd_watch_scheduler_installed() -> bool:
    artifact_path = _launchd_artifact_path()
    if not artifact_path.exists():
        return False
    try:
        plist = artifact_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "<key>KeepAlive</key>" in plist


def _systemd_artifact_paths() -> tuple[Path, Path]:
    base = Path.home() / ".config" / "systemd" / "user"
    return base / "recall-daemon.service", base / "recall-daemon.timer"


def _systemd_scheduler_installed() -> bool:
    service_path, timer_path = _systemd_artifact_paths()
    if not service_path.exists():
        return False
    if timer_path.exists():
        return True
    return _systemd_watch_scheduler_installed()


def _systemd_watch_scheduler_installed() -> bool:
    service_path, _ = _systemd_artifact_paths()
    if not service_path.exists():
        return False
    try:
        service = service_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "Type=simple" in service


def _systemd_timer_installed() -> bool:
    _, timer_path = _systemd_artifact_paths()
    return timer_path.exists()


def _cron_scheduler_installed() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    return CRON_BEGIN_MARKER in _read_crontab_for_compact()


def _stop_cron_scheduler() -> None:
    global _stopped_crontab_for_compact
    existing = _read_crontab_for_compact()
    updated = _remove_cron_block(existing)
    if updated == existing:
        return
    _write_crontab(updated)
    _stopped_crontab_for_compact = existing


def _restore_cron_scheduler() -> None:
    global _stopped_crontab_for_compact
    if _stopped_crontab_for_compact is None:
        raise RuntimeError("cannot restart cron daemon scheduler without compact crontab snapshot")
    _write_crontab(_stopped_crontab_for_compact)
    _stopped_crontab_for_compact = None


def _read_crontab_for_compact() -> str:
    try:
        return _read_crontab()
    except OSError as err:
        raise RuntimeError(f"could not inspect cron daemon scheduler: {err}") from err


def _stop_systemd_daemon() -> None:
    if _systemd_timer_installed():
        # Timer-backed poll installs must be stopped first; otherwise systemd
        # can activate the service again while compaction owns the database.
        cmd = ["systemctl", "--user", "stop", "recall-daemon.timer"]
        _run_lifecycle_command(cmd, "stop systemd timer")
    cmd = ["systemctl", "--user", "stop", "recall-daemon.service"]
    _run_lifecycle_command(cmd, "stop systemd daemon")


def _launchd_daemon_pid() -> int | None:
    label = _active_launchd_label()
    print_result = _run_lifecycle_query(["launchctl", "print", _launchd_service_target(label)])
    if print_result is not None and print_result.returncode == 0:
        pid = _parse_launchd_print_pid(print_result.stdout)
        if pid is not None:
            return pid

    list_result = _run_lifecycle_query(["launchctl", "list", label])
    if list_result is None or list_result.returncode != 0:
        return None
    return _parse_launchd_list_pid(list_result.stdout)


def _systemd_daemon_pid() -> int | None:
    result = _run_lifecycle_query(
        [
            "systemctl",
            "--user",
            "show",
            "recall-daemon.service",
            "--property=MainPID",
            "--no-pager",
        ]
    )
    if result is None or result.returncode != 0:
        return None
    return _parse_systemd_main_pid(result.stdout)


def _run_lifecycle_query(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(args, check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _parse_launchd_print_pid(stdout: str) -> int | None:
    for line in stdout.splitlines():
        key, separator, value = line.strip().strip(";").partition("=")
        if separator != "=":
            continue
        if key.strip().strip('"').lower() != "pid":
            continue
        return _parse_positive_int(value.strip().strip('";'))
    return None


def _parse_launchd_list_pid(stdout: str) -> int | None:
    fields = stdout.strip().split()
    if not fields:
        return None
    return _parse_positive_int(fields[0])


def _parse_systemd_main_pid(stdout: str) -> int | None:
    stripped = stdout.strip()
    if "=" not in stripped:
        return _parse_positive_int(stripped)
    for line in stripped.splitlines():
        key, _, value = line.partition("=")
        if key == "MainPID":
            return _parse_positive_int(value)
    return None


def _parse_positive_int(value: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _installed_scheduler_from_status(config: AppConfig) -> SchedulerKind | None:
    status = daemon_status(config)
    return status.scheduler if status.installed else None


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


def _restart_daemon(config: AppConfig, scheduler: SchedulerKind | None) -> bool:
    if scheduler == SchedulerKind.LAUNCHD:
        plist_path = _launchd_artifact_path()
        cmd = ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)]
        _run_lifecycle_command(cmd, "restart launchd daemon")
        return True
    if scheduler == SchedulerKind.SYSTEMD:
        unit = "recall-daemon.timer" if _systemd_timer_installed() else "recall-daemon.service"
        cmd = ["systemctl", "--user", "start", unit]
        _run_lifecycle_command(cmd, "restart systemd daemon")
        return True
    if scheduler == SchedulerKind.CRON:
        _restore_cron_scheduler()
        return True

    _start_background(config, verbose=False)
    _wait_for_pid_file(config, timeout=30.0)
    return True


def _restart_daemon_for_cli(config: AppConfig, scheduler: SchedulerKind | None) -> bool:
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
    scheduler: SchedulerKind | None,
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


def _run_lifecycle_command(args: list[str], action: str) -> None:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as err:
        raise RuntimeError(f"could not {action}: {err}") from err
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"could not {action}{suffix}")


def _compaction_cli_error(err: CompactionError) -> CliError:
    message = str(err)
    exit_code = 3 if "replacement failed" in message else 1
    return CliError(code=ErrorCode.RUNTIME, message=message, exit_code=exit_code)
