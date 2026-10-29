from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import duckdb
from pydantic import TypeAdapter, ValidationError

from recall.core.config import (
    AppConfig,
    create_private_dir,
    resolve_recall_binary,
)
from recall.core.daemon_status import DaemonSchedulerStatus, RuntimeStatus, SourcePage
from recall.core.rpc_client import (
    RpcAdmissionBusyError,
    RpcCallError,
    RpcClient,
    RpcConnectionError,
)
from recall.core.types import DaemonMode, SchedulerKind, Source
from recall.db import (
    FtsSidecarUnavailableError,
    advisory_lock,
    connect,
    is_lock_conflict,
    open_sidecar,
    sidecar_path,
)
from recall.db.fts_sidecar import (
    fts_fields_signature,
    get_fts_fields_signature,
    set_fts_fields_signature,
)
from recall.db.ownership import SnapshotUnavailable, read_offline
from recall.services import service_records
from recall.services.compaction import CompactionError, compact, estimate_bloat_ratio
from recall.services.embed_phase import PendingSplit, count_pending_split
from recall.services.fts_sidecar_reconcile import rescope_sidecar_for_fields
from recall.services.indexer import IndexSummary
from recall.services.runtime_state import load_runtime_status, set_installed_scheduler
from recall.services.self_repair import read_refusal_marker
from recall.services.service_records import (
    ArtifactSnapshot as _ArtifactSnapshot,
)
from recall.services.service_records import (
    SchedulerRecord,
    Service,
    ServiceCapability,
    ServiceLookupError,
)
from recall.services.snapshots import gc_snapshots
from recall.services.watcher import resolve_daemon_mode

logger = logging.getLogger("recall.daemon")

LAUNCHD_LABEL = "it.send.recall.daemon"
# Earlier releases installed the launch agent under this label. Lifecycle
# commands keep acting on such an install until `recall daemon install`
# retires it (REQ-DAEMON-076).
LEGACY_LAUNCHD_LABEL = "xyz.metalrodeo.recall.daemon"
CRON_BEGIN_MARKER = "# BEGIN recall daemon"
CRON_END_MARKER = "# END recall daemon"

# How long to let an asynchronous `launchctl bootout` drain before bootstrapping
# over it. Teardown closes the DuckDB store, so it scales with database size.
_LAUNCHD_UNLOAD_WAIT_SECONDS_MAX = 30.0
_LAUNCHD_UNLOAD_POLL_SECONDS = 0.25
_LAUNCHD_BOOTSTRAP_ATTEMPTS = 3
_SCHEDULER_DATABASE_WAIT_SECONDS_MAX = 30.0
_SCHEDULER_DATABASE_POLL_SECONDS = 0.1


@dataclass(frozen=True)
class DaemonCycleSummary:
    index_summary: IndexSummary
    swapped: bool
    embed_summary: dict[str, int] | None = None
    record_status_persisted: bool = True


@dataclass(frozen=True)
class FtsSidecarStartupResult:
    enabled: bool
    bootstrap_messages_processed: int
    bootstrap_tool_calls_processed: int
    bootstrap_messages_done: bool
    bootstrap_tool_calls_done: bool
    reconcile_pending_drained: dict[str, int]
    reconcile_orphans_backfilled: dict[str, int]
    reconcile_ghosts_deleted: dict[str, int]
    reconcile_pending_remaining: dict[str, int]
    last_run_at: datetime | None
    error: str | None = None


@dataclass(frozen=True)
class DaemonStopResult:
    scheduler: str
    stopped: bool
    pid: int | None
    duration_seconds: float
    message: str | None = None


@dataclass(frozen=True)
class DaemonStartResult:
    scheduler: str
    started: bool
    pid: int | None
    duration_seconds: float
    message: str | None = None


@dataclass(frozen=True)
class DaemonRestartResult:
    scheduler: str
    stopped: bool
    started: bool
    pid: int | None
    duration_seconds: float
    message: str | None = None


@dataclass(frozen=True)
class ServiceScope:
    """Scheduler identity and selected data dir, measured once for an operation."""

    data_dir: Path
    record: SchedulerRecord
    identity_record: SchedulerRecord | None = None

    @property
    def service(self) -> Service | None:
        return (self.identity_record or self.record).service

    @property
    def other_service(self) -> Service | None:
        return next(
            (
                service
                for service in (self.identity_record or self.record).services
                if service.effective_data_dir != self.data_dir.resolve()
            ),
            None,
        )

    def authorize(
        self, config: AppConfig, action: str, *, durable: bool = False
    ) -> ServiceCapability:
        other = self.other_service
        if other is not None:
            raise RuntimeError(_other_service_refusal(config, other, action))
        return ServiceCapability(self.record if durable else self.identity_record or self.record)


def resolve_service_scope(
    config: AppConfig,
    scheduler: SchedulerKind | None = None,
) -> ServiceScope:
    selected = scheduler or config.daemon.scheduler
    if selected == SchedulerKind.AUTO:
        selected = SchedulerKind.LAUNCHD if sys.platform == "darwin" else SchedulerKind.SYSTEMD
    record = service_records.discover(selected, run=service_records.query)
    if (
        scheduler is None
        and config.daemon.scheduler == SchedulerKind.AUTO
        and not record.services
        and sys.platform.startswith("linux")
    ):
        identity = service_records.discover(SchedulerKind.CRON, run=service_records.query)
        return ServiceScope(config.data_dir, record, identity)
    return ServiceScope(config.data_dir, record)


def lifecycle_scope(config: AppConfig) -> ServiceScope | ServiceLookupError:
    """Capture the target and any discoverable cron identity without switching actions."""
    scheduler = _durable_scheduler(config) or config.daemon.scheduler
    try:
        scope = resolve_service_scope(config, scheduler)
        if (
            scheduler == SchedulerKind.SYSTEMD
            and config.daemon.scheduler == SchedulerKind.AUTO
            and not scope.record.services
            and shutil.which("crontab") is not None
        ):
            identity = service_records.discover(SchedulerKind.CRON, run=service_records.query)
            return ServiceScope(config.data_dir, scope.record, identity)
        return scope
    except ServiceLookupError as err:
        return err


def _scope_scheduler(scope: ServiceScope | ServiceLookupError) -> SchedulerKind:
    if isinstance(scope, ServiceLookupError):
        return scope.scheduler
    other = scope.other_service
    return other.scheduler if other is not None else scope.record.scheduler


def _authorize_lifecycle(
    config: AppConfig,
    action: str,
    scope: ServiceScope | ServiceLookupError,
) -> ServiceCapability:
    if isinstance(scope, ServiceLookupError):
        raise RuntimeError(
            f"refusing to {action} the installed {scope.scheduler.value} service: "
            f"cannot identify its data dir from the scheduler record: {scope}. "
            "No service action was taken."
        ) from scope
    return scope.authorize(config, action, durable=True)


_DATA_DIR_SCOPED_ALTERNATIVE = {
    "start": "Start this data dir's daemon with `recall daemon start --background`",
    "stop": "Stop this data dir's daemon with `recall daemon stop --soft`",
    "restart": (
        "Restart this data dir's daemon with `recall daemon stop --soft` "
        "then `recall daemon start --background`"
    ),
    "uninstall": "Nothing is installed for this data dir",
    "install": "To replace this service, select its data dir first",
}


def _other_service_refusal(config: AppConfig, service: Service, action: str) -> str:
    return (
        f"refusing to {action} the installed {service.scheduler.value} service: it runs "
        f"data dir {service.data_dir}, not the selected data dir {config.data_dir}. "
        f"{_DATA_DIR_SCOPED_ALTERNATIVE[action]}; to manage the service, select its data dir "
        f"(RECALL_DATA_DIR={service.data_dir})."
    )


def stop_daemon_soft(config: AppConfig | None = None) -> DaemonStopResult:
    """Stop only the selected PID, without any scheduler discovery."""
    config = config or AppConfig.load()
    return replace(
        _stop_pidfile(config, timeout=10, started_at=time.monotonic()), scheduler="sentinel"
    )


def _stop_pidfile(config: AppConfig, *, timeout: float, started_at: float) -> DaemonStopResult:
    """Signal the selected data dir's daemon through its PID file; never touches a scheduler."""
    pid = _running_daemon_pid_from_file(config)
    if pid is None:
        _cleanup_daemon_runtime_files(config)
        return DaemonStopResult(
            scheduler="pid",
            stopped=True,
            pid=None,
            duration_seconds=_elapsed(started_at),
            message="daemon already stopped",
        )
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _cleanup_daemon_runtime_files(config)
        return DaemonStopResult(
            scheduler="pid",
            stopped=True,
            pid=pid,
            duration_seconds=_elapsed(started_at),
            message="daemon process was already gone",
        )
    except PermissionError as err:
        return DaemonStopResult(
            scheduler="pid",
            stopped=False,
            pid=pid,
            duration_seconds=_elapsed(started_at),
            message=f"cannot signal daemon pid={pid}: {err}",
        )

    result = _wait_until_stopped(config, pid=pid, timeout=timeout, started_at=started_at)
    if result.stopped:
        _cleanup_daemon_runtime_files(config)
    return result


def stop_daemon_durable(
    config: AppConfig | None = None,
    *,
    timeout: float = 10.0,
    scope: ServiceScope | ServiceLookupError | None = None,
) -> DaemonStopResult:
    config = config or AppConfig.load()
    started_at = time.monotonic()
    scope = scope if scope is not None else lifecycle_scope(config)
    scheduler = _scope_scheduler(scope)
    try:
        capability = _authorize_lifecycle(config, "stop", scope)
    except RuntimeError as err:
        return DaemonStopResult(
            scheduler.value,
            False,
            _running_daemon_pid_from_file(config),
            _elapsed(started_at),
            str(err),
        )
    return _stop_service(config, capability, timeout=timeout, started_at=started_at)


def _stop_service(
    config: AppConfig,
    capability: ServiceCapability,
    *,
    timeout: float,
    started_at: float,
    stop_unowned: Callable[[int], None] | None = None,
) -> DaemonStopResult:
    record = capability.record
    pid = _running_daemon_pid_from_file(config)
    scheduler = record.scheduler
    if scheduler == SchedulerKind.CRON:
        return _stop_pidfile(config, timeout=timeout, started_at=started_at)
    if not record.services:
        if pid is None:
            _cleanup_daemon_runtime_files(config)
            return DaemonStopResult(
                scheduler.value, True, None, _elapsed(started_at), "daemon already stopped"
            )
        return DaemonStopResult(
            scheduler.value,
            False,
            pid,
            _elapsed(started_at),
            f"daemon pid={pid} is running but {scheduler.value} does not manage it (orphaned). "
            f"Stop it with `kill {pid}`, then run `recall daemon install` "
            "to reinstate the service.",
        )
    try:
        if scheduler == SchedulerKind.LAUNCHD:
            for service in record.services:
                if service.loaded:
                    _run_lifecycle_command(
                        ["launchctl", "bootout", _launchd_service_target(service.name)],
                        "stop launchd daemon",
                    )
        else:
            for unit in _systemd_stop_units(capability):
                _run_lifecycle_command(
                    ["systemctl", "--user", "stop", unit], f"stop systemd {unit}"
                )
    except RuntimeError as err:
        return DaemonStopResult(scheduler.value, False, pid, _elapsed(started_at), str(err))
    try:
        if (
            stop_unowned is not None
            and pid is not None
            and all(service.pid != pid for service in record.services)
        ):
            deadline = started_at + timeout
            while not _service_stopped(capability):
                if time.monotonic() >= deadline:
                    return DaemonStopResult(
                        scheduler.value, False, pid, _elapsed(started_at), "scheduler still loaded"
                    )
                time.sleep(0.05)
            stop_unowned(pid)
        return _wait_for_durable_stop(
            config,
            scheduler=scheduler.value,
            pid=pid,
            timeout=max(0, started_at + timeout - time.monotonic()),
            started_at=started_at,
            scheduler_stopped=lambda: _service_stopped(capability),
        )
    except RuntimeError as err:
        return DaemonStopResult(scheduler.value, False, pid, _elapsed(started_at), str(err))


def _service_stopped(capability: ServiceCapability) -> bool:
    record = capability.record
    if record.scheduler == SchedulerKind.LAUNCHD:
        return all(not _launchd_label_is_loaded(service.name) for service in record.services)
    return all(not _systemd_unit_active(unit) for unit in _systemd_stop_units(capability))


def start_daemon_durable(
    config: AppConfig | None = None,
    *,
    timeout: float = 10.0,
) -> DaemonStartResult:
    config = config or AppConfig.load()
    started_at = time.monotonic()
    scheduler = _durable_scheduler(config)
    pid = _running_daemon_pid_from_file(config)
    scope = lifecycle_scope(config)
    kind = _scope_scheduler(scope)
    try:
        capability = _authorize_lifecycle(config, "start", scope)
    except RuntimeError as err:
        return DaemonStartResult(kind.value, False, None, _elapsed(started_at), str(err))
    if pid is not None:
        return DaemonStartResult(
            scheduler.value if scheduler else "pid",
            True,
            pid,
            _elapsed(started_at),
            "daemon already running",
        )
    return _start_service(config, capability, timeout=timeout, started_at=started_at)


def _start_service(
    config: AppConfig,
    capability: ServiceCapability,
    *,
    timeout: float,
    started_at: float,
) -> DaemonStartResult:
    record = capability.record
    scheduler = record.scheduler
    service = record.service
    if scheduler == SchedulerKind.CRON:
        return DaemonStartResult(
            "pid",
            False,
            None,
            _elapsed(started_at),
            "no durable scheduler is configured on this platform",
        )
    if service is None:
        message = (
            f"launchd plist is not installed: {record.paths[0]}"
            if scheduler == SchedulerKind.LAUNCHD
            else "systemd service is not installed"
        )
        return DaemonStartResult(scheduler.value, False, None, _elapsed(started_at), message)
    try:
        already_loaded = (
            _launchd_label_is_loaded(service.name)
            if scheduler == SchedulerKind.LAUNCHD
            else _systemd_unit_active(_systemd_start_unit(capability))
        )
        if not already_loaded:
            try:
                if scheduler == SchedulerKind.LAUNCHD:
                    assert service.path is not None
                    capability.snapshot_artifacts(config.data_dir)
                    _run_lifecycle_command(
                        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(service.path)],
                        "start launchd daemon",
                    )
                else:
                    unit = _systemd_start_unit(capability)
                    _run_lifecycle_command(
                        ["systemctl", "--user", "start", unit], f"start systemd {unit}"
                    )
            except RuntimeError:
                loaded = (
                    _launchd_label_is_loaded(service.name)
                    if scheduler == SchedulerKind.LAUNCHD
                    else _systemd_unit_active(_systemd_start_unit(capability))
                )
                if not loaded:
                    raise
                already_loaded = True
        result = _wait_until_started(
            config,
            scheduler=scheduler.value,
            timeout=timeout,
            started_at=started_at,
            scheduler_started=(lambda: _launchd_label_is_loaded(service.name))
            if scheduler == SchedulerKind.LAUNCHD
            else lambda: _systemd_unit_active(_systemd_start_unit(capability)),
            expect_live_daemon=service.mode != DaemonMode.POLL,
        )
    except RuntimeError as err:
        return DaemonStartResult(scheduler.value, False, None, _elapsed(started_at), str(err))
    if already_loaded:
        return replace(
            result, message="already loaded" if result.started else "loaded but not responsive"
        )
    return result


def restart_daemon_durable(
    config: AppConfig | None = None,
    *,
    timeout: float = 10.0,
) -> DaemonRestartResult:
    config = config or AppConfig.load()
    started_at = time.monotonic()
    scope = lifecycle_scope(config)
    scheduler = _scope_scheduler(scope)
    try:
        capability = _authorize_lifecycle(config, "restart", scope)
        if capability.record.scheduler == SchedulerKind.LAUNCHD:
            capability.snapshot_artifacts(config.data_dir)
    except RuntimeError as err:
        return DaemonRestartResult(
            scheduler.value,
            False,
            False,
            _running_daemon_pid_from_file(config),
            _elapsed(started_at),
            str(err),
        )
    stop_result = _stop_service(config, capability, timeout=timeout, started_at=started_at)
    if not stop_result.stopped:
        return DaemonRestartResult(
            stop_result.scheduler,
            False,
            False,
            stop_result.pid,
            _elapsed(started_at),
            stop_result.message,
        )
    start_result = _start_service(config, capability, timeout=timeout, started_at=started_at)
    return DaemonRestartResult(
        start_result.scheduler,
        True,
        start_result.started,
        start_result.pid,
        _elapsed(started_at),
        start_result.message,
    )


def _maybe_run_auto_compact(state: Any, config: AppConfig) -> None:
    """Check bloat ratio and compact when the configured threshold is exceeded."""
    cfg = config.compaction
    if not cfg.auto_trigger:
        return

    now = time.monotonic()
    last_check = state.last_compact_check_at
    if last_check is not None:
        elapsed = now - last_check
        if elapsed < cfg.check_interval_hours * 3600:
            return
    state.last_compact_check_at = now

    had_connection = getattr(state, "_conn", None) is not None
    should_reopen = had_connection
    try:
        # DuckDB rejects same-process connections to the same file when the
        # daemon's lenient shared connection is open and the estimator opens a
        # read-only connection. Release it for the scheduled check as well as
        # for the compaction run itself.
        state.close_db_connection()
        try:
            stats = estimate_bloat_ratio(config.db_path, config)
        except Exception:
            logger.exception("auto-compact: failed to estimate bloat ratio")
            return

        # Cache for CLI health notices regardless of whether we compact below.
        state._last_bloat_ratio = stats.ratio

        if stats.file_size < cfg.min_bytes:
            logger.debug(
                "auto-compact: file_size=%d below min_bytes=%d, skipping "
                "(ratio %.2f is block-granularity noise at this size)",
                stats.file_size,
                cfg.min_bytes,
                stats.ratio,
            )
            return

        if stats.ratio < cfg.bloat_ratio_threshold:
            logger.debug(
                "auto-compact: ratio=%.2f below threshold=%.2f, skipping",
                stats.ratio,
                cfg.bloat_ratio_threshold,
            )
            return

        should_reopen = True
        logger.info(
            "auto-compact: ratio=%.2f exceeds threshold=%.2f, running compact",
            stats.ratio,
            cfg.bloat_ratio_threshold,
        )
        try:
            result = compact(config)
            # Refresh the cached ratio to the post-compaction value so CLI health
            # notices stop warning about bloat the compaction just reclaimed
            # (otherwise the stale pre-compaction ratio persists until the next
            # check interval).
            state._last_bloat_ratio = result.after.ratio
            logger.info(
                "auto-compact: complete, %.2f -> %.2f",
                stats.ratio,
                result.after.ratio,
            )
        except CompactionError:
            logger.exception("auto-compact: compaction failed; daemon continuing")
        except Exception:
            logger.exception("auto-compact: unexpected failure; daemon continuing")
    finally:
        if should_reopen:
            try:
                state.reopen_db_connection()
            except Exception:
                logger.exception("auto-compact: failed to reopen DB connection")
                raise


def run_startup_snapshot_gc(config: AppConfig) -> None:
    """Prune stale snapshots once during daemon startup and stay quiet on no-op."""
    result = gc_snapshots(config)
    if result.removed_paths:
        logger.info(
            "snapshots gc removed %d entries (%d bytes) on daemon startup",
            len(result.removed_paths),
            result.total_bytes_freed,
        )
    if result.failed_paths:
        logger.warning(
            "snapshots gc could not remove %d entries on daemon startup: %s",
            len(result.failed_paths),
            ", ".join(result.failed_paths[:5]),
        )


def run_startup_fts_sidecar_sync(config: AppConfig) -> FtsSidecarStartupResult:
    """Bootstrap + reconcile the SQLite FTS5 sidecar on daemon startup.

    Under backend="sqlite_sidecar": opens the sidecar, runs bootstrap to
    completion, then runs a reconcile pass to drain pending writes and repair
    sidecar drift. Under backend="duckdb": returns a disabled no-op result.
    """
    if config.fts.backend == "duckdb":
        return _zero_fts_sidecar_startup_result(enabled=False)

    duckdb_conn: duckdb.DuckDBPyConnection | None = None
    sidecar_conn: Any = None
    sidecar_db_path = sidecar_path(config.data_dir)
    sidecar_existed = sidecar_db_path.exists()
    try:
        with advisory_lock(config.lock_path):
            duckdb_conn = connect(config)
            sidecar_conn = open_sidecar(sidecar_db_path)

            from recall.services.fts_sidecar_bootstrap import bootstrap_sidecar
            from recall.services.fts_sidecar_reconcile import reconcile_sidecar

            bootstrap = bootstrap_sidecar(
                duckdb_conn,
                sidecar_conn,
                fts_fields=config.fts.fields,
            )
            _assert_legacy_fts_schemas_present(duckdb_conn)
            _rescope_sidecar_fields_if_needed(
                duckdb_conn,
                sidecar_conn,
                fts_fields=config.fts.fields,
                sidecar_existed=sidecar_existed,
            )
            reconcile = reconcile_sidecar(
                duckdb_conn,
                sidecar_conn,
                fts_fields=config.fts.fields,
            )
    except FtsSidecarUnavailableError as err:
        message = f"SQLite FTS5 sidecar unavailable during daemon startup: {err}"
        logger.error(message)
        return _zero_fts_sidecar_startup_result(enabled=True, error=message)
    finally:
        if sidecar_conn is not None:
            sidecar_conn.close()
        if duckdb_conn is not None:
            duckdb_conn.close()

    return FtsSidecarStartupResult(
        enabled=True,
        bootstrap_messages_processed=bootstrap.messages_processed,
        bootstrap_tool_calls_processed=bootstrap.tool_calls_processed,
        bootstrap_messages_done=bootstrap.messages_done,
        bootstrap_tool_calls_done=bootstrap.tool_calls_done,
        reconcile_pending_drained=reconcile.pending_drained,
        reconcile_orphans_backfilled=reconcile.orphans_backfilled,
        reconcile_ghosts_deleted=reconcile.ghosts_deleted,
        reconcile_pending_remaining=reconcile.pending_remaining,
        last_run_at=datetime.now(UTC),
    )


def _zero_fts_sidecar_startup_result(
    *,
    enabled: bool,
    error: str | None = None,
) -> FtsSidecarStartupResult:
    return FtsSidecarStartupResult(
        enabled=enabled,
        bootstrap_messages_processed=0,
        bootstrap_tool_calls_processed=0,
        bootstrap_messages_done=False,
        bootstrap_tool_calls_done=False,
        reconcile_pending_drained=_zero_sidecar_kind_counts(),
        reconcile_orphans_backfilled=_zero_sidecar_kind_counts(),
        reconcile_ghosts_deleted=_zero_sidecar_kind_counts(),
        reconcile_pending_remaining=_zero_sidecar_kind_counts(),
        last_run_at=None,
        error=error,
    )


def _rescope_sidecar_fields_if_needed(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: Any,
    *,
    fts_fields: tuple[str, ...],
    sidecar_existed: bool,
) -> None:
    current_signature = fts_fields_signature(fts_fields)
    stored_signature = get_fts_fields_signature(sidecar_conn)
    if stored_signature == current_signature:
        return

    if stored_signature is None and not sidecar_existed:
        set_fts_fields_signature(sidecar_conn, fts_fields)
        return

    stats = rescope_sidecar_for_fields(
        duckdb_conn,
        sidecar_conn,
        fts_fields=fts_fields,
    )
    set_fts_fields_signature(sidecar_conn, fts_fields)
    logger.info(
        "FTS sidecar field scope changed from %r to %r; rewrote %d messages and %d tool calls",
        stored_signature,
        current_signature,
        stats.messages_rewritten,
        stats.tool_calls_rewritten,
    )


def fts_sidecar_startup_status_fields(
    result: FtsSidecarStartupResult | None,
) -> dict[str, Any]:
    if result is None:
        result = _zero_fts_sidecar_startup_result(enabled=False)
    return {
        "fts_sidecar_enabled": result.enabled,
        "fts_sidecar_bootstrap_messages_processed": result.bootstrap_messages_processed,
        "fts_sidecar_bootstrap_tool_calls_processed": result.bootstrap_tool_calls_processed,
        "fts_sidecar_bootstrap_messages_done": result.bootstrap_messages_done,
        "fts_sidecar_bootstrap_tool_calls_done": result.bootstrap_tool_calls_done,
        "fts_sidecar_reconcile_pending_drained_messages": _kind_count(
            result.reconcile_pending_drained, "message"
        ),
        "fts_sidecar_reconcile_pending_drained_tool_calls": _kind_count(
            result.reconcile_pending_drained, "tool_call"
        ),
        "fts_sidecar_reconcile_orphans_backfilled_messages": _kind_count(
            result.reconcile_orphans_backfilled, "message"
        ),
        "fts_sidecar_reconcile_orphans_backfilled_tool_calls": _kind_count(
            result.reconcile_orphans_backfilled, "tool_call"
        ),
        "fts_sidecar_reconcile_ghosts_deleted_messages": _kind_count(
            result.reconcile_ghosts_deleted, "message"
        ),
        "fts_sidecar_reconcile_ghosts_deleted_tool_calls": _kind_count(
            result.reconcile_ghosts_deleted, "tool_call"
        ),
        "fts_sidecar_reconcile_pending_remaining_messages": _kind_count(
            result.reconcile_pending_remaining, "message"
        ),
        "fts_sidecar_reconcile_pending_remaining_tool_calls": _kind_count(
            result.reconcile_pending_remaining, "tool_call"
        ),
        "fts_sidecar_last_run_at": result.last_run_at,
        "fts_sidecar_error": result.error,
    }


def _zero_sidecar_kind_counts() -> dict[str, int]:
    return {"message": 0, "tool_call": 0}


def _kind_count(counts: dict[str, int], kind: str) -> int:
    return int(counts.get(kind, 0))


def _assert_legacy_fts_schemas_present(conn: duckdb.DuckDBPyConnection) -> None:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.schemata
        WHERE schema_name LIKE 'fts_main_%'
        """
    ).fetchone()
    legacy_schema_count = int(row[0]) if row is not None else 0
    if legacy_schema_count > 0:
        return

    runtime_row = conn.execute(
        """
        SELECT last_successful_at
        FROM runtime_state
        """
    ).fetchone()
    if runtime_row is not None and runtime_row[0] is not None:
        logger.info(
            "legacy DuckDB FTS schemas are absent after sidecar bootstrap; "
            "this may mean indexing never created them or a sidecar-only "
            "migration already ran"
        )


def install_scheduler(
    *, scheduler: SchedulerKind | None, config: AppConfig | None = None
) -> DaemonSchedulerStatus:
    config = config or AppConfig.load()
    requested_scheduler = scheduler or config.daemon.scheduler
    effective_scheduler = _resolve_scheduler_kind(requested_scheduler, config)
    program_args = _scheduler_command(config)
    _verify_scheduler_program(program_args[0])
    command = _command_string(program_args)
    try:
        scope = resolve_service_scope(config, effective_scheduler)
    except ServiceLookupError as err:
        _authorize_lifecycle(config, "install", err)
        raise AssertionError("unreachable") from err
    capability = scope.authorize(config, "install")

    scheduler_config = replace(
        config,
        data_dir=config.data_dir.absolute(),
        db_path=config.db_path.absolute(),
        lock_path=config.lock_path.absolute(),
        config_path=config.config_path.absolute(),
    )
    match effective_scheduler:
        case SchedulerKind.LAUNCHD:
            _install_launchd(scheduler_config, program_args, command, capability)
        case SchedulerKind.SYSTEMD:
            _install_systemd(scheduler_config, program_args, command, capability)
        case SchedulerKind.CRON:
            _install_cron(scheduler_config, program_args, command, capability)
        case _:
            raise ValueError(f"unsupported scheduler: {effective_scheduler.value}")

    starts_daemon = effective_scheduler in {
        SchedulerKind.LAUNCHD,
        SchedulerKind.SYSTEMD,
    } and _is_watch_mode(config)
    return _update_installed_scheduler(config, effective_scheduler, require_rpc=starts_daemon)


def uninstall_scheduler(config: AppConfig | None = None) -> DaemonSchedulerStatus:
    config = config or AppConfig.load()
    try:
        scope = resolve_service_scope(config)
    except ServiceLookupError as err:
        _authorize_lifecycle(config, "uninstall", err)
        raise AssertionError("unreachable") from err
    capability = scope.authorize(config, "uninstall")
    record = capability.record
    if record.services:
        if record.scheduler == SchedulerKind.LAUNCHD:
            _uninstall_launchd(capability)
        elif record.scheduler == SchedulerKind.SYSTEMD:
            _uninstall_systemd(capability)
        elif record.scheduler == SchedulerKind.CRON:
            _uninstall_cron(capability)

    return _update_installed_scheduler(config, None)


@dataclass(frozen=True)
class _LocalReads:
    """What `daemon_status` could read from the database on the local path."""

    runtime_status: RuntimeStatus
    daemon_pid: int | None
    runtime_unavailable_reason: str | None
    # Set only when the counts were actually read, so an unread count is never
    # dated as fresh.
    pending: PendingSplit | None = None


def _read_beside_daemon(config: AppConfig, conn: duckdb.DuckDBPyConnection | None) -> _LocalReads:
    if conn is not None:
        return _LocalReads(
            load_runtime_status(config, conn=conn), None, None, _pending_split(config, conn)
        )
    try:
        result = read_offline(config, lambda owned: _read_beside_daemon(config, owned))
    except duckdb.IOException as err:
        if not is_lock_conflict(err):
            raise
        return _unavailable_reads(_held_by_other_process(err))
    if isinstance(result, SnapshotUnavailable):
        return _unavailable_reads(result.reason)
    return result


def _unavailable_reads(reason: str) -> _LocalReads:
    # A pid file proves neither daemon identity nor ownership of this snapshot.
    return _LocalReads(RuntimeStatus.unread(), None, reason)


def _held_by_other_process(err: duckdb.IOException) -> str:
    """Name a read-write holder that wrote no pid file: a foreground `recall index`,
    or a daemon between its launch and its pid-file write. DuckDB's message
    carries the holder's pid, which the operator previously saw in the raw error."""
    match = re.search(r"\(PID (\d+)\)", str(err))
    holder = f"another process (PID {match.group(1)})" if match else "another process"
    return (
        f"{holder} holds the database read-write, e.g. a foreground `recall index` "
        "or a daemon that has not written its pid file yet; runtime fields unavailable"
    )


def _source_page(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, *, limit: int, cursor: str | None
) -> SourcePage | None:
    from recall.db.source_files import SourceCatalog
    from recall.services.coordinator import source_page

    try:
        position = (
            SourceCatalog(conn, clock=time.time).issued_cursor(cursor)
            if cursor is not None
            else None
        )
        return source_page(config, conn=conn, limit=limit, cursor=position)
    except (duckdb.CatalogException, duckdb.BinderException) as err:
        logger.warning("source catalog coverage unavailable: %s", err)
        return None


def offline_daemon_status(
    config: AppConfig, *, limit: int, cursor: str | None
) -> tuple[DaemonSchedulerStatus, SourcePage | None]:
    """Detach runtime, pending and catalog readings from one owned snapshot."""
    from recall.db.source_files import CatalogCursor

    if cursor is not None:
        CatalogCursor.parse(cursor)

    def read(conn: duckdb.DuckDBPyConnection) -> tuple[_LocalReads, SourcePage | None]:
        return _read_beside_daemon(config, conn), _source_page(
            config, conn, limit=limit, cursor=cursor
        )

    try:
        result = read_offline(config, read)
    except duckdb.IOException as err:
        if not is_lock_conflict(err):
            raise
        result = SnapshotUnavailable(_held_by_other_process(err))
    if isinstance(result, SnapshotUnavailable):
        reads, page = _unavailable_reads(result.reason), None
    else:
        reads, page = result
    return _daemon_status(config, reads), page


def read_source_page(config: AppConfig, *, limit: int, cursor: str | None) -> SourcePage | None:
    """Read an offline catalog page under the same ownership as runtime status."""
    return offline_daemon_status(config, limit=limit, cursor=cursor)[1]


def _pending_split(config: AppConfig, conn: duckdb.DuckDBPyConnection) -> PendingSplit | None:
    try:
        return count_pending_split(conn, config.daemon.embed_idle_session)
    except duckdb.Error as err:
        logger.warning("pending embed counts unavailable: %s", err)
        return None


@dataclass(frozen=True)
class ServiceView:
    """What status reports about the installed service, as measured at `measured_at`."""

    # The persisted installed-scheduler record the view was measured with.
    installed_scheduler_record: SchedulerKind | None
    scheduler: SchedulerKind | None
    installed: bool
    service_data_dir: str | None
    measured_at: float
    artifact_paths: tuple[str, ...] = ()
    installed_command: str | None = None
    installed_mode: DaemonMode | None = None
    installed_binary_path: str | None = None
    installed_binary_stale: bool = False
    launchd_legacy_label: bool = False
    launchd_legacy_leftover: bool = False
    scheduler_last_exit_status: int | None = None
    scheduler_health_state: str | None = None


def probe_service(config: AppConfig, installed_scheduler: SchedulerKind | None) -> ServiceView:
    """Measure manager-owned identity; report another data dir without claiming its service."""
    try:
        scope = resolve_service_scope(config, installed_scheduler)
    except ServiceLookupError:
        return ServiceView(
            installed_scheduler, None, False, None, time.time(), scheduler_health_state="unknown"
        )
    service = scope.service
    reported_service = scope.other_service or service
    data_dir = str(reported_service.data_dir) if reported_service is not None else None
    if scope.other_service is not None:
        return ServiceView(installed_scheduler, None, False, data_dir, time.time())
    if service is None:
        return ServiceView(installed_scheduler, scope.record.scheduler, False, None, time.time())
    record = scope.identity_record or scope.record
    if (
        record.scheduler == SchedulerKind.SYSTEMD
        and service.mode == DaemonMode.POLL
        and not record.timer
    ):
        return ServiceView(
            installed_scheduler,
            record.scheduler,
            False,
            data_dir,
            time.time(),
            artifact_paths=tuple(str(path) for path in record.paths if path.exists()),
        )
    binary = _program_path(service.command, record.scheduler)
    try:
        last_exit, health = _detect_scheduler_health(ServiceCapability(record), config)
    except ValueError:
        last_exit, health = None, None
    return ServiceView(
        installed_scheduler_record=installed_scheduler,
        scheduler=record.scheduler,
        installed=True,
        service_data_dir=data_dir,
        measured_at=time.time(),
        artifact_paths=tuple(
            str(path)
            for path in (
                (service.path,) if record.scheduler == SchedulerKind.LAUNCHD else record.paths
            )
            if path is not None and path.exists()
        ),
        installed_command=service.command,
        installed_mode=service.mode,
        installed_binary_path=binary,
        installed_binary_stale=_installed_binary_stale(binary),
        launchd_legacy_label=service.name == LEGACY_LAUNCHD_LABEL,
        launchd_legacy_leftover=service.name == LAUNCHD_LABEL and len(record.services) > 1,
        scheduler_last_exit_status=last_exit,
        scheduler_health_state=health,
    )


# How old a daemon's served service measurement may grow before its refresher
# replaces it. The fields describe the job this daemon runs as: its last exit
# status changes only when that job exits, which ends this process, so the
# bound only delays out-of-band changes (a manual launchctl/systemctl edit).
SERVICE_PROBE_MAX_AGE_SECONDS = 30.0


class ServiceProbeCache:
    """The daemon's service view, refreshed off the request path (REQ-DAEMON-081).

    The probes run bounded manager queries off the request path. The cache
    reserves time for the next probe before its measurement
    reaches SERVICE_PROBE_MAX_AGE_SECONDS, and status serves the last one. A status
    whose persisted installed-scheduler record differs from the one the view was
    measured with measures inline, so install and uninstall show at once.

    With an on-time wake and stable clocks, a future deadline keeps the previous
    measurement within the age bound if the next probe fits the reserved time.
    An already-due deadline instead leaves only the age bound minus the current
    age and one-second retry delay; failures and longer stalls can exceed it.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._view: ServiceView | None = None
        self._refresh_at = time.monotonic()

    def refresh_delay(self) -> float:
        """Time until the cached measurement needs its next background probe.

        An overdue or failed probe retries after one second, avoiding a busy loop.
        Monotonic deadlines also account for time spent away from the monitor.
        """
        with self._lock:
            remaining = self._refresh_at - time.monotonic()
            return remaining if remaining > 0 else 1.0

    def view_for(self, installed_scheduler: SchedulerKind | None) -> ServiceView:
        with self._lock:
            view = self._view
        if view is not None and view.installed_scheduler_record == installed_scheduler:
            return view
        return self.refresh(installed_scheduler)

    def refresh(self, installed_scheduler: SchedulerKind | None) -> ServiceView:
        started_at = time.monotonic()
        view = probe_service(self._config, installed_scheduler)
        finished_at = time.monotonic()
        age = max(0.0, time.time() - view.measured_at)
        # Reserve at least half the window, adapting to the latest success so
        # a transiently slow probe cannot permanently increase the probe rate.
        reserve_seconds = max(SERVICE_PROBE_MAX_AGE_SECONDS / 2, 2 * (finished_at - started_at))
        with self._lock:
            self._refresh_at = finished_at + SERVICE_PROBE_MAX_AGE_SECONDS - age - reserve_seconds
            self._view = view
        return view

    def refresh_current(self) -> ServiceView | None:
        """Re-measure with the record of the view being served; None before the first one."""
        with self._lock:
            view = self._view
        if view is None:
            return None
        return self.refresh(view.installed_scheduler_record)


def daemon_status(
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
    *,
    fts_sidecar_startup: FtsSidecarStartupResult | None = None,
    service_probes: ServiceProbeCache | None = None,
) -> DaemonSchedulerStatus:
    config = config or AppConfig.load()
    return _daemon_status(
        config,
        _read_beside_daemon(config, conn),
        fts_sidecar_startup=fts_sidecar_startup,
        service_probes=service_probes,
    )


def _daemon_status(
    config: AppConfig,
    reads: _LocalReads,
    *,
    fts_sidecar_startup: FtsSidecarStartupResult | None = None,
    service_probes: ServiceProbeCache | None = None,
) -> DaemonSchedulerStatus:
    runtime_status = reads.runtime_status
    binary_version = _package_version()
    service = (
        service_probes.view_for(runtime_status.installed_scheduler)
        if service_probes is not None
        else probe_service(config, runtime_status.installed_scheduler)
    )
    installed_command = service.installed_command

    # Determine the effective mode: prefer the running daemon over the
    # installed scheduler, and prefer the installed scheduler over config.
    installed_mode = service.installed_mode
    runtime_mode = _detect_runtime_mode(config)
    try:
        config_mode = resolve_daemon_mode(config.daemon.mode)
    except RuntimeError:
        config_mode = DaemonMode.POLL
    effective_mode = runtime_mode or installed_mode or config_mode
    mode_mismatch_reason: str | None = None
    if runtime_mode is not None and installed_mode is not None and runtime_mode != installed_mode:
        mode_mismatch_reason = f"running={runtime_mode.value}, installed={installed_mode.value}"

    # What auto would resolve to given current deps (for mode mismatch notices)
    try:
        auto_mode = resolve_daemon_mode(DaemonMode.AUTO)
    except RuntimeError:
        auto_mode = DaemonMode.POLL

    # Prefer the installed command from the unit file over config-derived
    command = installed_command or _command_string(_scheduler_command(config))

    watched_dirs: tuple[str, ...] = ()
    live_session_count = 0
    live_session_paths: tuple[str, ...] = ()
    discovery_interval_seconds = 0.0
    discovery_last_run_at: datetime | None = None
    discovery_last_promoted = 0
    discovery_last_demoted = 0
    watcher_subscription_count = 0
    catchup_in_progress = False
    catchup_total = 0
    catchup_done = 0
    if effective_mode == DaemonMode.WATCH:
        from recall.parsers import all_parsers, get_parser
        from recall.services.watcher import get_live_snapshot

        snapshot = get_live_snapshot()
        # REQ-DAEMON-050: prefer the ACTUAL scheduled watches the running
        # watcher reports. Falls back to parser roots only when the snapshot
        # has no scheduled keys yet (watcher not started, pre-first-tick, or
        # empty live set). This keeps daemon status honest about what is
        # being observed, including during REQ-DAEMON-055 poll-fallback.
        if snapshot.scheduled_watch_keys:
            watched_dirs = tuple(snapshot.scheduled_watch_keys)
        else:
            effective_source = (
                _detect_installed_source(installed_command)
                if installed_command
                else config.daemon.source
            )
            source_parsers = (
                [get_parser(effective_source, config.sources)]
                if effective_source
                else all_parsers(config.sources)
            )
            dirs = []
            for parser in source_parsers:
                dirs.extend(str(root) for root in parser.watch_roots())
            watched_dirs = tuple(dirs)

        live_session_count = snapshot.live_session_count
        live_session_paths = tuple(snapshot.live_session_paths[:10])
        discovery_interval_seconds = snapshot.discovery_interval_seconds
        if discovery_interval_seconds <= 0.0 and snapshot.discovery_last_run_at is None:
            discovery_interval_seconds = float(config.daemon.live_discovery_interval)
        discovery_last_run_at = snapshot.discovery_last_run_at
        discovery_last_promoted = snapshot.discovery_last_promoted
        discovery_last_demoted = snapshot.discovery_last_demoted
        watcher_subscription_count = snapshot.watcher_subscription_count
        catchup_in_progress = snapshot.catchup_in_progress
        catchup_total = snapshot.catchup_total
        catchup_done = snapshot.catchup_done

    return DaemonSchedulerStatus(
        configured_scheduler=config.daemon.scheduler,
        scheduler=service.scheduler,
        installed=service.installed,
        command=command,
        config_path=str(config.config_path),
        artifact_paths=service.artifact_paths,
        runtime_status=runtime_status,
        startup_refusal=read_refusal_marker(config.data_dir),
        daemon_pid=reads.daemon_pid,
        runtime_unavailable_reason=reads.runtime_unavailable_reason,
        mode=config.daemon.mode,
        resolved_mode=effective_mode,
        auto_resolved_mode=auto_mode,
        installed_mode=installed_mode,
        mode_mismatch_reason=mode_mismatch_reason,
        watched_dirs=watched_dirs,
        debounce=config.daemon.debounce,
        fts_debounce=config.daemon.fts_debounce,
        embed_phase_enabled=config.daemon.embed,
        embed_model_loaded=False,
        # Without a running embed loop, the fresh eligible count stands in for
        # the loop's snapshot, so both status paths give `embed_pending` one meaning.
        embed_pending=reads.pending.eligible if reads.pending else 0,
        embed_pending_at=reads.pending.measured_at if reads.pending else None,
        embed_pending_eligible=reads.pending.eligible if reads.pending else None,
        embed_pending_held=reads.pending.held if reads.pending else None,
        embed_pending_measured_at=reads.pending.measured_at if reads.pending else None,
        embed_last_batch_at=None,
        embed_last_batch_size=0,
        **fts_sidecar_startup_status_fields(fts_sidecar_startup),
        installed_binary_path=service.installed_binary_path,
        installed_binary_stale=service.installed_binary_stale,
        launchd_legacy_label=service.launchd_legacy_label,
        launchd_legacy_leftover=service.launchd_legacy_leftover,
        scheduler_last_exit_status=service.scheduler_last_exit_status,
        scheduler_health_state=service.scheduler_health_state,
        service_data_dir=service.service_data_dir,
        service_measured_at=service.measured_at,
        live_session_count=live_session_count,
        live_session_paths=live_session_paths,
        discovery_interval_seconds=discovery_interval_seconds,
        discovery_last_run_at=discovery_last_run_at,
        discovery_last_promoted=discovery_last_promoted,
        discovery_last_demoted=discovery_last_demoted,
        watcher_subscription_count=watcher_subscription_count,
        catchup_in_progress=catchup_in_progress,
        catchup_total=catchup_total,
        catchup_done=catchup_done,
        daemon_version=None,
        binary_version=binary_version,
        version_drift=False,
    )


def fts_rebuild_backoff_status(
    debouncer: Any | None,
    *,
    now_mono: float | None = None,
) -> dict[str, Any]:
    if debouncer is None:
        return {
            "last_fts_rebuild_failure_at": None,
            "last_fts_rebuild_failure_reason": None,
            "fts_rebuild_consecutive_failures": 0,
            "fts_rebuild_next_retry_at": None,
        }

    last_failure_at = _datetime_from_timestamp(debouncer.last_oom_at_wall)
    next_retry_at = None
    next_retry_at_mono = debouncer.next_retry_at_mono
    if next_retry_at_mono is not None:
        now_mono = now_mono if now_mono is not None else time.monotonic()
        if now_mono < next_retry_at_mono:
            next_retry_at = _datetime_from_timestamp(debouncer.next_retry_at_wall)

    return {
        "last_fts_rebuild_failure_at": last_failure_at,
        "last_fts_rebuild_failure_reason": debouncer.last_oom_reason,
        "fts_rebuild_consecutive_failures": debouncer.oom_count,
        "fts_rebuild_next_retry_at": next_retry_at,
    }


def _datetime_from_timestamp(value: float | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value)


def _package_version() -> str | None:
    try:
        return version("recall")
    except PackageNotFoundError:
        return None


def _resolve_scheduler_kind(requested: SchedulerKind, config: AppConfig) -> SchedulerKind:
    platform = sys.platform
    if platform == "darwin":
        if requested in {SchedulerKind.AUTO, SchedulerKind.LAUNCHD}:
            return SchedulerKind.LAUNCHD
        raise ValueError(f"scheduler {requested.value} is not supported on macOS")
    if platform.startswith("linux"):
        if requested in {SchedulerKind.AUTO, SchedulerKind.SYSTEMD}:
            return SchedulerKind.SYSTEMD
        if requested == SchedulerKind.CRON:
            return SchedulerKind.CRON
        raise ValueError(f"scheduler {requested.value} is not supported on Linux")
    raise ValueError(f"unsupported platform for daemon scheduler management: {platform}")


def _scheduler_command(config: AppConfig) -> list[str]:
    recall_path = _resolve_recall_binary()
    try:
        effective_mode = resolve_daemon_mode(config.daemon.mode)
    except RuntimeError:
        effective_mode = DaemonMode.POLL
    if effective_mode == DaemonMode.WATCH:
        command = [recall_path, "daemon", "--mode", "watch"]
    else:
        command = [recall_path, "daemon", "--once"]
    if config.daemon.source is not None:
        command.extend(["--source", config.daemon.source.value])
    return command


def _resolve_recall_binary() -> str:
    return resolve_recall_binary()


def _verify_scheduler_program(program_path: str) -> None:
    path = Path(program_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"resolved recall binary does not exist: {path}")


def _restore_artifacts(snapshots: tuple[_ArtifactSnapshot, ...]) -> None:
    for snapshot in snapshots:
        snapshot.path.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.content is None:
            if snapshot.path.exists():
                snapshot.path.unlink()
            continue
        snapshot.path.write_bytes(snapshot.content)


def _post_install_smoke_test(program_args: list[str]) -> None:
    # `--help` is the one flag Typer guarantees on every command path: it
    # short-circuits before RPC/DB init, exits 0 on success, and a non-zero
    # exit here means the interpreter could not even import the CLI — exactly
    # the silent-failure condition launchd would otherwise hit at runtime.
    subprocess.run(
        [program_args[0], "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _launchd_service_target(label: str) -> str:
    return f"gui/{os.getuid()}/{label}"


def _launchd_plist_contents(config: AppConfig, command: str) -> str:
    stdout_path, stderr_path = _log_paths(config)
    environment_xml = "\n".join(
        [
            "    <key>EnvironmentVariables</key>",
            "    <dict>",
            *[
                f"      <key>{escape(key)}</key>\n      <string>{escape(value)}</string>"
                for key, value in _scheduler_environment(config).items()
            ],
            "    </dict>",
        ]
    )
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
            '<plist version="1.0">',
            "<dict>",
            "  <key>Label</key>",
            f"  <string>{LAUNCHD_LABEL}</string>",
            "  <key>ProgramArguments</key>",
            "  <array>",
            *[f"    <string>{escape(part)}</string>" for part in shlex.split(command)],
            "  </array>",
            *_launchd_schedule_keys(config),
            "  <key>RunAtLoad</key>",
            f"  <{'true' if _is_watch_mode(config) else 'false'}/>",
            # Without a ProcessType, launchd clamps the job to utility QoS and
            # writer turns queue behind foreground work under host load.
            # Interactive lifts the clamp; background work demotes its own
            # threads instead (`scheduling.demote_current_thread`).
            "  <key>ProcessType</key>",
            "  <string>Interactive</string>",
            "  <key>StandardOutPath</key>",
            f"  <string>{escape(str(stdout_path))}</string>",
            "  <key>StandardErrorPath</key>",
            f"  <string>{escape(str(stderr_path))}</string>",
            environment_xml,
            "</dict>",
            "</plist>",
            "",
        ]
    )


def _install_launchd(
    config: AppConfig, program_args: list[str], command: str, capability: ServiceCapability
) -> None:
    """Install the launch agent under LAUNCHD_LABEL, retiring a legacy install.

    Install is the only place the legacy label is retired. Every step after the
    snapshot runs under one restore path, so a failure at any point leaves the
    host as it was: the new job booted out and unloaded, both plists restored,
    and the job that was running re-bootstrapped. The legacy label is never
    `launchctl disable`d: that override persists and would block the rollback.
    """
    plist_path, legacy_plist_path = capability.record.paths
    snapshots = capability.snapshot_artifacts(config.data_dir)
    new_loaded = _launchd_label_is_loaded(LAUNCHD_LABEL)
    legacy_loaded = _launchd_label_is_loaded(LEGACY_LAUNCHD_LABEL)
    legacy_installed = legacy_loaded or legacy_plist_path.exists()
    domain = f"gui/{os.getuid()}"
    service_target = _launchd_service_target(LAUNCHD_LABEL)
    new_label_touched = False
    try:
        if legacy_installed:
            _retire_legacy_launchd_job(config, legacy_plist_path)
        new_label_touched = True
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        _create_log_dir(config)
        plist_path.write_text(_launchd_plist_contents(config, command), encoding="utf-8")
        # Clear any disabled override before loading: a disabled label refuses
        # the bootstrap.
        _run_command(["launchctl", "enable", service_target], check=True)
        _bootstrap_launchd_label(config, domain, plist_path)
        # Bootstrap exiting 0 is not proof the job took: verify before reporting
        # an install, so a silently unloaded service cannot pass as success.
        if not _launchd_label_is_loaded(LAUNCHD_LABEL):
            raise RuntimeError(
                f"launchd accepted the bootstrap but {LAUNCHD_LABEL} is not loaded; "
                f"the daemon is not running. Retry `recall daemon install`, and check "
                f"`launchctl print {service_target}`."
            )
        _post_install_smoke_test(program_args)
    except _LaunchdProcessExitTimeout:
        # The job can already be unregistered while its process holds DuckDB.
        # Restoring files is safe; restarting even the previous job is not.
        _restore_artifacts_or_log(snapshots)
        raise
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError):
        _roll_back_launchd_install(
            config,
            snapshots,
            new_label_touched=new_label_touched,
            new_was_loaded=new_loaded,
            legacy_was_loaded=legacy_loaded,
        )
        raise


def _retire_legacy_launchd_job(config: AppConfig, legacy_plist_path: Path) -> None:
    """Boot out the legacy job, wait for it to drain, then remove its plist.

    Bootstrapping the new label while the legacy job is still loaded would run
    two daemons against one database, so a legacy job that does not drain
    fails the install before anything is bootstrapped.
    """
    _bootout_launchd_label(config, LEGACY_LAUNCHD_LABEL)
    if not _wait_for_launchd_unload(LEGACY_LAUNCHD_LABEL):
        raise RuntimeError(
            f"legacy launch agent {LEGACY_LAUNCHD_LABEL} is still loaded after bootout; "
            f"nothing was installed. Check "
            f"`launchctl print {_launchd_service_target(LEGACY_LAUNCHD_LABEL)}` and retry "
            "`recall daemon install`."
        )
    legacy_plist_path.unlink(missing_ok=True)


class _LaunchdProcessExitTimeout(RuntimeError):
    """A booted-out job still has a live process; no job may be bootstrapped."""


def _bootout_launchd_label(config: AppConfig, label: str) -> None:
    target = _launchd_service_target(label)
    try:
        result = service_records.query(["launchctl", "print", target])
    except (OSError, ValueError, TimeoutError, subprocess.TimeoutExpired) as err:
        raise RuntimeError(f"cannot inspect launchd job before bootout: {err}") from err
    if result.returncode and f'Could not find service "{label}"' not in result.stderr:
        raise RuntimeError(f"cannot inspect launchd job before bootout: {result.stderr.strip()}")
    match = (
        re.search(r"^\s*pid = (\d+)\s*$", result.stdout, re.MULTILINE)
        if result is not None and result.returncode == 0
        else None
    )
    pid = int(match.group(1)) if match is not None else None
    if pid is None or pid <= 0:
        # A prior attempt may have unregistered the job before its process
        # released the database. The data directory retains that owner's PID.
        pid = _running_daemon_pid_from_file(config)
    with suppress(subprocess.CalledProcessError):
        _run_command(["launchctl", "bootout", target], check=True)
    if pid is None:
        return
    deadline = time.monotonic() + _LAUNCHD_UNLOAD_WAIT_SECONDS_MAX
    while _process_is_alive(pid):
        if time.monotonic() >= deadline:
            raise _LaunchdProcessExitTimeout(
                f"launch agent {label} pid={pid} is still alive after "
                f"{_LAUNCHD_UNLOAD_WAIT_SECONDS_MAX:g}s following bootout; "
                "no replacement was bootstrapped. Wait for that process to exit "
                "before retrying `recall daemon install`."
            )
        time.sleep(_LAUNCHD_UNLOAD_POLL_SECONDS)


def _bootstrap_launchd_label(config: AppConfig, domain: str, plist_path: Path) -> None:
    _bootout_launchd_label(config, LAUNCHD_LABEL)
    # bootout is asynchronous, so the previous instance can still be draining
    # here. Bootstrapping into that window fails with EBUSY, and a probe cannot
    # tell a dying service from a live one -- both answer `launchctl print`.
    # Wait it out, then retry, so neither state is guessed at.
    for attempt in range(1, _LAUNCHD_BOOTSTRAP_ATTEMPTS + 1):
        _wait_for_launchd_unload(LAUNCHD_LABEL)
        try:
            _run_command(["launchctl", "bootstrap", domain, str(plist_path)], check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == _LAUNCHD_BOOTSTRAP_ATTEMPTS:
                raise
            # KeepAlive can win the race and reload the old job. Evict it and
            # take the next attempt.
            _bootout_launchd_label(config, LAUNCHD_LABEL)


def _roll_back_launchd_install(
    config: AppConfig,
    snapshots: tuple[_ArtifactSnapshot, ...],
    *,
    new_label_touched: bool,
    new_was_loaded: bool,
    legacy_was_loaded: bool,
) -> None:
    """Undo a failed `_install_launchd` without ever loading both labels.

    `snapshots` is (new plist, legacy plist). The job that was loaded before
    install is re-bootstrapped: the new label when it was loaded, else the
    legacy label.
    """
    new_snapshot, legacy_snapshot = snapshots
    if new_label_touched:
        try:
            _bootout_launchd_label(config, LAUNCHD_LABEL)
        except RuntimeError as err:
            logger.error("could not restore launchd job after a failed install: %s", err)
            _restore_artifacts_or_log((legacy_snapshot,))
            return
        if not _wait_for_launchd_unload(LAUNCHD_LABEL):
            # The new job is still loaded: keep its plist so every command
            # resolves to it, and leave the legacy job unloaded beside it.
            logger.error(
                "%s is still loaded after a failed install; restored the %s plist "
                "without loading it",
                LAUNCHD_LABEL,
                LEGACY_LAUNCHD_LABEL,
            )
            _restore_artifacts_or_log((legacy_snapshot,))
            return
    if not _restore_artifacts_or_log(snapshots):
        return
    if new_label_touched and new_was_loaded and new_snapshot.content is not None:
        reload_path = new_snapshot.path
    elif (
        legacy_was_loaded
        and legacy_snapshot.content is not None
        and not _launchd_label_is_loaded(LEGACY_LAUNCHD_LABEL)
    ):
        reload_path = legacy_snapshot.path
    else:
        return
    with suppress(subprocess.CalledProcessError):
        _run_command(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(reload_path)], check=True)


def _restore_artifacts_or_log(snapshots: tuple[_ArtifactSnapshot, ...]) -> bool:
    try:
        _restore_artifacts(snapshots)
    except OSError:
        logger.exception("could not restore launchd plists after a failed install")
        return False
    return True


def _uninstall_launchd(capability: ServiceCapability) -> None:
    for service in capability.record.services:
        if service.loaded:
            try:
                _run_command(
                    ["launchctl", "bootout", _launchd_service_target(service.name)], check=True
                )
            except subprocess.CalledProcessError:
                if _launchd_label_is_loaded(service.name):
                    raise
    for path in capability.record.paths:
        path.unlink(missing_ok=True)


def _install_systemd(
    config: AppConfig,
    program_args: list[str],
    command: str,
    capability: ServiceCapability,
) -> None:
    artifact_paths = capability.record.paths
    service_path = artifact_paths[0]
    snapshots = capability.snapshot_artifacts(config.data_dir)
    service_path.parent.mkdir(parents=True, exist_ok=True)
    env_lines = "\n".join(
        f"Environment={key}={shlex.quote(value)}"
        for key, value in _scheduler_environment(config).items()
    )
    timer_path = artifact_paths[1]
    try:
        if _is_watch_mode(config):
            stdout_path, stderr_path = _log_paths(config)
            _create_log_dir(config)
            service_path.write_text(
                "\n".join(
                    [
                        "[Unit]",
                        "Description=recall daemon (watch mode)",
                        "",
                        "[Service]",
                        "Type=simple",
                        "Restart=always",
                        "RestartSec=5",
                        # Exit 3 is a refused start (REQ-RESIL-016): relaunching
                        # it every 5 s is the loop the refusal exists to end.
                        "RestartPreventExitStatus=3",
                        env_lines,
                        f"ExecStart={command}",
                        f"StandardOutput=append:{stdout_path}",
                        f"StandardError=append:{stderr_path}",
                        "",
                        "[Install]",
                        "WantedBy=default.target",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            # Remove stale timer if switching from poll to watch mode.
            if timer_path.exists():
                with suppress(subprocess.CalledProcessError):
                    _run_command(
                        ["systemctl", "--user", "disable", "--now", "recall-daemon.timer"],
                        check=True,
                    )
                timer_path.unlink()
            _run_command(["systemctl", "--user", "daemon-reload"], check=True)
            _run_command(["systemctl", "--user", "enable", "recall-daemon.service"], check=True)
            _run_command(["systemctl", "--user", "restart", "recall-daemon.service"], check=True)
        else:
            # Disable prior watch service if switching from watch to poll mode.
            with suppress(subprocess.CalledProcessError):
                _run_command(
                    ["systemctl", "--user", "disable", "--now", "recall-daemon.service"],
                    check=True,
                )
            service_path.write_text(
                "\n".join(
                    [
                        "[Unit]",
                        "Description=recall daemon",
                        "",
                        "[Service]",
                        "Type=oneshot",
                        env_lines,
                        f"ExecStart={command}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            timer_path.write_text(
                "\n".join(
                    [
                        "[Unit]",
                        "Description=Run recall daemon periodically",
                        "",
                        "[Timer]",
                        f"OnUnitActiveSec={config.daemon.interval}",
                        "Unit=recall-daemon.service",
                        "",
                        "[Install]",
                        "WantedBy=timers.target",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            _run_command(["systemctl", "--user", "daemon-reload"], check=True)
            _run_command(
                ["systemctl", "--user", "enable", "--now", "recall-daemon.timer"],
                check=True,
            )
        _post_install_smoke_test(program_args)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        with suppress(subprocess.CalledProcessError):
            _run_command(
                ["systemctl", "--user", "disable", "--now", "recall-daemon.service"], check=True
            )
        with suppress(subprocess.CalledProcessError):
            _run_command(
                ["systemctl", "--user", "disable", "--now", "recall-daemon.timer"],
                check=True,
            )
        with suppress(OSError):
            _restore_artifacts(snapshots)
        with suppress(subprocess.CalledProcessError):
            _run_command(["systemctl", "--user", "daemon-reload"], check=True)
        raise


def _uninstall_systemd(capability: ServiceCapability) -> None:
    for unit in _systemd_stop_units(capability):
        _run_command(["systemctl", "--user", "disable", "--now", unit], check=True)
    for path in capability.record.paths:
        path.unlink(missing_ok=True)
    _run_command(["systemctl", "--user", "daemon-reload"], check=True)


def _install_cron(
    config: AppConfig, program_args: list[str], command: str, capability: ServiceCapability
) -> None:
    schedule = _cron_schedule(config.daemon.interval)
    stdout_path, stderr_path = _log_paths(config)
    _create_log_dir(config)
    cron_command = (
        f"{_scheduler_env_prefix(config)}{command} >> {shlex.quote(str(stdout_path))} "
        f"2>> {shlex.quote(str(stderr_path))}"
    )
    block = "\n".join([CRON_BEGIN_MARKER, f"{schedule} {cron_command}", CRON_END_MARKER])
    existing = capability.record.crontab
    assert existing is not None
    updated = _replace_cron_block(existing, block)
    _write_crontab(updated)
    try:
        _post_install_smoke_test(program_args)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        with suppress(RuntimeError):
            _write_crontab(existing)
        raise


def _uninstall_cron(capability: ServiceCapability) -> None:
    existing = capability.record.crontab
    assert existing is not None
    updated = _remove_cron_block(existing)
    if updated != existing:
        _write_crontab(updated)


def _cron_schedule(interval_seconds: int) -> str:
    if interval_seconds < 60:
        raise ValueError("sub-minute daemon intervals are not supported by cron")
    if interval_seconds % 60 != 0:
        raise ValueError("cron requires a minute-aligned daemon interval")
    minutes = interval_seconds // 60
    if minutes < 60:
        return "* * * * *" if minutes == 1 else f"*/{minutes} * * * *"
    if minutes % 60 == 0:
        hours = minutes // 60
        if hours < 24:
            return f"0 */{hours} * * *"
        if hours % 24 == 0:
            days = hours // 24
            if days <= 31:
                return f"0 0 */{days} * *"
    raise ValueError("cron cannot express this daemon interval directly")


def _scheduler_environment(config: AppConfig) -> dict[str, str]:
    return {
        "RECALL_CONFIG_PATH": str(config.config_path),
        "RECALL_DATA_DIR": str(config.data_dir),
        "RECALL_DB_PATH": str(config.db_path),
        "RECALL_LOCK_PATH": str(config.lock_path),
    }


def _scheduler_env_prefix(config: AppConfig) -> str:
    return "".join(
        f"{key}={shlex.quote(value)} " for key, value in _scheduler_environment(config).items()
    )


def _is_watch_mode(config: AppConfig) -> bool:
    try:
        return resolve_daemon_mode(config.daemon.mode) == DaemonMode.WATCH
    except RuntimeError:
        return False


def _durable_scheduler(config: AppConfig) -> SchedulerKind | None:
    if sys.platform == "darwin" and config.daemon.scheduler in {
        SchedulerKind.AUTO,
        SchedulerKind.LAUNCHD,
    }:
        return SchedulerKind.LAUNCHD
    if sys.platform.startswith("linux") and config.daemon.scheduler in {
        SchedulerKind.AUTO,
        SchedulerKind.SYSTEMD,
    }:
        return SchedulerKind.SYSTEMD
    return None


def _running_daemon_pid_from_file(config: AppConfig) -> int | None:
    pid_path = config.data_dir / "recall.pid"
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if pid > 0 and _process_is_alive(pid) else None


def _process_is_alive(pid: int) -> bool:
    """Whether `pid` is a running process; an exited one its parent has not reaped is not."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return not _process_is_zombie(pid)


def _process_is_zombie(pid: int) -> bool:
    """`kill(pid, 0)` succeeds for a zombie, so ask for the process state itself."""
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return False
        # The state follows the parenthesized command name, which may hold spaces.
        return stat.rpartition(")")[2].split()[:1] == ["Z"]
    result = _run_lifecycle_query(["ps", "-o", "stat=", "-p", str(pid)])
    return result is not None and result.stdout.strip().startswith("Z")


def _cleanup_daemon_runtime_files(config: AppConfig) -> None:
    (config.data_dir / "recall.sock").unlink(missing_ok=True)
    (config.data_dir / "recall.pid").unlink(missing_ok=True)


def _elapsed(started_at: float) -> float:
    return max(0.0, time.monotonic() - started_at)


def _wait_for_durable_stop(
    config: AppConfig,
    *,
    scheduler: str,
    pid: int | None,
    timeout: float,
    started_at: float,
    scheduler_stopped: Callable[[], bool],
) -> DaemonStopResult:
    deadline = time.monotonic() + timeout
    pid_path = config.data_dir / "recall.pid"
    while time.monotonic() < deadline:
        scheduler_is_stopped = scheduler_stopped()
        process_stopped = pid is None or not _process_is_alive(pid)
        pidfile_released, removed_stale_pidfile = _release_pidfile_if_stale(pid_path)
        if process_stopped and pidfile_released and scheduler_is_stopped:
            return DaemonStopResult(
                scheduler=scheduler,
                stopped=True,
                pid=pid,
                duration_seconds=_elapsed(started_at),
                message="removed stale daemon pidfile" if removed_stale_pidfile else None,
            )
        time.sleep(0.1)
    return DaemonStopResult(
        scheduler=scheduler,
        stopped=False,
        pid=pid,
        duration_seconds=_elapsed(started_at),
        message=f"daemon still alive or scheduler still loaded after {timeout:.1f}s",
    )


def _wait_until_stopped(
    config: AppConfig,
    *,
    pid: int,
    timeout: float,
    started_at: float,
) -> DaemonStopResult:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_is_alive(pid):
            _cleanup_daemon_runtime_files(config)
            return DaemonStopResult(
                scheduler="pid",
                stopped=True,
                pid=pid,
                duration_seconds=_elapsed(started_at),
                message=None,
            )
        time.sleep(0.1)
    return DaemonStopResult(
        scheduler="pid",
        stopped=False,
        pid=pid,
        duration_seconds=_elapsed(started_at),
        message=f"daemon pid={pid} still alive after {timeout:.1f}s",
    )


def _wait_until_started(
    config: AppConfig,
    *,
    scheduler: str,
    timeout: float,
    started_at: float,
    scheduler_started: Callable[[], bool],
    expect_live_daemon: bool,
) -> DaemonStartResult:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pid = _running_daemon_pid_from_file(config)
        if pid is not None:
            return DaemonStartResult(
                scheduler=scheduler,
                started=True,
                pid=pid,
                duration_seconds=_elapsed(started_at),
                message=None,
            )
        if scheduler_started() and not expect_live_daemon:
            return DaemonStartResult(
                scheduler=scheduler,
                started=True,
                pid=None,
                duration_seconds=_elapsed(started_at),
                message="scheduler started",
            )
        time.sleep(0.1)
    return DaemonStartResult(
        scheduler=scheduler,
        started=False,
        pid=None,
        duration_seconds=_elapsed(started_at),
        message=f"daemon did not become ready within {timeout:.1f}s",
    )


def _wait_for_launchd_unload(
    label: str,
    timeout_seconds: float = _LAUNCHD_UNLOAD_WAIT_SECONDS_MAX,
) -> bool:
    """Block until launchd finishes tearing the `label` job down.

    Returns False if the job is still loaded at the deadline; the caller
    decides whether that is fatal.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        if not _launchd_label_is_loaded(label):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_LAUNCHD_UNLOAD_POLL_SECONDS)


def _launchd_label_is_loaded(label: str) -> bool:
    try:
        result = service_records.query(["launchctl", "print", _launchd_service_target(label)])
    except (OSError, ValueError, TimeoutError, subprocess.TimeoutExpired) as err:
        raise RuntimeError(f"could not inspect launchd job {label}: {err}") from err
    if result.returncode == 0:
        return True
    if f'Could not find service "{label}"' in result.stderr:
        return False
    raise RuntimeError(f"could not inspect launchd job {label}: {result.stderr.strip()}")


def _systemd_stop_units(capability: ServiceCapability) -> tuple[str, ...]:
    return (("recall-daemon.timer",) if capability.record.timer else ()) + (
        "recall-daemon.service",
    )


def _systemd_start_unit(capability: ServiceCapability) -> str:
    service = capability.record.service
    assert service is not None
    return (
        "recall-daemon.timer"
        if service.mode == DaemonMode.POLL and capability.record.timer
        else service.name
    )


def _systemd_unit_active(unit: str) -> bool:
    try:
        result = service_records.query(["systemctl", "--user", "is-active", "--quiet", unit])
    except (OSError, ValueError, TimeoutError, subprocess.TimeoutExpired) as err:
        raise RuntimeError(f"could not inspect systemd unit {unit}: {err}") from err
    if result.returncode in {0, 3}:
        return result.returncode == 0
    raise RuntimeError(f"could not inspect systemd unit {unit}: {result.stderr.strip()}")


def _run_lifecycle_query(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return service_records.query(args)
    except (OSError, ValueError, TimeoutError, subprocess.TimeoutExpired):
        return None


def _run_lifecycle_command(args: list[str], action: str) -> None:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as err:
        raise RuntimeError(f"could not {action}: {err}") from err
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"could not {action}{suffix}")


def _pidfile_pid_is(pid_path: Path, expected_pid: int) -> bool:
    try:
        content = pid_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return False
    try:
        return int(content) == expected_pid
    except ValueError:
        return False


def _release_pidfile_if_stale(pid_path: Path) -> tuple[bool, bool]:
    """Return whether the pidfile no longer points at a live daemon.

    Durable scheduler stops can leave pidfiles behind when launchd/systemd has
    already killed the process. A non-live or invalid pidfile should not block a
    successful stop; unlink it so subsequent lifecycle operations see the same
    state as a clean daemon exit.
    """
    try:
        content = pid_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return True, False
    except OSError as err:
        logger.warning("daemon lifecycle: failed to read pidfile %s: %s", pid_path, err)
        return False, False

    try:
        pid = int(content)
    except ValueError:
        removed = _unlink_stale_pidfile(pid_path)
        return removed, removed

    if pid <= 0 or not _process_is_alive(pid):
        removed = _unlink_stale_pidfile(pid_path)
        return removed, removed
    return False, False


def _unlink_stale_pidfile(pid_path: Path) -> bool:
    try:
        pid_path.unlink(missing_ok=True)
    except OSError as err:
        logger.warning("daemon lifecycle: failed to unlink stale pidfile %s: %s", pid_path, err)
        return False
    return True


def _running_daemon_pid(config: AppConfig) -> int | None:
    """Return the running daemon PID when the RPC socket and pidfile are live."""
    socket_path = config.data_dir / "recall.sock"
    pid_path = config.data_dir / "recall.pid"
    if not socket_path.exists() or not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except (OSError, PermissionError, ProcessLookupError):
        return None
    return pid


def _detect_runtime_mode(config: AppConfig) -> DaemonMode | None:
    """Infer watch mode from the live daemon process plus watcher snapshot state.

    REQ-DAEMON-060: must return WATCH during the startup window on an idle
    host, before the first discovery tick runs and before any live session
    seeds. `runtime_started_at` is set exactly once in
    `start_live_watch_runtime` and cleared on shutdown, which is the
    unambiguous "watch runtime is live" signal. The discovery/count fallbacks
    remain for hosts where the daemon predates this field.
    """
    if _running_daemon_pid(config) is None:
        return None

    from recall.services.watcher import get_live_snapshot

    snapshot = get_live_snapshot()
    if snapshot.runtime_started_at is not None:
        return DaemonMode.WATCH
    if snapshot.discovery_last_run_at is not None:
        return DaemonMode.WATCH
    if snapshot.live_session_count > 0 or snapshot.watcher_subscription_count > 0:
        return DaemonMode.WATCH
    return None


def _program_path(command: str | None, scheduler: SchedulerKind) -> str | None:
    if command is None:
        return None
    parts = shlex.split(command)
    if scheduler == SchedulerKind.CRON:
        while parts and "=" in parts[0] and not parts[0].startswith("/"):
            parts.pop(0)
    return parts[0] if parts else None


def _installed_binary_stale(installed_binary_path: str | None) -> bool:
    if installed_binary_path is None:
        return False
    installed_path = Path(installed_binary_path).expanduser()
    if not installed_path.exists():
        return True
    try:
        resolved_binary = Path(_resolve_recall_binary()).expanduser()
    except (OSError, RuntimeError):
        return True
    # Both sides may be symlinks. Under uv's tool layout `~/.local/bin/recall`
    # is a symlink into `~/.local/share/uv/tools/recall/bin/recall`; under
    # launchd the daemon's `shutil.which("recall")` returns None (minimal PATH)
    # and `resolve_recall_binary()` falls through to argv[0].resolve(), which
    # is the symlink target. Comparing unresolved paths flagged that mismatch
    # as drift even though it was the same binary. Canonicalize before
    # comparing so genuine drift (different file) still trips the check.
    return installed_path.resolve() != resolved_binary.resolve()


def _detect_installed_source(command: str) -> Source | None:
    """Parse --source flag from an installed command string."""
    parts = shlex.split(command)
    for i, part in enumerate(parts):
        if part == "--source" and i + 1 < len(parts):
            try:
                return Source(parts[i + 1])
            except ValueError:
                return None
    return None


def _launchd_schedule_keys(config: AppConfig) -> list[str]:
    if _is_watch_mode(config):
        from recall.services.self_repair import refusal_marker_path

        # launchd cannot filter on exit status, so a refused start (exit 3,
        # REQ-RESIL-016) leaves a marker and KeepAlive holds only while that
        # path is absent; clearing the memory relaunches (REQ-RESIL-024).
        return [
            "  <key>KeepAlive</key>",
            "  <dict>",
            "    <key>PathState</key>",
            "    <dict>",
            f"      <key>{escape(str(refusal_marker_path(config.data_dir)))}</key>",
            "      <false/>",
            "    </dict>",
            "  </dict>",
        ]
    return [
        "  <key>StartInterval</key>",
        f"  <integer>{config.daemon.interval}</integer>",
    ]


def _log_paths(config: AppConfig) -> tuple[Path, Path]:
    log_dir = config.data_dir / "logs"
    return log_dir / "daemon.log", log_dir / "daemon.err.log"


def _create_log_dir(config: AppConfig) -> None:
    create_private_dir(config.data_dir)
    create_private_dir(_log_paths(config)[0].parent)


def _write_crontab(contents: str) -> None:
    if shutil.which("crontab") is None:
        raise RuntimeError("crontab is not available on this host")
    result = subprocess.run(
        ["crontab", "-"],
        input=contents,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "failed to update crontab")


def _replace_cron_block(existing: str, block: str) -> str:
    stripped = _remove_cron_block(existing).rstrip()
    if not stripped:
        return f"{block}\n"
    return f"{stripped}\n{block}\n"


def _remove_cron_block(existing: str) -> str:
    lines = existing.splitlines()
    filtered: list[str] = []
    inside_block = False
    for line in lines:
        if line == CRON_BEGIN_MARKER:
            inside_block = True
            continue
        if line == CRON_END_MARKER:
            inside_block = False
            continue
        if not inside_block:
            filtered.append(line)
    cleaned = "\n".join(line for line in filtered if line.strip())
    return f"{cleaned}\n" if cleaned else ""


def _detect_scheduler_health(
    capability: ServiceCapability, config: AppConfig
) -> tuple[int | None, str | None]:
    if capability.record.scheduler == SchedulerKind.LAUNCHD:
        service = capability.record.service
        assert service is not None
        return _detect_launchd_health(service.name)
    if capability.record.scheduler == SchedulerKind.SYSTEMD:
        return _detect_systemd_health(capability)
    return None, None


_LAUNCHCTL_LAST_EXIT_STATUS = re.compile(r'"LastExitStatus"\s*=\s*(-?\d+)\s*;')


def _detect_launchd_health(label: str) -> tuple[int | None, str | None]:
    result = _run_lifecycle_query(["launchctl", "list", label])
    if result is None or result.returncode != 0:
        return None, None
    parsed = _parse_launchctl_list_output(result.stdout)
    if parsed is not None:
        return parsed

    if result.stdout.strip():
        print_result = _run_lifecycle_query(["launchctl", "print", _launchd_service_target(label)])
        if print_result is not None and print_result.returncode == 0:
            fallback = _parse_launchctl_print_output(print_result.stdout)
            if fallback is not None:
                return fallback
    return None, "unknown"


def _parse_launchctl_list_output(stdout: str) -> tuple[int | None, str | None] | None:
    """Read `LastExitStatus` out of `launchctl list <label>`.

    Asked about one label, launchctl answers with a plist-style dictionary
    (`"LastExitStatus" = 0;`); only the bare `launchctl list` prints the tabular
    PID/Status/Label listing. The `launchctl print` fallback cannot stand in for
    this: a job that is still running reports `last exit code = (never exited)`.
    """
    match = _LAUNCHCTL_LAST_EXIT_STATUS.search(stdout)
    if match is None:
        return None
    last_exit_status = int(match.group(1))
    if last_exit_status == 0:
        return 0, "ok"
    return last_exit_status, "failed"


def _parse_launchctl_print_output(stdout: str) -> tuple[int | None, str | None] | None:
    for line in stdout.splitlines():
        stripped = line.strip().lower()
        if "last exit code" not in stripped:
            continue
        _, _, value = stripped.rpartition("=")
        value = value.strip()
        try:
            exit_status = int(value)
        except ValueError:
            return None
        if exit_status == 0:
            return 0, "ok"
        return exit_status, "failed"
    return None


def _detect_systemd_health(capability: ServiceCapability) -> tuple[int | None, str | None]:
    service = _run_lifecycle_query(
        [
            "systemctl",
            "--user",
            "show",
            "recall-daemon.service",
            "--property=Result,ExecMainStatus",
            "--no-pager",
        ]
    )
    if service is None or service.returncode != 0:
        return None, None
    properties = service_records.parse_systemd_properties(service.stdout)
    raw_result = properties.get("Result")
    exec_status = _parse_int(properties.get("ExecMainStatus"))
    health_state = _systemd_health_state(raw_result, exec_status)

    if capability.record.timer:
        timer = _run_lifecycle_query(
            [
                "systemctl",
                "--user",
                "show",
                "recall-daemon.timer",
                "--property=Result",
                "--no-pager",
            ]
        )
        if timer is not None and timer.returncode == 0:
            timer_result = service_records.parse_systemd_properties(timer.stdout).get("Result")
            if timer_result not in {None, "", "success"} and health_state in {None, "ok"}:
                health_state = timer_result

    return exec_status, health_state


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _systemd_health_state(result: str | None, exec_status: int | None) -> str | None:
    if result in {None, ""}:
        if exec_status is None:
            return None
        return "ok" if exec_status == 0 else "failed"
    if result == "success" and exec_status in {None, 0}:
        return "ok"
    if exec_status not in {None, 0}:
        return result or "failed"
    return result


def _command_string(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _run_command(args: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, capture_output=True, text=True, timeout=30)


def _connect_scheduler_client(config: AppConfig) -> RpcClient | None:
    client = RpcClient(config)
    try:
        client.connect(auto_fork=False)
    except RpcAdmissionBusyError as err:
        client.close()
        raise RuntimeError(err.message) from err
    except RpcConnectionError:
        client.close()
        return None
    return client


def _scheduler_status_from_rpc(client: RpcClient) -> DaemonSchedulerStatus:
    try:
        payload = client.call("recall.daemon_status")
    except (RpcCallError, RpcConnectionError) as err:
        raise RuntimeError(f"could not read scheduler status: {err.message}") from err
    try:
        return TypeAdapter(DaemonSchedulerStatus).validate_python(payload)
    except ValidationError as err:
        raise RuntimeError(f"daemon returned invalid scheduler status: {err}") from err


def _scheduler_management_status(config: AppConfig) -> DaemonSchedulerStatus:
    client = _connect_scheduler_client(config)
    if client is None:
        return daemon_status(config=config)
    try:
        return _scheduler_status_from_rpc(client)
    finally:
        client.close()


def _update_installed_scheduler(
    config: AppConfig, scheduler: SchedulerKind | None, *, require_rpc: bool = False
) -> DaemonSchedulerStatus:
    # Scheduler activation may already have handed DuckDB to the new daemon.
    # Never auto-fork here: an offline management command must stay offline.
    deadline = time.monotonic() + _SCHEDULER_DATABASE_WAIT_SECONDS_MAX
    while True:
        client = _connect_scheduler_client(config)
        if client is not None:
            try:
                try:
                    client.call(
                        "recall._set_installed_scheduler",
                        {"scheduler": scheduler.value if scheduler is not None else None},
                    )
                except (RpcCallError, RpcConnectionError) as err:
                    raise RuntimeError(
                        f"could not persist installed scheduler: {err.message}"
                    ) from err
                status = _scheduler_status_from_rpc(client)
                if status.runtime_status.installed_scheduler != scheduler:
                    raise RuntimeError("daemon did not persist the requested installed scheduler")
                return status
            finally:
                client.close()

        if require_rpc:
            # Bootstrap can return before the daemon opens DuckDB. Taking the
            # apparently free database here can make that daemon fail startup.
            if _running_daemon_pid_from_file(config) is not None:
                status = daemon_status(config=config)
                if status.installed and status.runtime_unavailable_reason is not None:
                    return status
            now = time.monotonic()
            if now >= deadline:
                # A unit can own DuckDB before writing its pid file. At the
                # readiness deadline only, distinguish that lock from a unit
                # that never started without racing the startup window.
                if config.db_path.exists():
                    status = daemon_status(config=config)
                    if status.installed and status.runtime_unavailable_reason is not None:
                        return status
                raise RuntimeError(
                    "daemon did not become ready after scheduler activation; "
                    f"could not persist installed scheduler via {config.data_dir / 'recall.sock'}"
                )
            time.sleep(min(_SCHEDULER_DATABASE_POLL_SECONDS, deadline - now))
            continue

        try:
            conn = connect(config)
        except duckdb.IOException as err:
            # The process can own the database before its RPC socket is ready,
            # or retain it briefly after shutdown removed the socket.
            if not is_lock_conflict(err):
                raise
            now = time.monotonic()
            if now >= deadline:
                return daemon_status(config=config)
            time.sleep(min(_SCHEDULER_DATABASE_POLL_SECONDS, deadline - now))
            continue
        try:
            set_installed_scheduler(conn, scheduler)
            return daemon_status(config=config, conn=conn)
        finally:
            conn.close()
