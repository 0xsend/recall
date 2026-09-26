from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
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

from recall.core.config import AppConfig, create_private_dir, resolve_recall_binary
from recall.core.rpc_client import RpcCallError, RpcClient, RpcConnectionError
from recall.core.types import DaemonMode, SchedulerKind, Source
from recall.db import (
    FtsSidecarUnavailableError,
    advisory_lock,
    connect,
    connect_readonly,
    is_lock_conflict,
    open_sidecar,
    sidecar_path,
)
from recall.db.fts_sidecar import (
    fts_fields_signature,
    get_fts_fields_signature,
    set_fts_fields_signature,
)
from recall.services.compaction import CompactionError, compact, estimate_bloat_ratio
from recall.services.fts_sidecar_reconcile import rescope_sidecar_for_fields
from recall.services.indexer import IndexSummary
from recall.services.runtime_state import (
    RuntimeStatus,
    load_runtime_status,
    set_installed_scheduler,
)
from recall.services.self_repair import read_refusal_marker
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
class DaemonSchedulerStatus:
    configured_scheduler: SchedulerKind
    scheduler: SchedulerKind | None
    installed: bool
    command: str
    config_path: str
    artifact_paths: tuple[str, ...]
    runtime_status: RuntimeStatus
    mode: DaemonMode = DaemonMode.AUTO
    resolved_mode: DaemonMode | None = None
    # What resolve_daemon_mode(AUTO) produces given current deps (e.g. watchdog)
    auto_resolved_mode: DaemonMode | None = None
    installed_mode: DaemonMode | None = None
    mode_mismatch_reason: str | None = None
    watched_dirs: tuple[str, ...] = ()
    debounce: int = 5
    fts_debounce: int = 10
    embed_phase_enabled: bool = False
    embed_model_loaded: bool = False
    embed_pending: int = 0
    # When the pending count was measured; a deferred cycle freezes the count
    # without looking again, so the age is what dates it (REQ-ADAPT-012).
    embed_pending_at: float | None = None
    embed_last_batch_at: float | None = None
    embed_last_batch_size: int = 0
    embed_last_batch_duration: float = 0.0
    # Embed loop liveness: a stalled loop and an idle loop both freeze the batch
    # fields above, so the stage it last entered is what makes a stall
    # diagnosable without process introspection (REQ-ADAPT-012).
    embed_loop_iterations: int = 0
    embed_loop_last_iteration_at: float | None = None
    # Which path drove the last cycle: the phase timer, or a requested
    # (index/recompute) enrichment that commits through the same state.
    embed_loop_last_trigger: str | None = None
    embed_loop_stage: str | None = None
    embed_loop_stage_at: float | None = None
    embed_loop_last_outcome: str | None = None
    embed_loop_next_interval: float = 0.0
    # Requested enrichment runs while the timer sleeps in its own stage, so its
    # cycles are counted and staged apart from the timer's (REQ-ADAPT-012).
    embed_requested_cycles: int = 0
    embed_requested_at: float | None = None
    embed_requested_stage: str | None = None
    embed_requested_stage_at: float | None = None
    # Why the last cycle deferred and when that was evaluated, beside the
    # outcome they explain. The reason is also served as
    # `reconciliation.enrichment_deferred` for existing readers.
    embed_deferred_reason: str | None = None
    embed_deferred_at: float | None = None
    embed_last_error: str | None = None
    # Sessions deprioritized after repeated no-progress embed cycles, and the
    # wall-clock time the last cooldown expires (REQ-ADAPT-016).
    embed_cooldown_sessions: int = 0
    embed_cooldown_until: float | None = None
    # Watch index metrics (live overlay from RPC server)
    watch_total_indexed: int = 0
    watch_total_failed: int = 0
    watch_avg_duration: float | None = None
    watch_min_duration: float | None = None
    watch_max_duration: float | None = None
    watch_last_event_at: float | None = None
    watch_last_event_duration: float | None = None
    watch_recent_events: tuple[dict[str, Any], ...] = ()
    # FTS sidecar (REQ-FTS-SIDECAR-007, -008)
    fts_sidecar_enabled: bool = False
    fts_sidecar_bootstrap_messages_processed: int = 0
    fts_sidecar_bootstrap_tool_calls_processed: int = 0
    fts_sidecar_bootstrap_messages_done: bool = False
    fts_sidecar_bootstrap_tool_calls_done: bool = False
    fts_sidecar_reconcile_pending_drained_messages: int = 0
    fts_sidecar_reconcile_pending_drained_tool_calls: int = 0
    fts_sidecar_reconcile_orphans_backfilled_messages: int = 0
    fts_sidecar_reconcile_orphans_backfilled_tool_calls: int = 0
    fts_sidecar_reconcile_ghosts_deleted_messages: int = 0
    fts_sidecar_reconcile_ghosts_deleted_tool_calls: int = 0
    fts_sidecar_reconcile_pending_remaining_messages: int = 0
    fts_sidecar_reconcile_pending_remaining_tool_calls: int = 0
    fts_sidecar_last_run_at: datetime | None = None
    fts_sidecar_error: str | None = None
    last_fts_rebuild_failure_at: datetime | None = None
    last_fts_rebuild_failure_reason: str | None = None
    fts_rebuild_consecutive_failures: int = 0
    fts_rebuild_next_retry_at: datetime | None = None
    installed_binary_path: str | None = None
    installed_binary_stale: bool = False
    # The launch agent is installed under LEGACY_LAUNCHD_LABEL; lifecycle
    # commands act on it until `recall daemon install` migrates it.
    launchd_legacy_label: bool = False
    # Installed under LAUNCHD_LABEL, but a legacy plist or loaded legacy job
    # remains (an interrupted migration); `recall daemon install` retires it.
    launchd_legacy_leftover: bool = False
    scheduler_last_exit_status: int | None = None
    scheduler_health_state: str | None = None
    live_session_count: int = 0
    live_session_paths: tuple[str, ...] = ()
    # REQ-LIVE-011: how much `--fresh` has actually cost, so `live.fresh_timeout`
    # is tuned against observed timeouts rather than a guess.
    live_fresh_requests: int = 0
    live_fresh_timeouts: int = 0
    # REQ-LIVE-011: `--follow` clients currently attached. A vanished follower
    # that leaked its slot shows up here as a count that never comes down.
    follow_subscriptions: int = 0
    discovery_interval_seconds: float = 0.0
    discovery_last_run_at: datetime | None = None
    discovery_last_promoted: int = 0
    discovery_last_demoted: int = 0
    watcher_subscription_count: int = 0
    catchup_in_progress: bool = False
    catchup_total: int = 0
    catchup_done: int = 0
    daemon_version: str | None = None
    binary_version: str | None = None
    version_drift: bool = False
    # Last bloat ratio the daemon estimated (startup + each auto-compact check),
    # so the CLI can warn about a bloated database before it degrades queries.
    bloat_ratio: float | None = None
    bloat_ratio_threshold: float | None = None
    bloat_auto_trigger: bool = False
    # Most recent index/table divergence probe (`IndexDivergenceReport.to_payload()`),
    # cached by the daemon at startup and on each `recall.check_indexes` call
    # (REQ-RESIL-018). None when no daemon has probed yet.
    index_divergence: dict[str, Any] | None = None
    # Message left by a refused start (`<data_dir>/daemon-refused`); while it
    # exists the scheduler will not relaunch the daemon, so the status the
    # operator can still run must say why (REQ-RESIL-024).
    startup_refusal: str | None = None
    # REQ-DAEMON-074: set on the local path when a live daemon holds the
    # database; the runtime fields above are then defaults, not readings.
    daemon_pid: int | None = None
    runtime_unavailable_reason: str | None = None


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


def stop_daemon_soft(config: AppConfig | None = None) -> DaemonStopResult:
    """Preserve the legacy PID-file shutdown path without touching scheduler state."""
    config = config or AppConfig.load()
    started_at = time.monotonic()
    pid = _running_daemon_pid_from_file(config)
    if pid is None:
        _cleanup_daemon_runtime_files(config)
        return DaemonStopResult(
            scheduler="sentinel",
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
            scheduler="sentinel",
            stopped=True,
            pid=pid,
            duration_seconds=_elapsed(started_at),
            message="daemon process was already gone",
        )
    except PermissionError as err:
        return DaemonStopResult(
            scheduler="sentinel",
            stopped=False,
            pid=pid,
            duration_seconds=_elapsed(started_at),
            message=f"cannot signal daemon pid={pid}: {err}",
        )

    result = _wait_until_stopped(config, pid=pid, timeout=10.0, started_at=started_at)
    if result.stopped:
        _cleanup_daemon_runtime_files(config)
        return replace(result, scheduler="sentinel")
    return replace(result, scheduler="sentinel")


def stop_daemon_durable(
    config: AppConfig | None = None, *, timeout: float = 10.0
) -> DaemonStopResult:
    """Stop the installed user scheduler and wait for the daemon process to exit."""
    config = config or AppConfig.load()
    started_at = time.monotonic()
    scheduler = _durable_scheduler(config)
    scheduler_name = scheduler.value if scheduler is not None else "pid"
    pid = _running_daemon_pid_from_file(config)

    if scheduler == SchedulerKind.LAUNCHD:
        label = _active_launchd_label()
        leftover_labels = _leftover_legacy_launchd_labels(label)
        for leftover in leftover_labels:
            # A legacy job still loaded beside the installed one would keep a
            # daemon running after stop; the wait below reports it if it stays.
            with suppress(RuntimeError):
                _run_lifecycle_command(
                    ["launchctl", "bootout", _launchd_service_target(leftover)],
                    "stop legacy launchd daemon",
                )
        try:
            _run_lifecycle_command(
                ["launchctl", "bootout", _launchd_service_target(label)], "stop launchd daemon"
            )
        except RuntimeError as err:
            unit_loaded = _launchd_unit_is_loaded() or any(
                _launchd_label_is_loaded(leftover) for leftover in leftover_labels
            )
            if pid is None and not unit_loaded:
                return DaemonStopResult(
                    scheduler=scheduler_name,
                    stopped=True,
                    pid=None,
                    duration_seconds=_elapsed(started_at),
                    message="daemon already stopped",
                )
            # A live daemon that launchd does not manage cannot be stopped
            # through launchctl at all -- boot-out reports "No such process"
            # while the process keeps holding the database. Say so, because
            # launchd's own error points nowhere.
            message = (
                (
                    f"daemon pid={pid} is running but launchd does not manage it "
                    f"(orphaned; {label} is not loaded). Stop it with "
                    f"`kill {pid}`, then run `recall daemon install` to reinstate "
                    f"the launch agent."
                )
                if pid is not None and not unit_loaded
                else str(err)
            )
            return DaemonStopResult(
                scheduler=scheduler_name,
                stopped=False,
                pid=pid,
                duration_seconds=_elapsed(started_at),
                message=message,
            )
        return _wait_for_durable_stop(
            config,
            scheduler=scheduler_name,
            pid=pid,
            timeout=timeout,
            started_at=started_at,
            scheduler_stopped=lambda: (
                not _launchd_unit_is_loaded()
                and not any(_launchd_label_is_loaded(leftover) for leftover in leftover_labels)
            ),
        )

    if scheduler == SchedulerKind.SYSTEMD:
        try:
            for unit in _systemd_stop_units(config):
                _run_lifecycle_command(
                    ["systemctl", "--user", "stop", unit],
                    f"stop systemd {unit}",
                )
        except RuntimeError as err:
            if pid is None and not _systemd_any_lifecycle_unit_active(config):
                return DaemonStopResult(
                    scheduler=scheduler_name,
                    stopped=True,
                    pid=None,
                    duration_seconds=_elapsed(started_at),
                    message="daemon already stopped",
                )
            return DaemonStopResult(
                scheduler=scheduler_name,
                stopped=False,
                pid=pid,
                duration_seconds=_elapsed(started_at),
                message=str(err),
            )
        return _wait_for_durable_stop(
            config,
            scheduler=scheduler_name,
            pid=pid,
            timeout=timeout,
            started_at=started_at,
            scheduler_stopped=lambda: not _systemd_any_lifecycle_unit_active(config),
        )

    if pid is None:
        _cleanup_daemon_runtime_files(config)
        return DaemonStopResult(
            scheduler=scheduler_name,
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
            scheduler=scheduler_name,
            stopped=True,
            pid=pid,
            duration_seconds=_elapsed(started_at),
            message="daemon process was already gone",
        )
    except PermissionError as err:
        return DaemonStopResult(
            scheduler=scheduler_name,
            stopped=False,
            pid=pid,
            duration_seconds=_elapsed(started_at),
            message=f"cannot signal daemon pid={pid}: {err}",
        )
    return _wait_until_stopped(config, pid=pid, timeout=timeout, started_at=started_at)


def start_daemon_durable(
    config: AppConfig | None = None, *, timeout: float = 10.0
) -> DaemonStartResult:
    """Start the installed user scheduler and wait until the daemon is live."""
    config = config or AppConfig.load()
    started_at = time.monotonic()
    already_running_pid = _running_daemon_pid_from_file(config)
    scheduler = _durable_scheduler(config)
    scheduler_name = scheduler.value if scheduler is not None else "pid"
    if already_running_pid is not None:
        return DaemonStartResult(
            scheduler=scheduler_name,
            started=True,
            pid=already_running_pid,
            duration_seconds=_elapsed(started_at),
            message="daemon already running",
        )

    if _scheduler_already_loaded(scheduler, config):
        return _wait_for_already_loaded_scheduler(
            config,
            scheduler=scheduler,
            scheduler_name=scheduler_name,
            timeout=timeout,
            started_at=started_at,
        )

    if scheduler == SchedulerKind.LAUNCHD:
        plist_path = _artifact_paths(config, SchedulerKind.LAUNCHD)[0]
        if not plist_path.exists():
            return DaemonStartResult(
                scheduler=scheduler_name,
                started=False,
                pid=None,
                duration_seconds=_elapsed(started_at),
                message=f"launchd plist is not installed: {plist_path}",
            )
        try:
            _run_lifecycle_command(
                ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)],
                "start launchd daemon",
            )
        except RuntimeError as err:
            if _scheduler_already_loaded(scheduler, config):
                return _wait_for_already_loaded_scheduler(
                    config,
                    scheduler=scheduler,
                    scheduler_name=scheduler_name,
                    timeout=timeout,
                    started_at=started_at,
                )
            return DaemonStartResult(
                scheduler=scheduler_name,
                started=False,
                pid=None,
                duration_seconds=_elapsed(started_at),
                message=str(err),
            )
        return _wait_until_started(
            config,
            scheduler=scheduler_name,
            timeout=timeout,
            started_at=started_at,
            scheduler_started=_launchd_unit_is_loaded,
            expect_live_daemon=_scheduler_expects_live_daemon(
                config, scheduler=SchedulerKind.LAUNCHD
            ),
        )

    if scheduler == SchedulerKind.SYSTEMD:
        unit = _systemd_start_unit(config)
        if unit is None:
            return DaemonStartResult(
                scheduler=scheduler_name,
                started=False,
                pid=None,
                duration_seconds=_elapsed(started_at),
                message="systemd service is not installed",
            )
        try:
            _run_lifecycle_command(
                ["systemctl", "--user", "start", unit],
                f"start systemd {unit}",
            )
        except RuntimeError as err:
            if _scheduler_already_loaded(scheduler, config):
                return _wait_for_already_loaded_scheduler(
                    config,
                    scheduler=scheduler,
                    scheduler_name=scheduler_name,
                    timeout=timeout,
                    started_at=started_at,
                )
            return DaemonStartResult(
                scheduler=scheduler_name,
                started=False,
                pid=None,
                duration_seconds=_elapsed(started_at),
                message=str(err),
            )
        return _wait_until_started(
            config,
            scheduler=scheduler_name,
            timeout=timeout,
            started_at=started_at,
            scheduler_started=lambda: _systemd_unit_active(unit),
            expect_live_daemon=_scheduler_expects_live_daemon(
                config, scheduler=SchedulerKind.SYSTEMD
            ),
        )

    return DaemonStartResult(
        scheduler=scheduler_name,
        started=False,
        pid=None,
        duration_seconds=_elapsed(started_at),
        message="no durable scheduler is configured on this platform",
    )


def restart_daemon_durable(
    config: AppConfig | None = None, *, timeout: float = 10.0
) -> DaemonRestartResult:
    config = config or AppConfig.load()
    started_at = time.monotonic()
    stop_result = stop_daemon_durable(config, timeout=timeout)
    if not stop_result.stopped:
        return DaemonRestartResult(
            scheduler=stop_result.scheduler,
            stopped=False,
            started=False,
            pid=stop_result.pid,
            duration_seconds=_elapsed(started_at),
            message=stop_result.message,
        )
    start_result = start_daemon_durable(config, timeout=timeout)
    return DaemonRestartResult(
        scheduler=start_result.scheduler,
        stopped=True,
        started=start_result.started,
        pid=start_result.pid,
        duration_seconds=_elapsed(started_at),
        message=start_result.message,
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


@dataclass(frozen=True)
class _ArtifactSnapshot:
    path: Path
    content: bytes | None


def install_scheduler(
    *, scheduler: SchedulerKind | None, config: AppConfig | None = None
) -> DaemonSchedulerStatus:
    config = config or AppConfig.load()
    requested_scheduler = scheduler or config.daemon.scheduler
    effective_scheduler = _resolve_scheduler_kind(requested_scheduler, config)
    program_args = _scheduler_command(config)
    _verify_scheduler_program(program_args[0])
    command = _command_string(program_args)

    match effective_scheduler:
        case SchedulerKind.LAUNCHD:
            _install_launchd(config, program_args, command)
        case SchedulerKind.SYSTEMD:
            _install_systemd(
                config, program_args, command, _artifact_paths(config, effective_scheduler)
            )
        case SchedulerKind.CRON:
            _install_cron(config, program_args, command)
        case _:
            raise ValueError(f"unsupported scheduler: {effective_scheduler.value}")

    starts_daemon = effective_scheduler in {
        SchedulerKind.LAUNCHD,
        SchedulerKind.SYSTEMD,
    } and _is_watch_mode(config)
    return _update_installed_scheduler(config, effective_scheduler, require_rpc=starts_daemon)


def uninstall_scheduler(config: AppConfig | None = None) -> DaemonSchedulerStatus:
    config = config or AppConfig.load()
    status = _scheduler_management_status(config)
    scheduler = status.scheduler

    if scheduler == SchedulerKind.LAUNCHD:
        _uninstall_launchd()
    elif scheduler == SchedulerKind.SYSTEMD:
        _uninstall_systemd(_artifact_paths(config, SchedulerKind.SYSTEMD))
    elif scheduler == SchedulerKind.CRON:
        _uninstall_cron()

    return _update_installed_scheduler(config, None)


@dataclass(frozen=True)
class _LocalReads:
    """What `daemon_status` could read from the database on the local path."""

    runtime_status: RuntimeStatus
    embed_pending: int
    daemon_pid: int | None
    runtime_unavailable_reason: str | None
    # Set only when the count was actually queried, so an unread or skipped
    # count is never dated as fresh.
    embed_pending_at: float | None = None


def _read_beside_daemon(
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection | None,
    *,
    skip_embed_pending: bool,
) -> _LocalReads:
    """Read runtime_state and the pending-embed count without opening the file under a live daemon.

    DuckDB's lock is exclusive to a read-write holder, so while a daemon holds
    the database the runtime fields are only served by its RPC; the local path
    reports the holder instead of raising (REQ-DAEMON-074). The pid file is
    written before the daemon opens the database, so it is checked first; a
    lock conflict from a lost race (the daemon relaunched between the check
    and the open) is reported the same way. A caller that passes `conn` is
    the daemon itself and reads directly.
    """
    holder_pid = None if conn is not None else _running_daemon_pid_from_file(config)
    if holder_pid is not None:
        return _LocalReads(RuntimeStatus.unread(), 0, holder_pid, _held_by_daemon(holder_pid))
    try:
        runtime_status = load_runtime_status(config, conn=conn)
    except duckdb.IOException as err:
        if conn is not None or not is_lock_conflict(err):
            raise
        holder_pid = _running_daemon_pid_from_file(config)
        if holder_pid is not None:
            return _LocalReads(RuntimeStatus.unread(), 0, holder_pid, _held_by_daemon(holder_pid))
        return _LocalReads(RuntimeStatus.unread(), 0, None, _held_by_other_process(err))
    if skip_embed_pending:
        return _LocalReads(runtime_status, 0, None, None)
    return _LocalReads(
        runtime_status,
        _pending_embed_count(config, conn),
        None,
        None,
        embed_pending_at=time.time(),
    )


def _held_by_daemon(holder_pid: int) -> str:
    return f"daemon pid {holder_pid} holds the database; runtime fields are served by its RPC"


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


def _pending_embed_count(config: AppConfig, conn: duckdb.DuckDBPyConnection | None) -> int:
    """Pending embed count (REQ-ADAPT-012); the RPC path skips this because the
    live EmbedPhaseState cache already has it."""
    try:
        from recall.services.embed_phase import find_pending_embeds

        owned_conn = conn is None
        status_conn = conn or connect_readonly(config)
        try:
            return find_pending_embeds(status_conn, idle_threshold=0).total
        finally:
            if owned_conn:
                status_conn.close()
    except Exception:
        return 0


def daemon_status(
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
    *,
    skip_embed_pending: bool = False,
    fts_sidecar_startup: FtsSidecarStartupResult | None = None,
) -> DaemonSchedulerStatus:
    config = config or AppConfig.load()
    reads = _read_beside_daemon(config, conn, skip_embed_pending=skip_embed_pending)
    runtime_status = reads.runtime_status
    binary_version = _package_version()
    resolved_scheduler = _status_scheduler(config, runtime_status)
    installed = resolved_scheduler is not None and _scheduler_installed(config, resolved_scheduler)
    installed_binary_path = (
        _detect_installed_program_path(config, resolved_scheduler) if installed else None
    )
    installed_binary_stale = _installed_binary_stale(installed_binary_path)
    launchd_installed = installed and resolved_scheduler == SchedulerKind.LAUNCHD
    active_launchd_label = _active_launchd_label() if launchd_installed else None
    launchd_legacy_label = active_launchd_label == LEGACY_LAUNCHD_LABEL
    launchd_legacy_leftover = active_launchd_label == LAUNCHD_LABEL and (
        _launchd_plist_path(LEGACY_LAUNCHD_LABEL).exists()
        or _launchd_label_is_loaded(LEGACY_LAUNCHD_LABEL)
    )
    scheduler_last_exit_status, scheduler_health_state = (
        _detect_scheduler_health(resolved_scheduler, config) if installed else (None, None)
    )

    # Determine the effective mode: prefer the running daemon over the
    # installed scheduler, and prefer the installed scheduler over config.
    installed_mode = _detect_installed_mode(config, resolved_scheduler) if installed else None
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

    # Filter artifact_paths to only existing files
    all_paths = (
        _artifact_paths(config, resolved_scheduler) if resolved_scheduler is not None else tuple()
    )
    existing_paths = tuple(p for p in all_paths if p.exists())

    # Prefer the installed command from the unit file over config-derived
    installed_command = _detect_installed_command(config, resolved_scheduler) if installed else None
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
        scheduler=resolved_scheduler,
        installed=installed,
        command=command,
        config_path=str(config.config_path),
        artifact_paths=tuple(str(path) for path in existing_paths),
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
        embed_pending=reads.embed_pending,
        embed_pending_at=reads.embed_pending_at,
        embed_last_batch_at=None,
        embed_last_batch_size=0,
        **fts_sidecar_startup_status_fields(fts_sidecar_startup),
        installed_binary_path=installed_binary_path,
        installed_binary_stale=installed_binary_stale,
        launchd_legacy_label=launchd_legacy_label,
        launchd_legacy_leftover=launchd_legacy_leftover,
        scheduler_last_exit_status=scheduler_last_exit_status,
        scheduler_health_state=scheduler_health_state,
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
        if requested == SchedulerKind.AUTO:
            if _systemd_user_available():
                return SchedulerKind.SYSTEMD
            raise ValueError(
                "systemd --user is unavailable on this host; use --scheduler cron explicitly"
            )
        if requested == SchedulerKind.SYSTEMD:
            if not _systemd_user_available():
                raise ValueError("systemd --user is unavailable on this host")
            return SchedulerKind.SYSTEMD
        if requested == SchedulerKind.CRON:
            return SchedulerKind.CRON
        raise ValueError(f"scheduler {requested.value} is not supported on Linux")
    raise ValueError(f"unsupported platform for daemon scheduler management: {platform}")


def _systemd_user_available() -> bool:
    if shutil.which("systemctl") is None:
        return False
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


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


def _artifact_paths(config: AppConfig, scheduler: SchedulerKind) -> tuple[Path, ...]:
    home = Path.home()
    if scheduler == SchedulerKind.LAUNCHD:
        return (_launchd_plist_path(_active_launchd_label()),)
    if scheduler == SchedulerKind.SYSTEMD:
        base = home / ".config" / "systemd" / "user"
        return (base / "recall-daemon.service", base / "recall-daemon.timer")
    if scheduler == SchedulerKind.CRON:
        return (
            config.data_dir / "logs" / "daemon.log",
            config.data_dir / "logs" / "daemon.err.log",
        )
    raise ValueError(f"unsupported scheduler: {scheduler.value}")


def _verify_scheduler_program(program_path: str) -> None:
    path = Path(program_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"resolved recall binary does not exist: {path}")


def _snapshot_artifacts(paths: tuple[Path, ...]) -> tuple[_ArtifactSnapshot, ...]:
    snapshots: list[_ArtifactSnapshot] = []
    for path in paths:
        content = path.read_bytes() if path.exists() else None
        snapshots.append(_ArtifactSnapshot(path=path, content=content))
    return tuple(snapshots)


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


def _launchd_plist_path(label: str) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def _launchd_service_target(label: str) -> str:
    return f"gui/{os.getuid()}/{label}"


def _leftover_legacy_launchd_labels(active_label: str) -> tuple[str, ...]:
    """Legacy jobs still loaded beside the installed label (an interrupted migration)."""
    if active_label == LAUNCHD_LABEL and _launchd_label_is_loaded(LEGACY_LAUNCHD_LABEL):
        return (LEGACY_LAUNCHD_LABEL,)
    return ()


def _active_launchd_label() -> str:
    """The launchd label every lifecycle command acts on.

    An install made under the legacy label keeps working until `recall daemon
    install` migrates it, so every command resolves the label here instead of
    splitting between two jobs. A legacy job whose plist is gone but which is
    still loaded counts too: it is the daemon that is running.
    """
    if _launchd_plist_path(LAUNCHD_LABEL).exists():
        return LAUNCHD_LABEL
    if _launchd_plist_path(LEGACY_LAUNCHD_LABEL).exists() or _launchd_label_is_loaded(
        LEGACY_LAUNCHD_LABEL
    ):
        return LEGACY_LAUNCHD_LABEL
    return LAUNCHD_LABEL


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


def _install_launchd(config: AppConfig, program_args: list[str], command: str) -> None:
    """Install the launch agent under LAUNCHD_LABEL, retiring a legacy install.

    Install is the only place the legacy label is retired. Every step after the
    snapshot runs under one restore path, so a failure at any point leaves the
    host as it was: the new job booted out and unloaded, both plists restored,
    and the job that was running re-bootstrapped. The legacy label is never
    `launchctl disable`d: that override persists and would block the rollback.
    """
    plist_path = _launchd_plist_path(LAUNCHD_LABEL)
    legacy_plist_path = _launchd_plist_path(LEGACY_LAUNCHD_LABEL)
    snapshots = _snapshot_artifacts((plist_path, legacy_plist_path))
    new_loaded = _launchd_label_is_loaded(LAUNCHD_LABEL)
    legacy_loaded = _launchd_label_is_loaded(LEGACY_LAUNCHD_LABEL)
    legacy_installed = legacy_loaded or legacy_plist_path.exists()
    domain = f"gui/{os.getuid()}"
    service_target = _launchd_service_target(LAUNCHD_LABEL)
    new_label_touched = False
    try:
        if legacy_installed:
            _retire_legacy_launchd_job(legacy_plist_path)
        new_label_touched = True
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        _create_log_dir(config)
        plist_path.write_text(_launchd_plist_contents(config, command), encoding="utf-8")
        # Clear any disabled override before loading: a disabled label refuses
        # the bootstrap.
        _run_command(["launchctl", "enable", service_target], check=True)
        _bootstrap_launchd_label(domain, plist_path)
        # Bootstrap exiting 0 is not proof the job took: verify before reporting
        # an install, so a silently unloaded service cannot pass as success.
        if not _launchd_label_is_loaded(LAUNCHD_LABEL):
            raise RuntimeError(
                f"launchd accepted the bootstrap but {LAUNCHD_LABEL} is not loaded; "
                f"the daemon is not running. Retry `recall daemon install`, and check "
                f"`launchctl print {service_target}`."
            )
        _post_install_smoke_test(program_args)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError):
        _roll_back_launchd_install(
            snapshots,
            new_label_touched=new_label_touched,
            new_was_loaded=new_loaded,
            legacy_was_loaded=legacy_loaded,
        )
        raise


def _retire_legacy_launchd_job(legacy_plist_path: Path) -> None:
    """Boot out the legacy job, wait for it to drain, then remove its plist.

    Bootstrapping the new label while the legacy job is still loaded would run
    two daemons against one database, so a legacy job that does not drain
    fails the install before anything is bootstrapped.
    """
    with suppress(subprocess.CalledProcessError):
        _run_command(
            ["launchctl", "bootout", _launchd_service_target(LEGACY_LAUNCHD_LABEL)], check=True
        )
    if not _wait_for_launchd_unload(LEGACY_LAUNCHD_LABEL):
        raise RuntimeError(
            f"legacy launch agent {LEGACY_LAUNCHD_LABEL} is still loaded after bootout; "
            f"nothing was installed. Check "
            f"`launchctl print {_launchd_service_target(LEGACY_LAUNCHD_LABEL)}` and retry "
            "`recall daemon install`."
        )
    legacy_plist_path.unlink(missing_ok=True)


def _bootstrap_launchd_label(domain: str, plist_path: Path) -> None:
    service_target = _launchd_service_target(LAUNCHD_LABEL)
    with suppress(subprocess.CalledProcessError):
        _run_command(["launchctl", "bootout", service_target], check=True)
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
            with suppress(subprocess.CalledProcessError):
                _run_command(["launchctl", "bootout", service_target], check=True)


def _roll_back_launchd_install(
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
        with suppress(subprocess.CalledProcessError):
            _run_command(
                ["launchctl", "bootout", _launchd_service_target(LAUNCHD_LABEL)], check=True
            )
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


def _uninstall_launchd() -> None:
    for label in (LAUNCHD_LABEL, LEGACY_LAUNCHD_LABEL):
        with suppress(subprocess.CalledProcessError):
            _run_command(["launchctl", "bootout", _launchd_service_target(label)], check=True)
        _launchd_plist_path(label).unlink(missing_ok=True)


def _install_systemd(
    config: AppConfig,
    program_args: list[str],
    command: str,
    artifact_paths: tuple[Path, ...],
) -> None:
    service_path = artifact_paths[0]
    snapshots = _snapshot_artifacts(artifact_paths)
    service_path.parent.mkdir(parents=True, exist_ok=True)
    env_lines = "\n".join(
        f"Environment={key}={shlex.quote(value)}"
        for key, value in _scheduler_environment(config).items()
    )
    timer_path = service_path.parent / "recall-daemon.timer"
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


def _uninstall_systemd(artifact_paths: tuple[Path, ...]) -> None:
    # Try disabling both service and timer; one or both may be active depending on mode
    with suppress(subprocess.CalledProcessError):
        _run_command(
            ["systemctl", "--user", "disable", "--now", "recall-daemon.service"], check=True
        )
    with suppress(subprocess.CalledProcessError):
        _run_command(["systemctl", "--user", "disable", "--now", "recall-daemon.timer"], check=True)
    # Remove all artifact files plus any stale timer from a previous poll install
    for path in artifact_paths:
        if path.exists():
            path.unlink()
    if artifact_paths:
        timer_path = artifact_paths[0].parent / "recall-daemon.timer"
        if timer_path.exists() and timer_path not in artifact_paths:
            timer_path.unlink()
    with suppress(subprocess.CalledProcessError):
        _run_command(["systemctl", "--user", "daemon-reload"], check=True)


def _install_cron(config: AppConfig, program_args: list[str], command: str) -> None:
    schedule = _cron_schedule(config.daemon.interval)
    stdout_path, stderr_path = _log_paths(config)
    _create_log_dir(config)
    cron_command = (
        f"{_scheduler_env_prefix(config)}{command} >> {shlex.quote(str(stdout_path))} "
        f"2>> {shlex.quote(str(stderr_path))}"
    )
    block = "\n".join([CRON_BEGIN_MARKER, f"{schedule} {cron_command}", CRON_END_MARKER])
    existing = _read_crontab()
    updated = _replace_cron_block(existing, block)
    _write_crontab(updated)
    try:
        _post_install_smoke_test(program_args)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        with suppress(RuntimeError):
            _write_crontab(existing)
        raise


def _uninstall_cron() -> None:
    existing = _read_crontab()
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
    return pid if _process_is_alive(pid) else None


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


def _scheduler_expects_live_daemon(config: AppConfig, *, scheduler: SchedulerKind) -> bool:
    installed_mode = _detect_installed_mode(config, scheduler)
    if installed_mode is not None:
        return installed_mode == DaemonMode.WATCH
    try:
        return resolve_daemon_mode(config.daemon.mode) == DaemonMode.WATCH
    except RuntimeError:
        return True


def _scheduler_already_loaded(scheduler: SchedulerKind | None, config: AppConfig) -> bool:
    if scheduler == SchedulerKind.LAUNCHD:
        return _launchd_unit_is_loaded()
    if scheduler == SchedulerKind.SYSTEMD:
        unit = _systemd_start_unit(config)
        return unit is not None and _systemd_unit_active(unit)
    return False


def _wait_for_already_loaded_scheduler(
    config: AppConfig,
    *,
    scheduler: SchedulerKind | None,
    scheduler_name: str,
    timeout: float,
    started_at: float,
) -> DaemonStartResult:
    if scheduler == SchedulerKind.LAUNCHD:
        result = _wait_until_started(
            config,
            scheduler=scheduler_name,
            timeout=timeout,
            started_at=started_at,
            scheduler_started=_launchd_unit_is_loaded,
            expect_live_daemon=_scheduler_expects_live_daemon(
                config, scheduler=SchedulerKind.LAUNCHD
            ),
        )
    elif scheduler == SchedulerKind.SYSTEMD:
        unit = _systemd_start_unit(config)
        result = _wait_until_started(
            config,
            scheduler=scheduler_name,
            timeout=timeout,
            started_at=started_at,
            scheduler_started=lambda: unit is not None and _systemd_unit_active(unit),
            expect_live_daemon=_scheduler_expects_live_daemon(
                config, scheduler=SchedulerKind.SYSTEMD
            ),
        )
    else:
        return DaemonStartResult(
            scheduler=scheduler_name,
            started=False,
            pid=None,
            duration_seconds=_elapsed(started_at),
            message="no durable scheduler is configured on this platform",
        )

    if result.started:
        return replace(result, message="already loaded")
    return replace(result, message="loaded but not responsive")


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


def _launchd_unit_is_loaded() -> bool:
    return _launchd_label_is_loaded(_active_launchd_label())


def _launchd_label_is_loaded(label: str) -> bool:
    print_result = _run_lifecycle_query(["launchctl", "print", _launchd_service_target(label)])
    if print_result is not None and print_result.returncode == 0:
        return True
    list_result = _run_lifecycle_query(["launchctl", "list", label])
    return list_result is not None and list_result.returncode == 0


def _systemd_stop_units(config: AppConfig) -> tuple[str, ...]:
    service_path, timer_path = _artifact_paths(config, SchedulerKind.SYSTEMD)
    units: list[str] = []
    if timer_path.exists():
        units.append("recall-daemon.timer")
    if service_path.exists() or not units:
        units.append("recall-daemon.service")
    return tuple(units)


def _systemd_start_unit(config: AppConfig) -> str | None:
    service_path, timer_path = _artifact_paths(config, SchedulerKind.SYSTEMD)
    if (
        _detect_installed_mode(config, SchedulerKind.SYSTEMD) == DaemonMode.POLL
        and timer_path.exists()
    ):
        return "recall-daemon.timer"
    if service_path.exists():
        return "recall-daemon.service"
    return None


def _systemd_any_lifecycle_unit_active(config: AppConfig) -> bool:
    return any(_systemd_unit_active(unit) for unit in _systemd_stop_units(config))


def _systemd_unit_active(unit: str) -> bool:
    result = _run_lifecycle_query(["systemctl", "--user", "is-active", "--quiet", unit])
    return result is not None and result.returncode == 0


def _run_lifecycle_query(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(args, check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
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


def _read_scheduler_artifact(config: AppConfig, scheduler: SchedulerKind | None) -> str | None:
    """Read the installed scheduler artifact (plist or service file), or None."""
    if scheduler is None:
        return None
    if scheduler not in {SchedulerKind.LAUNCHD, SchedulerKind.SYSTEMD}:
        return None
    artifact_path = _artifact_paths(config, scheduler)[0]
    if not artifact_path.exists():
        return None
    try:
        return artifact_path.read_text(encoding="utf-8")
    except OSError:
        return None


def _detect_installed_mode(config: AppConfig, scheduler: SchedulerKind | None) -> DaemonMode | None:
    """Infer the installed daemon mode from the scheduler artifact on disk.

    Supports both launchd plists (KeepAlive vs StartInterval) and systemd
    service files (Type=simple vs Type=oneshot).
    """
    content = _read_scheduler_artifact(config, scheduler)
    if content is None:
        return None
    if scheduler == SchedulerKind.LAUNCHD:
        if "<key>KeepAlive</key>" in content:
            return DaemonMode.WATCH
        if "<key>StartInterval</key>" in content:
            return DaemonMode.POLL
    elif scheduler == SchedulerKind.SYSTEMD:
        if "Type=simple" in content:
            return DaemonMode.WATCH
        if "Type=oneshot" in content:
            return DaemonMode.POLL
    return None


def _detect_installed_command(config: AppConfig, scheduler: SchedulerKind | None) -> str | None:
    """Extract the installed command from the scheduler artifact.

    Parses ProgramArguments from launchd plists and ExecStart from systemd
    service files.
    """
    if scheduler == SchedulerKind.CRON:
        return _extract_cron_command(_read_crontab())
    content = _read_scheduler_artifact(config, scheduler)
    if content is None:
        return None
    if scheduler == SchedulerKind.SYSTEMD:
        for line in content.splitlines():
            if line.startswith("ExecStart="):
                return line[len("ExecStart=") :]
    elif scheduler == SchedulerKind.LAUNCHD:
        return _extract_launchd_command(content)
    return None


def _extract_launchd_command(plist_content: str) -> str | None:
    """Extract ProgramArguments from a launchd plist as a shell command string.

    Uses simple line-by-line parsing to avoid a plistlib dependency. Looks
    for the <key>ProgramArguments</key> block and collects <string> values
    until the closing </array>.
    """
    lines = plist_content.splitlines()
    in_program_args = False
    args: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "<key>ProgramArguments</key>":
            in_program_args = True
            continue
        if in_program_args:
            if stripped == "<array>":
                continue
            if stripped == "</array>":
                break
            if stripped.startswith("<string>") and stripped.endswith("</string>"):
                # Extract value between <string> and </string>
                value = stripped[len("<string>") : -len("</string>")]
                args.append(value)
    if not args:
        return None
    return _command_string(args)


def _extract_cron_command(crontab: str) -> str | None:
    inside_block = False
    for line in crontab.splitlines():
        stripped = line.strip()
        if stripped == CRON_BEGIN_MARKER:
            inside_block = True
            continue
        if stripped == CRON_END_MARKER:
            inside_block = False
            continue
        if not inside_block or not stripped:
            continue
        parts = shlex.split(stripped)
        if len(parts) < 6:
            return None
        return " ".join(shlex.quote(part) for part in parts[5:])
    return None


def _detect_installed_program_path(
    config: AppConfig,
    scheduler: SchedulerKind | None,
) -> str | None:
    command = _detect_installed_command(config, scheduler)
    if command is None:
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if scheduler == SchedulerKind.CRON:
        while parts and "=" in parts[0] and not parts[0].startswith("/"):
            parts.pop(0)
    if not parts:
        return None
    return parts[0]


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


def _read_crontab() -> str:
    if shutil.which("crontab") is None:
        return ""
    result = _run_command(["crontab", "-l"], check=False)
    if result.returncode == 0:
        return result.stdout
    stderr = (result.stderr or "").lower()
    if "no crontab" in stderr:
        return ""
    raise RuntimeError(result.stderr.strip() or "failed to read current crontab")


def _write_crontab(contents: str) -> None:
    if shutil.which("crontab") is None:
        raise RuntimeError("crontab is not available on this host")
    result = subprocess.run(
        ["crontab", "-"],
        input=contents,
        check=False,
        capture_output=True,
        text=True,
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


def _status_scheduler(config: AppConfig, runtime_status: RuntimeStatus) -> SchedulerKind | None:
    if runtime_status.installed_scheduler is not None:
        return runtime_status.installed_scheduler
    for candidate in _status_candidates(config):
        if _scheduler_installed(config, candidate):
            return candidate
    try:
        return _resolve_scheduler_kind(config.daemon.scheduler, config)
    except ValueError:
        return None


def _status_candidates(config: AppConfig) -> tuple[SchedulerKind, ...]:
    if sys.platform == "darwin":
        return (SchedulerKind.LAUNCHD,)
    if sys.platform.startswith("linux"):
        return (SchedulerKind.SYSTEMD, SchedulerKind.CRON)
    return tuple()


def _scheduler_installed(config: AppConfig, scheduler: SchedulerKind) -> bool:
    if scheduler == SchedulerKind.CRON:
        return CRON_BEGIN_MARKER in _read_crontab()
    if scheduler == SchedulerKind.SYSTEMD:
        service_path, timer_path = _artifact_paths(config, scheduler)
        if not service_path.exists():
            return False
        # Watch mode: service-only is valid; poll mode: both service + timer required
        installed_mode = _detect_installed_mode(config, scheduler)
        if installed_mode == DaemonMode.WATCH:
            return True
        return timer_path.exists()
    return all(path.exists() for path in _artifact_paths(config, scheduler))


def _detect_scheduler_health(
    scheduler: SchedulerKind | None,
    config: AppConfig,
) -> tuple[int | None, str | None]:
    if scheduler is None or scheduler == SchedulerKind.CRON:
        return None, None
    if scheduler == SchedulerKind.LAUNCHD:
        return _detect_launchd_health()
    if scheduler == SchedulerKind.SYSTEMD:
        return _detect_systemd_health(config)
    return None, None


_LAUNCHCTL_LAST_EXIT_STATUS = re.compile(r'"LastExitStatus"\s*=\s*(-?\d+)\s*;')


def _detect_launchd_health() -> tuple[int | None, str | None]:
    label = _active_launchd_label()
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


def _detect_systemd_health(config: AppConfig) -> tuple[int | None, str | None]:
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
    properties = _parse_systemd_show_output(service.stdout)
    raw_result = properties.get("Result")
    exec_status = _parse_int(properties.get("ExecMainStatus"))
    health_state = _systemd_health_state(raw_result, exec_status)

    if _detect_installed_mode(config, SchedulerKind.SYSTEMD) == DaemonMode.POLL:
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
            timer_result = _parse_systemd_show_output(timer.stdout).get("Result")
            if timer_result not in {None, "", "success"} and health_state in {None, "ok"}:
                health_state = timer_result

    return exec_status, health_state


def _parse_systemd_show_output(stdout: str) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        properties[key] = value
    return properties


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
    return subprocess.run(args, check=check, capture_output=True, text=True)


def _connect_scheduler_client(config: AppConfig) -> RpcClient | None:
    client = RpcClient(config)
    try:
        client.connect(auto_fork=False)
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
