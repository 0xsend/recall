"""The `daemon status` payload: the scheduler model and the reconciliation block.

The daemon builds these and the CLI renders and projects them, so they live
below both layers; the command manifest derives its field lists from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypedDict

from recall.core.types import DaemonMode, RunKind, SchedulerKind


class WriteAdmissionStatus(TypedDict):
    allowed: bool
    reason: Literal["storage_maintenance"] | None
    action: str | None


class StorageMaintenanceStatus(TypedDict):
    phase: Literal["running", "failed", "succeeded"]
    snapshot: bool
    operation_id: str | None
    error: str | None


@dataclass(frozen=True)
class IndexRunCounts:
    total: int
    indexed: int
    skipped: int
    failed: int
    changed: int = 0
    total_seconds: float | None = None


@dataclass(frozen=True)
class RuntimeStatus:
    last_attempted_at: datetime | None
    last_successful_at: datetime | None
    last_run_kind: RunKind | None
    last_index_summary: IndexRunCounts | None
    last_failure_message: str | None
    last_failure_at: datetime | None
    installed_scheduler: SchedulerKind | None
    # Failure-signature memory (REQ-RESIL-014..019). Defaulted so a database that
    # predates migration 0023 still loads.
    last_fatal_signature: str | None = None
    fatal_repeat_count: int = 0
    last_fatal_at: datetime | None = None
    last_index_repair_at: datetime | None = None
    last_index_repair_signature: str | None = None
    needs_index_verification: bool = False

    @classmethod
    def unread(cls) -> RuntimeStatus:
        """The status of a runtime_state that was not (or could not be) read."""
        return cls(
            last_attempted_at=None,
            last_successful_at=None,
            last_run_kind=None,
            last_index_summary=None,
            last_failure_message=None,
            last_failure_at=None,
            installed_scheduler=None,
        )


@dataclass(frozen=True)
class DaemonSchedulerStatus:
    configured_scheduler: SchedulerKind
    scheduler: SchedulerKind | None
    installed: bool
    command: str
    config_path: str
    artifact_paths: tuple[str, ...]
    runtime_status: RuntimeStatus
    duckdb_memory_limit_requested: str | None = None
    duckdb_memory_limit_applied: str | None = None
    duckdb_memory_limit_clamped: bool | None = None
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
    # The eligible count the embed loop last measured; a deferred cycle freezes
    # it without looking again, so the age is what dates it (REQ-ADAPT-012).
    embed_pending: int = 0
    embed_pending_at: float | None = None
    # Eligible and held pending targets read fresh for this status call, and
    # when; None when the database could not be read (REQ-ADAPT-022).
    embed_pending_eligible: int | None = None
    embed_pending_held: int | None = None
    embed_pending_measured_at: float | None = None
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
    # The data dir the installed service runs against; null when none is
    # installed. When it is another data dir, `installed` is false and the
    # service fields stay empty: that service is not this daemon's
    # (REQ-DAEMON-080).
    service_data_dir: str | None = None
    # When the service fields were measured. A daemon serves a measurement
    # refreshed off the request path, at most SERVICE_PROBE_MAX_AGE_SECONDS
    # old in steady state (REQ-DAEMON-081).
    service_measured_at: float | None = None
    # Why the installed service could not be identified, measured with the
    # other service fields; null when it was identified or nothing was
    # measured (REQ-DAEMON-085).
    service_identity_error: str | None = None
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
    write_admission: WriteAdmissionStatus | None = None
    storage_maintenance: StorageMaintenanceStatus | None = None


class SourcePage(TypedDict):
    """One page of the durable source catalog; `next_cursor` continues it."""

    source_page: list[dict[str, object]]
    next_cursor: str | None


class ReconciliationCoverage(SourcePage):
    """Backlog and coverage as of the catalog's durable observations."""

    coverage: list[dict[str, object]]
    pending: int
    out_of_scope_discovered: int
    out_of_scope_pending: int
    catalog_scan_complete: bool
    raw_indexing_ready: bool
    keyword_search_ready: bool
    index_only_sessions: dict[str, int]
    unsupported_summary: dict[str, object]
    index_migration: dict[str, object]


class ReconciliationStatus(ReconciliationCoverage):
    """The `reconciliation` block a running daemon's status carries.

    The `daemon status` manifest derives its `reconciliation.<key>` projection
    fields from this type, so every key the daemon reports is projectable.
    """

    noticed_pending: int
    keyword_repair_error: str | None
    paused: bool
    runtime_mode: str
    rpc_ready: bool
    live_observation_ready: bool
    active_observation_error: str | None
    enrichment_ready: bool
    enrichment_error: str | None
    enrichment_deferred: str | None
    error: str | None
    checkpoint_maintenance: dict[str, Any]
    checkpoint_contention_retries: int
    wal_pressure_blocked: bool
    activity_hints_saturated: bool
    raw_preparation_reserved_bytes: int
    raw_preparation_historical_reserved_bytes: int
    ordinary_read_active: int
    ordinary_read_oldest_age_seconds: float | None
    recovery_waiting: int
    recovery_waiting_oldest_age_seconds: float | None
