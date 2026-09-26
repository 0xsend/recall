from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from recall.core.config import AppConfig, resolve_recall_binary
from recall.core.types import RunKind, SchedulerKind

logger = logging.getLogger("recall.cli.status_notices")

# A notice reads one number out of `daemon_status`; the bounded source page it
# would otherwise return is 100 rows of coverage detail nobody here looks at.
NOTICE_SOURCE_PAGE_LIMIT = 1
NOTICE_DETAIL_LIMIT = 3

# How long a discovered transcript must stay uncommitted before the backlog is
# worth naming. On a watch host every append leaves its session pending for a
# few seconds, so `pending > 0` alone keeps the warning permanently on and
# nobody reads a warning that never clears (REQ-CLI-023).
PENDING_COVERAGE_MIN_AGE_SECONDS = 300.0


def fetch_daemon_status(config: AppConfig | None = None) -> dict[str, Any] | None:
    """Read `daemon_status` once, treating any failure as "nothing to say".

    Callers that render more than one notice from the same reading pass the
    result to `render_status_notice_for_status` and `pending_coverage_guidance`
    instead of calling either one's own fetch (REQ-CLI-019/021).

    The `[cli] status_notices = false` opt-out is enforced here rather than at
    each notice, so no advisory path can reach the daemon behind it. Advisory
    reads never fork either: a daemon the operator did not start must not be
    launched to decorate a command's stderr.
    """
    config = config or AppConfig.load()
    if not config.cli.status_notices:
        return None
    try:
        from recall.cli.rpc import rpc_call_or_error

        status = rpc_call_or_error(
            "recall.daemon_status",
            {"limit": NOTICE_SOURCE_PAGE_LIMIT},
            auto_fork=False,
        )
    except Exception as exc:
        logger.debug("status notice RPC failed: %s", exc)
        return None
    return status if isinstance(status, dict) else None


def render_status_notice(
    config: AppConfig | None = None, *, include_informational: bool = True
) -> str | None:
    config = config or AppConfig.load()
    status = fetch_daemon_status(config)
    if status is None:
        return None

    return render_status_notice_for_status(
        status,
        config=config,
        include_informational=include_informational,
    )


def render_status_notice_for_status(
    status: dict[str, Any],
    *,
    config: AppConfig | None = None,
    include_informational: bool = True,
    include_pending_coverage: bool = True,
) -> str | None:
    """Render the notice from a status the caller already read.

    `include_pending_coverage` is how a command that already stated the backlog
    in its own guidance keeps the fact from reaching stderr twice (REQ-CLI-019).
    """
    config = config or AppConfig.load()
    if not config.cli.status_notices:
        return None
    return _render_status_notice_from_status(
        status,
        interval_seconds=config.daemon.interval,
        include_informational=include_informational,
        include_pending_coverage=include_pending_coverage,
    )


def pending_coverage_guidance(status: dict[str, Any] | None) -> str | None:
    """One clause naming the reconciliation backlog, for zero-result guidance.

    `recall search` answering "nothing" is only evidence of absence when recall
    has finished discovering and committing what exists (REQ-CLI-019). This
    reads the same reconciliation coverage the status notice warns from, so a
    caller reading the empty result learns why it may be incomplete.
    """
    if status is None:
        return None
    backlog = _coverage_backlog(status)
    if backlog is None:
        return None
    pending, scan_complete = backlog
    clause = f"{pending} transcripts are discovered but not yet indexed"
    if not scan_complete:
        clause += " and the catalog scan has not finished"
    return f"Coverage is incomplete: {clause} — an empty result is not proof of absence."


def _render_status_notice_from_status(
    status: dict[str, Any],
    *,
    interval_seconds: int,
    include_informational: bool = True,
    include_pending_coverage: bool = True,
) -> str | None:
    if not isinstance(status, dict):
        return None

    runtime = status.get("runtime_status")
    if not isinstance(runtime, dict):
        return None

    warning = (
        _binary_stale_notice(status)
        or _scheduler_exit_notice(status)
        or _freshness_staleness_notice(status, interval_seconds=interval_seconds)
        or _mode_mismatch_notice(status)
    )
    # REQ-DAEMON-074: on the local path under a live daemon the runtime dict
    # holds defaults, not readings; a notice that reads it would report a
    # daemon that "has not completed a successful run" for every start.
    runtime_read = not status.get("runtime_unavailable_reason")
    # Health notices surface independently of the lifecycle `warning` above: a
    # daemon can be fresh and correctly scheduled yet still be running stale code
    # (version drift) or sitting on a bloated database. Both silently went
    # unnoticed until they caused visible failures, so they always show.
    version_drift = _version_drift_notice(status)
    bloat = _bloat_notice(status)
    index_divergence = _index_divergence_notice(status)
    startup_refusal = _startup_refusal_notice(status)
    legacy_launchd_label = _legacy_launchd_label_notice(status)
    runtime_mode_notice = _runtime_mode_notice(status)
    unsupported = _unsupported_coverage_notice(status)
    pending_coverage = _pending_coverage_notice(status) if include_pending_coverage else None
    # The informational line (index freshness) is human-oriented status; callers
    # rendering structured output pass include_informational=False so agents get
    # only actionable warnings on stderr, not a status line on every --json call.
    informational = (
        _informational_notice(runtime) if include_informational and runtime_read else None
    )

    parts = [
        part
        for part in (
            warning,
            version_drift,
            bloat,
            index_divergence,
            startup_refusal,
            legacy_launchd_label,
            runtime_mode_notice,
            unsupported,
            pending_coverage,
            informational,
        )
        if part
    ]
    if not parts:
        return None
    return "; ".join(parts)


def _coverage_backlog(status: dict[str, Any]) -> tuple[int, bool] | None:
    """Return `(pending, catalog_scan_complete)` while reconciliation is behind.

    `recall live` calls a page incomplete when rows it discovered are not
    accounted for; `reconciliation.pending` is that same statement for the whole
    index — transcripts recall found and has not committed. An unfinished
    catalog scan rides along as detail rather than as the gate: a host that
    indexes without ever reconciling reports it permanently, and a warning that
    is always on is one nobody reads.

    A nonzero count is not by itself a backlog: on a watch host each append puts
    its own live session back in `pending` for the few seconds before the
    daemon commits it. `oldest_pending_age` separates that churn from a real
    backlog, so the count must also have been waiting past the noise floor.
    """
    reconciliation = status.get("reconciliation")
    if not isinstance(reconciliation, dict):
        return None
    pending = reconciliation.get("pending")
    if not isinstance(pending, int) or pending <= 0:
        return None
    oldest_age = _oldest_pending_age(reconciliation)
    if oldest_age is not None and oldest_age < PENDING_COVERAGE_MIN_AGE_SECONDS:
        return None
    return pending, bool(reconciliation.get("catalog_scan_complete"))


def _oldest_pending_age(reconciliation: dict[str, Any]) -> float | None:
    """Seconds the longest-waiting discovered transcript has gone uncommitted.

    Only roots that `pending` itself counts are eligible: a disabled root is
    excluded from the total, so its age cannot decide whether to warn about it.
    `None` means the payload carries no age — the caller then warns on the count
    alone, because under-warning about coverage is the costlier error.
    """
    coverage = reconciliation.get("coverage")
    if not isinstance(coverage, list):
        return None
    ages = [
        float(row["oldest_pending_age"])
        for row in coverage
        if isinstance(row, dict)
        and row.get("configuration") != "disabled"
        and isinstance(row.get("oldest_pending_age"), (int, float))
    ]
    return max(ages) if ages else None


def _pending_coverage_notice(status: dict[str, Any]) -> str | None:
    """Warn while reconciliation is still behind the sources it discovered.

    Search is where an empty answer is most easily misread as "it never
    happened". A backlog — or a catalog scan that has not finished enumerating
    the roots — means the index does not yet cover what exists.
    """
    backlog = _coverage_backlog(status)
    if backlog is None:
        return None
    pending, scan_complete = backlog
    detail = f"{pending} discovered transcripts are not indexed yet"
    if not scan_complete:
        detail += " and the catalog scan has not finished"
    return (
        f"Warning: coverage is incomplete — {detail}; inspect "
        "`recall daemon status --json --fields reconciliation` before inferring absence."
    )


def notice_details(summary: object) -> str:
    """One clause naming the largest parked groups, or empty when none are known."""
    if not isinstance(summary, dict):
        return ""
    groups = summary.get("groups")
    if not isinstance(groups, list) or not groups:
        return ""
    parts: list[str] = []
    for group in groups[:NOTICE_DETAIL_LIMIT]:
        if not isinstance(group, dict):
            continue
        source = group.get("source") or "unknown"
        files = group.get("files")
        count = f" ({files} files)" if isinstance(files, int) else ""
        if group.get("detail_omitted"):
            paths = group.get("sample_paths")
            sample = ""
            if isinstance(paths, list) and paths:
                sample = f"; sample {paths[0]}"
            parts.append(f"{source} detail omitted by an older payload{count}{sample}")
            continue
        detail = group.get("detail") or "unsupported_record"
        parts.append(f"{source} {detail}{count}")
    if not parts:
        return ""
    suffix = "; summary truncated" if summary.get("truncated") else ""
    return "; " + ", ".join(parts) + suffix


def _unsupported_coverage_notice(status: dict[str, Any]) -> str | None:
    """Warn when sources are parked on parser diagnostics, not missing.

    Coverage ``unsupported`` counts complete records the parser could not
    commit. Search and list then look empty even though the transcript exists.
    Name the gap so an agent files a recall issue or fixes the parser.
    """
    reconciliation = status.get("reconciliation")
    if not isinstance(reconciliation, dict):
        return None
    coverage = reconciliation.get("coverage")
    if not isinstance(coverage, list):
        return None
    total = 0
    for row in coverage:
        if not isinstance(row, dict):
            continue
        count = row.get("unsupported")
        if isinstance(count, int) and count > 0:
            total += count
    if total <= 0:
        return None
    noun = "source" if total == 1 else "sources"
    if "unsupported_summary" in reconciliation:
        summary = reconciliation.get("unsupported_summary")
    else:
        summary = None
    details = notice_details(summary)
    # An older daemon has coverage counts and source_page, not the summary field.
    # Naming a field it does not return sends the caller to an empty read.
    if summary is None:
        inspect = "reconciliation.source_page diagnostics"
    else:
        inspect = "reconciliation.unsupported_summary"
    return (
        f"Warning: {total} {noun} parked as unsupported parser diagnostics, not missing"
        f"{details}; inspect `recall daemon status --json` "
        f"{inspect}, then file a recall issue or fix the parser."
    )


def _version_drift_notice(status: dict[str, Any]) -> str | None:
    """Warn when the live daemon runs different code than the installed binary.

    `version_drift` is set when the daemon's package version no longer matches
    the resolved binary — e.g. after a `recall` upgrade whose daemon was never
    restarted. The daemon keeps serving the old code (stale parsers, missing
    fixes) until restarted, so an agent must be told to cycle it.
    """
    if not status.get("version_drift"):
        return None
    daemon_version = status.get("daemon_version") or "an unknown version"
    binary_version = status.get("binary_version") or "unknown (metadata missing)"
    return (
        f"Warning: recall daemon is running {daemon_version} but the installed binary is "
        f"{binary_version}; run `recall daemon restart` to load the new code."
    )


def _bloat_notice(status: dict[str, Any]) -> str | None:
    """Warn when the daemon's last bloat estimate exceeds the compaction threshold.

    Database bloat (dead row-versions DuckDB has not reclaimed) is otherwise
    invisible until it degrades queries or trips an out-of-memory failure. The
    daemon caches its most recent `estimate_bloat_ratio` in `bloat_ratio`; when
    it clears the threshold the CLI names the fix so it cannot silently grow.
    """
    ratio = status.get("bloat_ratio")
    threshold = status.get("bloat_ratio_threshold")
    if not isinstance(ratio, (int, float)) or not isinstance(threshold, (int, float)):
        return None
    if ratio < threshold:
        return None
    detail = f"database is ~{ratio:.1f}x bloated (compaction threshold {threshold:.1f}x)"
    if status.get("bloat_auto_trigger"):
        return f"Warning: {detail}; the daemon will auto-compact on its next check."
    return f"Warning: {detail}; run `recall compact --yes` to reclaim space."


def _startup_refusal_notice(status: dict[str, Any]) -> str | None:
    """Warn while a refused start holds the scheduler down (REQ-RESIL-024).

    The refusal marker is the one condition under which the daemon is down and
    will not come back on its own; the message it carries names the manual fix.
    """
    refusal = status.get("startup_refusal")
    if not isinstance(refusal, str) or not refusal:
        return None
    return (
        "Warning: the daemon refused to start and the scheduler will not relaunch it "
        f"until the failure memory is cleared: {refusal}"
    )


def _legacy_launchd_label_notice(status: dict[str, Any]) -> str | None:
    """Name the pending launch agent migration (REQ-DAEMON-076)."""
    if not status.get("installed"):
        return None
    if status.get("launchd_legacy_label"):
        return (
            "Warning: the daemon is installed under the legacy launchd label; "
            "run `recall daemon install` to migrate."
        )
    if status.get("launchd_legacy_leftover"):
        return (
            "Warning: a legacy launchd job or plist remains beside the installed daemon; "
            "run `recall daemon install` to retire it."
        )
    return None


def _index_divergence_notice(status: dict[str, Any]) -> str | None:
    """Warn when the daemon's index probe found ART/table divergence (REQ-RESIL-018).

    Divergence otherwise surfaces only as a fatal `Failed to delete all rows
    from index` on the next write to the affected key, which invalidates the
    whole DuckDB instance. Naming the fix here lets the operator rebuild before
    that happens.
    """
    divergence = status.get("index_divergence")
    diverged_count = 0
    if isinstance(divergence, dict):
        diverged = divergence.get("diverged")
        count = divergence.get("diverged_count")
        if isinstance(count, int):
            diverged_count = count
        elif isinstance(diverged, list):
            diverged_count = len(diverged)
    if diverged_count > 0:
        return (
            f"Warning: DuckDB index/table divergence on {diverged_count} sampled key(s); "
            "run `recall daemon stop`, then `recall db rebuild-indexes`, then "
            "`recall daemon start`."
        )
    runtime = status.get("runtime_status")
    if isinstance(runtime, dict) and runtime.get("needs_index_verification"):
        return (
            "Warning: the database needs index verification after a disk-full error; "
            "the daemon probes and repairs at its next start (`recall daemon restart`)."
        )
    return None


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _informational_notice(runtime: dict[str, Any]) -> str | None:
    last_successful_at = _parse_datetime(runtime.get("last_successful_at"))
    if last_successful_at is None:
        return None

    last_run_kind_raw = runtime.get("last_run_kind")
    if last_run_kind_raw is None:
        return None

    try:
        last_run_kind = RunKind(last_run_kind_raw)
    except ValueError:
        return None

    age = _format_age(last_successful_at)
    successful_run_kind = _last_successful_run_kind(runtime, last_run_kind)

    index_summary = runtime.get("last_index_summary")
    if isinstance(index_summary, dict):
        notice = f"Index: last updated {age} ago"
        if successful_run_kind is not None:
            notice += f" via {successful_run_kind.value}"
        notice += (
            f" (changed {index_summary.get('changed', 0)}, "
            f"indexed {index_summary.get('indexed', 0)}, "
            f"skipped {index_summary.get('skipped', 0)}, "
            f"failed {index_summary.get('failed', 0)}"
        )
        total_seconds = index_summary.get("total_seconds")
        if total_seconds is not None:
            notice += f", {total_seconds:.2f}s"
        notice += ")"
    else:
        notice = f"Index: last updated {age} ago"
        if successful_run_kind is not None:
            notice += f" via {successful_run_kind.value}"

    installed_scheduler = runtime.get("installed_scheduler")
    if installed_scheduler is not None and successful_run_kind in {
        RunKind.DAEMON_ONCE,
        RunKind.DAEMON_SCHEDULED,
    }:
        notice += f"; Daemon: installed via {installed_scheduler}, last success {age} ago"
    return notice


def _format_age(timestamp: datetime) -> str:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    delta = max((now - timestamp).total_seconds(), 0)
    if delta < 60:
        return "0m"
    if delta < 3600:
        return f"{int(delta // 60)}m"
    if delta < 86400:
        return f"{int(delta // 3600)}h"
    return f"{int(delta // 86400)}d"


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _attach_utc_to_naive(value)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        return _attach_utc_to_naive(parsed)
    return None


def _attach_utc_to_naive(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp


def _binary_stale_notice(status: dict[str, Any]) -> str | None:
    if not status.get("installed") or not status.get("installed_binary_stale"):
        return None
    installed_binary = status.get("installed_binary_path") or "unknown"
    try:
        resolved_binary = resolve_recall_binary()
    except Exception:
        resolved_binary = "an unresolved current recall binary"
    return (
        f"Warning: installed daemon is pinned to {installed_binary}, but recall currently "
        f"resolves to {resolved_binary}. Run `recall daemon install` to refresh the scheduler."
    )


def _scheduler_exit_notice(status: dict[str, Any]) -> str | None:
    if not status.get("installed"):
        return None
    exit_status = status.get("scheduler_last_exit_status")
    if exit_status in {None, 0}:
        return None

    scheduler = status.get("scheduler")
    scheduler = _enum_value(scheduler)
    health_state = status.get("scheduler_health_state")
    if scheduler == SchedulerKind.LAUNCHD.value or scheduler == "launchd":
        detail = "launchd last exited"
        if exit_status == 78:
            detail += " with status 78: launchd could not execute the configured program"
        else:
            detail += f" with status {exit_status}"
        return f"Warning: {detail}. Run `recall daemon install` to refresh the launch agent."

    if scheduler == SchedulerKind.SYSTEMD.value or scheduler == "systemd":
        state_suffix = f" ({health_state})" if health_state else ""
        return (
            f"Warning: systemd scheduler last exited with status {exit_status}{state_suffix}. "
            "Run `recall daemon install` to refresh the user unit."
        )

    return f"Warning: scheduler last exited with status {exit_status}."


def _freshness_staleness_notice(status: dict[str, Any], *, interval_seconds: int) -> str | None:
    if not status.get("installed"):
        return None
    if status.get("runtime_unavailable_reason"):
        # The runtime fields were not read (REQ-DAEMON-074); `last_successful_at`
        # being None is the default, not evidence of a run that never happened.
        return None

    runtime = status.get("runtime_status")
    if not isinstance(runtime, dict):
        return None

    last_successful_at = _parse_datetime(runtime.get("last_successful_at"))
    if last_successful_at is None:
        return (
            "Warning: Daemon appears installed but has not completed a successful run yet. "
            "Run `recall daemon install` to refresh the scheduler."
        )

    resolved_mode = _enum_value(status.get("resolved_mode"))
    threshold_seconds = 3600 if resolved_mode == "watch" else max(2 * interval_seconds, 600)
    age_seconds = max((datetime.now(UTC) - last_successful_at).total_seconds(), 0)
    if age_seconds <= threshold_seconds:
        return None

    return (
        "Warning: daemon appears stale; "
        f"the last successful run was {_format_age(last_successful_at)} ago. "
        "Run `recall daemon install` to refresh the scheduler."
    )


def _mode_mismatch_notice(status: dict[str, Any]) -> str | None:
    """Emit a notice when the installed daemon mode differs from what auto would resolve to.

    This alerts users who installed the daemon before watch mode was available
    (or before watchdog was installed) to re-run `recall daemon install`.

    Uses ``auto_resolved_mode`` from the daemon status response (computed in the
    services layer) so the CLI module avoids importing recall.services directly.
    """
    if not status.get("installed"):
        return None

    # Skip notice when the user explicitly configured poll mode — they
    # intentionally chose it and the "upgrade to watch" suggestion is noise.
    # But if they configured "watch" or "auto" and the installed scheduler is
    # still poll, that's a real mismatch worth surfacing.
    configured_mode = _enum_value(status.get("mode"))
    if configured_mode == "poll":
        return None

    resolved_mode = _enum_value(status.get("resolved_mode"))
    installed_mode = _enum_value(status.get("installed_mode")) or resolved_mode
    auto_mode = _enum_value(status.get("auto_resolved_mode"))
    if installed_mode is None or auto_mode is None:
        return None

    if installed_mode == auto_mode:
        return None

    # Only notify when upgrading is possible (poll → watch)
    if installed_mode == "poll" and auto_mode == "watch":
        return (
            "Daemon installed in poll mode but watch mode is available."
            " Run `recall daemon install` to upgrade."
        )

    return None


def _runtime_mode_notice(status: dict[str, Any]) -> str | None:
    """Explain when the live daemon mode overrides the installed scheduler mode."""
    reason = status.get("mode_mismatch_reason")
    if not reason:
        return None

    resolved_mode = _enum_value(status.get("resolved_mode"))
    installed_mode = _enum_value(status.get("installed_mode"))
    if resolved_mode is not None and installed_mode is not None:
        return (
            f"Daemon is running in {resolved_mode} mode but the installed scheduler runs "
            f"{installed_mode}; status reports the running daemon's mode."
        )

    return f"Daemon mode differs between the running process and installed scheduler ({reason})."


def _last_successful_run_kind(runtime: dict[str, Any], last_run_kind: RunKind) -> RunKind | None:
    last_successful_at = runtime.get("last_successful_at")
    if last_successful_at is None:
        return None
    last_failure_at = runtime.get("last_failure_at")
    if last_failure_at is None:
        return last_run_kind
    if isinstance(last_successful_at, str):
        last_successful_at = datetime.fromisoformat(last_successful_at)
    if isinstance(last_failure_at, str):
        last_failure_at = datetime.fromisoformat(last_failure_at)
    if isinstance(last_successful_at, datetime):
        last_successful_at = _attach_utc_to_naive(last_successful_at)
    if isinstance(last_failure_at, datetime):
        last_failure_at = _attach_utc_to_naive(last_failure_at)
    if last_successful_at >= last_failure_at:
        return last_run_kind
    return None
