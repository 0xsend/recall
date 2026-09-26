"""Health notices: version drift and database bloat.

Both conditions were silent until they caused visible failures — a daemon
serving stale code after an upgrade, and a database bloating unbounded. These
notices surface them on ordinary CLI commands.
"""

from __future__ import annotations

from recall.cli.status_notices import (
    _bloat_notice,
    _freshness_staleness_notice,
    _pending_coverage_notice,
    _render_status_notice_from_status,
    _unsupported_coverage_notice,
    _version_drift_notice,
    fetch_daemon_status,
    pending_coverage_guidance,
)

# --- version drift -------------------------------------------------------


def test_version_drift_notice_warns_when_drift() -> None:
    notice = _version_drift_notice(
        {"version_drift": True, "daemon_version": "0.21.3", "binary_version": "0.21.6"}
    )
    assert notice is not None
    assert "0.21.3" in notice and "0.21.6" in notice
    assert "recall daemon restart" in notice


def test_version_drift_notice_silent_when_aligned() -> None:
    assert (
        _version_drift_notice(
            {"version_drift": False, "daemon_version": "0.21.6", "binary_version": "0.21.6"}
        )
        is None
    )


def test_version_drift_notice_handles_missing_binary_version() -> None:
    # binary_version is None when importlib.metadata can't resolve the dist
    # (reinstall pulled the venv out from under the live daemon) — still drift.
    notice = _version_drift_notice(
        {"version_drift": True, "daemon_version": "0.21.6", "binary_version": None}
    )
    assert notice is not None and "0.21.6" in notice


# --- bloat ---------------------------------------------------------------


def test_bloat_notice_warns_over_threshold_auto_trigger() -> None:
    notice = _bloat_notice(
        {"bloat_ratio": 22.8, "bloat_ratio_threshold": 2.0, "bloat_auto_trigger": True}
    )
    assert notice is not None
    assert "22.8x" in notice
    assert "auto-compact" in notice


def test_bloat_notice_warns_over_threshold_manual() -> None:
    notice = _bloat_notice(
        {"bloat_ratio": 5.0, "bloat_ratio_threshold": 2.0, "bloat_auto_trigger": False}
    )
    assert notice is not None
    assert "recall compact" in notice


def test_bloat_notice_silent_when_dense() -> None:
    assert (
        _bloat_notice(
            {"bloat_ratio": 1.11, "bloat_ratio_threshold": 2.0, "bloat_auto_trigger": True}
        )
        is None
    )


def test_unsupported_coverage_notice_warns_with_count() -> None:
    notice = _unsupported_coverage_notice(
        {
            "runtime_status": {},
            "reconciliation": {
                "coverage": [
                    {"source": "claude_code", "unsupported": 1776},
                    {"source": "grok", "unsupported": 899},
                    {"source": "codex", "unsupported": 0},
                ]
            },
        }
    )
    assert notice is not None
    assert "2675 sources" in notice
    assert "not missing" in notice
    assert "file a recall issue or fix the parser" in notice


def test_unsupported_coverage_notice_singular() -> None:
    notice = _unsupported_coverage_notice(
        {"reconciliation": {"coverage": [{"source": "pi_agent", "unsupported": 1}]}}
    )
    assert notice is not None
    assert "1 source parked" in notice
    assert "reconciliation.source_page diagnostics" in notice
    assert "unsupported_summary" not in notice


def test_unsupported_coverage_notice_names_summary_groups() -> None:
    notice = _unsupported_coverage_notice(
        {
            "reconciliation": {
                "coverage": [{"source": "kimi_code", "unsupported": 2}],
                "unsupported_summary": {
                    "files": 2,
                    "truncated": False,
                    "groups": [
                        {
                            "source": "kimi_code",
                            "detail": "record: 'swarm_mode.enter'",
                            "detail_omitted": False,
                            "files": 2,
                            "sample_paths": ["/tmp/wire.jsonl"],
                        }
                    ],
                },
            }
        }
    )
    assert notice is not None
    assert "kimi_code record: 'swarm_mode.enter' (2 files)" in notice
    assert "reconciliation.unsupported_summary" in notice
    assert "source_page" not in notice


def test_status_notice_includes_unsupported_in_structured_warnings() -> None:
    notice = _render_status_notice_from_status(
        {
            "runtime_status": {},
            "reconciliation": {"coverage": [{"unsupported": 3}]},
        },
        interval_seconds=300,
        include_informational=False,
    )
    assert notice is not None
    assert "3 sources parked as unsupported" in notice


def test_unsupported_coverage_notice_silent_when_zero_or_absent() -> None:
    assert (
        _unsupported_coverage_notice({"reconciliation": {"coverage": [{"unsupported": 0}]}}) is None
    )
    assert _unsupported_coverage_notice({"reconciliation": {"coverage": []}}) is None
    assert _unsupported_coverage_notice({"reconciliation": {}}) is None
    assert _unsupported_coverage_notice({}) is None


# --- pending coverage (REQ-CLI-023) --------------------------------------


def test_pending_coverage_notice_names_the_backlog_and_the_scan() -> None:
    notice = _pending_coverage_notice(
        {
            "runtime_status": {},
            "reconciliation": {"pending": 27362, "catalog_scan_complete": False},
        }
    )
    assert notice is not None
    assert "27362" in notice
    assert "catalog scan" in notice
    assert "inferring absence" in notice
    assert "recall daemon status" in notice


def test_pending_coverage_notice_warns_on_backlog_with_a_finished_scan() -> None:
    notice = _pending_coverage_notice(
        {"reconciliation": {"pending": 12, "catalog_scan_complete": True}}
    )
    assert notice is not None
    assert "12" in notice
    assert "catalog scan" not in notice


def test_pending_coverage_notice_silent_without_a_backlog() -> None:
    """A host that indexes without reconciling reports an unfinished scan forever;
    a warning that is always on is one nobody reads, so pending is the gate."""
    assert (
        _pending_coverage_notice({"reconciliation": {"pending": 0, "catalog_scan_complete": False}})
        is None
    )
    assert (
        _pending_coverage_notice({"reconciliation": {"pending": 0, "catalog_scan_complete": True}})
        is None
    )
    assert _pending_coverage_notice({"reconciliation": {}}) is None
    assert _pending_coverage_notice({}) is None


def test_fetch_daemon_status_is_silent_when_the_daemon_is_unreachable(monkeypatch) -> None:
    """`search`'s zero-result guidance must never fail because the daemon is down."""
    monkeypatch.setenv("RECALL_CLI_STATUS_NOTICES", "true")

    def explode(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("daemon unreachable")

    monkeypatch.setattr("recall.cli.rpc.rpc_call_or_error", explode)
    assert fetch_daemon_status() is None
    assert pending_coverage_guidance(None) is None


def test_fetch_daemon_status_requests_a_single_source_row(monkeypatch) -> None:
    """REQ-CLI-021: the notice reads one number; a 100-row source page is waste."""
    monkeypatch.setenv("RECALL_CLI_STATUS_NOTICES", "true")
    seen: list[tuple[object, object]] = []

    def record(_method: str, params: object = None, **kwargs: object) -> object:
        seen.append((params, kwargs.get("auto_fork")))
        return {"runtime_status": {}}

    monkeypatch.setattr("recall.cli.rpc.rpc_call_or_error", record)
    assert fetch_daemon_status() == {"runtime_status": {}}
    assert seen == [({"limit": 1}, False)]


def test_fetch_daemon_status_makes_no_rpc_when_notices_are_disabled(monkeypatch) -> None:
    """An advisory read must not reach — or fork — a daemon the operator opted out of."""
    monkeypatch.setenv("RECALL_CLI_STATUS_NOTICES", "false")

    def explode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("status notices are disabled; no RPC may be made")

    monkeypatch.setattr("recall.cli.rpc.rpc_call_or_error", explode)
    assert fetch_daemon_status() is None


def test_pending_coverage_guidance_names_the_backlog() -> None:
    guidance = pending_coverage_guidance(
        {
            "reconciliation": {
                "pending": 27362,
                "catalog_scan_complete": False,
                "coverage": [{"configuration": "enabled", "oldest_pending_age": 86_400.0}],
            }
        }
    )
    assert guidance is not None
    assert "27362" in guidance
    assert "not yet indexed" in guidance


# --- pending coverage noise floor (REQ-CLI-023) --------------------------


def _pending_status(*, pending: int, oldest_pending_age: float | None) -> dict:
    return {
        "runtime_status": {},
        "reconciliation": {
            "pending": pending,
            "catalog_scan_complete": True,
            "coverage": [
                {
                    "source": "claude_code",
                    "configuration": "enabled",
                    "pending": pending,
                    "oldest_pending_age": oldest_pending_age,
                }
            ],
        },
    }


def test_pending_coverage_notice_silent_while_a_live_session_is_briefly_pending() -> None:
    """On a watch host every append is pending for seconds; an always-on warning is unread."""
    assert _pending_coverage_notice(_pending_status(pending=3, oldest_pending_age=4.0)) is None


def test_pending_coverage_notice_warns_once_the_backlog_ages_past_the_floor() -> None:
    notice = _pending_coverage_notice(_pending_status(pending=3, oldest_pending_age=86_400.0))
    assert notice is not None
    assert "3 discovered transcripts" in notice


def test_pending_coverage_notice_warns_when_no_age_is_reported() -> None:
    """Without the age signal the notice fails open: under-warning is the costlier error."""
    notice = _pending_coverage_notice(_pending_status(pending=9, oldest_pending_age=None))
    assert notice is not None
    assert "9 discovered transcripts" in notice


def test_pending_coverage_notice_ignores_ages_from_disabled_roots() -> None:
    """A disabled root is excluded from `pending`, so its age cannot gate the warning."""
    status = _pending_status(pending=4, oldest_pending_age=5.0)
    status["reconciliation"]["coverage"].append(
        {"source": "grok", "configuration": "disabled", "oldest_pending_age": 999_999.0}
    )
    assert _pending_coverage_notice(status) is None


def test_pending_coverage_guidance_respects_the_same_noise_floor() -> None:
    """A seconds-old append never explains why a query matched nothing."""
    assert pending_coverage_guidance(_pending_status(pending=3, oldest_pending_age=4.0)) is None


def test_bloat_notice_silent_when_unknown() -> None:
    # Fresh daemon before its first estimate: no ratio -> no notice.
    assert _bloat_notice({"bloat_ratio": None, "bloat_ratio_threshold": 2.0}) is None
    assert _bloat_notice({}) is None


# --- integration ---------------------------------------------------------


def test_render_surfaces_drift_and_bloat_together() -> None:
    status = {
        "installed": True,
        "resolved_mode": "watch",
        "version_drift": True,
        "daemon_version": "0.21.3",
        "binary_version": "0.21.6",
        "bloat_ratio": 22.8,
        "bloat_ratio_threshold": 2.0,
        "bloat_auto_trigger": True,
        "runtime_status": {},  # required non-null for the renderer to proceed
    }
    notice = _render_status_notice_from_status(status, interval_seconds=300)
    assert notice is not None
    assert "0.21.3" in notice  # drift part
    assert "22.8x" in notice  # bloat part


def test_daemon_status_manifest_exposes_bloat_fields() -> None:
    """The bloat fields are exposed by daemon_status, so `daemon status --fields
    bloat_ratio` must validate — the manifest must list them."""
    from recall.cli.manifest import output_fields_for

    fields = output_fields_for("daemon status")
    assert fields is not None
    assert {"bloat_ratio", "bloat_ratio_threshold", "bloat_auto_trigger"} <= fields


# --- unread runtime fields (REQ-DAEMON-074) ------------------------------


def _unread_local_status() -> dict:
    return {
        "installed": True,
        "resolved_mode": "watch",
        "daemon_pid": 4242,
        "runtime_unavailable_reason": "daemon pid 4242 holds the database",
        "runtime_status": {
            "last_attempted_at": None,
            "last_successful_at": None,
            "last_run_kind": None,
            "last_failure_message": None,
            "last_failure_at": None,
        },
    }


def test_freshness_notice_is_silent_while_the_runtime_fields_are_unread() -> None:
    """The defaults of an unread runtime_state are not a daemon that never ran."""
    assert _freshness_staleness_notice(_unread_local_status(), interval_seconds=300) is None


def test_no_runtime_derived_notice_while_the_runtime_fields_are_unread() -> None:
    notice = _render_status_notice_from_status(_unread_local_status(), interval_seconds=300)
    assert notice is None
