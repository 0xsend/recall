from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any

import pytest
from recall.cli.app import app
from typer.testing import CliRunner

# A bare epoch-seconds float such as "1789610011.5" — unreadable in text output.
_BARE_EPOCH = re.compile(r"\b\d{10}\.\d+\b")

_FTS_SIDECAR_FIELDS = {
    "fts_sidecar_enabled",
    "fts_sidecar_bootstrap_messages_processed",
    "fts_sidecar_bootstrap_tool_calls_processed",
    "fts_sidecar_bootstrap_messages_done",
    "fts_sidecar_bootstrap_tool_calls_done",
    "fts_sidecar_reconcile_pending_drained_messages",
    "fts_sidecar_reconcile_pending_drained_tool_calls",
    "fts_sidecar_reconcile_orphans_backfilled_messages",
    "fts_sidecar_reconcile_orphans_backfilled_tool_calls",
    "fts_sidecar_reconcile_ghosts_deleted_messages",
    "fts_sidecar_reconcile_ghosts_deleted_tool_calls",
    "fts_sidecar_reconcile_pending_remaining_messages",
    "fts_sidecar_reconcile_pending_remaining_tool_calls",
    "fts_sidecar_last_run_at",
    "fts_sidecar_error",
}


def _base_status(*, sidecar_enabled: bool = True) -> dict[str, Any]:
    return {
        "configured_scheduler": "auto",
        "scheduler": None,
        "installed": False,
        "command": "/tmp/recall daemon --once",
        "config_path": "/tmp/config.toml",
        "artifact_paths": [],
        "runtime_status": {},
        "mode": "poll",
        "resolved_mode": "poll",
        "daemon_version": None,
        "binary_version": "0.10.4",
        "version_drift": False,
        "debounce": 5,
        "fts_debounce": 10,
        "embed_phase_enabled": False,
        "embed_model_loaded": False,
        "embed_pending": 0,
        "last_fts_rebuild_failure_at": None,
        "last_fts_rebuild_failure_reason": None,
        "fts_rebuild_consecutive_failures": 0,
        "fts_rebuild_next_retry_at": None,
        "fts_sidecar_enabled": sidecar_enabled,
        "fts_sidecar_bootstrap_messages_processed": 7,
        "fts_sidecar_bootstrap_tool_calls_processed": 3,
        "fts_sidecar_bootstrap_messages_done": True,
        "fts_sidecar_bootstrap_tool_calls_done": True,
        "fts_sidecar_reconcile_pending_drained_messages": 1,
        "fts_sidecar_reconcile_pending_drained_tool_calls": 2,
        "fts_sidecar_reconcile_orphans_backfilled_messages": 3,
        "fts_sidecar_reconcile_orphans_backfilled_tool_calls": 4,
        "fts_sidecar_reconcile_ghosts_deleted_messages": 5,
        "fts_sidecar_reconcile_ghosts_deleted_tool_calls": 6,
        "fts_sidecar_reconcile_pending_remaining_messages": 0,
        "fts_sidecar_reconcile_pending_remaining_tool_calls": 0,
        "fts_sidecar_last_run_at": "2026-05-28T15:30:00+00:00",
        "fts_sidecar_error": None,
    }


@pytest.fixture(autouse=True)
def _status_notice_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "recall.cli.daemon.AppConfig.load",
        lambda: type("Config", (), {"daemon": type("Daemon", (), {"interval": 300})()})(),
    )


def test_daemon_status_json_includes_fts_sidecar_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error",
        lambda *_args, **_kwargs: _base_status(sidecar_enabled=True),
    )

    result = CliRunner().invoke(app, ["daemon", "status", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert _FTS_SIDECAR_FIELDS.issubset(payload)
    assert payload["fts_sidecar_enabled"] is True
    assert payload["fts_sidecar_bootstrap_messages_processed"] == 7
    assert payload["fts_sidecar_reconcile_pending_drained_tool_calls"] == 2
    assert payload["last_fts_rebuild_failure_at"] is None
    assert payload["last_fts_rebuild_failure_reason"] is None
    assert payload["fts_rebuild_consecutive_failures"] == 0
    assert payload["fts_rebuild_next_retry_at"] is None


def test_daemon_status_json_emits_health_warning_on_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Structured daemon status keeps stdout clean JSON and surfaces the
    version-drift/bloat health warning on stderr (REQ-CLI-021/022, REQ-CLI-012)."""
    status = _base_status()
    status["version_drift"] = True
    status["daemon_version"] = "0.21.3"
    status["binary_version"] = "0.22.0"
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error",
        lambda *_args, **_kwargs: status,
    )

    result = CliRunner().invoke(app, ["daemon", "status", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)  # stdout stays clean, parseable JSON
    assert payload["version_drift"] is True
    assert "Warning" not in result.stdout
    assert "0.21.3" in result.stderr
    assert "recall daemon restart" in result.stderr


def test_daemon_status_text_renders_embed_loop_liveness(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stalled embed loop must be visible as stage age, not just a frozen batch."""
    status = _base_status()
    status.update(
        {
            "embed_phase_enabled": True,
            "embed_pending": 639051,
            "embed_last_batch_at": 1789610011.5,
            "embed_last_batch_size": 467,
            "embed_loop_iterations": 12,
            "embed_loop_last_iteration_at": 1789610011.5,
            "embed_loop_stage": "snapshot",
            "embed_loop_stage_at": time.time() - 90,
            "embed_loop_last_outcome": "drained",
            "embed_loop_next_interval": 0.01,
            "embed_last_error": None,
        }
    )
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error",
        lambda *_args, **_kwargs: status,
    )

    result = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

    assert result.exit_code == 0
    assert "Embed loop: iteration=12 stage=snapshot" in result.stdout
    assert "stage_age=9" in result.stdout
    assert "outcome=drained" in result.stdout


def test_daemon_status_reports_embed_deferral_and_pending_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reason a cycle deferred belongs beside the outcome it explains, and a
    frozen pending count must be dated so staleness is readable (REQ-ADAPT-012)."""
    status = _base_status()
    status.update(
        {
            "embed_phase_enabled": True,
            "embed_pending": 639051,
            "embed_pending_at": time.time() - 900,
            "embed_deferred_reason": (
                "load 41.0 > battery threshold 5.4 (on battery: Battery Power)"
            ),
            "embed_deferred_at": time.time() - 600,
            "embed_loop_iterations": 5,
            "embed_loop_stage": "wait",
            "embed_loop_stage_at": time.time() - 240,
            "embed_loop_last_outcome": "deferred",
            "embed_loop_last_trigger": "timer",
            "embed_loop_next_interval": 300.0,
        }
    )
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error",
        lambda *_args, **_kwargs: status,
    )

    result = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

    assert result.exit_code == 0
    assert "pending: 639051, measured 900s ago)" in result.stdout
    assert "trigger=timer" in result.stdout
    assert (
        "Embed deferred: load 41.0 > battery threshold 5.4 (on battery: Battery Power)"
        in result.stdout
    )
    # The reason names a ceiling an operator can check against `pmset`, so it
    # must also say when it was evaluated (REQ-ADAPT-006).
    assert "evaluated" in result.stdout
    assert "(10m ago)" in result.stdout

    result = CliRunner().invoke(app, ["daemon", "status", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert (
        payload["embed_deferred_reason"]
        == "load 41.0 > battery threshold 5.4 (on battery: Battery Power)"
    )
    assert payload["embed_deferred_at"] == status["embed_deferred_at"]
    assert payload["embed_pending_at"] == status["embed_pending_at"]
    assert payload["embed_loop_last_trigger"] == "timer"


def test_daemon_status_text_separates_requested_enrichment_from_the_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requested cycles report their own count and stage, not the timer's.

    A `--full` reparse drives thousands of requested cycles while the timer
    sleeps in `wait`; reading them as timer iterations, or as a timer stage
    whose age keeps growing, is what made the loop look wedged (REQ-ADAPT-012).
    """
    status = _base_status()
    status.update(
        {
            "embed_phase_enabled": True,
            "embed_loop_iterations": 5,
            "embed_loop_stage": "wait",
            "embed_loop_stage_at": time.time() - 240,
            "embed_loop_last_outcome": "drained",
            "embed_loop_last_trigger": "requested",
            "embed_loop_next_interval": 300.0,
            "embed_requested_cycles": 812,
            "embed_requested_at": time.time() - 3,
            "embed_requested_stage": "commit",
            "embed_requested_stage_at": time.time() - 3,
        }
    )
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error",
        lambda *_args, **_kwargs: status,
    )

    result = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

    assert result.exit_code == 0
    assert "Embed loop: iteration=5 stage=wait" in result.stdout
    assert "Embed requested: cycles=812 stage=commit stage_age=3s" in result.stdout


def test_daemon_status_text_reports_embed_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    status = _base_status()
    status.update({"embed_last_error": "RuntimeError: model unavailable"})
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error",
        lambda *_args, **_kwargs: status,
    )

    result = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

    assert result.exit_code == 0
    assert "Embed last error: RuntimeError: model unavailable" in result.stdout


def test_daemon_status_text_reports_embed_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sessions skipped for cooldown must be visible, not merely slow (REQ-ADAPT-016)."""
    status = _base_status()
    status.update(
        {
            "embed_phase_enabled": True,
            "embed_cooldown_sessions": 2,
            "embed_cooldown_until": 1789613611.0,
        }
    )
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error",
        lambda *_args, **_kwargs: status,
    )

    result = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

    assert result.exit_code == 0
    assert "Embed cooldown: 2 session(s) skipped until" in result.stdout

    result = CliRunner().invoke(app, ["daemon", "status", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["embed_cooldown_sessions"] == 2
    assert payload["embed_cooldown_until"] == 1789613611.0


def test_daemon_status_text_renders_fts_sidecar_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error",
        lambda *_args, **_kwargs: _base_status(sidecar_enabled=True),
    )

    result = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

    assert result.exit_code == 0
    assert "FTS sidecar: enabled=true" in result.stdout
    assert "bootstrap: messages=7 done=true | tool_calls=3 done=true" in result.stdout
    assert "reconcile (last run 2026-05-28T15:30:00+00:00):" in result.stdout
    assert "drained=messages=1, tool_calls=2" in result.stdout
    assert "error: —" in result.stdout


def test_daemon_status_text_dates_the_last_fatal_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-RESIL-014: an undated signature reads as live when it is months old."""
    status = _base_status()
    status["runtime_status"] = {
        "last_fatal_signature": "fresh_index:FatalException:database has been invalidated",
        "fatal_repeat_count": 1,
        "last_fatal_at": "2026-09-17T09:54:02",
    }
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error",
        lambda *_args, **_kwargs: status,
    )

    result = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

    assert result.exit_code == 0
    assert (
        "Last fatal: fresh_index:FatalException:database has been invalidated "
        "(x1) at 2026-09-17T09:54:02" in result.stdout
    )


def test_daemon_status_text_renders_disabled_under_duckdb_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error",
        lambda *_args, **_kwargs: _base_status(sidecar_enabled=False),
    )

    result = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

    assert result.exit_code == 0
    assert "FTS sidecar: disabled (backend=duckdb)" in result.stdout
    assert "bootstrap: messages=" not in result.stdout


def _reconciliation(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rpc_ready": True,
        "catalog_scan_complete": False,
        "live_observation_ready": True,
        "raw_indexing_ready": False,
        "keyword_search_ready": False,
        "enrichment_ready": False,
        "enrichment_deferred": None,
        "pending": 27196,
        "coverage": [],
        "source_page": [],
        "next_cursor": None,
        "error": None,
    }
    payload.update(overrides)
    return payload


def test_daemon_status_text_renders_epoch_fields_as_iso_with_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text output states runtime timestamps the way its ISO siblings read."""
    last_batch_at = datetime(2026, 9, 17, 22, 8, 4).timestamp()
    last_iteration_at = datetime(2026, 9, 17, 22, 5, 12).timestamp()
    monkeypatch.setattr(time, "time", lambda: last_batch_at + 180.0)
    status = _base_status()
    status.update(
        {
            "reconciliation": _reconciliation(),
            "embed_phase_enabled": True,
            "embed_model_loaded": True,
            "embed_pending": 600924,
            "embed_last_batch_at": last_batch_at,
            "embed_last_batch_size": 169,
            "embed_last_batch_duration": 0.63,
            "embed_loop_iterations": 3,
            "embed_loop_last_iteration_at": last_iteration_at,
            "embed_loop_stage": "wait",
            "embed_loop_stage_at": last_batch_at + 168.0,
            "embed_loop_last_outcome": "deferred",
            "embed_loop_next_interval": 300,
        }
    )
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error",
        lambda *_args, **_kwargs: status,
    )

    result = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

    assert result.exit_code == 0, result.output
    assert _BARE_EPOCH.search(result.stdout) is None, result.stdout
    assert "last_batch=2026-09-17T22:08:04 (3m ago)" in result.stdout
    assert "last_iteration=2026-09-17T22:05:12 (6m ago)" in result.stdout
    assert "stage_age=12s" in result.stdout


def test_daemon_status_text_leads_with_an_operator_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first lines answer version, embed drain and reconciliation drain."""
    last_batch_at = datetime(2026, 9, 17, 22, 8, 4).timestamp()
    monkeypatch.setattr(time, "time", lambda: last_batch_at + 180.0)
    status = _base_status()
    status.update(
        {
            "reconciliation": _reconciliation(
                pending=27196,
                enrichment_deferred="load 20.3 > threshold 5.4",
            ),
            "daemon_version": "0.31.0",
            "binary_version": "0.31.0",
            "daemon_pid": 4242,
            "scheduler": "launchd",
            "resolved_mode": "watch",
            "embed_phase_enabled": True,
            "embed_pending": 600924,
            "embed_last_batch_at": last_batch_at,
            "embed_last_batch_size": 169,
            "embed_loop_last_outcome": "deferred",
            "bloat_ratio": 1.33,
            "bloat_ratio_threshold": 1.5,
        }
    )
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error",
        lambda *_args, **_kwargs: status,
    )

    result = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

    assert result.exit_code == 0, result.output
    summary = result.stdout.splitlines()[:5]
    assert summary[0] == "Daemon: version=0.31.0 mode=watch scheduler=launchd pid=4242"
    assert summary[1] == (
        "  embed: pending=600924 outcome=deferred "
        "last_batch=2026-09-17T22:08:04 (3m ago) deferred=load 20.3 > threshold 5.4"
    )
    assert summary[2] == (
        "  reconciliation: pending=27196 catalog_scan_complete=false keyword_search_ready=false"
    )
    assert summary[3] == "  health: version_drift=no bloat=1.33/1.50"
    assert summary[4] == ""


@pytest.mark.parametrize(
    ("counts", "suffix"),
    [
        ({"claude_code": 3, "grok": 1}, " index_only_sessions=4"),
        ({"claude_code": 0}, ""),
        ({}, ""),
    ],
)
def test_daemon_status_summary_counts_index_only_sessions(
    monkeypatch: pytest.MonkeyPatch, counts: dict[str, int], suffix: str
) -> None:
    """Sessions whose transcript survives only in the index are named when present."""
    status = _base_status()
    status["reconciliation"] = _reconciliation(index_only_sessions=counts)
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error",
        lambda *_args, **_kwargs: status,
    )

    result = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

    assert result.exit_code == 0, result.output
    assert (
        "  reconciliation: pending=27196 catalog_scan_complete=false keyword_search_ready=false"
        + suffix
    ) in result.stdout.splitlines()


def test_daemon_status_summary_names_unavailable_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An offline read reports missing coverage instead of a fabricated zero."""
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error",
        lambda *_args, **_kwargs: _base_status(),
    )

    result = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

    assert result.exit_code == 0, result.output
    assert "  reconciliation: coverage unavailable" in result.stdout
    # No daemon version answered, so the binary version is labelled as such.
    assert result.stdout.startswith("Daemon: version=unknown (binary 0.10.4) ")


def test_daemon_status_requests_a_small_source_page_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A default status must not carry a 100-row source page."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def rpc(method: str, params: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        calls.append((method, params))
        return _base_status()

    monkeypatch.setattr("recall.cli.daemon.rpc_call_or_error", rpc)

    result = CliRunner().invoke(app, ["daemon", "status", "--json"])

    assert result.exit_code == 0, result.output
    assert calls == [("recall.daemon_status", {"limit": 10})]


def test_daemon_status_default_page_still_falls_back_to_a_local_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forwarding the default limit must not break the daemon-down path (REQ-DAEMON-074)."""
    from recall.cli.contract import CliError, ErrorCode

    def fail_rpc(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise CliError(code=ErrorCode.RUNTIME, message="daemon down", exit_code=1)

    monkeypatch.setattr("recall.cli.daemon.rpc_call_or_error", fail_rpc)
    monkeypatch.setattr("recall.cli.daemon.daemon_status", lambda: _base_status())

    result = CliRunner().invoke(app, ["daemon", "status", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["reconciliation"]["rpc_ready"] is False
