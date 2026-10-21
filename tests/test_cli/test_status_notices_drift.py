from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from recall.cli.daemon import _print_daemon_status_text
from recall.cli.status_notices import _render_status_notice_from_status


def _status(*, installed: bool = True, mode: str = "poll") -> dict:
    return {
        "installed": installed,
        "scheduler": "launchd",
        "mode": mode,
        "resolved_mode": mode,
        "auto_resolved_mode": mode,
        "installed_binary_path": "/usr/local/bin/recall",
        "installed_binary_stale": False,
        "scheduler_last_exit_status": None,
        "scheduler_health_state": None,
        "runtime_status": {
            "last_successful_at": None,
            "last_failure_at": None,
            "last_run_kind": "daemon-scheduled",
            "last_index_summary": {
                "changed": 1,
                "indexed": 1,
                "skipped": 0,
                "failed": 0,
                "total_seconds": 0.42,
            },
            "installed_scheduler": "launchd",
        },
    }


def test_render_status_notice_warns_when_installed_but_never_successful() -> None:
    """REQ-DAEMON-037 supersedes the legacy silent None path."""
    status = _status()

    notice = _render_status_notice_from_status(status, interval_seconds=300)

    assert notice is not None
    assert "Daemon appears installed but has not completed a successful run yet" in notice
    assert "recall daemon install" in notice


def test_render_status_notice_prefers_binary_drift_over_other_warnings(tmp_path: Path) -> None:
    """REQ-DAEMON-038: drift notice wins over freshness and mode mismatch notices."""
    status = _status(mode="poll")
    status["auto_resolved_mode"] = "watch"
    status["installed_binary_stale"] = True
    status["installed_binary_path"] = str(tmp_path / "gone/recall")
    status["runtime_status"]["last_successful_at"] = (
        datetime.now(UTC) - timedelta(hours=12)
    ).isoformat()

    notice = _render_status_notice_from_status(status, interval_seconds=300)

    assert notice is not None
    assert f"installed daemon is pinned to {tmp_path / 'gone/recall'}" in notice
    assert "no longer exists" in notice
    assert "poll mode but watch mode is available" not in notice


def test_render_status_notice_reports_launchd_exit_78_with_human_hint() -> None:
    """REQ-DAEMON-039: launchd EX_CONFIG is mapped to a human-readable failure."""
    status = _status()
    status["scheduler_last_exit_status"] = 78
    status["scheduler_health_state"] = "failed"

    notice = _render_status_notice_from_status(status, interval_seconds=300)

    assert notice is not None
    assert "launchd last exited with status 78" in notice
    assert "could not execute the configured program" in notice
    assert "recall daemon install" in notice


def test_render_status_notice_includes_freshness_info_after_scheduler_warning() -> None:
    """Warnings may append the informational freshness line after the high-priority notice."""
    status = _status()
    status["scheduler_last_exit_status"] = 2
    status["scheduler_health_state"] = "failed"
    status["runtime_status"]["last_successful_at"] = (
        datetime.now(UTC) - timedelta(minutes=10)
    ).isoformat()

    notice = _render_status_notice_from_status(status, interval_seconds=300)

    assert notice is not None
    assert "launchd last exited with status 2" in notice
    assert "Index: last updated 10m ago" in notice


def test_daemon_status_text_surfaces_notice(capsys, monkeypatch) -> None:
    """`recall daemon status` surfaces the high-priority notice on stderr, keeping
    stdout reserved for the status report (REQ-CLI-012)."""
    status = _status()
    status["installed_binary_stale"] = True
    status["runtime_status"]["last_successful_at"] = (
        datetime.now(UTC) - timedelta(hours=6)
    ).isoformat()

    monkeypatch.setattr(
        "recall.cli.daemon.AppConfig.load",
        lambda: type("Config", (), {"daemon": type("Daemon", (), {"interval": 300})()})(),
    )
    monkeypatch.setattr(
        "recall.cli.status_notices.resolve_recall_binary", lambda: "~/.local/bin/recall"
    )

    _print_daemon_status_text(status)

    captured = capsys.readouterr()
    assert "Notice: Warning: installed daemon is pinned to /usr/local/bin/recall" in captured.err
    assert "Notice:" not in captured.out


def _answered_by_daemon(status: dict) -> dict:
    """The status as a running daemon's RPC serves it."""
    return {**status, "reconciliation": {"rpc_ready": True}}


def test_a_running_watch_daemon_idle_for_hours_is_not_called_stale() -> None:
    """REQ-DAEMON-037: an idle watch daemon records no run; reinstalling fixes nothing."""
    status = _answered_by_daemon(_status(mode="watch"))
    status["runtime_status"]["last_successful_at"] = (
        datetime.now(UTC) - timedelta(hours=22)
    ).isoformat()

    notice = _render_status_notice_from_status(status, interval_seconds=300) or ""

    assert "appears stale" not in notice
    assert "recall daemon install" not in notice


def test_a_running_daemon_that_never_committed_is_not_told_to_reinstall() -> None:
    status = _answered_by_daemon(_status(mode="watch"))

    notice = _render_status_notice_from_status(status, interval_seconds=300) or ""

    assert "has not completed a successful run" not in notice
    assert "recall daemon install" not in notice


def test_an_unreachable_installed_daemon_with_an_old_run_is_still_stale() -> None:
    status = _status(mode="watch")
    status["runtime_status"]["last_successful_at"] = (
        datetime.now(UTC) - timedelta(hours=22)
    ).isoformat()

    notice = _render_status_notice_from_status(status, interval_seconds=300)

    assert notice is not None
    assert "daemon appears stale" in notice
    assert "recall daemon install" in notice


def test_a_pinned_binary_that_exists_is_stated_without_install_advice(tmp_path: Path) -> None:
    """REQ-DAEMON-038: reinstalling from a dev shell would re-pin the service to it."""
    pinned = tmp_path / "stable/recall"
    pinned.parent.mkdir()
    pinned.write_text("#!/bin/sh\n")
    status = _answered_by_daemon(_status())
    status["installed_binary_path"] = str(pinned)
    status["installed_binary_stale"] = True

    notice = _render_status_notice_from_status(status, interval_seconds=300)
    structured = _render_status_notice_from_status(
        status, interval_seconds=300, include_informational=False
    )

    assert notice is not None
    assert f"installed daemon runs {pinned}" in notice
    assert "recall daemon install" not in notice
    assert structured is None or str(pinned) not in structured
