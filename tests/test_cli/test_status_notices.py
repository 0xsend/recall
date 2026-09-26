from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from conftest import _can_acquire_duckdb_lock, set_in_process_server
from recall.cli.app import app
from recall.cli.status_notices import (
    _freshness_staleness_notice,
    _mode_mismatch_notice,
    render_status_notice,
)
from recall.core.config import AppConfig
from recall.core.types import RunKind
from recall.services.indexer import index_sessions
from recall.services.rpc_server import RpcServer
from recall.services.runtime_state import (
    IndexRunCounts,
    record_run_failure,
    record_run_success,
)
from typer.testing import CliRunner

_requires_duckdb_lock = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)


def _install_codex_fixture(tmp_path: Path) -> None:
    target = tmp_path / ".codex" / "sessions" / "s1"
    target.mkdir(parents=True)
    fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "codex" / "session1" / "rollout.jsonl"
    )
    shutil.copy(fixture, target / "rollout.jsonl")


def _setup_env(tmp_path: Path, monkeypatch) -> AppConfig:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    monkeypatch.setenv("RECALL_CLI_STATUS_NOTICES", "true")
    # These tests cover notice rendering through in-process RPC; host crontab
    # permissions are exercised by daemon scheduler tests.
    monkeypatch.setattr("recall.services.daemon._read_crontab", lambda: "")
    _install_codex_fixture(tmp_path)
    index_sessions(source=None, full=True, recreate=True, verbose=False)
    config = AppConfig.load()
    server = RpcServer(config=config)
    set_in_process_server(server)
    return config


@_requires_duckdb_lock
def test_list_prints_status_notice_for_human_output(tmp_path, monkeypatch) -> None:
    _setup_env(tmp_path, monkeypatch)

    runner = CliRunner()
    result = runner.invoke(app, ["list", "--format", "text"])

    assert result.exit_code == 0
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert lines[0].startswith("Index:")
    assert "Daemon:" not in lines[0]


@_requires_duckdb_lock
def test_json_commands_suppress_status_notices(tmp_path, monkeypatch) -> None:
    _setup_env(tmp_path, monkeypatch)

    runner = CliRunner()

    list_result = runner.invoke(app, ["list", "--json"])
    assert list_result.exit_code == 0
    assert "Index:" not in list_result.stdout
    assert isinstance(json.loads(list_result.stdout), list)

    search_result = runner.invoke(app, ["search", "git", "--json"])
    assert search_result.exit_code == 0
    assert "Index:" not in search_result.stdout
    json.loads(search_result.stdout)


def test_daemon_json_requires_once(monkeypatch) -> None:
    started = False

    def fake_start(**_kwargs):
        nonlocal started
        started = True

    monkeypatch.setattr("recall.cli.daemon._start_foreground_server", fake_start)

    runner = CliRunner()
    result = runner.invoke(app, ["daemon", "--json"])

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "VALIDATION"
    assert "structured output is only supported with --once" in payload["error"]["message"]
    assert started is False


@_requires_duckdb_lock
def test_status_notice_reports_last_success_not_later_failure(tmp_path, monkeypatch) -> None:
    config = _setup_env(tmp_path, monkeypatch)

    conn = duckdb.connect(str(config.db_path))
    try:
        success_at = datetime.now(UTC) - timedelta(hours=3)
        record_run_success(
            conn,
            run_kind=RunKind.INDEX,
            index_summary=IndexRunCounts(total=1, indexed=1, skipped=0, failed=0, changed=1),
            attempted_at=success_at,
            successful_at=success_at,
        )
        failure_at = success_at + timedelta(hours=1)
        record_run_failure(
            conn,
            run_kind=RunKind.DAEMON_SCHEDULED,
            message="simulated failure",
            attempted_at=failure_at,
            failed_at=failure_at,
        )
    finally:
        conn.close()

    # Re-create server so it picks up the new DB state
    server = RpcServer(config=config)
    set_in_process_server(server)

    notice = render_status_notice(config)

    assert notice is not None
    assert "via daemon-scheduled" not in notice
    assert "Daemon:" not in notice


@_requires_duckdb_lock
def test_status_notice_suppressed_for_legacy_embed_run_kind(tmp_path, monkeypatch) -> None:
    """Historical DBs with last_run_kind='embed' must not produce a misleading index notice."""
    config = _setup_env(tmp_path, monkeypatch)

    conn = duckdb.connect(str(config.db_path))
    try:
        conn.execute(
            """
            UPDATE runtime_state
            SET last_run_kind = 'embed',
                last_successful_at = CURRENT_TIMESTAMP,
                last_attempted_at = CURRENT_TIMESTAMP
            """
        )
    finally:
        conn.close()

    # Re-create server so it picks up the new DB state
    server = RpcServer(config=config)
    set_in_process_server(server)

    assert render_status_notice(config) is None


def test_freshness_notice_accepts_naive_last_successful_at(monkeypatch) -> None:
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 5, 26, 4, 0, tzinfo=UTC)

    monkeypatch.setattr("recall.cli.status_notices.datetime", FrozenDateTime)
    status = {
        "installed": True,
        "resolved_mode": "poll",
        "runtime_status": {
            "last_successful_at": "2026-05-26T02:49:50.703888",
        },
    }

    notice = _freshness_staleness_notice(status, interval_seconds=300)

    assert isinstance(notice, str)
    assert notice


def test_freshness_notice_accepts_naive_datetime_object(monkeypatch) -> None:
    """The status dict passes last_successful_at as a naive datetime object, not a
    string; it must be normalized to UTC before subtracting from now(UTC), else
    `recall daemon status` crashes with offset-naive/offset-aware TypeError."""

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 5, 26, 4, 0, tzinfo=UTC)

    monkeypatch.setattr("recall.cli.status_notices.datetime", FrozenDateTime)
    status = {
        "installed": True,
        "resolved_mode": "watch",
        "runtime_status": {
            # naive datetime object (no tzinfo) — the shape that crashed in the wild
            "last_successful_at": datetime(2026, 5, 26, 2, 49, 50, 703888),
        },
    }

    notice = _freshness_staleness_notice(status, interval_seconds=300)

    assert isinstance(notice, str)
    assert notice


# ---------------------------------------------------------------------------
# Mode mismatch notice (REQ-DAEMON-032)
# ---------------------------------------------------------------------------


def test_mode_mismatch_notice_poll_with_watch_available() -> None:
    """When installed in poll mode but auto resolves to watch, emit upgrade notice."""
    status = {
        "installed": True,
        "mode": "auto",
        "resolved_mode": "poll",
        "auto_resolved_mode": "watch",
    }
    notice = _mode_mismatch_notice(status)
    assert notice is not None
    assert "poll mode" in notice
    assert "watch mode" in notice
    assert "recall daemon install" in notice


@pytest.mark.parametrize(
    ("installed", "mode", "resolved_mode", "auto_resolved_mode"),
    [
        pytest.param(True, "auto", "watch", "watch", id="already-watch"),
        pytest.param(False, "auto", "poll", "watch", id="not-installed"),
        pytest.param(True, "auto", "poll", "poll", id="watchdog-unavailable"),
        pytest.param(True, "poll", "poll", "watch", id="explicit-poll-config"),
    ],
)
def test_mode_mismatch_notice_is_silent(
    installed: bool, mode: str, resolved_mode: str, auto_resolved_mode: str
) -> None:
    """No upgrade notice when already on watch, not installed, watch is
    unavailable, or the user explicitly configured poll."""
    status = {
        "installed": installed,
        "mode": mode,
        "resolved_mode": resolved_mode,
        "auto_resolved_mode": auto_resolved_mode,
    }
    assert _mode_mismatch_notice(status) is None


def test_mode_mismatch_notice_explicit_watch_with_poll_installed() -> None:
    """Notice when user configured watch but installed scheduler is still poll."""
    status = {
        "installed": True,
        "mode": "watch",
        "resolved_mode": "poll",
        "auto_resolved_mode": "watch",
    }
    notice = _mode_mismatch_notice(status)
    assert notice is not None, "watch config with poll install should trigger notice"
    assert "recall daemon install" in notice


def test_mode_mismatch_uses_installed_mode_not_resolved() -> None:
    """A running watch daemon must not mask a poll scheduler install."""
    status = {
        "installed": True,
        "mode": "auto",
        "installed_mode": "poll",
        "resolved_mode": "watch",
        "auto_resolved_mode": "watch",
    }

    notice = _mode_mismatch_notice(status)

    assert notice is not None
    assert "poll mode" in notice
    assert "watch mode" in notice
