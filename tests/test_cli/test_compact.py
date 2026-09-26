from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest
from conftest import _can_acquire_duckdb_lock, _can_invoke_crontab
from recall.cli.app import app
from recall.core.config import AppConfig
from recall.core.types import SchedulerKind
from recall.db import connect
from typer.testing import CliRunner

_requires_duckdb_lock = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)
_requires_callable_crontab = pytest.mark.skipif(
    not _can_invoke_crontab(),
    reason="crontab binary not callable by current user",
)


def _init_empty_db(tmp_path: Path, monkeypatch) -> AppConfig:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    monkeypatch.setenv("RECALL_EMBED_BACKEND", "onnx")
    config = AppConfig.load()
    conn = connect(config)
    conn.close()
    return config


def _mark_daemon_running(config: AppConfig) -> None:
    (config.data_dir / "recall.sock").touch()
    (config.data_dir / "recall.pid").write_text(str(os.getpid()), encoding="utf-8")


@_requires_duckdb_lock
def test_compact_dry_run_reports_before_stats_without_touching_files(tmp_path, monkeypatch) -> None:
    config = _init_empty_db(tmp_path, monkeypatch)
    before_mtime = config.db_path.stat().st_mtime_ns

    result = CliRunner().invoke(app, ["compact", "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["before_bytes"] > 0
    assert payload["before_live_bytes"] >= 0
    assert isinstance(payload["before_ratio"], float)
    assert payload["after_bytes"] is None
    assert payload["elapsed_seconds"] is None
    assert payload["tables_copied"] == {}
    assert payload["daemon_was_running"] is False
    assert payload["daemon_restarted"] is False
    assert payload["skipped_reason"] == "dry_run"
    assert config.db_path.stat().st_mtime_ns == before_mtime
    assert not config.db_path.with_name(f"{config.db_path.name}.compact").exists()


@_requires_duckdb_lock
def test_compact_dry_run_daemon_lock_conflict_is_read_only(tmp_path, monkeypatch) -> None:
    config = _init_empty_db(tmp_path, monkeypatch)
    _mark_daemon_running(config)

    monkeypatch.setattr(
        "recall.cli.compact.estimate_bloat_ratio",
        lambda _db_path, _config: (_ for _ in ()).throw(duckdb.IOException("Conflicting lock")),
    )
    monkeypatch.setattr(
        "recall.cli.compact._stop_daemon",
        lambda _config: pytest.fail("dry-run must not stop daemon"),
    )

    result = CliRunner().invoke(app, ["compact", "--dry-run", "--json"])

    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "RUNTIME"
    assert "recall daemon stop" in payload["error"]["message"]
    assert (config.data_dir / "recall.sock").exists()
    assert (config.data_dir / "recall.pid").exists()


@_requires_duckdb_lock
@_requires_callable_crontab
def test_compact_threshold_skip_reports_below_threshold(tmp_path, monkeypatch) -> None:
    _init_empty_db(tmp_path, monkeypatch)

    result = CliRunner().invoke(app, ["compact", "--threshold", "100", "--yes", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["before_bytes"] > 0
    assert payload["after_bytes"] is None
    assert payload["tables_copied"] == {}
    assert payload["skipped_reason"] == "below_threshold"


@_requires_duckdb_lock
def test_compact_stops_daemon_holding_advisory_lock(tmp_path, monkeypatch) -> None:
    config = _init_empty_db(tmp_path, monkeypatch)
    pid_path = config.data_dir / "recall.pid"
    socket_path = config.data_dir / "recall.sock"
    lock_holder = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import fcntl, pathlib, signal, sys, time\n"
                "path = pathlib.Path(sys.argv[1])\n"
                "path.parent.mkdir(parents=True, exist_ok=True)\n"
                "handle = path.open('w', encoding='utf-8')\n"
                "fcntl.flock(handle, fcntl.LOCK_EX)\n"
                "print('locked', flush=True)\n"
                "signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))\n"
                "while True:\n"
                "    time.sleep(0.1)\n"
            ),
            str(config.lock_path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert lock_holder.stdout is not None
        assert lock_holder.stdout.readline().strip() == "locked"
        pid_path.write_text(str(lock_holder.pid), encoding="utf-8")
        socket_path.touch()

        def stop_lock_holding_daemon(_config, *, scheduler, pid, scheduler_pid):
            assert scheduler is None
            assert pid == lock_holder.pid
            assert scheduler_pid is None
            os.kill(lock_holder.pid, signal.SIGTERM)
            lock_holder.wait(timeout=5)
            pid_path.unlink(missing_ok=True)
            socket_path.unlink(missing_ok=True)

        monkeypatch.setattr(
            "recall.cli.compact._process_is_alive",
            lambda pid: pid == lock_holder.pid,
        )
        monkeypatch.setattr(
            "recall.cli.compact.daemon_status",
            lambda _config: SimpleNamespace(installed=False, scheduler=None),
        )
        monkeypatch.setattr(
            "recall.cli.compact._stop_daemon_from_lifecycle",
            stop_lock_holding_daemon,
        )
        monkeypatch.setattr(
            "recall.cli.compact.estimate_bloat_ratio",
            lambda _db_path, _config: SimpleNamespace(file_size=10, live_bytes=5, ratio=2.0),
        )
        monkeypatch.setattr(
            "recall.cli.compact.compact",
            lambda _config: SimpleNamespace(
                before=SimpleNamespace(file_size=10, live_bytes=5, ratio=2.0),
                after=SimpleNamespace(file_size=6, live_bytes=5, ratio=1.2),
                tables_copied={"sessions": 1},
                elapsed_seconds=0.5,
                replaced=True,
                skipped_reason=None,
            ),
        )
        monkeypatch.setattr(
            "recall.cli.compact._restart_daemon_for_cli",
            lambda _config, _scheduler: True,
        )

        result = CliRunner().invoke(app, ["compact", "--threshold", "1.0", "--yes", "--json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload.get("error", {}).get("code") != "LOCKED"
        assert payload["daemon_was_running"] is True
        assert lock_holder.poll() is not None
    finally:
        if lock_holder.poll() is None:
            lock_holder.terminate()
            lock_holder.wait(timeout=5)


@_requires_duckdb_lock
def test_compact_does_not_leave_service_sentinel_on_failure(tmp_path, monkeypatch) -> None:
    config = _init_empty_db(tmp_path, monkeypatch)
    sentinel_path = config.data_dir / "recall.compacting"

    from recall.services.compaction import CompactionError

    monkeypatch.setattr("recall.cli.compact._stop_daemon", lambda _config: (False, None))
    monkeypatch.setattr(
        "recall.cli.compact.estimate_bloat_ratio",
        lambda _db_path, _config: SimpleNamespace(file_size=10, live_bytes=5, ratio=2.0),
    )

    def fail_while_sentinel_exists(_config):
        assert sentinel_path.exists()
        raise CompactionError("replacement failed")

    monkeypatch.setattr("recall.cli.compact.compact", fail_while_sentinel_exists)

    result = CliRunner().invoke(app, ["compact", "--yes", "--json"])

    assert result.exit_code == 3, result.output
    assert not sentinel_path.exists()


@_requires_duckdb_lock
def test_compact_requires_confirmation_before_daemon_stop(tmp_path, monkeypatch) -> None:
    config = _init_empty_db(tmp_path, monkeypatch)
    _mark_daemon_running(config)

    monkeypatch.setattr(
        "recall.cli.compact.compact",
        lambda _config: pytest.fail("confirmation must run before compaction"),
    )
    monkeypatch.setattr(
        "recall.cli.compact._stop_daemon",
        lambda _config: pytest.fail("confirmation must run before daemon stop"),
    )

    result = CliRunner().invoke(app, ["compact", "--json"])

    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "CONFIRMATION_REQUIRED"
    assert "Pass --yes" in payload["error"]["message"]
    assert (config.data_dir / "recall.sock").exists()
    assert (config.data_dir / "recall.pid").exists()


@_requires_duckdb_lock
def test_compact_requires_confirmation_without_daemon(tmp_path, monkeypatch) -> None:
    _init_empty_db(tmp_path, monkeypatch)

    monkeypatch.setattr(
        "recall.cli.compact.compact",
        lambda _config: pytest.fail("confirmation must run before compaction"),
    )

    result = CliRunner().invoke(app, ["compact", "--json"])

    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "CONFIRMATION_REQUIRED"
    assert "Pass --yes" in payload["error"]["message"]


def test_compact_schema_and_llms_manifest_expose_command() -> None:
    runner = CliRunner()

    schema_result = runner.invoke(app, ["schema", "compact"])
    assert schema_result.exit_code == 0
    schema = json.loads(schema_result.stdout)
    assert schema["command"] == "compact"
    assert any(option["name"] == "dry_run" for option in schema["options"])
    assert any(option["name"] == "no_restart" for option in schema["options"])
    assert any(option["name"] == "threshold" for option in schema["options"])

    manifest_result = runner.invoke(app, ["--llms"])
    assert manifest_result.exit_code == 0
    manifest = json.loads(manifest_result.stdout)
    assert "compact" in manifest["commands"]


@_requires_duckdb_lock
@_requires_callable_crontab
def test_compact_restarts_running_daemon(tmp_path, monkeypatch) -> None:
    config = _init_empty_db(tmp_path, monkeypatch)
    calls: list[str] = []

    _mark_daemon_running(config)
    monkeypatch.setattr("recall.cli.compact._process_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        "recall.cli.compact.daemon_status",
        lambda _config: SimpleNamespace(installed=False, scheduler=None),
    )
    monkeypatch.setattr(
        "recall.cli.compact._stop_auto_forked_daemon",
        lambda _pid: calls.append("stop"),
    )
    monkeypatch.setattr(
        "recall.cli.compact._wait_for_daemon_release",
        lambda _config, original_pid: None,
    )
    monkeypatch.setattr(
        "recall.cli.compact.compact",
        lambda cfg: SimpleNamespace(
            before=SimpleNamespace(file_size=10, live_bytes=5, ratio=2.0),
            after=SimpleNamespace(file_size=6, live_bytes=5, ratio=1.2),
            tables_copied={"sessions": 1},
            elapsed_seconds=0.5,
            replaced=True,
            skipped_reason=None,
        ),
    )
    monkeypatch.setattr(
        "recall.cli.compact._start_background",
        lambda _config, verbose: (
            calls.append("restart")
            or SimpleNamespace(pid=22222, log_path=str(config.data_dir / "logs/daemon.log"))
        ),
    )
    monkeypatch.setattr("recall.cli.compact._wait_for_pid_file", lambda _config, timeout: None)

    result = CliRunner().invoke(app, ["compact", "--yes", "--json"])

    assert result.exit_code == 0, result.output
    assert calls == ["stop", "restart"]
    payload = json.loads(result.stdout)
    assert payload["daemon_was_running"] is True
    assert payload["daemon_restarted"] is True


@_requires_duckdb_lock
def test_compact_sentinel_covers_daemon_stop_and_restart(tmp_path, monkeypatch) -> None:
    config = _init_empty_db(tmp_path, monkeypatch)
    sentinel_path = config.data_dir / "recall.compacting"
    observations: list[tuple[str, bool]] = []

    def stop_daemon_while_sentinel_exists(_config):
        observations.append(("stop", sentinel_path.exists()))
        return True, None

    def compact_while_sentinel_exists(_config):
        observations.append(("compact", sentinel_path.exists()))
        return SimpleNamespace(
            before=SimpleNamespace(file_size=10, live_bytes=5, ratio=2.0),
            after=SimpleNamespace(file_size=6, live_bytes=5, ratio=1.2),
            tables_copied={"sessions": 1},
            elapsed_seconds=0.5,
            replaced=True,
            skipped_reason=None,
        )

    def restart_daemon_while_sentinel_exists(_config, _scheduler):
        observations.append(("restart", sentinel_path.exists()))
        return True

    monkeypatch.setattr("recall.cli.compact._stop_daemon", stop_daemon_while_sentinel_exists)
    monkeypatch.setattr(
        "recall.cli.compact.estimate_bloat_ratio",
        lambda _db_path, _config: SimpleNamespace(file_size=10, live_bytes=5, ratio=2.0),
    )
    monkeypatch.setattr("recall.cli.compact.compact", compact_while_sentinel_exists)
    monkeypatch.setattr(
        "recall.cli.compact._restart_daemon_for_cli",
        restart_daemon_while_sentinel_exists,
    )

    result = CliRunner().invoke(app, ["compact", "--yes", "--json"])

    assert result.exit_code == 0, result.output
    assert observations == [("stop", True), ("compact", True), ("restart", True)]
    assert not sentinel_path.exists()


@_requires_duckdb_lock
@_requires_callable_crontab
def test_compact_no_restart_leaves_running_daemon_stopped(tmp_path, monkeypatch) -> None:
    config = _init_empty_db(tmp_path, monkeypatch)
    calls: list[str] = []

    _mark_daemon_running(config)
    monkeypatch.setattr("recall.cli.compact._process_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        "recall.cli.compact.daemon_status",
        lambda _config: SimpleNamespace(installed=False, scheduler=None),
    )
    monkeypatch.setattr(
        "recall.cli.compact._stop_auto_forked_daemon",
        lambda _pid: calls.append("stop"),
    )
    monkeypatch.setattr(
        "recall.cli.compact._wait_for_daemon_release",
        lambda _config, original_pid: None,
    )
    monkeypatch.setattr(
        "recall.cli.compact.compact",
        lambda cfg: SimpleNamespace(
            before=SimpleNamespace(file_size=10, live_bytes=5, ratio=2.0),
            after=SimpleNamespace(file_size=6, live_bytes=5, ratio=1.2),
            tables_copied={"sessions": 1},
            elapsed_seconds=0.5,
            replaced=True,
            skipped_reason=None,
        ),
    )
    monkeypatch.setattr("recall.cli.compact._restart_daemon", lambda _config, _scheduler: False)

    result = CliRunner().invoke(app, ["compact", "--no-restart", "--yes", "--json"])

    assert result.exit_code == 0, result.output
    assert calls == ["stop"]
    payload = json.loads(result.stdout)
    assert payload["daemon_was_running"] is True
    assert payload["daemon_restarted"] is False


@_requires_duckdb_lock
@_requires_callable_crontab
def test_compact_stops_pid_only_daemon_contract(tmp_path, monkeypatch) -> None:
    config = _init_empty_db(tmp_path, monkeypatch)
    calls: list[object] = []
    (config.data_dir / "recall.pid").write_text("12345", encoding="utf-8")

    monkeypatch.setattr("recall.cli.compact._process_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        "recall.cli.compact.daemon_status",
        lambda _config: SimpleNamespace(installed=False, scheduler=None),
    )
    monkeypatch.setattr(
        "recall.cli.compact._stop_auto_forked_daemon",
        lambda pid: calls.append(("stop", pid)),
    )
    monkeypatch.setattr(
        "recall.cli.compact._wait_for_daemon_release",
        lambda _config, original_pid: calls.append(("wait", original_pid)),
    )

    from recall.cli.compact import _stop_daemon

    daemon_was_running, scheduler = _stop_daemon(config)

    assert daemon_was_running is True
    assert scheduler is None
    assert calls == [("stop", 12345), ("wait", 12345)]


@_requires_duckdb_lock
@_requires_callable_crontab
def test_compact_ignores_dead_pidfile(tmp_path, monkeypatch) -> None:
    config = _init_empty_db(tmp_path, monkeypatch)
    (config.data_dir / "recall.pid").write_text("999999999\n", encoding="utf-8")

    monkeypatch.setattr(
        "recall.cli.compact._stop_auto_forked_daemon",
        lambda _pid: pytest.fail("dead pidfile must not be stopped"),
    )
    monkeypatch.setattr(
        "recall.cli.compact._wait_for_daemon_release",
        lambda _config, original_pid: pytest.fail("dead pidfile must not be waited on"),
    )
    monkeypatch.setattr(
        "recall.cli.compact.estimate_bloat_ratio",
        lambda _db_path, _config: SimpleNamespace(file_size=10, live_bytes=5, ratio=2.0),
    )
    monkeypatch.setattr(
        "recall.cli.compact.compact",
        lambda cfg: SimpleNamespace(
            before=SimpleNamespace(file_size=10, live_bytes=5, ratio=2.0),
            after=SimpleNamespace(file_size=6, live_bytes=5, ratio=1.2),
            tables_copied={"sessions": 1},
            elapsed_seconds=0.5,
            replaced=True,
            skipped_reason=None,
        ),
    )

    result = CliRunner().invoke(app, ["compact", "--yes", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["daemon_was_running"] is False
    assert payload["daemon_restarted"] is False


@_requires_duckdb_lock
@_requires_callable_crontab
def test_compact_ignores_socket_only_daemon_contract(tmp_path, monkeypatch) -> None:
    config = _init_empty_db(tmp_path, monkeypatch)
    calls: list[object] = []
    (config.data_dir / "recall.sock").touch()

    monkeypatch.setattr(
        "recall.cli.compact.daemon_status",
        lambda _config: SimpleNamespace(installed=False, scheduler=None),
    )
    monkeypatch.setattr(
        "recall.cli.compact._stop_auto_forked_daemon",
        lambda pid: calls.append(("stop", pid)),
    )
    monkeypatch.setattr(
        "recall.cli.compact._wait_for_daemon_release",
        lambda _config, original_pid: calls.append(("wait", original_pid)),
    )

    from recall.cli.compact import _stop_daemon

    daemon_was_running, scheduler = _stop_daemon(config)

    assert daemon_was_running is False
    assert scheduler is None
    assert calls == []


@_requires_duckdb_lock
@_requires_callable_crontab
def test_compact_threshold_skip_restarts_running_daemon(tmp_path, monkeypatch) -> None:
    config = _init_empty_db(tmp_path, monkeypatch)
    calls: list[str] = []

    _mark_daemon_running(config)
    monkeypatch.setattr("recall.cli.compact._process_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        "recall.cli.compact.daemon_status",
        lambda _config: SimpleNamespace(installed=False, scheduler=None),
    )
    monkeypatch.setattr(
        "recall.cli.compact._stop_auto_forked_daemon",
        lambda _pid: calls.append("stop"),
    )
    monkeypatch.setattr(
        "recall.cli.compact._wait_for_daemon_release",
        lambda _config, original_pid: calls.append("wait"),
    )
    monkeypatch.setattr(
        "recall.cli.compact.estimate_bloat_ratio",
        lambda _db_path, _config: SimpleNamespace(file_size=10, live_bytes=5, ratio=2.0),
    )
    monkeypatch.setattr(
        "recall.cli.compact.compact",
        lambda _config: pytest.fail("below-threshold compact must not rebuild"),
    )
    monkeypatch.setattr(
        "recall.cli.compact._start_background",
        lambda _config, verbose: (
            calls.append("restart")
            or SimpleNamespace(pid=22222, log_path=str(config.data_dir / "logs/daemon.log"))
        ),
    )
    monkeypatch.setattr("recall.cli.compact._wait_for_pid_file", lambda _config, timeout: None)

    result = CliRunner().invoke(app, ["compact", "--yes", "--threshold", "1000", "--json"])

    assert result.exit_code == 0, result.output
    assert calls == ["stop", "wait", "restart"]
    payload = json.loads(result.stdout)
    assert payload["skipped_reason"] == "below_threshold"
    assert payload["daemon_was_running"] is True
    assert payload["daemon_restarted"] is True


@_requires_duckdb_lock
def test_compact_stops_and_restarts_systemd_timer_lifecycle(tmp_path, monkeypatch) -> None:
    config = _init_empty_db(tmp_path, monkeypatch)
    calls: list[object] = []
    systemd_dir = tmp_path / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True)
    (systemd_dir / "recall-daemon.service").write_text(
        "[Service]\nType=oneshot\nExecStart=recall daemon --once\n",
        encoding="utf-8",
    )
    (systemd_dir / "recall-daemon.timer").write_text(
        "[Timer]\nOnUnitActiveSec=300\n", encoding="utf-8"
    )
    (config.data_dir / "recall.pid").write_text("12345", encoding="utf-8")
    (config.data_dir / "recall.sock").touch()

    monkeypatch.setattr("recall.cli.compact._process_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        "recall.cli.compact._lifecycle_scheduler_candidates",
        lambda _config: (SchedulerKind.SYSTEMD,),
    )
    monkeypatch.setattr("recall.cli.compact._systemd_daemon_pid", lambda: 12345)
    monkeypatch.setattr(
        "recall.cli.compact._run_lifecycle_command",
        lambda cmd, action: calls.append((cmd, action)),
    )
    monkeypatch.setattr(
        "recall.cli.compact._wait_for_daemon_release",
        lambda _config, original_pid: calls.append(("wait", original_pid)),
    )

    from recall.cli.compact import _restart_daemon, _stop_daemon

    daemon_was_running, scheduler = _stop_daemon(config)
    restarted = _restart_daemon(config, scheduler)

    assert daemon_was_running is True
    assert scheduler == SchedulerKind.SYSTEMD
    assert restarted is True
    assert calls == [
        (["systemctl", "--user", "stop", "recall-daemon.timer"], "stop systemd timer"),
        (["systemctl", "--user", "stop", "recall-daemon.service"], "stop systemd daemon"),
        ("wait", 12345),
        (["systemctl", "--user", "start", "recall-daemon.timer"], "restart systemd daemon"),
    ]


@_requires_duckdb_lock
def test_compact_wait_for_release_requires_pidfile_removal_and_pid_exit(
    tmp_path, monkeypatch
) -> None:
    config = _init_empty_db(tmp_path, monkeypatch)
    pid_path = config.data_dir / "recall.pid"
    pid_path.write_text("12345", encoding="utf-8")
    (config.data_dir / "recall.sock").touch()
    alive_checks = [True, False]

    def process_is_alive(_pid: int) -> bool:
        alive = alive_checks.pop(0)
        if not alive:
            pid_path.unlink()
        return alive

    monkeypatch.setattr("recall.cli.compact.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("recall.cli.compact._process_is_alive", process_is_alive)

    from recall.cli.compact import _wait_for_daemon_release

    _wait_for_daemon_release(config, original_pid=12345)

    assert alive_checks == []
    assert (config.data_dir / "recall.sock").exists()


def test_wait_for_daemon_release_unlinks_stale_pidfile_after_sigkill(tmp_path, monkeypatch) -> None:
    """launchd bootout SIGKILLs the daemon if it doesn't exit cleanly,
    so atexit pidfile cleanup never runs. The wait helper must detect that the
    pid in the pidfile matches the (dead) original_pid and unlink the stale file
    rather than waiting forever for atexit that will never fire."""
    config = _init_empty_db(tmp_path, monkeypatch)
    pid_path = config.data_dir / "recall.pid"
    pid_path.write_text("99999", encoding="utf-8")

    monkeypatch.setattr("recall.cli.compact.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("recall.cli.compact._process_is_alive", lambda _pid: False)

    from recall.cli.compact import _wait_for_daemon_release

    _wait_for_daemon_release(config, original_pid=99999, timeout=1.0)

    assert not pid_path.exists()


def test_wait_for_daemon_release_does_not_unlink_respawn_pidfile(tmp_path, monkeypatch) -> None:
    """If the pidfile holds a DIFFERENT pid than original (respawn), keep
    waiting for the new daemon to release it itself — do not nuke it."""
    config = _init_empty_db(tmp_path, monkeypatch)
    pid_path = config.data_dir / "recall.pid"
    # Pidfile points at a respawn PID, not the original we killed.
    pid_path.write_text("99998", encoding="utf-8")

    monkeypatch.setattr("recall.cli.compact.time.sleep", lambda _seconds: None)
    # Original pid is dead; respawn pid is alive.
    monkeypatch.setattr("recall.cli.compact._process_is_alive", lambda pid: pid == 99998)

    from recall.cli.compact import _wait_for_daemon_release

    with pytest.raises(RuntimeError, match="did not release pidfile"):
        _wait_for_daemon_release(config, original_pid=99999, timeout=0.05)

    # Respawn's pidfile must remain — we did not own it.
    assert pid_path.read_text(encoding="utf-8").strip() == "99998"


def test_wait_for_daemon_release_keeps_waiting_when_unlink_fails(tmp_path, monkeypatch) -> None:
    """If stale-pidfile unlink fails, the wait must NOT claim success — fall
    through and ultimately time out so compact never proceeds while the
    authoritative pidfile is still on disk."""
    config = _init_empty_db(tmp_path, monkeypatch)
    pid_path = config.data_dir / "recall.pid"
    pid_path.write_text("99999", encoding="utf-8")

    monkeypatch.setattr("recall.cli.compact.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("recall.cli.compact._process_is_alive", lambda _pid: False)
    # Force unlink to fail (e.g., simulate read-only filesystem).
    monkeypatch.setattr("recall.cli.compact._unlink_stale_pidfile", lambda _path: False)

    from recall.cli.compact import _wait_for_daemon_release

    with pytest.raises(RuntimeError, match="did not release pidfile"):
        _wait_for_daemon_release(config, original_pid=99999, timeout=0.05)

    # Pidfile still on disk — we did not lie about a clean release.
    assert pid_path.exists()


def test_wait_for_launchd_unit_unloaded_polls_keepalive_plists(tmp_path, monkeypatch) -> None:
    """The wait helper must poll launchctl for KeepAlive plists, not
    only StartInterval ones. _launchd_poll_scheduler_loaded() gates on
    StartInterval and is therefore unsuitable — verify the helper uses the
    plist-mode-independent probe instead."""
    poll_calls: list[bool] = []

    fake_responses = [
        SimpleNamespace(returncode=0, stdout="loaded", stderr=""),
        SimpleNamespace(returncode=1, stdout="", stderr="Could not find service"),
    ]

    def fake_lifecycle_query(args: list[str]) -> SimpleNamespace | None:
        if args[:2] == ["launchctl", "print"]:
            response = fake_responses.pop(0)
            poll_calls.append(response.returncode == 0)
            return response
        return None

    monkeypatch.setattr("recall.cli.compact.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("recall.cli.compact._run_lifecycle_query", fake_lifecycle_query)

    from recall.cli.compact import _wait_for_launchd_unit_unloaded

    _wait_for_launchd_unit_unloaded(timeout=5.0, interval=0.01)

    # Must have polled at least twice: once seeing loaded, then unloaded.
    assert poll_calls == [True, False]


def test_wait_for_launchd_unit_unloaded_returns_when_unloaded(monkeypatch) -> None:
    """Bootout is async; the wait helper polls until the unit is gone."""
    loaded_results = [True, True, False]

    def fake_poll() -> bool:
        return loaded_results.pop(0)

    monkeypatch.setattr("recall.cli.compact.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("recall.cli.compact._launchctl_unit_is_loaded", fake_poll)

    from recall.cli.compact import _wait_for_launchd_unit_unloaded

    _wait_for_launchd_unit_unloaded(timeout=5.0, interval=0.01)

    assert loaded_results == []


def test_wait_for_launchd_unit_unloaded_raises_on_timeout(monkeypatch) -> None:
    """If launchctl never reports the unit unloaded, raise so compact aborts cleanly."""
    monkeypatch.setattr("recall.cli.compact.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("recall.cli.compact._launchctl_unit_is_loaded", lambda: True)

    from recall.cli.compact import _wait_for_launchd_unit_unloaded

    with pytest.raises(RuntimeError, match="still loaded"):
        _wait_for_launchd_unit_unloaded(timeout=0.05, interval=0.01)


def test_compact_launchd_stop_waits_for_unit_unloaded_before_pidfile_wait(
    tmp_path, monkeypatch
) -> None:
    """Bootout returns async; wait-for-unloaded must run before the pidfile poll.

    Without the wait, _wait_for_daemon_release could
    observe a respawned daemon's fresh pidfile and time out with LOCKED.
    """
    config = _init_empty_db(tmp_path, monkeypatch)
    pid_path = config.data_dir / "recall.pid"
    pid_path.write_text("12345", encoding="utf-8")
    (config.data_dir / "recall.sock").touch()

    call_order: list[str] = []

    def fake_run_command(args: list[str], action: str) -> None:
        if action == "stop launchd daemon":
            call_order.append("bootout")

    loaded_results = [True, False]

    def fake_poll_loaded() -> bool:
        call_order.append("poll_loaded")
        return loaded_results.pop(0)

    def fake_stop_unowned(_pid, _sched_pid) -> None:
        call_order.append("stop_unowned")
        pid_path.unlink(missing_ok=True)

    def fake_process_is_alive(_pid: int) -> bool:
        return False

    monkeypatch.setattr("recall.cli.compact.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("recall.cli.compact._run_lifecycle_command", fake_run_command)
    monkeypatch.setattr("recall.cli.compact._launchctl_unit_is_loaded", fake_poll_loaded)
    monkeypatch.setattr("recall.cli.compact._stop_pid_file_daemon_if_unowned", fake_stop_unowned)
    monkeypatch.setattr("recall.cli.compact._process_is_alive", fake_process_is_alive)

    from recall.cli.compact import _stop_daemon_from_lifecycle

    _stop_daemon_from_lifecycle(
        config, scheduler=SchedulerKind.LAUNCHD, pid=12345, scheduler_pid=12345
    )

    # Bootout must happen first, THEN the wait-for-unloaded poll, THEN the
    # pid-file fallback. Pre-fix the wait step was missing, so a respawn between
    # bootout and pidfile poll could wedge the wait.
    assert call_order[0] == "bootout"
    assert "poll_loaded" in call_order
    assert call_order.index("poll_loaded") < call_order.index("stop_unowned")
    assert loaded_results == []


def test_compact_text_output_is_human_readable(capsys) -> None:
    from recall.cli.compact import _emit_compact_result
    from recall.cli.contract import OutputFormat

    _emit_compact_result(
        {
            "before_bytes": 10,
            "before_live_bytes": 5,
            "before_ratio": 2.0,
            "after_bytes": 6,
            "after_live_bytes": 5,
            "after_ratio": 1.2,
            "elapsed_seconds": 0.5,
            "tables_copied": {"sessions": 1},
            "daemon_was_running": True,
            "daemon_restarted": True,
            "skipped_reason": None,
        },
        output_format=OutputFormat.TEXT,
        fields=None,
        cta=False,
    )
    captured = capsys.readouterr()
    output = captured.out

    assert "Before size: 10 bytes" in output
    assert "Before ratio: 2.00" in output
    assert "sessions: 1 rows" in captured.err
    assert "After size: 6 bytes" in output
    assert "Elapsed seconds: 0.500" in output
    assert "Daemon restart: restarted" in output
    assert not output.startswith("{")
