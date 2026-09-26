from __future__ import annotations

import functools
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import duckdb
import pytest
import recall.services.daemon as daemon_module
from conftest import _can_invoke_crontab
from recall.cli.daemon import _print_daemon_status_text
from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.types import DaemonMode, RunKind, SchedulerKind, Source
from recall.db import FtsSidecarUnavailableError, connect, create_fts_indexes, is_lock_conflict
from recall.services.daemon import (
    daemon_status,
    install_scheduler,
    restart_daemon_durable,
    run_startup_fts_sidecar_sync,
    start_daemon_durable,
    stop_daemon_durable,
    stop_daemon_soft,
    uninstall_scheduler,
)
from recall.services.runtime_state import record_run_success, set_installed_scheduler
from recall.services.watcher import WatcherLiveSnapshot

pytestmark = pytest.mark.usefixtures("launchd_without_legacy_job")


def _copy_codex_fixture(tmp_path: Path, name: str = "rollout.jsonl") -> Path:
    target = tmp_path / ".codex" / "sessions" / "s1"
    target.mkdir(parents=True, exist_ok=True)
    fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "codex" / "session1" / "rollout.jsonl"
    )
    destination = target / name
    shutil.copy(fixture, destination)
    return destination


def _app_config(tmp_path: Path, *, interval: int = 300) -> AppConfig:
    binary_path = tmp_path / "bin" / "recall"
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    binary_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    data_dir = tmp_path / ".local" / "share" / "recall"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / ".config" / "recall" / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(interval=interval, mode=DaemonMode.POLL),
        cli=CliConfig(),
    )


@contextmanager
def _database_held_by_live_daemon(config: AppConfig, *, pid_file: bool = True) -> Iterator[int]:
    """Hold the DuckDB file read-write from another process, as the daemon does.

    DuckDB's file lock is exclusive per process, so even a read-only open
    from this process conflicts; the pid file names the holder, as the
    daemon's does. Yields the holder's pid.
    """
    config.data_dir.mkdir(parents=True, exist_ok=True)
    connect(config).close()
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import duckdb, sys, time\n"
            "conn = duckdb.connect(sys.argv[1])\n"
            "print('held', flush=True)\n"
            "time.sleep(120)\n",
            str(config.db_path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        if pid_file:
            (config.data_dir / "recall.pid").write_text(f"{holder.pid}\n", encoding="utf-8")
        yield holder.pid
    finally:
        holder.kill()
        holder.wait(timeout=10)
        assert holder.stdout is not None
        holder.stdout.close()


def test_daemon_status_reports_a_live_daemon_instead_of_opening_its_database(
    tmp_path: Path,
) -> None:
    """REQ-DAEMON-074: the runtime fields live behind the daemon's RPC while a
    daemon holds the database; a local read under it is a lock error,
    not a status."""
    config = _app_config(tmp_path)

    with _database_held_by_live_daemon(config) as pid:
        status = daemon_status(config=config)

    assert status.daemon_pid == pid
    assert status.runtime_unavailable_reason is not None
    assert str(pid) in status.runtime_unavailable_reason
    assert status.runtime_status.last_attempted_at is None
    assert status.embed_pending == 0


def test_daemon_status_names_another_process_when_the_holder_wrote_no_pid_file(
    tmp_path: Path,
) -> None:
    """REQ-DAEMON-074: the lost race (or a foreground `recall index`) is a lock
    conflict with no live pid file; it must name another process and DuckDB's
    holder pid, never a daemon."""
    config = _app_config(tmp_path)

    with _database_held_by_live_daemon(config, pid_file=False) as pid:
        status = daemon_status(config=config)

    assert status.daemon_pid is None
    assert status.runtime_unavailable_reason is not None
    assert "another process" in status.runtime_unavailable_reason
    assert f"PID {pid}" in status.runtime_unavailable_reason
    assert "daemon pid" not in status.runtime_unavailable_reason
    assert status.runtime_status.last_attempted_at is None


def test_daemon_status_attributes_a_lost_lock_race_to_the_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-DAEMON-074: if the daemon writes its pid after the initial check but
    before the local database read, the lock conflict is attributed to that
    daemon rather than an unspecified process."""
    config = _app_config(tmp_path)

    with _database_held_by_live_daemon(config, pid_file=False) as pid:
        checks = 0

        def pid_appears_after_initial_check(_config: AppConfig) -> int | None:
            nonlocal checks
            checks += 1
            return None if checks == 1 else pid

        monkeypatch.setattr(
            daemon_module,
            "_running_daemon_pid_from_file",
            pid_appears_after_initial_check,
        )

        status = daemon_status(config=config)

    assert checks >= 2
    assert status.daemon_pid == pid
    assert status.runtime_unavailable_reason is not None
    assert f"daemon pid {pid}" in status.runtime_unavailable_reason
    assert "another process" not in status.runtime_unavailable_reason
    assert status.runtime_status.last_attempted_at is None


def test_daemon_status_reads_the_database_when_no_daemon_holds_it(tmp_path: Path) -> None:
    """REQ-DAEMON-074: a stale pid file (dead process) must not hide the runtime fields."""
    config = _app_config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(config)
    try:
        record_run_success(
            conn,
            run_kind=RunKind.INDEX,
            attempted_at=datetime(2026, 9, 1, 12, 0, 0),
            successful_at=datetime(2026, 9, 1, 12, 0, 5),
        )
    finally:
        conn.close()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=10)
    (config.data_dir / "recall.pid").write_text(f"{dead.pid}\n", encoding="utf-8")

    status = daemon_status(config=config)

    assert status.daemon_pid is None
    assert status.runtime_unavailable_reason is None
    assert status.runtime_status.last_successful_at is not None


def _sidecar_app_config(tmp_path: Path, *, backend: str = "sqlite_sidecar") -> AppConfig:
    return replace(_app_config(tmp_path), fts=FtsConfig(backend=backend))


def _seed_sidecar_sync_rows(config: AppConfig, *, messages: int = 2, tool_calls: int = 2) -> None:
    conn = connect(config)
    try:
        for index in range(messages):
            conn.execute(
                """
                INSERT INTO message_state (
                    message_id, role, content, thinking, has_thinking,
                    fts_content, fts_thinking
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    f"msg-{index:03d}",
                    "assistant",
                    f"raw {index}",
                    f"thinking {index}",
                    True,
                    f"content {index}",
                    f"thoughts {index}",
                ],
            )
        for index in range(tool_calls):
            conn.execute(
                """
                INSERT INTO tool_calls (
                    id, session_id, message_id, idx, tool_name, bash_command, is_compound
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    f"tc-{index:03d}",
                    "session-1",
                    None,
                    index,
                    "bash",
                    f"echo {index}",
                    False,
                ],
            )
    finally:
        conn.close()


def _legacy_fts_schema_count(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.schemata
        WHERE schema_name LIKE 'fts_main_%'
        """
    ).fetchone()
    assert row is not None
    return int(row[0])


@pytest.fixture(autouse=True)
def _force_onnx_embedding_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Daemon tests do not exercise MLX and should avoid host-specific MLX probe crashes."""
    monkeypatch.setenv("RECALL_EMBED_BACKEND", "onnx")


@pytest.fixture(autouse=True)
def _launchd_unload_wait_without_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Faked `launchctl print` always answers "loaded", so the real 30s unload
    deadline would expire on every launchd install. Keep the real probe but
    give it no deadline: the outcome under these fakes is identical."""
    monkeypatch.setattr(
        daemon_module,
        "_wait_for_launchd_unload",
        functools.partial(daemon_module._wait_for_launchd_unload, timeout_seconds=0.0),
    )


@pytest.fixture(autouse=True)
def _scheduler_metadata_without_external_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Artifact tests fake the scheduler, so no child daemon owns their metadata.

    Real process readiness and metadata persistence are exercised together in
    test_daemon_install_db_owner.py. Keep the artifact assertions independent
    of that transport while preserving their actual persisted state.
    """

    real_update = daemon_module._update_installed_scheduler

    def update(config, scheduler, *, require_rpc=False):
        try:
            conn = daemon_module.connect(config)
        except duckdb.IOException as err:
            if not is_lock_conflict(err):
                raise
            return real_update(config, scheduler, require_rpc=require_rpc)
        try:
            daemon_module.set_installed_scheduler(conn, scheduler)
            return daemon_status(config=config, conn=conn)
        finally:
            conn.close()

    monkeypatch.setattr(daemon_module, "_update_installed_scheduler", update)


def _skip_without_crontab_capability() -> None:
    if not _can_invoke_crontab():
        pytest.skip("crontab binary not callable; daemon status scheduler detection requires it")


class CompletedProcess:
    def __init__(self, args: list[str], returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.args = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _read_runtime_state(db_path: Path) -> tuple:
    """Read runtime_state row.

    Index layout:
      0: last_attempted_at
      1: last_successful_at
      2: last_run_kind
      3: last_index_total
      4: last_index_indexed
      5: last_index_skipped
      6: last_index_failed
      7: last_index_changed
      8: last_index_total_seconds
      9: last_failure_message
     10: last_failure_at
     11: installed_scheduler
    """
    conn = duckdb.connect(str(db_path))
    try:
        row = conn.execute(
            """
            SELECT
                last_attempted_at,
                last_successful_at,
                last_run_kind,
                last_index_total,
                last_index_indexed,
                last_index_skipped,
                last_index_failed,
                last_index_changed,
                last_index_total_seconds,
                last_failure_message,
                last_failure_at,
                installed_scheduler
            FROM runtime_state
            """
        ).fetchone()
        assert row is not None
        return tuple(row)
    finally:
        conn.close()


def test_run_startup_fts_sidecar_sync_runs_bootstrap_and_reconcile_under_sidecar_backend(
    tmp_path: Path,
) -> None:
    config = _sidecar_app_config(tmp_path)
    _seed_sidecar_sync_rows(config, messages=3, tool_calls=2)

    result = run_startup_fts_sidecar_sync(config)

    assert result.enabled is True
    assert result.error is None
    assert result.bootstrap_messages_processed == 3
    assert result.bootstrap_tool_calls_processed == 2
    assert result.bootstrap_messages_done is True
    assert result.bootstrap_tool_calls_done is True
    assert result.reconcile_pending_remaining == {"message": 0, "tool_call": 0}
    assert result.last_run_at is not None


def test_run_startup_fts_sidecar_sync_is_noop_under_duckdb_backend(tmp_path: Path) -> None:
    result = run_startup_fts_sidecar_sync(_sidecar_app_config(tmp_path, backend="duckdb"))

    assert result.enabled is False
    assert result.bootstrap_messages_processed == 0
    assert result.bootstrap_tool_calls_processed == 0
    assert result.reconcile_pending_drained == {"message": 0, "tool_call": 0}
    assert result.reconcile_orphans_backfilled == {"message": 0, "tool_call": 0}
    assert result.reconcile_ghosts_deleted == {"message": 0, "tool_call": 0}
    assert result.reconcile_pending_remaining == {"message": 0, "tool_call": 0}
    assert result.last_run_at is None
    assert result.error is None


def test_run_startup_fts_sidecar_sync_handles_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _sidecar_app_config(tmp_path)

    def unavailable(_path: Path):
        raise FtsSidecarUnavailableError("probe failed")

    monkeypatch.setattr(daemon_module, "open_sidecar", unavailable)

    result = run_startup_fts_sidecar_sync(config)

    assert result.enabled is True
    assert "probe failed" in (result.error or "")
    assert result.bootstrap_messages_processed == 0
    assert result.bootstrap_tool_calls_processed == 0
    assert result.reconcile_pending_remaining == {"message": 0, "tool_call": 0}
    assert result.last_run_at is None


def test_run_startup_fts_sidecar_sync_is_idempotent(tmp_path: Path) -> None:
    config = _sidecar_app_config(tmp_path)
    _seed_sidecar_sync_rows(config, messages=2, tool_calls=2)

    first = run_startup_fts_sidecar_sync(config)
    second = run_startup_fts_sidecar_sync(config)

    assert first.bootstrap_messages_processed == 2
    assert first.bootstrap_tool_calls_processed == 2
    assert second.bootstrap_messages_processed == 0
    assert second.bootstrap_tool_calls_processed == 0
    assert second.reconcile_pending_drained == {"message": 0, "tool_call": 0}
    assert second.reconcile_orphans_backfilled == {"message": 0, "tool_call": 0}
    assert second.reconcile_ghosts_deleted == {"message": 0, "tool_call": 0}


def test_run_startup_fts_sidecar_sync_preserves_legacy_fts_schemas(tmp_path: Path) -> None:
    config = _sidecar_app_config(tmp_path)
    _seed_sidecar_sync_rows(config, messages=1, tool_calls=1)
    conn = connect(config)
    try:
        create_fts_indexes(conn, replace(config.fts, backend="duckdb"))
        before = _legacy_fts_schema_count(conn)
    finally:
        conn.close()
    assert before > 0

    run_startup_fts_sidecar_sync(config)

    conn = connect(config)
    try:
        after = _legacy_fts_schema_count(conn)
    finally:
        conn.close()
    assert after == before


def test_install_scheduler_writes_launchd_plist_with_it_send_label(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    commands: list[list[str]] = []

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    status = install_scheduler(scheduler=SchedulerKind.AUTO, config=config)

    plist_path = tmp_path / "Library" / "LaunchAgents" / "it.send.recall.daemon.plist"
    assert plist_path.exists()
    contents = plist_path.read_text(encoding="utf-8")
    assert "<string>it.send.recall.daemon</string>" in contents
    assert "metalrodeo" not in contents
    assert "0xbigboss" not in contents
    assert status.scheduler == SchedulerKind.LAUNCHD
    assert status.installed is True
    assert str(plist_path) in status.artifact_paths
    assert [
        "launchctl",
        "bootstrap",
        f"gui/{daemon_module.os.getuid()}",
        str(plist_path),
    ] in commands
    assert [
        "launchctl",
        "enable",
        f"gui/{daemon_module.os.getuid()}/it.send.recall.daemon",
    ] in commands
    assert _read_runtime_state(config.db_path)[11] == "launchd"


def test_install_scheduler_succeeds_while_the_relaunched_daemon_holds_the_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-074: loading the unit relaunches the daemon, which takes the
    database before install records the scheduler kind and reads status; both
    must tolerate the live holder instead of failing a completed install."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = replace(_app_config(tmp_path), daemon=DaemonConfig(mode=DaemonMode.WATCH))

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    with _database_held_by_live_daemon(config) as pid:
        status = install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    plist_path = daemon_module._artifact_paths(config, SchedulerKind.LAUNCHD)[0]
    assert plist_path.exists()
    assert status.installed is True
    assert status.daemon_pid == pid
    assert status.runtime_unavailable_reason is not None

    # The lost race: the relaunched daemon holds the file before its pid file
    # exists, so the installed-scheduler record and the status read see only a
    # lock conflict. The install still completes.
    monkeypatch.setattr(daemon_module, "_SCHEDULER_DATABASE_WAIT_SECONDS_MAX", 0.0)
    with _database_held_by_live_daemon(config, pid_file=False):
        status = install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    assert status.installed is True
    assert status.daemon_pid is None
    assert status.runtime_unavailable_reason is not None
    assert "another process" in status.runtime_unavailable_reason


def test_install_scheduler_launchd_keepalive_yields_to_refusal_marker(
    tmp_path, monkeypatch
) -> None:
    """REQ-RESIL-024: launchd has no exit-status filter, so KeepAlive is
    conditioned on the refusal marker being absent; while a refused start's
    marker exists the job is not relaunched, and removing it relaunches."""
    import plistlib

    from recall.services import self_repair

    monkeypatch.setenv("HOME", str(tmp_path))
    config = replace(_app_config(tmp_path), daemon=DaemonConfig(mode=DaemonMode.WATCH))

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    plist_path = tmp_path / "Library" / "LaunchAgents" / "it.send.recall.daemon.plist"
    plist = plistlib.loads(plist_path.read_bytes())
    marker = str(self_repair.refusal_marker_path(config.data_dir))
    assert plist["KeepAlive"] == {"PathState": {marker: False}}
    assert plist["RunAtLoad"] is True
    assert daemon_module._detect_installed_mode(config, SchedulerKind.LAUNCHD) == DaemonMode.WATCH


def test_install_scheduler_carries_resolved_daemon_runtime_settings(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = replace(
        _app_config(tmp_path),
        daemon=DaemonConfig(interval=300, embed=False, source=Source.CODEX, mode=DaemonMode.POLL),
    )

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    status = install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    plist_path = tmp_path / "Library" / "LaunchAgents" / "it.send.recall.daemon.plist"
    contents = plist_path.read_text(encoding="utf-8")
    assert "<string>--source</string>" in contents
    assert "<string>codex</string>" in contents
    assert "<string>--no-embed</string>" not in contents
    assert status.command.endswith("daemon --once --source codex")


def test_stop_daemon_durable_invokes_launchctl_bootout(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    commands: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)
    monkeypatch.setattr(daemon_module, "_launchd_unit_is_loaded", lambda: False)

    result = stop_daemon_durable(config, timeout=1.0)

    assert result.stopped is True
    assert result.scheduler == "launchd"
    assert [
        "launchctl",
        "bootout",
        f"gui/{daemon_module.os.getuid()}/{daemon_module.LAUNCHD_LABEL}",
    ] in commands


def test_stop_daemon_durable_names_the_orphaned_daemon(tmp_path, monkeypatch) -> None:
    """REQ-DAEMON-073: a live daemon launchd does not manage is reported as orphaned.

    `launchctl bootout` answers "No such process" because the service is not
    loaded, but the daemon itself is still running and holding the database.
    Returning launchd's error verbatim hides the only actionable fact.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)

    def fake_run(args: list[str], **_kwargs) -> CompletedProcess:
        if args[:2] == ["launchctl", "bootout"]:
            return CompletedProcess(
                args, returncode=3, stderr="Boot-out failed: 3: No such process\n"
            )
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)
    monkeypatch.setattr(daemon_module, "_launchd_unit_is_loaded", lambda: False)
    monkeypatch.setattr(daemon_module, "_running_daemon_pid_from_file", lambda _config: 4242)

    result = stop_daemon_durable(config, timeout=1.0)

    assert result.stopped is False
    assert result.pid == 4242
    assert "orphan" in (result.message or "").lower()
    assert "4242" in (result.message or "")


def test_stop_daemon_durable_invokes_systemctl_stop(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    service_path, _ = daemon_module._artifact_paths(config, SchedulerKind.SYSTEMD)
    service_path.parent.mkdir(parents=True, exist_ok=True)
    service_path.write_text("[Service]\nType=simple\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)
    monkeypatch.setattr(daemon_module, "_systemd_any_lifecycle_unit_active", lambda _config: False)

    result = stop_daemon_durable(config, timeout=1.0)

    assert result.stopped is True
    assert result.scheduler == "systemd"
    assert ["systemctl", "--user", "stop", "recall-daemon.service"] in commands


def test_stop_daemon_durable_polls_until_pid_is_gone(tmp_path, monkeypatch) -> None:
    config = replace(
        _app_config(tmp_path),
        daemon=replace(_app_config(tmp_path).daemon, scheduler=SchedulerKind.CRON),
    )
    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / "recall.pid").write_text("12345", encoding="utf-8")
    alive_values = iter([True, True, True, False])

    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(
        daemon_module,
        "_process_is_alive",
        lambda _pid: next(alive_values, False),
    )
    monkeypatch.setattr(daemon_module.os, "kill", lambda _pid, _signal: None)

    result = stop_daemon_durable(config, timeout=1.0)

    assert result.stopped is True
    assert result.pid == 12345


def test_stop_daemon_durable_returns_false_on_timeout(tmp_path, monkeypatch) -> None:
    config = replace(
        _app_config(tmp_path),
        daemon=replace(_app_config(tmp_path).daemon, scheduler=SchedulerKind.CRON),
    )
    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / "recall.pid").write_text("12345", encoding="utf-8")

    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(daemon_module, "_process_is_alive", lambda _pid: True)
    monkeypatch.setattr(daemon_module.os, "kill", lambda _pid, _signal: None)

    result = stop_daemon_durable(config, timeout=0.01)

    assert result.stopped is False
    assert "still alive" in str(result.message)


def test_wait_for_durable_stop_treats_stale_pidfile_as_stopped(tmp_path, monkeypatch) -> None:
    config = _app_config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    pid_path = config.data_dir / "recall.pid"
    pid_path.write_text("999999999", encoding="utf-8")

    monkeypatch.setattr(daemon_module, "_process_is_alive", lambda _pid: False)

    result = daemon_module._wait_for_durable_stop(
        config,
        scheduler="systemd",
        pid=None,
        timeout=1.0,
        started_at=0.0,
        scheduler_stopped=lambda: True,
    )

    assert result.stopped is True
    assert result.message == "removed stale daemon pidfile"
    assert not pid_path.exists()


def test_wait_for_durable_stop_unlinks_invalid_pidfile_on_first_inactive_tick(
    tmp_path, monkeypatch
) -> None:
    config = _app_config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    pid_path = config.data_dir / "recall.pid"
    pid_path.write_text("not-a-pid", encoding="utf-8")
    scheduler_checks = 0

    def scheduler_stopped() -> bool:
        nonlocal scheduler_checks
        scheduler_checks += 1
        return True

    monkeypatch.setattr(
        daemon_module,
        "_process_is_alive",
        lambda _pid: pytest.fail("invalid pidfile content must not be probed"),
    )

    result = daemon_module._wait_for_durable_stop(
        config,
        scheduler="launchd",
        pid=None,
        timeout=1.0,
        started_at=0.0,
        scheduler_stopped=scheduler_stopped,
    )

    assert result.stopped is True
    assert scheduler_checks == 1
    assert not pid_path.exists()


def test_wait_for_durable_stop_times_out_with_live_pid_and_active_scheduler(
    tmp_path, monkeypatch
) -> None:
    config = _app_config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    pid_path = config.data_dir / "recall.pid"
    pid_path.write_text("12345", encoding="utf-8")

    monkeypatch.setattr(daemon_module, "_process_is_alive", lambda _pid: True)
    monkeypatch.setattr(daemon_module.time, "sleep", lambda _seconds: None)

    result = daemon_module._wait_for_durable_stop(
        config,
        scheduler="systemd",
        pid=12345,
        timeout=0.01,
        started_at=0.0,
        scheduler_stopped=lambda: False,
    )

    assert result.stopped is False
    assert pid_path.exists()
    assert "scheduler still loaded" in str(result.message)


def test_stop_daemon_soft_uses_pidfile_without_scheduler(tmp_path, monkeypatch) -> None:
    config = _app_config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / "recall.pid").write_text("12345", encoding="utf-8")
    kill_calls: list[tuple[int, int]] = []
    alive_values = iter([True, False])

    monkeypatch.setattr(daemon_module, "_process_is_alive", lambda _pid: next(alive_values, False))

    def fake_kill(pid: int, sig: int) -> None:
        kill_calls.append((pid, sig))

    monkeypatch.setattr(daemon_module.os, "kill", fake_kill)

    result = stop_daemon_soft(config)

    assert result.stopped is True
    assert result.scheduler == "sentinel"
    assert kill_calls == [(12345, daemon_module.signal.SIGTERM)]


def test_start_daemon_durable_invokes_launchctl_bootstrap(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = replace(
        _app_config(tmp_path),
        daemon=replace(_app_config(tmp_path).daemon, mode=DaemonMode.WATCH),
    )
    plist_path = daemon_module._artifact_paths(config, SchedulerKind.LAUNCHD)[0]
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text("<plist />", encoding="utf-8")
    commands: list[list[str]] = []
    pid_values = iter([None, None, 24680])
    loaded_values = iter([False, True])

    def fake_run(args: list[str], **_kwargs) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)
    monkeypatch.setattr(daemon_module, "_launchd_unit_is_loaded", lambda: next(loaded_values, True))
    monkeypatch.setattr(
        daemon_module,
        "_running_daemon_pid_from_file",
        lambda _config: next(pid_values, 24680),
    )

    result = start_daemon_durable(config, timeout=1.0)

    assert result.started is True
    assert result.pid == 24680
    assert [
        "launchctl",
        "bootstrap",
        f"gui/{daemon_module.os.getuid()}",
        str(plist_path),
    ] in commands


def test_start_daemon_durable_treats_launchd_bootstrap_already_loaded_as_started(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    plist_path = daemon_module._artifact_paths(config, SchedulerKind.LAUNCHD)[0]
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text("<plist />", encoding="utf-8")
    loaded_values = iter([False, True, True])
    commands: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess(args, returncode=1, stderr="service target already exists")

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)
    monkeypatch.setattr(daemon_module, "_launchd_unit_is_loaded", lambda: next(loaded_values, True))
    monkeypatch.setattr(daemon_module, "_running_daemon_pid_from_file", lambda _config: None)

    result = start_daemon_durable(config, timeout=1.0)

    assert result.started is True
    assert result.pid is None
    assert result.message == "already loaded"
    assert [
        "launchctl",
        "bootstrap",
        f"gui/{daemon_module.os.getuid()}",
        str(plist_path),
    ] in commands


def test_start_daemon_durable_skips_bootstrap_when_launchd_already_loaded(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_module, "_launchd_unit_is_loaded", lambda: True)
    monkeypatch.setattr(daemon_module, "_running_daemon_pid_from_file", lambda _config: None)
    monkeypatch.setattr(
        daemon_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("already-loaded scheduler must not bootstrap"),
    )

    result = start_daemon_durable(config, timeout=1.0)

    assert result.started is True
    assert result.pid is None
    assert result.message == "already loaded"


def test_start_daemon_durable_reports_already_running(tmp_path, monkeypatch) -> None:
    config = _app_config(tmp_path)
    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_module, "_running_daemon_pid_from_file", lambda _config: 24680)
    monkeypatch.setattr(
        daemon_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("already running must not start scheduler"),
    )

    result = start_daemon_durable(config, timeout=1.0)

    assert result.started is True
    assert result.pid == 24680
    assert result.message == "daemon already running"


def test_restart_daemon_durable_short_circuits_failed_stop(tmp_path, monkeypatch) -> None:
    config = _app_config(tmp_path)
    calls: list[str] = []

    def fake_stop(_config, *, timeout: float):
        calls.append(f"stop:{timeout}")
        return daemon_module.DaemonStopResult(
            scheduler="launchd",
            stopped=False,
            pid=12345,
            duration_seconds=0.1,
            message="still running",
        )

    def fake_start(_config, *, timeout: float):
        calls.append(f"start:{timeout}")
        return daemon_module.DaemonStartResult(
            scheduler="launchd",
            started=True,
            pid=12346,
            duration_seconds=0.1,
            message=None,
        )

    monkeypatch.setattr(daemon_module, "stop_daemon_durable", fake_stop)
    monkeypatch.setattr(daemon_module, "start_daemon_durable", fake_start)

    result = restart_daemon_durable(config, timeout=3.0)

    assert result.stopped is False
    assert result.started is False
    assert calls == ["stop:3.0"]


def test_install_scheduler_auto_on_linux_without_systemd_requires_explicit_cron(
    tmp_path, monkeypatch
) -> None:
    config = _app_config(tmp_path)
    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(daemon_module, "_systemd_user_available", lambda: False)

    with pytest.raises(ValueError, match="--scheduler cron"):
        install_scheduler(scheduler=SchedulerKind.AUTO, config=config)


def test_install_scheduler_cron_rejects_subminute_intervals(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path, interval=30)
    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )

    with pytest.raises(ValueError, match="sub-minute"):
        install_scheduler(scheduler=SchedulerKind.CRON, config=config)


def test_install_scheduler_systemd_watch_mode_generates_simple_service(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = replace(
        _app_config(tmp_path),
        daemon=DaemonConfig(interval=300, mode=DaemonMode.WATCH),
    )
    commands: list[list[str]] = []

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(daemon_module, "_systemd_user_available", lambda: True)
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    status = install_scheduler(scheduler=SchedulerKind.SYSTEMD, config=config)

    base = tmp_path / ".config" / "systemd" / "user"
    service_path = base / "recall-daemon.service"
    timer_path = base / "recall-daemon.timer"

    assert service_path.exists()
    assert not timer_path.exists()

    contents = service_path.read_text(encoding="utf-8")
    assert "Type=simple" in contents
    assert "Restart=always" in contents
    assert "RestartSec=5" in contents
    # REQ-RESIL-024: an exit-3 refusal must not be relaunched every 5 seconds.
    assert "RestartPreventExitStatus=3" in contents
    assert "WantedBy=default.target" in contents
    assert "Type=oneshot" not in contents

    assert status.scheduler == SchedulerKind.SYSTEMD
    assert status.installed is True
    assert str(service_path) in status.artifact_paths

    assert ["systemctl", "--user", "daemon-reload"] in commands
    assert ["systemctl", "--user", "enable", "recall-daemon.service"] in commands
    assert ["systemctl", "--user", "restart", "recall-daemon.service"] in commands
    assert ["systemctl", "--user", "enable", "--now", "recall-daemon.timer"] not in commands


def test_install_scheduler_systemd_poll_mode_generates_oneshot_with_timer(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = replace(
        _app_config(tmp_path),
        daemon=DaemonConfig(interval=300, mode=DaemonMode.POLL),
    )
    commands: list[list[str]] = []

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(daemon_module, "_systemd_user_available", lambda: True)
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    status = install_scheduler(scheduler=SchedulerKind.SYSTEMD, config=config)

    base = tmp_path / ".config" / "systemd" / "user"
    service_path = base / "recall-daemon.service"
    timer_path = base / "recall-daemon.timer"

    assert service_path.exists()
    assert timer_path.exists()

    service_contents = service_path.read_text(encoding="utf-8")
    assert "Type=oneshot" in service_contents
    assert "Type=simple" not in service_contents

    timer_contents = timer_path.read_text(encoding="utf-8")
    assert "OnUnitActiveSec=300" in timer_contents
    assert "WantedBy=timers.target" in timer_contents

    assert status.scheduler == SchedulerKind.SYSTEMD
    assert status.installed is True
    assert str(service_path) in status.artifact_paths
    assert str(timer_path) in status.artifact_paths

    assert ["systemctl", "--user", "daemon-reload"] in commands
    assert ["systemctl", "--user", "enable", "--now", "recall-daemon.timer"] in commands


def test_install_scheduler_systemd_watch_to_poll_disables_service(tmp_path, monkeypatch) -> None:
    """Watch-to-poll transition must disable the watch service before enabling timer."""
    monkeypatch.setenv("HOME", str(tmp_path))
    watch_config = replace(
        _app_config(tmp_path),
        daemon=DaemonConfig(interval=300, mode=DaemonMode.WATCH),
    )
    poll_config = replace(
        _app_config(tmp_path),
        daemon=DaemonConfig(interval=300, mode=DaemonMode.POLL),
    )
    commands: list[list[str]] = []

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(daemon_module, "_systemd_user_available", lambda: True)
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    # Install watch mode first
    install_scheduler(scheduler=SchedulerKind.SYSTEMD, config=watch_config)

    base = tmp_path / ".config" / "systemd" / "user"
    service_path = base / "recall-daemon.service"
    assert "Type=simple" in service_path.read_text(encoding="utf-8")

    # Reset command log, then switch to poll mode
    commands.clear()
    install_scheduler(scheduler=SchedulerKind.SYSTEMD, config=poll_config)

    # Must have disabled the watch service before enabling the timer
    assert ["systemctl", "--user", "disable", "--now", "recall-daemon.service"] in commands
    assert ["systemctl", "--user", "enable", "--now", "recall-daemon.timer"] in commands

    # Service should now be oneshot, timer should exist
    assert "Type=oneshot" in service_path.read_text(encoding="utf-8")
    timer_path = base / "recall-daemon.timer"
    assert timer_path.exists()


def test_install_scheduler_systemd_poll_to_watch_removes_timer(tmp_path, monkeypatch) -> None:
    """Switching from poll to watch mode must disable the timer and remove its file."""
    monkeypatch.setenv("HOME", str(tmp_path))
    poll_config = replace(
        _app_config(tmp_path),
        daemon=DaemonConfig(interval=300, mode=DaemonMode.POLL),
    )
    watch_config = replace(
        _app_config(tmp_path),
        daemon=DaemonConfig(interval=300, mode=DaemonMode.WATCH),
    )
    commands: list[list[str]] = []

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(daemon_module, "_systemd_user_available", lambda: True)
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    # Install poll mode first
    install_scheduler(scheduler=SchedulerKind.SYSTEMD, config=poll_config)

    base = tmp_path / ".config" / "systemd" / "user"
    timer_path = base / "recall-daemon.timer"
    assert timer_path.exists()

    # Reset command log, then switch to watch mode
    commands.clear()
    install_scheduler(scheduler=SchedulerKind.SYSTEMD, config=watch_config)

    # Must have disabled the timer
    assert ["systemctl", "--user", "disable", "--now", "recall-daemon.timer"] in commands
    assert ["systemctl", "--user", "enable", "recall-daemon.service"] in commands
    assert ["systemctl", "--user", "restart", "recall-daemon.service"] in commands

    # Timer file should be gone, service should be Type=simple
    assert not timer_path.exists()
    service_path = base / "recall-daemon.service"
    assert "Type=simple" in service_path.read_text(encoding="utf-8")


def test_uninstall_scheduler_systemd_watch_mode(tmp_path, monkeypatch) -> None:
    """Uninstalling a systemd watch install removes service and cleans up stale timer."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = replace(
        _app_config(tmp_path),
        daemon=DaemonConfig(interval=300, mode=DaemonMode.WATCH),
    )
    commands: list[list[str]] = []

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(daemon_module, "_systemd_user_available", lambda: True)
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    install_scheduler(scheduler=SchedulerKind.SYSTEMD, config=config)

    base = tmp_path / ".config" / "systemd" / "user"
    service_path = base / "recall-daemon.service"
    assert service_path.exists()

    commands.clear()
    uninstall_scheduler(config=config)

    assert not service_path.exists()
    assert ["systemctl", "--user", "disable", "--now", "recall-daemon.service"] in commands
    assert _read_runtime_state(config.db_path)[11] is None


def test_daemon_status_reports_systemd_watch_install_regardless_of_current_mode(
    tmp_path, monkeypatch
) -> None:
    """Status must detect a systemd watch install even if config currently says poll."""
    monkeypatch.setenv("HOME", str(tmp_path))
    watch_config = replace(
        _app_config(tmp_path),
        daemon=DaemonConfig(interval=300, mode=DaemonMode.WATCH),
    )
    poll_config = replace(
        _app_config(tmp_path),
        daemon=DaemonConfig(interval=300, mode=DaemonMode.POLL),
    )

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(daemon_module, "_systemd_user_available", lambda: True)
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    # Install with watch mode
    install_scheduler(scheduler=SchedulerKind.SYSTEMD, config=watch_config)

    # Check status using a poll-mode config — must still detect the install
    # AND report the actual installed mode AND installed command, not config-derived
    status = daemon_status(config=poll_config)
    assert status.scheduler == SchedulerKind.SYSTEMD
    assert status.installed is True
    assert status.resolved_mode == DaemonMode.WATCH
    assert "--mode" in status.command and "watch" in status.command
    assert "--once" not in status.command

    base = tmp_path / ".config" / "systemd" / "user"
    service_path = base / "recall-daemon.service"
    timer_path = base / "recall-daemon.timer"
    # artifact_paths should only include existing files (service, not timer)
    assert str(service_path) in status.artifact_paths
    assert str(timer_path) not in status.artifact_paths


def test_reinstall_systemd_watch_mode_restarts_service(tmp_path, monkeypatch) -> None:
    """Re-installing watch mode must restart the service to pick up new config."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config_v1 = replace(
        _app_config(tmp_path),
        daemon=DaemonConfig(interval=300, mode=DaemonMode.WATCH, source=Source.CODEX),
    )
    config_v2 = replace(
        _app_config(tmp_path),
        daemon=DaemonConfig(interval=300, mode=DaemonMode.WATCH, source=Source.CLAUDE_CODE),
    )
    commands: list[list[str]] = []

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(daemon_module, "_systemd_user_available", lambda: True)
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    # First install
    install_scheduler(scheduler=SchedulerKind.SYSTEMD, config=config_v1)

    # Re-install with different source
    commands.clear()
    install_scheduler(scheduler=SchedulerKind.SYSTEMD, config=config_v2)

    # Must restart, not just enable
    assert ["systemctl", "--user", "restart", "recall-daemon.service"] in commands

    # Service file should have the updated source in ExecStart
    base = tmp_path / ".config" / "systemd" / "user"
    contents = (base / "recall-daemon.service").read_text(encoding="utf-8")
    assert "--source" in contents and Source.CLAUDE_CODE.value in contents


def test_daemon_status_watched_dirs_reflect_installed_source(tmp_path, monkeypatch) -> None:
    """watched_dirs must reflect the installed --source, not the current config source."""
    monkeypatch.setenv("HOME", str(tmp_path))

    # Create watched directories so parsers can resolve them
    (tmp_path / ".codex" / "sessions").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".claude" / "projects").mkdir(parents=True, exist_ok=True)

    codex_watch_config = replace(
        _app_config(tmp_path),
        daemon=DaemonConfig(interval=300, mode=DaemonMode.WATCH, source=Source.CODEX),
    )

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(daemon_module, "_systemd_user_available", lambda: True)
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    # Install watch mode with source=codex
    install_scheduler(scheduler=SchedulerKind.SYSTEMD, config=codex_watch_config)

    # Query status with config changed to source=claude-code
    claude_watch_config = replace(
        _app_config(tmp_path),
        daemon=DaemonConfig(interval=300, mode=DaemonMode.WATCH, source=Source.CLAUDE_CODE),
    )
    status = daemon_status(config=claude_watch_config)

    # watched_dirs should reflect the installed source (codex), not the config source
    assert status.installed is True
    assert "--source" in status.command and "codex" in status.command
    assert any(".codex" in d for d in status.watched_dirs)
    assert not any(".claude" in d for d in status.watched_dirs)


def test_uninstall_scheduler_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    commands: list[list[str]] = []

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        commands.append(args)
        if args[:2] == ["launchctl", "bootout"]:
            raise subprocess.CalledProcessError(returncode=3, cmd=args)
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)
    uninstall_scheduler(config=config)
    uninstall_scheduler(config=config)

    plist_path = tmp_path / "Library" / "LaunchAgents" / "it.send.recall.daemon.plist"
    assert not plist_path.exists()
    assert _read_runtime_state(config.db_path)[11] is None
    assert any(command[:2] == ["launchctl", "bootout"] for command in commands)


def test_uninstall_scheduler_clears_runtime_kind_after_a_transient_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-DAEMON-074: launchd teardown is asynchronous, so uninstall waits for
    a transient database lock to clear before removing the persisted scheduler
    kind; otherwise status keeps describing an uninstalled daemon."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    plist_path = daemon_module._artifact_paths(config, SchedulerKind.LAUNCHD)[0]
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text("<plist></plist>\n", encoding="utf-8")
    conn = connect(config)
    try:
        set_installed_scheduler(conn, SchedulerKind.LAUNCHD)
    finally:
        conn.close()

    attempts = 0

    def connect_after_daemon_releases(active_config: AppConfig) -> duckdb.DuckDBPyConnection:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise duckdb.IOException(
                "IO Error: Could not set lock on file: Conflicting lock is held in daemon"
            )
        return connect(active_config)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        daemon_module.subprocess,
        "run",
        lambda args, *, check, capture_output, text, timeout=None: CompletedProcess(args),
    )
    monkeypatch.setattr(daemon_module, "connect", connect_after_daemon_releases)
    monkeypatch.setattr(daemon_module.time, "sleep", lambda _seconds: None)

    status = uninstall_scheduler(config=config)

    assert attempts == 2
    assert status.installed is False
    assert status.runtime_status.installed_scheduler is None
    assert _read_runtime_state(config.db_path)[11] is None


def test_uninstall_scheduler_stops_retrying_at_the_database_release_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-DAEMON-074: a holder that outlives teardown cannot make uninstall
    retry forever; scheduler artifacts are authoritative after the bound."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    plist_path = daemon_module._artifact_paths(config, SchedulerKind.LAUNCHD)[0]
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text("<plist></plist>\n", encoding="utf-8")
    conn = connect(config)
    try:
        set_installed_scheduler(conn, SchedulerKind.LAUNCHD)
    finally:
        conn.close()

    attempts = 0
    now = 0.0

    def database_remains_held(_config: AppConfig) -> duckdb.DuckDBPyConnection:
        nonlocal attempts
        attempts += 1
        raise duckdb.IOException(
            "IO Error: Could not set lock on file: Conflicting lock is held in daemon"
        )

    def monotonic() -> float:
        return now

    def advance(seconds: float) -> None:
        nonlocal now
        assert seconds > 0
        now += seconds
        assert now <= 30.0

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        daemon_module.subprocess,
        "run",
        lambda args, *, check, capture_output, text, timeout=None: CompletedProcess(args),
    )
    monkeypatch.setattr(daemon_module, "connect", database_remains_held)
    monkeypatch.setattr(daemon_module.time, "monotonic", monotonic)
    monkeypatch.setattr(daemon_module.time, "sleep", advance)

    status = uninstall_scheduler(config=config)

    assert attempts > 1
    assert now == 30.0
    assert status.installed is False
    assert status.runtime_status.installed_scheduler == SchedulerKind.LAUNCHD
    assert _read_runtime_state(config.db_path)[11] == "launchd"


def test_uninstall_scheduler_propagates_a_non_lock_duckdb_io_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-DAEMON-074: only the expected teardown lock is retried; an unrelated
    DuckDB I/O failure remains an operating error for the caller to diagnose."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    plist_path = daemon_module._artifact_paths(config, SchedulerKind.LAUNCHD)[0]
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text("<plist></plist>\n", encoding="utf-8")
    conn = connect(config)
    try:
        set_installed_scheduler(conn, SchedulerKind.LAUNCHD)
    finally:
        conn.close()

    def fail_with_io_error(_config: AppConfig) -> duckdb.DuckDBPyConnection:
        raise duckdb.IOException("IO Error: disk read failed")

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        daemon_module.subprocess,
        "run",
        lambda args, *, check, capture_output, text, timeout=None: CompletedProcess(args),
    )
    monkeypatch.setattr(daemon_module, "connect", fail_with_io_error)

    with pytest.raises(duckdb.IOException, match="disk read failed"):
        uninstall_scheduler(config=config)


def test_daemon_status_reports_installed_scheduler_and_command(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    command_path = str(tmp_path / "bin" / "recall")

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_module.shutil, "which", lambda _name: command_path)
    monkeypatch.setattr(
        daemon_module.subprocess,
        "run",
        lambda args, *, check, capture_output, text, timeout=None: CompletedProcess(args),
    )

    install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)
    status = daemon_status(config=config)

    assert status.scheduler == SchedulerKind.LAUNCHD
    assert status.installed is True
    assert status.command == f"{command_path} daemon --once"
    assert status.config_path == str(config.config_path)


def test_daemon_status_running_watch_overrides_installed_poll(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(
        daemon_module.subprocess,
        "run",
        lambda args, *, check, capture_output, text, timeout=None: CompletedProcess(args),
    )

    install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    snapshot = WatcherLiveSnapshot(
        live_session_count=3,
        live_session_paths=["/tmp/live-1.jsonl", "/tmp/live-2.jsonl"],
        discovery_interval_seconds=15.0,
        discovery_last_run_at=datetime(2026, 4, 14, 11, 0, 0),
        discovery_last_promoted=2,
        discovery_last_demoted=0,
        watcher_subscription_count=4,
    )
    monkeypatch.setattr(daemon_module, "_detect_runtime_mode", lambda _config: DaemonMode.WATCH)
    monkeypatch.setattr("recall.services.watcher.get_live_snapshot", lambda: snapshot)

    status = daemon_status(config=config)

    assert status.resolved_mode == DaemonMode.WATCH
    assert status.installed_mode == DaemonMode.POLL
    assert status.mode_mismatch_reason is not None
    assert "running=watch" in status.mode_mismatch_reason
    assert "installed=poll" in status.mode_mismatch_reason
    assert status.live_session_count == 3
    assert status.discovery_last_run_at == snapshot.discovery_last_run_at


def test_daemon_status_no_running_daemon_falls_back_to_installed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = replace(
        _app_config(tmp_path),
        daemon=replace(_app_config(tmp_path).daemon, mode=DaemonMode.POLL),
    )

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        daemon_module.shutil, "which", lambda _name: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(
        daemon_module.subprocess,
        "run",
        lambda args, *, check, capture_output, text, timeout=None: CompletedProcess(args),
    )

    install_scheduler(
        scheduler=SchedulerKind.LAUNCHD,
        config=replace(config, daemon=replace(config.daemon, mode=DaemonMode.WATCH)),
    )
    monkeypatch.setattr(daemon_module, "_detect_runtime_mode", lambda _config: None)

    status = daemon_status(config=config)

    assert status.resolved_mode == DaemonMode.WATCH
    assert status.installed_mode == DaemonMode.WATCH
    assert status.mode_mismatch_reason is None


def test_detect_runtime_mode_reports_watch_during_startup_window(tmp_path, monkeypatch) -> None:
    """REQ-DAEMON-060: immediately after `start_live_watch_runtime` on an idle
    host, live counts are zero and no discovery tick has run. Runtime-mode
    detection must still report WATCH so status does not temporarily misreport
    `poll` during the startup window that runs until the first discovery tick.
    """
    from recall.services.watcher import (
        _mark_runtime_started,
        reset_live_snapshot,
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)

    monkeypatch.setattr(daemon_module, "_running_daemon_pid", lambda _config: 424242)

    reset_live_snapshot()
    try:
        # Startup window: no discovery tick, no live sessions, no subscriptions.
        assert daemon_module._detect_runtime_mode(config) is None
        _mark_runtime_started(datetime(2026, 4, 15, 10, 0, 0))
        assert daemon_module._detect_runtime_mode(config) == DaemonMode.WATCH
    finally:
        reset_live_snapshot()
    # After shutdown `reset_live_snapshot` clears runtime_started_at, so
    # detection falls back to None again.
    assert daemon_module._detect_runtime_mode(config) is None


def test_daemon_status_no_daemon_no_install_uses_config(tmp_path, monkeypatch) -> None:
    _skip_without_crontab_capability()

    monkeypatch.setenv("HOME", str(tmp_path))
    config = replace(
        _app_config(tmp_path),
        daemon=replace(_app_config(tmp_path).daemon, mode=DaemonMode.WATCH),
    )
    monkeypatch.setattr(daemon_module, "_detect_runtime_mode", lambda _config: None)

    status = daemon_status(config=config)

    assert status.installed is False
    assert status.resolved_mode == DaemonMode.WATCH
    assert status.installed_mode is None
    assert status.mode_mismatch_reason is None


def test_daemon_status_watch_snapshot_defaults_to_config_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-050: empty watcher state still reports the configured discovery interval."""
    _skip_without_crontab_capability()

    monkeypatch.setenv("HOME", str(tmp_path))
    config = replace(
        _app_config(tmp_path),
        daemon=replace(
            _app_config(tmp_path).daemon,
            mode=DaemonMode.WATCH,
            live_discovery_interval=45,
        ),
    )
    empty_snapshot = WatcherLiveSnapshot(
        live_session_count=0,
        live_session_paths=[],
        discovery_interval_seconds=0.0,
        discovery_last_run_at=None,
        discovery_last_promoted=0,
        discovery_last_demoted=0,
        watcher_subscription_count=0,
    )
    monkeypatch.setattr("recall.services.watcher.get_live_snapshot", lambda: empty_snapshot)

    status = daemon_status(config=config)

    assert status.live_session_count == 0
    assert status.live_session_paths == ()
    assert status.discovery_interval_seconds == 45.0
    assert status.discovery_last_run_at is None
    assert status.discovery_last_promoted == 0
    assert status.discovery_last_demoted == 0
    assert status.watcher_subscription_count == 0


def test_daemon_status_watch_snapshot_populates_live_session_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-050: watch-mode status surfaces the live watcher snapshot."""
    _skip_without_crontab_capability()

    monkeypatch.setenv("HOME", str(tmp_path))
    config = replace(
        _app_config(tmp_path),
        daemon=replace(_app_config(tmp_path).daemon, mode=DaemonMode.WATCH),
    )
    last_run_at = datetime(2026, 4, 14, 9, 30, 0)
    snapshot = WatcherLiveSnapshot(
        live_session_count=12,
        live_session_paths=[f"/tmp/session-{index}.jsonl" for index in range(12)],
        discovery_interval_seconds=12.5,
        discovery_last_run_at=last_run_at,
        discovery_last_promoted=4,
        discovery_last_demoted=1,
        watcher_subscription_count=7,
        catchup_in_progress=True,
        catchup_total=9,
        catchup_done=3,
    )
    monkeypatch.setattr("recall.services.watcher.get_live_snapshot", lambda: snapshot)

    status = daemon_status(config=config)

    assert status.live_session_count == 12
    assert status.live_session_paths == tuple(snapshot.live_session_paths[:10])
    assert len(status.live_session_paths) == 10
    assert status.discovery_interval_seconds == 12.5
    assert status.discovery_last_run_at == last_run_at
    assert status.discovery_last_promoted == 4
    assert status.discovery_last_demoted == 1
    assert status.watcher_subscription_count == 7
    assert status.catchup_in_progress is True
    assert status.catchup_total == 9
    assert status.catchup_done == 3


def test_daemon_status_poll_mode_ignores_live_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-050: poll-mode status leaves live watcher fields at their safe defaults."""
    _skip_without_crontab_capability()

    monkeypatch.setenv("HOME", str(tmp_path))
    config = replace(
        _app_config(tmp_path),
        daemon=replace(_app_config(tmp_path).daemon, mode=DaemonMode.POLL),
    )
    snapshot = WatcherLiveSnapshot(
        live_session_count=5,
        live_session_paths=["/tmp/live.jsonl"],
        discovery_interval_seconds=20.0,
        discovery_last_run_at=datetime(2026, 4, 14, 10, 0, 0),
        discovery_last_promoted=2,
        discovery_last_demoted=1,
        watcher_subscription_count=3,
    )
    monkeypatch.setattr("recall.services.watcher.get_live_snapshot", lambda: snapshot)

    status = daemon_status(config=config)

    assert status.discovery_interval_seconds == 0.0
    assert status.live_session_count == 0
    assert status.live_session_paths == ()
    assert status.discovery_last_run_at is None
    assert status.discovery_last_promoted == 0
    assert status.discovery_last_demoted == 0
    assert status.watcher_subscription_count == 0
    assert status.catchup_in_progress is False
    assert status.catchup_total == 0
    assert status.catchup_done == 0


def test_print_daemon_status_text_renders_live_session_block_for_watch_or_live_data(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-050: text status prints live watcher metrics when live data exists."""
    watch_status = {
        "configured_scheduler": "launchd",
        "scheduler": "launchd",
        "installed": True,
        "command": "/tmp/recall daemon --once",
        "config_path": "/tmp/config.toml",
        "artifact_paths": (),
        "runtime_status": {},
        "mode": "watch",
        "resolved_mode": "watch",
        "watched_dirs": (),
        "debounce": 5,
        "fts_debounce": 10,
        "embed_phase_enabled": False,
        "embed_model_loaded": False,
        "embed_pending": 0,
        "watch_total_indexed": 1,
        "watch_total_failed": 0,
        "live_session_count": 2,
        "live_session_paths": ("/tmp/live-a.jsonl", "/tmp/live-b.jsonl"),
        "discovery_interval_seconds": 30.0,
        "discovery_last_run_at": None,
        "discovery_last_promoted": 3,
        "discovery_last_demoted": 1,
        "watcher_subscription_count": 4,
        "catchup_in_progress": True,
        "catchup_total": 9,
        "catchup_done": 3,
    }
    poll_status = dict(
        watch_status,
        mode="poll",
        resolved_mode="poll",
        installed_mode="poll",
        mode_mismatch_reason="running=watch, installed=poll",
    )
    empty_poll_status = dict(
        poll_status,
        live_session_count=0,
        live_session_paths=(),
        discovery_last_run_at=None,
    )

    monkeypatch.setattr(
        "recall.cli.daemon.AppConfig.load",
        lambda: type("Config", (), {"daemon": type("Daemon", (), {"interval": 300})()})(),
    )

    _print_daemon_status_text(watch_status)
    watch_output = capsys.readouterr().out
    assert "Live sessions: 2 (subs=4)" in watch_output
    assert "Discovery: interval=30.0s last_run=never promoted=3 demoted=1" in watch_output
    assert "Catch-up: running done=3 total=9" in watch_output
    assert "Live paths:" in watch_output
    assert "  /tmp/live-a.jsonl" in watch_output
    assert "  /tmp/live-b.jsonl" in watch_output

    _print_daemon_status_text(poll_status)
    poll_output = capsys.readouterr().out
    assert "Mode: poll (resolved: poll) [installed: poll]" in poll_output
    assert "Live sessions: 2 (subs=4)" in poll_output
    assert "Discovery:" in poll_output

    _print_daemon_status_text(empty_poll_status)
    empty_poll_output = capsys.readouterr().out
    assert "Mode: poll (resolved: poll) [installed: poll]" in empty_poll_output
    assert "Live sessions:" not in empty_poll_output
    assert "Discovery:" not in empty_poll_output
    assert "Live paths:" not in empty_poll_output


def test_print_daemon_status_text_renders_active_fts_rebuild_backoff(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    status = {
        "configured_scheduler": "launchd",
        "scheduler": "launchd",
        "installed": True,
        "command": "/tmp/recall daemon --mode watch",
        "config_path": "/tmp/config.toml",
        "artifact_paths": (),
        "runtime_status": {},
        "mode": "watch",
        "resolved_mode": "watch",
        "watched_dirs": (),
        "debounce": 5,
        "fts_debounce": 10,
        "embed_phase_enabled": False,
        "embed_model_loaded": False,
        "embed_pending": 0,
        "watch_total_indexed": 3,
        "watch_total_failed": 1,
        "last_fts_rebuild_failure_at": "2026-05-27T01:00:00Z",
        "last_fts_rebuild_failure_reason": "FTS rebuild exhausted DuckDB memory_limit",
        "fts_rebuild_consecutive_failures": 2,
        "fts_rebuild_next_retry_at": "2026-05-27T01:02:00Z",
    }
    monkeypatch.setattr(
        "recall.cli.daemon.AppConfig.load",
        lambda: type("Config", (), {"daemon": type("Daemon", (), {"interval": 300})()})(),
    )

    _print_daemon_status_text(status)

    output = capsys.readouterr().out
    assert "FTS rebuild: 2 consecutive OOM failure(s)" in output
    assert "last at 2026-05-27T01:00:00Z" in output
    assert "next retry at 2026-05-27T01:02:00Z" in output
    assert "  reason: FTS rebuild exhausted DuckDB memory_limit" in output


def test_print_daemon_status_text_omits_inactive_fts_rebuild_backoff(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    status = {
        "configured_scheduler": "launchd",
        "scheduler": "launchd",
        "installed": True,
        "command": "/tmp/recall daemon --mode watch",
        "config_path": "/tmp/config.toml",
        "artifact_paths": (),
        "runtime_status": {},
        "mode": "watch",
        "resolved_mode": "watch",
        "watched_dirs": (),
        "debounce": 5,
        "fts_debounce": 10,
        "embed_phase_enabled": False,
        "embed_model_loaded": False,
        "embed_pending": 0,
        "watch_total_indexed": 3,
        "watch_total_failed": 1,
        "last_fts_rebuild_failure_at": None,
        "last_fts_rebuild_failure_reason": None,
        "fts_rebuild_consecutive_failures": 0,
        "fts_rebuild_next_retry_at": None,
    }
    monkeypatch.setattr(
        "recall.cli.daemon.AppConfig.load",
        lambda: type("Config", (), {"daemon": type("Daemon", (), {"interval": 300})()})(),
    )

    _print_daemon_status_text(status)

    output = capsys.readouterr().out
    assert "FTS rebuild:" not in output
    assert "consecutive OOM failure" not in output


_RESTART_HINT = (
    "Binary upgraded since daemon started — run `recall daemon restart` to load the new version."
)


@pytest.mark.parametrize(
    ("overrides", "version_line", "hint_shown"),
    [
        pytest.param(
            {"daemon_version": "0.10.3", "binary_version": "0.10.4", "version_drift": True},
            "Version: binary=0.10.4 daemon=0.10.3 drift=yes",
            True,
            id="drift-hint",
        ),
        pytest.param(
            {"daemon_version": "0.10.4", "binary_version": "0.10.4", "version_drift": False},
            "Version: binary=0.10.4 daemon=0.10.4 drift=no",
            False,
            id="equal-versions",
        ),
        pytest.param(
            {
                "scheduler": None,
                "installed": False,
                "daemon_version": None,
                "binary_version": "0.10.4",
                "version_drift": False,
            },
            "Version: binary=0.10.4 daemon=unknown drift=n/a",
            False,
            id="no-daemon-version",
        ),
    ],
)
def test_print_daemon_status_text_renders_versions(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    version_line: str,
    hint_shown: bool,
) -> None:
    status = {
        "configured_scheduler": "launchd",
        "scheduler": "launchd",
        "installed": True,
        "command": "/tmp/recall daemon --once",
        "config_path": "/tmp/config.toml",
        "artifact_paths": (),
        "runtime_status": {},
        "mode": "poll",
        "resolved_mode": "poll",
        "watched_dirs": (),
        "debounce": 5,
        "fts_debounce": 10,
        "embed_phase_enabled": False,
        "embed_model_loaded": False,
        "embed_pending": 0,
        **overrides,
    }
    monkeypatch.setattr(
        "recall.cli.daemon.AppConfig.load",
        lambda: type("Config", (), {"daemon": type("Daemon", (), {"interval": 300})()})(),
    )

    _print_daemon_status_text(status)

    output = capsys.readouterr().out
    assert version_line in output
    if hint_shown:
        assert _RESTART_HINT in output
    else:
        assert "recall daemon restart" not in output


def test_historical_embed_run_kind_does_not_crash(tmp_path, monkeypatch) -> None:
    """Databases with last_run_kind='embed' from before the unified pipeline must not crash.

    Legacy run kinds must deserialize to None and must not surface stale
    index summary data from a prior run.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)

    from recall.db import connect
    from recall.services.runtime_state import load_runtime_status

    conn = connect(config)
    try:
        conn.execute(
            """
            UPDATE runtime_state
            SET last_run_kind = 'embed',
                last_index_total = 10,
                last_index_indexed = 5
            """
        )
    finally:
        conn.close()

    status = load_runtime_status(config)
    assert status.last_run_kind is None
    assert status.last_index_summary is None


# ---------------------------------------------------------------------------
# Launchd mode detection (REQ-DAEMON-031)
# ---------------------------------------------------------------------------

LAUNCHD_POLL_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>it.send.recall.daemon</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/local/bin/recall</string>
    <string>daemon</string>
    <string>--once</string>
  </array>
  <key>StartInterval</key>
  <integer>300</integer>
  <key>RunAtLoad</key>
  <false/>
</dict>
</plist>
"""

LAUNCHD_WATCH_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>it.send.recall.daemon</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/local/bin/recall</string>
    <string>daemon</string>
    <string>--mode</string>
    <string>watch</string>
  </array>
  <key>KeepAlive</key>
  <true/>
  <key>RunAtLoad</key>
  <true/>
</dict>
</plist>
"""


def test_detect_installed_mode_launchd_poll(tmp_path, monkeypatch) -> None:
    """Launchd plist with StartInterval is detected as poll mode."""
    from recall.services.daemon import _detect_installed_mode

    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    plist_path = _artifact_paths_for_launchd(tmp_path)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(LAUNCHD_POLL_PLIST, encoding="utf-8")

    mode = _detect_installed_mode(config, SchedulerKind.LAUNCHD)
    assert mode == DaemonMode.POLL


def test_detect_installed_mode_launchd_watch(tmp_path, monkeypatch) -> None:
    """Launchd plist with KeepAlive is detected as watch mode."""
    from recall.services.daemon import _detect_installed_mode

    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    plist_path = _artifact_paths_for_launchd(tmp_path)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(LAUNCHD_WATCH_PLIST, encoding="utf-8")

    mode = _detect_installed_mode(config, SchedulerKind.LAUNCHD)
    assert mode == DaemonMode.WATCH


def test_detect_installed_command_launchd(tmp_path, monkeypatch) -> None:
    """Launchd plist ProgramArguments are extracted as a command string."""
    from recall.services.daemon import _detect_installed_command

    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    plist_path = _artifact_paths_for_launchd(tmp_path)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(LAUNCHD_WATCH_PLIST, encoding="utf-8")

    command = _detect_installed_command(config, SchedulerKind.LAUNCHD)
    assert command is not None
    assert "/usr/local/bin/recall" in command
    assert "--mode" in command
    assert "watch" in command


def test_detect_installed_command_launchd_poll(tmp_path, monkeypatch) -> None:
    """Launchd poll plist extracts --once command."""
    from recall.services.daemon import _detect_installed_command

    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    plist_path = _artifact_paths_for_launchd(tmp_path)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(LAUNCHD_POLL_PLIST, encoding="utf-8")

    command = _detect_installed_command(config, SchedulerKind.LAUNCHD)
    assert command is not None
    assert "--once" in command


def _artifact_paths_for_launchd(tmp_path: Path) -> Path:
    """Return a tmp-scoped launchd plist path for artifact parsing tests."""
    return tmp_path / "Library" / "LaunchAgents" / "it.send.recall.daemon.plist"


class TestInstalledBinaryStale:
    """REQ-STALE-RESOLVE: stale check must compare resolved paths.

    Under uv's tool layout the user-visible `~/.local/bin/recall` is a symlink
    into `~/.local/share/uv/tools/recall/bin/recall`. The launchd plist
    records the symlink path while `resolve_recall_binary()` (running inside
    the daemon with launchd's minimal PATH) falls through to argv[0].resolve()
    and returns the symlink target. Comparing the raw paths flagged that as
    drift on every healthy install.
    """

    def test_returns_false_when_installed_path_resolves_to_resolved_binary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "share" / "uv" / "tools" / "recall" / "bin" / "recall"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        installed_symlink = tmp_path / "local" / "bin" / "recall"
        installed_symlink.parent.mkdir(parents=True, exist_ok=True)
        installed_symlink.symlink_to(target)

        # Daemon resolves to the symlink target (post argv[0].resolve()).
        monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(target))

        assert daemon_module._installed_binary_stale(str(installed_symlink)) is False

    def test_returns_true_when_paths_resolve_to_different_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed = tmp_path / "old" / "recall"
        installed.parent.mkdir(parents=True, exist_ok=True)
        installed.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        other = tmp_path / "new" / "recall"
        other.parent.mkdir(parents=True, exist_ok=True)
        other.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

        monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(other))

        assert daemon_module._installed_binary_stale(str(installed)) is True

    def test_returns_true_when_installed_path_does_not_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Genuine drift: the recorded path was deleted (e.g. the user
        # uninstalled the older binary the scheduler still references).
        existing = tmp_path / "share" / "recall"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(existing))

        missing = tmp_path / "does-not-exist" / "recall"
        assert daemon_module._installed_binary_stale(str(missing)) is True

    def test_returns_true_when_resolve_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If resolution itself fails we cannot prove the binary is current,
        # so the safer signal is to report stale and let the user reinstall.
        installed = tmp_path / "local" / "bin" / "recall"
        installed.parent.mkdir(parents=True, exist_ok=True)
        installed.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

        def _raise() -> str:
            raise RuntimeError("could not resolve")

        monkeypatch.setattr(daemon_module, "_resolve_recall_binary", _raise)

        assert daemon_module._installed_binary_stale(str(installed)) is True

    def test_returns_false_when_path_is_none(self) -> None:
        assert daemon_module._installed_binary_stale(None) is False
