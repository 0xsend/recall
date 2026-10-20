"""Launch agent label migration (REQ-DAEMON-076).

A host installed by an earlier release runs the daemon under the legacy label.
Every lifecycle command must act on that job until `recall daemon install`
retires it, and install must never leave both labels loaded.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import recall.services.daemon as daemon_module
from recall.cli.status_notices import _render_status_notice_from_status
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.core.types import DaemonMode, SchedulerKind
from recall.db import connect
from recall.services.daemon import (
    daemon_status,
    install_scheduler,
    restart_daemon_durable,
    start_daemon_durable,
    stop_daemon_durable,
    uninstall_scheduler,
)

NEW_LABEL = "it.send.recall.daemon"
LEGACY_LABEL = "xyz.metalrodeo.recall.daemon"
LEGACY_BINARY = "/opt/legacy/bin/recall"
DAEMON_PID = 4242


def _plist(label: str, *, mode: DaemonMode) -> str:
    schedule = (
        "  <key>KeepAlive</key>\n  <true/>\n"
        if mode == DaemonMode.WATCH
        else "  <key>StartInterval</key>\n  <integer>300</integer>\n"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0">\n<dict>\n'
        f"  <key>Label</key>\n  <string>{label}</string>\n"
        "  <key>ProgramArguments</key>\n  <array>\n"
        f"    <string>{LEGACY_BINARY}</string>\n    <string>daemon</string>\n"
        f"  </array>\n{schedule}</dict>\n</plist>\n"
    )


class FakeLaunchd:
    """launchd for several labels at once.

    `bootout` unloads a label unless it is `stuck`; `bootstrap` loads the label
    named by the plist file unless the label is `unbootable`. While a label is loaded
    its job is the daemon: `print` reports DAEMON_PID and the pid file exists.
    """

    def __init__(
        self,
        home: Path,
        config: AppConfig,
        *,
        loaded: set[str],
        stuck: frozenset[str] = frozenset(),
        unbootable: frozenset[str] = frozenset(),
        smoke_test_fails: bool = False,
    ) -> None:
        self.home = home
        self.pid_path = config.data_dir / "recall.pid"
        self.loaded = set()
        self.pids: dict[str, int] = {}
        self.stuck = stuck
        self.unbootable = unbootable
        self.smoke_test_fails = smoke_test_fails
        self.commands: list[list[str]] = []
        for label in loaded:
            self._load(label)

    def _load(self, label: str) -> None:
        self.pids.setdefault(label, DAEMON_PID + len(self.pids))
        self.loaded.add(label)
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(self.pids[label]), encoding="utf-8")

    def _unload(self, label: str) -> None:
        self.loaded.discard(label)
        if not self.loaded:
            self.pid_path.unlink(missing_ok=True)

    def daemon_alive(self, pid: int) -> bool:
        return any(self.pids[label] == pid for label in self.loaded)

    def launchctl_commands(self) -> list[list[str]]:
        return [command for command in self.commands if command[0] == "launchctl"]

    def run(
        self, args: list[str], *, check: bool = False, **_kwargs
    ) -> subprocess.CompletedProcess:
        self.commands.append(list(args))
        returncode, stdout = self._respond(args)
        if check and returncode != 0:
            raise subprocess.CalledProcessError(returncode, args, output=stdout, stderr="failed")
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")

    def _respond(self, args: list[str]) -> tuple[int, str]:
        if args[0] != "launchctl":
            # the post-install `recall --help` smoke test
            return (1 if self.smoke_test_fails else 0), ""
        verb = args[1]
        if verb == "bootout":
            label = args[2].rsplit("/", 1)[1]
            if label not in self.loaded:
                return 3, ""
            if label not in self.stuck:
                self._unload(label)
            return 0, ""
        if verb == "bootstrap":
            label = Path(args[3]).stem
            if label in self.unbootable or label in self.loaded:
                return 5, ""
            self._load(label)
            return 0, ""
        if verb in {"print", "list"}:
            label = args[2].rsplit("/", 1)[-1]
            if label not in self.loaded:
                return 113, ""
            pid = self.pids[label]
            return 0, f"pid = {pid}\n" if verb == "print" else f'"PID" = {pid};\n'
        return 0, ""  # enable


def _config(tmp_path: Path, *, mode: DaemonMode = DaemonMode.POLL) -> AppConfig:
    data_dir = tmp_path / ".local" / "share" / "recall"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / ".config" / "recall" / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(interval=300, mode=mode),
        cli=CliConfig(),
    )


def _plist_path(home: Path, label: str) -> Path:
    return home / "Library" / "LaunchAgents" / f"{label}.plist"


def _write_plist(home: Path, label: str, *, mode: DaemonMode = DaemonMode.POLL) -> Path:
    path = _plist_path(home, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_plist(label, mode=mode), encoding="utf-8")
    return path


def _mentions(commands: list[list[str]], label: str) -> list[list[str]]:
    return [command for command in commands if any(label in part for part in command)]


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    binary = tmp_path / "bin" / "recall"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(binary))
    # A job that never drains would otherwise hold each install for the full
    # 30s unload deadline.
    monkeypatch.setattr(
        daemon_module,
        "_wait_for_launchd_unload",
        functools.partial(daemon_module._wait_for_launchd_unload, timeout_seconds=0.0),
    )
    monkeypatch.setattr(daemon_module, "_LAUNCHD_UNLOAD_WAIT_SECONDS_MAX", 0.0)
    return tmp_path


def _fake_launchd(
    monkeypatch: pytest.MonkeyPatch, host: Path, config: AppConfig, **kwargs
) -> FakeLaunchd:
    launchd = FakeLaunchd(host, config, **kwargs)
    monkeypatch.setattr(subprocess, "run", launchd.run)
    monkeypatch.setattr(daemon_module, "_process_is_alive", launchd.daemon_alive)
    monkeypatch.setattr("recall.cli.compact._process_is_alive", launchd.daemon_alive)
    return launchd


def test_compaction_cycles_a_legacy_only_install_through_its_label(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.cli.compact import _restart_daemon, _stop_daemon

    config = _config(host, mode=DaemonMode.WATCH)
    legacy_plist = _write_plist(host, LEGACY_LABEL, mode=DaemonMode.WATCH)
    launchd = _fake_launchd(monkeypatch, host, config, loaded={LEGACY_LABEL})
    monkeypatch.setattr(
        "recall.cli.compact._stop_auto_forked_daemon",
        lambda pid: pytest.fail(f"launchd owns pid {pid}; a bare SIGTERM would respawn it"),
    )

    daemon_was_running, scheduler = _stop_daemon(config)
    assert daemon_was_running is True
    assert scheduler == SchedulerKind.LAUNCHD
    assert launchd.loaded == set()

    assert _restart_daemon(config, scheduler) is True

    assert launchd.loaded == {LEGACY_LABEL}
    assert ["launchctl", "bootout", f"gui/{os.getuid()}/{LEGACY_LABEL}"] in launchd.commands
    assert ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(legacy_plist)] in launchd.commands
    assert _mentions(launchd.commands, NEW_LABEL) == []


def test_start_acts_on_a_legacy_only_install(host: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(host)
    legacy_plist = _write_plist(host, LEGACY_LABEL)
    launchd = _fake_launchd(monkeypatch, host, config, loaded=set())

    result = start_daemon_durable(config, timeout=1.0)

    assert result.started is True, result.message
    assert launchd.loaded == {LEGACY_LABEL}
    assert ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(legacy_plist)] in launchd.commands
    assert _mentions(launchd.commands, NEW_LABEL) == []


def test_restart_acts_on_a_legacy_only_install(host: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(host)
    legacy_plist = _write_plist(host, LEGACY_LABEL)
    launchd = _fake_launchd(monkeypatch, host, config, loaded={LEGACY_LABEL})

    result = restart_daemon_durable(config, timeout=1.0)

    assert result.stopped is True, result.message
    assert result.started is True, result.message
    assert launchd.loaded == {LEGACY_LABEL}
    assert ["launchctl", "bootout", f"gui/{os.getuid()}/{LEGACY_LABEL}"] in launchd.commands
    assert ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(legacy_plist)] in launchd.commands
    assert _mentions(launchd.commands, NEW_LABEL) == []


def test_status_reports_a_legacy_install_without_touching_launchd(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(host)
    connect(config).close()
    legacy_plist = _write_plist(host, LEGACY_LABEL, mode=DaemonMode.WATCH)
    launchd = _fake_launchd(monkeypatch, host, config, loaded={LEGACY_LABEL})

    status = daemon_status(config=config)

    assert status.installed is True
    assert status.scheduler == SchedulerKind.LAUNCHD
    assert status.installed_mode == DaemonMode.WATCH
    assert status.installed_binary_path == LEGACY_BINARY
    assert status.artifact_paths == (str(legacy_plist),)
    notice = _render_status_notice_from_status(asdict(status), interval_seconds=300)
    assert notice is not None
    assert "legacy launchd label" in notice
    assert "recall daemon install" in notice
    verbs = {command[1] for command in launchd.launchctl_commands()}
    assert verbs <= {"print", "list"}
    assert launchd.loaded == {LEGACY_LABEL}
    assert legacy_plist.exists()


def test_install_migrates_a_legacy_install_to_the_new_label(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(host)
    legacy_plist = _write_plist(host, LEGACY_LABEL)
    launchd = _fake_launchd(monkeypatch, host, config, loaded={LEGACY_LABEL})

    status = install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    new_plist = _plist_path(host, NEW_LABEL)
    assert launchd.loaded == {NEW_LABEL}
    assert not legacy_plist.exists()
    assert f"<string>{NEW_LABEL}</string>" in new_plist.read_text(encoding="utf-8")
    assert "metalrodeo" not in new_plist.read_text(encoding="utf-8")
    assert status.installed is True
    assert status.launchd_legacy_label is False
    assert status.artifact_paths == (str(new_plist),)
    commands = launchd.commands
    legacy_bootout = commands.index(["launchctl", "bootout", f"gui/{os.getuid()}/{LEGACY_LABEL}"])
    enable = commands.index(["launchctl", "enable", f"gui/{os.getuid()}/{NEW_LABEL}"])
    bootstrap = commands.index(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(new_plist)])
    assert legacy_bootout < enable < bootstrap
    assert not [command for command in commands if command[:2] == ["launchctl", "disable"]]


def test_install_aborts_before_bootstrap_when_the_legacy_job_never_drains(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(host)
    legacy_plist = _write_plist(host, LEGACY_LABEL)
    legacy_contents = legacy_plist.read_bytes()
    launchd = _fake_launchd(
        monkeypatch, host, config, loaded={LEGACY_LABEL}, stuck=frozenset({LEGACY_LABEL})
    )

    with pytest.raises(RuntimeError, match=rf"{LEGACY_LABEL} pid={DAEMON_PID} is still alive"):
        install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    assert not [c for c in launchd.commands if c[:2] == ["launchctl", "bootstrap"]]
    assert launchd.loaded == {LEGACY_LABEL}
    assert legacy_plist.read_bytes() == legacy_contents
    assert not _plist_path(host, NEW_LABEL).exists()


def test_install_restores_the_legacy_job_when_bootstrap_retries_run_out(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(host)
    legacy_plist = _write_plist(host, LEGACY_LABEL)
    legacy_contents = legacy_plist.read_bytes()
    launchd = _fake_launchd(
        monkeypatch, host, config, loaded={LEGACY_LABEL}, unbootable=frozenset({NEW_LABEL})
    )

    with pytest.raises(subprocess.CalledProcessError):
        install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    assert legacy_plist.read_bytes() == legacy_contents
    assert not _plist_path(host, NEW_LABEL).exists()
    assert launchd.loaded == {LEGACY_LABEL}
    assert launchd.commands[-1] == [
        "launchctl",
        "bootstrap",
        f"gui/{os.getuid()}",
        str(legacy_plist),
    ]


def test_interrupted_migration_status_prefers_new_label_and_install_converges(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(host)
    connect(config).close()
    legacy_plist = _write_plist(host, LEGACY_LABEL)
    new_plist = _write_plist(host, NEW_LABEL)
    launchd = _fake_launchd(monkeypatch, host, config, loaded={NEW_LABEL, LEGACY_LABEL})

    status = daemon_status(config=config)
    assert status.installed is True
    assert status.launchd_legacy_label is False
    assert status.launchd_legacy_leftover is True
    assert status.artifact_paths == (str(new_plist),)
    notice = _render_status_notice_from_status(asdict(status), interval_seconds=300)
    assert notice is not None
    assert "recall daemon install" in notice
    assert "retire" in notice

    install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    assert launchd.loaded == {NEW_LABEL}
    assert not legacy_plist.exists()
    assert new_plist.exists()


def test_uninstall_removes_both_labels(host: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(host)
    connect(config).close()
    legacy_plist = _write_plist(host, LEGACY_LABEL)
    new_plist = _write_plist(host, NEW_LABEL)
    launchd = _fake_launchd(monkeypatch, host, config, loaded={NEW_LABEL, LEGACY_LABEL})

    status = uninstall_scheduler(config=config)

    assert launchd.loaded == set()
    assert not legacy_plist.exists()
    assert not new_plist.exists()
    assert status.installed is False


def test_stop_boots_out_a_legacy_job_left_beside_the_new_one(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(host)
    _write_plist(host, NEW_LABEL)
    launchd = _fake_launchd(monkeypatch, host, config, loaded={NEW_LABEL, LEGACY_LABEL})

    result = stop_daemon_durable(config, timeout=1.0)

    assert result.stopped is True, result.message
    assert launchd.loaded == set()


def test_rollback_keeps_the_new_job_alone_when_it_will_not_unload(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(host)
    legacy_plist = _write_plist(host, LEGACY_LABEL)
    legacy_contents = legacy_plist.read_bytes()
    launchd = _fake_launchd(
        monkeypatch,
        host,
        config,
        loaded={LEGACY_LABEL},
        stuck=frozenset({NEW_LABEL}),
        smoke_test_fails=True,
    )

    with pytest.raises(subprocess.CalledProcessError):
        install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    assert launchd.loaded == {NEW_LABEL}
    legacy_bootstrap = ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(legacy_plist)]
    assert legacy_bootstrap not in launchd.commands
    # The plist of the job still loaded stays, so every command resolves to it.
    assert _plist_path(host, NEW_LABEL).exists()
    assert legacy_plist.read_bytes() == legacy_contents


def test_rollback_leaves_a_stopped_legacy_install_stopped(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(host)
    legacy_plist = _write_plist(host, LEGACY_LABEL)
    legacy_contents = legacy_plist.read_bytes()
    launchd = _fake_launchd(
        monkeypatch, host, config, loaded=set(), unbootable=frozenset({NEW_LABEL})
    )

    with pytest.raises(subprocess.CalledProcessError):
        install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    assert launchd.loaded == set()
    assert legacy_plist.read_bytes() == legacy_contents
    assert not _plist_path(host, NEW_LABEL).exists()


class LingeringLaunchd(FakeLaunchd):
    """Unregister a job while its process still owns the database."""

    def __init__(self, *args, exit_after_polls: int | None, **kwargs) -> None:
        self.process_running = True
        self.exit_after_polls = exit_after_polls
        self.exit_polls = 0
        self.retiring_pid: int | None = None
        self.bootstrap_while_alive = False
        super().__init__(*args, **kwargs)

    def _unload(self, label: str) -> None:
        if label in self.loaded:
            self.retiring_pid = self.pids[label]
            self.process_running = True
            self.exit_polls = 0
        super()._unload(label)

    def daemon_alive(self, pid: int) -> bool:
        if self.loaded:
            return super().daemon_alive(pid)
        assert pid == self.retiring_pid
        self.exit_polls += 1
        if self.exit_after_polls is not None and self.exit_polls >= self.exit_after_polls:
            self.process_running = False
        return self.process_running

    def _respond(self, args: list[str]) -> tuple[int, str]:
        if args[:2] == ["launchctl", "bootstrap"] and self.process_running:
            self.bootstrap_while_alive = True
            return 5, "old process holds the database"
        return super()._respond(args)


@pytest.mark.parametrize("label", [NEW_LABEL, LEGACY_LABEL])
@pytest.mark.parametrize("poll_seconds", [0.01, 0.1])
def test_install_waits_for_unregistered_process_to_exit(
    host: Path, monkeypatch: pytest.MonkeyPatch, label: str, poll_seconds: float
) -> None:
    """REQ-DAEMON-078: label removal alone does not release the database owner."""
    config = _config(host)
    _write_plist(host, label)
    launchd = LingeringLaunchd(host, config, loaded={label}, exit_after_polls=3)
    monkeypatch.setattr(subprocess, "run", launchd.run)
    monkeypatch.setattr(daemon_module, "_process_is_alive", launchd.daemon_alive)
    monkeypatch.setattr(daemon_module, "_LAUNCHD_UNLOAD_POLL_SECONDS", poll_seconds)
    monkeypatch.setattr(daemon_module, "_LAUNCHD_UNLOAD_WAIT_SECONDS_MAX", 1.0)

    status = install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    assert status.installed is True
    assert launchd.loaded == {NEW_LABEL}
    assert launchd.process_running is False
    assert launchd.bootstrap_while_alive is False


@pytest.mark.parametrize("label", [NEW_LABEL, LEGACY_LABEL])
def test_install_does_not_bootstrap_or_restore_over_a_process_that_will_not_exit(
    host: Path, monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    """REQ-DAEMON-078: a bounded teardown failure must not start a second owner."""
    config = _config(host)
    plist = _write_plist(host, label)
    original = plist.read_bytes()
    launchd = LingeringLaunchd(host, config, loaded={label}, exit_after_polls=None)
    monkeypatch.setattr(subprocess, "run", launchd.run)
    monkeypatch.setattr(daemon_module, "_process_is_alive", launchd.daemon_alive)
    monkeypatch.setattr(daemon_module, "_LAUNCHD_UNLOAD_WAIT_SECONDS_MAX", 0.0)

    with pytest.raises(RuntimeError, match=rf"{label}.*pid={DAEMON_PID}.*still alive"):
        install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    assert launchd.bootstrap_while_alive is False
    assert not [c for c in launchd.commands if c[:2] == ["launchctl", "bootstrap"]]
    assert plist.read_bytes() == original


def test_failed_install_waits_for_replacement_process_before_restoring_previous_job(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-078: failure restore obeys the same process-exit barrier."""
    config = _config(host)
    plist = _write_plist(host, NEW_LABEL)
    original = plist.read_bytes()
    launchd = LingeringLaunchd(
        host, config, loaded={NEW_LABEL}, exit_after_polls=3, smoke_test_fails=True
    )
    monkeypatch.setattr(subprocess, "run", launchd.run)
    monkeypatch.setattr(daemon_module, "_process_is_alive", launchd.daemon_alive)
    monkeypatch.setattr(daemon_module, "_LAUNCHD_UNLOAD_POLL_SECONDS", 0.01)
    monkeypatch.setattr(daemon_module, "_LAUNCHD_UNLOAD_WAIT_SECONDS_MAX", 1.0)

    with pytest.raises(subprocess.CalledProcessError):
        install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    assert launchd.loaded == {NEW_LABEL}
    assert launchd.bootstrap_while_alive is False
    assert plist.read_bytes() == original


def test_failed_install_does_not_restore_over_a_replacement_process_that_will_not_exit(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-078: rollback also fails closed on process-exit timeout."""
    config = _config(host)
    _write_plist(host, NEW_LABEL)
    launchd = LingeringLaunchd(host, config, loaded={NEW_LABEL}, exit_after_polls=1)

    def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
        if args[0] != "launchctl":
            # The installed job is live, but its binary fails the smoke test;
            # this process then outlives its registration during rollback.
            launchd.exit_after_polls = None
            if kwargs.get("check"):
                raise subprocess.CalledProcessError(1, args, stderr="smoke failed")
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="smoke failed")
        return launchd.run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(daemon_module, "_process_is_alive", launchd.daemon_alive)

    with pytest.raises(subprocess.CalledProcessError):
        install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    bootstraps = [c for c in launchd.commands if c[:2] == ["launchctl", "bootstrap"]]
    assert len(bootstraps) == 1, "only the candidate was loaded; restore must not start another"
    assert launchd.bootstrap_while_alive is False
    assert launchd.process_running is True
