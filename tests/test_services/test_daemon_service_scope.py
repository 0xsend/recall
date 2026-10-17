"""Service actions and probes follow the data dir the installed service runs (REQ-DAEMON-080, -081).

A user has one service. These tests install it for one data dir through the product's own
install path, then select another data dir, as a scratch or test daemon does.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
import recall.cli.compact as compact_module
import recall.services.daemon as daemon_module
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.core.types import DaemonMode, SchedulerKind
from recall.services.daemon import (
    ServiceProbeCache,
    daemon_status,
    install_scheduler,
    restart_daemon_durable,
    start_daemon_durable,
    stop_daemon_durable,
    stop_daemon_soft,
    uninstall_scheduler,
)

pytestmark = pytest.mark.usefixtures("launchd_without_legacy_job")

SCHEDULER_TOOLS = {"launchctl", "systemctl", "crontab"}
MUTATING_VERBS = {"bootout", "bootstrap", "enable", "disable", "kickstart", "stop", "start"}


@dataclass
class Scheduler:
    """Records scheduler calls and answers as a host whose service is loaded and healthy."""

    calls: list[list[str]] = field(default_factory=list)
    crontab: str = ""
    service_kind: SchedulerKind | None = None
    booted_out_labels: set[str] = field(default_factory=set)
    stopped_units: set[str] = field(default_factory=set)

    def run(
        self, args: list[str], *_args: object, input: str | None = None, **_kwargs: object
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        if args == ["crontab", "-"]:
            self.crontab = input or ""
        if args == ["crontab", "-l"]:
            return subprocess.CompletedProcess(args, 0, self.crontab, "")
        if args[:2] == ["launchctl", "bootout"]:
            self.booted_out_labels.add(args[2].rpartition("/")[2])
        if args[:2] == ["launchctl", "bootstrap"]:
            self.booted_out_labels.clear()
        if args[:2] in (["launchctl", "print"], ["launchctl", "list"]) and (
            args[2].rpartition("/")[2] in self.booted_out_labels
        ):
            return subprocess.CompletedProcess(args, 113, "", "Could not find service")
        if args[:3] == ["systemctl", "--user", "stop"]:
            self.stopped_units.update(args[3:])
        if args[:3] in (["systemctl", "--user", "start"], ["systemctl", "--user", "restart"]):
            self.stopped_units.difference_update(args[3:])
        if args[:3] == ["systemctl", "--user", "is-active"] and args[-1] in self.stopped_units:
            return subprocess.CompletedProcess(args, 3, "inactive\n", "")
        stdout = '{\n\t"LastExitStatus" = 0;\n};\n' if args[:2] == ["launchctl", "list"] else ""
        if args[:2] == ["systemctl", "--user"] and "show" in args:
            stdout = "Result=success\nExecMainStatus=0\n"
        return subprocess.CompletedProcess(args, 0, stdout, "")

    def scheduler_calls(self) -> list[list[str]]:
        return [call for call in self.calls if call and call[0] in SCHEDULER_TOOLS]

    def mutating_calls(self) -> list[list[str]]:
        return [
            call
            for call in self.scheduler_calls()
            if any(word in MUTATING_VERBS for word in call[1:3]) or call == ["crontab", "-"]
        ]


def _config(home: Path, data_dir: Path) -> AppConfig:
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=home / ".config" / "recall" / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(interval=300, mode=DaemonMode.WATCH),
        cli=CliConfig(),
    )


HOST_SCHEDULERS = {
    "linux": SchedulerKind.SYSTEMD,
    "darwin": SchedulerKind.LAUNCHD,
    "cron": SchedulerKind.CRON,
}


@pytest.fixture(params=["linux", "darwin"])
def host(request, tmp_path, monkeypatch) -> tuple[Scheduler, AppConfig, AppConfig]:
    """A host whose service runs `service`; returns the recorder, service and scratch configs."""
    return _installed_host(request.param, tmp_path, monkeypatch)


@pytest.fixture(params=["linux", "darwin", "cron"])
def any_host(request, tmp_path, monkeypatch) -> tuple[Scheduler, AppConfig, AppConfig]:
    """`host`, plus a Linux host whose service is the crontab block."""
    return _installed_host(request.param, tmp_path, monkeypatch)


def _installed_host(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Scheduler, AppConfig, AppConfig]:
    platform = "darwin" if name == "darwin" else "linux"
    monkeypatch.setenv("HOME", str(tmp_path))
    binary = tmp_path / "bin" / "recall"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    scheduler = Scheduler()
    monkeypatch.setattr(daemon_module.sys, "platform", platform)
    monkeypatch.setattr(compact_module.sys, "platform", platform)
    monkeypatch.setattr(daemon_module.shutil, "which", lambda _name: str(binary))
    monkeypatch.setattr(daemon_module, "_systemd_user_available", lambda: True)
    monkeypatch.setattr(subprocess, "run", scheduler.run)
    service = _config(tmp_path, tmp_path / "service-data")
    scratch = _config(tmp_path, tmp_path / "scratch-data")
    scratch.data_dir.mkdir(parents=True)
    # The relaunched service would take the database; this one never starts.
    monkeypatch.setattr(daemon_module, "_update_installed_scheduler", lambda *_a, **_k: None)
    scheduler.service_kind = HOST_SCHEDULERS[name]
    install_scheduler(scheduler=scheduler.service_kind, config=service)
    scheduler.calls.clear()
    return scheduler, service, scratch


def test_stop_from_another_data_dir_refuses_and_leaves_the_service_running(any_host) -> None:
    scheduler, service, scratch = any_host

    result = stop_daemon_durable(scratch, timeout=0.2)

    assert result.stopped is False
    assert str(service.data_dir) in (result.message or "")
    assert "recall daemon stop --soft" in (result.message or "")
    assert scheduler.mutating_calls() == []


def test_restart_and_start_from_another_data_dir_refuse(any_host) -> None:
    scheduler, _service, scratch = any_host

    restarted = restart_daemon_durable(scratch, timeout=0.2)
    started = start_daemon_durable(scratch, timeout=0.2)

    assert (restarted.stopped, restarted.started, started.started) == (False, False, False)
    assert "recall daemon start --background" in (started.message or "")
    assert scheduler.mutating_calls() == []


def test_uninstall_from_another_data_dir_refuses_and_keeps_the_service(any_host) -> None:
    scheduler, service, scratch = any_host

    with pytest.raises(RuntimeError, match="refusing to uninstall"):
        uninstall_scheduler(scratch)

    kept = daemon_module.resolve_service_scope(scratch).other_service
    assert kept == daemon_module.InstalledService(scheduler.service_kind, service.data_dir)
    assert scheduler.mutating_calls() == []


def test_status_from_another_data_dir_does_not_claim_the_service(any_host) -> None:
    scheduler, service, scratch = any_host

    status = daemon_status(scratch)

    assert status.installed is False
    assert status.scheduler is None
    assert status.artifact_paths == ()
    assert status.installed_binary_path is None
    assert status.scheduler_health_state is None
    assert status.service_data_dir == str(service.data_dir)
    # Reading the crontab is how a cron artifact is read; nothing else runs.
    artifact_reads = [["crontab", "-l"]] if scheduler.service_kind == SchedulerKind.CRON else []
    assert scheduler.scheduler_calls() == artifact_reads


def test_compact_from_another_data_dir_leaves_the_service_alone(any_host) -> None:
    scheduler, _service, scratch = any_host

    assert compact_module._stop_daemon(scratch) == (False, None)
    assert scheduler.mutating_calls() == []


def test_the_service_data_dir_keeps_acting_on_the_service(host) -> None:
    scheduler, service, _scratch = host

    status = daemon_status(service)

    assert status.installed is True
    assert status.service_data_dir == str(service.data_dir)
    assert status.scheduler_health_state == "ok"
    for action in (
        lambda: stop_daemon_durable(service, timeout=0.2),
        lambda: start_daemon_durable(service, timeout=0.2),
        lambda: restart_daemon_durable(service, timeout=0.2),
        lambda: compact_module._stop_daemon(service),
    ):
        scheduler.calls.clear()
        outcome = action()
        assert "refusing" not in str(outcome)
        assert scheduler.mutating_calls() != []
    uninstall_scheduler(service)
    assert daemon_module.resolve_service_scope(service).service is None


def _respell_installed_data_dir(service: AppConfig, spelling: str) -> None:
    """Rewrite the install-written `RECALL_DATA_DIR` value, as a hand edit would."""
    for artifact in daemon_status(service).artifact_paths:
        path = Path(artifact)
        original = path.read_text(encoding="utf-8")
        text = original
        for end in ("\n", "</string>"):
            text = text.replace(f"{service.data_dir}{end}", f"{spelling}{end}")
        assert text != original
        path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("spelling", ["symlink", "trailing slash"])
def test_another_spelling_of_the_service_data_dir_is_the_service_data_dir(host, spelling) -> None:
    scheduler, service, scratch = host
    selected = service
    if spelling == "symlink":
        link = service.data_dir.with_name("service-link")
        link.symlink_to(service.data_dir, target_is_directory=True)
        selected = _config(link.parent, link)
    else:
        _respell_installed_data_dir(service, f"{service.data_dir}/")

    status = daemon_status(selected)
    result = stop_daemon_durable(selected, timeout=0.2)

    assert status.installed is True
    assert "refusing" not in (result.message or "")
    assert scheduler.mutating_calls() != []

    scheduler.calls.clear()
    refused = stop_daemon_durable(scratch, timeout=0.2)
    assert str(scratch.data_dir) in (refused.message or "")
    assert f"RECALL_DATA_DIR={service.data_dir}" in (refused.message or "")
    assert scheduler.mutating_calls() == []


def test_a_tilde_in_the_installed_data_dir_is_taken_literally(host) -> None:
    """The service process does not expand `~`, so `~/service-data` is not `$HOME/service-data`."""
    scheduler, service, _scratch = host
    _respell_installed_data_dir(service, "~/service-data")

    refused = stop_daemon_durable(service, timeout=0.2)

    assert str(service.data_dir) in (refused.message or "")
    assert "RECALL_DATA_DIR=~/service-data" in (refused.message or "")
    assert scheduler.mutating_calls() == []


def test_a_drop_in_environment_decides_the_service_data_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    unit = tmp_path / ".config" / "systemd" / "user" / "recall-daemon.service"
    (unit.parent / "recall-daemon.service.d").mkdir(parents=True)
    unit.write_text(
        f"[Service]\nType=simple\nEnvironment=RECALL_DATA_DIR={tmp_path / 'a'}\n", encoding="utf-8"
    )
    (unit.parent / "recall-daemon.service.d" / "override.conf").write_text(
        f'[Service]\nEnvironment="RECALL_DATA_DIR={tmp_path / "b c"}"\n', encoding="utf-8"
    )

    assert (
        daemon_module.resolve_service_scope(_config(tmp_path, tmp_path / "b c")).other_service
        is None
    )
    other = daemon_module.resolve_service_scope(_config(tmp_path, tmp_path / "a")).other_service
    assert other is not None and other.data_dir == tmp_path / "b c"


def test_a_service_without_a_data_dir_runs_the_default_one(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    unit = tmp_path / ".config" / "systemd" / "user" / "recall-daemon.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("[Service]\nType=simple\n", encoding="utf-8")

    default = _config(tmp_path, tmp_path / ".local" / "share" / "recall")
    assert daemon_module.resolve_service_scope(default).other_service is None
    assert daemon_module.resolve_service_scope(_config(tmp_path, tmp_path / "x")).other_service


def _crontab_with_block(block: str) -> str:
    return (
        "MAILTO=me\n0 * * * * /bin/other-job\n"
        f"# BEGIN recall daemon\n{block}\n# END recall daemon\n"
        "0 * * * * /bin/later-job\n"
    )


@pytest.mark.parametrize(
    ("block", "command"),
    [
        (
            "*/5 * * * * RECALL_DATA_DIR='{data}' /bin/recall daemon --once",
            ["RECALL_DATA_DIR={data}", "/bin/recall", "daemon", "--once"],
        ),
        (
            "# hand-written note\nSHELL=/bin/sh\n*/5 * * * * RECALL_DATA_DIR='{data}' /bin/recall",
            ["RECALL_DATA_DIR={data}", "/bin/recall"],
        ),
        (
            'RECALL_DATA_DIR = "{data}"\n\n*/5 * * * * /bin/recall daemon --once',
            ["/bin/recall", "daemon", "--once"],
        ),
        ("@reboot RECALL_DATA_DIR='{data}' /bin/recall", ["RECALL_DATA_DIR={data}", "/bin/recall"]),
        (
            "@reboot RECALL_DATA_DIR='{data}' /bin/recall daemon --once --foo",
            ["RECALL_DATA_DIR={data}", "/bin/recall", "daemon", "--once", "--foo"],
        ),
    ],
    ids=[
        "install-written",
        "comment and SHELL first",
        "environment line",
        "nickname",
        "long nickname",
    ],
)
def test_a_cron_service_is_read_from_the_entry_cron_runs(
    tmp_path, monkeypatch, block, command
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    data = tmp_path / "cron data"
    crontab = _crontab_with_block(block.format(data=data))
    monkeypatch.setattr(daemon_module, "_read_crontab", lambda: crontab)

    scope = daemon_module.resolve_service_scope(_config(tmp_path, tmp_path / "scratch"))
    assert scope.other_service == daemon_module.InstalledService(SchedulerKind.CRON, data)

    view = daemon_module.probe_service(_config(tmp_path, data), SchedulerKind.CRON)
    assert view.installed is True
    assert shlex.split(view.installed_command or "") == [word.format(data=data) for word in command]
    assert view.installed_binary_path == "/bin/recall"


@pytest.mark.parametrize(
    ("block", "service_dir"),
    [
        ("RECALL_DATA_DIR=/tmp/inside", "/tmp/inside"),
        ("# no entry", None),
        ("", None),
    ],
    ids=["assignment only", "comment only", "markers only"],
)
def test_a_cron_block_without_an_entry_ignores_lines_after_it(
    tmp_path, monkeypatch, block, service_dir
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    later = tmp_path / "later"
    scheduler = Scheduler(
        crontab=(
            f"# BEGIN recall daemon\n{block}\n# END recall daemon\n"
            f"RECALL_DATA_DIR={later}\n0 * * * * /bin/later-job\n"
        )
    )
    monkeypatch.setattr(daemon_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "run", scheduler.run)
    expected = Path(service_dir) if service_dir else tmp_path / ".local" / "share" / "recall"

    scope = daemon_module.resolve_service_scope(_config(tmp_path, later))
    assert scope.other_service == daemon_module.InstalledService(SchedulerKind.CRON, expected)

    with pytest.raises(RuntimeError, match="refusing to uninstall"):
        uninstall_scheduler(_config(tmp_path, later))
    assert scheduler.mutating_calls() == []


def test_a_daemon_serves_its_last_service_measurement_until_refreshed(host) -> None:
    scheduler, service, _scratch = host
    probes = ServiceProbeCache(service)
    first = daemon_status(service, service_probes=probes)
    probe_calls = len(scheduler.scheduler_calls())

    again = daemon_status(service, service_probes=probes)
    assert len(scheduler.scheduler_calls()) == probe_calls
    assert again.service_measured_at == first.service_measured_at
    assert again.scheduler_health_state == first.scheduler_health_state == "ok"

    refreshed = probes.refresh_current()
    assert refreshed is not None
    assert (
        daemon_status(service, service_probes=probes).service_measured_at == refreshed.measured_at
    )
    assert len(scheduler.scheduler_calls()) > probe_calls


def test_a_changed_installed_scheduler_record_is_measured_at_once(host) -> None:
    _scheduler, service, _scratch = host
    probes = ServiceProbeCache(service)
    before = probes.view_for(None)

    after = probes.view_for(SchedulerKind.CRON)

    assert after.installed_scheduler_record == SchedulerKind.CRON
    assert after.measured_at >= before.measured_at


@dataclass
class Signals:
    """Passes `kill(pid, 0)` probes through and records every real signal without sending it."""

    sent: list[tuple[int, int]] = field(default_factory=list)
    probe: Callable[[int, int], None] = os.kill

    def kill(self, pid: int, signal_number: int) -> None:
        if signal_number == 0:
            self.probe(pid, 0)
        else:
            self.sent.append((pid, signal_number))


@pytest.fixture
def signals(monkeypatch) -> Signals:
    recorder = Signals()
    monkeypatch.setattr(daemon_module.os, "kill", recorder.kill)
    return recorder


@pytest.fixture
def zombie_pid_file(tmp_path, monkeypatch):
    """A data dir whose `recall.pid` names a child that has exited and is not reaped."""
    if not hasattr(os, "waitid"):
        pytest.skip("os.waitid(WNOWAIT) is needed to keep a zombie")
    # No service is installed; the pid file decides.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(daemon_module, "_read_crontab", lambda: "")
    config = _config(tmp_path, tmp_path / "data")
    config.data_dir.mkdir(parents=True)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        # Wait for the exit without reaping it, so the pid stays a zombie.
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        (config.data_dir / "recall.pid").write_text(f"{child.pid}\n", encoding="utf-8")
        yield config, child.pid
    finally:
        child.wait(timeout=5)


def test_soft_stop_counts_an_exited_unreaped_daemon_as_stopped(zombie_pid_file, signals) -> None:
    config, _pid = zombie_pid_file

    result = stop_daemon_soft(config)

    assert result.stopped is True
    assert result.duration_seconds < 5
    assert signals.sent == []
    assert not (config.data_dir / "recall.pid").exists()


def test_durable_stop_counts_an_exited_unreaped_daemon_as_stopped(zombie_pid_file, signals) -> None:
    config, _pid = zombie_pid_file
    # A cron install has no durable stop verb, so the pid file alone decides.
    config = replace(config, daemon=replace(config.daemon, scheduler=SchedulerKind.CRON))

    result = stop_daemon_durable(config, timeout=0.5)

    assert result.stopped is True
    assert result.duration_seconds < 5
    assert signals.sent == []


@pytest.mark.parametrize(("state", "stopped"), [("Z+", True), ("S", False)])
def test_soft_stop_reads_the_process_state_from_ps_off_linux(
    tmp_path, monkeypatch, signals, state, stopped
) -> None:
    config = _config(tmp_path, tmp_path / "data")
    config.data_dir.mkdir(parents=True)
    # This test process stands in for the daemon: `kill(pid, 0)` succeeds for it.
    pid = os.getpid()
    (config.data_dir / "recall.pid").write_text(f"{pid}\n", encoding="utf-8")
    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        daemon_module,
        "_run_lifecycle_query",
        lambda args: (
            subprocess.CompletedProcess(args, 0, f"{state}\n", "")
            if args == ["ps", "-o", "stat=", "-p", str(pid)]
            else None
        ),
    )
    monkeypatch.setattr(daemon_module, "_wait_until_stopped", _report_still_running)

    result = stop_daemon_soft(config)

    assert result.stopped is stopped
    assert signals.sent == ([] if stopped else [(pid, signal.SIGTERM)])


def _report_still_running(config, *, pid, timeout, started_at):
    return daemon_module.DaemonStopResult(
        scheduler="sentinel", stopped=False, pid=pid, duration_seconds=0.0, message="alive"
    )
