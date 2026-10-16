"""Service actions and probes follow the data dir the installed service runs (REQ-DAEMON-080, -081).

A user has one service. These tests install it for one data dir through the product's own
install path, then select another data dir, as a scratch or test daemon does.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
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

    def run(
        self, args: list[str], *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
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
            if any(word in MUTATING_VERBS for word in call[1:3])
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


@pytest.fixture(params=["linux", "darwin"])
def host(request, tmp_path, monkeypatch) -> tuple[Scheduler, AppConfig, AppConfig]:
    """A host whose service runs `service`; returns the recorder, service and scratch configs."""
    monkeypatch.setenv("HOME", str(tmp_path))
    binary = tmp_path / "bin" / "recall"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    scheduler = Scheduler()
    monkeypatch.setattr(daemon_module.sys, "platform", request.param)
    monkeypatch.setattr(compact_module.sys, "platform", request.param)
    monkeypatch.setattr(daemon_module.shutil, "which", lambda _name: str(binary))
    monkeypatch.setattr(daemon_module, "_systemd_user_available", lambda: True)
    monkeypatch.setattr(subprocess, "run", scheduler.run)
    service = _config(tmp_path, tmp_path / "service-data")
    scratch = _config(tmp_path, tmp_path / "scratch-data")
    scratch.data_dir.mkdir(parents=True)
    kind = SchedulerKind.LAUNCHD if request.param == "darwin" else SchedulerKind.SYSTEMD
    # The relaunched service would take the database; this one never starts.
    monkeypatch.setattr(daemon_module, "_update_installed_scheduler", lambda *_a, **_k: None)
    install_scheduler(scheduler=kind, config=service)
    scheduler.calls.clear()
    return scheduler, service, scratch


def test_stop_from_another_data_dir_refuses_and_leaves_the_service_running(host) -> None:
    scheduler, service, scratch = host

    result = stop_daemon_durable(scratch, timeout=0.2)

    assert result.stopped is False
    assert str(service.data_dir) in (result.message or "")
    assert "recall daemon stop --soft" in (result.message or "")
    assert scheduler.mutating_calls() == []


def test_restart_and_start_from_another_data_dir_refuse(host) -> None:
    scheduler, _service, scratch = host

    restarted = restart_daemon_durable(scratch, timeout=0.2)
    started = start_daemon_durable(scratch, timeout=0.2)

    assert (restarted.stopped, restarted.started, started.started) == (False, False, False)
    assert "recall daemon start --background" in (started.message or "")
    assert scheduler.mutating_calls() == []


def test_uninstall_from_another_data_dir_refuses_and_keeps_the_service(host) -> None:
    scheduler, service, scratch = host
    before = daemon_status(service)

    with pytest.raises(RuntimeError, match="refusing to uninstall"):
        uninstall_scheduler(scratch)

    assert daemon_status(service).artifact_paths == before.artifact_paths != ()
    assert scheduler.mutating_calls() == []


def test_status_from_another_data_dir_does_not_claim_the_service(host) -> None:
    scheduler, service, scratch = host

    status = daemon_status(scratch)

    assert status.installed is False
    assert status.scheduler is None
    assert status.artifact_paths == ()
    assert status.installed_binary_path is None
    assert status.scheduler_health_state is None
    assert status.service_data_dir == str(service.data_dir)
    assert scheduler.scheduler_calls() == []


def test_compact_from_another_data_dir_leaves_the_service_alone(host) -> None:
    scheduler, _service, scratch = host

    assert compact_module._stop_daemon(scratch) == (False, None)
    assert scheduler.mutating_calls() == []


def test_the_service_data_dir_keeps_acting_on_the_service(host) -> None:
    scheduler, service, _scratch = host

    status = daemon_status(service)
    stop_daemon_durable(service, timeout=0.2)

    assert status.installed is True
    assert status.service_data_dir == str(service.data_dir)
    assert status.scheduler_health_state == "ok"
    assert scheduler.mutating_calls() != []


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


def test_a_cron_service_is_scoped_by_its_environment_prefix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    crontab = (
        "# BEGIN recall daemon\n"
        f"*/5 * * * * RECALL_DATA_DIR='{tmp_path / 'cron data'}' /bin/recall daemon --once\n"
        "# END recall daemon\n"
    )
    monkeypatch.setattr(daemon_module, "_read_crontab", lambda: crontab)

    scope = daemon_module.resolve_service_scope(_config(tmp_path, tmp_path / "scratch"))

    assert scope.other_service == daemon_module.InstalledService(
        SchedulerKind.CRON, tmp_path / "cron data"
    )


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


def test_soft_stop_counts_an_exited_unreaped_daemon_as_stopped(tmp_path) -> None:
    config = _config(tmp_path, tmp_path / "data")
    config.data_dir.mkdir(parents=True)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        # Wait for the exit without reaping it, so the pid stays a zombie.
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        (config.data_dir / "recall.pid").write_text(f"{child.pid}\n", encoding="utf-8")

        result = stop_daemon_soft(config)

        assert result.stopped is True
        assert result.duration_seconds < 5
        assert not (config.data_dir / "recall.pid").exists()
    finally:
        child.wait(timeout=5)
