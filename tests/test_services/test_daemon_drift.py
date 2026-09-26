from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import recall.services.daemon as daemon_module
from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.types import DaemonMode
from recall.services.daemon import daemon_status


class CompletedProcess:
    def __init__(self, args: list[str], returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.args = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _app_config(tmp_path: Path, *, mode: DaemonMode = DaemonMode.POLL) -> AppConfig:
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


LAUNCHD_STALE_PLIST = """\
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
</dict>
</plist>
"""


SYSTEMD_STALE_SERVICE = """\
[Unit]
Description=recall daemon

[Service]
Type=oneshot
ExecStart=/usr/local/bin/recall daemon --once
"""


# `launchctl list <label>` answers with a plist-style dictionary; only the bare
# `launchctl list` prints the tabular PID/Status/Label listing. The running
# sample is transcribed from Darwin 25.6 with the recall agent loaded; the
# failed one is the same shape without the `PID` key launchd omits while the
# job is not running.
LAUNCHCTL_LIST_EXITED_78 = """\
{
\t"StandardOutPath" = "/tmp/recall/logs/daemon.log";
\t"Label" = "it.send.recall.daemon";
\t"OnDemand" = true;
\t"LastExitStatus" = 78;
\t"Program" = "/usr/local/bin/recall";
};
"""

LAUNCHCTL_LIST_RUNNING = """\
{
\t"StandardOutPath" = "/tmp/recall/logs/daemon.log";
\t"Label" = "it.send.recall.daemon";
\t"OnDemand" = true;
\t"LastExitStatus" = 0;
\t"PID" = 65965;
\t"Program" = "/usr/local/bin/recall";
};
"""


def test_daemon_status_reports_launchd_binary_drift_and_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-035/036: launchd status exposes stale binary and exit status."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    plist_path = tmp_path / "Library" / "LaunchAgents" / "it.send.recall.daemon.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(LAUNCHD_STALE_PLIST, encoding="utf-8")

    current_binary = tmp_path / ".local" / "bin" / "recall"
    current_binary.parent.mkdir(parents=True, exist_ok=True)
    current_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        if args[:3] == ["launchctl", "list", "it.send.recall.daemon"]:
            return CompletedProcess(args, stdout=LAUNCHCTL_LIST_EXITED_78)
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(current_binary))
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    status = daemon_status(config=config)

    assert status.installed is True
    assert status.installed_binary_path == "/usr/local/bin/recall"
    assert status.installed_binary_stale is True
    assert status.scheduler_last_exit_status == 78
    assert status.scheduler_health_state == "failed"


def test_daemon_status_reports_launchd_health_for_a_running_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-036: a loaded, never-failed launchd job reads as exit 0 / ok."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    plist_path = tmp_path / "Library" / "LaunchAgents" / "it.send.recall.daemon.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(LAUNCHD_STALE_PLIST, encoding="utf-8")

    current_binary = tmp_path / ".local" / "bin" / "recall"
    current_binary.parent.mkdir(parents=True, exist_ok=True)
    current_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        if args[:3] == ["launchctl", "list", "it.send.recall.daemon"]:
            return CompletedProcess(args, stdout=LAUNCHCTL_LIST_RUNNING)
        if args[:2] == ["launchctl", "print"]:
            # A running job reports "last exit code = (never exited)", which is
            # why the list parse has to carry the answer.
            return CompletedProcess(args, stdout="\tlast exit code = (never exited)\n")
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(current_binary))
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    status = daemon_status(config=config)

    assert status.scheduler_last_exit_status == 0
    assert status.scheduler_health_state == "ok"


def test_daemon_status_reports_systemd_binary_drift_and_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-035/036: systemd status exposes stale binary and Result state."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = replace(_app_config(tmp_path), daemon=DaemonConfig(interval=300, mode=DaemonMode.POLL))
    base = tmp_path / ".config" / "systemd" / "user"
    base.mkdir(parents=True, exist_ok=True)
    (base / "recall-daemon.service").write_text(SYSTEMD_STALE_SERVICE, encoding="utf-8")
    (base / "recall-daemon.timer").write_text("[Timer]\nOnUnitActiveSec=300\n", encoding="utf-8")

    current_binary = tmp_path / ".local" / "bin" / "recall"
    current_binary.parent.mkdir(parents=True, exist_ok=True)
    current_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        if args[:4] == ["systemctl", "--user", "show", "recall-daemon.service"]:
            return CompletedProcess(args, stdout="Result=exit-code\nExecMainStatus=78\n")
        if args[:4] == ["systemctl", "--user", "show", "recall-daemon.timer"]:
            return CompletedProcess(args, stdout="Result=success\n")
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(daemon_module, "_systemd_user_available", lambda: True)
    monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(current_binary))
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    status = daemon_status(config=config)

    assert status.installed is True
    assert status.installed_binary_path == "/usr/local/bin/recall"
    assert status.installed_binary_stale is True
    assert status.scheduler_last_exit_status == 78
    assert status.scheduler_health_state == "exit-code"


def test_daemon_status_reports_unknown_launchd_health_on_unparseable_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-036: unparseable launchctl output degrades to unknown health."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    plist_path = tmp_path / "Library" / "LaunchAgents" / "it.send.recall.daemon.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(LAUNCHD_STALE_PLIST, encoding="utf-8")

    current_binary = tmp_path / ".local" / "bin" / "recall"
    current_binary.parent.mkdir(parents=True, exist_ok=True)
    current_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        if args[:3] == ["launchctl", "list", "it.send.recall.daemon"]:
            return CompletedProcess(args, stdout="nonsense\n")
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(current_binary))
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    status = daemon_status(config=config)

    assert status.scheduler_last_exit_status is None
    assert status.scheduler_health_state == "unknown"


def test_daemon_status_survives_a_missing_launchctl_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-036: a missing launchctl is unknown health, not a status crash."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    plist_path = tmp_path / "Library" / "LaunchAgents" / "it.send.recall.daemon.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(LAUNCHD_STALE_PLIST, encoding="utf-8")

    current_binary = tmp_path / ".local" / "bin" / "recall"
    current_binary.parent.mkdir(parents=True, exist_ok=True)
    current_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        raise FileNotFoundError(2, "No such file or directory", args[0])

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(current_binary))
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    status = daemon_status(config=config)

    assert status.installed is True
    assert status.scheduler_last_exit_status is None
    assert status.scheduler_health_state is None
