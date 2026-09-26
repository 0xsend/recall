from __future__ import annotations

import functools
import subprocess
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
from recall.core.types import DaemonMode, SchedulerKind
from recall.services.daemon import install_scheduler

pytestmark = pytest.mark.usefixtures("launchd_without_legacy_job")


@pytest.fixture
def _launchd_unload_wait_without_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Faked `launchctl print` always answers "loaded", so the real 30s unload
    deadline would expire on every launchd install. Keep the real probe but
    give it no deadline: the outcome under these fakes is identical."""
    monkeypatch.setattr(
        daemon_module,
        "_wait_for_launchd_unload",
        functools.partial(daemon_module._wait_for_launchd_unload, timeout_seconds=0.0),
    )


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


def test_install_scheduler_launchd_rejects_missing_binary_without_writing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-033: missing launchd binary fails before any plist write."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    plist_path = tmp_path / "Library" / "LaunchAgents" / "it.send.recall.daemon.plist"
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
        daemon_module, "_resolve_recall_binary", lambda: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="does not exist"):
        install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    assert not plist_path.exists()
    assert commands == []


@pytest.mark.usefixtures("_launchd_unload_wait_without_deadline")
def test_install_scheduler_launchd_smoke_test_failure_rolls_back_artifact_and_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-034: failed post-install smoke test restores the previous plist."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    binary_path = tmp_path / "bin" / "recall"
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    binary_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    plist_path = tmp_path / "Library" / "LaunchAgents" / "it.send.recall.daemon.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    previous_contents = "<plist>old</plist>\n"
    plist_path.write_text(previous_contents, encoding="utf-8")

    commands: list[list[str]] = []
    # The previous install is running; bootout unloads it and bootstrap loads.
    loaded = [True]

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        commands.append(args)
        if args == [str(binary_path), "--help"]:
            raise subprocess.CalledProcessError(returncode=1, cmd=args, stderr="boom")
        verb = args[1] if args[0] == "launchctl" else None
        if verb == "bootout":
            loaded[0] = False
        elif verb == "bootstrap":
            loaded[0] = True
        elif verb in {"print", "list"}:
            return CompletedProcess(args, returncode=0 if loaded[0] else 113)
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(binary_path))
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)

    assert plist_path.read_text(encoding="utf-8") == previous_contents
    # The restored install is loaded again, not left stopped.
    assert loaded == [True]
    assert commands[-1] == [
        "launchctl",
        "bootstrap",
        f"gui/{daemon_module.os.getuid()}",
        str(plist_path),
    ]
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
    assert [str(binary_path), "--help"] in commands
    bootouts = [command for command in commands if command[:2] == ["launchctl", "bootout"]]
    assert len(bootouts) >= 2


class FakeLaunchd:
    """Simulates launchd's asynchronous bootout.

    `launchctl bootout` returns before teardown finishes, so for a bounded
    number of subsequent calls the service is still present: `print` answers 0
    and `bootstrap` refuses with EBUSY. Once the teardown drains, the service is
    genuinely gone and `bootstrap` is accepted.
    """

    def __init__(self, *, teardown_calls: int) -> None:
        self.loaded = True
        self._teardown_remaining = 0
        self._teardown_calls = teardown_calls
        self.commands: list[list[str]] = []

    def _drain(self) -> None:
        if self._teardown_remaining > 0:
            self._teardown_remaining -= 1
            if self._teardown_remaining == 0:
                self.loaded = False

    def settle(self) -> None:
        """Complete any teardown still pending, as launchd does moments later."""
        if self._teardown_remaining > 0:
            self._teardown_remaining = 0
            self.loaded = False

    def run(
        self,
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        self.commands.append(args)
        head = args[:2]
        if head == ["launchctl", "bootout"]:
            if not self.loaded:
                if check:
                    raise subprocess.CalledProcessError(
                        returncode=3, cmd=args, stderr="Boot-out failed: 3: No such process\n"
                    )
                return CompletedProcess(
                    args, returncode=3, stderr="Boot-out failed: 3: No such process\n"
                )
            self._teardown_remaining = self._teardown_calls
            return CompletedProcess(args)
        if head == ["launchctl", "bootstrap"]:
            self._drain()
            if self.loaded:
                raise subprocess.CalledProcessError(
                    returncode=37, cmd=args, stderr="Bootstrap failed: 37: Operation in progress\n"
                )
            self.loaded = True
            return CompletedProcess(args)
        if head == ["launchctl", "print"]:
            self._drain()
            return CompletedProcess(args, returncode=0 if self.loaded else 1)
        if head == ["launchctl", "list"]:
            self._drain()
            return CompletedProcess(args, returncode=0 if self.loaded else 1)
        return CompletedProcess(args)


def test_install_scheduler_launchd_bootstraps_service_despite_async_bootout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-071: a draining bootout must not be mistaken for a live service.

    `bootstrap` fails with EBUSY while the previous instance tears down. Probing
    `launchctl print` at that moment answers 0 -- the service still exists, but
    only because it is dying. Treating that as "already running" leaves nothing
    loaded once the teardown drains, and reports success.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    binary_path = tmp_path / "bin" / "recall"
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    binary_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    launchd = FakeLaunchd(teardown_calls=3)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(binary_path))
    monkeypatch.setattr(daemon_module.subprocess, "run", launchd.run)

    install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)
    launchd.settle()

    assert launchd.loaded, "install reported success but left no service loaded in launchd"


def test_install_scheduler_launchd_fails_closed_when_service_never_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-072: an install that cannot load the service must raise."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    binary_path = tmp_path / "bin" / "recall"
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    binary_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int | None = None,
    ) -> CompletedProcess:
        if args[:2] == ["launchctl", "bootstrap"]:
            raise subprocess.CalledProcessError(
                returncode=5, cmd=args, stderr="Bootstrap failed: 5: Input/output error\n"
            )
        if args[:2] in (["launchctl", "print"], ["launchctl", "list"]):
            return CompletedProcess(args, returncode=1)
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(binary_path))
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)


def test_install_scheduler_systemd_rejects_missing_binary_without_writing_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-033: missing systemd binary fails before service/timer writes."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _app_config(tmp_path)
    base = tmp_path / ".config" / "systemd" / "user"
    service_path = base / "recall-daemon.service"
    timer_path = base / "recall-daemon.timer"
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
        daemon_module, "_resolve_recall_binary", lambda: str(tmp_path / "bin" / "recall")
    )
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="does not exist"):
        install_scheduler(scheduler=SchedulerKind.SYSTEMD, config=config)

    assert not service_path.exists()
    assert not timer_path.exists()
    assert commands == []


def test_install_scheduler_systemd_smoke_test_failure_rolls_back_files_and_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-034: failed smoke test restores previous systemd units."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = replace(_app_config(tmp_path), daemon=DaemonConfig(interval=300, mode=DaemonMode.POLL))
    binary_path = tmp_path / "bin" / "recall"
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    binary_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    base = tmp_path / ".config" / "systemd" / "user"
    base.mkdir(parents=True, exist_ok=True)
    service_path = base / "recall-daemon.service"
    timer_path = base / "recall-daemon.timer"
    previous_service = "[Unit]\nDescription=old service\n"
    previous_timer = "[Timer]\nOnUnitActiveSec=60\n"
    service_path.write_text(previous_service, encoding="utf-8")
    timer_path.write_text(previous_timer, encoding="utf-8")

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
        if args == [str(binary_path), "--help"]:
            raise subprocess.CalledProcessError(returncode=2, cmd=args, stderr="smoke failed")
        return CompletedProcess(args)

    monkeypatch.setattr(daemon_module.sys, "platform", "linux")
    monkeypatch.setattr(daemon_module, "_systemd_user_available", lambda: True)
    monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(binary_path))
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        install_scheduler(scheduler=SchedulerKind.SYSTEMD, config=config)

    assert service_path.read_text(encoding="utf-8") == previous_service
    assert timer_path.read_text(encoding="utf-8") == previous_timer
    assert ["systemctl", "--user", "enable", "--now", "recall-daemon.timer"] in commands
    assert [str(binary_path), "--help"] in commands
    assert ["systemctl", "--user", "disable", "--now", "recall-daemon.service"] in commands
    assert ["systemctl", "--user", "disable", "--now", "recall-daemon.timer"] in commands
    daemon_reload_count = commands.count(["systemctl", "--user", "daemon-reload"])
    assert daemon_reload_count >= 2
