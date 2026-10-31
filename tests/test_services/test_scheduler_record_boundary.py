"""Unreadable scheduler responses never create service authority (REQ-DAEMON-083)."""

from __future__ import annotations

import signal
import subprocess
import sys

import pytest
from recall.core.config import AppConfig
from recall.services import daemon, service_records


def test_scheduler_query_caps_stdout_before_returning(scheduler_namespace) -> None:
    with pytest.raises(ValueError, match="exceeds 1 MiB"):
        scheduler_namespace(
            [sys.executable, "-c", "import os; os.write(1, b'x' * (1024 * 1024 + 1))"]
        )


def test_scheduler_query_timeout_reaps_its_child(scheduler_namespace, monkeypatch) -> None:
    children: list[subprocess.Popen[bytes]] = []

    class RecordedPopen(subprocess.Popen):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            children.append(self)

    monkeypatch.setattr(service_records.subprocess, "Popen", RecordedPopen)
    # The child never answers, so the bound expires however slowly the host runs it.
    silent = [sys.executable, "-c", "import time; time.sleep(3600)"]
    with pytest.raises(TimeoutError, match="timed out"):
        scheduler_namespace(silent, timeout_seconds=0.05)
    assert len(children) == 1
    assert children[0].returncode == -signal.SIGKILL


def test_missing_crontab_does_not_block_an_absent_systemd_action(
    monkeypatch, scheduler_platform
) -> None:
    scheduler_platform("linux")
    query = service_records.query
    calls = []

    def record(args):
        calls.append(args)
        return query(args)

    monkeypatch.setattr(service_records, "query", record)
    monkeypatch.setattr(daemon.shutil, "which", lambda name: None)
    result = daemon.stop_daemon_durable(AppConfig.load())
    assert result.stopped is True
    assert result.scheduler == "systemd"
    assert not any(call[0] == "crontab" for call in calls)


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        ('"RECALL_DATA_DIR=/data with spaces"', "/data with spaces"),
        (r'"RECALL_DATA_DIR=/data\nline"', "/data\nline"),
        (r'"RECALL_DATA_DIR=/data\\nline"', r"/data\nline"),
        (r'"RECALL_DATA_DIR=/data\$value"', "/data$value"),
        (r'"RECALL_DATA_DIR=/data\x09tab"', "/data\ttab"),
        (r'"RECALL_DATA_DIR=/data\011tab"', "/data\ttab"),
        ("'RECALL_DATA_DIR=/literal\\n'", r"/literal\n"),
    ],
)
def test_systemd_environment_preserves_the_scheduler_value(encoded, expected) -> None:
    assert service_records._environment_assignments(encoded) == {"RECALL_DATA_DIR": expected}


@pytest.mark.parametrize("working_directory", ["", "~", "absolute"])
@pytest.mark.parametrize("optional", [False, True])
def test_systemd_default_working_directory_is_account_home(
    tmp_path, monkeypatch, working_directory, optional
) -> None:
    import subprocess

    from recall.core.types import SchedulerKind

    account = tmp_path / "account"
    account.mkdir()
    manager_home = tmp_path / "manager-home"
    monkeypatch.setattr(service_records, "account_home", lambda: account)
    cwd = str(account) if working_directory == "absolute" else working_directory
    if optional:
        cwd = "!" + cwd

    def query(args):
        if args == ["systemctl", "--user", "show-environment"]:
            return subprocess.CompletedProcess(args, 0, f"HOME={manager_home}\n", "")
        if args[3].endswith("timer"):
            return subprocess.CompletedProcess(args, 0, "LoadState=not-found\n", "")
        values = {
            "LoadState": "loaded",
            "FragmentPath": str(account / "recall-daemon.service"),
            "Environment": "RECALL_DATA_DIR=relative",
            "UnsetEnvironment": "",
            "Type": "simple",
            "WorkingDirectory": cwd,
            "ExecStart": "{ path=/bin/recall ; argv[]=/bin/recall daemon ; }",
            "MainPID": "0",
            "ExecMainStartTimestampMonotonic": "0",
            "NotifyAccess": "none",
            "ActiveState": "inactive",
            "NeedDaemonReload": "no",
        }
        return subprocess.CompletedProcess(
            args, 0, "\n".join(f"{key}={value}" for key, value in values.items()), ""
        )

    record = service_records.discover(SchedulerKind.SYSTEMD, run=query)
    assert record.service is not None
    assert record.service.effective_data_dir == account / "relative"
