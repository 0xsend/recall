"""Manager records for scheduler tests whose fixtures contain installed artifacts.

The fake models a manager loading the fixture files. It never forwards a scheduler
process to the host. A supplied lifecycle fake still owns activation and PID state.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest
from recall.services import daemon

_scheduler_root: Path | None = None
_scheduler_guard_installed = False


def isolate_scheduler_account(root: Path) -> None:
    """Keep Python children in the same manager and account namespace as pytest."""
    from recall.services import service_records

    global _scheduler_root, _scheduler_guard_installed
    _scheduler_root = root

    def account_home() -> Path:
        home = Path.home().resolve()
        if not home.is_relative_to(root):
            raise RuntimeError(f"test account home escapes scheduler namespace: {home}")
        return home

    pytest.MonkeyPatch().setattr(service_records, "account_home", account_home)
    find_executable = shutil.which

    def guard(event: str, args: tuple[Any, ...]) -> None:
        namespace = _scheduler_root
        assert namespace is not None
        if event == "subprocess.Popen":
            executable = os.fsdecode(args[0])
            if Path(executable).name in {"systemctl", "launchctl", "crontab"}:
                env = args[3] or os.environ
                resolved = Path(
                    find_executable(executable, path=env.get("PATH", os.defpath)) or executable
                ).resolve()
                if not resolved.is_relative_to(namespace):
                    raise RuntimeError(f"test isolation forbids real scheduler: {resolved}")
        elif event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            if path.name in {
                "it.send.recall.daemon.plist",
                "xyz.metalrodeo.recall.daemon.plist",
            } and (
                not path.absolute().is_relative_to(namespace)
                or not path.resolve().is_relative_to(namespace)
            ):
                raise RuntimeError(f"test isolation forbids external service artifact: {path}")

    if not _scheduler_guard_installed:
        sys.addaudithook(guard)
        _scheduler_guard_installed = True


@dataclass(frozen=True)
class SchedulerPlatform:
    """One modeled platform shared by the test and its CLI children."""

    name: Literal["linux", "darwin"]

    @property
    def environ(self) -> dict[str, str]:
        return {"FAKE_PLATFORM": self.name}

    @staticmethod
    def cli_setup() -> str:
        # Run after imports: imported native libraries still belong to the host.
        return 'sys.platform = os.environ["FAKE_PLATFORM"]\n'


class ArtifactScheduler:
    def __init__(self) -> None:
        self.run = subprocess.run
        self.lifecycle_query = daemon._run_lifecycle_query

    def _state(self, args: list[str]) -> subprocess.CompletedProcess[str] | None:
        if daemon._run_lifecycle_query is not self.lifecycle_query:
            return daemon._run_lifecycle_query(args)
        if subprocess.run is not self.run:
            return subprocess.run(args, check=False, capture_output=True, text=True, timeout=5)
        return None

    def query(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        home = Path.home()

        def result(
            stdout: str = "", returncode: int = 0, stderr: str = ""
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args, returncode, stdout, stderr)

        if args == ["systemctl", "--user", "show-environment"]:
            return result(f"HOME={shlex.quote(str(home))}\n")
        if args[:3] == ["systemctl", "--user", "show"]:
            if any(
                arg in {"--property=Result,ExecMainStatus", "--property=Result"} for arg in args
            ):
                answer = self._state(args)
                return (
                    answer if answer is not None else result("Result=success\nExecMainStatus=0\n")
                )
            unit = args[3]
            path = home / ".config/systemd/user" / unit
            if not path.exists():
                return result("LoadState=not-found\n")
            if unit.endswith("timer"):
                return result(
                    f"LoadState=loaded\nFragmentPath={path}\nTriggers=recall-daemon.service\nNeedDaemonReload=no\n"
                )
            environment: dict[str, str] = {}
            kind = "oneshot"
            command = "/fixture/recall daemon --once"
            for file in [path, *sorted(path.with_name(path.name + ".d").glob("*.conf"))]:
                for line in file.read_text().splitlines():
                    key, _, value = line.partition("=")
                    if key == "Environment":
                        if not value:
                            environment.clear()
                        else:
                            for word in shlex.split(value):
                                key, setting = word.split("=", 1)
                                environment[key] = setting
                    elif key == "Type":
                        kind = value
                    elif key == "ExecStart":
                        command = value
            values = {
                "LoadState": "loaded",
                "FragmentPath": str(path),
                "Environment": shlex.join([f"{k}={v}" for k, v in environment.items()]),
                "EnvironmentFiles": "",
                "UnsetEnvironment": "",
                "Type": kind,
                "WorkingDirectory": str(Path.cwd()),
                "ExecStart": f"{{ argv[]={command} ; }}",
                "MainPID": "0",
                "ActiveState": "inactive",
                "NeedDaemonReload": "no",
            }
            return result("\n".join(f"{k}={v}" for k, v in values.items()) + "\n")
        if args[:3] == ["systemctl", "--user", "is-active"]:
            answer = self._state(args)
            return answer if answer is not None else result("inactive\n", 3)
        if args[:2] == ["launchctl", "print"]:
            if args[2].count("/") == 1:
                return result(f"{args[2]} = {{\n environment = {{\n HOME => {home}\n }}\n}}\n")
            label = args[2].rsplit("/", 1)[1]
            path = home / "Library/LaunchAgents" / f"{label}.plist"
            answer = self._state(args)
            if answer is not None and answer.returncode not in {0, 113, 3, 1}:
                return answer
            if (answer is not None and answer.returncode != 0) or (
                not path.exists() and (answer is None or not answer.stdout.strip())
            ):
                return result("", 113, f'Could not find service "{label}" in domain for user gui')
            value: dict[str, Any] = {}
            if path.exists():
                try:
                    value = plistlib.loads(path.read_bytes()) or {}
                except (ValueError, plistlib.InvalidFileException):
                    # A loaded job may retain a valid record after its file was edited.
                    value = {}
            argv = value.get(
                "ProgramArguments",
                ["/fixture/recall", "daemon"] + ([] if value.get("KeepAlive") else ["--once"]),
            )
            env = value.get("EnvironmentVariables", {})
            lines = [
                f"{args[2]} = {{",
                f" path = {path}",
                " arguments = {",
                *map(str, argv),
                " }",
                " environment = {",
            ]
            lines.extend(f" {key} => {value}" for key, value in env.items())
            lines.extend([" }", f" working directory = {Path.cwd()}"])
            if answer is not None:
                lines.extend(
                    line for line in answer.stdout.splitlines() if line.strip().startswith("pid = ")
                )
            lines.append("}")
            return result("\n".join(lines))
        if args[:2] == ["launchctl", "list"] or args[0] == "ps":
            answer = self._state(args)
            return answer if answer is not None else result()
        if args == ["crontab", "-l"]:
            answer = self._state(args)
            return answer if answer is not None else result("", 1, "no crontab for test-account")
        raise AssertionError(f"unexpected manager query: {args}")
