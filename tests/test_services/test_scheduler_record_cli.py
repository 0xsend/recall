"""Scheduler identity regressions at the executable CLI boundary (REQ-DAEMON-083)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

FAKE = r"""
import json, os, pathlib, plistlib, shlex, sys
state_path = pathlib.Path(os.environ["FAKE_STATE"])
state = json.loads(state_path.read_text())
args = sys.argv[1:]
name = pathlib.Path(sys.argv[0]).name
with pathlib.Path(os.environ["FAKE_CALLS"]).open("a") as log:
    log.write(json.dumps([name, *args]) + "\n")
if state.get("unreachable"):
    print("scheduler unreachable", file=sys.stderr)
    sys.exit(1)
account = pathlib.Path(state["account"])
kind = state["kind"]
label = "it.send.recall.daemon"
legacy = "xyz.metalrodeo.recall.daemon"
data = state["data"]
if name == "systemctl":
    if args == ["--user", "show-environment"]:
        print("HOME=" + str(account))
    elif args[:2] == ["--user", "show"]:
        unit = args[2]
        if unit.endswith("timer"):
            timer_path = account / ".config/systemd/user/recall-daemon.timer"
            default_target = "recall-daemon.service" if timer_path.exists() else None
            target = state.get("timer_target") or default_target
            if target:
                print("LoadState=loaded\nFragmentPath=" + str(timer_path))
                print("Triggers=" + target + "\nNeedDaemonReload=no")
            else:
                print("LoadState=not-found")
        else:
            path = account / ".config/systemd/user/recall-daemon.service"
            if not path.exists():
                print("LoadState=not-found")
                sys.exit(0)
            if state.get("read_artifacts"):
                for line in path.read_text().splitlines():
                    if line.startswith("Environment=RECALL_DATA_DIR="):
                        data = shlex.split(line[len("Environment="):])[0].split("=", 1)[1]
            print("LoadState=loaded\nFragmentPath=" + str(path))
            assignment = shlex.quote("RECALL_DATA_DIR=" + data) if data is not None else ""
            print("Environment=" + assignment)
            if state.get("environment_files"):
                print("EnvironmentFiles=" + state["environment_files"])
            print("UnsetEnvironment=\nType=oneshot")
            print("NeedDaemonReload=" + state.get("needs_reload", "no"))
            print("WorkingDirectory=!" + str(account))
            print("ExecStart={ path=/fixture/recall ; argv[]=/fixture/recall daemon --once ; }")
            print("MainPID=0\nActiveState=" + ("active" if state["loaded"] else "inactive"))
    elif args[:2] == ["--user", "is-active"]:
        sys.exit(0 if state["loaded"] else 3)
    elif args[:2] == ["--user", "stop"]:
        state["loaded"] = False
    elif args[:2] in (["--user", "start"], ["--user", "restart"]):
        state["loaded"] = True
elif name == "launchctl":
    if args[0] == "print":
        target = args[1]
        if target.count("/") == 1:
            print(target + " = {\n environment = {\n HOME => " + str(account) + "\n }\n}")
        elif ((target.endswith("/" + legacy) and "legacy_data" not in state)
              or not state["loaded"]
              or (target.endswith("/" + label) and state.get("new_absent"))):
            missing = target.rsplit("/", 1)[1]
            print('Could not find service "' + missing + '" in domain for user gui',
                  file=sys.stderr)
            sys.exit(113)
        else:
            actual_label = target.rsplit("/", 1)[1]
            path = account / "Library/LaunchAgents" / (actual_label + ".plist")
            if actual_label == legacy:
                data = state["legacy_data"]
            print(target + " = {\n path = " + str(path) + "\n program = /fixture/recall")
            print(" arguments = {\n /fixture/recall\n daemon\n --once\n }")
            print(" environment = {")
            if data is not None:
                print(" RECALL_DATA_DIR => " + data)
            print(" }\n state = not running\n}")
    elif args[0] == "bootout":
        state["loaded"] = False
    elif args[0] == "bootstrap":
        state["loaded"] = True
        if state.get("read_artifacts"):
            value = plistlib.loads(pathlib.Path(args[2]).read_bytes())
            state["data"] = value.get("EnvironmentVariables", {}).get("RECALL_DATA_DIR")
    elif args[0] == "list":
        print('"LastExitStatus" = 0;')
        sys.exit(0 if state["loaded"] else 113)
elif name == "crontab":
    if args == ["-l"]:
        print(state["crontab"], end="")
    elif args == ["-"]:
        state["crontab"] = sys.stdin.read()
else:
    raise AssertionError("no real scheduler is ever forwarded to")
state_path.write_text(json.dumps(state))
"""

WRAPPER = r"""
import os, pathlib, pwd, shutil, sys
from recall.cli.app import app
def scheduler_guard(event, args):
    if event == "subprocess.Popen":
        executable = os.fsdecode(args[0])
        if pathlib.Path(executable).name in {"systemctl", "launchctl", "crontab"}:
            resolved = pathlib.Path(shutil.which(executable) or executable).resolve()
            expected = pathlib.Path(os.environ["FAKE_BIN"]).resolve()
            assert resolved.parent == expected, "real scheduler forbidden"
sys.addaudithook(scheduler_guard)
fields = list(pwd.getpwuid(os.getuid()))
fields[5] = os.environ["FAKE_ACCOUNT"]
account = pwd.struct_passwd(fields)
pwd.getpwuid = lambda uid: account
if os.environ["FAKE_KIND"] == "launchd":
    sys.platform = "darwin"
app()
"""


@dataclass
class Host:
    root: Path
    kind: str
    account: Path
    selected: Path
    state: Path
    calls: Path
    config: Path
    wrapper: Path
    bin: Path

    def invoke(
        self, action: str, *, redirected: bool, soft: bool = False
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ | {
            "HOME": str(self.root / "redirected" if redirected else self.account),
            "RECALL_DATA_DIR": str(self.selected),
            "RECALL_CONFIG_PATH": str(self.config),
            "FAKE_BIN": str(self.bin),
            "FAKE_STATE": str(self.state),
            "FAKE_CALLS": str(self.calls),
            "FAKE_ACCOUNT": str(self.account),
            "FAKE_KIND": self.kind,
            "PATH": os.pathsep.join(
                (str(self.bin), str(Path(sys.executable).parent), os.environ["PATH"])
            ),
        }
        for name in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
            env.pop(name, None)
        args = [sys.executable, str(self.wrapper), "daemon", action, "--json"]
        if action in {"stop", "start", "restart"}:
            args += ["--timeout", "0.2"]
        if action == "install":
            args += ["--scheduler", self.kind]
        if action == "uninstall":
            args += ["--yes"]
        if soft:
            args += ["--soft"]
        return subprocess.run(args, env=env, capture_output=True, text=True, timeout=20)

    def mutations(self) -> list[list[str]]:
        calls = (
            [json.loads(line) for line in self.calls.read_text().splitlines()]
            if self.calls.exists()
            else []
        )
        return [
            call
            for call in calls
            if (call[0] == "crontab" and call[1:] == ["-"])
            or any(
                word
                in {
                    "stop",
                    "start",
                    "restart",
                    "bootout",
                    "bootstrap",
                    "enable",
                    "disable",
                    "daemon-reload",
                }
                for word in call[1:3]
            )
        ]


@pytest.fixture(params=["systemd", "launchd", "cron"])
def scheduler_host(request, tmp_path: Path) -> Host:
    import plistlib

    kind = request.param
    account = tmp_path / "account"
    selected = tmp_path / "selected"
    selected.mkdir()
    state = tmp_path / "state.json"
    calls = tmp_path / "calls.jsonl"
    config = tmp_path / "config.toml"
    config.write_text(f'[daemon]\nscheduler = "{kind}"\nmode = "poll"\n')
    unit = account / ".config/systemd/user/recall-daemon.service"
    if kind == "systemd":
        unit.parent.mkdir(parents=True)
        unit.write_text("[Service]\nType=oneshot\n")
    plist = account / "Library/LaunchAgents/it.send.recall.daemon.plist"
    plist.parent.mkdir(parents=True)
    if kind == "launchd":
        plist.write_bytes(
            plistlib.dumps(
                {
                    "Label": "it.send.recall.daemon",
                    "ProgramArguments": ["/fixture/recall", "daemon", "--once"],
                    "StartInterval": 300,
                }
            )
        )
    table = "# BEGIN recall daemon\n* * * * * /fixture/recall daemon --once\n# END recall daemon\n"
    state.write_text(
        json.dumps(
            {
                "account": str(account),
                "kind": kind,
                "data": None,
                "loaded": True,
                "crontab": table if kind == "cron" else "",
            }
        )
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in ("systemctl", "launchctl", "crontab"):
        path = binary / name
        path.write_text(f"#!{sys.executable}\n" + FAKE)
        path.chmod(0o755)
    wrapper = tmp_path / "cli.py"
    wrapper.write_text(WRAPPER)
    return Host(tmp_path, kind, account, selected, state, calls, config, wrapper, binary)


@pytest.mark.parametrize("redirected", [False, True])
@pytest.mark.parametrize("action", ["stop", "start", "restart", "install", "uninstall"])
def test_req_daemon_083_foreign_default_service_refuses(
    scheduler_host: Host, redirected: bool, action: str
) -> None:
    host = scheduler_host
    result = host.invoke(action, redirected=redirected)
    assert result.returncode == 1, (result.stdout, result.stderr)
    payload = json.loads(result.stdout)
    message = payload["error"]["message"] if "error" in payload else payload["message"]
    assert "refusing to " + action in message
    assert str(host.account / ".local/share/recall") in message
    assert str(host.selected) in message
    assert host.mutations() == []


def test_req_daemon_083_matching_default_still_stops(scheduler_host: Host) -> None:
    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    result = host.invoke("stop", redirected=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert json.loads(result.stdout)["stopped"] is True
    if host.kind != "cron":
        assert len(host.mutations()) == 1


@pytest.mark.parametrize("action", ["stop", "start", "restart", "install", "uninstall"])
def test_req_daemon_083_unreachable_refuses(scheduler_host: Host, action: str) -> None:
    host = scheduler_host
    state = json.loads(host.state.read_text())
    state["unreachable"] = True
    host.state.write_text(json.dumps(state))
    result = host.invoke(action, redirected=True)
    assert result.returncode == 1, (result.stdout, result.stderr)
    payload = json.loads(result.stdout)
    message = payload["error"]["message"] if "error" in payload else payload["message"]
    assert "scheduler" in message
    assert host.mutations() == []


def test_req_daemon_083_soft_stop_has_no_scheduler_query(scheduler_host: Host) -> None:
    result = scheduler_host.invoke("stop", redirected=True, soft=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert json.loads(result.stdout)["scheduler"] == "sentinel"
    assert not scheduler_host.calls.exists()


@pytest.mark.parametrize("unreachable", [False, True])
def test_start_with_a_live_pidfile_requires_matching_readable_scope(
    scheduler_host: Host, unreachable: bool
) -> None:
    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    host.selected.mkdir(parents=True, exist_ok=True)
    (host.selected / "recall.pid").write_text(str(os.getpid()))
    state = json.loads(host.state.read_text())
    state["unreachable"] = unreachable
    host.state.write_text(json.dumps(state))
    result = host.invoke("start", redirected=True)
    payload = json.loads(result.stdout)
    if unreachable:
        assert result.returncode == 1, (result.stdout, result.stderr)
        assert payload["started"] is False
        assert "cannot identify its data dir" in payload["message"]
    else:
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert payload["started"] is True
        assert payload["pid"] == os.getpid()
        assert payload["message"] == "daemon already running"
    assert host.mutations() == []


@pytest.mark.parametrize("action", ["start", "restart", "install", "uninstall"])
def test_matching_service_keeps_each_action(scheduler_host: Host, action: str) -> None:
    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    result = host.invoke(action, redirected=True)
    expected = 1 if host.kind == "cron" and action in {"start", "restart"} else 0
    assert result.returncode == expected, (result.stdout, result.stderr)
    payload = json.loads(result.stdout)
    assert "error" not in payload
    if action in {"start", "restart"}:
        assert payload["started"] is (expected == 0)
    else:
        assert payload["installed"] is (action == "install")


@pytest.mark.parametrize("scheduler_host", ["launchd"], indirect=True)
def test_stopped_launchd_uses_account_plist(scheduler_host: Host) -> None:
    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    state = json.loads(host.state.read_text())
    state["loaded"] = False
    host.state.write_text(json.dumps(state))
    result = host.invoke("start", redirected=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert host.mutations() == [
        [
            "launchctl",
            "bootstrap",
            f"gui/{os.getuid()}",
            str(host.account / "Library/LaunchAgents/it.send.recall.daemon.plist"),
        ]
    ]


def test_foreign_secondary_target_refuses(scheduler_host: Host) -> None:
    host = scheduler_host
    state = json.loads(host.state.read_text())
    host.selected = host.account / ".local/share/recall"
    if host.kind == "systemd":
        state["timer_target"] = "foreign.service"
    elif host.kind == "launchd":
        state["legacy_data"] = str(host.root / "other")
    else:
        state["crontab"] = state["crontab"].replace(
            "/fixture/recall", "RECALL_DATA_DIR=" + str(host.root / "other") + " /fixture/recall"
        )
    host.state.write_text(json.dumps(state))
    result = host.invoke("stop", redirected=True)
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert host.mutations() == []


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
def test_unknown_artifact_environment_refuses(scheduler_host: Host) -> None:
    host = scheduler_host
    state = json.loads(host.state.read_text())
    state["environment_files"] = "/unknown/environment (ignore_errors=no)"
    host.state.write_text(json.dumps(state))
    result = host.invoke("stop", redirected=True)
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert "EnvironmentFile" in json.loads(result.stdout)["message"]
    assert host.mutations() == []


def test_initial_install_uses_account_destination(scheduler_host: Host) -> None:
    host = scheduler_host
    state = json.loads(host.state.read_text())
    state.update(absent=True, loaded=False, read_artifacts=True, crontab="")
    for path in host.account.rglob("recall-daemon.service"):
        path.unlink()
    for path in host.account.rglob("it.send.recall.daemon.plist"):
        path.unlink()
    host.state.write_text(json.dumps(state))
    result = host.invoke("install", redirected=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert json.loads(result.stdout)["installed"] is True
    if host.kind == "launchd":
        assert (host.account / "Library/LaunchAgents/it.send.recall.daemon.plist").exists()
    elif host.kind == "systemd":
        assert (host.account / ".config/systemd/user/recall-daemon.service").exists()
    else:
        assert str(host.selected) in json.loads(host.state.read_text())["crontab"]
    assert not (host.root / "redirected/.config/systemd/user").exists()
    assert not (host.root / "redirected/Library/LaunchAgents").exists()


@pytest.mark.parametrize("scheduler_host", ["launchd"], indirect=True)
@pytest.mark.parametrize("action", ["restart", "install"])
def test_loaded_job_does_not_bootstrap_an_edited_foreign_plist(
    scheduler_host: Host, action: str
) -> None:
    import plistlib

    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    path = host.account / "Library/LaunchAgents/it.send.recall.daemon.plist"
    value = plistlib.loads(path.read_bytes())
    value["EnvironmentVariables"] = {"RECALL_DATA_DIR": str(host.root / "foreign")}
    path.write_bytes(plistlib.dumps(value))
    result = host.invoke(action, redirected=True)
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert "another data dir" in result.stdout
    assert host.mutations() == []


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
def test_install_does_not_restore_an_unloaded_unit_edit(scheduler_host: Host) -> None:
    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    state = json.loads(host.state.read_text())
    state["needs_reload"] = "yes"
    host.state.write_text(json.dumps(state))
    result = host.invoke("install", redirected=True)
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert "requires daemon-reload" in result.stdout
    assert host.mutations() == []


@pytest.mark.parametrize("scheduler_host", ["launchd"], indirect=True)
def test_start_prefers_a_loaded_legacy_job_over_a_stopped_new_plist(scheduler_host: Host) -> None:
    import plistlib

    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    state = json.loads(host.state.read_text())
    state.update(new_absent=True, legacy_data=str(host.selected))
    host.state.write_text(json.dumps(state))
    path = host.account / "Library/LaunchAgents/xyz.metalrodeo.recall.daemon.plist"
    path.write_bytes(
        plistlib.dumps(
            {
                "Label": path.stem,
                "ProgramArguments": ["/fixture/recall", "daemon", "--once"],
                "StartInterval": 300,
            }
        )
    )
    result = host.invoke("start", redirected=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert json.loads(result.stdout)["message"] == "already loaded"
    assert host.mutations() == []


@pytest.mark.parametrize("scheduler_host", ["launchd"], indirect=True)
def test_status_names_the_foreign_secondary_data_dir(scheduler_host: Host) -> None:
    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    state = json.loads(host.state.read_text())
    state["legacy_data"] = str(host.root / "foreign")
    host.state.write_text(json.dumps(state))
    result = host.invoke("status", redirected=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    payload = json.loads(result.stdout)
    assert payload["installed"] is False
    assert payload["service_data_dir"] == str(host.root / "foreign")
    assert host.mutations() == []


def test_initial_install_anchors_relative_data_paths_before_loading(
    scheduler_host: Host, monkeypatch
) -> None:
    import plistlib
    import shlex

    host = scheduler_host
    monkeypatch.chdir(host.root)
    host.selected = Path("selected")
    state = json.loads(host.state.read_text())
    state.update(absent=True, loaded=False, read_artifacts=True, crontab="")
    for path in [
        host.account / ".config/systemd/user/recall-daemon.service",
        host.account / "Library/LaunchAgents/it.send.recall.daemon.plist",
    ]:
        path.unlink(missing_ok=True)
    host.state.write_text(json.dumps(state))
    result = host.invoke("install", redirected=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert json.loads(result.stdout)["installed"] is True
    expected = str(host.root / "selected")
    if host.kind == "launchd":
        value = plistlib.loads(
            (host.account / "Library/LaunchAgents/it.send.recall.daemon.plist").read_bytes()
        )
        assert value["EnvironmentVariables"]["RECALL_DATA_DIR"] == expected
    elif host.kind == "systemd":
        content = (host.account / ".config/systemd/user/recall-daemon.service").read_text()
        assignment = next(
            line for line in content.splitlines() if line.startswith("Environment=RECALL_DATA_DIR=")
        )
        assert shlex.split(assignment[len("Environment=") :]) == ["RECALL_DATA_DIR=" + expected]
    else:
        assert "RECALL_DATA_DIR=" + expected in json.loads(host.state.read_text())["crontab"]
