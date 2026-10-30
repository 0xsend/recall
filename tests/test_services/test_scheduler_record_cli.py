"""Scheduler identity regressions at the executable CLI boundary (REQ-DAEMON-083)."""

from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from tests.scheduler_fakes import SchedulerPlatform

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
            if state.get("read_artifacts") and state.get("unreadable_after_reload"):
                print("Failed to get properties: Connection timed out", file=sys.stderr)
                sys.exit(1)
            path = account / ".config/systemd/user/recall-daemon.service"
            if not path.exists():
                print("LoadState=not-found")
                sys.exit(0)
            # A reloaded manager reports what the unit file says; otherwise the state.
            words = [] if data is None else ["RECALL_DATA_DIR=" + data]
            fragment_files = []
            if state.get("read_artifacts"):
                words = []
                for line in path.read_text().splitlines():
                    if line.startswith("Environment="):
                        words += shlex.split(line[len("Environment="):])
                    elif line.startswith("EnvironmentFile="):
                        file = line[len("EnvironmentFile="):]
                        optional = "yes" if file.startswith("-") else "no"
                        entry = file.lstrip("-") + " (ignore_errors=" + optional + ")"
                        fragment_files.append(entry)
            words += state.get("unit_environment", [])
            print("LoadState=loaded\nFragmentPath=" + str(path))
            print("Environment=" + " ".join(shlex.quote(word) for word in words))
            for entry in fragment_files + state.get("environment_files", []):
                print("EnvironmentFiles=" + entry)
            print("UnsetEnvironment=" + state.get("unset", ""))
            print("Type=oneshot")
            print("NeedDaemonReload=" + state.get("needs_reload", "no"))
            print("WorkingDirectory=" + state.get("working_directory", "!" + str(account)))
            print("ExecStart={ path=/fixture/recall ; argv[]=/fixture/recall daemon --once ; }")
            print("MainPID=" + str(state.get("main_pid", 0)))
            print("ControlGroup=" + state.get("control_group", ""))
            print("ActiveState=" + ("active" if state["loaded"] else "inactive"))
    elif args[:2] == ["--user", "is-active"]:
        sys.exit(0 if state["loaded"] else 3)
    elif args[:2] == ["--user", "daemon-reload"]:
        state["read_artifacts"] = True
    elif args[:2] == ["--user", "stop"]:
        state["loaded"] = False
    elif args[:2] in (["--user", "start"], ["--user", "restart"]):
        state["loaded"] = True
elif name == "launchctl":
    if args[0] == "print":
        target = args[1]
        if target.count("/") == 1:
            print(target + " = {")
            print(state.get("domain_blocks", "environment = {\n HOME => " + str(account) + "\n }"))
            print("}")
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
            print(state.get("schedule", "run interval = 300 seconds"))
            if "job_blocks" in state:
                print(state["job_blocks"])
            else:
                print(" environment = {")
                if data is not None:
                    print(" RECALL_DATA_DIR => " + data)
                print(" }")
            print(" state = not running\n}")
    elif args[0] == "bootout":
        state["loaded"] = False
    elif args[0] == "bootstrap":
        state["loaded"] = True
        if "bootstrap_pid" in state:
            pathlib.Path(state["pid_path"]).write_text(str(state["bootstrap_pid"]))
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
    platform: SchedulerPlatform

    def environ(self, *, redirected: bool) -> dict[str, str]:
        env = os.environ | {
            "HOME": str(self.root / "redirected" if redirected else self.account),
            "RECALL_DATA_DIR": str(self.selected),
            "RECALL_CONFIG_PATH": str(self.config),
            "FAKE_BIN": str(self.bin),
            "FAKE_STATE": str(self.state),
            "FAKE_CALLS": str(self.calls),
            "FAKE_ACCOUNT": str(self.account),
            "FAKE_KIND": self.kind,
            **self.platform.environ,
            "PATH": os.pathsep.join(
                (str(self.bin), str(Path(sys.executable).parent), os.environ["PATH"])
            ),
        }
        for name in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
            env.pop(name, None)
        return env

    def invoke(
        self, action: str, *, redirected: bool, soft: bool = False
    ) -> subprocess.CompletedProcess[str]:
        env = self.environ(redirected=redirected)
        command = ["compact", "--yes"] if action == "compact" else ["daemon", action]
        args = [sys.executable, str(self.wrapper), *command, "--json"]
        if action in {"stop", "start", "restart"}:
            # Each manager query starts a Python fake; allow macOS spawn overhead.
            args += ["--timeout", "5"]
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
def scheduler_host(request, tmp_path: Path, scheduler_platform) -> Host:
    import plistlib

    kind = request.param
    platform = scheduler_platform("darwin" if kind == "launchd" else "linux")
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
    wrapper.write_text(WRAPPER + platform.cli_setup() + "app()\n")
    return Host(tmp_path, kind, account, selected, state, calls, config, wrapper, binary, platform)


@pytest.mark.parametrize("scheduler_host", ["systemd", "cron"], indirect=True)
def test_macos_install_refuses_unsupported_scheduler(
    scheduler_host: Host, scheduler_platform
) -> None:
    host = scheduler_host
    host.platform = scheduler_platform("darwin")
    result = host.invoke("install", redirected=False)
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert json.loads(result.stdout) == {
        "error": {
            "code": "VALIDATION",
            "message": f"scheduler {host.kind} is not supported on macOS",
        }
    }
    assert result.stderr == ""
    assert host.mutations() == []


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


def _prepare_launchd_action(host: Host, *, loaded: bool) -> Path:
    from recall.core.config import AppConfig
    from recall.db import connect

    host.selected = host.account / ".local/share/recall"
    host.selected.mkdir(parents=True, exist_ok=True)
    config = replace(
        AppConfig.load(),
        data_dir=host.selected,
        db_path=host.selected / "recall.duckdb",
        lock_path=host.selected / "recall.lock",
    )
    connect(config).close()
    state = json.loads(host.state.read_text())
    state["loaded"] = loaded
    host.state.write_text(json.dumps(state))
    return host.account / "Library/LaunchAgents/it.send.recall.daemon.plist"


@pytest.mark.parametrize("scheduler_host", ["launchd"], indirect=True)
@pytest.mark.parametrize("action", ["start", "install", "uninstall", "compact"])
@pytest.mark.parametrize(
    "schedule",
    [
        {},
        {"KeepAlive": "yes"},
        {"KeepAlive": 1},
        {"KeepAlive": {}},
        {"KeepAlive": {"AfterInitialDemand": True}},
        {"KeepAlive": {"PathState": {"/fixture/refused": "false"}}},
        {"KeepAlive": {"Unsupported": True}},
        {"StartInterval": "300"},
        {"StartInterval": True},
        {"StartInterval": 0},
        {"StartInterval": -1},
        {"KeepAlive": True, "StartInterval": 300},
        {"StartInterval": 300, "StartCalendarInterval": {"Minute": 1}},
    ],
)
def test_stopped_launchd_malformed_schedule_refuses_without_effects(
    scheduler_host: Host, action: str, schedule: dict[str, object]
) -> None:
    host = scheduler_host
    path = _prepare_launchd_action(host, loaded=False)
    value = plistlib.loads(path.read_bytes())
    del value["StartInterval"]
    path.write_bytes(plistlib.dumps(value | schedule))
    before = path.read_bytes()
    result = host.invoke(action, redirected=True)
    assert result.returncode == (2 if action == "compact" else 1), result.stdout
    assert "launchd schedule" in result.stdout
    assert host.mutations() == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("scheduler_host", ["launchd"], indirect=True)
@pytest.mark.parametrize("action", ["start", "install", "uninstall", "compact"])
@pytest.mark.parametrize("source", ["domain", "job"])
@pytest.mark.parametrize("block", ["default environment", "inherited environment", "environment"])
def test_launchd_duplicate_environment_refuses_without_effects(
    scheduler_host: Host, action: str, source: str, block: str
) -> None:
    host = scheduler_host
    path = _prepare_launchd_action(host, loaded=True)
    state = json.loads(host.state.read_text())
    # The second assignment matches: last-wins parsing would authorize this action.
    state[source + "_blocks"] = (
        f"{block} = {{\n RECALL_DATA_DIR => {host.root / 'foreign'}\n"
        f" RECALL_DATA_DIR => {host.selected}\n }}"
    )
    host.state.write_text(json.dumps(state))
    before = path.read_bytes()
    result = host.invoke(action, redirected=True)
    assert result.returncode == (2 if action == "compact" else 1), result.stdout
    assert "duplicate launchctl" in result.stdout
    assert host.mutations() == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("scheduler_host", ["launchd"], indirect=True)
@pytest.mark.parametrize("action", ["start", "install", "uninstall", "compact"])
def test_stopped_launchd_duplicate_plist_environment_refuses_without_effects(
    scheduler_host: Host, action: str
) -> None:
    from xml.sax.saxutils import escape

    host = scheduler_host
    path = _prepare_launchd_action(host, loaded=False)
    value = plistlib.loads(path.read_bytes())
    value["EnvironmentVariables"] = {"RECALL_DATA_DIR": str(host.selected)}
    content = plistlib.dumps(value).decode()
    assignment = "<key>RECALL_DATA_DIR</key>"
    duplicate = f"{assignment}<string>{escape(str(host.root / 'foreign'))}</string>"
    path.write_text(content.replace(assignment, duplicate + assignment))
    before = path.read_bytes()
    result = host.invoke(action, redirected=True)
    assert result.returncode == (2 if action == "compact" else 1), result.stdout
    assert "duplicate launchd plist key" in result.stdout
    assert host.mutations() == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("scheduler_host", ["launchd"], indirect=True)
@pytest.mark.parametrize("action", ["start", "install"])
@pytest.mark.parametrize(
    "schedule",
    [
        {"KeepAlive": True},
        {"KeepAlive": {"PathState": {"/fixture/refused": False}}},
        {"StartInterval": 300},
        {"KeepAlive": False, "StartInterval": 300},
    ],
)
def test_stopped_launchd_supported_schedule_starts(
    scheduler_host: Host, action: str, schedule: dict[str, object]
) -> None:
    host = scheduler_host
    path = _prepare_launchd_action(host, loaded=False)
    value = plistlib.loads(path.read_bytes())
    del value["StartInterval"]
    path.write_bytes(plistlib.dumps(value | schedule))
    before = path.read_bytes()
    if action == "start" and schedule.get("KeepAlive"):
        state = json.loads(host.state.read_text())
        state.update(bootstrap_pid=os.getpid(), pid_path=str(host.selected / "recall.pid"))
        host.state.write_text(json.dumps(state))
    result = host.invoke(action, redirected=True)
    assert result.returncode == 0, result.stdout
    if action == "start":
        assert json.loads(result.stdout)["started"] is True
        assert host.mutations() == [["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)]]
        assert path.read_bytes() == before
    else:
        assert json.loads(result.stdout)["installed"] is True
        assert json.loads(result.stdout)["scheduler"] == "launchd"
        assert plistlib.loads(path.read_bytes())["StartInterval"] == 300


@pytest.mark.parametrize("scheduler_host", ["launchd"], indirect=True)
@pytest.mark.parametrize("winner", range(6))
def test_launchd_distinct_environment_blocks_keep_override_order(
    scheduler_host: Host, winner: int
) -> None:
    host = scheduler_host
    _prepare_launchd_action(host, loaded=True)
    state = json.loads(host.state.read_text())
    layers = ["default environment", "inherited environment", "environment"]
    for source, offset in [("domain", 0), ("job", 3)]:
        blocks = []
        for index, block in enumerate(layers, offset):
            if index <= winner:
                data = host.selected if index == winner else host.root / f"foreign-{index}"
                blocks.append(f"{block} = {{\n RECALL_DATA_DIR => {data}\n }}")
        state[source + "_blocks"] = "\n".join(reversed(blocks))
    host.state.write_text(json.dumps(state))
    result = host.invoke("stop", redirected=True)
    assert result.returncode == 0, result.stdout
    assert json.loads(result.stdout)["stopped"] is True
    assert host.mutations() == [
        ["launchctl", "bootout", f"gui/{os.getuid()}/it.send.recall.daemon"]
    ]


@pytest.mark.parametrize("scheduler_host", ["launchd"], indirect=True)
@pytest.mark.parametrize("action", ["start", "install", "uninstall", "compact"])
@pytest.mark.parametrize(
    "schedule",
    [
        "",
        "run interval = nope",
        "run interval = 0 seconds",
        "run interval = 300 seconds\n properties = keepalive",
        "event triggers = {\n com.apple.launchd.PathState => {\n keepalive = 1\n }\n }",
        "event triggers = {\n unknown => {\n keepalive = 1\n }\n }",
        "event triggers = {\n com.apple.launchd.PathState => {\n keepalive = 1\n"
        ' descriptor = {\n "PathState" => {\n "/fixture/refused" => 0\n }\n }\n }\n }',
    ],
)
def test_loaded_launchd_malformed_schedule_refuses_without_effects(
    scheduler_host: Host, action: str, schedule: str
) -> None:
    host = scheduler_host
    path = _prepare_launchd_action(host, loaded=True)
    state = json.loads(host.state.read_text())
    state["schedule"] = schedule
    host.state.write_text(json.dumps(state))
    before = path.read_bytes()
    result = host.invoke(action, redirected=True)
    assert result.returncode == (2 if action == "compact" else 1), result.stdout
    assert "launchd schedule" in result.stdout
    assert host.mutations() == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("scheduler_host", ["launchd"], indirect=True)
@pytest.mark.parametrize(
    "schedule",
    [
        "properties = keepalive",
        "semaphores = {\n successful exit => 0\n }",
        "semaphores = {\n after crash => 1\n }",
        "semaphores = {\n other service enabled: it.send.recall.test-other => 1\n }",
        "event triggers = {\n com.apple.launchd.PathState => {\n keepalive = 1\n"
        ' descriptor = {\n "PathState" => {\n "/fixture/refused" => false\n }\n }\n }\n }',
    ],
)
def test_loaded_launchd_keepalive_conditions_report_watch_mode(
    scheduler_host: Host, schedule: str
) -> None:
    host = scheduler_host
    _prepare_launchd_action(host, loaded=True)
    state = json.loads(host.state.read_text())
    state["schedule"] = schedule
    host.state.write_text(json.dumps(state))
    result = host.invoke("status", redirected=True)
    assert result.returncode == 0, result.stdout
    assert json.loads(result.stdout)["installed"] is True
    assert json.loads(result.stdout)["installed_mode"] == "watch"
    assert host.mutations() == []


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
    state["environment_files"] = ["/unknown/environment (ignore_errors=no)"]
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


SECRET = "sk-sentinel-7f3a9c"


def _set_state(host: Host, **values: object) -> None:
    state = json.loads(host.state.read_text())
    state.update(values)
    host.state.write_text(json.dumps(state))


def _environment_file(host: Host, name: str, body: str) -> Path:
    path = host.root / "environment" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(f"API_KEY={SECRET}\nOTHER='{SECRET}\nRECALL_DATA_DIR=/quoted'\n{body}")
    return path


def _entry(path: Path, *, ignore_errors: bool) -> str:
    return f"{path} (ignore_errors={'yes' if ignore_errors else 'no'})"


def _assert_secret_absent(host: Host, result: subprocess.CompletedProcess[str]) -> None:
    assert SECRET not in result.stdout
    assert SECRET not in result.stderr
    if host.calls.exists():
        assert SECRET not in host.calls.read_text()
    for path in [host.selected, host.account / ".local/share/recall"]:
        for file in path.rglob("*") if path.exists() else ():
            if file.is_file() and file.suffix in {".log", ".json", ".txt", ""}:
                assert SECRET.encode() not in file.read_bytes(), file


def _message(result: subprocess.CompletedProcess[str]) -> str:
    payload = json.loads(result.stdout)
    return payload["error"]["message"] if "error" in payload else payload["message"]


def _matching_scenario(host: Host, row: str) -> None:
    """Timer-driven units whose environment files leave the selected data dir provable."""
    host.selected = host.account / ".local/share/recall"
    timer = host.account / ".config/systemd/user/recall-daemon.timer"
    timer.write_text("[Timer]\nOnUnitActiveSec=300\nUnit=recall-daemon.service\n")
    selected = str(host.selected)
    match row:
        case "no-identity-keys":
            files = [_entry(_environment_file(host, "a.env", ""), ignore_errors=True)]
        case "sets-selected":
            path = _environment_file(host, "a.env", f"RECALL_DATA_DIR={selected}\n")
            files = [_entry(path, ignore_errors=False)]
        case "sets-selected-home":
            host.selected = host.root / "home2/.local/share/recall"
            path = _environment_file(host, "a.env", f"HOME={host.root / 'home2'}\n")
            files = [_entry(path, ignore_errors=False)]
        case "missing-optional":
            files = [_entry(host.root / "absent.env", ignore_errors=True)]
        case "later-file-wins":
            first = _environment_file(host, "a.env", f"RECALL_DATA_DIR={host.root / 'other'}\n")
            second = _environment_file(host, "b.env", f"RECALL_DATA_DIR={selected}\n")
            files = [_entry(first, ignore_errors=False), _entry(second, ignore_errors=True)]
        case "unset-after-file":
            path = _environment_file(host, "a.env", f"RECALL_DATA_DIR={host.root / 'other'}\n")
            files = [_entry(path, ignore_errors=False)]
            _set_state(host, unset="RECALL_DATA_DIR")
        case _:
            raise AssertionError(row)
    _set_state(host, environment_files=files)


MATCHING_ROWS = [
    "no-identity-keys",
    "sets-selected",
    "sets-selected-home",
    "missing-optional",
    "later-file-wins",
    "unset-after-file",
]


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize("row", MATCHING_ROWS)
@pytest.mark.parametrize("action", ["status", "stop", "start", "restart", "install", "uninstall"])
def test_req_daemon_084_provable_environment_file_keeps_each_action(
    scheduler_host: Host, row: str, action: str
) -> None:
    host = scheduler_host
    _matching_scenario(host, row)
    result = host.invoke(action, redirected=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    payload = json.loads(result.stdout)
    assert "error" not in payload
    match action:
        case "status":
            assert payload["installed"] is True
            assert payload["scheduler"] == "systemd"
            assert payload["service_data_dir"] == str(host.selected)
            assert payload["service_identity_error"] is None
            assert host.mutations() == []
        case "stop":
            assert payload["stopped"] is True
            assert host.mutations() == [
                ["systemctl", "--user", "stop", "recall-daemon.timer"],
                ["systemctl", "--user", "stop", "recall-daemon.service"],
            ]
        case "start" | "restart":
            assert payload["started"] is True
        case "install":
            assert payload["installed"] is True
            assert payload["service_identity_error"] is None
        case "uninstall":
            assert payload["installed"] is False
    _assert_secret_absent(host, result)


def _foreign_scenario(host: Host, row: str) -> Path:
    """Units whose environment files select another data dir than the selected one."""
    host.selected = host.account / ".local/share/recall"
    other = host.root / "other"
    match row:
        case "overrides-environment":
            _set_state(host, data=str(host.selected))
            path = _environment_file(host, "a.env", f"RECALL_DATA_DIR={other}\n")
            files = [_entry(path, ignore_errors=True)]
        case "later-file-wins":
            first = _environment_file(host, "a.env", f"RECALL_DATA_DIR={host.selected}\n")
            path = _environment_file(host, "b.env", f"RECALL_DATA_DIR={other}\n")
            files = [_entry(first, ignore_errors=False), _entry(path, ignore_errors=False)]
        case _:
            raise AssertionError(row)
    _set_state(host, environment_files=files)
    return path


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize("row", ["overrides-environment", "later-file-wins"])
@pytest.mark.parametrize("action", ["stop", "start", "restart", "install", "uninstall"])
def test_req_daemon_084_environment_file_on_another_data_dir_refuses(
    scheduler_host: Host, row: str, action: str
) -> None:
    host = scheduler_host
    path = _foreign_scenario(host, row)
    result = host.invoke(action, redirected=True)
    assert result.returncode == 1, (result.stdout, result.stderr)
    message = _message(result)
    assert f"refusing to {action}" in message
    assert f"runs data dir {host.root / 'other'} (set by EnvironmentFile {path})" in message
    assert str(host.selected) in message
    assert host.mutations() == []
    _assert_secret_absent(host, result)


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
def test_req_daemon_084_status_names_the_environment_file_data_dir(scheduler_host: Host) -> None:
    host = scheduler_host
    _foreign_scenario(host, "overrides-environment")
    result = host.invoke("status", redirected=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    payload = json.loads(result.stdout)
    assert payload["installed"] is False
    assert payload["scheduler"] is None
    assert payload["service_data_dir"] == str(host.root / "other")
    assert payload["service_identity_error"] is None
    _assert_secret_absent(host, result)


def _unprovable_scenario(host: Host, row: str) -> str:
    """Units whose environment files recall cannot prove; returns the expected cause."""
    host.selected = host.account / ".local/share/recall"
    directory = host.root / "environment"
    directory.mkdir(exist_ok=True)
    path = directory / "a.env"
    ignore_errors = True
    match row:
        case "missing-required":
            ignore_errors = False
            cause = f"EnvironmentFile {path} is missing and required; the service cannot start"
        case "directory":
            path.mkdir()
            cause = f"EnvironmentFile {path} is not a regular file"
        case "nul":
            path.write_bytes(f"API_KEY={SECRET}\0\n".encode())
            cause = f"EnvironmentFile {path} contains a NUL byte"
        case "invalid-utf8":
            path.write_bytes(f"API_KEY={SECRET}".encode() + b"\xff\n")
            cause = f"EnvironmentFile {path} contains a non-ASCII byte"
        case "noncharacter":
            path.write_text(f"BAD=\ufdd0{SECRET}\nRECALL_DATA_DIR={host.selected}\n")
            cause = f"EnvironmentFile {path} contains a non-ASCII byte"
        case "glob":
            path = directory / "*.env"
            cause = f"EnvironmentFile {path} uses a glob pattern recall does not expand"
        case "escaped-identity":
            _environment_file(host, "a.env", "RECALL_DATA_DIR=/a\\ b\n")
            cause = f"RECALL_DATA_DIR in EnvironmentFile {path} line 4 uses quoting or escapes"
        case "comment-backslash":
            _environment_file(host, "a.env", "# note \\\nRECALL_DATA_DIR=/x\n")
            cause = f"EnvironmentFile {path} comment line 4 ends with a backslash"
        case _:
            raise AssertionError(row)
    _set_state(host, environment_files=[_entry(path, ignore_errors=ignore_errors)])
    return cause


UNPROVABLE_ROWS = [
    "missing-required",
    "directory",
    "nul",
    "invalid-utf8",
    "noncharacter",
    "glob",
    "escaped-identity",
    "comment-backslash",
]


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize("row", UNPROVABLE_ROWS)
@pytest.mark.parametrize("action", ["stop", "start", "restart", "install", "uninstall"])
def test_req_daemon_084_unprovable_environment_file_refuses(
    scheduler_host: Host, row: str, action: str
) -> None:
    host = scheduler_host
    cause = _unprovable_scenario(host, row)
    result = host.invoke(action, redirected=True)
    assert result.returncode == 1, (result.stdout, result.stderr)
    message = _message(result)
    assert message.startswith(
        f"refusing to {action} the installed systemd service: cannot identify its data dir "
        "from the scheduler record: "
    )
    assert cause in message
    assert message.endswith("No service action was taken.")
    assert host.mutations() == []
    _assert_secret_absent(host, result)


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize("row", UNPROVABLE_ROWS)
def test_req_daemon_085_status_reports_why_the_service_is_unidentified(
    scheduler_host: Host, row: str
) -> None:
    host = scheduler_host
    cause = _unprovable_scenario(host, row)
    result = host.invoke("status", redirected=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    payload = json.loads(result.stdout)
    assert payload["installed"] is False
    assert payload["scheduler"] is None
    assert payload["scheduler_health_state"] == "unknown"
    assert cause in payload["service_identity_error"]
    _assert_secret_absent(host, result)


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
def test_req_daemon_085_text_status_names_the_identity_error(scheduler_host: Host) -> None:
    host = scheduler_host
    cause = _unprovable_scenario(host, "missing-required")
    env = host.environ(redirected=True)
    result = subprocess.run(
        [sys.executable, str(host.wrapper), "daemon", "status", "--format", "text"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "Installed: unknown (unknown)" in result.stdout
    assert f"Service: cannot identify: {cause}" in result.stdout


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize("key", ["RECALL_CONFIG_PATH", "RECALL_DB_PATH", "RECALL_LOCK_PATH"])
@pytest.mark.parametrize("override", ["drop-in-file", "unset-after-file"])
def test_req_daemon_084_install_rolls_back_when_the_reloaded_unit_differs(
    scheduler_host: Host, key: str, override: str
) -> None:
    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    unit = host.account / ".config/systemd/user/recall-daemon.service"
    before = unit.read_bytes()
    written = {
        "RECALL_CONFIG_PATH": str(host.config),
        "RECALL_DB_PATH": str(host.selected / "recall.duckdb"),
        "RECALL_LOCK_PATH": str(host.selected / "recall.lock"),
    }[key]
    if override == "drop-in-file":
        path = _environment_file(host, "a.env", f"{key}={host.root / 'elsewhere'}\n")
        expected = f"the reloaded service's {key} is {host.root / 'elsewhere'}, not {written}"
    else:
        path = _environment_file(host, "a.env", f"{key}={written}\n")
        _set_state(host, unset=key)
        expected = f"the reloaded service's {key} is unset, not {written}"
    _set_state(host, environment_files=[_entry(path, ignore_errors=False)])
    result = host.invoke("install", redirected=True)
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert f"install would not take effect: {expected}" in _message(result)
    mutations = host.mutations()
    assert not any(
        call[2] in {"enable", "restart", "start"} for call in mutations if call[0] == "systemctl"
    )
    assert mutations[-1] == ["systemctl", "--user", "daemon-reload"]
    assert unit.read_bytes() == before
    _assert_secret_absent(host, result)


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
def test_req_daemon_084_install_rolls_back_when_the_reloaded_unit_is_unidentified(
    scheduler_host: Host,
) -> None:
    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    unit = host.account / ".config/systemd/user/recall-daemon.service"
    before = unit.read_bytes()
    _set_state(host, unreadable_after_reload=True)
    result = host.invoke("install", redirected=True)
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert "install would not take effect: cannot identify the reloaded service" in _message(result)
    mutations = host.mutations()
    assert not any(
        call[2] in {"enable", "restart", "start"} for call in mutations if call[0] == "systemctl"
    )
    assert mutations[-1] == ["systemctl", "--user", "daemon-reload"]
    assert unit.read_bytes() == before


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
def test_req_daemon_084_install_drops_a_fragment_environment_file(scheduler_host: Host) -> None:
    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    unit = host.account / ".config/systemd/user/recall-daemon.service"
    path = _environment_file(host, "fragment.env", f"RECALL_CONFIG_PATH={host.root / 'old'}\n")
    unit.write_text(
        f"[Service]\nType=oneshot\nEnvironment=RECALL_DATA_DIR={host.selected}\n"
        f"EnvironmentFile={path}\n"
    )
    _set_state(host, read_artifacts=True)
    status = json.loads(host.invoke("status", redirected=True).stdout)
    assert (status["scheduler"], status["service_data_dir"]) == ("systemd", str(host.selected))
    result = host.invoke("install", redirected=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert json.loads(result.stdout)["installed"] is True
    assert "EnvironmentFile=" not in unit.read_text()
    _assert_secret_absent(host, result)


def _control_group(pid: int) -> str:
    """The unified-hierarchy cgroup systemd would report as the unit's ControlGroup."""
    lines = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    return next(line.removeprefix("0::") for line in lines if line.startswith("0::"))


@pytest.fixture
def running_service(scheduler_host: Host):
    """Start a real child as the unit's MainPID; it is stopped by its recorded PID."""
    children: list[subprocess.Popen[bytes]] = []

    def start(environment: dict[str, str], cwd: Path | None = None) -> int:
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.read()"],
            env=environment | {"API_KEY": SECRET},
            cwd=cwd,
            stdin=subprocess.PIPE,
        )
        children.append(child)
        _set_state(
            scheduler_host,
            main_pid=child.pid,
            loaded=True,
            control_group=_control_group(child.pid),
        )
        return child.pid

    yield start
    for child in children:
        assert child.stdin is not None
        child.stdin.close()
        child.wait(timeout=10)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="procfs is Linux-only")
@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize("action", ["status", "stop", "restart"])
def test_req_daemon_084_running_process_on_the_selected_data_dir_is_identified(
    scheduler_host: Host, running_service, action: str
) -> None:
    host = scheduler_host
    _matching_scenario(host, "sets-selected")
    running_service({"HOME": str(host.account), "RECALL_DATA_DIR": str(host.selected)})
    result = host.invoke(action, redirected=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    payload = json.loads(result.stdout)
    if action == "status":
        assert payload["installed"] is True
        assert payload["service_identity_error"] is None
    else:
        assert host.mutations() != []
    _assert_secret_absent(host, result)


def _drift_scenario(host: Host, row: str, running_service) -> tuple[Path, Path]:
    """A running process whose data dir differs from its unit's next start."""
    selected = host.account / ".local/share/recall"
    other = host.root / "other"
    host.selected = selected
    match row:
        case "file-changed-to-selected":
            path = _environment_file(host, "a.env", f"RECALL_DATA_DIR={selected}\n")
            _set_state(host, environment_files=[_entry(path, ignore_errors=False)])
            running_service({"HOME": str(host.account), "RECALL_DATA_DIR": str(other)})
            return other, selected
        case "file-changed-to-another":
            path = _environment_file(host, "a.env", f"RECALL_DATA_DIR={other}\n")
            _set_state(host, environment_files=[_entry(path, ignore_errors=False)])
            running_service({"HOME": str(host.account), "RECALL_DATA_DIR": str(selected)})
            return selected, other
        case "unit-environment-changed":
            _set_state(host, data=str(selected))
            running_service({"HOME": str(host.account), "RECALL_DATA_DIR": str(other)})
            return other, selected
        case "working-directory-changed":
            started, edited = host.root / "started-in", host.root / "edited-to"
            started.mkdir()
            edited.mkdir()
            host.selected = edited / "relative"
            _set_state(host, data="relative", working_directory=str(edited))
            running_service({"HOME": str(host.account), "RECALL_DATA_DIR": "relative"}, started)
            return started.resolve() / "relative", edited.resolve() / "relative"
        case _:
            raise AssertionError(row)


DRIFT_ROWS = [
    "file-changed-to-selected",
    "file-changed-to-another",
    "unit-environment-changed",
    "working-directory-changed",
]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="procfs is Linux-only")
@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize("row", DRIFT_ROWS)
@pytest.mark.parametrize("action", ["stop", "start", "restart", "install", "uninstall"])
def test_req_daemon_084_running_data_dir_drift_refuses(
    scheduler_host: Host, running_service, row: str, action: str
) -> None:
    host = scheduler_host
    running, next_start = _drift_scenario(host, row, running_service)
    result = host.invoke(action, redirected=True)
    assert result.returncode == 1, (result.stdout, result.stderr)
    message = _message(result)
    assert f"refusing to {action} the installed systemd service" in message
    assert (
        f"recall-daemon.service is running data dir {running} but its next start would use "
        f"{next_start} (changed since it started)"
    ) in message
    assert host.mutations() == []
    _assert_secret_absent(host, result)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="procfs is Linux-only")
@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize("row", DRIFT_ROWS)
def test_req_daemon_085_status_reports_running_data_dir_drift(
    scheduler_host: Host, running_service, row: str
) -> None:
    host = scheduler_host
    running, next_start = _drift_scenario(host, row, running_service)
    result = host.invoke("status", redirected=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    payload = json.loads(result.stdout)
    assert payload["installed"] is False
    assert payload["scheduler"] is None
    assert (
        f"running data dir {running} but its next start would use {next_start}"
        in payload["service_identity_error"]
    )
    _assert_secret_absent(host, result)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="procfs is Linux-only")
@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize("action", ["stop", "restart"])
def test_req_daemon_084_exited_main_process_refuses_and_asks_for_a_retry(
    scheduler_host: Host, action: str
) -> None:
    host = scheduler_host
    _matching_scenario(host, "no-identity-keys")
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)
    _set_state(host, main_pid=child.pid, loaded=True)
    result = host.invoke(action, redirected=True)
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert f"running process {child.pid} changed or cannot be read" in _message(result)
    assert "; retry" in _message(result)
    assert host.mutations() == []


AUTO_FORK_PROBE = """
from recall.cli.daemon import _refuse_managed_auto_fork
from recall.core.config import AppConfig
from recall.core.rpc_types import RpcConnectionError
try:
    _refuse_managed_auto_fork(AppConfig.load())
except RpcConnectionError as err:
    print("refused: " + err.message)
else:
    print("allowed")
"""


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("sets-selected", "has no reachable daemon; refusing to auto-fork"),
        ("overrides-environment", "allowed"),
        ("missing-required", "cannot identify installed service; refusing to auto-fork"),
    ],
)
def test_req_daemon_084_auto_fork_follows_the_environment_file_identity(
    scheduler_host: Host, scenario: str, expected: str
) -> None:
    host = scheduler_host
    if scenario == "sets-selected":
        _matching_scenario(host, scenario)
    elif scenario == "overrides-environment":
        _foreign_scenario(host, scenario)
    else:
        _unprovable_scenario(host, scenario)
    probe = host.root / "auto_fork_probe.py"
    probe.write_text(host.wrapper.read_text().removesuffix("app()\n") + AUTO_FORK_PROBE)
    env = host.environ(redirected=True)
    result = subprocess.run(
        [sys.executable, str(probe)], env=env, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert expected in result.stdout
    assert host.mutations() == []
    _assert_secret_absent(host, result)


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
def test_req_daemon_084_compaction_stops_and_restarts_a_provable_service(
    scheduler_host: Host,
) -> None:
    from recall.core.config import AppConfig
    from recall.db import connect

    host = scheduler_host
    _matching_scenario(host, "sets-selected")
    host.selected.mkdir(parents=True, exist_ok=True)
    config = replace(
        AppConfig.load(),
        data_dir=host.selected,
        db_path=host.selected / "recall.duckdb",
        lock_path=host.selected / "recall.lock",
    )
    connect(config).close()
    result = host.invoke("compact", redirected=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert host.mutations() == [
        ["systemctl", "--user", "stop", "recall-daemon.timer"],
        ["systemctl", "--user", "stop", "recall-daemon.service"],
        ["systemctl", "--user", "start", "recall-daemon.timer"],
    ]
    _assert_secret_absent(host, result)


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize("ignore_errors", [True, False])
@pytest.mark.parametrize("action", ["status", "start", "restart"])
def test_req_daemon_086_noncharacter_file_never_selects_an_inactive_service(
    scheduler_host: Host, ignore_errors: bool, action: str
) -> None:
    """systemd drops the whole file and starts the unit's own data dir instead."""
    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    path = host.root / "secrets.env"
    path.write_text(f"BAD=﷐{SECRET}\nRECALL_DATA_DIR={host.selected}\n")
    _set_state(
        host,
        data=str(host.root / "other"),
        loaded=False,
        environment_files=[_entry(path, ignore_errors=ignore_errors)],
    )
    result = host.invoke(action, redirected=True)
    cause = f"EnvironmentFile {path} contains a non-ASCII byte"
    if action == "status":
        assert result.returncode == 0, (result.stdout, result.stderr)
        payload = json.loads(result.stdout)
        assert (payload["installed"], payload["scheduler"]) == (False, None)
        assert cause in payload["service_identity_error"]
    else:
        assert result.returncode == 1, (result.stdout, result.stderr)
        assert cause in _message(result)
    assert host.mutations() == []
    _assert_secret_absent(host, result)


# A unit cgroup no test process belongs to: the number now names another process.
FOREIGN_UNIT_CGROUP = (
    "/user.slice/user-1000.slice/user@1000.service/app.slice/recall-daemon.service"
)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="procfs is Linux-only")
@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize("action", ["status", "stop", "start", "restart", "install", "uninstall"])
def test_req_daemon_086_main_pid_outside_the_units_control_group_refuses(
    scheduler_host: Host, running_service, action: str
) -> None:
    """The manager's MainPID exited and its number was reused before recall opened it."""
    host = scheduler_host
    _matching_scenario(host, "sets-selected")
    pid = running_service({"HOME": str(host.account), "RECALL_DATA_DIR": str(host.selected)})
    _set_state(host, control_group=FOREIGN_UNIT_CGROUP)
    result = host.invoke(action, redirected=True)
    cause = f"process {pid} is not in recall-daemon.service's control group"
    if action == "status":
        assert result.returncode == 0, (result.stdout, result.stderr)
        payload = json.loads(result.stdout)
        assert (payload["installed"], payload["scheduler"]) == (False, None)
        assert cause in payload["service_identity_error"]
    else:
        assert result.returncode == 1, (result.stdout, result.stderr)
        assert cause in _message(result)
    assert host.mutations() == []
    _assert_secret_absent(host, result)


REPLACE_BEFORE_CWD = r"""
import os, pathlib, signal
from recall.services import service_environment
service_environment.PROC_ROOT = pathlib.Path(os.environ["FAKE_PROC"])
_readlink = os.readlink
_entry = service_environment.PROC_ROOT / os.environ["FAKE_ORIGINAL"]
_reuser = "/proc/" + os.environ["FAKE_REUSER"]
def _replace_then_readlink(path, *args, **kwargs):
    # The main process exits after its environment was read; its number is reused.
    if os.fspath(path).endswith("cwd") and _readlink(_entry) != _reuser:
        os.kill(int(os.environ["FAKE_ORIGINAL"]), signal.SIGKILL)
        stat = pathlib.Path("/proc/" + os.environ["FAKE_ORIGINAL"] + "/stat")
        for _ in range(500):
            if stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                break
            os.sched_yield()
        else:
            raise AssertionError("the original process did not exit")
        _entry.unlink()
        _entry.symlink_to(_reuser)
    return _readlink(path, *args, **kwargs)
os.readlink = _replace_then_readlink
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="procfs is Linux-only")
@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize("action", ["status", "stop", "restart"])
def test_req_daemon_086_environment_and_cwd_come_from_one_process(
    scheduler_host: Host, running_service, action: str
) -> None:
    """A reader that reopened the number would pair the old environ with the new cwd."""
    host = scheduler_host
    started, edited = host.root / "started-in", host.root / "edited-to"
    started.mkdir()
    edited.mkdir()
    host.selected = edited / "relative"
    _set_state(host, data="relative", working_directory=str(edited))
    original = running_service({"HOME": str(host.account), "RECALL_DATA_DIR": "relative"}, started)
    reuser = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        env={"HOME": str(host.account), "RECALL_DATA_DIR": str(host.root / "reuser")},
        cwd=edited,
        stdin=subprocess.PIPE,
    )
    proc = host.root / "proc"
    proc.mkdir()
    (proc / str(original)).symlink_to(f"/proc/{original}")
    host.wrapper.write_text(
        host.wrapper.read_text().removesuffix("app()\n") + REPLACE_BEFORE_CWD + "app()\n"
    )
    env = host.environ(redirected=True) | {
        "FAKE_PROC": str(proc),
        "FAKE_ORIGINAL": str(original),
        "FAKE_REUSER": str(reuser.pid),
    }
    command = ["daemon", action, "--json"] + (["--timeout", "5"] if action != "status" else [])
    try:
        result = subprocess.run(
            [sys.executable, str(host.wrapper), *command],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
    finally:
        assert reuser.stdin is not None
        reuser.stdin.close()
        reuser.wait(timeout=10)
    if action == "status":
        assert result.returncode == 0, (result.stdout, result.stderr)
        payload = json.loads(result.stdout)
        assert (payload["installed"], payload["scheduler"]) == (False, None)
        assert payload["service_identity_error"] is not None
    else:
        assert result.returncode == 1, (result.stdout, result.stderr)
        assert "refusing to " + action in _message(result)
    assert host.mutations() == []
    _assert_secret_absent(host, result)


def _watch_config(host: Host) -> None:
    host.config.write_text('[daemon]\nscheduler = "systemd"\nmode = "watch"\n')


def _verification_failure(host: Host, failure: str) -> None:
    if failure == "drop-in-overrides-config":
        path = _environment_file(host, "a.env", f"RECALL_CONFIG_PATH={host.root / 'elsewhere'}\n")
        _set_state(host, environment_files=[_entry(path, ignore_errors=False)])
    else:
        _set_state(host, unreadable_after_reload=True)


UNCHANGED_ACTIVATION = [
    ["systemctl", "--user", "daemon-reload"],
    ["systemctl", "--user", "daemon-reload"],
]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="procfs is Linux-only")
@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize("failure", ["drop-in-overrides-config", "lookup-fails-after-reload"])
@pytest.mark.parametrize("mode", ["watch", "watch-to-poll"])
def test_req_daemon_086_install_refusal_keeps_the_running_watch_service(
    scheduler_host: Host, running_service, failure: str, mode: str
) -> None:
    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    unit = host.account / ".config/systemd/user/recall-daemon.service"
    timer = unit.with_name("recall-daemon.timer")
    unit.write_text("[Service]\nType=simple\n")
    before = unit.read_bytes()
    if mode == "watch":
        _watch_config(host)
    _set_state(host, data=str(host.selected))
    pid = running_service({"HOME": str(host.account), "RECALL_DATA_DIR": str(host.selected)})
    _verification_failure(host, failure)
    result = host.invoke("install", redirected=True)
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert "install would not take effect" in _message(result)
    assert host.mutations() == UNCHANGED_ACTIVATION
    state = json.loads(host.state.read_text())
    assert (state["loaded"], state["main_pid"]) == (True, pid)
    assert unit.read_bytes() == before
    assert not timer.exists()
    _assert_secret_absent(host, result)


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
@pytest.mark.parametrize("failure", ["drop-in-overrides-config", "lookup-fails-after-reload"])
@pytest.mark.parametrize("mode", ["poll", "poll-to-watch"])
def test_req_daemon_086_install_refusal_keeps_the_timer(
    scheduler_host: Host, failure: str, mode: str
) -> None:
    """The active timer is paused while the candidate is unverified, then resumed."""
    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    unit = host.account / ".config/systemd/user/recall-daemon.service"
    timer = unit.with_name("recall-daemon.timer")
    timer.write_text("[Timer]\nOnUnitActiveSec=300\nUnit=recall-daemon.service\n")
    before = (unit.read_bytes(), timer.read_bytes())
    if mode == "poll-to-watch":
        _watch_config(host)
    _set_state(host, data=str(host.selected))
    _verification_failure(host, failure)
    result = host.invoke("install", redirected=True)
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert "install would not take effect" in _message(result)
    assert host.mutations() == [
        ["systemctl", "--user", "stop", "recall-daemon.timer"],
        *UNCHANGED_ACTIVATION,
        ["systemctl", "--user", "start", "recall-daemon.timer"],
    ]
    assert (unit.read_bytes(), timer.read_bytes()) == before


@pytest.mark.parametrize("scheduler_host", ["systemd"], indirect=True)
def test_req_daemon_086_poll_install_activates_only_after_verification(
    scheduler_host: Host,
) -> None:
    """Watch installs wait for the daemon's RPC; the native rehearsal drives those."""
    host = scheduler_host
    host.selected = host.account / ".local/share/recall"
    timer = host.account / ".config/systemd/user/recall-daemon.timer"
    timer.write_text("[Timer]\nOnUnitActiveSec=300\nUnit=recall-daemon.service\n")
    _set_state(host, data=str(host.selected))
    result = host.invoke("install", redirected=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert host.mutations() == [
        ["systemctl", "--user", "stop", "recall-daemon.timer"],
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "disable", "--now", "recall-daemon.service"],
        ["systemctl", "--user", "enable", "--now", "recall-daemon.timer"],
    ]
    assert timer.exists()
