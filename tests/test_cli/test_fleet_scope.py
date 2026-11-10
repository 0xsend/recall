"""Public endpoint membership and invocation lifecycle contracts."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from recall.cli.app import app
from recall.core.rpc_client import RpcConnectionError
from recall.core.types import Source
from recall.services.fleet_transport import FleetRemoteResult
from typer.testing import CliRunner

COMMANDS = ["list", "search", "show", "live", "usage", "skills"]


def payload(command: str, label: str) -> Any:
    row = {
        "id": label,
        "session_id": label,
        "source": "codex",
        "host": "local",
        "started_at": "2026-11-10T12:00:00Z",
        "last_activity_at": "2026-11-10T12:00:00Z",
        "liveness": "active",
        "score": 2 if label == "caller-item" else 1,
        "lexical_match": True,
    }
    if command == "show":
        return {**row, "messages": [], "tool_calls": []}
    if command == "live":
        return {
            "schema_version": 2,
            "sessions": [row],
            "watching": True,
            "coverage": {"complete": True},
            "next_cursor": None,
        }
    if command == "usage":
        return [
            {
                "source": "codex",
                "model": "fixture",
                "host": "local",
                "input_tokens": 20 if label == "caller-item" else 10,
                "output_tokens": 2,
                "cached_input_tokens": 0,
                "fresh_input_tokens": None,
                "session_count": 1,
            }
        ]
    if command == "skills":
        counts = {
            "considered_sessions": 1,
            "attributed_invocations": 2,
            "unattributed_candidates": 0,
        }
        return {
            "rows": [
                {
                    "skill_name": "fixture",
                    "source": "codex",
                    "host": "aem5",
                    "invocations": 2,
                    "sessions": 1,
                }
            ],
            "coverage": {
                "scope": "local",
                "expected_hosts": ["aem5"],
                "successful_hosts": ["aem5"],
                "covered_sources": [s.value for s in Source],
                **counts,
                "control": counts,
            },
        }
    return [row]


def argv(command: str, inventory: Path, *extra: str) -> list[str]:
    prefix = {
        "search": ["search", "fixture"],
        "show": ["show", "caller-item"],
        "usage": ["stats", "usage"],
        "skills": ["stats", "skills"],
    }.get(command, [command])
    return [
        *prefix,
        *([] if command == "skills" else ["--fleet"]),
        "--fleet-config",
        str(inventory),
        *extra,
    ]


@contextmanager
def peer(
    directory: Path,
    command: str,
    contacts: list[str],
    *,
    status_value: dict[str, Any] | None = None,
    status_error: bool = False,
) -> Iterator[None]:
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(directory / "recall.sock"))
    listener.listen(8)
    listener.settimeout(0.05)
    stopped = threading.Event()
    errors: list[Exception] = []

    def serve() -> None:
        while not stopped.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            with conn:
                conn.settimeout(2)
                try:
                    request = json.loads(conn.makefile("rb").readline())
                    method = request["method"]
                    contacts.append(method)
                    result = (
                        (status_value or {})
                        if method == "recall.daemon_status"
                        else payload(command, "caller-item")
                    )
                    response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
                    if method == "recall.daemon_status" and status_error:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request["id"],
                            "error": {"code": -32603, "message": "status unavailable"},
                        }
                    conn.sendall((json.dumps(response) + "\n").encode())
                except Exception as error:
                    errors.append(error)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(3)
        listener.close()
        (directory / "recall.sock").unlink()
        assert not thread.is_alive()
        assert not errors


@pytest.fixture
def fleet_fixture(socket_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("recall.cli.rpc._in_process_server", None)
    monkeypatch.setenv("RECALL_DATA_DIR", str(socket_dir))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(socket_dir / "config.toml"))
    monkeypatch.setattr("socket.gethostname", lambda: "aem5.example.com")
    contacts: list[str] = []
    inventory = socket_dir / "fleet.toml"
    inventory.write_text('[[host]]\nname="edge"\nssh="edge"\n')
    return socket_dir, inventory, contacts


def remote_fixture(monkeypatch: pytest.MonkeyPatch, command: str, contacts: list[str]) -> None:
    def remote(host, remote_argv, **kwargs):
        contacts.append("ssh:" + host.name)
        if command == "show":
            return FleetRemoteResult(False, "", "session not found", 1, "session not found")
        return FleetRemoteResult(True, json.dumps(payload(command, "edge-item")), "", 0)

    # Patching existing effect boundaries keeps the full CLI/advisory path intact on base.
    monkeypatch.setattr("recall.services.fleet_transport.run_remote", remote)


def assert_contributions(command: str, value: Any, expected_hosts: list[str]) -> None:
    if command == "skills":
        assert value["coverage"]["expected_hosts"] == expected_hosts
        assert value["coverage"]["considered_sessions"] == len(expected_hosts)
        assert sum(row["invocations"] for row in value["rows"]) == 2 * len(expected_hosts)
    elif command == "usage":
        assert sorted(row["host"] for row in value) == sorted(expected_hosts)
        assert sum(row["input_tokens"] for row in value) == 20 + 10 * (len(expected_hosts) - 1)
    elif command == "show":
        assert value["id"] == "caller-item"
        assert value["host"] == "aem5"
    else:
        rows = value["sessions"] if command == "live" else value
        assert sorted(row["host"] for row in rows) == sorted(expected_hosts)
        assert rows[0]["id"] == "caller-item"
        if command == "live":
            assert value["watching"] is None
            assert [h["name"] for h in value["coverage"]["hosts"]] == expected_hosts
            assert value["coverage"]["hosts"][0]["watching"] is True


@pytest.mark.parametrize("command", COMMANDS)
def test_fleet_includes_caller_absent_from_inventory(fleet_fixture, monkeypatch, command):
    directory, inventory, contacts = fleet_fixture
    remote_fixture(monkeypatch, command, contacts)
    with peer(directory, command, contacts):
        result = CliRunner().invoke(app, argv(command, inventory, "--json"))
    assert result.exit_code == 0, (result.output, result.exception)
    assert_contributions(command, json.loads(result.stdout), ["aem5", "edge"])


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize(
    "name,declared",
    [
        ("aem5", False),
        ("laptop", True),
        ("aem5.example.com", True),
        ("AEM5", True),
        ("AEM5", False),
        ("aem5.example.com", False),
    ],
)
def test_fleet_self_inventory_contributes_once(fleet_fixture, monkeypatch, command, name, declared):
    directory, inventory, contacts = fleet_fixture
    inventory.write_text(
        f'[[host]]\nname="{name}"\nssh="self"\n' + ('hostname="aem5"\n' if declared else "")
    )
    remote_fixture(monkeypatch, command, contacts)
    folded = declared or name == "aem5"
    with peer(directory, command, contacts):
        result = CliRunner().invoke(app, argv(command, inventory, "--json"))
    assert result.exit_code == 0, (result.output, result.exception)
    assert_contributions(command, json.loads(result.stdout), ["aem5"] if folded else ["aem5", name])
    assert contacts.count("ssh:" + name) == (0 if folded else 1)


@pytest.mark.parametrize("command", COMMANDS[:-1])
@pytest.mark.parametrize("missing", [True, False])
def test_fleet_empty_inventory_scope(fleet_fixture, monkeypatch, command, missing):
    directory, inventory, contacts = fleet_fixture
    inventory.unlink() if missing else inventory.write_text("")
    remote_fixture(monkeypatch, command, contacts)
    with peer(directory, command, contacts):
        result = CliRunner().invoke(app, argv(command, inventory, "--json"))
    assert result.exit_code == 0, result.output
    assert_contributions(command, json.loads(result.stdout), ["aem5"])
    assert "no fleet hosts" not in result.stderr


@pytest.mark.parametrize("command", COMMANDS[:-1])
def test_fleet_inventory_only_requires_fleet(fleet_fixture, command):
    _, inventory, contacts = fleet_fixture
    args = argv(command, inventory, "--fleet-inventory-only", "--json")
    args.remove("--fleet")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    assert "requires --fleet" in result.stdout + result.stderr
    assert contacts == []


@pytest.mark.parametrize("output", ["--json", "--format=text"])
@pytest.mark.parametrize("notices", [True, False])
@pytest.mark.parametrize("inactive_poll", [True, False])
def test_fleet_zero_tool_results_have_no_local_side_effects(
    fleet_fixture,
    monkeypatch,
    output,
    notices,
    inactive_poll,
):
    directory, inventory, _contacts = fleet_fixture
    (directory / "config.toml").write_text(f"[cli]\nstatus_notices={str(notices).lower()}\n")
    effects: list[str] = []

    def startup(*args, **kwargs):
        effects.append("startup")
        raise RpcConnectionError(message="contained startup")

    def snapshot(config, request):
        from recall.core.poll_reads import PollSnapshotSelected

        effects.append("snapshot-selection")
        raise PollSnapshotSelected(config, request, "scratch-inactive-poll")

    def execute(*args, **kwargs):
        from recall.core.poll_reads import PollSnapshot

        effects.extend(["snapshot-execution", "snapshot-storage-read"])
        return PollSnapshot([{"id": "synthetic-snapshot-only"}], None, None)

    monkeypatch.setattr("recall.cli.daemon.execute_poll_snapshot", execute)

    monkeypatch.setattr("recall.core.rpc_client.RpcClient._auto_fork_daemon", startup)
    monkeypatch.setattr("recall.cli.daemon.select_poll_snapshot", snapshot)
    if not inactive_poll:
        monkeypatch.setattr("recall.cli.daemon.select_poll_snapshot", lambda *a: None)

    def remote(*args, **kwargs):
        return FleetRemoteResult(True, "[]", "", 0)

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", remote)
    result = CliRunner().invoke(app, argv("search", inventory, "--tool", "Bash", output))
    assert effects == [], (effects, result.stdout, result.stderr, result.exit_code)
    assert result.exit_code == 0, result.output
    if output == "--json":
        assert json.loads(result.stdout) == []
    assert result.stderr.count("fleet: skip aem5:") == 1
    assert "retry without --tool" in result.stderr
    assert "results without it" not in result.stderr
    assert "Poll mode:" not in result.stderr


@pytest.mark.parametrize("command", COMMANDS[:-1])
@pytest.mark.parametrize("self_entry", [False, True])
def test_fleet_inventory_only_membership(fleet_fixture, monkeypatch, command, self_entry):
    directory, inventory, contacts = fleet_fixture
    if self_entry:
        with inventory.open("a") as stream:
            stream.write('[[host]]\nname="laptop"\nssh="self"\nhostname="aem5"\n')
    remote_fixture(monkeypatch, command, contacts)
    with peer(directory, command, contacts):
        result = CliRunner().invoke(
            app, argv(command, inventory, "--fleet-inventory-only", "--json")
        )
    if command == "show" and not self_entry:
        assert result.exit_code == 1
    else:
        assert result.exit_code == 0, result.output
        value = json.loads(result.stdout)
        if self_entry:
            assert_contributions(command, value, ["aem5", "edge"])
        elif command == "usage":
            assert value[0]["input_tokens"] == 10
            assert value[0]["host"] == "edge"
        else:
            rows = value["sessions"] if command == "live" else value
            assert [row["id"] for row in rows] == ["edge-item"]
    if not self_entry:
        assert all(contact.startswith("ssh:") for contact in contacts)
    assert "ssh:laptop" not in contacts


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize(
    "config",
    [
        '[[host]]\nname="one"\nssh="one"\nhostname="aem5"\n[[host]]\nname="TWO"\nssh="two"\nhostname="aem5"\n',
        '[[host]]\nname="aem5"\nssh="one"\n[[host]]\nname="aem5.example.com"\nssh="two"\nhostname="aem5"\n',
        '[[host]]\nname="one"\nssh="one"\nhostname="edge"\n[[host]]\nname="two"\nssh="two"\nhostname="edge"\n',
        '[[host]]\nname="aem5"\nssh="foreign"\nhostname="foreign"\n',
        '[[host]]\nname="edge"\nssh="edge"\nhostname="edge.example.com"\n',
        '[[host]]\nname="edge"\nssh="edge"\nhostname="local"\n',
        '[[host]]\nname="edge"\nssh="edge"\nhostname=" "\n',
        '[[host]]\nname="edge"\nssh="edge"\nhostname="white space"\n',
        '[[host]]\nname="edge"\nssh="edge"\nhostname=42\n',
        "invalid = [",
    ],
)
def test_fleet_invalid_inventory_prevents_reads(fleet_fixture, monkeypatch, command, config):
    directory, inventory, contacts = fleet_fixture
    inventory.write_text(config)
    remote_fixture(monkeypatch, command, contacts)
    with peer(directory, command, contacts):
        result = CliRunner().invoke(app, argv(command, inventory, "--json"))
    assert result.exit_code == 2, result.output
    assert "error" in json.loads(result.stdout)
    assert contacts == []


@pytest.mark.parametrize("command", ["list", "live", "show"])
@pytest.mark.parametrize("selector", ["aem5", "laptop", "edge", "unknown"])
def test_fleet_host_selects_endpoint(fleet_fixture, monkeypatch, command, selector):
    directory, inventory, contacts = fleet_fixture
    with inventory.open("a") as stream:
        stream.write('[[host]]\nname="laptop"\nssh="self"\nhostname="aem5"\n')
    remote_fixture(monkeypatch, command, contacts)
    with peer(directory, command, contacts):
        result = CliRunner().invoke(app, argv(command, inventory, "--host", selector, "--json"))
    if selector == "unknown" or (command == "show" and selector == "edge"):
        assert result.exit_code == 1
    else:
        assert result.exit_code == 0, result.output
        if selector in ("aem5", "laptop"):
            assert_contributions(command, json.loads(result.stdout), ["aem5"])
    if selector in ("edge", "unknown"):
        assert not any(contact.startswith("recall.") for contact in contacts)
    else:
        assert not any(contact.startswith("ssh:") for contact in contacts)


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("healthy_remote", [False, True])
def test_fleet_local_failure_diagnostic_and_exit(
    fleet_fixture, monkeypatch, command, healthy_remote
):
    _, inventory, contacts = fleet_fixture

    def remote(*args, **kwargs):
        if not healthy_remote:
            return FleetRemoteResult(False, "", "remote down", 255, "remote down")
        value = payload(command, "edge-item")
        return FleetRemoteResult(True, json.dumps(value), "", 0)

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", remote)
    result = CliRunner().invoke(app, argv(command, inventory, "--json"))
    assert result.stderr.count("fleet: skip aem5:") == 1
    assert "no socket" in result.stderr
    assert "recall.daemon_status" not in contacts
    if command == "skills" or not healthy_remote:
        assert result.exit_code == 1
        assert "error" in json.loads(result.stdout)
        assert "rows" not in json.loads(result.stdout)
    else:
        assert result.exit_code == 0, result.output


@pytest.mark.parametrize("command", COMMANDS[:-1])
def test_fleet_params_and_missing_identity(fleet_fixture, monkeypatch, command):
    directory, inventory, contacts = fleet_fixture
    remote_fixture(monkeypatch, command, contacts)
    monkeypatch.setattr("socket.gethostname", lambda: "")
    result = CliRunner().invoke(app, argv(command, inventory, "--json"))
    assert result.exit_code == 2, result.output
    assert contacts == []
    args = argv(command, inventory, "--params", '{"fleet_inventory_only":true,"json":true}')
    result = CliRunner().invoke(app, args)
    assert result.exit_code == (1 if command == "show" else 0), result.output
    assert contacts == ["ssh:edge"]
    assert not (directory / "recall.sock").exists()


@pytest.mark.parametrize("command", COMMANDS[:-1])
def test_fleet_inventory_only_empty_refuses(fleet_fixture, command):
    _, inventory, contacts = fleet_fixture
    inventory.write_text("")
    result = CliRunner().invoke(app, argv(command, inventory, "--fleet-inventory-only", "--json"))
    assert result.exit_code == 2
    assert "no fleet hosts configured" in result.stdout
    assert contacts == []


@pytest.mark.parametrize("command", COMMANDS[:-1])
@pytest.mark.parametrize(
    "option", [["--fleet-exclude-local"], ["--params", '{"fleet_exclude_local":true}']]
)
def test_unselected_compatibility_option_rejects(fleet_fixture, command, option):
    _, inventory, contacts = fleet_fixture
    result = CliRunner().invoke(app, argv(command, inventory, *option, "--json"))
    assert result.exit_code == 2
    assert contacts == []


def test_fleet_read_policy_ends_at_invocation_close(fleet_fixture, monkeypatch):
    from recall.cli import rpc

    directory, inventory, contacts = fleet_fixture
    effects: list[str] = []

    def startup(*args, **kwargs):
        effects.append("ordinary-startup")
        raise RpcConnectionError(message="contained startup")

    monkeypatch.setattr("recall.core.rpc_client.RpcClient._auto_fork_daemon", startup)
    monkeypatch.setattr("recall.cli.daemon.select_poll_snapshot", lambda *args: None)
    remote_fixture(monkeypatch, "list", contacts)
    runner = CliRunner()
    for arguments in (
        ["list", "--json"],
        argv("list", inventory, "--json"),
        argv("list", inventory, "--host", "unknown", "--json"),
        ["list", "--json"],
    ):
        runner.invoke(app, arguments)
    assert effects == ["ordinary-startup", "ordinary-startup"]
    assert rpc._in_process_server is None
    assert not (directory / "recall.sock").exists()


@pytest.mark.parametrize("command", ["list", "search", "usage"])
@pytest.mark.parametrize("selected", [True, False])
@pytest.mark.parametrize("output", ["--json", "--format=text"])
def test_fleet_notices_use_only_live_selected_local_status(
    fleet_fixture, monkeypatch, command, selected, output
):
    directory, inventory, contacts = fleet_fixture
    remote_fixture(monkeypatch, command, contacts)
    extra = [] if selected else ["--fleet-inventory-only"]
    with peer(directory, command, contacts):
        result = CliRunner().invoke(app, argv(command, inventory, *extra, output))
    assert result.exit_code == 0, (result.output, result.exception)
    assert contacts.count("recall.daemon_status") == int(selected)


@pytest.mark.parametrize("command", ["list", "search", "usage"])
@pytest.mark.parametrize("output", ["json", "jsonl", "toon", "text"])
def test_fleet_formats_keep_selected_contributions(fleet_fixture, monkeypatch, command, output):
    directory, inventory, contacts = fleet_fixture
    remote_fixture(monkeypatch, command, contacts)
    with peer(directory, command, contacts):
        result = CliRunner().invoke(app, argv(command, inventory, "--format", output))
    assert result.exit_code == 0, (result.output, result.exception)
    if output == "json":
        assert_contributions(command, json.loads(result.stdout), ["aem5", "edge"])
    elif command == "usage":
        assert "aem5" in result.stdout and "edge" in result.stdout
    else:
        assert "caller-item" in result.stdout and "edge-item" in result.stdout


@pytest.mark.parametrize("status_error", [False, True])
def test_fleet_notice_status_is_best_effort(fleet_fixture, monkeypatch, status_error):
    directory, inventory, contacts = fleet_fixture
    remote_fixture(monkeypatch, "search", contacts)
    with peer(
        directory,
        "search",
        contacts,
        status_error=status_error,
        status_value={
            "version_drift": True,
            "daemon_version": "0.1",
            "binary_version": "0.2",
            "runtime_status": {},
        },
    ):
        result = CliRunner().invoke(app, argv("search", inventory, "--json"))
    assert result.exit_code == 0, result.output
    assert_contributions("search", json.loads(result.stdout), ["aem5", "edge"])
    assert contacts.count("recall.daemon_status") == 1
    assert result.stderr.count("running 0.1") == int(not status_error)
    assert "fleet: skip" not in result.stderr


@pytest.mark.parametrize("output", ["json", "text"])
def test_stats_skills_default_unavailable_never_starts_or_snapshots(
    fleet_fixture,
    monkeypatch,
    output,
):
    from recall.core.poll_reads import PollSnapshot, PollSnapshotSelected

    directory, inventory, contacts = fleet_fixture
    remote_fixture(monkeypatch, "skills", contacts)
    effects = []

    def select(config, request):
        effects.append("snapshot-selection")
        raise PollSnapshotSelected(config, request, "scratch-inactive-service")

    def execute(*args):
        effects.extend(["snapshot-execution", "snapshot-storage-read"])
        return PollSnapshot(payload("skills", "caller-item"), None, None)

    def start(*args, **kwargs):
        effects.append("startup")
        raise RpcConnectionError(message="contained startup")

    monkeypatch.setattr("recall.cli.daemon.select_poll_snapshot", select)
    monkeypatch.setattr("recall.cli.daemon.execute_poll_snapshot", execute)
    monkeypatch.setattr("recall.core.rpc_client.RpcClient._auto_fork_daemon", start)
    result = CliRunner().invoke(app, argv("skills", inventory, "--format", output))
    assert effects == []
    assert result.exit_code == 1
    if output == "text":
        assert result.stdout == ""
    else:
        assert "error" in json.loads(result.stdout)
        assert "rows" not in json.loads(result.stdout)
    assert not (directory / "recall.sock").exists()
    ordinary = CliRunner().invoke(app, ["stats", "skills", "--local", "--json"])
    assert ordinary.exit_code == 0, ordinary.output
    assert effects == ["snapshot-selection", "snapshot-execution", "snapshot-storage-read"]
    assert json.loads(ordinary.stdout)["rows"] == payload("skills", "caller-item")["rows"]
