"""Responsive transport expiry must not become successful offline status."""

from __future__ import annotations

import json
import socket

import pytest
from recall.cli.app import app
from recall.cli.rpc import DaemonUnreachableError, rpc_call_or_error
from recall.core.config import AppConfig
from recall.core.rpc_client import RpcClient
from toon_format import decode as toon_decode
from typer.testing import CliRunner


@pytest.mark.parametrize("output_format", ["text", "json", "jsonl", "toon"])
@pytest.mark.parametrize(
    "command",
    [
        ["daemon", "status"],
        ["daemon", "status", "--limit", "1"],
        ["daemon", "status", "--limit", "10"],
        ["daemon", "status", "--params", '{"limit":1}'],
        ["live"],
        ["stats"],
        ["db", "check-indexes"],
        ["db", "check-indexes", "--params", '{"sample":1}'],
    ],
)
def test_socket_admission_busy_is_an_explicit_error(
    monkeypatch, saturated_rpc_listener, command, output_format
):
    monkeypatch.setattr("recall.cli.rpc._in_process_server", None)

    def forbidden_local_read(*_args, **_kwargs):
        pytest.fail("admission refusal must not open a local database or read offline status")

    monkeypatch.setattr("recall.cli.db.connect_readonly", forbidden_local_read)
    monkeypatch.setattr("recall.cli.daemon.offline_daemon_status", forbidden_local_read)
    result = CliRunner().invoke(app, [*command, "--format", output_format])

    assert result.exit_code == 1, result.output
    message = "daemon busy: socket admission temporarily unavailable; retry this command"
    if output_format == "text":
        assert result.stdout == ""
        assert result.stderr == f"error: {message}\n"
    else:
        decode = toon_decode if output_format == "toon" else json.loads
        assert decode(result.stdout) == {
            "error": {
                "code": "RUNTIME",
                "message": message,
                "details": {"kind": "daemon_busy", "phase": "connect", "retryable": True},
            }
        }
        assert result.stderr == ""


@pytest.mark.parametrize("output_format", ["json", "jsonl", "toon"])
@pytest.mark.parametrize(
    "command,projection",
    [
        (["daemon", "status"], ["--fields", "daemon_pid"]),
        (["daemon", "status"], ["--params", '{"limit":1,"fields":"daemon_pid"}']),
        (["db", "check-indexes"], ["--fields", "complete"]),
        (["db", "check-indexes"], ["--params", '{"sample":1,"fields":"complete"}']),
    ],
)
def test_busy_error_survives_projection(
    monkeypatch, saturated_rpc_listener, output_format, command, projection
):
    test_socket_admission_busy_is_an_explicit_error(
        monkeypatch, saturated_rpc_listener, [*command, *projection], output_format
    )


def test_busy_live_mark_is_a_nonfatal_skip(monkeypatch, saturated_rpc_listener):
    monkeypatch.setattr("recall.cli.rpc._in_process_server", None)
    result = CliRunner().invoke(
        app, ["live", "mark", "--session", "owned", "--pid", "4242", "--json"]
    )
    assert result.exit_code == 0, result.output
    message = "daemon busy: socket admission temporarily unavailable; retry this command"
    assert json.loads(result.stdout) == {
        "marked": False,
        "source_session_id": "owned",
        "pid": 4242,
        "reason": message,
    }
    from recall.cli.live import MARK_SKIPPED_NOTE

    assert result.stderr == MARK_SKIPPED_NOTE.format(reason=message) + "\n"


def test_busy_advisory_does_not_replace_a_successful_command(monkeypatch, saturated_rpc_listener):
    from recall.cli import stats as stats_cli

    monkeypatch.setattr("recall.cli.rpc._in_process_server", None)
    monkeypatch.setattr(stats_cli, "rpc_call_or_error", lambda *_args, **_kwargs: {})
    result = CliRunner().invoke(app, ["stats", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {}
    assert result.stderr == ""


@pytest.mark.parametrize(
    "command, stage",
    [
        (command, stage)
        for command in (["daemon", "status"], ["live"])
        for stage in ("connect", "retry", "send")
        if command == ["live"] or stage != "retry"
    ],
)
def test_responsive_transport_expiry_is_an_error(monkeypatch, stage, command):
    monkeypatch.setattr("recall.cli.rpc._in_process_server", None)
    config = AppConfig.load()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / "recall.sock").touch()
    client = RpcClient(config=config)
    monkeypatch.setattr("recall.cli.rpc.RpcClient", lambda **_kwargs: client)
    connections = 0

    def connect(_socket, _address):
        nonlocal connections
        connections += 1
        if stage == "retry" and connections == 1:
            raise ConnectionRefusedError("stale socket")
        if stage != "send":
            raise TimeoutError("socket deadline")

    def sendall(_socket, _data):
        raise TimeoutError("socket deadline")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "sendall", sendall)
    monkeypatch.setattr(client, "_auto_fork_daemon", lambda **_kwargs: None)
    result = CliRunner().invoke(app, [*command, "--json"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "RUNTIME"
    assert "responsive read deadline expired" in payload["error"]["message"]
    assert "reconciliation" not in payload
    assert "runtime coverage unavailable" not in result.output
    assert client._sock is None


@pytest.mark.parametrize("idle_timeout, seconds", [(None, 30), (7.0, 7)])
def test_ordinary_rpc_reports_default_or_explicit_silence_limit(monkeypatch, idle_timeout, seconds):
    monkeypatch.setattr("recall.cli.rpc._in_process_server", None)
    local, peer = socket.socketpair()
    client = RpcClient()
    client._sock = local
    monkeypatch.setattr("recall.cli.rpc.RpcClient", lambda **_kwargs: client)

    def silent(_socket, _size):
        raise TimeoutError("no frame")

    monkeypatch.setattr(socket.socket, "recv", silent)
    try:
        with pytest.raises(DaemonUnreachableError) as exc_info:
            rpc_call_or_error("recall.list", idle_timeout=idle_timeout)
        assert f"no frames from the daemon for {seconds}s" in exc_info.value.message
        assert "idle timeout, not a total deadline" in exc_info.value.message
        assert client._sock is None
    finally:
        client.close()
        peer.close()
