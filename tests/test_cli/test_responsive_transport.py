"""Responsive transport expiry must not become successful offline status."""

from __future__ import annotations

import json
import socket

import pytest
from recall.cli.app import app
from recall.cli.rpc import DaemonUnreachableError, rpc_call_or_error
from recall.core.config import AppConfig
from recall.core.rpc_client import RpcClient
from typer.testing import CliRunner


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
