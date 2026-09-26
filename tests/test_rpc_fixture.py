from __future__ import annotations

import duckdb
import pytest
from conftest import _RpcServerLifecycle
from recall.cli import rpc as rpc_module
from recall.core.config import AppConfig
from recall.services.rpc_server import RpcServer


def test_autouse_rpc_server_stays_unrealized_without_a_cli_call() -> None:
    lazy = rpc_module._in_process_server

    assert lazy._server is None


def test_lazy_rpc_server_loads_test_environment_at_first_use(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    data_dir = tmp_path / ".local/share/recall"
    monkeypatch.setenv("RECALL_DATA_DIR", str(data_dir))

    lazy = rpc_module._in_process_server
    server = lazy._real

    assert server._config.data_dir == data_dir


def test_rpc_lifecycle_closes_each_replacement_connection_once(tmp_path, monkeypatch) -> None:
    """Teardown owns replacements even when a later server supersedes one."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

    config = AppConfig.load()
    first = RpcServer(config=config)
    first_conn = first._get_conn()
    second = RpcServer(config=config)
    second_conn = second._get_conn()

    lifecycle = _RpcServerLifecycle()
    lifecycle.install(first)
    lifecycle.install(first)
    lifecycle.install(second)
    lifecycle.cleanup()

    assert first._conn is None
    assert second._conn is None
    with pytest.raises(duckdb.ConnectionException):
        first_conn.execute("SELECT 1")
    with pytest.raises(duckdb.ConnectionException):
        second_conn.execute("SELECT 1")
    assert rpc_module._in_process_server is None
