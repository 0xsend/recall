from __future__ import annotations

import os
import socket

import pytest
from recall.core.config import AppConfig
from recall.core.rpc_client import STALE_COMPACTION_SENTINEL_AGE, RpcClient, RpcConnectionError


def _load_config_with_data_dir(tmp_path, monkeypatch) -> AppConfig:
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path))
    config = AppConfig.load()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    return config


def test_connect_refuses_autofork_when_sentinel_present(tmp_path, monkeypatch) -> None:
    config = _load_config_with_data_dir(tmp_path, monkeypatch)
    (config.data_dir / "recall.compacting").touch()
    client = RpcClient(config=config)

    monkeypatch.setattr(
        client,
        "_auto_fork_daemon",
        lambda: pytest.fail("fresh compaction sentinel must block auto-fork"),
    )

    with pytest.raises(RpcConnectionError) as exc_info:
        client.connect(auto_fork=True)

    assert "compact is in progress" in exc_info.value.message


def test_connect_ignores_stale_sentinel(socket_dir, monkeypatch) -> None:
    config = _load_config_with_data_dir(socket_dir, monkeypatch)
    sentinel = config.data_dir / "recall.compacting"
    sentinel.touch()
    stale_mtime = sentinel.stat().st_mtime - STALE_COMPACTION_SENTINEL_AGE - 1
    os.utime(sentinel, (stale_mtime, stale_mtime))
    client = RpcClient(config=config)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.settimeout(2)

        def start_daemon(**_kwargs):
            listener.bind(str(client.socket_path))
            listener.listen(1)

        monkeypatch.setattr(client, "_auto_fork_daemon", start_daemon)
        try:
            client.connect()
            peer, _ = listener.accept()
            with peer:
                client.close()
                assert peer.recv(1) == b""
        finally:
            client.close()
