from __future__ import annotations

import os

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


def test_connect_ignores_stale_sentinel(tmp_path, monkeypatch) -> None:
    config = _load_config_with_data_dir(tmp_path, monkeypatch)
    sentinel = config.data_dir / "recall.compacting"
    sentinel.touch()
    stale_mtime = sentinel.stat().st_mtime - STALE_COMPACTION_SENTINEL_AGE - 1
    os.utime(sentinel, (stale_mtime, stale_mtime))
    calls: list[str] = []

    class FakeSocket:
        def settimeout(self, _timeout: float) -> None:
            calls.append("settimeout")

        def connect(self, _path: str) -> None:
            calls.append("connect")

        def close(self) -> None:
            calls.append("close")

    client = RpcClient(config=config)

    def auto_fork() -> None:
        calls.append("auto_fork")
        client.socket_path.touch()

    monkeypatch.setattr(client, "_auto_fork_daemon", auto_fork)
    monkeypatch.setattr("recall.core.rpc_client.socket.socket", lambda *_args: FakeSocket())

    client.connect(auto_fork=True)

    assert calls == ["auto_fork", "settimeout", "connect"]
    assert client._sock is not None
