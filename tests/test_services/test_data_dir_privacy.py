"""The data directory and RPC socket are private to the owning user."""

from __future__ import annotations

import asyncio
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from recall.core.config import AppConfig
from recall.db import connect
from recall.services.rpc_server import RpcServer


@pytest.fixture
def short_home() -> Iterator[Path]:
    # macOS pytest paths exceed the Unix socket path limit.
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="recall-priv-") as directory:
        yield Path(directory)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _load_config(home: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("RECALL_DATA_DIR", str(data_dir))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(home / "config.toml"))
    monkeypatch.delenv("RECALL_DB_PATH", raising=False)
    monkeypatch.delenv("RECALL_LOCK_PATH", raising=False)
    return AppConfig.load()


def test_connect_creates_missing_data_dir_owner_only(
    short_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = short_home / "share" / "recall"
    config = _load_config(short_home, data_dir, monkeypatch)

    connect(config).close()

    assert _mode(data_dir) == 0o700


def test_connect_leaves_existing_data_dir_mode_alone(
    short_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = short_home / "recall"
    data_dir.mkdir(mode=0o755)
    data_dir.chmod(0o755)
    config = _load_config(short_home, data_dir, monkeypatch)

    connect(config).close()

    assert _mode(data_dir) == 0o755


async def _socket_modes_while_listening(server: RpcServer) -> tuple[int, int]:
    """Start *server*, read the data-dir and socket modes once it listens, then stop it."""
    task = asyncio.create_task(server.start(watch=False))
    try:
        async with asyncio.timeout(10):
            while server._server is None:
                if task.done():
                    await task
                    raise AssertionError("server exited before listening")
                await asyncio.sleep(0.01)
        return _mode(server.socket_path.parent), _mode(server.socket_path)
    finally:
        server.request_shutdown()
        await asyncio.wait_for(task, timeout=10)
        await server.stop()


def test_rpc_server_creates_private_data_dir_and_owner_only_socket(
    short_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = short_home / "data"
    config = _load_config(short_home, data_dir, monkeypatch)

    dir_mode, socket_mode = asyncio.run(_socket_modes_while_listening(RpcServer(config=config)))

    assert dir_mode == 0o700
    assert socket_mode == 0o600


def test_rpc_socket_is_owner_only_in_existing_shared_dir(
    short_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = short_home / "data"
    data_dir.mkdir(mode=0o755)
    data_dir.chmod(0o755)
    config = _load_config(short_home, data_dir, monkeypatch)

    dir_mode, socket_mode = asyncio.run(_socket_modes_while_listening(RpcServer(config=config)))

    assert dir_mode == 0o755
    assert socket_mode == 0o600
