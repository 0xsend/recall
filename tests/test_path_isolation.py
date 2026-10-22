from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.services.rpc_server import RpcServer
from recall.services.self_repair import read_failure_marker

# Collection happens before autouse fixtures. Defaults must already be isolated.
_COLLECTED_CONFIG = AppConfig.load()


def test_default_config_and_server_are_isolated(recall_test_root: Path) -> None:
    for config in (_COLLECTED_CONFIG, AppConfig.load(), RpcServer()._config):
        for path in (config.data_dir, config.config_path, config.db_path, config.lock_path):
            assert path.resolve().is_relative_to(recall_test_root)


def test_default_server_fatal_marker_is_inside_fake_home() -> None:
    server = RpcServer()
    assert server._config.data_dir == Path.home() / ".local/share/recall"
    assert server._stop_on_fatal_db(duckdb.FatalException("isolated failure"), "isolation")
    marker = read_failure_marker(Path.home() / ".local/share/recall")
    assert marker is not None and marker.failure is not None
    assert marker.failure.message == "isolated failure"


@pytest.mark.parametrize(
    "name", ["RECALL_DATA_DIR", "RECALL_CONFIG_PATH", "RECALL_DB_PATH", "RECALL_LOCK_PATH"]
)
def test_resolver_rejects_external_path(
    name: str, recall_test_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = recall_test_root.parent / "outside-recall-test-root"
    monkeypatch.setenv(name, str(outside))
    with pytest.raises(pytest.fail.Exception, match=f"recall test isolation: {name}=.*escapes"):
        AppConfig.load()


@pytest.mark.parametrize("name", ["data_dir", "config_path", "db_path", "lock_path"])
def test_explicit_config_rejects_external_path(name: str, recall_test_root: Path) -> None:
    config = AppConfig.load()
    outside = recall_test_root / ".." / "outside-recall-test-root"
    with pytest.raises(pytest.fail.Exception, match=f"recall test isolation: {name}=.*escapes"):
        replace(config, **{name: outside})


def test_config_symlink_cannot_escape_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "external.toml"
    link.symlink_to(Path(__file__).resolve().parents[1] / "pyproject.toml")
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(link))
    with pytest.raises(pytest.fail.Exception, match=r"RECALL_CONFIG_PATH=.*escapes"):
        AppConfig.load()


def test_default_resolution_uses_an_explicit_fake_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "explicit-home"
    monkeypatch.setenv("HOME", str(home))
    for name in ("RECALL_DATA_DIR", "RECALL_CONFIG_PATH", "RECALL_DB_PATH", "RECALL_LOCK_PATH"):
        monkeypatch.delenv(name, raising=False)
    config = AppConfig.load()
    assert config.data_dir == home / ".local/share/recall"
    assert config.db_path == home / ".local/share/recall/recall.duckdb"
    assert config.lock_path == home / ".local/share/recall/recall.lock"
    assert config.config_path == home / ".config/recall/config.toml"
    assert not any(name in os.environ for name in ("RECALL_DB_PATH", "RECALL_LOCK_PATH"))


@pytest.mark.parametrize("escape", [False, True])
def test_pytest_isolates_collection_and_fails_on_escape(tmp_path: Path, escape: bool) -> None:
    """A fresh pytest process replaces caller paths before importing test code."""
    caller_home = tmp_path / "caller-home"
    caller_data = caller_home / ".local/share/recall"
    caller_config = caller_home / ".config/recall/config.toml"
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "import duckdb\n"
        "from recall.core.config import AppConfig\n"
        "from recall.services.rpc_server import RpcServer\n"
        "server = RpcServer()\n"
        "assert not server._config.data_dir.is_relative_to(Path(os.environ['CALLER_HOME']))\n"
        "assert server._stop_on_fatal_db(duckdb.FatalException('probe failure'), 'probe')\n"
        "assert (server._config.data_dir / 'daemon-failure.json').is_file()\n"
        + (
            "os.environ['RECALL_CONFIG_PATH'] = os.environ['CALLER_CONFIG']\nAppConfig.load()\n"
            if escape
            else ""
        )
        + "def test_probe(): pass\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "HOME": str(caller_home),
        "CALLER_HOME": str(caller_home),
        "CALLER_CONFIG": str(caller_config),
        "RECALL_DATA_DIR": str(caller_data),
        "RECALL_CONFIG_PATH": str(caller_config),
        "RECALL_DB_PATH": str(caller_data / "recall.duckdb"),
        "RECALL_LOCK_PATH": str(caller_data / "recall.lock"),
        "PYTHONPATH": str(root),
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "conftest", "--noconftest", str(probe)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if escape:
        assert result.returncode == 2, result.stdout + result.stderr
        assert "recall test isolation: RECALL_CONFIG_PATH=" in result.stdout
        assert "escapes session root" in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
    assert not caller_data.exists()
    assert not caller_config.exists()
