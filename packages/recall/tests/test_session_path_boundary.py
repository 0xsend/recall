from __future__ import annotations

from pathlib import Path

from recall.core.config import AppConfig
from recall.services.rpc_server import RpcServer


def test_package_selection_inherits_session_isolation(recall_test_root: Path) -> None:
    """Selecting package tests alone must retain the repository's path boundary."""
    assert Path.home().resolve().is_relative_to(recall_test_root)
    for config in (AppConfig.load(), RpcServer()._config):
        for path in (config.data_dir, config.config_path, config.db_path, config.lock_path):
            assert path.resolve().is_relative_to(recall_test_root)
