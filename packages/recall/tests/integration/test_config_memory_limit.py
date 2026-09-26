from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.db.connection import connect, connect_readonly


def _load_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_toml: str) -> AppConfig:
    data_dir = tmp_path / "data"
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_toml, encoding="utf-8")
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(data_dir))
    monkeypatch.delenv("RECALL_DB_PATH", raising=False)
    monkeypatch.delenv("RECALL_LOCK_PATH", raising=False)
    return AppConfig.load()


def _memory_limit_setting(conn: duckdb.DuckDBPyConnection) -> str:
    row = conn.execute("SELECT current_setting('memory_limit')").fetchone()
    if row is None:
        raise AssertionError("DuckDB did not return current memory_limit setting")
    return str(row[0])


def test_connect_applies_configured_duckdb_memory_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RECALL_DUCKDB_MEMORY_LIMIT", raising=False)
    config = _load_config(
        tmp_path,
        monkeypatch,
        """
[duckdb]
memory_limit = "3GB"
""",
    )

    conn = connect(config)
    try:
        assert _memory_limit_setting(conn) == "3.0 GiB"
    finally:
        conn.close()


def test_connect_readonly_prefers_env_memory_limit_over_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RECALL_DUCKDB_MEMORY_LIMIT", raising=False)
    config = _load_config(
        tmp_path,
        monkeypatch,
        """
[duckdb]
memory_limit = "4GB"
""",
    )
    conn = connect(config)
    conn.close()

    monkeypatch.setenv("RECALL_DUCKDB_MEMORY_LIMIT", "3GB")
    readonly = connect_readonly(config)
    try:
        assert _memory_limit_setting(readonly) == "3.0 GiB"
    finally:
        readonly.close()
