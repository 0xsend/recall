from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.db.connection import connect


def _temp_directory_setting(conn: duckdb.DuckDBPyConnection) -> Path:
    row = conn.execute("SELECT current_setting('temp_directory')").fetchone()
    if row is None:
        raise AssertionError("DuckDB did not return current temp_directory setting")
    return Path(str(row[0]))


def test_connect_creates_spill_dir_with_pragma_applied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RECALL_DUCKDB_TEMP_DIR", raising=False)
    monkeypatch.delenv("RECALL_DUCKDB_MEMORY_LIMIT", raising=False)
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.delenv("RECALL_DB_PATH", raising=False)
    monkeypatch.delenv("RECALL_LOCK_PATH", raising=False)
    config = AppConfig.load()
    expected_spill_dir = tmp_path / "data" / "duckdb_spill"

    conn = connect(config)
    try:
        assert expected_spill_dir.is_dir()
        assert _temp_directory_setting(conn) == expected_spill_dir
    finally:
        conn.close()
