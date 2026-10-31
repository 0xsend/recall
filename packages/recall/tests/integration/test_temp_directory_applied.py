from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.db.connection import connect, connect_readonly


def _temp_directory_setting(conn: duckdb.DuckDBPyConnection) -> Path:
    row = conn.execute("SELECT current_setting('temp_directory')").fetchone()
    if row is None:
        raise AssertionError("DuckDB did not return current temp_directory setting")
    return Path(str(row[0]))


@pytest.fixture
def config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> AppConfig:
    monkeypatch.delenv("RECALL_DUCKDB_TEMP_DIR", raising=False)
    monkeypatch.delenv("RECALL_DUCKDB_MEMORY_LIMIT", raising=False)
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.delenv("RECALL_DB_PATH", raising=False)
    monkeypatch.delenv("RECALL_LOCK_PATH", raising=False)
    return AppConfig.load()


def test_connect_creates_spill_dir_with_pragma_applied(config: AppConfig, tmp_path: Path) -> None:
    expected_spill_dir = tmp_path / "data" / "duckdb_spill"

    conn = connect(config)
    try:
        assert expected_spill_dir.is_dir()
        assert _temp_directory_setting(conn) == expected_spill_dir
    finally:
        conn.close()


@pytest.mark.parametrize("opener", [connect, connect_readonly], ids=["read-write", "read-only"])
def test_reconnect_after_read_spills(
    config: AppConfig,
    tmp_path: Path,
    opener: Callable[[AppConfig], duckdb.DuckDBPyConnection],
) -> None:
    with connect(config):
        pass

    profile_path = tmp_path / "spill-profile.json"
    with opener(config) as conn:
        conn.execute("SET threads = 1")
        conn.execute("SET memory_limit = '16MiB'")
        conn.execute("SET enable_profiling = 'json'")
        conn.execute("SET profiling_output = ?", [str(profile_path)])
        assert conn.execute(
            "SELECT sum(rn) FROM ("
            "SELECT row_number() OVER (ORDER BY hash(i)) rn FROM range(1000000) t(i))"
        ).fetchall() == [(500000500000,)]
        conn.execute("PRAGMA disable_profiling")
        assert json.loads(profile_path.read_text())["system_peak_temp_dir_size"] > 0

        for _ in range(2):
            with opener(config) as reopened:
                assert _temp_directory_setting(reopened) == config.data_dir / "duckdb_spill"
                assert reopened.execute("SELECT count(*) FROM schema_version").fetchone() == (1,)
        assert conn.execute("SELECT count(*) FROM schema_version").fetchone() == (1,)
