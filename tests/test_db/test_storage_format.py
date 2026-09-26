"""REQ-RECON-017: new files use a fixed format; ordinary opens preserve old files."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.db.connection import connect, connect_readonly


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.setenv("HOME", str(tmp_path))
    return replace(
        AppConfig.load(),
        data_dir=tmp_path,
        db_path=tmp_path / "recall.duckdb",
        lock_path=tmp_path / "recall.lock",
    )


def reopened_format(path: Path) -> str:
    """A fresh process observes the persisted header, not a requested ATTACH tag."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import duckdb,json,sys; "
            "c=duckdb.connect(sys.argv[1],read_only=True); "
            "print(json.dumps(c.execute(\"SELECT tags['storage_version'] "
            'FROM duckdb_databases() WHERE database_name=current_database()").fetchone()[0])); '
            "c.close()",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return json.loads(result.stdout)


def test_new_database_persists_the_selected_fixed_format(config: AppConfig) -> None:
    conn = connect(config)
    conn.execute("CHECKPOINT")
    conn.close()
    assert reopened_format(config.db_path) == "v1.2.0+"


def test_new_database_allows_later_ordinary_connections(config: AppConfig) -> None:
    first = connect(config)
    second = connect(config)
    assert second.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
    second.close()
    assert first.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
    first.close()


def test_existing_legacy_database_keeps_its_format_on_ordinary_open(config: AppConfig) -> None:
    conn = duckdb.connect(str(config.db_path))
    conn.execute("CREATE TABLE owned_marker AS SELECT 7 AS value")
    conn.close()
    assert reopened_format(config.db_path) == "v1.0.0+"
    conn = connect(config)
    assert conn.execute("SELECT value FROM owned_marker").fetchone() == (7,)
    conn.close()
    assert reopened_format(config.db_path) == "v1.0.0+"
    conn = connect_readonly(config)
    assert conn.execute("SELECT value FROM owned_marker").fetchone() == (7,)
    conn.close()
    assert reopened_format(config.db_path) == "v1.0.0+"
