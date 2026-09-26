from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    DuckDBConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.db.connection import (
    WAL_CHECKPOINT_HIGH_WATER_BYTES,
    _apply_runtime_pragmas,
    checkpoint_wal_if_due,
    wal_size_bytes,
)


def _config(
    tmp_path: Path,
    *,
    memory_limit: str | None = None,
    temp_directory: str | None = None,
) -> AppConfig:
    data_dir = tmp_path / "data"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
        duckdb=DuckDBConfig(memory_limit=memory_limit, temp_directory=temp_directory),
    )


def _settings(conn: duckdb.DuckDBPyConnection) -> tuple[str, str, str]:
    row = conn.execute(
        "SELECT current_setting('memory_limit'), current_setting('temp_directory'), "
        "current_setting('wal_autocheckpoint')"
    ).fetchone()
    if row is None:
        raise AssertionError("DuckDB did not return runtime pragma settings")
    return str(row[0]), str(row[1]), str(row[2])


def test_apply_runtime_pragmas_sets_both_pragmas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spill_path = tmp_path / "configured-spill"
    monkeypatch.delenv("RECALL_DUCKDB_MEMORY_LIMIT", raising=False)
    monkeypatch.delenv("RECALL_DUCKDB_TEMP_DIR", raising=False)
    config = _config(tmp_path, memory_limit="3GB", temp_directory=str(spill_path))
    conn = duckdb.connect(":memory:")
    try:
        _apply_runtime_pragmas(conn, config)

        memory_limit, temp_directory, wal_autocheckpoint = _settings(conn)
        assert memory_limit == "3.0 GiB"
        assert Path(temp_directory) == spill_path
        assert wal_autocheckpoint == "256.0 MiB"
    finally:
        conn.close()


def test_checkpoint_wal_if_due_persists_a_real_wal(tmp_path: Path) -> None:
    db_path = tmp_path / "checkpoint.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("SET wal_autocheckpoint='256MiB'")
        conn.execute("CREATE TABLE durable_rows AS SELECT range AS id FROM range(100000)")
        before = wal_size_bytes(db_path)
        assert before > 0

        skipped = checkpoint_wal_if_due(conn, db_path, high_water_bytes=before + 1)
        assert skipped.attempted is False
        assert wal_size_bytes(db_path) == before

        completed = checkpoint_wal_if_due(conn, db_path, high_water_bytes=1)
        assert completed.attempted is True
        assert completed.wal_bytes_before == before
        assert completed.wal_bytes_after < before
        assert completed.duration_seconds >= 0
    finally:
        conn.close()

    reopened = duckdb.connect(str(db_path), read_only=True)
    try:
        assert reopened.execute("SELECT COUNT(*) FROM durable_rows").fetchone() == (100000,)
    finally:
        reopened.close()


def test_checkpoint_high_water_keeps_engine_fallback_headroom() -> None:
    assert WAL_CHECKPOINT_HIGH_WATER_BYTES == 64 * 1024 * 1024
    assert WAL_CHECKPOINT_HIGH_WATER_BYTES < 256 * 1024 * 1024
