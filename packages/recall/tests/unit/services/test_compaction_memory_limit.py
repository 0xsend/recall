from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import duckdb
import pytest
from recall.cli import compact as compact_cli
from recall.cli.app import app
from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    DuckDBConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.db.connection import connect
from recall.services import compaction as compaction_module
from recall.services.compaction import compact
from typer.testing import CliRunner

runner = CliRunner()


class TrackingConnection:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        path: Path,
        read_only: bool,
    ) -> None:
        self._conn = conn
        self.path = path
        self.read_only = read_only
        self.memory_limit_statements: list[str] = []

    def execute(self, query: str, *args: Any, **kwargs: Any) -> duckdb.DuckDBPyConnection:
        if query.startswith("SET memory_limit"):
            self.memory_limit_statements.append(query)
        return self._conn.execute(query, *args, **kwargs)

    def close(self) -> None:
        self._conn.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def _config(tmp_path: Path, *, memory_limit: str | None = None) -> AppConfig:
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
        duckdb=DuckDBConfig(memory_limit=memory_limit),
    )


def _track_raw_compaction_connections(monkeypatch: pytest.MonkeyPatch) -> list[TrackingConnection]:
    real_connect = duckdb.connect
    tracked: list[TrackingConnection] = []

    def connect_spy(database: str, *args: Any, **kwargs: Any) -> TrackingConnection:
        raw_conn = real_connect(database, *args, **kwargs)
        tracking_conn = TrackingConnection(
            raw_conn,
            path=Path(database),
            read_only=bool(kwargs.get("read_only", False)),
        )
        tracked.append(tracking_conn)
        return tracking_conn

    monkeypatch.setattr(compaction_module.duckdb, "connect", connect_spy)
    return tracked


def test_compact_applies_resolved_memory_limit_to_raw_duckdb_connections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, memory_limit="4GB")
    conn = connect(config)
    conn.close()

    monkeypatch.setenv("RECALL_DUCKDB_MEMORY_LIMIT", "3GB")
    tracked = _track_raw_compaction_connections(monkeypatch)

    result = compact(config)

    expected = "SET memory_limit = '3GiB'"
    compact_path = config.db_path.with_name(f"{config.db_path.name}.compact")
    assert result.replaced is True
    assert [conn.memory_limit_statements for conn in tracked] == [
        [expected],
        [expected],
        [expected],
        [expected],
    ]
    assert [(conn.path, conn.read_only) for conn in tracked] == [
        (config.db_path, True),  # estimate the source
        (config.db_path, True),  # retain the source storage format
        (compact_path, False),
        (compact_path, True),
    ]


def test_compact_runs_snapshot_gc_after_successful_replace(tmp_path: Path) -> None:
    config = _config(tmp_path)
    conn = connect(config)
    conn.close()
    snapshots = config.data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    stale = snapshots / "stale"
    stale.write_text("stale")
    stale_time = time.time() - (10 * 86_400)
    os.utime(stale, (stale_time, stale_time))

    result = compact(config)

    assert result.replaced is True
    assert not stale.exists()


def test_compact_dry_run_applies_resolved_memory_limit_to_estimate_connection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, memory_limit="4GB")
    conn = connect(config)
    conn.close()

    monkeypatch.setenv("RECALL_DUCKDB_MEMORY_LIMIT", "3GB")
    monkeypatch.setattr(compact_cli.AppConfig, "load", staticmethod(lambda: config))
    tracked = _track_raw_compaction_connections(monkeypatch)

    result = runner.invoke(app, ["compact", "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    assert [conn.memory_limit_statements for conn in tracked] == [["SET memory_limit = '3GiB'"]]
    assert [(conn.path, conn.read_only) for conn in tracked] == [(config.db_path, True)]


def test_compact_cli_estimate_helper_applies_resolved_memory_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, memory_limit="4GB")
    conn = connect(config)
    conn.close()

    monkeypatch.setenv("RECALL_DUCKDB_MEMORY_LIMIT", "3GB")
    tracked = _track_raw_compaction_connections(monkeypatch)

    compact_cli._estimate_bloat_ratio_for_cli(config)

    assert [conn.memory_limit_statements for conn in tracked] == [["SET memory_limit = '3GiB'"]]
    assert [(conn.path, conn.read_only) for conn in tracked] == [(config.db_path, True)]
