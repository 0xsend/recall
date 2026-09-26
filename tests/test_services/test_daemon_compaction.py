from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import pytest
import recall.services.daemon as daemon_module
from conftest import _can_acquire_duckdb_lock
from recall.core.config import AppConfig, CompactionConfig
from recall.db.schema import ensure_schema
from recall.services.compaction import CompactionError, estimate_bloat_ratio
from recall.services.rpc_server import RpcServer
from test_compaction import _bloat_session_state, _seed_fresh_rows

_TRAILING_BLOAT_BYTES = 10_000_000
_requires_duckdb_lock = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    compaction: CompactionConfig | None = None,
) -> AppConfig:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / ".config/recall/config.toml"))
    monkeypatch.setenv("RECALL_EMBED_BACKEND", "onnx")
    monkeypatch.delenv("RECALL_COMPACTION_AUTO", raising=False)
    monkeypatch.delenv("RECALL_COMPACTION_THRESHOLD", raising=False)
    monkeypatch.delenv("RECALL_COMPACTION_INTERVAL_HOURS", raising=False)
    config = AppConfig.load()
    if compaction is None:
        return config
    return replace(config, compaction=compaction)


def _connect_schema(config: AppConfig) -> duckdb.DuckDBPyConnection:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(config.db_path))
    ensure_schema(conn, embed_dim=config.embedding.dimensions)
    return conn


def _close_server(server: RpcServer) -> None:
    if server._conn is not None:
        server._conn.close()
        server._conn = None


def _append_trailing_bloat(db_path: Path) -> None:
    with db_path.open("ab") as db_file:
        db_file.write(b"\0" * _TRAILING_BLOAT_BYTES)


def test_auto_compact_disabled_when_auto_trigger_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        compaction=CompactionConfig(auto_trigger=False, bloat_ratio_threshold=0.0),
    )
    server = RpcServer(config=config)
    compact_calls = 0

    def compact_spy(_config: AppConfig) -> Any:
        nonlocal compact_calls
        compact_calls += 1
        raise AssertionError("compact must not be called when auto-trigger is disabled")

    monkeypatch.setattr(daemon_module, "compact", compact_spy)

    daemon_module._maybe_run_auto_compact(server, config)

    assert compact_calls == 0
    assert server.last_compact_check_at is None


@_requires_duckdb_lock
def test_auto_compact_skipped_when_ratio_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        compaction=CompactionConfig(bloat_ratio_threshold=999.0),
    )
    conn = _connect_schema(config)
    try:
        _seed_fresh_rows(conn)
    finally:
        conn.close()
    server = RpcServer(config=config)
    compact_calls = 0

    def compact_spy(_config: AppConfig) -> Any:
        nonlocal compact_calls
        compact_calls += 1
        raise AssertionError("compact must not be called below threshold")

    monkeypatch.setattr(daemon_module, "compact", compact_spy)

    daemon_module._maybe_run_auto_compact(server, config)

    assert compact_calls == 0
    assert server.last_compact_check_at is not None


def test_auto_compact_skipped_when_interval_not_elapsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        compaction=CompactionConfig(min_bytes=0, bloat_ratio_threshold=0.0, check_interval_hours=1),
    )
    server = RpcServer(config=config)
    server.last_compact_check_at = 100.0
    monkeypatch.setattr(daemon_module.time, "monotonic", lambda: 100.5)
    compact_calls = 0

    def compact_spy(_config: AppConfig) -> Any:
        nonlocal compact_calls
        compact_calls += 1
        raise AssertionError("compact must not be called before interval elapses")

    monkeypatch.setattr(daemon_module, "compact", compact_spy)

    daemon_module._maybe_run_auto_compact(server, config)

    assert compact_calls == 0
    assert server.last_compact_check_at == 100.0


@_requires_duckdb_lock
def test_auto_compact_fires_when_ratio_exceeds_threshold_and_interval_elapsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        compaction=CompactionConfig(min_bytes=0, bloat_ratio_threshold=0.0, check_interval_hours=1),
    )
    conn = _connect_schema(config)
    try:
        _bloat_session_state(conn, config)
    finally:
        conn.close()
    _append_trailing_bloat(config.db_path)
    before = estimate_bloat_ratio(config.db_path, config)
    assert before.ratio > 5.0
    server = RpcServer(config=config)
    server._get_conn()
    compact_results = []
    actual_compact = daemon_module.compact

    def compact_spy(config_arg: AppConfig) -> Any:
        result = actual_compact(config_arg)
        compact_results.append(result)
        return result

    monkeypatch.setattr(daemon_module, "compact", compact_spy)

    try:
        daemon_module._maybe_run_auto_compact(server, config)
    finally:
        _close_server(server)

    after = estimate_bloat_ratio(config.db_path, config)
    assert len(compact_results) == 1
    assert after.ratio < before.ratio
    assert compact_results[0].after.ratio == pytest.approx(after.ratio)
    # The cached bloat ratio (surfaced by CLI health notices) must be refreshed
    # to the post-compaction value, not left at the pre-compaction ratio — else
    # `daemon status` keeps warning about bloat the compaction already reclaimed.
    cached_ratio = server._last_bloat_ratio
    assert cached_ratio is not None
    assert cached_ratio == pytest.approx(compact_results[0].after.ratio)
    assert cached_ratio < before.ratio


@_requires_duckdb_lock
def test_auto_compact_failure_does_not_crash_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        compaction=CompactionConfig(min_bytes=0, bloat_ratio_threshold=0.0, check_interval_hours=1),
    )
    conn = _connect_schema(config)
    conn.close()
    server = RpcServer(config=config)

    def fail_compact(_config: AppConfig) -> Any:
        raise CompactionError("simulated compaction failure")

    monkeypatch.setattr(daemon_module, "compact", fail_compact)

    try:
        daemon_module._maybe_run_auto_compact(server, config)
    finally:
        _close_server(server)

    assert server.last_compact_check_at is not None


@_requires_duckdb_lock
def test_auto_compact_releases_and_reopens_db_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        compaction=CompactionConfig(min_bytes=0, bloat_ratio_threshold=0.0, check_interval_hours=1),
    )
    conn = _connect_schema(config)
    conn.close()
    server = RpcServer(config=config)
    original_conn = server._get_conn()
    observed_closed_during_compact: list[bool] = []

    def estimate_spy(_db_path: Path, _config: AppConfig) -> Any:
        # file_size must clear CompactionConfig.min_bytes, or the size floor
        # short-circuits before the ratio is ever compared.
        return SimpleNamespace(ratio=2.0, file_size=8 * 1024 * 1024 * 1024)

    def compact_spy(_config: AppConfig) -> Any:
        observed_closed_during_compact.append(server._conn is None)
        return SimpleNamespace(after=SimpleNamespace(ratio=1.0))

    monkeypatch.setattr(daemon_module, "estimate_bloat_ratio", estimate_spy)
    monkeypatch.setattr(daemon_module, "compact", compact_spy)

    try:
        daemon_module._maybe_run_auto_compact(server, config)
        assert observed_closed_during_compact == [True]
        assert server._conn is not None
        assert server._conn is not original_conn
        assert server._conn.execute("SELECT 1").fetchone() == (1,)
    finally:
        _close_server(server)
