from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import pytest
import recall.services.daemon as daemon_module
from recall.core.config import DEFAULT_COMPACTION_BLOAT_THRESHOLD, AppConfig, CompactionConfig
from recall.db.schema import ensure_schema
from recall.services.compaction import CompactionError, estimate_bloat_ratio
from recall.services.rpc_server import RpcServer
from test_compaction import _seed_fresh_rows

from conftest import _can_acquire_duckdb_lock

# One DuckDB row group holds 122,880 rows, so this table is a single row group
# and every commit that touches it makes the next checkpoint rewrite all of
# it: the shape of a small host whose busiest tables fit in one row group.
_EMBEDDING_DIMENSIONS = 128
_SINGLE_ROW_GROUP_ROWS = 60_000
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


def _single_row_group_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    base = _config(
        tmp_path,
        monkeypatch,
        compaction=CompactionConfig(
            min_bytes=0,
            bloat_ratio_threshold=DEFAULT_COMPACTION_BLOAT_THRESHOLD,
            check_interval_hours=1,
        ),
    )
    return replace(base, embedding=replace(base.embedding, dimensions=_EMBEDDING_DIMENSIONS))


def _seed_single_row_group(config: AppConfig) -> None:
    """Build a freshly written file: one dense row group, no free blocks."""
    conn = _connect_schema(config)
    try:
        conn.execute(
            f"""
            INSERT INTO tool_call_embeddings
            SELECT 'tc' || i,
                   list_transform(
                       range(0, {_EMBEDDING_DIMENSIONS}), x -> (random() + i)::FLOAT
                   )::FLOAT[{_EMBEDDING_DIMENSIONS}]
            FROM range(0, {_SINGLE_ROW_GROUP_ROWS}) t(i)
            """
        )
    finally:
        conn.close()


def _rewrite_one_embedding(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO tool_call_embeddings "
        "SELECT * FROM tool_call_embeddings WHERE tool_call_id = 'tc5'"
    )


def _table_fingerprints(db_path: Path) -> dict[str, tuple[int, int]]:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        tables = [
            str(row[0])
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' "
                "ORDER BY table_name"
            ).fetchall()
        ]
        fingerprints = {}
        for table in tables:
            row = conn.execute(
                f'SELECT count(*), coalesce(bit_xor(hash(t)), 0) FROM main."{table}" AS t'
            ).fetchone()
            assert row is not None
            fingerprints[table] = (int(row[0]), int(row[1]))
        return fingerprints


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
def test_auto_compact_fires_on_dead_row_churn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _single_row_group_config(tmp_path, monkeypatch)
    _seed_single_row_group(config)
    # INSERT OR REPLACE leaves the replaced versions dead inside row groups
    # that still hold live rows: the churn behind every real bloat incident.
    conn = duckdb.connect(str(config.db_path))
    try:
        for _ in range(3):
            conn.execute(
                "INSERT OR REPLACE INTO tool_call_embeddings "
                "SELECT * FROM tool_call_embeddings "
                f"WHERE tool_call_id IN "
                f"(SELECT 'tc' || i FROM range(0, {_SINGLE_ROW_GROUP_ROWS // 2}) t(i))"
            )
            conn.execute("CHECKPOINT")
    finally:
        conn.close()
    before = estimate_bloat_ratio(config.db_path, config)
    assert before.ratio >= DEFAULT_COMPACTION_BLOAT_THRESHOLD
    expected = _table_fingerprints(config.db_path)
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
    assert after.ratio < DEFAULT_COMPACTION_BLOAT_THRESHOLD
    assert after.file_size < before.file_size
    assert compact_results[0].after.ratio == pytest.approx(after.ratio)
    assert _table_fingerprints(config.db_path) == expected
    # The cached bloat ratio (surfaced by CLI health notices) must be refreshed
    # to the post-compaction value, not left at the pre-compaction ratio — else
    # `daemon status` keeps warning about bloat the compaction already reclaimed.
    assert server._last_bloat_ratio == pytest.approx(compact_results[0].after.ratio)


@_requires_duckdb_lock
def test_auto_compact_check_does_not_compact_its_own_checkpoint_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check closes the shared connection, and DuckDB checkpoints on close.

    That checkpoint rewrites every row group a commit touched into fresh blocks
    before freeing the old ones, so after one ordinary write the file is about
    twice the live data while holding no dead rows. A check that counted the
    freed blocks compacted, and compaction leaves no free blocks, so the next
    write repeated it at every check.
    """
    config = _single_row_group_config(tmp_path, monkeypatch)
    _seed_single_row_group(config)
    fresh = estimate_bloat_ratio(config.db_path, config)
    server = RpcServer(config=config)
    compact_calls = 0

    def compact_spy(_config: AppConfig) -> Any:
        nonlocal compact_calls
        compact_calls += 1
        raise CompactionError("compact must not run on checkpoint headroom")

    monkeypatch.setattr(daemon_module, "compact", compact_spy)

    file_sizes = []
    try:
        for _cycle in range(3):
            _rewrite_one_embedding(server._get_conn())
            server.last_compact_check_at = None
            daemon_module._maybe_run_auto_compact(server, config)
            file_sizes.append(config.db_path.stat().st_size)
            assert compact_calls == 0
            assert server._conn is not None
            assert server._last_bloat_ratio == pytest.approx(fresh.ratio, abs=0.1)
    finally:
        _close_server(server)

    # The headroom is real: the close-checkpoint grew the file well past the
    # threshold's multiple of the live data, which is what used to trigger.
    assert min(file_sizes) > DEFAULT_COMPACTION_BLOAT_THRESHOLD * fresh.live_bytes


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
