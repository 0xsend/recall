from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb
import pytest
import recall.services.compaction as compaction_module
from conftest import _can_acquire_duckdb_lock
from recall.core.config import (
    DEFAULT_COMPACTION_BLOAT_THRESHOLD,
    AppConfig,
    FtsConfig,
)
from recall.db import advisory_lock
from recall.db.connection import STORAGE_VERSION, storage_upgrade_needed, storage_version
from recall.db.fts_sidecar import open_sidecar, sidecar_path, upsert_message_fts
from recall.db.queries import create_fts_indexes
from recall.db.schema import ensure_schema
from recall.services.compaction import (
    DATA_TABLES,
    SINGLETON_TABLES,
    CompactionError,
    _compaction_sentinel,
    _CompactionSentinel,
    compact,
    estimate_bloat_ratio,
)

_BLOAT_MODEL_PAYLOAD_SIZE = 1000
_FRESH_RATIO_CONTENT_SIZE = 5000
_FRESH_RATIO_ROWS = 2000
_STALE_WAL_BYTES = b"stale wal placeholder"

_requires_duckdb_lock = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)


def test_every_schema_table_is_registered_with_compaction() -> None:
    """Compaction drops the tables it knows about and then COPY FROM DATABASE
    recreates them. A table declared in schema.sql but absent from these lists
    survives the drop and makes the copy fail with "table already exists", taking
    every compaction down with it. Keeping the lists exhaustive is the invariant;
    this catches the next table added to schema.sql without registering it."""
    import re as _re

    schema_sql = (
        Path(__file__).resolve().parents[1].parent / "packages/recall/src/recall/db/schema.sql"
    ).read_text(encoding="utf-8")
    declared = set(_re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", schema_sql))
    registered = set(DATA_TABLES) | set(SINGLETON_TABLES)
    assert declared - registered == set(), (
        "tables in schema.sql not registered in compaction DATA_TABLES/SINGLETON_TABLES"
    )
    assert registered - declared == set(), (
        "compaction registers tables that schema.sql does not declare"
    )


def test_compaction_preserves_pending_reconciliation_and_completed_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RECON-001/004: maintenance cannot discard durable work or its checkpoint."""
    from recall.db.source_files import SourceCatalog, SourceSignature

    config = _config_with_fts_backend(tmp_path, monkeypatch, backend="sqlite_sidecar")
    conn = _connect_schema(config)
    catalog = SourceCatalog(conn, clock=lambda: 100.0)
    catalog.observe("codex", "/sources", "/sources/pending.jsonl", SourceSignature(1, 2, 3, 4, 90))
    catalog.acknowledge("codex", "/sources/pending.jsonl", 1, 80, "a" * 64)
    catalog.observe("codex", "/sources", "/sources/pending.jsonl", SourceSignature(1, 2, 5, 6, 100))
    catalog.fail("codex", "/sources/pending.jsonl", "capture changed", {"kind": "source_changed"})
    conn.execute(
        """INSERT INTO reconciliation_roots
           (source, root_path, scan_started_at, scan_finished_at, discovered_count,
            scan_complete, scan_generation)
           VALUES ('codex', '/sources', 98, 99, 1, TRUE, 7)"""
    )
    conn.close()

    result = compact(config)

    with duckdb.connect(str(config.db_path)) as conn:
        row = SourceCatalog(conn, clock=lambda: 100.0).get("codex", "/sources/pending.jsonl")
        assert row is not None
        assert (row.desired_generation, row.committed_generation, row.committed_offset) == (
            2,
            1,
            80,
        )
        assert row.committed_prefix_sha256 == "a" * 64
        assert row.retry_count == 1
        assert row.next_retry_at == 102
        assert row.last_error == "capture changed"
        assert conn.execute(
            "SELECT scan_complete, scan_generation, discovered_count FROM reconciliation_roots"
        ).fetchall() == [(True, 7, 1)]
    assert result.tables_copied["source_files"] == 1
    assert result.tables_copied["reconciliation_roots"] == 1


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / ".config/recall/config.toml"))
    monkeypatch.delenv("RECALL_DB_PATH", raising=False)
    monkeypatch.delenv("RECALL_LOCK_PATH", raising=False)
    monkeypatch.delenv("RECALL_FTS_FIELDS", raising=False)
    monkeypatch.delenv("RECALL_EMBED_BACKEND", raising=False)
    monkeypatch.delenv("RECALL_EMBED_MODEL", raising=False)
    monkeypatch.delenv("RECALL_EMBED_BATCH_SIZE", raising=False)
    return AppConfig.load()


def _config_with_fts_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str,
) -> AppConfig:
    config = _config(tmp_path, monkeypatch)
    return replace(config, fts=FtsConfig(backend=backend))


def _connect_schema(config: AppConfig) -> duckdb.DuckDBPyConnection:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(config.db_path))
    ensure_schema(conn, embed_dim=config.embedding.dimensions)
    return conn


def _table_counts(db_path: Path, tables: tuple[str, ...] = DATA_TABLES) -> dict[str, int]:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        return {table: _count_rows(conn, table) for table in tables}
    finally:
        conn.close()


def _count_rows(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def _legacy_fts_schema_count(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.schemata
        WHERE schema_name LIKE 'fts\\_main\\_%' ESCAPE '\\'
        """
    ).fetchone()
    assert row is not None
    return int(row[0])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _advisory_lock_probe(lock_path: Path) -> subprocess.CompletedProcess[str]:
    script = """
import sys
from pathlib import Path

from recall.db import RecallLockError, advisory_lock

try:
    with advisory_lock(Path(sys.argv[1])):
        pass
except RecallLockError:
    raise SystemExit(23)
"""
    return subprocess.run(
        [sys.executable, "-c", script, str(lock_path)],
        check=False,
        capture_output=True,
        text=True,
    )


def _bloat_session_state(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
    *,
    sessions: int = 50,
    cycles: int = 50,
) -> None:
    """Grow row-version chains so the compacted file has measurable shrinkage."""
    for i in range(sessions):
        sid = f"sess-{i:04d}"
        msg_id = f"msg-{i:04d}"
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) VALUES (?, 'claude_code', ?)",
            [sid, f"/tmp/{sid}.jsonl"],
        )
        conn.execute(
            """
            INSERT INTO session_state (
                session_id, file_mtime, file_size, last_byte_offset
            ) VALUES (?, ?, ?, ?)
            """,
            [sid, 0.0, 0, 0],
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
            [msg_id, sid, 0],
        )
        conn.execute(
            """
            INSERT INTO message_state (
                message_id, role, content, thinking, has_thinking
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [msg_id, "assistant", "initial compaction content", None, False],
        )
    create_fts_indexes(conn, config.fts)
    conn.execute("CHECKPOINT")
    for _cycle in range(cycles):
        payload = f"cycle-{_cycle}-" + ("x" * _BLOAT_MODEL_PAYLOAD_SIZE)
        conn.execute(
            """
            UPDATE session_state
            SET model = ?,
                last_byte_offset = last_byte_offset + 1,
                indexed_at = CURRENT_TIMESTAMP
            """,
            [payload],
        )
    conn.execute("CHECKPOINT")


def _seed_fresh_rows(
    conn: duckdb.DuckDBPyConnection,
    *,
    rows: int = 3,
    content_size: int = 32,
) -> None:
    content = "fresh searchable content " + ("x" * content_size)
    for i in range(rows):
        sid = f"fresh-{i}"
        msg_id = f"fresh-msg-{i}"
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) VALUES (?, 'codex', ?)",
            [sid, f"/tmp/{sid}.jsonl"],
        )
        conn.execute(
            """
            INSERT INTO session_state (
                session_id, message_count, file_mtime, file_size
            ) VALUES (?, ?, ?, ?)
            """,
            [sid, 1, 1.0, 1],
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
            [msg_id, sid, 0],
        )
        conn.execute(
            """
            INSERT INTO message_state (
                message_id, role, content, thinking, has_thinking
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [msg_id, "assistant", f"{content} {i}", None, False],
        )
    conn.execute("CHECKPOINT")


def _seed_fts_rows(conn: duckdb.DuckDBPyConnection) -> None:
    rows = [
        ("fts-alpha", "fts-msg-alpha", "alpha planning database compaction"),
        ("fts-beta", "fts-msg-beta", "beta validates database compaction"),
        ("fts-gamma", "fts-msg-gamma", "gamma unrelated release note"),
    ]
    for sid, msg_id, content in rows:
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) VALUES (?, 'codex', ?)",
            [sid, f"/tmp/{sid}.jsonl"],
        )
        conn.execute(
            """
            INSERT INTO session_state (
                session_id, message_count, file_mtime, file_size
            ) VALUES (?, ?, ?, ?)
            """,
            [sid, 1, 1.0, 1],
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
            [msg_id, sid, 0],
        )
        conn.execute(
            """
            INSERT INTO message_state (
                message_id, role, content, thinking, has_thinking
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [msg_id, "assistant", content, None, False],
        )
    conn.execute("CHECKPOINT")


def _seed_sidecar_row(config: AppConfig) -> None:
    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        upsert_message_fts(
            sidecar_conn,
            "fresh-msg-0",
            "fresh searchable content",
            "",
        )
    finally:
        sidecar_conn.close()


def _message_fts_ids(conn: duckdb.DuckDBPyConnection, query: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT ms.message_id
        FROM message_state ms
        WHERE fts_main_message_state.match_bm25(
            ms.message_id, ?, fields := 'fts_content'
        ) IS NOT NULL
        """,
        [query],
    ).fetchall()
    return {str(row[0]) for row in rows}


def test_estimate_bloat_ratio_zero_for_missing_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    stats = estimate_bloat_ratio(tmp_path / "missing.duckdb", config)

    assert stats.file_size == 0
    assert stats.live_bytes == 0
    assert stats.ratio == 0.0
    assert stats.block_size == 0


@_requires_duckdb_lock
def test_estimate_bloat_ratio_on_fresh_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path, monkeypatch)
    conn = _connect_schema(config)
    try:
        _seed_fresh_rows(
            conn,
            rows=_FRESH_RATIO_ROWS,
            content_size=_FRESH_RATIO_CONTENT_SIZE,
        )
    finally:
        conn.close()

    stats = estimate_bloat_ratio(config.db_path, config)

    assert stats.file_size > 0
    assert stats.live_bytes > 0
    assert stats.block_size > 0
    assert stats.ratio == pytest.approx(1.0, rel=0.75)


@_requires_duckdb_lock
def test_estimate_bloat_ratio_detects_insert_or_replace_churn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dead row-versions from INSERT OR REPLACE stay *referenced* — DuckDB
    reclaims a row group only once all its rows are dead. The old block-count
    measure therefore read a 95%-dead database as pristine (the 1.01x wedge on
    the 145 GiB production DB). Scaling by live/physical rows must surface the
    bloat once churn crosses the compaction threshold, while a dense table
    still reads ~1x (no false trigger)."""
    base = _config(tmp_path, monkeypatch)
    config = replace(base, embedding=replace(base.embedding, dimensions=32))

    # Seed enough varied (low-compressibility) embedding data that the fresh
    # file is dominated by live rows, not fixed schema overhead (a few hundred
    # KB), so a dense database reads ~1x rather than tripping on the baseline.
    conn = _connect_schema(config)
    try:
        conn.execute(
            """
            INSERT INTO tool_call_embeddings
            SELECT 'tc' || i,
                   list_transform(range(0, 32), x -> (random() + i)::FLOAT)::FLOAT[32]
            FROM range(0, 30000) t(i)
            """
        )
        conn.execute("CHECKPOINT")
    finally:
        conn.close()
    # estimate_bloat_ratio opens a read-only connection; the writer must be
    # closed first or DuckDB rejects the differing-configuration open.
    fresh = estimate_bloat_ratio(config.db_path, config)

    # Re-REPLACE only the first half repeatedly: its row groups stay mixed
    # (live + dead) and cannot be reclaimed, mirroring watch-mode churn.
    conn = duckdb.connect(str(config.db_path))
    try:
        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE _churn AS
            SELECT * FROM tool_call_embeddings
            WHERE tool_call_id IN (SELECT 'tc' || i FROM range(0, 15000) t(i))
            """
        )
        for _ in range(15):
            conn.execute("INSERT OR REPLACE INTO tool_call_embeddings SELECT * FROM _churn")
            conn.execute("CHECKPOINT")
    finally:
        conn.close()

    churned = estimate_bloat_ratio(config.db_path, config)

    # At fixture scale (a few MB against 256 KiB blocks) the ratio carries
    # real block-granularity noise -- `fresh` reads ~1.65 with nothing dead --
    # which is exactly why auto-compaction also gates on `min_bytes`. So the
    # invariant worth asserting here is the *separation*, not either side's
    # position relative to the production trigger.
    assert churned.ratio > fresh.ratio
    assert churned.ratio >= DEFAULT_COMPACTION_BLOAT_THRESHOLD


def test_compaction_sentinel_is_process_reentrant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = config.data_dir / "recall.compacting"

    with _compaction_sentinel(config):
        assert sentinel_path.exists()
        with _compaction_sentinel(config):
            assert sentinel_path.exists()
        assert sentinel_path.exists()

    assert not sentinel_path.exists()


def test_compaction_sentinel_creates_missing_parent_dir(tmp_path: Path) -> None:
    sentinel_path = tmp_path / "missing-data-dir" / "recall.compacting"
    assert not sentinel_path.parent.exists()

    with _CompactionSentinel(sentinel_path) as active_path:
        assert active_path == sentinel_path.resolve(strict=False)
        assert sentinel_path.parent.exists()
        assert sentinel_path.exists()

    assert sentinel_path.parent.exists()
    assert not sentinel_path.exists()


@_requires_duckdb_lock
def test_compact_preserves_the_selected_storage_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit maintenance must not downgrade the selected fixed format."""
    config = _config_with_fts_backend(tmp_path, monkeypatch, backend="sqlite_sidecar")
    config.data_dir.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(
        str(config.db_path), config={"storage_compatibility_version": STORAGE_VERSION}
    ) as conn:
        ensure_schema(conn, embed_dim=config.embedding.dimensions)
        assert not storage_upgrade_needed(storage_version(conn))

    compact(config)

    with duckdb.connect(str(config.db_path), read_only=True) as conn:
        assert not storage_upgrade_needed(storage_version(conn))


def test_compact_preserves_a_legacy_storage_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only explicit storage migration may advance an existing file's format."""
    config = _config_with_fts_backend(tmp_path, monkeypatch, backend="sqlite_sidecar")
    with _connect_schema(config) as conn:
        before = storage_version(conn)
        assert storage_upgrade_needed(before)

    compact(config)

    with duckdb.connect(str(config.db_path), read_only=True) as conn:
        assert storage_version(conn) == before


def test_compact_round_trip_preserves_row_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    conn = _connect_schema(config)
    try:
        _bloat_session_state(conn, config)
    finally:
        conn.close()
    expected_counts = _table_counts(config.db_path)
    before = estimate_bloat_ratio(config.db_path, config)
    assert before.ratio > 1.5

    result = compact(config)
    actual_counts = _table_counts(config.db_path)

    assert result.replaced is True
    assert result.skipped_reason is None
    assert result.before == before
    assert result.after.file_size < result.before.file_size
    assert set(DATA_TABLES).issubset(result.tables_copied)
    for table in DATA_TABLES:
        assert result.tables_copied[table] == expected_counts[table]
    assert actual_counts == expected_counts


@_requires_duckdb_lock
def test_compact_calls_sidecar_optimize_when_backend_is_sqlite_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_with_fts_backend(tmp_path, monkeypatch, backend="sqlite_sidecar")
    conn = _connect_schema(config)
    try:
        _seed_fresh_rows(conn)
    finally:
        conn.close()
    _seed_sidecar_row(config)
    calls = 0

    def record_optimize(_sidecar_conn: sqlite3.Connection) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr("recall.db.fts_sidecar.optimize_sidecar", record_optimize)

    result = compact(config)

    assert result.replaced is True
    assert calls == 1


@_requires_duckdb_lock
def test_compact_skips_sidecar_optimize_when_backend_is_duckdb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_with_fts_backend(tmp_path, monkeypatch, backend="duckdb")
    conn = _connect_schema(config)
    try:
        _seed_fresh_rows(conn)
    finally:
        conn.close()
    calls = 0

    def record_optimize(_sidecar_conn: sqlite3.Connection) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr("recall.db.fts_sidecar.optimize_sidecar", record_optimize)

    result = compact(config)

    assert result.replaced is True
    assert calls == 0


@_requires_duckdb_lock
def test_compact_preserves_legacy_fts_schemas_when_backend_is_sqlite_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_with_fts_backend(tmp_path, monkeypatch, backend="sqlite_sidecar")
    conn = _connect_schema(config)
    try:
        _seed_fts_rows(conn)
        create_fts_indexes(conn, FtsConfig(backend="duckdb"))
        before = _legacy_fts_schema_count(conn)
    finally:
        conn.close()
    assert before > 0
    drop_calls = 0
    real_drop_fts_shadow_schemas = compaction_module._drop_fts_shadow_schemas

    def spy_drop_fts_shadow_schemas(spy_conn: duckdb.DuckDBPyConnection) -> None:
        nonlocal drop_calls
        drop_calls += 1
        real_drop_fts_shadow_schemas(spy_conn)

    monkeypatch.setattr(
        compaction_module,
        "_drop_fts_shadow_schemas",
        spy_drop_fts_shadow_schemas,
    )

    result = compact(config)

    conn = duckdb.connect(str(config.db_path), read_only=True)
    try:
        after = _legacy_fts_schema_count(conn)
    finally:
        conn.close()
    assert result.replaced is True
    assert drop_calls == 0
    assert after == before


@_requires_duckdb_lock
def test_compact_succeeds_even_if_sidecar_optimize_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config_with_fts_backend(tmp_path, monkeypatch, backend="sqlite_sidecar")
    conn = _connect_schema(config)
    try:
        _seed_fresh_rows(conn)
    finally:
        conn.close()
    _seed_sidecar_row(config)

    def fail_optimize(_sidecar_conn: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("simulated optimize failure")

    monkeypatch.setattr("recall.db.fts_sidecar.optimize_sidecar", fail_optimize)

    with caplog.at_level("WARNING"):
        result = compact(config)

    assert result.replaced is True
    assert "sidecar optimize failed" in caplog.text


@_requires_duckdb_lock
def test_compact_succeeds_when_sidecar_file_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_with_fts_backend(tmp_path, monkeypatch, backend="sqlite_sidecar")
    conn = _connect_schema(config)
    try:
        _seed_fresh_rows(conn)
    finally:
        conn.close()
    assert not sidecar_path(config.data_dir).exists()

    result = compact(config)

    assert result.replaced is True
    assert sidecar_path(config.data_dir).exists()


@_requires_duckdb_lock
def test_compact_uses_copy_from_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path, monkeypatch)
    conn = _connect_schema(config)
    try:
        _seed_fresh_rows(conn)
    finally:
        conn.close()
    executed_sql: list[str] = []
    original_connect = compaction_module.duckdb.connect

    class SpyConnection:
        def __init__(self, inner: duckdb.DuckDBPyConnection) -> None:
            self._inner = inner

        def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
            executed_sql.append(" ".join(sql.split()).upper())
            return self._inner.execute(sql, *args, **kwargs)

        def close(self) -> None:
            self._inner.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    def connect_spy(*args: Any, **kwargs: Any) -> SpyConnection:
        return SpyConnection(original_connect(*args, **kwargs))

    monkeypatch.setattr(compaction_module.duckdb, "connect", connect_spy)

    result = compact(config)

    assert result.replaced is True
    copy_index = next(
        index for index, sql in enumerate(executed_sql) if "COPY FROM DATABASE" in sql
    )
    for table in SINGLETON_TABLES:
        delete_index = next(
            index for index, sql in enumerate(executed_sql) if sql == f"DELETE FROM {table.upper()}"
        )
        assert delete_index < copy_index
    for table in DATA_TABLES:
        assert f"INSERT INTO {table.upper()} SELECT * FROM OLD.{table.upper()}" not in executed_sql


@_requires_duckdb_lock
def test_compact_aborts_on_row_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    conn = _connect_schema(config)
    try:
        _seed_fresh_rows(conn)
    finally:
        conn.close()
    before_size = config.db_path.stat().st_size
    before_hash = _sha256(config.db_path)
    before_counts = _table_counts(config.db_path)

    def fail_verify(_conn: duckdb.DuckDBPyConnection, table: str) -> int:
        raise CompactionError(f"row count mismatch for {table}: source=1 dest=0")

    monkeypatch.setattr(compaction_module, "_verify_table_count", fail_verify)

    with pytest.raises(CompactionError, match="row count mismatch"):
        compact(config)

    assert config.db_path.stat().st_size == before_size
    assert _sha256(config.db_path) == before_hash
    assert _table_counts(config.db_path) == before_counts
    assert not config.db_path.with_name(f"{config.db_path.name}.compact").exists()


@_requires_duckdb_lock
def test_compact_atomic_replace_failure_retains_pre_compact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    conn = _connect_schema(config)
    try:
        _seed_fresh_rows(conn)
    finally:
        conn.close()
    compact_path = config.db_path.with_name(f"{config.db_path.name}.compact")
    backup_path = config.db_path.with_name(f"{config.db_path.name}.pre-compact")
    original_replace: Any = os.replace

    def fail_main_replace(src: Any, dst: Any) -> None:
        if Path(src) == compact_path and Path(dst) == config.db_path:
            raise OSError("simulated atomic replacement failure")
        original_replace(src, dst)

    monkeypatch.setattr(compaction_module.os, "replace", fail_main_replace)

    with pytest.raises(CompactionError, match="replacement") as exc_info:
        compact(config)

    assert "atomic replacement" in str(exc_info.value.__cause__)
    conn = duckdb.connect(str(config.db_path), read_only=True)
    try:
        assert _count_rows(conn, "sessions") == 3
    finally:
        conn.close()
    assert backup_path.exists()


@_requires_duckdb_lock
def test_compact_sentinel_exists_during_compaction_and_removed_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    conn = _connect_schema(config)
    try:
        _seed_fresh_rows(conn)
    finally:
        conn.close()
    sentinel_path = config.data_dir / "recall.compacting"
    sentinel_seen = False
    original_replace = compaction_module._replace_database_files

    def verify_sentinel_during_replacement(replacement_path: Path, db_path: Path) -> None:
        nonlocal sentinel_seen
        sentinel_seen = sentinel_path.exists()
        original_replace(replacement_path, db_path)

    monkeypatch.setattr(
        compaction_module,
        "_replace_database_files",
        verify_sentinel_during_replacement,
    )

    compact(config)

    assert sentinel_seen is True
    assert not sentinel_path.exists()


@_requires_duckdb_lock
def test_compact_failure_releases_lock_and_sentinel_and_retains_pre_compact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    conn = _connect_schema(config)
    try:
        _seed_fresh_rows(conn)
    finally:
        conn.close()
    sentinel_path = config.data_dir / "recall.compacting"
    backup_path = config.db_path.with_name(f"{config.db_path.name}.pre-compact")

    def fail_replacement(_replacement_path: Path, _db_path: Path) -> None:
        assert sentinel_path.exists()
        result = _advisory_lock_probe(config.lock_path)
        assert result.returncode == 23, result.stderr
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr(compaction_module, "_replace_database_files", fail_replacement)

    with pytest.raises(CompactionError, match="replacement"):
        compact(config)

    assert backup_path.exists()
    assert not sentinel_path.exists()
    with advisory_lock(config.lock_path):
        pass


@_requires_duckdb_lock
def test_fts_search_equivalence_after_compact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_with_fts_backend(tmp_path, monkeypatch, backend="duckdb")
    conn = _connect_schema(config)
    try:
        _seed_fts_rows(conn)
        create_fts_indexes(conn, config.fts)
        before_ids = _message_fts_ids(conn, "database compaction")
    finally:
        conn.close()

    compact(config)

    conn = duckdb.connect(str(config.db_path), read_only=True)
    try:
        after_ids = _message_fts_ids(conn, "database compaction")
    finally:
        conn.close()

    assert after_ids == before_ids


@_requires_duckdb_lock
def test_compact_handles_existing_wal_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    conn = _connect_schema(config)
    try:
        _seed_fresh_rows(conn)
    finally:
        conn.close()
    wal_path = config.db_path.with_suffix(config.db_path.suffix + ".wal")
    staged_wal_path = wal_path.with_suffix(wal_path.suffix + ".pre-compact")
    wal_path.write_bytes(_STALE_WAL_BYTES)

    result = compact(config)

    assert result.replaced is True
    assert not wal_path.exists() or wal_path.read_bytes() != _STALE_WAL_BYTES
    assert not staged_wal_path.exists()


@_requires_duckdb_lock
def test_successful_compact_clears_fatal_failure_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RESIL-016: COPY FROM DATABASE rebuilds every index, so a compacted database
    must not be refused at the next daemon start on the strength of the old marker."""
    from recall.services.runtime_state import load_runtime_status
    from recall.services.self_repair import read_failure_marker, remember_fatal_failure

    config = _config(tmp_path, monkeypatch)
    conn = _connect_schema(config)
    try:
        _seed_fresh_rows(conn)
    finally:
        conn.close()
    remember_fatal_failure(
        config.data_dir,
        duckdb.FatalException("Failed to delete all rows from index. Only deleted 0 out of 1"),
        site="catch_up",
    )

    result = compact(config)

    assert result.replaced is True
    assert read_failure_marker(config.data_dir) is None
    assert load_runtime_status(config).last_fatal_signature is None
