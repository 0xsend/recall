"""Schema 42 migration (0042_claude_response_usage) and rollback qualification."""

from __future__ import annotations

import importlib
import sqlite3
from contextlib import closing
from pathlib import Path

import duckdb
import pytest
from recall.core.schema_version import RecoveryStateUnavailableError
from recall.db.lease import StandaloneLifetime
from recall.db.migrations import get_migrations, run_pending_migrations
from recall.db.schema import (
    SCHEMA_VERSION,
    ensure_schema,
    ensure_schema_lenient,
    get_schema_version,
)
from recall.services.index_migration import begin_storage_migration, rollback_migration

from tests.test_services.test_run_completion import _config


def _schema_41_db(path: Path, *, reject_version: bool = False) -> None:
    with duckdb.connect(str(path)) as conn:
        ensure_schema(conn)
        conn.execute("DROP TABLE claude_usage_scopes")
        conn.execute("DROP TABLE claude_response_usage")
        conn.execute("ALTER TABLE runtime_state DROP COLUMN claude_usage_incarnation")
        conn.execute(
            "DELETE FROM schema_migrations WHERE migration_id = '0042_claude_response_usage'"
        )
        conn.execute("DELETE FROM schema_version WHERE version > 41")
        if reject_version:
            conn.execute("DROP TABLE schema_version")
            conn.execute(
                "CREATE TABLE schema_version(version INTEGER PRIMARY KEY CHECK(version <= 41))"
            )
        conn.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (41)")
        conn.execute(
            "INSERT INTO sessions(id, source, source_path, source_session_id) "
            "VALUES ('hist', 'claude_code', '/tmp/hist.jsonl', 'native')"
        )
        conn.execute(
            "INSERT INTO session_state(session_id, input_tokens, output_tokens, "
            "file_mtime, file_size) VALUES ('hist', 9999, 777, 1.0, 100)"
        )


def test_migration_0042_upgrades_schema_41_preserving_historical_totals_and_replays(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v41.duckdb"
    _schema_41_db(path)
    with duckdb.connect(str(path)) as conn:
        ensure_schema(conn)
        assert get_schema_version(conn) == SCHEMA_VERSION == 42
        assert conn.execute(
            "SELECT migration_id FROM schema_migrations WHERE migration_id = ?",
            ["0042_claude_response_usage"],
        ).fetchone() == ("0042_claude_response_usage",)
        assert conn.execute(
            "SELECT input_tokens, output_tokens FROM session_state WHERE session_id = 'hist'"
        ).fetchone() == (9999, 777)
        assert conn.execute("SELECT claude_usage_incarnation FROM runtime_state").fetchone() == (0,)
        # Idempotent replay on second open and direct upgrade call.
        ensure_schema(conn)
        migration = next(m for m in get_migrations() if m.id == "0042_claude_response_usage")
        with StandaloneLifetime().lease(conn) as lease:
            assert migration.upgrade(lease)
        assert conn.execute(
            "SELECT count(*) FROM schema_migrations WHERE migration_id = ?",
            ["0042_claude_response_usage"],
        ).fetchone() == (1,)


def test_migration_0042_rolls_back_atomically_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "v41_fail.duckdb"
    _schema_41_db(path, reject_version=True)
    with duckdb.connect(str(path)) as conn:
        run_pending_migrations(conn, 41)
        assert get_schema_version(conn) == 41
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        assert "claude_usage_scopes" not in tables
        assert "claude_response_usage" not in tables
        assert conn.execute(
            "SELECT count(*) FROM schema_migrations WHERE migration_id = ?",
            ["0042_claude_response_usage"],
        ).fetchone() == (0,)
        assert conn.execute(
            "SELECT input_tokens, output_tokens FROM session_state WHERE session_id = 'hist'"
        ).fetchone() == (9999, 777)


def test_migration_0042_refuses_pre_fence_and_damaged_schema_41(tmp_path: Path) -> None:
    path = tmp_path / "v36.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE schema_version(version INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO schema_version VALUES (36)")
        migration = importlib.import_module("recall.db.migrations.0042_claude_response_usage")
        with StandaloneLifetime().lease(conn) as lease:
            assert migration._upgrade(lease) is False
        assert get_schema_version(conn) == 36
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        assert "claude_usage_scopes" not in tables

    for mode in ("table_absent", "column_missing", "column_wrong_type", "column_wrong_type_null"):
        damaged = tmp_path / f"v41_damaged_{mode}.duckdb"
        _schema_41_db(damaged)
        with duckdb.connect(str(damaged)) as conn:
            conn.execute("DROP TABLE daemon_index_repair")
            if mode == "column_missing":
                conn.execute("CREATE TABLE daemon_index_repair (singleton BOOLEAN)")
                conn.execute("INSERT INTO daemon_index_repair VALUES (TRUE)")
            elif mode in {"column_wrong_type", "column_wrong_type_null"}:
                conn.execute(
                    "CREATE TABLE daemon_index_repair ("
                    "singleton BOOLEAN, attempt_no UBIGINT, phase VARCHAR, trigger_id UUID, "
                    "trigger_kind VARCHAR, signature VARCHAR, trigger_json INTEGER, "
                    "target_manifest JSON, oracle_ref VARCHAR, oracle_hash VARCHAR, "
                    "success_since_attempt BOOLEAN, last_folded_event_id UUID, "
                    "success_since_failure BOOLEAN, acknowledged_events JSON, "
                    "release_certificate JSON)"
                )
                trigger_val = "NULL" if mode == "column_wrong_type_null" else "1"
                conn.execute(
                    "INSERT INTO daemon_index_repair VALUES "
                    f"(TRUE, 0, 'idle', NULL, NULL, NULL, {trigger_val}, NULL, NULL, NULL, "
                    "TRUE, NULL, TRUE, '[]', NULL)"
                )
            with StandaloneLifetime().lease(conn) as lease:
                assert migration._upgrade(lease) is False
            assert get_schema_version(conn) == 41
            tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
            assert "claude_usage_scopes" not in tables
            assert "claude_response_usage" not in tables
            assert conn.execute(
                "SELECT count(*) FROM schema_migrations WHERE migration_id = ?",
                ["0042_claude_response_usage"],
            ).fetchone() == (0,)


@pytest.mark.parametrize(
    "damage",
    [
        "table_absent",
        "column_missing",
        "column_wrong_type",
        "column_wrong_type_null",
        "row_absent",
        "row_invalid",
    ],
)
@pytest.mark.parametrize("lenient", [False, True])
def test_ensure_schema_refuses_damaged_schema_41_before_running_0042(
    tmp_path: Path, damage: str, lenient: bool
) -> None:
    path = tmp_path / "damaged41.duckdb"
    _schema_41_db(path)
    with duckdb.connect(str(path)) as conn:
        if damage == "table_absent":
            conn.execute("DROP TABLE daemon_index_repair")
        elif damage == "column_missing":
            conn.execute("DROP TABLE daemon_index_repair")
            conn.execute("CREATE TABLE daemon_index_repair (singleton BOOLEAN)")
            conn.execute("INSERT INTO daemon_index_repair VALUES (TRUE)")
        elif damage in {"column_wrong_type", "column_wrong_type_null"}:
            conn.execute("DROP TABLE daemon_index_repair")
            conn.execute(
                "CREATE TABLE daemon_index_repair ("
                "singleton BOOLEAN, attempt_no UBIGINT, phase VARCHAR, trigger_id UUID, "
                "trigger_kind VARCHAR, signature VARCHAR, trigger_json INTEGER, "
                "target_manifest JSON, oracle_ref VARCHAR, oracle_hash VARCHAR, "
                "success_since_attempt BOOLEAN, last_folded_event_id UUID, "
                "success_since_failure BOOLEAN, acknowledged_events JSON, "
                "release_certificate JSON)"
            )
            trigger_val = "NULL" if damage == "column_wrong_type_null" else "1"
            conn.execute(
                "INSERT INTO daemon_index_repair VALUES "
                f"(TRUE, 0, 'idle', NULL, NULL, NULL, {trigger_val}, NULL, NULL, NULL, "
                "TRUE, NULL, TRUE, '[]', NULL)"
            )
        elif damage == "row_absent":
            conn.execute("DELETE FROM daemon_index_repair")
        else:
            conn.execute("UPDATE daemon_index_repair SET acknowledged_events = '{}'::JSON")
    ensure = ensure_schema_lenient if lenient else ensure_schema
    with duckdb.connect(str(path)) as conn, pytest.raises(RecoveryStateUnavailableError):
        ensure(conn)
    with duckdb.connect(str(path), read_only=True) as conn:
        assert get_schema_version(conn) == 41
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        assert "claude_usage_scopes" not in tables
        assert "claude_response_usage" not in tables
        assert conn.execute(
            "SELECT count(*) FROM schema_migrations WHERE migration_id = ?",
            ["0042_claude_response_usage"],
        ).fetchone() == (0,)


def test_rollback_migration_accepts_schema_42_and_refuses_schema_41_backup(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    config = _config(data, None)
    with duckdb.connect(
        str(config.db_path), config={"storage_compatibility_version": "v1.0.0"}
    ) as conn:
        ensure_schema(conn)
        job = begin_storage_migration(conn, config, observer=StandaloneLifetime())
    with closing(sqlite3.connect(data / "recall.fts.sqlite")) as conn:
        conn.execute("CREATE TABLE witness(v TEXT)")
        conn.commit()
    assert job.backup_path is not None

    # A schema-41 backup cannot be installed into a schema-42 DB (no logical downgrade).
    with duckdb.connect(str(Path(job.backup_path) / "recall.duckdb")) as backup:
        backup.execute("DELETE FROM schema_version WHERE version > 41")
        backup.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (41)")
    before = config.db_path.read_bytes()
    with pytest.raises(RuntimeError, match="cannot lower the logical schema"):
        rollback_migration(config, preserve=tmp_path / "failed-41")
    assert config.db_path.read_bytes() == before

    # Restoring the backup to schema 42 succeeds cleanly.
    with duckdb.connect(str(Path(job.backup_path) / "recall.duckdb")) as backup:
        backup.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (42)")
    rollback_migration(config, preserve=tmp_path / "failed-42")
    assert (config.data_dir / "daemon-refused").exists()
