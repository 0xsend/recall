"""Compaction cannot discard canonical namespaces or alter recovery authority."""

from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from recall.db.repair import load_repair_row
from recall.db.schema import ensure_schema
from recall.services import compaction

from tests.test_services.test_run_completion import _config


@pytest.mark.parametrize("namespace", ["fts_main_acknowledged", "fts_main_message_state"])
def test_native_compact_preserves_unknown_data_or_holds(tmp_path: Path, namespace: str):
    config = _config(tmp_path, None)
    config = replace(config, fts=replace(config.fts, backend="duckdb"))
    with duckdb.connect(str(config.db_path)) as conn:
        ensure_schema(conn)
        conn.execute(f'CREATE SCHEMA "{namespace}"')
        conn.execute(
            f'CREATE TABLE "{namespace}".acknowledged AS SELECT 9223372036854775807 AS value'
        )
    if namespace == "fts_main_message_state":
        with pytest.raises(compaction.CompactionError, match="custom table"):
            compaction.compact(config)
    else:
        assert compaction.compact(config).replaced
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        assert fresh.execute(f'SELECT * FROM "{namespace}".acknowledged').fetchall() == [
            (9223372036854775807,)
        ]


def test_fresh_candidate_recovery_drift_holds_before_install(tmp_path: Path, monkeypatch):
    config = _config(tmp_path, None)
    with duckdb.connect(str(config.db_path)) as conn:
        ensure_schema(conn)
        before = load_repair_row(conn)
    copy = compaction._copy_database

    def changed(conn, config):
        counts = copy(conn, config)
        conn.execute("UPDATE daemon_index_repair SET attempt_no=42")
        return counts

    monkeypatch.setattr(compaction, "_copy_database", changed)
    with pytest.raises(compaction.CompactionError, match="recovery"):
        compaction.compact(config)
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        assert load_repair_row(fresh) == before


@pytest.mark.parametrize("backend", ["sqlite_sidecar", "duckdb"])
def test_pre_fence_compaction_upgrades_through_owned_schema_admission(tmp_path: Path, backend: str):
    config = _config(tmp_path, None)
    config = replace(config, fts=replace(config.fts, backend=backend))
    with duckdb.connect(str(config.db_path)) as conn:
        ensure_schema(conn)
        conn.execute("CREATE TABLE acknowledged(value BIGINT)")
        conn.execute("INSERT INTO acknowledged VALUES (9223372036854775807)")
        conn.execute("DROP TABLE daemon_index_repair")
        conn.execute("DELETE FROM schema_version WHERE version>36")
        conn.execute("INSERT INTO schema_version(version) VALUES (36)")
        conn.execute("DELETE FROM schema_migrations WHERE migration_id='0041_daemon_index_repair'")
    assert compaction.compact(config).replaced
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        assert fresh.execute("SELECT max(version) FROM schema_version").fetchone() == (41,)
        assert load_repair_row(fresh).phase == "idle"
        assert fresh.execute("SELECT * FROM acknowledged").fetchall() == [(9223372036854775807,)]


def test_native_compact_copies_unrelated_enum_data_without_oracle_encoding(tmp_path: Path):
    config = _config(tmp_path, None)
    config = replace(config, fts=replace(config.fts, backend="duckdb"))
    with duckdb.connect(str(config.db_path)) as conn:
        ensure_schema(conn)
        conn.execute("CREATE TYPE acknowledged_kind AS ENUM ('first','second')")
        conn.execute("CREATE TABLE acknowledged(value acknowledged_kind)")
        conn.execute("INSERT INTO acknowledged VALUES ('second'),('first')")
    assert compaction.compact(config).replaced
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        assert fresh.execute("SELECT * FROM acknowledged ORDER BY value").fetchall() == [
            ("first",),
            ("second",),
        ]
