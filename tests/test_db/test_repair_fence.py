"""The recovery table, logical fence, and its audit are one durable upgrade."""

import importlib
from pathlib import Path

import duckdb
import pytest
from recall.db.schema import ensure_schema, ensure_schema_lenient

RECREATE = "recall index --recreate --yes"


def test_fresh_database_is_fenced_with_seeded_recovery_state(tmp_path: Path) -> None:
    path = tmp_path / "fresh.duckdb"
    with duckdb.connect(str(path)) as conn:
        ensure_schema(conn)
    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute("SELECT max(version) FROM schema_version").fetchone() == (41,)
        assert conn.execute(
            "SELECT phase, attempt_no, success_since_attempt, success_since_failure "
            "FROM daemon_index_repair WHERE singleton"
        ).fetchone() == ("idle", 0, True, True)
        assert conn.execute(
            "SELECT migration_id FROM schema_migrations WHERE migration_id = ?",
            ["0041_daemon_index_repair"],
        ).fetchone() == ("0041_daemon_index_repair",)


def test_existing_database_upgrade_preserves_acknowledged_rows(tmp_path: Path) -> None:
    path = tmp_path / "existing.duckdb"
    with duckdb.connect(str(path)) as conn:
        ensure_schema(conn)
        conn.execute("CREATE TABLE witness(id INTEGER, value VARCHAR)")
        conn.execute("INSERT INTO witness VALUES (1, 'acknowledged')")
        conn.execute("DROP TABLE IF EXISTS daemon_index_repair")
        conn.execute("DELETE FROM schema_version WHERE version > 36")
        conn.execute("INSERT OR IGNORE INTO schema_version VALUES (36, CURRENT_TIMESTAMP)")
        conn.execute(
            "DELETE FROM schema_migrations WHERE migration_id = '0041_daemon_index_repair'"
        )
    with duckdb.connect(str(path)) as conn:
        ensure_schema_lenient(conn)
    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute("SELECT max(version) FROM schema_version").fetchone() == (41,)
        assert conn.execute("SELECT * FROM witness").fetchall() == [(1, "acknowledged")]
        assert conn.execute("SELECT phase FROM daemon_index_repair").fetchone() == ("idle",)


def _downgrade_to_unfenced(path: Path) -> None:
    with duckdb.connect(str(path)) as conn:
        ensure_schema(conn)
        conn.execute("CREATE TABLE witness(id INTEGER)")
        conn.execute("INSERT INTO witness VALUES (1)")
        conn.execute("DROP TABLE IF EXISTS daemon_index_repair")
        conn.execute("DELETE FROM schema_version WHERE version > 36")
        conn.execute("INSERT OR IGNORE INTO schema_version VALUES (36, CURRENT_TIMESTAMP)")
        conn.execute(
            "DELETE FROM schema_migrations WHERE migration_id = '0041_daemon_index_repair'"
        )


def _assert_prohibits_recreate(message: str) -> None:
    """The released destructive reset may be named only as prohibited."""
    assert "db rebuild-indexes" in message
    assert "matched restore" in message
    assert f"Never run the released 0.46.1 `{RECREATE}`" in message
    assert f"Run `{RECREATE}`" not in message


@pytest.mark.parametrize("lenient", [False, True])
def test_failed_fence_diagnostic_prohibits_the_released_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lenient: bool
) -> None:
    path = tmp_path / "unfenced.duckdb"
    _downgrade_to_unfenced(path)

    def fail(_transaction: object) -> None:
        raise duckdb.Error("injected fence failure")

    migration = importlib.import_module("recall.db.migrations.0041_daemon_index_repair")
    monkeypatch.setattr(migration, "initialize_repair_in", fail)
    ensure = ensure_schema_lenient if lenient else ensure_schema
    with duckdb.connect(str(path)) as conn, pytest.raises(RuntimeError) as caught:
        ensure(conn)
    _assert_prohibits_recreate(str(caught.value))
    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute("SELECT max(version) FROM schema_version").fetchone() == (36,)
        assert conn.execute("SELECT id FROM witness").fetchall() == [(1,)]


@pytest.mark.parametrize("damage", ["table_absent", "row_absent", "row_invalid"])
@pytest.mark.parametrize("lenient", [False, True])
def test_schema_41_without_a_valid_recovery_singleton_is_not_admitted(
    tmp_path: Path, damage: str, lenient: bool
) -> None:
    path = tmp_path / "damaged.duckdb"
    with duckdb.connect(str(path)) as conn:
        ensure_schema(conn)
        conn.execute("CREATE TABLE witness(id INTEGER)")
        conn.execute("INSERT INTO witness VALUES (1)")
        if damage == "table_absent":
            conn.execute("DROP TABLE daemon_index_repair")
        elif damage == "row_absent":
            conn.execute("DELETE FROM daemon_index_repair")
        else:
            conn.execute("UPDATE daemon_index_repair SET acknowledged_events = '{}'::JSON")
    ensure = ensure_schema_lenient if lenient else ensure_schema
    with duckdb.connect(str(path)) as conn, pytest.raises(RuntimeError) as caught:
        ensure(conn)
    _assert_prohibits_recreate(str(caught.value))
    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute("SELECT max(version) FROM schema_version").fetchone() == (41,)
        assert conn.execute("SELECT id FROM witness").fetchall() == [(1,)]
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if damage == "table_absent":
            assert "daemon_index_repair" not in tables
        elif damage == "row_absent":
            assert conn.execute("SELECT count(*) FROM daemon_index_repair").fetchone() == (0,)


def test_connect_returns_no_writer_for_schema_41_without_a_recovery_singleton(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from recall.core.config import AppConfig
    from recall.db.connection import connect

    config = replace(
        AppConfig.load(),
        data_dir=tmp_path,
        db_path=tmp_path / "recall.duckdb",
        lock_path=tmp_path / "recall.lock",
    )
    with connect(config):
        pass
    with duckdb.connect(str(config.db_path)) as conn:
        conn.execute("DROP TABLE daemon_index_repair")
    with pytest.raises(RuntimeError) as caught:
        connect(config)
    _assert_prohibits_recreate(str(caught.value))
