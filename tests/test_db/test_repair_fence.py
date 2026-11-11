"""The recovery table, logical fence, and its audit are one durable upgrade."""

from pathlib import Path

import duckdb
from recall.db.schema import ensure_schema, ensure_schema_lenient


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
