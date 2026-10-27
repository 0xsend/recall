from pathlib import Path

import duckdb
import pytest
from recall.db.migrations import get_migrations, run_pending_migrations
from recall.db.schema import ensure_schema, get_schema_version


def _legacy(path: Path, *, reject_version: bool = False) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(path))
    ensure_schema(conn)
    conn.execute("ALTER TABLE usage_log_cursors DROP COLUMN file_device")
    conn.execute("ALTER TABLE usage_log_cursors DROP COLUMN file_inode")
    conn.execute("DELETE FROM schema_version")
    if reject_version:
        conn.execute("DROP TABLE schema_version")
        conn.execute(
            "CREATE TABLE schema_version(version INTEGER PRIMARY KEY CHECK(version <= 35))"
        )
    conn.execute("INSERT INTO schema_version(version) VALUES (35)")
    conn.execute("INSERT INTO usage_log_cursors VALUES ('/log', 123, 456, 'tail', '2026-10-27')")
    conn.execute("INSERT INTO sessions VALUES ('s1', 'grok', '/chat', 'sid')")
    conn.execute(
        "INSERT INTO usage_events(id, source, source_session_id, prompt_tokens, harvested_at) "
        "VALUES ('e1', 'grok', 'sid', 17, '2026-10-27')"
    )
    return conn


def test_usage_file_identity_migration_preserves_rows_and_reopens(tmp_path: Path) -> None:
    path = tmp_path / "legacy.duckdb"
    with _legacy(path) as conn:
        cursors = conn.execute("SELECT * FROM usage_log_cursors").fetchall()
        events = conn.execute("SELECT * FROM usage_events").fetchall()
        sessions = conn.execute("SELECT * FROM sessions").fetchall()
        run_pending_migrations(conn, 35)
        migration = next(m for m in get_migrations() if m.target_version == 36)
        assert migration.upgrade(conn)
        run_pending_migrations(conn, 35)
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE migration_id = ?", [migration.id]
        ).fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM schema_version WHERE version=36").fetchone() == (
            1,
        )
    with duckdb.connect(str(path)) as conn:
        assert get_schema_version(conn) == 36
        assert conn.execute("SELECT * FROM usage_log_cursors").fetchall() == [
            (*cursors[0], None, None)
        ]
        assert conn.execute("SELECT * FROM usage_events").fetchall() == events
        assert conn.execute("SELECT * FROM sessions").fetchall() == sessions


def test_usage_file_identity_migration_version_failure_rolls_back(tmp_path: Path) -> None:
    path = tmp_path / "legacy.duckdb"
    with _legacy(path, reject_version=True) as conn:
        before = conn.execute("SELECT * FROM usage_log_cursors").fetchall()
        run_pending_migrations(conn, 35)
        assert get_schema_version(conn) == 35
        assert conn.execute("SELECT * FROM usage_log_cursors").fetchall() == before
        assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone() == (0,)
        with pytest.raises(duckdb.BinderException, match="file_device"):
            conn.execute("SELECT file_device FROM usage_log_cursors")
    with duckdb.connect(str(path)) as conn:
        assert get_schema_version(conn) == 35
        assert conn.execute("SELECT id, prompt_tokens FROM usage_events").fetchall() == [("e1", 17)]
        assert conn.execute("SELECT id FROM sessions").fetchall() == [("s1",)]


def test_usage_file_identity_fresh_and_upgraded_nullable_pair(tmp_path: Path) -> None:
    with duckdb.connect(":memory:") as fresh, _legacy(tmp_path / "legacy.duckdb") as migrated:
        ensure_schema(fresh)
        run_pending_migrations(migrated, 35)
        for conn in (fresh, migrated):
            conn.execute("INSERT INTO usage_log_cursors(path, updated_at) VALUES ('/new', now())")
            assert conn.execute(
                "SELECT byte_offset, file_size, consumed_tail_sha256, file_device, file_inode "
                "FROM usage_log_cursors WHERE path='/new'"
            ).fetchone() == (0, 0, None, None, None)
            conn.execute(
                "UPDATE usage_log_cursors SET file_device=18446744073709551615, "
                "file_inode=18446744073709551615 WHERE path='/new'"
            )
            assert conn.execute(
                "SELECT file_device, file_inode FROM usage_log_cursors WHERE path='/new'"
            ).fetchone() == (18446744073709551615, 18446744073709551615)


def test_usage_file_identity_migration_ddl_failure_keeps_version_and_data(tmp_path: Path) -> None:
    with _legacy(tmp_path / "legacy.duckdb") as conn:
        before = conn.execute("SELECT * FROM usage_log_cursors").fetchall()
        # A malformed legacy relation must not mark an unsuccessful upgrade applied.
        conn.execute("ALTER TABLE usage_log_cursors RENAME TO retained_cursors")
        conn.execute("CREATE VIEW usage_log_cursors AS SELECT * FROM retained_cursors")
        run_pending_migrations(conn, 35)
        assert get_schema_version(conn) == 35
        assert conn.execute("SELECT * FROM usage_log_cursors").fetchall() == before
        assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone() == (0,)
        conn.execute("DROP VIEW usage_log_cursors")
        conn.execute("ALTER TABLE retained_cursors RENAME TO usage_log_cursors")
        run_pending_migrations(conn, 35)
        assert get_schema_version(conn) == 36
