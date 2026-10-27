from __future__ import annotations

import duckdb
import pytest
from recall.db.migrations import get_migrations, run_pending_migrations
from recall.db.schema import get_schema_version


def _legacy(path, *, reject_version=False):
    conn = duckdb.connect(str(path))
    version_constraint = "CHECK (version <= 32)" if reject_version else ""
    conn.execute(f"CREATE TABLE schema_version (version INTEGER PRIMARY KEY {version_constraint})")
    conn.execute("INSERT INTO schema_version VALUES (32)")
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT NOT NULL, "
        "source_path TEXT UNIQUE NOT NULL, source_session_id TEXT)"
    )
    conn.execute("CREATE INDEX idx_sessions_source ON sessions(source)")
    conn.execute("INSERT INTO sessions VALUES ('retained', 'claude_code', '/shared', 'native')")
    conn.execute("CREATE TABLE messages (id TEXT PRIMARY KEY, session_id TEXT, content TEXT)")
    conn.execute("INSERT INTO messages VALUES ('child', 'retained', 'preserved content')")
    return conn


def test_source_path_migration_preserves_rows_and_source_uniqueness_after_restart(tmp_path):
    path = tmp_path / "recall.duckdb"
    with _legacy(path) as conn:
        before = conn.execute("SELECT * FROM sessions").fetchall()
        run_pending_migrations(conn, 32)
        assert get_schema_version(conn) == 36
        assert conn.execute("SELECT * FROM sessions").fetchall() == before
        assert conn.execute("SELECT * FROM messages").fetchall() == [
            ("child", "retained", "preserved content")
        ]
        migration = next(m for m in get_migrations() if m.id == "0035_source_path_identity")
        assert migration.upgrade(conn)
        run_pending_migrations(conn, 32)
        assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone() == (2,)
    with duckdb.connect(str(path)) as reopened:
        reopened.execute("INSERT INTO sessions VALUES ('other', 'codex', '/shared', 'native')")
        with pytest.raises(duckdb.ConstraintException):
            reopened.execute("INSERT INTO sessions VALUES ('duplicate', 'codex', '/shared', NULL)")
        assert reopened.execute("SELECT id FROM sessions ORDER BY id").fetchall() == [
            ("other",),
            ("retained",),
        ]


def test_source_path_migration_rolls_back_all_ddl_if_version_advance_fails(tmp_path):
    with _legacy(tmp_path / "recall.duckdb", reject_version=True) as conn:
        run_pending_migrations(conn, 32)
        assert get_schema_version(conn) == 32
        assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone() == (0,)
        assert conn.execute("SELECT * FROM sessions").fetchall() == [
            ("retained", "claude_code", "/shared", "native")
        ]
        with pytest.raises(duckdb.ConstraintException):
            conn.execute("INSERT INTO sessions VALUES ('other', 'codex', '/shared', NULL)")
        assert conn.execute("SELECT * FROM messages").fetchall() == [
            ("child", "retained", "preserved content")
        ]
