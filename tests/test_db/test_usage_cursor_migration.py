from pathlib import Path

import duckdb
import pytest
from recall.db.migrations import get_migrations, run_pending_migrations
from recall.db.schema import ensure_schema, get_schema_version


def _v31_cursor_db(path: Path, *, reject_version: bool = False) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(path))
    constraint = " CHECK (version <= 31)" if reject_version else ""
    conn.execute(f"CREATE TABLE schema_version (version INTEGER PRIMARY KEY{constraint})")
    conn.execute("INSERT INTO schema_version VALUES (31)")
    conn.execute(
        """CREATE TABLE usage_log_cursors (
            path TEXT PRIMARY KEY, byte_offset BIGINT NOT NULL DEFAULT 0,
            file_size BIGINT NOT NULL DEFAULT 0, updated_at TIMESTAMP NOT NULL
        )"""
    )
    conn.execute("INSERT INTO usage_log_cursors VALUES ('/log', 123, 456, '2026-10-17')")
    return conn


def test_usage_cursor_migration_preserves_legacy_cursor_and_replays_idempotently(tmp_path: Path):
    path = tmp_path / "recall.duckdb"
    conn = _v31_cursor_db(path)
    try:
        run_pending_migrations(conn, 31, through_version=32)
        migration = next(m for m in get_migrations() if m.target_version == 32)
        assert migration.upgrade(conn)
        run_pending_migrations(conn, 31, through_version=32)
        assert get_schema_version(conn) == 32
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_version WHERE version = 32"
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE migration_id = ?", [migration.id]
        ).fetchone() == (1,)
    finally:
        conn.close()

    with duckdb.connect(str(path)) as reopened:
        assert reopened.execute(
            "SELECT path, byte_offset, file_size, updated_at::DATE::TEXT, consumed_tail_sha256 "
            "FROM usage_log_cursors"
        ).fetchall() == [("/log", 123, 456, "2026-10-17", None)]


def test_usage_cursor_migration_rolls_back_ddl_when_version_cannot_advance(tmp_path: Path):
    conn = _v31_cursor_db(tmp_path / "recall.duckdb", reject_version=True)
    try:
        run_pending_migrations(conn, 31, through_version=32)
        assert get_schema_version(conn) == 31
        assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone() == (0,)
        assert conn.execute(
            "SELECT path, byte_offset, file_size FROM usage_log_cursors"
        ).fetchall() == [("/log", 123, 456)]
        with pytest.raises(duckdb.BinderException, match="consumed_tail_sha256"):
            conn.execute("SELECT consumed_tail_sha256 FROM usage_log_cursors")
    finally:
        conn.close()


def test_fresh_and_migrated_usage_cursors_accept_nullable_identity(tmp_path: Path):
    with duckdb.connect(":memory:") as fresh, _v31_cursor_db(tmp_path / "old.duckdb") as upgraded:
        ensure_schema(fresh)
        run_pending_migrations(upgraded, 31, through_version=32)
        for conn in (fresh, upgraded):
            conn.execute(
                "INSERT INTO usage_log_cursors (path, updated_at) VALUES ('/empty', '2026-10-17')"
            )
            assert conn.execute(
                "SELECT byte_offset, file_size, consumed_tail_sha256 "
                "FROM usage_log_cursors WHERE path = '/empty'"
            ).fetchone() == (0, 0, None)
