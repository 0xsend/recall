"""Scope session pathname uniqueness to its source, preserving all row identities.

Only sessions is copied; children, catalog bindings, usage and keyword documents
keep their existing IDs. DDL, copied rows and the version advance commit together.
An interrupted transaction rolls back and startup retries the migration.

Rollback restores the paired pre-upgrade DuckDB and keyword-sidecar backup.
An automatic downgrade cannot represent shared paths in the old schema.
"""

from __future__ import annotations

import duckdb

from recall.db.migrations import Migration, register, table_exists

MIGRATION_ID = "0035_source_path_identity"
TARGET_VERSION = 35


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < 32:
        return False
    conn.execute("BEGIN TRANSACTION")
    try:
        if table_exists(conn, "sessions"):
            unique_columns = conn.execute(
                "SELECT constraint_column_names FROM duckdb_constraints() "
                "WHERE schema_name = 'main' AND table_name = 'sessions' "
                "AND constraint_type = 'UNIQUE'"
            ).fetchall()
            if unique_columns != [(["source", "source_path"],)]:
                conn.execute(
                    """CREATE TABLE sessions_source_path (
                        id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        source_session_id TEXT,
                        UNIQUE (source, source_path)
                    )"""
                )
                conn.execute("INSERT INTO sessions_source_path SELECT * FROM sessions")
                conn.execute("DROP TABLE sessions")
                conn.execute("ALTER TABLE sessions_source_path RENAME TO sessions")
                conn.execute("CREATE INDEX idx_sessions_source ON sessions(source)")
        if current < TARGET_VERSION:
            set_schema_version_to(conn, TARGET_VERSION)
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (migration_id) VALUES (?)", [MIGRATION_ID]
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return True


register(Migration(id=MIGRATION_ID, target_version=TARGET_VERSION, upgrade=_upgrade))
