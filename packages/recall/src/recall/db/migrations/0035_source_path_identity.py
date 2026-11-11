"""Scope session pathname uniqueness to its source, preserving all row identities.

Only sessions is copied; children, catalog bindings, usage and keyword documents
keep their existing IDs. DDL, copied rows and the version advance commit together.
An interrupted transaction rolls back and startup retries the migration.

Rollback restores the paired pre-upgrade DuckDB and keyword-sidecar backup.
An automatic downgrade cannot represent shared paths in the old schema.
"""

from __future__ import annotations

from recall.db.lease import DbLease
from recall.db.migrations import Migration, register, table_exists

MIGRATION_ID = "0035_source_path_identity"
TARGET_VERSION = 35


def _upgrade(lease: DbLease) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = lease.call(
        "0035_source_path_identity:_upgrade", lambda conn: get_schema_version(conn)
    )
    if current < 32:
        return False
    lease.call(
        "0035_source_path_identity:_upgrade",
        lambda conn: conn.execute("BEGIN TRANSACTION").fetchone(),
    )
    try:
        if lease.call(
            "0035_source_path_identity:_upgrade", lambda conn: table_exists(conn, "sessions")
        ):
            unique_columns = lease.call(
                "0035_source_path_identity:_upgrade",
                lambda conn: conn.execute(
                    "SELECT constraint_column_names FROM duckdb_constraints() "
                    "WHERE schema_name = 'main' AND table_name = 'sessions' "
                    "AND constraint_type = 'UNIQUE'"
                ).fetchall(),
            )
            if unique_columns != [(["source", "source_path"],)]:
                lease.call(
                    "0035_source_path_identity:_upgrade",
                    lambda conn: conn.execute(
                        """CREATE TABLE sessions_source_path (
                        id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        source_session_id TEXT,
                        UNIQUE (source, source_path)
                    )"""
                    ).fetchone(),
                )
                lease.call(
                    "0035_source_path_identity:_upgrade",
                    lambda conn: conn.execute(
                        "INSERT INTO sessions_source_path SELECT * FROM sessions"
                    ).fetchone(),
                )
                lease.call(
                    "0035_source_path_identity:_upgrade",
                    lambda conn: conn.execute("DROP TABLE sessions").fetchone(),
                )
                lease.call(
                    "0035_source_path_identity:_upgrade",
                    lambda conn: conn.execute(
                        "ALTER TABLE sessions_source_path RENAME TO sessions"
                    ).fetchone(),
                )
                lease.call(
                    "0035_source_path_identity:_upgrade",
                    lambda conn: conn.execute(
                        "CREATE INDEX idx_sessions_source ON sessions(source)"
                    ).fetchone(),
                )
        if current < TARGET_VERSION:
            lease.call(
                "0035_source_path_identity:_upgrade",
                lambda conn: set_schema_version_to(conn, TARGET_VERSION),
            )
        lease.call(
            "0035_source_path_identity:_upgrade",
            lambda conn: conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (migration_id) VALUES (?)", [MIGRATION_ID]
            ).fetchone(),
        )
        lease.call(
            "0035_source_path_identity:_upgrade", lambda conn: conn.execute("COMMIT").fetchone()
        )
    except BaseException:
        lease.cleanup_call(
            "0035_source_path_identity:_upgrade", lambda conn: conn.execute("ROLLBACK").fetchone()
        )
        raise
    return True


register(Migration(id=MIGRATION_ID, target_version=TARGET_VERSION, upgrade=_upgrade))
