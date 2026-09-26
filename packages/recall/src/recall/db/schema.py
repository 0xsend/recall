from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from .migrations import run_pending_migrations

logger = logging.getLogger("recall.schema")

SCHEMA_VERSION = 31
INDEX_MIGRATION_VERSION = 1


# Public wrappers for migration authors (see db/migrations/*.py).
# These are defined early enough for runtime lookup; the migration import is at top
# of file (E402) because upgrade() fns only call the wrappers via lazy import at
# execution time, after the whole module is loaded.
def get_schema_version(conn: duckdb.DuckDBPyConnection) -> int:
    """Return the current logical schema version (MAX from schema_version table).

    Intended for use inside migration upgrade() callables via lazy import.
    """
    return _get_schema_version(conn)


def set_schema_version(conn: duckdb.DuckDBPyConnection) -> None:
    """Record that the DB has been advanced to SCHEMA_VERSION.

    Intended for fresh schema creation and full recreate paths.
    Inserts a new row (history is kept; callers use MAX).
    """
    _set_schema_version(conn)


def set_schema_version_to(conn: duckdb.DuckDBPyConnection, version: int) -> None:
    """Record that the DB has been advanced to a migration's target version.

    Migration modules must call this instead of set_schema_version() so replaying
    an older migration cannot jump straight to the latest global SCHEMA_VERSION.
    """
    _set_schema_version_to(conn, version)


def ensure_schema(conn: duckdb.DuckDBPyConnection, *, embed_dim: int = 384) -> None:
    if not _schema_version_table_exists(conn):
        _apply_schema(conn, embed_dim)
        _set_schema_version(conn)
        _store_embedding_dimensions(conn, embed_dim)
        _initialize_fresh_index_migration(conn)
        return

    current = _get_schema_version(conn)

    # Run any pending migrations (e.g. 0014 v13→v14 CHECK relaxation for Grok).
    # This replaces the old one-off _relax_v13... special case. The framework is
    # idempotent, safe for large DBs, and only bumps version when appropriate
    # (very old <13 DBs still hit the mismatch error below and require --recreate).
    run_pending_migrations(conn, current)
    current = _get_schema_version(conn)

    if current != SCHEMA_VERSION:
        raise RuntimeError(
            f"schema version mismatch (expected {SCHEMA_VERSION}, found {current}). "
            "Run `recall index --recreate --yes` to rebuild the database."
        )
    _check_embedding_dimensions(conn, embed_dim)


def ensure_schema_lenient(conn: duckdb.DuckDBPyConnection, *, embed_dim: int = 384) -> None:
    """Apply schema to a fresh DB but tolerate version mismatches on existing ones.

    Used by the daemon startup path so it can open a stale-schema database
    without crashing.  A fresh (empty) database still gets the full schema
    applied so all tables exist for normal operation.
    """
    if not _schema_version_table_exists(conn):
        _apply_schema(conn, embed_dim)
        _set_schema_version(conn)
        _store_embedding_dimensions(conn, embed_dim)
        _initialize_fresh_index_migration(conn)
        return

    current = _get_schema_version(conn)

    # Run pending migrations (same path as strict ensure; warnings only on mismatch).
    # The 0014 migration (and future ones) log their own "Applying migration..." messages.
    run_pending_migrations(conn, current)
    current = _get_schema_version(conn)

    if current != SCHEMA_VERSION:
        logger.warning(
            "schema version mismatch (expected %d, found %d); "
            "daemon will start — run `recall index --recreate --yes` to fix",
            SCHEMA_VERSION,
            current,
        )
        return
    _check_embedding_dimensions(conn, embed_dim)


def recreate_embedding_tables(conn: duckdb.DuckDBPyConnection, embed_dim: int) -> None:
    """Drop and recreate embedding tables with the given dimensions."""
    conn.execute("DROP TABLE IF EXISTS message_embeddings")
    conn.execute("DROP TABLE IF EXISTS tool_call_embeddings")
    conn.execute("DROP TABLE IF EXISTS embedding_cache")
    conn.execute(
        f"""
        CREATE TABLE message_embeddings (
            message_id TEXT PRIMARY KEY,
            content_embedding FLOAT[{embed_dim}],
            thinking_embedding FLOAT[{embed_dim}]
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE tool_call_embeddings (
            tool_call_id TEXT PRIMARY KEY,
            bash_embedding FLOAT[{embed_dim}]
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE embedding_cache (
            cache_key TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('content', 'thinking', 'bash')),
            raw_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            embedding FLOAT[{embed_dim}] NOT NULL,
            normalization_version INTEGER NOT NULL DEFAULT 1,
            context_version INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding_cache_kind ON embedding_cache(kind)")
    _store_embedding_dimensions(conn, embed_dim)


def _schema_version_table_exists(conn: duckdb.DuckDBPyConnection) -> bool:
    rows = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'schema_version'"
    ).fetchone()
    return bool(rows and rows[0])


def _apply_schema(conn: duckdb.DuckDBPyConnection, embed_dim: int = 384) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    sql = schema_path.read_text(encoding="utf-8")
    sql = sql.replace("__EMBED_DIM__", str(embed_dim))
    conn.execute(sql)


def _initialize_fresh_index_migration(conn: duckdb.DuckDBPyConnection) -> None:
    """Record the current index-migration version on a newly created database.

    Existing databases keep applied version 0 until versioned historical
    index migration completes (REQ-RECON-010).
    """
    conn.execute("INSERT OR IGNORE INTO runtime_state (singleton) VALUES (TRUE)")
    conn.execute(
        "UPDATE runtime_state SET index_migration_version = ? WHERE singleton",
        [INDEX_MIGRATION_VERSION],
    )
    conn.execute(
        """INSERT OR IGNORE INTO index_migration_jobs (singleton, target_version, phase)
           VALUES (TRUE, 0, 'idle')"""
    )


def _set_schema_version(conn: duckdb.DuckDBPyConnection) -> None:
    _set_schema_version_to(conn, SCHEMA_VERSION)


def _set_schema_version_to(conn: duckdb.DuckDBPyConnection, version: int) -> None:
    conn.execute("INSERT INTO schema_version (version) VALUES (?)", [version])


def _get_schema_version(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    if row is None or row[0] is None:
        raise RuntimeError("schema_version table is empty")
    return int(row[0])


def _store_embedding_dimensions(conn: duckdb.DuckDBPyConnection, dimensions: int) -> None:
    conn.execute("UPDATE runtime_state SET embedding_dimensions = ?", [dimensions])


def _check_embedding_dimensions(conn: duckdb.DuckDBPyConnection, configured: int) -> None:
    row = conn.execute("SELECT embedding_dimensions FROM runtime_state WHERE singleton").fetchone()
    if row is None or row[0] is None:
        _store_embedding_dimensions(conn, configured)
        return
    stored = int(row[0])
    if stored != configured:
        logger.warning(
            "embedding dimensions changed (%d → %d), recreating embedding tables",
            stored,
            configured,
        )
        recreate_embedding_tables(conn, configured)
