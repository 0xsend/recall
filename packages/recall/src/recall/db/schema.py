from __future__ import annotations

import logging
from enum import Enum, auto
from pathlib import Path

import duckdb

from recall.core.schema_version import SchemaTooNewError
from recall.db.lease import DbLease, StandaloneLifetime, TransactionCapability

from .migrations import run_pending_migrations_on

logger = logging.getLogger("recall.schema")

SCHEMA_VERSION = 41
INDEX_MIGRATION_VERSION = 1


class SchemaChange(Enum):
    UNCHANGED = auto()
    CHANGED = auto()


def check_schema_compatibility(conn: duckdb.DuckDBPyConnection) -> int | None:
    """Read the logical version without modifying or migrating the database."""
    if not _schema_version_table_exists(conn):
        return None
    current = _get_schema_version(conn)
    if current > SCHEMA_VERSION:
        raise SchemaTooNewError(current, SCHEMA_VERSION)
    return current


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
    with StandaloneLifetime().lease(conn) as lease:
        ensure_schema_on(lease, embed_dim=embed_dim)


def ensure_schema_lenient(conn: duckdb.DuckDBPyConnection, *, embed_dim: int = 384) -> None:
    with StandaloneLifetime().lease(conn) as lease:
        ensure_schema_on(lease, embed_dim=embed_dim, lenient=True)


def ensure_schema_on(
    lease: DbLease, *, embed_dim: int = 384, lenient: bool = False
) -> SchemaChange:
    """Observe schema/migration units before any runner's fail-soft policy."""
    current = lease.call("schema_admission", check_schema_compatibility)
    if current is None:
        _initialize_schema_on(lease, embed_dim)
        return SchemaChange.CHANGED
    previous = current
    run_pending_migrations_on(lease, current)
    current = lease.call("postmigration_version", _get_schema_version)
    if current != SCHEMA_VERSION:
        if lenient:
            raise RuntimeError(
                f"incomplete required schema (expected {SCHEMA_VERSION}, found {current}); "
                "use verified offline repair or matched restore"
            )
        raise RuntimeError(
            f"schema version mismatch (expected {SCHEMA_VERSION}, found {current}). "
            "Run `recall index --recreate --yes` to rebuild the database."
        )
    embedding_change = lease.call(
        "schema_embedding_dimensions", lambda conn: _check_embedding_dimensions(conn, embed_dim)
    )
    if current != previous or embedding_change is SchemaChange.CHANGED:
        return SchemaChange.CHANGED
    return SchemaChange.UNCHANGED


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


def _initialize_schema_on(lease: DbLease, embed_dim: int) -> None:
    with lease.transaction("initialize_schema") as transaction:
        initialize_schema_in(transaction, embed_dim)


def initialize_schema_in(transaction: TransactionCapability, embed_dim: int) -> None:
    """Create table, logical version and audit inside the caller's fence transaction."""
    from recall.db.repair import initialize_repair_in

    transaction.call("schema_body", lambda conn: _apply_schema(conn, embed_dim))
    initialize_repair_in(transaction)
    transaction.call("schema_version", _set_schema_version)
    transaction.call(
        "schema_embedding_dimensions", lambda conn: _store_embedding_dimensions(conn, embed_dim)
    )
    transaction.call("schema_index_migration", _initialize_fresh_index_migration)


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


def _check_embedding_dimensions(conn: duckdb.DuckDBPyConnection, configured: int) -> SchemaChange:
    row = conn.execute("SELECT embedding_dimensions FROM runtime_state WHERE singleton").fetchone()
    if row is None or row[0] is None:
        _store_embedding_dimensions(conn, configured)
        return SchemaChange.UNCHANGED
    stored = int(row[0])
    if stored != configured:
        logger.warning(
            "embedding dimensions changed (%d → %d), recreating embedding tables",
            stored,
            configured,
        )
        recreate_embedding_tables(conn, configured)
        return SchemaChange.CHANGED
    return SchemaChange.UNCHANGED
