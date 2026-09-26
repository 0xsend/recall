"""Migration framework for recall DuckDB schema upgrades.

Versioned migrations live as 00NN_name.py modules in this package. They are
auto-discovered via pkgutil on package import (modeled on parsers/registry.py).

Each migration registers a Migration(id, target_version, upgrade) where upgrade(conn)
is idempotent, may use lazy imports for schema helpers, and only mutates small tables
(sessions, message_state) when necessary for CHECK or schema-shape changes.

The runner is invoked from db/schema.py:ensure_schema / ensure_schema_lenient.
See SPEC.md "Migration Policy" (REQ-MIG-001..007) for full requirements.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass

import duckdb

logger = logging.getLogger("recall.schema")


@dataclass(frozen=True)
class Migration:
    """A single schema migration step.

    - id: stable identifier, e.g. "0014_relax_open_enum_constraints" (lexical sort order)
    - target_version: the SCHEMA_VERSION this migration advances toward (or achieves)
    - upgrade: callable that performs the work (must be idempotent and safe)
    """

    id: str
    target_version: int
    upgrade: Callable[[duckdb.DuckDBPyConnection], bool]


_MIGRATIONS: list[Migration] = []


def register(migration: Migration) -> None:
    """Register a migration (called at import time from 00NN_*.py modules)."""
    _MIGRATIONS.append(migration)


def get_migrations() -> list[Migration]:
    """Return all registered migrations, sorted by id for deterministic order."""
    return sorted(_MIGRATIONS, key=lambda m: m.id)


def table_exists(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    """Whether `table` is present in the main schema.

    A migration runs against whatever shape the database already has, so most
    of them have to ask this before touching a table.
    """
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [table],
    ).fetchone()
    return bool(row and row[0])


# Auto-discover and import all 00NN_*.py modules when this package is loaded.
# This populates _MIGRATIONS without explicit imports in schema.py.
def _discover_migrations() -> None:
    package_name = __name__
    for _importer, modname, ispkg in pkgutil.iter_modules(__path__):
        if modname[0].isdigit() and not ispkg:
            importlib.import_module(f".{modname}", package_name)


_discover_migrations()


def _ensure_migrations_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Ensure the audit table exists (called by runner; safe on v13 DBs that lack it)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def get_pending_migrations(
    conn: duckdb.DuckDBPyConnection, current_version: int
) -> list[Migration]:
    """Return migrations that have not been applied and advance the current version."""
    _ensure_migrations_table(conn)
    applied = {
        row[0] for row in conn.execute("SELECT migration_id FROM schema_migrations").fetchall()
    }
    pending: list[Migration] = []
    for m in get_migrations():
        if m.id not in applied and current_version < m.target_version:
            pending.append(m)
    return pending


def run_pending_migrations(conn: duckdb.DuckDBPyConnection, current_version: int) -> None:
    """Run all pending migrations for the given current_version, recording each only on success.

    The upgrade() callable must return True if the desired schema state was achieved
    (either by performing work or because the DB was already in the desired state).
    It should return False (or raise) if the migration could not complete successfully.
    Only successful migrations are recorded in schema_migrations.
    """
    for m in get_pending_migrations(conn, current_version):
        logger.info("Applying migration %s (target version %d)", m.id, m.target_version)
        try:
            success = m.upgrade(conn)
        except Exception:
            logger.exception("Migration %s failed with exception", m.id)
            success = False

        if success:
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (migration_id) VALUES (?)",
                [m.id],
            )
            logger.info("Migration %s applied successfully", m.id)
        else:
            logger.warning(
                "Migration %s did not complete successfully; halting migration pass",
                m.id,
            )
            break


# Public list for introspection / tests (populated by discovery + register calls).
MIGRATIONS: list[Migration] = _MIGRATIONS
