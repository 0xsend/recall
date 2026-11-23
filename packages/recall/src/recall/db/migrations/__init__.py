"""Migration framework for recall DuckDB schema upgrades.

Versioned migrations live as 00NN_name.py modules in this package. They are
auto-discovered via pkgutil on package import (modeled on parsers/registry.py).

Each migration registers a Migration(id, target_version, upgrade) where upgrade(lease)
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

from recall.core.time import utcnow_naive
from recall.db.lease import DbLease, StandaloneLifetime

logger = logging.getLogger("recall.schema")


@dataclass(frozen=True)
class AlreadyAudited:
    migration_id: str
    target_version: int


@dataclass(frozen=True)
class Migration:
    """A single schema migration step.

    - id: stable identifier, e.g. "0014_relax_open_enum_constraints" (lexical sort order)
    - target_version: the SCHEMA_VERSION this migration advances toward (or achieves)
    - upgrade: idempotent callable requiring an active observed DB lease
    """

    id: str
    target_version: int
    upgrade: Callable[[DbLease], bool | AlreadyAudited]


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


def record_migration_audit(conn: duckdb.DuckDBPyConnection, migration_id: str) -> None:
    """Record a migration in ``schema_migrations`` with naive-UTC ``applied_at`` when present."""
    has_applied_at = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = 'main'
          AND table_name = 'schema_migrations'
          AND column_name = 'applied_at'
        """
    ).fetchone()
    if has_applied_at and has_applied_at[0]:
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (migration_id, applied_at) VALUES (?, ?)",
            [migration_id, utcnow_naive()],
        )
    else:
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (migration_id) VALUES (?)",
            [migration_id],
        )


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
            applied_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
        )
        """
    )


def get_pending_migrations(
    conn: duckdb.DuckDBPyConnection, current_version: int
) -> list[Migration]:
    """Return migrations that have not been applied and advance the current version."""
    with StandaloneLifetime().lease(conn) as lease:
        return get_pending_migrations_on(lease, current_version)


def get_pending_migrations_on(lease: DbLease, current_version: int) -> list[Migration]:
    lease.call("migration_audit_bootstrap", _ensure_migrations_table)
    applied = {
        row[0]
        for row in lease.call(
            "migration_audit_read",
            lambda conn: conn.execute("SELECT migration_id FROM schema_migrations").fetchall(),
        )
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
    It returns False only when the database is not eligible for it (for example an
    earlier schema version) and raises when its work fails. A migration that opens a
    transaction does so through ``DbLease.transaction``, so a failed commit is
    observed and rolled back, and a failed rollback is reported with it.
    Only successful migrations are recorded in schema_migrations.
    """
    with StandaloneLifetime().lease(conn) as lease:
        run_pending_migrations_on(lease, current_version)


def run_pending_migrations_on(lease: DbLease, current_version: int) -> None:
    for m in get_pending_migrations_on(lease, current_version):
        logger.info("Applying migration %s (target version %d)", m.id, m.target_version)
        try:
            success = lease.call(f"migration_body:{m.id}", lambda conn, m=m: m.upgrade(lease))
        except BaseExceptionGroup:
            # A transaction whose rollback also failed may still be open on this
            # connection: a soft failure here would let later work commit it.
            raise
        except Exception:
            logger.exception("Migration %s failed with exception", m.id)
            success = False

        if isinstance(success, AlreadyAudited):
            assert (success.migration_id, success.target_version) == (m.id, m.target_version)
            logger.info("Migration %s applied and audited atomically", m.id)
        elif success:
            lease.call(
                f"migration_audit:{m.id}",
                lambda conn, m=m: record_migration_audit(conn, m.id),
            )
            logger.info("Migration %s applied successfully", m.id)
        else:
            # A body may swallow a native error after its statement observed it.
            # Admission re-raises a recognized fatal here instead of reporting
            # a soft failure that the caller can mistake for a normal return.
            lease.call(f"migration_outcome:{m.id}", lambda conn: None)
            logger.warning(
                "Migration %s did not complete successfully; halting migration pass",
                m.id,
            )
            break


# Public list for introspection / tests (populated by discovery + register calls).
MIGRATIONS: list[Migration] = _MIGRATIONS
