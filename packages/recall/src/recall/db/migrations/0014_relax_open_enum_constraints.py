"""Migration 0014: relax CHECK constraints on sessions.source and message_state.role.

This is the first migration in the new framework. It replaces the one-off
_relax_v13_open_enum_constraints hack that was wired directly into ensure_schema.

Why this migration exists (see handoff 2356 and SPEC Migration Policy):
- Adding Grok (and future agents) required extending the open enum for `source`.
- DuckDB 1.5.x does not support ALTER TABLE ... DROP CONSTRAINT for CHECK constraints
  (NotImplementedException). The only safe way to widen the constraint is to
  recreate the table with the new definition.
- sessions and message_state are tiny (even on years of history) so the
  CREATE new + INSERT SELECT * + DROP + RENAME is fast and safe.
- message_state contains full message content/thinking text; we still accept the
  copy cost because the alternative (--recreate for every user with history) is worse.
- messages and tool_calls are deliberately left alone (large); they never had
  CHECKs on role/source.

The migration is idempotent:
- If not v13, do nothing (record is still written by runner).
- If v13 but CHECKs already absent (e.g. manual relax or re-run), the
  duckdb_constraints() probe returns needs=False and we skip DDL.
- INSERT OR IGNORE on schema_migrations (done by runner after upgrade returns).

Version bump only happens for actual v13 DBs (inside this upgrade). v11 DBs
that reach here record the migration but leave version=11 so the final
mismatch check in ensure_schema still forces --recreate (structural drift
between schema.sql revisions cannot be replayed safely).

This pattern lets "add support for NewAgentX" be: parser + Source entry +
(optional) small migration that relaxes/extends something + SCHEMA_VERSION bump.
No user ever has to --recreate again for this class of change.
"""

from __future__ import annotations

import logging

import duckdb

from recall.db.migrations import Migration, register

logger = logging.getLogger("recall.schema")


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    """Perform the v13→v14 CHECK relaxation (or no-op if not applicable).

    Returns True if the DB is now in the desired state for this migration
    (either we successfully relaxed the constraints, or they were already absent).
    Returns False if a relaxation was attempted but failed.

    The runner will only record this migration as applied when True is returned.
    """
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current != 13:
        return True

    # Check current constraint state
    rows = conn.execute(
        """
        SELECT expression FROM duckdb_constraints()
        WHERE table_name = 'sessions' AND constraint_type = 'CHECK'
        """
    ).fetchall()
    # For a v13 DB, any remaining CHECK on the column means we need to recreate
    # the table with the open definition. We no longer rely on detecting the
    # exact old v13 shape, because users (or previous runs) may have partially
    # widened the CHECK already.
    needs_sessions = bool(rows)

    rows = conn.execute(
        """
        SELECT expression FROM duckdb_constraints()
        WHERE table_name = 'message_state' AND constraint_type = 'CHECK'
        """
    ).fetchall()
    needs_message_state = bool(rows)

    # No-op case: v13 DB that is already in the desired state
    # (e.g. user previously ran the old _relax hack, or constraints were never tight).
    # We must still bump the version for proper idempotency.
    if not needs_sessions and not needs_message_state:
        set_schema_version_to(conn, 14)
        return True

    sessions_ok = True
    message_state_ok = True

    if needs_sessions:
        sessions_ok = _relax_sessions_table(conn)

    if needs_message_state:
        message_state_ok = _relax_message_state_table(conn)

    if sessions_ok and message_state_ok:
        set_schema_version_to(conn, 14)
        return True
    else:
        # At least one relaxation failed. Do not bump version.
        # The migration will remain pending and can be retried on next ensure_schema.
        return False


def _relax_sessions_table(conn: duckdb.DuckDBPyConnection) -> bool:
    """Attempt to relax the sessions.source CHECK constraint.

    Returns True on success.
    Returns False if the table recreation failed (after rollback).
    """
    logger.info("Relaxing CHECK constraint on sessions table (v13 → v14 compatibility)")
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(
            """
            CREATE TABLE sessions_new (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_path TEXT UNIQUE NOT NULL,
                source_session_id TEXT
            )
            """
        )
        conn.execute("INSERT INTO sessions_new SELECT * FROM sessions")
        conn.execute("DROP TABLE sessions")
        conn.execute("ALTER TABLE sessions_new RENAME TO sessions")
        # Recreate indexes lost during table recreation
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source)")
        conn.execute("COMMIT")
        logger.info("sessions table relaxed successfully")
        return True
    except Exception:
        conn.execute("ROLLBACK")
        logger.exception("Failed to relax sessions table constraint")
        return False


def _relax_message_state_table(conn: duckdb.DuckDBPyConnection) -> bool:
    """Attempt to relax the message_state.role CHECK constraint.

    Returns True on success.
    Returns False if the table recreation failed (after rollback).
    """
    logger.info("Relaxing CHECK constraint on message_state.role (v13 → v14 compatibility)")
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(
            """
            CREATE TABLE message_state_new (
                message_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT,
                thinking TEXT,
                timestamp TIMESTAMP,
                has_thinking BOOLEAN DEFAULT FALSE
            )
            """
        )
        conn.execute("INSERT INTO message_state_new SELECT * FROM message_state")
        conn.execute("DROP TABLE message_state")
        conn.execute("ALTER TABLE message_state_new RENAME TO message_state")
        # Recreate indexes lost during table recreation
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_message_state_has_thinking "
            "ON message_state(has_thinking)"
        )
        conn.execute("COMMIT")
        logger.info("message_state table relaxed successfully")
        return True
    except Exception:
        conn.execute("ROLLBACK")
        logger.exception("Failed to relax message_state table constraint")
        return False


# Register at import time (triggered by pkgutil discovery in migrations/__init__.py).
register(
    Migration(
        id="0014_relax_open_enum_constraints",
        target_version=14,
        upgrade=_upgrade,
    )
)
