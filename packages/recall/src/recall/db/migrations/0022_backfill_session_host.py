"""Migration 0022: attribute unattributed sessions to this machine.

Migration 0019 added `session_state.host` and backfilled every pre-existing row
to the literal `local`. That value is the REQ-HOST-API-004 *sentinel* meaning
"unattributed", not a machine identity, so those rows have been indistinguishable
from genuinely-unattributable ones ever since — and until the REQ-FLEET-MERGE-002
fix they surfaced in fleet merges as a phantom host named `local`.

Rewrite the sentinel to this machine's short hostname. Rows already carrying a
real label are left alone, and the match is exact so hostnames that merely look
like the sentinel (`localhost`, `local-dev`) are untouched.

This is the first migration to mutate user *data* rather than schema shape, so it
records a pre-image first (REQ-MIG-008). A whole-database copy would be the wrong
instrument here: the pre-image for a single column is a few kilobytes, while a
production database can be well over 100 GiB. To reverse this migration:

    UPDATE session_state SET host = u.old_value
    FROM schema_migration_undo u
    WHERE u.migration_id = '0022_backfill_session_host'
      AND u.table_name = 'session_state'
      AND u.row_key = session_state.session_id;
"""

from __future__ import annotations

import logging

import duckdb

from recall.core.types import UNATTRIBUTED_HOST
from recall.db.migrations import Migration, register, table_exists

logger = logging.getLogger("recall.schema")

MIGRATION_ID = "0022_backfill_session_host"


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.core.types import default_session_host
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < 21:
        logger.warning(
            "%s requires schema_version >= 21; DB is at %d. "
            "Halting; earlier migrations must complete first.",
            MIGRATION_ID,
            current,
        )
        return False

    # The undo table is part of the v22 schema shape, so it exists at v22 whether
    # or not this particular host had anything to record.
    _ensure_undo_table(conn)

    hostname = default_session_host()
    if hostname == UNATTRIBUTED_HOST:
        # This machine cannot name itself, so there is nothing better than the
        # sentinel to write. Advancing without touching data is correct: a later
        # host that can name itself is not blocked, and no pre-image is owed.
        logger.info("%s: hostname unobtainable; leaving host labels as-is", MIGRATION_ID)
        set_schema_version_to(conn, 22)
        return True

    conn.execute("BEGIN TRANSACTION")
    try:
        if not table_exists(conn, "session_state"):
            logger.info("%s: session_state is absent; nothing to backfill", MIGRATION_ID)
            set_schema_version_to(conn, 22)
            conn.execute("COMMIT")
            return True

        # Record the pre-image before mutating. INSERT OR IGNORE keeps a re-run
        # from double-recording (REQ-MIG-003 idempotency pattern).
        conn.execute(
            """
            INSERT OR IGNORE INTO schema_migration_undo
                (migration_id, table_name, row_key, column_name, old_value)
            SELECT ?, 'session_state', session_id, 'host', host
            FROM session_state
            WHERE host = ?
            """,
            [MIGRATION_ID, UNATTRIBUTED_HOST],
        )
        row = conn.execute(
            "SELECT COUNT(*) FROM session_state WHERE host = ?",
            [UNATTRIBUTED_HOST],
        ).fetchone()
        affected = int(row[0]) if row else 0
        conn.execute(
            "UPDATE session_state SET host = ? WHERE host = ?",
            [hostname, UNATTRIBUTED_HOST],
        )
        set_schema_version_to(conn, 22)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        logger.exception("Failed to apply %s", MIGRATION_ID)
        return False

    logger.info(
        "%s: attributed %d unattributed session(s) to host=%s",
        MIGRATION_ID,
        affected,
        hostname,
    )
    return True


def _ensure_undo_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the pre-image table (present in schema.sql for fresh DBs)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migration_undo (
            migration_id TEXT NOT NULL,
            table_name TEXT NOT NULL,
            row_key TEXT NOT NULL,
            column_name TEXT NOT NULL,
            old_value TEXT,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (migration_id, table_name, row_key, column_name)
        )
        """
    )


register(
    Migration(
        id=MIGRATION_ID,
        target_version=22,
        upgrade=_upgrade,
    )
)
