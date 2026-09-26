"""Migration 0023: failure-signature memory columns on runtime_state.

A daemon that dies on a persistent DuckDB fault cannot record the failure in
the invalidated instance, so it spools a marker file and folds it into
`runtime_state` on the next start (REQ-RESIL-014). These columns hold what the
next start needs to decide between repairing and refusing (REQ-RESIL-015/016)
and the ENOSPC verification flag (REQ-RESIL-019). Additive and idempotent:
`ADD COLUMN IF NOT EXISTS` with defaults, no data rewritten.
"""

from __future__ import annotations

import logging

import duckdb

from recall.db.migrations import Migration, register, table_exists

logger = logging.getLogger("recall.schema")

MIGRATION_ID = "0023_fatal_failure_memory"

# DuckDB's ALTER TABLE ADD COLUMN rejects constraints ("not yet supported"), so
# the counters are nullable with a default; readers coalesce NULL to 0 / FALSE.
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("last_fatal_signature", "TEXT"),
    ("fatal_repeat_count", "INTEGER DEFAULT 0"),
    ("last_index_repair_at", "TIMESTAMP"),
    ("last_index_repair_signature", "TEXT"),
    ("needs_index_verification", "BOOLEAN DEFAULT FALSE"),
)


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < 22:
        logger.warning(
            "%s requires schema_version >= 22; DB is at %d. "
            "Halting; earlier migrations must complete first.",
            MIGRATION_ID,
            current,
        )
        return False

    if not table_exists(conn, "runtime_state"):
        logger.info("%s: runtime_state is absent; nothing to add", MIGRATION_ID)
        set_schema_version_to(conn, 23)
        return True

    try:
        for name, definition in _COLUMNS:
            conn.execute(f"ALTER TABLE runtime_state ADD COLUMN IF NOT EXISTS {name} {definition}")
        set_schema_version_to(conn, 23)
    except Exception:
        logger.exception("Failed to apply %s", MIGRATION_ID)
        return False
    return True


register(
    Migration(
        id=MIGRATION_ID,
        target_version=23,
        upgrade=_upgrade,
    )
)
