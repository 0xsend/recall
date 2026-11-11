"""Migration 0029: date the fatal signature on runtime_state.

`last_fatal_signature` and `fatal_repeat_count` carried no timestamp, so a
signature from a repair that landed months ago read as a live failure in
`recall daemon status` (REQ-RESIL-014). Additive and idempotent: one nullable
`ADD COLUMN IF NOT EXISTS`, no data rewritten. Existing rows stay NULL — the
timestamp of a failure recorded before this migration is not recoverable.
"""

from __future__ import annotations

import logging

from recall.db.lease import DbLease
from recall.db.migrations import Migration, register, table_exists

logger = logging.getLogger("recall.schema")

MIGRATION_ID = "0029_fatal_timestamp"


def _upgrade(lease: DbLease) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = lease.call("migration:0029:version_read", lambda conn: get_schema_version(conn))
    if current < 28:
        logger.warning(
            "%s requires schema_version >= 28; DB is at %d. "
            "Halting; earlier migrations must complete first.",
            MIGRATION_ID,
            current,
        )
        return False

    if not lease.call("migration:0029:catalog", lambda conn: table_exists(conn, "runtime_state")):
        logger.info("%s: runtime_state is absent; nothing to add", MIGRATION_ID)
        lease.call("migration:0029:version_write", lambda conn: set_schema_version_to(conn, 29))
        return True

    try:
        lease.call(
            "migration:0029:alter",
            lambda conn: conn.execute(
                "ALTER TABLE runtime_state ADD COLUMN IF NOT EXISTS last_fatal_at TIMESTAMP"
            ).fetchone(),
        )
        lease.call("migration:0029:version_write", lambda conn: set_schema_version_to(conn, 29))
    except Exception:
        logger.exception("Failed to apply %s", MIGRATION_ID)
        return False
    return True


register(
    Migration(
        id=MIGRATION_ID,
        target_version=29,
        upgrade=_upgrade,
    )
)
