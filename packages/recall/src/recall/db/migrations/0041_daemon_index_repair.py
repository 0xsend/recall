"""Install the recovery state, logical fence and audit in one transaction."""

from __future__ import annotations

import duckdb

from recall.db.migrations import AlreadyAudited, Migration, register
from recall.db.repair import FENCE_VERSION, MIGRATION_ID, initialize_repair


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool | AlreadyAudited:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    # Schema 36 is the oldest layout with all prerequisite migrations.
    if current < 36:
        return False
    conn.execute("BEGIN")
    try:
        initialize_repair(conn)
        if current < FENCE_VERSION:
            set_schema_version_to(conn, FENCE_VERSION)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return AlreadyAudited(MIGRATION_ID, FENCE_VERSION)


register(Migration(MIGRATION_ID, FENCE_VERSION, _upgrade))
