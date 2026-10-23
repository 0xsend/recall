"""Backed-up semantic work metadata; payloads are unchanged.

Supported rollback restores the matched DuckDB/SQLite preimage with 5fa253b.
Offline preimage-equivalent fixtures only may reverse, in one transaction:
DROP VIEW semantic_pending; DROP TABLE semantic_work_counts;
DROP TABLE semantic_work; DROP TABLE semantic_work_state;
DELETE FROM schema_migrations WHERE migration_id='0033_semantic_work';
DELETE FROM schema_version WHERE version=33;
Never perform an in-place version decrement on an active database.
"""

from __future__ import annotations

import duckdb

from recall.db.migrations import Migration, register
from recall.db.semantic_work import SemanticSchemaUpgrade, SemanticWorkUnavailable


def upgrade(
    conn: duckdb.DuckDBPyConnection, capability: SemanticSchemaUpgrade | None = None
) -> bool:
    if capability is None:
        raise SemanticWorkUnavailable(
            "schema 33 requires a verified pair preimage; start the approved daemon owner"
        )
    capability.install(conn)
    return True


register(Migration("0033_semantic_work", 33, upgrade))
