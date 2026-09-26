"""Migration 0018: add the SQLite FTS sidecar pending queue."""

from __future__ import annotations

import logging

import duckdb

from recall.db.migrations import Migration, register

logger = logging.getLogger("recall.schema")


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < 17:
        logger.warning(
            "0018_fts_sidecar_pending requires schema_version >= 17; DB is at %d. "
            "Halting; earlier migrations must complete first.",
            current,
        )
        return False

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fts_sidecar_pending (
                kind TEXT NOT NULL,
                id TEXT NOT NULL,
                op TEXT NOT NULL,
                queued_at TIMESTAMP NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_fts_sidecar_pending_kind_id
            ON fts_sidecar_pending(kind, id)
            """
        )
    except Exception:
        logger.exception("Failed to add SQLite FTS sidecar pending queue")
        return False

    if current < 18:
        set_schema_version_to(conn, 18)
    return True


register(
    Migration(
        id="0018_fts_sidecar_pending",
        target_version=18,
        upgrade=_upgrade,
    )
)
