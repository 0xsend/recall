"""Migration 0020: fingerprint parser sidecars in the index change signal.

Adds session_state.sidecar_mtime.

The change signal was the session file's mtime/size alone, so metadata a parser
sourced from a sibling sidecar (Grok's summary.json / signals.json, REQ-PARSE-014)
could never refresh: chat_history.jsonl stops changing when the session ends, and
the row is skipped as unchanged from then on. Rows written before this column
therefore hold whatever the parser produced at first index -- for Grok that left
started_at/ended_at NULL on every session indexed before sidecar support landed.

The column is left NULL here rather than defaulted to 0. A NULL reads as "no
sidecar stamp recorded" and compares unequal to any real fingerprint, so exactly
the sources that declare sidecars re-index once and backfill. Sources with no
sidecars fingerprint to 0.0, which equals the column default for rows written
after this migration, so they never churn.

See SPEC REQ-INDEX-018.
"""

from __future__ import annotations

import logging

import duckdb

from recall.db.migrations import Migration, register

logger = logging.getLogger("recall.schema")


def _upgrade(conn: duckdb.DuckDBPyConnection) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = get_schema_version(conn)
    if current < 19:
        logger.warning(
            "0020_sidecar_mtime requires schema_version >= 19; DB is at %d. "
            "Halting; earlier migrations must complete first.",
            current,
        )
        return False

    tables = conn.execute(
        """
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_name = 'session_state'
        """
    ).fetchone()
    if not tables or not tables[0]:
        # Minimal simulated DBs in cutover tests omit session_state; real
        # installs always have it from schema.sql.
        logger.info("session_state missing; skipping ADD COLUMN sidecar_mtime")
        set_schema_version_to(conn, 20)
        return True

    existing = {
        str(row[1]) for row in conn.execute("PRAGMA table_info('session_state')").fetchall()
    }
    if "sidecar_mtime" not in existing:
        try:
            conn.execute("ALTER TABLE session_state ADD COLUMN sidecar_mtime DOUBLE")
        except duckdb.Error:
            logger.exception("0020_sidecar_mtime: ADD COLUMN sidecar_mtime failed")
            return False

    set_schema_version_to(conn, 20)
    return True


register(
    Migration(
        id="0020_sidecar_mtime",
        target_version=20,
        upgrade=_upgrade,
    )
)
