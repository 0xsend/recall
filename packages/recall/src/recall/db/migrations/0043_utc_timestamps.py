"""Migration 0043: standardize stored DuckDB timestamps to naive UTC (#356).

DuckDB ``TIMESTAMP`` columns have historically stored host-local wall-clock
timestamps because aware Python datetimes and ``CURRENT_TIMESTAMP`` down-convert
to the process-local timezone. This migration establishes the naive-UTC storage
invariant for operational tables and installed table defaults:

1. Alters column defaults on existing upgraded tables to
   ``(CURRENT_TIMESTAMP AT TIME ZONE 'UTC')`` so no future insert can inherit a
   host-local default.
2. Converts the three small operational tables (``usage_events``,
   ``usage_log_cursors``, ``live_marks``) from host-local naive ``TIMESTAMP`` to
   naive UTC, recording exact pre-images in ``schema_migration_undo`` before
   each mutation (``REQ-MIG-008``). On UTC hosts, the ``IS DISTINCT FROM`` guard
   skips every row and records zero undo rows.
3. Leaves large historical tables (``session_state``, ``message_state``,
   ``tool_results``) untouched on startup; bumping ``INDEX_MIGRATION_VERSION``
   to ``2`` makes existing hosts eligible for backed-up historical index
   migration (``REQ-RECON-010``) to restore exact transcript UTC without
   corrupting sessions originally indexed in a different timezone.

All DDL defaults, pre-image inserts, data updates, schema-version advance, and
audit insert commit atomically inside one observed lease transaction after
verifying schema 42 and the schema-41 recovery fence.

To reverse the operational table mutations from ``schema_migration_undo`` (using
an atomic rebuild for ``usage_events`` to avoid DuckDB 1.5.5's multi-row-group
ART ``UPDATE`` bug on ``idx_usage_events_ts``):

    BEGIN;
    DROP TABLE IF EXISTS usage_events_restore;
    CREATE TABLE usage_events_restore (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        source_session_id TEXT NOT NULL,
        session_id TEXT,
        ts TIMESTAMP,
        prompt_tokens BIGINT,
        cached_prompt_tokens BIGINT,
        completion_tokens BIGINT,
        reasoning_tokens BIGINT,
        host TEXT,
        harvested_at TIMESTAMP NOT NULL
    );
    INSERT INTO usage_events_restore (
        id, source, source_session_id, session_id, ts,
        prompt_tokens, cached_prompt_tokens, completion_tokens,
        reasoning_tokens, host, harvested_at
    )
    SELECT
        e.id, e.source, e.source_session_id, e.session_id,
        COALESCE(CAST(u_ts.old_value AS TIMESTAMP), e.ts),
        e.prompt_tokens, e.cached_prompt_tokens, e.completion_tokens,
        e.reasoning_tokens, e.host,
        COALESCE(CAST(u_h.old_value AS TIMESTAMP), e.harvested_at)
    FROM usage_events e
    LEFT JOIN schema_migration_undo u_ts
      ON u_ts.migration_id = '0043_utc_timestamps'
     AND u_ts.table_name = 'usage_events'
     AND u_ts.column_name = 'ts'
     AND u_ts.row_key = e.id
    LEFT JOIN schema_migration_undo u_h
      ON u_h.migration_id = '0043_utc_timestamps'
     AND u_h.table_name = 'usage_events'
     AND u_h.column_name = 'harvested_at'
     AND u_h.row_key = e.id;
    DROP TABLE usage_events;
    ALTER TABLE usage_events_restore RENAME TO usage_events;
    CREATE INDEX idx_usage_events_source_sid
        ON usage_events(source, source_session_id);
    CREATE INDEX idx_usage_events_session ON usage_events(session_id);
    CREATE INDEX idx_usage_events_ts ON usage_events(ts);

    UPDATE usage_log_cursors
    SET updated_at = CAST(u.old_value AS TIMESTAMP)
    FROM schema_migration_undo u
    WHERE u.migration_id = '0043_utc_timestamps'
      AND u.table_name = 'usage_log_cursors'
      AND u.column_name = 'updated_at'
      AND u.row_key = usage_log_cursors.path;

    UPDATE live_marks
    SET marked_at = CAST(u.old_value AS TIMESTAMP)
    FROM schema_migration_undo u
    WHERE u.migration_id = '0043_utc_timestamps'
      AND u.table_name = 'live_marks'
      AND u.column_name = 'marked_at'
      AND json_extract_string(u.row_key, '$[0]') = live_marks.source
      AND json_extract_string(u.row_key, '$[1]') = live_marks.source_session_id
      AND json_extract_string(u.row_key, '$[2]') = live_marks.host;
    COMMIT;
"""

from __future__ import annotations

import duckdb

from recall.core.time import utcnow_naive
from recall.db.lease import DbLease, TransactionCapability
from recall.db.migrations import (
    AlreadyAudited,
    Migration,
    record_migration_audit,
    register,
    table_exists,
)
from recall.db.repair import require_schema_fence

MIGRATION_ID = "0043_utc_timestamps"
REQUIRED_VERSION = 42
TARGET_VERSION = 43

_UTC_DEFAULT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("schema_version", "applied_at"),
    ("schema_migrations", "applied_at"),
    ("schema_migration_undo", "recorded_at"),
    ("session_state", "indexed_at"),
    ("embedding_cache", "created_at"),
    ("fts_sidecar_pending", "queued_at"),
)


def _column_exists(conn: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ? AND column_name = ?
        """,
        [table, column],
    ).fetchone()
    return bool(row and row[0])


def _ensure_undo_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migration_undo (
            migration_id TEXT NOT NULL,
            table_name TEXT NOT NULL,
            row_key TEXT NOT NULL,
            column_name TEXT NOT NULL,
            old_value TEXT,
            recorded_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
            PRIMARY KEY (migration_id, table_name, row_key, column_name)
        )
        """
    )


def _already_audited(conn: duckdb.DuckDBPyConnection) -> bool:
    if not table_exists(conn, "schema_migrations"):
        return False
    row = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE migration_id = ?",
        [MIGRATION_ID],
    ).fetchone()
    return row is not None


def _convert_operational_tables(transaction: TransactionCapability) -> None:
    recorded_at = utcnow_naive()

    if transaction.call("utc:has_usage_events", lambda conn: table_exists(conn, "usage_events")):
        needs_usage_events = transaction.call(
            "utc:needs_usage_events",
            lambda conn: bool(
                (
                    conn.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM usage_events
                            WHERE (
                                ts IS NOT NULL
                                AND (ts::TIMESTAMPTZ AT TIME ZONE 'UTC') IS DISTINCT FROM ts
                            ) OR (
                                harvested_at IS NOT NULL
                                AND (harvested_at::TIMESTAMPTZ AT TIME ZONE 'UTC')
                                    IS DISTINCT FROM harvested_at
                            )
                        )
                        """
                    ).fetchone()
                    or (False,)
                )[0]
            ),
        )
        if needs_usage_events:
            transaction.call(
                "utc:undo_usage_events_ts",
                lambda conn: conn.execute(
                    """
                    INSERT OR IGNORE INTO schema_migration_undo
                        (migration_id, table_name, row_key, column_name, old_value, recorded_at)
                    SELECT ?, 'usage_events', id, 'ts', CAST(ts AS VARCHAR), ?
                    FROM usage_events
                    WHERE ts IS NOT NULL
                      AND (ts::TIMESTAMPTZ AT TIME ZONE 'UTC') IS DISTINCT FROM ts
                    """,
                    [MIGRATION_ID, recorded_at],
                ).fetchone(),
            )
            transaction.call(
                "utc:undo_usage_events_harvested_at",
                lambda conn: conn.execute(
                    """
                    INSERT OR IGNORE INTO schema_migration_undo
                        (migration_id, table_name, row_key, column_name, old_value, recorded_at)
                    SELECT ?, 'usage_events', id, 'harvested_at', CAST(harvested_at AS VARCHAR), ?
                    FROM usage_events
                    WHERE harvested_at IS NOT NULL
                      AND (harvested_at::TIMESTAMPTZ AT TIME ZONE 'UTC')
                          IS DISTINCT FROM harvested_at
                    """,
                    [MIGRATION_ID, recorded_at],
                ).fetchone(),
            )
            # Rebuild usage_events instead of bulk-UPDATEing an ART-indexed column (`ts`)
            # alongside its PRIMARY KEY: DuckDB 1.5.5 aborts commit with a FATAL internal
            # duplicate-key error on multi-row-group UPDATEs against PRIMARY_usage_events_0.
            transaction.call(
                "utc:rebuild_usage_events",
                lambda conn: conn.execute(
                    """
                    DROP TABLE IF EXISTS usage_events_utc;
                    CREATE TABLE usage_events_utc (
                        id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        source_session_id TEXT NOT NULL,
                        session_id TEXT,
                        ts TIMESTAMP,
                        prompt_tokens BIGINT,
                        cached_prompt_tokens BIGINT,
                        completion_tokens BIGINT,
                        reasoning_tokens BIGINT,
                        host TEXT,
                        harvested_at TIMESTAMP NOT NULL
                    );
                    INSERT INTO usage_events_utc (
                        id, source, source_session_id, session_id, ts,
                        prompt_tokens, cached_prompt_tokens, completion_tokens,
                        reasoning_tokens, host, harvested_at
                    )
                    SELECT
                        id, source, source_session_id, session_id,
                        CASE
                            WHEN ts IS NULL THEN NULL
                            ELSE (ts::TIMESTAMPTZ AT TIME ZONE 'UTC')
                        END,
                        prompt_tokens, cached_prompt_tokens, completion_tokens,
                        reasoning_tokens, host,
                        (harvested_at::TIMESTAMPTZ AT TIME ZONE 'UTC')
                    FROM usage_events;
                    DROP TABLE usage_events;
                    ALTER TABLE usage_events_utc RENAME TO usage_events;
                    CREATE INDEX idx_usage_events_source_sid
                        ON usage_events(source, source_session_id);
                    CREATE INDEX idx_usage_events_session ON usage_events(session_id);
                    CREATE INDEX idx_usage_events_ts ON usage_events(ts);
                    """
                ).fetchone(),
            )

    if transaction.call(
        "utc:has_usage_log_cursors", lambda conn: table_exists(conn, "usage_log_cursors")
    ):
        transaction.call(
            "utc:undo_usage_log_cursors",
            lambda conn: conn.execute(
                """
                INSERT OR IGNORE INTO schema_migration_undo
                    (migration_id, table_name, row_key, column_name, old_value, recorded_at)
                SELECT ?, 'usage_log_cursors', path, 'updated_at', CAST(updated_at AS VARCHAR), ?
                FROM usage_log_cursors
                WHERE updated_at IS NOT NULL
                  AND (updated_at::TIMESTAMPTZ AT TIME ZONE 'UTC') IS DISTINCT FROM updated_at
                """,
                [MIGRATION_ID, recorded_at],
            ).fetchone(),
        )
        transaction.call(
            "utc:update_usage_log_cursors",
            lambda conn: conn.execute(
                """
                UPDATE usage_log_cursors
                SET updated_at = (updated_at::TIMESTAMPTZ AT TIME ZONE 'UTC')
                WHERE updated_at IS NOT NULL
                  AND (updated_at::TIMESTAMPTZ AT TIME ZONE 'UTC') IS DISTINCT FROM updated_at
                """
            ).fetchone(),
        )

    if transaction.call("utc:has_live_marks", lambda conn: table_exists(conn, "live_marks")):
        transaction.call(
            "utc:undo_live_marks",
            lambda conn: conn.execute(
                """
                INSERT OR IGNORE INTO schema_migration_undo
                    (migration_id, table_name, row_key, column_name, old_value, recorded_at)
                SELECT
                    ?,
                    'live_marks',
                    to_json([source, source_session_id, host])::VARCHAR,
                    'marked_at',
                    CAST(marked_at AS VARCHAR),
                    ?
                FROM live_marks
                WHERE marked_at IS NOT NULL
                  AND (marked_at::TIMESTAMPTZ AT TIME ZONE 'UTC') IS DISTINCT FROM marked_at
                """,
                [MIGRATION_ID, recorded_at],
            ).fetchone(),
        )
        transaction.call(
            "utc:update_live_marks",
            lambda conn: conn.execute(
                """
                UPDATE live_marks
                SET marked_at = (marked_at::TIMESTAMPTZ AT TIME ZONE 'UTC')
                WHERE marked_at IS NOT NULL
                  AND (marked_at::TIMESTAMPTZ AT TIME ZONE 'UTC') IS DISTINCT FROM marked_at
                """
            ).fetchone(),
        )

    # Convert transient pending keyword intents so pre-upgrade local timestamps
    # cannot sort after post-upgrade UTC intents on UTC+N hosts (queued_at has
    # no ART index, so in-place UPDATE touches zero secondary indexes).
    if transaction.call(
        "utc:has_fts_sidecar_pending",
        lambda conn: table_exists(conn, "fts_sidecar_pending"),
    ):
        transaction.call(
            "utc:update_fts_sidecar_pending",
            lambda conn: conn.execute(
                """
                UPDATE fts_sidecar_pending
                SET queued_at = (queued_at::TIMESTAMPTZ AT TIME ZONE 'UTC')
                WHERE queued_at IS NOT NULL
                  AND (queued_at::TIMESTAMPTZ AT TIME ZONE 'UTC') IS DISTINCT FROM queued_at
                """
            ).fetchone(),
        )


def _upgrade(lease: DbLease) -> bool | AlreadyAudited:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = lease.call("0043_utc_timestamps:_upgrade", lambda conn: get_schema_version(conn))
    if current < REQUIRED_VERSION:
        return False
    try:
        lease.call("0043_utc_timestamps:fence", require_schema_fence)
    except (ValueError, duckdb.CatalogException):
        return False

    with lease.transaction("migration:0043") as transaction:
        tx_version = transaction.call("utc:tx_version", get_schema_version)
        already_applied = tx_version >= TARGET_VERSION or transaction.call(
            "utc:already_audited", _already_audited
        )
        transaction.call("utc:undo_table", _ensure_undo_table)
        for table, column in _UTC_DEFAULT_COLUMNS:
            if transaction.call(
                f"utc:col_check:{table}.{column}",
                lambda conn, t=table, c=column: _column_exists(conn, t, c),
            ):
                transaction.call(
                    f"utc:default:{table}.{column}",
                    lambda conn, t=table, c=column: conn.execute(
                        f"ALTER TABLE {t} ALTER COLUMN {c} "
                        "SET DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')"
                    ).fetchone(),
                )
        if not already_applied:
            _convert_operational_tables(transaction)
        if tx_version < TARGET_VERSION:
            transaction.call(
                "utc:version", lambda conn: set_schema_version_to(conn, TARGET_VERSION)
            )
        transaction.call(
            "utc:audit",
            lambda conn: record_migration_audit(conn, MIGRATION_ID),
        )
    return AlreadyAudited(MIGRATION_ID, TARGET_VERSION)


register(Migration(id=MIGRATION_ID, target_version=TARGET_VERSION, upgrade=_upgrade))
