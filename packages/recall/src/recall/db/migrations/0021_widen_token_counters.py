"""Migration 0021: widen token counters from INTEGER to BIGINT.

Token counters are cumulative for some sources and can exceed the signed
32-bit range even though an individual turn remains valid. The migration is
lossless and keeps the existing explicit indexes around DuckDB's dependency
restriction on ALTER COLUMN TYPE.
"""

from __future__ import annotations

import logging

from recall.db.lease import DbLease
from recall.db.migrations import Migration, register, table_exists

logger = logging.getLogger("recall.schema")

_TARGET_COLUMNS: dict[str, tuple[str, ...]] = {
    "session_state": ("input_tokens", "output_tokens", "cached_input_tokens"),
    "usage_events": (
        "prompt_tokens",
        "cached_prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
    ),
    "runtime_state": ("last_context_input_tokens", "last_context_output_tokens"),
}


def _upgrade(lease: DbLease) -> bool:
    from recall.db.schema import get_schema_version, set_schema_version_to

    current = lease.call(
        "0021_widen_token_counters:_upgrade", lambda conn: get_schema_version(conn)
    )
    if current < 20:
        logger.warning(
            "0021_widen_token_counters requires schema_version >= 20; DB is at %d. "
            "Halting; earlier migrations must complete first.",
            current,
        )
        return False

    pending: dict[str, tuple[str, ...]] = {}
    for table, columns in _TARGET_COLUMNS.items():
        if not lease.call(
            "0021_widen_token_counters:_upgrade",
            lambda conn, table=table: table_exists(conn, table),
        ):
            logger.info("0021_widen_token_counters: %s is absent; skipping", table)
            continue
        table_columns = _column_types(lease, table)
        missing = [column for column in columns if column not in table_columns]
        if missing:
            logger.error(
                "0021_widen_token_counters: %s is missing required columns: %s",
                table,
                ", ".join(missing),
            )
            return False
        needs_widening = tuple(column for column in columns if table_columns[column] != "BIGINT")
        if needs_widening:
            pending[table] = needs_widening

    if not pending:
        lease.call(
            "0021_widen_token_counters:_upgrade", lambda conn: set_schema_version_to(conn, 21)
        )
        return True

    indexes = _table_indexes(lease, pending)
    dropped_indexes: dict[str, str] = {}
    try:
        # DuckDB only refreshes ALTER COLUMN's index-dependency catalog after
        # the DROP INDEX statements commit. Keep the committed drop phase
        # compensatable, then make the type changes, index recreation, and
        # schema-version write one transaction.
        for index_name, sql in indexes.items():
            lease.call(
                "0021_widen_token_counters:_upgrade",
                lambda conn, index_name=index_name: conn.execute(
                    f"DROP INDEX {_quote_identifier(index_name)}"
                ).fetchone(),
            )
            dropped_indexes[index_name] = sql
    except Exception:
        _restore_indexes(lease, dropped_indexes)
        logger.exception("Failed to apply 0021_widen_token_counters")
        return False

    lease.call(
        "0021_widen_token_counters:_upgrade",
        lambda conn: conn.execute("BEGIN TRANSACTION").fetchone(),
    )
    try:
        for table, columns in pending.items():
            for column in columns:
                lease.call(
                    "0021_widen_token_counters:_upgrade",
                    lambda conn, column=column, table=table: conn.execute(
                        f"ALTER TABLE {_quote_identifier(table)} "
                        f"ALTER COLUMN {_quote_identifier(column)} TYPE BIGINT"
                    ).fetchone(),
                )
        for sql in indexes.values():
            lease.call(
                "0021_widen_token_counters:_upgrade",
                lambda conn, sql=sql: conn.execute(sql).fetchone(),
            )
        lease.call(
            "0021_widen_token_counters:_upgrade", lambda conn: set_schema_version_to(conn, 21)
        )
        lease.call(
            "0021_widen_token_counters:_upgrade", lambda conn: conn.execute("COMMIT").fetchone()
        )
    except Exception:
        lease.cleanup_call(
            "0021_widen_token_counters:_upgrade", lambda conn: conn.execute("ROLLBACK").fetchone()
        )
        _restore_indexes(lease, indexes)
        logger.exception("Failed to apply 0021_widen_token_counters")
        return False
    return True


def _column_types(lease: DbLease, table: str) -> dict[str, str]:
    return {
        str(row[1]): str(row[2]).upper()
        for row in lease.call(
            "0021_widen_token_counters:_column_types",
            lambda conn: conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall(),
        )
    }


def _table_indexes(
    lease: DbLease,
    tables: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    indexes: dict[str, str] = {}
    for table in tables:
        rows = lease.call(
            "0021_widen_token_counters:_table_indexes",
            lambda conn, table=table: conn.execute(
                """
            SELECT index_name, sql
            FROM duckdb_indexes()
            WHERE schema_name = 'main' AND table_name = ?
            """,
                [table],
            ).fetchall(),
        )
        for index_name, sql in rows:
            indexes[str(index_name)] = str(sql)
    return indexes


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _restore_indexes(lease: DbLease, indexes: dict[str, str]) -> None:
    for sql in indexes.values():
        lease.cleanup_call(
            "0021_widen_token_counters:_restore_indexes",
            lambda conn, sql=sql: conn.execute(sql).fetchone(),
        )


register(
    Migration(
        id="0021_widen_token_counters",
        target_version=21,
        upgrade=_upgrade,
    )
)
