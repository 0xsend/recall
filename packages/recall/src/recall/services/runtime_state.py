from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import duckdb

from recall.core.config import AppConfig
from recall.core.time import to_naive_utc, utcnow_naive
from recall.core.types import RunKind, SchedulerKind


@dataclass(frozen=True)
class IndexRunCounts:
    total: int
    indexed: int
    skipped: int
    failed: int
    changed: int = 0
    total_seconds: float | None = None


@dataclass(frozen=True)
class RuntimeStatus:
    last_attempted_at: datetime | None
    last_successful_at: datetime | None
    last_run_kind: RunKind | None
    last_index_summary: IndexRunCounts | None
    last_failure_message: str | None
    last_failure_at: datetime | None
    installed_scheduler: SchedulerKind | None
    # Failure-signature memory (REQ-RESIL-014..019). Defaulted so a database that
    # predates migration 0023 still loads.
    last_fatal_signature: str | None = None
    fatal_repeat_count: int = 0
    last_fatal_at: datetime | None = None
    last_index_repair_at: datetime | None = None
    last_index_repair_signature: str | None = None
    needs_index_verification: bool = False

    @classmethod
    def unread(cls) -> RuntimeStatus:
        """The status of a runtime_state that was not (or could not be) read."""
        return cls(
            last_attempted_at=None,
            last_successful_at=None,
            last_run_kind=None,
            last_index_summary=None,
            last_failure_message=None,
            last_failure_at=None,
            installed_scheduler=None,
        )


def load_runtime_status(
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> RuntimeStatus:
    config = config or AppConfig.load()
    owned_conn = conn is None
    if conn is None:
        from recall.db import connect_readonly

        conn = connect_readonly(config)
    try:
        return load_runtime_status_from_conn(conn)
    finally:
        if owned_conn:
            conn.close()


def load_runtime_status_from_conn(conn: duckdb.DuckDBPyConnection) -> RuntimeStatus:
    try:
        row = conn.execute(
            """
            SELECT
                last_attempted_at,
                last_successful_at,
                last_run_kind,
                last_index_total,
                last_index_indexed,
                last_index_skipped,
                last_index_failed,
                last_index_changed,
                last_index_total_seconds,
                last_failure_message,
                last_failure_at,
                installed_scheduler
            FROM runtime_state
            WHERE singleton = TRUE
            """
        ).fetchone()
    except duckdb.Error:
        row = None
    if row is None:
        return RuntimeStatus.unread()

    legacy_run_kind = False
    try:
        last_run_kind = RunKind(row[2]) if row[2] is not None else None
    except ValueError:
        last_run_kind = None
        legacy_run_kind = True
    index_summary = None
    if not legacy_run_kind and any(row[idx] is not None for idx in range(3, 9)):
        index_summary = IndexRunCounts(
            total=int(row[3] or 0),
            indexed=int(row[4] or 0),
            skipped=int(row[5] or 0),
            failed=int(row[6] or 0),
            changed=int(row[7] or 0),
            total_seconds=float(row[8]) if row[8] is not None else None,
        )
    installed_scheduler = SchedulerKind(row[11]) if row[11] is not None else None
    fatal = _load_fatal_memory_columns(conn)

    return RuntimeStatus(
        last_attempted_at=row[0],
        last_successful_at=row[1],
        last_run_kind=last_run_kind,
        last_index_summary=index_summary,
        last_failure_message=row[9],
        last_failure_at=row[10],
        installed_scheduler=installed_scheduler,
        last_fatal_signature=fatal.last_fatal_signature,
        fatal_repeat_count=fatal.fatal_repeat_count,
        last_fatal_at=fatal.last_fatal_at,
        last_index_repair_at=fatal.last_index_repair_at,
        last_index_repair_signature=fatal.last_index_repair_signature,
        needs_index_verification=fatal.needs_index_verification,
    )


@dataclass(frozen=True)
class _FatalMemory:
    """The 0023/0029 runtime_state columns; defaults are the pre-0023 shape."""

    last_fatal_signature: str | None = None
    fatal_repeat_count: int = 0
    last_fatal_at: datetime | None = None
    last_index_repair_at: datetime | None = None
    last_index_repair_signature: str | None = None
    needs_index_verification: bool = False


_FATAL_MEMORY_COLUMNS = (
    "last_fatal_signature, fatal_repeat_count, last_index_repair_at, "
    "last_index_repair_signature, needs_index_verification"
)


def _load_fatal_memory_columns(conn: duckdb.DuckDBPyConnection) -> _FatalMemory:
    """Read the fatal-memory columns in one round-trip, degrading by generation.

    A read-only open cannot migrate, so a database one generation behind must
    still yield the columns it does have: the 0029 read falls back to the 0023
    shape, and that to the pre-0023 defaults. Status is loaded on every
    `daemon status`, so the columns that are present are read together.
    """
    row = _select_fatal_memory(conn, f"{_FATAL_MEMORY_COLUMNS}, last_fatal_at")
    last_fatal_at: datetime | None = row[5] if row is not None else None
    if row is None:
        row = _select_fatal_memory(conn, _FATAL_MEMORY_COLUMNS)
    if row is None:
        return _FatalMemory()
    return _FatalMemory(
        last_fatal_signature=row[0],
        fatal_repeat_count=int(row[1] or 0),
        last_fatal_at=last_fatal_at,
        last_index_repair_at=row[2],
        last_index_repair_signature=row[3],
        needs_index_verification=bool(row[4]),
    )


def _select_fatal_memory(conn: duckdb.DuckDBPyConnection, columns: str) -> tuple[Any, ...] | None:
    """Read the singleton row, or None when the database lacks one of `columns`."""
    try:
        return conn.execute(
            f"SELECT {columns} FROM runtime_state WHERE singleton = TRUE"
        ).fetchone()
    except duckdb.Error:
        return None


def record_run_attempt(
    conn: duckdb.DuckDBPyConnection,
    *,
    run_kind: RunKind,
    attempted_at: datetime | None = None,
) -> datetime:
    timestamp = to_naive_utc(attempted_at) if attempted_at is not None else utcnow_naive()
    _ensure_runtime_state_row(conn)
    conn.execute(
        """
        UPDATE runtime_state
        SET last_attempted_at = ?, last_run_kind = ?
        WHERE singleton = TRUE
        """,
        [timestamp, run_kind.value],
    )
    return timestamp


def record_run_success(
    conn: duckdb.DuckDBPyConnection,
    *,
    run_kind: RunKind,
    index_summary: IndexRunCounts | None = None,
    last_context_messages: int = 0,
    last_context_mode: str = "off",
    last_context_input_tokens: int = 0,
    last_context_output_tokens: int = 0,
    last_context_model: str | None = None,
    attempted_at: datetime | None = None,
    successful_at: datetime | None = None,
) -> datetime:
    attempt_timestamp = to_naive_utc(attempted_at) if attempted_at is not None else utcnow_naive()
    success_timestamp = to_naive_utc(successful_at) if successful_at is not None else utcnow_naive()
    _ensure_runtime_state_row(conn)
    conn.execute(
        """
        UPDATE runtime_state
        SET
            last_attempted_at = ?,
            last_successful_at = ?,
            last_run_kind = ?,
            last_context_messages = ?,
            last_context_mode = ?,
            last_context_input_tokens = ?,
            last_context_output_tokens = ?,
            last_context_model = ?,
            last_failure_message = NULL,
            last_failure_at = NULL
        WHERE singleton = TRUE
        """,
        [
            attempt_timestamp,
            success_timestamp,
            run_kind.value,
            last_context_messages,
            last_context_mode,
            last_context_input_tokens,
            last_context_output_tokens,
            last_context_model,
        ],
    )
    if index_summary is not None:
        conn.execute(
            """
            UPDATE runtime_state
            SET
                last_index_total = ?,
                last_index_indexed = ?,
                last_index_skipped = ?,
                last_index_failed = ?,
                last_index_changed = ?,
                last_index_total_seconds = ?
            WHERE singleton = TRUE
            """,
            [
                index_summary.total,
                index_summary.indexed,
                index_summary.skipped,
                index_summary.failed,
                index_summary.changed,
                index_summary.total_seconds,
            ],
        )
    return success_timestamp


def record_run_failure(
    conn: duckdb.DuckDBPyConnection,
    *,
    run_kind: RunKind,
    message: str,
    attempted_at: datetime | None = None,
    failed_at: datetime | None = None,
) -> datetime:
    attempt_timestamp = to_naive_utc(attempted_at) if attempted_at is not None else utcnow_naive()
    failure_timestamp = to_naive_utc(failed_at) if failed_at is not None else utcnow_naive()
    # Some exceptions stringify empty (a dataclass exception has no `args`), and a
    # dated failure with no reason is unreadable — say which run died instead
    # (REQ-RESIL-026).
    reason = message.strip() or f"unknown failure during the {run_kind.value} run"
    _ensure_runtime_state_row(conn)
    conn.execute(
        """
        UPDATE runtime_state
        SET
            last_attempted_at = ?,
            last_run_kind = ?,
            last_failure_message = ?,
            last_failure_at = ?
        WHERE singleton = TRUE
        """,
        [attempt_timestamp, run_kind.value, reason, failure_timestamp],
    )
    return failure_timestamp


def set_installed_scheduler(
    conn: duckdb.DuckDBPyConnection, scheduler: SchedulerKind | None
) -> None:
    _ensure_runtime_state_row(conn)
    conn.execute(
        """
        UPDATE runtime_state
        SET installed_scheduler = ?
        WHERE singleton = TRUE
        """,
        [scheduler.value if scheduler is not None else None],
    )


def record_fatal_failure(
    conn: duckdb.DuckDBPyConnection,
    *,
    message: str,
    signature: str,
    failed_at: datetime,
    repeat_count: int,
) -> None:
    """Fold a spooled fatal record into runtime_state (REQ-RESIL-014).

    Written by the *next* daemon start, not by the run that died: the
    invalidated instance rejects every statement.
    """
    assert repeat_count >= 1, repeat_count
    _ensure_runtime_state_row(conn)
    failure_timestamp = to_naive_utc(failed_at)
    conn.execute(
        """
        UPDATE runtime_state
        SET
            last_failure_message = ?,
            last_failure_at = ?,
            last_fatal_signature = ?,
            fatal_repeat_count = ?,
            last_fatal_at = ?
        WHERE singleton = TRUE
        """,
        [message, failure_timestamp, signature, repeat_count, failure_timestamp],
    )


def record_index_repair(
    conn: duckdb.DuckDBPyConnection,
    *,
    signature: str | None,
    repaired_at: datetime,
) -> None:
    """Remember that the indexes were rebuilt in response to `signature` (REQ-RESIL-015)."""
    _ensure_runtime_state_row(conn)
    conn.execute(
        """
        UPDATE runtime_state
        SET last_index_repair_at = ?, last_index_repair_signature = ?
        WHERE singleton = TRUE
        """,
        [to_naive_utc(repaired_at), signature],
    )


def set_needs_index_verification(conn: duckdb.DuckDBPyConnection, flag: bool) -> None:
    """Mark (or clear) the ENOSPC-driven index verification flag (REQ-RESIL-019)."""
    _ensure_runtime_state_row(conn)
    conn.execute(
        "UPDATE runtime_state SET needs_index_verification = ? WHERE singleton = TRUE",
        [flag],
    )


def clear_fatal_memory_columns(conn: duckdb.DuckDBPyConnection) -> None:
    """Forget the fatal signature, repair record, and refusal message (REQ-RESIL-016).

    Called after a manual index rebuild or a successful compaction, both of which
    rebuild every index, so the next start must not refuse on stale memory.
    """
    _ensure_runtime_state_row(conn)
    conn.execute(
        """
        UPDATE runtime_state
        SET
            last_failure_message = NULL,
            last_failure_at = NULL,
            last_fatal_signature = NULL,
            fatal_repeat_count = 0,
            last_fatal_at = NULL,
            last_index_repair_at = NULL,
            last_index_repair_signature = NULL,
            needs_index_verification = FALSE
        WHERE singleton = TRUE
        """
    )


def _ensure_runtime_state_row(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("INSERT OR IGNORE INTO runtime_state (singleton) VALUES (TRUE)")
