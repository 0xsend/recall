from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import duckdb

from recall.db.fts_sidecar import scope_message_fts_columns, should_index_bash_fts

_MESSAGE_KIND = "message"
_TOOL_CALL_KIND = "tool_call"
_MESSAGE_START_ID = ""
_TOOL_CALL_START_ID = ""
_MAX_BATCH_SIZE = 10_000


@dataclass(frozen=True)
class BootstrapProgress:
    messages_processed: int
    tool_calls_processed: int
    messages_done: bool
    tool_calls_done: bool


@dataclass(frozen=True)
class _ProgressRow:
    last_id: str | None
    done: bool


def bootstrap_sidecar(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    fts_fields: tuple[str, ...] | None = None,
    batch_size: int = 10_000,
    progress_callback: Callable[[BootstrapProgress], None] | None = None,
) -> BootstrapProgress:
    if batch_size <= 0 or batch_size > _MAX_BATCH_SIZE:
        raise ValueError("batch_size must be between 1 and 10_000")

    progress_rows = _read_progress(sidecar_conn)
    messages_done = progress_rows[_MESSAGE_KIND].done
    tool_calls_done = progress_rows[_TOOL_CALL_KIND].done
    messages_processed = 0
    tool_calls_processed = 0

    if messages_done and tool_calls_done:
        return BootstrapProgress(
            messages_processed=0,
            tool_calls_processed=0,
            messages_done=True,
            tool_calls_done=True,
        )

    def increment_messages(count: int) -> None:
        nonlocal messages_processed
        messages_processed += count
        emit_progress()

    def increment_tool_calls(count: int) -> None:
        nonlocal tool_calls_processed
        tool_calls_processed += count
        emit_progress()

    def emit_progress() -> None:
        if progress_callback is not None:
            progress_callback(
                BootstrapProgress(
                    messages_processed=messages_processed,
                    tool_calls_processed=tool_calls_processed,
                    messages_done=messages_done,
                    tool_calls_done=tool_calls_done,
                )
            )

    if not messages_done:
        _, messages_done = _bootstrap_messages(
            duckdb_conn,
            sidecar_conn,
            start_after=progress_rows[_MESSAGE_KIND].last_id,
            fts_fields=fts_fields,
            batch_size=batch_size,
            after_batch=increment_messages,
        )

    if not tool_calls_done:
        _, tool_calls_done = _bootstrap_tool_calls(
            duckdb_conn,
            sidecar_conn,
            start_after=progress_rows[_TOOL_CALL_KIND].last_id,
            fts_fields=fts_fields,
            batch_size=batch_size,
            after_batch=increment_tool_calls,
        )

    return BootstrapProgress(
        messages_processed=messages_processed,
        tool_calls_processed=tool_calls_processed,
        messages_done=messages_done,
        tool_calls_done=tool_calls_done,
    )


def _read_progress(conn: sqlite3.Connection) -> dict[str, _ProgressRow]:
    rows = {
        str(row[0]): _ProgressRow(
            last_id=str(row[1]) if row[1] is not None else None,
            done=row[2] is not None,
        )
        for row in conn.execute(
            """
            SELECT kind, last_id, completed_at
            FROM bootstrap_progress
            WHERE kind IN (?, ?)
            """,
            [_MESSAGE_KIND, _TOOL_CALL_KIND],
        )
    }
    return {
        _MESSAGE_KIND: rows.get(_MESSAGE_KIND, _ProgressRow(last_id=None, done=False)),
        _TOOL_CALL_KIND: rows.get(_TOOL_CALL_KIND, _ProgressRow(last_id=None, done=False)),
    }


def _bootstrap_messages(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    start_after: str | None,
    fts_fields: Iterable[str] | None,
    batch_size: int,
    after_batch: Callable[[int], None],
) -> tuple[int, bool]:
    processed = 0
    last_id = start_after or _MESSAGE_START_ID

    while True:
        batch = duckdb_conn.execute(
            """
            SELECT message_id, COALESCE(fts_content, ''), COALESCE(fts_thinking, '')
            FROM message_state
            WHERE message_id > ?
            ORDER BY message_id
            LIMIT ?
            """,
            [last_id, batch_size],
        ).fetchmany(batch_size)
        if not batch:
            _mark_completed(sidecar_conn, _MESSAGE_KIND, start_after)
            return processed, True

        batch_last_id = str(batch[-1][0])
        with sidecar_conn:
            for message_id, fts_content, fts_thinking in batch:
                rowid = _insert_message_mapping(sidecar_conn, str(message_id))
                if rowid is None:
                    continue
                scoped_content, scoped_thinking = scope_message_fts_columns(
                    str(fts_content),
                    str(fts_thinking),
                    fts_fields,
                )
                sidecar_conn.execute(
                    """
                    INSERT OR REPLACE INTO message_fts(rowid, fts_content, fts_thinking)
                    VALUES (?, ?, ?)
                    """,
                    [rowid, scoped_content, scoped_thinking],
                )
            _record_batch_progress(sidecar_conn, _MESSAGE_KIND, batch_last_id)

        processed += len(batch)
        last_id = batch_last_id
        after_batch(len(batch))


def _bootstrap_tool_calls(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    start_after: str | None,
    fts_fields: Iterable[str] | None,
    batch_size: int,
    after_batch: Callable[[int], None],
) -> tuple[int, bool]:
    processed = 0
    last_id = start_after or _TOOL_CALL_START_ID
    if not should_index_bash_fts(fts_fields):
        _mark_completed(sidecar_conn, _TOOL_CALL_KIND, start_after)
        return processed, True

    while True:
        batch = duckdb_conn.execute(
            """
            SELECT id, bash_command
            FROM tool_calls
            WHERE bash_command IS NOT NULL
              AND id > ?
            ORDER BY id
            LIMIT ?
            """,
            [last_id, batch_size],
        ).fetchmany(batch_size)
        if not batch:
            _mark_completed(sidecar_conn, _TOOL_CALL_KIND, start_after)
            return processed, True

        batch_last_id = str(batch[-1][0])
        with sidecar_conn:
            for tool_call_id, bash_command in batch:
                rowid = _insert_tool_call_mapping(sidecar_conn, str(tool_call_id))
                if rowid is None:
                    continue
                sidecar_conn.execute(
                    """
                    INSERT OR REPLACE INTO tool_calls_fts(rowid, bash_command)
                    VALUES (?, ?)
                    """,
                    [rowid, bash_command],
                )
            _record_batch_progress(sidecar_conn, _TOOL_CALL_KIND, batch_last_id)

        processed += len(batch)
        last_id = batch_last_id
        after_batch(len(batch))


def _insert_message_mapping(conn: sqlite3.Connection, message_id: str) -> int | None:
    cursor = conn.execute(
        "INSERT OR IGNORE INTO message_fts_rowid(message_id) VALUES (?)",
        [message_id],
    )
    if cursor.rowcount == 0:
        return None
    row = conn.execute("SELECT last_insert_rowid()").fetchone()
    if row is None:
        raise RuntimeError(f"failed to assign sidecar rowid for message {message_id}")
    return int(row[0])


def _insert_tool_call_mapping(conn: sqlite3.Connection, tool_call_id: str) -> int | None:
    cursor = conn.execute(
        "INSERT OR IGNORE INTO tool_calls_fts_rowid(tool_call_id) VALUES (?)",
        [tool_call_id],
    )
    if cursor.rowcount == 0:
        return None
    row = conn.execute("SELECT last_insert_rowid()").fetchone()
    if row is None:
        raise RuntimeError(f"failed to assign sidecar rowid for tool call {tool_call_id}")
    return int(row[0])


def _record_batch_progress(conn: sqlite3.Connection, kind: str, last_id: str) -> None:
    conn.execute(
        """
        INSERT INTO bootstrap_progress(kind, last_id, completed_at)
        VALUES (?, ?, NULL)
        ON CONFLICT(kind) DO UPDATE SET
            last_id = excluded.last_id,
            completed_at = NULL
        """,
        [kind, last_id],
    )


def _mark_completed(conn: sqlite3.Connection, kind: str, last_id: str | None) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO bootstrap_progress(kind, last_id, completed_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(kind) DO UPDATE SET
                last_id = COALESCE(bootstrap_progress.last_id, excluded.last_id),
                completed_at = CURRENT_TIMESTAMP
            """,
            [kind, last_id],
        )
