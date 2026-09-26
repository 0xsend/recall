from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import duckdb

from recall.db.fts_sidecar import (
    delete_message_fts,
    delete_tool_call_fts,
    scope_message_fts_columns,
    should_index_bash_fts,
    upsert_message_fts,
    upsert_message_fts_batch,
    upsert_tool_call_fts,
    upsert_tool_call_fts_batch,
)

logger = logging.getLogger("recall.fts_sidecar_reconcile")

_MESSAGE_KIND = "message"
_TOOL_CALL_KIND = "tool_call"
_KINDS = (_MESSAGE_KIND, _TOOL_CALL_KIND)

_UPSERT_OP = "upsert"
_DELETE_OP = "delete"
_DEFAULT_RECONCILE_BATCH_SIZE = 10_000
_MAX_BATCH_SIZE = 10_000

_ReconcileKind = Literal["message", "tool_call"]


@dataclass(frozen=True)
class ReconcileStats:
    pending_drained: dict[str, int]
    orphans_backfilled: dict[str, int]
    ghosts_deleted: dict[str, int]
    pending_remaining: dict[str, int]


@dataclass(frozen=True)
class RescopeStats:
    messages_rewritten: int
    tool_calls_rewritten: int


@dataclass(frozen=True)
class _PendingRow:
    rowid: int
    kind: str
    entity_id: str
    op: str
    queued_at: datetime


@dataclass(frozen=True)
class PendingRepair:
    cursor: tuple[datetime, int] | None
    attempted: int
    applied: int
    remaining: int
    error: str | None


def repair_pending_batch(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    cursor: tuple[datetime, int] | None = None,
    fts_fields: tuple[str, ...] | None = None,
    limit: int = 256,
) -> PendingRepair:
    """Repair one bounded page, advancing past failures and wrapping for retry.

    The caller excludes concurrent writers through sidecar publication and
    acknowledgement. A failed publication retains its durable pending row.
    """
    if not 1 <= limit <= 256:
        raise ValueError("pending repair limit must be between 1 and 256")
    rows = _fetch_pending_rows_after(
        duckdb_conn, cursor[0] if cursor else None, cursor[1] if cursor else -1, limit
    )
    applied: list[int] = []
    error = None
    for row in rows:
        try:
            _apply_pending_row(duckdb_conn, sidecar_conn, row, fts_fields=fts_fields)
        except (sqlite3.Error, ValueError) as err:
            error = f"{row.kind} {row.op} {row.entity_id}: {err}"
            logger.warning("pending keyword repair failed: %s", error)
        else:
            applied.append(row.rowid)
    _delete_pending_rowids(duckdb_conn, applied, batch_size=limit)
    count = duckdb_conn.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone()
    assert count is not None
    next_cursor = (rows[-1].queued_at, rows[-1].rowid) if len(rows) == limit else None
    return PendingRepair(next_cursor, len(rows), len(applied), int(count[0]), error)


def reconcile_sidecar(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    fts_fields: tuple[str, ...] | None = None,
    batch_size: int = _DEFAULT_RECONCILE_BATCH_SIZE,
) -> ReconcileStats:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    pending_drained = _zero_counts()
    orphans_backfilled = _zero_counts()
    ghosts_deleted = _zero_counts()

    _drain_pending(
        duckdb_conn,
        sidecar_conn,
        pending_drained,
        fts_fields=fts_fields,
        batch_size=batch_size,
    )
    message_orphans, message_ghosts = _reconcile_message_differences(
        duckdb_conn,
        sidecar_conn,
        fts_fields=fts_fields,
        batch_size=batch_size,
    )
    tool_call_orphans, tool_call_ghosts = _reconcile_tool_call_differences(
        duckdb_conn,
        sidecar_conn,
        fts_fields=fts_fields,
        batch_size=batch_size,
    )
    orphans_backfilled[_MESSAGE_KIND] = message_orphans
    orphans_backfilled[_TOOL_CALL_KIND] = tool_call_orphans
    ghosts_deleted[_MESSAGE_KIND] = message_ghosts
    ghosts_deleted[_TOOL_CALL_KIND] = tool_call_ghosts

    return ReconcileStats(
        pending_drained=pending_drained,
        orphans_backfilled=orphans_backfilled,
        ghosts_deleted=ghosts_deleted,
        pending_remaining=_pending_counts(duckdb_conn),
    )


def rescope_sidecar_for_fields(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    fts_fields: tuple[str, ...] | None,
    batch_size: int = 10_000,
) -> RescopeStats:
    """Rewrite sidecar FTS rows so existing indexed content matches `fts_fields`.

    Field changes are rare, but the source database can be large. The pass walks
    DuckDB in key order and commits at most one configured batch at a time.
    """
    if batch_size <= 0 or batch_size > _MAX_BATCH_SIZE:
        raise ValueError("batch_size must be between 1 and 10_000")

    messages_rewritten = _rescope_message_rows(
        duckdb_conn,
        sidecar_conn,
        fts_fields=fts_fields,
        batch_size=batch_size,
    )
    tool_calls_rewritten = _rescope_tool_call_rows(
        duckdb_conn,
        sidecar_conn,
        fts_fields=fts_fields,
        batch_size=batch_size,
    )
    return RescopeStats(
        messages_rewritten=messages_rewritten,
        tool_calls_rewritten=tool_calls_rewritten,
    )


def _rescope_message_rows(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    fts_fields: tuple[str, ...] | None,
    batch_size: int,
) -> int:
    processed = 0
    last_id = ""

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
        ).fetchall()
        if not batch:
            return processed

        with sidecar_conn:
            for message_id, fts_content, fts_thinking in batch:
                rowid = _ensure_message_rowid(sidecar_conn, str(message_id))
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

        processed += len(batch)
        last_id = str(batch[-1][0])


def _rescope_tool_call_rows(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    fts_fields: tuple[str, ...] | None,
    batch_size: int,
) -> int:
    if not should_index_bash_fts(fts_fields):
        return _delete_all_tool_call_rows(sidecar_conn, batch_size=batch_size)

    processed = 0
    last_id = ""

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
        ).fetchall()
        if not batch:
            return processed

        with sidecar_conn:
            for tool_call_id, bash_command in batch:
                rowid = _ensure_tool_call_rowid(sidecar_conn, str(tool_call_id))
                sidecar_conn.execute(
                    """
                    INSERT OR REPLACE INTO tool_calls_fts(rowid, bash_command)
                    VALUES (?, ?)
                    """,
                    [rowid, str(bash_command)],
                )

        processed += len(batch)
        last_id = str(batch[-1][0])


def _delete_all_tool_call_rows(
    sidecar_conn: sqlite3.Connection,
    *,
    batch_size: int,
) -> int:
    deleted = 0
    last_rowid = -1

    while True:
        batch = sidecar_conn.execute(
            """
            SELECT rowid
            FROM tool_calls_fts_rowid
            WHERE rowid > ?
            ORDER BY rowid
            LIMIT ?
            """,
            [last_rowid, batch_size],
        ).fetchall()
        if not batch:
            return deleted

        rowids = [int(row[0]) for row in batch]
        placeholders = ", ".join("?" for _ in rowids)
        with sidecar_conn:
            sidecar_conn.execute(
                f"DELETE FROM tool_calls_fts WHERE rowid IN ({placeholders})",
                rowids,
            )
            sidecar_conn.execute(
                f"DELETE FROM tool_calls_fts_rowid WHERE rowid IN ({placeholders})",
                rowids,
            )
        deleted += len(rowids)
        last_rowid = rowids[-1]


def _ensure_message_rowid(conn: sqlite3.Connection, message_id: str) -> int:
    conn.execute(
        "INSERT OR IGNORE INTO message_fts_rowid(message_id) VALUES (?)",
        [message_id],
    )
    row = conn.execute(
        "SELECT rowid FROM message_fts_rowid WHERE message_id = ?",
        [message_id],
    ).fetchone()
    if row is None:
        raise sqlite3.IntegrityError(f"failed to resolve sidecar rowid for message {message_id}")
    return int(row[0])


def _ensure_tool_call_rowid(conn: sqlite3.Connection, tool_call_id: str) -> int:
    conn.execute(
        "INSERT OR IGNORE INTO tool_calls_fts_rowid(tool_call_id) VALUES (?)",
        [tool_call_id],
    )
    row = conn.execute(
        "SELECT rowid FROM tool_calls_fts_rowid WHERE tool_call_id = ?",
        [tool_call_id],
    ).fetchone()
    if row is None:
        raise sqlite3.IntegrityError(
            f"failed to resolve sidecar rowid for tool call {tool_call_id}"
        )
    return int(row[0])


def _zero_counts() -> dict[str, int]:
    return {_MESSAGE_KIND: 0, _TOOL_CALL_KIND: 0}


def _drain_pending(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    pending_drained: dict[str, int],
    *,
    fts_fields: tuple[str, ...] | None,
    batch_size: int,
) -> None:
    last_queued_at: datetime | None = None
    last_rowid = -1

    while True:
        rows = _fetch_pending_rows_after(
            duckdb_conn,
            last_queued_at,
            last_rowid,
            batch_size,
        )
        if not rows:
            return

        drained_rowids: list[int] = []
        for row in rows:
            try:
                _apply_pending_row(duckdb_conn, sidecar_conn, row, fts_fields=fts_fields)
            except sqlite3.Error as err:
                logger.warning(
                    "sidecar reconcile failed for pending %s %s %s: %s",
                    row.kind,
                    row.op,
                    row.entity_id,
                    err,
                )
                continue
            except ValueError as err:
                logger.warning(
                    "sidecar reconcile found invalid pending row %s %s %s: %s",
                    row.kind,
                    row.op,
                    row.entity_id,
                    err,
                )
                continue

            drained_rowids.append(row.rowid)
            if row.kind in pending_drained:
                pending_drained[row.kind] += 1

        _delete_pending_rowids(duckdb_conn, drained_rowids, batch_size=batch_size)
        last_queued_at = rows[-1].queued_at
        last_rowid = rows[-1].rowid
        if len(rows) < batch_size:
            return


def _fetch_pending_rows_after(
    duckdb_conn: duckdb.DuckDBPyConnection,
    last_queued_at: datetime | None,
    last_rowid: int,
    batch_size: int,
) -> list[_PendingRow]:
    if last_queued_at is None:
        rows = duckdb_conn.execute(
            """
            SELECT rowid, kind, id, op, queued_at
            FROM fts_sidecar_pending
            ORDER BY queued_at, rowid
            LIMIT ?
            """,
            [batch_size],
        ).fetchall()
    else:
        rows = duckdb_conn.execute(
            """
            SELECT rowid, kind, id, op, queued_at
            FROM fts_sidecar_pending
            WHERE (queued_at, rowid) > (?, ?)
            ORDER BY queued_at, rowid
            LIMIT ?
            """,
            [last_queued_at, last_rowid, batch_size],
        ).fetchall()

    return [
        _PendingRow(
            rowid=int(row[0]),
            kind=str(row[1]),
            entity_id=str(row[2]),
            op=str(row[3]),
            queued_at=row[4],
        )
        for row in rows
    ]


def _apply_pending_row(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    row: _PendingRow,
    *,
    fts_fields: tuple[str, ...] | None,
) -> None:
    if row.kind == _MESSAGE_KIND and row.op == _UPSERT_OP:
        _apply_message_upsert(duckdb_conn, sidecar_conn, row.entity_id, fts_fields=fts_fields)
        return
    if row.kind == _MESSAGE_KIND and row.op == _DELETE_OP:
        delete_message_fts(sidecar_conn, [row.entity_id])
        return
    if row.kind == _TOOL_CALL_KIND and row.op == _UPSERT_OP:
        _apply_tool_call_upsert(
            duckdb_conn,
            sidecar_conn,
            row.entity_id,
            fts_fields=fts_fields,
        )
        return
    if row.kind == _TOOL_CALL_KIND and row.op == _DELETE_OP:
        delete_tool_call_fts(sidecar_conn, [row.entity_id])
        return
    raise ValueError("unsupported kind/op")


def _apply_message_upsert(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    message_id: str,
    *,
    fts_fields: tuple[str, ...] | None,
) -> None:
    row = duckdb_conn.execute(
        """
        SELECT COALESCE(fts_content, ''), COALESCE(fts_thinking, '')
        FROM message_state
        WHERE message_id = ?
        """,
        [message_id],
    ).fetchone()
    if row is None:
        delete_message_fts(sidecar_conn, [message_id])
        return
    upsert_message_fts(sidecar_conn, message_id, str(row[0]), str(row[1]), fields=fts_fields)


def _apply_tool_call_upsert(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    tool_call_id: str,
    *,
    fts_fields: tuple[str, ...] | None,
) -> None:
    row = duckdb_conn.execute(
        """
        SELECT bash_command
        FROM tool_calls
        WHERE id = ?
        """,
        [tool_call_id],
    ).fetchone()
    if row is None or row[0] is None:
        delete_tool_call_fts(sidecar_conn, [tool_call_id])
        return
    upsert_tool_call_fts(sidecar_conn, tool_call_id, str(row[0]), fields=fts_fields)


def _delete_pending_rowids(
    duckdb_conn: duckdb.DuckDBPyConnection,
    rowids: list[int],
    *,
    batch_size: int,
) -> None:
    for batch in _chunks(rowids, batch_size):
        placeholders = ", ".join("?" for _ in batch)
        duckdb_conn.execute(
            f"DELETE FROM fts_sidecar_pending WHERE rowid IN ({placeholders})",
            batch,
        )


def _memberships_match(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    primary_sql: str,
    sidecar_sql: str,
    *,
    batch_size: int,
) -> bool:
    """Compare complete ordered identities without repeated primary-table scans."""
    batch_size = min(batch_size, _MAX_BATCH_SIZE)
    # Keep the caller's transaction visible. No other primary query may run
    # until this comparison ends, because execute replaces its active result.
    primary = duckdb_conn.execute(primary_sql)
    sidecar = sidecar_conn.execute(sidecar_sql)
    try:
        while batch := primary.fetchmany(batch_size):
            if batch != sidecar.fetchmany(batch_size):
                return False
        return not sidecar.fetchmany(1)
    finally:
        sidecar.close()


def _reconcile_message_differences(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    fts_fields: tuple[str, ...] | None,
    batch_size: int,
) -> tuple[int, int]:
    if _memberships_match(
        duckdb_conn,
        sidecar_conn,
        "SELECT message_id FROM message_state ORDER BY message_id",
        "SELECT message_id FROM message_fts_rowid ORDER BY message_id",
        batch_size=batch_size,
    ):
        return 0, 0
    duck_batch: list[str] = []
    sidecar_batch: list[str] = []
    duck_index = 0
    sidecar_index = 0
    duck_last_id = ""
    sidecar_last_id = ""
    duck_exhausted = False
    sidecar_exhausted = False
    has_pending_upserts = _has_pending_upserts(duckdb_conn, _MESSAGE_KIND)
    orphan_ids: list[str] = []
    ghost_ids: list[str] = []
    orphans_backfilled = 0
    ghosts_deleted = 0

    while True:
        if duck_index >= len(duck_batch) and not duck_exhausted:
            duck_batch = _fetch_message_ids_after(duckdb_conn, duck_last_id, batch_size)
            duck_index = 0
            if duck_batch:
                duck_last_id = duck_batch[-1]
            else:
                duck_exhausted = True

        if sidecar_index >= len(sidecar_batch) and not sidecar_exhausted:
            sidecar_batch = _fetch_sidecar_ids_after(
                sidecar_conn,
                "message_fts_rowid",
                "message_id",
                sidecar_last_id,
                batch_size,
            )
            sidecar_index = 0
            if sidecar_batch:
                sidecar_last_id = sidecar_batch[-1]
            else:
                sidecar_exhausted = True

        duck_id = None if duck_exhausted else duck_batch[duck_index]
        sidecar_id = None if sidecar_exhausted else sidecar_batch[sidecar_index]

        if duck_id is None and sidecar_id is None:
            break
        if sidecar_id is None or (duck_id is not None and duck_id < sidecar_id):
            assert duck_id is not None
            if not has_pending_upserts or not _has_pending_upsert(
                duckdb_conn,
                _MESSAGE_KIND,
                duck_id,
            ):
                orphan_ids.append(duck_id)
                if len(orphan_ids) >= batch_size:
                    orphans_backfilled += _upsert_missing_messages(
                        duckdb_conn,
                        sidecar_conn,
                        orphan_ids,
                        fts_fields=fts_fields,
                        batch_size=batch_size,
                    )
                    orphan_ids = []
            duck_index += 1
            continue
        if duck_id is None or sidecar_id < duck_id:
            ghost_ids.append(sidecar_id)
            if len(ghost_ids) >= batch_size:
                delete_message_fts(sidecar_conn, ghost_ids)
                ghosts_deleted += len(ghost_ids)
                ghost_ids = []
            sidecar_index += 1
            continue

        duck_index += 1
        sidecar_index += 1

    if orphan_ids:
        orphans_backfilled += _upsert_missing_messages(
            duckdb_conn,
            sidecar_conn,
            orphan_ids,
            fts_fields=fts_fields,
            batch_size=batch_size,
        )
    if ghost_ids:
        delete_message_fts(sidecar_conn, ghost_ids)
        ghosts_deleted += len(ghost_ids)

    return orphans_backfilled, ghosts_deleted


def _reconcile_tool_call_differences(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    fts_fields: tuple[str, ...] | None,
    batch_size: int,
) -> tuple[int, int]:
    primary_sql = (
        "SELECT id FROM tool_calls WHERE bash_command IS NOT NULL ORDER BY id"
        if should_index_bash_fts(fts_fields)
        else "SELECT id FROM tool_calls WHERE FALSE"
    )
    if _memberships_match(
        duckdb_conn,
        sidecar_conn,
        primary_sql,
        "SELECT tool_call_id FROM tool_calls_fts_rowid ORDER BY tool_call_id",
        batch_size=batch_size,
    ):
        return 0, 0
    duck_batch: list[str] = []
    sidecar_batch: list[str] = []
    duck_index = 0
    sidecar_index = 0
    duck_last_id = ""
    sidecar_last_id = ""
    duck_exhausted = False
    sidecar_exhausted = False
    has_pending_upserts = _has_pending_upserts(duckdb_conn, _TOOL_CALL_KIND)
    orphan_ids: list[str] = []
    ghost_ids: list[str] = []
    orphans_backfilled = 0
    ghosts_deleted = 0

    while True:
        if duck_index >= len(duck_batch) and not duck_exhausted:
            duck_batch = _fetch_tool_call_ids_after(
                duckdb_conn,
                duck_last_id,
                batch_size,
                include_bash=should_index_bash_fts(fts_fields),
            )
            duck_index = 0
            if duck_batch:
                duck_last_id = duck_batch[-1]
            else:
                duck_exhausted = True

        if sidecar_index >= len(sidecar_batch) and not sidecar_exhausted:
            sidecar_batch = _fetch_sidecar_ids_after(
                sidecar_conn,
                "tool_calls_fts_rowid",
                "tool_call_id",
                sidecar_last_id,
                batch_size,
            )
            sidecar_index = 0
            if sidecar_batch:
                sidecar_last_id = sidecar_batch[-1]
            else:
                sidecar_exhausted = True

        duck_id = None if duck_exhausted else duck_batch[duck_index]
        sidecar_id = None if sidecar_exhausted else sidecar_batch[sidecar_index]

        if duck_id is None and sidecar_id is None:
            break
        if sidecar_id is None or (duck_id is not None and duck_id < sidecar_id):
            assert duck_id is not None
            if not has_pending_upserts or not _has_pending_upsert(
                duckdb_conn,
                _TOOL_CALL_KIND,
                duck_id,
            ):
                orphan_ids.append(duck_id)
                if len(orphan_ids) >= batch_size:
                    orphans_backfilled += _upsert_missing_tool_calls(
                        duckdb_conn,
                        sidecar_conn,
                        orphan_ids,
                        fts_fields=fts_fields,
                        batch_size=batch_size,
                    )
                    orphan_ids = []
            duck_index += 1
            continue
        if duck_id is None or sidecar_id < duck_id:
            ghost_ids.append(sidecar_id)
            if len(ghost_ids) >= batch_size:
                delete_tool_call_fts(sidecar_conn, ghost_ids)
                ghosts_deleted += len(ghost_ids)
                ghost_ids = []
            sidecar_index += 1
            continue

        duck_index += 1
        sidecar_index += 1

    if orphan_ids:
        orphans_backfilled += _upsert_missing_tool_calls(
            duckdb_conn,
            sidecar_conn,
            orphan_ids,
            fts_fields=fts_fields,
            batch_size=batch_size,
        )
    if ghost_ids:
        delete_tool_call_fts(sidecar_conn, ghost_ids)
        ghosts_deleted += len(ghost_ids)

    return orphans_backfilled, ghosts_deleted


def _fetch_message_ids_after(
    duckdb_conn: duckdb.DuckDBPyConnection,
    last_id: str,
    batch_size: int,
) -> list[str]:
    return [
        str(message_id)
        for (message_id,) in duckdb_conn.execute(
            """
            SELECT message_id
            FROM message_state
            WHERE message_id > ?
            ORDER BY message_id
            LIMIT ?
            """,
            [last_id, batch_size],
        ).fetchall()
    ]


def _fetch_tool_call_ids_after(
    duckdb_conn: duckdb.DuckDBPyConnection,
    last_id: str,
    batch_size: int,
    *,
    include_bash: bool,
) -> list[str]:
    if not include_bash:
        return []
    return [
        str(tool_call_id)
        for (tool_call_id,) in duckdb_conn.execute(
            """
            SELECT id
            FROM tool_calls
            WHERE bash_command IS NOT NULL
              AND id > ?
            ORDER BY id
            LIMIT ?
            """,
            [last_id, batch_size],
        ).fetchall()
    ]


def _upsert_missing_messages(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    message_ids: list[str],
    *,
    fts_fields: tuple[str, ...] | None,
    batch_size: int,
) -> int:
    rows = _fetch_message_payloads(duckdb_conn, message_ids)
    if not rows:
        return 0
    upsert_message_fts_batch(
        sidecar_conn,
        rows,
        fields=fts_fields,
        batch_size=batch_size,
    )
    return len(rows)


def _upsert_missing_tool_calls(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    tool_call_ids: list[str],
    *,
    fts_fields: tuple[str, ...] | None,
    batch_size: int,
) -> int:
    rows = _fetch_tool_call_payloads(duckdb_conn, tool_call_ids)
    if not rows:
        return 0
    upsert_tool_call_fts_batch(
        sidecar_conn,
        rows,
        fields=fts_fields,
        batch_size=batch_size,
    )
    return len(rows)


def _fetch_message_payloads(
    duckdb_conn: duckdb.DuckDBPyConnection,
    message_ids: list[str],
) -> list[tuple[str, str, str]]:
    if not message_ids:
        return []
    placeholders = ", ".join("?" for _ in message_ids)
    fetched = {
        str(message_id): (str(message_id), str(fts_content), str(fts_thinking))
        for message_id, fts_content, fts_thinking in duckdb_conn.execute(
            f"""
            SELECT message_id, COALESCE(fts_content, ''), COALESCE(fts_thinking, '')
            FROM message_state
            WHERE message_id IN ({placeholders})
            """,
            message_ids,
        ).fetchall()
    }
    return [fetched[message_id] for message_id in message_ids if message_id in fetched]


def _fetch_tool_call_payloads(
    duckdb_conn: duckdb.DuckDBPyConnection,
    tool_call_ids: list[str],
) -> list[tuple[str, str | None]]:
    if not tool_call_ids:
        return []
    placeholders = ", ".join("?" for _ in tool_call_ids)
    fetched = {
        str(tool_call_id): (str(tool_call_id), None if bash_command is None else str(bash_command))
        for tool_call_id, bash_command in duckdb_conn.execute(
            f"""
            SELECT id, bash_command
            FROM tool_calls
            WHERE id IN ({placeholders})
              AND bash_command IS NOT NULL
            """,
            tool_call_ids,
        ).fetchall()
    }
    return [fetched[tool_call_id] for tool_call_id in tool_call_ids if tool_call_id in fetched]


def _fetch_sidecar_ids_after(
    sidecar_conn: sqlite3.Connection,
    mapping_table: str,
    id_column: str,
    last_id: str,
    batch_size: int,
) -> list[str]:
    return [
        str(row[0])
        for row in sidecar_conn.execute(
            f"""
            SELECT {id_column}
            FROM {mapping_table}
            WHERE {id_column} > ?
            ORDER BY {id_column}
            LIMIT ?
            """,
            [last_id, batch_size],
        ).fetchall()
    ]


def _has_pending_upserts(
    duckdb_conn: duckdb.DuckDBPyConnection,
    kind: _ReconcileKind,
) -> bool:
    return (
        duckdb_conn.execute(
            """
            SELECT 1
            FROM fts_sidecar_pending
            WHERE kind = ?
              AND op = 'upsert'
            LIMIT 1
            """,
            [kind],
        ).fetchone()
        is not None
    )


def _has_pending_upsert(
    duckdb_conn: duckdb.DuckDBPyConnection,
    kind: _ReconcileKind,
    entity_id: str,
) -> bool:
    return (
        duckdb_conn.execute(
            """
            SELECT 1
            FROM fts_sidecar_pending
            WHERE kind = ?
              AND id = ?
              AND op = 'upsert'
            LIMIT 1
            """,
            [kind, entity_id],
        ).fetchone()
        is not None
    )


def _pending_counts(duckdb_conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    counts = _zero_counts()
    for kind, count in duckdb_conn.execute(
        """
        SELECT kind, COUNT(*)
        FROM fts_sidecar_pending
        WHERE kind IN ('message', 'tool_call')
        GROUP BY kind
        """
    ).fetchall():
        counts[str(kind)] = int(count)
    return counts


def _chunks[T](items: list[T], chunk_size: int) -> list[list[T]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]
