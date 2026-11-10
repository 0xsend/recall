from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import duckdb

from recall.db.fts_sidecar import (
    FtsSidecarUnavailableError,
    KeywordIndexFailure,
    PublicationDeadline,
    _Publication,
    delete_message_fts,
    delete_tool_call_fts,
    is_lock_contention,
    publish_sidecar,
    should_index_bash_fts,
    upsert_message_fts_batch,
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
    message_columns: tuple[str, str] | None
    bash_command: str | None


@dataclass(frozen=True)
class PendingRepair:
    cursor: tuple[datetime, int] | None
    attempted: int
    applied: int
    remaining: int
    # The last row whose publication failed while the rest of its page committed.
    error: str | None
    # The page could not be published at all (lock, deadline, open, unavailable
    # sidecar): nothing was retired and `cursor` is the caller's own, so the
    # retry gate alone schedules the next attempt.
    deferral: KeywordIndexFailure | None = None


def repair_pending_batch(
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    cursor: tuple[datetime, int] | None = None,
    fts_fields: tuple[str, ...] | None = None,
    limit: int = 256,
    deadline: PublicationDeadline | None = None,
) -> PendingRepair:
    """Repair one bounded page, advancing past failed rows and wrapping for retry.

    The caller excludes concurrent writers through sidecar publication and
    acknowledgement. A failed publication retains its durable pending row. A
    failure of the whole page is a deferral: nothing is retired and the cursor
    does not move. `deadline` bounds the page's sidecar lock wait.
    """
    if not 1 <= limit <= 256:
        raise ValueError("pending repair limit must be between 1 and 256")
    rows = _fetch_pending_rows_after(
        duckdb_conn, cursor[0] if cursor else None, cursor[1] if cursor else -1, limit
    )
    applied: list[int] = []
    error = None
    try:
        outcomes = _publish_pending_page(
            sidecar_conn, rows, fts_fields=fts_fields, deadline=deadline
        )
    except (sqlite3.Error, FtsSidecarUnavailableError) as err:
        deferral = KeywordIndexFailure.of(f"publication of {len(rows)} pending rows deferred", err)
        logger.warning("pending keyword repair deferred: %s", deferral)
        return PendingRepair(cursor, len(rows), 0, pending_count(duckdb_conn), None, deferral)
    for row, failure in outcomes:
        if failure is None:
            applied.append(row.rowid)
        else:
            error = f"{row.kind} {row.op} {row.entity_id}: {failure}"
            logger.warning("pending keyword repair failed: %s", error)
    _delete_pending_rowids(duckdb_conn, applied, batch_size=limit)
    next_cursor = (rows[-1].queued_at, rows[-1].rowid) if len(rows) == limit else None
    return PendingRepair(next_cursor, len(rows), len(applied), pending_count(duckdb_conn), error)


def pending_count(duckdb_conn: duckdb.DuckDBPyConnection) -> int:
    count = duckdb_conn.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone()
    assert count is not None
    return int(count[0])


def has_pending(duckdb_conn: duckdb.DuckDBPyConnection) -> bool:
    row = duckdb_conn.execute("SELECT EXISTS(SELECT 1 FROM fts_sidecar_pending)").fetchone()
    assert row is not None
    return bool(row[0])


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

        processed += upsert_message_fts_batch(
            sidecar_conn,
            [(str(key), str(content), str(thinking)) for key, content, thinking in batch],
            fields=fts_fields,
            batch_size=batch_size,
        )

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

        processed += upsert_tool_call_fts_batch(
            sidecar_conn,
            [(str(key), str(command)) for key, command in batch],
            fields=fts_fields,
            batch_size=batch_size,
        )

        last_id = str(batch[-1][0])


def _delete_all_tool_call_rows(
    sidecar_conn: sqlite3.Connection,
    *,
    batch_size: int,
) -> int:
    deleted = 0
    last_id = ""

    while True:
        batch = sidecar_conn.execute(
            """
            SELECT tool_call_id
            FROM tool_calls_fts_rowid
            WHERE tool_call_id > ?
            ORDER BY tool_call_id
            LIMIT ?
            """,
            [last_id, batch_size],
        ).fetchall()
        if not batch:
            return deleted

        ids = [str(row[0]) for row in batch]
        delete_tool_call_fts(sidecar_conn, ids)
        deleted += len(ids)
        last_id = ids[-1]


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
        try:
            outcomes = _publish_pending_page(sidecar_conn, rows, fts_fields=fts_fields)
        except sqlite3.Error as err:
            logger.warning(
                "sidecar reconcile failed to publish %d pending rows: %s", len(rows), err
            )
            outcomes = []
        for row, failure in outcomes:
            if failure is not None:
                logger.warning(
                    "sidecar reconcile failed for pending %s %s %s: %s",
                    row.kind,
                    row.op,
                    row.entity_id,
                    failure,
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
        predicate = ""
        parameters = [batch_size]
    else:
        predicate = "WHERE (queued_at, rowid) > (?, ?)"
        parameters = [last_queued_at, last_rowid, batch_size]
    rows = duckdb_conn.execute(
        f"""
        SELECT rowid, kind, id, op, queued_at
        FROM fts_sidecar_pending
        {predicate}
        ORDER BY queued_at, rowid
        LIMIT ?
        """,
        parameters,
    ).fetchall()
    message_ids = [str(row[2]) for row in rows if row[1] == _MESSAGE_KIND and row[3] == _UPSERT_OP]
    tool_call_ids = [
        str(row[2]) for row in rows if row[1] == _TOOL_CALL_KIND and row[3] == _UPSERT_OP
    ]
    # Scalar IN parameters retain index scans for scattered IDs on DuckDB 1.5.5;
    # binding lists or relations instead scans the populated primary tables.
    message_placeholders = ", ".join("?" for _ in message_ids) or "NULL"
    tool_call_placeholders = ", ".join("?" for _ in tool_call_ids) or "NULL"
    payloads = (
        duckdb_conn.execute(
            f"""
            SELECT 'message', message_id, COALESCE(fts_content, ''), COALESCE(fts_thinking, '')
            FROM message_state
            WHERE message_id IN ({message_placeholders})
            UNION ALL
            SELECT 'tool_call', id, bash_command, NULL
            FROM tool_calls
            WHERE id IN ({tool_call_placeholders})
            """,
            message_ids + tool_call_ids,
        ).fetchall()
        if message_ids or tool_call_ids
        else []
    )
    messages = {
        str(key): (str(content), str(thinking))
        for kind, key, content, thinking in payloads
        if kind == _MESSAGE_KIND
    }
    tools = {
        str(key): str(command) if command is not None else None
        for kind, key, command, _thinking in payloads
        if kind == _TOOL_CALL_KIND
    }

    return [
        _PendingRow(
            rowid=int(row[0]),
            kind=str(row[1]),
            entity_id=str(row[2]),
            op=str(row[3]),
            queued_at=row[4],
            message_columns=messages.get(str(row[2])) if row[1] == _MESSAGE_KIND else None,
            bash_command=tools.get(str(row[2])) if row[1] == _TOOL_CALL_KIND else None,
        )
        for row in rows
    ]


def _publish_pending_page(
    sidecar_conn: sqlite3.Connection,
    rows: list[_PendingRow],
    *,
    fts_fields: tuple[str, ...] | None,
    deadline: PublicationDeadline | None = None,
) -> list[tuple[_PendingRow, Exception | None]]:
    """Publish a pending page in one commit, rolling a failed row back alone.

    Each row runs inside its own savepoint, so a row that SQLite or validation
    rejects leaves the rest of the page intact and stays pending. An error that
    ends the whole transaction, a lock another connection holds, or a failed
    commit propagates: no row of the page is published, so the caller must
    retain every row.
    """
    if not rows:
        return []
    outcomes: list[tuple[_PendingRow, Exception | None]] = []
    with publish_sidecar(sidecar_conn, deadline=deadline):
        for row in rows:
            sidecar_conn.execute("SAVEPOINT pending_row")
            try:
                _apply_pending_row(sidecar_conn, row, fts_fields=fts_fields)
            except (sqlite3.Error, ValueError) as err:
                if not sidecar_conn.in_transaction or (
                    isinstance(err, sqlite3.Error) and is_lock_contention(err)
                ):
                    raise
                sidecar_conn.execute("ROLLBACK TO pending_row")
                outcomes.append((row, err))
            else:
                outcomes.append((row, None))
            sidecar_conn.execute("RELEASE pending_row")
    return outcomes


def _apply_pending_row(
    sidecar_conn: sqlite3.Connection,
    row: _PendingRow,
    *,
    fts_fields: tuple[str, ...] | None,
) -> None:
    if row.kind == _MESSAGE_KIND and row.op == _UPSERT_OP:
        if row.message_columns is None:
            _page_publication(sidecar_conn).delete([row.entity_id], "message")
        else:
            upsert_message_fts(sidecar_conn, row.entity_id, *row.message_columns, fields=fts_fields)
        return
    if row.kind == _MESSAGE_KIND and row.op == _DELETE_OP:
        _page_publication(sidecar_conn).delete([row.entity_id], "message")
        return
    if row.kind == _TOOL_CALL_KIND and row.op == _UPSERT_OP:
        if row.bash_command is None:
            _page_publication(sidecar_conn).delete([row.entity_id], "tool_call")
        else:
            upsert_tool_call_fts(sidecar_conn, row.entity_id, row.bash_command, fields=fts_fields)
        return
    if row.kind == _TOOL_CALL_KIND and row.op == _DELETE_OP:
        _page_publication(sidecar_conn).delete([row.entity_id], "tool_call")
        return
    raise ValueError("unsupported kind/op")


def upsert_message_fts(
    sidecar_conn: sqlite3.Connection,
    message_id: str,
    fts_content: str,
    fts_thinking: str,
    *,
    fields: tuple[str, ...] | None,
) -> None:
    """Write one repaired message into the pending page's open publication."""
    _page_publication(sidecar_conn).messages(
        [(message_id, fts_content, fts_thinking)], fields=fields
    )


def upsert_tool_call_fts(
    sidecar_conn: sqlite3.Connection,
    tool_call_id: str,
    bash_command: str,
    *,
    fields: tuple[str, ...] | None,
) -> None:
    """Write one repaired tool call into the pending page's open publication."""
    _page_publication(sidecar_conn).tools([(tool_call_id, bash_command)], fields=fields)


def _page_publication(sidecar_conn: sqlite3.Connection) -> _Publication:
    """Return a write handle on the publication `_publish_pending_page` holds open.

    The page's `publish_sidecar` owns the transaction, its validation, commit and
    rollback; the handle only writes rows inside it.
    """
    assert sidecar_conn.in_transaction, "pending rows publish only inside a page publication"
    return _Publication(sidecar_conn)


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
