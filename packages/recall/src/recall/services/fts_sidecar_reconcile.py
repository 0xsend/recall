from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import duckdb

from recall.core.failure_observation import FailureObserver
from recall.core.fts_text import sql_derive_fts
from recall.core.tool_format import (
    TOOL_INPUT_MAX_PARSE_BYTES,
    TOOL_NAME_TRIM_CHARS,
    derive_tool_call_fts_text,
)
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
from recall.db.lease import DbLease, StandaloneLifetime

logger = logging.getLogger("recall.fts_sidecar_reconcile")

_MESSAGE_KIND = "message"
_TOOL_CALL_KIND = "tool_call"
_KINDS = (_MESSAGE_KIND, _TOOL_CALL_KIND)

_UPSERT_OP = "upsert"
_DELETE_OP = "delete"
_DEFAULT_RECONCILE_BATCH_SIZE = 10_000
_MAX_BATCH_SIZE = 10_000
_TOOL_PAYLOAD_PAGE_MAX_BYTES = 16 * 1024 * 1024
_TOOL_ID_LOOKUP_CHUNK = 500
_ALL_RESCOPE_KINDS = frozenset({_MESSAGE_KIND, _TOOL_CALL_KIND})

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
class _PendingEntry:
    rowid: int
    kind: str
    entity_id: str
    op: str
    queued_at: datetime


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
    with StandaloneLifetime().lease(duckdb_conn) as lease:
        return repair_pending_batch_on(
            lease,
            sidecar_conn=sidecar_conn,
            cursor=cursor,
            fts_fields=fts_fields,
            limit=limit,
            deadline=deadline,
        )


def repair_pending_batch_on(
    lease: DbLease,
    sidecar_conn: sqlite3.Connection,
    *,
    cursor: tuple[datetime, int] | None = None,
    fts_fields: tuple[str, ...] | None = None,
    limit: int = 256,
    deadline: PublicationDeadline | None = None,
) -> PendingRepair:
    return lease.call(
        "repair_pending_batch",
        lambda conn: _repair_pending_batch(
            lease,
            duckdb_conn=conn,
            sidecar_conn=sidecar_conn,
            cursor=cursor,
            fts_fields=fts_fields,
            limit=limit,
            deadline=deadline,
        ),
    )


def _repair_pending_batch(
    lease: DbLease,
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
    entries = _fetch_pending_entries_after(
        duckdb_conn, cursor[0] if cursor else None, cursor[1] if cursor else -1, limit
    )
    page_entries = next(_partition_pending_entries(duckdb_conn, entries), [])
    rows = _hydrate_pending_entries(duckdb_conn, page_entries)
    applied: list[int] = []
    error = None
    try:
        outcomes = _publish_pending_page(
            lease.observer, sidecar_conn, rows, fts_fields=fts_fields, deadline=deadline
        )
    except (sqlite3.Error, FtsSidecarUnavailableError) as err:
        if lease.observe(err, "sidecar:pending_page"):
            raise
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
    has_more = len(page_entries) < len(entries) or len(entries) == limit
    next_cursor = (rows[-1].queued_at, rows[-1].rowid) if (rows and has_more) else None
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
    with StandaloneLifetime().lease(duckdb_conn) as lease:
        return reconcile_sidecar_on(
            lease, sidecar_conn=sidecar_conn, fts_fields=fts_fields, batch_size=batch_size
        )


def reconcile_sidecar_on(
    lease: DbLease,
    sidecar_conn: sqlite3.Connection,
    *,
    fts_fields: tuple[str, ...] | None = None,
    batch_size: int = _DEFAULT_RECONCILE_BATCH_SIZE,
) -> ReconcileStats:
    return lease.call(
        "reconcile_sidecar",
        lambda conn: _reconcile_sidecar(
            lease,
            duckdb_conn=conn,
            sidecar_conn=sidecar_conn,
            fts_fields=fts_fields,
            batch_size=batch_size,
        ),
    )


def _reconcile_sidecar(
    lease: DbLease,
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
        lease,
        duckdb_conn,
        sidecar_conn,
        pending_drained,
        fts_fields=fts_fields,
        batch_size=batch_size,
    )
    message_orphans, message_ghosts = _reconcile_message_differences(
        lease,
        duckdb_conn,
        sidecar_conn,
        fts_fields=fts_fields,
        batch_size=batch_size,
    )
    tool_call_orphans, tool_call_ghosts = _reconcile_tool_call_differences(
        lease,
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
    batch_size: int = 10000,
    kinds: frozenset[str] | None = None,
) -> RescopeStats:
    with StandaloneLifetime().lease(duckdb_conn) as lease:
        return rescope_sidecar_for_fields_on(
            lease,
            sidecar_conn=sidecar_conn,
            fts_fields=fts_fields,
            batch_size=batch_size,
            kinds=kinds,
        )


def rescope_sidecar_for_fields_on(
    lease: DbLease,
    sidecar_conn: sqlite3.Connection,
    *,
    fts_fields: tuple[str, ...] | None,
    batch_size: int = 10000,
    kinds: frozenset[str] | None = None,
) -> RescopeStats:
    return lease.call(
        "rescope_sidecar_for_fields",
        lambda conn: _rescope_sidecar_for_fields(
            lease,
            duckdb_conn=conn,
            sidecar_conn=sidecar_conn,
            fts_fields=fts_fields,
            batch_size=batch_size,
            kinds=kinds,
        ),
    )


def _rescope_sidecar_for_fields(
    lease: DbLease,
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    fts_fields: tuple[str, ...] | None,
    batch_size: int = 10000,
    kinds: frozenset[str] | None = None,
) -> RescopeStats:
    """Rewrite sidecar FTS rows so existing indexed content matches `fts_fields`.

    Field changes are rare, but the source database can be large. The pass walks
    DuckDB in key order and commits at most one configured batch at a time.
    """
    if batch_size <= 0 or batch_size > _MAX_BATCH_SIZE:
        raise ValueError("batch_size must be between 1 and 10_000")

    target_kinds = _ALL_RESCOPE_KINDS if kinds is None else kinds
    messages_rewritten = (
        _rescope_message_rows(
            duckdb_conn,
            sidecar_conn,
            fts_fields=fts_fields,
            batch_size=batch_size,
        )
        if _MESSAGE_KIND in target_kinds
        else 0
    )
    tool_calls_rewritten = (
        _rescope_tool_call_rows(
            duckdb_conn,
            sidecar_conn,
            fts_fields=fts_fields,
            batch_size=batch_size,
        )
        if _TOOL_CALL_KIND in target_kinds
        else 0
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

    fts_content_expr = sql_derive_fts("ms.context_text", "ms.content")
    fts_thinking_expr = sql_derive_fts("ms.context_text", "ms.thinking")

    while True:
        batch = duckdb_conn.execute(
            f"""
            WITH page AS MATERIALIZED (
                SELECT message_id
                FROM message_state
                WHERE message_id > ?
                ORDER BY message_id
                LIMIT ?
            )
            SELECT ms.message_id, {fts_content_expr}, {fts_thinking_expr}
            FROM page
            JOIN message_state ms ON ms.message_id = page.message_id
            ORDER BY ms.message_id
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


def _partition_by_byte_budget(
    id_sizes: Sequence[tuple[object, object]],
    max_bytes: int,
) -> Iterator[list[str]]:
    """Split `(id, payload_bytes)` into sub-chunks within `max_bytes` (always advancing >= 1).

    # WAIVER(bounded-chunk-memory): A single pre-existing bash_command larger than
    # _TOOL_PAYLOAD_PAGE_MAX_BYTES (16 MiB) is isolated as its own 1-row batch rather
    # than truncated or dropped, because SQLite FTS5 requires the full document in one
    # bind parameter and capping bash_command would alter historical content_fingerprint
    # hashes. Permanent — SQLite FTS5 has no chunked document insert and Scenario 3
    # requires preserving existing Bash fingerprints verbatim. Non-bash tool_input rows
    # are capped at TOOL_INPUT_MAX_PARSE_BYTES (4 MiB) in SQL so every non-bash chunk
    # and every multi-row chunk stays strictly <= max_bytes.
    """
    chunk: list[str] = []
    used = 0
    for raw_id, raw_size in id_sizes:
        item_id = str(raw_id)
        item_bytes = max(0, raw_size) if isinstance(raw_size, int) else 0
        if chunk and used + item_bytes > max_bytes:
            yield chunk
            chunk = [item_id]
            used = item_bytes
        else:
            chunk.append(item_id)
            used += item_bytes
    if chunk:
        yield chunk


# Longest special-dispatch tool name in `core.tool_format` (`invoke_subagent` is 15 chars);
# bounding the SQL projection of `tool_name` prevents pathological multi-MiB names from
# crossing DuckDB into Python while preserving exact formatter dispatch semantics.
_TOOL_NAME_DISPATCH_MAX_BYTES = 64


def _fetch_tool_call_fts_chunk(
    duckdb_conn: duckdb.DuckDBPyConnection,
    tool_call_ids: Sequence[str],
) -> dict[str, str]:
    if not tool_call_ids:
        return {}
    placeholders = ", ".join("?" for _ in tool_call_ids)
    rows = duckdb_conn.execute(
        f"""
        SELECT
            id,
            CASE
                WHEN bash_command IS NOT NULL THEN NULL
                WHEN COALESCE(strlen(trim(tool_name, ?)), 0) > ? THEN ''
                ELSE trim(tool_name, ?)
            END,
            CASE
                WHEN bash_command IS NOT NULL THEN NULL
                WHEN COALESCE(strlen(CAST(tool_input AS VARCHAR)), 0) > ? THEN NULL
                ELSE tool_input
            END,
            bash_command
        FROM tool_calls
        WHERE id IN ({placeholders})
        """,
        [
            TOOL_NAME_TRIM_CHARS,
            _TOOL_NAME_DISPATCH_MAX_BYTES,
            TOOL_NAME_TRIM_CHARS,
            TOOL_INPUT_MAX_PARSE_BYTES,
            *tool_call_ids,
        ],
    ).fetchall()
    return {
        str(tc_id): derive_tool_call_fts_text(
            None if tool_name is None else str(tool_name),
            None if tool_input is None else str(tool_input),
            None if bash_command is None else str(bash_command),
        )
        for tc_id, tool_name, tool_input, bash_command in rows
    }


def _measure_tool_call_payload_bytes(
    duckdb_conn: duckdb.DuckDBPyConnection,
    tool_call_ids: Sequence[str],
) -> dict[str, int]:
    unique_ids = list(dict.fromkeys(str(tc_id) for tc_id in tool_call_ids))
    if not unique_ids:
        return {}
    sizes: dict[str, int] = {}
    for lookup_chunk in _chunks(unique_ids, _TOOL_ID_LOOKUP_CHUNK):
        placeholders = ", ".join("?" for _ in lookup_chunk)
        for tc_id, payload_bytes in duckdb_conn.execute(
            f"""
            SELECT
                id,
                CASE
                    WHEN bash_command IS NOT NULL THEN COALESCE(strlen(bash_command), 0)
                    ELSE (
                        CASE
                            WHEN COALESCE(strlen(trim(tool_name, ?)), 0) > ? THEN 0
                            ELSE COALESCE(strlen(trim(tool_name, ?)), 0)
                        END
                        + CASE
                            WHEN COALESCE(strlen(CAST(tool_input AS VARCHAR)), 0) > ? THEN 0
                            ELSE COALESCE(strlen(CAST(tool_input AS VARCHAR)), 0)
                        END
                    )
                END AS payload_bytes
            FROM tool_calls
            WHERE id IN ({placeholders})
            """,
            [
                TOOL_NAME_TRIM_CHARS,
                _TOOL_NAME_DISPATCH_MAX_BYTES,
                TOOL_NAME_TRIM_CHARS,
                TOOL_INPUT_MAX_PARSE_BYTES,
                *lookup_chunk,
            ],
        ).fetchall():
            sizes[str(tc_id)] = max(0, int(payload_bytes or 0))
    return sizes


def _measure_message_payload_bytes(
    duckdb_conn: duckdb.DuckDBPyConnection,
    message_ids: Sequence[str],
) -> dict[str, int]:
    unique_ids = list(dict.fromkeys(str(message_id) for message_id in message_ids))
    if not unique_ids:
        return {}
    sizes: dict[str, int] = {}
    for lookup_chunk in _chunks(unique_ids, _TOOL_ID_LOOKUP_CHUNK):
        placeholders = ", ".join("?" for _ in lookup_chunk)
        for message_id, payload_bytes in duckdb_conn.execute(
            f"""
            SELECT
                message_id,
                COALESCE(strlen(context_text), 0)
                + COALESCE(strlen(content), 0)
                + COALESCE(strlen(thinking), 0) AS payload_bytes
            FROM message_state
            WHERE message_id IN ({placeholders})
            """,
            lookup_chunk,
        ).fetchall():
            sizes[str(message_id)] = max(0, int(payload_bytes or 0))
    return sizes


def iter_tool_call_fts_chunks_for_ids(
    duckdb_conn: duckdb.DuckDBPyConnection,
    tool_call_ids: Sequence[str],
    *,
    max_bytes: int | None = None,
) -> Iterator[list[tuple[str, str]]]:
    """Stream derived tool-call FTS text in byte-bounded chunks without retaining prior chunks."""
    limit_bytes = _TOOL_PAYLOAD_PAGE_MAX_BYTES if max_bytes is None else max_bytes
    unique_ids = list(dict.fromkeys(str(tc_id) for tc_id in tool_call_ids))
    if not unique_ids:
        return
    sizes = _measure_tool_call_payload_bytes(duckdb_conn, unique_ids)
    ordered_sizes = [(tc_id, sizes[tc_id]) for tc_id in unique_ids if tc_id in sizes]
    for sub_ids in _partition_by_byte_budget(ordered_sizes, limit_bytes):
        fetched = _fetch_tool_call_fts_chunk(duckdb_conn, sub_ids)
        chunk = [(tc_id, fetched[tc_id]) for tc_id in sub_ids if tc_id in fetched]
        del fetched
        if chunk:
            yield chunk
        del chunk


def iter_tool_call_inputs_bounded(
    duckdb_conn: duckdb.DuckDBPyConnection,
    tool_call_ids: Sequence[str],
    *,
    max_bytes: int | None = None,
) -> Iterator[dict[str, str]]:
    """Stream bounded non-bash `tool_input` JSON chunks so callers retain only bounded snippets."""
    limit_bytes = _TOOL_PAYLOAD_PAGE_MAX_BYTES if max_bytes is None else max_bytes
    unique_ids = list(dict.fromkeys(str(tc_id) for tc_id in tool_call_ids))
    if not unique_ids:
        return
    sized_rows: list[tuple[object, object]] = []
    for lookup_chunk in _chunks(unique_ids, _TOOL_ID_LOOKUP_CHUNK):
        placeholders = ", ".join("?" for _ in lookup_chunk)
        sized_rows.extend(
            duckdb_conn.execute(
                f"""
                SELECT id, COALESCE(strlen(CAST(tool_input AS VARCHAR)), 0) AS payload_bytes
                FROM tool_calls
                WHERE id IN ({placeholders})
                  AND bash_command IS NULL
                  AND tool_input IS NOT NULL
                  AND COALESCE(strlen(CAST(tool_input AS VARCHAR)), 0) <= ?
                """,
                [*lookup_chunk, TOOL_INPUT_MAX_PARSE_BYTES],
            ).fetchall()
        )
    for sub_ids in _partition_by_byte_budget(sized_rows, limit_bytes):
        placeholders = ", ".join("?" for _ in sub_ids)
        rows = duckdb_conn.execute(
            f"""
            SELECT id, tool_input
            FROM tool_calls
            WHERE id IN ({placeholders})
              AND bash_command IS NULL
              AND tool_input IS NOT NULL
              AND COALESCE(strlen(CAST(tool_input AS VARCHAR)), 0) <= ?
            """,
            [*sub_ids, TOOL_INPUT_MAX_PARSE_BYTES],
        ).fetchall()
        chunk = {
            str(tc_id): str(tool_input) for tc_id, tool_input in rows if tool_input is not None
        }
        del rows
        if chunk:
            yield chunk
        del chunk


def iter_tool_call_fts_batches(
    duckdb_conn: duckdb.DuckDBPyConnection,
    *,
    start_after: str = "",
    batch_size: int = _DEFAULT_RECONCILE_BATCH_SIZE,
) -> Iterator[list[tuple[str, str]]]:
    """Walk all `tool_calls` in key order without sorting wide JSON or exceeding byte bounds."""
    last_id = start_after
    while True:
        id_rows = duckdb_conn.execute(
            """
            SELECT
                id,
                CASE
                    WHEN bash_command IS NOT NULL THEN COALESCE(strlen(bash_command), 0)
                    ELSE (
                        CASE
                            WHEN COALESCE(strlen(trim(tool_name, ?)), 0) > ? THEN 0
                            ELSE COALESCE(strlen(trim(tool_name, ?)), 0)
                        END
                        + CASE
                            WHEN COALESCE(strlen(CAST(tool_input AS VARCHAR)), 0) > ? THEN 0
                            ELSE COALESCE(strlen(CAST(tool_input AS VARCHAR)), 0)
                        END
                    )
                END AS payload_bytes
            FROM tool_calls
            WHERE id > ?
            ORDER BY id
            LIMIT ?
            """,
            [
                TOOL_NAME_TRIM_CHARS,
                _TOOL_NAME_DISPATCH_MAX_BYTES,
                TOOL_NAME_TRIM_CHARS,
                TOOL_INPUT_MAX_PARSE_BYTES,
                last_id,
                batch_size,
            ],
        ).fetchall()
        if not id_rows:
            return
        for sub_ids in _partition_by_byte_budget(id_rows, _TOOL_PAYLOAD_PAGE_MAX_BYTES):
            fetched = _fetch_tool_call_fts_chunk(duckdb_conn, sub_ids)
            batch = [(tc_id, fetched[tc_id]) for tc_id in sub_ids if tc_id in fetched]
            del fetched
            if batch:
                yield batch
            del batch
        last_id = str(id_rows[-1][0])


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
    for batch in iter_tool_call_fts_batches(duckdb_conn, batch_size=batch_size):
        processed += upsert_tool_call_fts_batch(
            sidecar_conn,
            batch,
            fields=fts_fields,
            batch_size=batch_size,
        )
        batch.clear()
    return processed


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


def _drain_pending_sub_page(
    lease: DbLease,
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    sub_entries: Sequence[_PendingEntry],
    pending_drained: dict[str, int],
    *,
    fts_fields: tuple[str, ...] | None,
    batch_size: int,
) -> None:
    """Hydrate, publish, and retire one byte-bounded sub-page so its frame releases on return."""
    rows = _hydrate_pending_entries(duckdb_conn, sub_entries)
    drained_rowids: list[int] = []
    try:
        outcomes = _publish_pending_page(lease.observer, sidecar_conn, rows, fts_fields=fts_fields)
    except sqlite3.Error as err:
        if lease.observe(err, "sidecar:drain_pending"):
            raise
        logger.warning("sidecar reconcile failed to publish %d pending rows: %s", len(rows), err)
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


def _drain_pending(
    lease: DbLease,
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
        entries = _fetch_pending_entries_after(
            duckdb_conn,
            last_queued_at,
            last_rowid,
            batch_size,
        )
        if not entries:
            return

        for sub_entries in _partition_pending_entries(duckdb_conn, entries):
            _drain_pending_sub_page(
                lease,
                duckdb_conn,
                sidecar_conn,
                sub_entries,
                pending_drained,
                fts_fields=fts_fields,
                batch_size=batch_size,
            )
        last_queued_at = entries[-1].queued_at
        last_rowid = entries[-1].rowid
        if len(entries) < batch_size:
            return


def _fetch_pending_entries_after(
    duckdb_conn: duckdb.DuckDBPyConnection,
    last_queued_at: datetime | None,
    last_rowid: int,
    batch_size: int,
) -> list[_PendingEntry]:
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
    return [
        _PendingEntry(
            rowid=int(row[0]),
            kind=str(row[1]),
            entity_id=str(row[2]),
            op=str(row[3]),
            queued_at=row[4],
        )
        for row in rows
    ]


def _partition_pending_entries(
    duckdb_conn: duckdb.DuckDBPyConnection,
    entries: Sequence[_PendingEntry],
    *,
    max_bytes: int | None = None,
) -> Iterator[list[_PendingEntry]]:
    """Split pending entries in queue order so hydrated payloads stay within `max_bytes`."""
    if not entries:
        return
    limit_bytes = _TOOL_PAYLOAD_PAGE_MAX_BYTES if max_bytes is None else max_bytes
    message_ids = [
        entry.entity_id
        for entry in entries
        if entry.kind == _MESSAGE_KIND and entry.op == _UPSERT_OP
    ]
    tool_call_ids = [
        entry.entity_id
        for entry in entries
        if entry.kind == _TOOL_CALL_KIND and entry.op == _UPSERT_OP
    ]
    message_sizes = _measure_message_payload_bytes(duckdb_conn, message_ids) if message_ids else {}
    tool_sizes = (
        _measure_tool_call_payload_bytes(duckdb_conn, tool_call_ids) if tool_call_ids else {}
    )
    chunk: list[_PendingEntry] = []
    used = 0
    for entry in entries:
        if entry.op != _UPSERT_OP:
            entry_bytes = 0
        elif entry.kind == _TOOL_CALL_KIND:
            entry_bytes = tool_sizes.get(entry.entity_id, 0)
        elif entry.kind == _MESSAGE_KIND:
            entry_bytes = message_sizes.get(entry.entity_id, 0)
        else:
            entry_bytes = 0
        if chunk and used + entry_bytes > limit_bytes:
            yield chunk
            chunk = [entry]
            used = entry_bytes
        else:
            chunk.append(entry)
            used += entry_bytes
    if chunk:
        yield chunk


def _hydrate_pending_entries(
    duckdb_conn: duckdb.DuckDBPyConnection,
    entries: Sequence[_PendingEntry],
) -> list[_PendingRow]:
    if not entries:
        return []
    message_ids = list(
        dict.fromkeys(
            entry.entity_id
            for entry in entries
            if entry.kind == _MESSAGE_KIND and entry.op == _UPSERT_OP
        )
    )
    tool_call_ids = list(
        dict.fromkeys(
            entry.entity_id
            for entry in entries
            if entry.kind == _TOOL_CALL_KIND and entry.op == _UPSERT_OP
        )
    )
    # Scalar IN parameters retain index scans for scattered IDs on DuckDB 1.5.5;
    # binding lists or relations instead scans the populated primary tables.
    messages: dict[str, tuple[str, str]] = {}
    if message_ids:
        fts_content_expr = sql_derive_fts("context_text", "content")
        fts_thinking_expr = sql_derive_fts("context_text", "thinking")
        for lookup_chunk in _chunks(message_ids, _TOOL_ID_LOOKUP_CHUNK):
            message_placeholders = ", ".join("?" for _ in lookup_chunk)
            msg_payloads = duckdb_conn.execute(
                f"""
                SELECT message_id, {fts_content_expr}, {fts_thinking_expr}
                FROM message_state
                WHERE message_id IN ({message_placeholders})
                """,
                lookup_chunk,
            ).fetchall()
            for key, content, thinking in msg_payloads:
                messages[str(key)] = (str(content), str(thinking))
    tools: dict[str, str] = {}
    if tool_call_ids:
        for lookup_chunk in _chunks(tool_call_ids, _TOOL_ID_LOOKUP_CHUNK):
            tools.update(_fetch_tool_call_fts_chunk(duckdb_conn, lookup_chunk))

    return [
        _PendingRow(
            rowid=entry.rowid,
            kind=entry.kind,
            entity_id=entry.entity_id,
            op=entry.op,
            queued_at=entry.queued_at,
            message_columns=messages.get(entry.entity_id) if entry.kind == _MESSAGE_KIND else None,
            bash_command=tools.get(entry.entity_id) if entry.kind == _TOOL_CALL_KIND else None,
        )
        for entry in entries
    ]


def _publish_pending_page(
    observer: FailureObserver,
    sidecar_conn: sqlite3.Connection,
    rows: list[_PendingRow],
    *,
    fts_fields: tuple[str, ...] | None,
    deadline: PublicationDeadline | None = None,
) -> list[tuple[_PendingRow, str | None]]:
    """Publish a pending page in one commit, rolling a failed row back alone.

    Each row runs inside its own savepoint, so a row that SQLite or validation
    rejects leaves the rest of the page intact and stays pending. An error that
    ends the whole transaction, a lock another connection holds, or a failed
    commit propagates: no row of the page is published, so the caller must
    retain every row.
    """
    if not rows:
        return []
    outcomes: list[tuple[_PendingRow, str | None]] = []
    with publish_sidecar(sidecar_conn, deadline=deadline):
        for row in rows:
            sidecar_conn.execute("SAVEPOINT pending_row")
            try:
                _apply_pending_row(sidecar_conn, row, fts_fields=fts_fields)
            except BaseException as err:
                if observer.observe(err, "sidecar:pending_row") or not isinstance(
                    err, (sqlite3.Error, ValueError)
                ):
                    raise
                if not sidecar_conn.in_transaction or (
                    isinstance(err, sqlite3.Error) and is_lock_contention(err)
                ):
                    raise
                sidecar_conn.execute("ROLLBACK TO pending_row")
                outcomes.append((row, str(err)))
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
    lease: DbLease,
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
        while batch := lease.call(
            "sidecar:membership_fetch", lambda _: primary.fetchmany(batch_size)
        ):
            if batch != sidecar.fetchmany(batch_size):
                matches = False
                break
        else:
            matches = not sidecar.fetchmany(1)
    except BaseException as primary_error:
        lease.observe(primary_error, "sidecar:membership_comparison")
        try:
            lease.cleanup_call("sidecar:membership_cursor_close", lambda _: sidecar.close())
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "membership read and close failed", [primary_error, cleanup]
            ) from None
        raise
    else:
        lease.cleanup_call("sidecar:membership_cursor_close", lambda _: sidecar.close())

    return matches


def _reconcile_message_differences(
    lease: DbLease,
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    fts_fields: tuple[str, ...] | None,
    batch_size: int,
) -> tuple[int, int]:
    if _memberships_match(
        lease,
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
    lease: DbLease,
    duckdb_conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    *,
    fts_fields: tuple[str, ...] | None,
    batch_size: int,
) -> tuple[int, int]:
    primary_sql = (
        "SELECT id FROM tool_calls ORDER BY id"
        if should_index_bash_fts(fts_fields)
        else "SELECT id FROM tool_calls WHERE FALSE"
    )
    if _memberships_match(
        lease,
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
            WHERE id > ?
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
    written = 0
    for chunk in iter_tool_call_fts_chunks_for_ids(duckdb_conn, tool_call_ids):
        upsert_tool_call_fts_batch(
            sidecar_conn,
            chunk,
            fields=fts_fields,
            batch_size=batch_size,
        )
        written += len(chunk)
        chunk.clear()
    return written


def _fetch_message_payloads(
    duckdb_conn: duckdb.DuckDBPyConnection,
    message_ids: list[str],
) -> list[tuple[str, str, str]]:
    if not message_ids:
        return []
    placeholders = ", ".join("?" for _ in message_ids)
    fts_content_expr = sql_derive_fts("context_text", "content")
    fts_thinking_expr = sql_derive_fts("context_text", "thinking")
    fetched = {
        str(message_id): (str(message_id), str(fts_content), str(fts_thinking))
        for message_id, fts_content, fts_thinking in duckdb_conn.execute(
            f"""
            SELECT message_id, {fts_content_expr}, {fts_thinking_expr}
            FROM message_state
            WHERE message_id IN ({placeholders})
            """,
            message_ids,
        ).fetchall()
    }
    return [fetched[message_id] for message_id in message_ids if message_id in fetched]


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
