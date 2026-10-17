"""Coalesced keyword writes to the SQLite FTS sidecar.

DuckDB is authoritative and the sidecar is derived from it. A keyword change is
staged inside the DuckDB transaction that makes it, and staging records an
`upsert` row in `fts_sidecar_pending` in that same transaction: the durable
intent. The change joins the shared buffer only when that transaction commits,
so a rollback publishes nothing. A flush applies everything buffered in as few
SQLite transactions as `FTS_SIDECAR_BATCH_SIZE` allows (each SQLite commit
writes a new FTS5 segment and journals the pages it overwrites, so rows per
commit set the sidecar's write amplification). It commits SQLite before it
retires the intents. A crash or a failed flush therefore leaves intents behind,
and the reconciler re-derives those rows from DuckDB.

The daemon's writers stage every keyword change into one buffer, but a few
paths queue pending rows themselves (moved-session supersession queues
deletes and carried-context upserts). So a flush retires an id's `upsert`
intents only when they are exactly the ones this buffer wrote; an id with any
other pending row keeps all of them, and the repair loop, which runs after a
flush, re-derives it from DuckDB.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

import duckdb

from recall.core.config import AppConfig
from recall.core.telemetry import Stage, measured
from recall.db.fts_sidecar import (
    FTS_SIDECAR_BATCH_SIZE,
    FtsKind,
    FtsSidecarUnavailableError,
    delete_fts_rows,
    open_sidecar,
    scope_message_fts_columns,
    should_index_bash_fts,
    sidecar_path,
    write_message_fts_rows,
    write_tool_call_fts_rows,
)

# Flush bounds. The age bound sits inside the 15 s active-freshness floor
# (BRIEF) with room for raw scheduling; the row and text bounds keep one flush
# inside a short writer turn and its rollback journal small.
FLUSH_AGE_SECONDS = 5.0
FLUSH_ROWS = 4_096
FLUSH_TEXT_CHARS = 16 * 1024 * 1024

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MessageRows:
    rows: tuple[tuple[str, str, str], ...]  # (message_id, content, thinking), already scoped


@dataclass(frozen=True)
class _ToolCallRows:
    rows: tuple[tuple[str, str | None], ...]  # a None command removes the row


@dataclass(frozen=True)
class _Deletions:
    kind: FtsKind
    ids: tuple[str, ...]


type _Change = _MessageRows | _ToolCallRows | _Deletions
type _Intents = Counter[tuple[FtsKind, str]]


def _change_ids(change: _Change) -> tuple[FtsKind, list[str]]:
    match change:
        case _MessageRows(rows=rows):
            return "message", [message_id for message_id, _, _ in rows]
        case _ToolCallRows(rows=rows):
            return "tool_call", [tool_call_id for tool_call_id, _ in rows]
        case _Deletions(kind=kind, ids=ids):
            return kind, list(ids)


def _change_chars(change: _Change) -> int:
    match change:
        case _MessageRows(rows=rows):
            return sum(len(content) + len(thinking) for _, content, thinking in rows)
        case _ToolCallRows(rows=rows):
            return sum(len(command or "") for _, command in rows)
        case _Deletions():
            return 0


class SidecarStage:
    """Keyword changes made by one open DuckDB transaction."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, fields: tuple[str, ...] | None) -> None:
        self._conn = conn
        self._fields = fields
        self.changes: list[_Change] = []
        self.intents: _Intents = Counter()

    def upsert_messages(self, rows: Iterable[tuple[str, str, str]]) -> None:
        scoped = tuple(
            (message_id, *scope_message_fts_columns(content, thinking, self._fields))
            for message_id, content, thinking in rows
        )
        self._stage(_MessageRows(scoped), len(scoped))

    def upsert_tool_calls(self, rows: Iterable[tuple[str, str | None]]) -> None:
        index_bash = should_index_bash_fts(self._fields)
        commands = tuple(
            (tool_call_id, command if index_bash else None) for tool_call_id, command in rows
        )
        self._stage(_ToolCallRows(commands), len(commands))

    def delete(self, kind: FtsKind, ids: Iterable[str]) -> None:
        materialized = tuple(dict.fromkeys(ids))
        self._stage(_Deletions(kind, materialized), len(materialized))

    def _stage(self, change: _Change, size: int) -> None:
        if size == 0:
            return
        kind, ids = _change_ids(change)
        # Set-based: DuckDB 1.5.5 binds a long executemany one parameter at a
        # time, which costs seconds of writer CPU on a large session.
        self._conn.execute(
            "INSERT INTO fts_sidecar_pending(kind, id, op) SELECT ?, unnest(?::TEXT[]), 'upsert'",
            [kind, ids],
        )
        self.changes.append(change)
        self.intents.update((kind, entity_id) for entity_id in ids)


class SidecarBuffer:
    """Committed keyword changes waiting for one coalesced sidecar flush.

    Staging and flushing run on the database writer; the due time may be read
    from any thread. With `flush_when_full`, the commit that fills the buffer
    flushes it inline; without it, a full buffer is due at once and its owner
    flushes it in a writer turn of its own, so no commit carries another
    turn's keyword work.
    """

    def __init__(
        self,
        path: Path,
        *,
        flush_when_full: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._path = path
        self._flush_when_full = flush_when_full
        self._clock = clock
        self._lock = threading.Lock()
        self._changes: list[_Change] = []
        self._intents: _Intents = Counter()
        self._rows = 0
        self._chars = 0
        self._oldest: float | None = None
        self._full = False
        self.last_error: str | None = None

    @contextmanager
    def transaction(
        self, conn: duckdb.DuckDBPyConnection, fields: tuple[str, ...] | None
    ) -> Iterator[SidecarStage]:
        """Stage the keyword changes of one DuckDB transaction.

        The body owns BEGIN and COMMIT. Its changes are published only when it
        returns normally, after COMMIT; an exception (and its ROLLBACK) discards
        them together with their intents.
        """
        stage = SidecarStage(conn, fields)
        yield stage
        if not stage.changes:
            return
        with self._lock:
            if self._oldest is None:
                self._oldest = self._clock()
            self._changes.extend(stage.changes)
            self._intents.update(stage.intents)
            self._rows += sum(len(_change_ids(change)[1]) for change in stage.changes)
            self._chars += sum(_change_chars(change) for change in stage.changes)
            self._full = self._rows >= FLUSH_ROWS or self._chars >= FLUSH_TEXT_CHARS
            flush_now = self._full and self._flush_when_full
        if flush_now:
            self.flush(conn)

    def seconds_until_due(self) -> float | None:
        """Seconds until the buffer is due (age or size bound); None when empty."""
        with self._lock:
            if self._oldest is None:
                return None
            if self._full:
                return 0.0
            return max(0.0, self._oldest + FLUSH_AGE_SECONDS - self._clock())

    def flush(self, conn: duckdb.DuckDBPyConnection) -> int:
        """Apply every buffered change; return the rows written.

        Call outside any DuckDB transaction: retiring intents must not join a
        transaction whose own intents are still unpublished.
        """
        with self._lock:
            changes = self._changes
            intents = self._intents
            rows = self._rows
            self._changes = []
            self._intents = Counter()
            self._rows = 0
            self._chars = 0
            self._oldest = None
            self._full = False
        if not changes:
            return 0
        try:
            _apply(self._path, changes)
        except (sqlite3.Error, FtsSidecarUnavailableError) as err:
            # The intents are still durable in DuckDB; the repair loop
            # re-derives these rows.
            self.last_error = f"{type(err).__name__}: {err}"
            logger.warning(
                "keyword sidecar flush failed for %d rows: %s; left for pending repair",
                rows,
                err,
            )
            return 0
        self.last_error = None
        _retire_intents(conn, intents)
        return rows


def keyword_buffer_for(config: AppConfig, *, flush_when_full: bool = True) -> SidecarBuffer | None:
    """A buffer for the configured keyword store; None when keyword search lives in DuckDB."""
    if config.fts.backend != "sqlite_sidecar":
        return None
    return SidecarBuffer(sidecar_path(config.data_dir), flush_when_full=flush_when_full)


def stage_keywords(
    buffer: SidecarBuffer | None,
    conn: duckdb.DuckDBPyConnection,
    fields: tuple[str, ...] | None,
) -> AbstractContextManager[SidecarStage | None]:
    """`SidecarBuffer.transaction`, or no stage when keyword search is not in the sidecar."""
    if buffer is None:
        return nullcontext()
    return buffer.transaction(conn, fields)


@contextmanager
def flushed_on_exit(
    buffer: SidecarBuffer | None, conn: duckdb.DuckDBPyConnection
) -> Iterator[SidecarBuffer | None]:
    """Flush what a caller-owned buffer published, also when the caller fails."""
    if buffer is None:
        yield None
        return
    try:
        yield buffer
    except BaseException:
        try:
            buffer.flush(conn)
        except Exception:
            # The caller's error is the one to report; unflushed changes keep
            # their durable intents for the next repair.
            logger.exception("keyword sidecar flush after a failed write also failed")
        raise
    buffer.flush(conn)


@measured(Stage.FTS)
def _apply(path: Path, changes: list[_Change]) -> None:
    sidecar = open_sidecar(path)
    try:
        for group in _bounded_groups(changes, FTS_SIDECAR_BATCH_SIZE):
            with sidecar:
                for change in group:
                    _apply_change(sidecar, change)
    finally:
        sidecar.close()


def _apply_change(sidecar: sqlite3.Connection, change: _Change) -> None:
    match change:
        case _MessageRows(rows=rows):
            write_message_fts_rows(sidecar, list(rows))
        case _ToolCallRows(rows=rows):
            write_tool_call_fts_rows(sidecar, list(rows))
        case _Deletions(kind=kind, ids=ids):
            delete_fts_rows(sidecar, ids, kind)


def _bounded_groups(changes: list[_Change], limit: int) -> Iterator[list[_Change]]:
    """Split changes, in order, into groups of at most `limit` rows."""
    group: list[_Change] = []
    size = 0
    for change in changes:
        for part in _split(change, limit):
            part_size = len(_change_ids(part)[1])
            if group and size + part_size > limit:
                yield group
                group, size = [], 0
            group.append(part)
            size += part_size
    if group:
        yield group


def _split(change: _Change, limit: int) -> Iterator[_Change]:
    match change:
        case _MessageRows(rows=rows):
            yield from (_MessageRows(part) for part in batched(rows, limit))
        case _ToolCallRows(rows=rows):
            yield from (_ToolCallRows(part) for part in batched(rows, limit))
        case _Deletions(kind=kind, ids=ids):
            yield from (_Deletions(kind, part) for part in batched(ids, limit))


def _retire_intents(conn: duckdb.DuckDBPyConnection, intents: _Intents) -> None:
    """Delete the `upsert` intents of ids whose every queued intent this buffer wrote."""
    for kind in ("message", "tool_call"):
        staged = sorted(
            (entity_id, count) for (k, entity_id), count in intents.items() if k == kind
        )
        if not staged:
            continue
        conn.execute(
            """
            DELETE FROM fts_sidecar_pending
            WHERE op = 'upsert' AND kind = $kind AND id IN (
                SELECT staged.id
                FROM (SELECT unnest($ids::TEXT[]) AS id, unnest($counts::BIGINT[]) AS n) staged
                JOIN (
                    SELECT id, count(*) AS n FROM fts_sidecar_pending
                    WHERE op = 'upsert' AND kind = $kind
                    GROUP BY id
                ) queued USING (id)
                WHERE queued.n = staged.n
            )
            """,
            {
                "kind": kind,
                "ids": [entity_id for entity_id, _ in staged],
                "counts": [count for _, count in staged],
            },
        )
