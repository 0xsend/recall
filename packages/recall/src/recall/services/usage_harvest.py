"""Harvest provider usage logs into usage_events and roll up onto sessions.

Grok Build writes rotating ``~/.grok/logs/unified.jsonl`` with
``shell.turn.inference_done`` events. Session transcripts do not carry tokens;
this module is the only path that fills Grok session token columns
(REQ-USAGE-010-015).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from recall.core.types import Source
from recall.db.mutations import execute_count
from recall.parsers.common import parse_timestamp

logger = logging.getLogger("recall.usage_harvest")

_GROK_INFERENCE_MSG = "shell.turn.inference_done"
_GROK_SOURCE = Source.GROK.value


# A batch stops at whichever bound it reaches first. Events bound the
# writer's work for the batch; bytes bound the reading between commits. The
# line that crosses the byte bound is finished, so a batch may read past it.
MAX_EVENTS_PER_BATCH = 2048
MAX_LOG_BYTES_PER_BATCH = 4 * 1024 * 1024


@dataclass(frozen=True)
class LogCursor:
    byte_offset: int
    file_size: int


@dataclass(frozen=True)
class _UsageEvent:
    id: str
    source_session_id: str
    ts: datetime | None
    prompt_tokens: int | None
    cached_prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None


@dataclass(frozen=True)
class UsageLogBatch:
    """Parsed log lines between two cursor positions, ready to commit."""

    path: str
    # The stored cursor the batch was read from; the commit applies only while
    # it is still the stored cursor.
    start: LogCursor
    end: LogCursor
    rotated: bool
    # Unique by id, in log order.
    events: tuple[_UsageEvent, ...]
    events_seen: int
    events_skipped: int
    caught_up: bool

    def __post_init__(self) -> None:
        assert len(self.events) <= MAX_EVENTS_PER_BATCH
        assert self.end.byte_offset >= (0 if self.rotated else self.start.byte_offset)


@dataclass(frozen=True)
class HarvestResult:
    events_seen: int = 0
    events_upserted: int = 0
    events_skipped: int = 0
    sessions_rolled_up: int = 0
    rotated: bool = False
    path: str | None = None
    # A missing log and a fully caught-up log both report zero counters. Without
    # this flag an install whose log never appeared is indistinguishable from a
    # healthy one, so the ledger can stall unnoticed (REQ-USAGE-010).
    log_present: bool = True
    # False while log past the stored cursor remains for a later call.
    caught_up: bool = True
    # Cursor offset and log size after the call.
    byte_offset: int = 0
    log_size_bytes: int = 0


def default_grok_unified_log(*, home: Path | None = None) -> Path:
    root = home if home is not None else Path.home()
    return root / ".grok" / "logs" / "unified.jsonl"


def harvest_grok_unified_log(
    conn: duckdb.DuckDBPyConnection,
    path: Path | None = None,
    *,
    host: str | None = None,
) -> HarvestResult:
    """Read the next batch of Grok unified.jsonl and commit it on one connection."""
    batch = read_grok_usage_batch(conn, path)
    if batch is None:
        return HarvestResult(path=str(path or default_grok_unified_log()), log_present=False)
    return commit_grok_usage_batch(conn, batch, host=host)


def read_grok_usage_batch(
    conn: duckdb.DuckDBPyConnection, path: Path | None = None
) -> UsageLogBatch | None:
    """Parse the next batch past the stored cursor; None when the log is absent.

    Writes nothing, so it runs on a read cursor outside the writer. Lines are
    parsed whole whatever their length; a final line without its newline is
    still being written and is left for a later batch.
    """
    log_path = path if path is not None else default_grok_unified_log()
    if not log_path.is_file():
        logger.debug("grok usage log absent at %s; nothing to harvest", log_path)
        return None
    absolute = str(log_path.expanduser().resolve())
    file_size = int(log_path.stat().st_size)
    start = _load_cursor(conn, absolute)
    # REQ-USAGE-013: rotation / truncation
    rotated = file_size < start.file_size or file_size < start.byte_offset
    offset = 0 if rotated else start.byte_offset

    events: dict[str, _UsageEvent] = {}
    events_seen = 0
    events_skipped = 0
    caught_up = False
    with log_path.open("rb") as handle:
        handle.seek(offset)
        first = offset
        while len(events) < MAX_EVENTS_PER_BATCH and offset - first < MAX_LOG_BYTES_PER_BATCH:
            # WAIVER(bounded allocation): a line is held whole to parse it, so a
            # pathological line costs its size in memory on this read path; the
            # writer receives only the parsed batch. Usage records are a few
            # hundred bytes and Grok's longest lines are under 2 KiB. Debt --
            # removed by an incremental JSON parser over a bounded buffer.
            raw = handle.readline()
            if not raw.endswith(b"\n"):
                caught_up = True
                break
            offset = handle.tell()
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            events_seen += 1
            event = _parse_inference_event(line)
            if event is None:
                events_skipped += 1
                continue
            # Lines with identical semantic payload are one event; the first wins.
            events.setdefault(event.id, event)
        # The file can grow during the read, past the size sampled before it. The
        # cursor records the largest size this read saw, so a later replacement
        # shorter than that is a rotation even when it is no shorter than the
        # offset this batch stopped at (REQ-USAGE-013).
        size_after = os.fstat(handle.fileno()).st_size
    return UsageLogBatch(
        path=absolute,
        start=start,
        end=LogCursor(byte_offset=offset, file_size=max(file_size, size_after, offset)),
        rotated=rotated,
        events=tuple(events.values()),
        events_seen=events_seen,
        events_skipped=events_skipped,
        caught_up=caught_up,
    )


def commit_grok_usage_batch(
    conn: duckdb.DuckDBPyConnection, batch: UsageLogBatch, *, host: str | None
) -> HarvestResult:
    """Commit a batch's events, its cursor and the rollup in one transaction.

    The writer's work is at most ``MAX_EVENTS_PER_BATCH`` inserts, one cursor
    write and the all-session rollup. When another harvest moved the cursor
    after the batch was read, nothing is written and the result is not caught
    up, so the caller reads again from the stored cursor.
    """
    conn.execute("BEGIN")
    try:
        if _load_cursor(conn, batch.path) != batch.start:
            conn.execute("COMMIT")
            return HarvestResult(
                path=batch.path,
                caught_up=False,
                byte_offset=batch.start.byte_offset,
                log_size_bytes=batch.end.file_size,
            )
        events_upserted = _insert_new_events(
            conn, batch.events, host=host, harvested_at=datetime.now(UTC)
        )
        if batch.end != batch.start:
            _save_cursor(conn, batch.path, batch.end)
        # Every call rolls up every sid, not only this batch's, so a session row
        # rewritten without tokens heals even when the log has nothing new
        # (REQ-USAGE-014). The rollup writes only rows whose totals differ.
        sessions_rolled = rollup_grok_sessions(conn)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return HarvestResult(
        events_seen=batch.events_seen,
        events_upserted=events_upserted,
        events_skipped=batch.events_skipped,
        sessions_rolled_up=sessions_rolled,
        rotated=batch.rotated,
        path=batch.path,
        caught_up=batch.caught_up,
        byte_offset=batch.end.byte_offset,
        log_size_bytes=batch.end.file_size,
    )


def harvest_calls_to_cover(log_size_bytes: int) -> int:
    """Harvest calls that together read every byte of a log of *log_size_bytes*.

    A call that is not caught up read ``MAX_LOG_BYTES_PER_BATCH`` bytes or
    ``MAX_EVENTS_PER_BATCH`` lines of at least one byte, or found its cursor
    moved by a concurrent harvest that did; hence the factor of two.
    """
    assert log_size_bytes >= 0
    per_call = min(MAX_LOG_BYTES_PER_BATCH, MAX_EVENTS_PER_BATCH)
    return 2 * (log_size_bytes // per_call + 1)


# A sid has several sessions when its workspace moved and the old path is still
# indexed. One of them, the smallest session id, owns the sid's usage: it alone
# carries the totals and its events link to it, so totals summed over sessions
# count each event once and the indexed link column is never alternated.
_SID_OWNERS = """
    SELECT source_session_id, MIN(id) AS session_id
    FROM sessions
    WHERE source = $source
    GROUP BY source_session_id
"""


def rollup_grok_sessions(conn: duckdb.DuckDBPyConnection) -> int:
    """Write each Grok sid's summed usage onto its owner session and link its events.

    The sid's other sessions carry no totals. Both statements change only rows
    whose value differs, so a rollup with nothing new writes nothing. Returns
    the number of sessions whose token totals changed.
    """
    changed = conn.execute(
        f"""
        UPDATE session_state
        SET input_tokens = rollup.prompt_sum,
            cached_input_tokens = rollup.cached_sum,
            output_tokens = rollup.completion_sum
        FROM (
            SELECT s.id AS session_id,
                   CASE WHEN s.id = owners.session_id THEN totals.prompt_sum END AS prompt_sum,
                   CASE WHEN s.id = owners.session_id THEN totals.cached_sum END AS cached_sum,
                   CASE WHEN s.id = owners.session_id THEN totals.completion_sum END
                       AS completion_sum
            FROM sessions s
            JOIN ({_SID_OWNERS}) AS owners ON owners.source_session_id = s.source_session_id
            JOIN (
                SELECT source_session_id,
                       COALESCE(SUM(prompt_tokens), 0) AS prompt_sum,
                       COALESCE(SUM(cached_prompt_tokens), 0) AS cached_sum,
                       COALESCE(SUM(completion_tokens), 0) AS completion_sum
                FROM usage_events
                WHERE source = $source
                GROUP BY source_session_id
            ) AS totals ON totals.source_session_id = s.source_session_id
            WHERE s.source = $source
        ) AS rollup
        WHERE session_state.session_id = rollup.session_id
          AND (session_state.input_tokens IS DISTINCT FROM rollup.prompt_sum
               OR session_state.cached_input_tokens IS DISTINCT FROM rollup.cached_sum
               OR session_state.output_tokens IS DISTINCT FROM rollup.completion_sum)
        """,
        {"source": _GROK_SOURCE},
    ).fetchone()
    conn.execute(
        f"""
        UPDATE usage_events
        SET session_id = owners.session_id
        FROM ({_SID_OWNERS}) AS owners
        WHERE usage_events.source = $source
          AND usage_events.source_session_id = owners.source_session_id
          AND usage_events.session_id IS DISTINCT FROM owners.session_id
        """,
        {"source": _GROK_SOURCE},
    )
    assert changed is not None
    return int(changed[0])


def _event_id(
    *,
    source: str,
    sid: str,
    ts: Any,
    loop_index: Any,
    prompt: int | None,
    completion: int | None,
) -> str:
    # Semantic identity (REQ-USAGE-012): same sid/ts/loop/tokens is one event.
    # Rotation re-reads must not double-count; distinct lines with identical
    # semantic payload are treated as the same event.
    material = "|".join(
        [
            source,
            sid,
            str(ts or ""),
            str(loop_index if loop_index is not None else ""),
            str(prompt if prompt is not None else ""),
            str(completion if completion is not None else ""),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_cursor(conn: duckdb.DuckDBPyConnection, path: str) -> LogCursor:
    row = conn.execute(
        "SELECT byte_offset, file_size FROM usage_log_cursors WHERE path = ?",
        [path],
    ).fetchone()
    if row is None:
        return LogCursor(byte_offset=0, file_size=0)
    return LogCursor(byte_offset=int(row[0] or 0), file_size=int(row[1] or 0))


def _save_cursor(conn: duckdb.DuckDBPyConnection, path: str, cursor: LogCursor) -> None:
    now = datetime.now(UTC)
    existing = conn.execute(
        "SELECT 1 FROM usage_log_cursors WHERE path = ?",
        [path],
    ).fetchone()
    if existing is None:
        conn.execute(
            """
            INSERT INTO usage_log_cursors (path, byte_offset, file_size, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            [path, cursor.byte_offset, cursor.file_size, now],
        )
    else:
        conn.execute(
            """
            UPDATE usage_log_cursors
            SET byte_offset = ?, file_size = ?, updated_at = ?
            WHERE path = ?
            """,
            [cursor.byte_offset, cursor.file_size, now, path],
        )


def _parse_inference_event(line: str) -> _UsageEvent | None:
    """Return the line's inference event, or None when REQ-USAGE-011 skips it."""
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(entry, dict) or entry.get("msg") != _GROK_INFERENCE_MSG:
        return None
    sid = entry.get("sid")
    ctx = entry.get("ctx")
    if not isinstance(sid, str) or not sid or not isinstance(ctx, dict):
        return None
    prompt = _optional_int(ctx.get("prompt_tokens"))
    completion = _optional_int(ctx.get("completion_tokens"))
    raw_ts = entry.get("ts")
    return _UsageEvent(
        id=_event_id(
            source=_GROK_SOURCE,
            sid=sid,
            ts=raw_ts,
            loop_index=ctx.get("loop_index"),
            prompt=prompt,
            completion=completion,
        ),
        source_session_id=sid,
        ts=parse_timestamp(raw_ts) if isinstance(raw_ts, str) else None,
        prompt_tokens=prompt,
        cached_prompt_tokens=_optional_int(ctx.get("cached_prompt_tokens")),
        completion_tokens=completion,
        reasoning_tokens=_optional_int(ctx.get("reasoning_tokens")),
    )


def _insert_new_events(
    conn: duckdb.DuckDBPyConnection,
    events: Sequence[_UsageEvent],
    *,
    host: str | None,
    harvested_at: datetime,
) -> int:
    """Insert events whose ids are not stored yet; return how many were inserted.

    One set-based statement per batch: per-event statements each paid a
    commit and, under concurrent reads, a scheduling wait, which held the
    writer for tens of seconds on a first harvest.
    """
    import pyarrow as pa

    if not events:
        return 0
    # An aware timestamp is cast to TIMESTAMP in the session time zone, the
    # same conversion a bound aware datetime receives; naive ones pass through.
    aware = [event.ts is not None and event.ts.utcoffset() is not None for event in events]
    table = pa.table(
        {
            "id": pa.array([event.id for event in events], type=pa.string()),
            "source_session_id": pa.array(
                [event.source_session_id for event in events], type=pa.string()
            ),
            "ts": pa.array(
                [
                    None if is_aware else event.ts
                    for event, is_aware in zip(events, aware, strict=True)
                ],
                type=pa.timestamp("us"),
            ),
            "ts_utc": pa.array(
                [
                    event.ts if is_aware else None
                    for event, is_aware in zip(events, aware, strict=True)
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "prompt_tokens": pa.array([event.prompt_tokens for event in events], type=pa.int64()),
            "cached_prompt_tokens": pa.array(
                [event.cached_prompt_tokens for event in events], type=pa.int64()
            ),
            "completion_tokens": pa.array(
                [event.completion_tokens for event in events], type=pa.int64()
            ),
            "reasoning_tokens": pa.array(
                [event.reasoning_tokens for event in events], type=pa.int64()
            ),
        }
    )
    conn.register("_recall_usage_event_inserts", table)
    try:
        return execute_count(
            conn,
            """
            INSERT INTO usage_events (
                id, source, source_session_id, session_id, ts,
                prompt_tokens, cached_prompt_tokens, completion_tokens, reasoning_tokens,
                host, harvested_at
            )
            SELECT id, ?, source_session_id, NULL,
                   COALESCE(CAST(ts_utc AS TIMESTAMP), ts),
                   prompt_tokens, cached_prompt_tokens, completion_tokens, reasoning_tokens,
                   ?, ?
            FROM _recall_usage_event_inserts
            ON CONFLICT DO NOTHING
            """,
            [_GROK_SOURCE, host, harvested_at],
        )
    finally:
        conn.unregister("_recall_usage_event_inserts")
