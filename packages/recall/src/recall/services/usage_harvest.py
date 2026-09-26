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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from recall.core.types import Source
from recall.parsers.common import parse_timestamp

logger = logging.getLogger("recall.usage_harvest")

_GROK_INFERENCE_MSG = "shell.turn.inference_done"
_GROK_SOURCE = Source.GROK.value


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


def default_grok_unified_log(*, home: Path | None = None) -> Path:
    root = home if home is not None else Path.home()
    return root / ".grok" / "logs" / "unified.jsonl"


def harvest_grok_unified_log(
    conn: duckdb.DuckDBPyConnection,
    path: Path | None = None,
    *,
    host: str | None = None,
) -> HarvestResult:
    """Incrementally harvest Grok unified.jsonl into usage_events and roll up.

    Cursor is stored in ``usage_log_cursors``. Size shrink vs last seen size
    resets the byte offset (rotation) without deleting existing events;
    stable event ids keep re-harvest idempotent.
    """
    log_path = path if path is not None else default_grok_unified_log()
    if not log_path.is_file():
        logger.debug("grok usage log absent at %s; nothing to harvest", log_path)
        return HarvestResult(path=str(log_path), log_present=False)

    absolute = str(log_path.expanduser().resolve())
    stat = log_path.stat()
    file_size = int(stat.st_size)

    cursor_offset, cursor_size = _load_cursor(conn, absolute)
    rotated = False
    offset = cursor_offset
    if file_size < cursor_size or file_size < cursor_offset:
        # REQ-USAGE-013: rotation / truncation
        offset = 0
        rotated = True

    events_seen = 0
    events_upserted = 0
    events_skipped = 0
    sids_touched: set[str] = set()
    harvested_at = datetime.now(UTC)

    with log_path.open("rb") as handle:
        if offset > 0:
            handle.seek(offset)
        while True:
            raw = handle.readline()
            if not raw:
                break
            next_offset = handle.tell()
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                offset = next_offset
                continue
            events_seen += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                events_skipped += 1
                offset = next_offset
                continue
            if not isinstance(entry, dict):
                events_skipped += 1
                offset = next_offset
                continue
            if entry.get("msg") != _GROK_INFERENCE_MSG:
                events_skipped += 1
                offset = next_offset
                continue
            sid = entry.get("sid")
            if not isinstance(sid, str) or not sid:
                events_skipped += 1
                offset = next_offset
                continue
            ctx = entry.get("ctx")
            if not isinstance(ctx, dict):
                events_skipped += 1
                offset = next_offset
                continue

            prompt = _optional_int(ctx.get("prompt_tokens"))
            cached = _optional_int(ctx.get("cached_prompt_tokens"))
            completion = _optional_int(ctx.get("completion_tokens"))
            reasoning = _optional_int(ctx.get("reasoning_tokens"))
            loop_index = ctx.get("loop_index")
            ts = parse_timestamp(entry.get("ts")) if isinstance(entry.get("ts"), str) else None

            event_id = _event_id(
                source=_GROK_SOURCE,
                sid=sid,
                ts=entry.get("ts"),
                loop_index=loop_index,
                prompt=prompt,
                completion=completion,
            )
            if _upsert_event(
                conn,
                event_id=event_id,
                source=_GROK_SOURCE,
                source_session_id=sid,
                ts=ts,
                prompt_tokens=prompt,
                cached_prompt_tokens=cached,
                completion_tokens=completion,
                reasoning_tokens=reasoning,
                host=host,
                harvested_at=harvested_at,
            ):
                events_upserted += 1
            sids_touched.add(sid)
            offset = next_offset

    if (offset, file_size) != (cursor_offset, cursor_size):
        _save_cursor(conn, absolute, offset, file_size)
    # Every tick rolls up every sid, not only this tick's, so a session row
    # rewritten without tokens heals even when the log has nothing new
    # (REQ-USAGE-014). The rollup writes only rows whose totals differ.
    sessions_rolled = rollup_grok_sessions(conn)
    return HarvestResult(
        events_seen=events_seen,
        events_upserted=events_upserted,
        events_skipped=events_skipped,
        sessions_rolled_up=sessions_rolled,
        rotated=rotated,
        path=absolute,
    )


def rollup_grok_sessions(conn: duckdb.DuckDBPyConnection) -> int:
    """Write each Grok sid's summed usage onto its sessions and link its events.

    Both statements change only rows whose value differs, so a rollup with
    nothing new writes nothing. Returns the number of sessions whose token
    totals changed.
    """
    changed = conn.execute(
        """
        UPDATE session_state
        SET input_tokens = totals.prompt_sum,
            cached_input_tokens = totals.cached_sum,
            output_tokens = totals.completion_sum
        FROM (
            SELECT s.id AS session_id,
                   COALESCE(SUM(e.prompt_tokens), 0) AS prompt_sum,
                   COALESCE(SUM(e.cached_prompt_tokens), 0) AS cached_sum,
                   COALESCE(SUM(e.completion_tokens), 0) AS completion_sum
            FROM usage_events e
            JOIN sessions s
              ON s.source = e.source AND s.source_session_id = e.source_session_id
            WHERE e.source = ?
            GROUP BY s.id
        ) AS totals
        WHERE session_state.session_id = totals.session_id
          AND (session_state.input_tokens IS DISTINCT FROM totals.prompt_sum
               OR session_state.cached_input_tokens IS DISTINCT FROM totals.cached_sum
               OR session_state.output_tokens IS DISTINCT FROM totals.completion_sum)
        """,
        [_GROK_SOURCE],
    ).fetchone()
    # A sid has several sessions when its workspace moved and the old path is
    # still indexed. Its events link to one owner, the smallest session id, so
    # the indexed link column is written once rather than alternated per session.
    conn.execute(
        """
        UPDATE usage_events
        SET session_id = owners.session_id
        FROM (
            SELECT source_session_id, MIN(id) AS session_id
            FROM sessions
            WHERE source = ?
            GROUP BY source_session_id
        ) AS owners
        WHERE usage_events.source = ?
          AND usage_events.source_session_id = owners.source_session_id
          AND usage_events.session_id IS DISTINCT FROM owners.session_id
        """,
        [_GROK_SOURCE, _GROK_SOURCE],
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


def _load_cursor(conn: duckdb.DuckDBPyConnection, path: str) -> tuple[int, int]:
    row = conn.execute(
        "SELECT byte_offset, file_size FROM usage_log_cursors WHERE path = ?",
        [path],
    ).fetchone()
    if row is None:
        return 0, 0
    return int(row[0] or 0), int(row[1] or 0)


def _save_cursor(
    conn: duckdb.DuckDBPyConnection,
    path: str,
    byte_offset: int,
    file_size: int,
) -> None:
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
            [path, byte_offset, file_size, now],
        )
    else:
        conn.execute(
            """
            UPDATE usage_log_cursors
            SET byte_offset = ?, file_size = ?, updated_at = ?
            WHERE path = ?
            """,
            [byte_offset, file_size, now, path],
        )


def _upsert_event(
    conn: duckdb.DuckDBPyConnection,
    *,
    event_id: str,
    source: str,
    source_session_id: str,
    ts: datetime | None,
    prompt_tokens: int | None,
    cached_prompt_tokens: int | None,
    completion_tokens: int | None,
    reasoning_tokens: int | None,
    host: str | None,
    harvested_at: datetime,
) -> bool:
    """Insert event if new. Returns True when a row was inserted."""
    existing = conn.execute(
        "SELECT 1 FROM usage_events WHERE id = ?",
        [event_id],
    ).fetchone()
    if existing is not None:
        return False
    conn.execute(
        """
        INSERT INTO usage_events (
            id, source, source_session_id, session_id, ts,
            prompt_tokens, cached_prompt_tokens, completion_tokens, reasoning_tokens,
            host, harvested_at
        ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            event_id,
            source,
            source_session_id,
            ts,
            prompt_tokens,
            cached_prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            host,
            harvested_at,
        ],
    )
    return True
