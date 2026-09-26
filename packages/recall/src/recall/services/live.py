"""Reads that answer "what is this agent doing right now" (REQ-LIVE-001).

Liveness is derived from recent catalog observations, watch events and process
marks independently of raw indexing. Committed transcript rows supply turn state
and carry separately validated freshness. SQL bounds roster pages and coverage.
"""

from __future__ import annotations

import base64
import binascii
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal, overload

import duckdb

from recall.core.models import Message, StopMarker, ToolCall
from recall.core.types import Role
from recall.services.live_events import SessionIndexed, live_path_key
from recall.services.live_marks import pid_alive
from recall.services.live_session_set import LiveMember
from recall.services.sessions import (
    MESSAGE_SELECT,
    message_from_row,
    tool_call_columns,
    tool_call_from_row,
)

# REQ-LIVE-005 truncates the quoted turn text. 400 chars is enough to read what
# the agent said last without turning a fleet listing into a transcript dump.
DEFAULT_TEXT_BUDGET = 400

# Bound on one IN-list. `daemon.live_max_subscriptions` already caps the live
# set, but the join must not inherit that cap as an assumption.
_PATH_LOOKUP_CHUNK = 500


@dataclass(frozen=True)
class LiveSessionRow:
    """One live-set member and whatever the index knows about the same path.

    Every indexed field is optional because the daemon promotes a transcript on
    its first write, routinely before any index pass has created a row for it.
    A member with no session id is live and unindexed, not absent.
    """

    path: str
    mtime: float
    last_event_at: float | None
    session_id: str | None = None
    source: str | None = None
    source_session_id: str | None = None
    host: str | None = None
    indexed_mtime: float | None = None
    indexed_size: int | None = None
    indexed_at: datetime | None = None


@dataclass(frozen=True)
class LiveSessionsView:
    """The live set as served over RPC.

    `watching` is False when the daemon runs without the live watcher, and that
    is a different answer from an empty `sessions`. Without a live set nothing
    can be called active, so a caller that read "none active" from a daemon
    which never watched would be reading a fact recall does not have.
    """

    watching: bool
    idle_threshold_seconds: float
    sessions: tuple[LiveSessionRow, ...] = ()


@dataclass(frozen=True)
class LiveCoverage:
    """What a metadata-filtered live roster could and could not inspect."""

    watched_paths: int
    indexed_paths: int
    filtered_unknown_paths: tuple[str, ...]

    observed_paths: int = 0
    unindexed_paths: int = 0
    unknown_count: int = 0
    catalog_scan_complete: bool = False
    complete: bool = False


@dataclass(frozen=True)
class LivePage:
    """A bounded, opaque page over an already deterministically ordered roster."""

    sessions: tuple[LiveSession, ...]
    next_cursor: str | None


_LIVE_PAGE_VERSION = "live-v1"


def paginate_live_sessions(
    sessions: Sequence[LiveSession], *, limit: int, cursor: str | None = None
) -> LivePage:
    """Page a roster without letting watcher subscription capacity define it."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    offset = decode_live_page_cursor(cursor)
    page = tuple(sessions[offset : offset + limit])
    next_offset = offset + len(page)
    next_cursor = None
    if next_offset < len(sessions):
        next_cursor = encode_live_page_cursor(next_offset)
    return LivePage(sessions=page, next_cursor=next_cursor)


def decode_live_page_cursor(cursor: str | None) -> int:
    """Decode a roster offset without accepting arbitrary pagination state."""
    if cursor is None:
        return 0
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(cursor + padding).decode("utf-8")
        version, offset_text = raw.split(":", maxsplit=1)
        offset = int(offset_text)
    except (ValueError, UnicodeDecodeError, binascii.Error) as err:
        raise ValueError("malformed live page cursor") from err
    if version != _LIVE_PAGE_VERSION or offset < 0:
        raise ValueError("malformed live page cursor")
    return offset


def encode_live_page_cursor(offset: int) -> str:
    if offset < 0:
        raise ValueError("live page offset must be non-negative")
    return (
        base64.urlsafe_b64encode(f"{_LIVE_PAGE_VERSION}:{offset}".encode())
        .decode("ascii")
        .rstrip("=")
    )


def live_metadata_coverage(
    conn: duckdb.DuckDBPyConnection,
    *,
    query: tuple[str, dict[str, object]],
    watched_count: int,
    catalog_scan_complete: bool,
) -> LiveCoverage:
    """Count the full SQL roster, retaining at most sixteen unknown paths."""
    sql, params = query
    counts = conn.execute(
        sql
        + """
        SELECT COUNT(*), COUNT(*) FILTER (WHERE id IS NOT NULL),
               COUNT(*) FILTER (WHERE id IS NULL),
               COUNT(*) FILTER (WHERE possible_match AND NOT matches)
        FROM classified
        """,
        params,
    ).fetchone()
    assert counts is not None
    sample = conn.execute(
        sql
        + """
        SELECT source_path FROM classified WHERE possible_match AND NOT matches
        ORDER BY last_activity_at DESC NULLS LAST, source_path ASC LIMIT 16
        """,
        params,
    ).fetchall()
    return LiveCoverage(
        watched_paths=watched_count,
        observed_paths=int(counts[0]),
        indexed_paths=int(counts[1]),
        unindexed_paths=int(counts[2]),
        filtered_unknown_paths=tuple(str(row[0]) for row in sample),
        unknown_count=int(counts[3]),
        catalog_scan_complete=catalog_scan_complete,
        complete=catalog_scan_complete and counts[3] == 0,
    )


@dataclass(frozen=True)
class _IndexedSession:
    """What the index knows about one watched path, read at the DB boundary."""

    session_id: str
    source: str
    source_session_id: str | None
    host: str | None
    file_mtime: float | None
    file_size: int | None
    indexed_at: datetime | None


_INDEXED_SELECT = """
    SELECT s.source_path, s.id, s.source, s.source_session_id,
           ss.host, ss.file_mtime, ss.file_size, ss.indexed_at
    FROM sessions s
    LEFT JOIN session_state ss ON ss.session_id = s.id
    WHERE s.source_path IN ({placeholders})
"""


def live_session_rows(
    members: Sequence[LiveMember],
    *,
    conn: duckdb.DuckDBPyConnection,
) -> tuple[LiveSessionRow, ...]:
    """Join live-set members to their indexed sessions, preserving member order."""
    if not members:
        return ()
    paths = [str(member.path) for member in members]
    indexed = _fetch_indexed_by_path(conn, paths)
    return tuple(_row_from(member, indexed.get(str(member.path))) for member in members)


def _fetch_indexed_by_path(
    conn: duckdb.DuckDBPyConnection, paths: Sequence[str]
) -> dict[str, _IndexedSession]:
    indexed: dict[str, _IndexedSession] = {}
    for start in range(0, len(paths), _PATH_LOOKUP_CHUNK):
        chunk = paths[start : start + _PATH_LOOKUP_CHUNK]
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            _INDEXED_SELECT.format(placeholders=placeholders), list(chunk)
        ).fetchall()
        indexed.update({str(row[0]): _indexed_from_row(row) for row in rows})
    return indexed


def _indexed_from_row(row: tuple[Any, ...]) -> _IndexedSession:
    _, session_id, source, source_session_id, host, file_mtime, file_size, indexed_at = row
    return _IndexedSession(
        session_id=str(session_id),
        source=str(source),
        source_session_id=None if source_session_id is None else str(source_session_id),
        host=None if host is None else str(host),
        file_mtime=None if file_mtime is None else float(file_mtime),
        file_size=None if file_size is None else int(file_size),
        indexed_at=indexed_at,
    )


def _row_from(member: LiveMember, indexed: _IndexedSession | None) -> LiveSessionRow:
    if indexed is None:
        return LiveSessionRow(
            path=str(member.path),
            mtime=member.mtime,
            last_event_at=member.last_event_at,
        )
    return LiveSessionRow(
        path=str(member.path),
        mtime=member.mtime,
        last_event_at=member.last_event_at,
        session_id=indexed.session_id,
        source=indexed.source,
        source_session_id=indexed.source_session_id,
        host=indexed.host,
        indexed_mtime=indexed.file_mtime,
        indexed_size=indexed.file_size,
        indexed_at=indexed.indexed_at,
    )


_HIGH_WATER_SELECT = """
    SELECT s.id, (SELECT MAX(m.idx) FROM messages m WHERE m.session_id = s.id)
    FROM sessions s
    WHERE s.source_path = ?
"""


def session_indexed_event(path: str, *, conn: duckdb.DuckDBPyConnection) -> SessionIndexed:
    """Read the high-water message idx for a just-indexed transcript (REQ-LIVE-011).

    Called after the write lock is released, so what it reads is exactly what a
    woken subscriber will read. The lookup goes through `live_path_key` because
    `sessions.source_path` holds the resolved spelling while a caller or the
    watcher may hold a symlinked one. A path with no row yet — a brand-new or
    unparseable transcript — yields an event with both fields None rather than
    no event, because the caller still needs to learn that the pass finished.
    """
    key = live_path_key(path)
    row = conn.execute(_HIGH_WATER_SELECT, [key]).fetchone()
    if row is None:
        return SessionIndexed(path=key, session_id=None, high_water_idx=None)
    session_id, high_water_idx = row
    return SessionIndexed(
        path=key,
        session_id=str(session_id),
        high_water_idx=None if high_water_idx is None else int(high_water_idx),
    )


class Liveness(StrEnum):
    """What recall can say about a session right now (REQ-LIVE-001)."""

    ACTIVE = "active"
    IDLE = "idle"
    ENDED = "ended"
    UNKNOWN = "unknown"


class TurnPhase(StrEnum):
    """Where the conversation stands, derived from the indexed tail (REQ-LIVE-005)."""

    WORKING = "working"
    AWAITING_INPUT = "awaiting_input"
    SUBAGENTS_RUNNING = "subagents_running"
    ENDED = "ended"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Freshness:
    """How far the index trails the file, measured now (REQ-LIVE-003).

    Every field is optional because either side can be absent: a transcript the
    harness deleted has no `file_*`, and one no pass has reached has no
    `indexed_*`. `lag_*` stays None in both cases rather than defaulting to
    zero, which a caller would read as "current".
    """

    file_mtime: float | None
    file_size: int | None
    indexed_mtime: float | None
    indexed_size: int | None
    lag_seconds: float | None
    lag_bytes: int | None
    current: bool
    # `current` is only meaningful when the durable catalog has validated the
    # same source identity and complete committed boundary.  A direct read of a
    # legacy database has no such proof, even if its byte count happens to
    # match the file today.
    validated: bool
    generation: int | None
    content_epoch: int | None
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogProgress:
    """Durable observation used to decide whether a source is fresh."""

    desired_generation: int
    committed_generation: int
    committed_offset: int
    content_epoch: int
    signature_dev: int | None
    signature_inode: int | None
    signature_ctime_ns: int
    signature_mtime_ns: int
    signature_size: int
    parser_revision: str
    source: str
    sidecar_signature: str
    missing: bool = False
    last_error: str | None = None


_CATALOG_PROGRESS_SELECT = """
    SELECT desired_generation, committed_generation, committed_offset, content_epoch,
           dev, inode, ctime_ns, mtime_ns, size, parser_revision, missing, last_error,
           source, sidecar_signature
    FROM source_files
    WHERE source_path = ? AND session_id = ?
"""


def catalog_progress_for_session(
    conn: duckdb.DuckDBPyConnection, *, session_id: str, source_path: str
) -> CatalogProgress | None:
    """Read the committed catalog boundary for one indexed session.

    The session binding makes duplicate paths under different configured roots
    unambiguous.  Missing legacy catalog rows remain observable as a limitation
    rather than being silently promoted to fresh.
    """
    row = conn.execute(_CATALOG_PROGRESS_SELECT, [source_path, session_id]).fetchone()
    if row is None:
        return None
    return CatalogProgress(
        desired_generation=int(row[0]),
        committed_generation=int(row[1]),
        committed_offset=int(row[2]),
        content_epoch=int(row[3]),
        signature_dev=None if row[4] is None else int(row[4]),
        signature_inode=None if row[5] is None else int(row[5]),
        signature_ctime_ns=int(row[6]),
        signature_mtime_ns=int(row[7]),
        signature_size=int(row[8]),
        parser_revision=str(row[9]),
        missing=bool(row[10]),
        last_error=None if row[11] is None else str(row[11]),
        source=str(row[12]),
        sidecar_signature=str(row[13]),
    )


@dataclass(frozen=True)
class RunningTool:
    """The tool_use recall has seen no result for."""

    name: str
    summary: str
    started_at: datetime | None


@dataclass(frozen=True)
class OpenToolCall:
    """A tool call with no paired `tool_results` row, with its message's position.

    `ToolCall.idx` is positional *within its message*, so it cannot order calls
    across messages on its own; `message_idx` supplies the outer key.
    """

    tool_call: ToolCall
    message_idx: int
    started_at: datetime | None


@dataclass(frozen=True)
class TurnState:
    """Turn state is computed from the tail at read time and never stored."""

    state: TurnPhase
    last_user_at: datetime | None = None
    last_assistant_at: datetime | None = None
    last_user_text: str | None = None
    last_assistant_text: str | None = None
    running_tool: RunningTool | None = None
    subagents_active: int = 0
    stop_reason: str | None = None


def derive_liveness(
    *,
    watched: bool,
    session_ended: bool,
    marked_pid_alive: bool | None,
    last_activity_at: datetime | None,
    now: datetime,
    idle_window_seconds: float,
) -> Liveness:
    """Rank the four answers REQ-LIVE-001 allows, by how recently each was observed.

    A marked pid probed dead wins outright. It is the only input re-observed on
    every read: `kill(pid, 0)` answers about *now*, whereas live-set membership
    is an inference from a write that may be up to `live_idle_threshold` (300 s)
    old. Ranking membership first, as this did until the U18 bug bash, left a
    SIGKILLed agent reading `active` for that whole window — the wrong-liveness
    answer this requirement exists to forbid. The stale-mark case it was
    protecting against is covered by the hook: a harness resuming into the same
    transcript re-marks at `SessionStart`, before it writes.

    `watched` still beats a harness end *marker*, which is a record of the past
    rather than a fresh observation: a restarted harness is live again whatever
    an earlier marker said.

    A window miss resolves `unknown`, never `ended`: nothing observed an end,
    and inventing one is the guess this requirement exists to forbid.
    """
    if marked_pid_alive is False:
        return Liveness.ENDED
    if watched:
        return Liveness.ACTIVE
    if session_ended:
        return Liveness.ENDED
    if last_activity_at is None:
        return Liveness.UNKNOWN
    if (now - last_activity_at).total_seconds() <= idle_window_seconds:
        return Liveness.IDLE
    return Liveness.UNKNOWN


def derive_freshness(
    path: str,
    *,
    indexed_mtime: float | None,
    indexed_size: int | None,
    catalog: CatalogProgress | None = None,
) -> Freshness:
    """Compare the file as it is right now against a validated catalog boundary.

    The `stat` happens here rather than being carried from the live set, so the
    answer is as of the read and not as of the last watch event. A missing file
    is reported as missing rather than raised on: the harness may have rotated
    or deleted a transcript recall still has rows for.
    """
    try:
        stat = os.stat(path)
    except OSError:
        file_mtime: float | None = None
        file_size: int | None = None
    else:
        file_mtime = stat.st_mtime
        file_size = stat.st_size

    known_both_sizes = file_size is not None and indexed_size is not None
    lag_bytes = file_size - indexed_size if known_both_sizes else None
    known_both_mtimes = file_mtime is not None and indexed_mtime is not None
    lag_seconds = file_mtime - indexed_mtime if known_both_mtimes else None
    limitations: list[str] = []
    validated = False
    generation: int | None = None
    content_epoch: int | None = None
    if catalog is None:
        if indexed_mtime is None and indexed_size is None:
            limitations.append("not_yet_indexed")
        else:
            limitations.append("catalog_progress_unavailable")
    elif file_mtime is None or file_size is None:
        limitations.append("source_unreadable")
        generation = catalog.committed_generation
        content_epoch = catalog.content_epoch
        if catalog.committed_generation == 0:
            limitations.append("not_yet_indexed")
    else:
        generation = catalog.committed_generation
        content_epoch = catalog.content_epoch
        if catalog.committed_generation == 0:
            limitations.append("not_yet_indexed")
        # ctime and identity catch a same-size/same-mtime replacement.  The
        # parser revision is part of the catalog generation boundary: a stale
        # parser observation can never be current merely because the file has
        # not changed on disk.
        source_signature_matches = (
            stat.st_ctime_ns == catalog.signature_ctime_ns
            and stat.st_mtime_ns == catalog.signature_mtime_ns
            and stat.st_size == catalog.signature_size
            and (catalog.signature_dev is None or stat.st_dev == catalog.signature_dev)
            and (catalog.signature_inode is None or stat.st_ino == catalog.signature_inode)
        )
        from pathlib import Path

        from recall.core.types import Source
        from recall.parsers import get_parser
        from recall.services.coordinator import parser_revision
        from recall.services.reconciler import capture_sidecars

        inputs_match = False
        try:
            parser = get_parser(Source(catalog.source))
            revision_matches = parser_revision(type(parser)) == catalog.parser_revision
            _, sidecar_signature = capture_sidecars(parser, Path(path))
            sidecars_match = sidecar_signature == catalog.sidecar_signature
        except (OSError, ValueError):
            limitations.append("parser_inputs_unavailable")
        else:
            if not revision_matches:
                limitations.append("parser_revision_mismatch")
            if not sidecars_match:
                limitations.append("sidecar_signature_mismatch")
            inputs_match = revision_matches and sidecars_match
        generation_committed = catalog.desired_generation == catalog.committed_generation
        complete_boundary = catalog.committed_offset == stat.st_size
        source_usable = not catalog.missing and catalog.last_error is None
        if not source_signature_matches:
            limitations.append("catalog_signature_mismatch")
        if not generation_committed:
            limitations.append("catalog_generation_pending")
        if not complete_boundary:
            limitations.append("catalog_boundary_incomplete")
        if catalog.missing:
            limitations.append("catalog_source_missing")
        if catalog.last_error is not None:
            limitations.append("catalog_source_error")
        validated = (
            source_signature_matches
            and inputs_match
            and generation_committed
            and complete_boundary
            and source_usable
            and catalog.committed_generation > 0
        )

    return Freshness(
        file_mtime=file_mtime,
        file_size=file_size,
        indexed_mtime=indexed_mtime,
        indexed_size=indexed_size,
        lag_seconds=lag_seconds,
        lag_bytes=lag_bytes,
        current=validated,
        validated=validated,
        generation=generation,
        content_epoch=content_epoch,
        limitations=tuple(limitations),
    )


def live_fresh_catch_up_paths(sessions: Sequence[LiveSession]) -> tuple[str, ...]:
    """Already-indexed behind rows `live --fresh` may catch up now (REQ-LIVE-003).

    First index of a never-committed source stays on the fair coordinator.
    """
    return tuple(
        row.path
        for row in sessions
        if not row.freshness.current
        and row.freshness.generation is not None
        and row.freshness.generation > 0
        and "not_yet_indexed" not in row.freshness.limitations
    )


def derive_turn_state(
    messages: Sequence[Message],
    *,
    open_tool_calls: Sequence[OpenToolCall],
    last_stop_reason: str | None,
    last_stop_ends_turn: bool,
    session_ended: bool,
    text_budget: int = DEFAULT_TEXT_BUDGET,
) -> TurnState:
    """Read the turn off the tail, generic over `Message`/`ToolCall` (REQ-LIVE-005).

    Precedence is most-terminal to least: an observed end marker settles the
    session; a subagent that is itself mid-tool is the specific reading of the
    parent's open `Agent` call, so it outranks the generic `working`; any other
    open tool call is `working`; a last stop with `ends_turn=False` and no
    later user record is still `working` (the harness left the turn open,
    even after the tool result landed); an end-of-turn stop with no later
    user record is `awaiting_input`. Anything else — including a harness
    that emits no markers at all — is `unknown`, never a guess.
    """
    if text_budget <= 0:
        raise ValueError(f"text_budget must be positive, got {text_budget}")

    last_user = _last_message(messages, Role.USER)
    last_assistant = _last_message(messages, Role.ASSISTANT)
    running_tool = _running_tool(open_tool_calls, text_budget=text_budget)
    subagents_active = _subagents_active(open_tool_calls)
    facts = {
        "last_user_at": last_user.timestamp if last_user else None,
        "last_assistant_at": last_assistant.timestamp if last_assistant else None,
        "last_user_text": _truncate(last_user.content if last_user else None, text_budget),
        "last_assistant_text": _truncate(
            last_assistant.content if last_assistant else None, text_budget
        ),
        "running_tool": running_tool,
        "subagents_active": subagents_active,
        "stop_reason": last_stop_reason,
    }

    if session_ended:
        return TurnState(state=TurnPhase.ENDED, **facts)
    if subagents_active:
        return TurnState(state=TurnPhase.SUBAGENTS_RUNNING, **facts)
    if running_tool is not None:
        return TurnState(state=TurnPhase.WORKING, **facts)
    if last_stop_ends_turn and not _user_spoke_last(messages):
        return TurnState(state=TurnPhase.AWAITING_INPUT, **facts)
    if last_stop_reason is not None and not last_stop_ends_turn and not _user_spoke_last(messages):
        return TurnState(state=TurnPhase.WORKING, **facts)
    return TurnState(state=TurnPhase.UNKNOWN, **facts)


def _last_message(messages: Sequence[Message], role: Role) -> Message | None:
    """Newest message in the given role, by `idx` — the order the tail is read in."""
    candidates = [message for message in messages if message.role is role and not message.agent_id]
    if not candidates:
        return None
    return max(candidates, key=lambda message: message.idx)


def _user_spoke_last(messages: Sequence[Message]) -> bool:
    main_thread = [message for message in messages if not message.agent_id]
    if not main_thread:
        return False
    return max(main_thread, key=lambda message: message.idx).role is Role.USER


def _running_tool(
    open_tool_calls: Sequence[OpenToolCall], *, text_budget: int
) -> RunningTool | None:
    """The newest unanswered call — with several open, the last one is the live one."""
    if not open_tool_calls:
        return None
    newest = max(open_tool_calls, key=lambda call: (call.message_idx, call.tool_call.idx))
    return RunningTool(
        name=newest.tool_call.tool_name,
        summary=_tool_summary(newest.tool_call, text_budget),
        started_at=newest.started_at,
    )


def _tool_summary(tool_call: ToolCall, text_budget: int) -> str:
    if tool_call.bash_command:
        return _truncate(tool_call.bash_command, text_budget) or ""
    if not tool_call.tool_input:
        return ""
    return (
        _truncate(
            ", ".join(f"{key}={value}" for key, value in tool_call.tool_input.items()), text_budget
        )
        or ""
    )


def _subagents_active(open_tool_calls: Sequence[OpenToolCall]) -> int:
    """Distinct subagents that are themselves waiting on a tool right now.

    Counted from *open calls*, not from messages carrying an `agent_id`: a
    finished subagent's messages stay in the transcript forever, so their
    presence can never distinguish running from finished. An unanswered
    tool_use attributed to a subagent is direct evidence that one is live.
    """
    return len({call.tool_call.agent_id for call in open_tool_calls if call.tool_call.agent_id})


def _truncate(text: str | None, budget: int) -> str | None:
    if text is None:
        return None
    return text[:budget]


# The stop marker in force is the newest one that still has a message under it.
# A rewritten, shorter transcript leaves markers above its surviving messages;
# reading past `MAX(idx)` would hold a turn open that no longer exists.
# Join a bounded tail relation: a scalar MAX subquery triggered a DuckDB internal
# error for message-free sessions during live startup. An empty tail has no match.
_LAST_STOP_MARKER_SELECT = """
    SELECT sm.message_idx, sm.reason, sm.ends_turn
    FROM session_stop_markers sm
    JOIN (
        SELECT idx FROM messages WHERE session_id = ? ORDER BY idx DESC LIMIT 1
    ) tail ON sm.message_idx <= tail.idx
    WHERE sm.session_id = ?
    ORDER BY sm.message_idx DESC
    LIMIT 1
"""

# A call counts as open only when recall holds its harness id: without the
# mapping there is no key a result could ever arrive on, so an unanswered call
# is `unknown`, not `working` (REQ-LIVE-006).
_OPEN_TOOL_CALLS_SELECT = f"""
    SELECT {tool_call_columns("tc")}, m.idx, ms.timestamp
    FROM tool_calls tc
    JOIN tool_use_ids tu ON tu.tool_call_id = tc.id
    JOIN messages m ON m.id = tc.message_id
    LEFT JOIN message_state ms ON ms.message_id = m.id
    LEFT JOIN tool_results tr ON tr.tool_call_id = tc.id
"""


def fetch_last_stop_marker(conn: duckdb.DuckDBPyConnection, session_id: str) -> StopMarker | None:
    """The stop marker governing the end of the session's indexed tail."""
    row = conn.execute(_LAST_STOP_MARKER_SELECT, [session_id, session_id]).fetchone()
    if row is None:
        return None
    message_idx, reason, ends_turn = row
    return StopMarker(idx=int(message_idx), reason=str(reason), ends_turn=bool(ends_turn))


def fetch_open_tool_calls(
    conn: duckdb.DuckDBPyConnection, session_id: str, *, limit: int
) -> list[OpenToolCall]:
    """Tool calls this session is still waiting on, newest first (REQ-LIVE-005).

    `limit` is required rather than defaulted: a session that fans out to a
    hundred subagents would otherwise make an unbounded read on the hot `live`
    path, and only the caller knows how many it can use.
    """
    return _fetch_open_tool_call_groups(conn, [session_id], limit=limit).get(session_id, [])


def _fetch_open_tool_call_groups(
    conn: duckdb.DuckDBPyConnection, session_ids: Sequence[str], *, limit: int
) -> dict[str, list[OpenToolCall]]:
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    groups: dict[str, list[OpenToolCall]] = {}
    for start in range(0, len(session_ids), _PATH_LOOKUP_CHUNK):
        chunk = session_ids[start : start + _PATH_LOOKUP_CHUNK]
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"""{_OPEN_TOOL_CALLS_SELECT}
            WHERE tc.session_id IN ({placeholders}) AND tr.tool_call_id IS NULL
            QUALIFY row_number() OVER (
                PARTITION BY tc.session_id ORDER BY m.idx DESC, tc.idx DESC
            ) <= ?
            ORDER BY tc.session_id, m.idx DESC, tc.idx DESC
            """,
            [*chunk, limit],
        ).fetchall()
        for row in rows:
            call = _open_tool_call_from_row(row)
            groups.setdefault(call.tool_call.session_id, []).append(call)
    return groups


def _open_tool_call_from_row(row: tuple[Any, ...]) -> OpenToolCall:
    return OpenToolCall(
        tool_call=tool_call_from_row(row[:15]),
        message_idx=int(row[15]),
        started_at=row[16],
    )


# Cursors are opaque to callers but cheap for the daemon to check: the version
# tag makes a format change detectable instead of silently misread, and the
# session id inside it is what makes a cursor from another session a
# validation error rather than a wrong answer (REQ-LIVE-004).
_CURSOR_VERSION = "v2"


@dataclass(frozen=True)
class Cursor:
    """A versioned continuation position bound to derived content.

    ``content_epoch=None`` represents a legacy v1 cursor.  It is deliberately
    distinguishable from epoch zero so the RPC boundary can reset it safely
    after a rewrite without guessing whether indexes still mean the same text.
    """

    session_id: str
    last_idx: int
    content_epoch: int | None

    # Keep source-compatible unpacking for callers moving from v1 while the
    # explicit attribute carries the new safety boundary.  This is intentionally
    # a two-item sequence: a third positional item would let old callers lose
    # the epoch through broad exception handling.
    def __iter__(self):
        yield self.session_id
        yield self.last_idx

    @overload
    def __getitem__(self, index: Literal[0]) -> str: ...

    @overload
    def __getitem__(self, index: Literal[1]) -> int: ...

    def __getitem__(self, index: Literal[0, 1]) -> str | int:
        return (self.session_id, self.last_idx)[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple):
            return (self.session_id, self.last_idx) == other
        if not isinstance(other, Cursor):
            return NotImplemented
        return (
            self.session_id,
            self.last_idx,
            self.content_epoch,
        ) == (other.session_id, other.last_idx, other.content_epoch)


def encode_cursor(session_id: str, last_idx: int, content_epoch: int = 0) -> str:
    """Opaque cursor naming a session, message position, and content epoch."""
    if content_epoch < 0:
        raise ValueError("content epoch must be non-negative")
    raw = f"{_CURSOR_VERSION}:{content_epoch}:{session_id}:{last_idx}".encode()
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> Cursor:
    """Read a cursor back, raising ``ValueError`` on anything recall did not write."""
    padding = "=" * (-len(cursor) % 4)
    try:
        raw = base64.urlsafe_b64decode(cursor + padding).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError) as err:
        raise ValueError(f"malformed cursor: {cursor}") from err
    version, _, rest = raw.partition(":")
    if version == "v1":
        session_id, _, idx_text = rest.rpartition(":")
        epoch: int | None = None
    elif version == _CURSOR_VERSION:
        epoch_text, _, session_and_idx = rest.partition(":")
        session_id, _, idx_text = session_and_idx.rpartition(":")
        try:
            epoch = int(epoch_text)
        except ValueError as err:
            raise ValueError(f"malformed cursor: {cursor}") from err
        if epoch < 0:
            raise ValueError(f"malformed cursor: {cursor}")
    else:
        session_id = ""
        idx_text = ""
        epoch = None
    if not session_id or not idx_text:
        raise ValueError(f"malformed cursor: {cursor}")
    try:
        return Cursor(session_id=session_id, last_idx=int(idx_text), content_epoch=epoch)
    except ValueError as err:
        raise ValueError(f"malformed cursor: {cursor}") from err


@dataclass(frozen=True)
class LiveSession:
    """One row of `recall live` (REQ-LIVE-002).

    Every indexed field is optional for the same reason `LiveSessionRow`'s are:
    the daemon promotes a transcript on its first write, so the most
    interesting row in the list is routinely the one with no session id yet.
    `path` and `liveness` are the two facts that always exist.
    """

    path: str
    liveness: Liveness
    freshness: Freshness
    turn: TurnState
    id: str | None = None
    source: str | None = None
    source_session_id: str | None = None
    host: str | None = None
    cwd: str | None = None
    git_repo: str | None = None
    git_branch: str | None = None
    model: str | None = None
    last_activity_at: datetime | None = None
    cursor: str | None = None
    # The writer pid a harness hook marked (REQ-LIVE-008); feeds `ended` detection.
    writer_pid: int | None = None


# Messages read per session to derive turn state. The derivation only needs the
# last user and assistant records; eight leaves room for interleaved subagent
# and system rows without turning a fleet listing into a transcript read.
_TURN_TAIL_MESSAGES = 8

# Open calls read per session. A session fanning out to more subagents than
# this is already `subagents_running`; the count is what the row reports, and
# reading every one of them would make the hot path unbounded.
_OPEN_TOOL_CALL_LIMIT = 32

# The mark joins on the session's own host, so a row attributed to another
# machine can only match a mark stamped for that machine -- and `live_view`
# then declines to probe its pid. Both halves are needed: the join keeps the
# rows apart, the probe guard keeps a copied database from reading another
# host's pid numbers as local ones.
_CANDIDATE_SELECT = """
    SELECT s.id, s.source, s.source_path, s.source_session_id,
           ss.cwd, ss.git_repo, ss.git_branch, ss.model, ss.host,
           ss.file_mtime, ss.file_size,
           GREATEST(ss.ended_at, CAST(to_timestamp(ss.file_mtime) AS TIMESTAMP))
             AS last_activity_at,
           lm.pid
    FROM sessions s
    JOIN session_state ss ON ss.session_id = s.id
    LEFT JOIN live_marks lm
      ON lm.source = s.source
     AND lm.source_session_id = s.source_session_id
     AND lm.host = ss.host
    WHERE {where}
    ORDER BY last_activity_at DESC, s.source_path ASC
    LIMIT {limit}
    OFFSET {offset}
"""


@dataclass(frozen=True)
class _Candidate:
    """An indexed session that may belong in the live view."""

    id: str | None
    source: str | None
    source_path: str
    source_session_id: str | None
    cwd: str | None
    git_repo: str | None
    git_branch: str | None
    model: str | None
    host: str | None
    file_mtime: float
    file_size: int
    last_activity_at: datetime | None
    marked_pid: int | None = None
    observed_write: bool = False


def live_view(
    *,
    conn: duckdb.DuckDBPyConnection,
    watched_paths: Sequence[str],
    include_idle: bool,
    now: datetime,
    idle_window_seconds: float,
    limit: int,
    local_host: str,
    source: str | None = None,
    project: str | None = None,
    host: str | None = None,
) -> list[LiveSession]:
    """Join the daemon's live set to the index and derive a row for each session.

    Without `include_idle` only watched paths whose derived liveness is active
    are returned: a marked process may have exited before its path leaves the
    live set. `include_idle` also returns those ended rows and widens membership
    to indexed sessions whose last activity falls inside the idle window. The
    metadata filters compose on both halves.

    `local_host` is this machine's label, the one the indexer writes into
    `session_state.host`. It is passed in rather than resolved here so the pid
    probe has an explicit subject: a mark stamped on another machine names a
    pid number that means nothing locally, and probing it would report a
    running agent as `ended`.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    watched = {live_path_key(path): path for path in watched_paths}
    candidates = _fetch_candidates(
        conn,
        watched_keys=list(watched),
        include_idle=include_idle,
        cutoff=now - timedelta(seconds=idle_window_seconds),
        # A marked pid may have exited while its path remains watched. Inspect
        # the bounded watched set before limiting active rows, or ended rows
        # can consume the page and hide less-recent active sessions.
        limit=limit if include_idle else len(watched),
        source=source,
        project=project,
        host=host,
    )

    session_ids = [candidate.id for candidate in candidates if candidate.id]
    tails = _fetch_turn_tails(conn, session_ids)
    open_calls = _fetch_open_tool_call_groups(conn, session_ids, limit=_OPEN_TOOL_CALL_LIMIT)
    rows = [
        _live_session_from_candidate(
            candidate,
            conn=conn,
            messages=tails.get(candidate.id, ()) if candidate.id else (),
            open_tool_calls=open_calls.get(candidate.id, ()) if candidate.id else (),
            watched=live_path_key(candidate.source_path) in watched,
            now=now,
            idle_window_seconds=idle_window_seconds,
            local_host=local_host,
        )
        for candidate in candidates
    ]
    if source is None and not project and host is None:
        # Metadata filters require an indexed identity. Without filters, a
        # candidate can still be missing because of LIMIT, so establish index
        # membership separately before calling a watched path unindexed.
        candidate_keys = {live_path_key(candidate.source_path) for candidate in candidates}
        missing_keys = [key for key in watched if key not in candidate_keys]
        indexed_keys = _fetch_indexed_by_path(conn, missing_keys)
        rows.extend(
            _live_session_unindexed(watched[key], now=now)
            for key in missing_keys
            if key not in indexed_keys
        )
    if not include_idle:
        rows = [row for row in rows if row.liveness is Liveness.ACTIVE]
    rows.sort(key=_activity_sort_key, reverse=True)
    return rows[:limit]


def live_roster_query(
    *,
    watched_paths: Sequence[str],
    include_idle: bool,
    now: datetime,
    idle_window_seconds: float,
    active_window_seconds: float = 300.0,
    source: str | None = None,
    project: str | None = None,
    host: str | None = None,
) -> tuple[str, dict[str, object]]:
    """Compose indexed and observed paths before filtering or limiting any page.

    Poll observations and watcher events have the same activity window. Kernel
    subscription capacity never defines the roster. Unknown metadata is retained
    for coverage accounting but cannot satisfy a metadata filter.
    """
    params: dict[str, object] = {
        "watched": sorted({live_path_key(path) for path in watched_paths}),
        "active_cutoff": now.timestamp() - active_window_seconds,
        "idle_cutoff": now - timedelta(seconds=idle_window_seconds),
        "include_idle": include_idle,
        "source": source,
        "project": f"%{project}%" if project else None,
        "host": host,
    }
    sql = """
        WITH watched AS (SELECT unnest($watched::VARCHAR[]) AS source_path),
        roster AS (
            SELECT s.id, s.source, s.source_path, s.source_session_id,
                   ss.cwd, ss.git_repo, ss.git_branch, ss.model, ss.host,
                   ss.file_mtime, ss.file_size,
                   GREATEST(ss.ended_at, CAST(to_timestamp(ss.file_mtime) AS TIMESTAMP),
                            CAST(to_timestamp(sf.mtime_ns / 1e9) AS TIMESTAMP))
                       AS last_activity_at,
                   lm.pid,
                   (s.source_path IN (SELECT source_path FROM watched)
                    OR COALESCE(NOT sf.missing AND sf.mtime_ns / 1e9 >= $active_cutoff, FALSE))
                       AS observed_write
            FROM sessions s JOIN session_state ss ON ss.session_id = s.id
            LEFT JOIN source_files sf ON sf.source_path = s.source_path AND sf.source = s.source
            LEFT JOIN live_marks lm ON lm.source = s.source
                AND lm.source_session_id = s.source_session_id AND lm.host = ss.host
            UNION ALL
            SELECT NULL, sf.source, sf.source_path, NULL, NULL, NULL, NULL, NULL, NULL,
                   sf.mtime_ns / 1e9, sf.size,
                   CAST(to_timestamp(sf.mtime_ns / 1e9) AS TIMESTAMP), NULL,
                   (sf.source_path IN (SELECT source_path FROM watched)
                    OR sf.mtime_ns / 1e9 >= $active_cutoff)
            FROM source_files sf WHERE NOT sf.missing
                AND NOT EXISTS (SELECT 1 FROM sessions s WHERE s.source_path = sf.source_path)
            UNION ALL
            SELECT NULL, NULL, w.source_path, NULL, NULL, NULL, NULL, NULL, NULL,
                   0.0, 0, NULL, NULL, TRUE
            FROM watched w
            WHERE NOT EXISTS (SELECT 1 FROM sessions s WHERE s.source_path = w.source_path)
              AND NOT EXISTS (SELECT 1 FROM source_files sf WHERE sf.source_path = w.source_path
                              AND NOT sf.missing)
        ), members AS (
            SELECT * FROM roster
            WHERE observed_write OR ($include_idle AND last_activity_at >= $idle_cutoff)
        ), classified AS (
            SELECT *,
                COALESCE(($source::VARCHAR IS NULL OR source = $source)
                AND ($project::VARCHAR IS NULL OR git_repo ILIKE $project)
                AND ($host::VARCHAR IS NULL OR host = $host), FALSE) AS matches,
                COALESCE(($source::VARCHAR IS NULL OR source = $source
                          OR (id IS NULL AND source IS NULL))
                AND ($project::VARCHAR IS NULL OR git_repo ILIKE $project
                     OR (id IS NULL AND git_repo IS NULL))
                AND ($host::VARCHAR IS NULL OR host = $host
                     OR (id IS NULL AND host IS NULL)), FALSE) AS possible_match
            FROM members
        )
    """
    return sql, params


def live_view_page(
    *,
    conn: duckdb.DuckDBPyConnection,
    watched_paths: Sequence[str],
    include_idle: bool,
    now: datetime,
    idle_window_seconds: float,
    limit: int,
    cursor: str | None,
    local_host: str,
    active_window_seconds: float = 300.0,
    source: str | None = None,
    project: str | None = None,
    host: str | None = None,
) -> LivePage:
    """Inspect at most limit candidates; continuation advances past inspected rows.

    A dead process can remove an active candidate after the SQL page was chosen.
    The continuation still covers subsequent rows, including after an empty page.
    """
    if not 1 <= limit <= 256:
        raise ValueError("limit must be 1..256")
    offset = decode_live_page_cursor(cursor)
    sql, params = live_roster_query(
        watched_paths=watched_paths,
        include_idle=include_idle,
        now=now,
        idle_window_seconds=idle_window_seconds,
        active_window_seconds=active_window_seconds,
        source=source,
        project=project,
        host=host,
    )
    raw_rows = conn.execute(
        sql
        + """
        SELECT id, source, source_path, source_session_id, cwd, git_repo, git_branch,
               model, host, file_mtime, file_size, last_activity_at, pid,
               observed_write
        FROM classified WHERE matches
        ORDER BY last_activity_at DESC NULLS LAST, source_path ASC
        LIMIT $page_limit OFFSET $page_offset
        """,
        {**params, "page_limit": limit + 1, "page_offset": offset},
    ).fetchall()
    candidates = [_candidate_from_row(row) for row in raw_rows[:limit]]
    session_ids = [candidate.id for candidate in candidates if candidate.id]
    tails = _fetch_turn_tails(conn, session_ids)
    open_calls = _fetch_open_tool_call_groups(conn, session_ids, limit=_OPEN_TOOL_CALL_LIMIT)
    rows = tuple(
        _live_session_from_candidate(
            candidate,
            conn=conn,
            messages=tails.get(candidate.id, ()) if candidate.id else (),
            open_tool_calls=open_calls.get(candidate.id, ()) if candidate.id else (),
            watched=candidate.observed_write,
            now=now,
            idle_window_seconds=idle_window_seconds,
            local_host=local_host,
        )
        for candidate in candidates
    )
    if not include_idle:
        rows = tuple(row for row in rows if row.liveness is Liveness.ACTIVE)
    next_cursor = (
        encode_live_page_cursor(offset + len(candidates)) if len(raw_rows) > limit else None
    )
    return LivePage(sessions=rows, next_cursor=next_cursor)


def _activity_sort_key(row: LiveSession) -> tuple[float, str]:
    """Newest first; a row with no timestamp sorts last but stays deterministic."""
    stamp = row.last_activity_at.timestamp() if row.last_activity_at else float("-inf")
    return (stamp, row.path)


def _fetch_candidates(
    conn: duckdb.DuckDBPyConnection,
    *,
    watched_keys: Sequence[str],
    include_idle: bool,
    cutoff: datetime,
    limit: int,
    source: str | None,
    project: str | None,
    host: str | None,
    offset: int = 0,
) -> list[_Candidate]:
    membership: list[str] = []
    params: list[object] = []
    if watched_keys:
        placeholders = ", ".join("?" for _ in watched_keys)
        membership.append(f"s.source_path IN ({placeholders})")
        params.extend(watched_keys)
    if include_idle:
        membership.append(
            "GREATEST(ss.ended_at, CAST(to_timestamp(ss.file_mtime) AS TIMESTAMP)) >= ?"
        )
        params.append(cutoff)
    if not membership:
        return []

    where = [f"({' OR '.join(membership)})"]
    if source is not None:
        where.append("s.source = ?")
        params.append(source)
    if project:
        where.append("ss.git_repo ILIKE ?")
        params.append(f"%{project}%")
    if host is not None:
        where.append("ss.host = ?")
        params.append(host)

    sql = _CANDIDATE_SELECT.format(where=" AND ".join(where), limit=int(limit), offset=int(offset))
    return [_candidate_from_row(row) for row in conn.execute(sql, params).fetchall()]


def _candidate_from_row(row: tuple[Any, ...]) -> _Candidate:
    return _Candidate(
        id=None if row[0] is None else str(row[0]),
        source=None if row[1] is None else str(row[1]),
        source_path=str(row[2]),
        source_session_id=None if row[3] is None else str(row[3]),
        cwd=row[4],
        git_repo=row[5],
        git_branch=row[6],
        model=row[7],
        host=None if row[8] is None else str(row[8]),
        file_mtime=float(row[9]),
        file_size=int(row[10]),
        last_activity_at=row[11],
        marked_pid=None if row[12] is None else int(row[12]),
        observed_write=bool(row[13]) if len(row) > 13 else False,
    )


def _live_session_from_candidate(
    candidate: _Candidate,
    *,
    conn: duckdb.DuckDBPyConnection,
    messages: Sequence[Message],
    open_tool_calls: Sequence[OpenToolCall],
    watched: bool,
    now: datetime,
    idle_window_seconds: float,
    local_host: str,
) -> LiveSession:
    if candidate.id is None:
        return LiveSession(
            path=candidate.source_path,
            source=candidate.source,
            liveness=derive_liveness(
                watched=watched,
                session_ended=False,
                marked_pid_alive=None,
                last_activity_at=candidate.last_activity_at,
                now=now,
                idle_window_seconds=idle_window_seconds,
            ),
            freshness=derive_freshness(
                candidate.source_path, indexed_mtime=None, indexed_size=None
            ),
            turn=TurnState(state=TurnPhase.UNKNOWN),
            last_activity_at=candidate.last_activity_at,
        )
    catalog = catalog_progress_for_session(
        conn, session_id=candidate.id, source_path=candidate.source_path
    )
    marker = fetch_last_stop_marker(conn, candidate.id)
    turn = derive_turn_state(
        messages,
        open_tool_calls=open_tool_calls,
        last_stop_reason=marker.reason if marker else None,
        last_stop_ends_turn=marker.ends_turn if marker else False,
        # No parser on any of the five harnesses emits a session-end marker
        # today (censused), and `ended` comes from the REQ-LIVE-008 pid stamp
        # instead. Storage lands with the first parser that emits one.
        session_ended=False,
    )
    return LiveSession(
        path=candidate.source_path,
        liveness=derive_liveness(
            watched=watched,
            session_ended=False,
            marked_pid_alive=_marked_pid_alive(candidate, local_host=local_host),
            last_activity_at=candidate.last_activity_at,
            now=now,
            idle_window_seconds=idle_window_seconds,
        ),
        freshness=derive_freshness(
            candidate.source_path,
            indexed_mtime=candidate.file_mtime,
            indexed_size=candidate.file_size,
            catalog=catalog,
        ),
        turn=turn,
        id=candidate.id,
        source=candidate.source,
        source_session_id=candidate.source_session_id,
        host=candidate.host,
        cwd=candidate.cwd,
        git_repo=candidate.git_repo,
        git_branch=candidate.git_branch,
        model=candidate.model,
        last_activity_at=candidate.last_activity_at,
        cursor=encode_cursor(
            candidate.id,
            max((message.idx for message in messages), default=-1),
            content_epoch=catalog.content_epoch if catalog is not None else 0,
        ),
        writer_pid=candidate.marked_pid,
    )


def _marked_pid_alive(candidate: _Candidate, *, local_host: str) -> bool | None:
    """Probe a marked pid only when this machine is the one that stamped it."""
    if candidate.marked_pid is None or candidate.host != local_host:
        return None
    return pid_alive(candidate.marked_pid)


def _live_session_unindexed(path: str, *, now: datetime) -> LiveSession:
    """A watched transcript with no row yet: live, and honestly empty otherwise."""
    freshness = derive_freshness(path, indexed_mtime=None, indexed_size=None)
    return LiveSession(
        path=path,
        liveness=Liveness.ACTIVE,
        freshness=freshness,
        turn=TurnState(state=TurnPhase.UNKNOWN),
        last_activity_at=(
            datetime.fromtimestamp(freshness.file_mtime) if freshness.file_mtime else None
        ),
    )


def _fetch_turn_tails(
    conn: duckdb.DuckDBPyConnection, session_ids: Sequence[str]
) -> dict[str, list[Message]]:
    """Read each page member's bounded tail without one wide join per session."""
    tails: dict[str, list[Message]] = {}
    for start in range(0, len(session_ids), _PATH_LOOKUP_CHUNK):
        chunk = session_ids[start : start + _PATH_LOOKUP_CHUNK]
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"""{MESSAGE_SELECT}
            WHERE m.session_id IN ({placeholders})
            QUALIFY row_number() OVER (PARTITION BY m.session_id ORDER BY m.idx DESC) <= ?
            ORDER BY m.session_id, m.idx
            """,
            [*chunk, _TURN_TAIL_MESSAGES],
        ).fetchall()
        for row in rows:
            message = message_from_row(row)
            tails.setdefault(message.session_id, []).append(message)
    return tails
