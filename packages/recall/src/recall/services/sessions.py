from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import duckdb

from recall.core.config import AppConfig
from recall.core.models import Message, Session, ToolCall
from recall.core.types import UNATTRIBUTED_HOST, Role, Source
from recall.db import connect_readonly


def _session_host_label(raw: object) -> str:
    """Non-empty host for agent/fleet surfaces (REQ-HOST-API-004)."""
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return UNATTRIBUTED_HOST


@dataclass(frozen=True)
class SessionSummary:
    id: str
    source: str
    source_session_id: str | None
    started_at: datetime | None
    ended_at: datetime | None
    cwd: str | None
    git_repo: str | None
    git_branch: str | None
    model: str | None
    message_count: int
    tool_count: int
    input_tokens: int | None
    output_tokens: int | None
    is_complete: bool
    # REQ-LIVE-003: freshness inputs, projectable before `live` exists.
    file_mtime: float
    file_size: int
    indexed_at: datetime | None
    # REQ-LIVE-002: newer of the last indexed message and the transcript's
    # mtime, so a session still being written outranks its last parsed row.
    last_activity_at: datetime
    host: str = UNATTRIBUTED_HOST


def list_sessions(
    *,
    source: Source | None,
    since: datetime | None,
    project: str | None,
    host: str | None = None,
    limit: int = 50,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[SessionSummary]:
    config = config or AppConfig.load()
    owned_conn = conn is None
    conn = conn or connect_readonly(config)
    try:
        where_parts: list[str] = []
        params: list[object] = []
        if source is not None:
            where_parts.append("s.source = ?")
            params.append(source.value)
        if since is not None:
            where_parts.append("COALESCE(ss.ended_at, ss.started_at, ss.indexed_at) >= ?")
            params.append(since)
        if project:
            where_parts.append("ss.git_repo ILIKE ?")
            params.append(f"%{project}%")
        if host is not None:
            where_parts.append("ss.host = ?")
            params.append(host)
        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        sql = f"""
            SELECT s.id, s.source, s.source_session_id, ss.started_at, ss.ended_at,
                   ss.cwd, ss.git_repo, ss.git_branch,
                   ss.model, ss.message_count, ss.tool_count, ss.input_tokens, ss.output_tokens,
                   ss.is_complete, ss.host, ss.file_mtime, ss.file_size, ss.indexed_at,
                   GREATEST(ss.ended_at, CAST(to_timestamp(ss.file_mtime) AS TIMESTAMP))
                     AS last_activity_at
            FROM sessions s
            JOIN session_state ss ON ss.session_id = s.id
            {where_clause}
            ORDER BY ss.started_at DESC NULLS LAST, ss.indexed_at DESC
            LIMIT {limit}
        """
        rows = conn.execute(sql, params).fetchall()
        return [
            SessionSummary(
                id=row[0],
                source=row[1],
                source_session_id=row[2],
                started_at=row[3],
                ended_at=row[4],
                cwd=row[5],
                git_repo=row[6],
                git_branch=row[7],
                model=row[8],
                message_count=int(row[9]),
                tool_count=int(row[10]),
                input_tokens=row[11],
                output_tokens=row[12],
                is_complete=bool(row[13]),
                host=_session_host_label(row[14]),
                file_mtime=float(row[15]),
                file_size=int(row[16]),
                indexed_at=row[17],
                last_activity_at=row[18],
            )
            for row in rows
        ]
    finally:
        if owned_conn:
            conn.close()


# Bare lowercase hex, shorter than a full 32-hex session id: treated as an id
# prefix. Minimum 6 chars keeps prefix matches meaningful; full 32-hex ids and
# dashed UUIDs (Codex source ids) intentionally do not match.
_HEX_ID_PREFIX_RE = re.compile(r"[0-9a-f]{6,31}")

_SESSION_SELECT = """
    SELECT s.id, s.source, s.source_path, s.source_session_id,
           ss.started_at, ss.ended_at, ss.duration_seconds,
           ss.model, ss.cwd, ss.git_repo, ss.git_branch,
           ss.message_count, ss.tool_count, ss.input_tokens,
           ss.output_tokens, ss.is_complete, ss.file_mtime, ss.file_size, ss.indexed_at,
           ss.host
    FROM sessions s
    JOIN session_state ss ON ss.session_id = s.id
"""

# Public because `services/live.py` reads the same tail with the same decoder;
# the aliases `m` and `ms` are part of the contract, so a caller adds its own
# WHERE and ORDER BY against them.
MESSAGE_SELECT = """
    SELECT m.id, m.session_id, m.idx, ms.role, ms.content, ms.thinking, ms.timestamp,
           ms.has_thinking, m.agent_id, COALESCE(ms.context_text, ''),
           COALESCE(ms.context_mode, 'off')
    FROM messages m
    JOIN message_state ms ON ms.message_id = m.id
"""

# The projection `tool_call_from_row` decodes. Kept as data rather than as SQL
# text so another reader (`services/live.py`) can qualify it with its own alias
# without restating the column order the decoder depends on.
TOOL_CALL_COLUMNS: tuple[str, ...] = (
    "id",
    "session_id",
    "message_id",
    "idx",
    "tool_name",
    "tool_input",
    "bash_command",
    "bash_base",
    "bash_sub",
    "is_compound",
    "agent_id",
    "subagent_type",
    "subagent_description",
    "subagent_model",
    "skill_name",
)


def tool_call_columns(alias: str) -> str:
    """`TOOL_CALL_COLUMNS` qualified by a table alias, in decoder order."""
    return ", ".join(f"{alias}.{column}" for column in TOOL_CALL_COLUMNS)


_TOOL_CALL_SELECT = f"SELECT {tool_call_columns('tool_calls')} FROM tool_calls"


def _session_from_row(row: tuple[Any, ...]) -> Session:
    """Build the session envelope; messages and tool calls are attached by the caller."""
    return Session(
        id=row[0],
        source=Source(row[1]),
        source_path=row[2],
        source_session_id=row[3],
        started_at=row[4],
        ended_at=row[5],
        duration_seconds=row[6],
        model=row[7],
        cwd=row[8],
        git_repo=row[9],
        git_branch=row[10],
        message_count=int(row[11]),
        tool_count=int(row[12]),
        input_tokens=row[13],
        output_tokens=row[14],
        is_complete=bool(row[15]),
        file_mtime=float(row[16]),
        file_size=int(row[17]),
        indexed_at=row[18],
        host=_session_host_label(row[19]),
        messages=[],
        orphan_tool_calls=[],
    )


def message_from_row(row: tuple[Any, ...]) -> Message:
    return Message(
        id=row[0],
        session_id=row[1],
        idx=int(row[2]),
        role=Role(row[3]),
        content=row[4],
        thinking=row[5],
        timestamp=row[6],
        has_thinking=bool(row[7]),
        agent_id=row[8],
        context_text=row[9],
        context_mode=row[10],
        tool_calls=[],
    )


def tool_call_from_row(row: tuple[Any, ...]) -> ToolCall:
    return ToolCall(
        id=row[0],
        session_id=row[1],
        message_id=row[2],
        idx=int(row[3]),
        tool_name=row[4],
        tool_input=_parse_tool_input(row[5]),
        bash_command=row[6],
        bash_base=row[7],
        bash_sub=row[8],
        is_compound=bool(row[9]),
        agent_id=row[10],
        subagent_type=row[11],
        subagent_description=row[12],
        subagent_model=row[13],
        skill_name=row[14],
    )


def _resolve_session_row(conn: duckdb.DuckDBPyConnection, session_id: str) -> tuple[Any, ...]:
    """Find one session by internal id, harness session id, or hex id prefix.

    Raises ``ValueError`` when nothing matches or the identifier is ambiguous.
    """
    session_row = conn.execute(
        f"{_SESSION_SELECT} WHERE s.id = ?",
        [session_id],
    ).fetchone()

    # Fallback: try matching by source_session_id (e.g. Claude Code UUID)
    if session_row is None:
        rows = conn.execute(
            f"{_SESSION_SELECT} WHERE s.source_session_id = ?",
            [session_id],
        ).fetchall()
        if len(rows) == 1:
            session_row = rows[0]
        elif len(rows) > 1:
            ids = ", ".join(r[0] for r in rows)
            raise ValueError(
                f"ambiguous source_session_id: {session_id} matches {len(rows)} sessions ({ids})"
            )

    # Prefix fallback: a bare hex string shorter than a full 32-hex id is
    # resolved as a prefix of sessions.id. The prefix is validated pure
    # hex, so it is safe to append the LIKE wildcard directly.
    is_hex_prefix = _HEX_ID_PREFIX_RE.fullmatch(session_id) is not None
    if session_row is None and is_hex_prefix:
        rows = conn.execute(
            f"{_SESSION_SELECT} WHERE s.id LIKE ?",
            [session_id + "%"],
        ).fetchall()
        if len(rows) == 1:
            session_row = rows[0]
        elif len(rows) > 1:
            ids = ", ".join(sorted(r[0] for r in rows))
            raise ValueError(
                f"ambiguous session id prefix: {session_id} matches {len(rows)} sessions ({ids})"
            )

    if session_row is None:
        # An indexed lookup cannot prove transcript absence. Discovery belongs
        # to the coordinator, never to a reader holding a database cursor.
        raise ValueError(
            f"session not found in indexed metadata: {session_id}; "
            "unindexed sources were not searched"
        )

    return session_row


def resolve_session_header(session_id: str, *, conn: duckdb.DuckDBPyConnection) -> Session:
    """The session row one identifier names, without loading its messages.

    Callers that must act *before* reading — `--fresh` indexing the transcript,
    `--after` checking a cursor names this session and not another — need the
    resolved row, and the read is the expensive half. Sharing `load_session`'s
    identifier resolution (internal id, harness id, or hex prefix) is what keeps
    them from ever disagreeing about which session was asked for.
    """
    return _session_from_row(_resolve_session_row(conn, session_id))


def _attach_tool_calls(session: Session, tool_calls: list[ToolCall]) -> None:
    """Hang each tool call off its message, collecting the unmatched as orphans."""
    message_lookup = {message.id: message for message in session.messages}
    for tool_call in tool_calls:
        owner = message_lookup.get(tool_call.message_id) if tool_call.message_id else None
        if owner is None:
            session.orphan_tool_calls.append(tool_call)
        else:
            owner.tool_calls.append(tool_call)


def load_session(
    session_id: str,
    *,
    include_tools: bool,
    message_limit: int | None = None,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> Session:
    config = config or AppConfig.load()
    owned_conn = conn is None
    conn = conn or connect_readonly(config)
    try:
        session_row = _resolve_session_row(conn, session_id)
        session = _session_from_row(session_row)
        # The caller may have passed a source_session_id or a prefix; every
        # subsequent query uses the resolved internal id.
        resolved_id = session.id

        message_rows = conn.execute(
            f"{MESSAGE_SELECT} WHERE m.session_id = ? ORDER BY m.idx ASC",
            [resolved_id],
        ).fetchall()
        messages = [message_from_row(row) for row in message_rows]
        if message_limit is not None:
            messages = messages[:message_limit]
        session.messages = messages

        if include_tools:
            tool_rows = conn.execute(
                f"{_TOOL_CALL_SELECT} WHERE session_id = ? ORDER BY idx ASC",
                [resolved_id],
            ).fetchall()
            _attach_tool_calls(session, [tool_call_from_row(row) for row in tool_rows])

        return session
    finally:
        if owned_conn:
            conn.close()


def load_session_tail(
    session_id: str,
    *,
    tail: int | None = None,
    after_idx: int | None = None,
    tools: bool = False,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> Session:
    """Read a bounded window at the *end* of a session (REQ-LIVE-004).

    ``tail`` takes the last N messages by ``idx``; ``after_idx`` takes only what
    landed after a cursor; together they take the last N of that delta. Both
    omitted reads the whole session. Unlike ``load_session``'s head-anchored
    ``message_limit``, the window is pushed into SQL, so a monitor loop never
    materializes a 10 MB session to look at its last few messages.

    Tool calls are scoped to the returned messages: the window stays bounded, so
    a tool call whose message falls outside it — or which has no message at all —
    is not an orphan of this read and is simply absent.
    """
    if tail is not None and tail <= 0:
        raise ValueError(f"tail must be positive, got {tail}")

    config = config or AppConfig.load()
    owned_conn = conn is None
    conn = conn or connect_readonly(config)
    try:
        session_row = _resolve_session_row(conn, session_id)
        session = _session_from_row(session_row)
        resolved_id = session.id

        where = "WHERE m.session_id = ?"
        params: list[object] = [resolved_id]
        if after_idx is not None:
            where += " AND m.idx > ?"
            params.append(after_idx)

        if tail is None:
            message_rows = conn.execute(
                f"{MESSAGE_SELECT} {where} ORDER BY m.idx ASC",
                params,
            ).fetchall()
        else:
            # Take the newest rows, then restore ascending order for the caller.
            message_rows = list(
                reversed(
                    conn.execute(
                        f"{MESSAGE_SELECT} {where} ORDER BY m.idx DESC LIMIT ?",
                        [*params, tail],
                    ).fetchall()
                )
            )

        session.messages = [message_from_row(row) for row in message_rows]

        if tools and session.messages:
            message_ids = [message.id for message in session.messages]
            placeholders = ", ".join("?" for _ in message_ids)
            tool_rows = conn.execute(
                f"{_TOOL_CALL_SELECT} WHERE session_id = ?"
                f" AND message_id IN ({placeholders}) ORDER BY idx ASC",
                [resolved_id, *message_ids],
            ).fetchall()
            _attach_tool_calls(session, [tool_call_from_row(row) for row in tool_rows])

        return session
    finally:
        if owned_conn:
            conn.close()


def _parse_tool_input(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    if isinstance(value, str):
        try:
            import json

            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return None
