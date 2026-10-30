from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from itertools import batched
from typing import Any, cast

import duckdb

from recall.core.config import AppConfig
from recall.core.models import Message, Session, SessionSummary, ToolCall
from recall.core.types import UNATTRIBUTED_HOST, Role, Source
from recall.db import connect_readonly
from recall.db.queries import id_set_predicate
from recall.db.read_snapshot import read_snapshot
from recall.services.session_ids import UnknownSessionError, resolve_session_identifier


def _session_host_label(raw: object) -> str:
    """Non-empty host for agent/fleet surfaces (REQ-HOST-API-004)."""
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return UNATTRIBUTED_HOST


def list_sessions(
    *,
    source: Source | None,
    since: datetime | None,
    project: str | None,
    host: str | None = None,
    session: str | None = None,
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
        if session is not None:
            identity = resolve_session_identifier(
                conn, session, source=source.value if source is not None else None, host=host
            )
            where_parts.append("s.id = ?")
            params.append(identity.id)
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

# The projection `message_from_row` decodes. The aliases `m` and `ms` are
# part of the contract, so a caller adds its own
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
    identity = resolve_session_identifier(conn, session_id)
    row = conn.execute(f"{_SESSION_SELECT} WHERE s.id = ?", [identity.id]).fetchone()
    if row is None:
        raise UnknownSessionError(
            f"session not indexed: {session_id}; session state is unavailable"
        )
    return row


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

        message_rows = session_message_rows(conn, resolved_id, limit=message_limit)
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


def session_message_rows(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    *,
    limit: int | None = None,
) -> list[tuple[Any, ...]]:
    """Resolve identities before reading state in bounded primary-key groups."""
    if limit is not None and limit <= 0:
        raise ValueError("message limit must be positive")
    statement = "SELECT id FROM messages WHERE session_id = ? ORDER BY idx, id"
    parameters: list[object] = [session_id]
    if limit is not None:
        statement += " LIMIT ?"
        parameters.append(limit)
    identities = conn.execute(statement, parameters).fetchall()
    return _message_rows_by_id(conn, (str(row[0]) for row in identities))


def _message_rows_by_id(
    conn: duckdb.DuckDBPyConnection, message_ids: Iterable[str]
) -> list[tuple[Any, ...]]:
    """Read an ascending identity window in bounded primary-key groups."""
    rows: list[tuple[Any, ...]] = []
    for group in batched(message_ids, 256):
        selected, bindings = id_set_predicate("ms.message_id", list(group))
        rows.extend(
            conn.execute(
                f"{MESSAGE_SELECT} WHERE {selected} ORDER BY m.idx, m.id",
                bindings,
            ).fetchall()
        )
    return rows


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

    A supplied connection belongs to the caller, which owns the read snapshot
    across metadata, identities, payload pages and tools. Without one, this
    function opens and finishes its own snapshot.
    """
    if tail is not None and tail <= 0:
        raise ValueError(f"tail must be positive, got {tail}")

    config = config or AppConfig.load()
    if conn is not None:
        return _load_session_tail(conn, session_id, tail=tail, after_idx=after_idx, tools=tools)
    conn = connect_readonly(config)
    try:
        return read_snapshot(
            conn,
            lambda snapshot: _load_session_tail(
                snapshot, session_id, tail=tail, after_idx=after_idx, tools=tools
            ),
        )
    finally:
        conn.close()


def _load_session_tail(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    *,
    tail: int | None,
    after_idx: int | None,
    tools: bool,
) -> Session:
    session = _session_from_row(_resolve_session_row(conn, session_id))
    statement = "SELECT id FROM messages WHERE session_id = ?"
    params: list[object] = [session.id]
    if after_idx is not None:
        statement += " AND idx > ?"
        params.append(after_idx)
    if tail is None:
        statement += " ORDER BY idx, id"
    else:
        statement += " ORDER BY idx DESC, id DESC LIMIT ?"
        params.append(tail)
    identities = conn.execute(statement, params).fetchall()
    if tail is not None:
        identities.reverse()
    message_rows = _message_rows_by_id(conn, (str(row[0]) for row in identities))
    session.messages = [message_from_row(row) for row in message_rows]

    if tools:
        by_id = {message.id: message for message in session.messages}
        for group in batched(session.messages, 256):
            message_ids = [message.id for message in group]
            selected, bindings = id_set_predicate("message_id", message_ids)
            rows = conn.execute(
                f"{_TOOL_CALL_SELECT} WHERE session_id = ? AND {selected} ORDER BY idx ASC",
                [session.id, *bindings],
            ).fetchall()
            for row in rows:
                tool = tool_call_from_row(row)
                assert tool.message_id is not None
                by_id[tool.message_id].tool_calls.append(tool)

    return session


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
