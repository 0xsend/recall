"""Capacity-limited database inputs for detached enrichment work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import duckdb

from recall.core.models import Session
from recall.db.queries import id_set_predicate
from recall.db.scans import execute_scan
from recall.services.sessions import (
    TOOL_CALL_COLUMNS,
    message_from_row,
    resolve_session_header,
    tool_call_from_row,
)

TARGET_ROWS_MAX: Final = 128
BATCH_BYTES_MAX: Final = 8 * 1024 * 1024
ITEM_BYTES_MAX: Final = 64 * 1024 * 1024
CONTEXT_BYTES_MAX: Final = 128 * 1024 * 1024
CONTEXT_ROWS_MAX: Final = 65536


def selected_predicate(column: str, identifiers: tuple[str, ...]) -> tuple[str, list[str]]:
    return id_set_predicate(column, identifiers) if identifiers else ("FALSE", [])


TOOL_PAYLOAD_BYTES_SQL: Final = (
    "4 * ("
    + " + ".join(
        f"COALESCE(length(CAST({name} AS VARCHAR)), 0)"
        for name in TOOL_CALL_COLUMNS
        if name not in {"idx", "is_compound"}
    )
    + ")"
)

MESSAGE_PAYLOAD_BYTES_SQL: Final = (
    "4 * (COALESCE(length(ms.content), 0) + COALESCE(length(ms.thinking), 0) "
    "+ COALESCE(length(ms.context_text), 0))"
)


class EnrichmentCapacityError(RuntimeError):
    """An input cannot be admitted without exceeding the enrichment capacity."""


@dataclass(frozen=True)
class EnrichmentTargets:
    message_ids: tuple[str, ...]
    tool_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        assert self.count <= TARGET_ROWS_MAX
        assert len(set(self.message_ids)) == len(self.message_ids)
        assert len(set(self.tool_ids)) == len(self.tool_ids)

    @property
    def count(self) -> int:
        return len(self.message_ids) + len(self.tool_ids)


@dataclass(frozen=True)
class PendingTargetPage:
    targets: EnrichmentTargets
    cursor: tuple[str, int, str] | None

    @property
    def scanning(self) -> bool:
        return self.targets.count == 0 and self.cursor is not None


def select_pending_targets(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    cursor: tuple[str, int, str] | None = None,
) -> PendingTargetPage:
    """Page identities first, then inspect only explicit-key payload sizes."""
    rows = execute_scan(
        conn,
        """
        WITH pending AS (
            SELECT 'message' AS kind, m.id, m.idx FROM messages m
            WHERE m.session_id = ? AND NOT EXISTS (
                SELECT 1 FROM message_embeddings e WHERE e.message_id = m.id
            )
            UNION ALL
            SELECT 'tool', t.id, t.idx FROM tool_calls t
            WHERE t.session_id = ? AND NOT EXISTS (
                SELECT 1 FROM tool_call_embeddings e WHERE e.tool_call_id = t.id
            )
        )
        SELECT kind, id, idx FROM pending
        WHERE ? IS NULL OR (kind, idx, id) > (?, ?, ?)
        ORDER BY kind, idx, id LIMIT ?
    """,
        [
            session_id,
            session_id,
            cursor[0] if cursor else None,
            *(cursor or (None, None, None)),
            TARGET_ROWS_MAX + 1,
        ],
    )
    if not rows:
        # Wrap once so newly arrived earlier identities cannot be omitted.
        if cursor is not None:
            return select_pending_targets(conn, session_id)
        return PendingTargetPage(EnrichmentTargets((), ()), None)
    candidates = rows[:TARGET_ROWS_MAX]
    targets = EnrichmentTargets(
        tuple(str(row[1]) for row in candidates if row[0] == "message"),
        tuple(str(row[1]) for row in candidates if row[0] == "tool"),
    )
    try:
        selected = admit_targets(conn, targets)
    except EnrichmentCapacityError:
        if len(rows) <= TARGET_ROWS_MAX:
            raise
        last = candidates[-1]
        return PendingTargetPage(
            EnrichmentTargets((), ()), (str(last[0]), int(last[2]), str(last[1]))
        )
    selected_keys = {("message", value) for value in selected.message_ids} | {
        ("tool", value) for value in selected.tool_ids
    }
    last_position = max(i for i, row in enumerate(candidates) if (row[0], row[1]) in selected_keys)
    last = candidates[last_position]
    next_cursor = (
        (str(last[0]), int(last[2]), str(last[1])) if last_position + 1 < len(rows) else None
    )
    return PendingTargetPage(selected, next_cursor)


def oversized_targets(
    conn: duckdb.DuckDBPyConnection,
    targets: EnrichmentTargets,
) -> EnrichmentTargets:
    messages, tools = input_sizes(conn, targets)
    return EnrichmentTargets(
        tuple(
            identifier
            for identifier in targets.message_ids
            if messages.get(identifier, 0) > ITEM_BYTES_MAX
        ),
        tuple(
            identifier
            for identifier in targets.tool_ids
            if tools.get(identifier, 0) > ITEM_BYTES_MAX
        ),
    )


def admit_targets(
    conn: duckdb.DuckDBPyConnection,
    targets: EnrichmentTargets,
) -> EnrichmentTargets:
    """Admit a prefix of targets by conservative UTF-8 payload size.

    Four bytes per character bounds UTF-8 without materializing encoded copies.
    A single large input gets an exclusive slot; an input above that slot's
    ceiling is an operating error, never a processed NULL marker.
    """
    assert targets.count <= TARGET_ROWS_MAX
    message_sizes, tool_sizes = input_sizes(conn, targets)
    selected_messages: list[str] = []
    selected_tools: list[str] = []
    total = 0
    oversized: list[str] = []
    for identifiers, sizes, selected in (
        (targets.message_ids, message_sizes, selected_messages),
        (targets.tool_ids, tool_sizes, selected_tools),
    ):
        for identifier in identifiers:
            size = sizes.get(identifier, 0)
            if total + size > BATCH_BYTES_MAX and (selected_messages or selected_tools):
                return EnrichmentTargets(tuple(selected_messages), tuple(selected_tools))
            if size > ITEM_BYTES_MAX:
                oversized.append(identifier)
                continue
            selected.append(identifier)
            total += size
    if oversized and not (selected_messages or selected_tools):
        raise EnrichmentCapacityError(
            f"enrichment input {oversized[0]} exceeds the {ITEM_BYTES_MAX} byte "
            "capacity; raw history is preserved and enrichment remains pending"
        )
    return EnrichmentTargets(tuple(selected_messages), tuple(selected_tools))


def input_sizes(
    conn: duckdb.DuckDBPyConnection,
    targets: EnrichmentTargets,
) -> tuple[dict[str, int], dict[str, int]]:
    messages, message_params = selected_predicate("message_id", targets.message_ids)
    tools, tool_params = selected_predicate("id", targets.tool_ids)
    message_rows = conn.execute(
        f"SELECT message_id, 4 * (COALESCE(length(content), 0) + "
        f"COALESCE(length(thinking), 0) + COALESCE(length(context_text), 0)) "
        f"FROM message_state WHERE {messages}",
        message_params,
    ).fetchall()
    tool_rows = conn.execute(
        f"SELECT id, {TOOL_PAYLOAD_BYTES_SQL} FROM tool_calls WHERE {tools}",
        tool_params,
    ).fetchall()
    return (
        {str(row[0]): int(row[1]) for row in message_rows},
        {str(row[0]): int(row[1]) for row in tool_rows},
    )


def read_enrichment_session(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    targets: EnrichmentTargets,
    *,
    full_context: bool,
) -> Session:
    """Load selected rows, or an admitted full-history context dependency."""
    session = resolve_session_header(session_id, conn=conn)
    if not full_context and admit_targets(conn, targets) != targets:
        raise EnrichmentCapacityError("enrichment inputs grew beyond the admitted batch capacity")
    if full_context:
        message_ids = tuple(
            str(row[0])
            for row in conn.execute(
                "SELECT id FROM messages WHERE session_id = ? ORDER BY idx, id LIMIT ?",
                [session_id, CONTEXT_ROWS_MAX + 1],
            ).fetchall()
        )
        tool_ids = tuple(
            str(row[0])
            for row in conn.execute(
                "SELECT id FROM tool_calls WHERE session_id = ? ORDER BY idx, id LIMIT ?",
                [session_id, CONTEXT_ROWS_MAX + 1],
            ).fetchall()
        )
        if len(message_ids) + len(tool_ids) > CONTEXT_ROWS_MAX:
            raise EnrichmentCapacityError("LLM context exceeds the 65536-row dependency capacity")
    else:
        message_ids, tool_ids = targets.message_ids, targets.tool_ids
    selected, params = selected_predicate("message_id", message_ids)
    tool_selected, tool_params = selected_predicate("id", tool_ids)
    if full_context:
        sizes = conn.execute(
            f"SELECT COALESCE(SUM(4 * (COALESCE(length(content), 0) + "
            f"COALESCE(length(thinking), 0) + COALESCE(length(context_text), 0))), 0) "
            f"FROM message_state WHERE {selected}",
            params,
        ).fetchone()
        # Dependency rosters can exceed target capacity; estimate the complete
        # typed tool projection without constructing an EnrichmentTargets batch.
        tool_size = conn.execute(
            f"SELECT COALESCE(SUM({TOOL_PAYLOAD_BYTES_SQL}), 0) "
            f"FROM tool_calls WHERE {tool_selected}",
            tool_params,
        ).fetchone()
        assert sizes is not None and tool_size is not None
        if int(sizes[0]) + int(tool_size[0]) > CONTEXT_BYTES_MAX:
            raise EnrichmentCapacityError("LLM context exceeds the 128 MiB dependency capacity")
    # The explicit payload predicate prevents a session join from turning a
    # small read into an unrestricted message_state scan on a populated index.
    session.messages = [
        message_from_row(row)
        for row in conn.execute(
            f"SELECT m.id, m.session_id, m.idx, ms.role, ms.content, ms.thinking, ms.timestamp, "
            f"ms.has_thinking, m.agent_id, COALESCE(ms.context_text, ''), "
            f"COALESCE(ms.context_mode, 'off') FROM messages m JOIN "
            f"(SELECT * FROM message_state WHERE {selected}) ms ON ms.message_id = m.id "
            "ORDER BY m.idx, m.id",
            params,
        ).fetchall()
    ]
    tools = [
        tool_call_from_row(row)
        for row in conn.execute(
            f"SELECT {', '.join(TOOL_CALL_COLUMNS)} FROM tool_calls WHERE {tool_selected} "
            "ORDER BY idx, id",
            tool_params,
        ).fetchall()
    ]
    by_id = {message.id: message for message in session.messages}
    for tool in tools:
        if tool.message_id in by_id:
            by_id[tool.message_id].tool_calls.append(tool)
        else:
            session.orphan_tool_calls.append(tool)
    if not set(targets.message_ids) <= set(by_id) or not set(targets.tool_ids) <= {
        tool.id for tool in tools
    }:
        raise ValueError("enrichment targets changed before preparation; retry the request")
    return session
