"""Capacity-limited database inputs for detached enrichment work."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final, Literal

import duckdb

from recall.core.models import Message, Session, ToolCall
from recall.db.queries import id_set_predicate
from recall.db.sessions import (
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
class MessageTarget:
    id: str
    content: bool
    thinking: bool
    marker: bool


@dataclass(frozen=True)
class ToolTarget:
    id: str
    bash: bool
    marker: bool


@dataclass(frozen=True)
class EnrichmentTargets:
    messages: tuple[MessageTarget, ...]
    tools: tuple[ToolTarget, ...]

    def __post_init__(self) -> None:
        assert self.count <= TARGET_ROWS_MAX
        assert len(set(self.message_ids)) == len(self.messages)
        assert len(set(self.tool_ids)) == len(self.tools)

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(target.id for target in self.messages)

    @property
    def tool_ids(self) -> tuple[str, ...]:
        return tuple(target.id for target in self.tools)

    @property
    def count(self) -> int:
        return len(self.messages) + len(self.tools)

    def subset(self, message_ids: tuple[str, ...], tool_ids: tuple[str, ...]) -> EnrichmentTargets:
        return EnrichmentTargets(
            tuple(target for target in self.messages if target.id in message_ids),
            tuple(target for target in self.tools if target.id in tool_ids),
        )


def read_target_masks(
    conn: duckdb.DuckDBPyConnection,
    message_ids: tuple[str, ...],
    tool_ids: tuple[str, ...],
    *,
    replace: bool = False,
) -> EnrichmentTargets:
    """Freeze field authority for an explicit bounded request or recomputation."""
    messages, params = selected_predicate("ms.message_id", message_ids)
    tools, tool_params = selected_predicate("t.id", tool_ids)
    embeddings, embedding_params = selected_predicate("message_id", message_ids)
    tool_embeddings, tool_embedding_params = selected_predicate("tool_call_id", tool_ids)
    rows = conn.execute(
        f"SELECT ms.message_id, COALESCE(ms.content, '') <> '' AND "
        f"(? OR COALESCE(e.content_null, true)), COALESCE(ms.thinking, '') <> '' AND "
        f"(? OR COALESCE(e.thinking_null, true)), e.message_id IS NULL "
        f"FROM message_state ms LEFT JOIN (SELECT message_id, "
        f"content_embedding IS NULL AS content_null, thinking_embedding IS NULL AS thinking_null "
        f"FROM message_embeddings WHERE {embeddings}) e ON e.message_id=ms.message_id "
        f"WHERE {messages}",
        [replace, replace, *embedding_params, *params],
    ).fetchall()
    tool_rows = conn.execute(
        f"SELECT t.id, COALESCE(t.bash_command, '') <> '' AND COALESCE(e.bash_null, true), "
        f"e.tool_call_id IS NULL FROM tool_calls t "
        f"LEFT JOIN (SELECT tool_call_id, bash_embedding IS NULL AS bash_null "
        f"FROM tool_call_embeddings WHERE {tool_embeddings}) e ON e.tool_call_id=t.id "
        f"WHERE {tools}",
        [*tool_embedding_params, *tool_params],
    ).fetchall()
    by_id = {
        str(row[0]): MessageTarget(str(row[0]), bool(row[1]), bool(row[2]), bool(row[3]))
        for row in rows
    }
    tool_by_id = {
        str(row[0]): ToolTarget(str(row[0]), bool(row[1]), bool(row[2])) for row in tool_rows
    }
    if not set(message_ids) <= by_id.keys() or not set(tool_ids) <= tool_by_id.keys():
        raise ValueError("enrichment raw inputs unavailable; retry after raw-state repair")
    return EnrichmentTargets(
        tuple(by_id[id] for id in message_ids), tuple(tool_by_id[id] for id in tool_ids)
    )


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
    from recall.db.semantic_work import require_complete

    require_complete(conn)
    rows = conn.execute(
        (
            "\n        SELECT kind, id, idx, missing_content, missing_thinking,"
            " missing_bash, needs_marker FROM semantic_pending\n        WHERE "
            "session_id = ? AND selectable\n          AND (? IS NULL OR (kind, "
            "idx, id) > (?, ?, ?))\n        ORDER BY kind, idx, id LIMIT ?"
            "\n        "
        ),
        [
            session_id,
            cursor[0] if cursor else None,
            *(cursor or (None, None, None)),
            TARGET_ROWS_MAX + 1,
        ],
    ).fetchall()
    if not rows:
        # Wrap once so newly arrived earlier identities cannot be omitted.
        if cursor is not None:
            return select_pending_targets(conn, session_id)
        return PendingTargetPage(EnrichmentTargets((), ()), None)
    candidates = rows[:TARGET_ROWS_MAX]
    targets = EnrichmentTargets(
        tuple(
            MessageTarget(str(row[1]), bool(row[3]), bool(row[4]), bool(row[6]))
            for row in candidates
            if row[0] == "message"
        ),
        tuple(
            ToolTarget(str(row[1]), bool(row[5]), bool(row[6]))
            for row in candidates
            if row[0] == "tool"
        ),
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
    return targets.subset(
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
                return targets.subset(tuple(selected_messages), tuple(selected_tools))
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
    return targets.subset(tuple(selected_messages), tuple(selected_tools))


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


def _tools(session: Session):
    return [tool for m in session.messages for tool in m.tool_calls] + session.orphan_tool_calls


@dataclass(frozen=True)
class EmbedInput:
    """Complete typed message input, excluding separately captured tools and results."""

    message_id: str
    document_input: str
    context_text: str
    context_mode: str


@dataclass(frozen=True)
class EmbedToolInput:
    """Canonical input from the complete typed tool, excluding generated vectors."""

    tool_call_id: str
    document_input: str


@dataclass(frozen=True)
class PreparedEmbedSession:
    session: Session
    source: str
    inputs: tuple[EmbedInput, ...]
    tool_inputs: tuple[EmbedToolInput, ...]
    context_targets: tuple[str, ...]
    context_metadata: tuple[str | None, str | None, str | None]
    targets: EnrichmentTargets
    full_context: bool = False
    publication: Literal["preserve", "replace"] = "preserve"
    generation: tuple[tuple[str, int, int, int], ...] = ()

    def __post_init__(self) -> None:
        assert set(self.context_targets) <= set(self.targets.message_ids)
        if self.publication == "replace":
            assert not self.targets.tool_ids
            assert set(self.context_targets) == set(self.targets.message_ids)


@dataclass(frozen=True)
class GeneratedEmbedSession:
    inputs: PreparedEmbedSession
    session: Session

    def __post_init__(self) -> None:
        assert self.session.id == self.inputs.session.id
        assert len(self.session.messages) == len(self.inputs.targets.message_ids)
        assert len(tuple(_tools(self.session))) == len(self.inputs.targets.tool_ids)
        assert {message.id for message in self.session.messages} == set(
            self.inputs.targets.message_ids
        )
        assert {tool.id for tool in _tools(self.session)} == set(self.inputs.targets.tool_ids)


def snapshot_embed_session(
    session: Session,
    *,
    context_targets: tuple[str, ...],
    targets: EnrichmentTargets,
    full_context: bool = False,
    publication: Literal["preserve", "replace"] = "preserve",
    generation: tuple[tuple[str, int, int, int], ...] = (),
) -> PreparedEmbedSession:
    """Capture the complete document and target context inputs for stale-result rejection."""
    return PreparedEmbedSession(
        session=session,
        source=session.source.value,
        inputs=tuple(
            EmbedInput(
                message_id=message.id,
                document_input=_document_input(message),
                context_text=message.context_text,
                context_mode=message.context_mode,
            )
            for message in session.messages
        ),
        tool_inputs=tuple(
            EmbedToolInput(
                tool_call_id=tool.id,
                document_input=_document_input(tool),
            )
            for tool in _tools(session)
        ),
        context_targets=context_targets,
        context_metadata=(session.cwd, session.git_repo, session.git_branch),
        targets=targets,
        full_context=full_context,
        publication=publication,
        generation=generation,
    )


def _document_input(value: Message | ToolCall) -> str:
    """Capture the complete loaded input type, with derived vectors excluded."""
    return json.dumps(
        value.model_dump(
            mode="json",
            exclude={
                "tool_calls",
                "content_embedding",
                "thinking_embedding",
                "bash_embedding",
            },
        ),
        sort_keys=True,
        ensure_ascii=False,
    )


def _inputs_still_match(conn: duckdb.DuckDBPyConnection, item: PreparedEmbedSession) -> bool:
    """Reject a whole session when its document/context input generation moved.

    Both sides are keyed by message/tool-call identity, never row order:
    tool_calls.idx is not unique within a session, so ORDER BY idx leaves ties
    whose physical order is plan-dependent, and identical content compared as
    ordered tuples could compare unequal.
    """
    if source_generation(conn, item.session.id) != item.generation:
        return False
    if item.publication == "preserve" and item.context_targets:
        # A newly published companion owns its context and any valid siblings.
        # Retry with that stored prefix rather than invalidating it for new context.
        present = conn.execute(
            "SELECT 1 FROM message_embeddings WHERE message_id IN "
            "(SELECT UNNEST(?::TEXT[])) LIMIT 1",
            [list(item.context_targets)],
        ).fetchone()
        if present is not None:
            return False
    try:
        current = read_enrichment_session(
            conn, item.session.id, item.targets, full_context=item.full_context
        )
    except (ValueError, EnrichmentCapacityError):
        return False
    snapshot = snapshot_embed_session(
        current,
        context_targets=item.context_targets,
        targets=item.targets,
        full_context=item.full_context,
    )
    if snapshot.source != item.source:
        return False
    if item.context_targets and snapshot.context_metadata != item.context_metadata:
        return False
    return {entry.message_id: entry for entry in snapshot.inputs} == {
        entry.message_id: entry for entry in item.inputs
    } and {entry.tool_call_id: entry for entry in snapshot.tool_inputs} == {
        entry.tool_call_id: entry for entry in item.tool_inputs
    }


def source_generation(
    conn: duckdb.DuckDBPyConnection, session_id: str
) -> tuple[tuple[str, int, int, int], ...]:
    return tuple(
        (str(key), int(epoch), int(desired), int(committed))
        for key, epoch, desired, committed in conn.execute(
            "SELECT "
            "source_key,content_epoch,desired_generation,committed_generation FROM "
            "source_files WHERE session_id=? ORDER BY source_key",
            [session_id],
        ).fetchall()
    )
