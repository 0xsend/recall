"""Archived single-transaction move oracle from fa89ba6 (REQ-INDEX-027)."""

from __future__ import annotations

import json
from typing import Final

import duckdb
from recall.core.ids import message_id, tool_call_id
from recall.db.queries import enqueue_session_sidecar_deletes
from recall.services.moves import MovedDuplicate, supersede


def supersede_moved_duplicate(
    conn: duckdb.DuckDBPyConnection, plan: MovedDuplicate, *, queue_sidecar_deletes: bool
) -> None:
    """Move the surviving history into the departed session's namespace atomically.

    A copy can have been indexed before its original disappeared. The original
    keeps its identity; the separately allocated copy id retires without an
    alias. All scalar state, vectors and tail facts move with their owner.
    """
    conn.execute("BEGIN")
    try:
        _carry_stored_contexts(
            conn, predecessor_id=plan.predecessor_id, successor_id=plan.successor_id
        )
        _carry_stored_vectors(
            conn, predecessor_id=plan.predecessor_id, successor_id=plan.successor_id
        )
        # A transcript without usage (Grok) must retain harvested totals on the
        # original identity, just as a direct reparse does (REQ-USAGE-014).
        conn.execute(
            """UPDATE session_state AS s SET
                   input_tokens = COALESCE(s.input_tokens, p.input_tokens),
                   output_tokens = COALESCE(s.output_tokens, p.output_tokens),
                   cached_input_tokens = COALESCE(s.cached_input_tokens, p.cached_input_tokens)
               FROM session_state AS p
               WHERE s.session_id = ? AND p.session_id = ?""",
            [plan.successor_id, plan.predecessor_id],
        )
        if queue_sidecar_deletes:
            enqueue_session_sidecar_deletes(conn, plan.successor_id)
        messages = [
            {"old_id": str(old), "id": message_id(plan.predecessor_id, int(idx))}
            for old, idx in conn.execute(
                "SELECT id, idx FROM messages WHERE session_id = ?", [plan.successor_id]
            ).fetchall()
        ]
        mapped_messages = {row["old_id"]: row["id"] for row in messages}
        tools = [
            {
                "old_id": str(old),
                "id": tool_call_id(
                    mapped_messages[str(parent)] if parent is not None else None,
                    int(idx),
                    plan.predecessor_id,
                ),
            }
            for old, parent, idx in conn.execute(
                "SELECT id, message_id, idx FROM tool_calls WHERE session_id = ?",
                [plan.successor_id],
            ).fetchall()
        ]
        supersede(
            conn,
            predecessor_ids=(plan.predecessor_id,),
            successor_id=plan.successor_id,
            queue_sidecar_deletes=queue_sidecar_deletes,
        )
        for kind, mappings, companions, owner in (
            ("message", messages, ("message_state", "message_embeddings"), "message_id"),
            (
                "tool",
                tools,
                ("tool_results", "tool_use_ids", "tool_call_embeddings"),
                "tool_call_id",
            ),
        ):
            conn.execute(
                "CREATE OR REPLACE TEMP TABLE move_ids AS SELECT unnest("
                'json_transform_strict(?, \'[{"old_id":"VARCHAR","id":"VARCHAR"}]\'),'
                " recursive := true)",
                [json.dumps(mappings)],
            )
            for table in companions:
                conn.execute(
                    f"UPDATE {table} SET {owner} = m.id FROM move_ids m WHERE {owner} = m.old_id"
                )
            if kind == "message":
                conn.execute(
                    "UPDATE tool_calls SET message_id = m.id FROM move_ids m"
                    " WHERE message_id = m.old_id"
                )
                conn.execute(
                    "UPDATE messages SET id = m.id FROM move_ids m WHERE messages.id = m.old_id"
                )
            else:
                conn.execute(
                    "UPDATE tool_calls SET id = m.id FROM move_ids m WHERE tool_calls.id = m.old_id"
                )
        conn.execute("DROP TABLE move_ids")
        for table in (
            "messages",
            "tool_calls",
            "tool_use_ids",
            "session_stop_markers",
            "session_state",
            "usage_events",
            "source_files",
        ):
            conn.execute(
                f"UPDATE {table} SET session_id = ? WHERE session_id = ?",
                [plan.predecessor_id, plan.successor_id],
            )
        conn.execute(
            "UPDATE sessions SET id = ? WHERE id = ?", [plan.predecessor_id, plan.successor_id]
        )
        conn.execute(
            "UPDATE source_files SET session_id = ? WHERE source_path = ? "
            "AND source = (SELECT source FROM sessions WHERE id = ?)",
            [plan.predecessor_id, plan.successor_path, plan.predecessor_id],
        )
        for kind, table in (("message", "messages"), ("tool_call", "tool_calls")):
            conn.execute(
                f"INSERT INTO fts_sidecar_pending(kind, id, op)"
                f" SELECT '{kind}', id, 'upsert' FROM {table} WHERE session_id = ?",
                [plan.predecessor_id],
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


CARRIED_CONTEXTS_SELECT: Final = """
        SELECT s.id AS successor_message, p.id AS predecessor_message,
               ps.context_text, ps.context_mode
        FROM messages p
        JOIN message_state ps ON ps.message_id = p.id
        JOIN messages s ON s.session_id = ? AND s.idx = p.idx
        JOIN message_state ss ON ss.message_id = s.id
        WHERE p.session_id = ?
          AND ps.context_mode IN ('llm-local', 'llm-remote', 'llm-codex')
          AND COALESCE(ps.context_text, '') <> ''
          AND COALESCE(ss.context_mode, 'off') NOT IN ('llm-local', 'llm-remote', 'llm-codex')
          AND ss.role = ps.role
          AND ss.content IS NOT DISTINCT FROM ps.content
          AND ss.thinking IS NOT DISTINCT FROM ps.thinking
"""


CARRIED_MESSAGE_VECTORS_SELECT: Final = """
    SELECT s.id AS successor_message, e.content_embedding, e.thinking_embedding
    FROM messages p
    JOIN message_state ps ON ps.message_id = p.id
    JOIN message_embeddings e ON e.message_id = p.id
    JOIN messages s ON s.session_id = ? AND s.idx = p.idx
    JOIN message_state ss ON ss.message_id = s.id
    LEFT JOIN message_embeddings target ON target.message_id = s.id
    WHERE p.session_id = ? AND target.message_id IS NULL
      AND ss.role = ps.role
      AND ss.content IS NOT DISTINCT FROM ps.content
      AND ss.thinking IS NOT DISTINCT FROM ps.thinking
      AND COALESCE(ss.context_text, '') = COALESCE(ps.context_text, '')
      AND COALESCE(ss.context_mode, 'off') = COALESCE(ps.context_mode, 'off')
"""


CARRIED_TOOL_VECTORS_SELECT: Final = """
    SELECT s.id AS successor_tool_call, e.bash_embedding
    FROM tool_calls p
    JOIN tool_call_embeddings e ON e.tool_call_id = p.id
    JOIN tool_calls s ON s.session_id = ? AND s.idx = p.idx
    LEFT JOIN messages pm ON pm.id = p.message_id
    LEFT JOIN messages sm ON sm.id = s.message_id
    LEFT JOIN tool_call_embeddings target ON target.tool_call_id = s.id
    WHERE p.session_id = ? AND target.tool_call_id IS NULL
      AND s.bash_command IS NOT DISTINCT FROM p.bash_command
      AND ((p.message_id IS NULL AND s.message_id IS NULL) OR pm.idx = sm.idx)
"""


def _carry_stored_vectors(
    conn: duckdb.DuckDBPyConnection, *, predecessor_id: str, successor_id: str
) -> None:
    """Fill absent vector rows only for matching stored inputs at the same position.

    Existing successor rows, including processed NULL markers, keep ownership.
    Context transfer runs first so its vectors remain tied to the paid context.
    """
    conn.execute(
        "INSERT INTO message_embeddings " + CARRIED_MESSAGE_VECTORS_SELECT,
        [successor_id, predecessor_id],
    )
    conn.execute(
        "INSERT INTO tool_call_embeddings " + CARRIED_TOOL_VECTORS_SELECT,
        [successor_id, predecessor_id],
    )


def _carry_stored_contexts(
    conn: duckdb.DuckDBPyConnection, *, predecessor_id: str, successor_id: str
) -> None:
    """Give the survivor's messages the LLM context the predecessor paid for.

    A proven move makes the predecessor's conversation a prefix of the
    survivor's, so the message at the same position with the same text had the
    same context inputs.  The predecessor's embedding was computed from that
    context, so it replaces the survivor's; without one, the survivor's stale
    embedding is dropped and embedded again.  The keyword-search text includes
    the context, so it is rebuilt here and the sidecar rows are queued for an
    upsert the daemon drains.
    """
    conn.execute(
        "CREATE OR REPLACE TEMP TABLE carried_contexts AS " + CARRIED_CONTEXTS_SELECT,
        [successor_id, predecessor_id],
    )
    conn.execute(
        """
        UPDATE message_state
        SET
            context_text = c.context_text,
            context_mode = c.context_mode,
            fts_content = COALESCE(c.context_text, '') || COALESCE(message_state.content, ''),
            fts_thinking = COALESCE(c.context_text, '') || COALESCE(message_state.thinking, '')
        FROM carried_contexts c
        WHERE message_state.message_id = c.successor_message
        """
    )
    conn.execute(
        "DELETE FROM message_embeddings"
        " WHERE message_id IN (SELECT successor_message FROM carried_contexts)"
    )
    conn.execute(
        """
        INSERT INTO message_embeddings (message_id, content_embedding, thinking_embedding)
        SELECT c.successor_message, e.content_embedding, e.thinking_embedding
        FROM carried_contexts c
        JOIN message_embeddings e ON e.message_id = c.predecessor_message
        """
    )
    conn.execute(
        "INSERT INTO fts_sidecar_pending(kind, id, op)"
        " SELECT DISTINCT 'message', successor_message, 'upsert' FROM carried_contexts"
    )
    conn.execute("DROP TABLE carried_contexts")
