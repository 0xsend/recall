"""Private predicate-changing SQL; called only inside Publication."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from itertools import batched

import duckdb

from recall.core.models import Message, Session, ToolCall
from recall.db.queries import (
    _bound_identifiers,
    _delete_session_sidecar_rows,
    id_set_predicate,
)

logger = logging.getLogger(__name__)


def delete_session(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    sidecar_conn: sqlite3.Connection | None = None,
) -> None:
    msg_subquery = "SELECT id FROM messages WHERE session_id = ?"
    tc_subquery = "SELECT id FROM tool_calls WHERE session_id = ?"
    message_ids = [str(row[0]) for row in conn.execute(msg_subquery, [session_id]).fetchall()]
    tool_call_ids = [str(row[0]) for row in conn.execute(tc_subquery, [session_id]).fetchall()]
    conn.execute(
        f"DELETE FROM tool_call_embeddings WHERE tool_call_id IN ({tc_subquery})",
        [session_id],
    )
    conn.execute(
        f"DELETE FROM message_embeddings WHERE message_id IN ({msg_subquery})",
        [session_id],
    )
    # `tool_results` is reachable only through `tool_use_ids` (see
    # `delete_tool_call_tail_facts`), so scoping the delete by the mapping —
    # not by `tool_calls` — also reaches results whose call row is already gone.
    conn.execute(
        "DELETE FROM tool_results WHERE tool_call_id IN"
        " (SELECT tool_call_id FROM tool_use_ids WHERE session_id = ?)",
        [session_id],
    )
    conn.execute("DELETE FROM tool_use_ids WHERE session_id = ?", [session_id])
    conn.execute("DELETE FROM session_stop_markers WHERE session_id = ?", [session_id])
    conn.execute("DELETE FROM tool_calls WHERE session_id = ?", [session_id])
    conn.execute(
        f"DELETE FROM message_state WHERE message_id IN ({msg_subquery})",
        [session_id],
    )
    conn.execute("DELETE FROM messages WHERE session_id = ?", [session_id])
    conn.execute("DELETE FROM session_state WHERE session_id = ?", [session_id])
    conn.execute("DELETE FROM sessions WHERE id = ?", [session_id])
    if sidecar_conn is not None and (message_ids or tool_call_ids):
        _delete_session_sidecar_rows(conn, sidecar_conn, message_ids, tool_call_ids)


def insert_session(
    conn: duckdb.DuckDBPyConnection, session: Session, *, last_byte_offset: int = 0
) -> None:
    conn.execute(
        """
        INSERT INTO sessions (id, source, source_path, source_session_id)
        VALUES (?, ?, ?, ?)
        """,
        [
            session.id,
            session.source.value,
            session.source_path,
            session.source_session_id,
        ],
    )
    conn.execute(
        """
        INSERT INTO session_state (
            session_id, started_at, ended_at, duration_seconds,
            model, cwd, git_repo, git_branch,
            message_count, tool_count, input_tokens, output_tokens,
            cached_input_tokens,
            is_complete, file_mtime, file_size, sidecar_mtime,
            last_byte_offset, indexed_at
        ) VALUES (
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?,
            ?, ?, ?, ?,
            ?, ?
        )
        """,
        [
            session.id,
            session.started_at,
            session.ended_at,
            session.duration_seconds,
            session.model,
            session.cwd,
            session.git_repo,
            session.git_branch,
            session.message_count,
            session.tool_count,
            session.input_tokens,
            session.output_tokens,
            session.cached_input_tokens,
            session.is_complete,
            session.file_mtime,
            session.file_size,
            session.sidecar_mtime,
            last_byte_offset,
            session.indexed_at or datetime.now(UTC),
        ],
    )


def _clear_message_rows(conn: duckdb.DuckDBPyConnection, message_ids: list[str]) -> None:
    """Drop any rows already standing on the ids about to be written.

    The prior-row inner join cannot see a message without its state, or state
    without its message. Clearing both owned identities prevents duplicate-key
    errors when repairing such historical inconsistencies. DuckDB 1.5.5 still
    raises ConstraintException without this cleanup in test_orphan_message_state.
    Retire it only when another writer-owned path reconciles orphan companions
    before insertion and those real-database repair cases still pass.
    """
    with _bound_identifiers(conn, message_ids):
        conn.execute(
            "DELETE FROM message_embeddings WHERE message_id IN (SELECT id FROM _recall_bound_ids)"
        )
        conn.execute(
            "DELETE FROM message_state WHERE message_id IN (SELECT id FROM _recall_bound_ids)"
        )
        conn.execute("DELETE FROM messages WHERE id IN (SELECT id FROM _recall_bound_ids)")


def insert_messages(
    conn: duckdb.DuckDBPyConnection,
    messages: Iterable[Message],
    *,
    context_text: str = "",
    context_mode: str = "off",
    context_by_message: Mapping[str, tuple[str, str]] | None = None,
) -> None:
    """Replace message rows, optionally with a complete per-message context map."""
    import pyarrow as pa

    materialized = list(messages)
    if not materialized:
        return
    # Clear all owned identities before inserting: a replacement may move an
    # existing (session_id, idx) onto an id in a different batch.
    for batch in batched(materialized, 256):
        _clear_message_rows(conn, [message.id for message in batch])
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("session_id", pa.string()),
            ("idx", pa.int64()),
            ("agent_id", pa.string()),
            ("role", pa.string()),
            ("content", pa.string()),
            ("thinking", pa.string()),
            ("timestamp", pa.timestamp("us")),
            ("timestamp_utc", pa.timestamp("us", tz="UTC")),
            ("has_thinking", pa.bool_()),
            ("context_text", pa.string()),
            ("context_mode", pa.string()),
        ]
    )
    for batch in batched(materialized, 256):
        rows = []
        for message in batch:
            text, mode = (
                context_by_message[message.id]
                if context_by_message is not None
                else (context_text, context_mode)
            )
            stamp = message.timestamp
            aware = stamp is not None and stamp.utcoffset() is not None
            rows.append(
                (
                    message.id,
                    message.session_id,
                    message.idx,
                    message.agent_id,
                    message.role.value,
                    message.content,
                    message.thinking,
                    None if aware else stamp,
                    stamp if aware else None,
                    message.has_thinking,
                    text,
                    mode,
                )
            )
        table = pa.Table.from_arrays(
            [
                pa.array(column, type=field.type)
                for column, field in zip(zip(*rows, strict=True), schema, strict=True)
            ],
            schema=schema,
        )
        conn.register("_recall_message_inserts", table)
        try:
            conn.execute(
                "INSERT INTO messages (id, session_id, idx, agent_id) "
                "SELECT id, session_id, idx, agent_id FROM _recall_message_inserts"
            )
            conn.execute(
                """
                INSERT INTO message_state (
                    message_id, role, content, thinking, timestamp, has_thinking,
                    context_text, context_mode, fts_content, fts_thinking
                )
                SELECT id, role, content, thinking,
                    COALESCE(CAST(timestamp_utc AS TIMESTAMP), timestamp), has_thinking,
                    context_text, context_mode,
                    context_text || COALESCE(content, ''),
                    context_text || COALESCE(thinking, '')
                FROM _recall_message_inserts
                """
            )
        finally:
            conn.unregister("_recall_message_inserts")


def insert_tool_calls(conn: duckdb.DuckDBPyConnection, tool_calls: Iterable[ToolCall]) -> None:
    import pyarrow as pa

    schema = pa.schema(
        [
            ("id", pa.string()),
            ("session_id", pa.string()),
            ("message_id", pa.string()),
            ("idx", pa.int64()),
            ("tool_name", pa.string()),
            ("tool_input", pa.string()),
            ("bash_command", pa.string()),
            ("bash_base", pa.string()),
            ("bash_sub", pa.string()),
            ("is_compound", pa.bool_()),
            ("agent_id", pa.string()),
            ("subagent_type", pa.string()),
            ("subagent_description", pa.string()),
            ("subagent_model", pa.string()),
            ("skill_name", pa.string()),
        ]
    )
    for batch in batched(tool_calls, 256):
        rows = [
            (
                tool.id,
                tool.session_id,
                tool.message_id,
                tool.idx,
                tool.tool_name,
                json.dumps(tool.tool_input) if tool.tool_input is not None else None,
                tool.bash_command,
                tool.bash_base,
                tool.bash_sub,
                tool.is_compound,
                tool.agent_id,
                tool.subagent_type,
                tool.subagent_description,
                tool.subagent_model,
                tool.skill_name,
            )
            for tool in batch
        ]
        table = pa.Table.from_arrays(
            [
                pa.array(column, type=field.type)
                for column, field in zip(zip(*rows, strict=True), schema, strict=True)
            ],
            schema=schema,
        )
        conn.register("_recall_tool_inserts", table)
        try:
            conn.execute(
                """
                INSERT INTO tool_calls (
                    id, session_id, message_id, idx, tool_name, tool_input,
                    bash_command, bash_base, bash_sub, is_compound,
                    agent_id, subagent_type, subagent_description, subagent_model, skill_name
                )
                SELECT id, session_id, message_id, idx, tool_name, tool_input,
                    bash_command, bash_base, bash_sub, is_compound,
                    agent_id, subagent_type, subagent_description, subagent_model, skill_name
                FROM _recall_tool_inserts
                """
            )
        finally:
            conn.unregister("_recall_tool_inserts")


def insert_message_embeddings(
    conn: duckdb.DuckDBPyConnection,
    rows: list[tuple[str, list[float] | None, list[float] | None]],
) -> None:
    """Bulk-insert embedding rows, skipping any that already exist.

    Transfer typed Arrow buffers instead of binding every vector element as a
    separate Python scalar. Arrow is a required storage dependency.
    """
    if not rows:
        return
    import pyarrow as pa

    ids = [r[0] for r in rows]
    content_embs = [r[1] for r in rows]
    thinking_embs = [r[2] for r in rows]
    # Variable-length list type is fastest — DuckDB ingests the Arrow
    # buffer directly without per-element fixed-size validation.
    # Use local variable name for DuckDB's automatic replacement scan.
    _emb_batch = pa.table(
        {
            "message_id": pa.array(ids, type=pa.string()),
            "content_embedding": pa.array(content_embs, type=pa.list_(pa.float32())),
            "thinking_embedding": pa.array(thinking_embs, type=pa.list_(pa.float32())),
        }
    )
    conn.execute(
        "INSERT OR IGNORE INTO message_embeddings SELECT * FROM _emb_batch",
    )


def insert_tool_call_embeddings(
    conn: duckdb.DuckDBPyConnection,
    rows: list[tuple[str, list[float] | None]],
) -> None:
    """Insert new or changed tool-call embeddings, skipping identical re-inserts.

    Only rows that are absent or whose stored vector actually differs are
    written; an unchanged re-insert is a no-op. A plain ``INSERT OR REPLACE``
    rewrote every row on every call (REPLACE == DELETE+INSERT in DuckDB), so
    each watch-mode re-index that re-presented a session's carried-forward
    embeddings left a fresh dead row-version. DuckDB reclaims a row group only
    once all its rows are dead, so those versions accumulated unbounded —
    ``tool_call_embeddings`` reached ~115 physical versions per live row
    (140 GiB of dead space) while ``message_embeddings`` stayed dense because
    its insert uses ``INSERT OR IGNORE``. The ``IS DISTINCT FROM`` guard keeps
    genuine updates (a changed vector still overwrites) without the churn.

    Transfer typed Arrow buffers through the same required storage dependency
    as raw writes. Retain the change guard until the current DuckDB engine
    passes test_tool_call_embedding_churn with an unconditional replacement.
    """
    if not rows:
        return
    selected, parameters = id_set_predicate("tool_call_id", [row[0] for row in rows])
    dedup_insert = f"""
        INSERT OR REPLACE INTO tool_call_embeddings
        SELECT b.tool_call_id, b.bash_embedding
        FROM _tc_emb_batch b
        LEFT JOIN (
            SELECT tool_call_id, bash_embedding FROM tool_call_embeddings WHERE {selected}
        ) e ON e.tool_call_id = b.tool_call_id
        WHERE e.tool_call_id IS NULL
           OR e.bash_embedding IS DISTINCT FROM b.bash_embedding
    """
    import pyarrow as pa

    ids = [r[0] for r in rows]
    embs = [r[1] for r in rows]
    _tc_emb_batch = pa.table(
        {
            "tool_call_id": pa.array(ids, type=pa.string()),
            "bash_embedding": pa.array(embs, type=pa.list_(pa.float32())),
        }
    )
    conn.execute(dedup_insert, parameters)


def fill_missing_message_embeddings(
    conn: duckdb.DuckDBPyConnection,
    rows: list[tuple[str, list[float] | None, list[float] | None, bool]],
) -> None:
    """Ordinary recovery fills NULL fields; concurrent non-NULL fields win.

    The caller holds the writer transaction and has validated input generations.
    None means no field publication authority. Marker creation is explicit.
    """
    if not rows:
        return
    import pyarrow as pa

    _missing_message_batch = pa.table(
        {
            "id": pa.array([r[0] for r in rows], type=pa.string()),
            "content": pa.array([r[1] for r in rows], type=pa.list_(pa.float32())),
            "thinking": pa.array([r[2] for r in rows], type=pa.list_(pa.float32())),
            "marker": pa.array([r[3] for r in rows], type=pa.bool_()),
        }
    )
    conn.execute(
        "INSERT OR IGNORE INTO message_embeddings "
        "SELECT id, content, thinking FROM _missing_message_batch WHERE marker"
    )
    selected, params = id_set_predicate("e.message_id", [row[0] for row in rows])
    for field in ("content", "thinking"):
        conn.execute(
            f"UPDATE message_embeddings AS e SET {field}_embedding = b.{field} "
            f"FROM _missing_message_batch b WHERE e.message_id=b.id AND {selected} "
            f"AND e.{field}_embedding IS NULL AND b.{field} IS NOT NULL",
            params,
        )


def fill_missing_tool_embeddings(
    conn: duckdb.DuckDBPyConnection,
    rows: list[tuple[str, list[float] | None, bool]],
) -> None:
    """Fill selected missing bash fields without replacing any valid vector."""
    if not rows:
        return
    import pyarrow as pa

    _missing_tool_batch = pa.table(
        {
            "id": pa.array([r[0] for r in rows], type=pa.string()),
            "bash": pa.array([r[1] for r in rows], type=pa.list_(pa.float32())),
            "marker": pa.array([r[2] for r in rows], type=pa.bool_()),
        }
    )
    conn.execute(
        "INSERT OR IGNORE INTO tool_call_embeddings "
        "SELECT id, bash FROM _missing_tool_batch WHERE marker"
    )
    selected, params = id_set_predicate("e.tool_call_id", [row[0] for row in rows])
    conn.execute(
        "UPDATE tool_call_embeddings AS e SET bash_embedding = b.bash "
        f"FROM _missing_tool_batch b WHERE e.tool_call_id=b.id AND {selected} "
        "AND e.bash_embedding IS NULL AND b.bash IS NOT NULL",
        params,
    )
