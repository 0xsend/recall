"""Exact file-local response state owned by the session publishing transaction.

These operations never begin or commit a transaction. The content writer calls
these only inside its KeywordTurn transaction, before its acknowledgement.
"""

from __future__ import annotations

from dataclasses import replace

import duckdb

from recall.core.models import (
    ClaudeCounters,
    ClaudeUsageFull,
    ClaudeUsageSuffix,
    NormalizationCheckpoint,
    ParseResult,
)
from recall.db.queries import _insert_arrow_rows

_COLUMNS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


class UnprovedClaudeScope(RuntimeError):
    """A suffix must be prepared again from the full source, outside the writer."""


def response_key(scope_token: str, response_id: str) -> str:
    """An exact length-framed key, with no delimiter or hash collision."""
    return f"{len(scope_token)}:{scope_token}{response_id}"


def delete_scope(conn: duckdb.DuckDBPyConnection, session_id: str) -> None:
    conn.execute(
        "DELETE FROM claude_response_usage WHERE scope_token IN "
        "(SELECT scope_token FROM claude_usage_scopes WHERE session_id = ?)",
        [session_id],
    )
    conn.execute("DELETE FROM claude_usage_scopes WHERE session_id = ?", [session_id])


def _insert_rows(conn: duckdb.DuckDBPyConnection, rows: list[tuple]) -> None:
    import pyarrow as pa

    schema = pa.schema(
        [("response_key", pa.string()), ("scope_token", pa.string())]
        + [(column, pa.int64()) for column in _COLUMNS]
    )
    _insert_arrow_rows(
        conn,
        rows,
        schema,
        "INSERT INTO claude_response_usage SELECT * FROM _recall_insert_rows",
    )


def replace_full(conn: duckdb.DuckDBPyConnection, session_id: str, result: ParseResult) -> None:
    projection = result.claude_usage
    assert isinstance(projection, ClaudeUsageFull)
    assert result.committed_prefix_sha256 is not None
    # Unlike a sequence, this allocator rolls back with a failed publication.
    allocation = conn.execute(
        "UPDATE runtime_state SET claude_usage_incarnation = "
        "COALESCE(claude_usage_incarnation, 0) + 1 WHERE singleton "
        "RETURNING claude_usage_incarnation"
    ).fetchone()
    assert allocation is not None
    token = f"{len(session_id)}:{session_id}:{allocation[0]}"
    delete_scope(conn, session_id)
    conn.execute(
        "INSERT INTO claude_usage_scopes VALUES (?, ?, 1, ?, ?)",
        [session_id, token, result.next_byte_offset, result.committed_prefix_sha256],
    )
    _insert_rows(
        conn,
        [
            (response_key(token, response.response_id), token, *response.counters.cells())
            for response in projection.responses
        ],
    )


def resolve_suffix(
    conn: duckdb.DuckDBPyConnection, session_id: str, result: ParseResult
) -> tuple[int | None, int | None]:
    patch = result.claude_usage
    assert isinstance(patch, ClaudeUsageSuffix)
    scope = conn.execute(
        "SELECT scope_token, format_revision, committed_offset, prefix_sha256 "
        "FROM claude_usage_scopes WHERE session_id = ?",
        [session_id],
    ).fetchone()
    expected = (patch.scope_token, 1, patch.base_offset, patch.base_prefix_sha256)
    if patch.scope_token is None or scope != expected:
        raise UnprovedClaudeScope("Claude usage scope does not cover the prepared prefix")
    assert result.committed_prefix_sha256 is not None
    totals = list(patch.anonymous.cells())
    inserted = []
    for response in patch.responses:
        key = response_key(patch.scope_token, response.response_id)
        # Single-column equality uses the primary-key ART. A bounded series of
        # point probes avoids a batched join's global materialization/scan.
        row = conn.execute(
            "SELECT input_tokens, cache_creation_input_tokens, cache_read_input_tokens, "
            "output_tokens FROM claude_response_usage WHERE response_key = ?",
            [key],
        ).fetchone()
        old = ClaudeCounters(*row) if row is not None else ClaudeCounters()
        current = old.patched(response.counters)
        changed = []
        for index, (before, after, supplied) in enumerate(
            zip(old.cells(), current.cells(), response.counters.cells(), strict=True)
        ):
            if supplied is not None:
                totals[index] = (totals[index] or 0) + supplied - (before or 0)
            if before != after:
                changed.append((_COLUMNS[index], after))
        if row is None:
            inserted.append((key, patch.scope_token, *current.cells()))
        elif changed:
            # Never name immutable indexed keys, even with their current values.
            assignments = ", ".join(f"{column} = ?" for column, _ in changed)
            conn.execute(
                f"UPDATE claude_response_usage SET {assignments} WHERE response_key = ?",
                [*(value for _, value in changed), key],
            )
    _insert_rows(conn, inserted)
    conn.execute(
        "UPDATE claude_usage_scopes SET committed_offset = ?, prefix_sha256 = ? "
        "WHERE session_id = ?",
        [result.next_byte_offset, result.committed_prefix_sha256, session_id],
    )
    inputs = [value for value in totals[:3] if value is not None]
    return (sum(inputs) if inputs else None, totals[3])


def bind_checkpoint(
    conn: duckdb.DuckDBPyConnection, session_id: str, checkpoint: NormalizationCheckpoint
) -> NormalizationCheckpoint:
    scope = conn.execute(
        "SELECT scope_token, format_revision, committed_offset, prefix_sha256 "
        "FROM claude_usage_scopes WHERE session_id = ?",
        [session_id],
    ).fetchone()
    assert scope is not None and scope[1:] == (1, checkpoint.offset, checkpoint.prefix_sha256)
    return replace(
        checkpoint,
        adapter_state={
            **checkpoint.adapter_state,
            "usage_scope": {"format": 1, "token": scope[0]},
        },
    )
