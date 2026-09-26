"""Recognize a transcript indexed again after its project directory moved.

A session's id hashes its absolute path, so renaming a project directory makes
the same transcript a new session while the row for the old path stays behind
with the same history.  The old row is superseded only when the move is proven:
its path is gone from disk and the new file begins with exactly the bytes the
catalog last committed for it.  Anything short of that proof keeps both rows.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import duckdb

from recall.core.ids import transcript_key
from recall.core.types import UNATTRIBUTED_HOST
from recall.db.queries import delete_session, enqueue_session_sidecar_deletes

__all__ = [
    "MovedDuplicate",
    "VanishedMoves",
    "apply_moved_duplicates",
    "find_moved_predecessors",
    "find_vanished_rows",
    "plan_moved_duplicates",
    "plan_vanished_moves",
    "supersede",
    "supersede_moved_duplicate",
]

# Vanished rows sharing a transcript's key; more than this is not a move
# history, and none of them is superseded.
_PREDECESSORS_MAX: Final = 16
# The largest committed prefix re-read to prove a continuation.  A longer one
# is left unproven rather than read in full.
_PREFIX_BYTES_MAX: Final = 512 * 1024 * 1024
_READ_CHUNK_BYTES: Final = 1024 * 1024
# The most vanished paths one call plans for: one catalog page.
_VANISHED_PAGE_MAX: Final = 256


@dataclass(frozen=True)
class _Row:
    session_id: str
    source: str
    source_path: str
    committed_offset: int | None
    committed_prefix_sha256: str | None


@dataclass(frozen=True)
class MovedDuplicate:
    """An indexed row whose transcript now lives at the successor's path."""

    predecessor_id: str
    predecessor_path: str
    successor_id: str
    successor_path: str


def find_moved_predecessors(
    conn: duckdb.DuckDBPyConnection,
    *,
    source: str,
    source_path: str,
    host: str,
    indexed_bytes: int,
) -> tuple[str, ...]:
    """Return the ids of rows the transcript at `source_path` provably continues.

    `indexed_bytes` is how much of the file the successor's rows were built
    from.  A predecessor that committed more than that holds history the
    successor does not, however the file on disk has grown since.
    """
    vanished = _vanished_rows(conn, source=source, source_path=source_path, host=host)
    if len(vanished) > _PREDECESSORS_MAX:
        return ()
    return tuple(
        row.session_id
        for row in vanished
        if row.committed_offset is not None
        and row.committed_offset <= indexed_bytes
        and _is_proven_move(row, successor_path=source_path)
    )


def find_vanished_rows(
    conn: duckdb.DuckDBPyConnection, *, source: str, source_path: str
) -> tuple[str, ...]:
    """Ids of rows of any host whose transcript key matches and whose path is gone."""
    vanished = _vanished_rows(conn, source=source, source_path=source_path, host=None)
    if len(vanished) > _PREDECESSORS_MAX:
        return ()
    return tuple(row.session_id for row in vanished)


def _vanished_rows(
    conn: duckdb.DuckDBPyConnection, *, source: str, source_path: str, host: str | None
) -> list[_Row]:
    key = transcript_key(source_path)
    return [
        row
        for row in _rows_ending_in(conn, source=source, key=key, host=host)
        if row.source_path != source_path
        and transcript_key(row.source_path) == key
        and not os.path.lexists(row.source_path)
    ]


def plan_moved_duplicates(conn: duckdb.DuckDBPyConnection, *, host: str) -> list[MovedDuplicate]:
    """Pair every provable predecessor on this host with its surviving row."""
    plans: list[MovedDuplicate] = []
    for rows in _groups(_local_rows(conn, host=host)).values():
        plans.extend(_plan_group(rows))
    return plans


@dataclass(frozen=True)
class VanishedMoves:
    """What one page of vanished paths proved, within a budget of proofs."""

    plans: tuple[MovedDuplicate, ...]
    # Paths with a candidate successor left unexamined because the budget ran out.
    deferred_paths: frozenset[str]
    proofs: int
    # Bytes of successor files the proofs covered, which also bounds the rows
    # the plans rewrite.
    proof_bytes: int


def plan_vanished_moves(
    conn: duckdb.DuckDBPyConnection,
    *,
    source: str,
    vanished_paths: list[str],
    host: str,
    proofs_max: int,
    proof_bytes_max: int,
) -> VanishedMoves:
    """Pair the rows at just-vanished paths with the one copy that provably holds them.

    A copy indexed while its original still existed proved nothing then; the
    original's disappearance is what completes the proof, so it is re-run here
    with the same terms `db supersede-moved` uses.  Each proof re-reads the
    successor's committed prefix, so proofs stop once `proofs_max` or
    `proof_bytes_max` is spent (the last may overrun the byte budget, so an
    oversized transcript still progresses) and the rest are deferred.
    """
    assert len(vanished_paths) <= _VANISHED_PAGE_MAX
    assert proofs_max >= 0
    indexed = {
        str(row[0])
        for row in conn.execute(
            "SELECT source_path FROM sessions"
            " WHERE source = ? AND source_path IN (SELECT UNNEST(?))",
            [source, vanished_paths],
        ).fetchall()
    }
    if not indexed or proofs_max == 0 or proof_bytes_max <= 0:
        return VanishedMoves(plans=(), deferred_paths=frozenset(indexed), proofs=0, proof_bytes=0)
    keys = {(source, transcript_key(path)) for path in indexed}
    groups = _groups(row for row in _local_rows(conn, host=host) if row.source == source)
    plans: list[MovedDuplicate] = []
    deferred: set[str] = set()
    proofs = 0
    proof_bytes = 0
    for key in sorted(keys):
        rows = groups.get(key, [])
        survivor = _single_survivor(rows)
        if survivor is None:
            continue
        for row in rows:
            if row.source_path not in indexed:
                continue
            if proofs == proofs_max or proof_bytes >= proof_bytes_max:
                deferred.add(row.source_path)
                continue
            proofs += 1
            proof_bytes += survivor.committed_offset or 0
            if _survivor_holds(survivor, row):
                plans.append(_moved(row, survivor))
    return VanishedMoves(
        plans=tuple(plans),
        deferred_paths=frozenset(deferred),
        proofs=proofs,
        proof_bytes=proof_bytes,
    )


def _groups(rows: Iterable[_Row]) -> dict[tuple[str, str], list[_Row]]:
    groups: dict[tuple[str, str], list[_Row]] = defaultdict(list)
    for row in rows:
        groups[(row.source, transcript_key(row.source_path))].append(row)
    return groups


def _single_survivor(rows: list[_Row]) -> _Row | None:
    """The one row of a transcript key still on disk, when its group can be a move history."""
    survivors = [row for row in rows if os.path.lexists(row.source_path)]
    if len(rows) < 2 or len(survivors) != 1 or len(rows) - 1 > _PREDECESSORS_MAX:
        return None
    return survivors[0]


def _plan_group(rows: list[_Row]) -> list[MovedDuplicate]:
    """Plan the rows of one transcript key that its single surviving row holds."""
    survivor = _single_survivor(rows)
    if survivor is None:
        return []
    return [
        _moved(row, survivor)
        for row in rows
        if row is not survivor and _survivor_holds(survivor, row)
    ]


def _moved(predecessor: _Row, survivor: _Row) -> MovedDuplicate:
    return MovedDuplicate(
        predecessor_id=predecessor.session_id,
        predecessor_path=predecessor.source_path,
        successor_id=survivor.session_id,
        successor_path=survivor.source_path,
    )


def _survivor_holds(survivor: _Row, predecessor: _Row) -> bool:
    """Whether the survivor's indexed rows contain everything the predecessor's do.

    The survivor's file must still begin with the bytes its rows were built
    from, and those bytes must extend the predecessor's committed prefix; the
    file alone proves nothing about rows that lag it.
    """
    if os.path.lexists(predecessor.source_path):
        return False
    if (
        survivor.committed_offset is None
        or survivor.committed_prefix_sha256 is None
        or predecessor.committed_offset is None
        or predecessor.committed_prefix_sha256 is None
        or predecessor.committed_offset > survivor.committed_offset
    ):
        return False
    digests = _prefix_digests(
        survivor.source_path, (predecessor.committed_offset, survivor.committed_offset)
    )
    return digests == (predecessor.committed_prefix_sha256, survivor.committed_prefix_sha256)


def supersede(
    conn: duckdb.DuckDBPyConnection,
    *,
    predecessor_ids: tuple[str, ...],
    successor_id: str,
    queue_sidecar_deletes: bool,
) -> None:
    """Remove superseded rows and point their references at the successor.

    Runs inside the caller's transaction, so the successor's insert and the
    predecessors' removal commit or roll back together.  Their keyword-search
    rows are never deleted here, since that would outlive a rollback: a running
    daemon queues the deletes with this transaction, and otherwise the
    daemon's startup reconciliation removes rows whose ids are gone.
    """
    for predecessor_id in predecessor_ids:
        if queue_sidecar_deletes:
            enqueue_session_sidecar_deletes(conn, predecessor_id)
        conn.execute(
            "UPDATE usage_events SET session_id = ? WHERE session_id = ?",
            [successor_id, predecessor_id],
        )
        conn.execute(
            "UPDATE source_files SET session_id = NULL WHERE session_id = ?",
            [predecessor_id],
        )
        delete_session(conn, predecessor_id)


def _is_proven_move(row: _Row, *, successor_path: str) -> bool:
    if os.path.lexists(row.source_path):
        return False
    if row.committed_offset is None or row.committed_prefix_sha256 is None:
        return False
    return _begins_with(successor_path, row.committed_offset, row.committed_prefix_sha256)


def _begins_with(path: str, offset: int, prefix_sha256: str) -> bool:
    """Whether the file's first `offset` bytes hash to `prefix_sha256`."""
    return _prefix_digests(path, (offset,)) == (prefix_sha256,)


def _prefix_digests(path: str, offsets: tuple[int, ...]) -> tuple[str, ...] | None:
    """SHA-256 of the file's first `offset` bytes for each ascending offset.

    None when an offset is out of bounds, the file is shorter, or unreadable:
    an unproven move, never an error.
    """
    assert list(offsets) == sorted(offsets)
    if offsets[0] <= 0 or offsets[-1] > _PREFIX_BYTES_MAX:
        return None
    digest = hashlib.sha256()
    digests: list[str] = []
    position = 0
    try:
        with open(path, "rb") as handle:
            for offset in offsets:
                while position < offset:
                    chunk = handle.read(min(offset - position, _READ_CHUNK_BYTES))
                    if not chunk:
                        return None
                    digest.update(chunk)
                    position += len(chunk)
                digests.append(digest.copy().hexdigest())
    except OSError:
        return None
    return tuple(digests)


_ROW_SELECT: Final = """
    SELECT s.id, s.source, s.source_path, f.committed_offset, f.committed_prefix_sha256
    FROM sessions s
    JOIN session_state ss ON ss.session_id = s.id
    LEFT JOIN source_files f ON f.source = s.source AND f.source_path = s.source_path
    WHERE (? OR ss.host IS NULL OR ss.host IN (?, ?))
"""


def _rows_ending_in(
    conn: duckdb.DuckDBPyConnection,
    *,
    source: str,
    key: str,
    host: str | None,
) -> list[_Row]:
    rows = conn.execute(
        _ROW_SELECT + " AND s.source = ? AND s.source_path LIKE ? ESCAPE '\\'",
        [host is None, host, UNATTRIBUTED_HOST, source, "%/" + _like_literal(key)],
    ).fetchall()
    return [_row(values) for values in rows]


def _local_rows(conn: duckdb.DuckDBPyConnection, *, host: str) -> list[_Row]:
    return [
        _row(values)
        for values in conn.execute(_ROW_SELECT, [False, host, UNATTRIBUTED_HOST]).fetchall()
    ]


def _row(values: tuple[object, ...]) -> _Row:
    offset = values[3]
    return _Row(
        session_id=str(values[0]),
        source=str(values[1]),
        source_path=str(values[2]),
        committed_offset=int(offset) if isinstance(offset, int) else None,
        committed_prefix_sha256=str(values[4]) if values[4] is not None else None,
    )


def _like_literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def apply_moved_duplicates(
    conn: duckdb.DuckDBPyConnection,
    plans: list[MovedDuplicate],
    *,
    queue_sidecar_deletes: bool,
) -> int:
    """Supersede each planned predecessor in its own transaction; return the count.

    A running daemon queues the superseded keyword-search rows for deletion in
    the same transaction.  `db supersede-moved` runs with the daemon stopped,
    which reconciles keyword search when it starts, so it queues none: hundreds
    of thousands of queue rows only delay that start, and an interrupted bulk
    insert can leave the queue's index unable to delete them.
    """
    for plan in plans:
        supersede_moved_duplicate(conn, plan, queue_sidecar_deletes=queue_sidecar_deletes)
    return len(plans)


def supersede_moved_duplicate(
    conn: duckdb.DuckDBPyConnection, plan: MovedDuplicate, *, queue_sidecar_deletes: bool
) -> None:
    """Carry the predecessor's stored context to the survivor and remove it, atomically."""
    conn.execute("BEGIN")
    try:
        _carry_stored_contexts(
            conn, predecessor_id=plan.predecessor_id, successor_id=plan.successor_id
        )
        supersede(
            conn,
            predecessor_ids=(plan.predecessor_id,),
            successor_id=plan.successor_id,
            queue_sidecar_deletes=queue_sidecar_deletes,
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


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
        """
        CREATE OR REPLACE TEMP TABLE carried_contexts AS
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
        """,
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
