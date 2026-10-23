"""Prove transcript moves without confusing them with independent copies.

Path hashes allocate session identities; a proven move keeps the departed
session's identity. Inode, native identity and first timestamp only bound the
candidate set. Absence, committed-prefix bytes and an unambiguous holder prove
a move. Existing duplicate copies reconcile into the departed namespace with
all derived state and catalog ownership in the same transaction.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

import duckdb

from recall.core.ids import message_id, tool_call_id
from recall.core.transcripts import transcript_file, transcript_identity
from recall.core.types import UNATTRIBUTED_HOST
from recall.db._predicate import delete_session
from recall.db.queries import enqueue_session_sidecar_deletes

__all__ = [
    "MovedDuplicate",
    "VanishedMoves",
    "apply_moved_duplicates",
    "find_moved_predecessors",
    "plan_moved_duplicates",
    "plan_vanished_moves",
    "supersede",
    "supersede_moved_duplicate",
]

# Departed rows sharing a transcript's first timestamp; more than this is not a
# move history, and none of them is superseded.
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
    started_at: datetime | None
    committed_offset: int | None
    committed_prefix_sha256: str | None
    source_session_id: str | None
    host: str | None


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
    started_at: datetime | None,
    host: str,
    indexed_bytes: int,
    source_session_id: str | None = None,
) -> tuple[str, ...]:
    """Return the ids of rows the transcript at `source_path` provably continues.

    `started_at` is the successor's first timestamp.  `indexed_bytes` is how
    much of the file the successor's rows were built from.  A predecessor that
    committed more than that holds history the successor does not, however the
    file on disk has grown since.
    """
    try:
        signature = transcript_file(Path(source_path)).stat()
    except OSError:
        return ()
    # Native identity handles cross-device moves without timestamps; inode
    # identity also handles sources whose native id came from the filename.
    # Both are candidate keys only: absence and committed bytes decide proof.
    values = conn.execute(
        _ROW_SELECT + " AND s.source = ? AND ("
        " (f.dev = ? AND f.inode = ?) OR (s.source_session_id = ?"
        " AND (ss.started_at IS NULL OR CAST(? AS TIMESTAMP) IS NULL))"
        " OR ss.started_at = CAST(? AS TIMESTAMP)) LIMIT ?",
        [
            False,
            host,
            UNATTRIBUTED_HOST,
            source,
            signature.st_dev,
            signature.st_ino,
            source_session_id,
            started_at,
            started_at,
            2 * _PREDECESSORS_MAX + 1,
        ],
    ).fetchall()
    if len(values) > 2 * _PREDECESSORS_MAX:
        return ()
    rows = [_row(value) for value in values]
    group = _MoveGroup.of(
        [row for row in rows if row.source_path != source_path], pending_successor=True
    )
    # The pending insert is another survivor. A timestamp collision cannot
    # authorize removing a predecessor that another committed copy also holds.
    if group is None:
        return ()
    proven = tuple(
        row.session_id
        for row in group.departed
        if row.committed_offset is not None
        and row.committed_offset <= indexed_bytes
        and _is_proven_move(row, successor_path=source_path)
        and not any(_survivor_holds(survivor, row) for survivor in group.survivors)
    )
    # Several departed byte-identical copies do not identify which one moved.
    return proven if len(proven) == 1 else ()


def _has_departed(source_path: str) -> bool:
    """Whether a recorded path no longer names a file of its own.

    Indexing records resolved paths, so a recorded path that is gone, or that
    now resolves somewhere else (its root replaced by a symlink to where the
    transcripts went), names no transcript of its own any more.
    """
    try:
        return str(transcript_identity(Path(source_path))) != source_path
    except FileNotFoundError:
        return True
    except OSError:
        # Unreadable bytes do not prove that the original has departed.
        return False


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
    with the same terms `db supersede-moved` uses.  Each proof re-reads a
    survivor's committed prefix, and a vanished row is checked against every
    survivor sharing its first timestamp.  Rows stop being examined once
    `proofs_max` or `proof_bytes_max` is spent (the last row examined may
    overrun the byte budget, so an oversized transcript still progresses) and
    the rest are deferred. A row is deferred when its complete set of survivor comparisons
    would exceed the remaining proof count.
    """
    assert len(vanished_paths) <= _VANISHED_PAGE_MAX
    assert proofs_max >= 0
    indexed = {
        str(row[0]): (row[1], row[2])
        for row in conn.execute(
            "SELECT s.source_path, ss.started_at, s.source_session_id FROM sessions s"
            " JOIN session_state ss ON ss.session_id = s.id"
            " WHERE s.source = ? AND s.source_path IN (SELECT UNNEST(?))",
            [source, vanished_paths],
        ).fetchall()
    }
    if not indexed or proofs_max == 0 or proof_bytes_max <= 0:
        return VanishedMoves(plans=(), deferred_paths=frozenset(indexed), proofs=0, proof_bytes=0)
    started_ats = sorted({stamp for stamp, _ in indexed.values() if stamp is not None})
    native_ids = sorted(
        {str(native) for stamp, native in indexed.values() if stamp is None and native}
    )
    groups = _groups(
        _rows_started_at(
            conn, source=source, started_ats=started_ats, host=host, native_ids=native_ids
        )
    )
    keys = set(groups)
    plans: list[MovedDuplicate] = []
    deferred: set[str] = set()
    proofs = 0
    proof_bytes = 0
    for key in sorted(keys):
        group = _MoveGroup.of(groups.get(key, []))
        if group is None:
            continue
        for row in group.departed:
            if row.source_path not in indexed:
                continue
            if proofs + len(group.survivors) > proofs_max or proof_bytes >= proof_bytes_max:
                deferred.add(row.source_path)
                continue
            proofs += len(group.survivors)
            proof_bytes += sum(survivor.committed_offset or 0 for survivor in group.survivors)
            holder = group.sole_holder(row)
            if holder is not None:
                plans.append(_moved(row, holder))
    return VanishedMoves(
        plans=tuple(plans),
        deferred_paths=frozenset(deferred),
        proofs=proofs,
        proof_bytes=proof_bytes,
    )


def _groups(rows: Iterable[_Row]) -> dict[tuple[str, str], list[_Row]]:
    """Candidate groups use native identity when a transcript has no timestamp."""
    groups: dict[tuple[str, str], list[_Row]] = defaultdict(list)
    for row in rows:
        if row.started_at is not None:
            groups[(row.source, "time:" + row.started_at.isoformat())].append(row)
        elif row.source_session_id:
            groups[(row.source, "native:" + row.source_session_id)].append(row)
    return groups


@dataclass(frozen=True)
class _MoveGroup:
    """Rows sharing a first timestamp, split by whether their path still names their file."""

    departed: tuple[_Row, ...]
    survivors: tuple[_Row, ...]

    @staticmethod
    def of(rows: list[_Row], *, pending_successor: bool = False) -> _MoveGroup | None:
        """The group, when it can be a move history: some of each, and few enough of each."""
        departed: list[_Row] = []
        survivors: list[_Row] = []
        for row in rows:
            (departed if _has_departed(row.source_path) else survivors).append(row)
        if not departed or not (survivors or pending_successor):
            return None
        if (
            len(departed) > _PREDECESSORS_MAX
            or len(survivors) + int(pending_successor) > _PREDECESSORS_MAX
        ):
            return None
        return _MoveGroup(departed=tuple(departed), survivors=tuple(survivors))

    def sole_holder(self, predecessor: _Row) -> _Row | None:
        """The one survivor holding everything the predecessor's rows do.

        Two survivors that both hold it are two copies, and prove nothing.
        """
        offsets = {row.committed_offset for row in self.departed}
        digests = {row.committed_prefix_sha256 for row in self.departed}
        # Equal-length, distinct committed prefixes cannot share a holder.
        # Otherwise several departed copies may claim one survivor; keep them
        # separate rather than let walk/page order choose an identity.
        if len(offsets) != 1 or len(digests) != len(self.departed):
            return None
        holders = [
            survivor for survivor in self.survivors if _survivor_holds(survivor, predecessor)
        ]
        return holders[0] if len(holders) == 1 else None


def _plan_group(rows: list[_Row]) -> list[MovedDuplicate]:
    """Plan each departed row of one group that exactly one survivor holds."""
    group = _MoveGroup.of(rows)
    if group is None:
        return []
    plans: list[MovedDuplicate] = []
    for row in group.departed:
        holder = group.sole_holder(row)
        if holder is not None:
            plans.append(_moved(row, holder))
    return plans


def _moved(predecessor: _Row, survivor: _Row) -> MovedDuplicate:
    return MovedDuplicate(
        predecessor_id=predecessor.session_id,
        predecessor_path=predecessor.source_path,
        successor_id=survivor.session_id,
        successor_path=survivor.source_path,
    )


ProofClass = Literal[
    "proven_move",
    "surviving_copy",
    "missing_catalog_proof",
    "prefix_limit_exceeded",
    "lagging_survivor",
    "unreadable_survivor",
    "divergent_content",
]


def _survivor_holds(survivor: _Row, predecessor: _Row) -> bool:
    """Whether the survivor's indexed rows contain everything the predecessor's do.

    The survivor's file must still begin with the bytes its rows were built
    from, and those bytes must extend the predecessor's committed prefix; the
    file alone proves nothing about rows that lag it.
    """
    return _survivor_proof(survivor, predecessor) == ("proven_move",)


def _survivor_proof(survivor: _Row, predecessor: _Row) -> tuple[ProofClass, ...]:
    """Return proof or retain reasons used by maintenance and reconciliation."""
    if not _has_departed(predecessor.source_path):
        return ("surviving_copy",)
    if (
        survivor.committed_offset is None
        or survivor.committed_prefix_sha256 is None
        or predecessor.committed_offset is None
        or predecessor.committed_prefix_sha256 is None
    ):
        return ("missing_catalog_proof",)
    if predecessor.committed_offset <= 0 or survivor.committed_offset <= 0:
        return ("missing_catalog_proof",)
    blockers: list[ProofClass] = []
    if max(predecessor.committed_offset, survivor.committed_offset) > _PREFIX_BYTES_MAX:
        blockers.append("prefix_limit_exceeded")
    if predecessor.committed_offset > survivor.committed_offset:
        blockers.append("lagging_survivor")
    if blockers:
        return tuple(blockers)
    digests = _prefix_digests(
        survivor.source_path, (predecessor.committed_offset, survivor.committed_offset)
    )
    if digests is None:
        return ("unreadable_survivor",)
    if digests != (predecessor.committed_prefix_sha256, survivor.committed_prefix_sha256):
        return ("divergent_content",)
    return ("proven_move",)


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
    if not _has_departed(row.source_path):
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
        with transcript_file(Path(path)).open("rb") as handle:
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
    SELECT s.id, s.source, s.source_path, ss.started_at,
           f.committed_offset, f.committed_prefix_sha256, s.source_session_id, ss.host
    FROM sessions s
    JOIN session_state ss ON ss.session_id = s.id
    LEFT JOIN source_files f ON f.source = s.source AND f.source_path = s.source_path
    WHERE (? OR ss.host IS NULL OR ss.host IN (?, ?))
"""


def _rows_started_at(
    conn: duckdb.DuckDBPyConnection,
    *,
    source: str,
    started_ats: list[datetime],
    host: str | None,
    native_ids: list[str] | None = None,
) -> list[_Row]:
    """Rows of `source` whose first timestamp is one of `started_ats`.

    The timestamps were read back from `session_state` or are about to be
    written to it, so they pass through the same conversion as the column.
    """
    if not started_ats and not native_ids:
        return []
    rows = conn.execute(
        _ROW_SELECT
        + " AND s.source = ? AND (ss.started_at IN (SELECT CAST(UNNEST(?) AS TIMESTAMP))"
        " OR (ss.started_at IS NULL AND s.source_session_id IN (SELECT UNNEST(?))))",
        [host is None, host, UNATTRIBUTED_HOST, source, started_ats, native_ids or []],
    ).fetchall()
    return [_row(values) for values in rows]


def _local_rows(conn: duckdb.DuckDBPyConnection, *, host: str) -> list[_Row]:
    return [
        _row(values)
        for values in conn.execute(
            _ROW_SELECT + " ORDER BY s.source, ss.started_at, s.id",
            [False, host, UNATTRIBUTED_HOST],
        ).fetchall()
    ]


def _row(values: tuple[object, ...]) -> _Row:
    started_at, offset, prefix_sha256 = values[3], values[4], values[5]
    return _Row(
        session_id=str(values[0]),
        source=str(values[1]),
        source_path=str(values[2]),
        started_at=started_at if isinstance(started_at, datetime) else None,
        committed_offset=int(offset) if isinstance(offset, int) else None,
        committed_prefix_sha256=str(prefix_sha256) if prefix_sha256 is not None else None,
        source_session_id=str(values[6]) if values[6] is not None else None,
        host=str(values[7]) if values[7] is not None else None,
    )


def apply_moved_duplicates(
    conn: duckdb.DuckDBPyConnection,
    plans: list[MovedDuplicate],
    *,
    queue_sidecar_deletes: bool,
) -> int:
    """Supersede each planned predecessor in its own transaction; return the count.

    A running daemon queues superseded keyword-search rows for deletion in
    the same transaction. Offline cleanup leaves deletion to startup membership
    reconciliation. Both paths queue retained identities for refresh.
    """
    for plan in plans:
        supersede_moved_duplicate(conn, plan, queue_sidecar_deletes=queue_sidecar_deletes)
    return len(plans)


def _supersede_moved_duplicate(
    conn: duckdb.DuckDBPyConnection, plan: MovedDuplicate, *, queue_sidecar_deletes: bool
) -> None:
    """Move the surviving history into the departed session's namespace atomically.

    A copy can have been indexed before its original disappeared. The original
    keeps its identity; the separately allocated copy id retires without an
    alias. All scalar state, vectors and tail facts move with their owner.
    """
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
            "UPDATE source_files SET session_id = ? WHERE source_path = ?",
            [plan.predecessor_id, plan.successor_path],
        )
        for kind, table in (("message", "messages"), ("tool_call", "tool_calls")):
            conn.execute(
                f"INSERT INTO fts_sidecar_pending(kind, id, op)"
                f" SELECT '{kind}', id, 'upsert' FROM {table} WHERE session_id = ?",
                [plan.predecessor_id],
            )
    except Exception:
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


def supersede_moved_duplicate(
    conn: duckdb.DuckDBPyConnection, plan: MovedDuplicate, *, queue_sidecar_deletes: bool
) -> None:
    from recall.db.semantic_work import publication

    publication(conn).move(plan, queue_sidecar_deletes=queue_sidecar_deletes)
