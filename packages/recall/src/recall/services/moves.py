"""Prove transcript moves without confusing them with independent copies.

Path hashes allocate session identities; a proven move keeps the departed
session's identity. Inode, native identity and first timestamp only bound the
candidate set. Absence, committed-prefix bytes and an unambiguous holder prove
a move. Existing duplicate copies reconcile into the departed namespace with
all derived state and catalog ownership in the same transaction.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal, cast

import duckdb

from recall.core.models import ParseResult
from recall.core.transcripts import transcript_file, transcript_identity
from recall.core.types import UNATTRIBUTED_HOST
from recall.db.queries import delete_session, enqueue_session_sidecar_deletes
from recall.services.transfers import BindingSnapshot, TransferCatalog

__all__ = [
    "MovedDuplicate",
    "VanishedMoves",
    "find_moved_predecessors",
    "plan_moved_duplicates",
    "plan_vanished_moves",
    "supersede",
]

# Maximum candidates on either side of a complete content-proof comparison.
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
    snapshot: BindingSnapshot | None


@dataclass(frozen=True)
class MovedDuplicate:
    """An indexed row whose transcript now lives at the successor's path."""

    predecessor_id: str
    predecessor_path: str
    successor_id: str
    successor_path: str
    predecessor_snapshot: BindingSnapshot
    successor_snapshot: BindingSnapshot
    hashed_identity: tuple[int, int]


def find_moved_predecessors(
    conn: duckdb.DuckDBPyConnection,
    capture: ParseResult,
    *,
    host: str,
    transfers: TransferCatalog | None = None,
) -> tuple[str, ...]:
    """Return identities uniquely continued by the bytes captured for publication.

    Every successor prefix used for proof is read together with the committed
    boundary and verified against the parser's digest. Later path mutations
    cannot authorize adoption for bytes outside that capture.
    """
    session = capture.session
    source = session.source.value
    source_path = session.source_path
    source_session_id = session.source_session_id
    started_at = session.started_at
    indexed_bytes = capture.next_byte_offset
    if capture.committed_prefix_sha256 is None:
        return ()
    try:
        signature = transcript_file(Path(source_path)).stat()
    except OSError:
        return ()
    # Native identity handles cross-device moves without timestamps; inode
    # identity also handles sources whose native id came from the filename.
    # Both are candidate keys only: absence and committed bytes decide proof.
    candidates = (
        _row_select(transfers) + " AND s.source = ? AND ("
        " (f.dev = ? AND f.inode = ?) OR (s.source_session_id = ?"
        " AND (ss.started_at IS NULL OR CAST(? AS TIMESTAMP) IS NULL))"
        " OR ss.started_at = CAST(? AS TIMESTAMP)) AND s.source_path <> ?"
    )
    parameters = [
        False,
        host,
        UNATTRIBUTED_HOST,
        source,
        signature.st_dev,
        signature.st_ino,
        source_session_id,
        started_at,
        started_at,
        source_path,
    ]
    limit = 2 * _PREDECESSORS_MAX + 1
    values = conn.execute(candidates + " LIMIT ?", [*parameters, limit]).fetchall()
    group = _MoveGroup.partition([_row(value) for value in values])
    exact_boundary = len(values) > 2 * _PREDECESSORS_MAX or group.overflows(pending_successor=True)
    if exact_boundary:
        # Only an equal-boundary mismatch rules out both an exact predecessor
        # and a committed holder of its full prefix. Unknown and unequal-length
        # histories must remain in the complete, bounded comparison set.
        values = conn.execute(
            candidates + " AND (f.committed_offset IS NULL OR f.committed_offset <> ?"
            " OR f.committed_prefix_sha256 IS NULL OR f.committed_prefix_sha256 = ?) LIMIT ?",
            [*parameters, indexed_bytes, capture.committed_prefix_sha256, limit],
        ).fetchall()
        if len(values) > 2 * _PREDECESSORS_MAX:
            return ()
        group = _MoveGroup.partition([_row(value) for value in values])
    # The pending insert is another survivor. A timestamp collision cannot
    # authorize removing a predecessor that another committed copy also holds.
    if not group.departed or group.overflows(pending_successor=True):
        return ()
    # Hash all eligible predecessor boundaries in one pass. The final digest
    # binds every shorter prefix to the same captured bytes as the SQL filter.
    offsets = tuple(
        sorted(
            {
                indexed_bytes,
                *(
                    row.committed_offset
                    for row in group.departed
                    if row.committed_offset is not None and row.committed_offset <= indexed_bytes
                ),
            }
        )
    )
    digests = _prefix_digests(source_path, offsets)
    if digests is None or digests.digests[-1] != capture.committed_prefix_sha256:
        return ()
    prefixes = dict(zip(offsets, digests.digests, strict=True))
    proven = tuple(
        row
        for row in group.departed
        if row.committed_offset is not None
        and row.committed_offset <= indexed_bytes
        and row.committed_prefix_sha256 is not None
        and _has_departed(row.source_path)
        and prefixes[row.committed_offset] == row.committed_prefix_sha256
        and not any(_survivor_holds(survivor, row) for survivor in group.survivors)
    )
    # Several departed byte-identical copies do not identify which one moved.
    if len(proven) != 1:
        return ()
    predecessor = proven[0]
    # Shorter departed prefixes still veto uniqueness, but the overflow
    # fallback may only adopt an identity at the verified exact boundary.
    if exact_boundary and predecessor.committed_offset != indexed_bytes:
        return ()
    return (predecessor.session_id,)


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


def plan_moved_duplicates(
    conn: duckdb.DuckDBPyConnection, *, host: str, transfers: TransferCatalog | None = None
) -> list[MovedDuplicate]:
    """Pair every provable predecessor on this host with its surviving row."""
    plans: list[MovedDuplicate] = []
    for rows in _groups(_local_rows(conn, host=host, transfers=transfers)).values():
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
    transfers: TransferCatalog | None = None,
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
            conn,
            source=source,
            started_ats=started_ats,
            host=host,
            native_ids=native_ids,
            transfers=transfers,
        )
    )
    keys = set(groups)
    plans: list[MovedDuplicate] = []
    deferred: set[str] = set()
    proofs = 0
    proof_bytes = 0
    for key in sorted(keys):
        candidates = _MoveCandidates.of(groups[key])
        for row in candidates.group.departed:
            if row.source_path not in indexed:
                continue
            group = candidates.for_predecessor(row)
            if group is None:
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
    exact_boundary: int | None = None

    @staticmethod
    def partition(rows: list[_Row]) -> _MoveGroup:
        departed: list[_Row] = []
        survivors: list[_Row] = []
        for row in rows:
            (departed if _has_departed(row.source_path) else survivors).append(row)
        return _MoveGroup(departed=tuple(departed), survivors=tuple(survivors))

    def overflows(self, *, pending_successor: bool = False) -> bool:
        return (
            len(self.departed) > _PREDECESSORS_MAX
            or len(self.survivors) + int(pending_successor) > _PREDECESSORS_MAX
        )

    def sole_holder(self, predecessor: _Row) -> _Holder | None:
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
        holders: list[_Holder] = []
        for survivor in self.survivors:
            proof = _survivor_proof_result(survivor, predecessor)
            if proof.reasons == ("proven_move",):
                assert proof.identity is not None
                holders.append(_Holder(survivor, proof.identity))
        if len(holders) != 1:
            return None
        holder = holders[0]
        if self.exact_boundary is not None and holder.row.committed_offset != self.exact_boundary:
            return None
        return holder


@dataclass(frozen=True)
class _BoundaryCandidates:
    """Count exclusions before materializing at most sixteen retained rows."""

    count: int
    offsets: dict[int | None, dict[str | None, list[_Row]]]
    offset_counts: dict[int | None, int]

    @staticmethod
    def of(rows: tuple[_Row, ...]) -> _BoundaryCandidates:
        offsets: dict[int | None, dict[str | None, list[_Row]]] = {}
        counts: dict[int | None, int] = defaultdict(int)
        for row in rows:
            offsets.setdefault(row.committed_offset, {}).setdefault(
                row.committed_prefix_sha256, []
            ).append(row)
            counts[row.committed_offset] += 1
        return _BoundaryCandidates(len(rows), offsets, dict(counts))

    def matching(self, predecessor: _Row) -> tuple[_Row, ...] | None:
        offset = predecessor.committed_offset
        digest = predecessor.committed_prefix_sha256
        assert offset is not None and digest is not None
        exact = self.offsets.get(offset, {})
        same = exact.get(digest, [])
        unknown = exact.get(None, [])
        retained = self.count - self.offset_counts.get(offset, 0) + len(same) + len(unknown)
        if retained > _PREDECESSORS_MAX:
            return None
        # When this fits, there can be at most sixteen other offset buckets.
        unequal = (
            row
            for boundary, digests in self.offsets.items()
            if boundary != offset
            for rows in digests.values()
            for row in rows
        )
        return (*same, *unknown, *unequal)


@dataclass(frozen=True)
class _MoveCandidates:
    """A complete group, indexed once for conservative exact-boundary exclusion."""

    group: _MoveGroup
    departed: _BoundaryCandidates
    survivors: _BoundaryCandidates

    @staticmethod
    def of(rows: list[_Row]) -> _MoveCandidates:
        group = _MoveGroup.partition(rows)
        return _MoveCandidates(
            group, _BoundaryCandidates.of(group.departed), _BoundaryCandidates.of(group.survivors)
        )

    def for_predecessor(self, predecessor: _Row) -> _MoveGroup | None:
        if not self.group.overflows():
            return self.group if self.group.survivors else None
        offset = predecessor.committed_offset
        digest = predecessor.committed_prefix_sha256
        if offset is None or not 0 < offset <= _PREFIX_BYTES_MAX or digest is None:
            return None
        departed = self.departed.matching(predecessor)
        survivors = self.survivors.matching(predecessor)
        if departed is None or survivors is None:
            return None
        # The fallback proves only an exact-boundary continuation. Longer and
        # unknown holders remain comparisons, but cannot alone authorize it.
        if not any(
            row.committed_offset == offset and row.committed_prefix_sha256 == digest
            for row in survivors
        ):
            return None
        return _MoveGroup(departed, survivors, exact_boundary=offset)


def _plan_group(rows: list[_Row]) -> list[MovedDuplicate]:
    """Plan each departed row of one group that exactly one survivor holds."""
    candidates = _MoveCandidates.of(rows)
    plans: list[MovedDuplicate] = []
    for row in candidates.group.departed:
        group = candidates.for_predecessor(row)
        if group is None:
            continue
        holder = group.sole_holder(row)
        if holder is not None:
            plans.append(_moved(row, holder))
    return plans


@dataclass(frozen=True)
class _Holder:
    row: _Row
    identity: tuple[int, int]


def _moved(predecessor: _Row, holder: _Holder) -> MovedDuplicate:
    survivor = holder.row
    assert predecessor.snapshot is not None and survivor.snapshot is not None
    return MovedDuplicate(
        predecessor_id=predecessor.session_id,
        predecessor_path=predecessor.source_path,
        successor_id=survivor.session_id,
        successor_path=survivor.source_path,
        predecessor_snapshot=predecessor.snapshot,
        successor_snapshot=survivor.snapshot,
        hashed_identity=holder.identity,
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


@dataclass(frozen=True)
class _SurvivorProof:
    reasons: tuple[ProofClass, ...]
    identity: tuple[int, int] | None = None


def _survivor_proof(survivor: _Row, predecessor: _Row) -> tuple[ProofClass, ...]:
    return _survivor_proof_result(survivor, predecessor).reasons


def _survivor_proof_result(survivor: _Row, predecessor: _Row) -> _SurvivorProof:
    """Return proof or retain reasons used by maintenance and reconciliation."""
    if not _has_departed(predecessor.source_path):
        return _SurvivorProof(("surviving_copy",))
    if (
        survivor.committed_offset is None
        or survivor.committed_prefix_sha256 is None
        or predecessor.committed_offset is None
        or predecessor.committed_prefix_sha256 is None
    ):
        return _SurvivorProof(("missing_catalog_proof",))
    if predecessor.committed_offset <= 0 or survivor.committed_offset <= 0:
        return _SurvivorProof(("missing_catalog_proof",))
    blockers: list[ProofClass] = []
    if max(predecessor.committed_offset, survivor.committed_offset) > _PREFIX_BYTES_MAX:
        blockers.append("prefix_limit_exceeded")
    if predecessor.committed_offset > survivor.committed_offset:
        blockers.append("lagging_survivor")
    if blockers:
        return _SurvivorProof(tuple(blockers))
    digests = _prefix_digests(
        survivor.source_path, (predecessor.committed_offset, survivor.committed_offset)
    )
    if digests is None:
        return _SurvivorProof(("unreadable_survivor",))
    if digests.digests != (predecessor.committed_prefix_sha256, survivor.committed_prefix_sha256):
        return _SurvivorProof(("divergent_content",))
    return _SurvivorProof(("proven_move",), digests.identity)


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
    proof = _prefix_digests(path, (offset,))
    return proof is not None and proof.digests == (prefix_sha256,)


@dataclass(frozen=True)
class _PrefixProof:
    digests: tuple[str, ...]
    identity: tuple[int, int]


def _prefix_digests(path: str, offsets: tuple[int, ...]) -> _PrefixProof | None:
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
            signature = os.fstat(handle.fileno())
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
    return _PrefixProof(tuple(digests), (signature.st_dev, signature.st_ino))


_ROW_SELECT: Final = """
    SELECT s.id, s.source, s.source_path, ss.started_at,
           f.committed_offset, f.committed_prefix_sha256, s.source_session_id, ss.host,
           f.normalization_checkpoint, f.desired_generation, f.committed_generation,
           f.content_epoch, f.dev, f.inode, ss.message_count, ss.tool_count, f.session_id
    FROM sessions s
    JOIN session_state ss ON ss.session_id = s.id
    LEFT JOIN source_files f ON f.source = s.source AND f.source_path = s.source_path
    WHERE (? OR ss.host IS NULL OR ss.host IN (?, ?))
"""


def _row_select(transfers: TransferCatalog | None) -> str:
    return (
        _ROW_SELECT
        if transfers is None
        else _ROW_SELECT + " AND " + transfers.excluded_sessions_clause()
    )


def _rows_started_at(
    conn: duckdb.DuckDBPyConnection,
    *,
    source: str,
    started_ats: list[datetime],
    host: str | None,
    native_ids: list[str] | None = None,
    transfers: TransferCatalog | None = None,
) -> list[_Row]:
    """Rows of `source` whose first timestamp is one of `started_ats`.

    The timestamps were read back from `session_state` or are about to be
    written to it, so they pass through the same conversion as the column.
    """
    if not started_ats and not native_ids:
        return []
    rows = conn.execute(
        _row_select(transfers)
        + " AND s.source = ? AND (ss.started_at IN (SELECT CAST(UNNEST(?) AS TIMESTAMP))"
        " OR (ss.started_at IS NULL AND s.source_session_id IN (SELECT UNNEST(?))))",
        [host is None, host, UNATTRIBUTED_HOST, source, started_ats, native_ids or []],
    ).fetchall()
    return [_row(values) for values in rows]


def _local_rows(
    conn: duckdb.DuckDBPyConnection, *, host: str, transfers: TransferCatalog | None = None
) -> list[_Row]:
    return [
        _row(values)
        for values in conn.execute(
            _row_select(transfers) + " ORDER BY s.source, ss.started_at, s.id",
            [False, host, UNATTRIBUTED_HOST],
        ).fetchall()
    ]


def _row(values: tuple[object, ...]) -> _Row:
    started_at, offset, prefix_sha256 = values[3], values[4], values[5]
    snapshot = None
    if isinstance(offset, int):
        snapshot = BindingSnapshot(
            str(values[0]),
            str(values[1]),
            str(values[2]),
            cast(str | None, values[6]),
            offset,
            cast(str | None, prefix_sha256),
            cast(str | None, values[8]),
            cast(int, values[9]),
            cast(int, values[10]),
            cast(int, values[11]),
            cast(int | None, values[12]),
            cast(int | None, values[13]),
            cast(int, values[14]),
            cast(int, values[15]),
            cast(str | None, values[16]),
        )
    return _Row(
        session_id=str(values[0]),
        source=str(values[1]),
        source_path=str(values[2]),
        started_at=started_at if isinstance(started_at, datetime) else None,
        committed_offset=int(offset) if isinstance(offset, int) else None,
        committed_prefix_sha256=str(prefix_sha256) if prefix_sha256 is not None else None,
        source_session_id=str(values[6]) if values[6] is not None else None,
        host=str(values[7]) if values[7] is not None else None,
        snapshot=snapshot,
    )


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
