"""Durable, bounded ownership of an already indexed transcript move.

Only a catalog row authorizes transfer work. A byte witness and a publication-
fenced normalized pass precede ownership and deletion; covered raw writes keep
those facts until the finish transaction removes the authority.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import duckdb

from recall.core.ids import message_id, tool_call_id
from recall.core.models import Message, NormalizationCheckpoint, ParseResult, ToolCall
from recall.core.transcripts import parked_path
from recall.core.types import HARVESTED_TOKEN_SOURCES, Source
from recall.db.fts_sidecar_stage import KeywordTurn, Publication
from recall.db.queries import id_set_predicate
from recall.db.schema import get_schema_version
from recall.db.source_files import SourceCatalog, SourceFile, SourceSignature
from recall.services.context_inputs import ContextInputs
from recall.services.sessions import (
    MESSAGE_SELECT,
    message_from_row,
    tool_call_columns,
    tool_call_from_row,
)

if TYPE_CHECKING:
    import pyarrow as pa

    from recall.core.config import AppConfig
    from recall.parsers.protocol import SessionParser
    from recall.services.coordinator import PreparedRawSource
    from recall.services.enrichment_inputs import EnrichmentTargets
    from recall.services.moves import MovedDuplicate

logger = logging.getLogger(__name__)
PAGE_ROWS = 512
TRANSFERS_MAX = 65536
PUBLISH_SEQ_MAX = (1 << 63) - 1
Phase = Literal["catch_up", "owning", "shrinking", "stuck"]
StuckReason = Literal[
    "witness_mismatch",
    "moved_file_missing",
    "moved_file_truncated",
    "normalized_mismatch",
    "predecessor_refused",
]
RefusalKind = Literal["stopping_diagnostic", "torn_history_changed", "unsupported_history_dropped"]


@dataclass(frozen=True)
class RefusalObservation:
    dev: int | None
    inode: int | None
    size: int
    mtime_ns: int
    generation: int
    ctime_ns: int
    sidecar_sha256: str
    parser_revision: str

    @classmethod
    def capture(
        cls, item: SourceFile, signature: SourceSignature, revision: str
    ) -> RefusalObservation:
        return cls(
            signature.dev,
            signature.inode,
            signature.size,
            signature.mtime_ns,
            item.desired_generation,
            signature.ctime_ns,
            hashlib.sha256(signature.sidecar_signature.encode()).hexdigest(),
            revision,
        )

    def encode(self) -> str:
        encoded = json.dumps(asdict(self), separators=(",", ":"))
        if len(encoded) > 1024:
            raise ValueError("transfer refusal observation exceeds 1024 characters")
        return encoded

    @classmethod
    def decode(cls, value: str) -> RefusalObservation:
        fields = json.loads(value)
        if not isinstance(fields, dict) or set(fields) != set(cls.__dataclass_fields__):
            raise ValueError("invalid transfer refusal observation")
        for key in ("dev", "inode", "size", "mtime_ns", "generation", "ctime_ns"):
            if fields[key] is None and key in ("dev", "inode"):
                continue
            if type(fields[key]) is not int or fields[key] < 0:
                raise ValueError("invalid transfer refusal observation integer")
        if (
            not isinstance(fields["parser_revision"], str)
            or not isinstance(fields["sidecar_sha256"], str)
            or len(fields["sidecar_sha256"]) != 64
        ):
            raise ValueError("invalid transfer refusal parser/sidecar identity")
        result = cls(**fields)
        result.encode()
        return result


class OrdinaryCaptureRefused(RuntimeError):
    def __init__(self, kind: RefusalKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class TransferPreservationRefused(RuntimeError):
    def __init__(self, transfer: DuplicateTransfer, mismatch: str) -> None:
        super().__init__(
            f"transfer {transfer.survivor_id} preserves indexed history until finish: {mismatch}"
        )
        self.transfer = transfer


@dataclass(frozen=True)
class BindingSnapshot:
    """Keyed catalog/session facts revalidated without content reads."""

    session_id: str
    source: str
    source_path: str
    source_session_id: str | None
    committed_offset: int
    committed_prefix_sha256: str | None
    normalization_checkpoint: str | None
    desired_generation: int
    committed_generation: int
    content_epoch: int
    dev: int | None
    inode: int | None
    message_count: int
    tool_count: int
    catalog_session_id: str | None


def binding_snapshot(conn: duckdb.DuckDBPyConnection, session_id: str) -> BindingSnapshot | None:
    row = conn.execute(
        """SELECT s.id, s.source, s.source_path, s.source_session_id,
        f.committed_offset, f.committed_prefix_sha256, f.normalization_checkpoint,
        f.desired_generation, f.committed_generation, f.content_epoch, f.dev, f.inode,
        ss.message_count, ss.tool_count, f.session_id
        FROM sessions s JOIN session_state ss ON ss.session_id=s.id
        JOIN source_files f ON f.source=s.source AND f.source_path=s.source_path
        WHERE s.id=?""",
        [session_id],
    ).fetchone()
    return BindingSnapshot(*row) if row is not None else None


@dataclass(frozen=True)
class DuplicateTransfer:
    survivor_id: str
    predecessor_id: str
    source: str
    original_path: str
    moved_path: str
    survivor_offset: int
    survivor_prefix_sha256: str
    survivor_message_count: int
    survivor_tool_count: int
    context_message_bound: int
    survivor_orphan_count: int | None
    phase: Phase
    predecessor_publish_seq: int
    predecessor_refusal_kind: RefusalKind | None
    predecessor_refusal_observation: str | None
    witness_epoch: int | None
    coverage_proven_at: datetime | None
    predecessor_content_epoch: int | None
    stuck_reason: StuckReason | None
    created_at: datetime
    updated_at: datetime

    @property
    def covered(self) -> bool:
        return self.coverage_proven_at is not None


@dataclass(frozen=True)
class CoveragePassFence:
    transfer_id: str
    predecessor_publish_seq: int
    witness_epoch: int


@dataclass(frozen=True)
class CoverageResult:
    fence: CoveragePassFence
    messages: int
    attached: int
    orphans: int
    mismatch: str | None


@dataclass(frozen=True)
class TransferCatalog:
    """Schema-38 capability. Statements naming transfer authority live here."""

    conn: duckdb.DuckDBPyConnection

    def load(self) -> tuple[DuplicateTransfer, ...]:
        rows = self.conn.execute(
            "SELECT * FROM duplicate_transfers ORDER BY created_at, survivor_id LIMIT ?",
            [TRANSFERS_MAX + 1],
        ).fetchall()
        if len(rows) > TRANSFERS_MAX:
            raise RuntimeError(f"transfer catalog exceeds {TRANSFERS_MAX} rows")
        return tuple(DuplicateTransfer(*row) for row in rows)

    def by_id(self, transfer_id: str) -> DuplicateTransfer | None:
        row = self.conn.execute(
            "SELECT * FROM duplicate_transfers WHERE survivor_id=?", [transfer_id]
        ).fetchone()
        return DuplicateTransfer(*row) if row else None

    def clear_refusal(self, transfer: DuplicateTransfer) -> None:
        self.conn.execute(
            """UPDATE duplicate_transfers SET predecessor_refusal_kind=NULL,
                predecessor_refusal_observation=NULL
                WHERE survivor_id=? AND coverage_proven_at IS NULL""",
            [transfer.survivor_id],
        )

    def record_refusal(
        self, transfer: DuplicateTransfer, kind: RefusalKind, observation: RefusalObservation
    ) -> None:
        assert transfer.phase == "catch_up" and not transfer.covered
        self.conn.execute(
            """UPDATE duplicate_transfers SET predecessor_refusal_kind=?,
                predecessor_refusal_observation=?, updated_at=CURRENT_TIMESTAMP
                WHERE survivor_id=?""",
            [kind, observation.encode(), transfer.survivor_id],
        )

    def begin_shrink(self, transfer: DuplicateTransfer) -> None:
        self.conn.execute(
            """UPDATE duplicate_transfers SET phase='shrinking', updated_at=CURRENT_TIMESTAMP
                WHERE survivor_id=?""",
            [transfer.survivor_id],
        )

    def remove(self, transfer: DuplicateTransfer) -> None:
        self.conn.execute(
            "DELETE FROM duplicate_transfers WHERE survivor_id=?", [transfer.survivor_id]
        )

    def for_predecessor(self, session_id: str) -> DuplicateTransfer | None:
        row = self.conn.execute(
            "SELECT * FROM duplicate_transfers WHERE predecessor_id=?", [session_id]
        ).fetchone()
        return DuplicateTransfer(*row) if row else None

    def for_path(self, source: str, path: str) -> DuplicateTransfer | None:
        row = self.conn.execute(
            "SELECT * FROM duplicate_transfers WHERE source=? AND moved_path=?", [source, path]
        ).fetchone()
        return DuplicateTransfer(*row) if row else None

    def excluded_sessions_clause(self) -> str:
        return """NOT starts_with(s.source_path, 'recall-transfer:') AND NOT EXISTS (
            SELECT 1 FROM duplicate_transfers dt
            WHERE s.id IN (dt.survivor_id, dt.predecessor_id))"""

    def custody_clause(
        self, message_alias: str | None = None, tool_alias: str | None = None
    ) -> str:
        assert (message_alias is None) != (tool_alias is None)
        if message_alias is not None:
            identity = f"{message_alias}.session_id"
            protected = f"{message_alias}.idx < dt.context_message_bound"
        else:
            identity = f"{tool_alias}.session_id"
            protected = f"""({tool_alias}.message_id IS NULL OR EXISTS (
                SELECT 1 FROM messages custody_m WHERE custody_m.id={tool_alias}.message_id
                AND custody_m.idx < dt.context_message_bound))"""
        return f"""EXISTS (SELECT 1 FROM duplicate_transfers dt WHERE dt.phase <> 'stuck'
            AND ({identity}=dt.survivor_id
            OR ({identity}=dt.predecessor_id AND {protected})))"""

    def survivor_events_clause(self) -> str:
        # A path-derived native id can differ until P's first publication.
        # Never assign new events to S in that window, or refill a shrink page.
        return """NOT EXISTS (SELECT 1 FROM duplicate_transfers dt WHERE dt.phase <> 'stuck'
            AND dt.survivor_id IN (usage_events.session_id, owners.session_id))"""

    def holder_join(self) -> str:
        return """LEFT JOIN (
            SELECT survivor_id AS session_id,
                   CASE WHEN coverage_proven_at IS NULL THEN survivor_id
                        ELSE predecessor_id END AS holder_id,
                   moved_path, phase FROM duplicate_transfers
            UNION ALL SELECT predecessor_id,
                   CASE WHEN coverage_proven_at IS NULL THEN survivor_id ELSE predecessor_id END,
                   moved_path, phase FROM duplicate_transfers
        ) holder ON holder.session_id=s.id"""

    def backing_arm(self) -> str:
        return """UNION SELECT l.source, l.identity FROM local_sessions l
            JOIN duplicate_transfers dt ON
                l.source_path=('recall-transfer:' || dt.survivor_id || ':' || dt.moved_path)
            JOIN source_files f ON f.source=dt.source AND f.source_path=dt.moved_path
            WHERE dt.phase <> 'stuck' AND NOT f.missing"""

    def participates(self, ids: tuple[str, str]) -> bool:
        return (
            self.conn.execute(
                """SELECT 1 FROM duplicate_transfers
                    WHERE survivor_id IN (?, ?) OR predecessor_id IN (?, ?) LIMIT 1""",
                [*ids, *ids],
            ).fetchone()
            is not None
        )

    def recompute_deferred(self, since: datetime | None, only_mode: str | None) -> int:
        row = self.conn.execute(
            f"""SELECT COUNT(*) FROM messages m
            JOIN message_state ms ON ms.message_id=m.id
            JOIN session_state ss ON ss.session_id=m.session_id
            WHERE (ms.content IS NOT NULL OR ms.thinking IS NOT NULL)
            AND (? IS NULL OR COALESCE(ss.ended_at,ss.started_at,ss.indexed_at)>=?)
            AND (? IS NULL OR COALESCE(ms.context_mode,'off')=?)
            AND {self.custody_clause("m")}""",
            [since, since, only_mode, only_mode],
        ).fetchone()
        assert row is not None
        return int(row[0])

    def has_custody(self) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM duplicate_transfers WHERE phase <> 'stuck' LIMIT 1"
            ).fetchone()
            is not None
        )

    def mark_stuck(self, transfer: DuplicateTransfer, reason: StuckReason) -> None:
        assert not transfer.covered
        self.conn.execute(
            """UPDATE duplicate_transfers SET phase='stuck', stuck_reason=?,
                updated_at=CURRENT_TIMESTAMP WHERE survivor_id=? AND coverage_proven_at IS NULL""",
            [reason, transfer.survivor_id],
        )

    def end_stuck(self, transfer: DuplicateTransfer) -> None:
        assert transfer.phase == "stuck" and not transfer.covered
        self.conn.execute(
            """DELETE FROM duplicate_transfers WHERE survivor_id=? AND phase='stuck'
                AND coverage_proven_at IS NULL""",
            [transfer.survivor_id],
        )

    def copy_table(self, source_alias: str) -> None:
        assert source_alias.isidentifier()
        self.conn.execute(
            f"INSERT INTO duplicate_transfers SELECT * FROM {source_alias}.duplicate_transfers"
        )

    def insert(
        self, plan: MovedDuplicate, predecessor: BindingSnapshot, survivor: BindingSnapshot
    ) -> None:
        assert survivor.committed_offset > 0 and survivor.committed_prefix_sha256 is not None
        witness_epoch = (
            survivor.content_epoch + 1
            if (predecessor.committed_offset, predecessor.committed_prefix_sha256)
            == (survivor.committed_offset, survivor.committed_prefix_sha256)
            else None
        )
        self.conn.execute(
            """INSERT INTO duplicate_transfers(
            survivor_id, predecessor_id, source, original_path, moved_path,
            survivor_offset, survivor_prefix_sha256, survivor_message_count,
            survivor_tool_count, context_message_bound, phase, witness_epoch)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'catch_up', ?)""",
            [
                plan.successor_id,
                plan.predecessor_id,
                survivor.source,
                plan.predecessor_path,
                plan.successor_path,
                survivor.committed_offset,
                survivor.committed_prefix_sha256,
                survivor.message_count,
                survivor.tool_count,
                max(survivor.message_count, predecessor.message_count),
                witness_epoch,
            ],
        )

    def publish(self, transfer: DuplicateTransfer, result: ParseResult) -> None:
        if transfer.predecessor_publish_seq == PUBLISH_SEQ_MAX:
            raise RuntimeError(f"transfer {transfer.survivor_id} exhausted its publication counter")
        item = SourceCatalog(self.conn, clock=time.time).get(transfer.source, transfer.moved_path)
        assert item is not None
        epoch = item.content_epoch
        if transfer.phase == "catch_up":
            witness = None if result.is_full_parse else transfer.witness_epoch
            if witness is None:
                if result.witness.kind == "at":
                    if result.witness.digest == transfer.survivor_prefix_sha256:
                        witness = epoch
                    else:
                        self.mark_stuck(transfer, "witness_mismatch")
                elif result.witness.kind == "crossed":
                    self.mark_stuck(transfer, "witness_mismatch")
                elif result.is_full_parse:
                    self.mark_stuck(transfer, "moved_file_truncated")
            else:
                assert epoch == witness, "a suffix cannot advance the content epoch"
            self.conn.execute(
                """UPDATE duplicate_transfers SET predecessor_publish_seq=predecessor_publish_seq+1,
                witness_epoch=?, predecessor_refusal_kind=NULL,
                predecessor_refusal_observation=NULL,
                updated_at=CURRENT_TIMESTAMP WHERE survivor_id=?""",
                [witness, transfer.survivor_id],
            )
        else:
            self.conn.execute(
                """UPDATE duplicate_transfers SET predecessor_publish_seq=predecessor_publish_seq+1,
                predecessor_content_epoch=CASE WHEN coverage_proven_at IS NOT NULL
                                               THEN ? ELSE NULL END,
                updated_at=CURRENT_TIMESTAMP WHERE survivor_id=?""",
                [epoch, transfer.survivor_id],
            )

    def accept_coverage(self, result: CoverageResult) -> bool:
        row = self.conn.execute(
            "SELECT * FROM duplicate_transfers WHERE survivor_id=?", [result.fence.transfer_id]
        ).fetchone()
        if row is None:
            return False
        transfer = DuplicateTransfer(*row)
        item = SourceCatalog(self.conn, clock=time.time).get(transfer.source, transfer.moved_path)
        if (
            transfer.phase != "catch_up"
            or item is None
            or transfer.predecessor_publish_seq != result.fence.predecessor_publish_seq
            or transfer.witness_epoch != result.fence.witness_epoch
            or item.content_epoch != result.fence.witness_epoch
        ):
            return False
        if result.mismatch is not None:
            logger.warning(
                "transfer normalized mismatch transfer=%s fact=%s",
                transfer.survivor_id,
                result.mismatch,
            )
            self.mark_stuck(transfer, "normalized_mismatch")
        else:
            counts = self.conn.execute(
                "SELECT message_count, tool_count FROM session_state WHERE session_id=?",
                [transfer.survivor_id],
            ).fetchone()
            assert counts == (transfer.survivor_message_count, transfer.survivor_tool_count)
            assert result.messages == transfer.survivor_message_count
            assert result.attached + result.orphans == transfer.survivor_tool_count
            self.conn.execute(
                """UPDATE duplicate_transfers SET phase='owning',
                coverage_proven_at=CURRENT_TIMESTAMP, predecessor_content_epoch=witness_epoch,
                survivor_orphan_count=?, predecessor_refusal_kind=NULL,
                predecessor_refusal_observation=NULL, updated_at=CURRENT_TIMESTAMP
                WHERE survivor_id=?""",
                [result.orphans, transfer.survivor_id],
            )
        return True


def transfer_catalog(conn: duckdb.DuckDBPyConnection) -> TransferCatalog | None:
    return TransferCatalog(conn) if get_schema_version(conn) >= 38 else None


def transfer_for_predecessor(
    conn: duckdb.DuckDBPyConnection, session_id: str
) -> DuplicateTransfer | None:
    catalog = transfer_catalog(conn)
    return catalog.for_predecessor(session_id) if catalog else None


def transfer_for_moved_path(
    conn: duckdb.DuckDBPyConnection, source: str, path: str
) -> DuplicateTransfer | None:
    catalog = transfer_catalog(conn)
    return catalog.for_path(source, path) if catalog else None


def rebind(conn: duckdb.DuckDBPyConnection, plan: MovedDuplicate) -> None:
    catalog = transfer_catalog(conn)
    if catalog is None:
        raise RuntimeError(
            "duplicate transfer apply requires schema 38; migrate the database first"
        )
    with KeywordTurn(None, None).transaction(conn):
        predecessor = binding_snapshot(conn, plan.predecessor_id)
        survivor = binding_snapshot(conn, plan.successor_id)
        if (
            predecessor is None
            or survivor is None
            or predecessor != plan.predecessor_snapshot
            or survivor != plan.successor_snapshot
        ):
            raise RuntimeError("move plan catalog/session snapshot changed before rebind")
        if (
            predecessor.catalog_session_id != predecessor.session_id
            or survivor.catalog_session_id != survivor.session_id
        ):
            raise RuntimeError("move plan requires both catalog rows bound to their sessions")
        if (survivor.dev, survivor.inode) != plan.hashed_identity:
            raise RuntimeError("move proof handle no longer matches catalog identity")
        # Both role columns exclude either session, including cross-role chains.
        if catalog.participates((plan.predecessor_id, plan.successor_id)):
            raise RuntimeError("a planned session already belongs to a transfer")
        catalog.insert(plan, predecessor, survivor)
        conn.execute(
            """UPDATE source_files SET missing=TRUE, session_id=NULL,
                normalization_checkpoint=NULL WHERE source=? AND source_path=?""",
            [predecessor.source, predecessor.source_path],
        )
        conn.execute(
            "UPDATE sessions SET source_path=? WHERE id=?",
            [parked_path(survivor.session_id, survivor.source_path), survivor.session_id],
        )
        conn.execute(
            "UPDATE sessions SET source_path=? WHERE id=?",
            [survivor.source_path, predecessor.session_id],
        )
        checkpoint = NormalizationCheckpoint.decode(predecessor.normalization_checkpoint)
        if checkpoint is not None:
            assert survivor.dev is not None and survivor.inode is not None
            checkpoint = replace(checkpoint, source_dev=survivor.dev, source_inode=survivor.inode)
        conn.execute(
            """UPDATE source_files SET session_id=?, committed_offset=?, committed_prefix_sha256=?,
            normalization_checkpoint=?, desired_generation=desired_generation+1,
            content_epoch=content_epoch+1,
            first_pending_at=COALESCE(first_pending_at, ?) WHERE source=? AND source_path=?""",
            [
                predecessor.session_id,
                predecessor.committed_offset,
                predecessor.committed_prefix_sha256,
                checkpoint.encode() if checkpoint else None,
                time.time(),
                survivor.source,
                survivor.source_path,
            ],
        )


def damage_reason(conn: duckdb.DuckDBPyConnection, transfer: DuplicateTransfer) -> str | None:
    if transfer.phase not in ("catch_up", "owning", "shrinking", "stuck"):
        return "unknown phase"
    if transfer.source not in {source.value for source in Source}:
        return "unknown source"
    for session_id in (transfer.predecessor_id, transfer.survivor_id):
        state = conn.execute(
            """SELECT s.source FROM sessions s
                JOIN session_state ss ON ss.session_id=s.id WHERE s.id=?""",
            [session_id],
        ).fetchone()
        if state != (transfer.source,):
            return "session source/state disagrees"
    if (
        conn.execute(
            "SELECT 1 FROM source_files WHERE session_id=? LIMIT 1", [transfer.survivor_id]
        ).fetchone()
        is not None
    ):
        return "survivor still has a catalog binding"
    rows = dict(
        conn.execute(
            "SELECT id, source_path FROM sessions WHERE id IN (?, ?)",
            [transfer.survivor_id, transfer.predecessor_id],
        ).fetchall()
    )
    if rows.get(transfer.predecessor_id) != transfer.moved_path:
        return "predecessor does not own moved path"
    if rows.get(transfer.survivor_id) != parked_path(transfer.survivor_id, transfer.moved_path):
        return "survivor is absent or not parked"
    row = conn.execute(
        "SELECT session_id FROM source_files WHERE source=? AND source_path=?",
        [transfer.source, transfer.moved_path],
    ).fetchone()
    if row != (transfer.predecessor_id,):
        return "moved catalog binding disagrees"
    if transfer.covered:
        epoch = conn.execute(
            "SELECT content_epoch FROM source_files WHERE source=? AND source_path=?",
            [transfer.source, transfer.moved_path],
        ).fetchone()
        if epoch != (transfer.predecessor_content_epoch,):
            return "covered content epoch disagrees"
    if transfer.stuck_reason not in (
        None,
        "witness_mismatch",
        "moved_file_missing",
        "moved_file_truncated",
        "normalized_mismatch",
        "predecessor_refused",
    ):
        return "unknown stuck reason"
    if (transfer.phase == "stuck") != (transfer.stuck_reason is not None):
        return "phase and stuck reason disagree"
    if transfer.covered != (transfer.phase in ("owning", "shrinking")):
        return "phase and coverage disagree"
    if transfer.predecessor_refusal_kind not in (
        None,
        "stopping_diagnostic",
        "torn_history_changed",
        "unsupported_history_dropped",
    ):
        return "unknown refusal kind"
    if transfer.predecessor_refusal_observation is not None:
        try:
            RefusalObservation.decode(transfer.predecessor_refusal_observation)
        except (ValueError, TypeError):
            return "invalid refusal observation"
    return None


@dataclass(frozen=True)
class ToolFact:
    idx: int
    fields: tuple[object, ...]
    native_id: str | None
    result: tuple[object, ...] | None

    @classmethod
    def from_call(cls, call: ToolCall, result: tuple[object, ...] | None = None) -> ToolFact:
        return cls(
            call.idx,
            (
                call.tool_name,
                _json(call.tool_input),
                call.bash_command,
                call.bash_base,
                call.bash_sub,
                call.is_compound,
                call.agent_id,
                call.subagent_type,
                call.subagent_description,
                call.subagent_model,
                call.skill_name,
            ),
            call.tool_use_id,
            result,
        )


@dataclass(frozen=True)
class MessageFact:
    idx: int
    role: str
    agent_id: str | None
    timestamp: datetime | None
    has_thinking: bool
    content: str | None
    thinking: str | None
    tools: tuple[ToolFact, ...]

    @classmethod
    def from_message(cls, message: Message, tools: tuple[ToolFact, ...]) -> MessageFact:
        return cls(
            message.idx,
            message.role.value,
            message.agent_id,
            _timestamp(message.timestamp),
            message.has_thinking,
            message.content,
            message.thinking,
            tools,
        )


@dataclass(frozen=True)
class FactPage:
    messages: tuple[MessageFact, ...] = ()
    orphans: tuple[ToolFact, ...] = ()
    markers: frozenset[int] = frozenset()


def _json(value: Any) -> str | None:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) if value is not None else None


def _timestamp(value: datetime | None) -> datetime | None:
    return (
        value.astimezone(UTC).replace(tzinfo=None)
        if value is not None and value.tzinfo is not None
        else value
    )


def _holds_tools(old: tuple[ToolFact, ...], new: tuple[ToolFact, ...], *, prefix: bool) -> bool:
    if not prefix and len(old) != len(new):
        return False
    target = {call.idx: call for call in new}
    for call in old:
        other = target.get(call.idx)
        if (
            other is None
            or call.fields != other.fields
            or (call.native_id is not None and call.native_id != other.native_id)
            or (call.result is not None and call.result != other.result)
        ):
            return False
    return True


def holds_survivor_facts(survivor: FactPage, predecessor: FactPage, last_idx: int) -> str | None:
    """First unheld positional fact, or None. Metadata and enrichment are excluded."""
    target = {message.idx: message for message in predecessor.messages}
    for old in survivor.messages:
        new = target.get(old.idx)
        if new is None:
            return f"message {old.idx} missing"
        if (old.role, old.agent_id, old.timestamp) != (new.role, new.agent_id, new.timestamp):
            return f"message {old.idx} identity"
        complete = old.idx == last_idx
        if complete:
            content = old.content is None or (
                new.content is not None and new.content.startswith(old.content)
            )
            thinking = old.thinking is None or (
                new.thinking is not None and new.thinking.startswith(old.thinking)
            )
            has_thinking = not old.has_thinking or new.has_thinking
        else:
            content, thinking, has_thinking = (
                old.content == new.content,
                old.thinking == new.thinking,
                old.has_thinking == new.has_thinking,
            )
        if not (content and thinking and has_thinking):
            return f"message {old.idx} payload"
        if not _holds_tools(old.tools, new.tools, prefix=complete):
            return f"message {old.idx} tools"
    if not _holds_tools(survivor.orphans, predecessor.orphans, prefix=True):
        return "orphan payload/native id/result"
    if not survivor.markers <= predecessor.markers:
        return "stop marker position"
    return None


def _tool_facts(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    *,
    message_ids: list[str] | None = None,
    orphan_after: int | None = None,
    orphan_indices: tuple[int, ...] | None = None,
) -> tuple[ToolFact, ...]:
    predicate = "t.session_id=?"
    parameters: list[object] = [session_id]
    limit = ""
    if message_ids is not None:
        predicate += " AND t.message_id IN (SELECT UNNEST(?))"
        parameters.append(message_ids)
    elif orphan_indices is not None:
        predicate += " AND t.message_id IS NULL AND t.idx IN (SELECT UNNEST(?))"
        parameters.append(list(orphan_indices))
    else:
        assert orphan_after is not None
        predicate += " AND t.message_id IS NULL AND t.idx>?"
        parameters.append(orphan_after)
        limit = f" LIMIT {PAGE_ROWS}"
    rows = conn.execute(
        f"""SELECT {tool_call_columns("t")}, u.tool_use_id,
        r.tool_call_id IS NOT NULL, r.result_summary, r.is_error, r.completed_at
        FROM tool_calls t LEFT JOIN tool_use_ids u ON u.tool_call_id=t.id
        LEFT JOIN tool_results r ON r.tool_call_id=t.id
        WHERE {predicate} ORDER BY t.idx,t.id{limit}""",
        parameters,
    ).fetchall()
    result: list[ToolFact] = []
    for row in rows:
        call = tool_call_from_row(row[:15])
        call.tool_use_id = row[15]
        fact = (row[17], row[18], _timestamp(row[19])) if row[16] else None
        result.append(ToolFact.from_call(call, fact))
    return tuple(result)


def _message_page(
    conn: duckdb.DuckDBPyConnection, session_id: str, indices: tuple[int, ...]
) -> FactPage:
    if not indices:
        return FactPage()
    rows = conn.execute(
        MESSAGE_SELECT + " WHERE m.session_id=? AND m.idx IN (SELECT UNNEST(?)) ORDER BY m.idx",
        [session_id, list(indices)],
    ).fetchall()
    # Tool facts are grouped by parent in the query below, so every page keeps
    # repeated per-message tool indices distinct.
    grouped: dict[str, list[ToolFact]] = {}
    tool_rows = conn.execute(
        f"""SELECT {tool_call_columns("t")}, u.tool_use_id,
        r.tool_call_id IS NOT NULL, r.result_summary, r.is_error, r.completed_at
        FROM tool_calls t LEFT JOIN tool_use_ids u ON u.tool_call_id=t.id
        LEFT JOIN tool_results r ON r.tool_call_id=t.id
        WHERE t.message_id IN (SELECT UNNEST(?)) ORDER BY t.message_id,t.idx,t.id""",
        [[str(row[0]) for row in rows]],
    ).fetchall()
    for row in tool_rows:
        call = tool_call_from_row(row[:15])
        call.tool_use_id = row[15]
        fact = (row[17], row[18], _timestamp(row[19])) if row[16] else None
        assert call.message_id is not None
        grouped.setdefault(call.message_id, []).append(ToolFact.from_call(call, fact))
    return FactPage(
        tuple(
            MessageFact.from_message(message_from_row(row), tuple(grouped.get(str(row[0]), ())))
            for row in rows
        )
    )


def verify_pass(conn: duckdb.DuckDBPyConnection, transfer: DuplicateTransfer) -> CoverageResult:
    """Capture authority and normalized rows in one independent read snapshot."""
    from recall.db.read_snapshot import read_snapshot

    with conn.cursor() as reader:
        return read_snapshot(reader, lambda snapshot: _verify_pass(snapshot, transfer.survivor_id))


def _verify_pass(conn: duckdb.DuckDBPyConnection, transfer_id: str) -> CoverageResult:
    catalog = transfer_catalog(conn)
    assert catalog is not None
    transfer = catalog.by_id(transfer_id)
    assert transfer is not None
    """Read-only paged pass. The fence precedes the first predecessor row."""
    assert transfer.phase == "catch_up" and transfer.witness_epoch is not None
    fence = CoveragePassFence(
        transfer.survivor_id, transfer.predecessor_publish_seq, transfer.witness_epoch
    )
    messages = attached = orphans = 0
    mismatch: str | None = None
    for start in range(0, transfer.survivor_message_count, PAGE_ROWS):
        indices = tuple(range(start, min(start + PAGE_ROWS, transfer.survivor_message_count)))
        old = _message_page(conn, transfer.survivor_id, indices)
        new = _message_page(conn, transfer.predecessor_id, indices)
        mismatch = mismatch or holds_survivor_facts(old, new, transfer.survivor_message_count - 1)
        messages += len(old.messages)
        attached += sum(len(message.tools) for message in old.messages)
    cursor = -1
    for _ in range(transfer.survivor_tool_count // PAGE_ROWS + 1):
        old_tools = _tool_facts(conn, transfer.survivor_id, orphan_after=cursor)
        if not old_tools:
            break
        # Additional P orphans must not push an S position out of this page.
        new_tools = tuple(
            call
            for call in _tool_facts(
                conn, transfer.predecessor_id, orphan_indices=tuple(call.idx for call in old_tools)
            )
        )
        mismatch = mismatch or holds_survivor_facts(
            FactPage(orphans=old_tools), FactPage(orphans=new_tools), -1
        )
        orphans += len(old_tools)
        cursor = old_tools[-1].idx
    marker_cursor = -(1 << 31)
    for _ in range(transfer.survivor_message_count // PAGE_ROWS + 2):
        old_markers = frozenset(
            row[0]
            for row in conn.execute(
                """SELECT message_idx FROM session_stop_markers
                    WHERE session_id=? AND message_idx>? ORDER BY message_idx LIMIT ?""",
                [transfer.survivor_id, marker_cursor, PAGE_ROWS],
            ).fetchall()
        )
        if not old_markers:
            break
        new_markers = frozenset(
            row[0]
            for row in conn.execute(
                """SELECT message_idx FROM session_stop_markers
                    WHERE session_id=? AND message_idx IN (SELECT UNNEST(?))""",
                [transfer.predecessor_id, sorted(old_markers)],
            ).fetchall()
        )
        mismatch = mismatch or holds_survivor_facts(
            FactPage(markers=old_markers), FactPage(markers=new_markers), -1
        )
        marker_cursor = max(old_markers)
    return CoverageResult(fence, messages, attached, orphans, mismatch)


def record_coverage(conn: duckdb.DuckDBPyConnection, result: CoverageResult) -> bool:
    catalog = transfer_catalog(conn)
    assert catalog is not None
    with KeywordTurn(None, None).transaction(conn):
        return catalog.accept_coverage(result)


def refuse_rewrite(
    conn: duckdb.DuckDBPyConnection, transfer: DuplicateTransfer, result: ParseResult
) -> None:
    """Keep committed protected facts, including insert-only tail facts, on a full write."""
    if not transfer.covered or not result.is_full_parse:
        return
    last_idx = transfer.survivor_message_count - 1
    native_results: dict[tuple[int | None, int], ToolFact] = {}
    for start in range(0, transfer.survivor_message_count, PAGE_ROWS):
        indices = tuple(range(start, min(start + PAGE_ROWS, transfer.survivor_message_count)))
        previous = _message_page(conn, transfer.predecessor_id, indices)
        for message in previous.messages:
            native_results.update(((message.idx, call.idx), call) for call in message.tools)
        parsed = _parsed_page(result, indices, native_results)
        mismatch = holds_survivor_facts(previous, parsed, last_idx)
        if mismatch:
            raise TransferPreservationRefused(transfer, mismatch)
    assert transfer.survivor_orphan_count is not None
    cursor = -1
    for _ in range(transfer.survivor_orphan_count // PAGE_ROWS + 1):
        previous_tools = tuple(
            call
            for call in _tool_facts(conn, transfer.predecessor_id, orphan_after=cursor)
            if call.idx < transfer.survivor_orphan_count
        )
        if not previous_tools:
            break
        native_results.update(((None, call.idx), call) for call in previous_tools)
        parsed = _parsed_page(result, (), native_results)
        mismatch = holds_survivor_facts(FactPage(orphans=previous_tools), parsed, last_idx)
        if mismatch:
            raise TransferPreservationRefused(transfer, mismatch)
        cursor = previous_tools[-1].idx
    markers = frozenset(
        row[0]
        for row in conn.execute(
            "SELECT message_idx FROM session_stop_markers WHERE session_id=? AND message_idx<?",
            [transfer.predecessor_id, transfer.survivor_message_count],
        ).fetchall()
    )
    if not markers <= frozenset(marker.idx for marker in result.tail_facts.stop_markers):
        raise TransferPreservationRefused(transfer, "stop marker position")


def _parsed_page(
    result: ParseResult, indices: tuple[int, ...], retained: dict[tuple[int | None, int], ToolFact]
) -> FactPage:
    results = {
        item.tool_use_id: (
            item.result_summary,
            item.is_error,
            _timestamp(item.completed_at),
        )
        for item in result.tail_facts.tool_results
    }

    def tool(call: ToolCall, parent: int | None) -> ToolFact:
        old = retained.get((parent, call.idx))
        # The ordinary writer retains results only for an unchanged native mapping.
        previous_result = old.result if old and old.native_id == call.tool_use_id else None
        return ToolFact.from_call(
            call,
            previous_result if previous_result is not None else results.get(call.tool_use_id or ""),
        )

    messages = tuple(
        MessageFact.from_message(
            message, tuple(tool(call, message.idx) for call in message.tool_calls)
        )
        for message in result.session.messages
        if message.idx in indices
    )
    orphans = tuple(tool(call, None) for call in result.session.orphan_tool_calls)
    return FactPage(messages, orphans)


@dataclass(frozen=True)
class RawPublication:
    """Admission issued before context preparation, consumed by the raw transaction."""

    result: ParseResult
    rewrite: bool
    transfer: DuplicateTransfer | None
    parser: SessionParser | None = None

    def require_current(self, conn: duckdb.DuckDBPyConnection, session_id: str) -> None:
        current = transfer_for_predecessor(conn, session_id)
        if current != self.transfer:
            raise RuntimeError("transfer changed after raw publication admission; retry")

    def publish(self, conn: duckdb.DuckDBPyConnection, *, acknowledged: bool) -> None:
        if self.transfer is not None:
            if not acknowledged:
                from recall.services.coordinator import capture_path, observe_path

                assert self.parser is not None
                source = observe_path(
                    self.parser,
                    capture_path(self.parser, Path(self.transfer.moved_path)),
                    conn=conn,
                )
                result = self.result
                assert result.committed_prefix_sha256 is not None
                catalog = SourceCatalog(conn, clock=time.time)
                if source.current and source.last_error is None and result.is_full_parse:
                    # A library full write is an explicit publication even when
                    # discovery has no new bytes. Give its acknowledgement the
                    # same desired generation as an explicit daemon reparse.
                    catalog.force_reconcile(source.source, source.source_path)
                    source = catalog.get(source.source, source.source_path)
                    assert source is not None
                if not catalog.acknowledge(
                    source.source,
                    source.source_path,
                    source.desired_generation
                    if result.reached_end
                    else source.committed_generation,
                    result.next_byte_offset,
                    result.committed_prefix_sha256,
                    semantic_rewrite=self.rewrite,
                    checkpoint=result.normalization_checkpoint.encode()
                    if result.normalization_checkpoint
                    else None,
                    full_parse=result.is_full_parse,
                ):
                    raise RuntimeError("transfer raw publication lost catalog generation")
            catalog = transfer_catalog(conn)
            assert catalog is not None
            item = SourceCatalog(conn, clock=time.time).get(
                self.transfer.source, self.transfer.moved_path
            )
            if item is None or (item.committed_offset, item.committed_prefix_sha256) != (
                self.result.next_byte_offset,
                self.result.committed_prefix_sha256,
            ):
                raise RuntimeError("transfer raw publication requires its catalog acknowledgement")
            catalog.publish(self.transfer, self.result)


def admit_raw_publication(
    conn: duckdb.DuckDBPyConnection,
    result: ParseResult,
    rewrite: bool,
    parser: SessionParser | None = None,
) -> RawPublication:
    transfer = transfer_for_predecessor(conn, result.session.id)
    if transfer is not None:
        damage = damage_reason(conn, transfer)
        if damage is not None:
            raise RuntimeError(f"transfer {transfer.survivor_id} is damaged: {damage}")
        if transfer.phase == "catch_up" and result.witness_offset != transfer.survivor_offset:
            raise RuntimeError(f"transfer {transfer.survivor_id} requires its capture witness")
        refuse_rewrite(conn, transfer, result)
    return RawPublication(result, rewrite, transfer, parser)


def require_raw_publication(
    conn: duckdb.DuckDBPyConnection, session_id: str, publication: RawPublication | None
) -> None:
    if publication is not None:
        publication.require_current(conn, session_id)
    elif transfer_for_predecessor(conn, session_id) is not None:
        raise RuntimeError(
            "an active transfer predecessor requires checked raw publication admission"
        )


def in_custody(conn: duckdb.DuckDBPyConnection, targets: EnrichmentTargets) -> bool:
    return eligible_targets(conn, targets) != targets


def eligible_targets(
    conn: duckdb.DuckDBPyConnection, targets: EnrichmentTargets
) -> EnrichmentTargets:
    from recall.services.enrichment_inputs import EnrichmentTargets

    catalog = transfer_catalog(conn)
    if catalog is None:
        return targets
    messages = conn.execute(
        f"""SELECT m.id FROM messages m WHERE m.id IN (SELECT UNNEST(?))
            AND NOT {catalog.custody_clause("m")}""",
        [list(targets.message_ids)],
    ).fetchall()
    tools = conn.execute(
        f"""SELECT t.id FROM tool_calls t WHERE t.id IN (SELECT UNNEST(?))
            AND NOT {catalog.custody_clause(tool_alias="t")}""",
        [list(targets.tool_ids)],
    ).fetchall()
    message_ids, tool_ids = {row[0] for row in messages}, {row[0] for row in tools}
    return EnrichmentTargets(
        tuple(key for key in targets.message_ids if key in message_ids),
        tuple(key for key in targets.tool_ids if key in tool_ids),
    )


@dataclass(frozen=True)
class EmbeddingCompanion:
    content: tuple[float, ...] | None
    thinking: tuple[float, ...] | None


@dataclass(frozen=True)
class CompanionInputs:
    """The required effective texts authorized by one complete companion row."""

    content: str | None
    thinking: str | None
    context_text: str

    def holds(self, final: CompanionInputs) -> bool:
        for old, new in ((self.content, final.content), (self.thinking, final.thinking)):
            if not new:
                if old:
                    return False
            elif not old or self.context_text + old != final.context_text + new:
                return False
        return True


@dataclass(frozen=True)
class OwnershipMessage:
    id: str
    content: str | None
    thinking: str | None
    context_text: str
    context_mode: str

    @classmethod
    def from_message(cls, message: Message) -> OwnershipMessage:
        return cls(
            message.id,
            message.content,
            message.thinking,
            message.context_text,
            message.context_mode,
        )


@dataclass(frozen=True)
class OwnershipRow:
    message: OwnershipMessage
    companion: EmbeddingCompanion | None


@dataclass(frozen=True)
class OwnershipDecision:
    message_id: str
    context_text: str
    context_mode: str
    companion: EmbeddingCompanion | None


PAID_CONTEXT_MODES = frozenset({"llm-local", "llm-remote", "llm-codex"})


def ownership_decisions(
    survivor: OwnershipRow, predecessor: OwnershipRow, *, context_eligible: bool
) -> OwnershipDecision:
    old, new = survivor.message, predecessor.message
    if context_eligible and old.context_mode in PAID_CONTEXT_MODES:
        context, mode = old.context_text, old.context_mode
    elif new.context_mode in PAID_CONTEXT_MODES:
        context, mode = new.context_text, new.context_mode
    elif context_eligible:
        context, mode = old.context_text, old.context_mode
    else:
        context, mode = new.context_text, new.context_mode

    def compatible(candidate: OwnershipRow) -> bool:
        if candidate.companion is None:
            return False
        message = candidate.message
        return CompanionInputs(message.content, message.thinking, message.context_text).holds(
            CompanionInputs(new.content, new.thinking, context)
        )

    companion = (
        survivor.companion
        if context_eligible and compatible(survivor)
        else predecessor.companion
        if compatible(predecessor)
        else None
    )
    return OwnershipDecision(new.id, context, mode, companion)


def _context_messages(
    conn: duckdb.DuckDBPyConnection, session_id: str, indices: tuple[int, ...]
) -> list[Message]:
    rows = conn.execute(
        MESSAGE_SELECT + " WHERE m.session_id=? AND m.idx IN (SELECT UNNEST(?)) ORDER BY m.idx",
        [session_id, list(indices)],
    ).fetchall()
    ids = [str(row[0]) for row in rows]
    tools = conn.execute(
        f"""SELECT {tool_call_columns("t")} FROM tool_calls t
            WHERE message_id IN (SELECT UNNEST(?)) ORDER BY idx,id""",
        [ids],
    ).fetchall()
    attached: dict[str, list[ToolCall]] = {}
    for row in tools:
        call = tool_call_from_row(row)
        assert call.message_id is not None
        attached.setdefault(call.message_id, []).append(call)
    messages = []
    for row in rows:
        message = message_from_row(row)
        message.tool_calls = attached.get(message.id, [])
        messages.append(message)
    return messages


def _ownership_rows(
    conn: duckdb.DuckDBPyConnection, session_id: str, indices: tuple[int, ...]
) -> dict[int, OwnershipRow]:
    positions = {message_id(session_id, index): index for index in indices}
    ids = list(positions)
    selected, bindings = id_set_predicate("message_id", ids)
    messages = [
        OwnershipMessage(*row)
        for row in conn.execute(
            f"""SELECT message_id,content,thinking,COALESCE(context_text,''),
                COALESCE(context_mode,'off') FROM message_state WHERE {selected}""",
            bindings,
        ).fetchall()
    ]
    companions = {
        row[0]: EmbeddingCompanion(
            tuple(row[1]) if row[1] is not None else None,
            tuple(row[2]) if row[2] is not None else None,
        )
        for row in conn.execute(
            f"""SELECT message_id, content_embedding, thinking_embedding FROM message_embeddings
                WHERE {selected}""",
            bindings,
        ).fetchall()
    }
    return {
        positions[message.id]: OwnershipRow(message, companions.get(message.id))
        for message in messages
    }


@dataclass(frozen=True)
class OwnershipContextProof:
    transfer_id: str
    predecessor_publish_seq: int
    last_message_eligible: bool

    def current(self, transfer: DuplicateTransfer) -> bool:
        return (self.transfer_id, self.predecessor_publish_seq) == (
            transfer.survivor_id,
            transfer.predecessor_publish_seq,
        )


def _same_context_inputs(conn: duckdb.DuckDBPyConnection, transfer: DuplicateTransfer) -> bool:
    """Compare the complete S-range projection in bounded read pages, without vectors."""
    for start in range(0, transfer.survivor_message_count, PAGE_ROWS):
        indices = tuple(range(start, min(start + PAGE_ROWS, transfer.survivor_message_count)))
        previous, current = (
            ContextInputs.project(transfer.source, _context_messages(conn, session_id, indices))
            for session_id in (transfer.survivor_id, transfer.predecessor_id)
        )
        if previous != current:
            return False
    after = -1
    # Coverage bounds S's orphan count; one final page also checks for P-only calls.
    assert transfer.survivor_orphan_count is not None
    for _ in range(transfer.survivor_orphan_count // PAGE_ROWS + 1):
        pages = [
            [
                tool_call_from_row(row)
                for row in conn.execute(
                    f"""SELECT {tool_call_columns("t")} FROM tool_calls t
                    WHERE session_id=? AND message_id IS NULL AND idx>?
                    ORDER BY idx,id LIMIT ?""",
                    [session_id, after, PAGE_ROWS],
                ).fetchall()
            ]
            for session_id in (transfer.survivor_id, transfer.predecessor_id)
        ]
        previous, current = (ContextInputs.project(transfer.source, (), page) for page in pages)
        if previous != current:
            return False
        if len(previous.orphans) < PAGE_ROWS:
            return True
        after = previous.orphans[-1].idx
    raise AssertionError("context proof exceeded the frozen orphan count")


def prove_ownership_context(
    conn: duckdb.DuckDBPyConnection, transfer: DuplicateTransfer
) -> OwnershipContextProof:
    """Read outside the writer; every later raw publication invalidates this proof."""
    from recall.db.read_snapshot import read_snapshot

    def prove(reader: duckdb.DuckDBPyConnection) -> OwnershipContextProof:
        catalog = transfer_catalog(reader)
        assert catalog is not None
        current = catalog.by_id(transfer.survivor_id)
        assert current is not None and current.phase == "owning"
        return OwnershipContextProof(
            current.survivor_id,
            current.predecessor_publish_seq,
            _same_context_inputs(reader, current),
        )

    with conn.cursor() as reader:
        return read_snapshot(reader, prove)


@dataclass
class OwnershipCursor:
    family: Literal["messages", "attached", "orphans", "complete"] = "messages"
    message_idx: int = -1
    tool_position: tuple[int, int, str] = (-1, -1, "")
    messages: int = 0
    attached: int = 0
    orphans: int = 0
    searchable_changed: bool = False


@dataclass(frozen=True)
class MessageOwnershipPage:
    after: int
    indices: tuple[int, ...]


@dataclass(frozen=True)
class ToolOwnershipPage:
    family: Literal["attached", "orphans"]
    after: tuple[int, int, str]
    rows: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class PreparedOwnershipPage:
    proof: OwnershipContextProof
    selection: MessageOwnershipPage | ToolOwnershipPage

    def current(self, transfer: DuplicateTransfer, cursor: OwnershipCursor) -> bool:
        if not self.proof.current(transfer):
            return False
        match self.selection:
            case MessageOwnershipPage(after, _):
                return cursor.family == "messages" and cursor.message_idx == after
            case ToolOwnershipPage(family, after, _):
                return cursor.family == family and cursor.tool_position == after


def prepare_ownership_page(
    conn: duckdb.DuckDBPyConnection,
    transfer: DuplicateTransfer,
    cursor: OwnershipCursor,
    proof: OwnershipContextProof,
) -> PreparedOwnershipPage | None:
    """Resolve ordered source keys outside the writer; publication fences their use."""
    from recall.db.read_snapshot import read_snapshot

    def prepare(reader: duckdb.DuckDBPyConnection) -> PreparedOwnershipPage | None:
        catalog = transfer_catalog(reader)
        assert catalog is not None
        current = catalog.by_id(transfer.survivor_id)
        if current is None or current.phase != "owning" or not proof.current(current):
            return None
        if cursor.family == "messages":
            rows = reader.execute(
                "SELECT idx FROM messages WHERE session_id=? AND idx>? ORDER BY idx LIMIT ?",
                [current.survivor_id, cursor.message_idx, PAGE_ROWS],
            ).fetchall()
            selection = MessageOwnershipPage(cursor.message_idx, tuple(int(row[0]) for row in rows))
        else:
            assert cursor.family in ("attached", "orphans")
            attached = cursor.family == "attached"
            parent = "m.idx" if attached else "CAST(-1 AS BIGINT)"
            join = "JOIN messages m ON m.id=t.message_id" if attached else ""
            predicate = "t.message_id IS NOT NULL" if attached else "t.message_id IS NULL"
            rows = reader.execute(
                f"""SELECT t.id, {parent}, t.idx FROM tool_calls t {join}
                WHERE t.session_id=? AND {predicate} AND ({parent}, t.idx, t.id) > (?, ?, ?)
                ORDER BY {parent},t.idx,t.id LIMIT ?""",
                [current.survivor_id, *cursor.tool_position, PAGE_ROWS],
            ).fetchall()
            selection = ToolOwnershipPage(
                cursor.family,
                cursor.tool_position,
                tuple((str(row[0]), int(row[1]), int(row[2])) for row in rows),
            )
        return PreparedOwnershipPage(proof, selection)

    with conn.cursor() as reader:
        return read_snapshot(reader, prepare)


def _require_covered_epoch(
    conn: duckdb.DuckDBPyConnection, transfer: DuplicateTransfer, phase: Phase
) -> None:
    assert transfer.phase == phase and transfer.covered
    item = SourceCatalog(conn, clock=time.time).get(transfer.source, transfer.moved_path)
    assert item is not None and item.content_epoch == transfer.predecessor_content_epoch


def _replace_companion_page(
    conn: duckdb.DuckDBPyConnection,
    kind: Literal["message", "tool"],
    identities: list[str],
    values: pa.Table,
) -> None:
    """Replace by primary key; a conflict join would scan the entire vector table."""
    assert 0 < len(identities) <= PAGE_ROWS and values.num_rows <= len(identities)
    table, key = (
        ("message_embeddings", "message_id")
        if kind == "message"
        else ("tool_call_embeddings", "tool_call_id")
    )
    selected, bindings = id_set_predicate(key, identities)
    conn.execute(f"DELETE FROM {table} WHERE {selected}", bindings)
    if values.num_rows:
        conn.register("_transfer_companions", values)
        try:
            conn.execute(f"INSERT INTO {table} SELECT * FROM _transfer_companions")
        finally:
            conn.unregister("_transfer_companions")


def ownership_page(
    conn: duckdb.DuckDBPyConnection,
    transfer: DuplicateTransfer,
    cursor: OwnershipCursor,
    turn: KeywordTurn,
    page: PreparedOwnershipPage,
) -> OwnershipCursor:
    """Apply at most one bounded family page. Caller retains cursors only after COMMIT."""
    import pyarrow as pa

    next_cursor = replace(cursor, searchable_changed=False)
    with turn.transaction(conn) as stage:
        _require_covered_epoch(conn, transfer, "owning")
        assert page.current(transfer, cursor)
        if cursor.family == "messages":
            assert isinstance(page.selection, MessageOwnershipPage)
            indices = page.selection.indices
            if not indices:
                next_cursor.family = "attached"
                return next_cursor
            old = _ownership_rows(conn, transfer.survivor_id, indices)
            new = _ownership_rows(conn, transfer.predecessor_id, indices)
            decisions = [
                ownership_decisions(
                    old[index],
                    new[index],
                    context_eligible=index < transfer.survivor_message_count - 1
                    or page.proof.last_message_eligible,
                )
                for index in indices
            ]
            table = pa.table(
                {
                    "id": [item.message_id for item in decisions],
                    "context_text": [item.context_text for item in decisions],
                    "context_mode": [item.context_mode for item in decisions],
                    "has_embedding": [item.companion is not None for item in decisions],
                    "content_embedding": pa.array(
                        [
                            list(item.companion.content)
                            if item.companion and item.companion.content is not None
                            else None
                            for item in decisions
                        ],
                        type=pa.list_(pa.float32()),
                    ),
                    "thinking_embedding": pa.array(
                        [
                            list(item.companion.thinking)
                            if item.companion and item.companion.thinking is not None
                            else None
                            for item in decisions
                        ],
                        type=pa.list_(pa.float32()),
                    ),
                }
            )
            conn.register("_transfer_ownership", table)
            try:
                selected, bindings = id_set_predicate(
                    "message_id", [item.message_id for item in decisions]
                )
                changed = conn.execute(
                    f"""UPDATE message_state SET
                    context_text=o.context_text, context_mode=o.context_mode,
                    fts_content=o.context_text || COALESCE(content, ''),
                    fts_thinking=o.context_text || COALESCE(thinking, '')
                    FROM _transfer_ownership o WHERE {selected} AND message_id=o.id AND
                    (message_state.context_text IS DISTINCT FROM o.context_text
                     OR message_state.context_mode IS DISTINCT FROM o.context_mode
                     OR message_state.fts_content IS DISTINCT FROM
                        o.context_text || COALESCE(content, '')
                     OR message_state.fts_thinking IS DISTINCT FROM
                        o.context_text || COALESCE(thinking, ''))
                    RETURNING message_id, fts_content, fts_thinking""",
                    bindings,
                ).fetchall()
                next_cursor.searchable_changed = bool(changed)
                stage.messages(changed)
                _replace_companion_page(
                    conn,
                    "message",
                    [item.message_id for item in decisions],
                    table.select(["id", "content_embedding", "thinking_embedding"]).filter(
                        table["has_embedding"]
                    ),
                )
            finally:
                conn.unregister("_transfer_ownership")
            next_cursor.message_idx = indices[-1]
            next_cursor.messages += len(indices)
        else:
            assert cursor.family in ("attached", "orphans")
            assert isinstance(page.selection, ToolOwnershipPage)
            attached = cursor.family == "attached"
            rows = page.selection.rows
            if not rows:
                next_cursor.family = "orphans" if attached else "complete"
                next_cursor.tool_position = (-1, -1, "")
                return next_cursor
            # Coverage already proves the positional counterpart. Construct its
            # identity instead of joining every tool with the same local index.
            targets = {
                str(identifier): tool_call_id(
                    message_id(transfer.predecessor_id, int(parent_idx)) if attached else None,
                    int(index),
                    transfer.predecessor_id,
                )
                for identifier, parent_idx, index in rows
            }
            selected, bindings = id_set_predicate("tool_call_id", list(targets))
            companions = conn.execute(
                f"SELECT tool_call_id,bash_embedding FROM tool_call_embeddings WHERE {selected}",
                bindings,
            ).fetchall()
            if companions:
                table = pa.table(
                    {
                        "id": [targets[row[0]] for row in companions],
                        "bash_embedding": pa.array(
                            [row[1] for row in companions], type=pa.list_(pa.float32())
                        ),
                    }
                )
                _replace_companion_page(
                    conn, "tool", [targets[row[0]] for row in companions], table
                )
            last = rows[-1]
            next_cursor.tool_position = (int(last[1]), int(last[2]), str(last[0]))
            if attached:
                next_cursor.attached += len(rows)
            else:
                next_cursor.orphans += len(rows)
    return next_cursor


def begin_shrink(
    conn: duckdb.DuckDBPyConnection, transfer: DuplicateTransfer, cursor: OwnershipCursor
) -> None:
    assert cursor.family == "complete"
    assert cursor.messages == transfer.survivor_message_count
    assert cursor.attached + cursor.orphans == transfer.survivor_tool_count
    assert cursor.orphans == transfer.survivor_orphan_count
    with KeywordTurn(None, None).transaction(conn):
        _require_covered_epoch(conn, transfer, "owning")
        catalog = transfer_catalog(conn)
        assert catalog is not None
        catalog.begin_shrink(transfer)


def shrink_page(
    conn: duckdb.DuckDBPyConnection, transfer: DuplicateTransfer, turn: KeywordTurn
) -> Literal["tools", "markers", "messages", "usage"] | None:
    """One family of at most 512 deletions/relinks, with authoritative counts and intents."""
    with turn.transaction(conn) as stage:
        _require_covered_epoch(conn, transfer, "shrinking")
        tools = [
            row[0]
            for row in conn.execute(
                "SELECT id FROM tool_calls WHERE session_id=? ORDER BY id LIMIT ?",
                [transfer.survivor_id, PAGE_ROWS],
            ).fetchall()
        ]
        if tools:
            for table in ("tool_call_embeddings", "tool_results", "tool_use_ids"):
                conn.execute(
                    f"DELETE FROM {table} WHERE tool_call_id IN (SELECT UNNEST(?))", [tools]
                )
            conn.execute("DELETE FROM tool_calls WHERE id IN (SELECT UNNEST(?))", [tools])
            conn.execute(
                "UPDATE session_state SET tool_count=tool_count-? WHERE session_id=?",
                [len(tools), transfer.survivor_id],
            )
            stage.removed("tool_call", tools)
            return "tools"
        markers = [
            row[0]
            for row in conn.execute(
                """SELECT message_idx FROM session_stop_markers
                    WHERE session_id=? ORDER BY message_idx LIMIT ?""",
                [transfer.survivor_id, PAGE_ROWS],
            ).fetchall()
        ]
        if markers:
            conn.execute(
                """DELETE FROM session_stop_markers
                    WHERE session_id=? AND message_idx IN (SELECT UNNEST(?))""",
                [transfer.survivor_id, markers],
            )
            return "markers"
        messages = [
            row[0]
            for row in conn.execute(
                "SELECT id FROM messages WHERE session_id=? ORDER BY id LIMIT ?",
                [transfer.survivor_id, PAGE_ROWS],
            ).fetchall()
        ]
        if messages:
            for table in ("message_embeddings", "message_state"):
                conn.execute(
                    f"DELETE FROM {table} WHERE message_id IN (SELECT UNNEST(?))", [messages]
                )
            conn.execute("DELETE FROM messages WHERE id IN (SELECT UNNEST(?))", [messages])
            conn.execute(
                "UPDATE session_state SET message_count=message_count-? WHERE session_id=?",
                [len(messages), transfer.survivor_id],
            )
            stage.removed("message", messages)
            return "messages"
        usage = [
            row[0]
            for row in conn.execute(
                "SELECT id FROM usage_events WHERE session_id=? ORDER BY id LIMIT ?",
                [transfer.survivor_id, PAGE_ROWS],
            ).fetchall()
        ]
        if usage:
            conn.execute(
                "UPDATE usage_events SET session_id=? WHERE id IN (SELECT UNNEST(?))",
                [transfer.predecessor_id, usage],
            )
            return "usage"
    return None


def finish(conn: duckdb.DuckDBPyConnection, transfer: DuplicateTransfer) -> None:
    with KeywordTurn(None, None).transaction(conn):
        _require_covered_epoch(conn, transfer, "shrinking")
        for table in (
            "messages",
            "tool_calls",
            "session_stop_markers",
            "usage_events",
            "source_files",
        ):
            assert (
                conn.execute(
                    f"SELECT 1 FROM {table} WHERE session_id=? LIMIT 1", [transfer.survivor_id]
                ).fetchone()
                is None
            )
        first, second = (
            ("s", "p") if Source(transfer.source) in HARVESTED_TOKEN_SOURCES else ("p", "s")
        )
        assignments = ", ".join(
            f"{column}=COALESCE({first}.{column}, {second}.{column})"
            for column in ("input_tokens", "output_tokens", "cached_input_tokens")
        )
        conn.execute(
            f"""UPDATE session_state p SET {assignments} FROM session_state s
                WHERE p.session_id=? AND s.session_id=?""",
            [transfer.predecessor_id, transfer.survivor_id],
        )
        conn.execute("DELETE FROM session_state WHERE session_id=?", [transfer.survivor_id])
        conn.execute("DELETE FROM sessions WHERE id=?", [transfer.survivor_id])
        catalog = transfer_catalog(conn)
        assert catalog is not None
        catalog.remove(transfer)


@dataclass
class TransferDriver:
    """One writer turn per call. Verification results enter from a separate read pass."""

    transfer_id: str | None = None
    cursor: OwnershipCursor | None = None
    coverage: CoverageResult | None = None
    context_proof: OwnershipContextProof | None = None
    ownership: PreparedOwnershipPage | None = None
    discarded_passes: int = 0
    searchable_changed: bool = False
    keyword_publication: Publication | None = None

    def turn(
        self,
        conn: duckdb.DuckDBPyConnection,
        config: AppConfig,
        *,
        attempt_publication: bool = True,
    ) -> bool:
        self.searchable_changed = False
        self.keyword_publication = None
        catalog = transfer_catalog(conn)
        if catalog is None:
            return False
        transfer = next(
            (
                item
                for item in catalog.load()
                if item.phase != "stuck" and damage_reason(conn, item) is None
            ),
            None,
        )
        if transfer is None:
            self.transfer_id, self.cursor, self.coverage = None, None, None
            self.context_proof = None
            self.ownership = None
            return False
        if self.transfer_id != transfer.survivor_id:
            self.transfer_id, self.cursor, self.coverage = transfer.survivor_id, None, None
            self.context_proof = None
            self.ownership = None
        turn = KeywordTurn.for_config(config)
        if transfer.phase == "catch_up":
            item = SourceCatalog(conn, clock=time.time).get(transfer.source, transfer.moved_path)
            assert item is not None
            if item.missing or (
                item.signature is not None and item.signature.size < transfer.survivor_offset
            ):
                with turn.transaction(conn):
                    catalog.mark_stuck(
                        transfer, "moved_file_missing" if item.missing else "moved_file_truncated"
                    )
            elif self.coverage is not None:
                if not record_coverage(conn, self.coverage):
                    self.discarded_passes += 1
                    logger.info(
                        "discarded stale transfer coverage transfer=%s discarded_passes=%d",
                        transfer.survivor_id,
                        self.discarded_passes,
                    )
                self.coverage = None
            else:
                return False
        elif transfer.phase == "owning":
            if self.context_proof is None or not self.context_proof.current(transfer):
                return False
            cursor = self.cursor or OwnershipCursor()
            if cursor.family == "complete":
                begin_shrink(conn, transfer, cursor)
                self.cursor = None
            else:
                if self.ownership is None or not self.ownership.current(transfer, cursor):
                    return False
                self.cursor = ownership_page(conn, transfer, cursor, turn, self.ownership)
                self.ownership = None
                self.searchable_changed = self.cursor.searchable_changed
        else:
            assert transfer.phase == "shrinking"
            family = shrink_page(conn, transfer, turn)
            self.searchable_changed = family in ("messages", "tools")
            if family is None:
                finish(conn, transfer)
        publication = turn.publish(attempt=attempt_publication)
        self.keyword_publication = publication
        if publication.applied:
            with KeywordTurn(None, None).transaction(conn):
                publication.retire(conn)
        return True

    def prepare(self, conn: duckdb.DuckDBPyConnection) -> bool:
        """Run outside the writer, capturing a fence before any normalized page."""
        catalog = transfer_catalog(conn)
        if catalog is None:
            return False
        if self.coverage is not None:
            return True
        transfer = next(
            (
                item
                for item in catalog.load()
                if item.phase != "stuck" and damage_reason(conn, item) is None
            ),
            None,
        )
        if transfer is None:
            return False
        if self.transfer_id != transfer.survivor_id:
            self.transfer_id, self.cursor, self.coverage = transfer.survivor_id, None, None
            self.context_proof = None
            self.ownership = None
        if transfer.phase == "catch_up" and transfer.witness_epoch is not None:
            item = SourceCatalog(conn, clock=time.time).get(transfer.source, transfer.moved_path)
            if item is not None and item.content_epoch == transfer.witness_epoch:
                self.coverage = verify_pass(conn, transfer)
        elif transfer.phase == "owning":
            if self.context_proof is None or not self.context_proof.current(transfer):
                self.context_proof = prove_ownership_context(conn, transfer)
            cursor = self.cursor or OwnershipCursor()
            if cursor.family != "complete":
                self.ownership = prepare_ownership_page(conn, transfer, cursor, self.context_proof)
        return True


@dataclass(frozen=True)
class ValidatedRefusal:
    transfer_id: str
    kind: RefusalKind
    observation: RefusalObservation


def record_predecessor_refusal(
    conn: duckdb.DuckDBPyConnection, capture: PreparedRawSource, error: Exception
) -> ValidatedRefusal | None:
    """Record only a stable rejected capture in the catalog-failure transaction."""
    from recall.parsers.revision import parser_revision
    from recall.services.coordinator import capture_path
    from recall.services.indexer import _capture_still_matches

    catalog = transfer_catalog(conn)
    if catalog is None:
        return None
    transfer = catalog.for_path(capture.item.source, capture.item.source_path)
    if transfer is None or transfer.phase != "catch_up":
        return None
    if not isinstance(error, OrdinaryCaptureRefused):
        if str(error) == "source changed while parser capture was in flight":
            catalog.clear_refusal(transfer)
        return None
    if capture.result is None or capture.observed is None:
        return None
    item = SourceCatalog(conn, clock=time.time).get(transfer.source, transfer.moved_path)
    if (
        item is None
        or capture.item.signature != capture.observed[2]
        or item.signature != capture.observed[2]
        or item.desired_generation != capture.item.desired_generation
    ):
        catalog.clear_refusal(transfer)
        return None
    try:
        stable = (
            _capture_still_matches(Path(transfer.moved_path), capture.result)
            and capture_path(capture.parser, Path(transfer.moved_path)) == capture.observed
        )
    except OSError:
        stable = False
    if not stable:
        catalog.clear_refusal(transfer)
        return None
    observation = RefusalObservation.capture(
        item, capture.observed[2], parser_revision(type(capture.parser))
    )
    previous = (transfer.predecessor_refusal_kind, transfer.predecessor_refusal_observation)
    catalog.record_refusal(transfer, error.kind, observation)
    if previous == (error.kind, observation.encode()):
        return ValidatedRefusal(transfer.survivor_id, error.kind, observation)
    return None


def finish_predecessor_refusal(
    conn: duckdb.DuckDBPyConnection, event: ValidatedRefusal, parser: SessionParser
) -> None:
    """A second actual retry supplies this event; polls and restarts cannot invent it."""
    from recall.parsers.revision import parser_revision
    from recall.services.coordinator import capture_path

    catalog = transfer_catalog(conn)
    assert catalog is not None
    with KeywordTurn(None, None).transaction(conn):
        transfer = catalog.by_id(event.transfer_id)
        if transfer is None or transfer.phase != "catch_up" or transfer.covered:
            return
        if (transfer.predecessor_refusal_kind, transfer.predecessor_refusal_observation) != (
            event.kind,
            event.observation.encode(),
        ):
            return
        item = SourceCatalog(conn, clock=time.time).get(transfer.source, transfer.moved_path)
        assert item is not None
        try:
            observed = capture_path(parser, Path(transfer.moved_path))[2]
        except OSError:
            catalog.clear_refusal(transfer)
            return
        current = RefusalObservation.capture(item, observed, parser_revision(type(parser)))
        if current != event.observation or item.signature != observed:
            catalog.clear_refusal(transfer)
            return
        catalog.mark_stuck(transfer, "predecessor_refused")


def transfer_report(conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    catalog = transfer_catalog(conn)
    if catalog is None:
        return []
    reports: list[dict[str, Any]] = []
    for transfer in catalog.load():
        damage = damage_reason(conn, transfer)
        item = SourceCatalog(conn, clock=time.time).get(transfer.source, transfer.moved_path)
        counts = {}
        for name, table in (
            ("messages", "messages"),
            ("tools", "tool_calls"),
            ("markers", "session_stop_markers"),
            ("usage", "usage_events"),
        ):
            row = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id=?", [transfer.survivor_id]
            ).fetchone()
            assert row is not None
            counts[name] = row[0]
        reports.append(
            {
                **asdict(transfer),
                "resolution": "damaged"
                if damage
                else "apply_ends"
                if transfer.phase == "stuck"
                else "running",
                "damage_reason": damage,
                "remaining": counts,
                "predecessor_committed_offset": item.committed_offset if item else None,
                "predecessor_catalog_epoch": item.content_epoch if item else None,
                "predecessor_last_error": item.last_error if item else None,
            }
        )
    return reports


APPLY_TURNS_MAX = 1_000_000
APPLY_SECONDS_MAX = 3600


def apply_transfers(
    conn: duckdb.DuckDBPyConnection, config: AppConfig, plans: list[MovedDuplicate]
) -> int:
    """Drive bounded turns under the caller's exclusive offline ownership."""
    from recall.core.config import SourceConfig
    from recall.db.source_files import source_key
    from recall.parsers import get_parser
    from recall.services.coordinator import (
        capture_path,
        commit_prepared_raw_sources,
        observe_path,
        prepare_raw_sources,
        with_transfer_witness,
    )

    catalog = transfer_catalog(conn)
    if catalog is None:
        raise RuntimeError(
            "duplicate transfer apply requires schema 38; migrate the database first"
        )
    for transfer in catalog.load():
        if transfer.phase == "stuck" and damage_reason(conn, transfer) is None:
            with KeywordTurn(None, None).transaction(conn):
                catalog.end_stuck(transfer)
    for plan in plans:
        rebind(conn, plan)
    driver = TransferDriver()
    finished = 0
    deadline = time.monotonic() + APPLY_SECONDS_MAX
    for _ in range(APPLY_TURNS_MAX):
        transfers = [item for item in catalog.load() if damage_reason(conn, item) is None]
        if not transfers:
            return finished
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "duplicate transfer apply deadline reached; durable transfers remain resumable"
            )
        transfer = transfers[0]
        if transfer.phase == "stuck":
            with KeywordTurn(None, None).transaction(conn):
                catalog.end_stuck(transfer)
            continue
        if transfer.phase == "catch_up":
            item = SourceCatalog(conn, clock=time.time).get(transfer.source, transfer.moved_path)
            assert item is not None
            # Offline maintenance acts on this proven historical path, even
            # when current discovery roots no longer include its catalog root.
            parser = get_parser(
                Source(transfer.source),
                {**config.sources, transfer.source: SourceConfig(roots=(Path(item.root_path),))},
            )
            try:
                item = observe_path(
                    parser, capture_path(parser, Path(transfer.moved_path)), conn=conn
                )
            except FileNotFoundError:
                SourceCatalog(conn, clock=time.time).mark_missing(
                    [source_key(transfer.source, transfer.moved_path)]
                )
            else:
                if (
                    item.signature is not None
                    and item.signature.size >= transfer.survivor_offset
                    and (not item.current or transfer.witness_epoch is None)
                ):
                    if item.next_retry_at > time.time():
                        time.sleep(min(1.0, item.next_retry_at - time.time()))
                        continue
                    prepared = prepare_raw_sources(
                        (with_transfer_witness(conn, item),),
                        {transfer.source: parser},
                        full=item.current,
                    )
                    commit_prepared_raw_sources(prepared, config, conn=conn)
        driver.prepare(conn)
        driver.turn(conn, config)
        if catalog.by_id(transfer.survivor_id) is None:
            finished += 1
    raise RuntimeError(
        "duplicate transfer apply turn limit reached; durable transfers remain resumable"
    )
