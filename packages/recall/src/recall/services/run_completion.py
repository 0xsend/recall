"""Finite-scope outcomes and their single owner-arbitrated completion transaction."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Literal

import duckdb

from recall.core.daemon_status import IndexRunCounts
from recall.core.repair_state import Action, Admission, Event, Refusal, decide
from recall.core.source_path import SourcePath
from recall.core.time import to_naive_utc, utcnow_naive
from recall.core.types import RunKind
from recall.db.fts_sidecar_stage import Publication
from recall.db.lease import CompletionPermit, DbLease, TransactionCapability
from recall.db.repair import RepairRow, load_repair_row, save_repair_row
from recall.services.repair_events import EventStorageError, serialized_events
from recall.services.runtime_state import _ensure_runtime_state_row, record_run_attempt

if TYPE_CHECKING:
    from recall.services.indexer import IndexSummary


@dataclass(frozen=True)
class RunContext:
    messages: int = 0
    mode: str = "off"
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None


@dataclass(frozen=True)
class RunRecord:
    run_kind: RunKind
    attempted_at: datetime
    counts: IndexRunCounts | None = None
    context: RunContext | None = None


@dataclass(frozen=True)
class RunAttempt:
    permit: CompletionPermit
    run_kind: RunKind
    attempted_at: datetime

    def record(self) -> RunRecord:
        return RunRecord(self.run_kind, self.attempted_at)


def begin_run_on(lease: DbLease, run_kind: RunKind) -> RunAttempt:
    permit = lease.completion_permit()
    attempted_at = lease.call(
        "run:attempt", lambda conn: record_run_attempt(conn, run_kind=run_kind)
    )
    return RunAttempt(permit, run_kind, attempted_at)


@dataclass(frozen=True)
class MessageRoster:
    """Native frozen target roster retained through requested completion."""

    table: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"_recall_recompute_[0-9a-f]{32}", self.table) is None:
            raise ValueError("invalid native message roster identity")

    def _problem_in(self, transaction: TransactionCapability) -> str | None:
        row = transaction.call(
            "completion:message_roster",
            lambda conn: conn.execute(
                f"SELECT EXISTS(SELECT 1 FROM {self.table} WHERE NOT completed), "
                "EXISTS(SELECT 1 FROM fts_sidecar_pending p "
                f"JOIN {self.table} selected ON p.kind='message' AND p.id=selected.id)"
            ).fetchone(),
        )
        assert row is not None
        if row[0]:
            return "selected message targets remain incomplete"
        if row[1]:
            return "selected message targets retain unresolved keyword intents"
        return None


@dataclass(frozen=True)
class SourceBatch:
    paths: tuple[SourcePath, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.paths) <= 2 or len(set(self.paths)) != len(self.paths):
            raise ValueError(
                "a captured source batch contains one or two distinct source identities"
            )

    def _problem_in(self, transaction: TransactionCapability) -> str | None:
        row = transaction.call(
            "completion:source_keyword_membership",
            lambda conn: conn.execute(
                """
                WITH selected AS (
                    SELECT unnest(?::VARCHAR[]) AS source,
                           unnest(?::VARCHAR[]) AS source_path
                ), membership AS (
                    SELECT count(*) FILTER (WHERE s.id IS NULL) > 0 AS unknown,
                           count(*) FILTER (WHERE selected.source IS NOT NULL) > 0 AS pending
                    FROM fts_sidecar_pending p
                    LEFT JOIN messages m ON p.kind = 'message' AND m.id = p.id
                    LEFT JOIN tool_calls t ON p.kind = 'tool_call' AND t.id = p.id
                    LEFT JOIN sessions s ON s.id = coalesce(m.session_id, t.session_id)
                    LEFT JOIN selected ON selected.source = s.source
                                      AND selected.source_path = s.source_path
                ), bindings AS (
                    SELECT count(*) FILTER (WHERE s.id IS NULL) > 0 AS missing
                    FROM selected
                    LEFT JOIN source_files sf USING (source, source_path)
                    LEFT JOIN sessions s ON s.id = sf.session_id
                )
                SELECT bindings.missing, membership.unknown, membership.pending
                FROM bindings, membership
                """,
                [[path.source for path in self.paths], [path.path for path in self.paths]],
            ).fetchone(),
        )
        assert row is not None
        missing, unknown, pending = row
        if missing:
            return "captured source lacks its acknowledged canonical session"
        if unknown:
            return "unattributed keyword intents prevent proving the selected scope"
        if pending:
            return "the selected source scope still has unresolved keyword intents"
        return None


def index_record(run_kind: RunKind, attempted_at: datetime, summary: IndexSummary) -> RunRecord:
    return RunRecord(
        run_kind,
        attempted_at,
        IndexRunCounts(
            total=summary.total,
            indexed=summary.indexed,
            skipped=summary.skipped,
            failed=summary.failed,
            changed=summary.changed,
            total_seconds=summary.total_seconds,
        ),
        RunContext(
            summary.context_messages,
            summary.context_mode,
            summary.context_input_tokens,
            summary.context_output_tokens,
            summary.context_model,
        ),
    )


@dataclass(frozen=True)
class _SettledScope:
    """Only a finished scope can supply this constructor argument."""


@dataclass(frozen=True)
class Qualified:
    record: RunRecord
    _scope: _SettledScope
    kind: Literal["qualified"] = field(default="qualified", init=False)


@dataclass(frozen=True)
class Partial:
    record: RunRecord
    message: str
    kind: Literal["partial"] = field(default="partial", init=False)


@dataclass(frozen=True)
class Failed:
    record: RunRecord
    message: str
    kind: Literal["failed"] = field(default="failed", init=False)


@dataclass(frozen=True)
class Cancelled:
    record: RunRecord
    message: str
    kind: Literal["cancelled"] = field(default="cancelled", init=False)


type RunCompletion = Qualified | Partial | Failed | Cancelled


@dataclass(frozen=True)
class PreparedCompletion:
    record: RunRecord
    publication: Publication
    scope: RunScope

    def _finalize_in(
        self, transaction: TransactionCapability, evidence: _ScopeEvidence
    ) -> RunCompletion:
        self.publication.retire_on(transaction)
        if evidence.cancelled is not None:
            return Cancelled(self.record, evidence.cancelled)
        counts = self.record.counts
        if counts is not None and counts.failed > 0:
            return Partial(self.record, f"{counts.failed} selected items failed")
        problem = evidence.problem_reason()
        if problem is not None:
            return Partial(self.record, problem)
        if self.publication.outcome in {"deferred", "skipped"}:
            return Partial(self.record, "keyword publication for the selected scope is unresolved")
        if self.publication.scope_pending_on(transaction):
            return Partial(self.record, "published scope still has unresolved keyword intents")
        if evidence.source_batch is not None:
            problem = evidence.source_batch._problem_in(transaction)
            if problem is not None:
                return Partial(self.record, problem)
        if evidence.message_roster is not None:
            problem = evidence.message_roster._problem_in(transaction)
            if problem is not None:
                return Partial(self.record, problem)
        return Qualified(self.record, _SettledScope())


type CompletionRequest = PreparedCompletion | Failed | Cancelled


@dataclass(frozen=True)
class _ScopeEvidence:
    cancelled: str | None
    inventory_complete: bool
    keyword_complete: bool
    problem: str | None
    source_batch: SourceBatch | None
    message_roster: MessageRoster | None

    def problem_reason(self) -> str | None:
        if self.problem is not None:
            return self.problem
        if not self.inventory_complete:
            return "the intended inventory did not complete"
        if not self.keyword_complete:
            return "keyword reconciliation for the selected scope is unresolved"
        return None


class RunScope:
    """Collect bounded evidence while the declared inventory and items run.

    Counts and a returned payload alone supply neither inventory completeness
    nor publication acknowledgement. The producer records each obligation
    where it is observed; qualification follows retirement in the final transaction.
    """

    def __init__(self) -> None:
        self._arbiter = RLock()
        self._admitted = False
        self._cancelled: str | None = None
        self._inventory_complete = False
        self._keyword_complete = False
        self._problem: str | None = None
        self._source_batch: SourceBatch | None = None
        self._message_roster: MessageRoster | None = None

    def declare_messages(self, roster: MessageRoster) -> None:
        with self._arbiter:
            assert not self._admitted and self._message_roster is None, "scope is already declared"
            self._message_roster = roster

    def declare_sources(self, batch: SourceBatch) -> None:
        with self._arbiter:
            assert not self._admitted and self._source_batch is None, "scope is already declared"
            self._source_batch = batch

    def inventory_completed(self) -> None:
        with self._arbiter:
            assert not self._admitted, "run scope has finished"
            self._inventory_complete = True

    def keyword_acknowledged(self) -> None:
        with self._arbiter:
            assert not self._admitted, "run scope has finished"
            self._keyword_complete = True

    def incomplete(self, reason: str) -> None:
        with self._arbiter:
            if not self._admitted and self._problem is None:
                self._problem = reason[:500] or "the selected scope is incomplete"

    def cancel(self, reason: str) -> bool:
        """Order departure against completion; False means completion was admitted."""
        with self._arbiter:
            if self._admitted:
                return False
            self._cancelled = reason[:500] or "cancelled run"
            return True

    @contextmanager
    def completing(self) -> Iterator[_ScopeEvidence]:
        with self._arbiter:
            assert not self._admitted, "run scope has finished"
            self._admitted = True
            evidence = _ScopeEvidence(
                self._cancelled,
                self._inventory_complete,
                self._keyword_complete,
                self._problem,
                self._source_batch,
                self._message_roster,
            )
        yield evidence

    def publication(self, publication: Publication) -> None:
        if publication.outcome in {"deferred", "skipped"}:
            self.incomplete("keyword publication for the selected scope is unresolved")

    def prepare(self, record: RunRecord, publication: Publication) -> PreparedCompletion:
        return PreparedCompletion(record, publication, self)


class CompletionCommitUncertain(RuntimeError):
    """Never replay this completion; close and fresh-read before later admission."""


class CompletionAdmissionChanged(RuntimeError):
    """A new instruction invalidated tentative completion; do not replay it."""


def _completion_admission(lease: DbLease, data_dir: Path) -> Admission:
    from recall.services.self_repair import (
        DaemonStartupRefused,
        admitted_recovery_files,
        write_refusal_marker,
    )

    try:
        batch = admitted_recovery_files(data_dir)
        row = lease.call("completion:admission_state", load_repair_row)
        state = Admission(
            row.durable_phase, batch.classify(row.event_dispositions), Refusal.ABSENT, row.budget
        )
        if decide(state, Event.PARTIAL) != Action.PARTIAL:
            raise DaemonStartupRefused(
                "retained recovery state prevents run completion; "
                "use manual repair or matched restore"
            )
    except (DaemonStartupRefused, EventStorageError, ValueError, duckdb.Error) as error:
        # Native observations retain their original identity and owner policy.
        if lease.observe(error, "completion:admission"):
            raise
        refused = (
            error
            if isinstance(error, DaemonStartupRefused)
            else DaemonStartupRefused(f"invalid recovery admission: {error}")
        )
        write_refusal_marker(data_dir, str(refused))
        lease.reject_admission(refused, "completion:admission")
    return state


def _no_barrier(stage: str) -> None:
    pass


def record_completion_on(
    lease: DbLease,
    permit: CompletionPermit,
    completion: CompletionRequest,
    *,
    data_dir: Path,
    completed_at: datetime | None = None,
    barrier: Callable[[str], None] = _no_barrier,
) -> datetime:
    """Execute once, including retirement, with no automatic retry authority."""
    timestamp = to_naive_utc(completed_at) if completed_at is not None else utcnow_naive()
    scope = (
        completion.scope.completing()
        if isinstance(completion, PreparedCompletion)
        else nullcontext()
    )
    with lease.completing(permit), scope as evidence, serialized_events():
        admission = _completion_admission(lease, data_dir)
        commit_started = False
        try:
            with lease.transaction("run_completion") as transaction:
                result = _apply_completion_in(
                    transaction, completion, timestamp, barrier, admission, evidence
                )
                barrier("before_completion_commit")
                current = _completion_admission(lease, data_dir)
                if (
                    isinstance(result, Qualified)
                    and decide(current, Event.SUCCESS) != Action.SUCCESS
                ):
                    raise CompletionAdmissionChanged(
                        "recovery instruction arrived during completion"
                    )
                commit_started = True
            barrier("after_completion_commit")
        except Exception as error:
            if commit_started:
                raise CompletionCommitUncertain("run completion commit is uncertain") from error
            raise
    return timestamp


def _apply_completion_in(
    transaction: TransactionCapability,
    request: CompletionRequest,
    timestamp: datetime,
    barrier: Callable[[str], None],
    admission: Admission,
    evidence: _ScopeEvidence | None,
) -> RunCompletion:
    """Apply within the caller's active outer transaction; never begin/commit."""
    transaction.call("completion:singleton_seed", _ensure_runtime_state_row)
    barrier("after_completion_seed")
    if isinstance(request, PreparedCompletion):
        assert evidence is not None, "prepared completion requires admitted scope evidence"
        completion = request._finalize_in(transaction, evidence)
    else:
        completion = request
    if isinstance(completion, Qualified) and decide(admission, Event.SUCCESS) != Action.SUCCESS:
        completion = Partial(completion.record, "a retained recovery instruction prevents success")
    barrier("after_completion_retirement")
    record = completion.record
    assignments = ["last_attempted_at = ?", "last_run_kind = ?"]
    values: list[object] = [to_naive_utc(record.attempted_at), record.run_kind.value]
    if isinstance(completion, Qualified):
        assignments.extend(
            ["last_successful_at = ?", "last_failure_message = NULL", "last_failure_at = NULL"]
        )
        values.append(timestamp)
    else:
        assignments.extend(["last_failure_message = ?", "last_failure_at = ?"])
        values.extend([completion.message.strip() or f"{completion.kind} run", timestamp])
    if record.context is not None:
        context = record.context
        assignments.extend(
            [
                "last_context_messages = ?",
                "last_context_mode = ?",
                "last_context_input_tokens = ?",
                "last_context_output_tokens = ?",
                "last_context_model = ?",
            ]
        )
        values.extend(
            [
                context.messages,
                context.mode,
                context.input_tokens,
                context.output_tokens,
                context.model,
            ]
        )
    transaction.call(
        "completion:history",
        lambda conn: conn.execute(
            "UPDATE runtime_state SET " + ", ".join(assignments) + " WHERE singleton = TRUE", values
        ).fetchone(),
    )
    barrier("after_completion_history")
    if record.counts is not None:
        counts = record.counts
        transaction.call(
            "completion:counts",
            lambda conn: conn.execute(
                "UPDATE runtime_state SET last_index_total=?, last_index_indexed=?, "
                "last_index_skipped=?, last_index_failed=?, last_index_changed=?, "
                "last_index_total_seconds=? WHERE singleton = TRUE",
                [
                    counts.total,
                    counts.indexed,
                    counts.skipped,
                    counts.failed,
                    counts.changed,
                    counts.total_seconds,
                ],
            ).fetchone(),
        )
    barrier("after_completion_counts")
    if isinstance(completion, Qualified):
        row = transaction.call("completion:repair_state", load_repair_row)
        assert row.phase in {"idle", "verified"}, "unfinished recovery cannot complete a run"
        released = RepairRow.model_validate(
            row.model_dump() | {"success_since_attempt": True, "success_since_failure": True}
        )
        transaction.call("completion:repair_flags", lambda conn: save_repair_row(conn, released))
    barrier("after_completion_flags")
    return completion
