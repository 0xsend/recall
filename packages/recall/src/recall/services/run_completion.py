"""Finite-scope outcomes and their single owner-arbitrated completion transaction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from recall.core.daemon_status import IndexRunCounts
from recall.core.time import to_naive_utc, utcnow_naive
from recall.core.types import RunKind
from recall.db.fts_sidecar_stage import Publication
from recall.db.lease import CompletionPermit, DbLease, TransactionCapability
from recall.db.repair import RepairRow, load_repair_row, save_repair_row
from recall.services.runtime_state import _ensure_runtime_state_row

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
    problem: str | None

    def finalize_in(self, transaction: TransactionCapability) -> Qualified | Partial:
        self.publication.retire_on(transaction)
        counts = self.record.counts
        if counts is not None and counts.failed > 0:
            return Partial(self.record, f"{counts.failed} selected items failed")
        if self.problem is not None:
            return Partial(self.record, self.problem)
        if self.publication.outcome in {"deferred", "skipped"}:
            return Partial(self.record, "keyword publication for the selected scope is unresolved")
        if self.publication.scope_pending_on(transaction):
            return Partial(self.record, "published scope still has unresolved keyword intents")
        return Qualified(self.record, _SettledScope())


type CompletionRequest = PreparedCompletion | Failed | Cancelled


class RunScope:
    """Collect bounded evidence while the declared inventory and items run.

    Counts and a returned payload alone supply neither inventory completeness
    nor publication acknowledgement. The producer records each obligation
    where it is observed; qualification follows retirement in the final transaction.
    """

    def __init__(self) -> None:
        self._inventory_complete = False
        self._keyword_complete = False
        self._problem: str | None = None

    def inventory_completed(self) -> None:
        self._inventory_complete = True

    def keyword_acknowledged(self) -> None:
        self._keyword_complete = True

    def incomplete(self, reason: str) -> None:
        if self._problem is None:
            self._problem = reason[:500] or "the selected scope is incomplete"

    def publication(self, publication: Publication) -> None:
        if publication.outcome in {"deferred", "skipped"}:
            self.incomplete("keyword publication for the selected scope is unresolved")

    def prepare(self, record: RunRecord, publication: Publication) -> PreparedCompletion:
        problem = self._problem
        if problem is None and not self._inventory_complete:
            problem = "the intended inventory did not complete"
        if problem is None and not self._keyword_complete:
            problem = "keyword reconciliation for the selected scope is unresolved"
        return PreparedCompletion(record, publication, problem)


class CompletionCommitUncertain(RuntimeError):
    """Never replay this completion; close and fresh-read before later admission."""


def _no_barrier(stage: str) -> None:
    pass


def record_completion_on(
    lease: DbLease,
    permit: CompletionPermit,
    completion: CompletionRequest,
    *,
    completed_at: datetime | None = None,
    barrier: Callable[[str], None] = _no_barrier,
) -> datetime:
    """Execute once, including retirement, with no automatic retry authority."""
    timestamp = to_naive_utc(completed_at) if completed_at is not None else utcnow_naive()
    with lease.completing(permit):
        commit_started = False
        try:
            with lease.transaction("run_completion") as transaction:
                apply_completion_in(transaction, completion, timestamp, barrier)
                barrier("before_completion_commit")
                commit_started = True
            barrier("after_completion_commit")
        except Exception as error:
            if commit_started:
                raise CompletionCommitUncertain("run completion commit is uncertain") from error
            raise
    return timestamp


def apply_completion_in(
    transaction: TransactionCapability,
    request: CompletionRequest,
    timestamp: datetime,
    barrier: Callable[[str], None],
) -> None:
    """Apply within the caller's active outer transaction; never begin/commit."""
    transaction.call("completion:singleton_seed", _ensure_runtime_state_row)
    barrier("after_completion_seed")
    completion = (
        request.finalize_in(transaction) if isinstance(request, PreparedCompletion) else request
    )
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
