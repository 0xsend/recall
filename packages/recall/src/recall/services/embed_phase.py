"""Embed phase: adaptive batched embedding for the daemon pipeline."""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import duckdb

from recall.core.config import AppConfig
from recall.core.daemon_status import SemanticCoverage, SemanticWorkStatus
from recall.core.embeddings import EmbeddingBackend
from recall.core.models import Session, ToolCall
from recall.core.telemetry import Counter, Gauge, Stage, count, gauge, measured
from recall.db.enrichment_inputs import (
    TARGET_ROWS_MAX,
    EnrichmentCapacityError,
    EnrichmentTargets,
    admit_targets,
    read_enrichment_session,
    read_target_masks,
    select_pending_targets,
    source_generation,
)
from recall.db.enrichment_inputs import (
    EmbedInput as EmbedInput,
)
from recall.db.enrichment_inputs import (
    EmbedToolInput as EmbedToolInput,
)
from recall.db.enrichment_inputs import (
    GeneratedEmbedSession as GeneratedEmbedSession,
)
from recall.db.enrichment_inputs import (
    PreparedEmbedSession as PreparedEmbedSession,
)
from recall.db.enrichment_inputs import (
    _document_input as _document_input,
)
from recall.db.enrichment_inputs import (
    _inputs_still_match as _inputs_still_match,
)
from recall.db.enrichment_inputs import (
    snapshot_embed_session as snapshot_embed_session,
)
from recall.db.semantic_work import PENDING_COUNTS_SQL, read_snapshot, require_complete

logger = logging.getLogger("recall.embed_phase")

# Commit rejections one session may accumulate before it is deprioritized so it
# cannot stop the drain. Counted per session, and only where a commit was
# attempted: a cycle short-circuited by the no-progress backoff discards no
# result, so counting it would make this threshold fire on one real rejection
# plus an idle timer tick.
COMMIT_REJECTION_COOLDOWN_THRESHOLD = 2
# Bound on simultaneously deprioritized sessions; a full map evicts the
# soonest-expiring entry.
SESSION_COOLDOWN_LIMIT = 64
# Bound on sessions carrying a rejection count; a full map evicts the
# least-rejected entry, which is the one furthest from deprioritization.
SESSION_REJECTION_LIMIT = 64


@dataclass(frozen=True)
class PendingEmbeds:
    message_count: int
    tool_call_count: int
    session_ids: list[str]
    # Timing metadata does not change the eligible backlog signature/equality.
    eligible_after: float | None = field(default=None, compare=False)

    @property
    def total(self) -> int:
        return self.message_count + self.tool_call_count

    @property
    def signature(self) -> tuple[int, int, tuple[str, ...]]:
        return (self.message_count, self.tool_call_count, tuple(self.session_ids))


@dataclass(frozen=True)
class _IdleEmbedSnapshot:
    conn: duckdb.DuckDBPyConnection
    config: AppConfig
    generation: object
    eligible_after: float | None

    def matches(
        self,
        conn: duckdb.DuckDBPyConnection,
        config: AppConfig,
        generation: object | None,
        now: float,
    ) -> bool:
        return (
            self.conn is conn
            and self.config == config
            and self.generation is generation
            and (self.eligible_after is None or now <= self.eligible_after)
        )


@measured(Stage.EMBED_SELECT)
def find_pending_embeds(
    conn: duckdb.DuckDBPyConnection,
    idle_threshold: int,
    *,
    max_sessions: int | None = None,
    now: float | None = None,
) -> PendingEmbeds:
    """Pending work in sessions idle for at least `idle_threshold` seconds.

    The counts cover every eligible session; only the returned roster, oldest
    first, is bounded by `max_sessions`. `eligible_after` is the earliest
    held pending deadline; processed sessions never create a deadline.
    """
    if max_sessions is not None and max_sessions <= 0:
        raise ValueError("max_sessions must be positive")
    require_complete(conn)
    now_epoch = time.time() if now is None else now
    cutoff = now_epoch - idle_threshold
    # Window totals are taken before LIMIT, so a one-session roster still
    # carries the whole eligible backlog.
    rows = conn.execute(
        f"""
        WITH {PENDING_COUNTS_SQL}, by_session AS (
            SELECT session_id, file_mtime, SUM(messages) AS messages, SUM(tools) AS tools
            FROM pending WHERE selectable AND messages+tools > 0 AND file_mtime < ?
            GROUP BY session_id, file_mtime
        ), roster AS (
            SELECT session_id, file_mtime,
                   SUM(messages) OVER () AS messages, SUM(tools) OVER () AS tools
            FROM by_session ORDER BY file_mtime, session_id LIMIT ?
        ), held AS (
            SELECT MIN(file_mtime) AS file_mtime FROM pending
            WHERE selectable AND messages+tools > 0 AND file_mtime >= ?
        )
        SELECT r.session_id, r.messages, r.tools, h.file_mtime
        FROM held h LEFT JOIN roster r ON TRUE
        ORDER BY r.file_mtime, r.session_id
        """,
        [cutoff, max_sessions, cutoff],
    ).fetchall()
    assert rows, "the held aggregate returns one row even with no pending targets"
    return PendingEmbeds(
        message_count=int(rows[0][1]) if rows[0][1] is not None else 0,
        tool_call_count=int(rows[0][2]) if rows[0][2] is not None else 0,
        session_ids=[str(row[0]) for row in rows if row[0] is not None],
        eligible_after=float(rows[0][3]) + idle_threshold if rows[0][3] is not None else None,
    )


@dataclass(frozen=True)
class PendingSplit:
    """Unembedded targets read at `measured_at`, split by the idle hold.

    `eligible` counts targets the embed loop may select now; `held` counts
    targets in sessions still inside `embed_idle_session`. Embedding is drained
    only when both are zero.
    """

    eligible: int
    held: int
    measured_at: float
    semantic_coverage: SemanticCoverage | None = field(default=None, compare=False)
    semantic_work: SemanticWorkStatus | None = field(default=None, compare=False)


def count_pending_split(
    conn: duckdb.DuckDBPyConnection, idle_threshold: int, *, now: float | None = None
) -> PendingSplit:
    with read_snapshot(conn) as snapshot:
        return _count_pending_split(snapshot, idle_threshold, now=now)


def _count_pending_split(
    conn: duckdb.DuckDBPyConnection, idle_threshold: int, *, now: float | None = None
) -> PendingSplit:
    """Count eligible and held pending targets in one statement (REQ-ADAPT-022).

    A session whose mtime equals the cutoff is held, matching the strict `<`
    that `find_pending_embeds` selects with.
    """
    semantic_work = require_complete(conn)
    measured_at = time.time() if now is None else now
    cutoff = measured_at - idle_threshold
    row = conn.execute(
        f"""
        WITH {PENDING_COUNTS_SQL}
        SELECT COALESCE(SUM(messages+tools) FILTER (WHERE selectable AND file_mtime < ?),0),
               COALESCE(SUM(messages+tools) FILTER (WHERE selectable AND file_mtime >= ?),0),
               COALESCE(SUM(missing_content),0), COALESCE(SUM(missing_thinking),0),
               COALESCE(SUM(missing_bash),0), COALESCE(SUM(blocked),0)
        FROM pending
        """,
        [cutoff, cutoff],
    ).fetchone()
    assert row is not None, "an aggregate without GROUP BY returns one row"
    coverage: SemanticCoverage = {
        "scope": "indexed_history",
        "complete": all(int(value) == 0 for value in row[2:]),
        "missing_content": int(row[2]),
        "missing_thinking": int(row[3]),
        "missing_bash": int(row[4]),
        "unselectable_targets": int(row[5]),
        "measured_at": measured_at,
        "error": None,
    }
    return PendingSplit(
        eligible=int(row[0]),
        held=int(row[1]),
        measured_at=measured_at,
        semantic_coverage=coverage,
        semantic_work=semantic_work,
    )


def find_eligible_pending_embeds(
    conn: duckdb.DuckDBPyConnection,
    idle_threshold: int,
    *,
    max_sessions: int,
    cooled: frozenset[str],
    now: float | None = None,
) -> PendingEmbeds:
    """Pending work whose roster excludes the deprioritized sessions.

    A cooled-down session stays in the global backlog counts but must not
    occupy a roster slot, so the query overfetches by the bounded cooldown set
    and filters in memory. Every roster the embed phase compares goes through
    here: a deprioritized session keeps its place in the raw oldest-first
    order, so a signature taken from the unfiltered roster could never match
    the next cycle's snapshot and the backoff pacing would be bypassed
    (`REQ-ADAPT-016`).
    """
    assert max_sessions > 0, "max_sessions must be positive"
    pending = find_pending_embeds(
        conn, idle_threshold, max_sessions=max_sessions + len(cooled), now=now
    )
    if not cooled:
        return pending
    return PendingEmbeds(
        message_count=pending.message_count,
        tool_call_count=pending.tool_call_count,
        session_ids=[sid for sid in pending.session_ids if sid not in cooled][:max_sessions],
        eligible_after=pending.eligible_after,
    )


@dataclass
class EmbedPhaseState:
    """Mutable state for the embed phase timer."""

    _config: AppConfig = field(repr=False)
    _idle_snapshot: _IdleEmbedSnapshot | None = field(default=None, repr=False)
    _selection_cursors: dict[str, tuple[str, int, str]] = field(
        default_factory=dict, repr=False
    )  # required constructor argument
    last_used_at: float = 0.0
    last_batch_at: float = 0.0
    last_batch_size: int = 0
    last_batch_duration: float = 0.0
    last_pending: PendingEmbeds | None = None
    # When `last_pending` was observed. A deferred cycle short-circuits before
    # the snapshot, so the count stands still while the backlog grows; the
    # observation time is what separates a frozen count from a stale one.
    last_pending_at: float = 0.0
    stalled_pending_signature: tuple[int, int, tuple[str, ...]] | None = None
    stalled_until: float = 0.0
    backend: EmbeddingBackend | None = None
    last_error: str | None = None
    # The deferral reason and the moment it was evaluated. A reason naming a
    # power-dependent threshold reads the same live as hours stale, so it is
    # only checkable against the host beside its own timestamp.
    _deferred_reason: str | None = None
    _deferred_at: float = 0.0
    # Timer-loop liveness. A stalled loop and an idle loop both stop advancing
    # `last_batch_at`, so the stage it last entered is what makes a stall
    # diagnosable from `daemon status` alone. These fields belong to the timer:
    # the cycle count, stage, stage time and next interval describe its pacing
    # alone.
    loop_iterations: int = 0
    last_iteration_at: float = 0.0
    stage: str = ""
    stage_at: float = 0.0
    next_interval: float = 0.0
    # Requested-enrichment liveness. Indexing and context recomputation commit
    # through this same state while the timer sleeps in its own stage, so they
    # are counted and staged separately: wearing the timer's stage would age a
    # stage the daemon already left, and one enriched session is not one timer
    # iteration.
    requested_cycles: int = 0
    last_requested_at: float = 0.0
    requested_stage: str = ""
    requested_stage_at: float = 0.0
    _requested_open: bool = False
    # The last cycle either path completed, and which path that was.
    last_trigger: str = ""
    last_outcome: str = ""
    # Sessions deprioritized after repeated commit rejections, mapped to their
    # monotonic cooldown deadline, and the per-session rejection counts that
    # feed them. Both are bounded, and both are read by `daemon status` on the
    # event loop while an embed cycle mutates them on the writer executor, so
    # every access holds the lock and readers receive copies.
    _cooldown_sessions: dict[str, float] = field(default_factory=dict, repr=False)
    _rejections: dict[str, int] = field(default_factory=dict, repr=False)
    _cooldown_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    @property
    def deferred_reason(self) -> str | None:
        return self._deferred_reason

    @deferred_reason.setter
    def deferred_reason(self, reason: str | None) -> None:
        self._deferred_reason = reason
        self._deferred_at = time.time() if reason else 0.0

    @property
    def deferred_at(self) -> float:
        """When the current deferral reason was evaluated; 0.0 when none is set."""
        return self._deferred_at

    def begin_timer_cycle(self) -> None:
        """Open one timer cycle. Only the phase timer paces itself this way."""
        self.loop_iterations += 1
        self.last_iteration_at = time.time()
        self.last_trigger = "timer"
        self.enter_stage("iteration")

    def begin_requested_cycle(self) -> None:
        """Open one requested enrichment cycle. `close_requested_cycle` ends it."""
        self.requested_cycles += 1
        self.last_requested_at = time.time()
        self.last_trigger = "requested"
        self._requested_open = True
        self.enter_requested_stage("iteration")

    def close_requested_cycle(self) -> None:
        """End the open cycle, recording `failed` when it reached no outcome.

        Called from the caller's `finally`, so a cycle that raises before it
        commits still closes and `daemon status` never describes requested work
        that already stopped. This pairing is not a context manager because
        `RpcError` is a frozen dataclass, and `contextlib` assigns
        `__traceback__` on an exception it re-raises.
        """
        if self._requested_open:
            self.record_requested_outcome("failed")

    def enter_stage(self, stage: str) -> None:
        self.stage = stage
        self.stage_at = time.time()

    def enter_requested_stage(self, stage: str) -> None:
        self.requested_stage = stage
        self.requested_stage_at = time.time()

    def record_outcome(self, outcome: str) -> None:
        self.last_outcome = outcome

    def record_requested_outcome(self, outcome: str) -> None:
        """Close the open requested cycle with the outcome it reached."""
        self.last_outcome = outcome
        self._requested_open = False

    def record_pending(self, pending: PendingEmbeds) -> None:
        """Adopt a pending snapshot together with the time it was observed."""
        self.last_pending = pending
        self.last_pending_at = time.time()
        gauge(Gauge.PENDING, pending.total)
        gauge(Gauge.PENDING_AT, self.last_pending_at)

    def configure(self, config: AppConfig) -> None:
        if self._config != config:
            self._idle_snapshot = None
        previous = self._config.embedding
        current = config.embedding
        self._config = config
        if (previous.backend, previous.model, previous.dimensions) != (
            current.backend,
            current.model,
            current.dimensions,
        ):
            self.backend = None
            self.last_pending = None
            self.last_pending_at = 0.0
            self.last_error = None
            self.clear_stalled()
            # A different model is a different generation of work: which
            # sessions refused to commit under the previous one says nothing
            # about this one.
            self.clear_cooldowns()
            gc.collect()

    def load_backend(self) -> EmbeddingBackend:
        if self.backend is not None:
            return self.backend
        from recall.services.embeddings import get_backend

        start = time.monotonic()
        self.backend = get_backend(self._config.embedding)
        elapsed = time.monotonic() - start
        logger.info("embed model loaded (%.1fs)", elapsed)
        self.last_used_at = time.monotonic()
        return self.backend

    def maybe_unload(self) -> None:
        if self.backend is None:
            return
        idle = time.monotonic() - self.last_used_at
        timeout = self._config.daemon.embed_model_timeout
        if idle >= timeout:
            self.backend = None
            gc.collect()
            logger.warning("embed model unloaded (idle %.0fm)", idle / 60)

    def record_batch(self, count: int, duration: float = 0.0) -> None:
        self.last_used_at = time.monotonic()
        self.last_batch_at = time.time()
        self.last_batch_size = count
        self.last_batch_duration = duration

    def stalled_for(self, pending: PendingEmbeds) -> float:
        if self.stalled_pending_signature != pending.signature:
            return 0.0
        return max(self.stalled_until - time.monotonic(), 0.0)

    def mark_stalled(self, pending: PendingEmbeds, cooldown: float) -> None:
        self.stalled_pending_signature = pending.signature
        self.stalled_until = time.monotonic() + cooldown

    def clear_stalled(self) -> None:
        """Drop the no-progress pacing. Rejection counts belong to sessions."""
        self.stalled_pending_signature = None
        self.stalled_until = 0.0

    def note_nothing_pending(self) -> None:
        """No session is eligible: nothing to pace on, no rejection to remember."""
        self.clear_stalled()
        with self._cooldown_lock:
            self._rejections.clear()

    def clear_cooldowns(self) -> None:
        """Forget every deprioritized session and every rejection count."""
        with self._cooldown_lock:
            self._cooldown_sessions.clear()
            self._rejections.clear()

    def cooled_session_ids(self) -> frozenset[str]:
        """The sessions currently deprioritized, expired entries dropped."""
        with self._cooldown_lock:
            self._drop_expired_cooldowns()
            return frozenset(self._cooldown_sessions)

    def cooldown_status(self) -> tuple[int, float | None]:
        """Active cooldown count and the wall-clock time the last one expires."""
        with self._cooldown_lock:
            self._drop_expired_cooldowns()
            if not self._cooldown_sessions:
                return (0, None)
            latest = max(self._cooldown_sessions.values())
            return (len(self._cooldown_sessions), time.time() + (latest - time.monotonic()))

    def note_commit_rejections(
        self, *, committed: Iterable[str], rejected: Iterable[str], cooldown: float
    ) -> list[str]:
        """Record one commit attempt's per-session verdicts.

        Only an attempted commit is evidence. A session whose result the
        commit-time staleness check discarded counts one rejection; a session
        that committed forgets its count; a cycle that never reached the commit
        must not call this at all. Counting per session is what makes the
        backstop escalate the session that is actually stuck rather than
        whichever one happened to head the roster when a global counter tripped.

        A session that reaches ``COMMIT_REJECTION_COOLDOWN_THRESHOLD`` rejections is
        deprioritized for ``cooldown`` seconds; its count is dropped with it and
        rebuilds from zero after the cooldown expires. Returns the newly
        deprioritized session ids, which is what the caller logs.
        """
        assert cooldown > 0, "cooldown must be positive"
        newly: list[str] = []
        with self._cooldown_lock:
            for session_id in committed:
                self._rejections.pop(session_id, None)
            self._drop_expired_cooldowns()
            deadline = time.monotonic() + cooldown
            for session_id in rejected:
                # An already-deprioritized session keeps its original deadline,
                # so the caller's WARNING stays one per cooldown.
                if session_id in self._cooldown_sessions:
                    continue
                count = self._rejections.get(session_id, 0) + 1
                if count < COMMIT_REJECTION_COOLDOWN_THRESHOLD:
                    self._remember_rejection(session_id, count)
                    continue
                self._rejections.pop(session_id, None)
                self._deprioritize(session_id, deadline)
                newly.append(session_id)
        return newly

    def selection_cursor(self, session_id: str) -> tuple[str, int, str] | None:
        with self._cooldown_lock:
            return self._selection_cursors.get(session_id)

    def advance_selection(self, session_id: str, cursor: tuple[str, int, str] | None) -> None:
        with self._cooldown_lock:
            self._selection_cursors.pop(session_id, None)
            if cursor is not None:
                if len(self._selection_cursors) >= SESSION_COOLDOWN_LIMIT:
                    del self._selection_cursors[next(iter(self._selection_cursors))]
                self._selection_cursors[session_id] = cursor

    def defer_capacity(self, session_id: str, cooldown: float) -> None:
        """Let other sessions run after an input failed capacity admission."""
        assert cooldown > 0
        with self._cooldown_lock:
            self._drop_expired_cooldowns()
            self._deprioritize(session_id, time.monotonic() + cooldown)

    def _drop_expired_cooldowns(self) -> None:
        """Caller holds `_cooldown_lock`."""
        now = time.monotonic()
        for session_id in [
            sid for sid, deadline in self._cooldown_sessions.items() if deadline <= now
        ]:
            del self._cooldown_sessions[session_id]

    def _remember_rejection(self, session_id: str, count: int) -> None:
        """Caller holds `_cooldown_lock`."""
        if session_id not in self._rejections:
            while len(self._rejections) >= SESSION_REJECTION_LIMIT:
                least = min(self._rejections, key=lambda sid: self._rejections[sid])
                del self._rejections[least]
        self._rejections[session_id] = count

    def _deprioritize(self, session_id: str, deadline: float) -> None:
        """Caller holds `_cooldown_lock`."""
        while len(self._cooldown_sessions) >= SESSION_COOLDOWN_LIMIT:
            soonest = min(self._cooldown_sessions, key=lambda sid: self._cooldown_sessions[sid])
            del self._cooldown_sessions[soonest]
        self._cooldown_sessions[session_id] = deadline


@dataclass
class PreparedEmbedCycle:
    """A bounded read snapshot and its detached model result.

    ``error`` is deliberately data, rather than an exception escaping into the
    raw-index writer: model failures must be visible but must not block raw
    content, status, or the next coordinator pass.
    """

    pending: PendingEmbeds
    sessions: list[PreparedEmbedSession]
    error: str | None = None
    selection_scanning: bool = False
    results: dict[str, GeneratedEmbedSession] = field(default_factory=dict)
    committed_session_ids: set[str] = field(default_factory=set)
    embed: bool = True
    context_messages: int = 0
    context_input_tokens: int = 0
    context_output_tokens: int = 0
    context_model: str | None = None

    @property
    def generated_session_ids(self) -> set[str]:
        return set(self.results)

    @property
    def generated(self) -> bool:
        return len(self.results) == len(self.sessions)


@measured(Stage.EMBED_SNAPSHOT)
def prepare_embed_cycle(
    config: AppConfig,
    state: EmbedPhaseState,
    *,
    conn: duckdb.DuckDBPyConnection,
    max_sessions: int = 10,
    should_stop: Callable[[], bool] | None = None,
    now: float | None = None,
    work_generation: object | None = None,
) -> PreparedEmbedCycle:
    """Read a bounded immutable-enough input snapshot without loading models."""
    if max_sessions <= 0:
        raise ValueError("max_sessions must be positive")
    require_complete(conn)
    now_epoch = time.time() if now is None else now
    # Only a caller holding the generation's writer lock may reuse an empty
    # observation. Callers without that ownership always scan.
    idle = state._idle_snapshot
    if idle is not None and idle.matches(conn, config, work_generation, now_epoch):
        pending = PendingEmbeds(0, 0, [])
        state.record_pending(pending)
        return PreparedEmbedCycle(pending=pending, sessions=[])
    state._idle_snapshot = None
    pending = find_eligible_pending_embeds(
        conn,
        config.daemon.embed_idle_session,
        max_sessions=max_sessions,
        cooled=state.cooled_session_ids(),
        now=now_epoch,
    )
    if pending.total == 0 and work_generation is not None:
        state._idle_snapshot = _IdleEmbedSnapshot(
            conn, config, work_generation, pending.eligible_after
        )
    state.record_pending(pending)
    prepared: list[PreparedEmbedSession] = []
    scanning = False
    for session_id in pending.session_ids:
        if should_stop is not None and should_stop():
            break
        try:
            page = select_pending_targets(conn, session_id, state.selection_cursor(session_id))
            state.advance_selection(session_id, page.cursor)
            if page.scanning:
                scanning = True
                continue
            selected = page.targets
            item = prepare_target_enrichment(config, session_id, selected, conn=conn)
        except EnrichmentCapacityError as err:
            state.advance_selection(session_id, None)
            state.defer_capacity(session_id, float(config.daemon.embed_backoff))
            return PreparedEmbedCycle(pending=pending, sessions=[], error=str(err))
        if selected.count:
            prepared.append(item)
    if prepared:
        gauge(Gauge.SELECTED_IDLE, max(time.time() - item.session.file_mtime for item in prepared))
    return PreparedEmbedCycle(pending=pending, sessions=prepared, selection_scanning=scanning)


def prepare_target_enrichment(
    config: AppConfig,
    session_id: str,
    selected: EnrichmentTargets,
    *,
    conn: duckdb.DuckDBPyConnection,
) -> PreparedEmbedSession:
    session = read_enrichment_session(conn, session_id, selected, full_context=False)
    targets = (
        tuple(
            message.id
            for message in session.messages
            if message.id in {target.id for target in selected.messages if target.marker}
            if message.context_mode in {"off", "template"} and (message.content or message.thinking)
        )
        if config.embedding.context.mode.startswith("llm-")
        else ()
    )
    full_context = bool(targets)
    if full_context:
        session = read_enrichment_session(conn, session_id, selected, full_context=True)
    from recall.db.enrichment_inputs import source_generation

    return snapshot_embed_session(
        session,
        context_targets=targets,
        targets=selected,
        full_context=full_context,
        generation=source_generation(conn, session.id),
    )


def prepare_context_recompute(
    session_id: str,
    target_ids: tuple[str, ...],
    *,
    conn: duckdb.DuckDBPyConnection,
    embed: bool,
    only_mode: str | None = None,
    config: AppConfig | None = None,
) -> PreparedEmbedCycle:
    """Snapshot one selected session without reparsing or loading any model."""
    config = config or AppConfig.load()
    selected = admit_targets(
        conn, read_target_masks(conn, target_ids[:TARGET_ROWS_MAX], (), replace=True)
    )
    session = read_enrichment_session(
        conn, session_id, selected, full_context=config.embedding.context.mode.startswith("llm-")
    )
    targets = tuple(
        message.id
        for message in session.messages
        if message.id in selected.message_ids
        and (only_mode is None or message.context_mode == only_mode)
    )
    if set(targets) != set(selected.message_ids):
        raise RuntimeError("context targets changed before preparation; retry the request")
    return PreparedEmbedCycle(
        pending=PendingEmbeds(len(targets), 0, [session_id]),
        sessions=[
            snapshot_embed_session(
                session,
                context_targets=targets,
                targets=selected,
                full_context=config.embedding.context.mode.startswith("llm-"),
                publication="replace",
                generation=source_generation(conn, session_id),
            )
        ],
        embed=embed,
    )


@measured(Stage.EMBED_GENERATE)
def generate_prepared_embed_cycle(
    config: AppConfig,
    state: EmbedPhaseState,
    prepared: PreparedEmbedCycle,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> PreparedEmbedCycle:
    """Run context and embedding models with no DuckDB connection held."""
    from recall.services.embeddings import embed_session
    from recall.services.indexer import _prepare_context_run, _resolve_session_message_contexts

    if prepared.error is not None:
        return prepared
    identities = "backend initialization"
    try:
        backend = state.load_backend() if prepared.embed else None
        context_run = (
            _prepare_context_run(config.embedding.context)
            if any(item.context_targets for item in prepared.sessions)
            else None
        )
        for item in prepared.sessions:
            identities = (
                f"session {item.session.id} messages={item.targets.message_ids} "
                f"tools={item.targets.tool_ids}"
            )
            if should_stop is not None and should_stop():
                break
            context_session = item.session.model_copy(
                update={"messages": [message.model_copy() for message in item.session.messages]}
            )
            if item.context_targets:
                assert context_run is not None
                targets = [m for m in context_session.messages if m.id in item.context_targets]
                stats = _resolve_session_message_contexts(
                    context_session, context_run.config, context_run.backend, targets=targets
                )
                prepared.context_messages += stats.messages
                prepared.context_input_tokens += stats.input_tokens
                prepared.context_output_tokens += stats.output_tokens
                if stats.model is not None:
                    prepared.context_model = stats.model
            if should_stop is not None and should_stop():
                break
            # Context dependencies are read-only. Only explicit target rows
            # enter inference and the generated publication unit.
            messages = [
                message.model_copy(update={"tool_calls": []})
                for message in context_session.messages
                if message.id in item.targets.message_ids
            ]
            tools = [
                tool.model_copy()
                for tool in _collect_tool_calls(item.session)
                if tool.id in item.targets.tool_ids
            ]
            selected = context_session.model_copy(
                update={
                    "messages": messages,
                    "orphan_tool_calls": tools,
                }
            )
            if backend is not None:
                embed_session(selected, backend, config.embedding.batch_size, targets=item.targets)
                _validate_generated_fields(selected, item.targets, backend.dimensions)
            prepared.results[item.session.id] = GeneratedEmbedSession(item, selected)
            count(Counter.GENERATED, len(messages) + len(tools))
    except Exception as err:
        count(Counter.GENERATION_ERRORS, 1)
        logger.error("embed generation failed: %s", err, exc_info=True)
        prepared.error = f"{identities}: {type(err).__name__}: {err}"
    finally:
        # Release transient Metal allocations between bounded batches, including failures.
        try:
            import mlx.core as mx
        except ImportError:
            pass
        else:
            mx.clear_cache()
            from recall.services.mlx_embeddings import record_memory

            record_memory()
    return prepared


def _validate_generated_fields(
    session: Session, targets: EnrichmentTargets, dimensions: int
) -> None:
    import math
    from array import array

    def validate(vector: list[float] | None, identity: str, kind: str) -> None:
        # Validate the storage representation, including overflow/underflow on FLOAT cast.
        stored = array("f", vector) if vector is not None else array("f")
        if (
            len(stored) != dimensions
            or dimensions != 384
            or not all(math.isfinite(value) for value in stored)
            or not any(stored)
        ):
            raise ValueError(
                f"invalid {kind} vector for {identity}: expected 384 finite, nonzero floats"
            )

    messages = {message.id: message for message in session.messages}
    tools = {tool.id: tool for tool in _collect_tool_calls(session)}
    for target in targets.messages:
        if target.content:
            validate(messages[target.id].content_embedding, target.id, "content")
        if target.thinking:
            validate(messages[target.id].thinking_embedding, target.id, "thinking")
    for target in targets.tools:
        if target.bash:
            validate(tools[target.id].bash_embedding, target.id, "bash")


@measured(Stage.EMBED_COMMIT)
def commit_prepared_embed_cycle(
    prepared: PreparedEmbedCycle,
    *,
    conn: duckdb.DuckDBPyConnection,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    from recall.db.semantic_work import publication

    if prepared.error is not None:
        return 0
    committed = 0
    for result in prepared.results.values():
        if should_stop is not None and should_stop():
            break
        if not publication(conn).enrich(result, embed=prepared.embed):
            count(Counter.REJECTED, 1)
            continue
        item = result.inputs
        prepared.committed_session_ids.add(item.session.id)
        published = item.targets.count if prepared.embed else len(item.context_targets)
        committed += published
        count(Counter.COMMITTED, published)
    return committed


@measured(Stage.EMBED_PUBLISH)
def publish_prepared_embed_cycle(
    prepared: PreparedEmbedCycle,
    config: AppConfig,
    *,
    conn: duckdb.DuckDBPyConnection,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    """Commit authoritative context/vectors and publish changed context to keyword search."""
    from contextlib import closing

    from recall.db import open_sidecar, sidecar_path
    from recall.db.fts_sidecar import upsert_message_fts

    committed = commit_prepared_embed_cycle(prepared, conn=conn, should_stop=should_stop)
    if not committed or config.fts.backend != "sqlite_sidecar":
        return committed
    with closing(open_sidecar(sidecar_path(config.data_dir))) as sidecar:
        for result in prepared.results.values():
            item = result.inputs
            if item.session.id not in prepared.committed_session_ids:
                continue
            original = {entry.message_id: entry for entry in item.inputs}
            for message in result.session.messages:
                before = original[message.id]
                if (before.context_text, before.context_mode) == (
                    message.context_text,
                    message.context_mode,
                ):
                    continue
                # The durable pending row already exists if the sidecar write
                # fails. The daemon writer excludes newer context commits here.
                upsert_message_fts(
                    sidecar,
                    message.id,
                    message.context_text + (message.content or ""),
                    message.context_text + (message.thinking or ""),
                    fields=config.fts.fields,
                )
                conn.execute(
                    "DELETE FROM fts_sidecar_pending WHERE kind = 'message' AND id = ?",
                    [message.id],
                )
    return committed


def _collect_tool_calls(session: Session) -> Iterable[ToolCall]:
    """Gather tool calls from messages and orphan_tool_calls."""
    tool_calls: list[ToolCall] = []
    for message in session.messages:
        tool_calls.extend(message.tool_calls)
    tool_calls.extend(session.orphan_tool_calls)
    return tool_calls
