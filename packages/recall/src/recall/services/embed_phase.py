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
from recall.core.embeddings import EmbeddingBackend
from recall.core.models import Session, ToolCall

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

    @property
    def total(self) -> int:
        return self.message_count + self.tool_call_count

    @property
    def signature(self) -> tuple[int, int, tuple[str, ...]]:
        return (self.message_count, self.tool_call_count, tuple(self.session_ids))


def find_pending_embeds(
    conn: duckdb.DuckDBPyConnection,
    idle_threshold: int,
    *,
    max_sessions: int | None = None,
    now: float | None = None,
) -> PendingEmbeds:
    """Query for un-embedded content from idle sessions.

    Uses a fast EXISTS check before the expensive LEFT JOIN + GROUP BY
    to avoid 290%+ CPU spikes on large tables when nothing is pending.
    """
    if max_sessions is not None and max_sessions <= 0:
        raise ValueError("max_sessions must be positive")
    now_epoch = time.time() if now is None else now
    cutoff = now_epoch - idle_threshold

    # Fast check: bail out immediately if nothing is pending, avoiding
    # the expensive LEFT JOIN + GROUP BY on 460K+ row tables
    has_pending = conn.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM messages m
            JOIN session_state ss ON ss.session_id = m.session_id
            LEFT JOIN message_embeddings me ON me.message_id = m.id
            WHERE me.message_id IS NULL AND ss.file_mtime < ?
            LIMIT 1
        ) OR EXISTS (
            SELECT 1 FROM tool_calls tc
            JOIN session_state ss ON ss.session_id = tc.session_id
            LEFT JOIN tool_call_embeddings tce ON tce.tool_call_id = tc.id
            WHERE tce.tool_call_id IS NULL AND ss.file_mtime < ?
            LIMIT 1
        )
        """,
        [cutoff, cutoff],
    ).fetchone()
    if not has_pending or not has_pending[0]:
        return PendingEmbeds(message_count=0, tool_call_count=0, session_ids=[])

    # Aggregate in DuckDB, then bound the roster before it crosses into Python.
    # Window totals retain complete backlog accounting even for a one-session batch.
    rows = conn.execute(
        """
        WITH pending AS (
            SELECT m.session_id, COUNT(*) AS messages, 0 AS tools
            FROM messages m
            JOIN session_state ss ON ss.session_id = m.session_id
            LEFT JOIN message_embeddings me ON me.message_id = m.id
            WHERE me.message_id IS NULL AND ss.file_mtime < ?
            GROUP BY m.session_id
            UNION ALL
            SELECT tc.session_id, 0 AS messages, COUNT(*) AS tools
            FROM tool_calls tc
            JOIN session_state ss ON ss.session_id = tc.session_id
            LEFT JOIN tool_call_embeddings tce ON tce.tool_call_id = tc.id
            WHERE tce.tool_call_id IS NULL AND ss.file_mtime < ?
            GROUP BY tc.session_id
        ), by_session AS (
            SELECT session_id, SUM(messages) AS messages, SUM(tools) AS tools
            FROM pending GROUP BY session_id
        )
        SELECT p.session_id, SUM(p.messages) OVER (), SUM(p.tools) OVER ()
        FROM by_session p JOIN session_state ss ON ss.session_id = p.session_id
        ORDER BY ss.file_mtime ASC, p.session_id
        LIMIT ?
        """,
        [cutoff, cutoff, max_sessions],
    ).fetchall()
    return PendingEmbeds(
        message_count=int(rows[0][1]) if rows else 0,
        tool_call_count=int(rows[0][2]) if rows else 0,
        session_ids=[str(row[0]) for row in rows],
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
    )


@dataclass
class EmbedPhaseState:
    """Mutable state for the embed phase timer."""

    _config: AppConfig = field(repr=False)  # required constructor argument
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

    def configure(self, config: AppConfig) -> None:
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
        logger.warning("embed model loaded (%.1fs)", elapsed)
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


@dataclass(frozen=True)
class EmbedInput:
    """The generation a model result is allowed to update."""

    message_id: str
    content: str | None
    thinking: str | None
    context_text: str
    context_mode: str
    role: str
    agent_id: str | None
    idx: int


@dataclass(frozen=True)
class EmbedToolInput:
    """A tool-call field rendered into an LLM context document."""

    tool_call_id: str
    idx: int
    tool_name: str
    tool_input: str | None
    bash_command: str | None


@dataclass
class PreparedEmbedSession:
    session: Session
    source: str
    inputs: tuple[EmbedInput, ...]
    tool_inputs: tuple[EmbedToolInput, ...]
    context_targets: tuple[str, ...]
    context_metadata: tuple[str | None, str | None, str | None]
    embedding_targets: tuple[str, ...] | None = None


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
    generated: bool = False
    generated_session_ids: set[str] = field(default_factory=set)
    committed_session_ids: set[str] = field(default_factory=set)
    embed: bool = True
    context_messages: int = 0
    context_input_tokens: int = 0
    context_output_tokens: int = 0
    context_model: str | None = None


def prepare_embed_cycle(
    config: AppConfig,
    state: EmbedPhaseState,
    *,
    conn: duckdb.DuckDBPyConnection,
    max_sessions: int = 10,
    should_stop: Callable[[], bool] | None = None,
) -> PreparedEmbedCycle:
    """Read a bounded immutable-enough input snapshot without loading models."""
    from recall.services.sessions import load_session

    if max_sessions <= 0:
        raise ValueError("max_sessions must be positive")
    pending = find_eligible_pending_embeds(
        conn,
        config.daemon.embed_idle_session,
        max_sessions=max_sessions,
        cooled=state.cooled_session_ids(),
    )
    state.record_pending(pending)
    prepared: list[PreparedEmbedSession] = []
    for session_id in pending.session_ids:
        if should_stop is not None and should_stop():
            break
        session = load_session(session_id, include_tools=True, conn=conn)
        targets = (
            tuple(
                message.id
                for message in session.messages
                if message.context_mode in {"off", "template"}
            )
            if config.embedding.context.mode.startswith("llm-")
            else ()
        )
        prepared.append(snapshot_embed_session(session, context_targets=targets))
    return PreparedEmbedCycle(pending=pending, sessions=prepared)


def snapshot_embed_session(
    session: Session,
    *,
    context_targets: tuple[str, ...],
    embedding_targets: tuple[str, ...] | None = None,
) -> PreparedEmbedSession:
    """Capture the complete document and target context inputs for stale-result rejection."""
    from recall.services.indexer import _normalize_json_value

    return PreparedEmbedSession(
        session=session,
        source=session.source.value,
        inputs=tuple(
            EmbedInput(
                message_id=message.id,
                content=message.content,
                thinking=message.thinking,
                context_text=message.context_text,
                context_mode=message.context_mode,
                role=message.role.value,
                agent_id=message.agent_id,
                idx=message.idx,
            )
            for message in session.messages
        ),
        tool_inputs=tuple(
            EmbedToolInput(
                tool_call_id=tool.id,
                idx=tool.idx,
                tool_name=tool.tool_name,
                tool_input=_normalize_json_value(tool.tool_input),
                bash_command=tool.bash_command,
            )
            for tool in _collect_tool_calls(session)
        ),
        context_targets=context_targets,
        context_metadata=(session.cwd, session.git_repo, session.git_branch),
        embedding_targets=embedding_targets,
    )


def prepare_index_enrichment(
    config: AppConfig, session_id: str, *, conn: duckdb.DuckDBPyConnection, embed: bool
) -> PreparedEmbedCycle:
    """Snapshot newly indexed inputs while retaining equivalent stored LLM context."""
    from recall.services.sessions import load_session

    session = load_session(session_id, include_tools=True, conn=conn)
    targets = tuple(
        message.id
        for message in session.messages
        if config.embedding.context.mode.startswith("llm-")
        and message.context_mode in {"off", "template"}
        and (message.content or message.thinking)
    )
    return PreparedEmbedCycle(
        pending=PendingEmbeds(
            len(session.messages), len(tuple(_collect_tool_calls(session))), [session_id]
        ),
        sessions=[snapshot_embed_session(session, context_targets=targets)],
        embed=embed,
    )


def prepare_context_recompute(
    session_id: str,
    target_ids: tuple[str, ...],
    *,
    conn: duckdb.DuckDBPyConnection,
    embed: bool,
    only_mode: str | None = None,
) -> PreparedEmbedCycle:
    """Snapshot one selected session without reparsing or loading any model."""
    from recall.services.sessions import load_session

    session = load_session(session_id, include_tools=True, conn=conn)
    targets = tuple(
        message.id
        for message in session.messages
        if message.id in target_ids and (only_mode is None or message.context_mode == only_mode)
    )
    if set(targets) != set(target_ids):
        raise RuntimeError("context targets changed before preparation; retry the request")
    return PreparedEmbedCycle(
        pending=PendingEmbeds(len(targets), 0, [session_id]),
        sessions=[
            snapshot_embed_session(session, context_targets=targets, embedding_targets=targets)
        ],
        embed=embed,
    )


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
    try:
        backend = state.load_backend() if prepared.embed else None
        context_run = (
            _prepare_context_run(config.embedding.context)
            if any(item.context_targets for item in prepared.sessions)
            else None
        )
        for item in prepared.sessions:
            if should_stop is not None and should_stop():
                break
            if item.context_targets:
                assert context_run is not None
                targets = [m for m in item.session.messages if m.id in item.context_targets]
                stats = _resolve_session_message_contexts(
                    item.session, context_run.config, context_run.backend, targets=targets
                )
                prepared.context_messages += stats.messages
                prepared.context_input_tokens += stats.input_tokens
                prepared.context_output_tokens += stats.output_tokens
                if stats.model is not None:
                    prepared.context_model = stats.model
            if should_stop is not None and should_stop():
                break
            if backend is not None:
                if item.embedding_targets is None:
                    embed_session(item.session, backend, config.embedding.batch_size)
                else:
                    # Explicit context recomputation changes selected message vectors
                    # only. Tool embeddings and non-target message vectors survive.
                    messages = [
                        message.model_copy(update={"tool_calls": []})
                        for message in item.session.messages
                        if message.id in item.embedding_targets
                    ]
                    selected = item.session.model_copy(
                        update={"messages": messages, "orphan_tool_calls": []}
                    )
                    embed_session(selected, backend, config.embedding.batch_size)
                    by_id = {message.id: message for message in messages}
                    for message in item.session.messages:
                        if generated := by_id.get(message.id):
                            message.content_embedding = generated.content_embedding
                            message.thinking_embedding = generated.thinking_embedding
            prepared.generated_session_ids.add(item.session.id)
        prepared.generated = len(prepared.generated_session_ids) == len(prepared.sessions)
    except Exception as err:
        logger.error("embed generation failed: %s", err, exc_info=True)
        prepared.error = f"{type(err).__name__}: {err}"
    finally:
        # Release transient Metal allocations between bounded batches, including failures.
        try:
            import mlx.core as mx
        except ImportError:
            pass
        else:
            mx.clear_cache()
    return prepared


def _inputs_still_match(conn: duckdb.DuckDBPyConnection, item: PreparedEmbedSession) -> bool:
    """Reject a whole session when its document/context input generation moved.

    Both sides are keyed by message/tool-call identity, never row order:
    tool_calls.idx is not unique within a session, so ORDER BY idx leaves ties
    whose physical order is plan-dependent, and identical content compared as
    ordered tuples could compare unequal.
    """
    from recall.services.indexer import _normalize_json_value

    source_row = conn.execute(
        "SELECT s.source, ss.cwd, ss.git_repo, ss.git_branch FROM sessions s "
        "JOIN session_state ss ON ss.session_id = s.id WHERE s.id = ?",
        [item.session.id],
    ).fetchone()
    if source_row is None or str(source_row[0]) != item.source:
        return False
    if item.context_targets and tuple(source_row[1:]) != item.context_metadata:
        return False
    rows = conn.execute(
        """
        SELECT m.id, ms.content, ms.thinking, COALESCE(ms.context_text, ''),
               COALESCE(ms.context_mode, 'off'), ms.role, m.agent_id, m.idx
        FROM messages m JOIN message_state ms ON ms.message_id = m.id
        WHERE m.session_id = ?
        """,
        [item.session.id],
    ).fetchall()
    current = {
        str(row[0]): EmbedInput(
            str(row[0]), row[1], row[2], str(row[3]), str(row[4]), str(row[5]), row[6], int(row[7])
        )
        for row in rows
    }
    if current != {entry.message_id: entry for entry in item.inputs}:
        return False
    tools = conn.execute(
        """
        SELECT id, idx, tool_name, tool_input, bash_command
        FROM tool_calls WHERE session_id = ?
        """,
        [item.session.id],
    ).fetchall()
    current_tools = {
        str(row[0]): EmbedToolInput(
            tool_call_id=str(row[0]),
            idx=int(row[1]),
            tool_name=str(row[2]),
            tool_input=_normalize_json_value(row[3]),
            bash_command=row[4],
        )
        for row in tools
    }
    return current_tools == {entry.tool_call_id: entry for entry in item.tool_inputs}


def commit_prepared_embed_cycle(
    prepared: PreparedEmbedCycle,
    *,
    conn: duckdb.DuckDBPyConnection,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    """Publish only model work whose complete raw/context input still matches."""
    from recall.db.queries import enqueue_sidecar_pending
    from recall.services.indexer import _insert_session_embeddings

    if prepared.error is not None:
        return 0
    committed = 0
    for item in prepared.sessions:
        if should_stop is not None and should_stop():
            break
        if item.session.id not in prepared.generated_session_ids:
            continue
        if not _inputs_still_match(conn, item):
            logger.info("discarded stale embed result for session %s", item.session.id)
            continue
        conn.execute("BEGIN")
        try:
            # Context and FTS state advance together. The FTS sidecar's normal
            # pending-repair mechanism observes these authoritative fields.
            for message in item.session.messages:
                if item.embedding_targets is not None and message.id not in item.context_targets:
                    continue
                conn.execute(
                    """
                    UPDATE message_state SET context_text = ?, context_mode = ?,
                        fts_content = ? || COALESCE(content, ''),
                        fts_thinking = ? || COALESCE(thinking, '')
                    WHERE message_id = ?
                    """,
                    [
                        message.context_text,
                        message.context_mode,
                        message.context_text,
                        message.context_text,
                        message.id,
                    ],
                )
            original_inputs = {entry.message_id: entry for entry in item.inputs}
            context_changed_ids = [
                message.id
                for message in item.session.messages
                if (original := original_inputs[message.id]).context_text != message.context_text
                or original.context_mode != message.context_mode
            ]
            if context_changed_ids:
                conn.execute(
                    "DELETE FROM message_embeddings WHERE message_id IN (SELECT UNNEST(?::TEXT[]))",
                    [context_changed_ids],
                )
            enqueue_sidecar_pending(conn, "message", context_changed_ids, "upsert")
            tools = (
                list(_collect_tool_calls(item.session)) if item.embedding_targets is None else []
            )
            messages = [
                message
                for message in item.session.messages
                if item.embedding_targets is None or message.id in item.embedding_targets
            ]
            if prepared.embed:
                _insert_session_embeddings(conn, messages, tools, mark_all=True)
            conn.execute("COMMIT")
            prepared.committed_session_ids.add(item.session.id)
            committed += len(messages) + len(tools) if prepared.embed else len(item.context_targets)
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return committed


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
        for item in prepared.sessions:
            if item.session.id not in prepared.committed_session_ids:
                continue
            original = {entry.message_id: entry for entry in item.inputs}
            for message in item.session.messages:
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
