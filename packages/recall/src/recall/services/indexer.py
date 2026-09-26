from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path
from typing import Literal, TypedDict, cast

import duckdb

from recall.core.config import AppConfig, ContextConfig, EmbeddingConfig, SourceConfig
from recall.core.embeddings import EmbeddingBackend
from recall.core.models import Message, ParseResult, Session, TailFacts, ToolCall
from recall.core.types import ABSOLUTE_TOKEN_SOURCES, Role, RunKind, Source
from recall.db import (
    advisory_lock,
    connect,
    create_fts_indexes,
    delete_session,
    insert_message_embeddings,
    insert_messages,
    insert_session,
    insert_tool_call_embeddings,
    insert_tool_calls,
)
from recall.db.fatal import is_fatal_db_invalidation
from recall.db.fts_sidecar import (
    delete_message_fts,
    delete_tool_call_fts,
    should_index_bash_fts,
    upsert_message_fts_batch,
    upsert_tool_call_fts_batch,
)
from recall.db.queries import (
    delete_tool_call_tail_facts,
    enqueue_sidecar_pending,
    fetch_tool_use_ids,
    id_set_predicate,
    insert_tool_results,
    insert_tool_use_ids,
    resolve_tool_call_ids,
    retire_stop_markers,
    upsert_stop_markers,
)
from recall.parsers import SessionParser, all_parsers, get_parser
from recall.services.context import resolve_message_contexts
from recall.services.context_backends import ContextBackend, ensure_context_backend
from recall.services.moves import find_moved_predecessors, find_vanished_rows, supersede
from recall.services.runtime_state import (
    IndexRunCounts,
    record_run_attempt,
    record_run_failure,
    record_run_success,
)

logger = logging.getLogger("recall.indexer")

# TailFacts is frozen with only immutable fields, so one shared empty value is
# a safe default for every write path that has none.
_NO_TAIL_FACTS = TailFacts()


def _bootstrap_logging_for_bare_caller(verbose: bool) -> None:
    """Give a bare in-process caller a log handler, and only a bare one.

    A script or test that calls this module directly has no logging set up, so
    without this its output goes through Python's lastResort handler (bare
    WARNING+ text, no timestamp) or nowhere at all. An entry point that owns its
    own logging — the daemon (`recall.cli.daemon._configure_daemon_logging`) —
    has already put a handler on the root logger, and a library call must not
    re-level the process it happens to be running inside.
    """
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)


@contextmanager
def _noop_ctx():
    yield


def _prepare_context_run(context_config: ContextConfig) -> ContextRun:
    # ensure_context_backend hard-fails (ContextBackendUnavailableError) when an
    # llm-* mode is configured but its backend is unsupported on this host —
    # always, independent of `fallback`. `fallback` now governs only transient
    # per-message generation failures (resolve_message_context), not a missing
    # backend: silently degrading to template hid misconfiguration (REQ-CTX-020).
    backend = ensure_context_backend(context_config)
    return ContextRun(config=context_config, backend=backend, mode=context_config.mode)


def _resolve_session_message_contexts(
    session: Session,
    context_config: ContextConfig,
    backend: ContextBackend | None,
    *,
    targets: list[Message] | None = None,
) -> SessionContextStats:
    target_messages = session.messages if targets is None else targets
    input_tokens = 0
    output_tokens = 0
    model: str | None = None
    contextualized_messages = 0
    results = resolve_message_contexts(session, target_messages, context_config, backend)
    for message, result in zip(target_messages, results, strict=True):
        message.context_text = result.prefix
        message.context_mode = result.mode
        if result.mode != "off" and result.prefix and (message.content or message.thinking):
            contextualized_messages += 1
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        if result.model is not None:
            model = result.model
    return SessionContextStats(
        messages=contextualized_messages,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )


def _resolve_session_write_contexts(
    conn: duckdb.DuckDBPyConnection,
    session: Session,
    context_run: ContextRun,
    *,
    is_full_parse: bool,
) -> SessionContextStats:
    if is_full_parse:
        # Re-parsing is not re-summarizing.  Message ids are deterministic and
        # transcripts are append-only, so a stored LLM summary is still valid;
        # re-earning it costs a model call per message, and resolving it under
        # a cheaper mode would overwrite it with nothing (REQ-INDEX-021).
        reused = _apply_stored_contexts(conn, session)
        pending = [message for message in session.messages if not message.context_text]
        stats = _resolve_session_message_contexts(
            session,
            context_run.config,
            context_run.backend,
            targets=pending,
        )
        return replace(stats, messages=stats.messages + reused, reused=reused)
    # A suffix knows only what its own records said, and the first-wins fields
    # were elected by the prefix (REQ-INDEX-026). Context describes the
    # session, not the suffix, so resolve it against the committed metadata.
    session = _with_committed_session_metadata(conn, session)
    if context_run.config.mode not in {"llm-local", "llm-remote"}:
        return _resolve_session_message_contexts(
            session,
            context_run.config,
            context_run.backend,
        )

    context_session = session.model_copy(
        update={"messages": [*_load_session_messages(conn, session.id), *session.messages]}
    )
    return _resolve_session_message_contexts(
        context_session,
        context_run.config,
        context_run.backend,
        targets=session.messages,
    )


def _with_committed_session_metadata(conn: duckdb.DuckDBPyConnection, session: Session) -> Session:
    """Fill a suffix's unset session metadata from the row it merges into.

    Only fields the suffix left unset are filled, so an adapter that re-derives
    a value on every parse -- a sidecar working directory, a last-wins model --
    keeps the value it just read. The copy is shallow: the caller still writes
    the original parse result, whose NULLs are what preserve the committed
    first-wins values through the incremental merge.
    """
    row = conn.execute(
        "SELECT model, cwd, git_repo, git_branch FROM session_state WHERE session_id = ?",
        [session.id],
    ).fetchone()
    if row is None:
        return session
    model, cwd, git_repo, git_branch = row
    return session.model_copy(
        update={
            "model": session.model or model,
            "cwd": session.cwd or cwd,
            "git_repo": session.git_repo or git_repo,
            "git_branch": session.git_branch or git_branch,
        }
    )


def _apply_stored_contexts(conn: duckdb.DuckDBPyConnection, session: Session) -> int:
    """Stamp previously stored LLM context onto freshly parsed messages.

    Only `llm-*` context is carried forward: it costs a model call to produce
    and is the value a re-parse would otherwise destroy.  Template prefixes
    are derived and free to rebuild, so they stay out of this and refresh
    with the session metadata they are built from.

    Returns how many messages kept a stored context.
    """
    if not session.messages:
        return 0
    from recall.services.sessions import load_session

    if conn.execute("SELECT 1 FROM sessions WHERE id = ?", [session.id]).fetchone() is None:
        return _apply_moved_contexts(conn, session)
    previous = load_session(session.id, include_tools=True, conn=conn)
    # Legacy contexts have no saved dependency fingerprint. Require the whole
    # normalized conversation, including tools and their association, to match.
    if _context_inputs(previous) != _context_inputs(session):
        return 0
    return _stamp_contexts(previous, session)


def _apply_moved_contexts(conn: duckdb.DuckDBPyConnection, session: Session) -> int:
    """Carry stored LLM context from the row a moved transcript left behind.

    The moved file is a new session with new message ids, so contexts match by
    position instead.  The old conversation must be exactly the new one's
    prefix: the same messages an in-place append would have kept its context
    for (REQ-INDEX-027).
    """
    from recall.services.sessions import load_session

    current = _context_inputs(session)
    for predecessor_id in find_vanished_rows(
        conn, source=session.source.value, source_path=session.source_path
    ):
        previous = load_session(predecessor_id, include_tools=True, conn=conn)
        if not _is_context_prefix(_context_inputs(previous), current):
            continue
        reused = _stamp_contexts(previous, session, by_position=True)
        if reused:
            return reused
    return 0


def _is_context_prefix(previous: tuple[object, ...], current: tuple[object, ...]) -> bool:
    """Whether `current` is `previous` with messages and orphan calls appended."""
    source, messages, orphans = previous
    current_source, current_messages, current_orphans = current
    assert isinstance(messages, tuple) and isinstance(orphans, tuple)
    assert isinstance(current_messages, tuple) and isinstance(current_orphans, tuple)
    return (
        source == current_source
        and current_messages[: len(messages)] == messages
        and current_orphans[: len(orphans)] == orphans
    )


def _stamp_contexts(previous: Session, session: Session, *, by_position: bool = False) -> int:
    """Stamp `previous`'s stored `llm-*` contexts onto matching messages of `session`."""
    stored = {
        (message.idx if by_position else message.id): (message.context_text, message.context_mode)
        for message in previous.messages
        if message.context_mode in ("llm-local", "llm-remote", "llm-codex")
    }
    if not stored:
        return 0
    reused = 0
    for message in session.messages:
        entry = stored.get(message.idx if by_position else message.id)
        if entry is None or not entry[0]:
            continue
        message.context_text, message.context_mode = entry
        reused += 1
    return reused


def _context_inputs(session: Session) -> tuple[object, ...]:
    def tools(calls: list[ToolCall]) -> tuple[object, ...]:
        return tuple(
            (call.idx, call.tool_name, _normalize_json_value(call.tool_input), call.bash_command)
            for call in calls
        )

    return (
        session.source,
        tuple(
            (
                message.idx,
                message.role,
                message.content,
                message.thinking,
                message.agent_id,
                tools(message.tool_calls),
            )
            for message in session.messages
        ),
        tools(session.orphan_tool_calls),
    )


def _message_context(
    message: Message,
    *,
    fallback_context_text: str = "",
    fallback_context_mode: str = "off",
) -> tuple[str, str]:
    if message.context_text or message.context_mode != "off":
        return message.context_text, message.context_mode
    return fallback_context_text, fallback_context_mode


def _log_failed_transaction(exc: BaseException, *, session_id: str, site: str) -> None:
    """Record what failed a write transaction *before* the rollback runs.

    A rollback is not a safe place to learn the cause.  DuckDB reverts a failed
    commit internally, and when that revert throws it raises ``FatalException``
    -- an uncaught C++ exception that ``abort()``s the process.  The original
    error dies with it, the client sees only ``connection closed by daemon``,
    and ``is_fatal_db_invalidation`` never runs because there is no process
    left to run it.  Diagnosing one such abort took seven discarded hypotheses
    precisely because nothing had written the cause down.

    This is the last point that is guaranteed to execute while the cause is
    still in hand, so it logs unconditionally rather than depending on the
    exception propagating.
    """
    detail = ""
    if isinstance(exc, duckdb.FatalException):
        detail = (
            " (fatal: DuckDB has invalidated this database instance for the"
            " life of the process; the daemon must restart to recover)"
        )
    logger.error(
        "write transaction failed site=%s session=%s %s: %s%s",
        site,
        session_id,
        type(exc).__name__,
        exc,
        detail,
    )


def _insert_messages(
    conn: duckdb.DuckDBPyConnection,
    messages: Iterable[Message],
    *,
    context_text: str = "",
    context_mode: str = "off",
    sidecar_conn: sqlite3.Connection | None = None,
    fts_fields: tuple[str, ...] | None = None,
    sidecar_touched: FtsSidecarTouchedIds | None = None,
) -> None:
    materialized = list(messages)
    contexts = {
        message.id: _message_context(
            message,
            fallback_context_text=context_text,
            fallback_context_mode=context_mode,
        )
        for message in materialized
    }
    insert_messages(conn, materialized, context_by_message=contexts)
    sidecar_rows: list[tuple[str, str, str]] = []
    failed_ids: list[str] = []
    if sidecar_conn is not None:
        sidecar_rows = [
            (
                message.id,
                f"{contexts[message.id][0]}{message.content or ''}",
                f"{contexts[message.id][0]}{message.thinking or ''}",
            )
            for message in materialized
        ]
    if sidecar_conn is not None and sidecar_rows:
        if sidecar_touched is not None:
            sidecar_touched.add_messages(
                message_id for message_id, _content, _thinking in sidecar_rows
            )
        try:
            upsert_message_fts_batch(sidecar_conn, sidecar_rows, fields=fts_fields)
        except sqlite3.Error as err:
            failed_ids = [message_id for message_id, _content, _thinking in sidecar_rows]
            logger.warning(
                "sidecar dual-write failed for %d messages: %s; queueing for reconciliation",
                len(failed_ids),
                err,
            )
    if failed_ids:
        _enqueue_sidecar_pending_best_effort(conn, "message", failed_ids, "upsert")


def _enqueue_sidecar_pending_best_effort(
    conn: duckdb.DuckDBPyConnection,
    kind: Literal["message", "tool_call"],
    ids: Iterable[str],
    op: Literal["upsert", "delete"],
) -> None:
    try:
        enqueue_sidecar_pending(conn, kind, ids, op)
    except Exception:
        logger.exception(
            "failed to enqueue SQLite FTS sidecar reconciliation work "
            "(kind=%s, op=%s); continuing because DuckDB is authoritative",
            kind,
            op,
        )


def _enqueue_touched_sidecar_ids_for_reconcile(
    conn: duckdb.DuckDBPyConnection,
    touched: FtsSidecarTouchedIds,
) -> None:
    """Queue rollback-touched sidecar ids to be re-derived from DuckDB."""
    if not touched.has_touched_ids():
        return
    _enqueue_sidecar_pending_best_effort(conn, "message", touched.message_ids, "upsert")
    _enqueue_sidecar_pending_best_effort(conn, "tool_call", touched.tool_call_ids, "upsert")


def _sidecar_touched_accumulator(
    sidecar_conn: sqlite3.Connection | None,
    sidecar_touched: FtsSidecarTouchedIds | None,
) -> tuple[FtsSidecarTouchedIds | None, bool]:
    if sidecar_touched is not None:
        return sidecar_touched, False
    if sidecar_conn is None:
        return None, False
    # Callers that do not need cycle-level accounting still need rollback repair.
    return FtsSidecarTouchedIds(), True


def _enqueue_owned_sidecar_touched_after_rollback(
    conn: duckdb.DuckDBPyConnection,
    sidecar_touched: FtsSidecarTouchedIds | None,
    owns_sidecar_touched: bool,
) -> None:
    if owns_sidecar_touched and sidecar_touched is not None:
        _enqueue_touched_sidecar_ids_for_reconcile(conn, sidecar_touched)


def _context_version_for_mode(mode: str) -> int:
    from recall.services.embeddings import context_version_for_mode

    return context_version_for_mode(mode)


@dataclass(frozen=True)
class IndexSummary:
    total: int
    indexed: int
    skipped: int
    failed: int
    changed: int = 0
    fts_rebuilt: bool = False
    total_seconds: float = 0.0
    context_messages: int = 0
    context_reused: int = 0
    context_mode: str = "off"
    context_input_tokens: int = 0
    context_output_tokens: int = 0
    context_model: str | None = None
    # Sources reconciliation still owed when the request returned, and how many
    # it served in the last minute (None before it has served any). A plain
    # incremental request reports the backlog it handed to the shared drain
    # rather than waiting it out (`REQ-RECON-025`).
    backlog_pending: int = 0
    backlog_drain_per_minute: float | None = None
    # source -> count of sidecar-bearing sessions that parsed with no start
    # timestamp this run. Empty is the healthy case (REQ-INDEX-019).
    metadata_drift: dict[str, int] = field(default_factory=dict)
    backup_path: str | None = None


@dataclass
class FtsSidecarTouchedIds:
    """Entity ids whose live sidecar rows were touched by one index attempt."""

    message_ids: set[str] = field(default_factory=set)
    tool_call_ids: set[str] = field(default_factory=set)

    def add_messages(self, ids: Iterable[str]) -> None:
        self.message_ids.update(str(entity_id) for entity_id in ids)

    def add_tool_calls(self, ids: Iterable[str]) -> None:
        self.tool_call_ids.update(str(entity_id) for entity_id in ids)

    def merge(self, other: FtsSidecarTouchedIds) -> None:
        self.message_ids.update(other.message_ids)
        self.tool_call_ids.update(other.tool_call_ids)

    def has_touched_ids(self) -> bool:
        return bool(self.message_ids or self.tool_call_ids)


@dataclass(frozen=True)
class PersistedSessionRows:
    identity_row: tuple[object, ...]
    session_row: tuple[object, ...]
    message_rows: list[tuple[object, ...]]
    tool_call_rows: list[tuple[object, ...]]


@dataclass(frozen=True)
class ContextRun:
    config: ContextConfig
    backend: ContextBackend | None
    mode: str


@dataclass(frozen=True)
class SessionContextStats:
    messages: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None
    reused: int = 0


@dataclass(frozen=True)
class ContextRecomputeGroup:
    session: Session
    targets: list[Message]


@dataclass(frozen=True)
class DiscoveredSessionPath:
    parser: SessionParser
    path: Path
    resolved_path: str
    file_mtime: float
    file_size: int
    sidecar_mtime: float = 0.0


@dataclass(frozen=True)
class IndexProgress:
    processed: int
    total: int
    indexed: int
    skipped: int
    failed: int
    status: Literal["start", "indexed", "skipped", "failed", "done"]
    path: str | None = None


class _BoundedEmbeddingCache(OrderedDict[str, list[float]]):
    """OrderedDict with LRU eviction to cap in-memory embedding vector storage."""

    def __init__(self, max_size: int = 10_000) -> None:
        super().__init__()
        self._max_size = max_size

    def __getitem__(self, key: str) -> list[float]:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key: str, default: list[float] | None = None) -> list[float] | None:  # type: ignore
        # Intentional: narrows dict.get() to cache's concrete key/value types
        if key in self:
            self.move_to_end(key)
            return super().__getitem__(key)
        return default

    def __setitem__(self, key: str, value: list[float]) -> None:
        if key in self:
            super().__setitem__(key, value)
            self.move_to_end(key)
            return
        if len(self) >= self._max_size:
            self.popitem(last=False)
        super().__setitem__(key, value)


@dataclass
class PreparedSessionQueue:
    items: dict[str, ParseResult | Future[ParseResult]]
    executor: ThreadPoolExecutor | None = None
    _pending: deque[DiscoveredSessionPath] = field(default_factory=deque)
    _session_states: dict[str, SessionState] = field(default_factory=dict)


def index_sessions(
    *,
    source: Source | None,
    full: bool,
    recreate: bool,
    verbose: bool,
    embed: bool = False,
    workers: int | str = "auto",
    persist_runtime_state: bool = True,
    run_kind: RunKind = RunKind.INDEX,
    config: AppConfig | None = None,
    progress_callback: Callable[[IndexProgress], None] | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
    sidecar_touched: FtsSidecarTouchedIds | None = None,
    home_root: Path | None = None,
    host: str | None = None,
    since: datetime | None = None,
    project: str | None = None,
) -> IndexSummary:
    config = config or AppConfig.load()
    from recall.core.types import default_session_host
    from recall.parsers.common import use_home_root

    resolved_host = host if host is not None else default_session_host()
    _bootstrap_logging_for_bare_caller(verbose)
    context_run = _prepare_context_run(config.embedding.context)

    if recreate:
        full = True

    backend: EmbeddingBackend | None = None
    embedding_cache: _BoundedEmbeddingCache | None = None
    if embed:
        from recall.services.embeddings import get_backend

        backend = get_backend(config.embedding)
        embedding_cache = _BoundedEmbeddingCache(max_size=10_000)

    # When a connection is provided by the caller (e.g. daemon RPC server),
    # the caller owns the connection and lock lifecycle.
    owned_conn = conn is None
    start_time = time.perf_counter()
    lock_ctx = advisory_lock(config.lock_path) if owned_conn else _noop_ctx()
    with lock_ctx:
        if owned_conn:
            conn = connect(config, recreate=recreate)
        assert conn is not None
        sidecar_conn: sqlite3.Connection | None = None
        try:
            if config.fts.backend == "sqlite_sidecar":
                from recall.db import open_sidecar, sidecar_path

                sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
            attempted_at = (
                record_run_attempt(conn, run_kind=run_kind) if persist_runtime_state else None
            )
            with use_home_root(home_root):
                paths = _discover_paths(source, sources=config.sources)
            paths = _scope_paths(conn, paths, since=since, project=project)
            session_states = _load_session_states(conn) if not full else {}
            changed_paths = [
                discovered
                for discovered in paths
                if full or not _is_unchanged(discovered, session_states)
            ]
            resolved_workers = _resolve_worker_count(workers, len(changed_paths))
            total = len(paths)
            skipped_count = total - len(changed_paths)
            indexed = 0
            failed = 0
            context_messages = 0
            context_reused = 0
            context_input_tokens = 0
            context_output_tokens = 0
            context_model: str | None = None
            # Sidecar-bearing sessions that still parsed without a start time.
            sidecar_sessions_seen: dict[str, int] = {}
            sidecar_sessions_missing_start: dict[str, int] = {}
            fts_rebuild_needed = False
            _emit_progress(progress_callback, 0, total, indexed, skipped_count, failed, "start")
            prepared_sessions = _submit_parse_futures(
                changed_paths, resolved_workers, session_states
            )
            try:
                for idx, discovered in enumerate(changed_paths, start=1):
                    processed = skipped_count + idx
                    session_sidecar_touched = (
                        FtsSidecarTouchedIds() if sidecar_conn is not None else None
                    )
                    try:
                        state = session_states.get(discovered.resolved_path)
                        if resolved_workers == 1:
                            parse_offset = _resolve_parse_offset(state, discovered)
                            msg_base = state.message_count if state and parse_offset > 0 else 0
                            orphan_base = (
                                state.orphan_tool_count if state and parse_offset > 0 else 0
                            )
                            result = discovered.parser.parse(
                                discovered.path,
                                offset=parse_offset,
                                message_idx_base=msg_base,
                                orphan_tool_call_idx_base=orphan_base,
                            )
                        else:
                            result = _load_prepared_session(
                                prepared_sessions,
                                discovered,
                            )
                        # Discovery already stat'd the sidecars; carry that
                        # fingerprint onto the row so the next run can compare
                        # it (REQ-INDEX-018). Parsers stamp file_mtime from
                        # their own stat, but the sidecar set is the indexer's
                        # question, not theirs.
                        _require_committable_capture(
                            discovered.path,
                            result,
                            has_indexed_history=_load_session_state(conn, discovered.resolved_path)
                            is not None,
                            conn=conn,
                        )
                        session = result.session.model_copy(
                            update={"sidecar_mtime": discovered.sidecar_mtime}
                        )
                        if discovered.sidecar_mtime > 0.0:
                            source_name = str(discovered.parser.source)
                            sidecar_sessions_seen[source_name] = (
                                sidecar_sessions_seen.get(source_name, 0) + 1
                            )
                            if session.started_at is None:
                                sidecar_sessions_missing_start[source_name] = (
                                    sidecar_sessions_missing_start.get(source_name, 0) + 1
                                )
                        session_context = _resolve_session_write_contexts(
                            conn,
                            session,
                            context_run,
                            is_full_parse=result.is_full_parse,
                        )
                        context_messages += session_context.messages
                        context_reused += session_context.reused
                        context_input_tokens += session_context.input_tokens
                        context_output_tokens += session_context.output_tokens
                        if session_context.model is not None:
                            context_model = session_context.model
                        if backend is not None:
                            from recall.services.embeddings import embed_session

                            embed_session(
                                session,
                                backend,
                                config.embedding.batch_size,
                                conn=conn,
                                resolved_cache=embedding_cache,
                                context_version=_context_version_for_mode(context_run.mode),
                            )
                        if result.is_full_parse:
                            if _write_session(
                                conn,
                                session,
                                last_byte_offset=result.next_byte_offset,
                                sidecar_conn=sidecar_conn,
                                fts_fields=config.fts.fields,
                                sidecar_touched=session_sidecar_touched,
                                host=resolved_host,
                                tail_facts=result.tail_facts,
                            ):
                                fts_rebuild_needed = True
                        else:
                            if _incremental_write_session(
                                conn,
                                session,
                                last_byte_offset=result.next_byte_offset,
                                sidecar_conn=sidecar_conn,
                                fts_fields=config.fts.fields,
                                sidecar_touched=session_sidecar_touched,
                                host=resolved_host,
                                tail_facts=result.tail_facts,
                            ):
                                fts_rebuild_needed = True
                        if sidecar_touched is not None and session_sidecar_touched is not None:
                            sidecar_touched.merge(session_sidecar_touched)
                        indexed += 1
                        logger.info("indexed %s", discovered.path)
                        _emit_progress(
                            progress_callback,
                            processed,
                            total,
                            indexed,
                            skipped_count,
                            failed,
                            "indexed",
                            str(discovered.path),
                        )
                    except Exception as err:
                        if session_sidecar_touched is not None:
                            if sidecar_touched is not None:
                                sidecar_touched.merge(session_sidecar_touched)
                            _enqueue_touched_sidecar_ids_for_reconcile(
                                conn,
                                session_sidecar_touched,
                            )
                        failed += 1
                        logger.error("failed to index %s: %s", discovered.path, err)
                        _emit_progress(
                            progress_callback,
                            processed,
                            total,
                            indexed,
                            skipped_count,
                            failed,
                            "failed",
                            str(discovered.path),
                        )
            finally:
                if prepared_sessions.executor is not None:
                    prepared_sessions.executor.shutdown(wait=True, cancel_futures=True)
            # REQ-USAGE-010: harvest Grok usage log during index (fail soft).
            try:
                from recall.services.usage_harvest import (
                    default_grok_unified_log,
                    harvest_grok_unified_log,
                )

                log_path = default_grok_unified_log(home=home_root)
                harvest = harvest_grok_unified_log(conn, log_path, host=resolved_host)
                if harvest.events_upserted or harvest.rotated:
                    logger.info(
                        "usage harvest path=%s upserted=%s rolled_up=%s rotated=%s",
                        harvest.path,
                        harvest.events_upserted,
                        harvest.sessions_rolled_up,
                        harvest.rotated,
                    )
            except Exception as err:
                if is_fatal_db_invalidation(err):
                    raise
                # Names its site and carries the traceback: three loops swallow
                # this failure and `str(err)` alone hid which statement it was
                # (REQ-RESIL-023).
                logger.warning("usage harvest failed after index run: %s", err, exc_info=True)
            if fts_rebuild_needed or (config.fts.fields and not _fts_indexes_exist(conn)):
                create_fts_indexes(conn, config.fts)
                fts_rebuild_needed = True
            summary = IndexSummary(
                total=total,
                indexed=indexed,
                skipped=skipped_count,
                failed=failed,
                changed=len(changed_paths),
                fts_rebuilt=fts_rebuild_needed,
                total_seconds=time.perf_counter() - start_time,
                metadata_drift=dict(sidecar_sessions_missing_start),
                context_messages=context_messages,
                context_reused=context_reused,
                context_mode=context_run.mode,
                context_input_tokens=context_input_tokens,
                context_output_tokens=context_output_tokens,
                context_model=context_model,
            )
            _warn_on_metadata_drift(sidecar_sessions_missing_start, sidecar_sessions_seen)
            if persist_runtime_state:
                record_run_success(
                    conn,
                    run_kind=run_kind,
                    index_summary=IndexRunCounts(
                        total=summary.total,
                        indexed=summary.indexed,
                        skipped=summary.skipped,
                        failed=summary.failed,
                        changed=summary.changed,
                        total_seconds=summary.total_seconds,
                    ),
                    last_context_messages=summary.context_messages,
                    last_context_mode=summary.context_mode,
                    last_context_input_tokens=summary.context_input_tokens,
                    last_context_output_tokens=summary.context_output_tokens,
                    last_context_model=summary.context_model,
                    attempted_at=attempted_at,
                )
            _emit_progress(
                progress_callback,
                total,
                total,
                summary.indexed,
                summary.skipped,
                summary.failed,
                "done",
            )
            return summary
        except Exception as err:
            if persist_runtime_state:
                record_run_failure(conn, run_kind=run_kind, message=str(err))
            raise
        finally:
            if sidecar_conn is not None:
                sidecar_conn.close()
            if owned_conn:
                conn.close()


def recompute_context_for_rows(
    *,
    config: AppConfig | None = None,
    since: datetime | None = None,
    only_mode: str | None = None,
    embed: bool = False,
    get_backend: Callable[[EmbeddingConfig], EmbeddingBackend] | None = None,
    verbose: bool = False,
    persist_runtime_state: bool = True,
    run_kind: RunKind = RunKind.INDEX,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> IndexSummary:
    """Rewrite persisted CONTENT/THINKING context without reparsing JSONL."""
    config = config or AppConfig.load()
    context_run = _prepare_context_run(config.embedding.context)
    if only_mode is not None and only_mode not in {
        "off",
        "template",
        "llm-local",
        "llm-remote",
        "llm-codex",
    }:
        raise ValueError(f"unsupported context mode: {only_mode}")
    _bootstrap_logging_for_bare_caller(verbose)

    backend: EmbeddingBackend | None = None
    embedding_cache: _BoundedEmbeddingCache | None = None
    if embed:
        if get_backend is None:
            from recall.services.embeddings import get_backend as default_get_backend

            get_backend = default_get_backend

        backend = get_backend(config.embedding)
        embedding_cache = _BoundedEmbeddingCache(max_size=10_000)

    owned_conn = conn is None
    start_time = time.perf_counter()
    lock_ctx = advisory_lock(config.lock_path) if owned_conn else _noop_ctx()
    with lock_ctx:
        if owned_conn:
            conn = connect(config, recreate=False)
        assert conn is not None
        sidecar_conn: sqlite3.Connection | None = None
        try:
            if config.fts.backend == "sqlite_sidecar":
                from recall.db import open_sidecar, sidecar_path

                sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
            attempted_at = (
                record_run_attempt(conn, run_kind=run_kind) if persist_runtime_state else None
            )
            groups = _load_context_recompute_groups(conn, since=since, only_mode=only_mode)
            selected_count = sum(len(group.targets) for group in groups)
            context_messages = 0
            context_reused = 0
            context_input_tokens = 0
            context_output_tokens = 0
            context_model: str | None = None
            embedding_rows: list[tuple[str, list[float] | None, list[float] | None]] = []
            updates: list[tuple[str, str, str, str, str]] = []
            for group in groups:
                if not group.targets:
                    continue
                session_context = _resolve_session_message_contexts(
                    group.session,
                    context_run.config,
                    context_run.backend,
                    targets=group.targets,
                )
                context_messages += session_context.messages
                context_reused += session_context.reused
                context_input_tokens += session_context.input_tokens
                context_output_tokens += session_context.output_tokens
                if session_context.model is not None:
                    context_model = session_context.model
                if backend is not None:
                    from recall.services.embeddings import embed_session

                    embed_session(
                        group.session.model_copy(update={"messages": group.targets}),
                        backend,
                        config.embedding.batch_size,
                        conn=conn,
                        resolved_cache=embedding_cache,
                        context_version=_context_version_for_mode(context_run.mode),
                    )
                for message in group.targets:
                    context_text, context_mode = _message_context(message)
                    updates.append(
                        (
                            context_text,
                            context_mode,
                            f"{context_text}{message.content or ''}",
                            f"{context_text}{message.thinking or ''}",
                            message.id,
                        )
                    )
                    if backend is not None:
                        embedding_rows.append(
                            (
                                message.id,
                                message.content_embedding,
                                message.thinking_embedding,
                            )
                        )

            if selected_count:
                conn.execute("BEGIN")
                try:
                    conn.executemany(
                        """
                        UPDATE message_state
                        SET
                            context_text = ?,
                            context_mode = ?,
                            fts_content = ?,
                            fts_thinking = ?
                        WHERE message_id = ?
                        """,
                        updates,
                    )
                    message_ids = [row[4] for row in updates]
                    placeholders = ", ".join("?" for _ in message_ids)
                    conn.execute(
                        f"DELETE FROM message_embeddings WHERE message_id IN ({placeholders})",
                        message_ids,
                    )
                    if embedding_rows:
                        insert_message_embeddings(conn, embedding_rows)
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                if sidecar_conn is not None:
                    sidecar_rows = [
                        (message_id, fts_content, fts_thinking)
                        for (
                            _context_text,
                            _context_mode,
                            fts_content,
                            fts_thinking,
                            message_id,
                        ) in updates
                    ]
                    failed_ids: list[str] = []
                    try:
                        upsert_message_fts_batch(
                            sidecar_conn,
                            sidecar_rows,
                            fields=config.fts.fields,
                        )
                    except sqlite3.Error as err:
                        failed_ids = [
                            message_id for message_id, _content, _thinking in sidecar_rows
                        ]
                        logger.warning(
                            "sidecar dual-write failed for %d messages: %s; "
                            "queueing for reconciliation",
                            len(failed_ids),
                            err,
                        )
                    if failed_ids:
                        _enqueue_sidecar_pending_best_effort(
                            conn,
                            "message",
                            failed_ids,
                            "upsert",
                        )
                create_fts_indexes(conn, config.fts)

            summary = IndexSummary(
                total=selected_count,
                indexed=0,
                skipped=0,
                failed=0,
                changed=selected_count,
                fts_rebuilt=selected_count > 0 and bool(config.fts.fields),
                total_seconds=time.perf_counter() - start_time,
                context_messages=context_messages,
                context_reused=context_reused,
                context_mode=context_run.mode,
                context_input_tokens=context_input_tokens,
                context_output_tokens=context_output_tokens,
                context_model=context_model,
            )
            if persist_runtime_state:
                record_run_success(
                    conn,
                    run_kind=run_kind,
                    index_summary=IndexRunCounts(
                        total=summary.total,
                        indexed=summary.indexed,
                        skipped=summary.skipped,
                        failed=summary.failed,
                        changed=summary.changed,
                        total_seconds=summary.total_seconds,
                    ),
                    last_context_messages=summary.context_messages,
                    last_context_mode=summary.context_mode,
                    last_context_input_tokens=summary.context_input_tokens,
                    last_context_output_tokens=summary.context_output_tokens,
                    last_context_model=summary.context_model,
                    attempted_at=attempted_at,
                )
            return summary
        except Exception as err:
            if persist_runtime_state:
                record_run_failure(conn, run_kind=run_kind, message=str(err))
            raise
        finally:
            if sidecar_conn is not None:
                sidecar_conn.close()
            if owned_conn:
                conn.close()


def _submit_parse_futures(
    changed_paths: list[DiscoveredSessionPath],
    workers: int,
    session_states: dict[str, SessionState],
) -> PreparedSessionQueue:
    if workers == 1:
        return PreparedSessionQueue(items={}, _session_states=session_states)

    max_buffered = min(max(workers * 2, 4), 16)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="recall-index")
    initial = changed_paths[:max_buffered]
    pending = deque(changed_paths[max_buffered:])
    items: dict[str, ParseResult | Future[ParseResult]] = {}
    for d in initial:
        state = session_states.get(d.resolved_path)
        parse_offset = _resolve_parse_offset(state, d)
        msg_base = state.message_count if state and parse_offset > 0 else 0
        orphan_base = state.orphan_tool_count if state and parse_offset > 0 else 0
        items[d.resolved_path] = executor.submit(
            d.parser.parse,
            d.path,
            offset=parse_offset,
            message_idx_base=msg_base,
            orphan_tool_call_idx_base=orphan_base,
        )
    return PreparedSessionQueue(
        items=items, executor=executor, _pending=pending, _session_states=session_states
    )


def _replenish_parse_queue(prepared: PreparedSessionQueue) -> None:
    """Submit the next pending parse job to keep the sliding window full."""
    if prepared._pending and prepared.executor is not None:
        next_d = prepared._pending.popleft()
        state = prepared._session_states.get(next_d.resolved_path)
        offset = _resolve_parse_offset(state, next_d)
        msg_base = state.message_count if state and offset > 0 else 0
        orphan_base = state.orphan_tool_count if state and offset > 0 else 0
        prepared.items[next_d.resolved_path] = prepared.executor.submit(
            next_d.parser.parse,
            next_d.path,
            offset=offset,
            message_idx_base=msg_base,
            orphan_tool_call_idx_base=orphan_base,
        )


def _load_prepared_session(
    prepared: PreparedSessionQueue,
    discovered: DiscoveredSessionPath,
) -> ParseResult:
    future = prepared.items.pop(discovered.resolved_path)
    assert isinstance(future, Future)
    try:
        return cast(ParseResult, future.result())
    finally:
        _replenish_parse_queue(prepared)


def _set_session_host(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    host: str | None = None,
) -> None:
    """Stamp session_state.host for multi-host analytics (REQ-MULTIHOST-002/004).

    Defaults to the short hostname so watcher and index paths share one label
    (not the DDL default ``local`` mixed with hostname from CLI index runs).
    """
    from recall.core.types import default_session_host

    label = host if host is not None else default_session_host()
    try:
        current = conn.execute(
            "SELECT host FROM session_state WHERE session_id = ?",
            [session_id],
        ).fetchone()
        # Skip the write when already stamped: DuckDB UPDATE is delete+insert, so
        # an unconditional stamp rewrites the row on every re-index.
        if current is not None and current[0] == label:
            return
        conn.execute(
            "UPDATE session_state SET host = ? WHERE session_id = ?",
            [label, session_id],
        )
    except duckdb.Error:
        # Pre-v19 DBs without host column should not abort indexing mid-write;
        # ensure_schema normally advances before index runs.
        logger.debug("session host stamp skipped for %s", session_id, exc_info=True)


# Sidecar-sourced metadata that silently stops arriving is the failure this
# module already shipped twice. Report it, but as one aggregated line per source
# per interval -- a long-lived daemon re-indexes constantly, so a per-session
# warning would bury the log (REQ-INDEX-019).
_METADATA_DRIFT_WARN_INTERVAL_SECONDS = 3600.0
_last_metadata_drift_warning: dict[str, float] = {}


def _warn_on_metadata_drift(
    missing_by_source: dict[str, int], seen_by_source: dict[str, int]
) -> None:
    """Warn once per source per interval when sidecars yield no start timestamp.

    A sidecar that exists and parses, yet produces no ``started_at``, means the
    upstream format moved under the parser. That is indistinguishable from
    "session genuinely has no start time" at the row level, so it is only
    visible in aggregate -- hence the count rather than a per-session message.
    """
    now = time.monotonic()
    for source, missing in sorted(missing_by_source.items()):
        if missing <= 0:
            continue
        last_warned = _last_metadata_drift_warning.get(source)
        if last_warned is not None and now - last_warned < _METADATA_DRIFT_WARN_INTERVAL_SECONDS:
            continue
        _last_metadata_drift_warning[source] = now
        logger.warning(
            "%s: %d of %d indexed sessions parsed without a start timestamp "
            "despite a sidecar being present - the parser may be stale against "
            "an upstream format change",
            source,
            missing,
            seen_by_source.get(source, missing),
        )


def _sidecar_fingerprint(parser: SessionParser, path: Path) -> float:
    """Newest mtime across the parser's declared sidecars; 0.0 when it has none.

    A single float is enough: any edit to any sidecar moves the newest mtime,
    and that is the only question the change signal asks (REQ-INDEX-018).
    Missing sidecars contribute nothing, so a session that never had one
    fingerprints identically to a source that declares none.
    """
    newest = 0.0
    for sidecar in parser.sidecar_paths(path):
        try:
            newest = max(newest, sidecar.stat().st_mtime)
        except OSError:
            continue
    return newest


def _discover_paths(
    source: Source | None, *, sources: Mapping[str, SourceConfig] | None
) -> list[DiscoveredSessionPath]:
    """Every transcript the configured parsers own (REQ-LIVE-012).

    `sources` is required rather than defaulting to `None`: omitting it silently
    widens discovery back to the built-in roots, which is how the watch catch-up
    came to ignore a pinned lane and index the whole host.
    """
    parsers = [get_parser(source, sources)] if source else all_parsers(sources)
    paths: list[DiscoveredSessionPath] = []
    for parser in parsers:
        for path in parser.discover():
            try:
                stat = path.stat()
            except OSError:
                continue
            paths.append(
                DiscoveredSessionPath(
                    parser=parser,
                    path=path,
                    resolved_path=str(path.resolve()),
                    file_mtime=stat.st_mtime,
                    file_size=stat.st_size,
                    sidecar_mtime=_sidecar_fingerprint(parser, path),
                )
            )
    # Newest first: recent sessions get indexed before old history (REQ-INDEX-013)
    paths.sort(key=lambda p: p.file_mtime, reverse=True)
    return paths


def _scope_paths(
    conn: duckdb.DuckDBPyConnection | None,
    paths: list[DiscoveredSessionPath],
    *,
    since: datetime | None,
    project: str | None,
) -> list[DiscoveredSessionPath]:
    """Narrow discovered transcripts to a slice of the corpus (REQ-INDEX-022).

    A parser change only reaches already-indexed sessions through a full
    re-parse, which on a large corpus is one very long all-or-nothing run.
    Scoping lets it be done in chunks.

    ``since`` compares the transcript's mtime rather than the session's
    recorded time: it is the file we are deciding whether to re-parse, and
    using it needs no database, so a transcript that was never indexed is
    scoped the same way as one that was.
    """
    scoped = paths
    if since is not None:
        cutoff = since.timestamp()
        scoped = [path for path in scoped if path.file_mtime >= cutoff]
    if project:
        if conn is None:
            # Silently ignoring the scope would re-parse the whole corpus when
            # the caller asked for one repo -- the opposite of the request.
            raise ValueError("project scope requires a database connection")
        allowed = _source_paths_for_project(conn, project)
        scoped = [path for path in scoped if path.resolved_path in allowed]
    return scoped


def _source_paths_for_project(conn: duckdb.DuckDBPyConnection, project: str) -> set[str]:
    """Source paths whose session sits in a matching git repo.

    Mirrors ``recall list --project`` (``git_repo ILIKE %project%``) so the
    flag means the same thing wherever it appears.
    """
    rows = conn.execute(
        """
        SELECT s.source_path
        FROM sessions s
        JOIN session_state ss ON ss.session_id = s.id
        WHERE ss.git_repo ILIKE ?
        """,
        [f"%{project}%"],
    ).fetchall()
    return {str(row[0]) for row in rows}


@dataclass(frozen=True)
class SessionState:
    file_mtime: float
    file_size: int
    last_byte_offset: int
    message_count: int
    orphan_tool_count: int
    # None means the row predates sidecar fingerprinting (REQ-INDEX-018) and
    # must re-index once. Distinct from 0.0, which means "source has none".
    sidecar_mtime: float | None = None


def _load_session_states(conn: duckdb.DuckDBPyConnection) -> dict[str, SessionState]:
    rows = conn.execute(
        """
        SELECT s.source_path, ss.file_mtime, ss.file_size,
               COALESCE(ss.last_byte_offset, 0), ss.message_count,
               (SELECT COUNT(*) FROM tool_calls tc
                WHERE tc.session_id = s.id AND tc.message_id IS NULL),
               ss.sidecar_mtime
        FROM sessions s
        JOIN session_state ss ON ss.session_id = s.id
        """
    ).fetchall()
    return {
        str(row[0]): SessionState(
            file_mtime=float(row[1]),
            file_size=int(row[2]),
            last_byte_offset=int(row[3]),
            message_count=int(row[4]),
            orphan_tool_count=int(row[5]),
            sidecar_mtime=None if row[6] is None else float(row[6]),
        )
        for row in rows
    }


def _load_session_state(conn: duckdb.DuckDBPyConnection, source_path: str) -> SessionState | None:
    row = conn.execute(
        """
        SELECT ss.file_mtime, ss.file_size,
               COALESCE(ss.last_byte_offset, 0), ss.message_count,
               (SELECT COUNT(*) FROM tool_calls tc
                WHERE tc.session_id = s.id AND tc.message_id IS NULL),
               ss.sidecar_mtime
        FROM sessions s
        JOIN session_state ss ON ss.session_id = s.id
        WHERE s.source_path = ?
        """,
        [source_path],
    ).fetchone()
    if row is None:
        return None
    return SessionState(
        file_mtime=float(row[0]),
        file_size=int(row[1]),
        last_byte_offset=int(row[2]),
        message_count=int(row[3]),
        orphan_tool_count=int(row[4]),
        sidecar_mtime=None if row[5] is None else float(row[5]),
    )


def _message_from_persisted_row(
    session_id: str,
    row: tuple[object, ...],
    *,
    offset: int = 0,
) -> Message:
    return Message(
        id=str(row[offset]),
        session_id=session_id,
        idx=cast(int, row[offset + 1]),
        role=Role(cast(str, row[offset + 2])),
        content=cast(str | None, row[offset + 3]),
        thinking=cast(str | None, row[offset + 4]),
        timestamp=cast(datetime | None, row[offset + 5]),
        has_thinking=bool(row[offset + 6]),
        agent_id=cast(str | None, row[offset + 7]),
    )


def _load_session_messages(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
) -> list[Message]:
    rows = conn.execute(
        """
        SELECT
            m.id,
            m.idx,
            ms.role,
            ms.content,
            ms.thinking,
            ms.timestamp,
            ms.has_thinking,
            m.agent_id
        FROM messages m
        JOIN message_state ms ON ms.message_id = m.id
        WHERE m.session_id = ?
        ORDER BY m.idx
        """,
        [session_id],
    ).fetchall()
    return [_message_from_persisted_row(session_id, tuple(row)) for row in rows]


def _load_context_recompute_groups(
    conn: duckdb.DuckDBPyConnection,
    *,
    since: datetime | None,
    only_mode: str | None,
) -> list[ContextRecomputeGroup]:
    target_exists = """
        EXISTS (
            SELECT 1
            FROM messages target_m
            JOIN message_state target_ms ON target_ms.message_id = target_m.id
            WHERE target_m.session_id = s.id
              AND (target_ms.content IS NOT NULL OR target_ms.thinking IS NOT NULL)
        """
    session_where_parts: list[str] = []
    params: list[object] = []
    if since is not None:
        session_where_parts.append("COALESCE(ss.ended_at, ss.started_at, ss.indexed_at) >= ?")
        params.append(since)
    if only_mode is not None:
        target_exists += " AND COALESCE(target_ms.context_mode, 'off') = ?"
        params.append(only_mode)
    target_exists += "\n        )"
    session_where_parts.append(target_exists)
    target_predicate = "(ms.content IS NOT NULL OR ms.thinking IS NOT NULL)"
    if only_mode is not None:
        target_predicate += " AND COALESCE(ms.context_mode, 'off') = ?"
        params.append(only_mode)
    session_where_clause = " AND ".join(session_where_parts)
    rows = conn.execute(
        f"""
        WITH target_sessions AS (
            SELECT s.id
            FROM sessions s
            JOIN session_state ss ON ss.session_id = s.id
            WHERE {session_where_clause}
        )
        SELECT
            s.id,
            s.source,
            s.source_path,
            s.source_session_id,
            ss.started_at,
            ss.ended_at,
            ss.duration_seconds,
            ss.model,
            ss.cwd,
            ss.git_repo,
            ss.git_branch,
            ss.message_count,
            ss.tool_count,
            ss.input_tokens,
            ss.output_tokens,
            ss.is_complete,
            ss.file_mtime,
            ss.file_size,
            ss.indexed_at,
            m.id,
            m.idx,
            ms.role,
            ms.content,
            ms.thinking,
            ms.timestamp,
            ms.has_thinking,
            m.agent_id,
            {target_predicate} AS is_target
        FROM sessions s
        JOIN session_state ss ON ss.session_id = s.id
        JOIN messages m ON m.session_id = s.id
        JOIN message_state ms ON ms.message_id = m.id
        JOIN target_sessions ts ON ts.id = s.id
        ORDER BY s.id, m.idx
        """,
        params,
    ).fetchall()
    sessions: OrderedDict[str, Session] = OrderedDict()
    targets_by_session: dict[str, list[Message]] = {}
    for row in rows:
        session_id = str(row[0])
        session = sessions.get(session_id)
        if session is None:
            session = Session(
                id=session_id,
                source=Source(row[1]),
                source_path=str(row[2]),
                source_session_id=cast(str | None, row[3]),
                started_at=cast(datetime | None, row[4]),
                ended_at=cast(datetime | None, row[5]),
                duration_seconds=cast(int | None, row[6]),
                model=cast(str | None, row[7]),
                cwd=cast(str | None, row[8]),
                git_repo=cast(str | None, row[9]),
                git_branch=cast(str | None, row[10]),
                message_count=int(row[11] or 0),
                tool_count=int(row[12] or 0),
                input_tokens=cast(int | None, row[13]),
                output_tokens=cast(int | None, row[14]),
                is_complete=bool(row[15]),
                file_mtime=float(row[16] or 0.0),
                file_size=int(row[17] or 0),
                indexed_at=cast(datetime | None, row[18]),
            )
            sessions[session_id] = session
            targets_by_session[session_id] = []
        message = _message_from_persisted_row(session_id, tuple(row), offset=19)
        session.messages.append(message)
        if bool(row[27]):
            targets_by_session[session_id].append(message)
    return [
        ContextRecomputeGroup(session=session, targets=targets_by_session[session_id])
        for session_id, session in sessions.items()
        if targets_by_session[session_id]
    ]


def _resolve_worker_count(workers: int | str, changed_count: int) -> int:
    if isinstance(workers, int):
        if workers <= 0:
            raise ValueError("workers must be positive")
        return workers

    normalized = workers.strip().lower()
    if normalized != "auto":
        raise ValueError("workers must be a positive integer or 'auto'")
    if changed_count <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 8, changed_count))


def _fts_indexes_exist(conn: duckdb.DuckDBPyConnection) -> bool:
    """Check whether FTS indexes have been created."""
    row = conn.execute(
        "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name LIKE 'fts_main_%'"
    ).fetchone()
    return bool(row and row[0] > 0)


def _is_unchanged(
    path: DiscoveredSessionPath,
    session_states: dict[str, SessionState],
) -> bool:
    state = session_states.get(path.resolved_path)
    if state is None:
        return False
    if state.file_mtime != path.file_mtime or state.file_size != path.file_size:
        return False
    # A NULL stamp predates sidecar fingerprinting: re-index once so
    # sidecar-sourced metadata lands, then the stamp compares normally.
    if state.sidecar_mtime is None:
        return path.sidecar_mtime == 0.0
    return state.sidecar_mtime == path.sidecar_mtime


def _emit_progress(
    progress_callback: Callable[[IndexProgress], None] | None,
    processed: int,
    total: int,
    indexed: int,
    skipped: int,
    failed: int,
    status: Literal["start", "indexed", "skipped", "failed", "done"],
    path: str | None = None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        IndexProgress(
            processed=processed,
            total=total,
            indexed=indexed,
            skipped=skipped,
            failed=failed,
            status=status,
            path=path,
        )
    )


def _resolve_parse_offset(state: SessionState | None, discovered: DiscoveredSessionPath) -> int:
    """Return the safe parse boundary for a changed source on the legacy paths.

    ``session_state.last_byte_offset`` predates durable prefix validation.  It
    says where a previous reader stopped, not that the bytes before it are
    unchanged, so it cannot authorize an append merge.  The prefix digest and
    adapter resume proof that can authorize one live on the source catalog, and
    only raw reconciliation holds them (``coordinator.prepare_raw_sources``);
    a caller arriving here has neither, so it takes the full reference path.
    """
    del state, discovered
    return 0


def _capture_still_matches(path: Path, result: ParseResult) -> bool:
    """Verify the source prefix about to be committed is the parser's capture."""
    if result.captured_prefix_sha256 is None:
        # Adapters are migrated incrementally; without a capture digest they
        # take the full normalization path but cannot claim a validated prefix.
        return result.is_full_parse
    try:
        digest = hashlib.sha256()
        remaining = result.captured_size
        with path.open("rb") as source:
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    return False
                digest.update(chunk)
                remaining -= len(chunk)
    except OSError:
        return False
    return digest.hexdigest() == result.captured_prefix_sha256


def _require_committable_capture(
    path: Path,
    result: ParseResult,
    *,
    has_indexed_history: bool,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> None:
    """Reject a result that cannot safely replace an indexed session.

    A full parse is a replacement operation.  Publishing its supported prefix
    after a malformed, unsupported, or torn record would delete valid rows
    after that point.  The source remains retryable and its parser diagnostics
    remain available to the reconciliation layer, but this writer must leave
    the last known-good session intact.
    """
    if not _capture_still_matches(path, result):
        raise RuntimeError("source changed while parser capture was in flight")
    if result.diagnostics and has_indexed_history:
        if (
            conn is not None
            and all(item.kind == "unterminated_tail" for item in result.diagnostics)
            and _preserves_indexed_prefix(conn, result.session)
        ):
            return
        kinds = ", ".join(diagnostic.kind for diagnostic in result.diagnostics)
        raise RuntimeError(f"parser result is incomplete and cannot replace history: {kinds}")


def _preserves_indexed_prefix(conn: duckdb.DuckDBPyConnection, session: Session) -> bool:
    """A torn append may advance complete records without dropping any old fact."""
    previous = _load_persisted_session_rows(conn, session.id)
    if previous is None:
        return False
    desired_messages = {message.id: message for message in session.messages}
    desired_calls = {call.id: call for call in _collect_tool_calls(session)}
    return all(
        str(row[0]) in desired_messages
        and tuple(_normalize_timestamp_for_compare(value) for value in row[:9])
        == tuple(
            _normalize_timestamp_for_compare(value)
            for value in _message_row_values(desired_messages[str(row[0])])
        )
        for row in previous.message_rows
    ) and all(
        str(row[0]) in desired_calls
        and _tool_call_payload_signature(row)
        == _tool_call_payload_signature(_tool_call_row_values(desired_calls[str(row[0])]))
        for row in previous.tool_call_rows
    )


def _incremental_write_session(
    conn: duckdb.DuckDBPyConnection,
    session: Session,
    *,
    last_byte_offset: int = 0,
    context_text: str = "",
    context_mode: str = "off",
    sidecar_conn: sqlite3.Connection | None = None,
    fts_fields: tuple[str, ...] | None = None,
    sidecar_touched: FtsSidecarTouchedIds | None = None,
    host: str | None = None,
    tail_facts: TailFacts,
    on_commit: Callable[[], None] | None = None,
) -> bool:
    """Append-only write for incrementally parsed sessions (offset > 0).

    Only inserts NEW messages and tool calls. Merges session metadata
    (extends ended_at, merges token counts, updates mtime/size/offset).
    Does not delete or modify existing rows.
    """
    has_searchable = _session_has_searchable_content(session)
    is_absolute_token_source = session.source in ABSOLUTE_TOKEN_SOURCES
    session_sidecar_touched, owns_sidecar_touched = _sidecar_touched_accumulator(
        sidecar_conn,
        sidecar_touched,
    )
    conn.execute("BEGIN")
    try:
        # Merge session metadata incrementally
        merged = conn.execute(
            """
            UPDATE session_state
            SET
                -- Sidecar-sourced, not stream-derived: an append cannot move a
                -- start time, but a sidecar refresh can (REQ-INDEX-018). Parsers
                -- with no sidecar yield NULL here and keep the stored value.
                started_at = COALESCE(?, started_at),
                ended_at = GREATEST(ended_at, ?),
                duration_seconds = CASE
                    WHEN started_at IS NOT NULL AND ? IS NOT NULL
                    THEN CAST(EXTRACT(EPOCH FROM (? - started_at)) AS INTEGER)
                    ELSE duration_seconds
                END,
                model = COALESCE(?, model),
                cwd = COALESCE(?, cwd),
                git_repo = COALESCE(?, git_repo),
                git_branch = COALESCE(?, git_branch),
                message_count = message_count + ?,
                tool_count = tool_count + ?,
                input_tokens = CASE
                    WHEN ? IS NULL THEN input_tokens
                    WHEN ? THEN GREATEST(COALESCE(input_tokens, 0), ?)
                    ELSE COALESCE(input_tokens, 0) + ?
                END,
                output_tokens = CASE
                    WHEN ? IS NULL THEN output_tokens
                    WHEN ? THEN GREATEST(COALESCE(output_tokens, 0), ?)
                    ELSE COALESCE(output_tokens, 0) + ?
                END,
                -- Replaced, not ANDed. A diagnostic never acknowledges past
                -- itself, so every parser leaves `is_complete` false exactly
                -- when the bytes from the committed boundary onward diagnose
                -- something -- and those are the bytes this suffix re-reads.
                -- ANDing made a torn tail permanent: once a half-written line
                -- marked the session incomplete, no append could clear it,
                -- while a full parse of the same bytes reports it complete.
                is_complete = ?,
                file_mtime = ?,
                file_size = ?,
                sidecar_mtime = ?,
                last_byte_offset = ?,
                indexed_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
            RETURNING session_id
            """,
            [
                session.started_at,
                session.ended_at,
                session.ended_at,
                session.ended_at,
                session.model,
                session.cwd,
                session.git_repo,
                session.git_branch,
                session.message_count,
                session.tool_count,
                session.input_tokens,
                is_absolute_token_source,
                session.input_tokens,
                session.input_tokens,
                session.output_tokens,
                is_absolute_token_source,
                session.output_tokens,
                session.output_tokens,
                session.is_complete,
                session.file_mtime,
                session.file_size,
                session.sidecar_mtime,
                last_byte_offset,
                session.id,
            ],
        )
        if merged.fetchone() is None:
            # An append has no session to append to. Inserting the suffix here
            # would leave its rows under a parent that does not exist and let
            # the caller acknowledge the offset, so the prefix could never be
            # rebuilt. Fail the turn instead and leave the catalog alone.
            raise RuntimeError(f"no indexed session to append to: {session.id}")
        _set_session_host(conn, session.id, host)
        if session.messages:
            _insert_messages(
                conn,
                session.messages,
                context_text=context_text,
                context_mode=context_mode,
                sidecar_conn=sidecar_conn,
                fts_fields=fts_fields,
                sidecar_touched=session_sidecar_touched,
            )
        tool_calls = list(_collect_tool_calls(session))
        if tool_calls:
            insert_tool_calls(conn, tool_calls)
            _upsert_tool_call_sidecar_rows(
                conn,
                sidecar_conn,
                tool_calls,
                fts_fields=fts_fields,
                sidecar_touched=session_sidecar_touched,
            )
        _write_tail_facts(conn, session.id, tool_calls, tail_facts)
        _insert_session_embeddings(conn, session.messages, tool_calls)
        if on_commit is not None:
            on_commit()
        conn.execute("COMMIT")
    except Exception as exc:
        _log_failed_transaction(exc, session_id=session.id, site="_incremental_write_session")
        conn.execute("ROLLBACK")
        _enqueue_owned_sidecar_touched_after_rollback(
            conn,
            session_sidecar_touched,
            owns_sidecar_touched,
        )
        raise
    return has_searchable


def _write_session(
    conn: duckdb.DuckDBPyConnection,
    session: Session,
    *,
    last_byte_offset: int = 0,
    context_text: str = "",
    context_mode: str = "off",
    sidecar_conn: sqlite3.Connection | None = None,
    fts_fields: tuple[str, ...] | None = None,
    sidecar_touched: FtsSidecarTouchedIds | None = None,
    host: str | None = None,
    tail_facts: TailFacts,
    on_commit: Callable[[], None] | None = None,
) -> bool:
    session_sidecar_touched, owns_sidecar_touched = _sidecar_touched_accumulator(
        sidecar_conn,
        sidecar_touched,
    )
    previous_rows = _load_persisted_session_rows(conn, session.id)
    try:
        if previous_rows is not None:
            result = _sync_existing_session(
                conn,
                previous_rows,
                session,
                last_byte_offset=last_byte_offset,
                context_text=context_text,
                context_mode=context_mode,
                sidecar_conn=sidecar_conn,
                fts_fields=fts_fields,
                sidecar_touched=session_sidecar_touched,
                tail_facts=tail_facts,
                on_commit=on_commit,
            )
        else:
            predecessor_ids = _moved_predecessors(
                conn,
                session,
                host=host,
                indexed_bytes=last_byte_offset,
                context_text=context_text,
            )
            _insert_new_session(
                conn,
                session,
                predecessor_ids=predecessor_ids,
                last_byte_offset=last_byte_offset,
                context_text=context_text,
                context_mode=context_mode,
                sidecar_conn=sidecar_conn,
                fts_fields=fts_fields,
                sidecar_touched=session_sidecar_touched,
                tail_facts=tail_facts,
                on_commit=on_commit,
            )
            result = _session_has_searchable_content(session)
        _set_session_host(conn, session.id, host)
        return result
    except Exception:
        _enqueue_owned_sidecar_touched_after_rollback(
            conn,
            session_sidecar_touched,
            owns_sidecar_touched,
        )
        raise


def _moved_predecessors(
    conn: duckdb.DuckDBPyConnection,
    session: Session,
    *,
    host: str | None,
    indexed_bytes: int,
    context_text: str,
) -> tuple[str, ...]:
    """Rows this new transcript provably continues from a path that moved.

    Their embeddings carry over by text, so a moved transcript is not embedded
    again (REQ-INDEX-027).
    """
    from recall.core.types import default_session_host

    predecessor_ids = find_moved_predecessors(
        conn,
        source=session.source.value,
        source_path=session.source_path,
        host=host if host is not None else default_session_host(),
        indexed_bytes=indexed_bytes,
    )
    for predecessor_id in predecessor_ids:
        previous_rows = _load_persisted_session_rows(conn, predecessor_id)
        if previous_rows is not None:
            _preserve_existing_embeddings(previous_rows, session, context_text=context_text)
    return predecessor_ids


def _insert_new_session(
    conn: duckdb.DuckDBPyConnection,
    session: Session,
    *,
    predecessor_ids: tuple[str, ...] = (),
    last_byte_offset: int = 0,
    context_text: str = "",
    context_mode: str = "off",
    sidecar_conn: sqlite3.Connection | None = None,
    fts_fields: tuple[str, ...] | None = None,
    sidecar_touched: FtsSidecarTouchedIds | None = None,
    tail_facts: TailFacts,
    on_commit: Callable[[], None] | None = None,
) -> None:
    conn.execute("BEGIN")
    try:
        supersede(
            conn,
            predecessor_ids=predecessor_ids,
            successor_id=session.id,
            queue_sidecar_deletes=True,
        )
        insert_session(conn, session, last_byte_offset=last_byte_offset)
        _insert_messages(
            conn,
            session.messages,
            context_text=context_text,
            context_mode=context_mode,
            sidecar_conn=sidecar_conn,
            fts_fields=fts_fields,
            sidecar_touched=sidecar_touched,
        )
        tool_calls = list(_collect_tool_calls(session))
        insert_tool_calls(conn, tool_calls)
        _upsert_tool_call_sidecar_rows(
            conn,
            sidecar_conn,
            tool_calls,
            fts_fields=fts_fields,
            sidecar_touched=sidecar_touched,
        )
        _write_tail_facts(conn, session.id, tool_calls, tail_facts)
        _insert_session_embeddings(conn, session.messages, tool_calls)
        if on_commit is not None:
            on_commit()
        conn.execute("COMMIT")
    except Exception as exc:
        _log_failed_transaction(exc, session_id=session.id, site="_insert_new_session")
        conn.execute("ROLLBACK")
        raise


def _write_tail_facts(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    tool_calls: list[ToolCall],
    tail_facts: TailFacts,
) -> None:
    """Persist the harness id mapping and any results it lets us pair.

    The mapping is written first so a result in the same chunk pairs against
    rows this call just inserted, and one from an earlier chunk pairs against
    rows a previous call did. A result naming a call recall never saw — a codex
    exec wrapper's, or a call indexed before this schema — resolves to nothing
    and is dropped rather than stored against a guessed owner.

    Insert-only is the steady state, not an invariant of the table: a full
    re-parse of a rewritten transcript can hand the same positional tool_call
    id a different harness call, so a mapping that no longer matches is retired
    first, taking its stale result with it. A mapping that still matches is
    left alone, which is what keeps an unchanged re-index at zero new row
    versions (REQ-INDEX-017).

    Stop markers are keyed by message position. Re-presenting unchanged facts
    is a no-op; a later lifecycle event at that position replaces the marker.
    """
    upsert_stop_markers(conn, session_id, tail_facts.stop_markers)
    mapped = {
        tool_call.id: tool_call.tool_use_id for tool_call in tool_calls if tool_call.tool_use_id
    }
    stored = fetch_tool_use_ids(conn, list(mapped))
    delete_tool_call_tail_facts(
        conn,
        sorted(
            tool_call_id
            for tool_call_id, tool_use_id in stored.items()
            if mapped[tool_call_id] != tool_use_id
        ),
    )
    insert_tool_use_ids(
        conn,
        ((tool_call_id, session_id, tool_use_id) for tool_call_id, tool_use_id in mapped.items()),
    )
    if not tail_facts.tool_results:
        return
    wanted = [tool_result.tool_use_id for tool_result in tail_facts.tool_results]
    resolved = resolve_tool_call_ids(conn, session_id, wanted)
    insert_tool_results(
        conn,
        (
            (
                resolved[tool_result.tool_use_id],
                tool_result.result_summary,
                tool_result.is_error,
                tool_result.completed_at,
            )
            for tool_result in tail_facts.tool_results
            if tool_result.tool_use_id in resolved
        ),
    )


def _insert_session_embeddings(
    conn: duckdb.DuckDBPyConnection,
    messages: list[Message],
    tool_calls: list[ToolCall],
    *,
    mark_all: bool = False,
) -> None:
    """Insert embedding rows for messages and tool_calls.

    When mark_all=True (embed phase), inserts rows for ALL items
    including those with NULL embeddings, so they're marked as processed
    and won't be re-selected by find_pending_embeds on subsequent cycles.
    When mark_all=False (index path), only inserts rows that have actual
    embedding vectors.
    """
    if mark_all:
        msg_emb_rows: list[tuple[str, list[float] | None, list[float] | None]] = [
            (msg.id, msg.content_embedding, msg.thinking_embedding) for msg in messages
        ]
    else:
        msg_emb_rows = [
            (msg.id, msg.content_embedding, msg.thinking_embedding)
            for msg in messages
            if msg.content_embedding is not None or msg.thinking_embedding is not None
        ]
    if msg_emb_rows:
        insert_message_embeddings(conn, msg_emb_rows)

    tc_emb_rows: list[tuple[str, list[float] | None]]
    if mark_all:
        tc_emb_rows = [(tc.id, tc.bash_embedding) for tc in tool_calls]
    else:
        tc_emb_rows = [
            (tc.id, tc.bash_embedding) for tc in tool_calls if tc.bash_embedding is not None
        ]
    if tc_emb_rows:
        insert_tool_call_embeddings(conn, tc_emb_rows)


def _sync_existing_session(
    conn: duckdb.DuckDBPyConnection,
    previous_rows: PersistedSessionRows,
    session: Session,
    *,
    last_byte_offset: int = 0,
    context_text: str = "",
    context_mode: str = "off",
    sidecar_conn: sqlite3.Connection | None = None,
    fts_fields: tuple[str, ...] | None = None,
    sidecar_touched: FtsSidecarTouchedIds | None = None,
    tail_facts: TailFacts,
    on_commit: Callable[[], None] | None = None,
) -> bool:
    _preserve_existing_embeddings(previous_rows, session, context_text=context_text)
    searchable_changed = _searchable_rows_changed(
        previous_rows,
        session,
        context_text=context_text,
        context_mode=context_mode,
    )
    if _session_identity_changed(previous_rows.identity_row, session):
        _rewrite_session_with_new_identity(
            conn,
            previous_rows,
            session,
            context_text=context_text,
            context_mode=context_mode,
            sidecar_conn=sidecar_conn,
            fts_fields=fts_fields,
            sidecar_touched=sidecar_touched,
            tail_facts=tail_facts,
            on_commit=on_commit,
        )
        return searchable_changed

    existing_message_rows = {str(row[0]): tuple(row) for row in previous_rows.message_rows}
    existing_tool_call_rows = {str(row[0]): tuple(row) for row in previous_rows.tool_call_rows}
    desired_messages = {message.id: message for message in session.messages}
    desired_tool_calls = {tool_call.id: tool_call for tool_call in _collect_tool_calls(session)}

    conn.execute("BEGIN")
    try:
        _delete_removed_rows(
            conn,
            "tool_calls",
            existing_tool_call_rows.keys(),
            desired_tool_calls.keys(),
            sidecar_conn=sidecar_conn,
            sidecar_touched=sidecar_touched,
        )
        _delete_removed_rows(
            conn,
            "messages",
            existing_message_rows.keys(),
            desired_messages.keys(),
            sidecar_conn=sidecar_conn,
            sidecar_touched=sidecar_touched,
        )
        _update_session_row(
            conn, previous_rows.session_row, session, last_byte_offset=last_byte_offset
        )
        _upsert_messages(
            conn,
            existing_message_rows,
            desired_messages,
            context_text=context_text,
            context_mode=context_mode,
            sidecar_conn=sidecar_conn,
            fts_fields=fts_fields,
            sidecar_touched=sidecar_touched,
        )
        _upsert_tool_calls(
            conn,
            existing_tool_call_rows,
            desired_tool_calls,
            sidecar_conn=sidecar_conn,
            fts_fields=fts_fields,
            sidecar_touched=sidecar_touched,
        )
        # This parse replaces the session, so positions its predecessor left
        # behind go with the messages `_delete_removed_rows` just retired.
        retire_stop_markers(conn, session.id, tail_facts.stop_markers)
        _write_tail_facts(conn, session.id, list(desired_tool_calls.values()), tail_facts)
        if on_commit is not None:
            on_commit()
        conn.execute("COMMIT")
    except Exception as exc:
        _log_failed_transaction(exc, session_id=session.id, site="_sync_existing_session")
        conn.execute("ROLLBACK")
        raise
    return searchable_changed


def _load_persisted_session_rows(
    conn: duckdb.DuckDBPyConnection, session_id: str
) -> PersistedSessionRows | None:
    identity_row = conn.execute(
        """
        SELECT id, source, source_path, source_session_id
        FROM sessions
        WHERE id = ?
        """,
        [session_id],
    ).fetchone()
    if identity_row is None:
        return None

    session_row = conn.execute(
        """
        SELECT
            session_id, started_at, ended_at, duration_seconds,
            model, cwd, git_repo, git_branch,
            message_count, tool_count, input_tokens, output_tokens,
            is_complete, file_mtime, file_size, last_byte_offset, indexed_at,
            cached_input_tokens, host
        FROM session_state
        WHERE session_id = ?
        """,
        [session_id],
    ).fetchone()
    if session_row is None:
        return None

    return PersistedSessionRows(
        identity_row=tuple(identity_row),
        session_row=tuple(session_row),
        message_rows=_load_persisted_message_rows(conn, session_id),
        tool_call_rows=_load_persisted_tool_call_rows(conn, session_id),
    )


def _load_persisted_message_rows(
    conn: duckdb.DuckDBPyConnection, session_id: str
) -> list[tuple[object, ...]]:
    identities = conn.execute(
        "SELECT id, session_id, idx, agent_id FROM messages WHERE session_id = ? ORDER BY idx",
        [session_id],
    ).fetchall()
    rows: list[tuple[object, ...]] = []
    # Joining a session filter to the corpus-wide state/vector tables can scan
    # millions of unrelated payloads. Bound each lookup to explicit primary keys.
    for group in batched(identities, 256):
        ids = [row[0] for row in group]
        owned, params = id_set_predicate("message_id", ids)
        states = {
            row[0]: row
            for row in conn.execute(
                f"""
                SELECT message_id, role, content, thinking, timestamp, has_thinking,
                    COALESCE(context_text, ''), COALESCE(context_mode, 'off')
                FROM message_state WHERE {owned}
                """,
                params,
            ).fetchall()
        }
        embeddings = {
            row[0]: row[1:]
            for row in conn.execute(
                f"""
                SELECT message_id, content_embedding, thinking_embedding
                FROM message_embeddings WHERE {owned}
                """,
                params,
            ).fetchall()
        }
        for identity in group:
            state = states.get(identity[0])
            if state is None:
                continue
            rows.append(
                (
                    *identity[:3],
                    *state[1:6],
                    identity[3],
                    *state[6:],
                    *embeddings.get(identity[0], (None, None)),
                )
            )
    return rows


def _load_persisted_tool_call_rows(
    conn: duckdb.DuckDBPyConnection, session_id: str
) -> list[tuple[object, ...]]:
    calls = conn.execute(
        """
        SELECT id, session_id, message_id, idx, tool_name, tool_input,
            bash_command, bash_base, bash_sub, is_compound, agent_id,
            subagent_type, subagent_description, subagent_model, skill_name
        FROM tool_calls WHERE session_id = ? ORDER BY idx
        """,
        [session_id],
    ).fetchall()
    rows: list[tuple[object, ...]] = []
    for group in batched(calls, 256):
        ids = [row[0] for row in group]
        owned, params = id_set_predicate("tool_call_id", ids)
        embeddings = {
            row[0]: row[1]
            for row in conn.execute(
                f"""
                SELECT tool_call_id, bash_embedding
                FROM tool_call_embeddings WHERE {owned}
                """,
                params,
            ).fetchall()
        }
        rows.extend((*row, embeddings.get(row[0])) for row in group)
    return rows


def _delete_removed_rows(
    conn: duckdb.DuckDBPyConnection,
    table: Literal["messages", "tool_calls"],
    existing_ids: Iterable[str],
    desired_ids: Iterable[str],
    *,
    sidecar_conn: sqlite3.Connection | None = None,
    sidecar_touched: FtsSidecarTouchedIds | None = None,
) -> None:
    removed_ids = sorted(set(existing_ids) - set(desired_ids))
    if not removed_ids:
        return
    if table == "messages":
        owned, params = id_set_predicate("message_id", removed_ids)
        conn.execute(f"DELETE FROM message_embeddings WHERE {owned}", params)
        conn.execute(f"DELETE FROM message_state WHERE {owned}", params)
    elif table == "tool_calls":
        owned, params = id_set_predicate("tool_call_id", removed_ids)
        conn.execute(f"DELETE FROM tool_call_embeddings WHERE {owned}", params)
        delete_tool_call_tail_facts(conn, removed_ids)
    removed, params = id_set_predicate("id", removed_ids)
    conn.execute(f"DELETE FROM {table} WHERE {removed}", params)
    if sidecar_conn is None:
        return
    if sidecar_touched is not None:
        if table == "messages":
            sidecar_touched.add_messages(removed_ids)
        else:
            sidecar_touched.add_tool_calls(removed_ids)
    if table == "messages":
        try:
            delete_message_fts(sidecar_conn, removed_ids)
        except sqlite3.Error as err:
            logger.warning(
                "sidecar delete failed for removed messages: %s; queueing for reconciliation",
                err,
            )
            _enqueue_sidecar_pending_best_effort(conn, "message", removed_ids, "delete")
    elif table == "tool_calls":
        try:
            delete_tool_call_fts(sidecar_conn, removed_ids)
        except sqlite3.Error as err:
            logger.warning(
                "sidecar delete failed for removed tool_calls: %s; queueing for reconciliation",
                err,
            )
            _enqueue_sidecar_pending_best_effort(conn, "tool_call", removed_ids, "delete")


def _update_session_row(
    conn: duckdb.DuckDBPyConnection,
    existing_row: tuple[object, ...],
    session: Session,
    *,
    last_byte_offset: int = 0,
) -> None:
    values = _session_row_values(session, last_byte_offset=last_byte_offset)
    if _session_update_is_noop(existing_row, values):
        return
    # Preserve harvest-filled tokens when the parser leaves them None
    # (Grok chat_history has no usage — REQ-USAGE-014 / idempotency).
    conn.execute(
        """
        UPDATE session_state
        SET
            started_at = ?,
            ended_at = ?,
            duration_seconds = ?,
            model = ?,
            cwd = ?,
            git_repo = ?,
            git_branch = ?,
            message_count = ?,
            tool_count = ?,
            input_tokens = CASE WHEN ? IS NULL THEN input_tokens ELSE ? END,
            output_tokens = CASE WHEN ? IS NULL THEN output_tokens ELSE ? END,
            is_complete = ?,
            file_mtime = ?,
            file_size = ?,
            last_byte_offset = ?,
            indexed_at = ?
        WHERE session_id = ?
        """,
        [
            values[1],  # started_at
            values[2],  # ended_at
            values[3],  # duration_seconds
            values[4],  # model
            values[5],  # cwd
            values[6],  # git_repo
            values[7],  # git_branch
            values[8],  # message_count
            values[9],  # tool_count
            values[10],
            values[10],  # input_tokens null-preserve
            values[11],
            values[11],  # output_tokens null-preserve
            values[12],  # is_complete
            values[13],  # file_mtime
            values[14],  # file_size
            values[15],  # last_byte_offset
            values[16],  # indexed_at
            values[0],  # session_id
        ],
    )


class _MessageStateUpdate(TypedDict):
    message_id: str
    role: str
    content: str | None
    thinking: str | None
    timestamp: datetime | None
    timestamp_utc: datetime | None
    has_thinking: bool
    context_text: str
    context_mode: str
    agent_id: str | None
    changed_columns: int
    agent_changed: bool


_MESSAGE_STATE_UPDATE_ASSIGNMENTS = (
    "role = incoming.role",
    "content = incoming.content",
    "thinking = incoming.thinking",
    "timestamp = COALESCE(CAST(incoming.timestamp_utc AS TIMESTAMP), incoming.timestamp)",
    "has_thinking = incoming.has_thinking",
    "context_text = incoming.context_text",
    "context_mode = incoming.context_mode",
    "fts_content = incoming.context_text || COALESCE(incoming.content, '')",
    "fts_thinking = incoming.context_text || COALESCE(incoming.thinking, '')",
)


def _changed_column_mask(existing: tuple[object, ...], desired: tuple[object, ...]) -> int:
    assert len(existing) == len(desired)
    return sum(
        1 << index
        for index, (old_value, new_value) in enumerate(zip(existing, desired, strict=True))
        if old_value != new_value
    )


def _message_state_change_mask(
    existing_row: tuple[object, ...],
    values: tuple[object, ...],
    *,
    context_text: str,
    context_mode: str,
) -> int:
    existing_context = cast(str, existing_row[9] or "")
    desired_content = cast(str | None, values[4])
    desired_thinking = cast(str | None, values[5])
    existing = _canonicalize_scalar_row(
        (
            existing_row[3],
            existing_row[4],
            existing_row[5],
            existing_row[6],
            existing_row[7],
            existing_context,
            existing_row[10],
            f"{existing_context}{existing_row[4] or ''}",
            f"{existing_context}{existing_row[5] or ''}",
        )
    )
    desired = _canonicalize_scalar_row(
        (
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
            context_text,
            context_mode,
            f"{context_text}{desired_content or ''}",
            f"{context_text}{desired_thinking or ''}",
        )
    )
    return _changed_column_mask(existing, desired)


def _update_assignments(mask: int, assignments: tuple[str, ...]) -> str:
    assert 0 < mask < 1 << len(assignments)
    return ",\n                    ".join(
        assignment for index, assignment in enumerate(assignments) if mask & (1 << index)
    )


def _update_message_states(
    conn: duckdb.DuckDBPyConnection, updates: list[_MessageStateUpdate]
) -> None:
    if not updates:
        return
    import pyarrow as pa

    schema = pa.schema(
        [
            ("message_id", pa.string()),
            ("role", pa.string()),
            ("content", pa.string()),
            ("thinking", pa.string()),
            ("timestamp", pa.timestamp("us")),
            ("timestamp_utc", pa.timestamp("us", tz="UTC")),
            ("has_thinking", pa.bool_()),
            ("context_text", pa.string()),
            ("context_mode", pa.string()),
            ("agent_id", pa.string()),
            ("changed_columns", pa.uint16()),
            ("agent_changed", pa.bool_()),
        ]
    )
    for batch in batched(updates, 256):
        # DuckDB 1.5.5 replaces a full row when UPDATE names an indexed column,
        # even if that column keeps its value. Cohort exact masks so unchanged
        # has_thinking and payload columns create no durable index or column work.
        # Remove masks only when test_rewrite_updates_only_changed_columns_and_
        # preserves_related_rows passes without them on every supported runtime.
        cohorts: dict[int, list[_MessageStateUpdate]] = {}
        for update in batch:
            cohorts.setdefault(update["changed_columns"], []).append(update)
        assert len(cohorts) <= 256
        for changed_columns, cohort in sorted(cohorts.items()):
            if changed_columns == 0:
                continue
            table = pa.Table.from_pylist(cohort, schema=schema)
            # Explicit registration keeps the relation scoped to this connection,
            # including when a caller wraps it for fault injection or diagnostics.
            conn.register("_recall_message_updates", table)
            try:
                assignments = _update_assignments(
                    changed_columns, _MESSAGE_STATE_UPDATE_ASSIGNMENTS
                )
                conn.execute(
                    f"""
                UPDATE message_state
                SET {assignments}
                FROM _recall_message_updates AS incoming
                WHERE message_state.message_id = incoming.message_id
                """
                )
            finally:
                conn.unregister("_recall_message_updates")
        agent_updates = [update for update in batch if update["agent_changed"]]
        if agent_updates:
            table = pa.Table.from_pylist(agent_updates, schema=schema)
            conn.register("_recall_message_updates", table)
            try:
                conn.execute(
                    """
                UPDATE messages
                SET agent_id = incoming.agent_id
                FROM _recall_message_updates AS incoming
                WHERE messages.id = incoming.message_id
                """
                )
            finally:
                conn.unregister("_recall_message_updates")


def _upsert_messages(
    conn: duckdb.DuckDBPyConnection,
    existing_rows: dict[str, tuple[object, ...]],
    desired_messages: dict[str, Message],
    *,
    context_text: str = "",
    context_mode: str = "off",
    sidecar_conn: sqlite3.Connection | None = None,
    fts_fields: tuple[str, ...] | None = None,
    sidecar_touched: FtsSidecarTouchedIds | None = None,
) -> None:
    new_messages: list[Message] = []
    sidecar_rows: list[tuple[str, str, str]] = []
    emb_rows: list[tuple[str, list[float] | None, list[float] | None]] = []
    obsolete_embedding_ids: list[str] = []
    state_updates: list[_MessageStateUpdate] = []
    for message_id, message in desired_messages.items():
        values = _message_row_values(message)
        existing_row = existing_rows.get(message_id)
        message_context_text, message_context_mode = _message_context(
            message,
            fallback_context_text=context_text,
            fallback_context_mode=context_mode,
        )
        has_emb = message.content_embedding is not None or message.thinking_embedding is not None
        if existing_row is None:
            new_messages.append(message)
            if has_emb:
                emb_rows.append(
                    (
                        message_id,
                        message.content_embedding,
                        message.thinking_embedding,
                    )
                )
            continue
        state_values = (*values, message_context_text, message_context_mode)
        existing_embs = _canonicalize_scalar_row((existing_row[11], existing_row[12]))
        desired_embs = _canonicalize_scalar_row(
            (message.content_embedding, message.thinking_embedding)
        )
        if _canonicalize_scalar_row(existing_row[:11]) == _canonicalize_scalar_row(state_values):
            if existing_embs != desired_embs and has_emb:
                obsolete_embedding_ids.append(message_id)
                emb_rows.append(
                    (
                        message_id,
                        message.content_embedding,
                        message.thinking_embedding,
                    )
                )
            continue
        timestamp = message.timestamp
        aware = timestamp is not None and timestamp.utcoffset() is not None
        # Keep naive and aware timestamps distinct until DuckDB performs the
        # same TIMESTAMP conversion as its per-value parameter binding.
        state_updates.append(
            {
                "message_id": message_id,
                "role": message.role.value,
                "content": message.content,
                "thinking": message.thinking,
                "timestamp": None if aware else timestamp,
                "timestamp_utc": timestamp if aware else None,
                "has_thinking": message.has_thinking,
                "context_text": message_context_text,
                "context_mode": message_context_mode,
                "agent_id": message.agent_id,
                "changed_columns": _message_state_change_mask(
                    existing_row,
                    values,
                    context_text=message_context_text,
                    context_mode=message_context_mode,
                ),
                "agent_changed": existing_row[8] != message.agent_id,
            }
        )
        if sidecar_conn is not None:
            sidecar_rows.append(
                (
                    message_id,
                    f"{message_context_text}{message.content or ''}",
                    f"{message_context_text}{message.thinking or ''}",
                )
            )
        # Preservation already resolved the complete valid vector pair. Replace
        # a different pair together: INSERT OR IGNORE cannot fill a NULL field
        # in an existing processed row, including one retained for its sibling.
        if has_emb:
            if existing_embs != desired_embs:
                obsolete_embedding_ids.append(message_id)
                emb_rows.append(
                    (
                        message_id,
                        message.content_embedding,
                        message.thinking_embedding,
                    )
                )
        elif _invalidate_changed_message_embeddings(
            conn,
            message_id,
            existing_row,
            values,
            context_text=message_context_text,
        ):
            obsolete_embedding_ids.append(message_id)
    _update_message_states(conn, state_updates)
    if sidecar_conn is not None and sidecar_rows:
        if sidecar_touched is not None:
            sidecar_touched.add_messages(
                message_id for message_id, _content, _thinking in sidecar_rows
            )
        try:
            upsert_message_fts_batch(sidecar_conn, sidecar_rows, fields=fts_fields)
        except sqlite3.Error as err:
            failed_ids = [message_id for message_id, _content, _thinking in sidecar_rows]
            logger.warning(
                "sidecar dual-write failed for %d messages: %s; queueing for reconciliation",
                len(failed_ids),
                err,
            )
            _enqueue_sidecar_pending_best_effort(conn, "message", failed_ids, "upsert")
    if new_messages:
        _insert_messages(
            conn,
            new_messages,
            context_text=context_text,
            context_mode=context_mode,
            sidecar_conn=sidecar_conn,
            fts_fields=fts_fields,
            sidecar_touched=sidecar_touched,
        )
    # A per-message DELETE with nullable vector predicates scans the populated
    # embedding table repeatedly. Delete known-obsolete rows in bounded groups,
    # inside this transaction and before inserting any replacement vectors.
    for batch in batched(obsolete_embedding_ids, 256):
        obsolete, params = id_set_predicate("message_id", batch)
        conn.execute(f"DELETE FROM message_embeddings WHERE {obsolete}", params)
    if emb_rows:
        insert_message_embeddings(conn, emb_rows)


def _upsert_tool_calls(
    conn: duckdb.DuckDBPyConnection,
    existing_rows: dict[str, tuple[object, ...]],
    desired_tool_calls: dict[str, ToolCall],
    *,
    sidecar_conn: sqlite3.Connection | None = None,
    fts_fields: tuple[str, ...] | None = None,
    sidecar_touched: FtsSidecarTouchedIds | None = None,
) -> None:
    new_tool_calls: list[ToolCall] = []
    changed_tool_calls: list[ToolCall] = []
    embedding_rows: list[tuple[str, list[float] | None]] = []
    stale_tc_ids: list[str] = []
    for tool_call_id, tool_call in desired_tool_calls.items():
        values = _tool_call_row_values(tool_call)
        existing_row = existing_rows.get(tool_call_id)
        if existing_row is None:
            new_tool_calls.append(tool_call)
            if tool_call.bash_embedding is not None:
                embedding_rows.append((tool_call_id, tool_call.bash_embedding))
            continue
        # Compare content fields only (first 15 of persisted row vs 15-wide values)
        if _canonicalize_tool_call_row(existing_row[:15]) == _canonicalize_tool_call_row(values):
            if (
                existing_row[15] != tool_call.bash_embedding
                and tool_call.bash_embedding is not None
            ):
                embedding_rows.append((tool_call_id, tool_call.bash_embedding))
            continue
        changed_tool_calls.append(tool_call)
        if tool_call.bash_embedding is not None:
            embedding_rows.append((tool_call_id, tool_call.bash_embedding))
        elif existing_row[6] != values[6] and existing_row[15] is not None:
            # bash_command changed, no new embedding — delete stale row
            stale_tc_ids.append(tool_call_id)
    _update_tool_calls(conn, changed_tool_calls, existing_rows)
    if changed_tool_calls:
        _upsert_tool_call_sidecar_rows(
            conn,
            sidecar_conn,
            changed_tool_calls,
            fts_fields=fts_fields,
            sidecar_touched=sidecar_touched,
        )
    if stale_tc_ids:
        stale, params = id_set_predicate("tool_call_id", stale_tc_ids)
        conn.execute(f"DELETE FROM tool_call_embeddings WHERE {stale}", params)
    if new_tool_calls:
        insert_tool_calls(conn, new_tool_calls)
        _upsert_tool_call_sidecar_rows(
            conn,
            sidecar_conn,
            new_tool_calls,
            fts_fields=fts_fields,
            sidecar_touched=sidecar_touched,
        )
    if embedding_rows:
        insert_tool_call_embeddings(conn, embedding_rows)


_TOOL_CALL_UPDATE_ASSIGNMENTS = (
    "session_id = incoming.session_id",
    "message_id = incoming.message_id",
    "idx = incoming.idx",
    "tool_name = incoming.tool_name",
    "tool_input = incoming.tool_input",
    "bash_command = incoming.bash_command",
    "bash_base = incoming.bash_base",
    "bash_sub = incoming.bash_sub",
    "is_compound = incoming.is_compound",
    "agent_id = incoming.agent_id",
    "subagent_type = incoming.subagent_type",
    "subagent_description = incoming.subagent_description",
    "subagent_model = incoming.subagent_model",
    "skill_name = incoming.skill_name",
)


def _update_tool_calls(
    conn: duckdb.DuckDBPyConnection,
    tool_calls: list[ToolCall],
    existing_rows: Mapping[str, tuple[object, ...]],
) -> None:
    if not tool_calls:
        return
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
        # The six indexed fields make a broad UPDATE a full-row replacement.
        # Exact masks keep payload-only changes on DuckDB's in-place update path.
        cohorts: dict[int, list[tuple[object, ...]]] = {}
        for tool_call in batch:
            values = _tool_call_row_values(tool_call)
            existing = _canonicalize_tool_call_row(existing_rows[tool_call.id][:15])
            desired = _canonicalize_tool_call_row(values)
            changed_columns = _changed_column_mask(existing[1:], desired[1:])
            if changed_columns:
                cohorts.setdefault(changed_columns, []).append(values)
        assert len(cohorts) <= 256
        for changed_columns, rows in sorted(cohorts.items()):
            table = pa.Table.from_arrays(
                [
                    pa.array(column, type=field.type)
                    for column, field in zip(zip(*rows, strict=True), schema, strict=True)
                ],
                schema=schema,
            )
            conn.register("_recall_tool_updates", table)
            try:
                assignments = _update_assignments(changed_columns, _TOOL_CALL_UPDATE_ASSIGNMENTS)
                conn.execute(
                    f"""
                UPDATE tool_calls
                SET {assignments}
                FROM _recall_tool_updates AS incoming
                WHERE tool_calls.id = incoming.id
                """,
                )
            finally:
                conn.unregister("_recall_tool_updates")


def _upsert_tool_call_sidecar_rows(
    conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection | None,
    tool_calls: Iterable[ToolCall],
    *,
    fts_fields: tuple[str, ...] | None = None,
    sidecar_touched: FtsSidecarTouchedIds | None = None,
) -> None:
    if sidecar_conn is None:
        return
    materialized = list(tool_calls)
    if not should_index_bash_fts(fts_fields):
        delete_ids = [tool_call.id for tool_call in materialized]
        if delete_ids:
            if sidecar_touched is not None:
                sidecar_touched.add_tool_calls(delete_ids)
            try:
                delete_tool_call_fts(sidecar_conn, delete_ids)
            except sqlite3.Error as err:
                logger.warning(
                    "sidecar delete failed for excluded bash tool calls: %s; "
                    "queueing for reconciliation",
                    err,
                )
                _enqueue_sidecar_pending_best_effort(conn, "tool_call", delete_ids, "upsert")
        return
    failed_ids: list[str] = []
    sidecar_rows = [(tool_call.id, tool_call.bash_command) for tool_call in materialized]
    if sidecar_rows:
        if sidecar_touched is not None:
            sidecar_touched.add_tool_calls(
                tool_call_id for tool_call_id, _bash_command in sidecar_rows
            )
        try:
            upsert_tool_call_fts_batch(sidecar_conn, sidecar_rows, fields=fts_fields)
        except sqlite3.Error as err:
            failed_ids = [tool_call_id for tool_call_id, _bash_command in sidecar_rows]
            logger.warning(
                "sidecar dual-write failed for %d tool_calls: %s; queueing for reconciliation",
                len(failed_ids),
                err,
            )
    if failed_ids:
        _enqueue_sidecar_pending_best_effort(conn, "tool_call", failed_ids, "upsert")


def _session_identity_changed(existing_row: tuple[object, ...], session: Session) -> bool:
    return _canonicalize_scalar_row(existing_row) != _canonicalize_scalar_row(
        _session_identity_values(session)
    )


def _rewrite_session_with_new_identity(
    conn: duckdb.DuckDBPyConnection,
    previous_rows: PersistedSessionRows,
    session: Session,
    *,
    context_text: str = "",
    context_mode: str = "off",
    sidecar_conn: sqlite3.Connection | None = None,
    fts_fields: tuple[str, ...] | None = None,
    sidecar_touched: FtsSidecarTouchedIds | None = None,
    tail_facts: TailFacts,
    on_commit: Callable[[], None] | None = None,
) -> None:
    _preserve_existing_embeddings(previous_rows, session, context_text=context_text)
    # One transaction, so a partial rewrite is undone by ROLLBACK. The previous
    # shape ran unwrapped and hand-rolled its own compensation -- delete the
    # session again, then re-insert the saved rows. That was lossy (it rebuilt
    # session_state from an explicit column list, silently dropping columns
    # added later) and it issued a second delete against an index the failed
    # attempt had already disturbed, which is a fault DuckDB answers by
    # invalidating the entire instance for the life of the process.
    conn.execute("BEGIN")
    try:
        if sidecar_conn is not None and sidecar_touched is not None:
            sidecar_touched.add_messages(str(row[0]) for row in previous_rows.message_rows)
            sidecar_touched.add_tool_calls(str(row[0]) for row in previous_rows.tool_call_rows)
        delete_session(conn, session.id, sidecar_conn=sidecar_conn)
        insert_session(conn, session)
        _insert_messages(
            conn,
            session.messages,
            context_text=context_text,
            context_mode=context_mode,
            sidecar_conn=sidecar_conn,
            fts_fields=fts_fields,
            sidecar_touched=sidecar_touched,
        )
        tool_calls = list(_collect_tool_calls(session))
        insert_tool_calls(conn, tool_calls)
        _upsert_tool_call_sidecar_rows(
            conn,
            sidecar_conn,
            tool_calls,
            fts_fields=fts_fields,
            sidecar_touched=sidecar_touched,
        )
        _write_tail_facts(conn, session.id, tool_calls, tail_facts)
        _insert_session_embeddings(conn, session.messages, tool_calls)
        if on_commit is not None:
            on_commit()
        conn.execute("COMMIT")
    except Exception as exc:
        _log_failed_transaction(
            exc, session_id=session.id, site="_rewrite_session_with_new_identity"
        )
        conn.execute("ROLLBACK")
        raise


def _preserve_existing_embeddings(
    previous_rows: PersistedSessionRows,
    session: Session,
    *,
    context_text: str = "",
) -> None:
    # Embeddings depend on effective text, not positional message/tool IDs.
    # Keep independent lookups for content and thinking: changing one does not
    # invalidate the other, and a prepended message does not change either input.
    content_vectors: dict[str, list[float]] = {}
    thinking_vectors: dict[str, list[float]] = {}
    for row in previous_rows.message_rows:
        prefix = str(row[9] or "")
        if row[4] and row[11] is not None:
            content_vectors[prefix + str(row[4])] = cast(list[float], row[11])
        if row[5] and row[12] is not None:
            thinking_vectors[prefix + str(row[5])] = cast(list[float], row[12])
    for message in session.messages:
        prefix, _ = _message_context(message, fallback_context_text=context_text)
        if message.content_embedding is None and message.content:
            message.content_embedding = content_vectors.get(prefix + message.content)
        if message.thinking_embedding is None and message.thinking:
            message.thinking_embedding = thinking_vectors.get(prefix + message.thinking)

    bash_vectors = {
        str(row[6]): cast(list[float], row[15])
        for row in previous_rows.tool_call_rows
        if row[6] and row[15] is not None
    }
    for call in _collect_tool_calls(session):
        if call.bash_embedding is None and call.bash_command:
            call.bash_embedding = bash_vectors.get(call.bash_command)


def _session_identity_values(session: Session) -> tuple[object, ...]:
    return (
        session.id,
        session.source.value,
        session.source_path,
        session.source_session_id,
    )


def _session_row_values(session: Session, *, last_byte_offset: int = 0) -> tuple[object, ...]:
    return (
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
        session.is_complete,
        session.file_mtime,
        session.file_size,
        last_byte_offset,
        session.indexed_at or datetime.now(UTC),
    )


def _message_row_values(message: Message) -> tuple[object, ...]:
    return (
        message.id,
        message.session_id,
        message.idx,
        message.role.value,
        message.content,
        message.thinking,
        message.timestamp,
        message.has_thinking,
        message.agent_id,
    )


def _tool_call_row_values(tool_call: ToolCall) -> tuple[object, ...]:
    return (
        tool_call.id,
        tool_call.session_id,
        tool_call.message_id,
        tool_call.idx,
        tool_call.tool_name,
        (
            json.dumps(tool_call.tool_input, sort_keys=True)
            if tool_call.tool_input is not None
            else None
        ),
        tool_call.bash_command,
        tool_call.bash_base,
        tool_call.bash_sub,
        tool_call.is_compound,
        tool_call.agent_id,
        tool_call.subagent_type,
        tool_call.subagent_description,
        tool_call.subagent_model,
        tool_call.skill_name,
    )


# Token columns are null-preserving in the session UPDATE
# (``CASE WHEN ? IS NULL THEN <stored> ELSE ? END``): a None here means "keep the
# stored value" — e.g. Grok, whose transcript carries no usage and whose token
# totals arrive only from harvest rollup — so a None must not count as a change.
_SESSION_NULL_PRESERVE_INDICES = frozenset({10, 11})  # input_tokens, output_tokens
# Number of leading columns the session UPDATE actually writes as a change
# signal. Index 16 (``indexed_at``) is stamped fresh on every parse and is only a
# COALESCE fallback for --since ordering, so it is excluded: an otherwise
# identical re-index must not rewrite the row just to bump it.
_SESSION_CHANGE_COLUMNS = 16


def _normalize_timestamp_for_compare(value: object) -> object:
    """Match a parsed timestamp to how DuckDB round-trips it, for equality.

    ``session_state`` timestamp columns are naive ``TIMESTAMP``: DuckDB stores a
    timezone-aware datetime as its *local* wall-clock with ``tzinfo`` dropped, so
    the value read back is naive-local. Parsers supply aware datetimes, so an
    aware value must be down-converted to local-naive before comparing — otherwise
    every real session (with a ``started_at``/``ended_at``) looks changed on a
    non-UTC host and the no-op guard never fires. Naive datetimes round-trip
    unchanged; non-datetimes pass through.
    """
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone().replace(tzinfo=None)
    return value


def _session_update_is_noop(existing_row: tuple[object, ...], values: tuple[object, ...]) -> bool:
    """Whether applying the session UPDATE would leave the row unchanged.

    ``existing_row`` is the wider persisted SELECT (includes ``cached_input_tokens``
    and ``host``, which the UPDATE does not touch); ``values`` is the writer tuple.
    Compares only the columns the UPDATE writes as a change signal, honoring the
    null-preserve token semantics and DuckDB's naive-timestamp round-trip, so a
    no-op re-index skips the rewrite.
    """
    existing = _canonicalize_scalar_row(existing_row)
    new = _canonicalize_scalar_row(values)
    for index in range(_SESSION_CHANGE_COLUMNS):
        new_value = _normalize_timestamp_for_compare(new[index])
        if index in _SESSION_NULL_PRESERVE_INDICES and new_value is None:
            continue
        if new_value != _normalize_timestamp_for_compare(existing[index]):
            return False
    return True


def _canonicalize_scalar_row(values: tuple[object, ...]) -> tuple[object, ...]:
    canonical: list[object] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            canonical.append(tuple(float(cast(float, item)) for item in value))
            continue
        canonical.append(value)
    return tuple(canonical)


def _canonicalize_tool_call_row(values: tuple[object, ...]) -> tuple[object, ...]:
    canonical = list(_canonicalize_scalar_row(values))
    canonical[5] = _normalize_json_value(values[5])
    return tuple(canonical)


def _invalidate_changed_message_embeddings(
    conn: duckdb.DuckDBPyConnection,
    message_id: str,
    existing_row: tuple[object, ...],
    new_values: tuple[object, ...],
    *,
    context_text: str = "",
) -> bool:
    """Clear changed fields; return whether the entire row needs batch deletion."""
    # Row indices: 4=content, 5=thinking, 9=context_text,
    # 11=content_embedding, 12=thinking_embedding.
    context_changed = (existing_row[9] or "") != context_text
    content_changed = existing_row[4] != new_values[4] or context_changed
    thinking_changed = existing_row[5] != new_values[5] or context_changed
    if not content_changed and not thinking_changed:
        return False
    # An all-NULL row also marks this input processed. Retire that marker when
    # the input changes so a later enrichment pass can select the new content.
    if (content_changed or existing_row[11] is None) and (
        thinking_changed or existing_row[12] is None
    ):
        return True
    updates: list[str] = []
    if content_changed and existing_row[11] is not None:
        updates.append("content_embedding = NULL")
    if thinking_changed and existing_row[12] is not None:
        updates.append("thinking_embedding = NULL")
    if not updates:
        return False
    conn.execute(
        f"UPDATE message_embeddings SET {', '.join(updates)} WHERE message_id = ?",
        [message_id],
    )
    return False


def _message_payload_signature(values: tuple[object, ...]) -> tuple[object, ...]:
    return _canonicalize_scalar_row(values[:8])


def _tool_call_payload_signature(values: tuple[object, ...]) -> tuple[object, ...]:
    return _canonicalize_tool_call_row((*values[:10], None))[:10]


def _normalize_json_value(value: object) -> str | None:
    if value is None:
        return None
    parsed = json.loads(value) if isinstance(value, str) else value
    return json.dumps(parsed, sort_keys=True)


def _searchable_rows_changed(
    previous_rows: PersistedSessionRows | None,
    session: Session,
    *,
    context_text: str = "",
    context_mode: str = "off",
) -> bool:
    if previous_rows is None:
        return _session_has_searchable_content(session)
    previous_signature = _persisted_searchable_signature(previous_rows)
    next_signature = _session_searchable_signature(
        session,
        context_text=context_text,
        context_mode=context_mode,
    )
    return previous_signature != next_signature


def _session_has_searchable_content(session: Session) -> bool:
    return any(_session_searchable_signature(session))


def _contextualized_message_count(
    session: Session,
    *,
    context_text: str = "",
    context_mode: str = "off",
) -> int:
    count = 0
    for message in session.messages:
        message_context_text, message_context_mode = _message_context(
            message,
            fallback_context_text=context_text,
            fallback_context_mode=context_mode,
        )
        if (
            message_context_mode != "off"
            and message_context_text
            and (message.content or message.thinking)
        ):
            count += 1
    return count


def _persisted_searchable_signature(rows: PersistedSessionRows) -> tuple[tuple[str, ...], ...]:
    message_signature = tuple(
        f"{row[0]!s}|{row[9] or ''}|{row[10] or 'off'}|{row[4] or ''}|{row[5] or ''}"
        for row in rows.message_rows
        if row[4] or row[5]
    )
    tool_signature = tuple(f"{row[0]!s}|{row[6] or ''}" for row in rows.tool_call_rows if row[6])
    return (message_signature, tool_signature)


def _session_searchable_signature(
    session: Session,
    *,
    context_text: str = "",
    context_mode: str = "off",
) -> tuple[tuple[str, ...], ...]:
    message_parts: list[str] = []
    for message in session.messages:
        if not message.content and not message.thinking:
            continue
        message_context_text, message_context_mode = _message_context(
            message,
            fallback_context_text=context_text,
            fallback_context_mode=context_mode,
        )
        message_parts.append(
            f"{message.id}|{message_context_text}|{message_context_mode}|"
            f"{message.content or ''}|{message.thinking or ''}"
        )
    message_signature = tuple(message_parts)
    tool_signature = tuple(
        f"{tool_call.id}|{tool_call.bash_command or ''}"
        for tool_call in _collect_tool_calls(session)
        if tool_call.bash_command
    )
    return (message_signature, tool_signature)


def _collect_tool_calls(session: Session) -> Iterable[ToolCall]:
    tool_calls: list[ToolCall] = []
    for message in session.messages:
        tool_calls.extend(message.tool_calls)
    tool_calls.extend(session.orphan_tool_calls)
    return tool_calls
