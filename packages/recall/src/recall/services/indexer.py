from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from itertools import batched
from pathlib import Path
from typing import Literal, NamedTuple, cast

import duckdb

from recall.core.config import AppConfig, ContextConfig, EmbeddingConfig, SourceConfig
from recall.core.daemon_status import IndexRunCounts
from recall.core.embeddings import EmbeddingBackend
from recall.core.models import Message, ParseResult, Session, TailFacts, ToolCall
from recall.core.telemetry import Stage, measured
from recall.core.transcripts import transcript_file, transcript_identity
from recall.core.types import Role, RunKind, Source
from recall.db import (
    advisory_lock,
    connect,
    create_fts_indexes,
)
from recall.db._payload import (
    _LLM_CONTEXT_MODES as _LLM_CONTEXT_MODES,
)
from recall.db._payload import (
    _MESSAGE_STATE_UPDATE_ASSIGNMENTS as _MESSAGE_STATE_UPDATE_ASSIGNMENTS,
)
from recall.db._payload import (
    _SESSION_CHANGE_INDICES as _SESSION_CHANGE_INDICES,
)
from recall.db._payload import (
    _SESSION_NULL_PRESERVE_INDICES as _SESSION_NULL_PRESERVE_INDICES,
)
from recall.db._payload import (
    _TOOL_CALL_UPDATE_ASSIGNMENTS as _TOOL_CALL_UPDATE_ASSIGNMENTS,
)
from recall.db._payload import (
    FtsSidecarTouchedIds as FtsSidecarTouchedIds,
)
from recall.db._payload import (
    PersistedSessionRows as PersistedSessionRows,
)
from recall.db._payload import (
    SessionWriteSnapshot as SessionWriteSnapshot,
)
from recall.db._payload import (
    StoredVector as StoredVector,
)
from recall.db._payload import (
    _canonicalize_scalar_row as _canonicalize_scalar_row,
)
from recall.db._payload import (
    _canonicalize_tool_call_row as _canonicalize_tool_call_row,
)
from recall.db._payload import (
    _changed_column_mask as _changed_column_mask,
)
from recall.db._payload import (
    _collect_tool_calls as _collect_tool_calls,
)
from recall.db._payload import (
    _delete_removed_rows as _delete_removed_rows,
)
from recall.db._payload import (
    _embedding_input_changes as _embedding_input_changes,
)
from recall.db._payload import (
    _embedding_inputs_changed as _embedding_inputs_changed,
)
from recall.db._payload import (
    _enqueue_owned_sidecar_touched_after_rollback as _enqueue_owned_sidecar_touched_after_rollback,
)
from recall.db._payload import (
    _enqueue_sidecar_pending_best_effort as _enqueue_sidecar_pending_best_effort,
)
from recall.db._payload import (
    _enqueue_touched_sidecar_ids_for_reconcile as _enqueue_touched_sidecar_ids_for_reconcile,
)
from recall.db._payload import (
    _insert_messages as _insert_messages,
)
from recall.db._payload import (
    _insert_new_session as _insert_new_session,
)
from recall.db._payload import (
    _insert_session_embeddings as _insert_session_embeddings,
)
from recall.db._payload import (
    _invalidate_changed_message_embeddings as _invalidate_changed_message_embeddings,
)
from recall.db._payload import (
    _load_persisted_message_rows as _load_persisted_message_rows,
)
from recall.db._payload import (
    _load_persisted_session_rows as _load_persisted_session_rows,
)
from recall.db._payload import (
    _load_persisted_tool_call_rows as _load_persisted_tool_call_rows,
)
from recall.db._payload import (
    _load_session_write_snapshot as _load_session_write_snapshot,
)
from recall.db._payload import (
    _log_failed_transaction as _log_failed_transaction,
)
from recall.db._payload import (
    _message_context as _message_context,
)
from recall.db._payload import (
    _message_row_values as _message_row_values,
)
from recall.db._payload import (
    _message_state_change_mask as _message_state_change_mask,
)
from recall.db._payload import (
    _MessageStateUpdate as _MessageStateUpdate,
)
from recall.db._payload import (
    _normalize_json_value as _normalize_json_value,
)
from recall.db._payload import (
    _normalize_session_identity as _normalize_session_identity,
)
from recall.db._payload import (
    _normalize_timestamp_for_compare as _normalize_timestamp_for_compare,
)
from recall.db._payload import (
    _persisted_searchable_signature as _persisted_searchable_signature,
)
from recall.db._payload import (
    _preserve_existing_embeddings as _preserve_existing_embeddings,
)
from recall.db._payload import (
    _read_stored_vectors as _read_stored_vectors,
)
from recall.db._payload import (
    _same_vectors as _same_vectors,
)
from recall.db._payload import (
    _searchable_rows_changed as _searchable_rows_changed,
)
from recall.db._payload import (
    _session_has_searchable_content as _session_has_searchable_content,
)
from recall.db._payload import (
    _session_identity_changed as _session_identity_changed,
)
from recall.db._payload import (
    _session_identity_values as _session_identity_values,
)
from recall.db._payload import (
    _session_row_values as _session_row_values,
)
from recall.db._payload import (
    _session_searchable_signature as _session_searchable_signature,
)
from recall.db._payload import (
    _session_update_is_noop as _session_update_is_noop,
)
from recall.db._payload import (
    _SessionIndexedFields as _SessionIndexedFields,
)
from recall.db._payload import (
    _set_session_host as _set_session_host,
)
from recall.db._payload import (
    _sidecar_touched_accumulator as _sidecar_touched_accumulator,
)
from recall.db._payload import (
    _sync_existing_session as _sync_existing_session,
)
from recall.db._payload import (
    _tool_call_row_values as _tool_call_row_values,
)
from recall.db._payload import (
    _update_assignments as _update_assignments,
)
from recall.db._payload import (
    _update_message_states as _update_message_states,
)
from recall.db._payload import (
    _update_session_row as _update_session_row,
)
from recall.db._payload import (
    _update_tool_calls as _update_tool_calls,
)
from recall.db._payload import (
    _upsert_messages as _upsert_messages,
)
from recall.db._payload import (
    _upsert_tool_call_sidecar_rows as _upsert_tool_call_sidecar_rows,
)
from recall.db._payload import (
    _upsert_tool_calls as _upsert_tool_calls,
)
from recall.db._payload import (
    _with_read_vectors as _with_read_vectors,
)
from recall.db._payload import (
    _write_tail_facts as _write_tail_facts,
)
from recall.db.fatal import is_fatal_db_invalidation
from recall.db.fts_sidecar import (
    upsert_message_fts_batch,
)
from recall.db.semantic_work import RawAppend, RawInsert, RawRewrite, publication
from recall.parsers import SessionParser, all_parsers, get_parser
from recall.services.context import resolve_message_contexts
from recall.services.context_backends import ContextBackend, ensure_context_backend
from recall.services.runtime_state import record_run_attempt, record_run_failure, record_run_success

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


@measured(Stage.CONTEXT)
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
    snapshot: SessionWriteSnapshot | None = None,
) -> SessionContextStats:
    if is_full_parse:
        # Re-parsing is not re-summarizing.  Message ids are deterministic and
        # transcripts are append-only, so a stored LLM summary is still valid;
        # re-earning it costs a model call per message, and resolving it under
        # a cheaper mode would overwrite it with nothing (REQ-INDEX-021).
        reused = _apply_stored_contexts(conn, session, snapshot=snapshot)
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


def _apply_stored_contexts(
    conn: duckdb.DuckDBPyConnection,
    session: Session,
    *,
    snapshot: SessionWriteSnapshot | None = None,
) -> int:
    """Stamp previously stored LLM context onto freshly parsed messages.

    Only `llm-*` context is carried forward: it costs a model call to produce
    and is the value a re-parse would otherwise destroy.  Template prefixes
    are derived and free to rebuild, so they stay out of this and refresh
    with the session metadata they are built from.

    Legacy contexts have no saved dependency fingerprint, so the stored
    normalized conversation, tools and their association included, must be
    exactly a prefix of the parsed one. An append keeps every stored context,
    as the suffix write and a moved transcript's carry-over do (REQ-CTX-004,
    REQ-INDEX-027). The one stored message a later record may still complete
    is the last: a step stored mid-stream is amended in place by the records
    after it. When only that message differs, every earlier context is kept
    and the amended message is rendered again. A change anywhere earlier keeps
    none.

    Returns how many messages kept a stored context.
    """
    if not session.messages:
        return 0
    previous_rows = (
        snapshot.rows if snapshot is not None else _load_persisted_session_rows(conn, session.id)
    )
    if previous_rows is None:
        return 0
    if not any(row[10] in _LLM_CONTEXT_MODES for row in previous_rows.message_rows):
        return 0
    previous = _stored_contexts(previous_rows)
    parsed = _context_inputs(session)
    if _is_context_prefix(previous.inputs, parsed):
        return _stamp_message_contexts(previous.contexts, session)
    if _completes_last_message(previous.inputs, parsed):
        assert previous.last_message_id is not None
        previous.contexts.pop(previous.last_message_id, None)
        return _stamp_message_contexts(previous.contexts, session)
    return 0


@dataclass(frozen=True)
class _StoredContexts:
    inputs: tuple[object, ...]
    contexts: dict[str | int, tuple[str, str]]
    last_message_id: str | None


def _stored_contexts(previous: PersistedSessionRows) -> _StoredContexts:
    """Project the writer's stored rows onto the exact context dependency proof.

    The same snapshot supplies context reuse and the subsequent changed-column
    comparison. Vector references stay unread and no message models are built.
    """
    rows = previous.message_rows
    tools: dict[str | None, list[tuple[object, ...]]] = {str(row[0]): [] for row in rows}
    tools[None] = []
    for call in previous.tool_call_rows:
        # Calls whose message companion is absent are orphans, as in show.
        tools.get(cast(str | None, call[2]), tools[None]).append(
            (call[3], call[4], _normalize_json_value(call[5]), call[6])
        )
    return _StoredContexts(
        inputs=(
            previous.identity_row[1],
            tuple(
                (row[2], row[3], row[4], row[5], row[8], tuple(tools[str(row[0])])) for row in rows
            ),
            tuple(tools[None]),
        ),
        contexts={
            str(row[0]): (cast(str, row[9]), cast(str, row[10]))
            for row in rows
            if row[10] in _LLM_CONTEXT_MODES
        },
        last_message_id=str(rows[-1][0]) if rows else None,
    )


def _completes_last_message(stored: tuple[object, ...], parsed: tuple[object, ...]) -> bool:
    """Whether `parsed` is `stored` with its last message only extended, then appended to.

    Extended means what a step still streaming gains: text that grows from what
    was stored, and tool calls after the stored ones. Any other difference in
    that message, such as an edited text or a changed tool input, is a change.
    """
    source, messages, orphans = stored
    _, parsed_messages, _ = parsed
    assert isinstance(messages, tuple) and isinstance(parsed_messages, tuple)
    if not messages or len(parsed_messages) < len(messages):
        return False
    # Everything before it, and the orphan calls, must already be an exact prefix.
    if not _is_context_prefix((source, messages[:-1], orphans), parsed):
        return False
    idx, role, content, thinking, agent_id, tools = messages[-1]
    new_idx, new_role, new_content, new_thinking, new_agent_id, new_tools = parsed_messages[
        len(messages) - 1
    ]
    assert isinstance(tools, tuple) and isinstance(new_tools, tuple)

    def grows(old: object, new: object) -> bool:
        return old is None or (
            isinstance(old, str) and isinstance(new, str) and new.startswith(old)
        )

    return (
        (idx, role, agent_id) == (new_idx, new_role, new_agent_id)
        and grows(content, new_content)
        and grows(thinking, new_thinking)
        and new_tools[: len(tools)] == tools
    )


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


def _stamp_contexts(
    previous: Session,
    session: Session,
    *,
    by_position: bool = False,
    except_ids: frozenset[str] = frozenset(),
) -> int:
    """Stamp `previous`'s stored `llm-*` contexts onto matching messages of `session`."""
    stored = {
        (message.idx if by_position else message.id): (message.context_text, message.context_mode)
        for message in previous.messages
        if message.context_mode in _LLM_CONTEXT_MODES and message.id not in except_ids
    }
    return _stamp_message_contexts(stored, session, by_position=by_position)


def _stamp_message_contexts(
    stored: Mapping[str | int, tuple[str, str]], session: Session, *, by_position: bool = False
) -> int:
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
                        _normalize_session_identity(conn, result.session)
                        snapshot = (
                            _load_session_write_snapshot(
                                conn,
                                result.session,
                                host=resolved_host,
                                indexed_bytes=result.next_byte_offset,
                            )
                            if result.is_full_parse
                            else None
                        )
                        _require_committable_capture(
                            discovered.path,
                            result,
                            has_indexed_history=(snapshot is not None and snapshot.rows is not None)
                            or _load_session_state(conn, discovered.resolved_path) is not None,
                            conn=conn,
                            snapshot=snapshot,
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
                            snapshot=snapshot,
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
                                snapshot=snapshot,
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
                    HarvestResult,
                    default_grok_unified_log,
                    harvest_calls_to_cover,
                    harvest_grok_unified_log,
                )

                log_path = default_grok_unified_log(home=home_root)

                def harvest_next() -> HarvestResult:
                    harvest = harvest_grok_unified_log(conn, log_path, host=resolved_host)
                    if harvest.events_upserted or harvest.rotated:
                        logger.info(
                            "usage harvest path=%s upserted=%s rolled_up=%s rotated=%s",
                            harvest.path,
                            harvest.events_upserted,
                            harvest.sessions_rolled_up,
                            harvest.rotated,
                        )
                    return harvest

                harvest = harvest_next()
                for _ in range(harvest_calls_to_cover(harvest.log_size_bytes) - 1):
                    if harvest.caught_up:
                        break
                    harvest = harvest_next()
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
            from recall.db.enrichment_inputs import (
                GeneratedEmbedSession,
                read_enrichment_session,
                read_target_masks,
                snapshot_embed_session,
                source_generation,
            )

            prepared_publications = []
            updates: list[tuple[str, str, str, str, str]] = []
            for group in groups:
                if not group.targets:
                    continue
                snapshots = [
                    snapshot_embed_session(
                        read_enrichment_session(
                            conn,
                            group.session.id,
                            read_target_masks(
                                conn, tuple(message.id for message in chunk), (), replace=True
                            ),
                            full_context=True,
                        ),
                        context_targets=tuple(message.id for message in chunk),
                        targets=read_target_masks(
                            conn, tuple(message.id for message in chunk), (), replace=True
                        ),
                        full_context=True,
                        publication="replace",
                        generation=source_generation(conn, group.session.id),
                    )
                    for chunk in batched(group.targets, 128)
                ]
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
                by_id = {message.id: message for message in group.targets}
                for snapshot in snapshots:
                    generated = group.session.model_copy(
                        update={
                            "messages": [
                                by_id[identifier].model_copy(update={"tool_calls": []})
                                for identifier in snapshot.targets.message_ids
                            ],
                            "orphan_tool_calls": [],
                        }
                    )
                    prepared_publications.append(GeneratedEmbedSession(snapshot, generated))

            if selected_count:
                for generated in prepared_publications:
                    if publication(conn).enrich(generated, embed=embed) is None:
                        raise RuntimeError(
                            "context inputs changed during generation; recomputation remains "
                            "pending"
                        )
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
                stat = transcript_file(path).stat()
                resolved_path = str(transcript_identity(path))
            except OSError:
                continue
            paths.append(
                DiscoveredSessionPath(
                    parser=parser,
                    path=path,
                    resolved_path=resolved_path,
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
        with transcript_file(path).open("rb") as source:
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
    conn: duckdb.DuckDBPyConnection,
    snapshot: SessionWriteSnapshot | None = None,
) -> bool:
    """Reject a result that cannot safely replace an indexed session.

    Returns whether the result is a semantic rewrite: a full parse that drops
    or changes an indexed fact, which a live cursor must be told about
    (REQ-LIVE-004). The capture digest and the history comparison are each one
    pass inside the writer, so the publisher calls this once, immediately
    before its write, and acknowledges with the verdict.

    A full parse is a replacement operation.  Publishing a parse that stopped
    at a malformed or oversized record would delete the valid rows after that
    point, so the writer leaves the last known-good session intact and the
    source stays retryable with its diagnostics.

    A torn tail or an unsupported record does not stop the parse: the first
    was never yielded, the second was skipped, and every complete record
    around them is normalized as a clean parse would.  Neither may shorten the
    indexed history.  A torn tail must also leave every indexed fact
    unchanged, because the record it cut can still amend them once finished.
    An unsupported record is skipped the same way by every parse of these
    bytes, so an amendment the complete records make (a later record closing
    an open message) is the reference result and publishes as a rewrite.

    A verified suffix appends after the prefix it was proven against, so it
    can neither drop an indexed fact nor be a rewrite; it carries no
    diagnostic but an unterminated tail, which it never acknowledged.
    """
    torn = any(diagnostic.is_unterminated_tail for diagnostic in result.diagnostics)
    unsupported = any(d.kind == "unsupported_record" for d in result.diagnostics)
    stopping = dict.fromkeys(
        diagnostic.kind
        for diagnostic in result.diagnostics
        if not diagnostic.is_unterminated_tail and diagnostic.kind != "unsupported_record"
    )
    rewrites_history = False
    if not result.is_full_parse:
        if unsupported or stopping:
            kinds = ", ".join(dict.fromkeys(d.kind for d in result.diagnostics))
            raise RuntimeError(f"a resumed suffix cannot carry a diagnosed record: {kinds}")
    elif has_indexed_history:
        if stopping:
            raise RuntimeError(
                f"parser result is incomplete and cannot replace history: {', '.join(stopping)}"
            )
        # One comparison decides whether a diagnosed parse may advance and
        # names a semantic rewrite.
        history = _compare_indexed_history(conn, result.session, snapshot=snapshot)
        if torn and history is not IndexedHistory.PRESERVED:
            raise RuntimeError(
                "parser result ends in an unterminated record and would change indexed history"
            )
        if unsupported and history is IndexedHistory.DROPPED:
            raise RuntimeError(
                "parser result skips unsupported records and would drop indexed history"
            )
        rewrites_history = history is not IndexedHistory.PRESERVED
    # Last, so the bytes the verdict describes are the ones on disk as the
    # write begins, however long the history comparison took.
    if not _capture_still_matches(path, result):
        raise RuntimeError("source changed while parser capture was in flight")
    return rewrites_history


_PREFIX_BATCH_ROWS = 256


class _PrefixMessage(NamedTuple):
    id: str
    session_id: str
    idx: int
    role: str
    content: str | None
    thinking: str | None
    timestamp: datetime | None
    has_thinking: bool
    agent_id: str | None

    def matches(self, message: Message) -> bool:
        return (
            self.id == message.id
            and self.session_id == message.session_id
            and self.idx == message.idx
            and self.role == message.role.value
            and self.content == message.content
            and self.thinking == message.thinking
            and _normalize_timestamp_for_compare(self.timestamp)
            == _normalize_timestamp_for_compare(message.timestamp)
            and self.has_thinking == message.has_thinking
            and self.agent_id == message.agent_id
        )


class _PrefixToolCall(NamedTuple):
    id: str
    session_id: str
    message_id: str | None
    idx: int
    tool_name: str
    tool_input: str | None
    bash_command: str | None
    bash_base: str | None
    bash_sub: str | None
    is_compound: bool

    def matches(self, call: ToolCall) -> bool:
        desired_input = _normalize_json_value(call.tool_input)
        return (
            self.id == call.id
            and self.session_id == call.session_id
            and self.message_id == call.message_id
            and self.idx == call.idx
            and self.tool_name == call.tool_name
            and (
                self.tool_input == desired_input
                or _normalize_json_value(self.tool_input) == desired_input
            )
            and self.bash_command == call.bash_command
            and self.bash_base == call.bash_base
            and self.bash_sub == call.bash_sub
            and self.is_compound == call.is_compound
        )


def _iter_prefix_messages(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    *,
    snapshot: SessionWriteSnapshot | None = None,
) -> Generator[_PrefixMessage, None, None]:
    if snapshot is not None:
        if snapshot.rows is not None:
            yield from (_PrefixMessage._make(row[:9]) for row in snapshot.rows.message_rows)
        return
    # Read identities and state in one statement: executing another statement
    # on this connection would truncate an active relation's result stream.
    # Keep the caller's transaction and fetch payloads in bounded batches.
    with closing(
        conn.sql(
            """SELECT m.id, m.session_id, m.idx, s.role, s.content, s.thinking,
                      s.timestamp, s.has_thinking, m.agent_id
               FROM messages m JOIN message_state s ON s.message_id = m.id
               WHERE m.session_id = ? ORDER BY m.idx""",
            params=[session_id],
        )
    ) as messages:
        while group := messages.fetchmany(_PREFIX_BATCH_ROWS):
            yield from (_PrefixMessage(*row) for row in group)


def _iter_prefix_tool_calls(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    *,
    snapshot: SessionWriteSnapshot | None = None,
) -> Generator[_PrefixToolCall, None, None]:
    if snapshot is not None:
        if snapshot.rows is not None:
            yield from (_PrefixToolCall._make(row[:10]) for row in snapshot.rows.tool_call_rows)
        return
    with closing(
        conn.sql(
            """SELECT id, session_id, message_id, idx, tool_name, tool_input,
                  bash_command, bash_base, bash_sub, is_compound
           FROM tool_calls WHERE session_id = ? ORDER BY idx""",
            params=[session_id],
        )
    ) as calls:
        while group := calls.fetchmany(_PREFIX_BATCH_ROWS):
            yield from (_PrefixToolCall(*row) for row in group)


class IndexedHistory(Enum):
    """How a full parse relates to the session history already indexed."""

    PRESERVED = "preserved"  # every indexed fact is unchanged
    AMENDED = "amended"  # every indexed fact is kept, some with new values
    DROPPED = "dropped"  # an indexed fact, or the session itself, is gone


def _compare_indexed_history(
    conn: duckdb.DuckDBPyConnection,
    session: Session,
    *,
    snapshot: SessionWriteSnapshot | None = None,
) -> IndexedHistory:
    """Classify a full parse against the indexed messages and tool calls."""
    if (
        conn.execute(
            "SELECT 1 FROM sessions s JOIN session_state ss ON ss.session_id = s.id WHERE s.id = ?",
            [session.id],
        ).fetchone()
        is None
    ):
        return IndexedHistory.DROPPED
    verdict = IndexedHistory.PRESERVED
    desired_messages = {message.id: message for message in session.messages}
    with closing(_iter_prefix_messages(conn, session.id, snapshot=snapshot)) as messages:
        for message in messages:
            desired = desired_messages.get(message.id)
            if desired is None:
                return IndexedHistory.DROPPED
            if not message.matches(desired):
                verdict = IndexedHistory.AMENDED
    desired_calls = {call.id: call for call in _collect_tool_calls(session)}
    with closing(_iter_prefix_tool_calls(conn, session.id, snapshot=snapshot)) as calls:
        for call in calls:
            desired_call = desired_calls.get(call.id)
            if desired_call is None:
                return IndexedHistory.DROPPED
            if not call.matches(desired_call):
                verdict = IndexedHistory.AMENDED
    return verdict


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
    snapshot: SessionWriteSnapshot | None = None,
) -> bool:
    session_sidecar_touched, owns_sidecar_touched = _sidecar_touched_accumulator(
        sidecar_conn,
        sidecar_touched,
    )
    if snapshot is None:
        snapshot = _load_session_write_snapshot(
            conn, session, host=host, indexed_bytes=last_byte_offset
        )
    previous_rows = snapshot.rows
    try:
        if previous_rows is not None:
            result = publication(conn).raw(
                RawRewrite(session, previous_rows),
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
            publication(conn).raw(
                RawInsert(session),
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


# Token columns are null-preserving in the session UPDATE
# (``CASE WHEN ? IS NULL THEN <stored> ELSE ? END``): a None here means "keep the
# stored value" — e.g. Grok, whose transcript carries no usage and whose token
# totals arrive only from harvest rollup — so a None must not count as a change.
# input_tokens, output_tokens, cached_input_tokens
# Columns the session UPDATE writes as a change signal. Index 16 (``indexed_at``)
# is stamped fresh on every parse and is only a COALESCE fallback for --since
# ordering, so it is excluded: an otherwise identical re-index must not rewrite
# the row just to bump it.


def _message_payload_signature(values: tuple[object, ...]) -> tuple[object, ...]:
    return _canonicalize_scalar_row(values[:8])


def _tool_call_payload_signature(values: tuple[object, ...]) -> tuple[object, ...]:
    return _canonicalize_tool_call_row((*values[:10], None))[:10]


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


def _incremental_write_session(conn: duckdb.DuckDBPyConnection, session: Session, **options):
    return publication(conn).raw(RawAppend(session), **options)
