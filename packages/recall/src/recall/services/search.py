from __future__ import annotations

import dataclasses
import logging
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import duckdb

from recall.core.config import AppConfig
from recall.core.embeddings import EmbeddingBackend
from recall.core.fts_text import sql_is_non_blank
from recall.core.read_cancellation import check_read_cancelled
from recall.core.snippets import build_search_snippet
from recall.core.time import format_utc_iso, to_naive_utc
from recall.core.types import UNATTRIBUTED_HOST, Role, SearchMode, Source
from recall.db import FtsSidecarUnavailableError, load_fts_extension
from recall.db.fts_sidecar import (
    KeywordIndexFailure,
    is_lock_contention,
    open_sidecar,
    open_sidecar_readonly,
    prepare_candidate_ids,
    search_messages_fts,
    search_tool_calls_fts,
    sidecar_path,
)
from recall.db.lease import DbLease
from recall.db.ownership import standalone_read_lease
from recall.db.read_snapshot import SCHEMA_MISMATCH_ERRORS
from recall.services.fts_sidecar_reconcile import iter_tool_call_inputs_bounded
from recall.services.session_ids import resolve_session_identifier

logger = logging.getLogger("recall.search")

_SIDECAR_CANDIDATE_BATCH_SIZE = 500
_SIDECAR_FILTER_BATCH_SIZE = 100
_SIDECAR_SELECTIVE_FRACTION = 0.1
MIN_VECTOR_SIMILARITY = 0.50
STRONG_VECTOR_SIMILARITY = 0.75
LEXICAL_RRF_WEIGHT = 3.0
SYSTEM_ROLE_QUALITY_MULTIPLIER = 0.75
_KEYWORD_OVERSAMPLE_FACTOR = 3


class KeywordIndexLockedError(RuntimeError):
    """A keyword read met another connection's lock on the keyword index."""

    def __init__(self, failure: KeywordIndexFailure) -> None:
        super().__init__(f"search failed: {failure}")
        self.failure = failure


@dataclass(frozen=True)
class KeywordAccess:
    open: Callable[[Path], sqlite3.Connection]
    load_legacy: Callable[[duckdb.DuckDBPyConnection], None]


def _open_or_initialize_sidecar(path: Path) -> sqlite3.Connection:
    if not path.exists():
        # Library searches have always served an empty index before its first write.
        open_sidecar(path).close()
    return open_sidecar_readonly(path)


_COMPAT_KEYWORD_ACCESS = KeywordAccess(
    lambda path: _open_or_initialize_sidecar(path), lambda conn: load_fts_extension(conn)
)
# The daemon creates its index only at startup (REQ-FTS-SIDECAR-018).
DAEMON_KEYWORD_ACCESS = KeywordAccess(
    lambda path: open_sidecar_readonly(path), lambda conn: load_fts_extension(conn)
)


@dataclass(frozen=True)
class SearchResult:
    kind: Literal["message", "tool_call"]
    session_id: str
    source: str
    source_path: str | None
    score: float
    message_id: str | None
    tool_call_id: str | None
    role: str | None
    content: str | None
    thinking: str | None
    timestamp: str | None
    tool_name: str | None
    bash_command: str | None
    # REQ-CLI-017: True when this row matched the lexical (BM25/FTS) leg of the
    # query, False for vector-only (semantic) neighbors. Defaults False so the
    # vector constructors and any external callers get the conservative value;
    # the keyword/FTS leaf constructors set it True explicitly. Hybrid fusion
    # carries it through `result_map`, which prefers the BM25 row on collision.
    lexical_match: bool = False
    # REQ-HOST-API-003: stamped from session_state after the hit set is built.
    host: str = UNATTRIBUTED_HOST
    snippet: str | None = None


def search(
    *,
    query: str,
    source: Source | None,
    tool: str | None,
    session: str | None = None,
    roles: tuple[Role, ...] | None = None,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
    per_session: int | None = None,
    limit: int = 20,
    mode: SearchMode = SearchMode.AUTO,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
    embed_backend: EmbeddingBackend | None = None,
    embed_backend_factory: Callable[[], EmbeddingBackend] | None = None,
    query_embedding: list[float] | None = None,
    keyword_access: KeywordAccess = _COMPAT_KEYWORD_ACCESS,
) -> list[SearchResult]:
    config = config or AppConfig.load()
    with standalone_read_lease(config, conn) as lease:
        return search_on(
            lease,
            query=query,
            source=source,
            tool=tool,
            session=session,
            roles=roles,
            project=project,
            since=since,
            host=host,
            per_session=per_session,
            limit=limit,
            mode=mode,
            config=config,
            embed_backend=embed_backend,
            embed_backend_factory=embed_backend_factory,
            query_embedding=query_embedding,
            keyword_access=keyword_access,
        )


def search_on(
    lease: DbLease,
    *,
    query: str,
    source: Source | None,
    tool: str | None,
    session: str | None = None,
    roles: tuple[Role, ...] | None = None,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
    per_session: int | None = None,
    limit: int = 20,
    mode: SearchMode = SearchMode.AUTO,
    config: AppConfig,
    embed_backend: EmbeddingBackend | None = None,
    embed_backend_factory: Callable[[], EmbeddingBackend] | None = None,
    query_embedding: list[float] | None = None,
    keyword_access: KeywordAccess = _COMPAT_KEYWORD_ACCESS,
) -> list[SearchResult]:
    return lease.call(
        "search:search",
        lambda conn: _search(
            lease,
            conn,
            query=query,
            source=source,
            tool=tool,
            session=session,
            roles=roles,
            project=project,
            since=since,
            host=host,
            per_session=per_session,
            limit=limit,
            mode=mode,
            config=config,
            embed_backend=embed_backend,
            embed_backend_factory=embed_backend_factory,
            query_embedding=query_embedding,
            keyword_access=keyword_access,
        ),
    )


def _search(
    lease: DbLease,
    conn: duckdb.DuckDBPyConnection,
    *,
    query: str,
    source: Source | None,
    tool: str | None,
    session: str | None = None,
    roles: tuple[Role, ...] | None = None,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
    per_session: int | None = None,
    limit: int = 20,
    mode: SearchMode = SearchMode.AUTO,
    config: AppConfig,
    embed_backend: EmbeddingBackend | None = None,
    embed_backend_factory: Callable[[], EmbeddingBackend] | None = None,
    query_embedding: list[float] | None = None,
    keyword_access: KeywordAccess = _COMPAT_KEYWORD_ACCESS,
) -> list[SearchResult]:
    if tool is not None and roles:
        raise ValueError("--role cannot be combined with --tool")
    if per_session is not None and (isinstance(per_session, bool) or per_session <= 0):
        raise ValueError("per_session must be a positive integer")
    if since is not None:
        since = to_naive_utc(since)
    if session is not None:
        session = resolve_session_identifier(
            conn,
            session,
            source=source.value if source is not None else None,
            host=host,
        ).id
    if tool:
        return _finalize_results(
            lease,
            conn,
            _mark_lexical(
                _run_keyword_search(
                    lease,
                    config,
                    conn,
                    lambda: _search_tool_calls(
                        conn,
                        query,
                        source,
                        tool,
                        limit,
                        session=session,
                        project=project,
                        since=since,
                        host=host,
                        per_session=per_session,
                        config=config,
                        keyword_access=keyword_access,
                    ),
                    failure_prefix="search failed",
                    keyword_access=keyword_access,
                )
            ),
            query=query,
            fields=config.fts.fields,
        )

    # Only AUTO needs a capability probe. Concrete keyword requests never
    # touch model discovery, and concrete vector modes report failures.
    backend_available = True
    if mode == SearchMode.AUTO:
        backend_available = _probe_backend(
            embed_backend=embed_backend,
            embed_backend_factory=embed_backend_factory,
            config=config,
        )
    effective_mode = _resolve_mode(
        conn,
        mode,
        embed_backend_available=backend_available,
    )

    if effective_mode == SearchMode.KEYWORD:
        return _finalize_results(
            lease,
            conn,
            _run_keyword_search(
                lease,
                config,
                conn,
                lambda: _search_all(
                    conn,
                    query,
                    source,
                    limit,
                    config.fts.fields,
                    session=session,
                    roles=roles,
                    project=project,
                    since=since,
                    host=host,
                    per_session=per_session,
                    config=config,
                    keyword_access=keyword_access,
                ),
                failure_prefix="search failed",
                keyword_access=keyword_access,
            ),
            query=query,
            fields=config.fts.fields,
        )

    # Vector or hybrid — resolve the backend and embed the query
    if query_embedding is None:
        try:
            if embed_backend is not None:
                backend = embed_backend
            elif embed_backend_factory is not None:
                backend = embed_backend_factory()
            else:
                from recall.services.embeddings import get_backend

                backend = get_backend(config.embedding)
            query_with_prefix = backend.query_prefix + query
            query_embedding = backend.embed([query_with_prefix])[0]
        except (ValueError, ImportError, OSError, RuntimeError) as err:
            if lease.observe(err, "search:query_embedding"):
                raise
            if mode != SearchMode.AUTO:
                raise RuntimeError(f"embedding backend unavailable: {err}") from err
            # AUTO remains truthful and useful when model acquisition fails.
            logger.warning("embedding backend unavailable, falling back to keyword: %s", err)
            return _finalize_results(
                lease,
                conn,
                _run_keyword_search(
                    lease,
                    config,
                    conn,
                    lambda: _search_all(
                        conn,
                        query,
                        source,
                        limit,
                        config.fts.fields,
                        session=session,
                        roles=roles,
                        project=project,
                        since=since,
                        host=host,
                        per_session=per_session,
                        config=config,
                        keyword_access=keyword_access,
                    ),
                    failure_prefix="search failed",
                    keyword_access=keyword_access,
                ),
                query=query,
                fields=config.fts.fields,
            )

    if effective_mode == SearchMode.VECTOR:
        return _finalize_results(
            lease,
            conn,
            _vector_search_all(
                conn,
                query_embedding,
                source,
                limit,
                session=session,
                roles=roles,
                project=project,
                since=since,
                host=host,
                per_session=per_session,
            ),
            query=query,
            fields=config.fts.fields,
        )

    # Hybrid: RRF fusion of BM25 + vector
    return _finalize_results(
        lease,
        conn,
        _run_keyword_search(
            lease,
            config,
            conn,
            lambda: _hybrid_search(
                conn,
                query,
                query_embedding,
                source,
                limit,
                config.fts.fields,
                session=session,
                roles=roles,
                project=project,
                since=since,
                host=host,
                per_session=per_session,
                config=config,
                keyword_access=keyword_access,
            ),
            failure_prefix="hybrid search failed",
            keyword_access=keyword_access,
        ),
        query=query,
        fields=config.fts.fields,
    )


def _finalize_results(
    lease: DbLease,
    conn: duckdb.DuckDBPyConnection,
    results: list[SearchResult],
    *,
    query: str,
    fields: tuple[str, ...],
) -> list[SearchResult]:
    with_hosts = _attach_hosts(lease, conn, results)
    return _attach_snippets_streamed(
        lease,
        conn,
        with_hosts,
        query=query,
        fields=fields,
    )


def _attach_snippets_streamed(
    lease: DbLease,
    conn: duckdb.DuckDBPyConnection,
    results: list[SearchResult],
    *,
    query: str,
    fields: tuple[str, ...],
) -> list[SearchResult]:
    """Attach snippets while consuming non-bash `tool_input` JSON one bounded chunk at a time."""
    if not results:
        return results
    non_bash_indices: dict[str, list[int]] = {}
    for index, result in enumerate(results):
        if result.kind == "tool_call" and result.bash_command is None and result.tool_call_id:
            non_bash_indices.setdefault(result.tool_call_id, []).append(index)

    precomputed: dict[int, str] = {}
    if non_bash_indices:
        try:
            for chunk in iter_tool_call_inputs_bounded(conn, sorted(non_bash_indices)):
                while chunk:
                    tc_id, raw_input = chunk.popitem()
                    for index in non_bash_indices.get(tc_id, ()):
                        result = results[index]
                        precomputed[index] = build_search_snippet(
                            query=query,
                            kind=result.kind,
                            content=result.content,
                            thinking=result.thinking,
                            role=result.role,
                            tool_name=result.tool_name,
                            bash_command=result.bash_command,
                            tool_input=raw_input,
                            fields=fields,
                        )
                    del raw_input
        except SCHEMA_MISMATCH_ERRORS as err:
            if lease.observe(err, "search:tool_inputs"):
                raise
            logger.debug("search tool_input hydration failed", exc_info=True)

    return [
        dataclasses.replace(
            result,
            snippet=(
                precomputed[index]
                if index in precomputed
                else build_search_snippet(
                    query=query,
                    kind=result.kind,
                    content=result.content,
                    thinking=result.thinking,
                    role=result.role,
                    tool_name=result.tool_name,
                    bash_command=result.bash_command,
                    tool_input=None,
                    fields=fields,
                )
            ),
        )
        for index, result in enumerate(results)
    ]


def _attach_hosts(
    lease: DbLease,
    conn: duckdb.DuckDBPyConnection,
    results: list[SearchResult],
) -> list[SearchResult]:
    """Stamp session_state.host onto search hits (REQ-HOST-API-003/004)."""
    if not results:
        return results
    session_ids = sorted({r.session_id for r in results if r.session_id})
    if not session_ids:
        return [dataclasses.replace(r, host=UNATTRIBUTED_HOST) for r in results]
    placeholders = ", ".join("?" for _ in session_ids)
    try:
        rows = conn.execute(
            f"SELECT session_id, host FROM session_state WHERE session_id IN ({placeholders})",
            session_ids,
        ).fetchall()
    except SCHEMA_MISMATCH_ERRORS as err:
        if lease.observe(err, "search:hosts"):
            raise
        logger.debug("search host lookup failed", exc_info=True)
        return [dataclasses.replace(r, host=UNATTRIBUTED_HOST) for r in results]
    host_by_session = {
        str(sid): (label.strip() if isinstance(label, str) and label.strip() else UNATTRIBUTED_HOST)
        for sid, label in rows
    }
    return [
        dataclasses.replace(r, host=host_by_session.get(r.session_id, UNATTRIBUTED_HOST))
        for r in results
    ]


def _mark_lexical(results: list[SearchResult]) -> list[SearchResult]:
    """Flag every row as a lexical (BM25/FTS) match (REQ-CLI-017).

    Used by the keyword-only paths, where presence in the result set already
    means the row matched the query terms textually. Hybrid sets the flag
    per-row from component scores instead, so it does not route through here.
    """
    return [dataclasses.replace(result, lexical_match=True) for result in results]


def _run_keyword_search(
    lease: DbLease,
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection,
    operation: Callable[[], list[SearchResult]],
    *,
    failure_prefix: str,
    keyword_access: KeywordAccess = _COMPAT_KEYWORD_ACCESS,
) -> list[SearchResult]:
    if config.fts.backend == "duckdb":
        try:
            keyword_access.load_legacy(conn)
            return operation()
        except duckdb.Error as err:
            lease.observe(err, "search:keyword")
            raise RuntimeError(
                f"{failure_prefix}: run `recall index` to create FTS indexes"
            ) from err

    try:
        return operation()
    except FtsSidecarUnavailableError as err:
        lease.observe(err, "search:keyword")
        raise RuntimeError(
            "search failed: SQLite FTS5 sidecar unavailable on this runtime; "
            "set RECALL_FTS_BACKEND=duckdb to use the legacy path"
        ) from err
    except (sqlite3.Error, ValueError) as err:
        lease.observe(err, "search:keyword")
        check_read_cancelled()
        if isinstance(err, sqlite3.OperationalError) and "no such module: fts5" in str(err):
            raise FtsSidecarUnavailableError(
                "search failed: SQLite FTS5 sidecar unavailable on this runtime; "
                "set RECALL_FTS_BACKEND=duckdb to use the legacy path"
            ) from err
        if isinstance(err, sqlite3.Error) and is_lock_contention(err):
            raise KeywordIndexLockedError(KeywordIndexFailure.of("read failed", err)) from err
        raise RuntimeError(f"search failed: {err}") from err


def _resolve_mode(
    conn: duckdb.DuckDBPyConnection,
    mode: SearchMode,
    *,
    embed_backend_available: bool = True,
) -> SearchMode:
    """Resolve AUTO mode based on whether embeddings exist and a backend can serve queries.

    Falls back to KEYWORD when the DB has embeddings but no backend is available
    to embed the query — avoids erroring on hosts without an embedding extra.
    """
    if mode != SearchMode.AUTO:
        return mode
    if not embed_backend_available:
        return SearchMode.KEYWORD
    has_embeddings = conn.execute(
        """
        SELECT
            EXISTS(
                SELECT 1
                FROM message_embeddings
                WHERE content_embedding IS NOT NULL OR thinking_embedding IS NOT NULL
                LIMIT 1
            )
            OR EXISTS(
                SELECT 1
                FROM tool_call_embeddings
                WHERE bash_embedding IS NOT NULL
                LIMIT 1
            )
        """
    ).fetchone()
    if has_embeddings and has_embeddings[0]:
        return SearchMode.HYBRID
    return SearchMode.KEYWORD


# Public alias for mode resolution (used by rpc_server)
resolve_search_mode = _resolve_mode


def resolve_search_mode_on(lease: DbLease, mode: SearchMode) -> SearchMode:
    return lease.call("search:mode", lambda conn: _resolve_mode(conn, mode))


def _probe_backend(
    *,
    embed_backend: EmbeddingBackend | None,
    embed_backend_factory: Callable[[], EmbeddingBackend] | None,
    config: AppConfig,
) -> bool:
    """Check whether an embedding backend can be resolved without loading it.

    Returns True if a pre-built backend was passed, a factory was supplied, or
    the configured backend is available on this host.  Returns False otherwise,
    so callers can fall back to keyword search.
    """
    if embed_backend is not None:
        return True
    if embed_backend_factory is not None:
        # Factory was provided — assume the caller knows it works.  The actual
        # call is deferred to search time so a failure there still gets caught.
        return True
    from recall.services.embeddings import any_backend_available

    return any_backend_available()


def _needs_session_state_join(
    project: str | None,
    since: datetime | None,
    host: str | None,
) -> bool:
    return project is not None or since is not None or host is not None


def _session_state_join_sql(
    project: str | None,
    since: datetime | None,
    host: str | None,
) -> str:
    if _needs_session_state_join(project, since, host):
        return "JOIN session_state ss ON ss.session_id = s.id"
    return ""


def _project_match_sql(ss_alias: str = "ss") -> str:
    return f"contains(LOWER(COALESCE({ss_alias}.git_repo, '')), LOWER(?))"


def _append_session_state_predicates(
    where_parts: list[str],
    params: list[object],
    *,
    project: str | None,
    since: datetime | None,
    host: str | None,
    message_ts_expr: str | None,
    ss_alias: str = "ss",
) -> None:
    if project is not None:
        where_parts.append(_project_match_sql(ss_alias))
        params.append(project)
    if host is not None:
        where_parts.append(f"{ss_alias}.host = ?")
        params.append(host)
    if since is not None:
        naive_since = to_naive_utc(since)
        if message_ts_expr is not None:
            where_parts.append(
                f"COALESCE({message_ts_expr}, {ss_alias}.ended_at, "
                f"{ss_alias}.started_at, {ss_alias}.indexed_at) >= ?"
            )
        else:
            where_parts.append(
                f"COALESCE({ss_alias}.ended_at, {ss_alias}.started_at, {ss_alias}.indexed_at) >= ?"
            )
        params.append(naive_since)


def _append_role_predicates(
    where_parts: list[str],
    params: list[object],
    roles: tuple[Role, ...] | None,
) -> None:
    if not roles:
        return
    placeholders = ", ".join("?" for _ in roles)
    where_parts.append(f"ms.role IN ({placeholders})")
    params.extend(role.value for role in roles)


def _apply_per_session_cap(
    results: list[SearchResult],
    per_session: int | None,
    limit: int,
) -> list[SearchResult]:
    if limit <= 0:
        return []
    if per_session is None:
        return results[:limit]
    kept: list[SearchResult] = []
    counts: dict[str, int] = {}
    for result in results:
        seen = counts.get(result.session_id, 0)
        if seen >= per_session:
            continue
        counts[result.session_id] = seen + 1
        kept.append(result)
        if len(kept) >= limit:
            break
    return kept


# ---- Hybrid Search (RRF) ----


def _hybrid_search(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    query_embedding: list[float],
    source: Source | None,
    limit: int,
    fields: tuple[str, ...],
    k: int = 60,
    session: str | None = None,
    roles: tuple[Role, ...] | None = None,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
    per_session: int | None = None,
    config: AppConfig | None = None,
    keyword_access: KeywordAccess = _COMPAT_KEYWORD_ACCESS,
) -> list[SearchResult]:
    """Weighted Reciprocal Rank Fusion of BM25 + vector cosine similarity."""
    fetch_limit = limit * _KEYWORD_OVERSAMPLE_FACTOR

    bm25_results = _search_all(
        conn,
        query,
        source,
        fetch_limit,
        fields,
        session=session,
        roles=roles,
        project=project,
        since=since,
        host=host,
        per_session=per_session,
        config=config,
        keyword_access=keyword_access,
        apply_quality_adjustments=False,
    )
    vector_results = _vector_search_all(
        conn,
        query_embedding,
        source,
        fetch_limit,
        session=session,
        roles=roles,
        project=project,
        since=since,
        host=host,
        per_session=per_session,
        apply_quality_adjustments=False,
    )

    rrf_scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}
    component_scores: dict[str, tuple[float, float]] = {}

    for rank, result in enumerate(bm25_results, 1):
        key = _result_key(result)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + LEXICAL_RRF_WEIGHT / (k + rank)
        result_map[key] = result
        _update_component_score(component_scores, key, bm25_score=result.score, vector_score=None)

    for rank, result in enumerate(vector_results, 1):
        key = _result_key(result)
        clamped_cosine = max(0.0, min(1.0, float(result.score)))
        rrf_scores[key] = rrf_scores.get(key, 0.0) + clamped_cosine / (k + rank)
        if key not in result_map:
            result_map[key] = result
        _update_component_score(component_scores, key, bm25_score=None, vector_score=result.score)

    fused_results = [
        dataclasses.replace(
            result_map[rk],
            score=rrf_scores[rk],
            # REQ-CLI-017: lexical only when this row matched the BM25 leg
            # (its bm25 component is set), not merely the vector neighborhood.
            lexical_match=component_scores.get(rk, (float("-inf"), float("-inf")))[0]
            > float("-inf"),
        )
        for rk in rrf_scores
    ]
    ranked = _apply_result_quality_adjustments(
        fused_results,
        component_scores=component_scores,
        enforce_lexical_tier=True,
    )
    return _apply_per_session_cap(ranked, per_session, limit)


def _result_key(result: SearchResult) -> str:
    if result.kind == "message":
        return f"m:{result.message_id}"
    return f"t:{result.tool_call_id}"


def _update_component_score(
    scores: dict[str, tuple[float, float]],
    key: str,
    *,
    bm25_score: float | None,
    vector_score: float | None,
) -> None:
    previous_bm25, previous_vector = scores.get(key, (float("-inf"), float("-inf")))
    scores[key] = (
        bm25_score if bm25_score is not None else previous_bm25,
        vector_score if vector_score is not None else previous_vector,
    )


def _hybrid_tier(
    result: SearchResult,
    component_scores: dict[str, tuple[float, float]] | None,
) -> int:
    if result.lexical_match:
        return 0
    if component_scores is not None:
        _, vector_score = component_scores.get(
            _result_key(result),
            (float("-inf"), float("-inf")),
        )
        if vector_score >= STRONG_VECTOR_SIMILARITY:
            return 0
    return 1


def _apply_result_quality_adjustments(
    results: list[SearchResult],
    *,
    component_scores: dict[str, tuple[float, float]] | None = None,
    enforce_lexical_tier: bool = False,
) -> list[SearchResult]:
    adjusted = [
        dataclasses.replace(result, score=result.score * _result_quality_multiplier(result))
        for result in results
    ]
    return sorted(
        adjusted,
        key=lambda result: (
            _hybrid_tier(result, component_scores) if enforce_lexical_tier else 0,
            -result.score,
            *_component_score_key(result, component_scores),
            _stable_result_key(result),
        ),
    )


def _component_score_key(
    result: SearchResult,
    component_scores: dict[str, tuple[float, float]] | None,
) -> tuple[float, float]:
    if component_scores is None:
        return (float("inf"), float("inf"))
    bm25_score, vector_score = component_scores.get(
        _result_key(result),
        (float("-inf"), float("-inf")),
    )
    return (-bm25_score, -vector_score)


def _result_quality_multiplier(result: SearchResult) -> float:
    if result.kind != "message":
        return 1.0
    content = (result.content or "").strip().lower()
    if "<command-message>" in content or "<command-name>" in content:
        return 0.25
    if "allowed-tools:" in content and content.startswith("---"):
        return 0.35
    if result.role == "system" and "## context" in content:
        return 0.5
    if result.role == "system":
        return SYSTEM_ROLE_QUALITY_MULTIPLIER
    return 1.0


def _stable_result_key(result: SearchResult) -> str:
    """Order equal scores by id (REQ-SEM-011).

    Every ranked SQL cutoff here and in the FTS sidecar orders ties by the same
    id ascending before its LIMIT, so the candidates kept at a cutoff are the
    ones this key ranks first.
    """
    if result.kind == "message":
        return f"m:{result.message_id or ''}"
    return f"t:{result.tool_call_id or ''}"


# ---- Vector Search ----


def _vector_search_all(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    source: Source | None,
    limit: int,
    session: str | None = None,
    *,
    roles: tuple[Role, ...] | None = None,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
    per_session: int | None = None,
    apply_quality_adjustments: bool = True,
) -> list[SearchResult]:
    fetch_limit = limit * _KEYWORD_OVERSAMPLE_FACTOR if apply_quality_adjustments else limit
    results: list[SearchResult] = []
    results.extend(
        _vector_search_messages(
            conn,
            query_embedding,
            source,
            fetch_limit,
            session=session,
            roles=roles,
            project=project,
            since=since,
            host=host,
            per_session=per_session,
        )
    )
    if not roles:
        results.extend(
            _vector_search_tool_calls(
                conn,
                query_embedding,
                source,
                fetch_limit,
                session=session,
                project=project,
                since=since,
                host=host,
                per_session=per_session,
            )
        )
    if apply_quality_adjustments:
        ranked = _apply_result_quality_adjustments(results)
    else:
        ranked = sorted(results, key=lambda r: (-r.score, _stable_result_key(r)))
    return _apply_per_session_cap(ranked, per_session, limit)


def _vector_search_messages(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    source: Source | None,
    limit: int,
    session: str | None = None,
    *,
    roles: tuple[Role, ...] | None = None,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
    per_session: int | None = None,
) -> list[SearchResult]:
    scoped_where_parts = ["TRUE"]
    params: list[object] = [query_embedding, MIN_VECTOR_SIMILARITY, MIN_VECTOR_SIMILARITY]
    if source is not None:
        scoped_where_parts.append("s.source = ?")
        params.append(source.value)
    if session is not None:
        scoped_where_parts.append("m.session_id = ?")
        params.append(session)
    if project is not None:
        scoped_where_parts.append(_project_match_sql("ss"))
        params.append(project)
    if host is not None:
        scoped_where_parts.append("ss.host = ?")
        params.append(host)

    eligible_where_parts = ["TRUE"]
    _append_role_predicates(eligible_where_parts, params, roles)
    if since is not None:
        eligible_where_parts.append(
            "COALESCE(ms.timestamp, scoped.ended_at, scoped.started_at, scoped.indexed_at) >= ?"
        )
        params.append(to_naive_utc(since))

    params.append(MIN_VECTOR_SIMILARITY)
    qualify_clause = ""
    if per_session is not None:
        qualify_clause = (
            "QUALIFY ROW_NUMBER() OVER ("
            "PARTITION BY session_id ORDER BY score DESC, message_id ASC"
            ") <= ?"
        )
        params.append(per_session)

    session_state_join = _session_state_join_sql(project, since, host)
    since_cols = ", ss.ended_at, ss.started_at, ss.indexed_at" if since is not None else ""
    scoped_where_clause = "WHERE " + " AND ".join(scoped_where_parts)
    eligible_where_clause = "WHERE " + " AND ".join(eligible_where_parts)
    content_non_blank = sql_is_non_blank("ms.content")
    thinking_non_blank = sql_is_non_blank("ms.thinking")

    embed_dim = len(query_embedding)
    # Materialize scores before identity joins, materialize narrow session-scoped
    # candidates before reading message_state payloads, and rank before final
    # hydration. DuckDB 1.5.5 otherwise retains vectors or wide message_state
    # strings in hash-join build tables and exhausts memory.
    # test_vector_search_memory covers all three boundaries.
    sql = f"""
        WITH query_vector AS (
            SELECT ?::FLOAT[{embed_dim}] AS embedding
        ), scores AS MATERIALIZED (
            SELECT message_id, content_score, thinking_score
            FROM (
                SELECT
                    me.message_id,
                    CASE
                        WHEN me.content_embedding IS NOT NULL
                            THEN array_cosine_similarity(me.content_embedding, q.embedding)
                        ELSE NULL
                    END AS content_score,
                    CASE
                        WHEN me.thinking_embedding IS NOT NULL
                            THEN array_cosine_similarity(me.thinking_embedding, q.embedding)
                        ELSE NULL
                    END AS thinking_score
                FROM message_embeddings me
                CROSS JOIN query_vector q
                WHERE me.content_embedding IS NOT NULL OR me.thinking_embedding IS NOT NULL
            )
            WHERE COALESCE(content_score, -2.0) >= ?
               OR COALESCE(thinking_score, -2.0) >= ?
        ), scoped AS MATERIALIZED (
            SELECT
                m.id AS message_id,
                m.session_id AS session_id,
                scores.content_score,
                scores.thinking_score
                {since_cols}
            FROM scores
            JOIN messages m ON m.id = scores.message_id
            JOIN sessions s ON s.id = m.session_id
            {session_state_join}
            {scoped_where_clause}
        ), eligible AS (
            SELECT
                scoped.message_id,
                scoped.session_id,
                CASE
                    WHEN scoped.content_score IS NOT NULL AND {content_non_blank}
                        AND scoped.thinking_score IS NOT NULL AND {thinking_non_blank}
                        THEN GREATEST(scoped.content_score, scoped.thinking_score)
                    WHEN scoped.content_score IS NOT NULL AND {content_non_blank}
                        THEN scoped.content_score
                    WHEN scoped.thinking_score IS NOT NULL AND {thinking_non_blank}
                        THEN scoped.thinking_score
                    ELSE NULL
                END AS score
            FROM message_state ms
            JOIN scoped ON scoped.message_id = ms.message_id
            {eligible_where_clause}
        ), ranked AS MATERIALIZED (
            SELECT message_id, score
            FROM eligible
            WHERE score IS NOT NULL AND score >= ?
            {qualify_clause}
            ORDER BY score DESC, message_id ASC
            LIMIT {limit}
        )
        SELECT
            m.id AS message_id,
            m.session_id,
            ms.role,
            ms.content,
            ms.thinking,
            ms.timestamp,
            s.source,
            s.source_path,
            ranked.score
        FROM messages m
        JOIN ranked ON ranked.message_id = m.id
        JOIN message_state ms ON ms.message_id = m.id
        JOIN sessions s ON s.id = m.session_id
        ORDER BY ranked.score DESC, ranked.message_id ASC
    """
    rows = conn.execute(sql, params).fetchall()
    return [
        SearchResult(
            kind="message",
            session_id=row[1],
            source=row[6],
            source_path=row[7],
            score=float(row[8]),
            message_id=row[0],
            tool_call_id=None,
            role=row[2],
            content=row[3],
            thinking=row[4],
            timestamp=(
                format_utc_iso(row[5])
                if isinstance(row[5], datetime)
                else (str(row[5]) if row[5] is not None else None)
            ),
            tool_name=None,
            bash_command=None,
        )
        for row in rows
    ]


def _vector_search_tool_calls(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    source: Source | None,
    limit: int,
    session: str | None = None,
    *,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
    per_session: int | None = None,
) -> list[SearchResult]:
    where_parts = ["scores.score >= ?"]
    params: list[object] = [query_embedding, MIN_VECTOR_SIMILARITY]
    if source is not None:
        where_parts.append("s.source = ?")
        params.append(source.value)
    if session is not None:
        where_parts.append("tc.session_id = ?")
        params.append(session)
    _append_session_state_predicates(
        where_parts,
        params,
        project=project,
        since=since,
        host=host,
        message_ts_expr=None,
    )
    qualify_clause = ""
    if per_session is not None:
        qualify_clause = (
            "QUALIFY ROW_NUMBER() OVER ("
            "PARTITION BY tc.session_id ORDER BY scores.score DESC, tc.id ASC"
            ") <= ?"
        )
        params.append(per_session)

    session_state_join = _session_state_join_sql(project, since, host)
    where_clause = "WHERE " + " AND ".join(where_parts)

    sql = f"""
        WITH scores AS MATERIALIZED (
            SELECT tce.tool_call_id,
                array_cosine_similarity(
                    tce.bash_embedding, ?::FLOAT[{len(query_embedding)}]
                ) AS score
            FROM tool_call_embeddings tce
            WHERE tce.bash_embedding IS NOT NULL
        ), ranked AS MATERIALIZED (
            SELECT tc.id AS tool_call_id, scores.score
            FROM scores
            JOIN tool_calls tc ON tc.id = scores.tool_call_id
            JOIN sessions s ON s.id = tc.session_id
            {session_state_join}
            {where_clause}
            {qualify_clause}
            ORDER BY scores.score DESC, tc.id ASC
            LIMIT {limit}
        )
        SELECT
            tc.id AS tool_call_id,
            tc.session_id,
            tc.message_id,
            tc.tool_name,
            tc.bash_command,
            s.source,
            s.source_path,
            ranked.score
        FROM tool_calls tc
        JOIN ranked ON ranked.tool_call_id = tc.id
        JOIN sessions s ON s.id = tc.session_id
        ORDER BY ranked.score DESC, ranked.tool_call_id ASC
    """
    rows = conn.execute(sql, params).fetchall()
    return [
        SearchResult(
            kind="tool_call",
            session_id=row[1],
            source=row[5],
            source_path=row[6],
            score=float(row[7]),
            message_id=row[2],
            tool_call_id=row[0],
            role=None,
            content=None,
            thinking=None,
            timestamp=None,
            tool_name=row[3],
            bash_command=row[4],
        )
        for row in rows
    ]


# ---- Keyword Search (BM25) ----


def _search_all(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    source: Source | None,
    limit: int,
    fields: tuple[str, ...],
    session: str | None = None,
    *,
    roles: tuple[Role, ...] | None = None,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
    per_session: int | None = None,
    config: AppConfig | None = None,
    keyword_access: KeywordAccess = _COMPAT_KEYWORD_ACCESS,
    apply_quality_adjustments: bool = True,
) -> list[SearchResult]:
    fetch_limit = limit * _KEYWORD_OVERSAMPLE_FACTOR if apply_quality_adjustments else limit
    results: list[SearchResult] = []
    message_fields = [field for field in ("content", "thinking") if field in fields]
    if message_fields:
        results.extend(
            _search_messages(
                conn,
                query,
                source,
                fetch_limit,
                message_fields,
                session=session,
                roles=roles,
                project=project,
                since=since,
                host=host,
                per_session=per_session,
                config=config,
                keyword_access=keyword_access,
            )
        )
    if "bash" in fields and not roles:
        results.extend(
            _search_tool_calls(
                conn,
                query,
                source,
                None,
                fetch_limit,
                session=session,
                project=project,
                since=since,
                host=host,
                per_session=per_session,
                config=config,
                keyword_access=keyword_access,
            )
        )
    if apply_quality_adjustments:
        ranked = _apply_result_quality_adjustments(results)
    else:
        ranked = sorted(results, key=lambda r: (-r.score, _stable_result_key(r)))
    # REQ-CLI-017: every row here is a BM25/FTS hit. Hybrid reuses this as its
    # lexical leg and re-derives the flag per-row, so marking here is correct
    # for the keyword path and harmless for the fused path.
    return _mark_lexical(_apply_per_session_cap(ranked, per_session, limit))


def _search_messages(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    source: Source | None,
    limit: int,
    fields: list[str],
    session: str | None = None,
    *,
    roles: tuple[Role, ...] | None = None,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
    per_session: int | None = None,
    config: AppConfig | None = None,
    keyword_access: KeywordAccess = _COMPAT_KEYWORD_ACCESS,
) -> list[SearchResult]:
    if config is not None and config.fts.backend == "sqlite_sidecar":
        return _search_messages_sidecar(
            conn,
            query,
            source,
            limit,
            fields,
            session,
            config,
            roles=roles,
            project=project,
            since=since,
            host=host,
            per_session=per_session,
            keyword_access=keyword_access,
        )

    fields_value = ",".join(_message_fts_field(field) for field in fields)
    where_parts: list[str] = []
    params: list[object] = [query, fields_value]
    if source is not None:
        where_parts.append("s.source = ?")
        params.append(source.value)
    if session is not None:
        where_parts.append("m.session_id = ?")
        params.append(session)
    _append_role_predicates(where_parts, params, roles)
    _append_session_state_predicates(
        where_parts,
        params,
        project=project,
        since=since,
        host=host,
        message_ts_expr="ms.timestamp",
    )
    qualify_clause = ""
    if per_session is not None:
        qualify_clause = (
            "QUALIFY ROW_NUMBER() OVER ("
            "PARTITION BY session_id ORDER BY score DESC, message_id ASC"
            ") <= ?"
        )
        params.append(per_session)

    session_state_join = _session_state_join_sql(project, since, host)
    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    sql = f"""
        WITH ranked AS (
            SELECT
                m.id AS message_id,
                m.session_id,
                ms.role,
                ms.content,
                ms.thinking,
                ms.timestamp,
                s.source,
                s.source_path,
                fts_main_message_state.match_bm25(ms.message_id, ?, fields := ?) AS score
            FROM messages m
            JOIN message_state ms ON ms.message_id = m.id
            JOIN sessions s ON s.id = m.session_id
            {session_state_join}
            {where_clause}
        )
        SELECT * FROM ranked
        WHERE score IS NOT NULL
        {qualify_clause}
        ORDER BY score DESC, message_id ASC
        LIMIT {limit}
    """
    rows = conn.execute(sql, params).fetchall()
    return [
        SearchResult(
            kind="message",
            session_id=row[1],
            source=row[6],
            source_path=row[7],
            score=float(row[8]),
            message_id=row[0],
            tool_call_id=None,
            role=row[2],
            content=row[3],
            thinking=row[4],
            timestamp=(
                format_utc_iso(row[5])
                if isinstance(row[5], datetime)
                else (str(row[5]) if row[5] is not None else None)
            ),
            tool_name=None,
            bash_command=None,
        )
        for row in rows
    ]


def _message_fts_field(field: str) -> str:
    if field == "content":
        return "fts_content"
    if field == "thinking":
        return "fts_thinking"
    raise ValueError(f"unsupported message search field: {field}")


def _search_messages_sidecar(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    source: Source | None,
    limit: int,
    fields: list[str],
    session: str | None,
    config: AppConfig,
    *,
    roles: tuple[Role, ...] | None = None,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
    per_session: int | None = None,
    keyword_access: KeywordAccess = _COMPAT_KEYWORD_ACCESS,
) -> list[SearchResult]:
    if limit <= 0:
        return []
    has_membership_filter = (
        source is not None
        or session is not None
        or bool(roles)
        or project is not None
        or since is not None
        or host is not None
    )
    sidecar_conn = keyword_access.open(sidecar_path(config.data_dir))
    try:
        if has_membership_filter or per_session is not None:
            selective = (
                _sidecar_filter_is_selective(
                    conn,
                    "message",
                    source,
                    session,
                    roles=roles,
                    project=project,
                    since=since,
                    host=host,
                )
                if has_membership_filter
                else False
            )
            return _search_filtered_sidecar(
                limit,
                selective,
                lambda size, use_prepared, after: search_messages_fts(
                    sidecar_conn,
                    query,
                    fields,
                    size,
                    after=after,
                    use_prepared_candidates=use_prepared,
                ),
                lambda hits: _hydrate_message_sidecar_hits(
                    conn,
                    hits,
                    source,
                    session,
                    roles=roles,
                    project=project,
                    since=since,
                    host=host,
                ),
                lambda: prepare_candidate_ids(
                    sidecar_conn,
                    _sidecar_candidate_ids(
                        conn,
                        "message",
                        source,
                        session,
                        roles=roles,
                        project=project,
                        since=since,
                        host=host,
                    ),
                ),
                per_session=per_session,
            )
        hits = search_messages_fts(sidecar_conn, query, fields, limit)
        check_read_cancelled()
        return _hydrate_message_sidecar_hits(conn, hits, source, session)
    finally:
        sidecar_conn.close()


def _search_filtered_sidecar(
    limit: int,
    selective: bool,
    fetch_hits: Callable[[int, bool, tuple[float, str] | None], list[tuple[str, float]]],
    hydrate_hits: Callable[[list[tuple[str, float]]], list[SearchResult]],
    prepare_candidates: Callable[[], None],
    *,
    per_session: int | None = None,
) -> list[SearchResult]:
    """Rank selective membership with prepared candidates, or page global hits.

    Both strategies retain corpus scores and score/id ordering, use keyset
    continuation so SQLite's top-N sort stays bounded to ``page_size`` (never
    growing with ``OFFSET``), enforce per-session caps incrementally in
    O(limit) memory, and check cancellation per page.
    """
    if limit <= 0:
        return []
    if selective:
        prepare_candidates()
        if per_session is None:
            hits = fetch_hits(limit, True, None)
            check_read_cancelled()
            return hydrate_hits(hits)
    page_size = max(limit, _SIDECAR_FILTER_BATCH_SIZE)
    accepted: list[SearchResult] = []
    per_session_counts: dict[str, int] = {}
    after: tuple[float, str] | None = None
    while len(accepted) < limit:
        check_read_cancelled()
        hits = fetch_hits(page_size, selective, after)
        check_read_cancelled()
        hydrated = hydrate_hits(hits)
        for row in hydrated:
            if per_session is not None:
                seen = per_session_counts.get(row.session_id, 0)
                if seen >= per_session:
                    continue
                per_session_counts[row.session_id] = seen + 1
            accepted.append(row)
            if len(accepted) >= limit:
                break
        if len(hits) < page_size:
            break
        last_id, last_score = hits[-1]
        after = (-last_score, last_id)
    return accepted


def _sidecar_filter_is_selective(
    conn: duckdb.DuckDBPyConnection,
    kind: Literal["message", "tool_call"],
    source: Source | None,
    session: str | None,
    tool: str | None = None,
    *,
    roles: tuple[Role, ...] | None = None,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
) -> bool:
    """Estimate filter density without exporting eligible identities.

    Session counters keep source/session/project/since/host decisions independent
    of corpus size. Role-only queries without narrow session filters fall back to
    broad global paging rather than exporting a large fraction of message IDs.
    These estimates choose only the cost strategy; both paths enforce
    authoritative current membership.
    """
    if (
        roles
        and source is None
        and session is None
        and project is None
        and since is None
        and host is None
    ):
        return False
    if tool:
        assert kind == "tool_call"
        session_state_join = (
            "JOIN session_state ss ON ss.session_id = s.id"
            if _needs_session_state_join(project, since, host)
            else ""
        )
        rows = f"tool_calls item JOIN sessions s ON s.id = item.session_id {session_state_join}"
        weight = "1"
        ss_alias = "ss"
    else:
        rows = "session_state item JOIN sessions s ON s.id = item.session_id"
        weight = "item.message_count" if kind == "message" else "item.tool_count"
        ss_alias = "item"
    predicates = ["TRUE"]
    params: list[object] = []
    if source is not None:
        predicates.append("s.source = ?")
        params.append(source.value)
    if session is not None:
        predicates.append("item.session_id = ?")
        params.append(session)
    if tool:
        predicates.append("LOWER(item.tool_name) = LOWER(?)")
        params.append(tool)
    _append_session_state_predicates(
        predicates,
        params,
        project=project,
        since=since,
        host=host,
        message_ts_expr=None,
        ss_alias=ss_alias,
    )
    check_read_cancelled()
    row = conn.execute(
        f"SELECT COALESCE(SUM({weight}) FILTER (WHERE {' AND '.join(predicates)}), 0), "
        f"COALESCE(SUM({weight}), 0) FROM {rows}",
        params,
    ).fetchone()
    assert row is not None
    return row[0] <= row[1] * _SIDECAR_SELECTIVE_FRACTION


def _sidecar_candidate_ids(
    conn: duckdb.DuckDBPyConnection,
    kind: Literal["message", "tool_call"],
    source: Source | None,
    session: str | None,
    tool: str | None = None,
    *,
    roles: tuple[Role, ...] | None = None,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
) -> Iterator[str]:
    """Stream authoritative eligible identities, bounded by their current rows."""
    if kind == "message":
        table = "messages"
        state_join = "JOIN message_state ms ON ms.message_id = item.id"
    else:
        table = "tool_calls"
        state_join = ""
    session_state_join = _session_state_join_sql(project, since, host)
    predicates: list[str] = []
    params: list[object] = []
    if source is not None:
        predicates.append("s.source = ?")
        params.append(source.value)
    if session is not None:
        predicates.append("item.session_id = ?")
        params.append(session)
    if tool:
        assert kind == "tool_call"
        predicates.append("LOWER(item.tool_name) = LOWER(?)")
        params.append(tool)
    if kind == "message":
        _append_role_predicates(predicates, params, roles)
    _append_session_state_predicates(
        predicates,
        params,
        project=project,
        since=since,
        host=host,
        message_ts_expr="ms.timestamp" if kind == "message" else None,
    )
    where = "WHERE " + " AND ".join(predicates) if predicates else ""
    check_read_cancelled()
    conn.execute(
        f"SELECT item.id FROM {table} item {state_join} "
        f"JOIN sessions s ON s.id = item.session_id {session_state_join} {where}",
        params,
    )
    while rows := conn.fetchmany(_SIDECAR_CANDIDATE_BATCH_SIZE):
        check_read_cancelled()
        yield from (str(row[0]) for row in rows)


def _hydrate_message_sidecar_hits(
    conn: duckdb.DuckDBPyConnection,
    hits: list[tuple[str, float]],
    source: Source | None,
    session: str | None,
    *,
    roles: tuple[Role, ...] | None = None,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
) -> list[SearchResult]:
    if not hits:
        return []

    message_ids = [message_id for message_id, _score in hits]
    scores = dict(hits)
    placeholders = ", ".join("?" for _ in message_ids)
    where_parts = [f"m.id IN ({placeholders})"]
    params: list[object] = [*message_ids]
    if source is not None:
        where_parts.append("s.source = ?")
        params.append(source.value)
    if session is not None:
        where_parts.append("m.session_id = ?")
        params.append(session)
    _append_role_predicates(where_parts, params, roles)
    _append_session_state_predicates(
        where_parts,
        params,
        project=project,
        since=since,
        host=host,
        message_ts_expr="ms.timestamp",
    )
    session_state_join = _session_state_join_sql(project, since, host)
    where_clause = "WHERE " + " AND ".join(where_parts)

    rows = conn.execute(
        f"""
        SELECT
            m.id AS message_id,
            m.session_id,
            ms.role,
            ms.content,
            ms.thinking,
            ms.timestamp,
            s.source,
            s.source_path
        FROM messages m
        JOIN message_state ms ON ms.message_id = m.id
        JOIN sessions s ON s.id = m.session_id
        {session_state_join}
        {where_clause}
        """,
        params,
    ).fetchall()
    rows_by_id = {str(row[0]): row for row in rows}
    return [
        SearchResult(
            kind="message",
            session_id=row[1],
            source=row[6],
            source_path=row[7],
            score=scores[message_id],
            message_id=row[0],
            tool_call_id=None,
            role=row[2],
            content=row[3],
            thinking=row[4],
            timestamp=(
                format_utc_iso(row[5])
                if isinstance(row[5], datetime)
                else (str(row[5]) if row[5] is not None else None)
            ),
            tool_name=None,
            bash_command=None,
        )
        for message_id in message_ids
        if (row := rows_by_id.get(message_id)) is not None
    ]


def _search_tool_calls(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    source: Source | None,
    tool: str | None,
    limit: int,
    session: str | None = None,
    *,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
    per_session: int | None = None,
    config: AppConfig | None = None,
    keyword_access: KeywordAccess = _COMPAT_KEYWORD_ACCESS,
) -> list[SearchResult]:
    if config is not None and config.fts.backend == "sqlite_sidecar":
        return _search_tool_calls_sidecar(
            conn,
            query,
            source,
            tool,
            limit,
            session,
            config,
            project=project,
            since=since,
            host=host,
            per_session=per_session,
            keyword_access=keyword_access,
        )

    where_parts: list[str] = []
    params: list[object] = [query]
    if tool:
        where_parts.append("LOWER(tc.tool_name) = LOWER(?)")
        params.append(tool)
    if source is not None:
        where_parts.append("s.source = ?")
        params.append(source.value)
    if session is not None:
        where_parts.append("tc.session_id = ?")
        params.append(session)
    _append_session_state_predicates(
        where_parts,
        params,
        project=project,
        since=since,
        host=host,
        message_ts_expr=None,
    )
    qualify_clause = ""
    if per_session is not None:
        qualify_clause = (
            "QUALIFY ROW_NUMBER() OVER ("
            "PARTITION BY session_id ORDER BY score DESC, tool_call_id ASC"
            ") <= ?"
        )
        params.append(per_session)

    session_state_join = _session_state_join_sql(project, since, host)
    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    sql = f"""
        WITH ranked AS (
            SELECT
                tc.id AS tool_call_id,
                tc.session_id,
                tc.message_id,
                tc.tool_name,
                tc.bash_command,
                s.source,
                s.source_path,
                fts_main_tool_calls.match_bm25(tc.id, ?, fields := 'bash_command') AS score
            FROM tool_calls tc
            JOIN sessions s ON s.id = tc.session_id
            {session_state_join}
            {where_clause}
        )
        SELECT * FROM ranked
        WHERE score IS NOT NULL
        {qualify_clause}
        ORDER BY score DESC, tool_call_id ASC
        LIMIT {limit}
    """
    rows = conn.execute(sql, params).fetchall()
    return [
        SearchResult(
            kind="tool_call",
            session_id=row[1],
            source=row[5],
            source_path=row[6],
            score=float(row[7]),
            message_id=row[2],
            tool_call_id=row[0],
            role=None,
            content=None,
            thinking=None,
            timestamp=None,
            tool_name=row[3],
            bash_command=row[4],
        )
        for row in rows
    ]


def _search_tool_calls_sidecar(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    source: Source | None,
    tool: str | None,
    limit: int,
    session: str | None,
    config: AppConfig,
    *,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
    per_session: int | None = None,
    keyword_access: KeywordAccess = _COMPAT_KEYWORD_ACCESS,
) -> list[SearchResult]:
    if limit <= 0:
        return []
    has_membership_filter = (
        tool is not None
        or source is not None
        or session is not None
        or project is not None
        or since is not None
        or host is not None
    )
    sidecar_conn = keyword_access.open(sidecar_path(config.data_dir))
    try:
        if has_membership_filter or per_session is not None:
            selective = (
                _sidecar_filter_is_selective(
                    conn,
                    "tool_call",
                    source,
                    session,
                    tool,
                    project=project,
                    since=since,
                    host=host,
                )
                if has_membership_filter
                else False
            )
            return _search_filtered_sidecar(
                limit,
                selective,
                lambda size, use_prepared, after: search_tool_calls_fts(
                    sidecar_conn,
                    query,
                    size,
                    after=after,
                    use_prepared_candidates=use_prepared,
                ),
                lambda hits: _hydrate_tool_call_sidecar_hits(
                    conn,
                    hits,
                    source,
                    tool,
                    session,
                    project=project,
                    since=since,
                    host=host,
                ),
                lambda: prepare_candidate_ids(
                    sidecar_conn,
                    _sidecar_candidate_ids(
                        conn,
                        "tool_call",
                        source,
                        session,
                        tool,
                        project=project,
                        since=since,
                        host=host,
                    ),
                ),
                per_session=per_session,
            )
        hits = search_tool_calls_fts(sidecar_conn, query, limit)
        check_read_cancelled()
        return _hydrate_tool_call_sidecar_hits(conn, hits, source, tool, session)
    finally:
        sidecar_conn.close()


def _hydrate_tool_call_sidecar_hits(
    conn: duckdb.DuckDBPyConnection,
    hits: list[tuple[str, float]],
    source: Source | None,
    tool: str | None,
    session: str | None,
    *,
    project: str | None = None,
    since: datetime | None = None,
    host: str | None = None,
) -> list[SearchResult]:
    if not hits:
        return []

    tool_call_ids = [tool_call_id for tool_call_id, _score in hits]
    scores = dict(hits)
    placeholders = ", ".join("?" for _ in tool_call_ids)
    where_parts = [f"tc.id IN ({placeholders})"]
    params: list[object] = [*tool_call_ids]
    if tool:
        where_parts.append("LOWER(tc.tool_name) = LOWER(?)")
        params.append(tool)
    if source is not None:
        where_parts.append("s.source = ?")
        params.append(source.value)
    if session is not None:
        where_parts.append("tc.session_id = ?")
        params.append(session)
    _append_session_state_predicates(
        where_parts,
        params,
        project=project,
        since=since,
        host=host,
        message_ts_expr=None,
    )
    session_state_join = _session_state_join_sql(project, since, host)
    where_clause = "WHERE " + " AND ".join(where_parts)

    rows = conn.execute(
        f"""
        SELECT
            tc.id AS tool_call_id,
            tc.session_id,
            tc.message_id,
            tc.tool_name,
            tc.bash_command,
            s.source,
            s.source_path
        FROM tool_calls tc
        JOIN sessions s ON s.id = tc.session_id
        {session_state_join}
        {where_clause}
        """,
        params,
    ).fetchall()
    rows_by_id = {str(row[0]): row for row in rows}
    return [
        SearchResult(
            kind="tool_call",
            session_id=row[1],
            source=row[5],
            source_path=row[6],
            score=scores[tool_call_id],
            message_id=row[2],
            tool_call_id=row[0],
            role=None,
            content=None,
            thinking=None,
            timestamp=None,
            tool_name=row[3],
            bash_command=row[4],
        )
        for tool_call_id in tool_call_ids
        if (row := rows_by_id.get(tool_call_id)) is not None
    ]
