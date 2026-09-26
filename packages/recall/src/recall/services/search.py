from __future__ import annotations

import dataclasses
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import duckdb

from recall.core.config import AppConfig
from recall.core.embeddings import EmbeddingBackend
from recall.core.types import UNATTRIBUTED_HOST, SearchMode, Source
from recall.db import FtsSidecarUnavailableError, connect_readonly, load_fts_extension
from recall.db.fts_sidecar import (
    open_sidecar,
    search_messages_fts,
    search_tool_calls_fts,
    sidecar_path,
)

logger = logging.getLogger("recall.search")

_SIDECAR_FILTER_BATCH_SIZE = 100
_SIDECAR_FILTER_MAX_SCANNED_ROWS = 10_000


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


def search(
    *,
    query: str,
    source: Source | None,
    tool: str | None,
    session: str | None = None,
    limit: int = 20,
    mode: SearchMode = SearchMode.AUTO,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
    embed_backend: EmbeddingBackend | None = None,
    embed_backend_factory: Callable[[], EmbeddingBackend] | None = None,
    query_embedding: list[float] | None = None,
) -> list[SearchResult]:
    config = config or AppConfig.load()
    owned_conn = conn is None
    conn = conn or connect_readonly(config)
    try:
        if tool:
            return _attach_hosts(
                conn,
                _mark_lexical(
                    _run_keyword_search(
                        config,
                        conn,
                        lambda: _search_tool_calls(
                            conn,
                            query,
                            source,
                            tool,
                            limit,
                            session=session,
                            config=config,
                        ),
                        failure_prefix="search failed",
                    )
                ),
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
            return _attach_hosts(
                conn,
                _run_keyword_search(
                    config,
                    conn,
                    lambda: _search_all(
                        conn,
                        query,
                        source,
                        limit,
                        config.fts.fields,
                        session=session,
                        config=config,
                    ),
                    failure_prefix="search failed",
                ),
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
                if mode != SearchMode.AUTO:
                    raise RuntimeError(f"embedding backend unavailable: {err}") from err
                # AUTO remains truthful and useful when model acquisition fails.
                logger.warning("embedding backend unavailable, falling back to keyword: %s", err)
                return _attach_hosts(
                    conn,
                    _run_keyword_search(
                        config,
                        conn,
                        lambda: _search_all(
                            conn,
                            query,
                            source,
                            limit,
                            config.fts.fields,
                            session=session,
                            config=config,
                        ),
                        failure_prefix="search failed",
                    ),
                )

        if effective_mode == SearchMode.VECTOR:
            return _attach_hosts(
                conn,
                _vector_search_all(conn, query_embedding, source, limit, session=session),
            )

        # Hybrid: RRF fusion of BM25 + vector
        return _attach_hosts(
            conn,
            _run_keyword_search(
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
                    config=config,
                ),
                failure_prefix="hybrid search failed",
            ),
        )
    finally:
        if owned_conn:
            conn.close()


def _attach_hosts(
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
    except duckdb.Error:
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
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection,
    operation: Callable[[], list[SearchResult]],
    *,
    failure_prefix: str,
) -> list[SearchResult]:
    if config.fts.backend == "duckdb":
        try:
            load_fts_extension(conn)
            return operation()
        except duckdb.Error as err:
            raise RuntimeError(
                f"{failure_prefix}: run `recall index` to create FTS indexes"
            ) from err

    try:
        return operation()
    except FtsSidecarUnavailableError as err:
        raise RuntimeError(
            "search failed: SQLite FTS5 sidecar unavailable on this runtime; "
            "set RECALL_FTS_BACKEND=duckdb to use the legacy path"
        ) from err
    except (sqlite3.Error, ValueError) as err:
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
    config: AppConfig | None = None,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion of BM25 + vector cosine similarity."""
    fetch_limit = limit * 3

    bm25_results = _search_all(
        conn,
        query,
        source,
        fetch_limit,
        fields,
        session=session,
        config=config,
    )
    vector_results = _vector_search_all(conn, query_embedding, source, fetch_limit, session=session)

    rrf_scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}
    component_scores: dict[str, tuple[float, float]] = {}

    for rank, result in enumerate(bm25_results, 1):
        key = _result_key(result)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        result_map[key] = result
        _update_component_score(component_scores, key, bm25_score=result.score, vector_score=None)

    for rank, result in enumerate(vector_results, 1):
        key = _result_key(result)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
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
    ranked = _apply_result_quality_adjustments(fused_results, component_scores=component_scores)
    return ranked[:limit]


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


def _apply_result_quality_adjustments(
    results: list[SearchResult],
    *,
    component_scores: dict[str, tuple[float, float]] | None = None,
) -> list[SearchResult]:
    adjusted = [
        dataclasses.replace(result, score=result.score * _result_quality_multiplier(result))
        for result in results
    ]
    return sorted(
        adjusted,
        key=lambda result: (
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
    return 1.0


def _stable_result_key(result: SearchResult) -> str:
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
) -> list[SearchResult]:
    results: list[SearchResult] = []
    results.extend(_vector_search_messages(conn, query_embedding, source, limit, session=session))
    results.extend(_vector_search_tool_calls(conn, query_embedding, source, limit, session=session))
    ranked = _apply_result_quality_adjustments(results)
    return ranked[:limit]


def _vector_search_messages(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    source: Source | None,
    limit: int,
    session: str | None = None,
) -> list[SearchResult]:
    where_parts = ["TRUE"]
    params: list[object] = [query_embedding]
    if source is not None:
        where_parts.append("s.source = ?")
        params.append(source.value)
    if session is not None:
        where_parts.append("m.session_id = ?")
        params.append(session)

    where_clause = "WHERE " + " AND ".join(where_parts)

    embed_dim = len(query_embedding)
    # Materialize scores before identity joins, then rank before wide payloads.
    # DuckDB 1.5.5 otherwise retains vectors in hash joins and exhausts memory
    # after writes have loaded ART indexes. test_vector_search_memory covers
    # both boundaries; remove only when it and the populated benchmark pass.
    sql = f"""
        WITH query_vector AS (
            SELECT ?::FLOAT[{embed_dim}] AS embedding
        ), scores AS MATERIALIZED (
            SELECT me.message_id, CASE
                WHEN me.content_embedding IS NOT NULL
                    AND me.thinking_embedding IS NOT NULL
                    THEN GREATEST(
                        array_cosine_similarity(me.content_embedding, q.embedding),
                        array_cosine_similarity(me.thinking_embedding, q.embedding)
                    )
                WHEN me.content_embedding IS NOT NULL THEN array_cosine_similarity(
                    me.content_embedding, q.embedding
                )
                ELSE array_cosine_similarity(me.thinking_embedding, q.embedding)
            END AS score
            FROM message_embeddings me
            CROSS JOIN query_vector q
            WHERE me.content_embedding IS NOT NULL OR me.thinking_embedding IS NOT NULL
        ), ranked AS MATERIALIZED (
            SELECT m.id AS message_id, scores.score
            FROM scores
            JOIN messages m ON m.id = scores.message_id
            JOIN message_state ms ON ms.message_id = m.id
            JOIN sessions s ON s.id = m.session_id
            {where_clause}
            ORDER BY score DESC
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
        ORDER BY ranked.score DESC
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
            timestamp=str(row[5]) if row[5] is not None else None,
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
) -> list[SearchResult]:
    where_parts = ["TRUE"]
    params: list[object] = [query_embedding]
    if source is not None:
        where_parts.append("s.source = ?")
        params.append(source.value)
    if session is not None:
        where_parts.append("tc.session_id = ?")
        params.append(session)

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
            {where_clause}
            ORDER BY score DESC
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
        ORDER BY ranked.score DESC
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
    config: AppConfig | None = None,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    message_fields = [field for field in ("content", "thinking") if field in fields]
    if message_fields:
        results.extend(
            _search_messages(
                conn,
                query,
                source,
                limit,
                message_fields,
                session=session,
                config=config,
            )
        )
    if "bash" in fields:
        results.extend(
            _search_tool_calls(
                conn,
                query,
                source,
                None,
                limit,
                session=session,
                config=config,
            )
        )
    ranked = _apply_result_quality_adjustments(results)
    # REQ-CLI-017: every row here is a BM25/FTS hit. Hybrid reuses this as its
    # lexical leg and re-derives the flag per-row, so marking here is correct
    # for the keyword path and harmless for the fused path.
    return _mark_lexical(ranked[:limit])


def _search_messages(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    source: Source | None,
    limit: int,
    fields: list[str],
    session: str | None = None,
    config: AppConfig | None = None,
) -> list[SearchResult]:
    if config is not None and config.fts.backend == "sqlite_sidecar":
        return _search_messages_sidecar(conn, query, source, limit, fields, session, config)

    fields_value = ",".join(_message_fts_field(field) for field in fields)
    where_parts: list[str] = []
    params: list[object] = [query, fields_value]
    if source is not None:
        where_parts.append("s.source = ?")
        params.append(source.value)
    if session is not None:
        where_parts.append("m.session_id = ?")
        params.append(session)
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
            {where_clause}
        )
        SELECT * FROM ranked
        WHERE score IS NOT NULL
        ORDER BY score DESC
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
            timestamp=str(row[5]) if row[5] is not None else None,
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
) -> list[SearchResult]:
    if limit <= 0:
        return []
    has_duckdb_filters = source is not None or session is not None
    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        if not has_duckdb_filters:
            hits = search_messages_fts(sidecar_conn, query, fields, limit)
            return _hydrate_message_sidecar_hits(conn, hits, source, session)

        results: list[SearchResult] = []
        offset = 0
        while len(results) < limit and offset < _SIDECAR_FILTER_MAX_SCANNED_ROWS:
            batch_limit = _sidecar_filter_batch_limit(limit, offset)
            hits = search_messages_fts(sidecar_conn, query, fields, batch_limit, offset=offset)
            if not hits:
                break
            offset += len(hits)
            results.extend(_hydrate_message_sidecar_hits(conn, hits, source, session))

        if len(results) < limit and offset >= _SIDECAR_FILTER_MAX_SCANNED_ROWS:
            logger.warning(
                "sidecar message search scan cap reached before satisfying filtered limit; "
                "limit=%s scanned=%s source_filter=%s session_filter=%s",
                limit,
                offset,
                source is not None,
                session is not None,
            )
        return results[:limit]
    finally:
        sidecar_conn.close()


def _sidecar_filter_batch_limit(limit: int, offset: int) -> int:
    remaining_scan_budget = _SIDECAR_FILTER_MAX_SCANNED_ROWS - offset
    return min(max(limit, _SIDECAR_FILTER_BATCH_SIZE), remaining_scan_budget)


def _hydrate_message_sidecar_hits(
    conn: duckdb.DuckDBPyConnection,
    hits: list[tuple[str, float]],
    source: Source | None,
    session: str | None,
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
            timestamp=str(row[5]) if row[5] is not None else None,
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
    config: AppConfig | None = None,
) -> list[SearchResult]:
    if config is not None and config.fts.backend == "sqlite_sidecar":
        return _search_tool_calls_sidecar(conn, query, source, tool, limit, session, config)

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
            {where_clause}
        )
        SELECT * FROM ranked
        WHERE score IS NOT NULL
        ORDER BY score DESC
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
) -> list[SearchResult]:
    if limit <= 0:
        return []
    has_duckdb_filters = tool is not None or source is not None or session is not None
    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        if not has_duckdb_filters:
            hits = search_tool_calls_fts(sidecar_conn, query, limit)
            return _hydrate_tool_call_sidecar_hits(conn, hits, source, tool, session)

        results: list[SearchResult] = []
        offset = 0
        while len(results) < limit and offset < _SIDECAR_FILTER_MAX_SCANNED_ROWS:
            batch_limit = _sidecar_filter_batch_limit(limit, offset)
            hits = search_tool_calls_fts(sidecar_conn, query, batch_limit, offset=offset)
            if not hits:
                break
            offset += len(hits)
            results.extend(_hydrate_tool_call_sidecar_hits(conn, hits, source, tool, session))

        if len(results) < limit and offset >= _SIDECAR_FILTER_MAX_SCANNED_ROWS:
            logger.warning(
                "sidecar tool-call search scan cap reached before satisfying filtered limit; "
                "limit=%s scanned=%s tool_filter=%s source_filter=%s session_filter=%s",
                limit,
                offset,
                tool is not None,
                source is not None,
                session is not None,
            )
        return results[:limit]
    finally:
        sidecar_conn.close()


def _hydrate_tool_call_sidecar_hits(
    conn: duckdb.DuckDBPyConnection,
    hits: list[tuple[str, float]],
    source: Source | None,
    tool: str | None,
    session: str | None,
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
