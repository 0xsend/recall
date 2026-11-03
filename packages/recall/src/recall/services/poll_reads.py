"""Owned, existing-only reads admitted by the installed systemd poll service."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb

from recall.core.config import AppConfig
from recall.core.poll_reads import (
    BashRead,
    ListRead,
    PollRead,
    PollSnapshot,
    SearchRead,
    SessionListRead,
    ShowRead,
    SkillsRead,
    StatsRead,
    UsageRead,
)
from recall.core.read_cancellation import check_read_cancelled, read_cancellation
from recall.core.rpc_types import RpcError
from recall.core.types import SearchMode, parse_search_mode, parse_source
from recall.db.fts_sidecar import (
    FtsSidecarUnavailableError,
    fts_fields_signature,
    get_fts_fields_signature,
    open_existing_sidecar_readonly,
)
from recall.db.ownership import read_existing_snapshot
from recall.read_budget import ReadBudget
from recall.services import analytics
from recall.services.coordinator import reconciliation_status
from recall.services.read_handlers import ShowSelection, read_list, read_overview
from recall.services.runtime_state import load_runtime_status_from_conn
from recall.services.search import KeywordAccess, resolve_search_mode, search

POLL_READ_SECONDS = 30.0


class _ReadRefused(Exception):
    """Carry a frozen RPC refusal through engine/ownership cleanup."""

    def __init__(self, error: RpcError) -> None:
        super().__init__(error.message)
        self.error = error


class _Execution:
    """Propagate one operation deadline to both engines and drain before return."""

    def __init__(self, budget: ReadBudget) -> None:
        self.budget = budget
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.conn: duckdb.DuckDBPyConnection | None = None
        self.timer = threading.Timer(budget.remaining(), self.cancel)

    def cancel(self) -> None:
        with self.lock:
            self.cancelled.set()
            if self.conn is not None:
                self.conn.interrupt()

    def check(self) -> None:
        if self.budget.remaining() <= 0 or self.cancelled.is_set():
            raise RuntimeError("poll snapshot read timed out; retry this command")
        check_read_cancelled()

    def snapshot(self, config: AppConfig, read: Callable[[duckdb.DuckDBPyConnection], Any]) -> Any:
        self.check()

        def owned(conn: duckdb.DuckDBPyConnection) -> Any:
            with self.lock:
                self.conn = conn
            try:
                self.check()
                return read(conn)
            finally:
                with self.lock:
                    self.conn = None

        result = read_existing_snapshot(config, owned)
        self.check()
        return result


def _load_legacy(conn: duckdb.DuckDBPyConnection) -> None:
    try:
        conn.execute("SET autoinstall_known_extensions=false")
        conn.execute("SET autoload_known_extensions=false")
        conn.execute("LOAD fts")
    except duckdb.Error as err:
        raise RuntimeError(
            "installed DuckDB FTS extension unavailable; "
            "initialize keyword indexing with an authorized writer"
        ) from err


def _search(
    config: AppConfig, request: SearchRead, execution: _Execution
) -> Callable[[duckdb.DuckDBPyConnection], Any]:
    params = request.params
    if not params.query:
        raise ValueError("query is required")
    mode = parse_search_mode(params.mode) if params.mode else SearchMode.AUTO
    desired = SearchMode.KEYWORD if params.tool else mode
    if desired is SearchMode.AUTO:
        desired = execution.snapshot(config, lambda conn: resolve_search_mode(conn, mode))
    embedding = None
    backend_unavailable = False
    if desired in (SearchMode.VECTOR, SearchMode.HYBRID):
        from recall.services.embeddings import get_backend

        try:
            backend = get_backend(config.embedding)
            embedding = backend.embed([backend.query_prefix + params.query])[0]
        except (ValueError, ImportError, OSError, RuntimeError) as err:
            if mode is not SearchMode.AUTO:
                raise RuntimeError(f"embedding backend unavailable: {err}") from err
            backend_unavailable = True
    execution.check()

    def read(conn: duckdb.DuckDBPyConnection) -> Any:
        final_mode = (
            SearchMode.KEYWORD
            if params.tool or backend_unavailable
            else resolve_search_mode(conn, mode)
        )
        if final_mode in (SearchMode.VECTOR, SearchMode.HYBRID) and embedding is None:
            raise RuntimeError(
                "poll snapshot search mode changed during preparation; retry this command"
            )

        def open_keyword(path: Path) -> sqlite3.Connection:
            execution.check()
            pending = conn.execute("SELECT EXISTS(SELECT 1 FROM fts_sidecar_pending)").fetchone()
            assert pending is not None
            if pending[0]:
                raise RuntimeError(
                    "keyword publication is pending; finish indexing with an authorized writer "
                    "before a local lexical read"
                )
            try:
                sidecar = open_existing_sidecar_readonly(path, timeout=execution.budget.remaining())
            except (sqlite3.Error, FtsSidecarUnavailableError) as err:
                raise RuntimeError(
                    f"keyword index {path} cannot be read: {err}; "
                    "use an authorized writer for initialization or repair"
                ) from err
            try:
                if get_fts_fields_signature(sidecar) != fts_fields_signature(config.fts.fields):
                    raise RuntimeError(
                        "keyword field signature differs; "
                        "explicit canonical initialization is required "
                        "(docs/keyword-initialization.md)"
                    )
            except BaseException:
                sidecar.close()
                raise
            return sidecar

        return search(
            query=params.query,
            source=parse_source(params.source) if params.source else None,
            tool=params.tool,
            session=params.session,
            limit=params.limit,
            mode=final_mode,
            config=config,
            conn=conn,
            query_embedding=embedding,
            keyword_access=KeywordAccess(open_keyword, _load_legacy),
        )

    return read


def _reader(
    config: AppConfig, request: PollRead, execution: _Execution
) -> Callable[[duckdb.DuckDBPyConnection], Any]:
    match request:
        case SearchRead():
            return _search(config, request, execution)
        case ListRead() | SessionListRead():
            return lambda conn: read_list(config, conn, request.params)
        case ShowRead():
            selection = ShowSelection.prepare(request.params)
            assert not selection.params.fresh
            return lambda conn: selection.read(config, conn)
        case StatsRead(method="recall.stats"):
            return lambda conn: read_overview(config, conn)
        case StatsRead(method="recall.stats_tools"):
            return lambda conn: analytics.tool_usage(config=config, conn=conn)
        case StatsRead(method="recall.stats_tokens"):
            return lambda conn: analytics.token_usage(config=config, conn=conn)
        case BashRead():

            def bash(conn: duckdb.DuckDBPyConnection) -> Any:
                if request.params.suggest:
                    suggestions, skipped = analytics.bash_suggestions(config=config, conn=conn)
                    return {"suggestions": suggestions, "skipped": skipped}
                return analytics.bash_breakdown(config=config, conn=conn)

            return bash
        case UsageRead():
            from recall.core.time import parse_since

            since = parse_since(request.params.since) if request.params.since else None
            return lambda conn: [
                asdict(row)
                for row in analytics.usage_by_source(since=since, config=config, conn=conn)
            ]
        case SkillsRead():
            from recall.core.time import parse_since

            since = parse_since(request.params.since) if request.params.since else None
            sources = (
                tuple(dict.fromkeys(parse_source(value) for value in request.params.source))
                if request.params.source is not None
                else None
            )
            return lambda conn: asdict(
                analytics.skill_usage(sources=sources, since=since, config=config, conn=conn)
            )
    raise AssertionError("unhandled poll read")


def read_poll_snapshot(config: AppConfig, request: PollRead, budget: ReadBudget) -> PollSnapshot:
    from recall.core.rpc_types import serialize_rpc_value

    execution = _Execution(budget)
    execution.timer.start()
    try:
        with read_cancellation(execution.cancelled):
            read = _reader(config, request, execution)

            def detach(conn: duckdb.DuckDBPyConnection) -> PollSnapshot:
                try:
                    data = read(conn)
                except RpcError as err:
                    raise _ReadRefused(err) from err
                runtime = load_runtime_status_from_conn(conn)
                coverage = (
                    None
                    if isinstance(request, ShowRead)
                    else reconciliation_status(config, conn=conn, limit=1)
                )
                execution.check()
                return PollSnapshot(
                    serialize_rpc_value(data),
                    runtime.last_successful_at,
                    serialize_rpc_value(coverage),
                )

            return execution.snapshot(config, detach)
    except _ReadRefused as err:
        raise err.error from err
    finally:
        execution.timer.cancel()
        execution.timer.join()
