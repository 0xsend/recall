"""Read assembly shared by the resident RPC owner and owned poll snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import duckdb

from recall.core.config import AppConfig
from recall.core.rpc_params import ListParams, ShowParams
from recall.core.rpc_types import APP_NOT_FOUND, INVALID_PARAMS, RpcError
from recall.core.source_path import SourcePath
from recall.core.time import parse_since
from recall.core.types import parse_source
from recall.db.lease import DbLease, StandaloneLifetime
from recall.services.live import (
    Cursor,
    catalog_progress_for_session,
    decode_cursor,
    derive_freshness,
    encode_cursor,
)
from recall.services.sessions import (
    list_sessions_on,
    load_session_on,
    load_session_tail_on,
    resolve_session_header_on,
)


def read_list(config: AppConfig, conn: duckdb.DuckDBPyConnection, params: ListParams) -> Any:
    with StandaloneLifetime().lease(conn) as lease:
        return read_list_on(config, lease, params)


def read_list_on(config: AppConfig, lease: DbLease, params: ListParams) -> Any:
    return list_sessions_on(
        lease,
        source=parse_source(params.source) if params.source else None,
        since=parse_since(params.since) if params.since else None,
        project=params.project,
        host=params.host,
        session=params.session,
        limit=params.limit,
        config=config,
    )


@dataclass(frozen=True)
class ShowSelection:
    params: ShowParams
    cursor: Cursor | None
    after_idx: int | None
    cursor_session: str | None

    @classmethod
    def prepare(cls, params: ShowParams) -> ShowSelection:
        if not params.session_id:
            raise RpcError(code=INVALID_PARAMS, message="session_id is required")
        if (
            params.tail is not None or params.after is not None
        ) and params.message_limit is not None:
            raise RpcError(
                code=INVALID_PARAMS,
                message=(
                    "message_limit reads from the start of the session; "
                    "tail and after read from the end"
                ),
            )
        if params.tail is not None and params.tail <= 0:
            raise RpcError(code=INVALID_PARAMS, message="tail must be positive")
        try:
            cursor = decode_cursor(params.after) if params.after is not None else None
        except ValueError as err:
            raise RpcError(code=INVALID_PARAMS, message=str(err)) from err
        cursor_session, after_idx = cursor if cursor is not None else (None, None)
        return cls(params, cursor, after_idx, cursor_session)

    def header(self, conn: duckdb.DuckDBPyConnection) -> Any:
        with StandaloneLifetime().lease(conn) as lease:
            return self.header_on(lease)

    def header_on(self, lease: DbLease) -> Any:
        try:
            header = resolve_session_header_on(lease, self.params.session_id)
        except ValueError as err:
            if str(err).startswith(("session not found", "session not indexed")):
                raise RpcError(code=APP_NOT_FOUND, message=str(err)) from err
            raise
        if self.cursor_session is not None and self.cursor_session != header.id:
            raise RpcError(
                code=INVALID_PARAMS,
                message=f"cursor belongs to another session: {self.cursor_session}",
            )
        return header

    def read(
        self,
        config: AppConfig,
        conn: duckdb.DuckDBPyConnection,
        *,
        departed_paths: frozenset[SourcePath] = frozenset(),
    ) -> Any:
        with StandaloneLifetime().lease(conn) as lease:
            return self.read_on(config, lease, departed_paths=departed_paths)

    def read_on(
        self,
        config: AppConfig,
        lease: DbLease,
        *,
        departed_paths: frozenset[SourcePath] = frozenset(),
    ) -> Any:
        params = self.params
        header = self.header_on(lease)
        progress = lease.call(
            "sessions:catalog_progress",
            lambda conn: catalog_progress_for_session(
                conn, session_id=header.id, source_path=header.source_path
            ),
        )
        epoch = progress.content_epoch if progress is not None else 0
        reset = self.cursor is not None and (self.cursor.content_epoch or 0) != epoch
        if params.tail is not None or params.after is not None:
            session = load_session_tail_on(
                lease,
                params.session_id,
                tail=(params.tail or 100) if reset else params.tail,
                after_idx=None if reset else self.after_idx,
                tools=params.tools,
                config=config,
            )
        else:
            session = load_session_on(
                lease,
                params.session_id,
                include_tools=params.tools,
                message_limit=params.message_limit,
                config=config,
            )
        last_seen = max(
            (message.idx for message in session.messages),
            default=self.after_idx if self.after_idx is not None and not reset else -1,
        )
        return {
            **session.model_dump(),
            "freshness": derive_freshness(
                session.source_path,
                indexed_mtime=session.file_mtime,
                indexed_size=session.file_size,
                catalog=progress,
                source_departed=SourcePath(session.source.value, session.source_path)
                in departed_paths,
            ),
            "cursor": encode_cursor(session.id, last_seen, epoch),
            "cursor_reset": reset,
            "cursor_reset_reason": "content_rewritten" if reset else None,
        }


def read_overview(config: AppConfig, conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    with StandaloneLifetime().lease(conn) as lease:
        return read_overview_on(config, lease)


def read_overview_on(config: AppConfig, lease: DbLease) -> dict[str, Any]:
    from recall.services.analytics import overview_on

    stats = asdict(overview_on(lease, config=config))
    stats.update(runtime_context_stats_on(lease))
    return stats


def runtime_context_stats(conn: Any) -> dict[str, Any]:
    with StandaloneLifetime().lease(conn) as lease:
        return runtime_context_stats_on(lease)


def runtime_context_stats_on(lease: DbLease) -> dict[str, Any]:
    try:
        row = lease.call(
            "runtime_context_stats",
            lambda conn: conn.execute(
                """
            SELECT
                COALESCE(last_index_total, 0),
                COALESCE(last_index_indexed, 0),
                COALESCE(last_index_skipped, 0),
                COALESCE(last_index_failed, 0),
                COALESCE(last_context_messages, 0),
                COALESCE(last_context_mode, 'off'),
                COALESCE(last_context_input_tokens, 0),
                COALESCE(last_context_output_tokens, 0),
                last_context_model
            FROM runtime_state
            WHERE singleton = TRUE
            """
            ).fetchone(),
        )
    except Exception as error:
        if lease.observe(error, "runtime_context_stats"):
            raise
        row = None
    if row is None:
        return {
            "last_index_total": 0,
            "last_index_indexed": 0,
            "last_index_skipped": 0,
            "last_index_failed": 0,
            "last_context_messages": 0,
            "last_context_mode": "off",
            "last_context_input_tokens": 0,
            "last_context_output_tokens": 0,
            "last_context_model": None,
        }
    return {
        "last_index_total": int(row[0] or 0),
        "last_index_indexed": int(row[1] or 0),
        "last_index_skipped": int(row[2] or 0),
        "last_index_failed": int(row[3] or 0),
        "last_context_messages": int(row[4] or 0),
        "last_context_mode": str(row[5] or "off"),
        "last_context_input_tokens": int(row[6] or 0),
        "last_context_output_tokens": int(row[7] or 0),
        "last_context_model": str(row[8]) if row[8] is not None else None,
    }
