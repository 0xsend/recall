"""Write sessions in tests the way the daemon's raw turn does.

Every write function takes a keyword turn. These helpers run one turn and
publish its keyword rows after COMMIT, so a test sees the sidecar as a
finished writer turn leaves it (REQ-FTS-SIDECAR-018).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
from recall.core.config import AppConfig
from recall.core.models import ParseResult, Session
from recall.db import advisory_lock, connect
from recall.db.fts_sidecar import ensure_sidecar, sidecar_path
from recall.db.fts_sidecar_stage import KeywordStage, KeywordTurn, Publication
from recall.parsers import SessionParser
from recall.services.indexer import _write_session
from recall.services.watcher import SessionIndexResult, index_single_session


def without_keyword_index() -> KeywordTurn:
    """A turn for the DuckDB keyword backend: no sidecar and no intents."""
    return KeywordTurn(None, None)


def sidecar_turn(data_dir: Path, *, fields: tuple[str, ...] | None = None) -> KeywordTurn:
    """A turn publishing to `recall.fts.sqlite` under `data_dir`."""
    return KeywordTurn(sidecar_path(data_dir), fields)


def publish(conn: duckdb.DuckDBPyConnection, turn: KeywordTurn) -> Publication:
    """Publish the turn's committed rows and retire their intents, as a raw turn does."""
    publication = turn.publish(attempt=True)
    publication.retire(conn)
    return publication


@contextmanager
def staged(
    conn: duckdb.DuckDBPyConnection,
    data_dir: Path,
    *,
    fields: tuple[str, ...] | None = None,
) -> Iterator[KeywordStage]:
    """Run one DuckDB transaction and publish its keyword rows once it commits."""
    turn = sidecar_turn(data_dir, fields=fields)
    with turn.transaction(conn) as stage:
        yield stage
    publish(conn, turn)


def write_published_session(
    conn: duckdb.DuckDBPyConnection,
    session: Session,
    data_dir: Path,
    *,
    fields: tuple[str, ...] | None = None,
    **kwargs: Any,
) -> bool:
    """`_write_session` in its own turn, published to the sidecar under `data_dir`."""
    turn = sidecar_turn(data_dir, fields=fields)
    result = _write_session(conn, session, turn=turn, **kwargs)
    publish(conn, turn)
    return result


def index_session_file(
    path: Path,
    parser: SessionParser,
    config: AppConfig,
    *,
    on_commit: Callable[[Session, ParseResult, bool], None] | None = None,
    prepared_result: ParseResult | None = None,
    host: str | None = None,
    lightweight_context: bool = False,
) -> SessionIndexResult:
    """Index one transcript as one writer turn: lock, connection, publication, checkpoint."""
    ensure_sidecar(config)
    with advisory_lock(config.lock_path):
        conn = connect(config)
        try:
            turn = KeywordTurn.for_config(config)
            result = index_single_session(
                path,
                parser,
                config,
                conn=conn,
                turn=turn,
                on_commit=on_commit,
                prepared_result=prepared_result,
                host=host,
                lightweight_context=lightweight_context,
            )
            publish(conn, turn)
            conn.execute("CHECKPOINT")
            return result
        finally:
            conn.close()
