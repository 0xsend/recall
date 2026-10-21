"""Equal scores at a search cutoff resolve by the stable document key.

Every ranked query keeps the candidates with the lowest stable keys among
rows that tie at its LIMIT, so membership and order never depend on scan
order. The corpora insert rows in descending key order, so a cutoff ordered
by score alone keeps the wrong tied rows.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import duckdb
import pytest
from recall.core.config import (
    DEFAULT_EMBED_MODEL,
    KNOWN_MODEL_DIMENSIONS,
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.types import SearchMode
from recall.db.fts_sidecar import (
    open_sidecar,
    sidecar_path,
    upsert_message_fts_batch,
    upsert_tool_call_fts_batch,
)
from recall.db.queries import create_fts_indexes
from recall.db.schema import ensure_schema
from recall.services.search import SearchResult, search

Kind = Literal["message", "tool_call"]
Backend = Literal["duckdb", "sqlite_sidecar"]

_EMBED_DIM = KNOWN_MODEL_DIMENSIONS[DEFAULT_EMBED_MODEL]
_SESSION_ID = "session-ties"
_TIED_TEXT = "git rebase onto main"
_LIMIT = 5
# Hybrid fetches three times the limit from each leg; the tie must outlast it.
_TIED_COUNT = 4 * _LIMIT
_WINNER_ID_SUFFIX = "zz"


def _unit(index: int) -> list[float]:
    values = [0.0] * _EMBED_DIM
    values[index] = 1.0
    return values


_QUERY_VECTOR = _unit(0)
_WINNER_VECTOR = _unit(0)
# Cosine 0.6 with the query, identical for every tied row.
_TIED_VECTOR = [0.6, 0.8] + [0.0] * (_EMBED_DIM - 2)


def _ids(kind: Kind, count: int) -> list[str]:
    prefix = "m" if kind == "message" else "t"
    return [f"{prefix}-{index:02d}" for index in range(count)]


def _winner_id(kind: Kind) -> str:
    return f"{'m' if kind == 'message' else 't'}-{_WINNER_ID_SUFFIX}"


def _config(tmp_path: Path, backend: Backend) -> AppConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / "config.toml",
        fts=FtsConfig(backend=backend),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )


def _seed(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
    kind: Kind,
    *,
    with_vector_winner: bool,
) -> None:
    """Seed one session whose rows tie exactly, inserted in descending key order."""
    conn.execute(
        "INSERT INTO sessions (id, source, source_path, source_session_id) VALUES (?, ?, ?, ?)",
        [_SESSION_ID, "codex", "/tmp/session-ties.jsonl", None],
    )
    rows = [(row_id, _TIED_VECTOR) for row_id in reversed(_ids(kind, _TIED_COUNT))]
    if with_vector_winner:
        rows.insert(0, (_winner_id(kind), _WINNER_VECTOR))

    if kind == "message":
        for idx, (message_id, vector) in enumerate(rows):
            conn.execute(
                "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
                [message_id, _SESSION_ID, idx],
            )
            conn.execute(
                """
                INSERT INTO message_state (message_id, role, content, thinking, has_thinking)
                VALUES (?, ?, ?, ?, ?)
                """,
                [message_id, "assistant", _TIED_TEXT, "", False],
            )
            conn.execute(
                "INSERT INTO message_embeddings (message_id, content_embedding) VALUES (?, ?)",
                [message_id, vector],
            )
    else:
        anchor_message_id = "anchor"
        conn.execute(
            "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
            [anchor_message_id, _SESSION_ID, 0],
        )
        for idx, (tool_call_id, vector) in enumerate(rows):
            conn.execute(
                """
                INSERT INTO tool_calls (
                    id, session_id, message_id, idx, tool_name, bash_command, is_compound
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [tool_call_id, _SESSION_ID, anchor_message_id, idx, "Bash", _TIED_TEXT, False],
            )
            conn.execute(
                "INSERT INTO tool_call_embeddings (tool_call_id, bash_embedding) VALUES (?, ?)",
                [tool_call_id, vector],
            )

    if config.fts.backend == "duckdb":
        create_fts_indexes(conn, config.fts)
        return
    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        if kind == "message":
            upsert_message_fts_batch(sidecar_conn, [(row_id, _TIED_TEXT, "") for row_id, _ in rows])
        else:
            upsert_tool_call_fts_batch(sidecar_conn, [(row_id, _TIED_TEXT) for row_id, _ in rows])
    finally:
        sidecar_conn.close()


def _result_id(result: SearchResult) -> str | None:
    return result.message_id if result.kind == "message" else result.tool_call_id


def _run(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
    mode: SearchMode,
    session: str | None,
) -> list[str | None]:
    results = search(
        query=_TIED_TEXT,
        source=None,
        tool=None,
        session=session,
        limit=_LIMIT,
        mode=mode,
        config=config,
        conn=conn,
        query_embedding=_QUERY_VECTOR,
    )
    return [_result_id(result) for result in results]


@pytest.mark.parametrize("session", [None, _SESSION_ID], ids=["unfiltered", "session-filter"])
@pytest.mark.parametrize("backend", ["duckdb", "sqlite_sidecar"])
@pytest.mark.parametrize("kind", ["message", "tool_call"])
@pytest.mark.parametrize("mode", [SearchMode.KEYWORD, SearchMode.HYBRID])
def test_lexical_cutoff_keeps_lowest_keys_among_tied_scores(
    tmp_path: Path,
    mode: SearchMode,
    kind: Kind,
    backend: Backend,
    session: str | None,
) -> None:
    config = _config(tmp_path, backend)
    conn = duckdb.connect(str(config.db_path))
    try:
        ensure_schema(conn)
        _seed(conn, config, kind, with_vector_winner=False)

        assert _run(conn, config, mode, session) == _ids(kind, _LIMIT)
    finally:
        conn.close()


@pytest.mark.parametrize("session", [None, _SESSION_ID], ids=["unfiltered", "session-filter"])
@pytest.mark.parametrize("kind", ["message", "tool_call"])
def test_vector_cutoff_keeps_lowest_keys_among_tied_scores(
    tmp_path: Path,
    kind: Kind,
    session: str | None,
) -> None:
    config = _config(tmp_path, "duckdb")
    conn = duckdb.connect(str(config.db_path))
    try:
        ensure_schema(conn)
        _seed(conn, config, kind, with_vector_winner=True)

        results = search(
            query=_TIED_TEXT,
            source=None,
            tool=None,
            session=session,
            limit=_LIMIT,
            mode=SearchMode.VECTOR,
            config=config,
            conn=conn,
            query_embedding=_QUERY_VECTOR,
        )

        assert [_result_id(result) for result in results] == [
            _winner_id(kind),
            *_ids(kind, _LIMIT - 1),
        ]
        winner, *tied = (result.score for result in results)
        assert math.isclose(winner, 1.0, rel_tol=1e-6)
        assert len(set(tied)) == 1
        assert math.isclose(tied[0], 0.6, rel_tol=1e-6)
    finally:
        conn.close()
