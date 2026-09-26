from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
from recall.db.schema import ensure_schema
from recall.services.analytics import usage_by_source


def _conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "recall.duckdb"))
    ensure_schema(conn)
    return conn


def _seed(
    conn: duckdb.DuckDBPyConnection,
    *,
    session_id: str,
    source: str,
    model: str,
    host: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cached: int | None = None,
    ended_at: datetime | None = None,
) -> None:
    conn.execute(
        "INSERT INTO sessions (id, source, source_path, source_session_id) VALUES (?, ?, ?, ?)",
        [session_id, source, f"/tmp/{session_id}", session_id],
    )
    conn.execute(
        """
        INSERT INTO session_state (
            session_id, model, host, input_tokens, output_tokens, cached_input_tokens,
            ended_at, file_mtime, file_size
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 1.0, 1)
        """,
        [session_id, model, host, input_tokens, output_tokens, cached, ended_at],
    )


def test_usage_by_source_groups_and_excludes_null_tokens(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    try:
        now = datetime.now(UTC)
        _seed(
            conn,
            session_id="a",
            source="kimi_code",
            model="kimi-code/k3",
            host="laptop",
            input_tokens=100,
            output_tokens=10,
            cached=40,
            ended_at=now,
        )
        _seed(
            conn,
            session_id="b",
            source="kimi_code",
            model="kimi-code/k3",
            host="laptop",
            input_tokens=50,
            output_tokens=5,
            cached=0,
            ended_at=now,
        )
        _seed(
            conn,
            session_id="c",
            source="grok",
            model="grok-4.5",
            host="vm",
            input_tokens=None,
            output_tokens=None,
            ended_at=now,
        )
        _seed(
            conn,
            session_id="d",
            source="grok",
            model="grok-4.5",
            host="vm",
            input_tokens=300,
            output_tokens=30,
            cached=190,
            ended_at=now,
        )

        stats = usage_by_source(conn=conn)
        by_key = {(s.source, s.host): s for s in stats}
        assert ("kimi_code", "laptop") in by_key
        kimi = by_key[("kimi_code", "laptop")]
        assert kimi.input_tokens == 150
        assert kimi.output_tokens == 15
        assert kimi.session_count == 2
        # known cache sum 40+0=40 → fresh = 150-40
        assert kimi.cached_input_tokens == 40
        assert kimi.fresh_input_tokens == 110

        assert ("grok", "vm") in by_key
        grok = by_key[("grok", "vm")]
        assert grok.input_tokens == 300
        assert grok.cached_input_tokens == 190
        assert grok.fresh_input_tokens == 110
        assert grok.session_count == 1  # null-token session excluded
    finally:
        conn.close()


def test_usage_by_source_fresh_null_when_cache_unknown(tmp_path: Path) -> None:
    """Claude-style rows leave cached_input_tokens NULL — do not invent fresh."""
    conn = _conn(tmp_path)
    try:
        _seed(
            conn,
            session_id="cc",
            source="claude_code",
            model="claude",
            host="laptop",
            input_tokens=1000,
            output_tokens=50,
            cached=None,
            ended_at=datetime.now(UTC),
        )
        stats = usage_by_source(conn=conn)
        assert len(stats) == 1
        assert stats[0].input_tokens == 1000
        assert stats[0].fresh_input_tokens is None
        assert stats[0].cached_input_tokens == 0
    finally:
        conn.close()


def test_usage_by_source_respects_since(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    try:
        old = datetime.now(UTC) - timedelta(days=40)
        new = datetime.now(UTC)
        _seed(
            conn,
            session_id="old",
            source="codex",
            model="gpt",
            host="local",
            input_tokens=999,
            output_tokens=1,
            ended_at=old,
        )
        _seed(
            conn,
            session_id="new",
            source="codex",
            model="gpt",
            host="local",
            input_tokens=10,
            output_tokens=2,
            ended_at=new,
        )
        stats = usage_by_source(since=datetime.now(UTC) - timedelta(days=7), conn=conn)
        assert len(stats) == 1
        assert stats[0].input_tokens == 10
    finally:
        conn.close()
