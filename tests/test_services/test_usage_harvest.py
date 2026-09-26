from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.services.usage_harvest import harvest_grok_unified_log, rollup_grok_sessions


def _fixtures() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "grok" / "unified"


def _connect(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "recall.duckdb"))
    ensure_schema(conn)
    return conn


def _seed_grok_session(
    conn: duckdb.DuckDBPyConnection, *, session_id: str, sid: str, workspace: str = "/tmp"
) -> None:
    conn.execute(
        "INSERT INTO sessions (id, source, source_path, source_session_id) VALUES (?, ?, ?, ?)",
        [session_id, "grok", f"{workspace}/{sid}/chat_history.jsonl", sid],
    )
    conn.execute(
        """
        INSERT INTO session_state (
            session_id, file_mtime, file_size, input_tokens, output_tokens
        ) VALUES (?, 1.0, 1, NULL, NULL)
        """,
        [session_id],
    )


def test_harvest_upserts_events_and_rolls_up_session(tmp_path: Path) -> None:
    conn = _connect(tmp_path)
    try:
        _seed_grok_session(conn, session_id="s1", sid="sess-uuid-1")
        log = _fixtures() / "basic.jsonl"
        result = harvest_grok_unified_log(conn, log)

        # 5 lines: 3 inference_done with sid, 1 other msg, 1 missing sid
        assert result.events_upserted == 3
        count = conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()
        assert count == (3,)

        row = conn.execute(
            """
            SELECT input_tokens, cached_input_tokens, output_tokens
            FROM session_state WHERE session_id = 's1'
            """
        ).fetchone()
        # sess-uuid-1: prompt 100+200, cached 40+150, completion 10+20
        assert row == (300, 190, 30)
    finally:
        conn.close()


def test_harvest_is_idempotent(tmp_path: Path) -> None:
    conn = _connect(tmp_path)
    try:
        _seed_grok_session(conn, session_id="s1", sid="sess-uuid-1")
        log = _fixtures() / "basic.jsonl"
        harvest_grok_unified_log(conn, log)
        second = harvest_grok_unified_log(conn, log)
        assert second.events_upserted == 0
        assert conn.execute("SELECT COUNT(*) FROM usage_events").fetchone() == (3,)
        row = conn.execute(
            "SELECT input_tokens, output_tokens FROM session_state WHERE session_id = 's1'"
        ).fetchone()
        assert row == (300, 30)
    finally:
        conn.close()


def _wal_bytes(tmp_path: Path) -> int:
    wal = tmp_path / "recall.duckdb.wal"
    return wal.stat().st_size if wal.exists() else 0


def test_harvest_of_unchanged_log_writes_nothing(tmp_path: Path) -> None:
    """The daemon harvests every discovery tick, so a tick with nothing new must
    not write (REQ-USAGE-014).

    Two recall sessions share one sid when a Grok workspace moves: the old path
    stays indexed beside the new one. The events' session link must settle on
    one of them instead of being rewritten on every tick.
    """
    conn = _connect(tmp_path)
    try:
        _seed_grok_session(conn, session_id="s-old", sid="sess-uuid-1", workspace="/old")
        _seed_grok_session(conn, session_id="s-new", sid="sess-uuid-1", workspace="/new")
        log = _fixtures() / "basic.jsonl"
        harvest_grok_unified_log(conn, log)
        conn.execute("CHECKPOINT")

        second = harvest_grok_unified_log(conn, log)

        assert second.events_upserted == 0
        assert _wal_bytes(tmp_path) == 0
        rows = conn.execute(
            """
            SELECT session_id, input_tokens, cached_input_tokens, output_tokens
            FROM session_state ORDER BY session_id
            """
        ).fetchall()
        assert rows == [("s-new", 300, 190, 30), ("s-old", 300, 190, 30)]
        links = conn.execute(
            "SELECT DISTINCT session_id FROM usage_events WHERE source_session_id = 'sess-uuid-1'"
        ).fetchall()
        assert len(links) == 1
        assert links[0][0] in {"s-new", "s-old"}
    finally:
        conn.close()


def test_harvest_restores_tokens_a_reindex_cleared(tmp_path: Path) -> None:
    """A session row rewritten without tokens regains them on the next tick,
    even when the log has nothing new (REQ-USAGE-014, REQ-USAGE-015)."""
    conn = _connect(tmp_path)
    try:
        _seed_grok_session(conn, session_id="s1", sid="sess-uuid-1")
        log = _fixtures() / "basic.jsonl"
        harvest_grok_unified_log(conn, log)
        conn.execute(
            """
            UPDATE session_state
            SET input_tokens = NULL, cached_input_tokens = NULL, output_tokens = NULL
            WHERE session_id = 's1'
            """
        )

        harvest_grok_unified_log(conn, log)

        row = conn.execute(
            """
            SELECT input_tokens, cached_input_tokens, output_tokens
            FROM session_state WHERE session_id = 's1'
            """
        ).fetchone()
        assert row == (300, 190, 30)
    finally:
        conn.close()


def test_harvest_rotation_does_not_double_count(tmp_path: Path) -> None:
    conn = _connect(tmp_path)
    try:
        _seed_grok_session(conn, session_id="s1", sid="sess-uuid-1")
        log = tmp_path / "unified.jsonl"
        first = (
            '{"ts":"2026-07-22T03:21:43.681Z","src":"shell","sid":"sess-uuid-1",'
            '"msg":"shell.turn.inference_done","ctx":{"loop_index":1,"prompt_tokens":100,'
            '"cached_prompt_tokens":40,"completion_tokens":10,"reasoning_tokens":0}}\n'
        )
        log.write_text(first, encoding="utf-8")
        r1 = harvest_grok_unified_log(conn, log)
        assert r1.events_upserted == 1

        # Append then rotate (shrink) with the same semantic event + a new one
        second_line = (
            '{"ts":"2026-07-22T03:21:49.373Z","src":"shell","sid":"sess-uuid-1",'
            '"msg":"shell.turn.inference_done","ctx":{"loop_index":2,"prompt_tokens":200,'
            '"cached_prompt_tokens":150,"completion_tokens":20,"reasoning_tokens":0}}\n'
        )
        log.write_text(first + second_line, encoding="utf-8")
        harvest_grok_unified_log(conn, log)

        # Rotate: rewrite file smaller with only first event content again + new third
        third = (
            '{"ts":"2026-07-22T04:00:00.000Z","src":"shell","sid":"sess-uuid-1",'
            '"msg":"shell.turn.inference_done","ctx":{"loop_index":3,"prompt_tokens":50,'
            '"cached_prompt_tokens":10,"completion_tokens":5,"reasoning_tokens":0}}\n'
        )
        log.write_text(first + third, encoding="utf-8")
        r3 = harvest_grok_unified_log(conn, log)
        assert r3.rotated is True

        # Events: loop1, loop2 (from pre-rotate), loop3 — loop1 not duplicated
        assert conn.execute("SELECT COUNT(*) FROM usage_events").fetchone() == (3,)
        row = conn.execute(
            """
            SELECT input_tokens, cached_input_tokens, output_tokens
            FROM session_state WHERE session_id = 's1'
            """
        ).fetchone()
        # 100+200+50, 40+150+10, 10+20+5
        assert row == (350, 200, 35)
    finally:
        conn.close()


def test_rollup_joins_session_id_on_events(tmp_path: Path) -> None:
    conn = _connect(tmp_path)
    try:
        log = _fixtures() / "basic.jsonl"
        harvest_grok_unified_log(conn, log)
        # No session yet — events orphan
        assert conn.execute(
            "SELECT COUNT(*) FROM usage_events WHERE session_id IS NULL"
        ).fetchone() == (3,)
        _seed_grok_session(conn, session_id="s1", sid="sess-uuid-1")
        n = rollup_grok_sessions(conn)
        assert n >= 1
        assert conn.execute(
            "SELECT COUNT(*) FROM usage_events WHERE session_id = 's1'"
        ).fetchone() == (2,)
    finally:
        conn.close()


def test_harvest_reports_absent_log(tmp_path: Path) -> None:
    """An absent Grok log must be distinguishable from a no-op harvest.

    Both cases return zero counters, so without `log_present` a daemon whose
    log never appeared looks identical to one that is fully caught up.
    """
    conn = _connect(tmp_path)
    try:
        missing = tmp_path / "nope" / "unified.jsonl"
        absent = harvest_grok_unified_log(conn, missing)
        assert absent.log_present is False
        assert absent.events_seen == 0

        present = harvest_grok_unified_log(conn, _fixtures() / "basic.jsonl")
        assert present.log_present is True
    finally:
        conn.close()


def test_index_run_harvest_failure_carries_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """REQ-RESIL-023: the harvest swallowed by an index run is the third
    site logging `usage harvest failed`; it must name its site and carry the
    traceback like the daemon-resident ones."""
    import logging

    import recall.services.usage_harvest as harvest_module
    from recall.services.indexer import index_sessions

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / ".config/recall/config.toml"))
    monkeypatch.delenv("RECALL_DB_PATH", raising=False)
    monkeypatch.delenv("RECALL_LOCK_PATH", raising=False)

    def _short_row(*_args: object, **_kwargs: object) -> None:
        raise IndexError("tuple index out of range")

    monkeypatch.setattr(harvest_module, "harvest_grok_unified_log", _short_row)

    with caplog.at_level(logging.WARNING, logger="recall.indexer"):
        index_sessions(source=None, full=False, recreate=True, verbose=False, embed=False)

    record = next(r for r in caplog.records if "usage harvest failed" in r.getMessage())
    assert "index run" in record.getMessage()
    assert record.exc_info is not None
    assert record.exc_info[0] is IndexError
