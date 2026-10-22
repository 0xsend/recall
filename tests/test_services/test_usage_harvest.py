from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.services.analytics import usage_by_source
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
        assert rows == [("s-new", 300, 190, 30), ("s-old", None, None, None)]
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


def _usage_totals(conn: duckdb.DuckDBPyConnection) -> tuple[int, int, int]:
    """Grok input, cached and output tokens as `recall stats usage` sums them."""
    stats = [stat for stat in usage_by_source(conn=conn) if stat.source == "grok"]
    return (
        sum(stat.input_tokens for stat in stats),
        sum(stat.cached_input_tokens for stat in stats),
        sum(stat.output_tokens for stat in stats),
    )


# basic.jsonl's sess-uuid-1 events: prompt 100+200, cached 40+150, completion 10+20.
_BASIC_SID_TOTALS = (300, 190, 30)


def test_a_moved_workspace_counts_harvested_usage_once(tmp_path: Path) -> None:
    """The same sid indexed under its old and its new workspace path counts its
    harvested usage once (REQ-USAGE-014)."""
    conn = _connect(tmp_path)
    try:
        log = _fixtures() / "basic.jsonl"
        _seed_grok_session(conn, session_id="s-old", sid="sess-uuid-1", workspace="/old")
        harvest_grok_unified_log(conn, log)
        _seed_grok_session(conn, session_id="s-new", sid="sess-uuid-1", workspace="/new")

        harvest_grok_unified_log(conn, log)

        assert _usage_totals(conn) == _BASIC_SID_TOTALS
    finally:
        conn.close()


def test_usage_stored_on_every_session_of_a_sid_is_counted_once(tmp_path: Path) -> None:
    """A database where each session of a moved sid carries the sid's totals
    counts them once after the next harvest, even with nothing new in the log."""
    conn = _connect(tmp_path)
    try:
        log = _fixtures() / "basic.jsonl"
        _seed_grok_session(conn, session_id="s-old", sid="sess-uuid-1", workspace="/old")
        _seed_grok_session(conn, session_id="s-new", sid="sess-uuid-1", workspace="/new")
        harvest_grok_unified_log(conn, log)
        conn.execute(
            """
            UPDATE session_state
            SET input_tokens = 300, cached_input_tokens = 190, output_tokens = 30
            """
        )

        harvest_grok_unified_log(conn, log)

        assert _usage_totals(conn) == _BASIC_SID_TOTALS
    finally:
        conn.close()


def test_a_moved_sid_keeps_its_usage_when_the_session_carrying_it_is_removed(
    tmp_path: Path,
) -> None:
    conn = _connect(tmp_path)
    try:
        log = _fixtures() / "basic.jsonl"
        _seed_grok_session(conn, session_id="s-a", sid="sess-uuid-1", workspace="/old")
        _seed_grok_session(conn, session_id="s-b", sid="sess-uuid-1", workspace="/new")
        harvest_grok_unified_log(conn, log)
        [(carrier,)] = conn.execute(
            "SELECT session_id FROM session_state WHERE input_tokens IS NOT NULL"
        ).fetchall()
        conn.execute("DELETE FROM session_state WHERE session_id = ?", [carrier])
        conn.execute("DELETE FROM sessions WHERE id = ?", [carrier])

        harvest_grok_unified_log(conn, log)

        assert _usage_totals(conn) == _BASIC_SID_TOTALS
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


def _inference_line(*, sid: str, loop_index: int, ts: str = "2026-07-22T03:21:43.681Z") -> str:
    return (
        f'{{"ts":"{ts}","src":"shell","sid":"{sid}",'
        f'"msg":"shell.turn.inference_done","ctx":{{"loop_index":{loop_index},'
        f'"prompt_tokens":{loop_index * 10},"cached_prompt_tokens":{loop_index},'
        f'"completion_tokens":{loop_index * 2},"reasoning_tokens":0}}}}\n'
    )


def _session_totals(conn: duckdb.DuckDBPyConnection) -> list[tuple[object, ...]]:
    return conn.execute(
        """
        SELECT session_id, input_tokens, cached_input_tokens, output_tokens
        FROM session_state ORDER BY session_id
        """
    ).fetchall()


def test_harvest_drains_a_backlog_across_bounded_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One call is one writer turn: it reads a bounded stretch of log and a
    larger backlog drains over later calls to the same totals."""
    import recall.services.usage_harvest as harvest_module

    log = tmp_path / "unified.jsonl"
    lines = [_inference_line(sid="sess-uuid-1", loop_index=i) for i in range(1, 11)]
    log.write_text("".join(lines), encoding="utf-8")

    (tmp_path / "whole").mkdir()
    whole = _connect(tmp_path / "whole")
    try:
        _seed_grok_session(whole, session_id="s1", sid="sess-uuid-1")
        assert harvest_grok_unified_log(whole, log).caught_up is True
        expected = _session_totals(whole)
    finally:
        whole.close()

    monkeypatch.setattr(harvest_module, "MAX_LOG_BYTES_PER_BATCH", len(lines[0]) * 3)
    conn = _connect(tmp_path)
    try:
        _seed_grok_session(conn, session_id="s1", sid="sess-uuid-1")
        results = []
        while not results or not results[-1].caught_up:
            results.append(harvest_grok_unified_log(conn, log))
            assert len(results) <= len(lines)

        assert [r.events_upserted for r in results] == [3, 3, 3, 1]
        assert conn.execute("SELECT COUNT(*) FROM usage_events").fetchone() == (10,)
        assert _session_totals(conn) == expected == [("s1", 550, 55, 110)]
    finally:
        conn.close()


def test_harvest_leaves_a_partial_final_line_for_the_next_call(tmp_path: Path) -> None:
    """Grok may be mid-write: a last line without its newline is not consumed
    as malformed, and is harvested once it is complete."""
    conn = _connect(tmp_path)
    try:
        _seed_grok_session(conn, session_id="s1", sid="sess-uuid-1")
        log = tmp_path / "unified.jsonl"
        first = _inference_line(sid="sess-uuid-1", loop_index=1)
        second = _inference_line(sid="sess-uuid-1", loop_index=2)
        log.write_text(first + second[:40], encoding="utf-8")

        partial = harvest_grok_unified_log(conn, log)
        assert (partial.events_upserted, partial.events_skipped) == (1, 0)

        log.write_text(first + second, encoding="utf-8")
        completed = harvest_grok_unified_log(conn, log)

        assert (completed.events_upserted, completed.events_skipped) == (1, 0)
        assert _session_totals(conn) == [("s1", 30, 3, 6)]
    finally:
        conn.close()


def test_harvest_counts_a_repeated_line_within_one_call_once(tmp_path: Path) -> None:
    conn = _connect(tmp_path)
    try:
        _seed_grok_session(conn, session_id="s1", sid="sess-uuid-1")
        log = tmp_path / "unified.jsonl"
        line = _inference_line(sid="sess-uuid-1", loop_index=4)
        log.write_text(line + line, encoding="utf-8")

        result = harvest_grok_unified_log(conn, log)

        assert (result.events_seen, result.events_upserted) == (2, 1)
        assert _session_totals(conn) == [("s1", 40, 4, 8)]
    finally:
        conn.close()


def test_harvest_failure_leaves_events_and_cursor_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Events, cursor and rollup commit together, so a failed call is retried
    whole by the next one rather than skipping past unsaved events."""
    import recall.services.usage_harvest as harvest_module

    conn = _connect(tmp_path)
    try:
        _seed_grok_session(conn, session_id="s1", sid="sess-uuid-1")
        log = _fixtures() / "basic.jsonl"

        def failing_rollup(_conn: duckdb.DuckDBPyConnection) -> int:
            raise duckdb.ConversionException("rollup failed")

        with monkeypatch.context() as patch:
            patch.setattr(harvest_module, "rollup_grok_sessions", failing_rollup)
            with pytest.raises(duckdb.ConversionException):
                harvest_grok_unified_log(conn, log)

        assert conn.execute("SELECT COUNT(*) FROM usage_events").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM usage_log_cursors").fetchone() == (0,)
        assert harvest_grok_unified_log(conn, log).events_upserted == 3
        assert _session_totals(conn) == [("s1", 300, 190, 30)]
    finally:
        conn.close()


def test_harvest_stores_timestamps_as_bound_datetimes_are_stored(tmp_path: Path) -> None:
    """Offset timestamps are stored in the session time zone and offset-less
    ones as written, matching what a bound datetime parameter stores."""
    from datetime import datetime

    conn = _connect(tmp_path)
    try:
        log = tmp_path / "unified.jsonl"
        log.write_text(
            _inference_line(sid="a", loop_index=1, ts="2026-07-22T03:21:43.681Z")
            + _inference_line(sid="b", loop_index=1, ts="2026-07-22T03:21:43.681"),
            encoding="utf-8",
        )

        harvest_grok_unified_log(conn, log)

        stored = dict(conn.execute("SELECT source_session_id, ts FROM usage_events").fetchall())
        bound = conn.execute(
            "SELECT ?::TIMESTAMP, ?::TIMESTAMP",
            [
                datetime.fromisoformat("2026-07-22T03:21:43.681+00:00"),
                datetime.fromisoformat("2026-07-22T03:21:43.681"),
            ],
        ).fetchone()
        assert (stored["a"], stored["b"]) == bound
    finally:
        conn.close()


def test_requested_harvest_drains_the_log_in_bounded_writer_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, socket_dir: Path
) -> None:
    """`recall index` leaves the whole log harvested even though each writer
    turn reads only a bounded stretch of it (REQ-USAGE-010)."""
    import asyncio
    from dataclasses import replace

    import recall.services.usage_harvest as harvest_module
    from recall.core.config import AppConfig, DaemonConfig
    from recall.services.rpc_server import RpcServer

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", "data")
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.delenv("RECALL_DB_PATH", raising=False)
    monkeypatch.delenv("RECALL_LOCK_PATH", raising=False)
    config = replace(AppConfig.load(), daemon=DaemonConfig(embed=False))
    config.data_dir.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "unified.jsonl"
    lines = [_inference_line(sid="sess-uuid-1", loop_index=i) for i in range(1, 8)]
    log.write_text("".join(lines), encoding="utf-8")
    monkeypatch.setattr(harvest_module, "MAX_LOG_BYTES_PER_BATCH", len(lines[0]) * 2)
    server = RpcServer(config=config)
    try:
        asyncio.run(server._harvest_usage_until_caught_up(log, None, "requested harvest"))
        stored = server._get_conn().execute("SELECT COUNT(*) FROM usage_events").fetchone()
    finally:
        if server._conn is not None:
            server._conn.close()

    assert stored == (7,)


def test_harvest_keeps_an_oversized_valid_inference_event(tmp_path: Path) -> None:
    """Line length is not part of the input contract: a valid inference event
    padded past any buffer size is still harvested with its tokens."""
    conn = _connect(tmp_path)
    try:
        _seed_grok_session(conn, session_id="s1", sid="sess-uuid-1")
        log = tmp_path / "unified.jsonl"
        oversized = (
            '{"ts":"2026-07-22T03:21:43.681Z","src":"shell","sid":"sess-uuid-1",'
            '"msg":"shell.turn.inference_done","pad":"' + "x" * (8 * 1024 * 1024) + '",'
            '"ctx":{"loop_index":1,"prompt_tokens":7,"cached_prompt_tokens":2,'
            '"completion_tokens":3,"reasoning_tokens":0}}\n'
        )
        log.write_text(
            oversized + _inference_line(sid="sess-uuid-1", loop_index=2), encoding="utf-8"
        )

        results = [harvest_grok_unified_log(conn, log)]
        while not results[-1].caught_up:
            results.append(harvest_grok_unified_log(conn, log))
            assert len(results) <= 2

        assert sum(r.events_upserted for r in results) == 2
        assert sum(r.events_skipped for r in results) == 0
        assert _session_totals(conn) == [("s1", 27, 4, 7)]
    finally:
        conn.close()


def test_harvest_waits_for_an_oversized_final_line_to_end(tmp_path: Path) -> None:
    conn = _connect(tmp_path)
    try:
        log = tmp_path / "unified.jsonl"
        first = _inference_line(sid="sess-uuid-1", loop_index=1)
        second = _inference_line(sid="sess-uuid-1", loop_index=2)
        padded = second[:-2] + "," + '"pad":"' + "x" * 200_000 + '"}\n'
        log.write_text(first + padded[:100_000], encoding="utf-8")

        result = harvest_grok_unified_log(conn, log)

        assert (result.events_upserted, result.events_skipped, result.caught_up) == (1, 0, True)
        log.write_text(first + padded, encoding="utf-8")
        assert harvest_grok_unified_log(conn, log).events_upserted == 1
    finally:
        conn.close()


def test_a_writer_batch_holds_at_most_the_event_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer commits parsed batches only, each with at most
    MAX_EVENTS_PER_BATCH events, however the log is shaped."""
    import recall.services.usage_harvest as harvest_module
    from recall.services.usage_harvest import commit_grok_usage_batch, read_grok_usage_batch

    monkeypatch.setattr(harvest_module, "MAX_EVENTS_PER_BATCH", 4)
    conn = _connect(tmp_path)
    try:
        _seed_grok_session(conn, session_id="s1", sid="sess-uuid-1")
        log = tmp_path / "unified.jsonl"
        log.write_text(
            "".join(_inference_line(sid="sess-uuid-1", loop_index=i) for i in range(1, 11)),
            encoding="utf-8",
        )
        sizes = []
        while True:
            batch = read_grok_usage_batch(conn, log)
            assert batch is not None
            sizes.append(len(batch.events))
            if commit_grok_usage_batch(conn, batch, host=None).caught_up:
                break
            assert len(sizes) <= 10

        assert sizes == [4, 4, 2]
        assert _session_totals(conn) == [("s1", 550, 55, 110)]
    finally:
        conn.close()


def test_a_batch_read_before_another_harvest_commits_nothing(tmp_path: Path) -> None:
    """Reading runs outside the writer, so two harvests can read the same
    stretch; the second commit finds the cursor moved and writes nothing."""
    from recall.services.usage_harvest import commit_grok_usage_batch, read_grok_usage_batch

    conn = _connect(tmp_path)
    try:
        _seed_grok_session(conn, session_id="s1", sid="sess-uuid-1")
        log = _fixtures() / "basic.jsonl"
        first = read_grok_usage_batch(conn, log)
        stale = read_grok_usage_batch(conn, log)
        assert first is not None and stale is not None
        commit_grok_usage_batch(conn, first, host=None)

        late = commit_grok_usage_batch(conn, stale, host=None)

        assert (late.events_upserted, late.caught_up) == (0, False)
        assert conn.execute("SELECT COUNT(*) FROM usage_events").fetchone() == (3,)
        assert harvest_grok_unified_log(conn, log).caught_up is True
        assert _session_totals(conn) == [("s1", 300, 190, 30)]
    finally:
        conn.close()


def test_index_run_harvests_the_whole_log_across_bounded_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-shot `recall index` has no watch tick to finish later, so it
    keeps calling until the log present at its start is harvested."""
    import recall.services.usage_harvest as harvest_module
    from recall.services.indexer import index_sessions

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / ".config/recall/config.toml"))
    monkeypatch.delenv("RECALL_DB_PATH", raising=False)
    monkeypatch.delenv("RECALL_LOCK_PATH", raising=False)
    log = tmp_path / ".grok" / "logs" / "unified.jsonl"
    log.parent.mkdir(parents=True)
    lines = [_inference_line(sid="sess-uuid-1", loop_index=i) for i in range(1, 10)]
    log.write_text("".join(lines), encoding="utf-8")
    monkeypatch.setattr(harvest_module, "MAX_LOG_BYTES_PER_BATCH", len(lines[0]) * 2)

    index_sessions(source=None, full=False, recreate=True, verbose=False, embed=False)

    conn = duckdb.connect(str(tmp_path / ".local/share/recall/recall.duckdb"))
    try:
        assert conn.execute("SELECT COUNT(*) FROM usage_events").fetchone() == (9,)
    finally:
        conn.close()


def test_a_shrink_after_growth_during_a_read_is_seen_as_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log grew between the size sample and the read, and the batch stopped
    inside that growth. A replacement no shorter than where the batch stopped
    but shorter than what the read saw must still restart from its beginning."""
    import os

    import recall.services.usage_harvest as harvest_module

    log = tmp_path / "unified.jsonl"
    first = [_inference_line(sid="sess-uuid-1", loop_index=i) for i in range(1, 3)]
    growth = [_inference_line(sid="sess-uuid-1", loop_index=i) for i in range(3, 7)]
    replacement = [_inference_line(sid="sess-uuid-2", loop_index=i) for i in range(1, 5)]
    log.write_text("".join(first), encoding="utf-8")

    real_open = Path.open
    grown = False

    def open_after_growth(self: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal grown
        if self == log and not grown:
            grown = True
            with real_open(log, "a", encoding="utf-8") as stream:
                stream.write("".join(growth))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(harvest_module, "MAX_EVENTS_PER_BATCH", 3)
    conn = _connect(tmp_path)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", open_after_growth)
            harvest_grok_unified_log(conn, log)
        stopped_at = sum(len(line) for line in first + growth[:1])
        grown_to = sum(len(line) for line in first + growth)
        swapped = tmp_path / "unified.jsonl.next"
        swapped.write_text("".join(replacement), encoding="utf-8")
        assert stopped_at <= swapped.stat().st_size < grown_to
        os.replace(swapped, log)

        while not harvest_grok_unified_log(conn, log).caught_up:
            pass

        assert conn.execute(
            "SELECT COUNT(*) FROM usage_events WHERE source_session_id = 'sess-uuid-2'"
        ).fetchone() == (len(replacement),)
    finally:
        conn.close()
