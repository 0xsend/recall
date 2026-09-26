"""Persisting the stop markers turn state is derived from (REQ-LIVE-005).

The parser reads a harness stop reason at index time and the derivation needs
it at read time, so something has to hold it in between. It is a side table for
the same reason `tool_use_ids` is: a column on `messages` or `tool_calls` would
be NULL on every historical row and non-NULL after one re-parse, and a DuckDB
UPDATE is DELETE+INSERT (REQ-INDEX-017).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.core.types import DaemonMode
from recall.services.indexer import index_sessions
from recall.services.live import fetch_last_stop_marker, fetch_open_tool_calls

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _app_config(tmp_path: Path) -> AppConfig:
    data_dir = tmp_path / ".local" / "share" / "recall"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / ".config" / "recall" / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(mode=DaemonMode.POLL),
        cli=CliConfig(),
    )


def _install(tmp_path: Path, fixture_name: str) -> Path:
    projects = tmp_path / ".claude" / "projects" / "proj"
    projects.mkdir(parents=True, exist_ok=True)
    dest = projects / "live.jsonl"
    shutil.copy(FIXTURES / "claude_code" / fixture_name, dest)
    return dest


def _connect(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(_app_config(tmp_path).db_path), read_only=True)


def _session_id(conn: duckdb.DuckDBPyConnection) -> str:
    row = conn.execute("SELECT id FROM sessions").fetchone()
    assert row is not None
    return str(row[0])


def test_an_end_of_turn_stop_is_stored_with_the_message_it_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl")

    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        marker = fetch_last_stop_marker(conn, _session_id(conn))
        assert marker is not None
        assert marker.reason == "end_turn"
        assert marker.ends_turn is True
    finally:
        conn.close()


def test_a_mid_turn_stop_is_stored_but_does_not_claim_to_end_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`tool_use` is Claude Code's mid-turn reason; only the parser knows that."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_mid_tool.jsonl")

    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        marker = fetch_last_stop_marker(conn, _session_id(conn))
        assert marker is not None
        assert marker.reason == "tool_use"
        assert marker.ends_turn is False
    finally:
        conn.close()


def test_the_newest_marker_wins_when_a_session_has_several(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl")

    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        stored = conn.execute(
            "SELECT message_idx, reason FROM session_stop_markers ORDER BY message_idx"
        ).fetchall()
        assert stored == [(1, "tool_use"), (3, "end_turn")]
        marker = fetch_last_stop_marker(conn, _session_id(conn))
        assert marker is not None
        assert marker.idx == 3
    finally:
        conn.close()


def test_a_marker_above_the_surviving_messages_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rewritten, shorter transcript must not leave a dangling marker in force."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    write_conn = duckdb.connect(str(_app_config(tmp_path).db_path))
    try:
        session_id = _session_id(write_conn)
        write_conn.execute("DELETE FROM messages WHERE idx >= 2")
        marker = fetch_last_stop_marker(write_conn, session_id)
        assert marker is not None
        assert marker.idx == 1
        assert marker.reason == "tool_use"
    finally:
        write_conn.close()


def test_reindexing_an_unchanged_session_adds_no_marker_row_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        before = conn.execute("SELECT COUNT(*) FROM session_stop_markers").fetchone()
    finally:
        conn.close()

    for _ in range(3):
        index_sessions(source=None, full=True, recreate=False, verbose=False)

    conn = _connect(tmp_path)
    try:
        after = conn.execute("SELECT COUNT(*) FROM session_stop_markers").fetchone()
        assert before == after == (2,)
    finally:
        conn.close()


def test_deleting_a_session_takes_its_stop_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    write_conn = duckdb.connect(str(_app_config(tmp_path).db_path))
    try:
        from recall.db.queries import delete_session

        delete_session(write_conn, _session_id(write_conn))
        assert write_conn.execute("SELECT COUNT(*) FROM session_stop_markers").fetchone() == (0,)
    finally:
        write_conn.close()


def test_an_unanswered_tool_call_reads_as_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_mid_tool.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        open_calls = fetch_open_tool_calls(conn, _session_id(conn), limit=10)
        assert [call.tool_call.tool_name for call in open_calls] == ["Bash"]
        assert open_calls[0].message_idx == 1
        assert open_calls[0].started_at is not None
    finally:
        conn.close()


def test_an_answered_tool_call_is_not_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        assert fetch_open_tool_calls(conn, _session_id(conn), limit=10) == []
    finally:
        conn.close()


def test_a_tool_call_recall_cannot_pair_is_not_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No harness id means no way to know an answer arrived — `unknown`, not `working`."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_mid_tool.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    write_conn = duckdb.connect(str(_app_config(tmp_path).db_path))
    try:
        session_id = _session_id(write_conn)
        write_conn.execute("DELETE FROM tool_use_ids")
        assert fetch_open_tool_calls(write_conn, session_id, limit=10) == []
    finally:
        write_conn.close()


def test_fetch_open_tool_calls_rejects_a_non_positive_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_mid_tool.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        with pytest.raises(ValueError, match="limit"):
            fetch_open_tool_calls(conn, _session_id(conn), limit=0)
    finally:
        conn.close()
