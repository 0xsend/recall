"""Writing tool results and their harness id mapping (REQ-LIVE-006).

Both tables are insert-only on purpose. `tool_calls` is the row set whose
churn produced the 140 GiB `tool_call_embeddings` bloat (REQ-INDEX-017), and a
DuckDB UPDATE is DELETE+INSERT, so the pairing key had to land somewhere that
a re-parse never rewrites.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.core.types import DaemonMode
from recall.services.indexer import index_sessions

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


def _physical_row_versions(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    row = conn.execute(
        f"""
        SELECT MIN(column_rows) FROM (
            SELECT SUM(count) AS column_rows
            FROM pragma_storage_info('{table}')
            WHERE segment_type != 'VALIDITY'
            GROUP BY column_name
        )
        """
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def test_paired_tool_result_is_written_against_the_recall_tool_call_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl")

    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = duckdb.connect(str(_app_config(tmp_path).db_path), read_only=True)
    try:
        row = conn.execute(
            """
            SELECT tc.tool_name, tr.result_summary, tr.is_error
            FROM tool_results tr
            JOIN tool_calls tc ON tc.id = tr.tool_call_id
            """
        ).fetchone()
        assert row == ("Bash", "312 passed in 41.02s", False)
        assert conn.execute("SELECT tool_use_id FROM tool_use_ids").fetchall() == [
            ("toolu_end_bash",)
        ]
    finally:
        conn.close()


def test_failed_tool_result_keeps_its_error_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_tool_error.jsonl")

    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = duckdb.connect(str(_app_config(tmp_path).db_path), read_only=True)
    try:
        assert conn.execute("SELECT is_error FROM tool_results").fetchall() == [(True,)]
    finally:
        conn.close()


def test_unanswered_tool_use_writes_a_mapping_but_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session cut mid-tool is exactly the live case: the call is open."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_mid_tool.jsonl")

    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = duckdb.connect(str(_app_config(tmp_path).db_path), read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM tool_results").fetchone() == (0,)
        assert conn.execute("SELECT tool_use_id FROM tool_use_ids").fetchall() == [
            ("toolu_mid_bash",)
        ]
    finally:
        conn.close()


def test_result_arriving_in_a_later_chunk_still_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live case: the daemon indexes between the tool_use and its result."""
    monkeypatch.setenv("HOME", str(tmp_path))
    dest = _install(tmp_path, "live_mid_tool.jsonl")

    index_sessions(source=None, full=False, recreate=True, verbose=False)

    lines = (FIXTURES / "claude_code" / "live_end_turn.jsonl").read_text().splitlines()
    result_line = json.loads(lines[2])
    with dest.open("a", encoding="utf-8") as handle:
        result_line["message"]["content"][0]["tool_use_id"] = "toolu_mid_bash"
        handle.write(json.dumps(result_line) + "\n")

    index_sessions(source=None, full=False, recreate=False, verbose=False)

    conn = duckdb.connect(str(_app_config(tmp_path).db_path), read_only=True)
    try:
        row = conn.execute(
            """
            SELECT tc.tool_name, tr.result_summary
            FROM tool_results tr
            JOIN tool_calls tc ON tc.id = tr.tool_call_id
            """
        ).fetchone()
        assert row == ("Bash", "312 passed in 41.02s")
    finally:
        conn.close()


def test_reindexing_an_unchanged_session_does_not_churn_row_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INDEX-017: an insert-only table must not accumulate dead versions."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl")
    db_path = _app_config(tmp_path).db_path

    index_sessions(source=None, full=True, recreate=True, verbose=False)
    conn = duckdb.connect(str(db_path))
    conn.execute("CHECKPOINT")
    baseline = {
        table: _physical_row_versions(conn, table) for table in ("tool_results", "tool_use_ids")
    }
    conn.close()

    for _ in range(5):
        index_sessions(source=None, full=True, recreate=False, verbose=False)

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("CHECKPOINT")
        for table, before in baseline.items():
            assert _physical_row_versions(conn, table) == before, table
        assert conn.execute("SELECT COUNT(*) FROM tool_results").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM tool_use_ids").fetchone() == (1,)
    finally:
        conn.close()


def test_deleting_a_session_cascades_to_the_live_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DuckDB has no ON DELETE CASCADE, so the cascade is manual and must exist."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    from recall.db.queries import delete_session

    conn = duckdb.connect(str(_app_config(tmp_path).db_path))
    try:
        session_row = conn.execute("SELECT id FROM sessions").fetchone()
        assert session_row is not None
        session_id = session_row[0]
        assert conn.execute("SELECT COUNT(*) FROM tool_results").fetchone() == (1,)

        delete_session(conn, session_id)

        assert conn.execute("SELECT COUNT(*) FROM tool_results").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM tool_use_ids").fetchone() == (0,)
    finally:
        conn.close()


def test_full_reparse_repoints_a_reused_tool_call_id_at_the_new_tool_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rewritten transcript reuses the positional tool_call id for a different call.

    recall's tool_call id is a hash of the position, so the row at that message
    and block index keeps its id when the file is rewritten. Insert-only across
    a full re-parse would leave the mapping pointing at the old harness id and
    hand the previous call's success back as this call's failure.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    dest = _install(tmp_path, "live_end_turn.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    lines = [json.loads(line) for line in dest.read_text().splitlines()]
    lines[1]["message"]["content"][1]["id"] = "toolu_end_bash_v2"
    lines[1]["message"]["content"][1]["input"]["command"] = "uv run pytest -q tests/x"
    lines[2]["message"]["content"][0]["tool_use_id"] = "toolu_end_bash_v2"
    lines[2]["message"]["content"][0]["content"] = "1 failed, 311 passed"
    lines[2]["message"]["content"][0]["is_error"] = True
    dest.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")

    index_sessions(source=None, full=True, recreate=False, verbose=False)

    conn = duckdb.connect(str(_app_config(tmp_path).db_path), read_only=True)
    try:
        assert conn.execute(
            """
            SELECT tr.result_summary, tr.is_error
            FROM tool_results tr
            JOIN tool_calls tc ON tc.id = tr.tool_call_id
            """
        ).fetchall() == [("1 failed, 311 passed", True)]
        assert conn.execute("SELECT tool_use_id FROM tool_use_ids").fetchall() == [
            ("toolu_end_bash_v2",)
        ]
    finally:
        conn.close()


def test_full_reparse_that_drops_a_tool_use_removes_its_mapping_and_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A removed tool_call must not leave a result behind for the next call at that id."""
    monkeypatch.setenv("HOME", str(tmp_path))
    dest = _install(tmp_path, "live_end_turn.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    lines = [json.loads(line) for line in dest.read_text().splitlines()]
    lines[1]["message"]["content"] = [{"type": "text", "text": "Never mind, skipping the run."}]
    lines[2]["message"]["content"] = [{"type": "text", "text": "fine"}]
    lines[2].pop("toolUseResult", None)
    dest.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")

    index_sessions(source=None, full=True, recreate=False, verbose=False)

    conn = duckdb.connect(str(_app_config(tmp_path).db_path), read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM tool_use_ids").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM tool_results").fetchone() == (0,)
    finally:
        conn.close()


def test_full_reparse_backfills_a_session_indexed_before_the_live_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every session predating migration 0024 has empty tail-fact tables.

    A full re-parse is the documented way to pick up parser additions, so it
    has to write them; the sync path is the one a session already in the index
    takes.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl")
    db_path = _app_config(tmp_path).db_path
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = duckdb.connect(str(db_path))
    conn.execute("DELETE FROM tool_results")
    conn.execute("DELETE FROM tool_use_ids")
    conn.close()

    index_sessions(source=None, full=True, recreate=False, verbose=False)

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        assert conn.execute("SELECT tool_use_id FROM tool_use_ids").fetchall() == [
            ("toolu_end_bash",)
        ]
        assert conn.execute("SELECT result_summary FROM tool_results").fetchall() == [
            ("312 passed in 41.02s",)
        ]
    finally:
        conn.close()


def test_deleting_a_session_reaches_a_result_whose_tool_call_row_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cascade is scoped by the mapping, which is what makes it complete.

    Scoping it by `tool_calls` instead leaves any result whose call row was
    already removed behind forever — unreachable by every delete path, and
    positional ids mean the next call at that id inherits it.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    from recall.db.queries import delete_session

    conn = duckdb.connect(str(_app_config(tmp_path).db_path))
    try:
        session_row = conn.execute("SELECT id FROM sessions").fetchone()
        assert session_row is not None
        session_id = session_row[0]
        conn.execute("DELETE FROM tool_calls")

        delete_session(conn, session_id)

        assert conn.execute("SELECT COUNT(*) FROM tool_results").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM tool_use_ids").fetchone() == (0,)
    finally:
        conn.close()
