"""Tests for byte-offset incremental indexing (REQ-INDEX-013 through REQ-INDEX-016)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.core.types import DaemonMode
from recall.services.indexer import (
    DiscoveredSessionPath,
    SessionState,
    _discover_paths,
    _resolve_parse_offset,
    index_sessions,
)


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


def _codex_token_count_event(input_tokens: int, output_tokens: int) -> str:
    return json.dumps(
        {
            "type": "event_msg",
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                    "last_token_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                },
            },
        }
    )


def _fetch_session_tokens(db_path: Path) -> tuple[int | None, int | None]:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        row = conn.execute(
            "SELECT input_tokens, output_tokens FROM session_state LIMIT 1"
        ).fetchone()
        assert row is not None
        return row[0], row[1]
    finally:
        conn.close()


class TestResolveParseOffset:
    def test_grown_file_returns_zero_until_catalog_validates_prefix(self) -> None:
        state = SessionState(
            file_mtime=1.0,
            file_size=1000,
            last_byte_offset=800,
            message_count=10,
            orphan_tool_count=0,
        )
        discovered = DiscoveredSessionPath(
            parser=cast(Any, None),
            path=Path("/fake"),
            resolved_path="/fake",
            file_mtime=2.0,
            file_size=1500,  # larger than stored offset
        )
        assert _resolve_parse_offset(state, discovered) == 0


class TestNewestFirstDiscovery:
    def test_discover_paths_sorted_by_mtime_descending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))

        # Create codex session files with different mtimes
        codex_dir = tmp_path / ".codex" / "sessions"
        codex_dir.mkdir(parents=True, exist_ok=True)
        fixture = (
            Path(__file__).resolve().parents[2]
            / "fixtures"
            / "codex"
            / "session1"
            / "rollout.jsonl"
        )

        for i, name in enumerate(["rollout_old.jsonl", "rollout_mid.jsonl", "rollout_new.jsonl"]):
            dest = codex_dir / name
            shutil.copy(fixture, dest)
            import os
            import time

            # Set mtime: old=1000, mid=2000, new=3000
            mtime = 1000.0 + i * 1000.0
            os.utime(dest, (mtime, mtime))
            time.sleep(0.01)  # ensure different inodes

        from recall.core.types import Source

        paths = _discover_paths(Source.CODEX, sources=None)
        mtimes = [p.file_mtime for p in paths]
        assert mtimes == sorted(mtimes, reverse=True), "paths should be sorted newest-first"


class TestIncrementalIndexing:
    def test_last_byte_offset_stored_and_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Index a session, append content, re-index — verify offset-based incremental parse."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # Create a session file
        codex_dir = tmp_path / ".codex" / "sessions" / "s1"
        codex_dir.mkdir(parents=True, exist_ok=True)
        fixture = (
            Path(__file__).resolve().parents[2]
            / "fixtures"
            / "codex"
            / "session1"
            / "rollout.jsonl"
        )
        dest = codex_dir / "rollout.jsonl"
        shutil.copy(fixture, dest)

        # First index
        first = index_sessions(source=None, full=False, recreate=True, verbose=False)
        assert first.indexed >= 1

        # Check last_byte_offset was stored
        config = _app_config(tmp_path)
        conn = duckdb.connect(str(config.db_path), read_only=True)
        try:
            row = conn.execute(
                "SELECT last_byte_offset, message_count FROM session_state LIMIT 1"
            ).fetchone()
            assert row is not None
            stored_offset = row[0]
            stored_msg_count = row[1]
            assert stored_offset > 0, "last_byte_offset should be set after indexing"
            assert stored_msg_count > 0
        finally:
            conn.close()

        # Append new content
        with dest.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": "2026-12-31T23:59:59Z",
                        "payload": {"type": "user_message", "message": "appended message"},
                    }
                )
                + "\n"
            )

        # Re-index (incremental)
        second = index_sessions(source=None, full=False, recreate=False, verbose=False)
        assert second.indexed >= 1

        # Verify message count increased
        conn = duckdb.connect(str(config.db_path), read_only=True)
        try:
            row = conn.execute(
                "SELECT last_byte_offset, message_count FROM session_state LIMIT 1"
            ).fetchone()
            assert row is not None
            new_offset = row[0]
            new_msg_count = row[1]
            assert new_offset > stored_offset, "offset should advance after appending"
            assert new_msg_count > stored_msg_count, "message count should increase"
        finally:
            conn.close()

    def test_truncated_file_triggers_full_reparse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If file shrinks below stored offset, full reparse occurs."""
        monkeypatch.setenv("HOME", str(tmp_path))

        codex_dir = tmp_path / ".codex" / "sessions" / "s1"
        codex_dir.mkdir(parents=True, exist_ok=True)
        fixture = (
            Path(__file__).resolve().parents[2]
            / "fixtures"
            / "codex"
            / "session1"
            / "rollout.jsonl"
        )
        dest = codex_dir / "rollout.jsonl"
        shutil.copy(fixture, dest)

        # Index
        index_sessions(source=None, full=False, recreate=True, verbose=False)

        # Truncate the file (write smaller content)
        dest.write_text(
            json.dumps(
                {
                    "type": "event_msg",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "payload": {"type": "user_message", "message": "truncated"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        # Re-index — should do full parse (no crash)
        result = index_sessions(source=None, full=False, recreate=False, verbose=False)
        assert result.indexed >= 1


class TestIncrementalOrphanToolCalls:
    """Regression: orphan tool call IDs must not collide on incremental append."""

    def test_appending_orphan_tool_call_does_not_duplicate_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))

        # Use session2 fixture which has 4 orphan tool calls
        codex_dir = tmp_path / ".codex" / "sessions" / "s1"
        codex_dir.mkdir(parents=True, exist_ok=True)
        fixture = (
            Path(__file__).resolve().parents[2]
            / "fixtures"
            / "codex"
            / "session2"
            / "rollout-2024-01-16T12-00-00-abc123.jsonl"
        )
        dest = codex_dir / "rollout-2024-01-16T12-00-00-abc123.jsonl"
        shutil.copy(fixture, dest)

        # First index
        first = index_sessions(source=None, full=False, recreate=True, verbose=False)
        assert first.indexed == 1
        assert first.failed == 0

        config = _app_config(tmp_path)
        conn = duckdb.connect(str(config.db_path), read_only=True)
        try:
            row = conn.execute("SELECT tool_count FROM session_state LIMIT 1").fetchone()
            assert row is not None
            initial_tool_count = row[0]
            assert initial_tool_count == 4, "session2 fixture has 4 orphan tool calls"
        finally:
            conn.close()

        # Append a new orphan tool call
        with dest.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "response_item",
                        "timestamp": "2024-01-16T14:00:00Z",
                        "payload": {
                            "type": "function_call",
                            "name": "exec_command",
                            "arguments": '{"cmd":"echo appended"}',
                            "call_id": "call_new",
                        },
                    }
                )
                + "\n"
            )

        # Incremental re-index — this previously failed with duplicate key
        second = index_sessions(source=None, full=False, recreate=False, verbose=False)
        assert second.indexed == 1, f"incremental index should succeed, got {second}"
        assert second.failed == 0

        conn = duckdb.connect(str(config.db_path), read_only=True)
        try:
            row = conn.execute("SELECT tool_count FROM session_state LIMIT 1").fetchone()
            assert row is not None
            assert row[0] == initial_tool_count + 1, "tool count should increase by 1"

            orphan_count = conn.execute(
                "SELECT COUNT(*) FROM tool_calls WHERE message_id IS NULL"
            ).fetchone()
            assert orphan_count is not None
            assert orphan_count[0] == 5, "should have 5 orphan tool calls total"
        finally:
            conn.close()


class TestIncrementalTokenPreservation:
    """ISSUE-1 regression: NULL tokens must not become 0 after incremental append."""

    def test_codex_duplicate_cumulative_tokens_are_idempotent_across_offsets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Codex cumulative totals must not be added when duplicate events cross offsets."""
        monkeypatch.setenv("HOME", str(tmp_path))

        codex_dir = tmp_path / ".codex" / "sessions" / "s1"
        codex_dir.mkdir(parents=True, exist_ok=True)
        dest = codex_dir / "rollout.jsonl"
        dest.write_text(_codex_token_count_event(100, 10) + "\n", encoding="utf-8")

        index_sessions(source=None, full=False, recreate=True, verbose=False)

        config = _app_config(tmp_path)
        assert _fetch_session_tokens(config.db_path) == (100, 10)

        with dest.open("a", encoding="utf-8") as f:
            f.write(_codex_token_count_event(100, 10) + "\n")

        index_sessions(source=None, full=False, recreate=False, verbose=False)
        assert _fetch_session_tokens(config.db_path) == (100, 10)

        with dest.open("a", encoding="utf-8") as f:
            f.write(_codex_token_count_event(150, 15) + "\n")
            f.write(_codex_token_count_event(150, 15) + "\n")

        index_sessions(source=None, full=False, recreate=False, verbose=False)
        assert _fetch_session_tokens(config.db_path) == (150, 15)

    def test_null_tokens_stay_null_after_incremental_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))

        # Create Codex session (Codex has no token counts → NULL)
        codex_dir = tmp_path / ".codex" / "sessions" / "s1"
        codex_dir.mkdir(parents=True, exist_ok=True)
        fixture = (
            Path(__file__).resolve().parents[2]
            / "fixtures"
            / "codex"
            / "session1"
            / "rollout.jsonl"
        )
        dest = codex_dir / "rollout.jsonl"
        shutil.copy(fixture, dest)

        # First index
        index_sessions(source=None, full=False, recreate=True, verbose=False)

        config = _app_config(tmp_path)
        conn = duckdb.connect(str(config.db_path), read_only=True)
        try:
            row = conn.execute(
                "SELECT input_tokens, output_tokens FROM session_state LIMIT 1"
            ).fetchone()
            assert row is not None
            assert row[0] is None, "Codex sessions should have NULL input_tokens"
            assert row[1] is None, "Codex sessions should have NULL output_tokens"
        finally:
            conn.close()

        # Append a valid line (no token info)
        with dest.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": "2026-12-31T23:59:59Z",
                        "payload": {"type": "user_message", "message": "appended"},
                    }
                )
                + "\n"
            )

        # Re-index (incremental)
        index_sessions(source=None, full=False, recreate=False, verbose=False)

        conn = duckdb.connect(str(config.db_path), read_only=True)
        try:
            row = conn.execute(
                "SELECT input_tokens, output_tokens FROM session_state LIMIT 1"
            ).fetchone()
            assert row is not None
            assert row[0] is None, "NULL tokens must not become 0 after incremental append"
            assert row[1] is None, "NULL tokens must not become 0 after incremental append"
        finally:
            conn.close()


class TestIncrementalIsCompletePreservation:
    """ISSUE-2 regression: is_complete=false must not flip to true after valid append."""

    def test_is_complete_false_preserved_after_valid_append(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))

        # Create a session with one valid + one invalid line
        codex_dir = tmp_path / ".codex" / "sessions" / "s1"
        codex_dir.mkdir(parents=True, exist_ok=True)
        dest = codex_dir / "rollout.jsonl"
        dest.write_text(
            json.dumps(
                {
                    "type": "event_msg",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "payload": {"type": "user_message", "message": "hello"},
                }
            )
            + "\nnot valid json\n",
            encoding="utf-8",
        )

        # Index — should mark is_complete=false
        index_sessions(source=None, full=False, recreate=True, verbose=False)

        config = _app_config(tmp_path)
        conn = duckdb.connect(str(config.db_path), read_only=True)
        try:
            row = conn.execute("SELECT is_complete FROM session_state LIMIT 1").fetchone()
            assert row is not None
            assert row[0] is False, "Session with bad JSON should be is_complete=false"
        finally:
            conn.close()

        # Append a valid line
        with dest.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": "2026-01-01T00:01:00Z",
                        "payload": {"type": "user_message", "message": "followup"},
                    }
                )
                + "\n"
            )

        # Re-index (incremental)
        index_sessions(source=None, full=False, recreate=False, verbose=False)

        conn = duckdb.connect(str(config.db_path), read_only=True)
        try:
            row = conn.execute("SELECT is_complete FROM session_state LIMIT 1").fetchone()
            assert row is not None
            assert row[0] is False, (
                "is_complete must stay false — the bad line is still in the file"
            )
        finally:
            conn.close()
