from __future__ import annotations

import json
import os
import shutil
import stat
import textwrap
import threading
import time
from collections.abc import Mapping
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import duckdb
import pytest
import recall.services.indexer as indexer_module
from recall.core.config import (
    AppConfig,
    CliConfig,
    ContextConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.models import Message, ParseResult, Session, TailFacts, ToolCall
from recall.core.types import Role, Source
from recall.db.fts_sidecar import open_sidecar, search_tool_calls_fts
from recall.db.schema import ensure_schema
from recall.parsers.claude_code import ClaudeCodeParser
from recall.parsers.codex import CodexParser
from recall.services.context_backends import ContextBackendUnavailableError, ContextResult
from recall.services.indexer import IndexProgress, index_sessions


def test_rewrite_updates_only_changed_columns_and_preserves_related_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "selective-updates.duckdb"
    with duckdb.connect(str(database_path)) as conn:
        ensure_schema(conn, embed_dim=2)
        original = Session(
            id="selective",
            source=Source.CODEX,
            source_path="/owned/selective.jsonl",
            file_mtime=1.0,
            file_size=1,
            messages=[
                Message(
                    id="content-message",
                    session_id="selective",
                    idx=0,
                    role=Role.ASSISTANT,
                    content="old content",
                    tool_calls=[
                        ToolCall(
                            id="message-tool",
                            session_id="selective",
                            message_id="content-message",
                            idx=0,
                            tool_name="Bash",
                            tool_input={"old": 1},
                            bash_command="echo stable",
                            bash_base="echo",
                            tool_use_id="use-message",
                        ),
                        ToolCall(
                            id="input-tool",
                            session_id="selective",
                            message_id="content-message",
                            idx=1,
                            tool_name="Bash",
                            tool_input={"old": 2},
                            bash_command="echo stable",
                            bash_base="echo",
                            tool_use_id="use-input",
                        ),
                        ToolCall(
                            id="indexed-tool",
                            session_id="selective",
                            message_id="content-message",
                            idx=2,
                            tool_name="Bash",
                            tool_input={"old": 3},
                            bash_command="echo stable",
                            bash_base="echo",
                            tool_use_id="use-indexed",
                        ),
                    ],
                ),
                Message(
                    id="context-message",
                    session_id="selective",
                    idx=1,
                    role=Role.USER,
                    content="stable context content",
                    context_text="[old] ",
                    context_mode="template",
                ),
                Message(
                    id="timestamp-message",
                    session_id="selective",
                    idx=2,
                    role=Role.USER,
                    content="stable timestamp content",
                    timestamp=datetime(2026, 1, 2, 3, 4),
                ),
                Message(
                    id="indexed-message",
                    session_id="selective",
                    idx=3,
                    role=Role.ASSISTANT,
                    content="stable indexed content",
                ),
                Message(
                    id="agent-message",
                    session_id="selective",
                    idx=4,
                    role=Role.ASSISTANT,
                    content="stable agent content",
                    agent_id="old-agent",
                ),
            ],
        )
        indexer_module._write_session(conn, original, tail_facts=TailFacts())
        conn.execute("CHECKPOINT")
        message_rowids_before = dict(
            conn.execute(
                "SELECT message_id, rowid FROM message_state WHERE message_id LIKE '%-message'"
            ).fetchall()
        )
        tool_rowids_before = dict(
            conn.execute("SELECT id, rowid FROM tool_calls ORDER BY id").fetchall()
        )
        agent_rowid_before = conn.execute(
            "SELECT rowid FROM messages WHERE id = 'agent-message'"
        ).fetchone()

        changed = original.model_copy(deep=True)
        messages = {message.id: message for message in changed.messages}
        messages["content-message"].content = "new content"
        messages["context-message"].context_text = "[new] "
        messages["timestamp-message"].timestamp = datetime(2026, 5, 6, 7, 8)
        messages["indexed-message"].thinking = "new indexed thinking"
        messages["indexed-message"].has_thinking = True
        messages["agent-message"].agent_id = "new-agent"
        calls = {call.id: call for message in changed.messages for call in message.tool_calls}
        calls["message-tool"].message_id = "context-message"
        calls["input-tool"].tool_input = {"new": [2, None]}
        calls["indexed-tool"].tool_name = "Read"

        indexer_module._write_session(conn, changed, tail_facts=TailFacts())

        message_rowids_after = dict(
            conn.execute(
                "SELECT message_id, rowid FROM message_state WHERE message_id LIKE '%-message'"
            ).fetchall()
        )
        tool_rowids_after = dict(
            conn.execute("SELECT id, rowid FROM tool_calls ORDER BY id").fetchall()
        )
        agent_rowid_after = conn.execute(
            "SELECT rowid FROM messages WHERE id = 'agent-message'"
        ).fetchone()
        assert agent_rowid_before is not None
        assert agent_rowid_after is not None
        assert agent_rowid_after[0] != agent_rowid_before[0]
        assert message_rowids_after["agent-message"] == message_rowids_before["agent-message"]
        assert message_rowids_after["content-message"] == message_rowids_before["content-message"]
        assert message_rowids_after["context-message"] == message_rowids_before["context-message"]
        assert (
            message_rowids_after["timestamp-message"] == message_rowids_before["timestamp-message"]
        )
        assert message_rowids_after["indexed-message"] != message_rowids_before["indexed-message"]
        assert tool_rowids_after["message-tool"] == tool_rowids_before["message-tool"]
        assert tool_rowids_after["input-tool"] == tool_rowids_before["input-tool"]
        assert tool_rowids_after["indexed-tool"] != tool_rowids_before["indexed-tool"]
        assert conn.execute(
            """
            SELECT message_id, content, thinking, timestamp, has_thinking,
                   context_text, context_mode, fts_content, fts_thinking
            FROM message_state
            WHERE message_id LIKE '%-message'
            ORDER BY message_id
            """
        ).fetchall() == [
            (
                "agent-message",
                "stable agent content",
                None,
                None,
                False,
                "",
                "off",
                "stable agent content",
                "",
            ),
            (
                "content-message",
                "new content",
                None,
                None,
                False,
                "",
                "off",
                "new content",
                "",
            ),
            (
                "context-message",
                "stable context content",
                None,
                None,
                False,
                "[new] ",
                "template",
                "[new] stable context content",
                "[new] ",
            ),
            (
                "indexed-message",
                "stable indexed content",
                "new indexed thinking",
                None,
                True,
                "",
                "off",
                "stable indexed content",
                "new indexed thinking",
            ),
            (
                "timestamp-message",
                "stable timestamp content",
                None,
                datetime(2026, 5, 6, 7, 8),
                False,
                "",
                "off",
                "stable timestamp content",
                "",
            ),
        ]
        assert conn.execute(
            "SELECT agent_id FROM messages WHERE id = 'agent-message'"
        ).fetchone() == ("new-agent",)
        assert conn.execute(
            """
            SELECT id, message_id, tool_name, CAST(tool_input AS VARCHAR),
                   bash_command, bash_base
            FROM tool_calls ORDER BY id
            """
        ).fetchall() == [
            (
                "indexed-tool",
                "content-message",
                "Read",
                '{"old": 3}',
                "echo stable",
                "echo",
            ),
            (
                "input-tool",
                "content-message",
                "Bash",
                '{"new": [2, null]}',
                "echo stable",
                "echo",
            ),
            (
                "message-tool",
                "context-message",
                "Bash",
                '{"old": 1}',
                "echo stable",
                "echo",
            ),
        ]
        assert conn.execute(
            "SELECT tool_call_id, tool_use_id FROM tool_use_ids ORDER BY tool_call_id"
        ).fetchall() == [
            ("indexed-tool", "use-indexed"),
            ("input-tool", "use-input"),
            ("message-tool", "use-message"),
        ]


@pytest.mark.parametrize(
    ("connection_zone", "stored_hour"),
    [("UTC", 9), ("Europe/Berlin", 10)],
)
def test_aware_timestamp_update_uses_the_connection_timezone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    connection_zone: str,
    stored_hour: int,
) -> None:
    original_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "Europe/Berlin")
    time.tzset()
    try:
        database_path = tmp_path / f"timestamp-{connection_zone.replace('/', '-')}.duckdb"
        with duckdb.connect(str(database_path)) as conn:
            conn.execute("SET TimeZone = ?", [connection_zone])
            ensure_schema(conn, embed_dim=2)
            original = Session(
                id="timestamp-zone",
                source=Source.CODEX,
                source_path="/owned/timestamp-zone.jsonl",
                file_mtime=1.0,
                file_size=1,
                messages=[
                    Message(
                        id="timestamp-zone-message",
                        session_id="timestamp-zone",
                        idx=0,
                        role=Role.USER,
                        content="unchanged",
                        timestamp=datetime(2026, 1, 2, 3, 4),
                    )
                ],
            )
            indexer_module._write_session(conn, original, tail_facts=TailFacts())
            changed = original.model_copy(deep=True)
            changed.messages[0].timestamp = datetime(2026, 1, 2, 9, 4, tzinfo=UTC)

            indexer_module._write_session(conn, changed, tail_facts=TailFacts())

            assert conn.execute(
                "SELECT timestamp FROM message_state WHERE message_id = ?",
                ["timestamp-zone-message"],
            ).fetchone() == (datetime(2026, 1, 2, stored_hour, 4),)
    finally:
        if original_tz is None:
            monkeypatch.delenv("TZ")
        else:
            monkeypatch.setenv("TZ", original_tz)
        time.tzset()


def test_changed_tool_calls_update_in_batches_and_roll_back_late_invalid_batch(
    tmp_path: Path,
) -> None:
    with (
        duckdb.connect(":memory:") as conn,
        closing(open_sidecar(tmp_path / "fts.sqlite")) as sidecar,
    ):
        ensure_schema(conn, embed_dim=2)
        original_calls = [
            ToolCall(
                id=f"tool-{index}",
                session_id="session",
                message_id="message",
                idx=index,
                tool_name="Bash",
                tool_input={"nested": [index, None, "λ"]},
                bash_command=f"echo {index}",
                bash_base="echo",
                bash_sub=str(index),
                is_compound=False,
                agent_id="old-agent",
                subagent_type="old-type",
                subagent_description="old description",
                subagent_model="old-model",
                skill_name="old-skill",
                bash_embedding=[float(index), float(index + 1)],
                tool_use_id=f"use-{index}",
            )
            for index in range(514)
        ]
        original = Session(
            id="session",
            source=Source.CODEX,
            source_path="/tmp/session.jsonl",
            file_mtime=1.0,
            file_size=1,
            messages=[
                Message(
                    id="message",
                    session_id="session",
                    idx=0,
                    role=Role.ASSISTANT,
                    tool_calls=original_calls,
                )
            ],
        )
        indexer_module._write_session(conn, original, sidecar_conn=sidecar, tail_facts=TailFacts())
        assert len(search_tool_calls_fts(sidecar, "echo", limit=600)) == 514

        changed_calls = [
            call.model_copy(
                update={
                    "tool_name": "Read",
                    "message_id": "message-2" if call.idx % 2 else "message",
                    "idx": 513 - call.idx,
                    "tool_input": None if call.idx == 256 else {"changed": [call.idx, False, None]},
                    "bash_command": (
                        None
                        if call.idx == 257
                        else call.bash_command
                        if call.idx % 2 == 0
                        else "changed"
                    ),
                    "bash_base": "changed-base",
                    "bash_sub": None,
                    "is_compound": True,
                    "agent_id": None,
                    "subagent_type": "new-type",
                    "subagent_description": None,
                    "subagent_model": "new-model",
                    "skill_name": None,
                    "bash_embedding": None,
                }
            )
            for call in original_calls
        ]
        changed = original.model_copy(
            update={
                "messages": [
                    original.messages[0].model_copy(
                        update={"tool_calls": [call for call in changed_calls if call.idx % 2]}
                    ),
                    Message(
                        id="message-2",
                        session_id="session",
                        idx=1,
                        role=Role.ASSISTANT,
                        tool_calls=[call for call in changed_calls if call.idx % 2 == 0],
                    ),
                ]
            }
        )
        indexer_module._write_session(conn, changed, sidecar_conn=sidecar, tail_facts=TailFacts())
        assert {tool_id for tool_id, _ in search_tool_calls_fts(sidecar, "changed", limit=600)} == {
            f"tool-{index}" for index in range(1, 514, 2) if index != 257
        }
        assert search_tool_calls_fts(sidecar, "echo 257", limit=600) == []

        assert conn.execute(
            "SELECT message_id, idx, tool_name, CAST(tool_input AS VARCHAR), "
            "bash_command, bash_base, bash_sub, "
            "is_compound, agent_id, subagent_type, subagent_description, subagent_model, "
            "skill_name "
            "FROM tool_calls WHERE id='tool-513'"
        ).fetchone() == (
            "message-2",
            0,
            "Read",
            '{"changed": [513, false, null]}',
            "changed",
            "changed-base",
            None,
            True,
            None,
            "new-type",
            None,
            "new-model",
            None,
        )
        assert conn.execute("SELECT COUNT(*) FROM tool_call_embeddings").fetchone() == (257,)
        assert conn.execute("SELECT COUNT(*) FROM tool_use_ids").fetchone() == (514,)
        assert conn.execute(
            "SELECT tool_input IS NULL FROM tool_calls WHERE id='tool-256'"
        ).fetchone() == (True,)
        assert conn.execute(
            "SELECT bash_embedding FROM tool_call_embeddings WHERE tool_call_id='tool-512'"
        ).fetchone() == ((512.0, 513.0),)
        assert conn.execute(
            "SELECT tool_use_id FROM tool_use_ids WHERE tool_call_id='tool-0'"
        ).fetchone() == ("use-0",)

        invalid = [call.model_copy(update={"tool_name": "Write"}) for call in changed_calls]
        invalid[-1] = invalid[-1].model_copy(update={"idx": "invalid"})  # type: ignore[arg-type]
        existing_rows: dict[str, tuple[object, ...]] = {
            str(row[0]): row
            for row in indexer_module._load_persisted_tool_call_rows(conn, "session")
        }
        conn.execute("BEGIN")
        with pytest.raises((TypeError, ValueError), match="invalid"):
            indexer_module._update_tool_calls(conn, invalid, existing_rows)
        conn.execute("ROLLBACK")
        assert conn.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE tool_name='Read'"
        ).fetchone() == (514,)


class TextFingerprintBackend:
    dimensions = 384
    model_id = "text-fingerprint"
    query_prefix = ""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(sum(text.encode("utf-8")) % 997)] * self.dimensions for text in texts]


class RecordingContextBackend:
    def __init__(self, mode: str = "llm-local") -> None:
        self.mode = mode
        self.calls: list[tuple[str, list[str]]] = []

    def is_available(self) -> bool:
        return True

    def generate_prefix(self, session: Session, message: Message) -> ContextResult:
        self.calls.append(
            (
                message.id,
                [message.content or message.thinking or "" for message in session.messages],
            )
        )
        return ContextResult(
            prefix=f"[ctx {message.id}] ",
            mode=self.mode,
            input_tokens=5,
            output_tokens=2,
            model=f"recording-{self.mode}",
        )


def _write_batch_codex_stub(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fake codex executable that records calls and handles batch output."""
    log_path = tmp_path / "codex-invocations.json"
    script = tmp_path / "codex_stub.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, re, sys\n"
        f"_log_path = pathlib.Path({os.fsdecode(log_path)!r})\n"
        + textwrap.dedent(
            """\
            argv = sys.argv[1:]
            prompt = sys.stdin.read()
            out_path = None
            schema_path = None
            for idx, arg in enumerate(argv):
                if arg == "-o" and idx + 1 < len(argv):
                    out_path = argv[idx + 1]
                if arg == "--output-schema" and idx + 1 < len(argv):
                    schema_path = argv[idx + 1]
            if out_path is None:
                raise RuntimeError("missing -o output path")
            if schema_path is None:
                response = "single context"
            else:
                matches = re.findall(
                    r'<chunk index="(\\d+)">(.*?)</chunk>',
                    prompt,
                    flags=re.S,
                )
                response = json.dumps(
                    [
                        {"index": int(index), "context": "ctx:" + chunk.split()[0]}
                        for index, chunk in reversed(matches)
                    ]
                )
            pathlib.Path(out_path).write_text(response, encoding="utf-8")
            sys.stdout.write(json.dumps({
                "type": "turn.completed",
                "usage": {
                    "input_tokens": len(prompt),
                    "cached_input_tokens": 0,
                    "output_tokens": len(response),
                    "reasoning_output_tokens": 0,
                },
            }) + "\\n")
            record = {
                "argv": argv,
                "prompt": prompt,
                "schema_exists": pathlib.Path(schema_path).exists() if schema_path else False,
            }
            existing = json.loads(_log_path.read_text()) if _log_path.exists() else []
            existing.append(record)
            _log_path.write_text(json.dumps(existing))
            sys.exit(0)
            """
        ),
        encoding="utf-8",
    )
    launcher = tmp_path / "codex"
    launcher.write_text(f'#!/bin/sh\nexec {os.fsdecode(script)} "$@"\n', encoding="utf-8")
    for path in (script, launcher):
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return launcher, log_path


def test_indexer_indexes_sessions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

    claude_target = tmp_path / ".claude" / "projects" / "proj1"
    codex_target = tmp_path / ".codex" / "sessions" / "s1"
    pi_target = tmp_path / ".pi" / "agent" / "sessions" / "proj1"
    claude_target.mkdir(parents=True)
    codex_target.mkdir(parents=True)
    pi_target.mkdir(parents=True)

    claude_fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "claude_code" / "session1.jsonl"
    )
    codex_fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "codex" / "session1" / "rollout.jsonl"
    )
    pi_fixture = Path(__file__).resolve().parents[2] / "fixtures" / "pi_agent" / "session1.jsonl"

    shutil.copy(claude_fixture, claude_target / "session1.jsonl")
    shutil.copy(codex_fixture, codex_target / "rollout.jsonl")
    shutil.copy(pi_fixture, pi_target / "session1.jsonl")

    summary = index_sessions(source=None, full=True, recreate=True, verbose=False)
    assert summary.indexed == 3
    assert summary.failed == 0

    db_path = tmp_path / ".local/share/recall" / "recall.duckdb"
    db_path = tmp_path / ".local/share/recall" / "recall.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        sessions_row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        messages_row = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        tool_calls_row = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()
        assert sessions_row is not None and sessions_row[0] == 3
        assert messages_row is not None and messages_row[0] == 11
        assert tool_calls_row is not None and tool_calls_row[0] == 4
    finally:
        conn.close()

    summary2 = index_sessions(source=None, full=False, recreate=False, verbose=False)
    assert summary2.skipped == 3

    conn = duckdb.connect(str(db_path))
    try:
        runtime_state = conn.execute(
            """
            SELECT
                last_run_kind,
                last_index_total,
                last_index_indexed,
                last_index_skipped,
                last_index_failed,
                last_successful_at,
                last_failure_message,
                last_context_messages,
                last_context_mode
            FROM runtime_state
            """
        ).fetchone()
        assert runtime_state is not None
        assert runtime_state[0] == "index"
        assert runtime_state[1] == 3
        assert runtime_state[2] == 0
        assert runtime_state[3] == 3
        assert runtime_state[7] == 0
        assert runtime_state[8] == "off"
        assert runtime_state[4] == 0
        assert runtime_state[5] is not None
        assert runtime_state[6] is None
    finally:
        conn.close()


def test_indexer_full_reindex_succeeds_on_existing_sessions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

    claude_target = tmp_path / ".claude" / "projects" / "proj1"
    codex_target = tmp_path / ".codex" / "sessions" / "s1"
    pi_target = tmp_path / ".pi" / "agent" / "sessions" / "proj1"
    claude_target.mkdir(parents=True)
    codex_target.mkdir(parents=True)
    pi_target.mkdir(parents=True)

    claude_fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "claude_code" / "session1.jsonl"
    )
    codex_fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "codex" / "session1" / "rollout.jsonl"
    )
    pi_fixture = Path(__file__).resolve().parents[2] / "fixtures" / "pi_agent" / "session1.jsonl"

    shutil.copy(claude_fixture, claude_target / "session1.jsonl")
    shutil.copy(codex_fixture, codex_target / "rollout.jsonl")
    shutil.copy(pi_fixture, pi_target / "session1.jsonl")

    first = index_sessions(source=None, full=True, recreate=True, verbose=False)
    assert first.indexed == 3
    assert first.failed == 0

    second = index_sessions(source=None, full=True, recreate=False, verbose=False)
    assert second.indexed == 3
    assert second.failed == 0


def test_indexer_preserves_existing_rows_when_fallback_insert_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

    claude_target = tmp_path / ".claude" / "projects" / "proj1"
    claude_target.mkdir(parents=True)

    claude_fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "claude_code" / "session1.jsonl"
    )
    shutil.copy(claude_fixture, claude_target / "session1.jsonl")

    first = index_sessions(source=None, full=True, recreate=True, verbose=False)
    assert first.indexed == 1

    db_path = tmp_path / ".local/share/recall" / "recall.duckdb"
    assert first.failed == 0

    conn = duckdb.connect(str(db_path))
    try:
        before_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        before_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        before_tool_calls = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()
        assert before_sessions is not None
        assert before_messages is not None
        assert before_tool_calls is not None
        expected = (before_sessions[0], before_messages[0], before_tool_calls[0])
    finally:
        conn.close()

    def fail_insert_messages(_conn, _messages, **_kwargs) -> None:
        raise RuntimeError("simulated insert_messages failure")

    original_parse = ClaudeCodeParser.parse

    def parse_with_new_message(
        self,
        path: Path,
        *,
        offset: int = 0,
        **_kw: int,
    ):
        result = original_parse(self, path, offset=offset)
        session = result.session
        session.messages.append(
            Message(
                id=f"{session.id}-new-message",
                session_id=session.id,
                idx=len(session.messages),
                role=Role.ASSISTANT,
                content="new content",
            )
        )
        session.message_count = len(session.messages)
        return ParseResult(
            session=session,
            next_byte_offset=result.next_byte_offset,
            is_full_parse=result.is_full_parse,
        )

    monkeypatch.setattr(ClaudeCodeParser, "parse", parse_with_new_message)
    monkeypatch.setattr(indexer_module, "insert_messages", fail_insert_messages)

    second = index_sessions(source=None, full=True, recreate=False, verbose=False)
    assert second.indexed == 0
    assert second.failed == 1

    conn = duckdb.connect(str(db_path))
    try:
        after_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        after_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        after_tool_calls = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()
        assert after_sessions is not None
        assert after_messages is not None
        assert after_tool_calls is not None
        assert (after_sessions[0], after_messages[0], after_tool_calls[0]) == expected
    finally:
        conn.close()


def test_indexer_reports_progress_updates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

    codex_target = tmp_path / ".codex" / "sessions" / "s1"
    codex_target.mkdir(parents=True)
    codex_fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "codex" / "session1" / "rollout.jsonl"
    )
    shutil.copy(codex_fixture, codex_target / "rollout.jsonl")

    progress_events: list[IndexProgress] = []

    summary = index_sessions(
        source=None,
        full=True,
        recreate=True,
        verbose=False,
        progress_callback=progress_events.append,
    )

    assert summary.indexed == 1
    assert progress_events[0].status == "start"
    assert progress_events[0].processed == 0
    assert progress_events[-1].status == "done"
    assert progress_events[-1].processed == 1
    assert progress_events[-1].indexed == 1


def test_indexer_rejects_invalid_workers(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = AppConfig(
        data_dir=tmp_path / ".local/share/recall",
        db_path=tmp_path / ".local/share/recall/recall.duckdb",
        lock_path=tmp_path / ".local/share/recall/recall.lock",
        config_path=tmp_path / ".config/recall/config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )

    with pytest.raises(ValueError, match="workers must be positive"):
        index_sessions(
            source=None,
            full=False,
            recreate=False,
            verbose=False,
            workers=0,
            config=config,
        )

    with pytest.raises(ValueError, match="workers must be a positive integer or 'auto'"):
        index_sessions(
            source=None,
            full=False,
            recreate=False,
            verbose=False,
            workers="many",
            config=config,
        )


def test_indexer_workers_prepare_sessions_concurrently_and_preserve_rows(
    tmp_path, monkeypatch
) -> None:
    barrier = threading.Barrier(2)
    thread_names: list[str] = []

    class FakeParser:
        source = Source.CODEX

        @property
        def file_pattern(self) -> str:
            return "*.jsonl"

        roots: tuple[Path, ...] | None = None

        def default_roots(self) -> list[Path]:
            return []

        def watch_roots(self) -> list[Path]:
            return []

        def discover(self) -> list[Path]:
            return []

        def sidecar_paths(self, path: Path) -> list[Path]:
            _ = path
            return []

        def parse(
            self,
            path: Path,
            *,
            offset: int = 0,
            message_idx_base: int = 0,
            orphan_tool_call_idx_base: int = 0,
            resume_state: Mapping[str, Any] | None = None,
        ) -> ParseResult:
            del message_idx_base, offset, orphan_tool_call_idx_base, resume_state
            thread_names.append(threading.current_thread().name)
            barrier.wait(timeout=2)
            time.sleep(0.05)
            session_id = path.stem
            return ParseResult(
                session=Session(
                    id=session_id,
                    source=Source.CODEX,
                    source_path=str(path),
                    file_mtime=1.0,
                    file_size=1,
                    messages=[
                        Message(
                            id=f"{session_id}-msg-1",
                            session_id=session_id,
                            idx=0,
                            role=Role.ASSISTANT,
                            content=f"content for {session_id}",
                        )
                    ],
                ),
                next_byte_offset=1,
                is_full_parse=True,
            )

        def live_candidates(self, *, now: datetime, idle_threshold: float) -> list[Path]:
            del idle_threshold, now
            return []

    parser = FakeParser()
    discovered = [
        indexer_module.DiscoveredSessionPath(
            parser=parser,
            path=tmp_path / "session-a.jsonl",
            resolved_path=str(tmp_path / "session-a.jsonl"),
            file_mtime=1.0,
            file_size=1,
        ),
        indexer_module.DiscoveredSessionPath(
            parser=parser,
            path=tmp_path / "session-b.jsonl",
            resolved_path=str(tmp_path / "session-b.jsonl"),
            file_mtime=1.0,
            file_size=1,
        ),
    ]

    config = AppConfig(
        data_dir=tmp_path / ".local/share/recall",
        db_path=tmp_path / ".local/share/recall/recall.duckdb",
        lock_path=tmp_path / ".local/share/recall/recall.lock",
        config_path=tmp_path / ".config/recall/config.toml",
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )

    monkeypatch.setattr(indexer_module, "_discover_paths", lambda _source, **_kwargs: discovered)
    monkeypatch.setattr(indexer_module, "_load_session_states", lambda _conn: {})

    summary = index_sessions(
        source=None,
        full=False,
        recreate=True,
        verbose=False,
        workers="auto",
        config=config,
    )

    assert summary.indexed == 2
    assert len(set(thread_names)) == 2

    conn = duckdb.connect(str(config.db_path))
    try:
        rows = conn.execute("SELECT id, source_path FROM sessions ORDER BY id").fetchall()
        assert rows == [
            ("session-a", str(tmp_path / "session-a.jsonl")),
            ("session-b", str(tmp_path / "session-b.jsonl")),
        ]
    finally:
        conn.close()


def test_indexer_uses_configured_template_context_mode(tmp_path, monkeypatch) -> None:
    class TemplateParser:
        source = Source.CODEX

        @property
        def file_pattern(self) -> str:
            return "*.jsonl"

        roots: tuple[Path, ...] | None = None

        def default_roots(self) -> list[Path]:
            return []

        def watch_roots(self) -> list[Path]:
            return []

        def discover(self) -> list[Path]:
            return []

        def sidecar_paths(self, path: Path) -> list[Path]:
            _ = path
            return []

        def parse(
            self,
            path: Path,
            *,
            offset: int = 0,
            message_idx_base: int = 0,
            orphan_tool_call_idx_base: int = 0,
            resume_state: Mapping[str, Any] | None = None,
        ) -> ParseResult:
            del offset, message_idx_base, orphan_tool_call_idx_base, resume_state
            return ParseResult(
                session=Session(
                    id="template-session",
                    source=Source.CODEX,
                    source_path=str(path),
                    file_mtime=path.stat().st_mtime,
                    file_size=path.stat().st_size,
                    git_repo="acme/recall",
                    git_branch="main",
                    messages=[
                        Message(
                            id="template-message",
                            session_id="template-session",
                            idx=0,
                            role=Role.ASSISTANT,
                            content="hello",
                        )
                    ],
                    message_count=1,
                ),
                next_byte_offset=path.stat().st_size,
                is_full_parse=True,
            )

        def live_candidates(self, *, now: datetime, idle_threshold: float) -> list[Path]:
            del now, idle_threshold
            return []

    path = tmp_path / "template.jsonl"
    path.write_text("{}", encoding="utf-8")
    parser = TemplateParser()
    discovered = indexer_module.DiscoveredSessionPath(
        parser=parser,
        path=path,
        resolved_path=str(path.resolve()),
        file_mtime=path.stat().st_mtime,
        file_size=path.stat().st_size,
    )
    config = AppConfig(
        data_dir=tmp_path / ".local/share/recall",
        db_path=tmp_path / ".local/share/recall/recall.duckdb",
        lock_path=tmp_path / ".local/share/recall/recall.lock",
        config_path=tmp_path / ".config/recall/config.toml",
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(context=ContextConfig(mode="template")),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )

    monkeypatch.setattr(indexer_module, "_discover_paths", lambda _source, **_kwargs: [discovered])
    summary = index_sessions(
        source=None,
        full=False,
        recreate=True,
        verbose=False,
        config=config,
    )

    assert summary.indexed == 1
    assert summary.context_messages == 1
    assert summary.context_mode == "template"
    conn = duckdb.connect(str(config.db_path))
    try:
        row = conn.execute(
            """
            SELECT context_text, context_mode, fts_content
            FROM message_state
            WHERE message_id = 'template-message'
            """
        ).fetchone()
        runtime_row = conn.execute(
            """
            SELECT last_context_messages, last_context_mode
            FROM runtime_state
            WHERE singleton = TRUE
            """
        ).fetchone()
    finally:
        conn.close()

    assert row == (
        "[acme/recall main] ",
        "template",
        "[acme/recall main] hello",
    )
    assert runtime_row == (1, "template")


def test_full_reindex_rewrites_existing_rows_when_context_mode_changes(
    tmp_path, monkeypatch
) -> None:
    class TemplateParser:
        source = Source.CODEX

        @property
        def file_pattern(self) -> str:
            return "*.jsonl"

        roots: tuple[Path, ...] | None = None

        def default_roots(self) -> list[Path]:
            return []

        def watch_roots(self) -> list[Path]:
            return []

        def discover(self) -> list[Path]:
            return []

        def sidecar_paths(self, path: Path) -> list[Path]:
            _ = path
            return []

        def parse(
            self,
            path: Path,
            *,
            offset: int = 0,
            message_idx_base: int = 0,
            orphan_tool_call_idx_base: int = 0,
            resume_state: Mapping[str, Any] | None = None,
        ) -> ParseResult:
            del offset, message_idx_base, orphan_tool_call_idx_base, resume_state
            return ParseResult(
                session=Session(
                    id="template-session",
                    source=Source.CODEX,
                    source_path=str(path),
                    file_mtime=path.stat().st_mtime,
                    file_size=path.stat().st_size,
                    git_repo="acme/recall",
                    git_branch="main",
                    messages=[
                        Message(
                            id="template-message",
                            session_id="template-session",
                            idx=0,
                            role=Role.ASSISTANT,
                            content="hello",
                            thinking="think",
                        )
                    ],
                    message_count=1,
                ),
                next_byte_offset=path.stat().st_size,
                is_full_parse=True,
            )

        def live_candidates(self, *, now: datetime, idle_threshold: float) -> list[Path]:
            del now, idle_threshold
            return []

    path = tmp_path / "template.jsonl"
    path.write_text("{}", encoding="utf-8")
    parser = TemplateParser()
    discovered = indexer_module.DiscoveredSessionPath(
        parser=parser,
        path=path,
        resolved_path=str(path.resolve()),
        file_mtime=path.stat().st_mtime,
        file_size=path.stat().st_size,
    )
    base_config = AppConfig(
        data_dir=tmp_path / ".local/share/recall",
        db_path=tmp_path / ".local/share/recall/recall.duckdb",
        lock_path=tmp_path / ".local/share/recall/recall.lock",
        config_path=tmp_path / ".config/recall/config.toml",
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(context=ContextConfig(mode="off")),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )

    monkeypatch.setattr(indexer_module, "_discover_paths", lambda _source, **_kwargs: [discovered])
    index_sessions(source=None, full=True, recreate=True, verbose=False, config=base_config)
    template_config = AppConfig(
        data_dir=base_config.data_dir,
        db_path=base_config.db_path,
        lock_path=base_config.lock_path,
        config_path=base_config.config_path,
        fts=base_config.fts,
        embedding=EmbeddingConfig(context=ContextConfig(mode="template")),
        daemon=base_config.daemon,
        cli=base_config.cli,
    )
    summary = index_sessions(
        source=None,
        full=True,
        recreate=False,
        verbose=False,
        config=template_config,
    )

    assert summary.indexed == 1
    conn = duckdb.connect(str(base_config.db_path))
    try:
        row = conn.execute(
            """
            SELECT context_text, context_mode, fts_content, fts_thinking
            FROM message_state
            WHERE message_id = 'template-message'
            """
        ).fetchone()
    finally:
        conn.close()

    assert row == (
        "[acme/recall main] ",
        "template",
        "[acme/recall main] hello",
        "[acme/recall main] think",
    )


def _insert_recompute_fixture(
    conn: duckdb.DuckDBPyConnection,
    *,
    session_id: str,
    message_id: str,
    repo: str,
    started_at: datetime,
    context_mode: str = "off",
) -> None:
    context_text = f"[{repo} main] " if context_mode == "template" else ""
    content = "shared searchable content"
    conn.execute(
        "INSERT INTO sessions (id, source, source_path, source_session_id) VALUES (?, ?, ?, ?)",
        [session_id, "codex", f"/tmp/{session_id}.jsonl", None],
    )
    conn.execute(
        """
        INSERT INTO session_state (
            session_id, started_at, git_repo, git_branch, message_count,
            tool_count, is_complete, file_mtime, file_size
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [session_id, started_at, repo, "main", 1, 1, True, 1.0, 1],
    )
    conn.execute(
        "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
        [message_id, session_id, 0],
    )
    conn.execute(
        """
        INSERT INTO message_state (
            message_id, role, content, thinking, has_thinking,
            context_text, context_mode, fts_content, fts_thinking
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            message_id,
            "assistant",
            content,
            "thinking",
            True,
            context_text,
            context_mode,
            f"{context_text}{content}",
            f"{context_text}thinking",
        ],
    )
    conn.execute(
        "INSERT INTO message_embeddings (message_id, content_embedding, thinking_embedding) "
        "VALUES (?, ?, ?)",
        [message_id, [0.0] * 384, [0.0] * 384],
    )
    conn.execute(
        "INSERT INTO tool_calls "
        "(id, session_id, message_id, idx, tool_name, bash_command, is_compound) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [f"tc-{message_id}", session_id, message_id, 0, "bash", "git status", False],
    )
    conn.execute(
        "INSERT INTO tool_call_embeddings (tool_call_id, bash_embedding) VALUES (?, ?)",
        [f"tc-{message_id}", [9.0] * 384],
    )


def test_recompute_context_rewrites_message_rows_and_embeddings_only() -> None:
    from recall.db.schema import ensure_schema
    from recall.services.indexer import recompute_context_for_rows

    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    now = datetime.now(UTC)
    _insert_recompute_fixture(
        conn,
        session_id="session-recall",
        message_id="message-recall",
        repo="acme/recall",
        started_at=now,
    )
    _insert_recompute_fixture(
        conn,
        session_id="session-other",
        message_id="message-other",
        repo="example/other",
        started_at=now,
    )
    bash_before = conn.execute(
        """
        SELECT tc.*, tce.bash_embedding
        FROM tool_calls tc
        JOIN tool_call_embeddings tce ON tce.tool_call_id = tc.id
        ORDER BY tc.id
        """
    ).fetchall()
    config = AppConfig(
        data_dir=Path("/tmp/recall-test/data"),
        db_path=Path("/tmp/recall-test/data/recall.duckdb"),
        lock_path=Path("/tmp/recall-test/data/recall.lock"),
        config_path=Path("/tmp/recall-test/config.toml"),
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(context=ContextConfig(mode="template")),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )

    summary = recompute_context_for_rows(
        config=config,
        embed=True,
        conn=conn,
        get_backend=lambda _config: TextFingerprintBackend(),
    )

    rows = {
        row[0]: row[1:]
        for row in conn.execute(
            """
            SELECT ms.message_id, ms.context_text, ms.context_mode, ms.fts_content,
                   me.content_embedding
            FROM message_state ms
            JOIN message_embeddings me ON me.message_id = ms.message_id
            ORDER BY ms.message_id
            """
        ).fetchall()
    }
    bash_after = conn.execute(
        """
        SELECT tc.*, tce.bash_embedding
        FROM tool_calls tc
        JOIN tool_call_embeddings tce ON tce.tool_call_id = tc.id
        ORDER BY tc.id
        """
    ).fetchall()
    runtime = conn.execute(
        "SELECT last_context_messages, last_context_mode FROM runtime_state WHERE singleton"
    ).fetchone()

    assert summary.context_messages == 2
    assert rows["message-recall"][:3] == (
        "[acme/recall main] ",
        "template",
        "[acme/recall main] shared searchable content",
    )
    assert rows["message-other"][:3] == (
        "[example/other main] ",
        "template",
        "[example/other main] shared searchable content",
    )
    assert rows["message-recall"][3] != [0.0] * 384
    assert rows["message-other"][3] != [0.0] * 384
    assert bash_after == bash_before
    assert runtime == (2, "template")


def test_recompute_context_filters_since_and_only_mode() -> None:
    from recall.db.schema import ensure_schema
    from recall.services.indexer import recompute_context_for_rows

    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    now = datetime.now(UTC)
    _insert_recompute_fixture(
        conn,
        session_id="recent-off",
        message_id="recent-off-msg",
        repo="recent/off",
        started_at=now - timedelta(days=1),
        context_mode="off",
    )
    _insert_recompute_fixture(
        conn,
        session_id="old-off",
        message_id="old-off-msg",
        repo="old/off",
        started_at=now - timedelta(days=60),
        context_mode="off",
    )
    _insert_recompute_fixture(
        conn,
        session_id="recent-template",
        message_id="recent-template-msg",
        repo="recent/template",
        started_at=now - timedelta(days=1),
        context_mode="template",
    )
    config = AppConfig(
        data_dir=Path("/tmp/recall-test/data"),
        db_path=Path("/tmp/recall-test/data/recall.duckdb"),
        lock_path=Path("/tmp/recall-test/data/recall.lock"),
        config_path=Path("/tmp/recall-test/config.toml"),
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(context=ContextConfig(mode="template")),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )

    summary = recompute_context_for_rows(
        config=config,
        since=now - timedelta(days=30),
        only_mode="off",
        conn=conn,
    )

    rows = dict(
        conn.execute(
            "SELECT message_id, context_mode FROM message_state ORDER BY message_id"
        ).fetchall()
    )
    assert summary.context_messages == 1
    assert rows == {
        "old-off-msg": "off",
        "recent-off-msg": "template",
        "recent-template-msg": "template",
    }


def test_recompute_uses_full_session_history_for_selected_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recall.db.schema import ensure_schema
    from recall.services.indexer import recompute_context_for_rows

    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    now = datetime.now(UTC)
    conn.execute(
        "INSERT INTO sessions (id, source, source_path, source_session_id) VALUES (?, ?, ?, ?)",
        ["recompute-session", "codex", "/tmp/recompute.jsonl", None],
    )
    conn.execute(
        """
        INSERT INTO session_state (
            session_id, started_at, git_repo, git_branch, message_count,
            tool_count, is_complete, file_mtime, file_size
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["recompute-session", now, "acme/recall", "main", 2, 0, True, 1.0, 1],
    )
    conn.executemany(
        "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
        [
            ("recompute-prior", "recompute-session", 0),
            ("recompute-target", "recompute-session", 1),
        ],
    )
    conn.executemany(
        """
        INSERT INTO message_state (
            message_id, role, content, thinking, has_thinking,
            context_text, context_mode, fts_content, fts_thinking
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "recompute-prior",
                "user",
                "earlier recompute context",
                None,
                False,
                "[prior] ",
                "template",
                "[prior] earlier recompute context",
                "",
            ),
            (
                "recompute-target",
                "assistant",
                "target recompute content",
                None,
                False,
                "",
                "off",
                "target recompute content",
                "",
            ),
        ],
    )
    backend = RecordingContextBackend()
    config = AppConfig(
        data_dir=Path("/tmp/recall-test/data"),
        db_path=Path("/tmp/recall-test/data/recall.duckdb"),
        lock_path=Path("/tmp/recall-test/data/recall.lock"),
        config_path=Path("/tmp/recall-test/config.toml"),
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(context=ContextConfig(mode="llm-local")),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )

    monkeypatch.setattr(
        "recall.services.context_backends.get_context_backend", lambda _config: backend
    )
    summary = recompute_context_for_rows(
        config=config,
        since=now - timedelta(seconds=1),
        only_mode="off",
        conn=conn,
    )

    rows = dict(
        conn.execute(
            "SELECT message_id, context_mode FROM message_state ORDER BY message_id"
        ).fetchall()
    )
    assert summary.context_messages == 1
    assert backend.calls == [
        (
            "recompute-target",
            ["earlier recompute context", "target recompute content"],
        )
    ]
    assert rows == {
        "recompute-prior": "template",
        "recompute-target": "llm-local",
    }


def test_recompute_context_improves_keyword_rank_for_repo_name() -> None:
    from recall.db.queries import create_fts_indexes
    from recall.db.schema import ensure_schema
    from recall.services.indexer import recompute_context_for_rows
    from recall.services.search import _search_messages

    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    now = datetime.now(UTC)
    _insert_recompute_fixture(
        conn,
        session_id="session-recall",
        message_id="message-recall",
        repo="acme/recall",
        started_at=now,
    )
    _insert_recompute_fixture(
        conn,
        session_id="session-other",
        message_id="message-other",
        repo="example/other",
        started_at=now,
    )
    fts = FtsConfig(fields=("content",), backend="duckdb")
    create_fts_indexes(conn, fts)
    before = _search_messages(
        conn,
        "acme shared",
        source=None,
        limit=10,
        fields=["content"],
    )
    config = AppConfig(
        data_dir=Path("/tmp/recall-test/data"),
        db_path=Path("/tmp/recall-test/data/recall.duckdb"),
        lock_path=Path("/tmp/recall-test/data/recall.lock"),
        config_path=Path("/tmp/recall-test/config.toml"),
        fts=fts,
        embedding=EmbeddingConfig(context=ContextConfig(mode="template")),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )

    recompute_context_for_rows(config=config, conn=conn)
    after = _search_messages(
        conn,
        "acme shared",
        source=None,
        limit=10,
        fields=["content"],
    )

    assert len(before) == 2
    assert after[0].session_id == "session-recall"
    assert after[0].score > after[1].score


def test_llm_context_mode_unavailable_backend_is_fatal(tmp_path, monkeypatch) -> None:
    # REQ-CTX-020: an unavailable llm-* backend aborts the run with an actionable
    # error — it is NOT degraded to template, even when fallback="template".
    class UnavailableBackend:
        _load_failure = RuntimeError("mlx_lm is not installed; install recall[mlx]")

        def is_available(self) -> bool:
            return False

        def generate_prefix(self, session: Session, message: Message):
            raise AssertionError(f"unexpected generation for {session.id}/{message.id}")

    class LlmParser:
        source = Source.CODEX

        @property
        def file_pattern(self) -> str:
            return "*.jsonl"

        roots: tuple[Path, ...] | None = None

        def default_roots(self) -> list[Path]:
            return []

        def watch_roots(self) -> list[Path]:
            return []

        def discover(self) -> list[Path]:
            return []

        def sidecar_paths(self, path: Path) -> list[Path]:
            _ = path
            return []

        def parse(
            self,
            path: Path,
            *,
            offset: int = 0,
            message_idx_base: int = 0,
            orphan_tool_call_idx_base: int = 0,
            resume_state: Mapping[str, Any] | None = None,
        ) -> ParseResult:
            del offset, message_idx_base, orphan_tool_call_idx_base, resume_state
            return ParseResult(
                session=Session(
                    id="llm-fallback-session",
                    source=Source.CODEX,
                    source_path=str(path),
                    file_mtime=path.stat().st_mtime,
                    file_size=path.stat().st_size,
                    git_repo="acme/recall",
                    git_branch="main",
                    messages=[
                        Message(
                            id="llm-fallback-message",
                            session_id="llm-fallback-session",
                            idx=0,
                            role=Role.ASSISTANT,
                            content="eligible message content",
                        )
                    ],
                    message_count=1,
                ),
                next_byte_offset=path.stat().st_size,
                is_full_parse=True,
            )

        def live_candidates(self, *, now: datetime, idle_threshold: float) -> list[Path]:
            del now, idle_threshold
            return []

    path = tmp_path / "llm-fallback.jsonl"
    path.write_text("{}", encoding="utf-8")
    discovered = indexer_module.DiscoveredSessionPath(
        parser=LlmParser(),
        path=path,
        resolved_path=str(path.resolve()),
        file_mtime=path.stat().st_mtime,
        file_size=path.stat().st_size,
    )
    config = AppConfig(
        data_dir=tmp_path / ".local/share/recall",
        db_path=tmp_path / ".local/share/recall/recall.duckdb",
        lock_path=tmp_path / ".local/share/recall/recall.lock",
        config_path=tmp_path / ".config/recall/config.toml",
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(context=ContextConfig(mode="llm-local", fallback="template")),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )

    monkeypatch.setattr(indexer_module, "_discover_paths", lambda _source, **_kwargs: [discovered])
    monkeypatch.setattr(
        "recall.services.context_backends.get_context_backend",
        lambda _config: UnavailableBackend(),
    )

    with pytest.raises(ContextBackendUnavailableError) as exc_info:
        index_sessions(
            source=None,
            full=False,
            recreate=True,
            verbose=False,
            config=config,
        )

    err = exc_info.value
    assert err.mode == "llm-local"
    message = str(err)
    # Actionable: names the mode, the missing extra, and the escape hatch.
    assert "llm-local" in message
    assert "recall[mlx]" in message
    assert "template" in message and "off" in message


@pytest.mark.parametrize(
    ("kind", "input_tokens", "output_tokens"), [("local", 7, 3), ("remote", 29, 11)]
)
def test_indexer_uses_llm_message_contexts_and_token_counts(
    tmp_path, monkeypatch, kind: str, input_tokens: int, output_tokens: int
) -> None:
    mode = f"llm-{kind}"
    model = f"stub-{kind}-model"

    class StubLlmBackend:
        def __init__(self, min_chars: int) -> None:
            self.min_chars = min_chars
            self.generated_message_ids: list[str] = []

        def is_available(self) -> bool:
            return True

        def generate_prefix(self, session: Session, message: Message) -> ContextResult:
            del session
            if len(message.content or "") < self.min_chars:
                return ContextResult(prefix="", mode="off")
            self.generated_message_ids.append(message.id)
            return ContextResult(
                prefix=f"[{kind} context for {message.id}] ",
                mode=mode,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
            )

    class LlmParser:
        source = Source.CODEX

        @property
        def file_pattern(self) -> str:
            return "*.jsonl"

        roots: tuple[Path, ...] | None = None

        def default_roots(self) -> list[Path]:
            return []

        def watch_roots(self) -> list[Path]:
            return []

        def discover(self) -> list[Path]:
            return []

        def sidecar_paths(self, path: Path) -> list[Path]:
            _ = path
            return []

        def parse(
            self,
            path: Path,
            *,
            offset: int = 0,
            message_idx_base: int = 0,
            orphan_tool_call_idx_base: int = 0,
            resume_state: Mapping[str, Any] | None = None,
        ) -> ParseResult:
            del offset, message_idx_base, orphan_tool_call_idx_base, resume_state
            return ParseResult(
                session=Session(
                    id=f"{kind}-session",
                    source=Source.CODEX,
                    source_path=str(path),
                    file_mtime=path.stat().st_mtime,
                    file_size=path.stat().st_size,
                    git_repo="acme/recall",
                    git_branch="main",
                    messages=[
                        Message(
                            id=f"{kind}-long",
                            session_id=f"{kind}-session",
                            idx=0,
                            role=Role.ASSISTANT,
                            content=f"long enough content for {kind} contextual retrieval",
                        ),
                        Message(
                            id=f"{kind}-short",
                            session_id=f"{kind}-session",
                            idx=1,
                            role=Role.ASSISTANT,
                            content="short",
                        ),
                    ],
                    message_count=2,
                ),
                next_byte_offset=path.stat().st_size,
                is_full_parse=True,
            )

        def live_candidates(self, *, now: datetime, idle_threshold: float) -> list[Path]:
            del now, idle_threshold
            return []

    path = tmp_path / f"{mode}.jsonl"
    path.write_text("{}", encoding="utf-8")
    discovered = indexer_module.DiscoveredSessionPath(
        parser=LlmParser(),
        path=path,
        resolved_path=str(path.resolve()),
        file_mtime=path.stat().st_mtime,
        file_size=path.stat().st_size,
    )
    config = AppConfig(
        data_dir=tmp_path / ".local/share/recall",
        db_path=tmp_path / ".local/share/recall/recall.duckdb",
        lock_path=tmp_path / ".local/share/recall/recall.lock",
        config_path=tmp_path / ".config/recall/config.toml",
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(context=ContextConfig(mode=mode, min_chars=20)),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )
    stub_backend = StubLlmBackend(min_chars=20)

    monkeypatch.setattr(indexer_module, "_discover_paths", lambda _source, **_kwargs: [discovered])
    monkeypatch.setattr(
        "recall.services.context_backends.get_context_backend", lambda _config: stub_backend
    )
    summary = index_sessions(
        source=None,
        full=False,
        recreate=True,
        verbose=False,
        config=config,
    )

    assert summary.context_mode == mode
    assert summary.context_messages == 1
    assert summary.context_input_tokens == input_tokens
    assert summary.context_output_tokens == output_tokens
    assert summary.context_model == model
    assert stub_backend.generated_message_ids == [f"{kind}-long"]
    conn = duckdb.connect(str(config.db_path))
    try:
        rows = conn.execute(
            """
            SELECT message_id, context_text, context_mode, fts_content
            FROM message_state
            ORDER BY message_id
            """
        ).fetchall()
        runtime = conn.execute(
            """
            SELECT last_context_mode, last_context_input_tokens,
                   last_context_output_tokens, last_context_model
            FROM runtime_state
            WHERE singleton
            """
        ).fetchone()
    finally:
        conn.close()

    assert rows == [
        (
            f"{kind}-long",
            f"[{kind} context for {kind}-long] ",
            mode,
            f"[{kind} context for {kind}-long] long enough content for {kind} contextual retrieval",
        ),
        (f"{kind}-short", "", "off", "short"),
    ]
    assert runtime == (mode, input_tokens, output_tokens, model)


def test_indexer_batches_codex_context_for_full_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CodexBatchParser:
        source = Source.CODEX

        @property
        def file_pattern(self) -> str:
            return "*.jsonl"

        roots: tuple[Path, ...] | None = None

        def default_roots(self) -> list[Path]:
            return []

        def watch_roots(self) -> list[Path]:
            return []

        def discover(self) -> list[Path]:
            return []

        def sidecar_paths(self, path: Path) -> list[Path]:
            _ = path
            return []

        def parse(
            self,
            path: Path,
            *,
            offset: int = 0,
            message_idx_base: int = 0,
            orphan_tool_call_idx_base: int = 0,
            resume_state: Mapping[str, Any] | None = None,
        ) -> ParseResult:
            del offset, message_idx_base, orphan_tool_call_idx_base, resume_state
            stat = path.stat()
            return ParseResult(
                session=Session(
                    id="codex-batch-session",
                    source=Source.CODEX,
                    source_path=str(path.resolve()),
                    file_mtime=stat.st_mtime,
                    file_size=stat.st_size,
                    git_repo="acme/recall",
                    git_branch="main",
                    messages=[
                        Message(
                            id=f"codex-batch-{idx}",
                            session_id="codex-batch-session",
                            idx=idx,
                            role=Role.ASSISTANT,
                            content=f"chunk-{idx} body long enough for batch context",
                        )
                        for idx in range(4)
                    ],
                    message_count=4,
                ),
                next_byte_offset=stat.st_size,
                is_full_parse=True,
            )

        def live_candidates(self, *, now: datetime, idle_threshold: float) -> list[Path]:
            del now, idle_threshold
            return []

    executable, log_path = _write_batch_codex_stub(tmp_path)
    path = tmp_path / "codex-batch.jsonl"
    path.write_text("{}", encoding="utf-8")
    discovered = indexer_module.DiscoveredSessionPath(
        parser=CodexBatchParser(),
        path=path,
        resolved_path=str(path.resolve()),
        file_mtime=path.stat().st_mtime,
        file_size=path.stat().st_size,
    )
    config = AppConfig(
        data_dir=tmp_path / ".local/share/recall",
        db_path=tmp_path / ".local/share/recall/recall.duckdb",
        lock_path=tmp_path / ".local/share/recall/recall.lock",
        config_path=tmp_path / ".config/recall/config.toml",
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(
            context=ContextConfig(
                mode="llm-codex",
                batch_size=2,
                min_chars=1,
                max_document_chars=400000,
                executable=str(executable),
            )
        ),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )

    monkeypatch.setattr(indexer_module, "_discover_paths", lambda _source, **_kwargs: [discovered])
    summary = index_sessions(
        source=None,
        full=False,
        recreate=True,
        verbose=False,
        config=config,
    )

    invocations = json.loads(log_path.read_text(encoding="utf-8"))
    assert summary.context_messages == 4
    assert summary.context_mode == "llm-codex"
    assert len(invocations) == 2
    assert all("--output-schema" in invocation["argv"] for invocation in invocations)
    assert all(invocation["schema_exists"] for invocation in invocations)

    conn = duckdb.connect(str(config.db_path))
    try:
        rows = conn.execute(
            """
            SELECT message_id, context_text, context_mode
            FROM message_state
            ORDER BY message_id
            """
        ).fetchall()
    finally:
        conn.close()

    assert rows == [
        ("codex-batch-0", "[ctx:chunk-0] ", "llm-codex"),
        ("codex-batch-1", "[ctx:chunk-1] ", "llm-codex"),
        ("codex-batch-2", "[ctx:chunk-2] ", "llm-codex"),
        ("codex-batch-3", "[ctx:chunk-3] ", "llm-codex"),
    ]


def test_recompute_context_batches_codex_context(tmp_path: Path) -> None:
    from recall.db.schema import ensure_schema
    from recall.services.indexer import recompute_context_for_rows

    executable, log_path = _write_batch_codex_stub(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    now = datetime.now(UTC)
    conn.execute(
        "INSERT INTO sessions (id, source, source_path, source_session_id) VALUES (?, ?, ?, ?)",
        ["recompute-codex-batch", "codex", "/tmp/recompute-codex-batch.jsonl", None],
    )
    conn.execute(
        """
        INSERT INTO session_state (
            session_id, started_at, git_repo, git_branch, message_count,
            tool_count, is_complete, file_mtime, file_size
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["recompute-codex-batch", now, "acme/recall", "main", 4, 0, True, 1.0, 1],
    )
    conn.executemany(
        "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
        [(f"recompute-codex-{idx}", "recompute-codex-batch", idx) for idx in range(4)],
    )
    conn.executemany(
        """
        INSERT INTO message_state (
            message_id, role, content, thinking, has_thinking,
            context_text, context_mode, fts_content, fts_thinking
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                f"recompute-codex-{idx}",
                "assistant",
                f"chunk-{idx} recompute body long enough",
                None,
                False,
                "",
                "off",
                f"chunk-{idx} recompute body long enough",
                "",
            )
            for idx in range(4)
        ],
    )
    config = AppConfig(
        data_dir=tmp_path / ".local/share/recall",
        db_path=tmp_path / ".local/share/recall/recall.duckdb",
        lock_path=tmp_path / ".local/share/recall/recall.lock",
        config_path=tmp_path / ".config/recall/config.toml",
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(
            context=ContextConfig(
                mode="llm-codex",
                batch_size=2,
                min_chars=1,
                max_document_chars=400000,
                executable=str(executable),
            )
        ),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )

    summary = recompute_context_for_rows(config=config, conn=conn)

    invocations = json.loads(log_path.read_text(encoding="utf-8"))
    assert summary.context_messages == 4
    assert len(invocations) == 2
    assert all("--output-schema" in invocation["argv"] for invocation in invocations)
    assert all(invocation["schema_exists"] for invocation in invocations)
    rows = conn.execute(
        """
        SELECT message_id, context_text, context_mode, fts_content
        FROM message_state
        ORDER BY message_id
        """
    ).fetchall()
    assert rows == [
        (
            "recompute-codex-0",
            "[ctx:chunk-0] ",
            "llm-codex",
            "[ctx:chunk-0] chunk-0 recompute body long enough",
        ),
        (
            "recompute-codex-1",
            "[ctx:chunk-1] ",
            "llm-codex",
            "[ctx:chunk-1] chunk-1 recompute body long enough",
        ),
        (
            "recompute-codex-2",
            "[ctx:chunk-2] ",
            "llm-codex",
            "[ctx:chunk-2] chunk-2 recompute body long enough",
        ),
        (
            "recompute-codex-3",
            "[ctx:chunk-3] ",
            "llm-codex",
            "[ctx:chunk-3] chunk-3 recompute body long enough",
        ),
    ]


def test_incremental_index_contextualizes_delta_against_full_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IncrementalParser:
        source = Source.CODEX

        @property
        def file_pattern(self) -> str:
            return "*.jsonl"

        roots: tuple[Path, ...] | None = None

        def default_roots(self) -> list[Path]:
            return []

        def watch_roots(self) -> list[Path]:
            return []

        def discover(self) -> list[Path]:
            return []

        def sidecar_paths(self, path: Path) -> list[Path]:
            _ = path
            return []

        def parse(
            self,
            path: Path,
            *,
            offset: int = 0,
            message_idx_base: int = 0,
            orphan_tool_call_idx_base: int = 0,
            resume_state: Mapping[str, Any] | None = None,
        ) -> ParseResult:
            del orphan_tool_call_idx_base, resume_state
            stat = path.stat()
            if offset == 0:
                messages = [
                    Message(
                        id="incremental-prior",
                        session_id="incremental-session",
                        idx=0,
                        role=Role.USER,
                        content="earlier conversation context",
                    )
                ]
                is_full_parse = True
            else:
                messages = [
                    Message(
                        id="incremental-new",
                        session_id="incremental-session",
                        idx=message_idx_base,
                        role=Role.ASSISTANT,
                        content="new appended answer",
                    )
                ]
                is_full_parse = False
            return ParseResult(
                session=Session(
                    id="incremental-session",
                    source=Source.CODEX,
                    source_path=str(path.resolve()),
                    file_mtime=stat.st_mtime,
                    file_size=stat.st_size,
                    messages=messages,
                    message_count=len(messages),
                ),
                next_byte_offset=stat.st_size,
                is_full_parse=is_full_parse,
            )

        def live_candidates(self, *, now: datetime, idle_threshold: float) -> list[Path]:
            del now, idle_threshold
            return []

    path = tmp_path / "incremental.jsonl"
    path.write_text("first\n", encoding="utf-8")
    parser = IncrementalParser()
    backend = RecordingContextBackend()
    config = AppConfig(
        data_dir=tmp_path / ".local/share/recall",
        db_path=tmp_path / ".local/share/recall/recall.duckdb",
        lock_path=tmp_path / ".local/share/recall/recall.lock",
        config_path=tmp_path / ".config/recall/config.toml",
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(context=ContextConfig(mode="llm-local")),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )

    def discover(
        _source: Source | None, **_kwargs: object
    ) -> list[indexer_module.DiscoveredSessionPath]:
        stat = path.stat()
        return [
            indexer_module.DiscoveredSessionPath(
                parser=parser,
                path=path,
                resolved_path=str(path.resolve()),
                file_mtime=stat.st_mtime,
                file_size=stat.st_size,
            )
        ]

    def sidecar_paths(self, path: Path) -> list[Path]:
        _ = path
        return []

    monkeypatch.setattr(indexer_module, "_discover_paths", discover)
    monkeypatch.setattr(
        "recall.services.context_backends.get_context_backend", lambda _config: backend
    )
    first = index_sessions(source=None, full=False, recreate=True, verbose=False, config=config)

    conn = duckdb.connect(str(config.db_path))
    try:
        conn.execute(
            """
            UPDATE message_state
            SET context_text = ?, context_mode = ?
            WHERE message_id = ?
            """,
            ["[persisted prior] ", "template", "incremental-prior"],
        )
    finally:
        conn.close()

    path.write_text("first\nsecond\n", encoding="utf-8")
    second = index_sessions(source=None, full=False, recreate=False, verbose=False, config=config)

    assert first.context_messages == 1
    assert second.context_messages == 1
    assert backend.calls[-1] == ("incremental-prior", ["earlier conversation context"])
    conn = duckdb.connect(str(config.db_path))
    try:
        rows = conn.execute(
            """
            SELECT m.id, ms.context_text, ms.context_mode
            FROM messages m
            JOIN message_state ms ON ms.message_id = m.id
            ORDER BY m.idx
            """
        ).fetchall()
    finally:
        conn.close()

    assert rows == [("incremental-prior", "[ctx incremental-prior] ", "llm-local")]


def test_indexer_skips_fts_rebuild_when_only_metadata_changes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    monkeypatch.setenv("RECALL_FTS_BACKEND", "duckdb")

    codex_target = tmp_path / ".codex" / "sessions" / "s1"
    codex_target.mkdir(parents=True)
    codex_fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "codex" / "session1" / "rollout.jsonl"
    )
    target_path = codex_target / "rollout.jsonl"
    shutil.copy(codex_fixture, target_path)

    first = index_sessions(source=None, full=True, recreate=True, verbose=False)
    assert first.fts_rebuilt is True

    original_parse = CodexParser.parse

    def sidecar_paths(self, path: Path) -> list[Path]:
        _ = path
        return []

    def parse_without_searchable_changes(
        self,
        path: Path,
        *,
        offset: int = 0,
        **_kw: int,
    ):
        result = original_parse(self, path, offset=offset)
        result.session.git_branch = "different-branch"
        return result

    fts_calls: list[tuple] = []
    monkeypatch.setattr(CodexParser, "parse", parse_without_searchable_changes)
    monkeypatch.setattr(
        indexer_module,
        "create_fts_indexes",
        lambda conn, fts: fts_calls.append((conn, fts)),
    )
    target_path.write_text(target_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    second = index_sessions(source=None, full=False, recreate=False, verbose=False)

    assert second.indexed == 1
    assert second.fts_rebuilt is False
    assert fts_calls == []


def test_indexer_metadata_only_rewrite_updates_session_without_graph_replace(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

    codex_target = tmp_path / ".codex" / "sessions" / "s1"
    codex_target.mkdir(parents=True)
    codex_fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "codex" / "session1" / "rollout.jsonl"
    )
    target_path = codex_target / "rollout.jsonl"
    shutil.copy(codex_fixture, target_path)

    first = index_sessions(source=None, full=True, recreate=True, verbose=False)
    assert first.indexed == 1
    db_path = tmp_path / ".local/share/recall" / "recall.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        expected_counts = conn.execute(
            "SELECT (SELECT COUNT(*) FROM sessions), (SELECT COUNT(*) FROM messages), "
            "(SELECT COUNT(*) FROM tool_calls)"
        ).fetchone()
    finally:
        conn.close()

    original_parse = CodexParser.parse

    def parse_with_metadata_changes(
        self,
        path: Path,
        *,
        offset: int = 0,
        **_kw: int,
    ):
        result = original_parse(self, path, offset=offset)
        result.session.git_branch = "metadata-only-change"
        return result

    original_delete_removed_rows = indexer_module._delete_removed_rows

    def fail_graph_delete(
        conn, table: Literal["messages", "tool_calls"], existing_ids, desired_ids, **_kwargs
    ) -> None:
        if set(existing_ids) != set(desired_ids):
            raise AssertionError("metadata-only rewrite should not replace the session graph")
        original_delete_removed_rows(conn, table, existing_ids, desired_ids)

    monkeypatch.setattr(CodexParser, "parse", parse_with_metadata_changes)
    monkeypatch.setattr(indexer_module, "_delete_removed_rows", fail_graph_delete)
    target_path.write_text(target_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    second = index_sessions(source=None, full=False, recreate=False, verbose=False)

    assert second.indexed == 1

    conn = duckdb.connect(str(db_path))
    try:
        session_row = conn.execute("SELECT git_branch FROM session_state").fetchone()
        counts = conn.execute(
            "SELECT (SELECT COUNT(*) FROM sessions), (SELECT COUNT(*) FROM messages), "
            "(SELECT COUNT(*) FROM tool_calls)"
        ).fetchone()
        assert session_row == ("metadata-only-change",)
        assert counts == expected_counts
    finally:
        conn.close()


def test_indexer_updates_persisted_source_path_when_session_moves(tmp_path, monkeypatch) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "moved.jsonl"
    first_path.write_text("{}", encoding="utf-8")
    second_path.write_text("{}", encoding="utf-8")

    class StableParser:
        source = Source.CODEX

        @property
        def file_pattern(self) -> str:
            return "*.jsonl"

        roots: tuple[Path, ...] | None = None

        def default_roots(self) -> list[Path]:
            return []

        def watch_roots(self) -> list[Path]:
            return []

        def discover(self) -> list[Path]:
            return []

        def sidecar_paths(self, path: Path) -> list[Path]:
            _ = path
            return []

        def parse(
            self,
            path: Path,
            *,
            offset: int = 0,
            message_idx_base: int = 0,
            orphan_tool_call_idx_base: int = 0,
            resume_state: Mapping[str, Any] | None = None,
        ) -> ParseResult:
            del message_idx_base, offset, orphan_tool_call_idx_base, resume_state
            return ParseResult(
                session=Session(
                    id="stable-session",
                    source=Source.CODEX,
                    source_path=str(path.resolve()),
                    source_session_id="stable-source-session",
                    file_mtime=path.stat().st_mtime,
                    file_size=path.stat().st_size,
                    messages=[
                        Message(
                            id="stable-message",
                            session_id="stable-session",
                            idx=0,
                            role=Role.ASSISTANT,
                            content="same content",
                        )
                    ],
                    message_count=1,
                    tool_count=0,
                ),
                next_byte_offset=path.stat().st_size,
                is_full_parse=True,
            )

        def live_candidates(self, *, now: datetime, idle_threshold: float) -> list[Path]:
            del idle_threshold, now
            return []

    parser = StableParser()
    config = AppConfig(
        data_dir=tmp_path / ".local/share/recall",
        db_path=tmp_path / ".local/share/recall/recall.duckdb",
        lock_path=tmp_path / ".local/share/recall/recall.lock",
        config_path=tmp_path / ".config/recall/config.toml",
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )

    def discovered(path: Path) -> indexer_module.DiscoveredSessionPath:
        stat = path.stat()
        return indexer_module.DiscoveredSessionPath(
            parser=parser,
            path=path,
            resolved_path=str(path.resolve()),
            file_mtime=stat.st_mtime,
            file_size=stat.st_size,
        )

    monkeypatch.setattr(
        indexer_module, "_discover_paths", lambda _source, **_kwargs: [discovered(first_path)]
    )

    first = index_sessions(
        source=None,
        full=False,
        recreate=True,
        verbose=False,
        config=config,
    )
    assert first.indexed == 1

    monkeypatch.setattr(
        indexer_module,
        "_discover_paths",
        lambda _source, **_kwargs: [discovered(second_path)],
    )

    second = index_sessions(
        source=None,
        full=False,
        recreate=False,
        verbose=False,
        config=config,
    )
    assert second.indexed == 1

    third = index_sessions(
        source=None,
        full=False,
        recreate=False,
        verbose=False,
        config=config,
    )
    assert third.skipped == 1

    conn = duckdb.connect(str(config.db_path))
    try:
        session_row = conn.execute(
            "SELECT source_path, source_session_id FROM sessions WHERE id = ?",
            ["stable-session"],
        ).fetchone()
        assert session_row == (str(second_path.resolve()), "stable-source-session")
    finally:
        conn.close()


def test_indexer_persists_duration_and_changed_file_counts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

    codex_target = tmp_path / ".codex" / "sessions" / "s1"
    codex_target.mkdir(parents=True)
    codex_fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "codex" / "session1" / "rollout.jsonl"
    )
    shutil.copy(codex_fixture, codex_target / "rollout.jsonl")

    first = index_sessions(source=None, full=True, recreate=True, verbose=False)
    assert first.indexed == 1

    second = index_sessions(source=None, full=False, recreate=False, verbose=False)
    assert second.skipped == 1

    db_path = tmp_path / ".local/share/recall" / "recall.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        runtime_state = conn.execute(
            """
            SELECT
                last_index_total_seconds,
                last_index_changed,
                last_index_indexed,
                last_index_skipped
            FROM runtime_state
            """
        ).fetchone()
        assert runtime_state is not None
        assert float(runtime_state[0]) >= 0.0
        assert runtime_state[1] == 0
        assert runtime_state[2] == 0
        assert runtime_state[3] == 1
    finally:
        conn.close()


def test_identity_rewrite_preserves_embeddings() -> None:
    """Verify embeddings survive when session identity changes."""
    from recall.core.models import ToolCall
    from recall.db import (
        insert_message_embeddings,
        insert_tool_call_embeddings,
    )
    from recall.db.schema import ensure_schema
    from recall.services.indexer import _write_session

    conn = duckdb.connect(":memory:")
    ensure_schema(conn)

    embed_vec = [1.0] * 384

    # Insert initial session with embeddings
    session_v1 = Session(
        id="s1",
        source=Source.CODEX,
        source_path="/tmp/test.jsonl",
        source_session_id=None,
        file_mtime=1.0,
        file_size=100,
        messages=[
            Message(
                id="m1",
                session_id="s1",
                idx=0,
                role=Role.ASSISTANT,
                content="hello",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        session_id="s1",
                        message_id="m1",
                        idx=0,
                        tool_name="bash",
                        bash_command="git status",
                    )
                ],
            )
        ],
    )
    _write_session(conn, session_v1, tail_facts=TailFacts())

    # Insert embeddings
    insert_message_embeddings(conn, [("m1", embed_vec, None)])
    insert_tool_call_embeddings(conn, [("tc1", embed_vec)])

    before_me = conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()
    before_tce = conn.execute("SELECT COUNT(*) FROM tool_call_embeddings").fetchone()
    assert before_me == (1,)
    assert before_tce == (1,)

    # Reindex with identity change (source_session_id changed)
    session_v2 = Session(
        id="s1",
        source=Source.CODEX,
        source_path="/tmp/test.jsonl",
        source_session_id="new-source-id",
        file_mtime=2.0,
        file_size=100,
        messages=[
            Message(
                id="m1",
                session_id="s1",
                idx=0,
                role=Role.ASSISTANT,
                content="hello",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        session_id="s1",
                        message_id="m1",
                        idx=0,
                        tool_name="bash",
                        bash_command="git status",
                    )
                ],
            )
        ],
    )
    _write_session(conn, session_v2, tail_facts=TailFacts())

    after_me = conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()
    after_tce = conn.execute("SELECT COUNT(*) FROM tool_call_embeddings").fetchone()
    assert after_me == (1,), f"expected 1 message embedding, got {after_me}"
    assert after_tce == (1,), f"expected 1 tool_call embedding, got {after_tce}"


def test_content_change_invalidates_stale_embeddings() -> None:
    """When content changes but no new embeddings are provided, old ones must be deleted."""
    from recall.core.models import ToolCall
    from recall.db import (
        insert_message_embeddings,
        insert_tool_call_embeddings,
    )
    from recall.db.schema import ensure_schema
    from recall.services.indexer import _write_session

    conn = duckdb.connect(":memory:")
    ensure_schema(conn)

    embed_vec = [1.0] * 384

    # Insert initial session
    session_v1 = Session(
        id="s1",
        source=Source.CODEX,
        source_path="/tmp/test.jsonl",
        file_mtime=1.0,
        file_size=100,
        messages=[
            Message(
                id="m1",
                session_id="s1",
                idx=0,
                role=Role.ASSISTANT,
                content="original content",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        session_id="s1",
                        message_id="m1",
                        idx=0,
                        tool_name="bash",
                        bash_command="git status",
                    )
                ],
            )
        ],
    )
    _write_session(conn, session_v1, tail_facts=TailFacts())
    insert_message_embeddings(conn, [("m1", embed_vec, None)])
    insert_tool_call_embeddings(conn, [("tc1", embed_vec)])

    # Verify embeddings exist
    assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (1,)
    assert conn.execute("SELECT COUNT(*) FROM tool_call_embeddings").fetchone() == (1,)

    # Reindex with changed content but NO embeddings
    session_v2 = Session(
        id="s1",
        source=Source.CODEX,
        source_path="/tmp/test.jsonl",
        file_mtime=2.0,
        file_size=200,
        messages=[
            Message(
                id="m1",
                session_id="s1",
                idx=0,
                role=Role.ASSISTANT,
                content="changed content",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        session_id="s1",
                        message_id="m1",
                        idx=0,
                        tool_name="bash",
                        bash_command="git commit",
                    )
                ],
            )
        ],
    )
    _write_session(conn, session_v2, tail_facts=TailFacts())

    # Stale embeddings must be gone
    me_count = conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()
    tce_count = conn.execute("SELECT COUNT(*) FROM tool_call_embeddings").fetchone()
    assert me_count == (0,), f"stale message embedding not deleted: {me_count}"
    assert tce_count == (0,), f"stale tool_call embedding not deleted: {tce_count}"


def test_rewrite_reuses_vector_at_a_position_with_an_empty_processed_row() -> None:
    from recall.db import insert_message_embeddings
    from recall.db.schema import ensure_schema
    from recall.services.indexer import _write_session

    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=2)
        old = Session(
            id="shift",
            source=Source.CODEX,
            source_path="/owned/rollout-shift.jsonl",
            file_mtime=1,
            file_size=1,
            messages=[
                Message(
                    id="m0",
                    session_id="shift",
                    idx=0,
                    role=Role.ASSISTANT,
                    content="retained",
                    content_embedding=[1.0, 2.0],
                ),
                Message(id="m1", session_id="shift", idx=1, role=Role.ASSISTANT),
            ],
        )
        _write_session(conn, old, tail_facts=TailFacts())
        insert_message_embeddings(conn, [("m1", None, None)])
        rewritten = old.model_copy(deep=True)
        rewritten.messages[0].content = "new leading message"
        rewritten.messages[0].content_embedding = None
        rewritten.messages[1].content = "retained"

        _write_session(conn, rewritten, tail_facts=TailFacts())

        assert conn.execute(
            "SELECT content_embedding, thinking_embedding FROM message_embeddings "
            "WHERE message_id = 'm1'"
        ).fetchone() == ((1.0, 2.0), None)
        assert conn.execute(
            "SELECT COUNT(*) FROM message_embeddings WHERE message_id = 'm0'"
        ).fetchone() == (0,)


@pytest.mark.parametrize("changed_field", ["content", "thinking"])
def test_fresh_vector_replaces_one_field_while_preserving_its_sibling(
    changed_field: str,
) -> None:
    from recall.db.schema import ensure_schema
    from recall.services.indexer import _write_session

    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=2)
        old = Session(
            id="partial",
            source=Source.CODEX,
            source_path="/owned/rollout-partial.jsonl",
            file_mtime=1,
            file_size=1,
            messages=[
                Message(
                    id="m",
                    session_id="partial",
                    idx=0,
                    role=Role.ASSISTANT,
                    content="old content",
                    thinking="old thinking",
                    content_embedding=[1.0, 2.0],
                    thinking_embedding=[3.0, 4.0],
                )
            ],
        )
        _write_session(conn, old, tail_facts=TailFacts())
        rewritten = old.model_copy(deep=True)
        message = rewritten.messages[0]
        message.content_embedding = None
        message.thinking_embedding = None
        setattr(message, changed_field, "changed text")
        setattr(message, f"{changed_field}_embedding", [5.0, 6.0])

        _write_session(conn, rewritten, tail_facts=TailFacts())

        expected = (
            ((5.0, 6.0), (3.0, 4.0)) if changed_field == "content" else ((1.0, 2.0), (5.0, 6.0))
        )
        assert (
            conn.execute(
                "SELECT content_embedding, thinking_embedding FROM message_embeddings "
                "WHERE message_id = 'm'"
            ).fetchone()
            == expected
        )


def test_rewrite_removes_empty_processed_marker_when_embedding_input_changes() -> None:
    from recall.db import insert_message_embeddings
    from recall.db.schema import ensure_schema
    from recall.services.indexer import _write_session

    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=2)
        session = Session(
            id="processed",
            source=Source.CODEX,
            source_path="/owned/rollout-processed.jsonl",
            file_mtime=1,
            file_size=1,
            messages=[
                Message(
                    id="m",
                    session_id="processed",
                    idx=0,
                    role=Role.USER,
                    content="old",
                )
            ],
        )
        _write_session(conn, session, tail_facts=TailFacts())
        insert_message_embeddings(conn, [("m", None, None)])
        session.messages[0].content = "changed"

        _write_session(conn, session, tail_facts=TailFacts())

        assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (0,)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("zone, stored_aware_hour", [("UTC", 9), ("Europe/Berlin", 10)])
def test_large_message_writes_preserve_fields_and_rollback_on_invalid_tail(
    existing: bool, zone: str, stored_aware_hour: int
) -> None:
    from recall.db.schema import ensure_schema
    from recall.services.indexer import _write_session

    with duckdb.connect(":memory:") as conn:
        conn.execute("SET TimeZone = ?", [zone])
        ensure_schema(conn, embed_dim=2)
        old = Session(
            id="batch-states",
            source=Source.CODEX,
            source_path="/owned/rollout-states.jsonl",
            file_mtime=1,
            file_size=1,
            message_count=514,
            messages=[
                Message(
                    id=f"state-{index}",
                    session_id="batch-states",
                    idx=index,
                    role=Role.ASSISTANT,
                    content="old",
                    thinking="old thoughts",
                    timestamp=datetime(2026, 1, 1),
                    agent_id="old-agent",
                )
                for index in range(514)
            ],
        )
        if existing:
            _write_session(conn, old, tail_facts=TailFacts())
        outside = Session(
            id="outside-states",
            source=Source.CODEX,
            source_path="/owned/rollout-outside.jsonl",
            file_mtime=1,
            file_size=1,
            message_count=1,
            messages=[
                Message(
                    id="outside",
                    session_id="outside-states",
                    idx=0,
                    role=Role.USER,
                    content="untouched",
                )
            ],
        )
        _write_session(conn, outside, tail_facts=TailFacts())
        outside_before = conn.execute(
            "SELECT * FROM message_state WHERE message_id = 'outside'"
        ).fetchall()
        changed = old.model_copy(deep=True)
        for message in changed.messages:
            if message.idx % 3 == 0:
                message.role = Role.USER
                message.content = "changed\n'λ'"
                message.thinking = None
                message.timestamp = datetime(2026, 2, 3, 6, 10)
                message.agent_id = None
                message.context_text = "[a] "
                message.context_mode = "template"
            elif message.idx % 3 == 1:
                message.content = None
                message.thinking = "fresh thoughts"
                message.has_thinking = True
                message.timestamp = datetime(2026, 2, 3, 9, 10, tzinfo=UTC)
                message.agent_id = "new-agent"
            else:
                message.content = ""
                message.thinking = ""
                message.timestamp = None
                message.context_text = "[b] "
                message.context_mode = "llm-local"
        _write_session(conn, changed, tail_facts=TailFacts())
        query = """
            SELECT ms.role, ms.content, ms.thinking, ms.timestamp, ms.has_thinking,
                   ms.context_text, ms.context_mode, ms.fts_content, ms.fts_thinking,
                   m.agent_id
            FROM messages m JOIN message_state ms ON ms.message_id = m.id
            WHERE m.session_id = 'batch-states' ORDER BY m.idx
        """
        cases = [
            (
                "user",
                "changed\n'λ'",
                None,
                datetime(2026, 2, 3, 6, 10),
                False,
                "[a] ",
                "template",
                "[a] changed\n'λ'",
                "[a] ",
                None,
            ),
            (
                "assistant",
                None,
                "fresh thoughts",
                datetime(2026, 2, 3, stored_aware_hour, 10),
                True,
                "",
                "off",
                "",
                "fresh thoughts",
                "new-agent",
            ),
            ("assistant", "", "", None, False, "[b] ", "llm-local", "[b] ", "[b] ", "old-agent"),
        ]
        expected = [cases[index % 3] for index in range(514)]
        assert conn.execute(query).fetchall() == expected

        invalid = changed.model_copy(deep=True)
        for message in invalid.messages:
            message.content = "must roll back"
        invalid.messages[-1].context_mode = "invalid"
        with pytest.raises(duckdb.ConstraintException):
            _write_session(conn, invalid, tail_facts=TailFacts())
        assert conn.execute(query).fetchall() == expected
        assert (
            conn.execute("SELECT * FROM message_state WHERE message_id = 'outside'").fetchall()
            == outside_before
        )
        _write_session(conn, changed, tail_facts=TailFacts())
        assert conn.execute(query).fetchall() == expected


def test_large_rewrite_preserves_equivalent_and_fresh_vectors_across_cleanup_groups() -> None:
    from recall.db.schema import ensure_schema
    from recall.services.indexer import _write_session

    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=2)
        old = Session(
            id="many",
            source=Source.CODEX,
            source_path="/owned/rollout-many.jsonl",
            file_mtime=1,
            file_size=1,
            messages=[
                Message(
                    id=f"message-{index}",
                    session_id="many",
                    idx=index,
                    role=Role.ASSISTANT,
                    content=f"old-{index}",
                    thinking=f"thinking-{index}",
                    content_embedding=[1.0, 1.0],
                    thinking_embedding=[2.0, 2.0],
                )
                for index in range(514)
            ],
        )
        _write_session(conn, old, tail_facts=TailFacts())
        changed = old.model_copy(deep=True)
        for message in changed.messages[:-1]:
            message.content = f"new-{message.idx}"
            message.content_embedding = [3.0, 3.0] if message.idx % 3 == 2 else None
            message.thinking_embedding = None
            if message.idx % 3 != 1:
                message.thinking = f"new-thinking-{message.idx}"
        _write_session(conn, changed, tail_facts=TailFacts())
        rows = dict(
            conn.execute(
                "SELECT message_id, [content_embedding[1], thinking_embedding[1]] "
                "FROM message_embeddings"
            ).fetchall()
        )
        expected = {"message-513": [1.0, 2.0]}
        for index in range(513):
            if index % 3 == 1:
                expected[f"message-{index}"] = [None, 2.0]
            elif index % 3 == 2:
                expected[f"message-{index}"] = [3.0, None]
        assert rows == expected


def _context_reuse_parser(message_count: int = 1):
    """A parser that always reports a full parse of a fixed session."""

    class ReuseParser:
        source = Source.CODEX

        @property
        def file_pattern(self) -> str:
            return "*.jsonl"

        roots: tuple[Path, ...] | None = None

        def default_roots(self) -> list[Path]:
            return []

        def watch_roots(self) -> list[Path]:
            return []

        def discover(self) -> list[Path]:
            return []

        def sidecar_paths(self, path: Path) -> list[Path]:
            _ = path
            return []

        def live_candidates(self, *, now: datetime, idle_threshold: float) -> list[Path]:
            del now, idle_threshold
            return []

        def parse(
            self,
            path: Path,
            *,
            offset: int = 0,
            message_idx_base: int = 0,
            orphan_tool_call_idx_base: int = 0,
            resume_state: Mapping[str, Any] | None = None,
        ) -> ParseResult:
            del offset, message_idx_base, orphan_tool_call_idx_base, resume_state
            return ParseResult(
                session=Session(
                    id="reuse-session",
                    source=Source.CODEX,
                    source_path=str(path),
                    file_mtime=path.stat().st_mtime,
                    file_size=path.stat().st_size,
                    git_repo="acme/recall",
                    git_branch="main",
                    messages=[
                        Message(
                            id=f"reuse-message-{idx}",
                            session_id="reuse-session",
                            idx=idx,
                            role=Role.ASSISTANT,
                            content=f"hello {idx}",
                        )
                        for idx in range(message_count)
                    ],
                    message_count=message_count,
                ),
                next_byte_offset=path.stat().st_size,
                is_full_parse=True,
            )

    return ReuseParser()


class ReuseRecordingBackend:
    """Stands in for an LLM context backend, counting what it is asked to summarize."""

    model_id = "recording-context"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def is_available(self) -> bool:
        return True

    def generate_prefix(self, session: Session, message: Message) -> ContextResult:
        del session
        self.seen.append(message.id)
        return ContextResult(
            prefix=f"[summary of {message.id}] ",
            mode="llm-local",
            input_tokens=1,
            output_tokens=1,
            model=self.model_id,
        )


def _reuse_config(tmp_path, mode: str):
    return AppConfig(
        data_dir=tmp_path / ".local/share/recall",
        db_path=tmp_path / ".local/share/recall/recall.duckdb",
        lock_path=tmp_path / ".local/share/recall/recall.lock",
        config_path=tmp_path / ".config/recall/config.toml",
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(context=ContextConfig(mode=mode)),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )


def _stored_contexts(db_path: Path) -> list[tuple[str, str]]:
    conn = duckdb.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT context_text, context_mode FROM message_state ORDER BY message_id"
        ).fetchall()
    finally:
        conn.close()


def test_full_reparse_keeps_stored_llm_context_when_context_is_disabled(
    tmp_path, monkeypatch
) -> None:
    """REQ-INDEX-021: a full re-parse must not destroy summaries it did not
    author. Disabling context to skip a multi-day re-summarization used to
    overwrite every stored summary with an empty string."""
    path = tmp_path / "reuse.jsonl"
    path.write_text("{}", encoding="utf-8")
    parser = _context_reuse_parser()
    discovered = indexer_module.DiscoveredSessionPath(
        parser=parser,
        path=path,
        resolved_path=str(path.resolve()),
        file_mtime=path.stat().st_mtime,
        file_size=path.stat().st_size,
    )
    monkeypatch.setattr(indexer_module, "_discover_paths", lambda _source, **_kwargs: [discovered])

    backend = ReuseRecordingBackend()
    monkeypatch.setattr(indexer_module, "ensure_context_backend", lambda _config: backend)
    config = _reuse_config(tmp_path, "llm-local")
    index_sessions(source=None, full=False, recreate=True, verbose=False, config=config)
    assert _stored_contexts(config.db_path) == [("[summary of reuse-message-0] ", "llm-local")]

    index_sessions(
        source=None, full=True, recreate=False, verbose=False, config=_reuse_config(tmp_path, "off")
    )

    assert _stored_contexts(config.db_path) == [("[summary of reuse-message-0] ", "llm-local")]


def test_full_reparse_does_not_resummarize_unchanged_messages(tmp_path, monkeypatch) -> None:
    """REQ-INDEX-021: re-parsing is not re-summarizing. Message ids are
    deterministic and transcripts are append-only, so a stored summary is
    still valid and re-earning it costs an LLM call per message."""
    path = tmp_path / "reuse.jsonl"
    path.write_text("{}", encoding="utf-8")
    parser = _context_reuse_parser(message_count=2)
    discovered = indexer_module.DiscoveredSessionPath(
        parser=parser,
        path=path,
        resolved_path=str(path.resolve()),
        file_mtime=path.stat().st_mtime,
        file_size=path.stat().st_size,
    )
    monkeypatch.setattr(indexer_module, "_discover_paths", lambda _source, **_kwargs: [discovered])
    backend = ReuseRecordingBackend()
    monkeypatch.setattr(indexer_module, "ensure_context_backend", lambda _config: backend)
    config = _reuse_config(tmp_path, "llm-local")

    index_sessions(source=None, full=False, recreate=True, verbose=False, config=config)
    assert len(backend.seen) == 2

    summary = index_sessions(source=None, full=True, recreate=False, verbose=False, config=config)

    assert len(backend.seen) == 2, "stored summaries must be reused, not re-earned"
    assert summary.context_reused == 2
    assert _stored_contexts(config.db_path) == [
        ("[summary of reuse-message-0] ", "llm-local"),
        ("[summary of reuse-message-1] ", "llm-local"),
    ]


def test_full_reparse_summarizes_messages_that_have_no_stored_context(
    tmp_path, monkeypatch
) -> None:
    """Reuse must not starve genuinely new messages of a summary."""
    path = tmp_path / "reuse.jsonl"
    path.write_text("{}", encoding="utf-8")
    discovered_one = indexer_module.DiscoveredSessionPath(
        parser=_context_reuse_parser(message_count=1),
        path=path,
        resolved_path=str(path.resolve()),
        file_mtime=path.stat().st_mtime,
        file_size=path.stat().st_size,
    )
    monkeypatch.setattr(
        indexer_module, "_discover_paths", lambda _source, **_kwargs: [discovered_one]
    )
    backend = ReuseRecordingBackend()
    monkeypatch.setattr(indexer_module, "ensure_context_backend", lambda _config: backend)
    config = _reuse_config(tmp_path, "llm-local")
    index_sessions(source=None, full=False, recreate=True, verbose=False, config=config)
    assert backend.seen == ["reuse-message-0"]

    discovered_two = indexer_module.DiscoveredSessionPath(
        parser=_context_reuse_parser(message_count=2),
        path=path,
        resolved_path=str(path.resolve()),
        file_mtime=path.stat().st_mtime,
        file_size=path.stat().st_size,
    )
    monkeypatch.setattr(
        indexer_module, "_discover_paths", lambda _source, **_kwargs: [discovered_two]
    )
    summary = index_sessions(source=None, full=True, recreate=False, verbose=False, config=config)

    assert backend.seen == ["reuse-message-0", "reuse-message-0", "reuse-message-1"]
    assert summary.context_reused == 0
    assert _stored_contexts(config.db_path) == [
        ("[summary of reuse-message-0] ", "llm-local"),
        ("[summary of reuse-message-1] ", "llm-local"),
    ]


def test_full_reparse_refreshes_cheap_template_context(tmp_path, monkeypatch) -> None:
    """Template prefixes are derived and free to rebuild, so they must stay
    fresh rather than being pinned by reuse."""
    path = tmp_path / "reuse.jsonl"
    path.write_text("{}", encoding="utf-8")
    discovered = indexer_module.DiscoveredSessionPath(
        parser=_context_reuse_parser(),
        path=path,
        resolved_path=str(path.resolve()),
        file_mtime=path.stat().st_mtime,
        file_size=path.stat().st_size,
    )
    monkeypatch.setattr(indexer_module, "_discover_paths", lambda _source, **_kwargs: [discovered])
    config = _reuse_config(tmp_path, "template")

    index_sessions(source=None, full=False, recreate=True, verbose=False, config=config)
    summary = index_sessions(source=None, full=True, recreate=False, verbose=False, config=config)

    assert summary.context_reused == 0
    assert _stored_contexts(config.db_path) == [("[acme/recall main] ", "template")]
