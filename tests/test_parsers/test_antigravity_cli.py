from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from recall.cli.contract import validate_source
from recall.core.models import StopMarker
from recall.core.types import Role, Source, parse_source
from recall.parsers.antigravity_cli import AntigravityCliParser
from recall.parsers.common import use_home_root


def _fixture_transcript() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "antigravity_cli"
        / "session1"
        / ".system_generated"
        / "logs"
        / "transcript_full.jsonl"
    )


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _encode_len_field(field_num: int, payload: bytes) -> bytes:
    return _encode_varint((field_num << 3) | 2) + _encode_varint(len(payload)) + payload


def _build_metadata_blob(
    *,
    workspace_uri: str | None = None,
    git_root_uri: str | None = None,
    git_branch: str | None = None,
    conversation_id: str | None = None,
    fallback_workspace_uri: str | None = None,
) -> bytes:
    blob = bytearray()
    ws = bytearray()
    if workspace_uri is not None:
        ws.extend(_encode_len_field(1, workspace_uri.encode("utf-8")))
    if git_root_uri is not None:
        ws.extend(_encode_len_field(2, git_root_uri.encode("utf-8")))
    if git_branch is not None:
        ws.extend(_encode_len_field(4, git_branch.encode("utf-8")))
    if ws:
        blob.extend(_encode_len_field(1, bytes(ws)))
    if conversation_id is not None:
        blob.extend(_encode_len_field(6, conversation_id.encode("utf-8")))
    if fallback_workspace_uri is not None:
        blob.extend(_encode_len_field(7, fallback_workspace_uri.encode("utf-8")))
    return bytes(blob)


def _write_conversation_db(db_path: Path, blob: bytes) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE trajectory_metadata_blob (id TEXT PRIMARY KEY, data BLOB)")
        conn.execute(
            "INSERT INTO trajectory_metadata_blob (id, data) VALUES (?, ?)",
            ("main", blob),
        )
        conn.commit()
    finally:
        conn.close()


def test_parse_source_and_cli_validation_accept_antigravity_aliases() -> None:
    for alias in ("antigravity-cli", "antigravity_cli", "antigravity", "agy"):
        assert parse_source(alias) is Source.ANTIGRAVITY_CLI
        validate_source(alias)


def test_antigravity_cli_parser_parses_fixture_end_to_end() -> None:
    parser = AntigravityCliParser()
    result = parser.parse(_fixture_transcript())
    session = result.session

    assert result.is_full_parse is True
    assert result.diagnostics == ()
    assert result.normalization_checkpoint is not None
    assert session.source is Source.ANTIGRAVITY_CLI
    assert session.is_complete is True
    assert session.model == "gemini-4-neon-high"
    assert session.started_at is not None
    assert session.ended_at is not None
    assert session.duration_seconds == 15

    # Token accounting: billable input = uncached + cache_read across
    # PLANNER_RESPONSE and CHECKPOINT steps; cached is cache_read alone.
    assert session.input_tokens == (1000 + 0) + (200 + 800) + (500 + 1000)
    assert session.cached_input_tokens == 0 + 800 + 1000
    assert session.output_tokens == 150 + 100 + 80

    assert session.message_count == 9
    assert [m.role for m in session.messages] == [
        Role.USER,
        Role.ASSISTANT,
        Role.SYSTEM,
        Role.SYSTEM,
        Role.SYSTEM,
        Role.SYSTEM,
        Role.SYSTEM,
        Role.SYSTEM,
        Role.ASSISTANT,
    ]
    assert session.messages[0].content == "Inspect the repo and run tests."
    assert session.messages[1].thinking == "Let me read the skill guide and run pytest."
    assert session.messages[1].has_thinking is True
    assert session.messages[1].content is None
    assert session.messages[5].content == "[Message] sender=sub-123 content=Found 5 parsers."
    assert session.messages[6].content == "API error (attempt 1): transient timeout"
    assert "Resuming from a compaction" in (session.messages[7].content or "")
    assert session.messages[8].content == "All 12 tests passed and the codebase has 5 parsers."

    assert session.tool_count == 3
    read_call, cmd_call, subagent_call = session.messages[1].tool_calls
    assert read_call.tool_name == "read_file"
    assert read_call.skill_name == "antigravity_guide"

    assert cmd_call.tool_name == "run_command"
    assert cmd_call.bash_command == "pytest -q"
    assert cmd_call.bash_base == "pytest"

    assert subagent_call.tool_name == "invoke_subagent"
    assert subagent_call.subagent_type == "research"
    assert subagent_call.subagent_description == "Codebase Researcher"
    assert subagent_call.subagent_model == "flash"

    assert result.tail_facts.stop_markers == (
        StopMarker(idx=1, reason="tool_calls", ends_turn=False),
        StopMarker(idx=8, reason="stop", ends_turn=True),
    )


def test_antigravity_cli_discover_selects_only_full_transcripts(tmp_path: Path) -> None:
    app_dir = tmp_path / ".gemini" / "antigravity-cli"
    logs_dir = app_dir / "brain" / "conv-1" / ".system_generated" / "logs"
    (logs_dir / "chunks" / "transcript_full").mkdir(parents=True)
    full = logs_dir / "transcript_full.jsonl"
    compact = logs_dir / "transcript.jsonl"
    chunk = logs_dir / "chunks" / "transcript_full" / "00000000.jsonl"
    for path in (full, compact, chunk):
        path.write_text('{"type":"USER_INPUT","content":"hi"}\n', encoding="utf-8")

    with use_home_root(tmp_path):
        parser = AntigravityCliParser()
        assert parser.watch_roots() == [(app_dir / "brain").resolve()]
        assert parser.discover() == [full.resolve()]


def test_antigravity_cli_reads_sidecar_metadata_and_project_skills(tmp_path: Path) -> None:
    app_dir = tmp_path / ".gemini" / "antigravity-cli"
    cid = "93283c8a-90a0-4bd8-88e5-32d1758b23fd"
    transcript = app_dir / "brain" / cid / ".system_generated" / "logs" / "transcript_full.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        json.dumps(
            {
                "step_index": 0,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "created_at": "2026-11-21T20:00:00Z",
                "model": "gemini-4-neon-high",
                "tool_calls": [
                    {
                        "name": "read_file",
                        "args": {"Path": ".agents/skills/my-skill/SKILL.md"},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    db_path = app_dir / "conversations" / f"{cid}.db"
    _write_conversation_db(
        db_path,
        _build_metadata_blob(
            workspace_uri="file:///Users/dev/my%20repo/sub",
            git_root_uri="file:///Users/dev/my%20repo",
            git_branch="feat/antigravity",
            conversation_id=cid,
        ),
    )

    parser = AntigravityCliParser()
    assert parser.sidecar_paths(transcript) == [db_path]

    session = parser.parse(transcript).session
    assert session.source_session_id == cid
    assert session.cwd == "/Users/dev/my repo/sub"
    assert session.git_repo == "/Users/dev/my repo"
    assert session.git_branch == "feat/antigravity"
    assert session.messages[0].tool_calls[0].skill_name == "my-skill"


def test_antigravity_cli_sidecar_fallbacks_and_non_fatal_corruption(tmp_path: Path) -> None:
    app_dir = tmp_path / ".gemini" / "antigravity-cli"
    cid = "sess-non-git"
    transcript = app_dir / "brain" / cid / ".system_generated" / "logs" / "transcript_full.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        '{"step_index":0,"source":"USER_EXPLICIT","type":"USER_INPUT","status":"DONE","created_at":"2026-11-21T20:00:00Z","content":"hello"}\n',
        encoding="utf-8",
    )

    # Non-git workspace using fallback field 7 -> git_repo derives from cwd
    db_path = app_dir / "conversations" / f"{cid}.db"
    _write_conversation_db(
        db_path,
        _build_metadata_blob(fallback_workspace_uri="file:///tmp/scratch-ws"),
    )
    parser = AntigravityCliParser()
    session = parser.parse(transcript).session
    assert session.cwd == "/tmp/scratch-ws"
    assert session.git_repo == "/tmp/scratch-ws"
    assert session.git_branch is None

    # Corrupted sidecar DB must be non-fatal
    db_path.write_bytes(b"not a sqlite database")
    corrupt_session = parser.parse(transcript).session
    assert corrupt_session.cwd is None
    assert corrupt_session.git_repo is None
    assert corrupt_session.message_count == 1
    assert corrupt_session.is_complete is True


def test_antigravity_cli_error_planner_response_ends_turn(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript_full.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "step_index": 0,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "ERROR",
                "created_at": "2026-11-21T20:00:00Z",
                "content": "Partial reply before abort",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = AntigravityCliParser().parse(transcript)
    assert result.tail_facts.stop_markers == (StopMarker(idx=0, reason="error", ends_turn=True),)


def test_antigravity_cli_diagnoses_unknown_record_and_declines_checkpoint(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript_full.jsonl"
    transcript.write_text(
        '{"step_index":0,"source":"SYSTEM","type":"FUTURE_STEP","status":"DONE","created_at":"2026-11-21T20:00:00Z"}\n',
        encoding="utf-8",
    )
    result = AntigravityCliParser().parse(transcript)
    assert result.session.is_complete is False
    assert result.next_byte_offset == transcript.stat().st_size
    assert result.normalization_checkpoint is None
    assert [d.kind for d in result.diagnostics] == ["unsupported_record"]
