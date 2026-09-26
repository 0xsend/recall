from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from recall.core.types import Role
from recall.parsers.grok import GrokParser


def _fixtures() -> Path:
    return Path(__file__).resolve().parents[2] / "fixtures" / "grok"


def test_grok_parser_parses_messages_tool_calls_and_results() -> None:
    fixture = _fixtures() / "session1.jsonl"
    parser = GrokParser()
    session = parser.parse(fixture).session

    assert session.source == "grok"
    assert session.model == "grok-4.3"
    assert session.message_count == 5
    assert session.tool_count == 1
    # chat_history alone never invents tokens (REQ-PARSE-012)
    assert session.input_tokens is None
    assert session.output_tokens is None

    # system
    sys_msg = session.messages[0]
    assert sys_msg.role == Role.SYSTEM
    assert "Grok" in (sys_msg.content or "")

    # user
    user_msg = session.messages[1]
    assert user_msg.role == Role.USER
    assert user_msg.content == "List the files in this repo."

    # assistant with thinking + tool_call
    asst1 = session.messages[2]
    assert asst1.role == Role.ASSISTANT
    assert asst1.thinking == "I should run a shell command to list files."
    assert asst1.has_thinking is True
    assert len(asst1.tool_calls) == 1
    tc = asst1.tool_calls[0]
    assert tc.tool_name == "bash"
    assert tc.tool_input == {"command": "ls -la"}
    assert tc.bash_command == "ls -la"
    assert tc.bash_base == "ls"

    # tool_result mapped to SYSTEM message
    tool_res = session.messages[3]
    assert tool_res.role == Role.SYSTEM
    assert "README.md" in (tool_res.content or "")
    assert "call-ls-1" in (tool_res.content or "")

    # final assistant with content, no tool calls
    asst2 = session.messages[4]
    assert asst2.role == Role.ASSISTANT
    assert asst2.content == "The repo contains README.md and source files."
    assert asst2.thinking == "I have the listing now."
    assert len(asst2.tool_calls) == 0


def test_grok_parser_enriches_from_summary_and_signals_sidecars() -> None:
    """REQ-PARSE-014 / REQ-GROK-TS-*: sidecars fill timestamps, git, model, duration."""
    fixture = (
        _fixtures() / "with_sidecars" / "%2Fwork%2Fproject" / "sess-uuid-1" / "chat_history.jsonl"
    )
    session = GrokParser().parse(fixture).session

    assert session.source_session_id == "sess-uuid-1"
    assert session.started_at == datetime(2026, 7, 16, 3, 37, 52, 792717, tzinfo=UTC)
    assert session.ended_at == datetime(2026, 7, 16, 3, 38, 53, 398314, tzinfo=UTC)
    assert session.cwd == "/work/project"
    assert session.git_repo == "/work/project"
    assert session.git_branch == "feat/fleet-usage-ledger"
    # summary current_model_id preferred when present (non-empty)
    assert session.model == "grok-4.5"
    assert session.duration_seconds == 61
    # Never invent tokens from chat_history or contextTokensUsed
    assert session.input_tokens is None
    assert session.output_tokens is None


def test_grok_parser_sidecar_failures_are_non_fatal() -> None:
    """REQ-GROK-TS-004: corrupt sidecars must not fail the parse."""
    fixture = _fixtures() / "with_bad_sidecars" / "sess-bad" / "chat_history.jsonl"
    session = GrokParser().parse(fixture).session

    assert session.message_count == 5
    assert session.started_at is None
    assert session.ended_at is None
    assert session.git_repo is None
    assert session.input_tokens is None
    assert session.output_tokens is None


def test_grok_parser_ingests_reasoning_summary_and_backend_tool_call() -> None:
    fixture = _fixtures() / "session_reasoning.jsonl"
    result = GrokParser().parse(fixture)
    session = result.session
    encrypted = "opaque-encrypted-blob"

    assert result.diagnostics == ()
    assert session.is_complete is True
    assert session.model == "grok-4.5"
    assert session.message_count == 2
    assert session.tool_count == 1

    user_msg = session.messages[0]
    assert user_msg.role == Role.USER
    assert user_msg.content == "Search the Zig release notes."

    asst = session.messages[1]
    assert asst.role == Role.ASSISTANT
    assert asst.thinking == "I should search the web for the Zig 0.15.1 release notes."
    assert asst.has_thinking is True
    assert asst.content == "Here is what I found."
    assert encrypted not in (asst.content or "")
    assert encrypted not in (asst.thinking or "")
    assert len(asst.tool_calls) == 1
    call = asst.tool_calls[0]
    assert call.tool_name == "web_search"
    assert call.tool_input is not None
    assert call.tool_input.get("query") == "Zig 0.15.1 release notes"
    assert call.tool_use_id == "ws-1"


def test_grok_parser_skips_encrypted_only_reasoning_without_diagnostic(tmp_path: Path) -> None:
    blob = "I am secret thinking that must not be indexed"
    transcript = tmp_path / "chat_history.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "content": [{"type": "text", "text": "Hi"}]}),
                json.dumps(
                    {
                        "type": "reasoning",
                        "id": "rs-enc",
                        "summary": [],
                        "encrypted_content": blob,
                        "status": "completed",
                    }
                ),
                json.dumps({"type": "assistant", "content": "Hello."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = GrokParser().parse(transcript)
    session = result.session

    assert result.diagnostics == ()
    assert session.is_complete is True
    assert [msg.role for msg in session.messages] == [Role.USER, Role.ASSISTANT]
    asst = session.messages[1]
    assert asst.content == "Hello."
    assert asst.thinking is None
    assert asst.has_thinking is False
    assert blob not in (asst.content or "")
    assert all(blob not in (msg.content or "") for msg in session.messages)
    assert all(msg.thinking != blob for msg in session.messages)


def test_grok_parser_attaches_backend_tool_call_preceding_assistant(tmp_path: Path) -> None:
    transcript = tmp_path / "chat_history.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "content": [{"type": "text", "text": "Search docs."}]}),
                json.dumps(
                    {
                        "type": "reasoning",
                        "id": "rs-1",
                        "summary": [{"type": "summary_text", "text": "Search first."}],
                        "encrypted_content": "opaque",
                        "status": "completed",
                    }
                ),
                json.dumps(
                    {
                        "type": "backend_tool_call",
                        "kind": {
                            "tool_type": "web_search",
                            "action": {"type": "search", "query": "bun test preload"},
                            "id": "ws-live",
                            "status": "completed",
                        },
                    }
                ),
                json.dumps({"type": "assistant", "content": "Preload runs once per file."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = GrokParser().parse(transcript)
    session = result.session

    assert result.diagnostics == ()
    assert session.message_count == 2
    asst = session.messages[1]
    assert asst.thinking == "Search first."
    assert asst.content == "Preload runs once per file."
    assert len(asst.tool_calls) == 1
    assert asst.tool_calls[0].tool_name == "web_search"
    assert asst.tool_calls[0].tool_input is not None
    assert asst.tool_calls[0].tool_input.get("query") == "bun test preload"
    assert asst.tool_calls[0].tool_use_id == "ws-live"


def test_grok_parser_persists_runtime_skill_attribution_including_its_own_checkout(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / ".grok" / "sessions" / "%2Fwork%2Fproduct" / "sess"
    transcript = session_dir / "chat_history.jsonl"
    session_dir.mkdir(parents=True)
    transcript.write_text(
        "\n".join(
            [
                '{"type":"assistant","tool_calls":[{"name":"read","arguments":"{\\"path\\":\\"/opt/agent-profile/plugins/engineering-practices/skills/code-law/SKILL.md\\"}"}]}',
                '{"type":"assistant","tool_calls":[{"name":"read","arguments":"{\\"path\\":\\"/work/product/agent-profile/plugins/engineering-practices/skills/code-law/SKILL.md\\"}"}]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    session = GrokParser().parse(transcript).session

    assert session.cwd == "/work/product"
    assert [call.skill_name for call in session.messages[0].tool_calls] == [
        "engineering-practices:code-law"
    ]
    assert [call.skill_name for call in session.messages[1].tool_calls] == [
        "engineering-practices:code-law"
    ]
