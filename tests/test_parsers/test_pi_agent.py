from __future__ import annotations

import json
from pathlib import Path

from recall.core.types import Role
from recall.parsers.pi_agent import PiAgentParser

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pi_agent"


def test_pi_agent_parser_parses_messages_and_tool_calls() -> None:
    fixture = _FIXTURES / "session1.jsonl"
    parser = PiAgentParser()
    session = parser.parse(fixture).session

    assert session.source_session_id == "pi-session-123"
    assert session.cwd == "/repo/pi"
    assert session.model == "gpt-5.4"
    assert session.message_count == 4
    assert session.tool_count == 1
    assert session.input_tokens == 100
    assert session.output_tokens == 20
    assert session.cached_input_tokens == 0

    user_message = session.messages[0]
    assert user_message.role == Role.USER
    assert user_message.content == "List the files in this repo."

    assistant_message = session.messages[1]
    assert assistant_message.role == Role.ASSISTANT
    assert assistant_message.content == "Checking the repository contents."
    assert assistant_message.thinking == "Need to inspect the repository tree first."
    assert assistant_message.has_thinking is True
    assert len(assistant_message.tool_calls) == 1
    tool_call = assistant_message.tool_calls[0]
    assert tool_call.tool_name == "bash"
    assert tool_call.tool_input == {"command": "ls -la"}
    assert tool_call.bash_command == "ls -la"
    assert tool_call.bash_base == "ls"

    tool_result = session.messages[2]
    assert tool_result.role == Role.SYSTEM
    assert "README.md" in (tool_result.content or "")

    final_message = session.messages[3]
    assert final_message.role == Role.ASSISTANT
    assert final_message.content == "I found the project files."


def test_pi_agent_parser_maps_custom_message_and_compaction_to_system() -> None:
    result = PiAgentParser().parse(_FIXTURES / "session_context.jsonl")
    session = result.session

    assert [d.kind for d in result.diagnostics] == []
    assert session.is_complete is True
    assert session.source_session_id == "pi-session-456"
    assert session.message_count == 4

    fingerprint, user_message, bash, compaction = session.messages
    assert fingerprint.role == Role.SYSTEM
    assert fingerprint.content == "instruction-fingerprint: agent-profile@abc123"
    assert user_message.role == Role.USER
    assert user_message.content == "Continue the work."
    assert bash.role == Role.SYSTEM
    assert bash.content == "$ mytool --help\nUsage: mytool <command>"
    assert len(bash.tool_calls) == 1
    assert bash.tool_calls[0].tool_name == "bash"
    assert bash.tool_calls[0].bash_command == "mytool --help"
    assert compaction.role == Role.SYSTEM
    assert compaction.content == "No prior history."


def test_pi_agent_parser_skips_empty_custom_message_and_compaction(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    records = [
        {
            "type": "session",
            "id": "pi-session-empty",
            "timestamp": "2026-08-24T19:03:14.000Z",
            "cwd": "/repo/pi",
        },
        {
            "type": "custom_message",
            "customType": "instruction-fingerprint",
            "content": "",
            "display": True,
            "id": "custom-empty",
            "timestamp": "2026-08-24T19:03:14.577Z",
        },
        {
            "type": "custom_message",
            "customType": "instruction-fingerprint",
            "display": True,
            "id": "custom-missing",
            "timestamp": "2026-08-24T19:03:14.600Z",
        },
        {
            "type": "compaction",
            "id": "compact-empty",
            "timestamp": "2026-08-24T19:03:16.000Z",
            "summary": "",
            "firstKeptEntryId": "msg-user-1",
            "tokensBefore": 0,
        },
        {
            "type": "compaction",
            "id": "compact-missing",
            "timestamp": "2026-08-24T19:03:16.100Z",
            "firstKeptEntryId": "msg-user-1",
            "tokensBefore": 0,
        },
        {
            "type": "message",
            "id": "msg-user-1",
            "timestamp": "2026-08-24T19:03:15.000Z",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Continue the work."}],
            },
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    result = PiAgentParser().parse(path)
    session = result.session

    assert [d.kind for d in result.diagnostics] == []
    assert session.is_complete is True
    assert session.message_count == 1
    assert session.messages[0].role == Role.USER
    assert session.messages[0].content == "Continue the work."


def test_pi_agent_parser_accumulates_cache_read_as_subset_of_input(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    records = [
        {
            "type": "session",
            "id": "pi-session-cache",
            "timestamp": "2026-09-18T14:00:00.000Z",
            "cwd": "/repo/pi",
        },
        {
            "type": "model_change",
            "id": "model-1",
            "timestamp": "2026-09-18T14:00:00.000Z",
            "modelId": "grok-4.6",
        },
        {
            "type": "message",
            "id": "msg-user-1",
            "timestamp": "2026-09-18T14:00:01.000Z",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Hello."}],
            },
        },
        {
            "type": "message",
            "id": "msg-assistant-1",
            "timestamp": "2026-09-18T14:00:02.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi."}],
                "usage": {
                    "input": 100,
                    "output": 20,
                    "cacheRead": 5000,
                    "cacheWrite": 10,
                    "reasoning": 5,
                    "totalTokens": 5130,
                },
                "stopReason": "stop",
            },
        },
        {
            "type": "message",
            "id": "msg-assistant-2",
            "timestamp": "2026-09-18T14:00:03.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "More."}],
                "usage": {
                    "input": 50,
                    "output": 10,
                    "cacheRead": 2000,
                    "cacheWrite": 0,
                    "reasoning": 2,
                    "totalTokens": 2060,
                },
                "stopReason": "stop",
            },
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    session = PiAgentParser().parse(path).session

    assert session.input_tokens == 100 + 5000 + 10 + 50 + 2000
    assert session.cached_input_tokens == 5000 + 2000
    assert session.output_tokens == 20 + 10


def test_pi_agent_parser_leaves_tokens_unknown_without_usage(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    records = [
        {"type": "session", "id": "pi-no-usage", "timestamp": "2026-09-18T14:00:00.000Z"},
        {
            "type": "message",
            "id": "msg-user-1",
            "timestamp": "2026-09-18T14:00:01.000Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "Hello."}]},
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    session = PiAgentParser().parse(path).session

    assert session.input_tokens is None
    assert session.output_tokens is None
    assert session.cached_input_tokens is None
