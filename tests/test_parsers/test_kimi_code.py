from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from recall.core.types import Role
from recall.parsers.kimi_code import KimiCodeParser


def _fixture_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "fixtures" / "kimi_code" / "session1"


def _control_fixture() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "kimi_code"
        / "control"
        / "agents"
        / "main"
        / "wire.jsonl"
    )


def _agent_wire_fixture() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "kimi_code"
        / "agent_wire"
        / "agents"
        / "main"
        / "wire.jsonl"
    )


def test_kimi_code_parser_parses_messages_tool_calls_and_results() -> None:
    fixture = _fixture_dir() / "agents" / "main" / "wire.jsonl"
    parser = KimiCodeParser()
    session = parser.parse(fixture).session

    assert session.source == "kimi_code"
    assert session.source_session_id == "session1"
    assert session.model == "kimi-code/k3"
    # cwd comes from the session's state.json workDir, not the wire path
    assert session.cwd == "/work/project"
    assert session.message_count == 7
    assert session.tool_count == 2

    # usage.record deltas are additive; input includes cache counters
    assert session.input_tokens == 100 + 50 + 10 + 200 + 50
    assert session.output_tokens == 20 + 40 + 10

    # timestamps come from millisecond epoch event times
    assert session.started_at == datetime.fromtimestamp(1784300000000 / 1000, tz=UTC)
    assert session.ended_at == datetime.fromtimestamp(1784300003000 / 1000, tz=UTC)
    assert session.duration_seconds == 3

    # user (turn.prompt duplicate is not double-counted)
    user_msg = session.messages[0]
    assert user_msg.role == Role.USER
    assert user_msg.content == "List the files in this repo."
    assert user_msg.timestamp == datetime.fromtimestamp(1784300001001 / 1000, tz=UTC)
    assert user_msg.agent_id is None

    # assistant step with thinking + bash tool call
    asst1 = session.messages[1]
    assert asst1.role == Role.ASSISTANT
    assert asst1.thinking == "I should run a shell command to list files."
    assert asst1.has_thinking is True
    assert asst1.content == "Let me check."
    assert len(asst1.tool_calls) == 1
    tc = asst1.tool_calls[0]
    assert tc.tool_name == "Bash"
    assert tc.tool_input == {"command": "ls -la"}
    assert tc.bash_command == "ls -la"
    assert tc.bash_base == "ls"

    # tool_result mapped to SYSTEM message with tool_call_id prefix
    tool_res = session.messages[2]
    assert tool_res.role == Role.SYSTEM
    assert "[tool_call_id: tool_call_1]" in (tool_res.content or "")
    assert "README.md" in (tool_res.content or "")

    # assistant step with an Agent (subagent) tool call
    asst2 = session.messages[3]
    assert asst2.role == Role.ASSISTANT
    assert asst2.thinking == "Now I will delegate exploration."
    assert asst2.content is None
    assert len(asst2.tool_calls) == 1
    agent_tc = asst2.tool_calls[0]
    assert agent_tc.tool_name == "Agent"
    assert agent_tc.subagent_type == "explore"
    assert agent_tc.subagent_description == "Explore repo"

    assert session.messages[4].role == Role.SYSTEM
    assert "[tool_call_id: tool_call_2]" in (session.messages[4].content or "")

    # final assistant text, no tool calls
    asst3 = session.messages[5]
    assert asst3.role == Role.ASSISTANT
    assert asst3.content == "The repo contains README.md and source files."
    assert asst3.thinking is None
    assert len(asst3.tool_calls) == 0

    # compaction summary lands as a SYSTEM message
    compaction = session.messages[6]
    assert compaction.role == Role.SYSTEM
    assert compaction.content == "Compacted 3 steps."


def test_kimi_code_parser_marks_subagent_wire_messages(tmp_path: Path) -> None:
    wire = (
        tmp_path
        / "sessions"
        / "wd_proj_aaaa"
        / "session_bbbb"
        / "agents"
        / "agent-0"
        / "wire.jsonl"
    )
    wire.parent.mkdir(parents=True)
    lines = [
        {"type": "metadata", "protocol_version": "1.4", "created_at": 1784300000000},
        {
            "type": "context.append_message",
            "message": {"role": "user", "content": [{"type": "text", "text": "explore this"}]},
            "time": 1784300001000,
        },
        {
            "type": "context.append_loop_event",
            "event": {
                "type": "content.part",
                "uuid": "p1",
                "turnId": "0",
                "step": 1,
                "part": {"type": "text", "text": "done"},
            },
            "time": 1784300002000,
        },
    ]
    wire.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    session = KimiCodeParser().parse(wire).session

    assert session.source_session_id == "session_bbbb"
    assert session.message_count == 2
    assert all(msg.agent_id == "agent-0" for msg in session.messages)


def test_kimi_code_parser_persists_typed_skill_call(tmp_path: Path) -> None:
    wire = (
        tmp_path / "sessions" / "wd_proj_aaaa" / "session_bbbb" / "agents" / "main" / "wire.jsonl"
    )
    wire.parent.mkdir(parents=True)
    lines = [
        {"type": "metadata", "protocol_version": "1.4", "created_at": 1784300000000},
        {
            "type": "context.append_loop_event",
            "event": {
                "type": "tool.call",
                "name": "Skill",
                "args": {"skill": "engineering-practices:code-law"},
            },
            "time": 1784300001000,
        },
    ]
    wire.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    session = KimiCodeParser().parse(wire).session

    assert session.tool_count == 1
    assert session.messages[0].tool_calls[0].skill_name == "engineering-practices:code-law"


def test_kimi_code_discover_uses_home_and_kimi_code_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KIMI_CODE_HOME", raising=False)
    assert KimiCodeParser().discover() == []
    assert KimiCodeParser().watch_roots() == []

    wire = (
        tmp_path
        / ".kimi-code"
        / "sessions"
        / "wd_proj_aaaa"
        / "session_bbbb"
        / "agents"
        / "main"
        / "wire.jsonl"
    )
    wire.parent.mkdir(parents=True)
    wire.write_text("", encoding="utf-8")
    assert KimiCodeParser().discover() == [wire]
    assert KimiCodeParser().watch_roots() == [wire.parents[4]]

    # KIMI_CODE_HOME relocates the whole data root
    custom = tmp_path / "custom-home"
    custom_wire = (
        custom / "sessions" / "wd_x_cccc" / "session_dddd" / "agents" / "main" / "wire.jsonl"
    )
    custom_wire.parent.mkdir(parents=True)
    custom_wire.write_text("", encoding="utf-8")
    monkeypatch.setenv("KIMI_CODE_HOME", str(custom))
    assert KimiCodeParser().discover() == [custom_wire]


def test_kimi_code_parser_skips_control_telemetry_records() -> None:
    result = KimiCodeParser().parse(_control_fixture())
    session = result.session

    assert result.diagnostics == ()
    assert session.is_complete is True
    assert session.model == "kimi-code/kimi-for-coding"
    assert session.input_tokens == 10 + 2 + 1
    assert session.output_tokens == 4
    assert session.message_count == 3
    assert session.tool_count == 1

    user_msg = session.messages[0]
    assert user_msg.role == Role.USER
    assert user_msg.content == "List the files."

    asst = session.messages[1]
    assert asst.role == Role.ASSISTANT
    assert asst.content == "Checking."
    assert len(asst.tool_calls) == 1
    assert asst.tool_calls[0].tool_name == "Bash"
    assert asst.tool_calls[0].tool_input == {"command": "ls"}

    tool_res = session.messages[2]
    assert tool_res.role == Role.SYSTEM
    assert "[tool_call_id: tool-1]" in (tool_res.content or "")
    assert "README.md" in (tool_res.content or "")

    assert all("You are Kimi Code CLI." not in (msg.content or "") for msg in session.messages)


def test_kimi_code_parser_diagnoses_unknown_loop_event(tmp_path: Path) -> None:
    wire = (
        tmp_path / "sessions" / "wd_proj_aaaa" / "session_bbbb" / "agents" / "main" / "wire.jsonl"
    )
    wire.parent.mkdir(parents=True)
    lines = [
        {"type": "metadata", "protocol_version": "1.4", "created_at": 1784300000000},
        {
            "type": "context.append_loop_event",
            "event": {"type": "future.control", "uuid": "x1"},
            "time": 1784300001000,
        },
    ]
    wire.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    result = KimiCodeParser().parse(wire)

    assert result.session.is_complete is False
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].kind == "unsupported_record"
    assert "future.control" in result.diagnostics[0].detail


def test_kimi_code_parser_keeps_recorded_turns_across_compaction_and_undo(
    tmp_path: Path,
) -> None:
    wire = (
        tmp_path / "sessions" / "wd_proj_aaaa" / "session_cccc" / "agents" / "main" / "wire.jsonl"
    )
    wire.parent.mkdir(parents=True)

    def user(text: str, time: int) -> dict:
        return {
            "type": "context.append_message",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]},
            "time": time,
        }

    def loop(event: dict, time: int) -> dict:
        return {"type": "context.append_loop_event", "event": event, "time": time}

    lines = [
        {"type": "metadata", "protocol_version": "1.4", "created_at": 1784300000000},
        {
            "type": "mcp.tools_discovered",
            "agentId": "main",
            "serverName": "docs",
            "hash": "h1",
            "enabledNames": ["lookup"],
            "tools": [{"name": "lookup", "description": "d", "inputSchema": {}}],
            "time": 1784300000100,
        },
        user("First question.", 1784300001000),
        loop({"type": "step.begin", "uuid": "s1", "turnId": "0", "step": 1}, 1784300001100),
        {
            "type": "turn.step.retrying",
            "agentId": "main",
            "turnId": 0,
            "step": 1,
            "stepId": "s1",
            "failedAttempt": 1,
            "nextAttempt": 2,
            "maxAttempts": 3,
            "delayMs": 10.0,
            "errorName": "APIEmptyResponseError",
            "errorMessage": "empty",
            "time": 1784300001150,
        },
        loop(
            {"type": "content.part", "uuid": "p1", "part": {"type": "text", "text": "Answer one."}},
            1784300001200,
        ),
        loop(
            {"type": "step.end", "uuid": "s1", "turnId": "0", "step": 1, "finishReason": "stop"},
            1784300001300,
        ),
        {
            "type": "file_history.tracked",
            "agentId": "main",
            "turnId": 0,
            "path": "notes.md",
            "entry": {"key": None, "version": 0},
            "time": 1784300001310,
        },
        {
            "type": "file_history.checkpoint",
            "agentId": "main",
            "turnId": 0,
            "phase": "end",
            "entries": {"notes.md": {"key": "k", "contentHash": "c", "size": 1, "version": 1}},
            "time": 1784300001320,
        },
        {"type": "turn.ended", "agentId": "main", "turnId": 0, "time": 1784300001400},
        {"type": "swarm_mode.enter", "agentId": "main", "trigger": "tool", "time": 1784300001500},
        {"type": "swarm_mode.exit", "agentId": "main", "time": 1784300001600},
        {"type": "full_compaction.begin", "source": "auto", "time": 1784300002000},
        {
            "type": "context.apply_compaction",
            "summary": "Summary of the first turn.",
            "contextSummary": "<summary>Summary of the first turn.</summary>",
            "compactedCount": 3,
            "keptUserMessageCount": 1,
            "tokensBefore": 900,
            "tokensAfter": 100,
            "time": 1784300002100,
        },
        user("First question.", 1784300002200),
        {"type": "full_compaction.complete", "time": 1784300002300},
        user("Second question.", 1784300003000),
        {"type": "interruptionReminder.recorded", "turnId": 1, "time": 1784300003050},
        {
            "type": "turn.cancel",
            "agentId": "main",
            "turnId": 1,
            "reason": "user_cancelled",
            "target": "active",
            "time": 1784300003100,
        },
        {"type": "context.undo", "agentId": "main", "count": 1, "time": 1784300003200},
        {
            "type": "token_counting.truncated",
            "agentId": "main",
            "length": 4,
            "tokens": 120,
            "time": 1784300003300,
        },
        {"type": "plan_mode.exit", "time": 1784300003400},
        user("Third question.", 1784300004000),
    ]
    wire.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    result = KimiCodeParser().parse(wire)

    assert result.diagnostics == ()
    assert result.session.is_complete is True
    # Compaction adds its summary and the re-sent kept message as recorded;
    # context.undo is not replayed, so the undone question stays indexed.
    assert [(msg.role, msg.content) for msg in result.session.messages] == [
        (Role.USER, "First question."),
        (Role.ASSISTANT, "Answer one."),
        (Role.SYSTEM, "Summary of the first turn."),
        (Role.USER, "First question."),
        (Role.USER, "Second question."),
        (Role.USER, "Third question."),
    ]


def test_kimi_code_parser_indexes_agent_wire_records_without_duplication() -> None:
    result = KimiCodeParser().parse(_agent_wire_fixture())
    session = result.session

    assert result.diagnostics == ()
    assert session.is_complete is True
    assert session.model == "kimi-code/k3"
    assert session.cwd == "/work/project"

    # Each mirrored agent.message.appended record lands exactly once; the
    # notification that the wire stream never carried is indexed from the
    # agent record.
    assert [(msg.role, msg.content) for msg in session.messages] == [
        (Role.USER, "List the files in this repo."),
        (Role.USER, "<system-reminder>\nFixture reminder.</system-reminder>"),
        (Role.ASSISTANT, "Let me check."),
        (Role.SYSTEM, "[tool_call_id: tool_read_1]\nreadme text"),
        (Role.ASSISTANT, "The repo contains a README."),
        (
            Role.USER,
            '<notification id="task:bash-fixture1:completed" category="task"'
            ' type="task.completed">Background job done.</notification>',
        ),
    ]

    # The user message is indexed from the agent.message.appended record,
    # which precedes its context.append_message mirror in the file.
    assert session.messages[0].timestamp == datetime.fromtimestamp(1790000001000 / 1000, tz=UTC)
    assert session.messages[5].timestamp == datetime.fromtimestamp(1790000001341 / 1000, tz=UTC)

    # Exactly one Read call survives the assistant mirror's toolCalls copy.
    assert session.tool_count == 1
    asst = session.messages[2]
    assert asst.thinking == "I should read the README first."
    assert len(asst.tool_calls) == 1
    assert asst.tool_calls[0].tool_name == "Read"
    assert asst.tool_calls[0].tool_use_id == "tool_read_1"
    assert asst.tool_calls[0].tool_input == {"path": "/work/project/README.md"}

    # usage.record is the only token source; meta.usage on the assistant
    # mirrors is not counted again.
    assert session.input_tokens == 100 + 50 + 10 + 50
    assert session.output_tokens == 20 + 10

    # agent.turn.ended closes the turn on the last message; the step.end
    # markers keep their own vocabulary.
    assert [
        (marker.idx, marker.reason, marker.ends_turn) for marker in result.tail_facts.stop_markers
    ] == [(3, "tool_use", False), (4, "end_turn", False), (5, "done", True)]
    assert [tr.tool_use_id for tr in result.tail_facts.tool_results] == ["tool_read_1"]


def test_kimi_code_parser_agent_wire_suffix_parse_skips_mirrored_records() -> None:
    fixture = _agent_wire_fixture()
    data = fixture.read_bytes()
    lines = data.splitlines(keepends=True)
    batch_start = next(i for i, line in enumerate(lines) if b'"source":"llm"' in line)
    offset = sum(len(line) for line in lines[:batch_start])

    # Five messages commit before the turn-end agent batch (user, reminder,
    # two assistant steps, one tool result).
    result = KimiCodeParser().parse(fixture, offset=offset, message_idx_base=5)

    assert result.diagnostics == ()
    # Mirrored agent.message.appended records are not re-indexed from a
    # suffix; the wire stream already carried them.
    assert result.session.messages == []
    # agent.turn.ended still marks the boundary, anchored to the last
    # committed message.
    assert [
        (marker.idx, marker.reason, marker.ends_turn) for marker in result.tail_facts.stop_markers
    ] == [(4, "done", True)]


def test_kimi_code_parser_indexes_unmirrored_agent_messages(tmp_path: Path) -> None:
    wire = (
        tmp_path / "sessions" / "wd_proj_aaaa" / "session_bbbb" / "agents" / "main" / "wire.jsonl"
    )
    wire.parent.mkdir(parents=True)
    lines = [
        {"type": "metadata", "protocol_version": "1.5", "created_at": 1790000000000},
        {
            "message": {
                "message": {"role": "user", "content": [{"type": "text", "text": "Hi there"}]},
                "meta": {"source": "input"},
            },
            "type": "agent.message.appended",
            "time": 1790000001000,
            "kind": "event",
        },
        {"turnId": 0, "queueItemId": "msg_1", "type": "agent.turn.started", "time": 1790000001001},
        {
            "message": {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello!"}],
                    "toolCalls": [
                        {
                            "type": "function",
                            "id": "tool_bash_1",
                            "name": "Bash",
                            "arguments": '{"command":"ls -la"}',
                        }
                    ],
                },
                "meta": {"source": "llm"},
            },
            "type": "agent.message.appended",
            "time": 1790000001100,
            "kind": "event",
        },
        {
            "message": {
                "message": {
                    "role": "tool",
                    "toolCallId": "tool_bash_1",
                    "content": [{"type": "text", "text": "file.txt"}],
                },
                "meta": {"source": "tool"},
            },
            "type": "agent.message.appended",
            "time": 1790000001200,
            "kind": "event",
        },
        {"turnId": 0, "outcome": "done", "type": "agent.turn.ended", "time": 1790000001300},
    ]
    wire.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    result = KimiCodeParser().parse(wire)
    session = result.session

    assert result.diagnostics == ()
    assert session.is_complete is True
    assert [(msg.role, msg.content) for msg in session.messages] == [
        (Role.USER, "Hi there"),
        (Role.ASSISTANT, "Hello!"),
        (Role.SYSTEM, "[tool_call_id: tool_bash_1]\nfile.txt"),
    ]
    assert session.messages[0].timestamp == datetime.fromtimestamp(1790000001000 / 1000, tz=UTC)
    assert session.tool_count == 1
    tool_call = session.messages[1].tool_calls[0]
    assert tool_call.tool_name == "Bash"
    assert tool_call.tool_use_id == "tool_bash_1"
    assert tool_call.tool_input == {"command": "ls -la"}
    assert tool_call.bash_command == "ls -la"
    assert [tr.tool_use_id for tr in result.tail_facts.tool_results] == ["tool_bash_1"]
    assert result.tail_facts.tool_results[0].result_summary == "file.txt"
    assert [
        (marker.idx, marker.reason, marker.ends_turn) for marker in result.tail_facts.stop_markers
    ] == [(2, "done", True)]


def test_kimi_code_parser_marks_failed_turn_end(tmp_path: Path) -> None:
    wire = (
        tmp_path / "sessions" / "wd_proj_aaaa" / "session_bbbb" / "agents" / "main" / "wire.jsonl"
    )
    wire.parent.mkdir(parents=True)
    lines = [
        {"type": "metadata", "protocol_version": "1.5", "created_at": 1790000000000},
        {
            "type": "context.append_message",
            "message": {"role": "user", "content": [{"type": "text", "text": "do a thing"}]},
            "time": 1790000001000,
        },
        {
            "turnId": 0,
            "outcome": "failed",
            "errorMessage": "boom",
            "type": "agent.turn.ended",
            "time": 1790000002000,
        },
    ]
    wire.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    result = KimiCodeParser().parse(wire)

    assert result.diagnostics == ()
    # A failed turn is still an ended turn: the record declares the boundary.
    assert [
        (marker.idx, marker.reason, marker.ends_turn) for marker in result.tail_facts.stop_markers
    ] == [(0, "failed", True)]


def test_kimi_code_parser_diagnoses_unknown_agent_record(tmp_path: Path) -> None:
    wire = (
        tmp_path / "sessions" / "wd_proj_aaaa" / "session_bbbb" / "agents" / "main" / "wire.jsonl"
    )
    wire.parent.mkdir(parents=True)
    lines = [
        {"type": "metadata", "protocol_version": "1.5", "created_at": 1790000000000},
        {"type": "agent.future.record", "payload": {}, "time": 1790000001000},
    ]
    wire.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    result = KimiCodeParser().parse(wire)

    assert result.session.is_complete is False
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].kind == "unsupported_record"
    assert "agent.future.record" in result.diagnostics[0].detail
