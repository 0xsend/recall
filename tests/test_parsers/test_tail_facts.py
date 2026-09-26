"""Parser tail facts: tool results, stop reasons, session end (REQ-LIVE-005).

Turn state is derived at read time from the tail of a session, so the parsers
have to surface three things they used to drop: a tool_result paired to the
tool_use it answers, the harness's end-of-turn marker, and a session-end marker
when the harness emits one. Claude Code emits the first two and no third.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from recall.core.models import StopMarker
from recall.parsers.claude_code import ClaudeCodeParser

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _parse(name: str):
    return ClaudeCodeParser().parse(FIXTURES / "claude_code" / name)


def test_unanswered_tool_use_yields_no_tool_result() -> None:
    """A session cut mid-tool has the call but not its result."""
    result = _parse("live_mid_tool.jsonl")

    assert result.tail_facts.tool_results == ()
    assert result.tail_facts.stop_markers == (
        StopMarker(idx=1, reason="tool_use", ends_turn=False),
    )
    assert result.tail_facts.session_ended is False


def test_tool_use_id_is_carried_onto_the_tool_call() -> None:
    """The harness id is the only stable key for pairing a later result."""
    result = _parse("live_mid_tool.jsonl")

    calls = [call for message in result.session.messages for call in message.tool_calls]
    assert [call.tool_name for call in calls] == ["Bash"]
    assert [call.tool_use_id for call in calls] == ["toolu_mid_bash"]


def test_answered_tool_use_yields_a_paired_result() -> None:
    result = _parse("live_end_turn.jsonl")

    assert len(result.tail_facts.tool_results) == 1
    tool_result = result.tail_facts.tool_results[0]
    assert tool_result.tool_use_id == "toolu_end_bash"
    assert tool_result.result_summary == "312 passed in 41.02s"
    assert tool_result.is_error is False


def test_end_of_turn_stop_reason_is_recorded_against_its_message() -> None:
    """Only the parser knows which of a harness's reasons actually ends a turn."""
    result = _parse("live_end_turn.jsonl")

    assert result.tail_facts.stop_markers == (
        StopMarker(idx=1, reason="tool_use", ends_turn=False),
        StopMarker(idx=3, reason="end_turn", ends_turn=True),
    )


def test_failed_tool_result_is_flagged() -> None:
    result = _parse("live_tool_error.jsonl")

    assert len(result.tail_facts.tool_results) == 1
    tool_result = result.tail_facts.tool_results[0]
    assert tool_result.tool_use_id == "toolu_err_read"
    assert tool_result.is_error is True
    assert tool_result.result_summary == "ENOENT: no such file or directory"


def test_result_summary_is_truncated() -> None:
    from recall.parsers.common import TOOL_RESULT_SUMMARY_MAX_CHARS, summarize_tool_result

    assert summarize_tool_result("x" * (TOOL_RESULT_SUMMARY_MAX_CHARS + 500)) == (
        "x" * TOOL_RESULT_SUMMARY_MAX_CHARS
    )
    assert summarize_tool_result("short") == "short"


@pytest.mark.parametrize("name", ["session1.jsonl", "session_v2.jsonl", "session_subagent.jsonl"])
def test_existing_fixtures_keep_their_messages_and_tool_calls(name: str) -> None:
    """tail_facts is additive: no existing parsed row moves."""
    result = _parse(name)

    assert result.session.message_count == len(result.session.messages)
    assert result.session.tool_count == sum(
        len(message.tool_calls) for message in result.session.messages
    )
    assert result.tail_facts.session_ended is False


def _parse_codex(relative: str):
    from recall.parsers.codex import CodexParser

    return CodexParser().parse(FIXTURES / "codex" / relative)


def test_codex_pairs_tool_outputs_by_call_id() -> None:
    """REQ-LIVE-005: codex joins a call to its output on `call_id`."""
    result = _parse_codex("live_tool_pair/rollout.jsonl")

    by_id = {tool_result.tool_use_id: tool_result for tool_result in result.tail_facts.tool_results}
    assert sorted(by_id) == ["call_alpha", "call_beta"]
    assert by_id["call_alpha"].result_summary == "Applied patch"
    assert by_id["call_alpha"].is_error is False
    assert by_id["call_beta"].result_summary == "ls: /nope: No such file or directory"
    assert by_id["call_beta"].is_error is True


def test_codex_carries_call_id_onto_its_tool_calls() -> None:
    result = _parse_codex("live_tool_pair/rollout.jsonl")

    calls = result.session.orphan_tool_calls + [
        call for message in result.session.messages for call in message.tool_calls
    ]
    assert sorted(call.tool_use_id or "" for call in calls) == ["call_alpha", "call_beta"]


def test_codex_task_complete_is_the_end_of_turn_marker() -> None:
    result = _parse_codex("live_tool_pair/rollout.jsonl")

    assert result.tail_facts.stop_markers[-1].reason == "task_complete"
    assert result.tail_facts.stop_markers[-1].ends_turn is True


def test_codex_turn_aborted_is_the_end_of_turn_marker(tmp_path: Path) -> None:
    """Live Codex interrupts emit event_msg turn_aborted, not task_complete."""
    from recall.parsers.codex import CodexParser

    path = tmp_path / "rollout.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "abort-1"}}),
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {"type": "agent_message", "message": "working"},
                    }
                ),
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {"type": "turn_aborted", "reason": "interrupted"},
                    }
                ),
            ]
        )
        + "\n"
    )
    result = CodexParser().parse(path)

    assert result.diagnostics == ()
    assert [(marker.reason, marker.ends_turn) for marker in result.tail_facts.stop_markers] == [
        ("interrupted", True)
    ]


def test_codex_exec_wrapper_calls_carry_no_call_id() -> None:
    """One wrapper program fans out to N calls, so no single call owns its id.

    Attributing the wrapper's `call_id` to one of them would be a guess, and
    REQ-LIVE-005 forbids guessing; the result is simply unpairable.
    """
    result = _parse_codex("live_exec_wrapper/rollout.jsonl")

    calls = result.session.orphan_tool_calls
    assert len(calls) == 2
    assert [call.tool_use_id for call in calls] == [None, None]
    assert [tool_result.tool_use_id for tool_result in result.tail_facts.tool_results] == [
        "call_wrapper"
    ]


def test_codex_session_without_markers_reports_nothing() -> None:
    """A transcript with no outputs and no task_complete claims no turn state."""
    result = _parse_codex("session1/rollout.jsonl")

    assert result.tail_facts.tool_results == ()
    assert result.tail_facts.stop_markers == ()
    assert result.tail_facts.session_ended is False


def test_grok_pairs_tool_results_by_tool_call_id() -> None:
    """REQ-LIVE-005: grok emits results as their own events keyed by tool_call_id."""
    from recall.parsers.grok import GrokParser

    result = GrokParser().parse(FIXTURES / "grok" / "session1.jsonl")

    assert [tool_result.tool_use_id for tool_result in result.tail_facts.tool_results] == [
        "call-ls-1"
    ]
    assert result.tail_facts.tool_results[0].result_summary.startswith("total 12")
    calls = [call for message in result.session.messages for call in message.tool_calls]
    assert [call.tool_use_id for call in calls] == ["call-ls-1"]


def test_grok_reads_tool_calls_versus_text_as_turn_markers() -> None:
    """Grok has no named stop field; the assistant event itself is the vocabulary."""
    from recall.parsers.grok import GrokParser

    result = GrokParser().parse(FIXTURES / "grok" / "session1.jsonl")

    assert [
        (marker.idx, marker.reason, marker.ends_turn) for marker in result.tail_facts.stop_markers
    ] == [(2, "tool_calls", False), (4, "stop", True)]


def test_pi_agent_pairs_tool_results_by_tool_call_id() -> None:
    """Pi Agent's result is a whole message with role toolResult, not a block."""
    from recall.parsers.pi_agent import PiAgentParser

    result = PiAgentParser().parse(FIXTURES / "pi_agent" / "session1.jsonl")

    assert [tool_result.tool_use_id for tool_result in result.tail_facts.tool_results] == ["tool-1"]
    tool_result = result.tail_facts.tool_results[0]
    assert tool_result.result_summary.startswith("total 8")
    assert tool_result.is_error is False
    calls = [call for message in result.session.messages for call in message.tool_calls]
    assert [call.tool_use_id for call in calls] == ["tool-1"]


def test_pi_agent_records_native_stop_reason_and_a_text_only_fallback() -> None:
    """Live Pi sessions stamp stopReason=toolUse; a text-only assistant ends the turn."""
    from recall.parsers.pi_agent import PiAgentParser

    result = PiAgentParser().parse(FIXTURES / "pi_agent" / "session1.jsonl")

    assert [
        (marker.idx, marker.reason, marker.ends_turn) for marker in result.tail_facts.stop_markers
    ] == [(1, "toolUse", False), (3, "stop", True)]


def test_kimi_pairs_tool_results_and_records_finish_reasons() -> None:
    """Kimi emits tool.call / tool.result / step.end loop events."""
    from recall.parsers.kimi_code import KimiCodeParser

    result = KimiCodeParser().parse(
        FIXTURES / "kimi_code" / "session1" / "agents" / "main" / "wire.jsonl"
    )

    assert [tool_result.tool_use_id for tool_result in result.tail_facts.tool_results] == [
        "tool_call_1",
        "tool_call_2",
    ]
    assert result.tail_facts.tool_results[0].result_summary == "README.md\nsrc"
    # "stop" is kimi's end-of-turn reason; "tool_use" means the step continues.
    assert [(marker.reason, marker.ends_turn) for marker in result.tail_facts.stop_markers] == [
        ("tool_use", False),
        ("tool_use", False),
        ("stop", True),
    ]
