"""Per-adapter normalization checkpoint contract (REQ-INDEX-026).

Every supported adapter must say whether the boundary it just captured can be
resumed, and, when it can, return enough state that normalizing the next suffix
produces exactly what a full parse of the whole file would have produced.

Each case carries a record list whose shapes mirror ``fixtures/<source>/`` but
are declared inline: the split points below are the contract under test, and a
fixture edit must not silently move a boundary from closed to open.

- ``closed_split`` ends on a record the adapter can resume from.
- ``open_splits`` end where the adapter must decline (REQ-INDEX-026 names Grok
  reasoning/backend-tool state and Kimi Code's open assistant step). Grok has
  two: a bare reasoning record, and -- less obviously -- a trailing assistant
  message, because a later ``backend_tool_call`` amends the previous assistant
  message in place and a suffix result has no way to express that amendment.
- ``closed_split_carries_state`` marks the adapters whose resumable boundary
  still owes the suffix something -- Codex's in-progress turn.

The tests after ``CASES`` cover what a split-point table cannot say: the
per-adapter refusal to normalize against a resume state this build does not
understand, and the Codex boundaries whose openness is a property of the whole
file rather than of its last record.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from recall.core.models import (
    NORMALIZATION_CHECKPOINT_BYTES_MAX,
    NormalizationCheckpoint,
    ParseResult,
    StopMarker,
)
from recall.parsers.claude_code import ClaudeCodeParser
from recall.parsers.codex import CodexParser
from recall.parsers.grok import GrokParser
from recall.parsers.kimi_code import KimiCodeParser
from recall.parsers.pi_agent import PiAgentParser
from recall.parsers.protocol import SessionParser
from recall.services.coordinator import parser_revision


def _line(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


CLAUDE_RECORDS = (
    _line(
        {
            "type": "user",
            "uuid": "u1",
            "timestamp": "2026-01-15T10:00:00Z",
            "cwd": "/repo",
            "sessionId": "cc-1",
            "gitBranch": "main",
            "message": {"role": "user", "content": [{"type": "text", "text": "run the tests"}]},
        }
    ),
    _line(
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "timestamp": "2026-01-15T10:00:04Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [
                    {"type": "text", "text": "Running them now."},
                    {
                        "type": "tool_use",
                        "id": "toolu_bash",
                        "name": "Bash",
                        "input": {"command": "uv run pytest -q"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 120, "output_tokens": 40},
            },
        }
    ),
    _line(
        {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "a1",
            "timestamp": "2026-01-15T10:00:46Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_bash",
                        "content": "312 passed",
                        "is_error": False,
                    }
                ],
            },
        }
    ),
    # closed_split = 4: a completed assistant turn.
    _line(
        {
            "type": "assistant",
            "uuid": "a2",
            "parentUuid": "u2",
            "timestamp": "2026-01-15T10:00:48Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [{"type": "text", "text": "All 312 tests pass."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 180, "output_tokens": 12},
            },
        }
    ),
    _line(
        {
            "type": "user",
            "uuid": "u3",
            "parentUuid": "a2",
            "timestamp": "2026-01-15T10:01:00Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "now lint"}]},
        }
    ),
    _line(
        {
            "type": "assistant",
            "uuid": "a3",
            "parentUuid": "u3",
            "timestamp": "2026-01-15T10:01:06Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [
                    {"type": "thinking", "thinking": "Ruff first."},
                    {"type": "text", "text": "Lint is clean."},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 200, "output_tokens": 18},
            },
        }
    ),
)

CODEX_RECORDS = (
    _line(
        {
            "type": "session_meta",
            "payload": {
                "id": "codex-1",
                "timestamp": "2026-01-16T12:00:00Z",
                "cwd": "/repo",
                "git": {"branch": "main", "root": "/repo"},
            },
        }
    ),
    _line(
        {
            "type": "event_msg",
            "timestamp": "2026-01-16T12:00:01Z",
            "payload": {"type": "task_started"},
        }
    ),
    _line(
        {
            "type": "event_msg",
            "timestamp": "2026-01-16T12:00:02Z",
            "payload": {"type": "user_message", "message": "list the repo"},
        }
    ),
    # closed_split = 4: resumable, but the turn opened at record 2 is still
    # running, so the suffix cannot reproduce the task_started markers without
    # the carried state.
    _line(
        {
            "type": "event_msg",
            "timestamp": "2026-01-16T12:00:03Z",
            "payload": {"type": "agent_message", "message": "Working on it."},
        }
    ),
    _line(
        {
            "type": "response_item",
            "timestamp": "2026-01-16T12:00:04Z",
            "payload": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_alpha",
                "name": "shell",
                "parameters": {"command": "ls -la"},
            },
        }
    ),
    _line(
        {
            "type": "response_item",
            "timestamp": "2026-01-16T12:00:05Z",
            "payload": {
                "type": "function_call_output",
                "id": "fco_1",
                "call_id": "call_alpha",
                "output": {"content": "README.md", "success": True},
            },
        }
    ),
    # The appended assistant messages are why the carry matters: the first of
    # them keeps its task_started marker only if the suffix knows the turn that
    # opened before the checkpoint is still running. task_complete overwrites
    # the marker on the last message, so a single appended message would hide
    # a dropped carry.
    _line(
        {
            "type": "event_msg",
            "timestamp": "2026-01-16T12:00:06Z",
            "payload": {"type": "agent_message", "message": "The repo has a README."},
        }
    ),
    _line(
        {
            "type": "event_msg",
            "timestamp": "2026-01-16T12:00:07Z",
            "payload": {"type": "agent_message", "message": "Anything else?"},
        }
    ),
    _line(
        {
            "type": "event_msg",
            "timestamp": "2026-01-16T12:00:08Z",
            "payload": {"type": "task_complete"},
        }
    ),
)

GROK_RECORDS = (
    _line({"type": "system", "content": [{"type": "text", "text": "You are Grok."}]}),
    _line({"type": "user", "content": [{"type": "text", "text": "List the files."}]}),
    _line(
        {
            "type": "assistant",
            "model_id": "grok-4.3",
            "reasoning": {"text": "Run ls.", "encrypted": "enc1", "id": "r-1"},
            "tool_calls": [
                {"id": "call-ls-1", "name": "bash", "arguments": '{"command":"ls -la"}'}
            ],
        }
    ),
    # closed_split = 4: no pending reasoning or backend-tool state, and the
    # trailing message is not an assistant message a later record can amend.
    _line({"type": "tool_result", "tool_call_id": "call-ls-1", "content": "README.md\n"}),
    # open boundary at 5: nothing is pending, but this trailing assistant
    # message is exactly what the backend_tool_call two records later amends.
    _line({"type": "assistant", "model_id": "grok-4.3", "content": "The repo has a README."}),
    # open boundary at 6: a bare reasoning record is thinking the next
    # assistant message has not claimed yet.
    _line(
        {
            "type": "reasoning",
            "id": "rs-1",
            "summary": [{"type": "summary_text", "text": "Search the release notes."}],
            "encrypted_content": "opaque",
            "status": "completed",
        }
    ),
    _line(
        {
            "type": "backend_tool_call",
            "kind": {
                "tool_type": "web_search",
                "action": {"type": "search", "query": "zig release notes"},
                "id": "ws-1",
                "status": "completed",
            },
        }
    ),
    _line({"type": "assistant", "model_id": "grok-4.3", "content": "Here is what I found."}),
    # A following non-assistant record closes the otherwise open trailing-
    # assistant boundary. Without this, the full fixture would contradict
    # REQ-INDEX-026 by declaring a trailing assistant resumable.
    _line({"type": "user", "content": [{"type": "text", "text": "Thanks."}]}),
)

KIMI_RECORDS = (
    _line({"type": "metadata", "protocol_version": "1.4", "created_at": 1784300000000}),
    _line(
        {
            "type": "config.update",
            "profileName": "agent",
            "modelAlias": "kimi-code/k3",
            "time": 1784300000010,
        }
    ),
    _line(
        {
            "type": "context.append_message",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "List the files."}],
                "toolCalls": [],
                "origin": {"kind": "user"},
            },
            "time": 1784300001000,
        }
    ),
    _line(
        {
            "type": "context.append_loop_event",
            "event": {"type": "step.begin", "uuid": "step-1", "turnId": "0", "step": 1},
            "time": 1784300001100,
        }
    ),
    _line(
        {
            "type": "context.append_loop_event",
            "event": {
                "type": "content.part",
                "uuid": "part-1",
                "turnId": "0",
                "step": 1,
                "stepUuid": "step-1",
                "part": {"type": "think", "think": "Run ls."},
            },
            "time": 1784300001500,
        }
    ),
    _line(
        {
            "type": "context.append_loop_event",
            "event": {
                "type": "content.part",
                "uuid": "part-2",
                "turnId": "0",
                "step": 1,
                "stepUuid": "step-1",
                "part": {"type": "text", "text": "Listing the files."},
            },
            "time": 1784300001600,
        }
    ),
    _line(
        {
            "type": "context.append_loop_event",
            "event": {
                "type": "tool.call",
                "uuid": "call-1",
                "name": "Bash",
                "args": {"command": "ls -la"},
                "toolCallId": "tc-1",
            },
            "time": 1784300001700,
        }
    ),
    # closed_split = 8: step.end flushed the open assistant step.
    _line(
        {
            "type": "context.append_loop_event",
            "event": {
                "type": "step.end",
                "uuid": "step-1",
                "turnId": "0",
                "step": 1,
                "finishReason": "tool_calls",
            },
            "time": 1784300001800,
        }
    ),
    _line(
        {
            "type": "context.append_loop_event",
            "event": {"type": "tool.result", "toolCallId": "tc-1", "result": "README.md\n"},
            "time": 1784300002000,
        }
    ),
    _line(
        {
            "type": "context.append_loop_event",
            "event": {"type": "step.begin", "uuid": "step-2", "turnId": "0", "step": 2},
            "time": 1784300002100,
        }
    ),
    # open boundary at 11: the second assistant step is still accumulating parts.
    _line(
        {
            "type": "context.append_loop_event",
            "event": {
                "type": "content.part",
                "uuid": "part-3",
                "turnId": "0",
                "step": 2,
                "stepUuid": "step-2",
                "part": {"type": "text", "text": "The repo has a README."},
            },
            "time": 1784300002200,
        }
    ),
    _line(
        {
            "type": "context.append_loop_event",
            "event": {
                "type": "step.end",
                "uuid": "step-2",
                "turnId": "0",
                "step": 2,
                "finishReason": "stop",
            },
            "time": 1784300002300,
        }
    ),
)

PI_RECORDS = (
    _line(
        {
            "type": "session",
            "version": 3,
            "id": "pi-session-1",
            "timestamp": "2026-03-10T02:35:42.937Z",
            "cwd": "/repo",
        }
    ),
    _line(
        {
            "type": "model_change",
            "id": "model-1",
            "parentId": None,
            "timestamp": "2026-03-10T02:35:42.937Z",
            "provider": "openai-codex",
            "modelId": "gpt-5.4",
        }
    ),
    _line(
        {
            "type": "message",
            "id": "msg-user-1",
            "parentId": "model-1",
            "timestamp": "2026-03-10T02:35:43.000Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "List the files."}]},
        }
    ),
    _line(
        {
            "type": "message",
            "id": "msg-assistant-1",
            "parentId": "msg-user-1",
            "timestamp": "2026-03-10T02:35:44.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Inspect the tree first."},
                    {"type": "text", "text": "Checking the contents."},
                    {
                        "type": "toolCall",
                        "id": "tool-1",
                        "name": "bash",
                        "arguments": {"command": "ls -la"},
                    },
                ],
                "usage": {"input": 100, "output": 20, "cacheRead": 5000, "cacheWrite": 10},
                "stopReason": "toolUse",
                "timestamp": 1773110144000,
            },
        }
    ),
    # closed_split = 5: a completed tool result.
    _line(
        {
            "type": "message",
            "id": "msg-tool-1",
            "parentId": "msg-assistant-1",
            "timestamp": "2026-03-10T02:35:44.100Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "tool-1",
                "toolName": "bash",
                "content": [{"type": "text", "text": "README.md\n"}],
                "isError": False,
                "timestamp": 1773110144100,
            },
        }
    ),
    _line(
        {
            "type": "message",
            "id": "msg-assistant-2",
            "parentId": "msg-tool-1",
            "timestamp": "2026-03-10T02:35:45.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "I found the project files."}],
                "usage": {"input": 50, "output": 10, "cacheRead": 2000, "cacheWrite": 0},
            },
        }
    ),
)


@dataclass(frozen=True)
class AdapterCase:
    name: str
    records: tuple[str, ...]
    relative_path: str
    closed_split: int
    # (label, record count) pairs whose prefix the adapter must refuse to resume.
    open_splits: tuple[tuple[str, int], ...]
    closed_split_carries_state: bool
    sidecars: dict[str, str] = field(default_factory=dict)

    def parser(self, root: Path) -> SessionParser:
        factories = {
            "claude_code": ClaudeCodeParser,
            "codex": CodexParser,
            "grok": GrokParser,
            "kimi_code": KimiCodeParser,
            "pi_agent": PiAgentParser,
        }
        return factories[self.name](roots=(root,))

    def write(self, root: Path, count: int) -> Path:
        path = root / self.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        for name, body in self.sidecars.items():
            (path.parent.parent.parent / name).write_text(body, encoding="utf-8")
        path.write_text("".join(f"{record}\n" for record in self.records[:count]), encoding="utf-8")
        return path


CASES = (
    AdapterCase(
        name="claude_code",
        records=CLAUDE_RECORDS,
        relative_path="projects/repo/session.jsonl",
        closed_split=4,
        open_splits=(),
        closed_split_carries_state=False,
    ),
    AdapterCase(
        name="codex",
        records=CODEX_RECORDS,
        relative_path="sessions/2026/01/16/rollout-codex-1.jsonl",
        closed_split=4,
        open_splits=(),
        closed_split_carries_state=True,
    ),
    AdapterCase(
        name="grok",
        records=GROK_RECORDS,
        relative_path="sessions/sess-1/chat_history.jsonl",
        closed_split=4,
        open_splits=(("trailing assistant message", 5), ("pending reasoning", 6)),
        closed_split_carries_state=False,
    ),
    AdapterCase(
        name="kimi_code",
        records=KIMI_RECORDS,
        relative_path="workdir-key/sess-1/agents/main/wire.jsonl",
        closed_split=8,
        # 10 is the less obvious one: `step.begin` has opened a step that has
        # produced no content yet, and an empty open step is still open.
        open_splits=(("empty open step", 10), ("open assistant step", 11)),
        closed_split_carries_state=False,
        sidecars={"state.json": '{"workDir":"/repo"}'},
    ),
    AdapterCase(
        name="pi_agent",
        records=PI_RECORDS,
        relative_path="sessions/session.jsonl",
        closed_split=5,
        open_splits=(),
        closed_split_carries_state=False,
    ),
)

RESUMABLE_CASES = [pytest.param(case, id=case.name) for case in CASES]
OPEN_BOUNDARY_CASES = [
    pytest.param(case, split, id=f"{case.name}-{label.replace(' ', '-')}")
    for case in CASES
    for label, split in case.open_splits
]


def _message_facts(result: ParseResult) -> list[tuple[object, ...]]:
    """Every identity and payload field an append must not disturb."""
    return [
        (
            message.idx,
            message.id,
            message.session_id,
            message.role.value,
            message.content,
            message.thinking,
            message.has_thinking,
            message.timestamp,
            message.agent_id,
            tuple(
                (
                    call.idx,
                    call.id,
                    call.message_id,
                    call.session_id,
                    call.tool_name,
                    json.dumps(call.tool_input, sort_keys=True),
                    call.tool_use_id,
                    call.bash_command,
                    call.agent_id,
                    call.skill_name,
                )
                for call in message.tool_calls
            ),
        )
        for message in result.session.messages
    ]


def _orphan_facts(result: ParseResult) -> list[tuple[object, ...]]:
    return [
        (call.idx, call.id, call.session_id, call.tool_name, call.tool_use_id, call.bash_command)
        for call in result.session.orphan_tool_calls
    ]


def _stop_markers(*results: ParseResult) -> dict[int, StopMarker]:
    """The parser's own last-wins rule, applied across a resumed pair."""
    merged: dict[int, StopMarker] = {}
    for result in results:
        for marker in result.tail_facts.stop_markers:
            merged[marker.idx] = marker
    return merged


def _tool_results(*results: ParseResult) -> tuple[object, ...]:
    return tuple(
        (item.tool_use_id, item.result_summary, item.is_error, item.completed_at)
        for result in results
        for item in result.tail_facts.tool_results
    )


def _resume(parser: SessionParser, path: Path, checkpoint: Any) -> ParseResult:
    return parser.parse(
        path,
        offset=checkpoint.offset,
        message_idx_base=checkpoint.message_idx_base,
        orphan_tool_call_idx_base=checkpoint.orphan_tool_call_idx_base,
        resume_state=checkpoint.adapter_state,
    )


@pytest.mark.parametrize("case", RESUMABLE_CASES)
def test_full_parse_declares_a_verified_checkpoint(case: AdapterCase, tmp_path: Path) -> None:
    from recall.core.models import NORMALIZATION_CHECKPOINT_VERSION, NormalizationCheckpoint

    parser = case.parser(tmp_path)
    path = case.write(tmp_path, len(case.records))
    result = parser.parse(path)

    checkpoint = result.normalization_checkpoint
    assert checkpoint is not None
    assert checkpoint.version == NORMALIZATION_CHECKPOINT_VERSION
    # The revision the catalog stores; a parser change must invalidate resume.
    assert checkpoint.parser_revision == parser_revision(type(parser))
    assert checkpoint.offset == result.next_byte_offset
    assert checkpoint.prefix_sha256 == result.committed_prefix_sha256
    assert checkpoint.message_idx_base == len(result.session.messages)
    assert checkpoint.orphan_tool_call_idx_base == len(result.session.orphan_tool_calls)
    source_stat = path.stat()
    assert checkpoint.source_dev == source_stat.st_dev
    assert checkpoint.source_inode == source_stat.st_ino
    # The envelope is persisted as text in one catalog column, so it has to
    # survive the round trip exactly.
    assert NormalizationCheckpoint.decode(checkpoint.encode()) == checkpoint


@pytest.mark.parametrize("case", RESUMABLE_CASES)
def test_resumed_append_matches_the_full_parse(case: AdapterCase, tmp_path: Path) -> None:
    parser = case.parser(tmp_path)
    path = case.write(tmp_path, case.closed_split)
    prefix = parser.parse(path)
    checkpoint = prefix.normalization_checkpoint
    assert checkpoint is not None, "closed boundary must be resumable"

    case.write(tmp_path, len(case.records))
    reference = parser.parse(path)
    resumed = _resume(parser, path, checkpoint)

    assert resumed.is_full_parse is False
    assert resumed.next_byte_offset == reference.next_byte_offset
    assert _message_facts(prefix) + _message_facts(resumed) == _message_facts(reference)
    assert _orphan_facts(prefix) + _orphan_facts(resumed) == _orphan_facts(reference)
    assert _tool_results(prefix, resumed) == _tool_results(reference)
    assert _stop_markers(prefix, resumed) == _stop_markers(reference)
    assert resumed.tail_facts.session_ended == reference.tail_facts.session_ended
    assert resumed.diagnostics == reference.diagnostics
    assert resumed.committed_prefix_sha256 == reference.committed_prefix_sha256
    # The resumed pass must leave the same proof a full parse would have left,
    # or the next append inherits a weaker checkpoint than a reparse would.
    assert resumed.normalization_checkpoint == reference.normalization_checkpoint


@pytest.mark.parametrize("case", RESUMABLE_CASES)
def test_resumed_append_carries_only_the_state_the_adapter_owes(
    case: AdapterCase, tmp_path: Path
) -> None:
    parser = case.parser(tmp_path)
    path = case.write(tmp_path, case.closed_split)
    checkpoint = parser.parse(path).normalization_checkpoint
    assert checkpoint is not None

    # Adapter state is persisted as JSON, so it must already be JSON data.
    assert json.loads(json.dumps(checkpoint.adapter_state)) == checkpoint.adapter_state
    if case.closed_split_carries_state:
        assert checkpoint.adapter_state, "an open turn must be carried into the suffix"
    else:
        assert checkpoint.adapter_state == {}, "a stateless boundary must carry nothing"


@pytest.mark.parametrize("case", RESUMABLE_CASES)
def test_resumed_append_reports_the_same_tail_metadata_as_a_full_parse(
    case: AdapterCase, tmp_path: Path
) -> None:
    """The row the caller merges into must not disagree about the file's tail."""
    parser = case.parser(tmp_path)
    path = case.write(tmp_path, case.closed_split)
    checkpoint = parser.parse(path).normalization_checkpoint
    assert checkpoint is not None

    case.write(tmp_path, len(case.records))
    reference = parser.parse(path)
    resumed = _resume(parser, path, checkpoint)

    assert resumed.session.id == reference.session.id
    assert resumed.session.source_path == reference.session.source_path
    assert resumed.session.ended_at == reference.session.ended_at
    assert resumed.session.file_size == reference.session.file_size
    assert resumed.session.sidecar_mtime == reference.session.sidecar_mtime


@pytest.mark.parametrize("case", RESUMABLE_CASES)
def test_replaying_the_same_suffix_is_idempotent(case: AdapterCase, tmp_path: Path) -> None:
    """A retried writer turn re-parses the same suffix and must not drift."""
    parser = case.parser(tmp_path)
    path = case.write(tmp_path, case.closed_split)
    checkpoint = parser.parse(path).normalization_checkpoint
    assert checkpoint is not None

    case.write(tmp_path, len(case.records))
    first = _resume(parser, path, checkpoint)
    second = _resume(parser, path, checkpoint)

    assert _message_facts(first) == _message_facts(second)
    assert _orphan_facts(first) == _orphan_facts(second)
    assert _stop_markers(first) == _stop_markers(second)
    assert first.normalization_checkpoint == second.normalization_checkpoint


@pytest.mark.parametrize(("case", "split"), OPEN_BOUNDARY_CASES)
def test_open_boundary_declines_to_resume(case: AdapterCase, split: int, tmp_path: Path) -> None:
    """Declining is the correctness path, not an excuse to skip the adapter.

    The prefix still commits -- the adapter simply refuses to promise the
    suffix can be normalized without re-reading it.
    """
    parser = case.parser(tmp_path)
    path = case.write(tmp_path, split)
    result = parser.parse(path)

    assert result.normalization_checkpoint is None
    assert result.is_full_parse is True
    assert result.committed_prefix_sha256 is not None


@pytest.mark.parametrize(("case", "split"), OPEN_BOUNDARY_CASES)
def test_the_boundary_after_an_open_one_is_resumable_again(
    case: AdapterCase, split: int, tmp_path: Path
) -> None:
    """Declining is per boundary, not a permanent opt-out for the adapter."""
    parser = case.parser(tmp_path)
    path = case.write(tmp_path, split)
    assert parser.parse(path).normalization_checkpoint is None

    case.write(tmp_path, len(case.records))
    reopened = parser.parse(path)

    assert reopened.normalization_checkpoint is not None
    assert reopened.normalization_checkpoint.offset == reopened.next_byte_offset


def _codex_case() -> AdapterCase:
    return next(case for case in CASES if case.name == "codex")


def test_codex_carries_turn_in_progress_across_the_checkpoint(tmp_path: Path) -> None:
    """The split lands inside a started, uncompleted Codex turn.

    The first appended assistant message keeps its ``task_started`` marker only
    while the turn opened before the checkpoint is known to still be running.
    A suffix parsed without that carry loses the marker outright.
    """
    case = _codex_case()
    parser = case.parser(tmp_path)
    path = case.write(tmp_path, case.closed_split)
    checkpoint = parser.parse(path).normalization_checkpoint
    assert checkpoint is not None
    assert checkpoint.adapter_state

    case.write(tmp_path, len(case.records))
    reference = parser.parse(path)
    resumed = _resume(parser, path, checkpoint)

    first_appended = checkpoint.message_idx_base
    reference_markers = {marker.idx: marker.reason for marker in reference.tail_facts.stop_markers}
    assert reference_markers[first_appended] == "task_started"
    resumed_markers = {marker.idx: marker.reason for marker in resumed.tail_facts.stop_markers}
    assert resumed_markers[first_appended] == "task_started"


def test_a_stateless_adapter_resumes_without_adapter_state(tmp_path: Path) -> None:
    """Claude Code's boundary is stateless: an empty state must still resume."""
    case = next(item for item in CASES if item.name == "claude_code")
    parser = case.parser(tmp_path)
    path = case.write(tmp_path, case.closed_split)
    checkpoint = parser.parse(path).normalization_checkpoint
    assert checkpoint is not None

    case.write(tmp_path, len(case.records))
    resumed = parser.parse(
        path,
        offset=checkpoint.offset,
        message_idx_base=checkpoint.message_idx_base,
        orphan_tool_call_idx_base=checkpoint.orphan_tool_call_idx_base,
    )

    assert resumed.is_full_parse is False
    assert [message.idx for message in resumed.session.messages] == [
        checkpoint.message_idx_base,
        checkpoint.message_idx_base + 1,
    ]


def test_every_supported_adapter_is_covered() -> None:
    """A new source must not silently opt out of the checkpoint contract."""
    from recall.core.types import Source

    assert {case.name for case in CASES} == {source.value for source in Source}


@pytest.mark.parametrize("case", RESUMABLE_CASES)
@pytest.mark.parametrize(
    "state",
    [
        pytest.param({"invented_by_a_later_build": 1}, id="unknown-key"),
        pytest.param({"turn_in_progress": "yes"}, id="wrong-value-type"),
        pytest.param(["turn_in_progress"], id="not-an-object"),
    ],
)
def test_a_resume_state_this_build_cannot_read_is_distinguishable(
    case: AdapterCase, state: Any, tmp_path: Path
) -> None:
    """A stored state no adapter here understands must not be quietly ignored.

    Ignoring it would normalize the suffix against state the adapter had to
    guess at; raising something generic would let an arbitrary parser defect
    masquerade as a routine fallback. The caller gets one named error so it can
    take the full-parse path deliberately.
    """
    from recall.parsers.checkpoint import UnsupportedResumeState

    parser = case.parser(tmp_path)
    path = case.write(tmp_path, case.closed_split)
    checkpoint = parser.parse(path).normalization_checkpoint
    assert checkpoint is not None

    case.write(tmp_path, len(case.records))
    with pytest.raises(UnsupportedResumeState):
        parser.parse(path, offset=checkpoint.offset, resume_state=state)


@pytest.mark.parametrize("case", RESUMABLE_CASES)
def test_resume_state_offered_at_offset_zero_is_refused(case: AdapterCase, tmp_path: Path) -> None:
    """There is no prefix at offset zero for carried state to describe."""
    from recall.parsers.checkpoint import UnsupportedResumeState

    parser = case.parser(tmp_path)
    path = case.write(tmp_path, len(case.records))

    with pytest.raises(UnsupportedResumeState):
        parser.parse(path, resume_state={"turn_in_progress": True})


def _codex_file(tmp_path: Path, records: tuple[str, ...]) -> tuple[CodexParser, Path]:
    path = tmp_path / "sessions/2026/01/16/rollout-codex-2.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{record}\n" for record in records), encoding="utf-8")
    return CodexParser(roots=(tmp_path,)), path


CODEX_SESSION_META = _line(
    {
        "type": "session_meta",
        "payload": {"id": "codex-2", "timestamp": "2026-01-16T12:00:00Z", "cwd": "/repo"},
    }
)


def _codex_response_user(text: str) -> str:
    return _line(
        {
            "type": "response_item",
            "timestamp": "2026-01-16T12:00:01Z",
            "payload": {
                "type": "message",
                "role": "user",
                "turn_id": "turn-1",
                "content": [{"type": "input_text", "text": text}],
            },
        }
    )


def _codex_mirror_user(text: str) -> str:
    return _line(
        {
            "type": "event_msg",
            "timestamp": "2026-01-16T12:00:02Z",
            "payload": {
                "type": "item_completed",
                "turn_id": "turn-1",
                "item": {
                    "type": "UserMessage",
                    "id": "item-user-1",
                    "content": [{"type": "Text", "text": text}],
                },
            },
        }
    )


def test_codex_carries_an_unmatched_response_across_the_checkpoint(tmp_path: Path) -> None:
    """A native mirror appended after its SDK response must not duplicate it.

    This is the ordinary modern rollout ordering: the prefix can end after the
    response record and the matching item_completed event arrives in the next
    filesystem append. A suffix that starts with empty occurrence counters
    emits the mirror again, unlike a full parse.
    """
    response = _codex_response_user("list the repo")
    mirror = _codex_mirror_user("list the repo")
    parser, path = _codex_file(tmp_path, (CODEX_SESSION_META, response))
    prefix = parser.parse(path)
    checkpoint = prefix.normalization_checkpoint
    assert checkpoint is not None
    assert checkpoint.adapter_state, "the unmatched response must be carried"

    path.write_text(
        "".join(f"{record}\n" for record in (CODEX_SESSION_META, response, mirror)),
        encoding="utf-8",
    )
    reference = parser.parse(path)
    resumed = _resume(parser, path, checkpoint)

    assert _message_facts(prefix) + _message_facts(resumed) == _message_facts(reference)


def test_codex_balanced_mixed_representation_boundary_is_resumable(tmp_path: Path) -> None:
    """Once a response/mirror pair balances, mixed format is not permanently open."""
    parser, path = _codex_file(
        tmp_path,
        (
            CODEX_SESSION_META,
            _codex_response_user("list the repo"),
            _codex_mirror_user("list the repo"),
        ),
    )

    result = parser.parse(path)

    assert [message.content for message in result.session.messages] == ["list the repo"]
    assert result.normalization_checkpoint is not None


def test_codex_resumes_across_a_deferred_completion_item(tmp_path: Path) -> None:
    """Deferred native items are ordinary, so deferring cannot forbid resume.

    `CommandExecution` items appear in every modern rollout. The walk holds
    them back to the end, which is why the records they produce are restored
    to file order and why the identities they are matched against travel in
    the checkpoint.
    """
    item = _line(
        {
            "type": "event_msg",
            "timestamp": "2026-01-16T12:00:02Z",
            "payload": {
                "type": "item_completed",
                "item": {"type": "WebSearch", "id": "ws-1", "action": {"query": "zig"}},
            },
        }
    )
    prefix = (
        CODEX_SESSION_META,
        _line(
            {
                "type": "event_msg",
                "timestamp": "2026-01-16T12:00:01Z",
                "payload": {"type": "user_message", "message": "search"},
            }
        ),
        item,
    )
    parser, path = _codex_file(tmp_path, prefix)
    result = parser.parse(path)

    assert result.session.orphan_tool_calls, "the deferred item is still parsed"
    assert result.normalization_checkpoint is not None

    _codex_resumed_equals_full(
        tmp_path,
        prefix,
        (
            _line(
                {
                    "type": "event_msg",
                    "timestamp": "2026-01-16T12:00:03Z",
                    "payload": {"type": "agent_message", "message": "Found it."},
                }
            ),
        ),
    )


def test_codex_carries_unbalanced_mixed_representation_state(tmp_path: Path) -> None:
    """Unmatched mirror/response occurrences remain resumable when carried.

    Pairing spans records, but an open correlation is bounded normalization
    state rather than permission to make modern mixed-format rollouts reparse
    their full history forever.
    """
    parser, path = _codex_file(
        tmp_path,
        (
            CODEX_SESSION_META,
            # A turn id is what makes this boundary unbalanced rather than
            # uncorrelated: without one the normalizer never pairs the two
            # representations at all, so nothing is left open to carry.
            _codex_response_user("list the repo"),
            _line(
                {
                    "type": "event_msg",
                    "timestamp": "2026-01-16T12:00:02Z",
                    "payload": {"type": "agent_message", "message": "Working on it."},
                }
            ),
        ),
    )
    result = parser.parse(path)

    assert [message.role.value for message in result.session.messages] == ["user", "assistant"]
    checkpoint = result.normalization_checkpoint
    assert checkpoint is not None
    assert checkpoint.adapter_state


def _codex_unpaired_mirrors(count: int) -> tuple[str, ...]:
    return tuple(
        _line(
            {
                "type": "event_msg",
                "timestamp": "2026-01-16T12:00:02Z",
                "payload": {
                    "type": "agent_message",
                    "turn_id": "turn-1",
                    "message": f"unmatched-{index}",
                },
            }
        )
        for index in range(count)
    )


def test_codex_large_open_correlation_state_remains_durably_resumable(tmp_path: Path) -> None:
    """Compressible correlation state must not silently lose the resume path."""
    prefix_records = (CODEX_SESSION_META, *_codex_unpaired_mirrors(900))
    parser, path = _codex_file(tmp_path, prefix_records)
    prefix = parser.parse(path)
    checkpoint = prefix.normalization_checkpoint
    assert checkpoint is not None
    assert len(json.dumps(checkpoint.adapter_state)) > NORMALIZATION_CHECKPOINT_BYTES_MAX

    encoded = checkpoint.encode()
    durable = NormalizationCheckpoint.decode(encoded)
    assert durable == checkpoint
    assert len(encoded) <= NORMALIZATION_CHECKPOINT_BYTES_MAX

    response = _codex_response_message("unmatched-899")
    _codex_append(path, (*prefix_records, response))
    resumed = _resume(parser, path, durable)
    reference = parser.parse(path)

    assert _message_facts(prefix) + _message_facts(resumed) == _message_facts(reference)


def test_codex_checkpoint_decline_is_operator_visible(tmp_path: Path, caplog: Any) -> None:
    """A genuine size refusal must explain a full-parse performance fallback."""
    parser, path = _codex_file(tmp_path, (CODEX_SESSION_META, *_codex_unpaired_mirrors(3_000)))

    with caplog.at_level("WARNING", logger="recall.parsers.checkpoint"):
        result = parser.parse(path)

    assert result.normalization_checkpoint is None
    assert any(
        "normalization checkpoint declined" in record.message
        and str(path) in record.message
        and "durable size bound" in record.message
        for record in caplog.records
    )


def test_codex_refuses_to_replay_a_compaction_snapshot_in_a_suffix(tmp_path: Path) -> None:
    """A compaction snapshot is deduplicated against history a suffix lacks.

    Replaying it would re-emit the whole conversation it replaced. The adapter
    diagnoses instead, which stops the capture acknowledging that record, so
    the caller re-reads the source from zero rather than committing duplicates.
    """
    user_record = _line(
        {
            "type": "response_item",
            "timestamp": "2026-01-16T12:00:01Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "list the repo"}],
            },
        }
    )
    compaction_record = _line(
        {
            "type": "compacted",
            "timestamp": "2026-01-16T12:00:02Z",
            "payload": {
                "replacement_history": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "text", "text": "list the repo"}],
                    }
                ]
            },
        }
    )
    parser, path = _codex_file(tmp_path, (CODEX_SESSION_META, user_record, compaction_record))

    reference = parser.parse(path)
    assert [message.content for message in reference.session.messages] == ["list the repo"]

    prefix_bytes = len(f"{CODEX_SESSION_META}\n{user_record}\n".encode())
    resumed = parser.parse(path, offset=prefix_bytes, message_idx_base=1)

    assert resumed.session.messages == []
    assert [item.kind for item in resumed.diagnostics] == ["unsupported_record"]
    assert resumed.next_byte_offset == prefix_bytes
    assert resumed.normalization_checkpoint is None


def _codex_response_message(text: str, *, item_id: str | None = None) -> str:
    payload: dict[str, Any] = {
        "type": "message",
        "role": "assistant",
        "turn_id": "turn-1",
        "content": [{"type": "output_text", "text": text}],
    }
    if item_id is not None:
        payload["id"] = item_id
    return _line({"type": "response_item", "timestamp": "2026-01-16T12:00:01Z", "payload": payload})


def _codex_mirror_message(text: str, *, item_id: str) -> str:
    return _line(
        {
            "type": "event_msg",
            "timestamp": "2026-01-16T12:00:02Z",
            "payload": {
                "type": "item_completed",
                "turn_id": "turn-1",
                "item": {
                    "type": "AgentMessage",
                    "id": item_id,
                    "content": [{"type": "Text", "text": text}],
                },
            },
        }
    )


def _split_offset(records: tuple[str, ...]) -> int:
    return len("".join(f"{record}\n" for record in records).encode())


def _codex_append(path: Path, records: tuple[str, ...]) -> None:
    path.write_text("".join(f"{record}\n" for record in records), encoding="utf-8")


def _codex_resumed_equals_full(
    tmp_path: Path, prefix: tuple[str, ...], suffix: tuple[str, ...]
) -> None:
    """Parse `prefix`, append `suffix`, and require the pair to equal a full parse."""
    parser, path = _codex_file(tmp_path, prefix)
    head = parser.parse(path)
    checkpoint = head.normalization_checkpoint
    assert checkpoint is not None, "the boundary must be resumable"

    _codex_append(path, prefix + suffix)
    reference = parser.parse(path)
    resumed = _resume(parser, path, checkpoint)

    assert _message_facts(head) + _message_facts(resumed) == _message_facts(reference)
    assert _orphan_facts(head) + _orphan_facts(resumed) == _orphan_facts(reference)
    assert _tool_results(head, resumed) == _tool_results(reference)
    assert resumed.normalization_checkpoint == reference.normalization_checkpoint


def test_codex_carries_an_unmatched_mirror_across_the_checkpoint(tmp_path: Path) -> None:
    """The imbalance runs both ways: a native mirror can precede its response."""
    _codex_resumed_equals_full(
        tmp_path,
        (CODEX_SESSION_META, _codex_mirror_user("list the repo")),
        (_codex_response_user("list the repo"),),
    )


def test_codex_carries_correlated_ids_across_the_checkpoint(tmp_path: Path) -> None:
    """The id channel is separate from the occurrence balance and must travel.

    These two records share an item id but normalize to different content, so
    the balance never pairs them. Only the carried id suppresses the second.
    """
    prefix = (CODEX_SESSION_META, _codex_response_message("alpha", item_id="item-x"))
    suffix = (_codex_mirror_message("beta", item_id="item-x"),)

    parser, path = _codex_file(tmp_path, prefix + suffix)
    assert [message.content for message in parser.parse(path).session.messages] == ["alpha"]

    _codex_resumed_equals_full(tmp_path, prefix, suffix)


def test_codex_carries_the_sticky_turn_id_across_the_checkpoint(tmp_path: Path) -> None:
    """A suffix record inherits the turn its prefix last named.

    Losing the sticky turn id silently disables the occurrence pairing for
    every appended record, which reads as "no duplicates found" rather than as
    an error.
    """
    prefix = (CODEX_SESSION_META, _codex_response_user("list the repo"))
    checkpoint = (
        _codex_file(tmp_path, prefix)[0]
        .parse(_codex_file(tmp_path, prefix)[1])
        .normalization_checkpoint
    )
    assert checkpoint is not None
    assert checkpoint.adapter_state["correlation"]["turn"] == "turn-1"

    # The appended mirror names no turn of its own and must still pair.
    _codex_resumed_equals_full(
        tmp_path,
        prefix,
        (
            _line(
                {
                    "type": "event_msg",
                    "timestamp": "2026-01-16T12:00:02Z",
                    "payload": {
                        "type": "item_completed",
                        "item": {
                            "type": "UserMessage",
                            "id": "item-user-1",
                            "content": [{"type": "Text", "text": "list the repo"}],
                        },
                    },
                }
            ),
        ),
    )


def test_codex_chained_appends_each_resume_from_the_previous_checkpoint(tmp_path: Path) -> None:
    """The ordinary case: append, commit, append again, never reparsing history."""
    records = (
        CODEX_SESSION_META,
        _codex_response_user("list the repo"),
        _codex_mirror_user("list the repo"),
        _codex_response_message("alpha", item_id="item-a"),
        _codex_mirror_message("alpha", item_id="item-a"),
        _codex_response_message("beta", item_id="item-b"),
        _codex_mirror_message("beta", item_id="item-b"),
    )
    parser, path = _codex_file(tmp_path, records[:2])

    results = [parser.parse(path)]
    for end in (4, 6, len(records)):
        checkpoint = results[-1].normalization_checkpoint
        assert checkpoint is not None, f"boundary before record {end} must be resumable"
        _codex_append(path, records[:end])
        results.append(_resume(parser, path, checkpoint))

    _codex_append(path, records)
    reference = parser.parse(path)
    merged: list[tuple[object, ...]] = []
    for result in results:
        merged.extend(_message_facts(result))

    assert merged == _message_facts(reference)
    assert results[-1].normalization_checkpoint == reference.normalization_checkpoint


def test_codex_a_balanced_boundary_carries_no_open_occurrences(tmp_path: Path) -> None:
    """A resolved pair leaves no occurrence entry behind.

    The occurrence balance is the part that would otherwise hold message
    content, so it is the part that must not accumulate. Correlated ids do
    persist -- an id can still be repeated after its pair resolved, and
    dropping it would emit a copy a full parse suppresses -- but they are
    short, fixed-size, and one per message at most.
    """
    pairs = tuple(
        record
        for index in range(12)
        for record in (
            _codex_response_message(f"step {index}", item_id=f"item-{index}"),
            _codex_mirror_message(f"step {index}", item_id=f"item-{index}"),
        )
    )
    parser, path = _codex_file(tmp_path, (CODEX_SESSION_META, *pairs))

    result = parser.parse(path)
    checkpoint = result.normalization_checkpoint

    assert len(result.session.messages) == 12, "each pair is one message"
    assert checkpoint is not None
    carried = checkpoint.adapter_state["correlation"]
    assert "balances" not in carried, "a resolved pair must leave no occurrence entry"
    assert carried["ids"] == sorted(f"item-{index}" for index in range(12))


_CODEX_SDK_WEB_SEARCH = _line(
    {
        "type": "response_item",
        "timestamp": "2026-01-16T12:00:01Z",
        "payload": {
            "type": "web_search_call",
            "id": "ws-1",
            "call_id": "ws-1",
            "action": {"type": "search", "query": "zig"},
        },
    }
)

_CODEX_NATIVE_WEB_SEARCH = _line(
    {
        "type": "event_msg",
        "timestamp": "2026-01-16T12:00:02Z",
        "payload": {
            "type": "item_completed",
            "item": {"type": "WebSearch", "id": "ws-1", "action": {"query": "zig"}},
        },
    }
)


def test_codex_falls_back_on_a_collidable_completion_item_in_a_suffix(tmp_path: Path) -> None:
    """A completion item outside the execution-view namespace still falls back.

    Matching it needs the identities of SDK tool records anywhere in the file,
    which is corpus-sized and deliberately not carried. The suffix diagnoses
    instead, so the capture stops acknowledging at that record and the caller
    re-reads from zero rather than committing a second call for one search.
    """
    prefix = (CODEX_SESSION_META, _CODEX_SDK_WEB_SEARCH)
    parser, path = _codex_file(tmp_path, (*prefix, _CODEX_NATIVE_WEB_SEARCH))
    assert len(parser.parse(path).session.orphan_tool_calls) == 1, "one invocation, one call"

    _codex_append(path, prefix)
    checkpoint = parser.parse(path).normalization_checkpoint
    assert checkpoint is not None

    _codex_append(path, (*prefix, _CODEX_NATIVE_WEB_SEARCH))
    resumed = _resume(parser, path, checkpoint)

    assert resumed.session.orphan_tool_calls == []
    assert [item.kind for item in resumed.diagnostics] == ["unsupported_record"]
    assert resumed.next_byte_offset == _split_offset(prefix)
    assert resumed.normalization_checkpoint is None


def test_codex_stops_trusting_the_execution_namespace_once_an_sdk_claims_it(
    tmp_path: Path,
) -> None:
    """The namespace argument is checked, not assumed.

    An execution view can be matched from a suffix only because no SDK record
    names an identity like one. A rollout that breaks that assumption is
    recorded in the checkpoint, and its suffixes fall back.
    """
    claiming_call = _line(
        {
            "type": "response_item",
            "timestamp": "2026-01-16T12:00:01Z",
            "payload": {
                "type": "function_call",
                "id": "exec-abc",
                "call_id": "exec-abc",
                "turn_id": "turn-1",
                "name": "shell",
                "arguments": '{"command": "ls"}',
            },
        }
    )
    exec_view = _line(
        {
            "type": "event_msg",
            "timestamp": "2026-01-16T12:00:02Z",
            "payload": {
                "type": "item_completed",
                "turn_id": "turn-1",
                "item": {
                    "type": "CommandExecution",
                    "id": "exec-abc",
                    "command": "ls",
                    "aggregated_output": "README.md",
                    "exit_code": 0,
                },
            },
        }
    )
    prefix = (CODEX_SESSION_META, claiming_call)
    parser, path = _codex_file(tmp_path, prefix)
    checkpoint = parser.parse(path).normalization_checkpoint
    assert checkpoint is not None
    assert checkpoint.adapter_state["correlation"]["exec_claimed"] is True

    _codex_append(path, (*prefix, exec_view))
    resumed = _resume(parser, path, checkpoint)

    assert [item.kind for item in resumed.diagnostics] == ["unsupported_record"]
    assert resumed.normalization_checkpoint is None


def test_codex_orders_tool_results_by_file_position_across_a_boundary(tmp_path: Path) -> None:
    """A deferred item's result must not overtake results appended after it.

    Native completion items are held back to the end of a walk, so a prefix
    emits its own at the prefix's end while a full parse emits them after the
    appended records. Reading both back in file position keeps a prefix and
    its suffix concatenating to exactly the full parse's sequence.
    """
    prefix = (
        CODEX_SESSION_META,
        _line(
            {
                "type": "event_msg",
                "timestamp": "2026-01-16T12:00:01Z",
                "payload": {"type": "user_message", "message": "run it"},
            }
        ),
        _line(
            {
                "type": "event_msg",
                "timestamp": "2026-01-16T12:00:02Z",
                "payload": {
                    "type": "item_completed",
                    "item": {
                        "type": "CommandExecution",
                        "id": "exec-abc",
                        "command": "ls -la",
                        "aggregated_output": "README.md",
                        "exit_code": 0,
                    },
                },
            }
        ),
    )
    suffix = (
        _line(
            {
                "type": "response_item",
                "timestamp": "2026-01-16T12:00:03Z",
                "payload": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_beta",
                    "name": "shell",
                    "arguments": '{"command": "pwd"}',
                },
            }
        ),
        _line(
            {
                "type": "response_item",
                "timestamp": "2026-01-16T12:00:04Z",
                "payload": {
                    "type": "function_call_output",
                    "id": "fco_1",
                    "call_id": "call_beta",
                    "output": {"content": "/repo", "success": True},
                },
            }
        ),
    )
    parser, path = _codex_file(tmp_path, (*prefix, *suffix))
    reference = parser.parse(path)
    assert [result.tool_use_id for result in reference.tail_facts.tool_results] == [
        "exec-abc",
        "call_beta",
    ], "the deferred result belongs where the file put it"

    _codex_resumed_equals_full(tmp_path, prefix, suffix)


def test_codex_carries_exec_turns_across_the_checkpoint(tmp_path: Path) -> None:
    """An exec wrapper's turn governs completion items appended after it.

    `exec-*` items are execution views of a wrapper program's decoded calls
    (REQ-PARSE-018). Losing the turn that ran the wrapper turns each of those
    views back into a top-level invocation.
    """
    wrapper = _line(
        {
            "type": "response_item",
            "timestamp": "2026-01-16T12:00:01Z",
            "payload": {
                "type": "custom_tool_call",
                "id": "call_exec_1",
                "call_id": "call_exec_1",
                "turn_id": "turn-1",
                "name": "exec",
                "input": 'bash({"command": "ls -la"})',
            },
        }
    )
    exec_view = _line(
        {
            "type": "event_msg",
            "timestamp": "2026-01-16T12:00:02Z",
            "payload": {
                "type": "item_completed",
                "turn_id": "turn-1",
                "item": {
                    "type": "CommandExecution",
                    "id": "exec-abc",
                    "command": "ls -la",
                    "aggregated_output": "README.md",
                    "exit_code": 0,
                },
            },
        }
    )
    prefix = (CODEX_SESSION_META, wrapper)
    checkpoint = _codex_file(tmp_path, prefix)[0].parse(_codex_file(tmp_path, prefix)[1])
    assert checkpoint.normalization_checkpoint is not None
    assert checkpoint.normalization_checkpoint.adapter_state["correlation"]["exec_turns"] == [
        "turn-1"
    ]

    _codex_resumed_equals_full(tmp_path, prefix, (exec_view,))


@pytest.mark.parametrize(
    "correlation",
    [
        pytest.param({"turn": 7}, id="turn-not-a-string"),
        pytest.param({"aliases": {"a": 1}}, id="alias-value-not-a-string"),
        pytest.param({"aliases": []}, id="aliases-not-an-object"),
        pytest.param({"balances": None}, id="balances-null"),
        pytest.param({"balances": [["turn-1", "digest"]]}, id="balance-wrong-arity"),
        pytest.param({"balances": [["turn-1", "digest", "1"]]}, id="balance-not-an-int"),
        pytest.param({"balances": [["turn-1", "digest", True]]}, id="balance-a-bool"),
        pytest.param({"balances": [["turn-1", "digest", 0]]}, id="balance-zero"),
        pytest.param({"balances": [["turn-1", "not-a-digest", 1]]}, id="invalid-digest"),
        pytest.param(
            {"balances": [["turn-1", "a" * 64, 1], ["turn-1", "a" * 64, 2]]},
            id="duplicate-balance",
        ),
        pytest.param({"ids": None}, id="ids-null"),
        pytest.param({"ids": [3]}, id="id-not-a-string"),
        pytest.param({"exec_turns": None}, id="exec-turns-null"),
        pytest.param({"anchor": ["turn-1", "digest"]}, id="anchor-wrong-arity"),
        pytest.param({"alias_pending": "turn-1"}, id="alias-pending-not-a-list"),
        pytest.param({"invented_by_a_later_build": True}, id="unknown-key"),
        pytest.param([], id="not-an-object"),
    ],
)
def test_codex_correlation_state_is_validated_fail_closed(correlation: Any, tmp_path: Path) -> None:
    """A correlation shape this build cannot read is named, not guessed at.

    A zero balance is included deliberately: it is indistinguishable from an
    absent entry, so storing one means the writer disagreed with this build's
    pruning rule and the rest of its state cannot be trusted either.
    """
    from recall.parsers.checkpoint import UnsupportedResumeState

    prefix = (CODEX_SESSION_META, _codex_response_user("list the repo"))
    parser, path = _codex_file(tmp_path, prefix)
    parser.parse(path)

    with pytest.raises(UnsupportedResumeState):
        parser.parse(path, offset=_split_offset(prefix), resume_state={"correlation": correlation})


# --- The revision's dependency set (REQ-INDEX-025) ---------------------------
#
# A stored checkpoint is trusted only while the code that produced it is
# unchanged, so the revision has to cover every first-party module whose edit
# could move a normalized row. A hand-maintained list rots silently and the
# damage -- one session holding rows normalized by two different builds -- is
# never healed by a reparse, because the reparse is exactly what the missing
# invalidation was supposed to force. The closure below is derived from the
# adapters' own imports so an omission fails here instead.

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "packages" / "recall" / "src"

# `revision` is the hashing mechanism, not an input to it: its only first-party
# import is the deferred one it uses to reach the shared modules it hashes.
_NOT_A_NORMALIZATION_DEPENDENCY = frozenset({"recall.parsers.revision"})


def _module_file(module_name: str) -> Path | None:
    candidate = _SOURCE_ROOT / (module_name.replace(".", "/") + ".py")
    if candidate.exists():
        return candidate
    package = _SOURCE_ROOT / module_name.replace(".", "/") / "__init__.py"
    return package if package.exists() else None


def _imported_modules(path: Path) -> set[str]:
    """First-party module names this file imports, at module or function scope."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if alias.name.startswith("recall"))
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            if not node.module.startswith("recall"):
                continue
            found.add(node.module)
            # `from recall.core import ids` names a module, not an attribute.
            found.update(
                name
                for alias in node.names
                if _module_file(name := f"{node.module}.{alias.name}") is not None
            )
    return found


def _first_party_closure(module_name: str) -> set[Path]:
    """Every first-party file the adapter's normalized output can depend on."""
    seen: set[str] = set()
    pending = [module_name]
    while pending:
        current = pending.pop()
        if current in seen or current in _NOT_A_NORMALIZATION_DEPENDENCY:
            continue
        path = _module_file(current)
        if path is None:
            continue
        seen.add(current)
        pending.extend(_imported_modules(path))
    files = {_module_file(name) for name in seen}
    return {path for path in files if path is not None}


ADAPTER_MODULES = (
    ("claude_code", ClaudeCodeParser),
    ("codex", CodexParser),
    ("grok", GrokParser),
    ("kimi_code", KimiCodeParser),
    ("pi_agent", PiAgentParser),
)


@pytest.mark.parametrize(
    ("module", "parser_type"),
    [pytest.param(module, parser_type, id=module) for module, parser_type in ADAPTER_MODULES],
)
def test_parser_revision_covers_every_normalization_dependency(
    module: str, parser_type: type
) -> None:
    """Editing anything the adapter normalizes with must invalidate resume."""
    from recall.parsers.revision import revision_inputs

    expected = _first_party_closure(f"recall.parsers.{module}")
    assert expected, "the closure walk must find the adapter itself"
    missing = sorted(str(path) for path in expected - set(revision_inputs(parser_type)))
    assert missing == [], f"{module} normalizes with files the revision does not hash"


def test_the_revision_hashes_nothing_beyond_the_adapters_dependencies() -> None:
    """The shared list is the union of the adapters' closures, and no more.

    Over-hashing is safe but not free: every extra file invalidates every
    stored checkpoint on the next edit to it, so the list must not drift into
    configuration or embedding code an adapter never reads.
    """
    from recall.parsers.revision import revision_inputs

    closure: set[Path] = set()
    adapters: set[Path] = set()
    for module, parser_type in ADAPTER_MODULES:
        closure |= _first_party_closure(f"recall.parsers.{module}")
        adapters.add(revision_inputs(parser_type)[0])

    shared = set(revision_inputs(CodexParser)) - adapters
    assert sorted(str(path) for path in shared) == sorted(str(path) for path in closure - adapters)


# --- Diagnosed boundaries (REQ-INDEX-025) -----------------------------------
#
# A half-written final line is what a watcher sees whenever it observes a
# transcript an agent is still writing, so it is the steady state on a live
# host rather than damage. The capture stops before that record without
# acknowledging it, so `next_byte_offset` and `committed_prefix_sha256` still
# describe the last complete boundary the adapter reasoned about -- and the
# record was never yielded, so the adapter's carried state matches it exactly.
# Every other diagnostic keeps declining.


@pytest.mark.parametrize("case", RESUMABLE_CASES)
def test_an_unterminated_tail_keeps_the_last_complete_boundary(
    case: AdapterCase, tmp_path: Path
) -> None:
    parser = case.parser(tmp_path)
    path = case.write(tmp_path, case.closed_split)
    complete = parser.parse(path)
    assert complete.normalization_checkpoint is not None

    path.write_bytes(path.read_bytes() + b'{"type": "assistant", "uuid": "torn"')
    torn = parser.parse(path)

    assert [diagnostic.kind for diagnostic in torn.diagnostics] == ["unterminated_tail"]
    assert torn.next_byte_offset == complete.next_byte_offset
    # Identical, not merely present: the torn parse saw no record the clean one
    # did not, so it owes the next suffix exactly the same boundary.
    assert torn.normalization_checkpoint == complete.normalization_checkpoint


@pytest.mark.parametrize("case", RESUMABLE_CASES)
def test_a_malformed_tail_record_still_declines(case: AdapterCase, tmp_path: Path) -> None:
    """Only the half-written line is admitted; damage is still unresumable.

    The offset must stay behind the bad record as well. That is what lets the
    incremental merge replace `is_complete` rather than latch it: the next
    parse re-reads the diagnosed bytes and re-reaches the same verdict.
    """
    parser = case.parser(tmp_path)
    path = case.write(tmp_path, case.closed_split)
    clean = parser.parse(path)
    path.write_bytes(path.read_bytes() + b"{not json}\n")

    result = parser.parse(path)

    assert [diagnostic.kind for diagnostic in result.diagnostics] == ["malformed_record"]
    assert result.normalization_checkpoint is None
    assert result.next_byte_offset == clean.next_byte_offset
    assert result.session.is_complete is False
