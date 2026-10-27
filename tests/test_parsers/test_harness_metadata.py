"""Current harness metadata leaves the normalized conversation unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from recall.parsers.claude_code import ClaudeCodeParser
from recall.parsers.codex import CodexParser
from recall.parsers.kimi_code import KimiCodeParser
from recall.parsers.protocol import SessionParser

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "harness_metadata"
CASES = [
    pytest.param(CodexParser(), "codex/rollout-additional-tools.jsonl", id="additional-tools"),
    pytest.param(
        CodexParser(), "codex/rollout-configuration-update.jsonl", id="configuration-update"
    ),
    pytest.param(
        CodexParser(), "codex/rollout-replacement-history.jsonl", id="replacement-history"
    ),
    pytest.param(
        KimiCodeParser(),
        "kimi_code/change-notifier/agents/main/wire.jsonl",
        id="change-notifier",
    ),
    pytest.param(
        KimiCodeParser(), "kimi_code/plan-revision/agents/main/wire.jsonl", id="plan-revision"
    ),
    pytest.param(ClaudeCodeParser(), "claude_code/session-tools-used.jsonl", id="tools-used"),
]


@pytest.mark.parametrize(("parser", "fixture"), CASES)
def test_harness_metadata_preserves_conversation(
    tmp_path: Path, parser: SessionParser, fixture: str
) -> None:
    path = tmp_path / Path(fixture).name
    path.write_bytes((FIXTURES / fixture).read_bytes())
    result = parser.parse(path)
    assert result.diagnostics == ()
    assert result.session.is_complete
    assert result.next_byte_offset == path.stat().st_size
    assert result.normalization_checkpoint is not None
    expected = ["Parser orchard question", "Parser orchard answer"]
    if "replacement-history" in fixture:
        expected.insert(1, "Orchard compacted summary")
    assert [message.content for message in result.session.messages] == expected
    assert result.session.tool_count == 0
    assert result.session.input_tokens is None
    assert result.session.output_tokens is None
    if parser.source == "claude_code":
        assert result.session.source_session_id == "tools-used"


@pytest.mark.parametrize(("parser", "fixture"), CASES)
def test_harness_metadata_split_matches_full_parse(
    tmp_path: Path, parser: SessionParser, fixture: str
) -> None:
    lines = (FIXTURES / fixture).read_bytes().splitlines(keepends=True)
    path = tmp_path / Path(fixture).name
    path.write_bytes(b"".join(lines))
    full = parser.parse(path)
    assert full.diagnostics == ()
    for split in range(1, len(lines)):
        path.write_bytes(b"".join(lines[:split]))
        prefix = parser.parse(path)
        checkpoint = prefix.normalization_checkpoint
        assert checkpoint is not None
        path.write_bytes(b"".join(lines))
        suffix = parser.parse(
            path,
            offset=checkpoint.offset,
            message_idx_base=checkpoint.message_idx_base,
            orphan_tool_call_idx_base=checkpoint.orphan_tool_call_idx_base,
            resume_state=checkpoint.adapter_state,
        )
        if "replacement-history" in fixture and split < 3:
            # A replacement snapshot needs the full history's occurrence counts.
            assert [d.kind for d in suffix.diagnostics] == ["needs_full_parse"]
        else:
            assert suffix.diagnostics == ()
            assert prefix.session.messages + suffix.session.messages == full.session.messages
            assert (
                prefix.tail_facts.tool_results + suffix.tail_facts.tool_results
                == full.tail_facts.tool_results
            )
            assert (
                prefix.tail_facts.stop_markers + suffix.tail_facts.stop_markers
                == full.tail_facts.stop_markers
            )
            assert suffix.normalization_checkpoint == full.normalization_checkpoint


@pytest.mark.parametrize(
    ("parser", "record"),
    [
        (CodexParser(), {"type": "response_item", "payload": {"type": "additional_tools_future"}}),
        (
            CodexParser(),
            {"type": "response_item", "payload": {"type": "configuration_update_future"}},
        ),
        (KimiCodeParser(), {"type": "changeNotifier.future"}),
        (KimiCodeParser(), {"type": "plan.future"}),
        (ClaudeCodeParser(), {"type": "tools-used-future"}),
    ],
)
def test_unknown_harness_metadata_stays_diagnostic(
    tmp_path: Path, parser: SessionParser, record: dict[str, object]
) -> None:
    path = tmp_path / "unknown.jsonl"
    path.write_text(json.dumps(record) + "\n")
    result = parser.parse(path)
    assert [d.kind for d in result.diagnostics] == ["unsupported_record"]
    assert result.normalization_checkpoint is None
    assert result.next_byte_offset == path.stat().st_size
    assert not result.session.messages


@pytest.mark.parametrize(
    "item",
    [
        {"type": "future_item", "content": "unrecognized prose"},
        {"type": "function_call", "name": "sample", "arguments": "{}", "call_id": "one"},
        {"type": "other", "content": "unrecognized prose"},
        None,
    ],
)
def test_compaction_unknown_items_remain_diagnostic(tmp_path: Path, item: object) -> None:
    path = tmp_path / "rollout.jsonl"
    record = {
        "type": "compacted",
        "payload": {"replacement_history": [item], "message": "Readable summary"},
    }
    path.write_text(json.dumps(record) + "\n")
    result = CodexParser().parse(path)
    assert [d.kind for d in result.diagnostics] == ["unsupported_record"]
    assert result.normalization_checkpoint is None
    assert [m.content for m in result.session.messages] == ["Readable summary"]
