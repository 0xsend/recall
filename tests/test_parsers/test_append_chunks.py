"""Resumed captures compose at adapter-owned record boundaries (REQ-INDEX-028)."""

from pathlib import Path

import pytest
from recall.parsers.checkpoint import ChunkTarget

from tests.test_parsers.test_normalization_checkpoint_adapters import (
    CASES,
    AdapterCase,
    _message_facts,
    _orphan_facts,
    _stop_markers,
    _tool_results,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("target", [ChunkTarget(1, 8 * 1024 * 1024), ChunkTarget(99999, 1)])
def test_req_index_028_chunks_equal_reference(
    case: AdapterCase, target: ChunkTarget, tmp_path: Path
) -> None:
    parser = case.parser(tmp_path)
    path = case.write(tmp_path, 1)
    prefix = parser.parse(path)
    checkpoint = prefix.normalization_checkpoint
    assert checkpoint is not None
    case.write(tmp_path, len(case.records))
    reference = parser.parse(path)
    parts = [prefix]
    for _ in range(len(case.records)):
        part = parser.parse(
            path,
            offset=checkpoint.offset,
            message_idx_base=checkpoint.message_idx_base,
            orphan_tool_call_idx_base=checkpoint.orphan_tool_call_idx_base,
            resume_state=checkpoint.adapter_state,
            target=target,
        )
        assert not part.diagnostics
        assert part.next_byte_offset > checkpoint.offset
        parts.append(part)
        if part.reached_end:
            break
        checkpoint = part.normalization_checkpoint
        assert checkpoint is not None
        assert part.captured_size == part.next_byte_offset
        # A real EOF at this byte boundary must finalize identically.
        whole = path.read_bytes()
        path.write_bytes(whole[: part.next_byte_offset])
        ended_here = parser.parse(path)
        path.write_bytes(whole)
        assert part.normalization_checkpoint == ended_here.normalization_checkpoint
    else:
        pytest.fail("chunks did not finish")
    assert [fact for item in parts for fact in _message_facts(item)] == _message_facts(reference)
    assert [fact for item in parts for fact in _orphan_facts(item)] == _orphan_facts(reference)
    assert _tool_results(*parts) == _tool_results(reference)
    assert _stop_markers(*parts) == _stop_markers(reference)
    assert parts[-1].normalization_checkpoint == reference.normalization_checkpoint


def test_req_index_028_codex_charges_deferred_completions_at_acceptance(tmp_path: Path) -> None:
    import json

    from recall.parsers.codex import CodexParser

    path = tmp_path / "rollout.jsonl"
    path.write_text('{"type":"session_meta","payload":{"id":"deferred"}}\n')
    parser = CodexParser(roots=(tmp_path,))
    prefix = parser.parse(path)
    checkpoint = prefix.normalization_checkpoint
    assert checkpoint is not None
    with path.open("a") as stream:
        for idx in range(10000):
            stream.write(
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "item_completed",
                            "item": {
                                "type": "CommandExecution",
                                "id": f"exec-{idx}",
                                "command": "ls",
                                "aggregated_output": "ok",
                                "exit_code": 0,
                            },
                        },
                    }
                )
                + "\n"
            )
    reference = parser.parse(path)
    parts = []
    for _ in range(1001):
        part = parser.parse(
            path,
            offset=checkpoint.offset,
            message_idx_base=checkpoint.message_idx_base,
            orphan_tool_call_idx_base=checkpoint.orphan_tool_call_idx_base,
            resume_state=checkpoint.adapter_state,
            target=ChunkTarget(31, 8 * 1024 * 1024),
        )
        assert len(part.session.orphan_tool_calls) <= 16
        assert len(part.tail_facts.tool_results) <= 16
        assert part.normalization_checkpoint is not None
        parts.append(part)
        checkpoint = part.normalization_checkpoint
        if part.reached_end:
            break
    else:
        pytest.fail("deferred chunks did not finish")
    assert len(parts) > 1
    assert [fact for item in parts for fact in _orphan_facts(item)] == _orphan_facts(reference)
    assert _tool_results(*parts) == _tool_results(reference)
    assert checkpoint == reference.normalization_checkpoint


def test_req_index_028_one_record_keeps_all_4096_tools(tmp_path: Path) -> None:
    import json

    from recall.parsers.claude_code import ClaudeCodeParser

    parser = ClaudeCodeParser(roots=(tmp_path,))
    path = tmp_path / "session.jsonl"
    path.write_text('{"type":"user","message":{"role":"user","content":"start"}}\n')
    prefix = parser.parse(path)
    checkpoint = prefix.normalization_checkpoint
    assert checkpoint is not None
    record = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": f"call-{idx}",
                    "name": "Bash",
                    "input": {"command": "ls"},
                }
                for idx in range(4096)
            ],
        },
    }
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.write('{"type":"user","message":{"role":"user","content":"after"}}\n')
    chunk = parser.parse(
        path,
        offset=checkpoint.offset,
        message_idx_base=checkpoint.message_idx_base,
        resume_state=checkpoint.adapter_state,
        target=ChunkTarget(2, 1),
    )
    assert not chunk.reached_end
    assert len(chunk.session.messages) == 1
    assert len(chunk.session.messages[0].tool_calls) == 4096
    assert chunk.normalization_checkpoint is not None


def test_req_index_028_codex_envelope_decline_extends_to_eof(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from tests.test_parsers.test_normalization_checkpoint_adapters import (
        CODEX_SESSION_META,
        _codex_append,
        _codex_file,
        _codex_unpaired_mirrors,
    )

    parser, path = _codex_file(tmp_path, (CODEX_SESSION_META,))
    prefix = parser.parse(path)
    checkpoint = prefix.normalization_checkpoint
    assert checkpoint is not None
    _codex_append(path, (CODEX_SESSION_META, *_codex_unpaired_mirrors(3010)))
    with caplog.at_level("WARNING", logger="recall.parsers.checkpoint"):
        part = parser.parse(
            path,
            offset=checkpoint.offset,
            message_idx_base=checkpoint.message_idx_base,
            resume_state=checkpoint.adapter_state,
            target=ChunkTarget(3000, 8 * 1024 * 1024),
        )
    reference = parser.parse(path)
    assert part.reached_end
    assert part.normalization_checkpoint is None
    assert any("normalization checkpoint declined" in record.message for record in caplog.records)
    assert _message_facts(prefix) + _message_facts(part) == _message_facts(reference)


def test_req_index_028_codex_abort_after_chunk_marks_the_prior_message(tmp_path: Path) -> None:
    import json

    from recall.parsers.codex import CodexParser

    path = tmp_path / "rollout.jsonl"
    path.write_text('{"type":"session_meta","payload":{"id":"abort-boundary"}}\n')
    parser = CodexParser()
    prefix = parser.parse(path)
    checkpoint = prefix.normalization_checkpoint
    assert checkpoint is not None
    with path.open("a") as stream:
        for payload in (
            {"type": "agent_message", "message": "interrupted answer"},
            {"type": "turn_aborted", "reason": "user_interrupt"},
        ):
            stream.write(json.dumps({"type": "event_msg", "payload": payload}) + "\n")
    first = parser.parse(
        path,
        offset=checkpoint.offset,
        resume_state=checkpoint.adapter_state,
        target=ChunkTarget(1, 8 * 1024 * 1024),
    )
    assert not first.reached_end
    checkpoint = first.normalization_checkpoint
    assert checkpoint is not None
    last = parser.parse(
        path,
        offset=checkpoint.offset,
        message_idx_base=checkpoint.message_idx_base,
        resume_state=checkpoint.adapter_state,
        target=ChunkTarget(1, 8 * 1024 * 1024),
    )
    reference = parser.parse(path)
    assert last.reached_end
    assert [
        (marker.idx, marker.reason, marker.ends_turn) for marker in last.tail_facts.stop_markers
    ] == [(0, "user_interrupt", True)]
    assert _stop_markers(prefix, first, last) == _stop_markers(reference)
