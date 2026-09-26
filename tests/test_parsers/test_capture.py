from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from recall.parsers.capture import JsonlCapture
from recall.parsers.claude_code import ClaudeCodeParser
from recall.parsers.codex import CodexParser
from recall.parsers.grok import GrokParser
from recall.parsers.kimi_code import KimiCodeParser
from recall.parsers.pi_agent import PiAgentParser


def test_capture_is_a_finite_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(b'{"n": 1}\n')
    with JsonlCapture(path) as capture:
        path.write_bytes(b'{"n": 1}\n{"n": 2}\n')
        assert list(capture.records()) == [{"n": 1}]
        assert capture.captured_size == len(b'{"n": 1}\n')


def test_capture_keeps_complete_boundary_and_reports_torn_tail(tmp_path: Path) -> None:
    content = b'{"n": 1}\n' + b'{"text":"\xc3'
    path = tmp_path / "session.jsonl"
    path.write_bytes(content)
    with JsonlCapture(path) as capture:
        assert list(capture.records()) == [{"n": 1}]
        assert capture.next_byte_offset == len(b'{"n": 1}\n')
        assert [d.kind for d in capture.diagnostics] == ["unterminated_tail"]
        assert capture.captured_prefix_sha256 == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("raw", [b"{bad}\n", b"\xff\n", b"[]\n"])
def test_capture_reports_malformed_complete_records(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(raw)
    with JsonlCapture(path) as capture:
        assert list(capture.records()) == []
        assert [d.kind for d in capture.diagnostics] == ["malformed_record"]


def test_capture_ignores_blank_lines_and_hashes_full_prefix_at_offset(tmp_path: Path) -> None:
    prefix = b"\n{" + b'"n": 1}' + b"\n\n"
    suffix = b'{"n": 2}\n'
    path = tmp_path / "session.jsonl"
    path.write_bytes(prefix + suffix)
    with JsonlCapture(path, offset=len(prefix)) as capture:
        assert list(capture.records()) == [{"n": 2}]
        assert capture.committed_prefix_sha256 == hashlib.sha256(prefix + suffix).hexdigest()


def test_capture_enforces_record_limit(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(b'{"long": 1}\n')
    with JsonlCapture(path, max_record_bytes=3) as capture:
        assert list(capture.records()) == []
        assert [d.kind for d in capture.diagnostics] == ["resource_limit"]


@pytest.mark.parametrize(
    ("parser_type", "complete_record"),
    [
        (ClaudeCodeParser, {"type": "system"}),
        # An empty object is an unsupported Codex semantic record.  Exercise
        # capture with supported metadata so this shared contract does not
        # accidentally require adapters to acknowledge unsupported input.
        (CodexParser, {"type": "session_meta", "payload": {}}),
        (GrokParser, {"type": "system"}),
        (KimiCodeParser, {"type": "metadata"}),
        (PiAgentParser, {"type": "session"}),
    ],
)
def test_all_parsers_expose_capture_contract(
    tmp_path: Path, parser_type: type, complete_record: dict[str, object]
) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(json.dumps(complete_record) + "\n{" + '"torn":')
    result = parser_type().parse(path)
    assert result.next_byte_offset == len(json.dumps(complete_record) + "\n")
    assert result.captured_size == path.stat().st_size
    assert result.captured_prefix_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert [d.kind for d in result.diagnostics] == ["unterminated_tail"]
    assert result.session.is_complete is False
    assert (
        result.committed_prefix_sha256
        == hashlib.sha256(path.read_bytes()[: result.next_byte_offset]).hexdigest()
    )


@pytest.mark.parametrize(
    "parser_type", [ClaudeCodeParser, CodexParser, GrokParser, KimiCodeParser, PiAgentParser]
)
def test_unknown_semantic_records_prevent_clean_coverage(tmp_path: Path, parser_type: type) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text('{"type":"future_conversation","content":"unrecognized"}\n')
    result = parser_type().parse(path)
    assert result.session.is_complete is False
    assert result.next_byte_offset == 0
    assert result.diagnostics[0].kind == "unsupported_record"
