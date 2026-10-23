"""Cold Codex representations retain the plain rollout's identity and content."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from recall.parsers.codex import CodexParser
from recall.parsers.compressed import zstd


def rollout_bytes(text: str = "compressed question") -> bytes:
    return b"".join(
        json.dumps(record).encode() + b"\n"
        for record in [
            {"type": "session_meta", "payload": {"id": "compressed-session", "cwd": "/project"}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": text}},
        ]
    )


def test_compressed_rollout_discovery_and_plain_precedence(tmp_path: Path) -> None:
    plain = tmp_path / "rollout-test.jsonl"
    compressed = plain.with_suffix(".jsonl.zst")
    compressed.write_bytes(zstd.compress(rollout_bytes()))
    parser = CodexParser(roots=(tmp_path,))
    assert parser.discover() == [plain]
    compressed_result = parser.parse(compressed)
    plain.write_bytes(rollout_bytes())
    plain_result = parser.parse(plain)
    assert compressed_result.session.id == plain_result.session.id
    assert compressed_result.session.source_path == str(plain)
    assert compressed_result.session.messages == plain_result.session.messages
    assert compressed_result.next_byte_offset == compressed.stat().st_size
    assert compressed_result.normalization_checkpoint is None
    assert parser.discover() == [plain]
    compressed.write_bytes(zstd.compress(rollout_bytes("obsolete sibling")))
    assert parser.parse(compressed).session.messages[0].content == "compressed question"
    (tmp_path / "rollout-test.jsonl.zst.tmp").write_bytes(b"in flight")
    assert parser.discover() == [plain]


@pytest.mark.parametrize("failure", ["truncated", "invalid", "tail", "malformed", "limit"])
def test_compressed_capture_rejects_incomplete_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = tmp_path / "rollout-test.jsonl.zst"
    raw = rollout_bytes()
    encoded = zstd.compress(raw)
    if failure == "truncated":
        encoded = encoded[:-1]
    elif failure == "invalid":
        encoded += b"invalid frame"
    elif failure == "tail":
        encoded = zstd.compress(raw + b'{"type":')
    elif failure == "malformed":
        encoded = zstd.compress(raw + b"{malformed}\n")
    elif failure == "limit":
        monkeypatch.setattr("recall.parsers.compressed.MAX_COMPRESSED_CAPTURE_BYTES", len(raw) - 1)
    path.write_bytes(encoded)
    with pytest.raises((ValueError, zstd.ZstdError)):
        CodexParser().parse(path)


def test_streaming_frames_without_content_size_are_fully_captured(tmp_path: Path) -> None:
    path = tmp_path / "rollout-stream.jsonl.zst"
    compressor = zstd.ZstdCompressor()
    encoded = compressor.compress(rollout_bytes()) + compressor.flush()
    encoded += zstd.compress(
        b'{"type":"event_msg","payload":{"type":"agent_message","message":"answer"}}\n'
    )
    path.write_bytes(encoded)
    result = CodexParser().parse(path)
    assert [message.content for message in result.session.messages] == [
        "compressed question",
        "answer",
    ]
    assert result.next_byte_offset == len(encoded)
    assert not result.diagnostics


def test_plain_symlink_and_compressed_alias_have_one_identity(tmp_path: Path) -> None:
    target = tmp_path / "actual.jsonl"
    target.write_bytes(rollout_bytes("authoritative plain"))
    plain = tmp_path / "rollout-alias.jsonl"
    plain.symlink_to(target)
    compressed = plain.with_suffix(".jsonl.zst")
    compressed.write_bytes(zstd.compress(rollout_bytes("obsolete compressed")))
    alias = tmp_path / "rollout-sibling.jsonl.zst"
    alias.symlink_to(compressed)
    parser = CodexParser(roots=(tmp_path,))
    assert parser.discover() == [target]
    assert parser.parse(compressed).session == parser.parse(target).session
    assert parser.parse(alias).session == parser.parse(target).session
    target.unlink()
    plain.unlink()
    assert parser.discover() == [plain]
    assert parser.parse(alias).session.id == parser.parse(compressed).session.id


def test_dangling_plain_symlink_does_not_redirect_compressed_identity(tmp_path: Path) -> None:
    from recall.core.transcripts import transcript_file, transcript_identity

    plain = tmp_path / "rollout-dangling.jsonl"
    plain.symlink_to(tmp_path / "missing.jsonl")
    compressed = plain.with_suffix(".jsonl.zst")
    compressed.write_bytes(zstd.compress(rollout_bytes()))
    parser = CodexParser(roots=(tmp_path,))

    assert transcript_file(plain) == compressed
    assert transcript_identity(plain) == plain
    assert transcript_identity(compressed) == plain
    assert parser.discover() == [plain]
    result = parser.parse(plain)
    assert result.session.source_path == str(plain)
    assert result.session.messages[0].content == "compressed question"
