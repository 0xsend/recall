"""Plain and compressed rollouts share identity and normalization."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from recall.parsers.checkpoint import UnsupportedResumeState
from recall.parsers.codex import CodexParser
from recall.parsers.compressed import CompressedJsonlCapture, zstd


@pytest.mark.parametrize(
    "fixture",
    ["session1/rollout.jsonl", "retained_context/rollout.jsonl", "live_tool_pair/rollout.jsonl"],
)
def test_compressed_normalization_preserves_all_ids_and_record_positions(
    tmp_path: Path, fixture: str
) -> None:
    path = tmp_path / "rollout.jsonl"
    content = (Path(__file__).parents[1] / "fixtures/codex" / fixture).read_bytes()
    path.write_bytes(content)
    parser = CodexParser(roots=(tmp_path,))
    plain = parser.parse(path)
    physical = path.with_name(path.name + ".zst")
    physical.write_bytes(zstd.compress(content))
    path.unlink()
    packed = parser.parse(physical)
    assert packed.session.model_dump(
        exclude={"file_mtime", "file_size"}
    ) == plain.session.model_dump(exclude={"file_mtime", "file_size"})
    assert packed.next_byte_offset == physical.stat().st_size
    assert packed.captured_prefix_sha256 == hashlib.sha256(physical.read_bytes()).hexdigest()
    assert packed.committed_prefix_sha256 == packed.captured_prefix_sha256
    assert packed.normalization_checkpoint is None
    assert packed.diagnostics == ()
    assert parser.discover() == [path]
    with pytest.raises(UnsupportedResumeState):
        parser.parse(path, offset=1)


def test_discovery_prefers_plain_and_excludes_temporary_representations(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    content = b'{"type":"session_meta","payload":{}}\n'
    path.write_bytes(content)
    (tmp_path / "rollout.jsonl.zst").write_bytes(zstd.compress(content + b"{bad}\n"))
    for name in (
        "rollout.jsonl.zst.tmp",
        "rollout.jsonl.tmp",
        "other.jsonl.zst",
        "rollout.jsonl.zst.old",
    ):
        (tmp_path / name).write_bytes(b"bad")
    parser = CodexParser(roots=(tmp_path,))
    assert parser.discover() == [path]
    assert parser.parse(tmp_path / "rollout.jsonl.zst").diagnostics == ()


@pytest.mark.parametrize(
    "content", [b"{bad}\n", b"\xff\n", b"[]\n", b'{"type":"future"}\n', b'{"type":"session_meta"}']
)
def test_invalid_decoded_content_never_acknowledges_physical_bytes(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "rollout.jsonl.zst"
    path.write_bytes(zstd.compress(content))
    result = CodexParser().parse(path)
    assert result.diagnostics
    assert result.next_byte_offset == 0
    assert result.normalization_checkpoint is None


@pytest.mark.parametrize("damage", ["truncated", "corrupt", "trailing", "empty", "window"])
def test_invalid_frames_are_diagnostic_without_acknowledgment(tmp_path: Path, damage: str) -> None:
    content = zstd.compress(
        b'{"type":"session_meta","payload":{}}\n',
        options={zstd.CompressionParameter.checksum_flag: 1},
    )
    if damage == "truncated":
        content = content[:-1]
    elif damage == "corrupt":
        content = content[:-1] + bytes([content[-1] ^ 1])
    elif damage == "trailing":
        content += b"garbage"
    elif damage == "empty":
        content = b""
    else:
        # A valid empty frame declaring a 512 MiB decoder window.
        content = b"\x28\xb5\x2f\xfd\x00\x98\x01\x00\x00"
    path = tmp_path / "rollout.jsonl.zst"
    path.write_bytes(content)
    result = CodexParser().parse(path)
    assert result.diagnostics
    assert result.next_byte_offset == 0


def test_concatenated_and_unknown_size_frames_preserve_record_boundaries(tmp_path: Path) -> None:
    first = b'{"n":1}\n{"n":'
    encoder = zstd.ZstdCompressor()
    encoded = encoder.compress(first) + encoder.flush()
    path = tmp_path / "rollout.jsonl.zst"
    path.write_bytes(encoded + zstd.compress(b"2}\n"))
    with CompressedJsonlCapture(path) as capture:
        assert list(capture.records()) == [{"n": 1}, {"n": 2}]
        assert capture.diagnostics == []
        assert capture.next_byte_offset == path.stat().st_size


def test_decoded_expansion_limit_does_not_acknowledge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("recall.parsers.compressed.MAX_DECODED_BYTES", 64)
    path = tmp_path / "rollout.jsonl.zst"
    path.write_bytes(zstd.compress(b'{"n":1}\n' * 9))
    with CompressedJsonlCapture(path) as capture:
        list(capture.records())
        assert [d.kind for d in capture.diagnostics] == ["resource_limit"]
        assert capture.next_byte_offset == 0


def test_record_limit_does_not_acknowledge(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl.zst"
    path.write_bytes(zstd.compress(b'{"long":"' + b"x" * (64 * 1024 * 1024) + b'"}\n'))
    with CompressedJsonlCapture(path) as capture:
        assert list(capture.records()) == []
        assert [d.kind for d in capture.diagnostics] == ["resource_limit"]
        assert capture.next_byte_offset == 0


@pytest.mark.parametrize("target_name", ["capture.bin", "capture.jsonl"])
def test_compressed_alias_preserves_resolved_identity_across_representations(
    tmp_path: Path, target_name: str
) -> None:
    target = tmp_path / target_name
    content = b'{"type":"event_msg","payload":{"type":"user_message","message":"alias history"}}\n'
    target.write_bytes(content)
    alias = tmp_path / "rollout.jsonl"
    alias.symlink_to(target)
    parser = CodexParser(roots=(tmp_path,))
    plain = parser.parse(alias)
    alias.unlink()
    target.write_bytes(zstd.compress(content))
    compressed = tmp_path / "rollout.jsonl.zst"
    compressed.symlink_to(target)
    assert parser.discover() == [target]
    decoded = parser.parse(target)
    assert decoded.session.id == plain.session.id
    assert decoded.session.messages == plain.session.messages
    assert decoded.normalization_checkpoint is None and not decoded.diagnostics
    target.write_bytes(
        content + b'{"type":"event_msg","payload":{"type":"user_message","message":"append"}}\n'
    )
    compressed.unlink()
    alias.symlink_to(target)
    extended = parser.parse(target)
    assert extended.session.id == plain.session.id
    assert extended.session.messages[:1] == plain.session.messages
    assert [m.content for m in extended.session.messages] == ["alias history", "append"]
