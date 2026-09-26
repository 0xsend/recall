"""Checkpoint envelope encoding and fail-closed decoding (REQ-INDEX-025).

The stored envelope is the only thing standing between an append and a resume
that publishes rows a full parse would never produce. Anything that is not an
envelope this build wrote and understands must decode to "no proof", so the
caller takes the full parse path instead of handling an exception.

Whether a *well-formed* envelope may actually be trusted -- matching parser
revision, matching committed offset and digest -- is the resume decision, and
lives in ``tests/test_services/test_checkpoint_resume_fallbacks.py``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from typing import Any

import pytest
from recall.core.models import (
    NORMALIZATION_CHECKPOINT_BYTES_MAX,
    NORMALIZATION_CHECKPOINT_VERSION,
    NormalizationCheckpoint,
)

DIGEST = "a" * 64


def _checkpoint(**overrides: Any) -> NormalizationCheckpoint:
    fields: dict[str, Any] = {
        "parser_revision": "rev-a",
        "offset": 128,
        "prefix_sha256": DIGEST,
        "message_idx_base": 4,
        "orphan_tool_call_idx_base": 1,
        "source_dev": 42,
        "source_inode": 84,
        "adapter_state": {"turn": {"id": "t-1"}},
    }
    return NormalizationCheckpoint(**(fields | overrides))


def _envelope(**overrides: Any) -> str:
    decoded = json.loads(_checkpoint().encode())
    decoded.update(overrides)
    return json.dumps(decoded)


def _compressed_envelope(packed: bytes) -> str:
    return _envelope(adapter_state={"_recall_zlib_v1": base64.b64encode(packed).decode("ascii")})


def test_encoding_round_trips_through_storage() -> None:
    checkpoint = _checkpoint()
    encoded = checkpoint.encode()

    assert json.loads(encoded)["version"] == NORMALIZATION_CHECKPOINT_VERSION
    assert NormalizationCheckpoint.decode(encoded) == checkpoint


def test_an_empty_adapter_state_is_the_stateless_boundary() -> None:
    stateless = _checkpoint(adapter_state={})

    assert NormalizationCheckpoint.decode(stateless.encode()) == stateless


def test_compressible_state_stays_inside_the_durable_envelope() -> None:
    checkpoint = _checkpoint(adapter_state={"tail": "x" * NORMALIZATION_CHECKPOINT_BYTES_MAX})
    encoded = checkpoint.encode()

    assert len(encoded) <= NORMALIZATION_CHECKPOINT_BYTES_MAX
    assert NormalizationCheckpoint.decode(encoded) == checkpoint


def test_incompressible_oversized_state_is_refused_rather_than_stored() -> None:
    tail = "".join(hashlib.sha256(str(index).encode()).hexdigest() for index in range(4_000))
    oversized = _checkpoint(adapter_state={"tail": tail})

    with pytest.raises(ValueError):
        oversized.encode()


def _excessively_nested_state() -> dict[str, Any]:
    state: dict[str, Any] = {}
    cursor = state
    for _ in range(100):
        nested: dict[str, Any] = {}
        cursor["nested"] = nested
        cursor = nested
    return state


def test_excessively_nested_state_is_refused_before_storage() -> None:
    with pytest.raises(ValueError, match="nesting bound"):
        _checkpoint(adapter_state=_excessively_nested_state()).encode()


def test_excessively_nested_compressed_state_decodes_to_no_proof() -> None:
    packed = zlib.compress(json.dumps(_excessively_nested_state()).encode())
    encoded = _envelope(adapter_state={"_recall_zlib_v1": base64.b64encode(packed).decode("ascii")})

    assert NormalizationCheckpoint.decode(encoded) is None


def test_reserved_compression_key_is_refused_before_storage() -> None:
    with pytest.raises(ValueError, match="reserved key"):
        _checkpoint(adapter_state={"_recall_zlib_v1": "adapter-owned"}).encode()


def test_negative_file_identity_is_refused_before_storage() -> None:
    with pytest.raises(ValueError, match="file identity"):
        _checkpoint(source_dev=-1).encode()


def test_compressed_state_over_the_decompression_bound_decodes_to_no_proof() -> None:
    bomb = zlib.compress(b'{"tail":"' + b"x" * (4 * 1024 * 1024) + b'"}')

    assert NormalizationCheckpoint.decode(_compressed_envelope(bomb)) is None


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(None, id="legacy-row"),
        pytest.param("", id="empty"),
        pytest.param("{not json", id="not-json"),
        pytest.param('"rev-a"', id="not-an-object"),
        pytest.param("[]", id="json-array"),
        pytest.param("[" * 10_000 + "0" + "]" * 10_000, id="excessively-nested-json"),
        pytest.param(
            json.dumps({"parser_revision": "rev-a", "offset": 128}), id="truncated-envelope"
        ),
        pytest.param("x" * (NORMALIZATION_CHECKPOINT_BYTES_MAX + 1), id="oversized"),
        pytest.param(_envelope(version=NORMALIZATION_CHECKPOINT_VERSION + 1), id="unknown-version"),
        pytest.param(_envelope(version=True), id="boolean-version"),
        pytest.param(_envelope(offset=-1), id="negative-offset"),
        pytest.param(_envelope(offset="128"), id="string-offset"),
        pytest.param(_envelope(message_idx_base=-1), id="negative-message-base"),
        pytest.param(_envelope(orphan_tool_call_idx_base=None), id="null-orphan-base"),
        pytest.param(_envelope(source_dev=None), id="null-source-device"),
        pytest.param(_envelope(source_inode=-1), id="negative-source-inode"),
        pytest.param(_envelope(source_inode=True), id="boolean-source-inode"),
        pytest.param(_envelope(parser_revision=""), id="empty-revision"),
        pytest.param(_envelope(parser_revision=7), id="numeric-revision"),
        pytest.param(_envelope(prefix_sha256="A" * 64), id="uppercase-digest"),
        pytest.param(_envelope(prefix_sha256="a" * 63), id="short-digest"),
        pytest.param(_envelope(prefix_sha256=None), id="missing-digest"),
        pytest.param(_envelope(adapter_state=[]), id="adapter-state-not-an-object"),
        pytest.param(
            _envelope(adapter_state={"_recall_zlib_v1": "not base64"}),
            id="malformed-compressed-state",
        ),
        pytest.param(
            _compressed_envelope(zlib.compress(b"{}")[:-1]),
            id="truncated-compressed-state",
        ),
        pytest.param(
            _compressed_envelope(zlib.compress(b"{}") + zlib.compress(b"{}")),
            id="concatenated-compressed-streams",
        ),
        pytest.param(
            _compressed_envelope(zlib.compress(b"{}") + b"trailing-junk"),
            id="trailing-compressed-data",
        ),
    ],
)
def test_an_unusable_envelope_decodes_to_no_proof(raw: str | None) -> None:
    assert NormalizationCheckpoint.decode(raw) is None
