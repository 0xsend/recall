"""REQ-CTX-022 / REQ-CTX-024: the batched llm-codex `--output-schema` must be a
JSON *object* root.

`codex exec --output-schema <FILE>` relays the schema as the model's
`response_format`, which (per OpenAI structured outputs) requires the top-level
schema to be `type: "object"`. A top-level `type: "array"` is rejected
server-side with HTTP 400 `invalid_json_schema`, which turns *every* multi-chunk
batch into a `REQ-CTX-024` batch failure and silently degrades the whole
`llm-codex` backend to `template`. These tests pin the wire contract: the schema
root is an object wrapping a `contexts` array, and `_parse_batch_contexts` reads
that object shape back.
"""

from __future__ import annotations

import json

import pytest
from recall.services.context_backends.codex_cli import (
    _batch_output_schema,
    _parse_batch_contexts,
)


@pytest.mark.parametrize("expected_count", [1, 2, 8])
def test_batch_output_schema_root_is_object(expected_count: int) -> None:
    # codex --output-schema rejects a non-object root with invalid_json_schema;
    # the root MUST be an object or every batch 400s. This is the regression guard.
    schema = _batch_output_schema(expected_count)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["contexts"]


@pytest.mark.parametrize("expected_count", [1, 2, 8])
def test_batch_output_schema_wraps_sized_contexts_array(expected_count: int) -> None:
    # The array of per-chunk results moves under a `contexts` property; its size
    # bounds (REQ-CTX-024 count-match contract) and item shape are preserved.
    contexts = _batch_output_schema(expected_count)["properties"]["contexts"]
    assert contexts["type"] == "array"
    assert contexts["minItems"] == expected_count
    assert contexts["maxItems"] == expected_count
    item = contexts["items"]
    assert item["type"] == "object"
    assert item["additionalProperties"] is False
    assert sorted(item["required"]) == ["context", "index"]
    assert item["properties"]["index"]["maximum"] == expected_count - 1


def test_parse_batch_contexts_reads_object_payload() -> None:
    payload = json.dumps(
        {"contexts": [{"index": 0, "context": "first"}, {"index": 1, "context": "second"}]}
    )
    assert _parse_batch_contexts(payload, expected_count=2) == ["first", "second"]


def test_parse_batch_contexts_reorders_by_index() -> None:
    # Indices map results back to source chunks (REQ-CTX-024); order in the
    # array must not matter.
    payload = json.dumps(
        {"contexts": [{"index": 1, "context": "second"}, {"index": 0, "context": "first"}]}
    )
    assert _parse_batch_contexts(payload, expected_count=2) == ["first", "second"]


def test_parse_batch_contexts_tolerates_bare_array() -> None:
    # Defense in depth: a model that ignores the wrapper and returns a bare array
    # is still parsed rather than treated as a batch failure.
    payload = json.dumps([{"index": 0, "context": "only"}])
    assert _parse_batch_contexts(payload, expected_count=1) == ["only"]


def test_parse_batch_contexts_rejects_count_mismatch() -> None:
    payload = json.dumps({"contexts": [{"index": 0, "context": "only"}]})
    with pytest.raises(RuntimeError):
        _parse_batch_contexts(payload, expected_count=2)


def test_parse_batch_contexts_rejects_missing_contexts_key() -> None:
    payload = json.dumps({"results": [{"index": 0, "context": "only"}]})
    with pytest.raises(RuntimeError):
        _parse_batch_contexts(payload, expected_count=1)
