"""Native event shapes from the September 2026 unsupported-source census."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from recall.parsers.codex import CodexParser


def _parse(tmp_path: Path, records: list[dict]):
    path = tmp_path / "rollout.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return CodexParser().parse(path)


def _event(payload: dict) -> dict:
    return {"type": "event_msg", "payload": payload}


def test_legacy_review_events_preserve_target_and_readable_output(tmp_path: Path) -> None:
    result = _parse(
        tmp_path,
        [
            _event(
                {
                    "type": "entered_review_mode",
                    "target": {"type": "baseBranch", "branch": "main"},
                    "user_facing_hint": "Review changes",
                }
            ),
            _event({"type": "exited_review_mode", "review_output": None}),
            _event(
                {
                    "type": "exited_review_mode",
                    "review_output": {"findings": [], "overall_explanation": "Checked"},
                }
            ),
        ],
    )
    assert result.diagnostics == ()
    assert [message.role.value for message in result.session.messages] == ["system", "assistant"]
    assert json.loads(result.session.messages[0].content or "") == {
        "target": {"type": "baseBranch", "branch": "main"},
        "user_facing_hint": "Review changes",
    }
    assert json.loads(result.session.messages[1].content or "") == {
        "findings": [],
        "overall_explanation": "Checked",
    }


def test_image_view_retains_the_invocation_without_inventing_image_text(tmp_path: Path) -> None:
    result = _parse(
        tmp_path,
        [
            _event(
                {
                    "type": "item_completed",
                    "item": {
                        "type": "ImageView",
                        "id": "image-call",
                        "path": "/owned/screenshot.png",
                    },
                }
            )
        ],
    )
    assert result.diagnostics == ()
    assert result.session.messages == []
    assert [
        (call.tool_name, call.tool_input, call.tool_use_id)
        for call in result.session.orphan_tool_calls
    ] == [("view_image", {"path": "/owned/screenshot.png"}, "image-call")]


@pytest.mark.parametrize("canonical_first", [False, True])
def test_image_view_mirror_does_not_duplicate_a_canonical_call(
    tmp_path: Path, canonical_first: bool
) -> None:
    native = _event(
        {
            "type": "item_completed",
            "item": {"type": "ImageView", "id": "image-call", "path": "/owned/screenshot.png"},
        }
    )
    canonical = {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "call_id": "image-call",
            "name": "functions.view_image",
            "arguments": {"path": "/owned/screenshot.png"},
        },
    }
    result = _parse(tmp_path, [canonical, native] if canonical_first else [native, canonical])
    assert result.diagnostics == ()
    assert [(call.tool_name, call.tool_use_id) for call in result.session.orphan_tool_calls] == [
        ("functions.view_image", "image-call")
    ]


def test_legacy_web_completion_retains_action_and_results(tmp_path: Path) -> None:
    result = _parse(
        tmp_path,
        [
            _event(
                {
                    "type": "web_search_end",
                    "call_id": "web-call",
                    "query": "reference",
                    "action": {"type": "open", "url": "https://example.com/reference"},
                    "results": [
                        {"type": "web", "title": "Reference", "snippet": "Preserved excerpt"}
                    ],
                }
            )
        ],
    )
    assert result.diagnostics == ()
    calls = result.session.orphan_tool_calls
    assert len(calls) == 1
    assert calls[0].tool_name == "web_search"
    assert calls[0].tool_input == {"type": "open", "url": "https://example.com/reference"}
    assert len(result.tail_facts.tool_results) == 1
    output = result.tail_facts.tool_results[0]
    assert output.tool_use_id == "web-call"
    assert output.is_error is False
    assert "Preserved excerpt" in output.result_summary


@pytest.mark.parametrize(
    ("native_result", "is_error", "expected_text"),
    [
        (
            {"Ok": {"content": [{"type": "text", "text": "Found"}], "isError": False}},
            False,
            "Found",
        ),
        (
            {"Ok": {"content": [{"type": "text", "text": "Rejected"}], "isError": True}},
            True,
            "Rejected",
        ),
        ({"Err": "Unavailable"}, True, "Unavailable"),
    ],
)
def test_legacy_mcp_completion_retains_invocation_result_and_error(
    tmp_path: Path, native_result: dict, is_error: bool, expected_text: str
) -> None:
    result = _parse(
        tmp_path,
        [
            _event(
                {
                    "type": "mcp_tool_call_end",
                    "call_id": "mcp-call",
                    "invocation": {
                        "server": "example",
                        "tool": "lookup",
                        "arguments": {"key": "value"},
                    },
                    "result": native_result,
                }
            )
        ],
    )
    assert result.diagnostics == ()
    calls = result.session.orphan_tool_calls
    assert len(calls) == 1
    assert (calls[0].tool_name, calls[0].tool_input, calls[0].tool_use_id) == (
        "mcp__example__lookup",
        {"key": "value"},
        "mcp-call",
    )
    assert len(result.tail_facts.tool_results) == 1
    output = result.tail_facts.tool_results[0]
    assert output.tool_use_id == "mcp-call"
    assert output.is_error is is_error
    assert expected_text in output.result_summary


@pytest.mark.parametrize("canonical_first", [False, True])
def test_legacy_mcp_mirror_prefers_canonical_call_and_output(
    tmp_path: Path, canonical_first: bool
) -> None:
    native = _event(
        {
            "type": "mcp_tool_call_end",
            "call_id": "mcp-call",
            "invocation": {"server": "example", "tool": "lookup", "arguments": {"key": "value"}},
            "result": {"Ok": {"content": [{"type": "text", "text": "Mirror"}]}},
        }
    )
    canonical = [
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "mcp-call",
                "name": "canonical_lookup",
                "arguments": {"key": "value"},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "mcp-call",
                "output": "Canonical result",
            },
        },
    ]
    result = _parse(tmp_path, [*canonical, native] if canonical_first else [native, *canonical])
    assert result.diagnostics == ()
    assert [call.tool_name for call in result.session.orphan_tool_calls] == ["canonical_lookup"]
    assert [output.result_summary for output in result.tail_facts.tool_results] == [
        "Canonical result"
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "item_completed", "item": {"type": "ImageView", "id": "image-call"}},
        {"type": "web_search_end", "action": {"type": "search", "query": "topic"}},
        {
            "type": "mcp_tool_call_end",
            "call_id": "mcp-call",
            "invocation": None,
            "result": {"Ok": {}},
        },
        {
            "type": "mcp_tool_call_end",
            "call_id": "mcp-call",
            "invocation": {"server": "example", "tool": "lookup"},
            "result": {"Future": {}},
        },
    ],
)
def test_incomplete_native_tools_remain_diagnostic(tmp_path: Path, payload: dict) -> None:
    result = _parse(tmp_path, [_event(payload)])
    assert result.session.is_complete is False
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].kind == "unsupported_record"
    assert result.session.orphan_tool_calls == []
