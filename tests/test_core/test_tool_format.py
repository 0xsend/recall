"""Tests for harness-agnostic tool-call detail formatting."""

from __future__ import annotations

import pytest
from recall.core.tool_format import TOOL_DETAIL_MAX_CHARS, format_tool_call_line, format_tool_detail


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "bash_command", "expected"),
    [
        ("run_command", {"CommandLine": "pytest -q"}, "pytest -q", "pytest -q"),
        ("Bash", {"command": "git status\n--short"}, "git status\n--short", "git status --short"),
        ("read_file", {"Path": "/repo/README.md"}, None, "/repo/README.md"),
        ("Read", {"file_path": "/repo/app.py", "offset": 10}, None, "/repo/app.py"),
        (
            "Edit",
            {"file_path": "/repo/a.py", "old_string": "a", "new_string": "b"},
            None,
            "/repo/a.py",
        ),
        ("search_web", {"query": "strong inference"}, None, "strong inference"),
        ("Grep", {"pattern": "TODO", "path": "src/"}, None, "TODO in src/"),
        ("WebFetch", {"url": "https://example.com/docs"}, None, "https://example.com/docs"),
        (
            "invoke_subagent",
            {"TypeName": "research", "Role": "Codebase Researcher", "Prompt": "inspect"},
            None,
            "research: Codebase Researcher",
        ),
        (
            "Agent",
            {"subagent_type": "Explore", "description": "Find routes"},
            None,
            "Explore: Find routes",
        ),
        ("Skill", {"skill": "code-law"}, None, "code-law"),
        (
            "apply_patch",
            {
                "input": (
                    "*** Begin Patch\n"
                    "*** Update File: src/a.py\n"
                    "@@ -1 +1 @@\n-old\n+new\n"
                    "*** Add File: src/b.py\n+hi\n"
                    "*** End Patch"
                )
            },
            None,
            "src/a.py, src/b.py",
        ),
        (
            "mcp__slack__post",
            {"channel": "C123", "thread_ts": "1.2"},
            None,
            "channel=C123 thread_ts=1.2",
        ),
        ("noop", {}, None, ""),
        ("noop", None, None, ""),
    ],
)
def test_format_tool_detail_covers_harness_tool_shapes(
    tool_name: str,
    tool_input: dict[str, object] | None,
    bash_command: str | None,
    expected: str,
) -> None:
    assert format_tool_detail(tool_name, tool_input, bash_command) == expected


def test_format_tool_detail_bounds_long_payloads() -> None:
    long_path = "/repo/" + ("a" * 300)
    rendered = format_tool_detail("read_file", {"Path": long_path})
    assert len(rendered) == TOOL_DETAIL_MAX_CHARS
    assert rendered.endswith("...")


def test_format_tool_call_line_omits_trailing_space_when_empty() -> None:
    assert format_tool_call_line({"tool_name": "read_file", "tool_input": {}}) == "[read_file]"
    assert (
        format_tool_call_line({"tool_name": "read_file", "tool_input": {"Path": "/tmp/a.txt"}})
        == "[read_file] /tmp/a.txt"
    )
