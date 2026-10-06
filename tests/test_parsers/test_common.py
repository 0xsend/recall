"""Unit tests for parsers/common.py shared helpers."""

from __future__ import annotations

import pytest
from recall.parsers.common import accumulate_metric, build_tool_call, first_int, first_of
from recall.parsers.skills import derive_skill_name, derive_skill_names, is_skill_candidate

# -- first_of --


@pytest.mark.parametrize(
    "entry,paths,expected",
    [
        # Flat key lookup
        ({"model": "gpt-4"}, ("model",), "gpt-4"),
        # Nested tuple key lookup
        ({"message": {"model": "opus"}}, (("message", "model"),), "opus"),
        # First match wins — nested before flat
        (
            {"message": {"model": "opus"}, "model": "gpt-4"},
            (("message", "model"), "model"),
            "opus",
        ),
        # First match wins — flat before nested
        (
            {"gitBranch": "main", "git": {"branch": "dev"}},
            ("gitBranch", ("git", "branch")),
            "main",
        ),
        # Missing key returns None
        ({"other": "value"}, ("model", "model_name"), None),
        # Empty string is skipped (treated as absent)
        ({"model": "", "model_name": "fallback"}, ("model", "model_name"), "fallback"),
        # All empty strings returns None
        ({"model": ""}, ("model",), None),
        # Non-string value is skipped
        ({"model": 42}, ("model",), None),
        # Deeply nested path
        ({"a": {"b": {"c": "deep"}}}, (("a", "b", "c"),), "deep"),
        # Nested path with missing intermediate key
        ({"a": {"x": 1}}, (("a", "b", "c"),), None),
    ],
    ids=[
        "flat_key",
        "nested_key",
        "nested_wins_over_flat",
        "flat_wins_over_nested",
        "missing_key",
        "empty_string_skipped",
        "all_empty_none",
        "non_string_skipped",
        "deep_nested",
        "missing_intermediate",
    ],
)
def test_first_of(entry: dict, paths: tuple, expected: str | None) -> None:
    assert first_of(entry, *paths) == expected


# -- first_int --


@pytest.mark.parametrize(
    "entry,paths,expected",
    [
        # Flat key coerced to int
        ({"count": 42}, ("count",), 42),
        # String numeric coerced
        ({"count": "7"}, ("count",), 7),
        # Nested key
        ({"usage": {"tokens": 100}}, (("usage", "tokens"),), 100),
        # Non-numeric returns None
        ({"count": "abc"}, ("count",), None),
        # Missing key returns None
        ({}, ("count",), None),
        # First match wins
        ({"a": 1, "b": 2}, ("a", "b"), 1),
        # Skips non-coercible, finds next
        ({"a": "bad", "b": 5}, ("a", "b"), 5),
    ],
    ids=[
        "int_value",
        "string_coerced",
        "nested_key",
        "non_numeric_none",
        "missing_none",
        "first_wins",
        "skip_bad_find_next",
    ],
)
def test_first_int(entry: dict, paths: tuple, expected: int | None) -> None:
    assert first_int(entry, *paths) == expected


# -- accumulate_metric --


@pytest.mark.parametrize(
    "existing,value,expected",
    [
        (None, None, None),
        (None, 5, 5),
        (10, None, 10),
        (10, 5, 15),
        (0, 0, 0),
        (None, "3", 3),
        (10, "bad", 10),
    ],
    ids=[
        "none_plus_none",
        "none_plus_value",
        "value_plus_none",
        "sum",
        "zero_plus_zero",
        "string_coerced",
        "bad_string_ignored",
    ],
)
def test_accumulate_metric(existing: int | None, value, expected: int | None) -> None:
    assert accumulate_metric(existing, value) == expected


def test_string_tool_input_is_preserved() -> None:
    """REQ-PARSE-018: a tool whose payload is a bare string keeps its text.

    Codex sends `apply_patch` as a custom_tool_call whose `input` is the raw
    patch, not JSON.  Dropping it because it is not a dict left 38,122 rows
    recording only that a call happened.
    """
    patch = "*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n*** End Patch"
    call = build_tool_call("apply_patch", patch)
    assert call.tool_input == {"input": patch}


def test_string_tool_input_does_not_disturb_bash_extraction() -> None:
    """A bash tool given a bare string still parses as a command, not a blob."""
    call = build_tool_call("shell", "git status --short")
    assert call.bash_command == "git status --short"
    assert call.bash_base == "git"


def test_non_string_non_dict_tool_input_stays_empty() -> None:
    for value in (None, 42, ["a"]):
        assert build_tool_call("apply_patch", value).tool_input is None


def test_grok_runtime_source_skill_is_attributed_outside_working_repo() -> None:
    skill_path = "/opt/agent-profile/plugins/engineering-practices/skills/code-law/SKILL.md"

    assert (
        derive_skill_name(
            "read",
            {"path": skill_path},
            None,
            cwd="/work/product",
        )
        == "engineering-practices:code-law"
    )


def test_read_shaped_text_inside_heredoc_is_not_attributed() -> None:
    command = "cat <<'EOF' > /tmp/note\ncat /Users/dev/.codex/skills/demo/SKILL.md\nEOF"

    assert (
        derive_skill_name(
            "bash",
            {"command": command},
            command,
            cwd="/work/product",
        )
        is None
    )


def test_read_shaped_text_inside_line_continued_heredoc_is_not_attributed() -> None:
    command = "cat <<'EOF' \\\n> /tmp/note\ncat /Users/dev/.codex/skills/demo/SKILL.md\nEOF"

    assert (
        derive_skill_name(
            "bash",
            {"command": command},
            command,
            cwd="/work/product",
        )
        is None
    )


def test_read_shaped_text_inside_multiline_quote_is_not_attributed() -> None:
    command = "printf 'literal\ncat /Users/dev/.codex/skills/demo/SKILL.md\n'"

    assert (
        derive_skill_name(
            "bash",
            {"command": command},
            command,
            cwd="/work/product",
        )
        is None
    )


@pytest.mark.parametrize(
    "command",
    [
        ("cat <<'EOF' > /tmp/note\n' literal\nEOF\ncat /Users/dev/.codex/skills/demo/SKILL.md"),
        ("cat <<'EOF' > /tmp/note\nliteral \\\nEOF\ncat /Users/dev/.codex/skills/demo/SKILL.md"),
        (
            "# unmatched apostrophe is literal in a comment: '\n"
            "cat /Users/dev/.codex/skills/demo/SKILL.md"
        ),
    ],
    ids=["quoted-body", "continued-body", "quoted-comment"],
)
def test_real_skill_load_after_literal_shell_text_is_attributed(command: str) -> None:
    assert (
        derive_skill_name(
            "bash",
            {"command": command},
            command,
            cwd="/work/product",
        )
        == "demo"
    )


@pytest.mark.parametrize(
    "command",
    [
        "rg SKILL.md /opt/agent-profile/plugins/engineering-practices/skills",
        "git diff -- /opt/agent-profile/plugins/engineering-practices/skills/code-law/SKILL.md",
        "sed -i '' /opt/agent-profile/plugins/engineering-practices/skills/code-law/SKILL.md",
        (
            "printf 'sed -n 1,20p /opt/agent-profile/plugins/engineering-practices/"
            "skills/code-law/SKILL.md'"
        ),
    ],
)
def test_grok_discovery_editing_and_quoted_pseudo_calls_are_not_attributed(
    command: str,
) -> None:
    assert (
        derive_skill_name(
            "bash",
            {"command": command},
            command,
            cwd="/work/product",
        )
        is None
    )


def test_brace_pattern_read_loads_each_expanded_skill() -> None:
    root = "/Users/dev/.codex/plugins/cache/agent-profile/engineering-practices/4.1.1/skills"
    command = f"cat {root}/{{code-law,gh}}/SKILL.md"

    assert derive_skill_names("exec_command", {"cmd": command}, command) == (
        "engineering-practices:code-law",
        "engineering-practices:gh",
    )


def test_one_read_of_several_skill_files_loads_each_once() -> None:
    root = "/Users/dev/.claude/skills"
    command = f"cat {root}/a/SKILL.md {root}/b/SKILL.md && sed -n 1,9p {root}/a/SKILL.md"

    assert derive_skill_names("Bash", {"command": command}, command) == ("a", "b")


def test_agent_profile_source_skill_is_attributed_for_every_harness() -> None:
    command = "cat /opt/agent-profile/plugins/engineering-practices/skills/code-law/SKILL.md"

    assert (
        derive_skill_name("exec_command", None, command, cwd="/work/product")
        == "engineering-practices:code-law"
    )


def test_agents_skills_root_groups_skills_by_plugin() -> None:
    command = "cat /Users/dev/.agents/skills/engineering-practices/zig-best-practices/SKILL.md"

    assert derive_skill_name("exec_command", None, command) == (
        "engineering-practices:zig-best-practices"
    )


def test_linked_checkout_skill_is_attributed() -> None:
    command = "sed -n 1,200p /Users/dev/code/dotfiles/codex/.codex/skills/demo/SKILL.md"

    assert derive_skill_name("exec_command", None, command, cwd="/work/product") == "demo"


def test_project_skill_read_inside_its_repository_is_a_load() -> None:
    command = "sed -n 1,200p .agents/skills/design-system/SKILL.md"

    assert derive_skill_name("exec_command", None, command, cwd="/work/app") == "design-system"
    assert derive_skill_name("exec_command", None, command, cwd=None) is None


def test_code_mode_program_is_not_analyzed_for_skill_loads() -> None:
    read = "cat /Users/dev/.codex/skills/foo/SKILL.md"
    program = (
        f'const cmds = ["{read}"];\nfor (const c of cmds) await tools.exec_command({{cmd: c}});'
    )

    assert derive_skill_names("exec_command", {"source": program}, None) == ()
    assert not is_skill_candidate("exec_command", {"source": program}, None)
    assert derive_skill_names("exec_command", {"cmd": read}, read) == ("foo",)


def test_an_unexpanded_shell_word_names_no_skill() -> None:
    loop = "cat -n plugins/agent-workflows/skills/$f/SKILL.md"
    glob = "cat /Users/dev/.claude/plugins/cache/acme/acme/*/skills/*/SKILL.md"

    assert derive_skill_names("Bash", None, loop, cwd="/Users/dev/dotfiles/agent-profile") == ()
    assert derive_skill_names("Bash", None, glob) == ()
