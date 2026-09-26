"""REQ-PARSE-017: attributing a tool call to the skill it loaded.

Paths below are the real installed layouts: Claude Code and Codex plugin
caches, Codex's built-in `.system` dir, and Pi Agent's cloned plugin repos.
"""

from __future__ import annotations

import pytest
from recall.parsers.common import build_tool_call
from recall.parsers.skills import derive_skill_name

CODEX_CACHE = (
    "/Users/a/.codex/plugins/cache/agent-profile/engineering-practices/1.7.0"
    "/skills/code-law/SKILL.md"
)
CLAUDE_CACHE = "/Users/a/.claude/plugins/cache/acme-tools/tools/1.52.1/skills/review/SKILL.md"
CODEX_SYSTEM = "/Users/a/.codex/skills/.system/openai-docs/SKILL.md"
PI_PLUGIN = "/Users/a/.pi/agent/git/github.com/0xsend/recall/plugins/recall/skills/recall/SKILL.md"
PI_SINGLE_PLUGIN_REPO = "/Users/a/.pi/agent/git/github.com/acme-tools/tools/skills/review/SKILL.md"
PI_USER_SKILL = "/Users/a/.pi/agent/skills/zig-best-practices/SKILL.md"


@pytest.mark.parametrize(
    "path,expected",
    [
        # A plugin in the path namespaces the skill exactly as Claude Code's
        # typed Skill tool records it, so counts aggregate across harnesses.
        (CODEX_CACHE, "engineering-practices:code-law"),
        (CLAUDE_CACHE, "tools:review"),
        (PI_PLUGIN, "recall:recall"),
        # A single-plugin repo clone: the repo is the plugin, and naming it
        # keeps these rows aggregating with the same skill read from a cache.
        (PI_SINGLE_PLUGIN_REPO, "tools:review"),
        # No plugin to name: Claude records these bare too.
        (CODEX_SYSTEM, "openai-docs"),
        (PI_USER_SKILL, "zig-best-practices"),
    ],
)
def test_read_of_an_installed_skill_is_attributed(path: str, expected: str) -> None:
    assert derive_skill_name("bash", None, f"sed -n '1,240p' {path}") == expected


def test_read_tool_attributes_from_its_path_argument() -> None:
    """Pi loads a skill with its `read` tool, Claude Code with `Read`."""
    assert derive_skill_name("read", {"path": PI_PLUGIN}, None) == "recall:recall"
    assert derive_skill_name("Read", {"file_path": CLAUDE_CACHE}, None) == "tools:review"


@pytest.mark.parametrize(
    "command",
    [
        # Authoring, not loading: a working copy, under a git verb.
        "git diff -- skills/agent/SKILL.md skills/review/SKILL.md",
        "git status --short -- skills/weekly-reports/SKILL.md",
        # A working copy read is still not an installed-skill load.
        "nl -ba plugins/agent-workflows/skills/eval/SKILL.md",
        # Discovery over the cache, not a load.
        "find /Users/a/.codex/plugins/cache/acme-tools -name SKILL.md",
        "rg --files /Users/a/.codex/plugins/cache/acme-tools | rg SKILL",
        # Measuring is not reading.
        f"wc -l {CODEX_CACHE}",
    ],
)
def test_authoring_and_discovery_are_not_skill_loads(command: str) -> None:
    assert derive_skill_name("bash", None, command) is None


def test_only_the_segment_carrying_the_path_decides() -> None:
    """One command can hold an authoring act and a load; scope is the
    segment, so the leading verb of the whole command must not govern."""
    command = f"git diff --stat && sed -n '1,5p' {CODEX_CACHE}"
    assert derive_skill_name("bash", None, command) == "engineering-practices:code-law"


def test_a_batched_read_attributes_the_first_skill() -> None:
    command = f"sed -n '1,240p' {CODEX_CACHE} && sed -n '1,190p' {CLAUDE_CACHE}"
    assert derive_skill_name("bash", None, command) == "engineering-practices:code-law"


@pytest.mark.parametrize(
    "path",
    [
        # A sibling reference file is not the skill itself.
        "/Users/a/.codex/plugins/cache/mp/tools/1.0.0/skills/agent/references/loops.md",
        # The plugin manifest is not a skill load.
        "/Users/a/.codex/plugins/cache/mp/tools/1.0.0/.claude-plugin/plugin.json",
        # No `skills/` segment at all.
        "/Users/a/.codex/plugins/cache/mp/tools/1.0.0/SKILL.md",
    ],
)
def test_non_skill_files_under_an_installed_root_are_ignored(path: str) -> None:
    assert derive_skill_name("bash", None, f"cat {path}") is None


def test_build_tool_call_populates_skill_name_for_shell_loads() -> None:
    """The derivation must run at the shared chokepoint, so every source
    gets it — codex/pi report structural zeros without this."""
    call = build_tool_call("exec_command", {"cmd": f"sed -n '1,240p' {CODEX_CACHE}"})
    assert call.skill_name == "engineering-practices:code-law"
    assert call.bash_command == f"sed -n '1,240p' {CODEX_CACHE}"


def test_build_tool_call_populates_skill_name_for_read_tools() -> None:
    call = build_tool_call("read", {"path": PI_PLUGIN})
    assert call.skill_name == "recall:recall"


def test_build_tool_call_keeps_the_typed_skill_tool_authoritative() -> None:
    """Claude Code's typed Skill call already names the skill; deriving must
    not second-guess it."""
    call = build_tool_call("Skill", {"skill": "agent-workflows:afk", "args": "..."})
    assert call.skill_name == "agent-workflows:afk"


def test_build_tool_call_leaves_authoring_unattributed() -> None:
    command = "git diff -- skills/agent/SKILL.md"
    assert build_tool_call("bash", {"command": command}).skill_name is None


@pytest.mark.parametrize(
    "command",
    [
        # The path is inside a quoted string, so nothing read it.  Splitting
        # on an unquoted-looking `|` invents a `sed` segment that never ran.
        f"printf 'literal | sed -n 1p {CODEX_CACHE}'",
        f'echo "see | cat {CODEX_CACHE} for details"',
    ],
)
def test_quoted_text_is_not_an_executed_segment(command: str) -> None:
    assert derive_skill_name("bash", None, command) is None


def test_a_real_pipeline_still_attributes() -> None:
    command = f"cat {CODEX_CACHE} | head -40"
    assert derive_skill_name("bash", None, command) == "engineering-practices:code-law"


def test_newline_separated_commands_are_separate_segments() -> None:
    command = f"git diff --stat\nsed -n '1,5p' {CODEX_CACHE}"
    assert derive_skill_name("bash", None, command) == "engineering-practices:code-law"


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/project/.codex/skills/demo/SKILL.md",
        "/Users/a/code/myrepo/.claude/skills/demo/SKILL.md",
    ],
)
def test_a_repository_project_skill_read_is_a_load(path: str) -> None:
    assert derive_skill_name("read", {"path": path}, None) == "demo"


def test_a_plugin_cache_inside_a_repository_is_not_installed() -> None:
    path = "/Users/a/code/recall/.claude/plugins/cache/mp/p/1.0.0/skills/demo/SKILL.md"
    assert derive_skill_name("read", {"path": path}, None) is None


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/Users/a/.codex/skills/demo/SKILL.md", "demo"),
        ("/home/a/.codex/skills/demo/SKILL.md", "demo"),
        ("~/.codex/skills/demo/SKILL.md", "demo"),
        ("$HOME/.codex/skills/demo/SKILL.md", "demo"),
    ],
)
def test_home_rooted_paths_are_attributed(path: str, expected: str) -> None:
    assert derive_skill_name("read", {"path": path}, None) == expected


@pytest.mark.parametrize(
    "command",
    [
        f"sed -i 's/a/b/' {CODEX_CACHE}",
        f"sed -i.bak 's/a/b/' {CODEX_CACHE}",
        f"sed --in-place 's/a/b/' {CODEX_CACHE}",
    ],
)
def test_in_place_edits_are_authoring_not_loads(command: str) -> None:
    """`sed -i` writes the file. Counting that as a load attributes editing a
    skill to using it — the exact confusion the discriminator exists to stop."""
    assert derive_skill_name("bash", None, command) is None
