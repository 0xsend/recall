"""Unit tests for parsers/common.py shared helpers."""

from __future__ import annotations

import pytest
from recall.parsers.common import accumulate_metric, build_tool_call, first_int, first_of
from recall.parsers.skills import derive_skill_name, derive_skill_names

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


def test_code_mode_program_shell_literal_loads_the_skill() -> None:
    root = "/Users/dev/.codex/plugins/cache/agent-profile/engineering-practices/6.1.0/skills"
    program = (
        f'const cmds = [\n"cat {root}/code-law/SKILL.md",\n"git status"];\n'
        f'const paths = ["{root}/gh/SKILL.md"];\n'
        "for (const c of cmds) await exec(c);"
    )

    assert derive_skill_names("exec_command", {"source": program}, None) == (
        "engineering-practices:code-law",
    )


def test_an_unexpanded_shell_word_names_no_skill() -> None:
    loop = "cat -n plugins/agent-workflows/skills/$f/SKILL.md"
    glob = "cat /Users/dev/.claude/plugins/cache/acme/acme/*/skills/*/SKILL.md"

    assert derive_skill_names("Bash", None, loop, cwd="/Users/dev/dotfiles/agent-profile") == ()
    assert derive_skill_names("Bash", None, glob) == ()


def test_code_mode_command_data_counts_unless_an_inner_call_ran_it() -> None:
    root = "/Users/dev/.codex/skills"
    jobs = (
        f'const jobs = [{{cmd: "cat {root}/foo/SKILL.md"}}];\n'
        "for (const job of jobs) await tools.exec_command({cmd: job.cmd});\n"
    )
    direct = (
        f'const READ = "cat {root}/bar/SKILL.md";\n'
        "await tools.exec_command({cmd: READ});\n"
        "await tools.exec_command({cmd: `ls ${dir}`});\n"
    )

    assert derive_skill_names("exec_command", {"source": jobs}, None) == ("foo",)
    assert derive_skill_names("exec_command", {"source": direct}, None) == ()


_CODEX_PLUGIN_SKILLS = (
    "/Users/dev/.codex/plugins/cache/agent-profile/engineering-practices/3.0.0/skills"
)
_CAT_P = "await tools.exec_command({cmd: `cat ${p}`});"


def test_code_mode_loop_over_a_path_array_loads_each_skill() -> None:
    root = _CODEX_PLUGIN_SKILLS
    for_of = (
        f'const paths = [\n  "{root}/code-law/SKILL.md",\n  "{root}/gh/SKILL.md"\n];\n'
        "for (const p of paths) {\n"
        "  const r = await tools.exec_command({cmd: `sed -n '1,260p' '${p}'`, "
        'workdir: "/work/app"});\n'
        "  text(`FILE ${p}\\n${r.output}`);\n"
        "}\n"
    )
    mapped = (
        f'const paths=["{root}/code-law/SKILL.md","{root}/gh/SKILL.md"];\n'
        "const rs=await Promise.all(paths.map(p=>tools.exec_command("
        '{cmd:`wc -l "${p}" && sed -n \'1,1000p\' "${p}"`})));\n'
        "for(let i=0;i<rs.length;i++){text(`--- ${paths[i]}\\n${rs[i].output}`)}\n"
    )
    destructured = (
        f'const files = [\n  ["law", "{root}/code-law/SKILL.md"],\n'
        f'  ["gh", "{root}/gh/SKILL.md"]\n];\n'
        "const rs = await Promise.all(files.map(async ([name,path]) => [name, "
        "await tools.exec_command({cmd:`sed -n '1,320p' ${path}`})]));\n"
    )
    stringified = (
        f'const paths = ["{root}/code-law/SKILL.md", "{root}/gh/SKILL.md"];\n'
        "const results = await Promise.all(paths.map((path) => tools.exec_command({\n"
        "  cmd: `wc -l ${JSON.stringify(path)}; sed -n '1,260p' ${JSON.stringify(path)}`,\n"
        "})));\n"
    )
    expected = ("engineering-practices:code-law", "engineering-practices:gh")

    assert derive_skill_names("exec_command", {"source": stringified}, None) == expected
    assert derive_skill_names("exec_command", {"source": for_of}, None) == expected
    assert derive_skill_names("exec_command", {"source": mapped}, None) == expected
    assert derive_skill_names("exec_command", {"source": destructured}, None) == expected


def test_code_mode_template_over_a_path_const_loads_the_skill() -> None:
    root = _CODEX_PLUGIN_SKILLS
    whole_path = (
        f'const path = "{root}/code-law/SKILL.md";\n'
        "const r = await tools.exec_command({\n"
        "  cmd: `wc -l '${path}' && sed -n '1,700p' '${path}'`,\n"
        "});\ntext(r.output);\n"
    )
    base = (
        f'const base="{root}";\n'
        "const r = await tools.exec_command({cmd:`sed -n '1,260p' '${base}/gh/SKILL.md'\n"
        'rg -n "REQ" SPEC.md`}); text(r.output);\n'
    )

    assert derive_skill_names("exec_command", {"source": whole_path}, None) == (
        "engineering-practices:code-law",
    )
    assert derive_skill_names("exec_command", {"source": base}, None) == (
        "engineering-practices:gh",
    )


@pytest.mark.parametrize(
    "body",
    [
        # Measuring or testing for a file is not reading it.
        "for (const p of paths) await tools.exec_command("
        "{cmd: `if [ -f '${p}' ]; then wc -l '${p}'; fi`});",
        # The loop variable is rebound, so which array it walks is not static.
        f"for (const p of paths) {_CAT_P}\nfor (const p of others) {_CAT_P}",
        # The array changes before the loop runs.
        f'paths.push("/tmp/x");\nfor (const p of paths) {_CAT_P}',
        # Only the runtime knows what an expression produces.
        f"for (const p of paths.slice(1)) {_CAT_P}",
        "await tools.exec_command({cmd: `cat ${paths.map(p => `'${p}'`).join(' ')}`});",
        "for (let i = 0; i < paths.length; i++) "
        "await tools.exec_command({cmd: `cat ${paths[i]}`});",
        # Outside its loop the variable is not bound to the array.
        f"for (const p of paths) text(p);\n{_CAT_P}",
    ],
)
def test_code_mode_template_that_is_not_statically_resolvable_loads_nothing(body: str) -> None:
    root = _CODEX_PLUGIN_SKILLS
    program = f'const paths = ["{root}/code-law/SKILL.md", "{root}/gh/SKILL.md"];\n' + body

    assert derive_skill_names("exec_command", {"source": program}, None) == ()


_ALPHA = f"{_CODEX_PLUGIN_SKILLS}/alpha/SKILL.md"
_BETA = f"{_CODEX_PLUGIN_SKILLS}/beta/SKILL.md"
_READ_EACH = "for (const p of paths) await tools.exec_command({cmd: `cat ${p}`});"


@pytest.mark.parametrize(
    "program",
    [
        # An initializer the runtime keeps transforming is not the literal.
        f'const paths = ["{_ALPHA}", "{_BETA}"].filter(x => x.includes("alpha"));\n{_READ_EACH}',
        f'const paths = ["{_ALPHA}", "{_BETA}"].slice(0, 1);\n{_READ_EACH}',
        f'const p = "{_ALPHA}".replace("alpha", "beta");\n'
        "await tools.exec_command({cmd: `cat ${p}`});",
        # The array changes, directly or through another name, before the loop.
        f'const paths = ["{_ALPHA}", "{_BETA}"];\npaths.pop();\n{_READ_EACH}',
        f'const paths = ["{_ALPHA}", "{_BETA}"];\npaths.shift();\n{_READ_EACH}',
        f'const paths = ["{_ALPHA}", "{_BETA}"];\npaths[1] += "x";\n{_READ_EACH}',
        f'const paths = ["{_ALPHA}", "{_BETA}"];\nconst q = paths;\nq.length = 0;\n{_READ_EACH}',
        f'const paths = ["{_ALPHA}", "{_BETA}"];\nconst q = paths;\nq.splice(1);\n{_READ_EACH}',
        f'const paths = ["{_ALPHA}", "{_BETA}"];\ntrim(paths);\n{_READ_EACH}',
        # The loop variable is reassigned inside the body.
        f'const paths = ["{_ALPHA}"];\n'
        "for (let p of paths) { p = p.replace('alpha', 'beta'); "
        "await tools.exec_command({cmd: `cat ${p}`}); }",
    ],
)
def test_code_mode_template_over_a_changed_value_loads_nothing(program: str) -> None:
    assert derive_skill_names("exec_command", {"source": program}, None) == ()


@pytest.mark.parametrize(
    "reader",
    [
        "const io = { async read(p) { return tools.exec_command({cmd: `cat ${p}`}); } };",
        "class Io { async read(p) { return tools.exec_command({cmd: `cat ${p}`}); } }",
        "async function rd(p, n = Number(1)) { return tools.exec_command({cmd: `cat ${p}`}); }",
        "const rd = async (p, n = Number(1)) => tools.exec_command({cmd: `cat ${p}`});",
    ],
)
def test_code_mode_parameter_shadowing_a_path_const_loads_nothing(reader: str) -> None:
    program = f'const p = "{_ALPHA}";\n{reader}\nawait rd("{_BETA}");'

    assert derive_skill_names("exec_command", {"source": program}, None) == ()


@pytest.mark.parametrize(
    "use",
    [
        "// await tools.exec_command({cmd: `cat ${p}`});",
        "/* await tools.exec_command({cmd: `cat ${p}`}); */",
        "console.log(`cat ${p}`);",
        "await tools.write_file({path: 'notes.md', content: `cat ${p}`});",
    ],
)
def test_code_mode_template_that_is_not_an_exec_command_loads_nothing(use: str) -> None:
    program = f'const p = "{_ALPHA}";\n{use}'

    assert derive_skill_names("exec_command", {"source": program}, None) == ()


def test_code_mode_program_without_a_skill_file_is_never_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recall.parsers import skills

    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("resolved a program that names no SKILL.md")

    monkeypatch.setattr(skills, "StaticBindings", refuse)
    program = 'const xs = ["a"];\nfor (const x of xs) await tools.exec_command({cmd: `echo ${x}`});'

    assert derive_skill_names("exec_command", {"source": program}, None) == ()


_ADVERSARIAL_UNITS = (
    "const a{i} = [];\n",
    "for (const x of xs) {{\n",
    "xs.map(x => (\n",
    "const a{i} = [1];\n"
    "for (const x{i} of a{i}) await tools.exec_command({{cmd: `echo ${{x{i}}}`}});\n",
)


@pytest.mark.parametrize("unit", _ADVERSARIAL_UNITS)
@pytest.mark.parametrize("chars_max", [16_384, None])
def test_code_mode_resolution_stays_bounded_on_adversarial_programs(
    unit: str, chars_max: int | None
) -> None:
    import time

    # A quadratic pass over 8,000 units takes tens of seconds; a bounded one
    # takes milliseconds, whether the program fits the resolver or not.
    tail = f'await tools.exec_command({{cmd: `cat ${{p}}`}});\nconst p = "{_ALPHA}";\n'
    body = "".join(unit.format(i=i) for i in range(8000))
    if chars_max is not None:
        body = body[: chars_max - len(tail)]
    program = 'const xs = ["a"];\n' + body + tail
    wrapper = {"source": program}

    started = time.perf_counter()
    for _ in range(4):  # once per inner call that carries the program
        derive_skill_names("exec_command", wrapper, None)

    assert time.perf_counter() - started < 2.0


_PATHS = f'const paths = ["{_ALPHA}", "{_BETA}"];\n'
_CAT_EACH = "for (const p of paths) await tools.exec_command({cmd: `cat ${p}`});"


@pytest.mark.parametrize(
    "program",
    [
        f"{_PATHS}delete paths[1];\n{_CAT_EACH}",
        f"{_PATHS}paths.forEach((x, i, arr) => arr.pop());\n{_CAT_EACH}",
        f"{_PATHS}paths.map(function (x, i, arr) {{ arr.length = 1; }});\n{_CAT_EACH}",
        f'const paths = [["{_ALPHA}"], ["{_BETA}"]];\npaths[1][0] = "/tmp/x";\n'
        "for (const [p] of paths) await tools.exec_command({cmd: `cat ${p}`});",
        f'const paths = [["{_ALPHA}"], ["{_BETA}"]];\npaths[1][0] += "x";\n'
        "for (const [p] of paths) await tools.exec_command({cmd: `cat ${p}`});",
        f'const paths = [["{_ALPHA}"], ["{_BETA}"]];\npaths[1].pop();\n'
        "for (const [p] of paths) await tools.exec_command({cmd: `cat ${p}`});",
    ],
)
def test_code_mode_array_changed_through_an_element_or_callback_loads_nothing(
    program: str,
) -> None:
    assert derive_skill_names("exec_command", {"source": program}, None) == ()


def test_code_mode_loop_assigning_a_declared_name_does_not_read_the_const() -> None:
    padding = " " * 40
    program = (
        f'const p = "{_ALPHA}";\n'
        f'function g() {{ let{padding}p; for (p of ["{_BETA}"]) '
        "tools.exec_command({cmd: `cat ${p}`}); }\n"
    )

    assert "engineering-practices:alpha" not in derive_skill_names(
        "exec_command", {"source": program}, None
    )


@pytest.mark.parametrize(
    "use",
    [
        "const job = {cmd: `cat ${p}`};\ntext(JSON.stringify(job));",
        "await tools.write_stdin({session_id: 1, chars: JSON.stringify({cmd: `cat ${p}`})});",
        "await tools.write_stdin({command: `cat ${p}`});",
    ],
)
def test_code_mode_cmd_object_not_passed_to_exec_loads_nothing(use: str) -> None:
    program = f'const p = "{_ALPHA}";\n{use}'

    assert derive_skill_names("exec_command", {"source": program}, None) == ()


def test_code_mode_subscript_assignment_does_not_rebind_the_loop_variable() -> None:
    program = (
        f"{_PATHS}const out = {{}};\n"
        "for (const p of paths) {\n"
        "  const r = await tools.exec_command({cmd: `cat ${p}`});\n"
        "  out[p] = r.output;\n"
        "}\n"
    )

    assert derive_skill_names("exec_command", {"source": program}, None) == (
        "engineering-practices:alpha",
        "engineering-practices:beta",
    )


def test_code_mode_reading_an_element_in_an_expression_keeps_the_array_known() -> None:
    program = (
        f"{_PATHS}const results = await Promise.all(paths.map(path => "
        "tools.exec_command({cmd:`sed -n '1,260p' '${path}'`})));\n"
        'results.forEach((r,i)=>{text(paths[i]+"\\n"+r.output)});\n'
    )

    assert derive_skill_names("exec_command", {"source": program}, None) == (
        "engineering-practices:alpha",
        "engineering-practices:beta",
    )


_ROWS = f'const paths = [["{_ALPHA}"], ["{_BETA}"]];\n'
_CAT_EACH_ROW = "for (const [p] of paths) await tools.exec_command({cmd: `cat ${p}`});"


@pytest.mark.parametrize(
    "change",
    [
        "paths.forEach(r => r.pop());",
        'paths.forEach(r => { r[0] = "/tmp/x"; });',
        'for (const r of paths) r[0] = "/tmp/x";',
        'function m(row) { row[0] = "/tmp/x"; }\nm(paths[1]);',
        'const row = paths[1];\nrow[0] = "/tmp/x";',
    ],
)
def test_code_mode_rows_changed_through_an_alias_load_nothing(change: str) -> None:
    program = f"{_ROWS}{change}\n{_CAT_EACH_ROW}"

    assert derive_skill_names("exec_command", {"source": program}, None) == ()


@pytest.mark.parametrize(
    "call",
    [
        "await tools.exec_command({cmd: 'true', command: `cat ${p}`});",
        "await tools.exec_command({cmd: `cat ${p}`, cmd: 'true'});",
        "await tools.exec_command({cmd: `cat ${p}`, ...override});",
    ],
)
def test_code_mode_template_that_is_not_the_final_cmd_loads_nothing(call: str) -> None:
    program = f'const p = "{_ALPHA}";\n{call}'

    assert derive_skill_names("exec_command", {"source": program}, None) == ()


def test_code_mode_measuring_an_array_of_rows_keeps_it_known() -> None:
    program = (
        f'const refs = [["alpha", "{_ALPHA}"], ["beta", "{_BETA}"]];\n'
        "const res = await Promise.all(refs.map(([n,p])=>tools.exec_command({\n"
        "  cmd:`sed -n '1,320p' '${p}'`,\n})));\n"
        "for(let i=0;i<refs.length;i++) text(`===== ${refs[i][0]} =====\\n${res[i].output}`);\n"
    )

    assert derive_skill_names("exec_command", {"source": program}, None) == (
        "engineering-practices:alpha",
        "engineering-practices:beta",
    )
