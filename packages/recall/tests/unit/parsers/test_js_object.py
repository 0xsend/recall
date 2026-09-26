"""REQ-PARSE-016: the JS object-literal reader behind Codex ``exec`` wrappers."""

from __future__ import annotations

import json

import pytest
from recall.parsers.js_object import (
    UNRESOLVED,
    JsLiteralError,
    collect_string_consts,
    iter_tool_calls,
    parse_js_object_literal,
    prune_unresolved,
)


def test_reads_strict_json() -> None:
    value, end = parse_js_object_literal('{"cmd": "ls", "n": 10}')
    assert value == {"cmd": "ls", "n": 10}
    assert end == len('{"cmd": "ls", "n": 10}')


def test_reads_unquoted_keys_and_single_quotes() -> None:
    value, _ = parse_js_object_literal("{cmd: 'echo hi', workdir: \"/tmp\"}")
    assert value == {"cmd": "echo hi", "workdir": "/tmp"}


def test_reads_trailing_comma_and_nested_containers() -> None:
    value, _ = parse_js_object_literal('{plan: [{step: "a"}, {step: "b"}], done: true,}')
    assert value == {"plan": [{"step": "a"}, {"step": "b"}], "done": True}


def test_stops_at_the_matching_brace() -> None:
    source = '{cmd: "ls"});\ntext(r.output);'
    value, end = parse_js_object_literal(source)
    assert value == {"cmd": "ls"}
    assert source[end] == ")"


def test_decodes_string_escapes() -> None:
    value, _ = parse_js_object_literal(r'{chars: "y\n", quote: "a\"b", uni: "é"}')
    assert value == {"chars": "y\n", "quote": 'a"b', "uni": "é"}


def test_shorthand_resolves_against_consts() -> None:
    value, _ = parse_js_object_literal("{cmd, workdir: '/tmp'}", consts={"cmd": "git status"})
    assert value == {"cmd": "git status", "workdir": "/tmp"}


def test_unresolvable_values_are_marked_not_guessed() -> None:
    value, _ = parse_js_object_literal("{cmd, workdir: cwd}")
    assert value == {"cmd": UNRESOLVED, "workdir": UNRESOLVED}


def test_interpolated_template_is_unresolvable() -> None:
    value, _ = parse_js_object_literal("{cmd: `sed -n '1,240p' '${path}'`}")
    assert value == {"cmd": UNRESOLVED}


def test_plain_template_literal_reads_as_a_string() -> None:
    value, _ = parse_js_object_literal("{cmd: `git status`}")
    assert value == {"cmd": "git status"}


def test_comments_are_skipped() -> None:
    value, _ = parse_js_object_literal('{/* why */ cmd: "ls", // trailing\n n: 1}')
    assert value == {"cmd": "ls", "n": 1}


@pytest.mark.parametrize("source", ["[1, 2]", '"just a string"', "{cmd: ", "{"])
def test_non_object_and_truncated_literals_raise(source: str) -> None:
    with pytest.raises(JsLiteralError):
        parse_js_object_literal(source)


def test_collect_string_consts_reads_declared_strings() -> None:
    source = "const cmd = \"rg TODO\";\nconst dir = '/tmp';\nconst n = 3;\n"
    assert collect_string_consts(source) == {"cmd": "rg TODO", "dir": "/tmp"}


def test_collect_string_consts_keeps_the_first_binding() -> None:
    source = 'const cmd = "first";\nconst cmd = "second";\n'
    assert collect_string_consts(source) == {"cmd": "first"}


def test_collect_string_consts_skips_interpolated_templates() -> None:
    source = "const cmd = `sed -n '1,${n}p' file`;\n"
    assert collect_string_consts(source) == {}


def test_prune_unresolved_drops_nested_sentinels() -> None:
    """REQ-PARSE-016: an UNRESOLVED left anywhere in the decoded value is a
    crash at index time — `json.dumps` of a tool_input cannot serialize it —
    so pruning must reach nested containers, not just top-level keys."""
    value = {
        "cmd": "x",
        "meta": {"cwd": UNRESOLVED, "keep": 1},
        "items": [UNRESOLVED, "a"],
    }
    cleaned, dropped = prune_unresolved(value)
    assert cleaned == {"cmd": "x", "meta": {"keep": 1}, "items": [None, "a"]}
    assert dropped is True
    json.dumps(cleaned)


def test_prune_unresolved_reports_clean_values_untouched() -> None:
    value = {"cmd": "x", "meta": {"cwd": "/tmp"}}
    cleaned, dropped = prune_unresolved(value)
    assert cleaned == value
    assert dropped is False


def test_iter_tool_calls_ignores_calls_inside_string_literals() -> None:
    """REQ-PARSE-016: a patch body or echoed snippet can contain the text of
    a tool call.  Counting it invents a call that never ran."""
    source = (
        'const patch = "*** Begin Patch\\n+ tools.exec_command({cmd: \\"nope\\"})";\n'
        "const r = await tools.apply_patch(patch);"
    )
    assert [name for name, _ in iter_tool_calls(source)] == ["apply_patch"]


def test_iter_tool_calls_ignores_calls_inside_comments() -> None:
    source = (
        "// tools.exec_command({cmd: 'nope'})\n/* tools.wait({}) */\nawait tools.update_plan({});"
    )
    assert [name for name, _ in iter_tool_calls(source)] == ["update_plan"]


def test_iter_tool_calls_reports_each_batched_call_in_order() -> None:
    source = 'await Promise.all([tools.exec_command({cmd:"a"}), tools.write_stdin({chars:"b"})]);'
    assert [name for name, _ in iter_tool_calls(source)] == ["exec_command", "write_stdin"]


def test_iter_tool_calls_offsets_point_past_the_open_paren() -> None:
    source = 'await tools.exec_command({cmd:"a"});'
    ((_, offset),) = iter_tool_calls(source)
    assert source[offset] == "{"


def test_iter_tool_calls_yields_call_sites_not_executions() -> None:
    """REQ-PARSE-016: control flow is not evaluated, and the contract says so.

    A guarded call is still a call site, so it is yielded even when the guard
    never runs; a call in a loop is yielded once however many times it ran.
    Resolving either needs a JS engine, and the wrapper's output payload is
    one combined blob for the whole program, so nothing in the transcript can
    reconcile it. The under-count is the common shape in the real corpus:
    over 5,106 August wrappers, 11% contained a loop and none contained a
    literal dead branch.
    """
    guarded = 'if (false) { await tools.exec_command({cmd: "never-ran"}); }'
    assert [name for name, _ in iter_tool_calls(guarded)] == ["exec_command"]

    looped = 'for (const cmd of ["a", "b", "c"]) { await tools.exec_command({cmd}); }'
    assert [name for name, _ in iter_tool_calls(looped)] == ["exec_command"]


def test_collect_string_consts_ignores_commented_declarations() -> None:
    """REQ-PARSE-016: a commented-out declaration never bound anything, and
    taking it as the first binding persists a command that never ran."""
    source = '// const cmd = "fake"\nconst cmd = "real";'
    assert collect_string_consts(source) == {"cmd": "real"}


def test_collect_string_consts_ignores_declarations_inside_strings() -> None:
    source = 'const note = "const cmd = \\"fake\\"";\nconst cmd = "real";'
    assert collect_string_consts(source)["cmd"] == "real"


def test_iter_tool_calls_ignores_regex_literals() -> None:
    """REQ-PARSE-016: a regex literal is code, but its body is not — a call
    spelled inside one is a pattern, not an invocation."""
    source = 'const pattern = /tools.exec_command(foo)/; await tools.notify({value: "ok"});'
    assert [name for name, _ in iter_tool_calls(source)] == ["notify"]


def test_iter_tool_calls_still_sees_calls_after_a_division() -> None:
    """A `/` is not always a regex; treating division as one would swallow
    the rest of the program."""
    source = 'const half = total / 2; await tools.exec_command({cmd: "ls"});'
    assert [name for name, _ in iter_tool_calls(source)] == ["exec_command"]
