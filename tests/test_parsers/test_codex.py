from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recall.parsers.codex import CodexParser

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "codex"


def _write_and_parse(tmp_path: Path, entries: list[dict[str, Any]]):
    """Write a minimal rollout carrying `entries` and parse it."""
    meta = {
        "type": "session_meta",
        "payload": {
            "id": "codex-inline",
            "timestamp": "2026-07-10T09:00:00Z",
            "cwd": "/repo",
            "git": {"branch": "main", "root": "/repo"},
        },
    }
    path = tmp_path / "rollout-2026-07-10T09-00-00-inline.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in [meta, *entries]) + "\n")
    return CodexParser().parse(path).session


def test_codex_parser_parses_orphans() -> None:
    fixture_dir = FIXTURES / "session1"
    fixture = fixture_dir / "rollout.jsonl"
    parser = CodexParser()
    session = parser.parse(fixture).session

    assert session.source_session_id == "codex123"
    assert session.cwd == "/repo"
    assert session.git_branch == "main"
    assert session.message_count == 3

    assert len(session.orphan_tool_calls) == 1
    orphan = session.orphan_tool_calls[0]
    assert orphan.tool_name == "shell"
    assert orphan.bash_command == "ls -la"

    tool_calls = session.messages[-1].tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0].bash_command == "pwd"


def test_codex_parser_response_item_tool_calls() -> None:
    """Parse response_item entries: function_call, custom_tool_call, web_search_call."""
    fixture_dir = FIXTURES / "session2"
    fixture = next(fixture_dir.glob("rollout-*.jsonl"))
    parser = CodexParser()
    session = parser.parse(fixture).session

    assert session.source_session_id == "codex456"
    assert session.cwd == "/project"
    assert session.git_branch == "feature"
    assert session.message_count == 2

    # 2 exec_command + 1 apply_patch + 1 web_search = 4 orphan tool calls
    assert len(session.orphan_tool_calls) == 4
    assert session.tool_count == 4

    exec1 = session.orphan_tool_calls[0]
    assert exec1.tool_name == "exec_command"
    assert exec1.bash_command == "npm test"

    patch = session.orphan_tool_calls[1]
    assert patch.tool_name == "apply_patch"
    assert patch.bash_command is None

    web = session.orphan_tool_calls[2]
    assert web.tool_name == "web_search"
    assert web.tool_input == {"type": "open_page", "url": "https://docs.example.com"}

    exec2 = session.orphan_tool_calls[3]
    assert exec2.tool_name == "exec_command"
    assert exec2.bash_command == "git status"


def test_codex_parser_indexes_current_response_messages(tmp_path: Path) -> None:
    """Current rollout message records are conversation messages, not ignored metadata."""
    session = _write_and_parse(
        tmp_path,
        [
            {
                "type": "response_item",
                "timestamp": "2026-09-08T12:00:00Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "owned question"}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-09-08T12:01:00Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "owned answer"}],
                },
            },
        ],
    )

    assert [(message.role.value, message.content) for message in session.messages] == [
        ("user", "owned question"),
        ("assistant", "owned answer"),
    ]


def test_codex_parser_normalizes_mirrors_reasoning_and_compaction(tmp_path: Path) -> None:
    """Current rollout mirrors do not duplicate turns or poison completeness."""
    session = _write_and_parse(
        tmp_path,
        [
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "id": "turn-1", "message": "same turn"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "turn-1",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "same turn"}],
                },
            },
            {"type": "response_item", "payload": {"type": "reasoning", "summary": []}},
            {
                "type": "response_item",
                "payload": {"type": "compaction", "summary": "compact prior context"},
            },
        ],
    )

    assert [(message.role.value, message.content) for message in session.messages] == [
        ("user", "same turn"),
        ("system", "compact prior context"),
    ]
    assert session.is_complete is True


def test_codex_parser_response_item_shell_command_alias() -> None:
    fixture_dir = FIXTURES / "session3"
    fixture = next(fixture_dir.glob("rollout-*.jsonl"))
    parser = CodexParser()
    session = parser.parse(fixture).session

    assert len(session.orphan_tool_calls) == 1
    shell_call = session.orphan_tool_calls[0]
    assert shell_call.tool_name == "shell_command"
    assert shell_call.bash_command == "echo hello"


def test_codex_parser_extracts_token_usage() -> None:
    fixture_dir = FIXTURES / "session4"
    fixture = next(fixture_dir.glob("rollout-*.jsonl"))
    parser = CodexParser()
    session = parser.parse(fixture).session

    assert session.input_tokens == 150
    assert session.output_tokens == 15


def test_codex_parser_no_token_events_leaves_none() -> None:
    fixture = FIXTURES / "session1" / "rollout.jsonl"
    parser = CodexParser()
    session = parser.parse(fixture).session

    assert session.input_tokens is None
    assert session.output_tokens is None


def test_codex_parser_git_repo_falls_back_to_cwd(tmp_path: Path) -> None:
    """Real codex session_meta git payloads carry branch/commit_hash/
    repository_url but no local root key; git_repo must fall back to cwd
    (mirroring the Claude parser) instead of indexing every codex session
    with git_repo=NULL."""
    import json

    fixture = tmp_path / "rollout-2026-07-03T00-00-00-abc.jsonl"
    meta = {
        "type": "session_meta",
        "payload": {
            "id": "codex789",
            "timestamp": "2026-07-03T00:00:00Z",
            "cwd": "/work/myrepo",
            "git": {
                "branch": "main",
                "commit_hash": "3490b705502a203f5b30c236b1bd9884202531bb",
                "repository_url": "git@github.com:example/myrepo.git",
            },
        },
    }
    fixture.write_text(json.dumps(meta) + "\n")

    session = CodexParser().parse(fixture).session

    assert session.git_branch == "main"
    assert session.git_repo == "/work/myrepo"


def test_codex_parser_exec_wrapper_fans_out_inner_tool_calls() -> None:
    """REQ-PARSE-016: codex >=2026-07-09 wraps tool calls in a JS ``exec``
    program.  Each ``tools.<name>({...})`` inside it is a distinct tool call
    and must be recorded under its own name with its own arguments, or the
    command text is lost entirely."""
    fixture_dir = FIXTURES / "session5"
    fixture = next(fixture_dir.glob("rollout-*.jsonl"))
    session = CodexParser().parse(fixture).session

    calls = session.orphan_tool_calls
    # 1 exec_command + 2 batched exec_command + 1 write_stdin + 1 apply_patch
    # + 1 unparseable wrapper
    assert [call.tool_name for call in calls][:6] == [
        "exec_command",
        "exec_command",
        "exec_command",
        "write_stdin",
        "apply_patch",
        "exec",
    ]

    assert calls[0].bash_command == "npm test"
    assert calls[0].tool_input == {
        "cmd": "npm test",
        "workdir": "/project",
        "yield_time_ms": 10000,
    }

    # Promise.all batches stay one row per inner call, in source order.
    assert calls[1].bash_command == "git status"
    assert calls[2].bash_command == "git diff --stat"
    assert calls[2].bash_base == "git"
    assert calls[2].bash_sub == "diff"

    assert calls[3].tool_input == {"session_id": 7, "chars": "y\n"}
    assert calls[3].bash_command is None


def test_codex_parser_exec_wrapper_recovers_a_const_patch_argument() -> None:
    """REQ-PARSE-018: ``tools.apply_patch(patch)`` passes a JS variable.  When
    the declaration is a plain string const the reader already collected, the
    row carries the patch itself rather than the surrounding program, so it
    matches the direct custom_tool_call shape and both forms aggregate."""
    fixture = next((FIXTURES / "session5").glob("rollout-*.jsonl"))
    session = CodexParser().parse(fixture).session

    patch_call = session.orphan_tool_calls[4]
    assert patch_call.tool_name == "apply_patch"
    assert patch_call.tool_input is not None
    assert "source" not in patch_call.tool_input
    assert patch_call.tool_input["input"].startswith("*** Begin Patch")


def test_codex_parser_exec_wrapper_without_tool_calls_keeps_source() -> None:
    """REQ-PARSE-016: a wrapper that invokes no tool still keeps its source
    under the ``exec`` name; dropping it is what made 183k rows text-free."""
    fixture = next((FIXTURES / "session5").glob("rollout-*.jsonl"))
    session = CodexParser().parse(fixture).session

    bare = session.orphan_tool_calls[5]
    assert bare.tool_name == "exec"
    assert bare.tool_input == {"source": "const r = 1 + 1;"}


def test_codex_parser_exec_wrapper_reads_js_object_literals() -> None:
    """REQ-PARSE-016: codex writes JS object literals, not JSON — unquoted
    keys are the majority shape in the wild, and a strict JSON reader falls
    back on ~60% of real calls."""
    fixture = next((FIXTURES / "session5").glob("rollout-*.jsonl"))
    calls = CodexParser().parse(fixture).session.orphan_tool_calls

    unquoted = calls[6]
    assert unquoted.tool_name == "exec_command"
    assert unquoted.bash_command == "ls -la"
    assert unquoted.tool_input == {
        "cmd": "ls -la",
        "workdir": "/project",
        "yield_time_ms": 10000,
    }


def test_codex_parser_exec_wrapper_resolves_shorthand_properties() -> None:
    """REQ-PARSE-016: ``{cmd, workdir}`` shorthand names a const declared
    earlier in the same program; the command is recoverable from it."""
    fixture = next((FIXTURES / "session5").glob("rollout-*.jsonl"))
    calls = CodexParser().parse(fixture).session.orphan_tool_calls

    shorthand = calls[7]
    assert shorthand.bash_command == "rg --files-with-matches TODO"
    assert shorthand.bash_base == "rg"


def test_codex_parser_exec_wrapper_keeps_source_beside_partial_arguments() -> None:
    """REQ-PARSE-016: a value that is a live JS expression cannot be
    recovered.  Resolved keys are kept, the unresolved one is omitted, and
    the source rides along so no text is lost."""
    fixture = next((FIXTURES / "session5").glob("rollout-*.jsonl"))
    calls = CodexParser().parse(fixture).session.orphan_tool_calls

    partial = calls[8]
    assert partial.bash_command == "echo hi"
    assert partial.tool_input is not None
    assert "workdir" not in partial.tool_input
    assert "someDir" in partial.tool_input["source"]


def test_codex_parser_exec_wrapper_prunes_nested_unresolved_values() -> None:
    """REQ-PARSE-016: a sentinel left nested in tool_input is not a parse
    detail — `json.dumps` raises on it at insert time and the whole session
    fails to index."""
    import json as json_module

    from recall.parsers.codex import _exec_wrapper_tool_calls

    source = 'const r = await tools.exec_command({cmd: "x", meta: {cwd: dynamic}});'
    call = _exec_wrapper_tool_calls(source)[0]

    assert call.bash_command == "x"
    assert call.tool_input is not None
    assert call.tool_input["meta"] == {}
    assert call.tool_input["source"] == source
    json_module.dumps(call.tool_input)


def test_codex_parser_exec_wrapper_ignores_calls_quoted_inside_source() -> None:
    """REQ-PARSE-016: patch bodies carry code.  A tool call quoted inside one
    never ran, and counting it corrupts tool counts."""
    from recall.parsers.codex import _exec_wrapper_tool_calls

    source = (
        'const patch = "*** Begin Patch\\n+ tools.exec_command({cmd: \\"nope\\"})";\n'
        "const r = await tools.apply_patch(patch);"
    )
    assert [call.tool_name for call in _exec_wrapper_tool_calls(source)] == ["apply_patch"]


def test_exec_wrapper_resolves_a_bare_identifier_argument(tmp_path: Path) -> None:
    """REQ-PARSE-018: `tools.apply_patch(patch)` passes a variable, not an
    object literal.  The declaration is a plain string const the reader
    already collects, so the payload is recoverable."""
    from recall.parsers.codex import _exec_wrapper_tool_calls

    patch = "*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n*** End Patch"
    program = f"const patch = {json.dumps(patch)};\nconst r = await tools.apply_patch(patch);"
    calls = _exec_wrapper_tool_calls(program)
    assert [c.tool_name for c in calls] == ["apply_patch"]
    assert calls[0].tool_input == {"input": patch}


def test_exec_wrapper_identifier_that_resolves_to_nothing_keeps_the_source() -> None:
    """An identifier bound at runtime cannot be recovered, but the program
    text still carries the intent, so no row goes text-free."""
    from recall.parsers.codex import _exec_wrapper_tool_calls

    program = "const patch = await build();\nawait tools.apply_patch(patch);"
    calls = _exec_wrapper_tool_calls(program)
    assert calls[0].tool_input == {"source": program}


def test_web_search_preserves_a_search_query(tmp_path: Path) -> None:
    """REQ-PARSE-019: a search action has no url, it has a query.  Reading
    only `url` recorded nothing for every actual search."""
    session = _write_and_parse(
        tmp_path,
        [
            {
                "timestamp": "2026-07-10T09:00:00.000Z",
                "type": "response_item",
                "payload": {
                    "type": "web_search_call",
                    "action": {"type": "search", "query": "Zig 0.15.1 release notes"},
                },
            }
        ],
    )
    call = session.orphan_tool_calls[0]
    assert call.tool_name == "web_search"
    assert call.tool_input is not None
    assert call.tool_input.get("query") == "Zig 0.15.1 release notes"


def test_web_search_still_preserves_an_opened_page_url(tmp_path: Path) -> None:
    session = _write_and_parse(
        tmp_path,
        [
            {
                "timestamp": "2026-07-10T09:00:00.000Z",
                "type": "response_item",
                "payload": {
                    "type": "web_search_call",
                    "action": {"type": "open_page", "url": "https://playwright.dev/docs/auth"},
                },
            }
        ],
    )
    call = session.orphan_tool_calls[0]
    assert call.tool_input is not None
    assert call.tool_input.get("url") == "https://playwright.dev/docs/auth"


def test_codex_parser_ingests_unwrapped_rollout_records(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    records = [
        {
            "id": "db67a7de-0000-4000-8000-000000000001",
            "timestamp": "2025-09-05T22:29:07.947Z",
            "instructions": None,
            "git": {
                "commit_hash": "abc123",
                "branch": "main",
                "repository_url": "ssh://git@github.com/example/monorepo.git",
            },
        },
        {
            "type": "response_item",
            "timestamp": "2025-09-05T22:29:08.000Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "organize the monorepo"}],
            },
        },
        {
            "type": "reasoning",
            "id": "rs_test",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "**Organizing monorepo tasks**\nPlan the work.",
                }
            ],
            "content": None,
            "encrypted_content": "gAAAA-opaque",
        },
        {
            "type": "function_call",
            "id": "fc_test",
            "name": "update_plan",
            "arguments": json.dumps({"plan": [{"step": "inventory"}]}),
            "call_id": "call_plan",
        },
        {
            "type": "function_call_output",
            "call_id": "call_plan",
            "output": "Plan updated",
        },
        {
            "type": "response_item",
            "timestamp": "2025-09-05T22:29:09.000Z",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = CodexParser().parse(path)

    assert result.diagnostics == ()
    assert result.next_byte_offset == path.stat().st_size
    assert result.session.source_session_id == "db67a7de-0000-4000-8000-000000000001"
    assert result.session.git_branch == "main"
    assert [
        (message.role.value, message.content, message.thinking)
        for message in result.session.messages
    ] == [
        ("user", "organize the monorepo", None),
        ("assistant", None, "**Organizing monorepo tasks**\nPlan the work."),
        ("assistant", "done", None),
    ]
    assert [
        (call.tool_name, call.tool_use_id, call.tool_input)
        for call in result.session.orphan_tool_calls
    ] == [("update_plan", "call_plan", {"plan": [{"step": "inventory"}]})]
    assert [(item.tool_use_id, item.result_summary) for item in result.tail_facts.tool_results] == [
        ("call_plan", "Plan updated")
    ]


def test_codex_parser_skips_encrypted_only_reasoning(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "reasoning",
                "id": "rs_secret",
                "summary": [],
                "content": None,
                "encrypted_content": "gAAAA-do-not-decode",
            }
        )
        + "\n"
    )
    result = CodexParser().parse(path)

    assert result.diagnostics == ()
    assert result.next_byte_offset == path.stat().st_size
    assert result.session.messages == []
    assert result.session.orphan_tool_calls == []
    assert result.session.is_complete is True


def test_codex_retained_context_verified_answer_is_acknowledged() -> None:
    """retained_context/verified_answer restates a request_user_input exchange.

    The questions are already indexed from the paired function_call's
    arguments and the answers from its function_call_output, so the record
    is acknowledged without a diagnostic and indexes nothing itself
    (REQ-PARSE-032).
    """
    fixture = FIXTURES / "retained_context" / "rollout.jsonl"
    result = CodexParser().parse(fixture)
    session = result.session

    assert result.diagnostics == ()
    assert result.next_byte_offset == fixture.stat().st_size
    assert session.is_complete is True
    assert session.source_session_id == "codex-retained-context"
    assert session.messages == []

    assert [
        (call.tool_name, call.tool_use_id, call.tool_input) for call in session.orphan_tool_calls
    ] == [
        (
            "request_user_input",
            "call_retained_1",
            {
                "questions": [
                    {
                        "header": "Scope",
                        "id": "scope",
                        "question": "Which scope should the change cover?",
                        "options": [
                            {
                                "label": "Parser only (Recommended)",
                                "description": "Touch only the parser mapping.",
                            },
                            {
                                "label": "Parser and indexer",
                                "description": "Also change the indexer.",
                            },
                        ],
                    }
                ]
            },
        )
    ]
    assert [(item.tool_use_id,) for item in result.tail_facts.tool_results] == [
        ("call_retained_1",)
    ]
    assert "user_note: keep it minimal" in result.tail_facts.tool_results[0].result_summary


def test_codex_retained_context_unknown_payload_type_diagnoses(tmp_path: Path) -> None:
    """Only verified_answer is observed in rollouts; any other retained_context
    payload stays fail-closed (REQ-PARSE-032)."""
    path = tmp_path / "rollout.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "retained_context",
                "timestamp": "2026-10-03T22:00:03Z",
                "payload": {"type": "mystery", "turn_id": "turn-001"},
            }
        )
        + "\n"
    )
    result = CodexParser().parse(path)

    assert [d.kind for d in result.diagnostics] == ["unsupported_record"]
    assert result.session.is_complete is False
