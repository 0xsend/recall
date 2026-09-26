"""Current rollout record shapes observed in the bounded host sample census."""

from __future__ import annotations

import json
from pathlib import Path

from recall.parsers.codex import CodexParser


def test_current_records_preserve_conversation_and_collapse_correlated_mirrors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rollout.jsonl"
    turn = {"turn_id": "turn-one"}
    records = [
        {"type": "session_meta", "payload": {"id": "current"}},
        {"type": "world_state", "payload": {"full": True, "state": {}}},
        {"type": "turn_context", "payload": {"turn_id": "turn-one", "model": "model-a"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Project instructions"}],
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "turn_id": "turn-one",
                "item": {
                    "type": "UserMessage",
                    "id": "local-user",
                    "content": [{"type": "text", "text": "Question"}],
                },
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Question"}],
                "internal_chat_message_metadata_passthrough": turn,
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "turn_id": "turn-one",
                "item": {
                    "type": "Reasoning",
                    "id": "local-thought",
                    "summary_text": ["A summary"],
                    "raw_content": [],
                },
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "A summary"}],
                "encrypted_content": "opaque",
                "internal_chat_message_metadata_passthrough": turn,
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "agent_message",
                "author": "worker",
                "recipient": "driver",
                "content": [{"type": "input_text", "text": "Worker result"}],
            },
        },
        {
            "type": "event_msg",
            "payload": {"type": "thread_settings_applied", "thread_settings": {}},
        },
        {"type": "inter_agent_communication_metadata", "payload": {"trigger_turn": True}},
        {"type": "event_msg", "payload": {"type": "thread_goal_updated", "goal": {}}},
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "item": {
                    "type": "FileChange",
                    "id": "rendered-change",
                    "status": "completed",
                    "changes": {"a.py": {"type": "update", "unified_diff": "@@ -1 +1 @@"}},
                    "stdout": "",
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "item": {
                    "type": "Extension",
                    "kind": "web.search",
                    "id": "exec-ext-1",
                    "query": "docs",
                    "action": {"type": "search"},
                    "results": [],
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "item": {
                    "type": "CollabAgentToolCall",
                    "id": "call_wait_1",
                    "tool": "wait",
                    "status": "completed",
                },
            },
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = CodexParser().parse(path)
    assert result.diagnostics == ()
    assert result.next_byte_offset == path.stat().st_size
    assert [(m.role.value, m.content, m.thinking) for m in result.session.messages] == [
        ("system", "Project instructions", None),
        ("user", "Question", None),
        ("assistant", None, "A summary"),
        ("assistant", "Worker result", None),
    ]
    assert result.session.model == "model-a"


def test_compaction_snapshot_preserves_new_summary_without_replaying_prior_messages(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rollout.jsonl"
    question = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Question"}],
    }
    summary = {
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": "Compacted summary"}],
    }
    records = [
        {"type": "response_item", "payload": question},
        {
            "type": "compacted",
            "payload": {
                "message": "",
                "replacement_history": [
                    question,
                    {"type": "compaction", "encrypted_content": "opaque"},
                    summary,
                ],
            },
        },
        {"type": "response_item", "payload": question},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    parsed = CodexParser().parse(path)
    assert parsed.diagnostics == ()
    assert [m.content for m in parsed.session.messages] == [
        "Question",
        "Compacted summary",
        "Question",
    ]


def test_unknown_event_record_is_diagnosed_and_not_acknowledged(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    path.write_text(
        json.dumps({"type": "event_msg", "payload": {"type": "future_conversation"}}) + "\n"
    )
    parsed = CodexParser().parse(path)
    assert parsed.next_byte_offset == 0
    assert [item.kind for item in parsed.diagnostics] == ["unsupported_record"]


def test_tool_search_and_cumulative_usage_records_are_consumed(tmp_path: Path) -> None:
    records = [
        {
            "type": "response_item",
            "payload": {
                "type": "tool_search_call",
                "call_id": "search-one",
                "arguments": {"query": "find tools", "limit": 2},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "tool_search_output",
                "call_id": "search-one",
                "tools": [{"type": "function", "name": "example"}],
            },
        },
        {
            "type": "token_usage_record",
            "payload": {
                "usage": {"input_tokens": 7, "output_tokens": 2},
                "thread_token_usage": {"input_tokens": 120, "output_tokens": 30},
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {"total_token_usage": {"input_tokens": 100, "output_tokens": 20}},
            },
        },
    ]
    path = tmp_path / "rollout.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    result = CodexParser().parse(path)
    assert result.diagnostics == ()
    assert result.next_byte_offset == path.stat().st_size
    assert (result.session.input_tokens, result.session.output_tokens) == (120, 30)
    assert [
        (t.tool_name, t.tool_input, t.tool_use_id) for t in result.session.orphan_tool_calls
    ] == [("tool_search", {"query": "find tools", "limit": 2}, "search-one")]
    assert len(result.tail_facts.tool_results) == 1
    assert "example" in (result.tail_facts.tool_results[0].result_summary or "")


def test_replacement_history_accepts_agent_messages_without_replaying_them(tmp_path: Path) -> None:
    message = {
        "type": "agent_message",
        "id": "assistant-one",
        "author": "worker",
        "content": [
            {"type": "input_text", "text": "answer"},
            {"type": "encrypted_content", "encrypted_content": "opaque"},
        ],
    }
    path = tmp_path / "rollout.jsonl"
    path.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in [
                {"type": "response_item", "payload": message},
                {
                    "type": "compacted",
                    "payload": {"replacement_history": [message], "message": "retained summary"},
                },
            ]
        )
    )
    result = CodexParser().parse(path)
    assert result.diagnostics == ()
    assert [m.content for m in result.session.messages] == ["answer", "retained summary"]
    assert result.next_byte_offset == path.stat().st_size


def test_compaction_empty_other_placeholder_has_no_content_but_unknown_payload_is_diagnosed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rollout.jsonl"
    for item, expected_diagnostics in [
        ({"type": "other"}, 0),
        ({"type": "other", "content": "unrecognized content"}, 1),
    ]:
        path.write_text(
            json.dumps(
                {
                    "type": "compacted",
                    "payload": {"message": "summary", "replacement_history": [item]},
                }
            )
            + "\n"
        )
        result = CodexParser().parse(path)
        assert len(result.diagnostics) == expected_diagnostics
        assert [message.content for message in result.session.messages] == ["summary"]


def test_completed_activity_retains_unmirrored_content_and_calls_in_source_order(
    tmp_path: Path,
) -> None:
    def completed(item):
        return {
            "type": "event_msg",
            "payload": {"type": "item_completed", "turn_id": "turn", "item": item},
        }

    records = [
        completed(
            {"type": "HookPrompt", "id": "hook", "fragments": [{"text": "hook instructions"}]}
        ),
        completed(
            {
                "type": "CommandExecution",
                "id": "command",
                "command": ["printf", "hello"],
                "cwd": "/owned",
                "aggregated_output": "hello",
                "exit_code": 0,
            }
        ),
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "other",
                "name": "read_file",
                "arguments": {"path": "a.py"},
            },
        },
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": "question"},
        },
        completed({"type": "Plan", "id": "plan", "text": "inspect then verify"}),
        completed(
            {
                "type": "McpToolCall",
                "id": "mcp",
                "server": "example",
                "tool": "lookup",
                "arguments": {"key": "value"},
                "result": {"content": [{"type": "text", "text": "found"}], "isError": False},
            }
        ),
        completed(
            {
                "type": "WebSearch",
                "id": "web",
                "query": "topic",
                "action": {"type": "search", "query": "topic"},
            }
        ),
        completed(
            {
                "type": "EnteredReviewMode",
                "id": "entered",
                "target": {"type": "custom", "instructions": "check correctness"},
                "user_facing_hint": "reviewing",
            }
        ),
        completed(
            {
                "type": "ExitedReviewMode",
                "id": "exited",
                "review_output": {"findings": [], "overall_explanation": "checked"},
            }
        ),
        {"type": "event_msg", "payload": {"type": "task_complete"}},
    ]
    path = tmp_path / "rollout.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    result = CodexParser().parse(path)
    assert result.diagnostics == ()
    messages = result.session.messages
    assert [m.content for m in messages[:3]] == [
        "hook instructions",
        "question",
        "inspect then verify",
    ]
    assert "check correctness" in (messages[3].content or "")
    assert "checked" in (messages[4].content or "")
    assert [m.idx for m in messages] == list(range(5))
    assert result.tail_facts.stop_markers[0].idx == 4
    assert [t.tool_name for t in result.session.orphan_tool_calls] == [
        "exec_command",
        "read_file",
        "mcp__example__lookup",
        "web_search",
    ]
    assert {r.tool_use_id for r in result.tail_facts.tool_results} == {"command", "mcp"}
    assert result.next_byte_offset == path.stat().st_size


def test_completed_tool_mirrors_prefer_response_records_in_either_order(tmp_path: Path) -> None:
    native = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {
                "type": "McpToolCall",
                "id": "call",
                "server": "example",
                "tool": "lookup",
                "arguments": {"key": "value"},
                "result": {"content": [{"type": "text", "text": "native result"}]},
            },
        },
    }
    response = {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "call_id": "call",
            "name": "canonical_lookup",
            "arguments": {"key": "value"},
        },
    }
    output = {
        "type": "response_item",
        "payload": {
            "type": "function_call_output",
            "call_id": "call",
            "output": "canonical result",
        },
    }
    for index, records in enumerate(([native, response, output], [response, output, native])):
        path = tmp_path / f"rollout-{index}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        result = CodexParser().parse(path)
        assert result.diagnostics == ()
        assert [(t.tool_name, t.tool_input) for t in result.session.orphan_tool_calls] == [
            ("canonical_lookup", {"key": "value"})
        ]
        assert len(result.tail_facts.tool_results) == 1
        assert result.tail_facts.tool_results[0].result_summary == "canonical result"


def test_native_exec_activities_do_not_duplicate_their_observed_wrapper_calls(
    tmp_path: Path,
) -> None:
    wrapper = {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call",
            "name": "exec",
            "call_id": "wrapper",
            "input": (
                "await tools.mcp__example__lookup({key: 'value'}); "
                "await tools.web__run({search_query: [{q: 'topic'}]});"
            ),
            "internal_chat_message_metadata_passthrough": {"turn_id": "turn"},
        },
    }

    def completed(kind, identity, turn, **values):
        return {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "turn_id": turn,
                "item": {"type": kind, "id": identity, **values},
            },
        }

    native = [
        completed(
            "McpToolCall",
            "exec-mcp",
            "turn",
            server="example",
            tool="lookup",
            arguments={"key": "value"},
            result={"content": []},
        ),
        completed("WebSearch", "exec-web", "turn", query="topic"),
    ]
    for index, records in enumerate(([wrapper, *native], [*native, wrapper])):
        path = tmp_path / f"rollout-{index}.jsonl"
        path.write_text(
            "".join(
                json.dumps(r) + "\n"
                for r in [
                    *records,
                    completed(
                        "McpToolCall",
                        "exec-other-turn",
                        "other",
                        server="example",
                        tool="lookup",
                        arguments={"key": "other"},
                        result={"content": []},
                    ),
                ]
            )
        )
        result = CodexParser().parse(path)
        assert result.diagnostics == ()
        assert [(t.tool_name, t.tool_input) for t in result.session.orphan_tool_calls] == [
            ("mcp__example__lookup", {"key": "value"}),
            ("web__run", {"search_query": [{"q": "topic"}]}),
            ("mcp__example__lookup", {"key": "other"}),
        ]


def test_native_text_and_turn_alias_preserve_commentary_without_replaying_mirrors(
    tmp_path: Path,
) -> None:
    timestamp = "2026-08-20T09:02:09.148Z"
    records = [
        {
            "type": "response_item",
            "timestamp": timestamp,
            "payload": {
                "type": "message",
                "id": "message-user",
                "role": "user",
                "content": [{"type": "input_text", "text": "Question"}],
                "internal_chat_message_metadata_passthrough": {"turn_id": "engine-turn"},
            },
        },
        {
            "type": "event_msg",
            "timestamp": timestamp,
            "payload": {"type": "task_started", "turn_id": "rollout-2"},
        },
        {
            "type": "event_msg",
            "timestamp": timestamp,
            "payload": {
                "type": "item_completed",
                "turn_id": "rollout-2",
                "item": {
                    "type": "UserMessage",
                    "id": "item-1",
                    "content": [{"type": "text", "text": "Question", "text_elements": []}],
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "turn_id": "rollout-2",
                "item": {
                    "type": "AgentMessage",
                    "id": "item-2",
                    "phase": "commentary",
                    "content": [{"type": "Text", "text": "First commentary"}],
                },
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "message-assistant",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "First commentary"}],
                "internal_chat_message_metadata_passthrough": {"turn_id": "engine-turn"},
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "turn_id": "rollout-2",
                "item": {
                    "type": "AgentMessage",
                    "id": "item-3",
                    "phase": "commentary",
                    "content": [{"type": "Text", "text": "Second commentary"}],
                },
            },
        },
    ]
    path = tmp_path / "rollout.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = CodexParser().parse(path)

    assert result.diagnostics == ()
    assert [(message.role.value, message.content) for message in result.session.messages] == [
        ("user", "Question"),
        ("assistant", "First commentary"),
        ("assistant", "Second commentary"),
    ]


def test_equal_questions_in_distinct_unaliased_turns_remain_distinct(tmp_path: Path) -> None:
    for timestamps in (
        ("2026-08-20T09:02:09.148Z", "2026-08-20T09:02:10.148Z"),
        (None, None),
    ):
        path = tmp_path / "rollout.jsonl"
        records = [
            {
                "type": "response_item",
                "timestamp": timestamps[0],
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": "Again",
                    "internal_chat_message_metadata_passthrough": {"turn_id": "first"},
                },
            },
            {
                "type": "event_msg",
                "timestamp": timestamps[1],
                "payload": {"type": "task_started", "turn_id": "second"},
            },
            {
                "type": "event_msg",
                "timestamp": timestamps[1],
                "payload": {
                    "type": "item_completed",
                    "turn_id": "second",
                    "item": {
                        "type": "UserMessage",
                        "id": "second-user",
                        "content": [{"type": "text", "text": "Again"}],
                    },
                },
            },
        ]
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        result = CodexParser().parse(path)
        assert result.diagnostics == ()
        assert [message.content for message in result.session.messages] == ["Again", "Again"]


def test_unknown_readable_native_block_is_diagnostic_instead_of_an_empty_message(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rollout.jsonl"
    record = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {
                "type": "AgentMessage",
                "id": "future-item",
                "content": [{"type": "FutureText", "text": "Do not silently discard this"}],
            },
        },
    }
    path.write_text(json.dumps(record) + "\n")
    result = CodexParser().parse(path)

    assert result.session.messages == []
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].kind == "unsupported_record"
    assert "FutureText" in result.diagnostics[0].detail


def test_forked_rollout_keeps_its_own_identity_over_the_embedded_parent_meta(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rollout.jsonl"
    records = [
        {
            "type": "session_meta",
            "payload": {"id": "child", "forked_from_id": "parent", "cwd": "/work/child"},
        },
        {"type": "session_meta", "payload": {"id": "parent", "cwd": "/work/parent"}},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    session = CodexParser().parse(path).session

    assert session.source_session_id == "child"
    assert session.cwd == "/work/child"
