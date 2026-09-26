"""What `recall show` renders of each message (REQ-CLI-024).

Two defects this file pins. `--thinking` gated only the TEXT renderer, so the
structured output every agent actually reads carried thinking whether or not it
was asked for — the flag was a documented no-op. And a turn whose payload is
tool calls came back as a record with `content`, `thinking` and `tool_calls` all
empty, so a 185-message session rendered 169 informationally blank rows.

The daemon response is stubbed: what is under test is the projection the command
applies to it, not the read that produced it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from recall.cli import show as show_cli
from recall.cli.app import app
from typer.testing import CliRunner

CURSOR = "djE6YWJjMTIzOjM"


def _message(**overrides: Any) -> dict[str, Any]:
    """A full message record in the shape `session.model_dump()` emits."""
    payload: dict[str, Any] = {
        "id": "m1",
        "session_id": "abc123",
        "idx": 0,
        "role": "assistant",
        "content": None,
        "thinking": None,
        "timestamp": "2026-09-17T21:39:49",
        "has_thinking": False,
        "agent_id": None,
        "context_text": "[/repo main] ",
        "context_mode": "template",
        "tool_calls": [],
        "content_embedding": None,
        "thinking_embedding": None,
    }
    payload.update(overrides)
    return payload


def _session(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "abc123",
        "source": "claude_code",
        "source_path": "/home/a/.claude/projects/p/s.jsonl",
        "started_at": "2026-09-17T21:39:28",
        "message_count": len(messages),
        "tool_count": 0,
        "messages": messages,
        "cursor": CURSOR,
        "freshness": {"lag_bytes": 0, "current": True},
    }


def _install_rpc(monkeypatch: pytest.MonkeyPatch, messages: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(
        show_cli,
        "rpc_call_or_error",
        lambda _method, _params, **_kwargs: _session(messages),
    )


def _messages_from(stdout: str) -> list[dict[str, Any]]:
    return json.loads(stdout)["messages"]


# --- --thinking actually gates thinking ----------------------------------


def test_thinking_is_omitted_from_structured_output_by_default(monkeypatch) -> None:
    _install_rpc(
        monkeypatch,
        [_message(content="done", thinking="the private reasoning", has_thinking=True)],
    )

    result = CliRunner().invoke(app, ["show", "abc123", "--json"])

    assert result.exit_code == 0, result.output
    (row,) = _messages_from(result.stdout)
    assert "thinking" not in row
    assert "thinking_embedding" not in row
    # The signal that thinking exists survives; only the payload is withheld.
    assert row["has_thinking"] is True
    assert row["content"] == "done"


def test_thinking_is_present_with_the_flag(monkeypatch) -> None:
    _install_rpc(
        monkeypatch,
        [_message(content="done", thinking="the private reasoning", has_thinking=True)],
    )

    result = CliRunner().invoke(app, ["show", "abc123", "--thinking", "--json"])

    assert result.exit_code == 0, result.output
    (row,) = _messages_from(result.stdout)
    assert row["thinking"] == "the private reasoning"


def test_text_output_withholds_thinking_by_default(monkeypatch) -> None:
    _install_rpc(
        monkeypatch,
        [_message(content="done", thinking="the private reasoning", has_thinking=True)],
    )

    plain = CliRunner().invoke(app, ["show", "abc123", "--format", "text"])
    with_flag = CliRunner().invoke(app, ["show", "abc123", "--thinking", "--format", "text"])

    assert plain.exit_code == 0 and with_flag.exit_code == 0
    assert "the private reasoning" not in plain.stdout
    assert "the private reasoning" in with_flag.stdout


# --- payload-free turns collapse -----------------------------------------


def test_tool_turn_collapses_to_a_summary_row(monkeypatch) -> None:
    _install_rpc(monkeypatch, [_message(idx=2)])

    result = CliRunner().invoke(app, ["show", "abc123", "--json"])

    assert result.exit_code == 0, result.output
    (row,) = _messages_from(result.stdout)
    assert row == {
        "id": "m1",
        "idx": 2,
        "role": "assistant",
        "timestamp": "2026-09-17T21:39:49",
        "has_thinking": False,
        "summary": "no message text; omitted: tool calls (--tools)",
    }


def test_summary_row_reports_a_count_only_when_tool_calls_were_read(monkeypatch) -> None:
    """A read that never asked for tool calls cannot honestly report zero."""
    _install_rpc(monkeypatch, [_message()])

    without = CliRunner().invoke(app, ["show", "abc123", "--json"])
    with_tools = CliRunner().invoke(app, ["show", "abc123", "--tools", "--json"])

    assert without.exit_code == 0 and with_tools.exit_code == 0
    assert "tool_call_count" not in _messages_from(without.stdout)[0]
    collapsed = _messages_from(with_tools.stdout)[0]
    assert collapsed["tool_call_count"] == 0
    assert collapsed["summary"] == "no message text, thinking, or tool calls"


def test_suppressed_thinking_turn_names_the_flag_that_reveals_it(monkeypatch) -> None:
    _install_rpc(monkeypatch, [_message(thinking="private", has_thinking=True)])

    result = CliRunner().invoke(app, ["show", "abc123", "--json"])

    assert result.exit_code == 0, result.output
    (row,) = _messages_from(result.stdout)
    assert row["summary"] == (
        "no message text; omitted: tool calls (--tools), thinking (--thinking)"
    )
    assert row["has_thinking"] is True


def test_a_turn_carrying_tool_calls_is_left_whole(monkeypatch) -> None:
    tool_call = {"tool_name": "Bash", "bash_command": "ls -la"}
    _install_rpc(monkeypatch, [_message(tool_calls=[tool_call])])

    result = CliRunner().invoke(app, ["show", "abc123", "--tools", "--json"])

    assert result.exit_code == 0, result.output
    (row,) = _messages_from(result.stdout)
    assert "summary" not in row
    assert row["tool_calls"] == [tool_call]
    assert row["context_text"] == "[/repo main] "


def test_a_turn_carrying_content_is_left_whole(monkeypatch) -> None:
    _install_rpc(monkeypatch, [_message(role="user", content="fix the bug")])

    result = CliRunner().invoke(app, ["show", "abc123", "--json"])

    assert result.exit_code == 0, result.output
    (row,) = _messages_from(result.stdout)
    assert "summary" not in row
    assert row["session_id"] == "abc123"
    assert row["context_mode"] == "template"


def test_text_mode_renders_a_collapsed_turn_as_one_line(monkeypatch) -> None:
    _install_rpc(monkeypatch, [_message(), _message(id="m2", idx=1)])

    result = CliRunner().invoke(app, ["show", "abc123", "--format", "text"])

    assert result.exit_code == 0, result.output
    rendered = [line for line in result.stdout.splitlines() if "no message text" in line]
    assert len(rendered) == 2
    assert rendered[0].endswith("assistant: no message text; omitted: tool calls (--tools)")


# --- the same view applies to the follow stream --------------------------


def test_follow_frames_apply_the_same_view(monkeypatch) -> None:
    delta = {
        "event": "delta",
        "cursor": CURSOR,
        "messages": [_message(thinking="private", has_thinking=True), _message(id="m2", idx=1)],
    }

    def fake_rpc(
        _method: str,
        _params: dict[str, object],
        *,
        on_notification: Callable[[str, dict[str, Any]], None] | None = None,
        **_kwargs: object,
    ) -> dict[str, Any]:
        assert on_notification is not None
        on_notification("recall.show_delta", delta)
        return {"event": "closed", "reason": "timeout", "cursor": CURSOR, "watching": True}

    monkeypatch.setattr(show_cli, "rpc_call_or_error", fake_rpc)

    result = CliRunner().invoke(app, ["show", "abc123", "--follow"])

    assert result.exit_code == 0, result.output
    frame = json.loads(result.stdout.splitlines()[0])
    assert all("thinking" not in row for row in frame["messages"])
    assert all("summary" in row for row in frame["messages"])
