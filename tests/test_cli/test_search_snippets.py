"""CLI tests for search result snippet normalization, sanitization, and mixed-version peers."""

from __future__ import annotations

import json

from recall.cli.app import app
from typer.testing import CliRunner

runner = CliRunner()


def test_search_cli_sanitizes_peer_snippet_before_json_and_fields_projection(
    monkeypatch,
) -> None:
    dirty_snippet = "\x1b[31mMATCH\x1b[0m\r\n" + ("x" * 400)
    monkeypatch.setattr(
        "recall.cli.search.rpc_call_or_error",
        lambda *_args, **_kwargs: [
            {
                "kind": "message",
                "session_id": "s1",
                "source": "antigravity_cli",
                "source_path": "/tmp/s1.jsonl",
                "score": 0.9,
                "message_id": "m1",
                "tool_call_id": None,
                "role": "assistant",
                "content": "clean body",
                "thinking": None,
                "timestamp": "2026-11-22T00:00:00",
                "tool_name": None,
                "bash_command": None,
                "lexical_match": True,
                "host": "aem5",
                "snippet": dirty_snippet,
            }
        ],
    )
    monkeypatch.setattr("recall.cli.search.fetch_daemon_status", lambda *_args, **_kwargs: None)

    result = runner.invoke(
        app,
        ["search", "MATCH", "--json", "--fields", "session_id,snippet"],
    )
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert len(rows) == 1
    assert set(rows[0].keys()) == {"session_id", "snippet"}
    assert "\x1b" not in rows[0]["snippet"]
    assert "\n" not in rows[0]["snippet"]
    assert len(rows[0]["snippet"]) <= 240
    assert rows[0]["snippet"].startswith("MATCH ")


def test_search_cli_backfills_missing_snippet_from_older_fleet_peer(monkeypatch) -> None:
    long_content = ("intro filler " * 80) + "needle_from_old_peer appears late in message."
    monkeypatch.setattr(
        "recall.cli.fleet.run_fleet_search",
        lambda **_kwargs: (
            [
                {
                    "kind": "message",
                    "session_id": "s-remote",
                    "source": "antigravity_cli",
                    "source_path": "/tmp/remote.jsonl",
                    "score": 0.85,
                    "message_id": "m-remote",
                    "tool_call_id": None,
                    "role": "assistant",
                    "content": long_content,
                    "thinking": None,
                    "timestamp": "2026-11-22T00:00:00",
                    "tool_name": None,
                    "bash_command": None,
                    "lexical_match": True,
                    "host": "ae-dev",
                    # Note: older v0.48.0 peer omits "snippet"
                }
            ],
            "ae-dev",
        ),
    )
    monkeypatch.setattr("recall.cli.search.fetch_daemon_status", lambda *_args, **_kwargs: None)

    json_res = runner.invoke(
        app,
        ["search", "needle_from_old_peer", "--fleet", "--json", "--fields", "session_id,snippet"],
    )
    assert json_res.exit_code == 0, json_res.output
    rows = json.loads(json_res.stdout)
    assert "needle_from_old_peer" in rows[0]["snippet"]

    text_res = runner.invoke(
        app,
        ["search", "needle_from_old_peer", "--fleet", "--format", "text"],
    )
    assert text_res.exit_code == 0, text_res.output
    assert "needle_from_old_peer" in text_res.stdout


def test_search_cli_sanitizes_tool_call_osc_in_text_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "recall.cli.fleet.run_fleet_search",
        lambda **_kwargs: (
            [
                {
                    "kind": "tool_call",
                    "session_id": "s-tool",
                    "source": "antigravity_cli",
                    "source_path": "/tmp/tool.jsonl",
                    "score": 0.9,
                    "message_id": "m-tool",
                    "tool_call_id": "tc-1",
                    "role": None,
                    "content": None,
                    "thinking": None,
                    "timestamp": None,
                    "tool_name": "run_command",
                    "bash_command": "echo \x1b]0;evil\x07safe\x9d0;hidden\x9c",
                    "lexical_match": True,
                    "host": "ae-dev",
                }
            ],
            "ae-dev",
        ),
    )
    monkeypatch.setattr("recall.cli.search.fetch_daemon_status", lambda *_args, **_kwargs: None)

    text_res = runner.invoke(
        app,
        ["search", "echo", "--fleet", "--format", "text"],
    )
    assert text_res.exit_code == 0, text_res.output
    assert "\x1b" not in text_res.stdout
    assert "\x9d" not in text_res.stdout
    assert "evil" not in text_res.stdout
    assert "hidden" not in text_res.stdout
    assert "[run_command] echo safe" in text_res.stdout


def test_search_cli_sanitizes_metadata_and_survives_lone_surrogate_peer_row(monkeypatch) -> None:
    monkeypatch.setattr(
        "recall.cli.fleet.run_fleet_search",
        lambda **_kwargs: (
            [
                {
                    "kind": "message",
                    "session_id": "s-\x1b[31mbad\x1b[0m",
                    "source": "antigravity_cli",
                    "source_path": "/tmp/\x1b]0;pathosc\x07safe.jsonl",
                    "score": 0.9,
                    "message_id": "m1",
                    "tool_call_id": None,
                    "role": "assistant\x1b[31m",
                    "content": "intro\ud800 needle here",
                    "thinking": None,
                    "timestamp": "2026-11-22T00:00:00\x9d0;ts\x9c",
                    "tool_name": None,
                    "bash_command": None,
                    "lexical_match": True,
                    "host": "ae-dev",
                }
            ],
            "ae-dev",
        ),
    )
    monkeypatch.setattr("recall.cli.search.fetch_daemon_status", lambda *_args, **_kwargs: None)

    text_res = runner.invoke(app, ["search", "needle", "--fleet", "--format", "text"])
    assert text_res.exit_code == 0, text_res.output
    assert "\x1b" not in text_res.stdout
    assert "\x1b" not in text_res.stderr
    assert "\x9d" not in text_res.stdout
    assert "\x9d" not in text_res.stderr
    assert "pathosc" not in text_res.stdout
    assert "needle" in text_res.stdout

    json_res = runner.invoke(
        app, ["search", "needle", "--fleet", "--json", "--fields", "session_id,snippet"]
    )
    assert json_res.exit_code == 0, json_res.output
    rows = json.loads(json_res.stdout)
    assert "needle" in rows[0]["snippet"]


def test_search_cli_handles_deeply_nested_tool_input_without_recursion_error(monkeypatch) -> None:
    from recall.core.snippets import build_search_snippet

    nested: object = "leaf"
    for _ in range(1200):
        nested = [nested]
    assert (
        build_search_snippet(
            query="echo",
            kind="tool_call",
            content=None,
            thinking=None,
            role=None,
            tool_name="run_command",
            bash_command="echo safe",
            tool_input={"deep": nested},
        )
        == "[run_command] echo safe"
    )
    monkeypatch.setattr(
        "recall.cli.fleet.run_fleet_search",
        lambda **_kwargs: (
            [
                {
                    "kind": "tool_call",
                    "session_id": "s-deep",
                    "source": "antigravity_cli",
                    "source_path": "/tmp/deep.jsonl",
                    "score": 0.9,
                    "message_id": "m-deep",
                    "tool_call_id": "tc-deep",
                    "role": None,
                    "content": None,
                    "thinking": None,
                    "timestamp": None,
                    "tool_name": "run_command",
                    "bash_command": "echo safe",
                    "tool_input": {"deep": nested},
                    "lexical_match": True,
                    "host": "ae-dev",
                }
            ],
            "ae-dev",
        ),
    )
    monkeypatch.setattr("recall.cli.search.fetch_daemon_status", lambda *_args, **_kwargs: None)

    text_res = runner.invoke(app, ["search", "echo", "--fleet", "--format", "text"])
    assert text_res.exit_code == 0, text_res.output
    assert "[run_command] echo safe" in text_res.stdout
