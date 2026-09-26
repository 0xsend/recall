"""`recall show --fresh` and the freshness it now reports (REQ-LIVE-003/010).

`show` is the drill-down from `live`, so it owes the same two things: an answer
that says how far behind the index is, and a way to close that gap before
answering.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from recall.cli import show as show_cli
from recall.cli.app import app
from recall.cli.contract import CliError, ErrorCode
from typer.testing import CliRunner


def _session(*, current: bool) -> dict[str, Any]:
    return {
        "id": "abc123",
        "source": "claude_code",
        "source_path": "/home/a/.claude/projects/p/s.jsonl",
        "started_at": "2026-09-08T12:00:00",
        "message_count": 4,
        "tool_count": 1,
        "messages": [],
        "freshness": {
            "file_mtime": 1_700_000_060.0,
            "file_size": 4096,
            "indexed_mtime": 1_700_000_060.0 if current else 1_700_000_000.0,
            "indexed_size": 4096 if current else 2048,
            "lag_seconds": 0.0 if current else 60.0,
            "lag_bytes": 0 if current else 2048,
            "current": current,
        },
    }


def _install_rpc(
    monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]
) -> list[tuple[str, dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_rpc(method: str, params: dict[str, object], **_kwargs: object) -> dict[str, Any]:
        calls.append((method, params))
        return result

    monkeypatch.setattr(show_cli, "rpc_call_or_error", fake_rpc)
    return calls


def test_show_reports_freshness_without_being_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-LIVE-002: every read of a session that may be live reports its lag."""
    _install_rpc(monkeypatch, _session(current=False))

    result = CliRunner().invoke(app, ["show", "abc123", "--json", "--fields", "freshness"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["freshness"]["lag_bytes"] == 2048  # manifest exposes it


def test_fresh_is_forwarded_to_the_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_rpc(monkeypatch, _session(current=True))

    result = CliRunner().invoke(app, ["show", "abc123", "--fresh", "--json"])

    assert result.exit_code == 0, result.output
    assert calls == [("recall.show", {"session_id": "abc123", "tools": False, "fresh": True})]


def test_fresh_says_so_when_the_session_did_not_catch_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_rpc(monkeypatch, _session(current=False))

    result = CliRunner().invoke(app, ["show", "abc123", "--fresh", "--json"])

    assert result.exit_code == 0, result.output
    assert "still behind" in result.stderr


def test_fresh_adds_no_note_when_the_session_caught_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_rpc(monkeypatch, _session(current=True))

    result = CliRunner().invoke(app, ["show", "abc123", "--fresh", "--json"])

    assert result.exit_code == 0, result.output
    assert result.stderr == ""


def test_fresh_fails_closed_when_no_daemon_can_serve_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_rpc(method: str, params: dict[str, object], **_kwargs: object) -> dict[str, Any]:
        raise CliError(code=ErrorCode.RUNTIME, message="daemon unavailable", exit_code=1)

    monkeypatch.setattr(show_cli, "rpc_call_or_error", failing_rpc)

    result = CliRunner().invoke(app, ["show", "abc123", "--fresh", "--json"])

    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.RUNTIME.value


def test_fresh_travels_through_params(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_rpc(monkeypatch, _session(current=True))

    result = CliRunner().invoke(
        app, ["show", "--params", json.dumps({"session_id": "abc123", "fresh": True, "json": True})]
    )

    assert result.exit_code == 0, result.output
    assert calls == [("recall.show", {"session_id": "abc123", "tools": False, "fresh": True})]


@pytest.mark.parametrize("freshness", [None, {}, {"current": "true"}])
def test_fresh_refuses_a_daemon_answer_without_a_freshness_observation(
    monkeypatch: pytest.MonkeyPatch, freshness: object
) -> None:
    response = _session(current=True)
    response["freshness"] = freshness
    _install_rpc(monkeypatch, response)

    result = CliRunner().invoke(app, ["show", "abc123", "--fresh", "--json"])

    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.RUNTIME.value
