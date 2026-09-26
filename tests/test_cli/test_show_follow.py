"""`recall show --follow` at the CLI boundary (REQ-LIVE-004).

The stream itself is proved against a real index and a real event channel in
`tests/test_services/test_show_follow.py`. What the command adds is the wire
shape a consumer parses: NDJSON, one object per line, whatever `--format` says
— a follower is piped into `while read line`, and a TOON table or a pretty
JSON array would never terminate a line-oriented reader.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from recall.cli import show as show_cli
from recall.cli.app import app
from recall.cli.contract import CliError, ErrorCode
from typer.testing import CliRunner

CURSOR = "djE6YWJjMTIzOjM"


def _delta(text: str, idx: int) -> dict[str, Any]:
    return {
        "event": "delta",
        "cursor": CURSOR,
        "messages": [{"idx": idx, "role": "user", "content": text}],
    }


def _install_stream(
    monkeypatch: pytest.MonkeyPatch,
    notifications: list[tuple[str, dict[str, Any]]],
    closed: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, object]]]:
    """Replay a daemon stream: fire the notifications, then return the closed line."""
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_rpc(
        method: str,
        params: dict[str, object],
        *,
        on_notification: Callable[[str, dict[str, Any]], None] | None = None,
        **_kwargs: object,
    ) -> dict[str, Any]:
        calls.append((method, params))
        for name, payload in notifications:
            assert on_notification is not None, "follow must pass a notification sink"
            on_notification(name, payload)
        return closed or {
            "event": "closed",
            "reason": "timeout",
            "cursor": CURSOR,
            "watching": True,
        }

    monkeypatch.setattr(show_cli, "rpc_call_or_error", fake_rpc)
    return calls


def test_follow_emits_one_json_object_per_line(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stream(
        monkeypatch, [("live.delta", _delta("first", 2)), ("live.delta", _delta("second", 3))]
    )

    result = CliRunner().invoke(app, ["show", "abc123", "--follow"])

    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.stdout.splitlines() if line]
    assert [line["event"] for line in lines] == ["delta", "delta", "closed"]
    assert lines[0]["messages"][0]["content"] == "first"


def test_follow_is_ndjson_even_when_a_format_says_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A follower is a line-oriented pipe; a TOON table would never terminate."""
    _install_stream(monkeypatch, [("live.delta", _delta("first", 2))])

    result = CliRunner().invoke(app, ["show", "abc123", "--follow", "--format", "toon"])

    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.stdout.splitlines() if line]
    assert [line["event"] for line in lines] == ["delta", "closed"]


def test_the_closed_line_is_last_and_carries_the_resumable_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stream(monkeypatch, [])

    result = CliRunner().invoke(app, ["show", "abc123", "--follow"])

    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.stdout.splitlines() if line]
    assert lines == [{"event": "closed", "reason": "timeout", "cursor": CURSOR, "watching": True}]


def test_follow_forwards_its_window_and_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_stream(monkeypatch, [])

    result = CliRunner().invoke(
        app, ["show", "abc123", "--follow", "--after", CURSOR, "--timeout", "5"]
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            "recall.show_follow",
            {"session_id": "abc123", "tools": False, "after": CURSOR, "timeout": 5.0},
        )
    ]


def test_an_unwatched_daemon_says_nobody_is_looking(monkeypatch: pytest.MonkeyPatch) -> None:
    """No watcher means no event can ever fire; silence would read as 'nothing happened'."""
    _install_stream(
        monkeypatch,
        [],
        closed={"event": "closed", "reason": "timeout", "cursor": CURSOR, "watching": False},
    )

    result = CliRunner().invoke(app, ["show", "abc123", "--follow"])

    assert result.exit_code == 0, result.output
    assert "not running a live watcher" in result.stderr


def test_a_watching_daemon_adds_no_note(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stream(monkeypatch, [])

    result = CliRunner().invoke(app, ["show", "abc123", "--follow"])

    assert result.exit_code == 0, result.output
    assert result.stderr == ""


def test_follow_is_refused_with_fleet(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_fleet_show` is a one-shot SSH call; it has no stream to hold open."""
    _install_stream(monkeypatch, [])

    result = CliRunner().invoke(app, ["show", "abc123", "--follow", "--fleet", "--json"])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.VALIDATION.value


def test_a_non_positive_timeout_is_a_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stream(monkeypatch, [])

    result = CliRunner().invoke(app, ["show", "abc123", "--follow", "--timeout", "0", "--json"])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.VALIDATION.value


def test_a_timeout_without_follow_is_a_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing else on `show` has a deadline; accepting one would be a no-op that lies."""
    _install_stream(monkeypatch, [])

    result = CliRunner().invoke(app, ["show", "abc123", "--timeout", "5", "--json"])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.VALIDATION.value


def test_a_daemon_error_surfaces_as_the_cli_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_rpc(method: str, params: dict[str, object], **_kwargs: object) -> dict[str, Any]:
        raise CliError(code=ErrorCode.RUNTIME, message="daemon unavailable", exit_code=1)

    monkeypatch.setattr(show_cli, "rpc_call_or_error", failing_rpc)

    result = CliRunner().invoke(app, ["show", "abc123", "--follow", "--json"])

    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.RUNTIME.value


def test_follow_travels_through_params(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_stream(monkeypatch, [])

    result = CliRunner().invoke(
        app,
        ["show", "--params", json.dumps({"session_id": "abc123", "follow": True, "timeout": 5})],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        ("recall.show_follow", {"session_id": "abc123", "tools": False, "timeout": 5.0})
    ]


def test_the_socket_idle_bound_outlives_the_stream_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silence past the deadline means a dead daemon, not a quiet session."""
    seen: dict[str, Any] = {}

    def fake_rpc(method: str, params: dict[str, object], **kwargs: object) -> dict[str, Any]:
        seen.update(kwargs)
        return {"event": "closed", "reason": "timeout", "cursor": CURSOR, "watching": True}

    monkeypatch.setattr(show_cli, "rpc_call_or_error", fake_rpc)

    result = CliRunner().invoke(app, ["show", "abc123", "--follow", "--timeout", "5"])

    assert result.exit_code == 0, result.output
    assert seen["idle_timeout"] == 35.0
