"""`recall show --tail` / `--after` at the CLI boundary (REQ-LIVE-004).

The windowing itself is proved against a real index in
`tests/test_services/test_show_tail.py`; what the command adds is forwarding,
the bounds it refuses to combine, and the fact that a bound is never silently
dropped — a monitor loop that asked for 20 messages and got 10 MB is the
failure this file exists to prevent.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from recall.cli import show as show_cli
from recall.cli.app import app
from recall.cli.contract import ErrorCode
from typer.testing import CliRunner

CURSOR = "djE6YWJjMTIzOjM"


def _session(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "abc123",
        "source": "claude_code",
        "source_path": "/home/a/.claude/projects/p/s.jsonl",
        "started_at": "2026-09-08T12:00:00",
        "message_count": 4,
        "tool_count": 0,
        "messages": [],
        "cursor": CURSOR,
        "freshness": {"lag_bytes": 0, "current": True},
    }
    payload.update(overrides)
    return payload


def _install_rpc(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_rpc(method: str, params: dict[str, object], **_kwargs: object) -> dict[str, Any]:
        calls.append((method, params))
        return _session()

    monkeypatch.setattr(show_cli, "rpc_call_or_error", fake_rpc)
    return calls


def test_tail_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_rpc(monkeypatch)

    result = CliRunner().invoke(app, ["show", "abc123", "--tail", "20", "--json"])

    assert result.exit_code == 0, result.output
    assert calls == [("recall.show", {"session_id": "abc123", "tools": False, "tail": 20})]


def test_after_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_rpc(monkeypatch)

    result = CliRunner().invoke(app, ["show", "abc123", "--after", CURSOR, "--json"])

    assert result.exit_code == 0, result.output
    assert calls == [("recall.show", {"session_id": "abc123", "tools": False, "after": CURSOR})]


def test_tail_and_after_compose(monkeypatch: pytest.MonkeyPatch) -> None:
    """A follower that fell behind wants the newest N of the delta, not all of it."""
    calls = _install_rpc(monkeypatch)

    result = CliRunner().invoke(app, ["show", "abc123", "--after", CURSOR, "--tail", "5", "--json"])

    assert result.exit_code == 0, result.output
    assert calls == [
        ("recall.show", {"session_id": "abc123", "tools": False, "tail": 5, "after": CURSOR})
    ]


def test_the_cursor_is_projectable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A polling caller reads the next cursor out of the answer it just got."""
    _install_rpc(monkeypatch)

    result = CliRunner().invoke(app, ["show", "abc123", "--json", "--fields", "cursor"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"cursor": CURSOR}


@pytest.mark.parametrize(
    "args",
    [
        ["--tail", "5", "--message-limit", "5"],
        ["--after", CURSOR, "--message-limit", "5"],
    ],
    ids=["tail", "after"],
)
def test_a_tail_window_and_a_head_limit_are_refused_together(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    """`--message-limit` reads from the start; these read from the end."""
    _install_rpc(monkeypatch)

    result = CliRunner().invoke(app, ["show", "abc123", *args, "--json"])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.VALIDATION.value


@pytest.mark.parametrize("args", [["--tail", "5"], ["--after", CURSOR]], ids=["tail", "after"])
def test_a_window_is_refused_rather_than_dropped_on_the_fleet_path(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    """`run_fleet_show` cannot carry a window; silently ignoring one would lie."""
    _install_rpc(monkeypatch)

    result = CliRunner().invoke(app, ["show", "abc123", "--fleet", *args, "--json"])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.VALIDATION.value


def test_a_non_positive_tail_is_a_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_rpc(monkeypatch)

    result = CliRunner().invoke(app, ["show", "abc123", "--tail", "0", "--json"])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.VALIDATION.value


def test_the_window_travels_through_params(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_rpc(monkeypatch)

    result = CliRunner().invoke(
        app,
        [
            "show",
            "--params",
            json.dumps({"session_id": "abc123", "tail": 5, "after": CURSOR, "json": True}),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        ("recall.show", {"session_id": "abc123", "tools": False, "tail": 5, "after": CURSOR})
    ]


@pytest.mark.parametrize("args", [["--tail", "2"], ["--after", CURSOR]])
def test_an_incompatible_daemon_cannot_silently_return_the_full_session(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    """A running released daemon can accept unknown parameters and ignore them."""
    legacy = _session(messages=[{"content": "OUTSIDE_REQUESTED_WINDOW"}] * 3)
    del legacy["cursor"]
    del legacy["freshness"]
    monkeypatch.setattr(show_cli, "rpc_call_or_error", lambda *_args, **_kwargs: legacy)

    result = CliRunner().invoke(app, ["show", "abc123", *args, "--json"])

    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.RUNTIME.value
    assert "daemon" in json.loads(result.stdout)["error"]["message"]
    assert "OUTSIDE_REQUESTED_WINDOW" not in result.output


def test_a_daemon_response_exceeding_the_tail_bound_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = _session(messages=[{"content": "OUTSIDE_REQUESTED_WINDOW"}] * 3)
    monkeypatch.setattr(show_cli, "rpc_call_or_error", lambda *_args, **_kwargs: oversized)

    result = CliRunner().invoke(app, ["show", "abc123", "--tail", "2", "--json"])

    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.RUNTIME.value
    assert "OUTSIDE_REQUESTED_WINDOW" not in result.output


@pytest.mark.parametrize("cursor", [None, "", 12])
def test_a_window_without_a_usable_continuation_cursor_is_refused(
    monkeypatch: pytest.MonkeyPatch, cursor: object
) -> None:
    response = _session(cursor=cursor)
    monkeypatch.setattr(show_cli, "rpc_call_or_error", lambda *_args, **_kwargs: response)

    result = CliRunner().invoke(app, ["show", "abc123", "--tail", "2", "--json"])

    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.RUNTIME.value
