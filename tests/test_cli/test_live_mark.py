"""`recall live mark` — the hook-facing write (REQ-LIVE-008).

This command runs inside a harness `SessionStart` hook, which is the only
reason its failure mode is unusual: a mark is enrichment, so a daemon that
cannot take it is not an error the agent's session should ever hear about.
Everything else about it is an ordinary structured command.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from recall.cli import live as live_cli
from recall.cli.app import app
from recall.cli.contract import CliError, ErrorCode
from typer.testing import CliRunner

MARKED = {
    "marked": True,
    "source": "claude_code",
    "source_session_id": "sess-1",
    "host": "laptop",
    "pid": 4242,
    "marked_at": "2026-09-08T10:00:00",
}


def _install_rpc(
    monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]
) -> list[tuple[str, dict[str, object], dict[str, object]]]:
    calls: list[tuple[str, dict[str, object], dict[str, object]]] = []

    def fake_rpc(method: str, params: dict[str, object], **kwargs: object) -> dict[str, Any]:
        calls.append((method, params, kwargs))
        return result

    monkeypatch.setattr(live_cli, "rpc_call_or_error", fake_rpc)
    return calls


def test_mark_forwards_the_session_and_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_rpc(monkeypatch, MARKED)

    result = CliRunner().invoke(
        app,
        [
            "live",
            "mark",
            "--session",
            "sess-1",
            "--pid",
            "4242",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[0][0] == "recall.live_mark"
    assert calls[0][1] == {"session": "sess-1", "pid": 4242}
    assert json.loads(result.stdout) == MARKED


def test_mark_forwards_an_explicit_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only sent when given: the daemon owns the default, so the CLI states no opinion."""
    calls = _install_rpc(monkeypatch, MARKED)

    result = CliRunner().invoke(
        app, ["live", "mark", "--session", "sess-1", "--pid", "4242", "--source", "codex", "--json"]
    )

    assert result.exit_code == 0, result.output
    assert calls[0][1] == {"session": "sess-1", "pid": 4242, "source": "codex"}


def test_mark_never_forks_a_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hook that started a daemon would put an index pass in the agent's startup path."""
    calls = _install_rpc(monkeypatch, MARKED)

    result = CliRunner().invoke(
        app, ["live", "mark", "--session", "sess-1", "--pid", "4242", "--json"]
    )

    assert result.exit_code == 0, result.output
    assert calls[0][2]["auto_fork"] is False


def test_mark_without_a_daemon_exits_zero_and_says_it_did_not_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enrichment is optional; failing the hook would fail the agent's session start."""

    def failing_rpc(method: str, params: dict[str, object], **_kwargs: object) -> dict[str, Any]:
        raise CliError(code=ErrorCode.RUNTIME, message="daemon unavailable", exit_code=1)

    monkeypatch.setattr(live_cli, "rpc_call_or_error", failing_rpc)

    result = CliRunner().invoke(
        app, ["live", "mark", "--session", "sess-1", "--pid", "4242", "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["marked"] is False
    assert payload["source_session_id"] == "sess-1"
    assert "daemon unavailable" in result.stderr


def test_a_non_positive_pid_is_a_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal the caller can fix is not the fail-open case; it never reaches the daemon."""
    calls = _install_rpc(monkeypatch, MARKED)

    result = CliRunner().invoke(
        app, ["live", "mark", "--session", "sess-1", "--pid", "0", "--json"]
    )

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.VALIDATION.value
    assert calls == []


def test_an_empty_session_id_is_a_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_rpc(monkeypatch, MARKED)

    result = CliRunner().invoke(app, ["live", "mark", "--session", "", "--pid", "42", "--json"])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.VALIDATION.value
    assert calls == []


def test_mark_travels_through_params(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_rpc(monkeypatch, MARKED)

    result = CliRunner().invoke(
        app,
        [
            "live",
            "mark",
            "--params",
            json.dumps({"session": "sess-1", "pid": 4242, "json": True}),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[0][1] == {"session": "sess-1", "pid": 4242}


def test_mark_text_output_names_what_it_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_rpc(monkeypatch, MARKED)

    result = CliRunner().invoke(
        app, ["live", "mark", "--session", "sess-1", "--pid", "4242", "--format", "text"]
    )

    assert result.exit_code == 0, result.output
    assert "sess-1" in result.stdout
    assert "4242" in result.stdout
