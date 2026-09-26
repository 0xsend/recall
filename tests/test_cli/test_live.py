"""`recall live` output and its RPC contract (REQ-LIVE-002/009).

The derivations and the join are proved against a real database in
`tests/test_services/`; these tests cover what the command adds — parameter
forwarding, the structured contract, and the one thing the CLI must never do,
which is let an unwatched daemon read as "nothing is running".
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from recall.cli import fleet as fleet_cli
from recall.cli import live as live_cli
from recall.cli.app import app
from recall.cli.contract import CliError, ErrorCode
from typer.testing import CliRunner


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": "/home/a/.claude/projects/p/s.jsonl",
        "liveness": "active",
        "freshness": {
            "file_mtime": 1_700_000_060.0,
            "file_size": 4096,
            "indexed_mtime": 1_700_000_000.0,
            "indexed_size": 2048,
            "lag_seconds": 60.0,
            "lag_bytes": 2048,
            "current": False,
        },
        "turn": {
            "state": "working",
            "last_user_at": "2026-09-08T12:00:00",
            "last_assistant_at": "2026-09-08T12:00:04",
            "last_user_text": "run the tests",
            "last_assistant_text": "on it",
            "running_tool": {
                "name": "Bash",
                "summary": "pytest -q",
                "started_at": "2026-09-08T12:00:04",
            },
            "subagents_active": 0,
            "stop_reason": "tool_use",
        },
        "id": "abc123",
        "source": "claude_code",
        "source_session_id": "uuid-1",
        "host": "laptop",
        "cwd": "/home/a/work",
        "git_repo": "/home/a/work/recall",
        "git_branch": "main",
        "model": "claude-opus-5",
        "last_activity_at": "2026-09-08T12:00:04",
        "cursor": "djE6YWJjMTIzOjM",
    }
    row.update(overrides)
    return row


def _envelope(sessions, *, watching=True):
    return {
        "schema_version": 2,
        "watching": watching,
        "sessions": sessions,
        "next_cursor": None,
        "coverage": {"complete": True},
    }


def _install_rpc(
    monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]
) -> list[tuple[str, dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_rpc(method: str, params: dict[str, object], **_kwargs: object) -> dict[str, Any]:
        calls.append((method, params))
        return {**_envelope([]), **result}

    monkeypatch.setattr(live_cli, "rpc_call_or_error", fake_rpc)
    return calls


def test_live_emits_the_rows_the_daemon_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_rpc(monkeypatch, {"watching": True, "sessions": [_row()]})

    result = CliRunner().invoke(app, ["live", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == _envelope([_row()])


def test_live_forwards_every_filter_to_the_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_rpc(monkeypatch, {"watching": True, "sessions": []})

    result = CliRunner().invoke(
        app,
        [
            "live",
            "--all",
            "--source",
            "codex",
            "--project",
            "recall",
            "--host",
            "laptop",
            "--limit",
            "10",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            "recall.live",
            {
                "limit": 10,
                "all": True,
                "source": "codex",
                "project": "recall",
                "host": "laptop",
            },
        )
    ]


def test_an_unwatched_daemon_says_so_instead_of_reading_as_nothing_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty list from a daemon with no live set is not evidence of no agents."""
    _install_rpc(monkeypatch, {"watching": False, "sessions": []})

    result = CliRunner().invoke(app, ["live", "--json"])

    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout) == _envelope([], watching=False)
    assert "not running a live watcher" in result.stderr


def test_a_watching_daemon_adds_no_note(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_rpc(monkeypatch, {"watching": True, "sessions": []})

    result = CliRunner().invoke(app, ["live", "--json"])

    assert result.exit_code == 0, result.stdout
    assert result.stderr == ""


def test_text_output_names_the_turn_and_the_running_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_rpc(monkeypatch, {"watching": True, "sessions": [_row()]})

    result = CliRunner().invoke(app, ["live", "--format", "text"])

    assert result.exit_code == 0, result.stdout
    assert "active" in result.stdout
    assert "turn=working" in result.stdout
    assert "tool=Bash" in result.stdout
    assert "lag=2048" in result.stdout


def test_text_output_says_so_when_nothing_is_live(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_rpc(monkeypatch, {"watching": True, "sessions": []})

    result = CliRunner().invoke(app, ["live", "--format", "text"])

    assert result.exit_code == 0, result.stdout
    assert "No matching sessions on this page." in result.stdout


def test_fields_projects_the_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_rpc(monkeypatch, {"watching": True, "sessions": [_row()]})

    result = CliRunner().invoke(app, ["live", "--json", "--fields", "id,liveness"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == _envelope([{"id": "abc123", "liveness": "active"}])


def test_an_unknown_field_is_a_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_rpc(monkeypatch, {"watching": True, "sessions": [_row()]})

    result = CliRunner().invoke(app, ["live", "--json", "--fields", "id,nope"])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.VALIDATION.value


def test_a_non_positive_limit_is_a_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_rpc(monkeypatch, {"watching": True, "sessions": []})

    result = CliRunner().invoke(app, ["live", "--json", "--limit", "0"])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.VALIDATION.value


def test_an_unknown_source_is_a_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_rpc(monkeypatch, {"watching": True, "sessions": []})

    result = CliRunner().invoke(app, ["live", "--json", "--source", "nope"])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.VALIDATION.value


def test_params_carries_the_same_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_rpc(monkeypatch, {"watching": True, "sessions": []})

    result = CliRunner().invoke(
        app, ["live", "--params", json.dumps({"all": True, "limit": 5, "json": True})]
    )

    assert result.exit_code == 0, result.output
    assert calls == [("recall.live", {"limit": 5, "all": True})]


def test_a_daemon_error_surfaces_as_the_cli_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_rpc(method: str, params: dict[str, object], **_kwargs: object) -> dict[str, Any]:
        raise CliError(code=ErrorCode.RUNTIME, message="daemon unavailable", exit_code=1)

    monkeypatch.setattr(live_cli, "rpc_call_or_error", failing_rpc)

    result = CliRunner().invoke(app, ["live", "--json"])

    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.RUNTIME.value


def _stale(**overrides: Any) -> dict[str, Any]:
    freshness_override = overrides.pop("freshness", None)
    row = _row(**overrides)
    row["freshness"] = {**row["freshness"], "lag_bytes": 2048, "current": False}
    if freshness_override is not None:
        row["freshness"] = {**row["freshness"], **freshness_override}
    return row


def _current(**overrides: Any) -> dict[str, Any]:
    row = _row(**overrides)
    row["freshness"] = {**row["freshness"], "lag_seconds": 0.0, "lag_bytes": 0, "current": True}
    return row


def test_fresh_is_forwarded_to_the_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-LIVE-003: only the daemon may index, so `--fresh` is its job, not the CLI's."""
    calls = _install_rpc(monkeypatch, {"watching": True, "sessions": []})

    result = CliRunner().invoke(app, ["live", "--fresh", "--json"])

    assert result.exit_code == 0, result.output
    assert calls == [("recall.live", {"limit": 50, "all": False, "fresh": True})]


def test_fresh_says_which_rows_did_not_catch_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `--fresh` answer that is still behind must not read as up to date."""
    _install_rpc(monkeypatch, {"watching": True, "sessions": [_current(), _stale(id="behind")]})

    result = CliRunner().invoke(app, ["live", "--fresh", "--json"])

    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)["sessions"]) == 2
    assert "1 of 2" in result.stderr
    assert "still behind" in result.stderr


def test_fresh_adds_no_note_when_every_row_caught_up(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_rpc(monkeypatch, {"watching": True, "sessions": [_current()]})

    result = CliRunner().invoke(app, ["live", "--fresh", "--json"])

    assert result.exit_code == 0, result.output
    assert result.stderr == ""


def test_fresh_names_never_indexed_rows_without_claiming_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`live --fresh` did not fail; those rows were never eligible to catch up."""
    _install_rpc(
        monkeypatch,
        {
            "watching": True,
            "sessions": [
                _stale(
                    id=None,
                    freshness={
                        "lag_bytes": None,
                        "current": False,
                        "limitations": ["not_yet_indexed"],
                    },
                )
            ],
        },
    )

    result = CliRunner().invoke(app, ["live", "--fresh", "--json"])

    assert result.exit_code == 0, result.output
    assert "not yet indexed" in result.stderr
    assert "fresh_timeout" not in result.stderr


def test_a_stale_row_without_fresh_is_not_worth_a_note(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lag is the normal state of an unrefreshed read; only `--fresh` promised otherwise."""
    _install_rpc(monkeypatch, {"watching": True, "sessions": [_stale()]})

    result = CliRunner().invoke(app, ["live", "--json"])

    assert result.exit_code == 0, result.output
    assert result.stderr == ""


def test_fresh_fails_closed_when_no_daemon_can_serve_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LIVE-003: without a daemon there is no fresh answer, so there is no answer."""

    def failing_rpc(method: str, params: dict[str, object], **_kwargs: object) -> dict[str, Any]:
        raise CliError(code=ErrorCode.RUNTIME, message="daemon unavailable", exit_code=1)

    monkeypatch.setattr(live_cli, "rpc_call_or_error", failing_rpc)

    result = CliRunner().invoke(app, ["live", "--fresh", "--json"])

    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.RUNTIME.value


def test_fresh_travels_through_params(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_rpc(monkeypatch, {"watching": True, "sessions": []})

    result = CliRunner().invoke(
        app, ["live", "--params", json.dumps({"fresh": True, "json": True})]
    )

    assert result.exit_code == 0, result.output
    assert calls == [("recall.live", {"limit": 50, "all": False, "fresh": True})]


def _install_fleet(
    monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]
) -> list[dict[str, object]]:
    """Capture what the command asks the fan-out for, in place of SSH."""
    calls: list[dict[str, object]] = []

    def fake_fan_out(**kwargs: object) -> dict[str, Any]:
        calls.append(kwargs)
        return _envelope(rows, watching=None)

    monkeypatch.setattr(fleet_cli, "run_fleet_live", fake_fan_out)
    return calls


def test_fleet_asks_the_fan_out_and_never_the_local_daemon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LIVE-007: `--fleet` answers about every host, so the local RPC is not the source."""
    rpc_calls = _install_rpc(monkeypatch, {"watching": True, "sessions": [_row()]})
    _install_fleet(monkeypatch, [_row(id="remote1", host="devbox")])

    result = CliRunner().invoke(app, ["live", "--fleet", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == _envelope(
        [_row(id="remote1", host="devbox")], watching=None
    )
    assert rpc_calls == []


def test_fleet_forwards_every_filter_to_the_fan_out(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fleet(monkeypatch, [])

    result = CliRunner().invoke(
        app,
        [
            "live",
            "--fleet",
            "--all",
            "--source",
            "codex",
            "--project",
            "recall",
            "--host",
            "devbox",
            "--limit",
            "10",
            "--fleet-config",
            "/tmp/fleet.toml",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "include_idle": True,
            "source": "codex",
            "project": "recall",
            "host": "devbox",
            "limit": 10,
            "fleet_config": "/tmp/fleet.toml",
        }
    ]


def test_fleet_does_not_claim_the_local_daemon_is_unwatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The combined roster does not substitute local observation for remote evidence."""
    _install_fleet(monkeypatch, [])

    result = CliRunner().invoke(app, ["live", "--fleet", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == _envelope([], watching=None)
    assert "not running a live watcher" not in result.stderr


def test_fresh_with_fleet_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """No remote is asked to index, so a `--fresh --fleet` answer would be stale by construction."""
    calls = _install_fleet(monkeypatch, [])

    result = CliRunner().invoke(app, ["live", "--fleet", "--fresh", "--json"])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.VALIDATION.value
    assert calls == []


def test_fleet_travels_through_params(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fleet(monkeypatch, [])

    result = CliRunner().invoke(
        app, ["live", "--params", json.dumps({"fleet": True, "limit": 7, "json": True})]
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "include_idle": False,
            "source": None,
            "project": None,
            "host": None,
            "limit": 7,
            "fleet_config": None,
        }
    ]


def test_a_fan_out_failure_surfaces_as_the_cli_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_fan_out(**_kwargs: object) -> list[dict[str, Any]]:
        raise CliError(code=ErrorCode.RUNTIME, message="fleet live: all hosts failed", exit_code=1)

    monkeypatch.setattr(fleet_cli, "run_fleet_live", failing_fan_out)

    result = CliRunner().invoke(app, ["live", "--fleet", "--json"])

    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["error"]["code"] == ErrorCode.RUNTIME.value
