from __future__ import annotations

import json

import pytest
from recall.cli.app import app
from typer.testing import CliRunner


def _response(state: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation_id": "owned-operation",
        "plan_id": "owned-plan",
        "status": state,
        "accepted": state == "accepted",
        "dry_run": False,
        "error": None,
    }


def test_storage_cli_waits_for_observed_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    states = iter(["accepted", "running", "succeeded"])

    def rpc(method, params, **kwargs):
        assert method == "recall.migrate_storage"
        assert kwargs["auto_fork"] is False
        assert 0 < kwargs["idle_timeout"] <= 5
        return _response(next(states))

    monkeypatch.setattr("recall.cli.daemon.rpc_call_or_error", rpc)
    result = CliRunner().invoke(app, ["daemon", "migrate-storage", "--wait", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "succeeded"


def test_storage_cli_deadline_is_not_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "recall.cli.daemon.rpc_call_or_error", lambda *a, **kw: _response("running")
    )
    result = CliRunner().invoke(
        app, ["daemon", "migrate-storage", "--wait", "--timeout", "0.01", "--json"]
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "timed_out"


def test_storage_cli_rejects_missing_operation_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("recall.cli.daemon.rpc_call_or_error", lambda *a, **kw: {})
    result = CliRunner().invoke(app, ["daemon", "migrate-storage", "--json"])
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "args", [["--timeout", "nan"], ["--timeout", "0"], ["--dry-run", "--wait"]]
)
def test_storage_cli_rejects_invalid_request_before_apply(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid input reached daemon")

    monkeypatch.setattr("recall.cli.daemon.rpc_call_or_error", unexpected)
    result = CliRunner().invoke(app, ["daemon", "migrate-storage", "--json", *args])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "VALIDATION"
