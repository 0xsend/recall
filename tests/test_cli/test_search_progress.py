"""Search diagnostics preserve structured stdout (REQ-CLI-025)."""

from __future__ import annotations

import json

import pytest
from recall.cli import search as search_cli
from recall.cli.app import app
from recall.cli.contract import CliError, ErrorCode
from typer.testing import CliRunner


@pytest.mark.parametrize("format_name", ["text", "json", "jsonl", "toon"])
def test_search_progress_stays_on_stderr_in_every_format(monkeypatch, format_name):
    monkeypatch.setattr(search_cli, "fetch_daemon_status", lambda _config: None)
    enabled = False

    def call(_method, _params, *, on_notification):
        if enabled:
            on_notification(
                "search_progress", {"attempt": 1, "state": "waiting", "preference_limit_ms": 5000}
            )
            on_notification(
                "search_progress",
                {
                    "attempt": 1,
                    "state": "released",
                    "wait_ms": 5000,
                    "reason": "preference_elapsed",
                },
            )
            on_notification(
                "search_progress", {"attempt": 2, "state": "waiting", "preference_limit_ms": 5000}
            )
            on_notification(
                "search_progress",
                {"attempt": 2, "state": "released", "wait_ms": 123, "reason": "drain_idle"},
            )
        return []

    monkeypatch.setattr(search_cli, "rpc_call_or_error", call)
    runner = CliRunner()
    args = ["search", "test", "--format", format_name]
    baseline = runner.invoke(app, args)
    enabled = True
    result = runner.invoke(app, args)
    assert result.exit_code == baseline.exit_code == 0
    assert result.stdout == baseline.stdout
    assert result.stderr.startswith(
        "search: waiting for embedding drain (preference up to 5s; request deadline 30s)\n"
        "search: embedding preference ended after 5.00s (limit reached); joining scan admission\n"
        "search (attempt 2): waiting for embedding drain "
        "(preference up to 5s; request deadline 30s)\n"
        "search (attempt 2): embedding preference ended after 0.12s "
        "(drain idle); joining scan admission\n"
    )
    if format_name == "json":
        assert json.loads(result.stdout) == []


def test_search_timeout_details_reach_structured_cli_error(monkeypatch):
    data = {"kind": "search_timeout", "phase": "scan", "largest_phase": "scan"}

    def call(*_args, **_kwargs):
        raise CliError(
            code=ErrorCode.RUNTIME,
            message="recall.search timed out after 30 seconds (phase: scan; largest phase: scan)",
            details=data,
            exit_code=1,
        )

    monkeypatch.setattr(search_cli, "rpc_call_or_error", call)
    result = CliRunner().invoke(app, ["search", "test", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["details"] == data
    assert "phase: scan" in json.loads(result.stdout)["error"]["message"]


def test_unknown_notifications_are_ignored(capsys):
    search_cli._render_search_progress("future_notification", {"state": "waiting"})
    search_cli._render_search_progress("search_progress", {"attempt": 1, "state": "future_state"})
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
