"""A daemon without session-filter support must never return an unfiltered roster."""

from types import SimpleNamespace

import pytest
from recall.cli import rpc
from recall.cli.app import app
from typer.testing import CliRunner


@pytest.mark.parametrize("command", ["list", "live"])
@pytest.mark.parametrize("selector", [["--session", "requested"], ["--params", '{"session":""}']])
def test_session_filter_refuses_a_daemon_without_filter_methods(monkeypatch, command, selector):
    async def legacy_roster(params, client):
        rows = [{"id": "unrelated", "source_session_id": "unrelated"}]
        return rows if command == "list" else {"sessions": rows, "watching": False}

    monkeypatch.setattr(
        rpc,
        "_in_process_server",
        SimpleNamespace(_methods={f"recall.{command}": legacy_roster}),
    )
    result = CliRunner().invoke(app, [command, *selector, "--json"])

    assert result.exit_code == 1
    assert "unrelated" not in result.stdout
    assert "method not found" in result.output
