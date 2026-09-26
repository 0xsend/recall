"""The public reconciliation envelope remains actionable through CLI and fleet."""

from __future__ import annotations

import json

import pytest
from recall.cli.app import app
from recall.core.fleet import FleetHost
from recall.services.fleet_query import fleet_live
from recall.services.fleet_transport import FleetRemoteResult
from typer.testing import CliRunner


def _page():
    return {
        "schema_version": 2,
        "watching": False,
        "sessions": [{"id": "observed", "path": "/transcript", "liveness": "active"}],
        "next_cursor": "next-page",
        "coverage": {"complete": False, "unknown_count": 1},
    }


def test_live_preserves_coverage_continuation_and_projects_only_sessions(monkeypatch):
    from recall.cli import live

    calls = []

    def rpc(method, params):
        calls.append((method, params))
        return _page()

    monkeypatch.setattr(live, "rpc_call_or_error", rpc)
    result = CliRunner().invoke(app, ["live", "--json", "--cursor", "prior", "--fields", "id"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {**_page(), "sessions": [{"id": "observed"}]}
    assert calls[0][1]["cursor"] == "prior"


def test_status_forwards_coverage_cursor_and_exposes_projection(monkeypatch):
    from recall.cli import daemon

    calls = []
    status = {"reconciliation": {"source_page": [], "next_cursor": "next"}}

    def rpc(method, params, **kwargs):
        calls.append((method, params))
        return status

    monkeypatch.setattr(daemon, "rpc_call_or_error", rpc)
    result = CliRunner().invoke(
        app,
        [
            "daemon",
            "status",
            "--json",
            "--limit",
            "2",
            "--cursor",
            "prior",
            "--fields",
            "reconciliation",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == status
    assert calls == [("recall.daemon_status", {"limit": 2, "cursor": "prior"})]


def test_fleet_retains_per_host_coverage_and_continuation(monkeypatch):
    def remote(*args, **kwargs):
        return FleetRemoteResult(ok=True, stdout=json.dumps(_page()), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_query.run_remote", remote)
    result = fleet_live([FleetHost(name="edge", ssh="edge")])
    assert result.hosts_ok == 1
    assert result.rows[0]["id"] == "observed"
    assert result.coverage["complete"] is False
    host = result.coverage["hosts"][0]
    assert host["name"] == "edge"
    assert host["next_cursor"] == "next-page"
    assert host["coverage"]["unknown_count"] == 1
    assert host["watching"] is False


def test_legacy_fleet_array_cannot_claim_complete_coverage(monkeypatch):
    def remote(*args, **kwargs):
        return FleetRemoteResult(ok=True, stdout="[]", stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_query.run_remote", remote)
    result = fleet_live([FleetHost(name="legacy", ssh="legacy")])
    assert result.hosts_ok == 1
    assert result.coverage["complete"] is False
    assert result.coverage["hosts"][0]["coverage"] is None


def test_text_after_rewrite_names_the_cursor_reset(monkeypatch):
    from recall.cli import show

    response = {
        "id": "observed",
        "messages": [],
        "cursor": "new-epoch",
        "cursor_reset": True,
        "cursor_reset_reason": "content_rewritten",
        "freshness": {"current": True},
    }
    monkeypatch.setattr(show, "rpc_call_or_error", lambda *_args, **_kwargs: response)
    result = CliRunner().invoke(app, ["show", "observed", "--after", "prior", "--format", "text"])
    assert result.exit_code == 0, result.output
    assert "cursor reset" in result.stderr


def test_status_pagination_error_cannot_silently_return_a_local_first_page(monkeypatch):
    from recall.cli import daemon
    from recall.cli.contract import CliError, ErrorCode

    def fail_rpc(*args, **kwargs):
        raise CliError(code=ErrorCode.VALIDATION, message="bad cursor", exit_code=2)

    def forbidden_local_status():
        pytest.fail("pagination failure fell back to a different local result")

    monkeypatch.setattr(daemon, "rpc_call_or_error", fail_rpc)
    monkeypatch.setattr(daemon, "daemon_status", forbidden_local_status)
    result = CliRunner().invoke(app, ["daemon", "status", "--cursor", "bad", "--json"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["message"] == "bad cursor"


def test_fleet_global_truncation_and_invalid_host_output_are_explicit(monkeypatch):
    def remote(host, *args, **kwargs):
        page = {**_page(), "next_cursor": None, "coverage": {"complete": True}}
        if host.name == "invalid":
            page["coverage"] = None
        return FleetRemoteResult(ok=True, stdout=json.dumps(page), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_query.run_remote", remote)
    result = fleet_live(
        [FleetHost(name=name, ssh=name) for name in ("one", "two", "invalid")], limit=1
    )
    assert result.hosts_ok == 2
    assert result.hosts_failed == 1
    assert result.coverage["truncated"] is True
    assert result.coverage["received"] == 2
    assert result.coverage["returned"] == 1
    assert result.coverage["complete"] is False
    assert result.errors[0]["name"] == "invalid"
