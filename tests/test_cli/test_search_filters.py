"""CLI, fleet decoder, and manifest tests for PR 3 search filters."""

from __future__ import annotations

import json

import pytest
from recall.cli.app import app
from recall.core.fleet import FleetHost
from recall.core.types import Role
from recall.services.fleet_query import _decode_filtered_search_array, fleet_search
from recall.services.fleet_transport import FleetRemoteResult
from typer.testing import CliRunner

from tests.fleet_support import inventory_executor

runner = CliRunner()


def test_cli_search_rejects_role_and_tool_locally_before_dispatch(monkeypatch) -> None:
    called = False

    def fail_rpc(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("RPC should not be called")

    def fail_fleet(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("fleet should not be called")

    monkeypatch.setattr("recall.cli.search.rpc_call_or_error", fail_rpc)
    monkeypatch.setattr("recall.cli.fleet.run_fleet_search", fail_fleet)

    for argv in (
        ["search", "needle", "--role", "user", "--tool", "Bash", "--json"],
        ["search", "needle", "--role", "user", "--tool", "", "--json"],
        ["search", "needle", "--role", "user", "--tool", "Bash", "--fleet", "--json"],
        ["search", "needle", "--role", "bogus", "--json"],
        ["search", "needle", "--per-session", "0", "--json"],
        ["search", "needle", "--since", "", "--json"],
        ["search", "needle", "--project", "   ", "--json"],
        ["search", "needle", "--host", "", "--json"],
        ["search", "needle", "--params", '{"query":"needle","per_session":true}', "--json"],
    ):
        res = runner.invoke(app, argv)
        assert res.exit_code == 2, (argv, res.output)
        assert not called


def test_cli_search_routes_unfiltered_vs_filtered_rpc_methods(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_rpc(method: str, params: dict[str, object], **_kwargs):
        calls.append((method, dict(params)))
        return []

    monkeypatch.setattr("recall.cli.search.rpc_call_or_error", fake_rpc)
    monkeypatch.setattr("recall.cli.search.fetch_daemon_status", lambda *_a, **_k: None)

    res1 = runner.invoke(app, ["search", "needle", "--json"])
    assert res1.exit_code == 0, res1.output
    assert calls[-1][0] == "recall.search"

    res2 = runner.invoke(
        app,
        [
            "search",
            "needle",
            "--role",
            "user",
            "--role",
            "assistant",
            "--project",
            "recall",
            "--since",
            "7d",
            "--host",
            "aem5",
            "--per-session",
            "2",
            "--json",
        ],
    )
    assert res2.exit_code == 0, res2.output
    method, params = calls[-1]
    assert method == "recall.search_filtered"
    assert params["role"] == ["user", "assistant"]
    assert params["project"] == "recall"
    assert params["since"] == "7d"
    assert params["host"] == "aem5"
    assert params["per_session"] == 2

    res3 = runner.invoke(
        app,
        [
            "search",
            "--params",
            json.dumps(
                {
                    "query": "needle",
                    "role": ["user"],
                    "project": "recall",
                    "since": "7d",
                    "host": "aem5",
                    "per_session": 1,
                }
            ),
            "--json",
        ],
    )
    assert res3.exit_code == 0, res3.output
    method3, params3 = calls[-1]
    assert method3 == "recall.search_filtered"
    assert params3["role"] == ["user"]
    assert params3["per_session"] == 1


def test_cli_search_tool_probe_preserves_filters(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_rpc(method: str, params: dict[str, object], **_kwargs):
        calls.append((method, dict(params)))
        return []

    monkeypatch.setattr("recall.cli.search.rpc_call_or_error", fake_rpc)
    monkeypatch.setattr("recall.cli.search.fetch_daemon_status", lambda *_a, **_k: None)

    res = runner.invoke(
        app,
        [
            "search",
            "needle",
            "--tool",
            "Bash",
            "--project",
            "recall",
            "--since",
            "7d",
            "--host",
            "aem5",
            "--per-session",
            "1",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    assert len(calls) == 2
    assert calls[0][0] == "recall.search_filtered"
    assert calls[0][1]["tool"] == "Bash"
    assert calls[1][0] == "recall.search_filtered"
    assert "tool" not in calls[1][1]
    assert calls[1][1]["project"] == "recall"
    assert calls[1][1]["since"] == "7d"
    assert calls[1][1]["host"] == "aem5"
    assert calls[1][1]["per_session"] == 1


def test_decode_filtered_search_array_rejects_malformed_and_violating_rows() -> None:
    valid_row = {
        "kind": "message",
        "session_id": "s1",
        "role": "user",
        "score": 1.25,
        "host": "aem5",
    }
    decoded = _decode_filtered_search_array(
        [valid_row], expected_roles=frozenset({"user"}), per_session=1
    )
    assert decoded[0]["score"] == 1.25

    # Rejects role violation
    with pytest.raises(ValueError, match="--role"):
        _decode_filtered_search_array(
            [{**valid_row, "role": "system"}],
            expected_roles=frozenset({"user"}),
            per_session=None,
        )

    # Rejects tool_call when role filter set
    with pytest.raises(ValueError, match="--role"):
        _decode_filtered_search_array(
            [{**valid_row, "kind": "tool_call", "role": None}],
            expected_roles=frozenset({"user"}),
            per_session=None,
        )

    # Rejects non-string session_id
    with pytest.raises(ValueError, match="session_id"):
        _decode_filtered_search_array(
            [{**valid_row, "session_id": 123}],
            expected_roles=None,
            per_session=1,
        )

    # Rejects huge integer overflowing float (OverflowError -> ValueError)
    with pytest.raises(ValueError, match="finite number"):
        _decode_filtered_search_array(
            [{**valid_row, "score": 10**400}],
            expected_roles=None,
            per_session=None,
        )

    # Rejects bool / NaN / Inf score
    for bad_score in (True, float("nan"), float("inf"), "1.0"):
        with pytest.raises(ValueError, match="finite"):
            _decode_filtered_search_array(
                [{**valid_row, "score": bad_score}],
                expected_roles=None,
                per_session=None,
            )

    # Rejects per-session cap violation from a misbehaving peer
    with pytest.raises(ValueError, match="--per-session"):
        _decode_filtered_search_array(
            [valid_row, {**valid_row, "score": 0.9}],
            expected_roles=None,
            per_session=1,
        )

    # Rejects unhashable kind/role with ValueError (never TypeError)
    with pytest.raises(ValueError, match="kind"):
        _decode_filtered_search_array(
            [{**valid_row, "kind": ["message"]}],
            expected_roles=None,
            per_session=None,
        )
    with pytest.raises(ValueError, match="--role"):
        _decode_filtered_search_array(
            [{**valid_row, "role": ["user"]}],
            expected_roles=frozenset({"user"}),
            per_session=None,
        )


def test_fleet_search_skips_unhashable_malformed_peer_and_keeps_healthy_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding #1: malformed list-typed kind/role skips only the bad peer, not the whole query."""
    hosts = (
        FleetHost(name="bad-kind", ssh="bad-kind"),
        FleetHost(name="bad-role", ssh="bad-role"),
        FleetHost(name="good", ssh="good"),
    )

    def fake_run(host: FleetHost, remote_argv: list[str], **_k: object) -> FleetRemoteResult:
        if host.name == "bad-kind":
            payload = [{"kind": ["message"], "session_id": "s1", "role": "user", "score": 1.0}]
        elif host.name == "bad-role":
            payload = [{"kind": "message", "session_id": "s1", "role": ["user"], "score": 1.0}]
        else:
            payload = [
                {
                    "kind": "message",
                    "session_id": "s-ok",
                    "source": "codex",
                    "role": "user",
                    "score": 0.75,
                    "host": "good",
                }
            ]
        return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", fake_run)
    result = fleet_search(
        inventory_executor(hosts),
        query="needle",
        roles=(Role.USER,),
        limit=5,
    )
    assert result.hosts_ok == 1
    assert result.hosts_failed == 2
    assert len(result.rows) == 1
    assert result.rows[0]["session_id"] == "s-ok"


def test_cli_search_filtered_older_daemon_error_includes_upgrade_hint(monkeypatch) -> None:
    from recall.cli.contract import CliError, ErrorCode

    def fail_old_daemon(method: str, *_args, **_kwargs):
        assert method == "recall.search_filtered"
        raise CliError(
            code=ErrorCode.RUNTIME,
            message="method not found: recall.search_filtered",
            exit_code=1,
        )

    monkeypatch.setattr("recall.cli.search.rpc_call_or_error", fail_old_daemon)
    res = runner.invoke(app, ["search", "needle", "--role", "user", "--json"])
    assert res.exit_code == 1
    err = json.loads(res.stdout)["error"]
    assert "recall daemon restart" in err["message"]
    assert "method not found: recall.search_filtered" in err["message"]


def test_fleet_search_filtered_per_session_keys_by_endpoint_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosts = (
        FleetHost(name="h1", ssh="h1"),
        FleetHost(name="h2", ssh="h2"),
        FleetHost(name="old", ssh="old"),
    )
    seen_argv: dict[str, list[str]] = {}

    def fake_run(host: FleetHost, remote_argv: list[str], **_k: object) -> FleetRemoteResult:
        seen_argv[host.name] = list(remote_argv)
        if host.name == "old":
            # Older peer rejects unknown --role flag over SSH CLI
            return FleetRemoteResult(
                ok=False,
                stdout="",
                stderr="No such option: --role",
                returncode=2,
                error="No such option: --role",
            )
        # Both h1 and h2 happen to share session_id "same-sid" (REQ-FLEET-MERGE-006)
        payload = [
            {
                "kind": "message",
                "session_id": "same-sid",
                "source": "codex",
                "role": "user",
                "score": 0.9 if host.name == "h1" else 0.8,
                "host": host.name,
            }
        ]
        return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", fake_run)
    result = fleet_search(
        inventory_executor(hosts),
        query="needle",
        roles=(Role.USER,),
        project="recall",
        since="7d",
        per_session=1,
        limit=5,
    )
    assert result.hosts_ok == 2
    assert result.hosts_failed == 1
    assert len(result.rows) == 2
    assert [row["host"] for row in result.rows] == ["h1", "h2"]
    assert "--role" in seen_argv["h1"]
    assert "--project" in seen_argv["h1"]
    assert "--since" in seen_argv["h1"]
    assert "--per-session" in seen_argv["h1"]


def test_cli_search_manifest_includes_pr3_filter_options() -> None:
    res = runner.invoke(app, ["schema", "search"])
    assert res.exit_code == 0, res.output
    schema = json.loads(res.stdout)
    option_map = {opt["name"]: opt for opt in schema["options"]}
    assert option_map["role"]["type"] == "string[]"
    assert option_map["role"]["enum"] == ["user", "assistant", "system"]
    assert option_map["project"]["type"] == "string"
    assert option_map["since"]["type"] == "string"
    assert option_map["host"]["type"] == "string"
    assert option_map["per_session"]["type"] == "integer"
    assert option_map["per_session"]["minimum"] == 1
