"""CLI contract tests for PR 5 stats filtering, fleet fan-out, and fail-closed RPC hints."""

from __future__ import annotations

import json
from typing import Any

import pytest
from recall.cli.app import app
from recall.cli.contract import CliError, ErrorCode
from typer.testing import CliRunner


@pytest.mark.parametrize(
    ("argv", "expected_method", "expected_params", "fake_reply"),
    [
        (
            ["stats", "--source", "agy", "--json"],
            "recall.stats_filtered",
            {"source": "antigravity_cli"},
            {
                "sessions": 3,
                "messages": 12,
                "tool_calls": 4,
                "bash_calls": 1,
                "last_index_total": None,
                "last_index_indexed": None,
                "last_index_skipped": None,
                "last_index_failed": None,
                "last_context_messages": None,
                "last_context_mode": None,
                "last_context_input_tokens": None,
                "last_context_output_tokens": None,
                "last_context_model": None,
            },
        ),
        (
            ["stats", "tools", "--source", "agy", "--json"],
            "recall.stats_tools_filtered",
            {"source": "antigravity_cli"},
            [{"tool_name": "read_file", "count": 5}],
        ),
        (
            ["stats", "bash", "--source", "codex", "--suggest", "--json"],
            "recall.stats_bash_filtered",
            {"source": "codex", "suggest": True},
            {"suggestions": [], "skipped": []},
        ),
        (
            ["stats", "tokens", "--source", "claude-code", "--json"],
            "recall.stats_tokens_filtered",
            {"source": "claude_code"},
            [{"repo": "org/repo", "input_tokens": 10, "output_tokens": 5}],
        ),
        (
            ["stats", "usage", "--source", "agy", "--json"],
            "recall.stats_usage_filtered",
            {"source": "antigravity_cli"},
            [
                {
                    "source": "antigravity_cli",
                    "model": "gemini-2.5",
                    "host": "aem5",
                    "input_tokens": 100,
                    "cached_input_tokens": 0,
                    "output_tokens": 10,
                    "fresh_input_tokens": None,
                    "session_count": 1,
                }
            ],
        ),
        (
            ["stats", "usage", "--since", "2026-05-01T00:00:00Z", "--json"],
            "recall.stats_usage",
            {"since": "2026-05-01T00:00:00Z"},
            [],
        ),
    ],
)
def test_stats_cli_dispatches_versioned_or_legacy_rpc(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_method: str,
    expected_params: dict[str, Any],
    fake_reply: Any,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_rpc(method: str, params: dict[str, Any] | None = None, **_kw: Any) -> Any:
        calls.append((method, dict(params or {})))
        return fake_reply

    monkeypatch.setattr("recall.cli.stats.rpc_call_or_error", fake_rpc)
    monkeypatch.setattr("recall.cli.stats._emit_status_notice", lambda _fmt: None)

    runner = CliRunner()
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
    assert calls == [(expected_method, expected_params)]
    assert json.loads(result.stdout) == fake_reply


def test_stats_filtered_text_omits_last_index_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "recall.cli.stats.rpc_call_or_error",
        lambda _m, _p=None, **_kw: {
            "sessions": 2,
            "messages": 6,
            "tool_calls": 3,
            "bash_calls": 1,
            "last_index_indexed": None,
        },
    )
    monkeypatch.setattr("recall.cli.stats._emit_status_notice", lambda _fmt: None)
    runner = CliRunner()
    res = runner.invoke(app, ["stats", "--source", "agy", "--format", "text"])
    assert res.exit_code == 0, res.output
    assert "Sessions: 2" in res.stdout
    assert "Last index run:" not in res.stdout


def test_stats_filtered_method_not_found_emits_daemon_restart_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_rpc(method: str, _params: dict[str, Any] | None = None, **_kw: Any) -> Any:
        raise CliError(code=ErrorCode.RUNTIME, message=f"method not found: {method}", exit_code=1)

    monkeypatch.setattr("recall.cli.stats.rpc_call_or_error", fake_rpc)
    runner = CliRunner()
    res = runner.invoke(app, ["stats", "tools", "--source", "agy", "--json"])
    assert res.exit_code == 1
    payload = json.loads(res.stdout)
    assert "recall daemon restart" in payload["error"]["message"]


def test_stats_fleet_local_only_old_daemon_emits_restart_hint_on_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from recall.core.rpc_client import RpcCallError
    from recall.core.rpc_types import METHOD_NOT_FOUND

    empty_inventory = tmp_path / "fleet.toml"
    empty_inventory.write_text("", encoding="utf-8")

    def fake_local_call(_self: Any, method: str, _params: dict[str, Any], **_kw: Any) -> Any:
        raise RpcCallError(METHOD_NOT_FOUND, f"method not found: {method}")

    monkeypatch.setattr("recall.core.read_scope.AggregateRead.call", fake_local_call)
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "stats",
            "--source",
            "agy",
            "--fleet",
            "--fleet-config",
            str(empty_inventory),
            "--json",
        ],
    )
    assert res.exit_code == 1
    assert "recall daemon restart" in res.stderr
    payload = json.loads(res.stdout)
    assert payload["error"]["code"] == "RUNTIME"
    assert "data" not in payload


@pytest.mark.parametrize(
    "argv",
    [
        ["stats", "--source", "agy", "tools", "--json"],
        ["stats", "--source", "", "tools", "--json"],
        ["stats", "--fleet-inventory-only", "tools", "--json"],
        ["stats", "--source", "agy", "tools", "--source", "codex", "--json"],
        ["stats", "--since", "7d", "usage", "--json"],
        ["stats", "--fleet", "bash", "--suggest", "--json"],
    ],
)
def test_stats_cli_rejects_root_options_before_subcommand(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("RPC must not be called when options precede subcommand")

    monkeypatch.setattr("recall.cli.stats.rpc_call_or_error", boom)
    runner = CliRunner()
    res = runner.invoke(app, argv)
    assert res.exit_code == 2
    assert "VALIDATION" in (res.stdout + res.stderr)
    assert "after `recall stats" in (res.stdout + res.stderr)


@pytest.mark.parametrize(
    "argv",
    [
        ["stats", "--source", "", "--json"],
        ["stats", "--since", "   ", "--json"],
        ["stats", "--since", "9999999999d", "--json"],
        ["stats", "--since", "0001-01-01T00:00:00+01:00", "--json"],
        ["stats", "--fleet-inventory-only", "--json"],
        ["stats", "tools", "--source", "bogus", "--json"],
        ["stats", "bash", "--fleet-inventory-only", "--json"],
        ["stats", "tokens", "--since", "9999999999d", "--json"],
        ["stats", "usage", "--source", "   ", "--json"],
        ["stats", "usage", "--since", "", "--json"],
    ],
)
def test_stats_cli_rejects_invalid_filters_pre_dispatch(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("RPC must not be called for invalid filters")

    monkeypatch.setattr("recall.cli.stats.rpc_call_or_error", boom)
    runner = CliRunner()
    res = runner.invoke(app, argv)
    assert res.exit_code == 2
    payload = json.loads(res.stdout)
    assert payload["error"]["code"] == "VALIDATION"


def test_manifest_exposes_stats_source_since_and_fleet_options() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["--llms"])
    assert res.exit_code == 0
    manifest = json.loads(res.stdout)
    by_cmd = manifest["commands"]
    for cmd in ("stats", "stats tools", "stats bash", "stats tokens", "stats usage"):
        option_names = {opt["name"] for opt in by_cmd[cmd]["options"]}
        assert {"source", "since", "fleet", "fleet_inventory_only", "fleet_config"}.issubset(
            option_names
        )
