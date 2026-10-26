from __future__ import annotations

import json
from pathlib import Path

import pytest
from recall.cli import stats as stats_cli
from recall.cli.app import app
from recall.cli.contract import CliError, ErrorCode
from recall.cli.manifest import command_manifest
from typer.testing import CliRunner


def _payload(scope: str = "local") -> dict[str, object]:
    return {
        "rows": [
            {
                "skill_name": "engineering-practices:code-law",
                "source": "codex",
                "host": "control",
                "invocations": 2,
                "sessions": 1,
            }
        ],
        "coverage": {
            "scope": scope,
            "expected_hosts": ["control"],
            "successful_hosts": ["control"],
            "covered_sources": ["codex", "grok"],
            "considered_sessions": 3,
            "attributed_invocations": 2,
            "unattributed_candidates": 1,
            "control": {
                "considered_sessions": 7,
                "attributed_invocations": 4,
                "unattributed_candidates": 2,
            },
        },
    }


def test_stats_skills_local_forwards_repeatable_sources_and_since(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path))
    calls: list[tuple[str, dict[str, object], float | None]] = []

    def fake_rpc(
        method: str,
        params: dict[str, object],
        *,
        idle_timeout: float | None = None,
    ) -> dict[str, object]:
        calls.append((method, params, idle_timeout))
        return _payload()

    monkeypatch.setattr(stats_cli, "rpc_call_or_error", fake_rpc)

    result = CliRunner().invoke(
        app,
        [
            "stats",
            "skills",
            "--local",
            "--source",
            "codex",
            "--source",
            "grok",
            "--since",
            "7d",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == _payload()
    assert calls == [
        (
            "recall.stats_skills",
            {"since": "7d", "source": ["codex", "grok"]},
            600.0,
        )
    ]


def test_stats_skills_params_accept_source_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path))
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        stats_cli,
        "rpc_call_or_error",
        lambda _method, params, **_kwargs: captured.append(params) or _payload(),
    )

    result = CliRunner().invoke(
        app,
        [
            "stats",
            "skills",
            "--params",
            '{"local":true,"source":["codex"],"since":"30d","format":"json"}',
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == [{"since": "30d", "source": ["codex"]}]


def test_stats_skills_defaults_to_local_plus_fleet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path))
    local = _payload()
    merged = _payload("local+fleet")
    monkeypatch.setattr(stats_cli, "rpc_call_or_error", lambda *_args, **_kwargs: local)

    calls: list[dict[str, object]] = []

    def fake_fleet(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return merged

    monkeypatch.setattr("recall.cli.fleet.run_fleet_stats_skills", fake_fleet)

    result = CliRunner().invoke(app, ["stats", "skills", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == merged
    assert calls == [
        {
            "local_payload": local,
            "since": None,
            "sources": (),
            "fleet_config": None,
        }
    ]


def test_stats_skills_missing_inventory_fails_without_rows(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RECALL_FLEET_PATH", raising=False)
    monkeypatch.setattr(stats_cli, "rpc_call_or_error", lambda *_args, **_kwargs: _payload())

    result = CliRunner().invoke(app, ["stats", "skills", "--json"])

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "VALIDATION"
    assert "no fleet hosts configured" in payload["error"]["message"]
    assert "rows" not in payload


def test_stats_skills_propagates_fleet_failure_without_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stats_cli, "rpc_call_or_error", lambda *_args, **_kwargs: _payload())

    def fail_fleet(**_kwargs: object) -> dict[str, object]:
        raise CliError(code=ErrorCode.RUNTIME, message="fleet stats skills: edge failed")

    monkeypatch.setattr("recall.cli.fleet.run_fleet_stats_skills", fail_fleet)

    result = CliRunner().invoke(app, ["stats", "skills", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "RUNTIME"
    assert "rows" not in payload


def test_stats_skills_manifest_exposes_full_contract() -> None:
    manifest = command_manifest(["stats", "skills"])

    assert manifest["command"] == "stats skills"
    options = {option["name"]: option for option in manifest["options"]}
    assert options["source"]["type"] == "string[]"
    assert options["local"]["default"] is False
    assert {"since", "fleet_config", "format", "fields", "json", "params"} <= set(options)
    assert manifest["output"]["type"] == "object"
    assert manifest["output"]["fields"] == ["rows", "coverage"]
