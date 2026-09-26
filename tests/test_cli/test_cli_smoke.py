from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from conftest import _can_acquire_duckdb_lock, set_in_process_server
from recall.cli.app import app
from typer.testing import CliRunner

_requires_duckdb_lock = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)


@_requires_duckdb_lock
def test_cli_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

    claude_target = tmp_path / ".claude" / "projects" / "proj1"
    codex_target = tmp_path / ".codex" / "sessions" / "s1"
    claude_target.mkdir(parents=True)
    codex_target.mkdir(parents=True)

    claude_fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "claude_code" / "session1.jsonl"
    )
    codex_fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "codex" / "session1" / "rollout.jsonl"
    )

    shutil.copy(claude_fixture, claude_target / "session1.jsonl")
    shutil.copy(codex_fixture, codex_target / "rollout.jsonl")

    # Set up in-process RPC server with test config
    from recall.core.config import AppConfig
    from recall.services.rpc_server import RpcServer

    config = AppConfig.load()
    server = RpcServer(config=config)
    set_in_process_server(server)

    runner = CliRunner()
    result = runner.invoke(app, ["index", "--full", "--recreate", "--no-embed", "--yes"])
    assert result.exit_code == 0, f"index failed: {result.output}"

    result = runner.invoke(app, ["index", "--no-embed", "--json"])
    assert result.exit_code == 0
    index_payload = json.loads(result.stdout)
    assert isinstance(index_payload, dict)
    assert "total" in index_payload

    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload
