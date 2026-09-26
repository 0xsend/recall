"""Tests for call-to-action (CTA) support (REQ-CLI-015, REQ-CLI-016)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from conftest import _can_acquire_duckdb_lock, set_in_process_server
from recall.cli.app import app
from recall.cli.contract import Cta
from recall.cli.cta import (
    cta_for_daemon_status,
    cta_for_daemon_stop,
    cta_for_index,
    cta_for_list,
    cta_for_search,
    cta_for_show,
    cta_for_stats,
)
from typer.testing import CliRunner

_requires_duckdb_lock = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)

# -- Unit tests for CTA generators --


def test_cta_for_list_suggests_show_with_real_id() -> None:
    """REQ-CLI-016: CTAs use real IDs from output data."""
    sessions = [{"id": "abc123", "source": "claude_code"}]
    ctas = cta_for_list(sessions)
    assert any("recall show abc123" in c.command for c in ctas)


def test_cta_for_list_empty_still_suggests_search() -> None:
    ctas = cta_for_list([])
    assert any("recall search" in c.command for c in ctas)


def test_cta_for_search_suggests_show_top_result() -> None:
    """REQ-CLI-016: search CTA uses session ID from top result."""
    results = [{"session_id": "def456", "source": "codex", "score": 0.9}]
    ctas = cta_for_search(results)
    assert any("recall show def456" in c.command for c in ctas)


def test_cta_for_search_empty_returns_empty() -> None:
    ctas = cta_for_search([])
    assert ctas == []


def test_cta_for_show_suggests_project_list() -> None:
    session = {"id": "abc", "cwd": "/path/to/project"}
    ctas = cta_for_show(session)
    assert any("--project /path/to/project" in c.command for c in ctas)


def test_cta_for_show_without_cwd_skips_project() -> None:
    session = {"id": "abc", "cwd": None}
    ctas = cta_for_show(session)
    assert not any("--project" in c.command for c in ctas)
    assert any("recall search" in c.command for c in ctas)


def test_cta_for_index_suggests_list_and_search() -> None:
    summary = {"indexed": 5, "total": 10}
    ctas = cta_for_index(summary)
    assert any("recall list" in c.command for c in ctas)
    assert any("recall search" in c.command for c in ctas)


def test_cta_for_index_zero_indexed_skips_search() -> None:
    summary = {"indexed": 0, "total": 10}
    ctas = cta_for_index(summary)
    assert any("recall list" in c.command for c in ctas)
    assert not any("recall search" in c.command for c in ctas)


def test_cta_for_stats_root_suggests_subcommands() -> None:
    ctas = cta_for_stats(None)
    assert any("recall stats tools" in c.command for c in ctas)
    assert any("recall stats bash --suggest" in c.command for c in ctas)


def test_cta_for_stats_tools_suggests_bash() -> None:
    ctas = cta_for_stats("tools")
    assert any("recall stats bash" in c.command for c in ctas)


def test_cta_for_stats_bash_suggests_tokens() -> None:
    ctas = cta_for_stats("bash")
    assert any("recall stats tokens" in c.command for c in ctas)


def test_cta_for_stats_tokens_suggests_list() -> None:
    ctas = cta_for_stats("tokens")
    assert any("recall list" in c.command for c in ctas)


def test_cta_for_daemon_status_suggests_restart_on_version_drift() -> None:
    ctas = cta_for_daemon_status({"version_drift": True})

    assert any(c.command == "recall daemon restart" for c in ctas)


def test_cta_for_daemon_status_skips_restart_without_version_drift() -> None:
    ctas = cta_for_daemon_status({"version_drift": False})

    assert not any(c.command == "recall daemon restart" for c in ctas)


def test_cta_for_daemon_stop_suggests_durable_start() -> None:
    ctas = cta_for_daemon_stop()

    assert any(c.command == "recall daemon start" for c in ctas)
    assert not any("--background" in c.command for c in ctas)


# -- Integration tests --


def _install_fixtures(tmp_path: Path) -> None:
    claude_target = tmp_path / ".claude" / "projects" / "proj1"
    codex_target = tmp_path / ".codex" / "sessions" / "s1"
    claude_target.mkdir(parents=True)
    codex_target.mkdir(parents=True)

    root = Path(__file__).resolve().parents[2] / "fixtures"
    shutil.copy(root / "claude_code" / "session1.jsonl", claude_target / "session1.jsonl")
    shutil.copy(root / "codex" / "session1" / "rollout.jsonl", codex_target / "rollout.jsonl")


def _init_index(tmp_path: Path, monkeypatch) -> CliRunner:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    _install_fixtures(tmp_path)

    from recall.core.config import AppConfig
    from recall.services.rpc_server import RpcServer

    config = AppConfig.load()
    server = RpcServer(config=config)
    set_in_process_server(server)

    runner = CliRunner()
    # CTA assertions require indexed history without an external model dependency.
    result = runner.invoke(app, ["index", "--full", "--no-embed", "--yes"])
    assert result.exit_code == 0, f"index failed: {result.output}"
    return runner


@_requires_duckdb_lock
def test_cta_flag_wraps_json_in_envelope(tmp_path, monkeypatch) -> None:
    """REQ-CLI-015: --cta wraps output in {"data": ..., "cta": [...]}."""
    runner = _init_index(tmp_path, monkeypatch)

    result = runner.invoke(app, ["list", "--json", "--cta", "--limit", "1"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "data" in payload
    assert "cta" in payload
    assert isinstance(payload["data"], list)
    assert isinstance(payload["cta"], list)
    assert len(payload["cta"]) > 0
    # CTA should have command and description fields
    assert "command" in payload["cta"][0]
    assert "description" in payload["cta"][0]


@_requires_duckdb_lock
def test_cta_contextual_show_command_in_search(tmp_path, monkeypatch) -> None:
    """REQ-CLI-016: search CTA includes contextual recall show <session-id>."""
    runner = _init_index(tmp_path, monkeypatch)

    result = runner.invoke(app, ["search", "git", "--json", "--cta", "--limit", "1"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "cta" in payload
    # At least one CTA should contain "recall show"
    cta_commands = [c["command"] for c in payload["cta"]]
    assert any("recall show" in cmd for cmd in cta_commands)


@_requires_duckdb_lock
def test_cta_with_toon_format(tmp_path, monkeypatch) -> None:
    """REQ-CLI-015: CTA envelope works with TOON format."""
    from toon_format import decode as toon_decode

    runner = _init_index(tmp_path, monkeypatch)

    result = runner.invoke(app, ["list", "--format", "toon", "--cta", "--limit", "1"])

    assert result.exit_code == 0
    payload = toon_decode(result.stdout)
    assert isinstance(payload, dict)
    assert "data" in payload
    assert "cta" in payload


def test_cta_dataclass_serialization() -> None:
    """Verify Cta dataclass serializes correctly."""
    from recall.cli.contract import serialize_data

    cta = Cta(command="recall show abc", description="View session")
    serialized = serialize_data(cta)
    assert serialized == {"command": "recall show abc", "description": "View session"}
