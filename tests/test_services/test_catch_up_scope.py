"""Reconciliation stays inside configured roots, independently of host HOME."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
    SourceConfig,
)
from recall.core.types import DaemonMode
from recall.services.rpc_server import RpcServer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _config(tmp_path: Path, lane: Path) -> AppConfig:
    data_dir = tmp_path / ".local" / "share" / "recall"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / ".config" / "recall" / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(mode=DaemonMode.WATCH),
        cli=CliConfig(),
        sources={
            "claude_code": SourceConfig(roots=(lane,)),
            "codex": SourceConfig(roots=()),
            "pi_agent": SourceConfig(roots=()),
            "grok": SourceConfig(roots=()),
            "kimi_code": SourceConfig(roots=()),
        },
    )


def test_reconciliation_stays_inside_the_configured_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pinned lane means the lane, not the lane plus everything under $HOME."""
    monkeypatch.setenv("HOME", str(tmp_path))

    home_projects = tmp_path / ".claude" / "projects" / "real"
    home_projects.mkdir(parents=True)
    shutil.copy(FIXTURES / "claude_code" / "session1.jsonl", home_projects / "real.jsonl")

    lane = tmp_path / "lane"
    (lane / "proj").mkdir(parents=True)
    shutil.copy(FIXTURES / "claude_code" / "live_mid_tool.jsonl", lane / "proj" / "live.jsonl")

    config = _config(tmp_path, lane)
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    server = RpcServer(config)

    async def run():
        try:
            summary = await server._handle_daemon_run({"once": True, "embed": False}, None)
            assert summary.index_summary.total == summary.index_summary.indexed == 1
            assert server._get_conn().execute(
                "SELECT source_path FROM source_files"
            ).fetchall() == [(str((lane / "proj" / "live.jsonl").resolve()),)]
        finally:
            await server.stop()

    asyncio.run(run())


def test_reconciliation_with_a_source_pinned_to_nothing_finds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`roots = []` is the only way to tell recall to ignore a harness entirely."""
    monkeypatch.setenv("HOME", str(tmp_path))

    home_projects = tmp_path / ".claude" / "projects" / "real"
    home_projects.mkdir(parents=True)
    shutil.copy(FIXTURES / "claude_code" / "session1.jsonl", home_projects / "real.jsonl")

    config = _config(tmp_path, tmp_path / "absent-lane")
    config = replace(config, sources={**config.sources, "claude_code": SourceConfig(roots=())})
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    server = RpcServer(config)

    async def run():
        try:
            summary = await server._handle_daemon_run({"once": True, "embed": False}, None)
            assert summary.index_summary.total == 0
            assert server._get_conn().execute("SELECT COUNT(*) FROM source_files").fetchone() == (
                0,
            )
        finally:
            await server.stop()

    asyncio.run(run())
