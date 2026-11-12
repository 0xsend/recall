"""Configured maintenance entries retain ordinary recovery holds."""

import asyncio
import importlib.util
import sys
from pathlib import Path

import duckdb
import pytest
from recall.db.connection import connect
from recall.services.daemon import run_startup_fts_sidecar_sync
from recall.services.fts_sidecar_rebuild import KeywordRebuild
from recall.services.self_repair import DaemonStartupRefused, refusal_marker_path

from tests.test_services.test_rpc_owned_boundary import server_at


@pytest.mark.parametrize(
    "entry", ["backfill", "backfill_preview", "supersede", "supersede_preview", "keyword"]
)
def test_other_ordinary_maintenance_cannot_bypass_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    from recall.cli.contract import CliError
    from recall.cli.db import _supersede_moved_locally

    server = server_at(tmp_path)
    config = server._config
    with connect(config) as conn:
        conn.execute("CREATE TABLE witness(k INT)")
        conn.execute("INSERT INTO witness VALUES (17)")
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RECALL_DB_PATH", str(config.db_path))
    monkeypatch.setenv("RECALL_LOCK_PATH", str(config.lock_path))
    refusal = refusal_marker_path(tmp_path)
    refusal.write_text("retained recovery instruction")
    try:
        if entry.startswith("backfill"):
            script = Path(__file__).resolve().parents[2] / "scripts/backfill_codex_tokens.py"
            spec = importlib.util.spec_from_file_location("_maintenance_backfill", script)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            with pytest.raises(DaemonStartupRefused):
                module.backfill(dry_run=entry.endswith("preview"))
        elif entry.startswith("supersede"):
            with pytest.raises(CliError) as result:
                _supersede_moved_locally(config, apply=not entry.endswith("preview"))
            assert result.value.exit_code == 3
        else:
            with pytest.raises(DaemonStartupRefused):
                KeywordRebuild(config, tmp_path / "keyword-rebuild", "scratch-test").begin()
            assert not (tmp_path / "keyword-rebuild").exists()
        assert refusal.read_text() == "retained recovery instruction"
    finally:
        asyncio.run(server.stop())
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        assert fresh.execute("SELECT * FROM witness").fetchall() == [(17,)]


def test_standalone_sidecar_startup_honors_retained_refusal(tmp_path: Path) -> None:
    server = server_at(tmp_path)
    with connect(server._config) as conn:
        conn.execute("CREATE TABLE witness(k INT)")
        conn.execute("INSERT INTO witness VALUES (17)")
    refusal = refusal_marker_path(tmp_path)
    refusal.write_text("retained recovery instruction")
    try:
        with pytest.raises(DaemonStartupRefused):
            run_startup_fts_sidecar_sync(server._config)
        assert refusal.read_text() == "retained recovery instruction"
        assert not (tmp_path / "recall.fts.sqlite").exists()
    finally:
        asyncio.run(server.stop())
    with duckdb.connect(str(server._config.db_path), read_only=True) as fresh:
        assert fresh.execute("SELECT * FROM witness").fetchall() == [(17,)]
