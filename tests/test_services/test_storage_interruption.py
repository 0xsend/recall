"""Abrupt process death exercises actual WAL replay and matching-backup recovery."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.services.coordinator import set_paused
from recall.services.index_migration import migration_status, rollback_migration
from recall.services.rpc_server import RpcServer

_CHILD = r"""
import asyncio, json, os, sys
from pathlib import Path
from types import SimpleNamespace
import duckdb
from recall.core.config import AppConfig
from recall.services import index_migration
from recall.services.rpc_server import RpcServer

cfg = AppConfig.load()
phase = sys.argv[1]
server = RpcServer(config=cfg)
server._get_conn()
real_transition = index_migration.perform_storage_transition
real_connect = duckdb.connect

def die(conn):
    row = conn.execute("SELECT backup_path, storage_attempt FROM index_migration_jobs").fetchone()
    with (cfg.data_dir / 'interruption.json').open('w') as handle:
        json.dump({'phase':phase, 'backup_path':row[0], 'attempt':row[1]}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os._exit(23)

class ObservedConnection:
    def __init__(self, conn): self.conn = conn
    def __enter__(self): return self
    def __exit__(self, *args): return self.conn.__exit__(*args)
    def execute(self, sql, *args):
        result = self.conn.execute(sql, *args)
        if 'storage_attempt = storage_attempt + 1' in sql:
            die(self.conn)
        return result

def transition(config):
    if phase == 'before':
        with real_connect(str(config.db_path), read_only=True) as conn: die(conn)
    if phase == 'wal':
        index_migration.duckdb = SimpleNamespace(
            connect=lambda *a, **kw: ObservedConnection(real_connect(*a, **kw)))
    real_transition(config)
    if phase == 'after':
        with real_connect(str(config.db_path), read_only=True) as conn: die(conn)
    raise AssertionError('interruption point was not reached')

index_migration.perform_storage_transition = transition
asyncio.run(server._run_storage_maintenance())
"""


@pytest.mark.parametrize("phase", ["before", "wal", "after"])
def test_storage_interruption_resumes_original_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    config_file = tmp_path / "config.toml"
    config_file.write_text("[daemon]\nembed=false\n[embedding.context]\nmode='template'\n")
    monkeypatch.setenv("RECALL_DATA_DIR", str(data))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_file))
    cfg = AppConfig.load()
    with duckdb.connect(str(cfg.db_path)) as conn:
        conn.execute("CREATE TABLE owned_marker AS SELECT 73 AS value")
    set_paused(cfg, True)
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, phase],
        capture_output=True,
        text=True,
        timeout=15,
        env=os.environ.copy(),
    )
    (tmp_path / "child.stdout").write_text(child.stdout)
    (tmp_path / "child.stderr").write_text(child.stderr)
    assert child.returncode == 23, child.stderr
    receipt = json.loads((data / "interruption.json").read_text())
    if phase == "wal":
        assert Path(str(cfg.db_path) + ".wal").stat().st_size > 0
        assert receipt["attempt"] == 1
    server = RpcServer(config=cfg)
    asyncio.run(server._maybe_begin_index_migration())
    status = migration_status(server._get_conn())
    assert status.phase == "idle"
    assert status.storage_version == "v1.2.0+"
    assert status.storage_attempt >= max(1, receipt["attempt"])
    assert status.backup_path == receipt["backup_path"]
    assert status.applied_version == 1
    assert status.captured == 0
    assert server._get_conn().execute("SELECT value FROM owned_marker").fetchone() == (73,)
    server._conn.close()
    server._conn = None

    # Restore into another owned directory; retain the successfully resumed file.
    from dataclasses import replace
    from shutil import copy2

    rollback_dir = tmp_path / "rollback"
    rollback_dir.mkdir()
    rollback_config = replace(cfg, data_dir=rollback_dir, db_path=rollback_dir / "recall.duckdb")
    copy2(cfg.db_path, rollback_config.db_path)
    rollback_migration(rollback_config)
    with duckdb.connect(str(rollback_config.db_path), read_only=True) as conn:
        restored = migration_status(conn)
        assert restored.phase == "idle"
        assert restored.storage_version == "v1.0.0+"
        assert conn.execute("SELECT value FROM owned_marker").fetchone() == (73,)
