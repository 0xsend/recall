"""Explicit fixed-format maintenance preserves the original matching backup."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import duckdb
import pytest
from lane_harness import lane_config
from recall.core.config import AppConfig
from recall.db.connection import connect
from recall.services.index_migration import (
    applied_version,
    begin_storage_migration,
    captured_keys,
    migration_status,
    perform_storage_transition,
    rollback_migration,
    verify_and_complete,
)


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    cfg = lane_config(tmp_path)
    cfg.data_dir.mkdir(parents=True)
    with duckdb.connect(str(cfg.db_path)) as conn:
        conn.execute("CREATE TABLE owned_marker AS SELECT 17 AS value")
    return cfg


def test_storage_only_requires_backup_and_persisted_format(config: AppConfig) -> None:
    sidecar = config.data_dir / "recall.fts.sqlite"
    with sqlite3.connect(sidecar) as sqlite:
        sqlite.execute("CREATE TABLE owned_keyword (value TEXT)")
        sqlite.execute("INSERT INTO owned_keyword VALUES ('before')")
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    conn = connect(config)
    version = applied_version(conn)
    status = begin_storage_migration(conn, config)
    assert status.storage_version == "v1.0.0+"
    assert status.storage_target == "v1.2.0"
    assert status.captured == 0
    assert status.backup_path is not None
    backup = Path(status.backup_path)
    assert json.loads((backup / "manifest.json").read_text())["complete"] is True
    assert (backup / "recall.fts.sqlite").is_file()
    with pytest.raises(RuntimeError, match="persisted storage"):
        verify_and_complete(conn, verified=True)
    conn.close()

    perform_storage_transition(config)
    conn = connect(config)
    assert conn.execute("SELECT value FROM owned_marker").fetchone() == (17,)
    status = verify_and_complete(conn, verified=True)
    assert status.phase == "idle"
    assert status.storage_version == "v1.2.0+"
    assert status.storage_attempt == 1
    assert applied_version(conn) == version
    assert captured_keys(conn) == ()
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == digest
    again = begin_storage_migration(conn, config)
    assert again == status
    conn.close()

    with sqlite3.connect(sidecar) as sqlite:
        sqlite.execute("UPDATE owned_keyword SET value='after'")
    rollback_migration(config)
    conn = connect(config)
    assert migration_status(conn).storage_version == "v1.0.0+"
    assert migration_status(conn).phase == "idle"
    assert conn.execute("SELECT value FROM owned_marker").fetchone() == (17,)
    with sqlite3.connect(sidecar) as sqlite:
        assert sqlite.execute("SELECT value FROM owned_keyword").fetchone() == ("before",)
    conn.close()


def test_failed_storage_resume_keeps_backup_and_start(config: AppConfig) -> None:
    conn = connect(config)
    first = begin_storage_migration(conn, config)
    conn.execute("UPDATE index_migration_jobs SET phase='failed', error='interrupted'")
    second = begin_storage_migration(conn, config)
    assert second.backup_path == first.backup_path
    assert second.started_at == first.started_at
    assert second.captured == 0
    assert second.error is None
    conn.close()


def test_missing_original_backup_fails_closed(config: AppConfig) -> None:
    conn = connect(config)
    first = begin_storage_migration(conn, config)
    conn.execute(
        "UPDATE index_migration_jobs SET phase='failed', backup_path='/absent/owned-backup'"
    )
    with pytest.raises(RuntimeError, match="backup"):
        begin_storage_migration(conn, config)
    assert first.backup_path is not None
    assert len(list((config.data_dir / "snapshots").iterdir())) == 1
    conn.close()
