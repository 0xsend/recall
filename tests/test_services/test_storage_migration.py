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
from recall.db.lease import StandaloneLifetime
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
    status = begin_storage_migration(conn, config, observer=StandaloneLifetime())
    assert status.storage_version == "v1.0.0+"
    assert status.storage_target == "v1.2.0"
    assert status.captured == 0
    assert status.backup_path is not None
    backup = Path(status.backup_path)
    assert json.loads((backup / "manifest.json").read_text())["complete"] is True
    assert (backup / "recall.fts.sqlite").is_file()
    with pytest.raises(RuntimeError, match="persisted storage"):
        verify_and_complete(conn, verified=True, observer=StandaloneLifetime())
    conn.close()

    perform_storage_transition(config)
    conn = connect(config)
    assert conn.execute("SELECT value FROM owned_marker").fetchone() == (17,)
    status = verify_and_complete(conn, verified=True, observer=StandaloneLifetime())
    assert status.phase == "idle"
    assert status.storage_version == "v1.2.0+"
    assert status.storage_attempt == 1
    assert applied_version(conn) == version
    assert captured_keys(conn) == ()
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == digest
    again = begin_storage_migration(conn, config, observer=StandaloneLifetime())
    assert again == status
    conn.close()

    with sqlite3.connect(sidecar) as sqlite:
        sqlite.execute("UPDATE owned_keyword SET value='after'")
    rollback_migration(config, preserve=config.data_dir.resolve().parent / "failed-set")
    conn = connect(config)
    assert migration_status(conn).storage_version == "v1.0.0+"
    assert migration_status(conn).phase == "idle"
    assert conn.execute("SELECT value FROM owned_marker").fetchone() == (17,)
    with sqlite3.connect(sidecar) as sqlite:
        assert sqlite.execute("SELECT value FROM owned_keyword").fetchone() == ("before",)
    conn.close()


def test_failed_storage_resume_keeps_backup_and_start(config: AppConfig) -> None:
    conn = connect(config)
    first = begin_storage_migration(conn, config, observer=StandaloneLifetime())
    conn.execute("UPDATE index_migration_jobs SET phase='failed', error='interrupted'")
    second = begin_storage_migration(conn, config, observer=StandaloneLifetime())
    assert second.backup_path == first.backup_path
    assert second.started_at == first.started_at
    assert second.captured == 0
    assert second.error is None
    conn.close()


def test_missing_original_backup_fails_closed(config: AppConfig) -> None:
    conn = connect(config)
    first = begin_storage_migration(conn, config, observer=StandaloneLifetime())
    conn.execute(
        "UPDATE index_migration_jobs SET phase='failed', backup_path='/absent/owned-backup'"
    )
    with pytest.raises(RuntimeError, match="backup"):
        begin_storage_migration(conn, config, observer=StandaloneLifetime())
    assert first.backup_path is not None
    assert len(list((config.data_dir / "snapshots").iterdir())) == 1
    conn.close()


@pytest.mark.parametrize(
    "operation",
    ["inspect", "index-resume", "storage-resume", "transition", "rollback", "raw-commit"],
)
def test_foreign_backup_refuses_operational_readers_without_effects(
    tmp_path: Path,
    config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    from recall.services.coordinator import commit_prepared_raw_sources
    from recall.services.index_migration import begin_migration, plan_storage_migration

    from tests.backup_access_guard import ForeignAccessGuard

    source = tmp_path / "source"
    source.mkdir()
    # A complete foreign backup and an identically named local collision must
    # never be inferred as permission to use either backup.
    from recall.services.recreation import consistent_backup

    conn = connect(config)
    backup = consistent_backup(
        conn, config, kind="storage-migration", observer=StandaloneLifetime()
    )
    import shutil

    shutil.copytree(backup, source / backup.name)
    foreign = str(source / backup.name)
    conn.execute("UPDATE runtime_state SET index_migration_version=0")
    conn.execute(
        """UPDATE index_migration_jobs SET phase='running', backup_path=?,
        captured_count=1, completed_count=0, storage_target='v1.2.0+'""",
        [foreign],
    )
    before_job = conn.execute("SELECT * FROM index_migration_jobs").fetchall()
    before_scope = conn.execute("SELECT * FROM index_migration_scope").fetchall()
    if operation in {"transition", "rollback"}:
        conn.close()
    paths = [
        config.db_path,
        Path(str(config.db_path) + ".wal"),
        config.data_dir / "recall.fts.sqlite",
    ]
    before_files = {p: p.read_bytes() if p.exists() else None for p in paths}
    try:
        guard = ForeignAccessGuard(monkeypatch, source)
        calls = {
            "inspect": lambda: plan_storage_migration(conn, config),
            "index-resume": lambda: begin_migration(conn, config, observer=StandaloneLifetime()),
            "storage-resume": lambda: begin_storage_migration(
                conn, config, observer=StandaloneLifetime()
            ),
            "transition": lambda: perform_storage_transition(config),
            "rollback": lambda: rollback_migration(
                config, preserve=config.data_dir.resolve().parent / "failed-set"
            ),
            "raw-commit": lambda: commit_prepared_raw_sources((), config, conn=conn),
        }
        with pytest.raises(RuntimeError, match=r"foreign or unqualified.*uninspected"):
            calls[operation]()
        assert guard.attempts == []
        if operation in {"transition", "rollback"}:
            conn = connect(config)
        assert conn.execute("SELECT * FROM index_migration_jobs").fetchall() == before_job
        assert conn.execute("SELECT * FROM index_migration_scope").fetchall() == before_scope
        assert migration_status(conn).applied_version == 0
        if operation == "rollback":
            assert {p: p.read_bytes() if p.exists() else None for p in paths} == before_files
    finally:
        conn.close()


@pytest.mark.parametrize("operation", ["index", "storage"])
@pytest.mark.parametrize("root_spelling", ["absolute", "relative", "dotted", "linked"])
def test_created_backup_remains_local_through_selected_root_spellings(
    config: AppConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    root_spelling: str,
) -> None:
    from recall.db.source_files import SourceCatalog, SourceSignature
    from recall.services.index_migration import begin_migration
    from recall.services.migration_backup import backup_presence

    root = config.data_dir
    alias = tmp_path / "linked"
    alias.symlink_to(root, target_is_directory=True)
    selected = {
        "absolute": root,
        "relative": Path(".local/share/recall"),
        "dotted": Path(".local/share/../share/recall"),
        "linked": alias,
    }[root_spelling]
    object.__setattr__(config, "data_dir", selected)
    monkeypatch.chdir(tmp_path)
    conn = connect(config)
    begin = begin_migration if operation == "index" else begin_storage_migration
    try:
        if operation == "index":
            source = tmp_path / "source.jsonl"
            source.write_text("captured input")
            SourceCatalog(conn, clock=lambda: 1.0).observe(
                "codex",
                str(tmp_path),
                str(source),
                SourceSignature(dev=1, inode=1, ctime_ns=1, mtime_ns=1, size=14),
            )
            conn.execute("UPDATE runtime_state SET index_migration_version=0")
        first = begin(conn, config, observer=StandaloneLifetime())
        assert first.backup_path is not None
        assert Path(first.backup_path).parent == root / "snapshots"
        assert backup_presence(config, first.backup_path) is True
        scope = captured_keys(conn)
        resumed = begin(conn, config, observer=StandaloneLifetime())
        assert resumed.backup_path == first.backup_path
        assert captured_keys(conn) == scope
        assert resumed.captured == first.captured
        assert len(list((root / "snapshots").iterdir())) == 1
    finally:
        conn.close()
    assert rollback_migration(
        config, preserve=config.data_dir.resolve().parent / "failed-set"
    ) == Path(first.backup_path)
    with connect(config) as restored:
        assert restored.execute("SELECT value FROM owned_marker").fetchone() == (17,)
        assert migration_status(restored).phase == "idle"
