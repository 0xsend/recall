from __future__ import annotations

import os
from pathlib import Path

import pytest
import recall.services.snapshots as snapshots_module
from recall.core.config import AppConfig
from recall.services.snapshots import RECALL_SIDECAR_INTRODUCED_AT, gc_snapshots


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    data_dir = tmp_path / ".local" / "share" / "recall"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(data_dir))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / ".config/recall/config.toml"))
    monkeypatch.delenv("RECALL_DB_PATH", raising=False)
    monkeypatch.delenv("RECALL_LOCK_PATH", raising=False)
    return AppConfig.load()


def _snapshots_dir(config: AppConfig) -> Path:
    path = config.data_dir / "snapshots"
    path.mkdir(parents=True)
    return path


def _set_mtime(path: Path, timestamp: float) -> None:
    os.utime(path, (timestamp, timestamp), follow_symlinks=False)


def _freeze_after_sidecar_cutoff(monkeypatch: pytest.MonkeyPatch) -> float:
    now = RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 10_000
    monkeypatch.setattr(snapshots_module.time, "time", lambda: now)
    return now


@pytest.mark.parametrize("kind", ["index-migration", "storage-migration", "recreate"])
def test_gc_retains_aged_recovery_backups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import sqlite3

    from recall.db.connection import connect
    from recall.services.recreation import consistent_backup

    config = _config(tmp_path, monkeypatch)
    now = _freeze_after_sidecar_cutoff(monkeypatch)
    conn = connect(config)
    with sqlite3.connect(config.data_dir / "recall.fts.sqlite") as sidecar:
        sidecar.execute("CREATE TABLE owned_marker AS SELECT 19 AS value")
    backup = consistent_backup(conn, config, kind=kind)
    conn.close()
    _set_mtime(backup, now - 8 * 86400)
    result = gc_snapshots(config, dry_run=True)
    assert str(backup) in result.kept_paths
    assert result.removed_paths == ()
    assert (backup / "recall.duckdb").is_file()
    assert (backup / "recall.fts.sqlite").is_file()


def test_gc_refuses_partial_snapshot_with_only_duckdb_after_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _freeze_after_sidecar_cutoff(monkeypatch)
    config = _config(tmp_path, monkeypatch)
    snap = _snapshots_dir(config) / "snap1.duckdb"
    snap.write_text("duckdb")
    _set_mtime(snap, now - 1)

    result = gc_snapshots(config, days=0)

    assert result.partial_paths == (str(snap),)
    assert str(snap) not in result.removed_paths
    assert snap.exists()


def test_gc_treats_pre_sidecar_duckdb_only_as_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_after_sidecar_cutoff(monkeypatch)
    config = _config(tmp_path, monkeypatch)
    legacy = _snapshots_dir(config) / "legacy.duckdb"
    legacy.write_text("legacy")
    _set_mtime(legacy, RECALL_SIDECAR_INTRODUCED_AT.timestamp() - 1)

    result = gc_snapshots(config, days=0)

    assert result.removed_paths == (str(legacy),)
    assert result.partial_paths == ()
    assert not legacy.exists()


def test_gc_removes_complete_paired_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _freeze_after_sidecar_cutoff(monkeypatch)
    config = _config(tmp_path, monkeypatch)
    snapshots = _snapshots_dir(config)
    duckdb = snapshots / "snap2.duckdb"
    sidecar = snapshots / "snap2.fts.sqlite"
    wal = snapshots / "snap2.duckdb.wal"
    duckdb.write_text("duck")
    sidecar.write_text("sidecar")
    wal.write_text("wal")
    _set_mtime(duckdb, now - 1)
    _set_mtime(sidecar, now - 1)
    _set_mtime(wal, now - 1)

    result = gc_snapshots(config, days=0)

    assert result.removed_paths == (str(duckdb), str(sidecar))
    assert result.total_bytes_freed == 11
    assert not duckdb.exists()
    assert not sidecar.exists()
    assert wal.exists()


def test_gc_refuses_partial_snapshot_with_only_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_after_sidecar_cutoff(monkeypatch)
    config = _config(tmp_path, monkeypatch)
    sidecar = _snapshots_dir(config) / "snap3.fts.sqlite"
    sidecar.write_text("sidecar")
    _set_mtime(sidecar, RECALL_SIDECAR_INTRODUCED_AT.timestamp() - 1)

    result = gc_snapshots(config, days=0)

    assert result.partial_paths == (str(sidecar),)
    assert result.removed_paths == ()
    assert sidecar.exists()


def test_gc_handles_per_snapshot_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _freeze_after_sidecar_cutoff(monkeypatch)
    config = _config(tmp_path, monkeypatch)
    snapshots = _snapshots_dir(config)
    complete = snapshots / "snap4"
    complete.mkdir()
    (complete / "recall.duckdb").write_text("duck")
    (complete / "recall.fts.sqlite").write_text("sidecar")
    (complete / "manifest.json").write_text("{}")
    partial = snapshots / "snap5"
    partial.mkdir()
    (partial / "recall.duckdb").write_text("duck")
    _set_mtime(complete, now - 1)
    _set_mtime(partial, now - 1)

    result = gc_snapshots(config, days=0)

    assert str(complete) in result.removed_paths
    assert str(partial) in result.partial_paths
    assert not complete.exists()
    assert partial.exists()
