from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.services import snapshots as snapshots_service
from recall.services.snapshots import gc_snapshots


def _config(tmp_path: Path) -> AppConfig:
    data_dir = tmp_path / "data"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )


def _age(path: Path, *, days: int) -> None:
    when = time.time() - (days * 86_400)
    os.utime(path, (when, when), follow_symlinks=False)


def test_gc_missing_snapshots_dir_is_noop(tmp_path: Path) -> None:
    result = gc_snapshots(_config(tmp_path))

    assert result.snapshots_dir_missing is True
    assert result.removed_paths == ()
    assert result.failed_paths == ()
    assert result.total_bytes_freed == 0


def test_gc_records_snapshots_dir_when_listing_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    snapshots = config.data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    original_iterdir = Path.iterdir

    def fail_iterdir(path: Path) -> Iterator[Path]:
        if path == snapshots:
            raise PermissionError(f"simulated listing failure for {path}")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_iterdir)

    result = gc_snapshots(config)

    assert result.failed_paths == (str(snapshots),)
    assert result.removed_paths == ()
    assert result.kept_paths == ()
    assert result.total_bytes_freed == 0


def test_gc_removes_stale_file_and_reports_size(tmp_path: Path) -> None:
    config = _config(tmp_path)
    snapshots = config.data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    stale = snapshots / "old.bin"
    stale.write_bytes(b"abcdef")
    _age(stale, days=10)

    result = gc_snapshots(config, days=7)

    assert result.removed_paths == (str(stale),)
    assert result.failed_paths == ()
    assert result.total_bytes_freed == 6
    assert not stale.exists()


def test_gc_keeps_fresh_file_and_removes_stale_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    snapshots = config.data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    fresh = snapshots / "fresh"
    stale = snapshots / "stale"
    fresh.write_text("fresh")
    stale.write_text("stale")
    _age(stale, days=10)

    result = gc_snapshots(config, days=7)

    assert result.removed_paths == (str(stale),)
    assert result.kept_paths == (str(fresh),)
    assert result.failed_paths == ()
    assert fresh.exists()
    assert not stale.exists()


def test_gc_directory_uses_top_level_mtime_and_recursive_size(tmp_path: Path) -> None:
    config = _config(tmp_path)
    snapshots = config.data_dir / "snapshots"
    snapshot_dir = snapshots / "old-dir"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "a").write_bytes(b"a")
    (snapshot_dir / "b").write_bytes(b"bb")
    nested = snapshot_dir / "nested"
    nested.mkdir()
    (nested / "c").write_bytes(b"ccc")
    _age(snapshot_dir, days=10)

    result = gc_snapshots(config, days=7)

    assert result.removed_paths == (str(snapshot_dir),)
    assert result.failed_paths == ()
    assert result.total_bytes_freed == 6
    assert not snapshot_dir.exists()


def test_gc_records_failed_path_when_stale_delete_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    snapshots = config.data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    stale = snapshots / "stale"
    stale.write_bytes(b"stale")
    _age(stale, days=10)

    def fail_remove(path: Path) -> None:
        raise OSError(f"simulated delete failure for {path}")

    monkeypatch.setattr(snapshots_service, "_remove", fail_remove)

    result = gc_snapshots(config, days=7)

    assert result.failed_paths == (str(stale),)
    assert result.removed_paths == ()
    assert result.total_bytes_freed == 0
    assert stale.exists()


def test_gc_dry_run_reports_stale_file_without_removing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    snapshots = config.data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    stale = snapshots / "stale"
    stale.write_bytes(b"stale")
    _age(stale, days=10)

    result = gc_snapshots(config, days=7, dry_run=True)

    assert result.dry_run is True
    assert result.removed_paths == (str(stale),)
    assert result.failed_paths == ()
    assert result.total_bytes_freed == 5
    assert stale.exists()


def test_gc_unlinks_stale_symlink_inside_snapshots(tmp_path: Path) -> None:
    config = _config(tmp_path)
    snapshots = config.data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    target = snapshots / "target"
    target.write_text("target")
    link = snapshots / "target-link"
    link.symlink_to(target)
    _age(link, days=10)

    result = gc_snapshots(config, days=7)

    assert result.removed_paths == (str(link),)
    assert result.failed_paths == ()
    assert not link.exists()
    assert target.exists()


def test_gc_records_symlink_loop_as_failed_path(tmp_path: Path) -> None:
    config = _config(tmp_path)
    snapshots = config.data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    loop = snapshots / "loop"
    loop.symlink_to("loop")

    result = gc_snapshots(config, days=7)

    assert result.failed_paths == (str(loop),)
    assert result.removed_paths == ()
    assert loop.is_symlink()


def test_gc_refuses_symlink_that_resolves_outside_snapshots(tmp_path: Path) -> None:
    config = _config(tmp_path)
    snapshots = config.data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    outside = tmp_path / "outside-target"
    outside.write_text("outside")
    link = snapshots / "outside-link"
    link.symlink_to(outside)
    _age(link, days=10)

    result = gc_snapshots(config, days=7)

    assert result.removed_paths == ()
    assert result.kept_paths == (str(link),)
    assert result.failed_paths == ()
    assert link.exists()
    assert outside.read_text() == "outside"


def test_gc_refuses_snapshots_dir_symlink(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.data_dir.mkdir(parents=True)
    outside_dir = tmp_path / "outside-snapshots"
    outside_dir.mkdir()
    outside_file = outside_dir / "stale"
    outside_file.write_text("outside")
    _age(outside_file, days=10)
    (config.data_dir / "snapshots").symlink_to(outside_dir, target_is_directory=True)

    result = gc_snapshots(config, days=7)

    assert result.removed_paths == ()
    assert result.failed_paths == ()
    assert outside_file.exists()
