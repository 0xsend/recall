"""A restored file stays non-authoritative while post-backup work is unsettled."""

import errno
import sqlite3
from contextlib import closing
from pathlib import Path

import duckdb
import pytest
from recall.db.lease import StandaloneLifetime
from recall.db.schema import ensure_schema
from recall.services import recreation
from recall.services.index_migration import begin_storage_migration, rollback_migration
from recall.services.self_repair import DaemonStartupRefused, run_startup_self_repair

from tests.test_services.test_run_completion import _config


def _prepared(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    config = _config(data, None)
    with duckdb.connect(
        str(config.db_path), config={"storage_compatibility_version": "v1.0.0"}
    ) as conn:
        ensure_schema(conn)
        job = begin_storage_migration(conn, config, observer=StandaloneLifetime())
        conn.execute("CREATE TABLE acknowledged AS SELECT 9223372036854775807 AS value")
    with closing(sqlite3.connect(data / "recall.fts.sqlite")) as conn:
        conn.execute("CREATE TABLE acknowledged(value TEXT)")
        conn.execute("INSERT INTO acknowledged VALUES ('post-backup')")
        conn.commit()
    return config, job


def _assert_preserved(archive):
    with duckdb.connect(str(archive / "recall.duckdb"), read_only=True) as conn:
        assert conn.execute("SELECT * FROM acknowledged").fetchall() == [(9223372036854775807,)]
    with closing(
        sqlite3.connect(f"{(archive / 'recall.fts.sqlite').as_uri()}?mode=ro", uri=True)
    ) as conn:
        assert conn.execute("SELECT * FROM acknowledged").fetchall() == [("post-backup",)]


def test_migration_restore_preserves_post_backup_work_and_holds(tmp_path: Path):
    config, _ = _prepared(tmp_path)
    archive = tmp_path / "failed-set"
    rollback_migration(config, preserve=archive)
    assert (config.data_dir / "daemon-refused").exists()
    _assert_preserved(archive)
    restored = config.db_path.read_bytes()
    with pytest.raises(DaemonStartupRefused):
        run_startup_self_repair(config)
    assert config.db_path.read_bytes() == restored


def test_migration_restore_cannot_install_a_pre_fence_backup(tmp_path: Path):
    config, job = _prepared(tmp_path)
    assert job.backup_path is not None
    with duckdb.connect(str(Path(job.backup_path) / "recall.duckdb")) as old:
        old.execute("DROP TABLE daemon_index_repair")
        old.execute("DELETE FROM schema_version WHERE version>36")
        old.execute("INSERT INTO schema_version(version) VALUES (36)")
        old.execute("DELETE FROM schema_migrations WHERE migration_id='0041_daemon_index_repair'")
    before = config.db_path.read_bytes()
    with pytest.raises(RuntimeError, match="cannot lower the logical schema"):
        rollback_migration(config, preserve=tmp_path / "failed-set")
    assert config.db_path.read_bytes() == before


def test_restore_copy_failure_retains_failed_work_and_refuses_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config, _ = _prepared(tmp_path)
    archive = tmp_path / "failed-set"
    copy = recreation.shutil.copyfileobj
    failures = []

    def fail(source, target, *args, **kwargs):
        if Path(target.name) == config.db_path:
            failures.append(True)
            target.write(b"interrupted restore")
            target.flush()
            raise OSError(errno.EIO, "restore device unavailable")
        return copy(source, target, *args, **kwargs)

    monkeypatch.setattr(recreation.shutil, "copyfileobj", fail)
    with pytest.raises(OSError, match="restore device unavailable"):
        rollback_migration(config, preserve=archive)
    assert failures == [True]
    assert (config.data_dir / "daemon-refused").exists()
    _assert_preserved(archive)
    with pytest.raises(DaemonStartupRefused):
        run_startup_self_repair(config)
