"""The operator's codec conversion preserves exact history and fails closed."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pytest
from lane_harness import lane_config
from recall.core.rpc_types import RpcError
from recall.db.connection import STORAGE_VERSION, connect
from recall.services.rpc_server import RpcServer
from recall.services.storage_codec import (
    _array_equal,
    codec_required,
    digest,
    load_journal,
    perform_codec_transition,
    prepare_codec,
    recover_codec,
    validate_candidate,
)


def legacy(tmp_path):
    cfg = lane_config(tmp_path)
    conn = connect(cfg)
    # An older writer reintroduces persisted masks at the current fixed header.
    conn.execute("SET disabled_compression_methods=''")
    conn.execute("CREATE TABLE historical (id VARCHAR PRIMARY KEY, v FLOAT[384], body VARCHAR)")
    conn.execute(
        "INSERT INTO historical SELECT md5(i::VARCHAR), "
        "list_transform(range(384), j -> CASE WHEN i%4=0 THEN NULL::FLOAT "
        "ELSE -0.0::FLOAT END)::FLOAT[384], 'payload-'||i "
        "FROM range(10000) t(i)"
    )
    conn.execute("CREATE MACRO preserved(x) AS x+17")
    conn.execute("CREATE INDEX historical_body ON historical(body)")
    bits = np.full(384, 0x80000000, dtype=np.uint32)
    bits[1] = 0x7FC00001
    special = pa.table(
        {"v": pa.FixedSizeListArray.from_arrays(pa.array(bits.view(np.float32)), 384)}
    )
    conn.register("special_vector", special)
    conn.execute("UPDATE historical SET v=s.v FROM special_vector s WHERE id=md5('1')")
    conn.execute("CHECKPOINT")
    assert codec_required(conn)
    conn.close()
    with sqlite3.connect(cfg.data_dir / "recall.fts.sqlite") as sql:
        sql.execute("CREATE TABLE pending_repair(id VARCHAR PRIMARY KEY)")
        sql.execute("INSERT INTO pending_repair VALUES ('historical-key')")
    return cfg


def test_codec_public_flow_refuses_writes_then_preserves_history(tmp_path: Path) -> None:
    cfg = legacy(tmp_path)
    side_hash = digest(cfg.data_dir / "recall.fts.sqlite")
    server = RpcServer(config=cfg)

    async def scenario():
        try:
            preview = await server._handle_migrate_storage({"dry_run": True}, None)
            assert preview["codec_eligible"] is True
            assert preview["desired_storage_version"] == STORAGE_VERSION
            with pytest.raises(RpcError) as denied:
                await server._writer_call(
                    lambda: server._get_conn().execute("DELETE FROM historical"), "ingest"
                )
            assert denied.value.data["reason"] == "codec_maintenance_required"
            accepted = await server._handle_migrate_storage({"plan_id": preview["plan_id"]}, None)
            assert accepted["accepted"] is True
            assert server._storage_maintenance_task is not None
            await asyncio.wait_for(server._storage_maintenance_task, 30)
            done = await server._handle_migrate_storage({"dry_run": True}, None)
            assert done["status"] == "succeeded", done
            assert done["codec_eligible"] is False
            assert server._get_conn().execute("SELECT count(*) FROM historical").fetchone() == (
                10000,
            )
            assert server._get_conn().execute("SELECT preserved(3)").fetchone() == (20,)
            persisted = (
                server._get_conn()
                .execute("SELECT v FROM historical WHERE id=md5('1')")
                .to_arrow_table()
                .column(0)
                .combine_chunks()
                .flatten()
                .view(pa.uint32())
                .to_pylist()
            )
            assert persisted[:3] == [0x80000000, 0x7FC00001, 0x80000000]
            with pytest.raises(duckdb.ConstraintException):
                server._get_conn().execute(
                    "INSERT INTO historical VALUES ('late', NULL, 'duplicate'), "
                    "('late', NULL, 'duplicate')"
                )
            assert digest(cfg.data_dir / "recall.fts.sqlite") == side_hash
            await server._writer_call(
                lambda: server._get_conn().execute(
                    "INSERT INTO historical VALUES ('late', NULL, 'new')"
                ),
                "append",
            )
            again = await server._handle_migrate_storage({}, None)
            assert again["accepted"] is False
            backups = list((cfg.data_dir / "snapshots").glob("storage-codec-*/manifest.json"))
            assert len(backups) == 1
        finally:
            await server.stop()

    asyncio.run(scenario())
    with connect(cfg) as conn:
        assert conn.execute("SELECT body FROM historical WHERE id='late'").fetchone() == ("new",)
        # Old writers cannot evade inspection with a completed marker.
        conn.execute("SET disabled_compression_methods=''")
        conn.execute("UPDATE historical SET body=body||'changed'")
        conn.execute("CHECKPOINT")
    journal = load_journal(cfg)
    assert journal is not None and journal.phase == "complete"


def test_codec_requires_verified_backup_and_recovers_prepublication(tmp_path: Path) -> None:
    cfg = legacy(tmp_path)
    with pytest.raises(RuntimeError, match="backup"):
        perform_codec_transition(cfg)
    with connect(cfg) as conn:
        journal = prepare_codec(conn, cfg, operation_id="operation", plan_id="plan")
    before = digest(cfg.db_path)
    backup = Path(journal.backup_path) / "recall.duckdb"
    original = backup.read_bytes()
    backup.write_bytes(b"corrupt backup")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        perform_codec_transition(cfg)
    assert digest(cfg.db_path) == before
    backup.write_bytes(original)
    candidate = Path(journal.candidate_path)
    candidate.write_bytes(b"interrupted candidate")
    perform_codec_transition(cfg)
    completed = load_journal(cfg)
    assert completed is not None and completed.phase == "complete"
    assert (
        candidate.parent / "attempt-1" / candidate.name
    ).read_bytes() == b"interrupted candidate"
    recover_codec(cfg)
    with connect(cfg) as conn:
        assert not codec_required(conn)
        assert conn.execute("SELECT count(*) FROM historical").fetchone() == (10000,)


def test_exact_validation_detects_float_payload_null_and_context_changes(tmp_path: Path) -> None:
    cfg = legacy(tmp_path)
    with connect(cfg) as conn:
        journal = prepare_codec(conn, cfg, operation_id="operation", plan_id="plan")
    perform_codec_transition(cfg)
    # Validator takes a same-basename candidate in a distinct directory.
    import shutil

    other = tmp_path / "other" / cfg.db_path.name
    other.parent.mkdir()
    shutil.copyfile(cfg.db_path, other)
    validate_candidate(cfg, other)
    with duckdb.connect(str(other)) as conn:
        conn.execute("SET disabled_compression_methods='roaring'")
        conn.execute("UPDATE historical SET body='different' WHERE id=md5('1')")
    with pytest.raises(RuntimeError, match="exact row mismatch"):
        validate_candidate(cfg, other)
    assert Path(journal.backup_path).exists()
    for bits in ([0, 0x80000000], [0x7FC00001, 0x7FC00002]):
        left = pa.array(np.array([bits[0]], dtype=np.uint32).view(np.float32))
        right = pa.array(np.array([bits[1]], dtype=np.uint32).view(np.float32))
        assert not _array_equal(left, right)
        assert _array_equal(left, left)
    assert not _array_equal(
        pa.array([[None]], type=pa.list_(pa.float32(), 1)),
        pa.array([None], type=pa.list_(pa.float32(), 1)),
    )


@pytest.mark.parametrize("phase", ["before_replace", "after_replace", "parent_fsync"])
def test_publication_failure_recovers_complete_pair(tmp_path, monkeypatch, phase):
    import os

    from recall.services import storage_codec

    cfg = legacy(tmp_path)
    with connect(cfg) as conn:
        original = prepare_codec(conn, cfg, operation_id="operation", plan_id="plan")
    real_replace = os.replace
    real_sync = storage_codec.fsync_path
    injected = False

    def replace_file(source, target):
        nonlocal injected
        if Path(target) == cfg.db_path:
            if phase == "before_replace":
                injected = True
                raise OSError("injected publication failure")
            real_replace(source, target)
            if phase == "after_replace":
                injected = True
                raise OSError("injected publication failure")
            return
        real_replace(source, target)

    def sync(path):
        nonlocal injected
        state = load_journal(cfg)
        if (
            phase == "parent_fsync"
            and state is not None
            and state.phase == "publishing"
            and Path(path) == cfg.db_path.parent
        ):
            injected = True
            raise OSError("injected publication failure")
        real_sync(path)

    with monkeypatch.context() as faults:
        faults.setattr(storage_codec.os, "replace", replace_file)
        faults.setattr(storage_codec, "fsync_path", sync)
        with pytest.raises(OSError, match="injected publication"):
            perform_codec_transition(cfg)
    assert injected
    state = load_journal(cfg)
    assert state is not None
    assert digest(cfg.db_path) in {original.source_sha256, state.candidate_sha256}
    recover_codec(cfg)
    if phase == "before_replace":
        assert digest(cfg.db_path) == original.source_sha256
        perform_codec_transition(cfg)
    completed = load_journal(cfg)
    assert completed is not None and completed.phase == "complete"
    with connect(cfg) as conn:
        assert conn.execute("SELECT count(*) FROM historical").fetchone() == (10000,)
        assert not codec_required(conn)


def test_recovery_never_initializes_missing_or_overwrites_later_progress(tmp_path):
    cfg = legacy(tmp_path)
    with connect(cfg) as conn:
        prepare_codec(conn, cfg, operation_id="operation", plan_id="plan")
    with duckdb.connect(str(cfg.db_path)) as conn:
        conn.execute("INSERT INTO historical VALUES ('later', NULL, 'durable progress')")
    later = digest(cfg.db_path)
    with pytest.raises(RuntimeError, match="authoritative database changed"):
        connect(cfg)
    assert digest(cfg.db_path) == later
    preserved = cfg.db_path.with_suffix(".preserved")
    cfg.db_path.rename(preserved)
    with pytest.raises(RuntimeError, match="database missing"):
        connect(cfg)
    assert not cfg.db_path.exists()


def test_space_admission_and_backup_retention(tmp_path, monkeypatch):
    import os
    import shutil

    from recall.services import storage_codec
    from recall.services.snapshots import gc_snapshots

    cfg = legacy(tmp_path)
    with connect(cfg) as conn:
        with monkeypatch.context() as faults:
            faults.setattr(
                storage_codec.shutil,
                "disk_usage",
                lambda path: shutil._ntuple_diskusage(100, 99, 1),
            )
            with pytest.raises(RuntimeError, match="insufficient"):
                prepare_codec(conn, cfg, operation_id="operation", plan_id="plan")
        assert not (cfg.data_dir / "snapshots").exists()
        journal = prepare_codec(conn, cfg, operation_id="operation", plan_id="plan")
    for entry in (cfg.data_dir / "snapshots").iterdir():
        os.utime(entry, (1, 1))
    gc_snapshots(cfg)
    assert Path(journal.backup_path).is_dir()
    assert load_journal(cfg) == journal


def test_original_index_job_finishes_before_codec_admission_fence(tmp_path):
    from recall.services.recreation import consistent_backup

    cfg = legacy(tmp_path)
    with connect(cfg) as conn:
        backup = consistent_backup(conn, cfg, kind="index-migration")
        conn.execute(
            "UPDATE index_migration_jobs SET phase='running', captured_count=1, backup_path=?",
            [str(backup)],
        )
    server = RpcServer(config=cfg)

    async def scenario():
        try:
            await server._writer_call(
                lambda: server._get_conn().execute(
                    "UPDATE index_migration_jobs SET phase='idle', completed_count=1"
                ),
                "finish original migration",
            )
            with pytest.raises(RpcError) as refused:
                await server._writer_call(
                    lambda: server._get_conn().execute("DELETE FROM historical"), "normal ingestion"
                )
            assert refused.value.data["reason"] == "codec_maintenance_required"
            assert server._get_conn().execute("SELECT count(*) FROM historical").fetchone() == (
                10000,
            )
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_committed_sqlite_wal_is_settled_and_preserved_before_backup(tmp_path):
    import signal
    import subprocess
    import sys

    cfg = legacy(tmp_path)
    sidecar = cfg.data_dir / "recall.fts.sqlite"
    writer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sqlite3,signal,sys; c=sqlite3.connect(sys.argv[1]); "
            "c.execute('PRAGMA journal_mode=WAL'); "
            "c.execute(\"INSERT INTO pending_repair VALUES ('durable-wal')\"); "
            "c.commit(); print('committed',flush=True); signal.pause()",
            str(sidecar),
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert writer.stdout is not None and writer.stdout.readline().strip() == "committed"
        writer.send_signal(signal.SIGKILL)
        assert writer.wait(timeout=10) == -signal.SIGKILL
    finally:
        if writer.poll() is None:
            writer.terminate()
            writer.wait(timeout=10)
    assert Path(f"{sidecar}-wal").stat().st_size > 0
    with connect(cfg) as conn:
        journal = prepare_codec(conn, cfg, operation_id="operation", plan_id="wal-plan")
    perform_codec_transition(cfg)
    with sqlite3.connect(sidecar) as conn:
        assert conn.execute("SELECT id FROM pending_repair ORDER BY id").fetchall() == [
            ("durable-wal",),
            ("historical-key",),
        ]
    assert journal.sidecar_sha256 == digest(sidecar)
