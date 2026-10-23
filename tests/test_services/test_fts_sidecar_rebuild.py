from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from recall.core.config import AppConfig, FtsConfig
from recall.core.rpc_client import RpcClient, RpcConnectionError
from recall.db.connection import advisory_lock, connect
from recall.db.fts_sidecar import open_sidecar, upsert_message_fts
from recall.services.fts_sidecar_rebuild import KeywordRebuild, file_hash


def _operation(tmp_path: Path) -> KeywordRebuild:
    data = tmp_path / "data"
    data.mkdir()
    config = replace(
        AppConfig.load(),
        data_dir=data,
        db_path=data / "recall.duckdb",
        lock_path=data / "recall.lock",
        fts=FtsConfig(),
    )
    conn = connect(config)
    conn.execute(
        "INSERT INTO message_state(message_id, role, content, fts_content, fts_thinking) "
        "VALUES ('m', 'assistant', 'preserved raw', 'needle alpha', '')"
    )
    conn.execute(
        "INSERT INTO message_state(message_id, role, content, fts_content) "
        "VALUES ('empty', 'assistant', 'preserved empty', '')"
    )
    conn.close()
    sidecar = open_sidecar(data / "recall.fts.sqlite")
    upsert_message_fts(sidecar, "stale", "stale keyword", "")
    sidecar.close()
    return KeywordRebuild(config, tmp_path / "rebuild", "declared-test-build")


def test_canonical_install_and_retained_rollback_before_release(tmp_path: Path) -> None:
    operation = _operation(tmp_path)
    with advisory_lock(operation.config.lock_path):
        assert operation.inspect() is None
        captured = operation.begin()
        with pytest.raises(RuntimeError, match="maintenance is held"):
            connect(operation.config)
        with pytest.raises(RpcConnectionError, match="maintenance is held"):
            RpcClient(operation.config).connect()
        built = operation.build()
        assert built.state == "verified"
        assert file_hash(operation.sidecar) == captured.original_hash
        assert file_hash(operation.config.db_path) == captured.database_hash
        installed = operation.install()
        assert installed.state == "installed"
        assert operation.install() == installed
        with sqlite3.connect(operation.sidecar) as reader:
            assert reader.execute(
                "SELECT message_id FROM message_fts_rowid ORDER BY message_id"
            ).fetchall() == [("empty",), ("m",)]
            assert reader.execute(
                "SELECT count(*) FROM message_fts WHERE message_fts MATCH 'needle'"
            ).fetchone() == (1,)
            assert reader.execute(
                "SELECT count(*) FROM message_fts WHERE message_fts MATCH 'stale'"
            ).fetchone() == (0,)
        assert operation.rollback().state == "rolled_back"
        assert file_hash(operation.sidecar) == captured.original_hash
        assert operation.finish().state == "released"
        assert operation.finish().state == "released"
        assert not operation.marker.exists()
        with pytest.raises(RuntimeError, match="begin an explicit"):
            operation.rollback()
        assert (Path(captured.backup) / "recall.duckdb").exists()


def test_candidate_or_snapshot_changes_refuse_install(tmp_path: Path) -> None:
    operation = _operation(tmp_path)
    with advisory_lock(operation.config.lock_path):
        captured = operation.begin()
        operation.build()
        with operation.candidate.open("ab") as handle:
            handle.write(b"corrupted after verification")
        with pytest.raises(RuntimeError, match="candidate changed"):
            operation.install()
        assert file_hash(operation.sidecar) == captured.original_hash
        changed = replace(operation, build_id="other-build")
        with pytest.raises(RuntimeError, match="differs from captured identity"):
            changed.build()
