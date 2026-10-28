from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig, FtsConfig
from recall.core.rpc_client import RpcClient, RpcConnectionError
from recall.db.connection import advisory_lock, connect, connect_existing
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
        with pytest.raises(RuntimeError, match=r"released; choose a fresh operation"):
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


def test_partial_build_can_roll_back_and_release(tmp_path: Path) -> None:
    operation = _operation(tmp_path)
    captured = operation.begin()
    assert operation.rollback().state == "rolled_back"
    assert file_hash(operation.sidecar) == captured.original_hash
    assert operation.finish().state == "released"
    with connect(operation.config) as source:
        assert source.execute("SELECT count(*) FROM message_state").fetchone() == (2,)


@pytest.mark.parametrize("sidecar_changed", [False, True])
def test_interrupted_abort_rechecks_original_before_releasing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sidecar_changed: bool
) -> None:
    operation = _operation(tmp_path)
    captured = operation.begin()

    def interrupt_release(self: KeywordRebuild) -> None:
        raise InterruptedError("interrupted before marker removal")

    with monkeypatch.context() as patch:
        patch.setattr(KeywordRebuild, "_release_marker", interrupt_release)
        with pytest.raises(InterruptedError):
            operation.abort()
    manifest = operation.inspect()
    assert manifest is not None and manifest.state == "aborted"
    assert operation.marker.exists()
    if sidecar_changed:
        with sqlite3.connect(operation.sidecar) as sidecar:
            sidecar.execute("DELETE FROM message_fts")
            sidecar.execute("DELETE FROM message_fts_rowid")
        changed_hash = file_hash(operation.sidecar)
        assert changed_hash != captured.original_hash
        with pytest.raises(RuntimeError, match=r"original sidecar.*use replace"):
            operation.abort()
        assert operation.marker.read_text() == str(operation.operation_dir.resolve())
        assert file_hash(operation.sidecar) == changed_hash
        replacement = replace(operation, operation_dir=tmp_path / "replacement")
        replacement.replace(operation.operation_dir)
        replacement.build()
        replacement.install()
        replacement.finish()
        with sqlite3.connect(operation.sidecar) as sidecar:
            assert sidecar.execute(
                "SELECT count(*) FROM message_fts WHERE message_fts MATCH 'needle'"
            ).fetchone() == (1,)
    else:
        assert operation.abort().state == "aborted"
        assert file_hash(operation.sidecar) == captured.original_hash
    assert not operation.marker.exists()
    assert file_hash(operation.config.db_path) == captured.database_hash
    assert file_hash(Path(captured.backup) / "recall.fts.sqlite") == captured.original_hash


@pytest.mark.parametrize("action", ["abort", "finish"])
def test_terminal_operation_with_held_marker_names_working_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    operation = _operation(tmp_path)
    operation.begin()
    if action == "finish":
        operation.rollback()

    def interrupt_release(self: KeywordRebuild) -> None:
        raise InterruptedError("interrupted before marker removal")

    with monkeypatch.context() as patch:
        patch.setattr(KeywordRebuild, "_release_marker", interrupt_release)
        with pytest.raises(InterruptedError):
            getattr(operation, action)()
    with pytest.raises(RuntimeError, match=f"retry {action}.*use replace"):
        operation.begin()
    getattr(operation, action)()
    assert not operation.marker.exists()
    with pytest.raises(RuntimeError, match="choose a fresh operation directory"):
        operation.begin()
    fresh = replace(operation, operation_dir=tmp_path / "fresh")
    assert fresh.begin().state == "building"
    fresh.abort()


@pytest.mark.parametrize("drift", ["database", "wal"])
def test_untouched_original_can_abort_after_source_drift(tmp_path: Path, drift: str) -> None:
    operation = _operation(tmp_path)
    captured = operation.begin()
    with ExitStack() as stack:
        source = stack.enter_context(connect_existing(operation.config, read_only=False))
        source.execute("UPDATE message_state SET fts_content='current needle'")
        if drift == "database":
            source.close()
        else:
            assert Path(str(operation.config.db_path) + ".wal").stat().st_size > 0
        with pytest.raises(RuntimeError, match="canonical database"):
            operation.build()
        assert operation.abort().state == "aborted"
        assert operation.abort().state == "aborted"
    assert file_hash(operation.sidecar) == captured.original_hash
    assert not operation.marker.exists()
    with pytest.raises(RuntimeError, match=r"aborted; choose a fresh operation"):
        operation.begin()
    assert Path(captured.backup).is_dir()
    with connect(operation.config) as source:
        assert source.execute("SELECT count(*) FROM message_state").fetchone() == (2,)


@pytest.mark.parametrize("sidecar_changed", [False, True])
@pytest.mark.parametrize("source_changed", [False, True])
def test_replacement_reconstructs_current_generation(
    tmp_path: Path, sidecar_changed: bool, source_changed: bool
) -> None:
    operation = _operation(tmp_path)
    captured = operation.begin()
    operation.build()
    operation.install()
    if sidecar_changed:
        # Prior binaries can delete without invoking the insertion guards.
        with sqlite3.connect(operation.sidecar) as sidecar:
            sidecar.execute(
                "DELETE FROM message_fts WHERE rowid IN (SELECT rowid FROM message_fts_rowid)"
            )
            sidecar.execute("DELETE FROM message_fts_rowid")
    if source_changed:
        with duckdb.connect(str(operation.config.db_path)) as source:
            source.execute(
                "UPDATE message_state SET fts_content='replacement needle' WHERE message_id='m'"
            )
    current_database = file_hash(operation.config.db_path)
    if source_changed or sidecar_changed:
        with pytest.raises(RuntimeError, match=r"canonical database|sidecar changed"):
            operation.rollback()
    with pytest.raises(RuntimeError, match="original sidecar"):
        operation.abort()
    replacement = replace(operation, operation_dir=tmp_path / "replacement")
    with pytest.raises(RuntimeError, match="another keyword maintenance"):
        replacement.begin()
    replacement.replace(operation.operation_dir)
    # Retrying a transferred operation converges without overwriting its capture.
    assert replacement.replace(operation.operation_dir) == replacement.inspect()
    with pytest.raises(RuntimeError, match="another operation"):
        operation.build()
    replacement.build()
    replacement.install()
    replacement.finish()
    assert file_hash(operation.config.db_path) == current_database
    assert file_hash(Path(captured.backup) / "recall.fts.sqlite") == captured.original_hash
    with sqlite3.connect(operation.sidecar) as sidecar:
        assert sidecar.execute(
            "SELECT count(*) FROM message_fts WHERE message_fts MATCH 'needle'"
        ).fetchone() == (1,)


def test_replacement_requires_the_named_owner(tmp_path: Path) -> None:
    operation = _operation(tmp_path)
    operation.begin()
    replacement = replace(operation, operation_dir=tmp_path / "replacement")
    with pytest.raises(RuntimeError, match="named previous operation"):
        replacement.replace(tmp_path / "wrong-owner")
    assert operation.marker.read_text() == str(operation.operation_dir.resolve())


def test_capture_records_snapshot_and_pending_identity(tmp_path: Path) -> None:
    operation = _operation(tmp_path)
    with duckdb.connect(str(operation.config.db_path)) as source:
        source.execute(
            "INSERT INTO sessions(id, source, source_path) "
            "VALUES ('s', 'codex', '/fixture/session')"
        )
        source.execute(
            "INSERT INTO fts_sidecar_pending "
            "VALUES ('message', 'm', 'upsert', '2026-10-23 01:00:00')"
        )
    captured = operation.begin()
    assert captured.format_version == 2
    identity = captured.capture
    assert identity is not None
    assert identity.database_bytes == (Path(captured.backup) / "recall.duckdb").stat().st_size
    assert identity.sidecar_bytes == operation.sidecar.stat().st_size
    assert identity.source_sessions == {"codex": 1}
    assert identity.pending_count == 1
    assert (
        identity.pending_sha256
        == hashlib.sha256(b'["message", "m", "upsert", "2026-10-23 01:00:00"]\n').hexdigest()
    )
    operation.abort()


def test_format_one_operation_remains_resumable(tmp_path: Path) -> None:
    operation = _operation(tmp_path)
    captured = operation.begin()
    legacy_manifest = captured.model_dump(exclude={"capture", "replaces_operation"})
    legacy_manifest["format_version"] = 1
    operation.manifest_path.write_text(json.dumps(legacy_manifest))
    assert operation.build().state == "verified"
    operation.install()
    operation.rollback()
    operation.finish()
    assert file_hash(operation.sidecar) == captured.original_hash
