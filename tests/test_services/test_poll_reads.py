"""Real-store ownership, preservation and read-only search (REQ-CONC-009)."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from recall.core.poll_reads import poll_read_request
from recall.db import connect
from recall.db.fts_sidecar import (
    open_existing_sidecar_readonly,
    open_sidecar,
    set_fts_fields_signature,
    sidecar_path,
)
from recall.db.ownership import (
    DatabaseOwnershipBusyError,
    daemon_ownership,
    offline_ownership,
    read_existing_snapshot,
)
from recall.read_budget import ReadBudget
from recall.services.poll_reads import read_poll_snapshot

from tests.test_services.test_database_ownership import child, config_at, event, release
from tests.test_services.test_fts_sidecar_search import _seed_session


def snapshot(config, method="recall.stats", params=None, *, budget=None):
    request = poll_read_request(method, params or {})
    assert request is not None
    return read_poll_snapshot(config, request, budget or ReadBudget.start(30))


@pytest.fixture
def index(tmp_path):
    config = config_at(tmp_path)
    with connect(config) as conn:
        sidecar = open_sidecar(sidecar_path(config.data_dir))
        try:
            _seed_session(
                conn,
                sidecar,
                session_id="fixture",
                message_id="message",
                content="poll snapshot searchable phrase",
                tool_call_id="bash",
                bash_command="git status",
            )
            conn.execute(
                "INSERT INTO session_state(session_id,message_count,tool_count,is_complete,"
                "file_mtime,file_size) VALUES ('fixture',1,1,true,0,0)"
            )
            set_fts_fields_signature(sidecar, config.fts.fields)
            sidecar.commit()
        finally:
            sidecar.close()
    return config


def test_keyword_and_filtered_reads_preserve_persistent_stores(index):
    db_before = index.db_path.read_bytes()
    sidecar = sidecar_path(index.data_dir)
    fts_before = sidecar.read_bytes()
    for params in [
        {"query": "searchable", "mode": "keyword"},
        {"query": "searchable", "mode": "auto", "session": "fixture"},
        {"query": "git", "mode": "vector", "tool": "bash", "session": "fixture"},
    ]:
        rows = snapshot(index, "recall.search", params).data
        assert len(rows) == 1
        assert rows[0]["session_id"] == "fixture"
        assert rows[0]["lexical_match"] is True
    assert index.db_path.read_bytes() == db_before
    assert sidecar.read_bytes() == fts_before
    with offline_ownership(index.db_path):
        pass


@pytest.mark.parametrize("defect", ["missing", "corrupt", "version", "fields", "pending"])
def test_invalid_keyword_store_fails_without_repair(index, defect):
    path = sidecar_path(index.data_dir)
    if defect == "missing":
        path.unlink()
    elif defect == "corrupt":
        path.write_bytes(b"not a sqlite database")
    elif defect in {"version", "fields"}:
        with sqlite3.connect(path) as conn:
            key = "schema_version" if defect == "version" else "fts_fields_signature"
            conn.execute("UPDATE fts_sidecar_meta SET value='wrong' WHERE key=?", [key])
    else:
        with connect(index) as conn:
            conn.execute(
                "INSERT INTO fts_sidecar_pending(kind,id,op) VALUES ('message','message','upsert')"
            )
    before = path.read_bytes() if path.exists() else None
    db_before = index.db_path.read_bytes()
    with pytest.raises(RuntimeError, match=r"keyword|Keyword"):
        snapshot(index, "recall.search", {"query": "searchable", "mode": "keyword"})
    assert (path.read_bytes() if path.exists() else None) == before
    assert index.db_path.read_bytes() == db_before
    assert snapshot(index).data["sessions"] == 1
    assert snapshot(index, "recall.list").data[0]["id"] == "fixture"


def test_valid_empty_index_succeeds_without_creating_data(tmp_path):
    config = config_at(tmp_path)
    with connect(config):
        pass
    sidecar = open_sidecar(sidecar_path(config.data_dir))
    set_fts_fields_signature(sidecar, config.fts.fields)
    sidecar.commit()
    sidecar.close()
    result = snapshot(config, "recall.search", {"query": "absent", "mode": "keyword"})
    assert result.data == []
    assert result.last_successful_at is None


def test_missing_database_stays_missing(tmp_path):
    config = config_at(tmp_path)
    with pytest.raises(RuntimeError, match="database does not exist"):
        snapshot(config)
    assert not config.db_path.exists()
    assert not sidecar_path(config.data_dir).exists()


@pytest.mark.parametrize("alias", ["canonical", "symlink", "relative"])
def test_reader_loses_gate_without_opening_database(index, alias, monkeypatch):
    selected = index.db_path
    if alias == "symlink":
        selected = index.data_dir / "alias.duckdb"
        selected.symlink_to(index.db_path)
    elif alias == "relative":
        monkeypatch.chdir(index.data_dir)
        selected = Path(index.db_path.name)
    with offline_ownership(index.db_path), pytest.raises(DatabaseOwnershipBusyError):
        snapshot(replace(index, db_path=selected))
    assert snapshot(index).data["sessions"] == 1


def test_read_failure_releases_connection_and_ownership(index):
    def fail(conn):
        conn.execute("SELECT error('read failed')")

    with pytest.raises(duckdb.Error, match="read failed"):
        read_existing_snapshot(index, fail)
    with offline_ownership(index.db_path), connect(index) as conn:
        conn.execute("UPDATE runtime_state SET last_failure_message='writer recovered'")
    assert snapshot(index).data["sessions"] == 1


def test_expired_budget_opens_no_database(tmp_path):
    config = config_at(tmp_path)
    with pytest.raises(RuntimeError, match=r"timed out|cancelled"):
        snapshot(config, budget=ReadBudget.start(0))
    assert not config.db_path.exists()
    assert not Path(f"{config.db_path}.ownership").exists()


def test_tick_starting_mid_read_waits_then_persists(index):
    with child(
        """
import sys
from recall.core.config import AppConfig
from recall.db.ownership import read_existing_snapshot
config = AppConfig.load()
def read(conn):
    print('reader-open', flush=True)
    assert sys.stdin.readline().strip() == 'release'
    return conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]
assert read_existing_snapshot(config, read) == 1
print('reader-closed', flush=True)
""",
        index,
    ) as reader:
        assert event(reader) == "reader-open"

        async def tick():
            async with daemon_ownership(index.db_path):
                with connect(index) as conn:
                    conn.execute("INSERT INTO sessions VALUES ('next', 'codex', '/next', NULL)")

        async def run():
            task = asyncio.create_task(tick())
            await asyncio.sleep(0)  # let startup attempt its nonblocking gate
            assert not task.done()
            release(reader)
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(run())
        assert event(reader) == "reader-closed"
        assert reader.wait(timeout=5) == 0
    assert snapshot(index).data["sessions"] == 2


def test_killed_reader_releases_the_kernel_gate(index):
    with child(
        """
import sys
from recall.core.config import AppConfig
from recall.db.ownership import read_existing_snapshot
def read(conn):
    print('reader-open', flush=True)
    sys.stdin.readline()
read_existing_snapshot(AppConfig.load(), read)
""",
        index,
    ) as reader:
        assert event(reader) == "reader-open"
        reader.kill()
        assert reader.wait(timeout=5) < 0
    assert snapshot(index).data["sessions"] == 1


def test_existing_readonly_sidecar_handles_escaped_path_and_wal(tmp_path):
    path = tmp_path / "read ?#% index.sqlite"
    writer = open_sidecar(path)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO fts_sidecar_meta(key,value) VALUES ('wal-proof','committed')")
        writer.commit()
        reader = open_existing_sidecar_readonly(path)
        try:
            assert reader.execute(
                "SELECT value FROM fts_sidecar_meta WHERE key='wal-proof'"
            ).fetchone() == ("committed",)
            reader.execute("CREATE TEMP TABLE candidates(id TEXT)")
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                reader.execute("DELETE FROM fts_sidecar_meta")
        finally:
            reader.close()
    finally:
        writer.close()


def test_deadline_interrupts_duckdb_and_drains_before_writer(index, monkeypatch):
    import recall.services.poll_reads as reads

    def expensive(_config, conn):
        return conn.execute("SELECT SUM(i) FROM range(1000000000000) t(i)").fetchone()

    monkeypatch.setattr(reads, "read_overview", expensive)
    with pytest.raises((RuntimeError, duckdb.InterruptException)):
        snapshot(index, budget=ReadBudget.start(0.05))
    with offline_ownership(index.db_path), connect(index) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)


def test_vector_needs_no_sidecar_and_prepares_outside_ownership(index, monkeypatch):
    from tests.test_services.test_fts_sidecar_search import _vec

    with connect(index) as conn:
        conn.execute(
            "INSERT INTO message_embeddings(message_id,content_embedding) VALUES ('message',?)",
            [_vec(0)],
        )
    sidecar_path(index.data_dir).unlink()

    class Backend:
        query_prefix = "query: "

        def embed(self, texts):
            assert texts == ["query: semantic query"]
            with offline_ownership(index.db_path):
                pass
            return [_vec(0)]

    monkeypatch.setattr("recall.services.embeddings.get_backend", lambda _config: Backend())
    rows = snapshot(index, "recall.search", {"query": "semantic query", "mode": "vector"}).data
    assert [row["message_id"] for row in rows] == ["message"]
    assert rows[0]["lexical_match"] is False
    assert not sidecar_path(index.data_dir).exists()


def test_auto_revalidates_a_new_embedding_requirement(index, monkeypatch):
    import recall.services.poll_reads as reads

    from tests.test_services.test_fts_sidecar_search import _vec

    original = reads.read_existing_snapshot
    added = False

    def changed_between_snapshots(config, read):
        nonlocal added
        result = original(config, read)
        if not added:
            with offline_ownership(config.db_path), connect(config) as conn:
                conn.execute(
                    "INSERT INTO message_embeddings(message_id,content_embedding) "
                    "VALUES ('message',?)",
                    [_vec(0)],
                )
            added = True
        return result

    monkeypatch.setattr(reads, "read_existing_snapshot", changed_between_snapshots)
    with pytest.raises(RuntimeError, match="mode changed during preparation; retry"):
        snapshot(index, "recall.search", {"query": "searchable", "mode": "auto"})
    assert snapshot(index).data["sessions"] == 1


def test_newer_schema_is_read_only_refusal(index):
    from recall.core.schema_version import SchemaTooNewError

    with connect(index) as conn:
        conn.execute("INSERT INTO schema_version(version) VALUES (999)")
    before = index.db_path.read_bytes()
    with pytest.raises(SchemaTooNewError, match="Upgrade recall"):
        snapshot(index)
    assert index.db_path.read_bytes() == before


def test_unusable_gate_opens_no_index(tmp_path):
    from recall.db.ownership import DatabaseOwnershipError

    config = config_at(tmp_path)
    Path(f"{config.db_path}.ownership").mkdir()
    with pytest.raises(DatabaseOwnershipError, match="cannot open database ownership gate"):
        snapshot(config)
    assert not config.db_path.exists()


def test_show_cursor_errors_keep_rpc_classification_after_cleanup(index):
    from recall.core.rpc_types import INVALID_PARAMS, RpcError
    from recall.services.live import encode_cursor

    wrong = encode_cursor("another-session", 0, 0)
    with pytest.raises(RpcError) as error:
        snapshot(index, "recall.show", {"session_id": "fixture", "after": wrong})
    assert error.value.code == INVALID_PARAMS
    assert "cursor belongs to another session" in error.value.message
    with offline_ownership(index.db_path), connect(index) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
