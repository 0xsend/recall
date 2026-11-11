"""Raw coordinator completion records only its captured and published scope."""

import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path

import duckdb
import pytest
from recall.db.fts_sidecar import sidecar_path
from recall.parsers.codex import CodexParser
from recall.services import indexer
from recall.services.coordinator import capture_path, observe_path, prepare_raw_sources
from recall.services.daemon import run_startup_fts_sidecar_sync_on
from recall.services.rpc_server import KeywordRetryGate, RpcServer

from tests.test_services.test_rpc_owned_boundary import server_at
from tests.test_services.test_run_completion import _spent_database


def _raw_setup(tmp_path: Path) -> tuple[Path, RpcServer, Path, CodexParser]:
    database = _spent_database(tmp_path)
    server = server_at(tmp_path)
    sync = server._database.run(
        "fixture:startup_sync",
        lambda lease: run_startup_fts_sidecar_sync_on(lease, server._config),
    )
    server._record_keyword_sync(sync, server._config)
    path = tmp_path / "rollout-source.jsonl"
    path.write_text(
        '{"type":"session_meta","payload":{"id":"source"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user","content":"hello"}}\n'
    )
    parser = CodexParser(roots=(tmp_path,))
    return database, server, path, parser


@pytest.mark.parametrize("failing", [False, True])
def test_actual_raw_completion_releases_only_a_settled_successful_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: bool
) -> None:
    database, server, path, parser = _raw_setup(tmp_path)
    item = server._database.call(
        "fixture:observe", lambda conn: observe_path(parser, capture_path(parser, path), conn=conn)
    )
    prepared = prepare_raw_sources((item,), {"codex": parser})
    assert prepared[0].error is None
    if failing:

        def fail(*args, **kwargs):
            raise RuntimeError("source write failed")

        monkeypatch.setattr(indexer, "insert_session", fail)

    async def scenario() -> None:
        try:
            await server._writer_call(
                lambda: server._commit_raw_sources(prepared, server._config),
                "raw_source_completion",
            )
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = fresh.execute(
            "SELECT last_successful_at,last_index_failed,last_failure_message FROM runtime_state"
        ).fetchone()
        assert row is not None
        successful, failed, reason = row
        assert failed == int(failing)
        if failing:
            assert successful == datetime(2026, 11, 1)
            assert reason
        else:
            assert successful > datetime(2026, 11, 1)
            assert reason is None
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (not failing, not failing)


def test_old_keyword_obligation_for_the_same_source_prevents_release(tmp_path: Path) -> None:
    database, server, path, parser = _raw_setup(tmp_path)

    def prepare():
        item = server._database.call(
            "fixture:observe",
            lambda conn: observe_path(parser, capture_path(parser, path), conn=conn),
        )
        return prepare_raw_sources((item,), {"codex": parser})

    async def scenario() -> None:
        try:
            prepared = prepare()
            holder = sqlite3.connect(
                sidecar_path(server._config.data_dir), isolation_level=None, timeout=0
            )
            try:
                holder.execute("BEGIN EXCLUSIVE")
                await server._writer_call(
                    lambda: server._commit_raw_sources(prepared, server._config), "deferred_source"
                )
            finally:
                holder.rollback()
                holder.close()
            prepared = prepare()
            await server._writer_call(
                lambda: server._commit_raw_sources(prepared, server._config), "unchanged_source"
            )
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        pending = fresh.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone()
        assert pending is not None and pending[0] > 0
        assert fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone() == (
            datetime(2026, 11, 1),
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)


def _prepare(server: RpcServer, parser: CodexParser, path: Path):
    item = server._database.call(
        "fixture:observe", lambda conn: observe_path(parser, capture_path(parser, path), conn=conn)
    )
    return prepare_raw_sources((item,), {"codex": parser})


def test_attributed_unrelated_keyword_backlog_does_not_block_a_successful_scope(
    tmp_path: Path,
) -> None:
    database, server, path, parser = _raw_setup(tmp_path)
    now = [0.0]
    server._keyword_gate = KeywordRetryGate(clock=lambda: now[0])
    other = path.with_name("rollout-other.jsonl")
    other.write_text(path.read_text().replace('"source"', '"other"'))

    async def scenario() -> None:
        try:
            first = _prepare(server, parser, path)
            holder = sqlite3.connect(
                sidecar_path(server._config.data_dir), isolation_level=None, timeout=0
            )
            try:
                holder.execute("BEGIN EXCLUSIVE")
                await server._writer_call(
                    lambda: server._commit_raw_sources(first, server._config), "deferred_source"
                )
            finally:
                holder.rollback()
                holder.close()
            now[0] = 10.0
            second = _prepare(server, parser, other)
            await server._writer_call(
                lambda: server._commit_raw_sources(second, server._config), "independent_source"
            )
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        assert fresh.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (1,)
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (True, True)
        row = fresh.execute(
            "SELECT last_successful_at,last_failure_message FROM runtime_state"
        ).fetchone()
        assert row is not None and row[0] > datetime(2026, 11, 1) and row[1] is None


def test_unknown_pending_membership_cannot_be_declared_unrelated(tmp_path: Path) -> None:
    database, server, path, parser = _raw_setup(tmp_path)
    server._database.call(
        "fixture:unattributed_intent",
        lambda conn: conn.execute(
            "INSERT INTO fts_sidecar_pending(kind,id,op) "
            "VALUES ('message','unknown-old-id','delete')"
        ).fetchone(),
    )
    prepared = _prepare(server, parser, path)

    async def scenario() -> None:
        try:
            await server._writer_call(
                lambda: server._commit_raw_sources(prepared, server._config), "source_completion"
            )
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        assert fresh.execute("SELECT id FROM fts_sidecar_pending").fetchall() == [
            ("unknown-old-id",)
        ]
        assert fresh.execute(
            "SELECT last_successful_at,last_failure_message FROM runtime_state"
        ).fetchone() == (
            datetime(2026, 11, 1),
            "unattributed keyword intents prevent proving the selected scope",
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)
