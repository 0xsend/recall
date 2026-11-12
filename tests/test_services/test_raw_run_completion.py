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


@pytest.mark.parametrize("cancelled", [False, True])
def test_raw_global_producer_failure_records_its_outcome_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool
) -> None:
    from recall.services import index_migration

    database, server, path, parser = _raw_setup(tmp_path)
    prepared = _prepare(server, parser, path)
    origin = (
        asyncio.CancelledError("raw producer interrupted")
        if cancelled
        else ValueError("raw producer failed")
    )
    original = index_migration.migration_status
    failures = 0

    def fail_once(conn):
        nonlocal failures
        if failures == 0:
            failures += 1
            raise origin
        return original(conn)

    monkeypatch.setattr(index_migration, "migration_status", fail_once)

    async def scenario():
        try:
            with pytest.raises(type(origin)) as caught:
                await server._writer_call(
                    lambda: server._commit_raw_sources(prepared, server._config),
                    "raw global failure",
                )
            assert caught.value is origin
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = fresh.execute(
            "SELECT last_successful_at,last_failure_message FROM runtime_state"
        ).fetchone()
        assert row is not None and row[0] == datetime(2026, 11, 1)
        assert row[1] is not None and (
            row[1].startswith("cancelled:") if cancelled else row[1] == str(origin)
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)


@pytest.mark.parametrize("admitted_first", [False, True])
def test_raw_actor_departure_reaches_its_live_scope_before_worker_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, admitted_first: bool
) -> None:
    import threading

    from recall.services import coordinator, run_completion

    from tests.test_services.test_rpc_run_completion import _inventory_server

    database, server, path, parser = _inventory_server(tmp_path, monkeypatch)
    server._database.call(
        "fixture:observe", lambda conn: observe_path(parser, capture_path(parser, path), conn=conn)
    )
    entered = threading.Event()
    departed = threading.Event()
    release = threading.Event()
    original_record = coordinator.record_completion_on
    original_cancel = run_completion.RunScope.cancel

    def cancel(scope, reason):
        result = original_cancel(scope, reason)
        departed.set()
        return result

    def record(*args, **kwargs):
        if not admitted_first:
            entered.set()
            assert release.wait(5)
        else:

            def barrier(stage):
                if stage == "after_completion_history":
                    entered.set()
                    assert release.wait(5)

            kwargs["barrier"] = barrier
        return original_record(*args, **kwargs)

    monkeypatch.setattr(run_completion.RunScope, "cancel", cancel)
    monkeypatch.setattr(coordinator, "record_completion_on", record)

    async def scenario():
        task = asyncio.create_task(server._serve_raw_turn())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            assert await asyncio.to_thread(departed.wait, 5)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone()
        assert row is not None and (row[0] > datetime(2026, 11, 1)) == admitted_first
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (admitted_first, admitted_first)


@pytest.mark.parametrize(
    "stage", ["before_admission", "after_completion_history", "after_completion_commit"]
)
def test_raw_shutdown_preserves_acknowledged_data_without_uncommitted_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from recall.services import coordinator

    database, server, path, parser = _raw_setup(tmp_path)
    prepared = _prepare(server, parser, path)
    original = coordinator.record_completion_on

    def record(*args, **kwargs):
        if stage == "before_admission":
            server._shutdown_event.set()

        def stop(at):
            if at == stage:
                server._shutdown_event.set()

        return original(*args, **kwargs, barrier=stop)

    monkeypatch.setattr(coordinator, "record_completion_on", record)

    async def scenario():
        try:
            await server._writer_call(
                lambda: server._commit_raw_sources(prepared, server._config), "raw shutdown"
            )
        except Exception:
            if stage == "after_completion_commit":
                raise
        finally:
            await server.stop()

    asyncio.run(scenario())
    committed = stage == "after_completion_commit"
    with duckdb.connect(str(database), read_only=True) as fresh:
        assert fresh.execute("SELECT count(*) FROM messages").fetchone() == (1,)
        assert fresh.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (
            0 if committed else 1,
        )
        row = fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone()
        assert row is not None and (row[0] > datetime(2026, 11, 1)) == committed
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (committed, committed)


@pytest.mark.parametrize("diagnostics", [False, True])
def test_raw_reporting_failure_keeps_original_and_never_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, diagnostics: bool
) -> None:
    from recall.services import coordinator, index_migration, run_completion

    database, server, path, parser = _raw_setup(tmp_path)
    prepared = _prepare(server, parser, path)
    origin = ValueError("raw producer failed before completion")
    reports = diagnostic_attempts = 0
    record = coordinator.record_completion_on

    def producer_failed(conn):
        raise origin

    def unavailable(*args, **kwargs):
        nonlocal reports
        reports += 1

        def fault(stage):
            if stage == "after_completion_history":
                args[0].call(
                    "injected:raw_reporting",
                    lambda conn: conn.execute("SELECT CAST('unavailable' AS INT)").fetchone(),
                )

        return record(*args, **kwargs, barrier=fault)

    def diagnostic_unavailable(*args, **kwargs):
        nonlocal diagnostic_attempts
        diagnostic_attempts += 1
        raise OSError("outcome diagnostic unavailable")

    monkeypatch.setattr(index_migration, "migration_status", producer_failed)
    monkeypatch.setattr(coordinator, "record_completion_on", unavailable)
    if diagnostics:
        monkeypatch.setattr(run_completion.logger, "error", diagnostic_unavailable)

    async def scenario():
        try:
            with pytest.raises(ValueError) as caught:
                await server._writer_call(
                    lambda: server._commit_raw_sources(prepared, server._config), "raw reporting"
                )
            assert caught.value is origin
        finally:
            await server.stop()

    asyncio.run(scenario())
    assert reports == 1 and diagnostic_attempts == int(diagnostics)
    with duckdb.connect(str(database), read_only=True) as fresh:
        assert fresh.execute(
            "SELECT last_successful_at,last_failure_message FROM runtime_state"
        ).fetchone() == (datetime(2026, 11, 1), None)
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)
