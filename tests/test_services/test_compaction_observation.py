from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb
import pytest
import recall.services.compaction as compaction
from recall.core.config import CompactionConfig
from recall.services.rpc_server import RpcServer
from test_daemon_compaction import _config, _connect_schema


@pytest.mark.parametrize("entry", ["monitor", "once"])
def test_negative_observation_keeps_readers_and_writers_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        compaction=CompactionConfig(check_interval_hours=0, bloat_ratio_threshold=999),
    )
    conn = _connect_schema(config)
    conn.close()
    server = RpcServer(config=config)
    entered, release = threading.Event(), threading.Event()
    actual = compaction._estimate_live_bytes

    def estimate(cursor, block_size):
        entered.set()
        assert release.wait(10), "metadata barrier was not released"
        return actual(cursor, block_size)

    monkeypatch.setattr(compaction, "_estimate_live_bytes", estimate)

    async def run() -> None:
        original = server._get_conn()
        original.execute("UPDATE runtime_state SET last_failure_message = 'before'")
        wal = config.db_path.with_suffix(".duckdb.wal")
        wal_before = wal.read_bytes()
        tick = asyncio.create_task(
            server._compact_monitor()
            if entry == "monitor"
            else server._handle_daemon_run({"once": True, "embed": False}, None)
        )
        read = write = None
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            read = asyncio.create_task(
                server._run_readonly(lambda cursor: cursor.execute("SELECT 42").fetchone())
            )
            write = asyncio.create_task(
                server._writer_call(
                    lambda: server._get_conn().execute(
                        "UPDATE runtime_state SET last_failure_message = 'during'"
                    ),
                    "observation regression",
                    checkpoint_wakeup=False,
                )
            )
            completed, _ = await asyncio.wait([read, write], timeout=1)
            progress = len(completed) == 2
        finally:
            release.set()
            if entry == "monitor":
                server.request_shutdown()
            await asyncio.wait_for(tick, 10)
            if read is not None and write is not None:
                await asyncio.gather(read, write)
        try:
            assert progress, "a negative check excluded independent reads or a committed writer"
            assert server._conn is original
            assert wal.exists() and wal.stat().st_size >= len(wal_before)
            assert original.execute(
                "SELECT last_failure_message FROM runtime_state"
            ).fetchone() == ("during",)
            assert not list(config.data_dir.glob("*.compact*"))
        finally:
            await server.stop()

    asyncio.run(run())


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("entry", ["monitor", "once"])
@pytest.mark.parametrize(
    "failure_site", ["operation", "rollback", "cursor_close", "cursor_close_fatal"]
)
def test_fatal_metadata_observation_stops_and_persists_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str, forced: bool, failure_site: str
) -> None:
    config = _config(tmp_path, monkeypatch, compaction=CompactionConfig(check_interval_hours=0))
    conn = _connect_schema(config)
    conn.close()
    server = RpcServer(config=config)
    server._get_conn()
    server._compact_unsuccessful_checks = 3 if forced else 0
    calls = 0
    fault = threading.local()
    real_open = server._open_conn_unlocked

    class Cursor:
        def __init__(self, cursor: duckdb.DuckDBPyConnection) -> None:
            self.cursor = cursor

        def execute(self, sql: str, *args: Any) -> duckdb.DuckDBPyConnection:
            result = self.cursor.execute(sql, *args)
            if sql == "ROLLBACK" and getattr(fault, "rollback", False):
                fault.rollback = False
                raise duckdb.FatalException(
                    "FATAL Error: metadata observation invalidated database"
                )
            return result

        def close(self) -> None:
            fail = getattr(fault, "close", "")
            fault.close = ""
            fault.rollback = False
            self.cursor.close()
            if fail == "cursor_close_fatal":
                raise duckdb.FatalException(
                    "FATAL Error: metadata observation invalidated database"
                )
            if fail:
                raise RuntimeError("cursor cleanup failed")

        def __getattr__(self, name: str) -> Any:
            return getattr(self.cursor, name)

    class Connection:
        def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
            self.connection = connection

        def cursor(self) -> Cursor:
            return Cursor(self.connection.cursor())

        def __getattr__(self, name: str) -> Any:
            return getattr(self.connection, name)

    # SQL, transactions and close use real DuckDB. Only the cleanup result is
    # induced; native engine invalidation itself is not simulated by this test.
    monkeypatch.setattr(server, "_open_conn_unlocked", lambda: Connection(real_open()))

    def fatal(cursor, block_size):
        nonlocal calls
        calls += 1
        fault.close = failure_site if failure_site.startswith("cursor_close") else ""
        if failure_site in ("rollback", "cursor_close_fatal"):
            fault.rollback = failure_site == "rollback"
            cursor.execute("SELECT error('metadata observation query failed')")
            raise AssertionError("the real DuckDB query must fail")
        raise duckdb.FatalException("FATAL Error: metadata observation invalidated database")

    monkeypatch.setattr(compaction, "_estimate_live_bytes", fatal)

    async def run() -> None:
        server._last_bloat_ratio = 7.0
        try:
            if entry == "once":
                with pytest.raises(duckdb.FatalException):
                    await server._handle_daemon_run({"once": True, "embed": False}, None)
            else:
                monitor = asyncio.create_task(server._compact_monitor())
                try:
                    await asyncio.wait_for(server._shutdown_event.wait(), 1)
                finally:
                    server.request_shutdown()
                    await monitor
            assert calls == 1
            assert server._shutdown_event.is_set()
            assert server._last_bloat_ratio == 7.0
            marker = config.data_dir / "daemon-failure.json"
            assert "metadata observation invalidated database" in marker.read_text()
        finally:
            await server.stop()

    asyncio.run(run())


class _MetadataBarrier:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from queue import Queue

        self.entered: Queue[int] = Queue()
        self.release = threading.Semaphore(0)
        self.calls = 0
        actual = compaction._estimate_live_bytes

        def estimate(cursor, block_size):
            self.calls += 1
            self.entered.put(self.calls)
            assert self.release.acquire(timeout=10), "metadata barrier not released"
            return actual(cursor, block_size)

        monkeypatch.setattr(compaction, "_estimate_live_bytes", estimate)

    async def next(self) -> int:
        return await asyncio.to_thread(self.entered.get, True, 10)


@pytest.mark.parametrize("writer", ["direct", "non_enrichment", "failed", "checkpoint"])
def test_interference_defers_once_per_due_check_then_forces_with_continuous_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
) -> None:
    import time
    from types import SimpleNamespace

    config = _config(
        tmp_path,
        monkeypatch,
        compaction=CompactionConfig(
            check_interval_hours=1,
            bloat_ratio_threshold=999,
        ),
    )
    conn = _connect_schema(config)
    conn.close()
    server = RpcServer(config=config)
    barrier = _MetadataBarrier(monkeypatch)
    now = 100.0
    monkeypatch.setattr(
        "recall.services.rpc_server.time",
        SimpleNamespace(
            monotonic=lambda: now,
            time=time.time,
        ),
    )

    async def run() -> None:
        original = server._get_conn()
        server._last_bloat_ratio = 99
        tick = None
        scan = None
        old_reader = None
        reader_entered, reader_release = threading.Event(), threading.Event()
        try:
            for check in range(3):
                tick = asyncio.create_task(server._maybe_run_auto_compact())
                for _attempt in range(3):
                    await barrier.next()
                    assert await server._run_readonly(
                        lambda cursor: cursor.execute("SELECT 42").fetchone()
                    ) == (42,)

                    def mutate() -> None:
                        original.execute("UPDATE runtime_state SET last_failure_message = 'kept'")
                        if writer == "failed":
                            raise ValueError("failed after a committed statement")
                        if writer == "checkpoint":
                            try:
                                original.execute("CHECKPOINT")
                            except duckdb.TransactionException as err:
                                assert "other write transactions active" in str(err)

                    if writer in {"direct", "checkpoint"}:
                        async with server._write_lock:
                            await server._await_shared_conn_work(
                                asyncio.get_running_loop().run_in_executor(None, mutate),
                                "direct",
                            )
                    elif writer == "failed":
                        with pytest.raises(ValueError, match="committed"):
                            await server._writer_call(mutate, "failed", checkpoint_wakeup=False)
                    else:
                        await server._writer_call(
                            mutate, writer, affects_enrichment=False, checkpoint_wakeup=False
                        )
                    barrier.release.release()
                await tick
                assert server._last_bloat_ratio == 99
                assert server._compact_unsuccessful_checks == check + 1
                # A not-due caller neither counts nor clears a failure.
                await server._maybe_run_auto_compact()
                assert server._compact_unsuccessful_checks == check + 1
                now_check = check + 1
                # The closure's clock advances only between completed due checks.
                advance(now_check)
            await server._writer_call(
                lambda: original.execute("UPDATE runtime_state SET last_failure_message='dirty'"),
                "dirty WAL",
                checkpoint_wakeup=False,
            )
            wal = config.db_path.with_suffix(".duckdb.wal")
            wal_before = wal.read_bytes()

            def hold_reader(cursor):
                value = cursor.execute("SELECT count(*) FROM runtime_state").fetchone()
                reader_entered.set()
                assert reader_release.wait(10)
                return value

            old_reader = asyncio.create_task(server._run_readonly(hold_reader))
            assert await asyncio.to_thread(reader_entered.wait, 10)
            # Scan capacity stays occupied throughout the forced observation.
            scan = await server._database_admission.acquire(writer=False, scan=True)
            tick = asyncio.create_task(server._maybe_run_auto_compact())
            assert await barrier.next() == 10
            assert server._write_lock.locked()
            assert not server._index_request_lock.locked()
            for _ in range(3):
                status = await server._handle_daemon_status({}, None)
                assert status["bloat_ratio"] == 99
                await server._handle_live({"fresh": False}, None)
                assert await server._run_readonly(
                    lambda cursor: cursor.execute("SELECT count(*) FROM runtime_state").fetchone()
                ) == (1,)
            published = []

            async def waiting_index() -> None:
                async with server._index_turn():
                    await server._writer_call(
                        lambda: published.append(server._last_bloat_ratio),
                        "queued index",
                        checkpoint_wakeup=False,
                    )

            index = asyncio.create_task(waiting_index())
            await asyncio.sleep(0)
            assert not index.done()
            barrier.release.release()
            await tick
            await index
            assert not old_reader.done()
            reader_release.set()
            assert await old_reader == (1,)
            assert len(published) == 1 and published[0] != 99
            assert server._compact_unsuccessful_checks == 0
            assert server._conn is original
            assert wal.read_bytes() == wal_before
            assert original.execute(
                "SELECT last_failure_message FROM runtime_state"
            ).fetchone() == ("dirty",)
            assert not list(config.data_dir.glob("*.compact*"))
            assert not list(config.data_dir.glob("*.pre-compact*"))
        finally:
            for _ in range(12):
                barrier.release.release()
            reader_release.set()
            if old_reader is not None:
                await old_reader
            if scan is not None:
                server._database_admission.release(scan)
            if tick is not None:
                await tick
            await server.stop()

    def advance(check: int) -> None:
        nonlocal now
        now = 100.0 + check * 3600

    asyncio.run(run())


@pytest.mark.parametrize("forced", [False, True])
def test_joiner_cancellation_does_not_abandon_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forced: bool
) -> None:
    config = _config(tmp_path, monkeypatch, compaction=CompactionConfig())
    conn = _connect_schema(config)
    conn.close()
    server = RpcServer(config=config)
    barrier = _MetadataBarrier(monkeypatch)

    async def run() -> None:
        server._compact_unsuccessful_checks = 3 if forced else 0
        first = asyncio.create_task(server._maybe_run_auto_compact())
        second = None
        try:
            await barrier.next()
            second = asyncio.create_task(server._maybe_run_auto_compact())
            await asyncio.sleep(0)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert server._compact_check_task is not None
            assert not server._compact_check_task.done()
            assert server._write_lock.locked() == forced
            barrier.release.release()
            await second
            assert barrier.calls == 1
            assert server._last_bloat_ratio is not None
            assert server._compact_unsuccessful_checks == 0
        finally:
            barrier.release.release()
            if second is not None:
                await second
            await server.stop()

    asyncio.run(run())


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("fatal", [False, True])
def test_shutdown_drains_observation_and_fatal_cleanup_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forced: bool, fatal: bool
) -> None:
    config = _config(tmp_path, monkeypatch)
    conn = _connect_schema(config)
    conn.close()
    server = RpcServer(config=config)
    entered, release = threading.Event(), threading.Event()

    def estimate(cursor, block_size):
        entered.set()
        assert release.wait(10)
        if fatal:
            raise duckdb.FatalException("FATAL Error: invalidation during cursor cleanup")
        return 1

    monkeypatch.setattr(compaction, "_estimate_live_bytes", estimate)

    async def run() -> None:
        server._last_bloat_ratio = 99
        server._compact_unsuccessful_checks = 3 if forced else 0
        original = server._get_conn()
        joined = asyncio.create_task(server._maybe_run_auto_compact())
        stop = None
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            stop = asyncio.create_task(server.stop())
            await asyncio.sleep(0)
            assert not stop.done()
            assert server._conn is original
            assert server._write_lock.locked() == forced
            release.set()
            await stop
            result = await asyncio.gather(joined, return_exceptions=True)
            assert isinstance(result[0], duckdb.FatalException if fatal else asyncio.CancelledError)
            assert server._last_bloat_ratio == 99
            if fatal:
                assert (
                    "invalidation during cursor cleanup"
                    in (config.data_dir / "daemon-failure.json").read_text()
                )
            assert server._conn is None
            assert not server._read_executions
            assert not server._write_lock.locked()
        finally:
            release.set()
            if stop is not None:
                await stop
            await asyncio.gather(joined, return_exceptions=True)

    asyncio.run(run())


def test_tiny_database_refreshes_cache_without_compaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch, compaction=CompactionConfig(bloat_ratio_threshold=0))
    conn = _connect_schema(config)
    conn.close()
    server = RpcServer(config=config)

    async def run() -> None:
        original = server._get_conn()
        try:
            await server._maybe_run_auto_compact()
            assert server._last_bloat_ratio is not None
            assert server._conn is original
            assert not config.db_path.with_suffix(".duckdb.pre-compact").exists()
            assert not (config.data_dir / "recall.compacting").exists()
        finally:
            await server.stop()

    asyncio.run(run())


def test_failed_forced_check_retries_forced_and_success_resets_streak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    conn = _connect_schema(config)
    conn.close()
    server = RpcServer(config=config)
    actual = compaction._estimate_live_bytes
    ownership = []

    def fail(cursor, block_size):
        ownership.append(server._write_lock.locked())
        raise duckdb.InvalidInputException("recoverable metadata failure")

    monkeypatch.setattr(compaction, "_estimate_live_bytes", fail)

    async def run() -> None:
        original = server._get_conn()
        server._last_bloat_ratio = 99
        try:
            for _ in range(5):
                server.last_compact_check_at = None
                await server._maybe_run_auto_compact()
                assert server._last_bloat_ratio == 99
            assert ownership == [False, False, False, True, True]
            assert server._conn is original
            assert server._compact_unsuccessful_checks == 3
            disabled = replace(config, compaction=replace(config.compaction, auto_trigger=False))
            await server._maybe_run_auto_compact(disabled)
            assert server._compact_unsuccessful_checks == 3
            await server._recover_shared_conn_locked(ValueError("no native error"), "recovery")
            assert server._compact_unsuccessful_checks == 3
            monkeypatch.setattr(compaction, "_estimate_live_bytes", actual)
            server.last_compact_check_at = None
            await server._maybe_run_auto_compact()
            assert server._compact_unsuccessful_checks == 0
            assert server._last_bloat_ratio != 99
            assert server._conn is original
        finally:
            await server.stop()

    asyncio.run(run())


def test_probe_budget_does_not_own_or_cancel_a_queued_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    conn = _connect_schema(config)
    conn.close()
    server = RpcServer(config=config)
    monkeypatch.setattr("recall.services.rpc_server._COMPACT_OBSERVATION_SECONDS", 0.01)

    async def run() -> None:
        server._last_bloat_ratio = 99
        try:
            async with server._write_lock:
                await server._maybe_run_auto_compact()
                assert server._last_bloat_ratio == 99
                assert server._compact_unsuccessful_checks == 1
            monkeypatch.setattr("recall.services.rpc_server._COMPACT_OBSERVATION_SECONDS", 30.0)
            server.last_compact_check_at = None
            await server._maybe_run_auto_compact()
            assert server._last_bloat_ratio != 99
            assert server._compact_unsuccessful_checks == 0
        finally:
            await server.stop()

    asyncio.run(run())


@pytest.mark.parametrize("forced", [False, True])
def test_positive_handoff_copies_intervening_committed_content_and_drains_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forced: bool
) -> None:
    from test_daemon_compaction import _seed_single_row_group, _single_row_group_config

    config = _single_row_group_config(tmp_path, monkeypatch)
    _seed_single_row_group(config)
    server = RpcServer(config=config)
    original = server._get_conn()
    for _ in range(3):
        original.execute(
            "INSERT OR REPLACE INTO tool_call_embeddings "
            "SELECT * FROM tool_call_embeddings WHERE tool_call_id IN "
            "(SELECT 'tc' || i FROM range(0, 30000) t(i))"
        )
        original.execute("CHECKPOINT")
    entered, release, reader_entered, reader_release = (threading.Event() for _ in range(4))
    actual = compaction._estimate_live_bytes
    calls = 0

    def estimate(cursor, block_size):
        nonlocal calls
        calls += 1
        result = actual(cursor, block_size)
        if calls == 1:
            entered.set()
            assert release.wait(10)
        return result

    monkeypatch.setattr(compaction, "_estimate_live_bytes", estimate)

    def read(cursor):
        result = cursor.execute("SELECT count(*) FROM tool_call_embeddings").fetchone()
        reader_entered.set()
        assert reader_release.wait(10)
        return result

    async def run() -> None:
        server._compact_unsuccessful_checks = 3 if forced else 0
        reader = asyncio.create_task(server._run_readonly(read))
        tick = index = None
        try:
            assert await asyncio.to_thread(reader_entered.wait, 10)
            tick = asyncio.create_task(server._maybe_run_auto_compact())
            assert await asyncio.to_thread(entered.wait, 10)
            committed = asyncio.Event()

            def commit():
                conn = server._get_conn()
                conn.execute("UPDATE runtime_state SET last_index_total=31337")
                if forced:
                    conn.execute("SELECT setseed(0.42)")
                    conn.execute(
                        "INSERT INTO tool_call_embeddings SELECT 'dense-' || i, "
                        "list_transform(range(0,128), x -> random()::FLOAT)::FLOAT[128] "
                        "FROM range(0,300000) t(i)"
                    )
                    conn.execute("CHECKPOINT")
                    assert compaction.estimate_bloat_from_cursor(conn, config.db_path).ratio < 1.5

            async def waiting_index() -> None:
                async with server._index_turn():
                    await server._writer_call(
                        commit,
                        "intervening index",
                        checkpoint_wakeup=False,
                    )
                    committed.set()

            # For ordinary observation, an index writer must wait until it has
            # been accepted; forced observation supplies that exclusion itself.
            if not forced:
                await server._write_lock.acquire(affects_enrichment=False)
                # This invalidates the first optimistic result, so the next
                # estimate after release must settle before the index turn.
                server._write_lock.release()
            if forced:
                # This scenario makes the accepted positive healthy at handoff;
                # checkpoint needs the old native reader to have completed.
                reader_release.set()
                assert await reader == (60000,)
            index = asyncio.create_task(waiting_index())
            await asyncio.sleep(0)
            release.set()
            await asyncio.wait_for(asyncio.shield(index), 5)
            assert committed.is_set()
            if not forced:
                assert not tick.done(), "maintenance replaced storage under a live reader"
                assert server._conn is original
                reader_release.set()
                assert await reader == (60000,)
            await asyncio.wait_for(tick, 10)
            assert server._conn is not original
            assert server._last_bloat_ratio is not None and server._last_bloat_ratio < 1.5
            assert server._get_conn().execute(
                "SELECT last_index_total FROM runtime_state"
            ).fetchone() == (31337,)
            assert server._get_conn().execute(
                "SELECT count(*) FROM tool_call_embeddings"
            ).fetchone() == ((360000,) if forced else (60000,))
            await server._writer_call(lambda: server._get_conn().execute("CHECKPOINT"), "after")
        finally:
            release.set()
            reader_release.set()
            await asyncio.gather(
                *(task for task in (reader, index, tick) if task is not None),
                return_exceptions=True,
            )
            await server.stop()

    asyncio.run(run())


@pytest.mark.parametrize("forced", [False, True])
def test_missing_database_observation_does_not_create_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forced: bool
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        compaction=CompactionConfig(
            min_bytes=0,
            bloat_ratio_threshold=0,
        ),
    )
    server = RpcServer(config=config)

    async def run() -> None:
        server._compact_unsuccessful_checks = 3 if forced else 0
        try:
            await server._maybe_run_auto_compact()
            assert server._last_bloat_ratio == 0
            assert not config.db_path.exists()
            assert server._conn is None
            assert server._compact_unsuccessful_checks == 0
        finally:
            await server.stop()

    asyncio.run(run())


def test_real_row_version_interference_cannot_publish_a_torn_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_daemon_compaction import _seed_single_row_group, _single_row_group_config

    config = _single_row_group_config(tmp_path, monkeypatch)
    _seed_single_row_group(config)
    server = RpcServer(config=config)
    barrier = _MetadataBarrier(monkeypatch)

    async def run() -> None:
        original = server._get_conn()
        server._last_bloat_ratio = 99
        task = asyncio.create_task(server._maybe_run_auto_compact())
        try:
            for _ in range(3):
                await barrier.next()
                await server._writer_call(
                    lambda: original.execute(
                        "INSERT OR REPLACE INTO tool_call_embeddings "
                        "SELECT * FROM tool_call_embeddings WHERE tool_call_id IN "
                        "(SELECT 'tc' || i FROM range(0,30000) t(i))"
                    ),
                    "replace 30k",
                    checkpoint_wakeup=False,
                )
                barrier.release.release()
            await task
            assert server._last_bloat_ratio == 99
            assert server._compact_unsuccessful_checks == 1
            assert server._conn is original
            assert original.execute("SELECT count(*) FROM tool_call_embeddings").fetchone() == (
                60000,
            )
            assert not (config.data_dir / "recall.compacting").exists()
        finally:
            for _ in range(3):
                barrier.release.release()
            await task
            await server.stop()

    asyncio.run(run())


def test_positive_candidate_is_discarded_when_engine_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    config = _config(
        tmp_path, monkeypatch, compaction=CompactionConfig(min_bytes=0, bloat_ratio_threshold=0)
    )
    conn = _connect_schema(config)
    conn.close()
    server = RpcServer(config=config)
    actual_estimate = compaction._estimate_live_bytes
    calls = 0

    def estimate(cursor, block_size):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise duckdb.InvalidInputException("replacement requires a fresh observation")
        return actual_estimate(cursor, block_size)

    monkeypatch.setattr(compaction, "_estimate_live_bytes", estimate)

    async def run() -> None:
        published = asyncio.Event()
        actual_publish = server._publish_bloat

        def publish(observation):
            actual_publish(observation)
            published.set()

        monkeypatch.setattr(server, "_publish_bloat", publish)
        original = server._get_conn()
        task = None
        try:
            async with server._index_turn():
                task = asyncio.create_task(server._maybe_run_auto_compact())
                await asyncio.wait_for(published.wait(), 5)

                def replace_engine():
                    server.close_db_connection()
                    server.reopen_db_connection()

                await server._writer_call(replace_engine, "replace engine", checkpoint_wakeup=False)
                assert server._conn is not original
            replacement = server._conn
            await task
            assert calls == 2
            assert server._conn is replacement
            assert server._compact_unsuccessful_checks == 1
            assert not (config.data_dir / "recall.compacting").exists()
            assert not config.db_path.with_suffix(".duckdb.pre-compact").exists()
        finally:
            if task is not None:
                await task
            await server.stop()

    asyncio.run(run())


def test_failed_positive_reopen_is_truthful_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import recall.services.daemon as daemon

    config = _config(
        tmp_path, monkeypatch, compaction=CompactionConfig(min_bytes=0, bloat_ratio_threshold=0)
    )
    conn = _connect_schema(config)
    conn.close()
    server = RpcServer(config=config)
    actual_compact = daemon.compact

    def compact(config):
        result = actual_compact(config)

        def fail_open():
            raise duckdb.IOException("replacement cannot be reopened")

        monkeypatch.setattr(server, "_open_conn_unlocked", fail_open)
        return result

    monkeypatch.setattr(daemon, "compact", compact)

    async def run() -> None:
        server._get_conn()
        try:
            with pytest.raises(duckdb.IOException, match="cannot be reopened"):
                await server._maybe_run_auto_compact()
            assert server._shutdown_event.is_set()
            assert server._conn is None
            assert not server._conn_closed_for_maintenance
            assert not server._write_lock.locked()
            assert not server._index_request_lock.locked()
        finally:
            await server.stop()
        with duckdb.connect(str(config.db_path), read_only=True) as reopened:
            assert reopened.execute("SELECT count(*) FROM runtime_state").fetchone() == (1,)

    asyncio.run(run())


def test_partial_close_failure_reopens_a_proven_usable_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path, monkeypatch, compaction=CompactionConfig(min_bytes=0, bloat_ratio_threshold=0)
    )
    conn = _connect_schema(config)
    conn.close()
    server = RpcServer(config=config)
    close = server.close_db_connection

    def partial_close():
        close()
        raise duckdb.IOException("close completed before an operating error")

    monkeypatch.setattr(server, "close_db_connection", partial_close)

    async def run() -> None:
        original = server._get_conn()
        try:
            await server._maybe_run_auto_compact()
            assert not server._shutdown_event.is_set()
            assert server._conn is not None and server._conn is not original
            assert server._conn.execute("SELECT count(*) FROM runtime_state").fetchone() == (1,)
            assert not server._conn_closed_for_maintenance
            assert not server._write_lock.locked()
            assert not server._index_request_lock.locked()
        finally:
            await server.stop()

    asyncio.run(run())
