"""The daemon's fatal funnel remembers the failure and always terminates.

REQ-RESIL-014: a fatal invalidation leaves `<data_dir>/daemon-failure.json`.
REQ-RESIL-019: an ENOSPC on the shared connection flags index verification.
REQ-RESIL-020 / INV-RESIL-006: the shutdown event is set even when persisting
the marker or emitting the log record raises.
REQ-RESIL-015 / REQ-RESIL-018: startup runs the self-repair step before any
other DuckDB work and caches the probe for `daemon status`.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest
from conftest import _can_acquire_duckdb_lock
from recall.core.config import AppConfig, CompactionConfig, DaemonConfig
from recall.db.maintenance import DivergedKey, IndexDivergenceReport
from recall.services import self_repair
from recall.services.rpc_server import ClientConnection, RpcServer
from recall.services.runtime_state import load_runtime_status
from recall.services.self_repair import (
    DaemonStartupRefused,
    StartupRepairOutcome,
    read_failure_marker,
)

_requires_duckdb_lock = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)

INDEX_FATAL = (
    "FATAL Error: Invalid Input Error: Failed to delete all rows from index. "
    "Only deleted 0 out of 1 rows.\nChunk: Chunk - [20 Columns] - FLAT VARCHAR: 1 = [ eae5a5d3 ]"
)
ENOSPC_WAL = (
    "TransactionContext Error: Failed to commit: Could not write file "
    '"/home/dev/.local/share/recall/recall.duckdb.wal": No space left on device'
)


def _tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    # Unix socket paths are length-limited; keep the data dir short.
    short_root = Path("/tmp") / f"recall-fatal-{abs(hash(tmp_path)) % 10**8}"
    monkeypatch.setenv("HOME", str(short_root))
    monkeypatch.setenv("RECALL_DATA_DIR", str(short_root / "data"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(short_root / "config.toml"))
    monkeypatch.delenv("RECALL_DB_PATH", raising=False)
    monkeypatch.delenv("RECALL_LOCK_PATH", raising=False)
    monkeypatch.setenv("RECALL_EMBED_BACKEND", "onnx")
    config = AppConfig.load()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    return replace(
        config,
        compaction=CompactionConfig(auto_trigger=False),
        daemon=DaemonConfig(embed=False),
    )


class _RaisingHandler(logging.Handler):
    """A handler that fails the way a stderr handler fails on a full disk.

    `Handler.handle` (not `emit`) raises so the error bypasses `handleError`
    and reaches the caller -- the shape that would preempt shutdown.
    """

    def handle(self, record: logging.LogRecord) -> bool:
        raise OSError(errno.ENOSPC, "No space left on device")


class _FakeStreamWriter:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        return None


class TestFatalFunnelRemembers:
    def test_stop_on_fatal_db_writes_marker_with_site_and_class(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _tmp_config(tmp_path, monkeypatch)
        server = RpcServer(config=config)

        stopped = server._stop_on_fatal_db(duckdb.FatalException(INDEX_FATAL), "usage_harvest")

        assert stopped is True
        assert server._shutdown_event.is_set()
        marker = read_failure_marker(config.data_dir)
        assert marker is not None and marker.failure is not None
        assert marker.failure.site == "usage_harvest"
        assert marker.failure.failure_class == "index-divergence"
        assert marker.failure.exception_type == "FatalException"

    def test_ordinary_error_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _tmp_config(tmp_path, monkeypatch)
        server = RpcServer(config=config)

        assert server._stop_on_fatal_db(RuntimeError("boom"), "usage_harvest") is False
        assert read_failure_marker(config.data_dir) is None
        assert not server._shutdown_event.is_set()

    def test_shutdown_is_set_even_when_marker_and_logging_both_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _tmp_config(tmp_path, monkeypatch)
        server = RpcServer(config=config)

        def marker_write_fails(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(self_repair, "remember_fatal_failure", marker_write_fails)
        rpc_logger = logging.getLogger("recall.rpc_server")
        handler = _RaisingHandler()
        rpc_logger.addHandler(handler)
        try:
            stopped = server._stop_on_fatal_db(duckdb.FatalException(INDEX_FATAL), "catch_up")
        finally:
            rpc_logger.removeHandler(handler)

        assert stopped is True
        assert server._shutdown_event.is_set()

    def test_process_request_fatal_leaves_a_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _tmp_config(tmp_path, monkeypatch)
        server = RpcServer(config=config)

        async def _fatal_handler(params: dict, client: object) -> None:
            raise duckdb.FatalException(INDEX_FATAL)

        server._methods["recall.fatal_test"] = _fatal_handler
        writer = _FakeStreamWriter()
        request = b'{"jsonrpc":"2.0","method":"recall.fatal_test","params":{},"id":1}'
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                server._process_request(
                    request, cast(asyncio.StreamWriter, writer), cast(ClientConnection, None)
                )
            )
        finally:
            loop.close()

        assert writer.chunks and b"-32603" in writer.chunks[0]
        assert server._shutdown_event.is_set()
        marker = read_failure_marker(config.data_dir)
        assert marker is not None and marker.failure is not None
        assert marker.failure.site == "recall.fatal_test"

    def test_watch_task_crash_with_fatal_leaves_a_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _tmp_config(tmp_path, monkeypatch)
        server = RpcServer(config=config)

        async def crash() -> None:
            raise duckdb.FatalException(INDEX_FATAL)

        loop = asyncio.new_event_loop()
        try:
            task = loop.create_task(crash())
            with pytest.raises(duckdb.FatalException):
                loop.run_until_complete(task)
            server._fail_on_watch_task_exit("drain", task)
        finally:
            loop.close()

        assert server._shutdown_event.is_set()
        marker = read_failure_marker(config.data_dir)
        assert marker is not None and marker.failure is not None
        assert marker.failure.site == "watch_drain"


class TestDiskFullFlag:
    def test_recover_shared_conn_flags_verification_on_enospc(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _tmp_config(tmp_path, monkeypatch)
        server = RpcServer(config=config)

        server._recover_shared_conn(duckdb.TransactionException(ENOSPC_WAL), "usage_harvest")

        marker = read_failure_marker(config.data_dir)
        assert marker is not None
        assert marker.failure is None, "a non-fatal ENOSPC is a flag, not a fatal record"
        assert marker.needs_index_verification is True
        assert not server._shutdown_event.is_set()

    def test_recover_shared_conn_ignores_ordinary_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _tmp_config(tmp_path, monkeypatch)
        server = RpcServer(config=config)

        server._recover_shared_conn(duckdb.TransactionException("aborted"), "usage_harvest")

        assert read_failure_marker(config.data_dir) is None

    @_requires_duckdb_lock
    def test_harvest_enospc_flags_marker_and_runtime_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import recall.services.watcher as watcher_module

        config = _tmp_config(tmp_path, monkeypatch)
        server = RpcServer(config=config)

        def wal_full(conn: object) -> None:
            raise duckdb.TransactionException(ENOSPC_WAL)

        monkeypatch.setattr(watcher_module, "_maybe_harvest_usage", wal_full)
        try:
            with pytest.raises(duckdb.TransactionException, match="No space left on device"):
                asyncio.run(
                    server._writer_call(server._harvest_usage_on_shared_conn, "usage harvest")
                )
            status = load_runtime_status(config, conn=server._get_conn())
        finally:
            if server._conn is not None:
                server._conn.close()

        assert not server._shutdown_event.is_set()
        marker = read_failure_marker(config.data_dir)
        assert marker is not None and marker.needs_index_verification is True
        assert status.needs_index_verification is True


def _report(diverged: tuple[DivergedKey, ...] = ()) -> IndexDivergenceReport:
    return IndexDivergenceReport(
        checked_at=datetime(2026, 8, 30, 12, 0, 0),
        indexes_probed=13,
        samples_checked=20,
        samples_unverifiable=3,
        diverged=diverged,
        complete=True,
        elapsed_seconds=0.42,
    )


@_requires_duckdb_lock
class TestStartupWiring:
    def test_self_repair_runs_before_other_startup_db_work_and_caches_probe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import recall.services.daemon as daemon_module

        config = _tmp_config(tmp_path, monkeypatch)
        order: list[str] = []
        report = _report()

        def fake_repair(cfg: AppConfig, conn: Any, **_kwargs: Any) -> StartupRepairOutcome:
            assert cfg.data_dir == config.data_dir
            assert conn is server._conn, "the repair runs on the shared connection"
            order.append("self_repair")
            return StartupRepairOutcome(action="none", record=None, probe=report, rebuild=None)

        monkeypatch.setattr(self_repair, "run_startup_self_repair_on", fake_repair)
        monkeypatch.setattr(
            daemon_module,
            "run_startup_fts_sidecar_sync",
            lambda _cfg: order.append("sidecar_sync"),
        )
        monkeypatch.setattr(
            daemon_module, "run_startup_snapshot_gc", lambda _cfg: order.append("snapshot_gc")
        )

        async def init_fts_then_stop(self: RpcServer) -> None:
            order.append("init_fts")
            self.request_shutdown()

        monkeypatch.setattr(RpcServer, "_init_fts", init_fts_then_stop)
        server = RpcServer(config=config)

        asyncio.run(asyncio.wait_for(server.start(watch=False), timeout=10))

        assert order[0] == "self_repair"
        assert order.index("self_repair") < order.index("sidecar_sync")
        assert order.index("self_repair") < order.index("init_fts")
        assert server._last_index_probe == report

    def test_self_repair_runs_as_locked_shared_conn_work(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-RESIL-021: the repair executes DDL on the shared connection, so
        it is shared-connection work like any other -- under the write lock and
        off the event-loop thread, not only behind the lifecycle gate."""
        import recall.services.daemon as daemon_module

        config = _tmp_config(tmp_path, monkeypatch)
        observed: list[tuple[bool, bool]] = []
        loop_thread = threading.current_thread()
        report = _report()

        def fake_repair(cfg: AppConfig, conn: Any, **_kwargs: Any) -> StartupRepairOutcome:
            observed.append(
                (server._write_lock.locked(), threading.current_thread() is loop_thread)
            )
            return StartupRepairOutcome(action="none", record=None, probe=report, rebuild=None)

        monkeypatch.setattr(self_repair, "run_startup_self_repair_on", fake_repair)
        monkeypatch.setattr(daemon_module, "run_startup_fts_sidecar_sync", lambda _cfg: None)
        monkeypatch.setattr(daemon_module, "run_startup_snapshot_gc", lambda _cfg: None)

        async def init_fts_then_stop(self: RpcServer) -> None:
            self.request_shutdown()

        monkeypatch.setattr(RpcServer, "_init_fts", init_fts_then_stop)
        server = RpcServer(config=config)

        asyncio.run(asyncio.wait_for(server.start(watch=False), timeout=10))

        assert observed == [(True, False)]

    def test_refusal_stops_server_and_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import recall.services.daemon as daemon_module

        config = _tmp_config(tmp_path, monkeypatch)

        def refuse(cfg: AppConfig, conn: Any, **_kwargs: Any) -> StartupRepairOutcome:
            raise DaemonStartupRefused("index divergence recurred after rebuild")

        monkeypatch.setattr(self_repair, "run_startup_self_repair_on", refuse)
        monkeypatch.setattr(daemon_module, "run_startup_fts_sidecar_sync", lambda _cfg: None)
        monkeypatch.setattr(daemon_module, "run_startup_snapshot_gc", lambda _cfg: None)
        server = RpcServer(config=config)

        with pytest.raises(DaemonStartupRefused):
            asyncio.run(asyncio.wait_for(server.start(watch=False), timeout=10))

        assert not server._socket_path.exists(), "a refused start releases the socket"
        assert not server._pid_path.exists()

    def test_fatal_during_self_repair_stops_server_and_leaves_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-RESIL-015: a fatal invalidation inside the repair step is not a
        step failure to log past -- it routes through the fatal funnel, so the
        daemon neither serves a dead instance nor forgets why it died."""
        import recall.services.daemon as daemon_module

        config = _tmp_config(tmp_path, monkeypatch)

        def dead_instance(cfg: AppConfig, conn: Any, **_kwargs: Any) -> StartupRepairOutcome:
            raise duckdb.FatalException(
                "database has been invalidated because of a previous fatal error"
            )

        monkeypatch.setattr(self_repair, "run_startup_self_repair_on", dead_instance)
        monkeypatch.setattr(daemon_module, "run_startup_fts_sidecar_sync", lambda _cfg: None)
        monkeypatch.setattr(daemon_module, "run_startup_snapshot_gc", lambda _cfg: None)
        server = RpcServer(config=config)

        with pytest.raises(duckdb.FatalException):
            asyncio.run(asyncio.wait_for(server.start(watch=False), timeout=10))

        assert server._shutdown_event.is_set()
        marker = read_failure_marker(config.data_dir)
        assert marker is not None and marker.failure is not None
        assert marker.failure.site == "startup_self_repair"
        assert not server._socket_path.exists(), "a dead start releases the socket"
        assert not server._pid_path.exists()

    def test_repair_step_failure_does_not_block_listening(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import recall.services.daemon as daemon_module

        config = _tmp_config(tmp_path, monkeypatch)

        def explode(cfg: AppConfig, conn: Any, **_kwargs: Any) -> StartupRepairOutcome:
            raise RuntimeError("catalog unavailable")

        monkeypatch.setattr(self_repair, "run_startup_self_repair_on", explode)
        monkeypatch.setattr(daemon_module, "run_startup_fts_sidecar_sync", lambda _cfg: None)
        monkeypatch.setattr(daemon_module, "run_startup_snapshot_gc", lambda _cfg: None)
        reached_fts = False

        async def init_fts_then_stop(self: RpcServer) -> None:
            nonlocal reached_fts
            reached_fts = True
            self.request_shutdown()

        monkeypatch.setattr(RpcServer, "_init_fts", init_fts_then_stop)
        server = RpcServer(config=config)

        asyncio.run(asyncio.wait_for(server.start(watch=False), timeout=10))

        assert reached_fts is True
        assert server._last_index_probe is None

    def test_self_repair_holds_the_connection_gate_so_requests_wait(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-RESIL-015: the socket is bound before the repair, so a request can
        arrive mid-repair. It must wait on the shared connection rather than open
        a second one whose schema bootstrap races the repair's (observed as
        `Catalog write-write conflict on create ... schema_version`)."""
        import recall.services.daemon as daemon_module

        config = _tmp_config(tmp_path, monkeypatch)
        seen: dict[str, Any] = {}
        request_conn: list[Any] = []
        request_done = threading.Event()

        def request() -> None:
            request_conn.append(server._get_conn())
            request_done.set()

        def fake_repair(cfg: AppConfig, conn: Any, **_kwargs: Any) -> StartupRepairOutcome:
            seen["conn"] = conn
            seen["shared"] = server._conn
            threading.Thread(target=request, daemon=True).start()
            seen["request_finished_during_repair"] = request_done.wait(0.3)
            return StartupRepairOutcome(action="none", record=None, probe=_report(), rebuild=None)

        monkeypatch.setattr(self_repair, "run_startup_self_repair_on", fake_repair)
        monkeypatch.setattr(daemon_module, "run_startup_fts_sidecar_sync", lambda _cfg: None)
        monkeypatch.setattr(daemon_module, "run_startup_snapshot_gc", lambda _cfg: None)

        async def init_fts_then_stop(self: RpcServer) -> None:
            self.request_shutdown()

        monkeypatch.setattr(RpcServer, "_init_fts", init_fts_then_stop)
        server = RpcServer(config=config)

        asyncio.run(asyncio.wait_for(server.start(watch=False), timeout=10))

        assert request_done.wait(5), "the request proceeds once the repair releases the gate"
        assert seen["request_finished_during_repair"] is False, "a request must wait for the repair"
        assert seen["conn"] is not None and seen["conn"] is seen["shared"]
        assert request_conn[0] is seen["conn"], "the request reuses the repaired connection"

    def test_read_waits_for_startup_bloat_probe_to_close_readonly_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import recall.services.compaction as compaction_module
        import recall.services.daemon as daemon_module
        from recall.db.schema import ensure_schema

        config = _tmp_config(tmp_path, monkeypatch)
        with duckdb.connect(str(config.db_path)) as conn:
            ensure_schema(conn)
        server = RpcServer(config=config)
        result: dict[str, Any] = {}
        request_started = threading.Event()
        request_done = threading.Event()

        def read_request() -> None:
            request_started.set()
            try:
                result["row"] = server._run_with_read_cursor_sync(
                    lambda conn: conn.execute("SELECT 42").fetchone()
                )
            except Exception as err:
                result["error"] = repr(err)
            finally:
                request_done.set()

        request = threading.Thread(target=read_request)

        def probe_with_read_in_flight(conn: Any, block_size: int) -> int:
            assert conn.execute("SELECT 42").fetchone() == (42,)
            assert block_size > 0
            request.start()
            assert request_started.wait(2)
            result["completed_during_probe"] = request_done.wait(0.3)
            return 0

        monkeypatch.setattr(compaction_module, "_estimate_live_bytes", probe_with_read_in_flight)
        monkeypatch.setattr(daemon_module, "run_startup_fts_sidecar_sync", lambda _cfg: None)
        monkeypatch.setattr(daemon_module, "run_startup_snapshot_gc", lambda _cfg: None)

        async def init_fts_then_stop(self: RpcServer) -> None:
            self.request_shutdown()

        monkeypatch.setattr(RpcServer, "_init_fts", init_fts_then_stop)
        asyncio.run(asyncio.wait_for(server.start(watch=False), timeout=10))
        request.join(timeout=5)

        assert not request.is_alive()
        assert result.get("error") is None, result
        assert result["row"] == (42,)
        assert result["completed_during_probe"] is False


@_requires_duckdb_lock
class TestCheckIndexesRpc:
    def test_check_indexes_rpc_runs_live_probe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _tmp_config(tmp_path, monkeypatch)
        server = RpcServer(config=config)
        assert "recall.check_indexes" in server._methods

        async def call() -> Any:
            return await server._handle_check_indexes({"sample": 1}, None)

        try:
            result = asyncio.run(call())
        finally:
            if server._conn is not None:
                server._conn.close()

        # The RPC returns the same payload shape `daemon status` and the CLI use.
        assert result["diverged"] == []
        assert result["diverged_count"] == 0
        assert result["indexes_probed"] > 0
        assert isinstance(result["checked_at"], str)

    def test_check_indexes_rpc_rejects_bad_sample(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _tmp_config(tmp_path, monkeypatch)
        server = RpcServer(config=config)

        async def call() -> Any:
            return await server._handle_check_indexes({"sample": 0}, None)

        with pytest.raises(ValueError, match="sample"):
            asyncio.run(call())

    def test_daemon_status_overlay_carries_cached_probe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _tmp_config(tmp_path, monkeypatch)
        server = RpcServer(config=config)
        key = DivergedKey(
            table="session_state",
            column="git_repo",
            key="/Users/dev/code/app",
            index_count=1790,
            full_count=1797,
        )
        server._last_index_probe = _report((key,))

        async def call() -> Any:
            return await server._handle_daemon_status({}, None)

        try:
            status = asyncio.run(call())
        finally:
            if server._conn is not None:
                server._conn.close()

        assert status["index_divergence"] is not None
        assert status["index_divergence"]["diverged_count"] == 1
        assert status["index_divergence"]["diverged"][0]["key"] == "/Users/dev/code/app"
        assert status["index_divergence"]["checked_at"] == "2026-08-30T12:00:00"
