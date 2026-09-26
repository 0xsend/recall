"""REQ-RECON-008 and REQ-RESIL-021 against real database and worker lifetimes."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import duckdb
import pytest
from lane_harness import build_lane
from recall.services import embed_phase
from recall.services.embed_phase import EmbedPhaseState
from recall.services.rpc_server import RpcServer
from recall.services.system_state import EmbedPreconditionResult, LoadThreshold


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[RpcServer]:
    lane = build_lane(tmp_path, monkeypatch)
    srv = lane.server
    srv._config = replace(
        srv._config, daemon=replace(srv._config.daemon, embed=True, embed_idle_session=0)
    )
    # The lane fixture is deliberately future-dated; these tests exercise idle enrichment.
    srv._get_conn().execute("UPDATE session_state SET file_mtime = 1")
    monkeypatch.setattr(
        "recall.services.system_state.check_power", lambda: EmbedPreconditionResult(ok=True)
    )
    monkeypatch.setattr(
        "recall.services.system_state.check_load", lambda _: EmbedPreconditionResult(ok=True)
    )
    yield srv
    asyncio.run(srv.stop())


def test_aborted_writer_is_rolled_back_before_the_next_job(server: RpcServer) -> None:
    def invalid_write() -> None:
        conn = server._get_conn()
        conn.execute("BEGIN TRANSACTION")
        conn.execute("UPDATE session_state SET file_size = 999999")
        conn.execute("SELECT CAST('invalid-number' AS INTEGER)")

    async def scenario() -> None:
        with pytest.raises(duckdb.ConversionException):
            await server._writer_call(invalid_write, "injected invalid transaction")
        sizes = await server._writer_call(
            lambda: server._get_conn().execute("SELECT file_size FROM session_state").fetchall(),
            "next writer reads committed state",
        )
        assert all(row[0] != 999999 for row in sizes)
        await server._writer_call(
            lambda: server._get_conn().execute("UPDATE session_state SET file_size = 17"),
            "next writer commits",
        )
        assert await server._run_readonly(
            lambda conn: conn.execute("SELECT DISTINCT file_size FROM session_state").fetchall()
        ) == [(17,)]

    asyncio.run(scenario())


def test_writer_wakes_checkpoint_monitor_after_releasing_lock(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.db.connection import WalCheckpointResult

    checkpoint_started = threading.Event()
    release_checkpoint = threading.Event()
    checkpoint_calls = 0

    def checkpoint(_conn: object, _path: Path) -> WalCheckpointResult:
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        if checkpoint_calls > 1:
            return WalCheckpointResult(False, 0, 0, 0.0)
        checkpoint_started.set()
        assert release_checkpoint.wait(timeout=2)
        return WalCheckpointResult(True, 70 * 1024**2, 0, 0.01)

    monkeypatch.setattr("recall.db.checkpoint_wal_if_due", checkpoint)
    monkeypatch.setattr(
        "recall.db.connection.wal_size_bytes",
        lambda _path: 70 * 1024**2 if checkpoint_calls == 0 else 0,
    )

    async def scenario() -> None:
        monitor = asyncio.create_task(server._checkpoint_monitor())
        try:
            await server._writer_call(lambda: None, "test write")
            assert await asyncio.to_thread(checkpoint_started.wait, 1)
            queued = asyncio.create_task(server._writer_call(lambda: "next", "queued writer"))
            await asyncio.sleep(0.02)
            assert queued.done() is False
            release_checkpoint.set()
            assert await queued == "next"
            for _ in range(50):
                if server._checkpoint_status["successes"] == 1:
                    break
                await asyncio.sleep(0.01)
            assert server._checkpoint_status == {
                "attempts": 1,
                "successes": 1,
                "running": False,
                "last_started_at": server._checkpoint_status["last_started_at"],
                "last_completed_at": server._checkpoint_status["last_completed_at"],
                "last_duration_seconds": 0.01,
                "last_wal_bytes_before": 70 * 1024**2,
                "last_wal_bytes_after": 0,
                "last_error": None,
            }
        finally:
            server._shutdown_event.set()
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("wal_bytes", "active_allowed"),
    [(70 * 1024**2, True), (200 * 1024**2, False)],
)
def test_checkpoint_contention_throttles_admission_without_connection_recovery(
    server: RpcServer,
    monkeypatch: pytest.MonkeyPatch,
    wal_bytes: int,
    active_allowed: bool,
) -> None:
    from recall.db.connection import WalCheckpointResult

    monkeypatch.setattr("recall.db.connection.wal_size_bytes", lambda _path: wal_bytes)
    monkeypatch.setattr(
        "recall.db.checkpoint_wal_if_due",
        lambda _conn, _path: WalCheckpointResult(True, wal_bytes, wal_bytes, 0.01, contended=True),
    )
    recover = Mock(side_effect=AssertionError("checkpoint contention triggered recovery"))
    monkeypatch.setattr(server, "_recover_shared_conn", recover)
    server._checkpoint_retry_seconds = 10.0

    async def scenario() -> None:
        monitor = asyncio.create_task(server._checkpoint_monitor())
        try:
            server._checkpoint_wakeup.set()
            async with asyncio.timeout(1):
                while server._checkpoint_status["attempts"] == 0:
                    await asyncio.sleep(0.01)
            assert server._checkpoint_status["successes"] == 0
            assert "deferred" in server._checkpoint_status["last_error"]
            from recall.db.source_files import SourceCatalog

            candidate = (
                SourceCatalog(server._get_conn(), clock=time.time).status_page(limit=1).rows[0]
            )
            assert (
                server._raw_candidate_admitted(
                    candidate, {candidate.source_path}, history_busy=False
                )
                is active_allowed
            )
            assert not server._raw_candidate_admitted(candidate, set(), history_busy=False)
            assert recover.call_count == 0
        finally:
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor

    asyncio.run(scenario())


def test_checkpoint_contention_waits_for_reader_drain_after_three_attempts(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.db.connection import WalCheckpointResult

    wal_bytes = 70 * 1024**2
    calls = 0
    reader_entered = threading.Event()
    reader_release = threading.Event()

    def checkpoint(_conn: object, _path: Path) -> WalCheckpointResult:
        nonlocal calls
        calls += 1
        if calls <= 3:
            return WalCheckpointResult(True, wal_bytes, wal_bytes, 0.001, contended=True)
        return WalCheckpointResult(True, wal_bytes, 0, 0.001)

    def held_read(conn: object) -> None:
        reader_entered.set()
        assert reader_release.wait(5)

    monkeypatch.setattr("recall.db.connection.wal_size_bytes", lambda _path: wal_bytes)
    monkeypatch.setattr("recall.db.checkpoint_wal_if_due", checkpoint)
    server._checkpoint_retry_seconds = 0

    async def scenario() -> None:
        reader = asyncio.create_task(server._run_readonly(held_read))
        monitor = asyncio.create_task(server._checkpoint_monitor())
        try:
            assert await asyncio.to_thread(reader_entered.wait, 1)
            async with asyncio.timeout(1):
                while calls < 3:
                    server._checkpoint_wakeup.set()
                    await asyncio.sleep(0.01)
            server._checkpoint_wakeup.set()
            await asyncio.sleep(0.05)
            assert calls == 3
            assert server._checkpoint_status["successes"] == 0
            reader_release.set()
            async with asyncio.timeout(1):
                while server._checkpoint_status["successes"] == 0:
                    await asyncio.sleep(0.01)
            assert calls == 4
        finally:
            reader_release.set()
            await reader
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor

    asyncio.run(scenario())


def test_checkpoint_monitor_records_failure_and_keeps_retrying(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.db.connection import WalCheckpointResult

    calls = 0

    def checkpoint(_conn: object, _path: Path) -> WalCheckpointResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise duckdb.IOException("injected checkpoint failure")
        return WalCheckpointResult(True, 70 * 1024**2, 0, 0.02)

    monkeypatch.setattr("recall.db.checkpoint_wal_if_due", checkpoint)
    monkeypatch.setattr("recall.db.connection.wal_size_bytes", lambda _path: 70 * 1024**2)
    server._checkpoint_retry_seconds = 0.1

    async def scenario() -> None:
        monitor = asyncio.create_task(server._checkpoint_monitor())
        try:
            server._checkpoint_wakeup.set()
            for _ in range(100):
                if server._checkpoint_status["last_error"] is not None:
                    break
                await asyncio.sleep(0.01)
            assert "injected checkpoint failure" in server._checkpoint_status["last_error"]
            await server._writer_call(lambda: None, "wake during cooldown")
            await asyncio.sleep(0.03)
            assert calls == 1
            for _ in range(100):
                if server._checkpoint_status["successes"] == 1:
                    break
                await asyncio.sleep(0.01)
            assert server._checkpoint_status["attempts"] == 2
            assert server._checkpoint_status["last_error"] is None
        finally:
            server._shutdown_event.set()
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor

    asyncio.run(scenario())


def test_checkpoint_monitor_clears_running_when_locked_recheck_is_noop(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.db.connection import WalCheckpointResult

    monkeypatch.setattr("recall.db.connection.wal_size_bytes", lambda _path: 70 * 1024**2)
    rechecked = threading.Event()

    def checkpoint(_conn: object, _path: Path) -> WalCheckpointResult:
        rechecked.set()
        return WalCheckpointResult(False, 0, 0, 0.0)

    monkeypatch.setattr("recall.db.checkpoint_wal_if_due", checkpoint)

    async def scenario() -> None:
        monitor = asyncio.create_task(server._checkpoint_monitor())
        server._checkpoint_wakeup.set()
        assert await asyncio.to_thread(rechecked.wait, 1)
        await asyncio.sleep(0.02)
        assert server._checkpoint_status["running"] is False
        assert server._checkpoint_status["attempts"] == 0
        assert monitor.done() is False
        server._shutdown_event.set()
        monitor.cancel()
        with suppress(asyncio.CancelledError):
            await monitor

    asyncio.run(scenario())


def test_checkpoint_monitor_records_wal_stat_failure_and_survives(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(_path: Path) -> int:
        raise OSError("injected WAL stat failure")

    monkeypatch.setattr("recall.db.connection.wal_size_bytes", unreadable)
    server._checkpoint_retry_seconds = 10.0

    async def scenario() -> None:
        monitor = asyncio.create_task(server._checkpoint_monitor())
        server._checkpoint_wakeup.set()
        for _ in range(100):
            if server._checkpoint_status["last_error"] is not None or monitor.done():
                break
            await asyncio.sleep(0.01)
        assert "injected WAL stat failure" in server._checkpoint_status["last_error"]
        assert server._checkpoint_status["attempts"] == 1
        assert server._checkpoint_status["running"] is False
        assert monitor.done() is False
        server._shutdown_event.set()
        monitor.cancel()
        with suppress(asyncio.CancelledError):
            await monitor

    asyncio.run(scenario())


@pytest.mark.parametrize("fail_final_stat", [False, True])
def test_checkpoint_monitor_cancellation_waits_for_inflight_turn(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch, fail_final_stat: bool
) -> None:
    from recall.db.connection import WalCheckpointResult

    started = threading.Event()
    release = threading.Event()

    def checkpoint(_conn: object, _path: Path) -> WalCheckpointResult:
        started.set()
        assert release.wait(timeout=2)
        return WalCheckpointResult(True, 70 * 1024**2, 0, 0.01)

    def wal_size(_path: Path) -> int:
        if fail_final_stat and started.is_set():
            raise PermissionError("WAL unavailable after cancellation")
        return 70 * 1024**2

    monkeypatch.setattr("recall.db.checkpoint_wal_if_due", checkpoint)
    monkeypatch.setattr("recall.db.connection.wal_size_bytes", wal_size)

    async def scenario() -> None:
        monitor = asyncio.create_task(server._checkpoint_monitor())
        server._checkpoint_wakeup.set()
        assert await asyncio.to_thread(started.wait, 1)
        monitor.cancel()
        queued = asyncio.create_task(server._writer_call(lambda: "available", "queued writer"))
        await asyncio.sleep(0.02)
        assert monitor.done() is False
        assert queued.done() is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await monitor
        assert await queued == "available"
        assert await server._writer_call(lambda: "reused", "lock reuse") == "reused"
        assert server._checkpoint_status["running"] is False

    asyncio.run(scenario())


def test_checkpoint_monitor_runs_while_paused_and_preserves_real_wal_rows(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.db.connection import checkpoint_wal_if_due, wal_size_bytes
    from recall.services.coordinator import set_paused

    db_path = server._config.db_path
    monkeypatch.setattr("recall.db.connection.WAL_CHECKPOINT_HIGH_WATER_BYTES", 1)
    monkeypatch.setattr(
        "recall.db.checkpoint_wal_if_due",
        lambda conn, path: checkpoint_wal_if_due(conn, path, high_water_bytes=1),
    )
    assert set_paused(server._config, True) is True

    async def scenario() -> None:
        monitor = asyncio.create_task(server._checkpoint_monitor())
        try:
            await server._writer_call(
                lambda: server._get_conn().execute(
                    "CREATE TABLE checkpoint_probe AS SELECT range AS id FROM range(10000)"
                ),
                "real WAL write",
            )
            for _ in range(200):
                if server._checkpoint_status["successes"] == 1:
                    break
                await asyncio.sleep(0.01)
            assert server._checkpoint_status["successes"] == 1
            assert server._checkpoint_status["last_wal_bytes_before"] > 0
            assert server._checkpoint_status["last_wal_bytes_after"] == wal_size_bytes(db_path)
            assert wal_size_bytes(db_path) == 0
            assert await server._run_readonly(
                lambda conn: conn.execute("SELECT COUNT(*) FROM checkpoint_probe").fetchone()
            ) == (10000,)
        finally:
            server._shutdown_event.set()
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor

    asyncio.run(scenario())


class DeterministicBackend:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.model_id = "lifecycle-verifier"
        self.query_prefix = ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.25] * self.dimensions for _ in texts]


@pytest.mark.parametrize("on_ac", [True, False])
def test_detached_enrichment_respects_load_and_recovers(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch, on_ac: bool
) -> None:
    state = EmbedPhaseState(_config=server._config)
    state.backend = DeterministicBackend(server._config.embedding.dimensions)
    server._embed_state = state
    monkeypatch.setattr(
        "recall.services.system_state.check_power",
        lambda: EmbedPreconditionResult(ok=on_ac, reason="battery" if not on_ac else ""),
    )
    thresholds: list[LoadThreshold] = []

    def busy(threshold: LoadThreshold) -> EmbedPreconditionResult:
        thresholds.append(threshold)
        return EmbedPreconditionResult(ok=False, reason="injected load above threshold")

    monkeypatch.setattr("recall.services.system_state.check_load", busy)

    async def scenario() -> None:
        assert await server._run_embed_batch(server._config, state) == -1
        status = await server._handle_daemon_status({}, None)
        assert "injected load" in status["reconciliation"]["enrichment_deferred"]
        assert status["reconciliation"]["enrichment_error"] is None
        assert server._get_conn().execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (
            0,
        )
        await server._writer_call(
            lambda: server._get_conn().execute("UPDATE session_state SET file_size = 31"),
            "raw write while enrichment deferred",
        )
        monkeypatch.setattr(
            "recall.services.system_state.check_load", lambda _: EmbedPreconditionResult(ok=True)
        )
        assert await server._run_embed_batch(server._config, state) > 0
        status = await server._handle_daemon_status({}, None)
        assert status["reconciliation"]["enrichment_deferred"] is None

    asyncio.run(scenario())
    daemon = server._config.daemon
    # The ceiling the power state put in force, and the words the deferral
    # reason uses to name it (REQ-ADAPT-006).
    assert [threshold.fraction for threshold in thresholds] == [
        daemon.load_threshold if on_ac else daemon.battery_threshold
    ]
    assert [threshold.name for threshold in thresholds] == [
        "load threshold" if on_ac else "battery threshold"
    ]


def test_pending_selection_materializes_only_the_requested_session_bound(server: RpcServer) -> None:
    conn = server._get_conn()

    class BoundedResult:
        def execute(self, *args, **kwargs):
            conn.execute(*args, **kwargs)
            return self

        def fetchone(self):
            return conn.fetchone()

        def fetchall(self):
            rows = conn.fetchall()
            assert len(rows) <= 1, "pending discovery materialized the entire session roster"
            return rows

    pending = embed_phase.find_pending_embeds(cast(Any, BoundedResult()), 0, max_sessions=1)
    assert len(pending.session_ids) == 1
    assert pending.message_count == conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert pending.tool_call_count == conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]


def test_detached_enrichment_counts_drain_and_backs_off_stalled_publication(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecordingBackend(DeterministicBackend):
        calls = 0

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return super().embed(texts)

    state = EmbedPhaseState(_config=server._config)
    backend = RecordingBackend(server._config.embedding.dimensions)
    state.backend = backend
    # An acknowledged but lost vector write must not manufacture progress or spin models.
    from recall.services import indexer

    original = indexer._insert_session_embeddings
    monkeypatch.setattr(indexer, "_insert_session_embeddings", lambda *_args, **_kwargs: None)

    async def scenario() -> None:
        assert await server._run_embed_batch(server._config, state) == 0
        assert state.last_batch_size == 0
        calls = backend.calls
        assert calls > 0
        assert await server._run_embed_batch(server._config, state) == 0
        assert backend.calls == calls
        assert state.last_pending is not None and state.stalled_for(state.last_pending) > 0
        monkeypatch.setattr(indexer, "_insert_session_embeddings", original)
        state.stalled_until = time.monotonic() - 1
        before = embed_phase.find_pending_embeds(server._get_conn(), 0).total
        drained = await server._run_embed_batch(server._config, state)
        after = embed_phase.find_pending_embeds(server._get_conn(), 0).total
        assert drained == before - after > 0
        assert state.stalled_pending_signature is None

    asyncio.run(scenario())


def test_concurrent_batches_share_model_ownership_without_duplicate_work(
    server: RpcServer,
) -> None:
    # These transcripts share example commands. Give each document distinct
    # model input so repeated text from different sessions is not a false duplicate.
    server._get_conn().execute(
        "UPDATE message_state SET content = message_id || ':' || content WHERE content IS NOT NULL"
    )
    server._get_conn().execute(
        "UPDATE tool_calls SET bash_command = id || ':' || bash_command "
        "WHERE bash_command IS NOT NULL"
    )
    entered, release = threading.Event(), threading.Event()

    class RecordingBackend(DeterministicBackend):
        batches: list[tuple[str, ...]]

        def __init__(self, dimensions: int) -> None:
            super().__init__(dimensions)
            self.batches = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.batches.append(tuple(texts))
            entered.set()
            assert release.wait(3)
            return super().embed(texts)

    state = EmbedPhaseState(_config=server._config)
    backend = RecordingBackend(server._config.embedding.dimensions)
    state.backend = backend

    async def scenario() -> None:
        first = asyncio.create_task(server._run_embed_batch(server._config, state))
        assert await asyncio.to_thread(entered.wait, 2)
        second = asyncio.create_task(server._run_embed_batch(server._config, state))
        try:
            await server._writer_call(
                lambda: server._get_conn().execute("UPDATE session_state SET file_size = 47"),
                "raw writer during competing enrichment requests",
            )
        finally:
            release.set()
            results = await asyncio.gather(first, second)
        assert all(result > 0 for result in results)
        assert len(backend.batches) == len(set(backend.batches)), "same inputs generated twice"
        assert embed_phase.find_pending_embeds(server._get_conn(), 0).total == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["snapshot", "generation", "commit"])
def test_cancellation_waits_for_active_work_and_keeps_writer_ownership(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    state = EmbedPhaseState(_config=server._config)
    state.backend = DeterministicBackend(server._config.embedding.dimensions)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    function_name = {
        "snapshot": "prepare_embed_cycle",
        "generation": "generate_prepared_embed_cycle",
        "commit": "commit_prepared_embed_cycle",
    }[stage]
    original = getattr(embed_phase, function_name)

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5), "verifier did not release active work"
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(embed_phase, function_name, blocked)

    async def scenario() -> None:
        task = asyncio.create_task(server._run_embed_batch(server._config, state))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            assert server._write_lock.locked() is (stage != "generation")
            if stage == "generation":
                # Raw writer can commit while the model is blocked.
                await server._writer_call(
                    lambda: server._get_conn().execute("UPDATE session_state SET file_size = 23"),
                    "raw progress during model work",
                )
                assert await server._run_readonly(
                    lambda conn: conn.execute(
                        "SELECT DISTINCT file_size FROM session_state"
                    ).fetchall()
                ) == [(23,)]
            task.cancel()
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=0.02)
            assert server._write_lock.locked() is (stage != "generation")
        finally:
            release.set()
            with suppress(asyncio.CancelledError):
                await task
        assert finished.is_set()
        assert not server._write_lock.locked()
        assert server._get_conn().execute("SELECT 1").fetchone() == (1,)
        if stage != "commit":
            assert server._get_conn().execute(
                "SELECT COUNT(*) FROM message_embeddings"
            ).fetchone() == (0,)

    asyncio.run(scenario())


def test_backend_failure_is_visible_and_a_later_batch_recovers(server: RpcServer) -> None:
    class FailingBackend(DeterministicBackend):
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("injected unavailable backend")

    state = EmbedPhaseState(_config=server._config)
    state.backend = FailingBackend(server._config.embedding.dimensions)
    server._embed_state = state

    async def scenario() -> None:
        assert await server._run_embed_batch(server._config, state) == -1
        status = await server._handle_daemon_status({}, None)
        assert "injected unavailable backend" in status["reconciliation"]["enrichment_error"]
        assert status["reconciliation"]["enrichment_ready"] is False
        assert server._get_conn().execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (
            0,
        )
        state.backend = DeterministicBackend(server._config.embedding.dimensions)
        assert await server._run_embed_batch(server._config, state) > 0
        status = await server._handle_daemon_status({}, None)
        assert status["reconciliation"]["enrichment_error"] is None
        assert state.last_batch_size > 0
        assert (
            server._get_conn().execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] > 0
        )

    asyncio.run(scenario())


async def wait_until(predicate, *, timeout: float = 5) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


def test_inventory_recovers_an_aborted_write_and_finishes_its_scan(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.services import coordinator

    server._config = replace(server._config, daemon=replace(server._config.daemon, interval=1))
    original = coordinator.persist_prepared_inventory
    injected = False

    def invalid_first_batch(*args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            conn = kwargs["conn"]
            conn.execute("BEGIN TRANSACTION")
            conn.execute("SELECT CAST('invalid-inventory' AS INTEGER)")
        return original(*args, **kwargs)

    monkeypatch.setattr(coordinator, "persist_prepared_inventory", invalid_first_batch)

    async def scenario() -> None:
        task = asyncio.create_task(server._run_inventory_loop())
        try:
            await wait_until(lambda: server._inventory_complete)
            assert injected
            assert server._inventory_error is None
            rows = await server._run_readonly(
                lambda conn: conn.execute(
                    "SELECT discovered_count, scan_complete FROM reconciliation_roots "
                    "WHERE source = 'claude_code'"
                ).fetchall()
            )
            assert rows == [(2, True)]
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_inventory_catalogues_an_old_import_on_its_next_walk(server: RpcServer) -> None:
    """REQ-RECON-002: only the walk finds an old-mtime import in a directory nothing
    watches, including when every other source under the root is unchanged."""
    import os

    from recall.db.source_files import SourceCatalog

    server._config = replace(server._config, daemon=replace(server._config.daemon, interval=1))
    imported = Path.home() / ".claude" / "projects" / "imported" / "old-session.jsonl"

    def catalogued(conn: duckdb.DuckDBPyConnection) -> bool:
        catalog = SourceCatalog(conn, clock=time.time)
        return catalog.get("claude_code", str(imported.resolve())) is not None

    async def scenario() -> None:
        task = asyncio.create_task(server._run_inventory_loop())
        try:
            await wait_until(lambda: server._inventory_complete)
            imported.parent.mkdir(parents=True)
            imported.write_text("{}\n", encoding="utf-8")
            os.utime(imported, (1, 1))
            async with asyncio.timeout(10):
                while not await server._run_readonly(catalogued):
                    await asyncio.sleep(0.05)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_watch_commit_updates_durable_runtime_metrics(server: RpcServer) -> None:
    import json

    from recall.services.watcher import get_live_snapshot

    server._runtime_watch = True
    server._config = replace(
        server._config, daemon=replace(server._config.daemon, embed=False, debounce=0)
    )
    path = Path.home() / ".codex/sessions/rollout-metrics.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "watch-metrics"}})
        + "\n"
        + json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "watch runtime metrics"}],
                },
            }
        )
        + "\n"
    )

    async def scenario() -> None:
        runtime, drain = await server._start_watch_mode()
        try:
            await wait_until(lambda: str(path) in get_live_snapshot().live_session_paths)
            with server._session_indexed.subscribe(str(path)) as sub:
                runtime.queue.mark(str(path), now=0)
                event = await sub.wait(timeout=5)
                assert event is not None
                assert event.high_water_idx == 0
            status = await server._handle_daemon_status({}, None)
            assert status["watch_total_indexed"] >= 1
            assert status["runtime_status"]["last_run_kind"] == "daemon-watch"
            assert status["runtime_status"]["last_index_summary"]["indexed"] == 1
            assert status["runtime_status"]["last_index_summary"]["failed"] == 0
            assert status["runtime_status"]["last_successful_at"] is not None
        finally:
            drain.cancel()
            with suppress(asyncio.CancelledError):
                await drain
            await server._stop_watch_mode(runtime)

    asyncio.run(scenario())


def test_watch_fts_rebuild_recovers_and_searches_committed_content(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    import recall.db
    from recall.core.config import FtsConfig

    server._config = replace(
        server._config,
        fts=FtsConfig(backend="duckdb", fields=("content",)),
        daemon=replace(server._config.daemon, embed=False, fts_debounce=0),
    )
    server._get_conn().execute("UPDATE message_state SET content = 'recoverytoken' ")
    original = recall.db.create_fts_indexes
    injected, rebuilt = threading.Event(), threading.Event()

    def fail_first(conn, config):
        if not injected.is_set():
            injected.set()
            conn.execute("BEGIN TRANSACTION")
            conn.execute("SELECT CAST('broken-fts' AS INTEGER)")
        original(conn, config)
        rebuilt.set()

    monkeypatch.setattr(recall.db, "create_fts_indexes", fail_first)

    async def scenario() -> None:
        runtime, drain = await server._start_watch_mode()
        try:
            runtime.fts_debouncer.mark_dirty()
            assert await asyncio.to_thread(rebuilt.wait, 5)
            rows = await server._run_readonly(
                lambda conn: conn.execute(
                    "SELECT COUNT(*) FROM message_state WHERE "
                    "fts_main_message_state.match_bm25(message_id, 'recoverytoken', "
                    "fields := 'fts_content') IS NOT NULL"
                ).fetchone()
            )
            assert rows == (6,)
            assert injected.is_set()
        finally:
            drain.cancel()
            with suppress(asyncio.CancelledError):
                await drain
            await server._stop_watch_mode(runtime)

    asyncio.run(scenario())


def test_watch_harvest_recovers_after_observation_and_database_failures(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.services import watcher

    server._config = replace(
        server._config, daemon=replace(server._config.daemon, live_discovery_interval=1)
    )
    tick_failed, harvested = threading.Event(), threading.Event()
    original_tick = watcher.run_live_discovery_tick
    attempts = 0

    def fail_first_tick(runtime):
        if not tick_failed.is_set():
            tick_failed.set()
            raise OSError("injected observer failure")
        original_tick(runtime)

    def harvest() -> None:
        nonlocal attempts
        attempts += 1
        conn = server._get_conn()
        if attempts == 1:
            conn.execute("BEGIN TRANSACTION")
            conn.execute("SELECT CAST('broken-harvest' AS INTEGER)")
        conn.execute("UPDATE session_state SET file_size = 29")
        harvested.set()

    monkeypatch.setattr(watcher, "run_live_discovery_tick", fail_first_tick)
    monkeypatch.setattr(server, "_harvest_usage_on_shared_conn", harvest)

    async def scenario() -> None:
        runtime, drain = await server._start_watch_mode()
        try:
            assert await asyncio.to_thread(harvested.wait, 5)
            assert tick_failed.is_set()
            assert attempts == 2
            assert await server._run_readonly(
                lambda conn: conn.execute("SELECT DISTINCT file_size FROM session_state").fetchall()
            ) == [(29,)]
        finally:
            drain.cancel()
            with suppress(asyncio.CancelledError):
                await drain
            await server._stop_watch_mode(runtime)

    asyncio.run(scenario())
