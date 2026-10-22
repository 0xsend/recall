"""REQ-RECON-008 and REQ-RESIL-021 against real database and worker lifetimes."""

from __future__ import annotations

import asyncio
import os
import sys
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
from recall.services.scheduling import demoted_threads
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
                server._raw_candidate_admitted(candidate, is_active=True).admitted is active_allowed
            )
            assert not server._raw_candidate_admitted(candidate, is_active=False).admitted
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


def test_enrichment_inference_runs_on_a_demoted_thread_and_writer_work_does_not(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    inference_threads: set[str] = set()

    class RecordingBackend(DeterministicBackend):
        def embed(self, texts: list[str]) -> list[list[float]]:
            inference_threads.add(threading.current_thread().name)
            return super().embed(texts)

    state = EmbedPhaseState(_config=server._config)
    state.backend = RecordingBackend(server._config.embedding.dimensions)
    monkeypatch.setattr(
        "recall.services.system_state.check_load", lambda _: EmbedPreconditionResult(ok=True)
    )

    async def scenario() -> str:
        assert await server._run_embed_batch(server._config, state) > 0
        return await server._writer_call(lambda: threading.current_thread().name, "thread probe")

    writer_thread = asyncio.run(scenario())

    demoted = demoted_threads()
    assert inference_threads
    assert inference_threads <= demoted.keys()
    assert writer_thread not in demoted
    expected = "applied" if sys.platform == "darwin" else "unsupported"
    assert {demoted[name] for name in inference_threads} == {expected}


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


def test_enrichment_is_not_ready_while_a_session_waits_out_the_idle_hold(
    server: RpcServer,
) -> None:
    """The loop's cached count is eligible-only, so it reads 0 while a fresh
    session sits inside `embed_idle_session`; readiness must not (REQ-ADAPT-022)."""
    idle = 600
    server._config = replace(
        server._config, daemon=replace(server._config.daemon, embed_idle_session=idle)
    )
    state = EmbedPhaseState(_config=server._config)
    state.backend = DeterministicBackend(server._config.embedding.dimensions)
    server._embed_state = state

    async def age_sessions(seconds_since_write: float) -> None:
        mtime = time.time() - seconds_since_write
        await server._writer_call(
            lambda: server._get_conn().execute("UPDATE session_state SET file_mtime = ?", [mtime]),
            "set session age",
        )

    async def scenario() -> None:
        # Enrichment is drained only once raw indexing is (REQ-DAEMON-079).
        await complete_one_inventory_walk(server)
        await age_sessions(0)
        assert await server._run_embed_batch(server._config, state) == 0
        held = await server._handle_daemon_status({}, None)
        assert held["embed_pending"] == 0
        assert held["embed_pending_eligible"] == 0
        assert held["embed_pending_held"] > 0
        assert held["reconciliation"]["enrichment_ready"] is False

        await age_sessions(idle + 1)
        eligible = await server._handle_daemon_status({}, None)
        assert eligible["embed_pending_eligible"] == held["embed_pending_held"]
        assert eligible["embed_pending_held"] == 0
        assert eligible["reconciliation"]["enrichment_ready"] is False

        for _ in range(16):
            if await server._run_embed_batch(server._config, state) == 0:
                break
        drained = await server._handle_daemon_status({}, None)
        assert drained["embed_pending"] == 0
        assert (drained["embed_pending_eligible"], drained["embed_pending_held"]) == (0, 0)
        assert drained["embed_pending_measured_at"] >= eligible["embed_pending_measured_at"]
        assert drained["reconciliation"]["enrichment_ready"] is True

    asyncio.run(scenario())


def test_enrichment_is_ready_with_embedding_disabled_whatever_is_unembedded(
    server: RpcServer,
) -> None:
    """Work that will never be embedded is nothing to wait for (REQ-ADAPT-022)."""
    server._config = replace(server._config, daemon=replace(server._config.daemon, embed=False))

    status = asyncio.run(server._handle_daemon_status({}, None))

    assert status["embed_pending_eligible"] + status["embed_pending_held"] > 0
    assert status["reconciliation"]["enrichment_ready"] is True


async def complete_one_inventory_walk(server: RpcServer) -> None:
    task = asyncio.create_task(server._run_inventory_loop())
    try:
        await wait_until(lambda: server._inventory_complete)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


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
        raw = None
        try:
            await wait_until(lambda: str(path) in get_live_snapshot().live_session_paths)
            with server._session_indexed.subscribe(str(path)) as sub:
                runtime.queue.mark(str(path), now=0)
                raw = asyncio.create_task(server._run_reconciliation_poll_loop())
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
            if raw is not None:
                raw.cancel()
                with suppress(asyncio.CancelledError):
                    await raw
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
    from recall.services import usage_harvest, watcher

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

    def harvest(conn, batch, *, host) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            conn.execute("BEGIN TRANSACTION")
            conn.execute("SELECT CAST('broken-harvest' AS INTEGER)")
        conn.execute("UPDATE session_state SET file_size = 29")
        harvested.set()

    monkeypatch.setattr(watcher, "run_live_discovery_tick", fail_first_tick)
    monkeypatch.setattr(usage_harvest, "read_grok_usage_batch", lambda conn, path: object())
    monkeypatch.setattr(usage_harvest, "commit_grok_usage_batch", harvest)

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


def test_readiness_waits_for_a_noticed_write_to_be_observed(server: RpcServer) -> None:
    """REQ-DAEMON-079: between a write the watcher reports and its observation the
    catalog still reads current, so no readiness flag may read true."""
    from recall.services.watcher import get_live_snapshot

    # The drain loop must not reach the path on its own while the window is probed.
    server._config = replace(server._config, daemon=replace(server._config.daemon, debounce=600))
    state = EmbedPhaseState(_config=server._config)
    state.backend = DeterministicBackend(server._config.embedding.dimensions)
    server._embed_state = state
    transcript = Path.home() / ".claude" / "projects" / "proj" / "live.jsonl"
    flags = ("raw_indexing_ready", "keyword_search_ready", "enrichment_ready")

    async def reconciliation() -> dict[str, Any]:
        return (await server._handle_daemon_status({}, None))["reconciliation"]

    async def embed_all() -> None:
        for _ in range(16):
            if await server._run_embed_batch(server._config, state) == 0:
                return
        raise AssertionError("embedding did not drain")

    async def scenario() -> None:
        # Recently written, so watch start seeds it into the live set however
        # long ago the lane was built.
        os.utime(transcript)
        await server.index_session_now(transcript)
        inventory = asyncio.create_task(server._run_inventory_loop())
        runtime, drain = await server._start_watch_mode()
        try:
            await wait_until(lambda: server._inventory_complete)
            await wait_until(lambda: str(transcript) in get_live_snapshot().live_session_paths)
            await embed_all()
            settled = await reconciliation()
            assert all(settled[flag] for flag in flags), settled

            with transcript.open("a", encoding="utf-8") as handle:
                handle.write(transcript.read_text(encoding="utf-8").splitlines()[-1] + "\n")
            await wait_until(lambda: runtime.queue.pending_count() == 1)
            noticed = await reconciliation()
            assert not any(noticed[flag] for flag in flags), noticed
            assert noticed["noticed_pending"] == 1

            await server.index_session_now(transcript)
            indexed = await reconciliation()
            assert indexed["noticed_pending"] == 0
            assert indexed["raw_indexing_ready"] is True
            await embed_all()
            assert (await reconciliation())["enrichment_ready"] is True
        finally:
            drain.cancel()
            inventory.cancel()
            for task in (drain, inventory):
                with suppress(asyncio.CancelledError):
                    await task
            await server._stop_watch_mode(runtime)

    asyncio.run(scenario())


def test_a_restarted_daemon_is_not_ready_before_its_own_inventory_walk(server: RpcServer) -> None:
    """REQ-DAEMON-079: the scan records a previous process left say nothing about
    what was written while no daemon was running."""
    from concurrent.futures import ThreadPoolExecutor

    server._config = replace(server._config, daemon=replace(server._config.daemon, embed=False))
    transcript = Path.home() / ".claude" / "projects" / "proj" / "other.jsonl"
    flags = ("raw_indexing_ready", "keyword_search_ready")

    async def first_run() -> dict[str, Any]:
        await complete_one_inventory_walk(server)
        status = (await server._handle_daemon_status({}, None))["reconciliation"]
        await server.stop()
        return status

    before_stop = asyncio.run(first_run())
    assert all(before_stop[flag] for flag in flags), before_stop

    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(transcript.read_text(encoding="utf-8").splitlines()[-1] + "\n")
    restarted = RpcServer(server._config)
    restarted._executor = ThreadPoolExecutor(max_workers=2)

    async def second_run() -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            before_walk = (await restarted._handle_daemon_status({}, None))["reconciliation"]
            await complete_one_inventory_walk(restarted)
            after_walk = (await restarted._handle_daemon_status({}, None))["reconciliation"]
            return before_walk, after_walk
        finally:
            await restarted.stop()

    before_walk, after_walk = asyncio.run(second_run())
    assert before_walk["catalog_scan_complete"] is True
    assert not any(before_walk[flag] for flag in flags), before_walk
    # The walk found the append, so the catalog now says so itself.
    assert after_walk["pending"] == 1
    assert not any(after_walk[flag] for flag in flags), after_walk


@pytest.mark.parametrize("queue_name", ["queue", "discovered"])
@pytest.mark.parametrize("already_noticed", [True, False])
def test_status_does_not_mix_coverage_with_a_later_watcher_settlement(
    server: RpcServer,
    monkeypatch: pytest.MonkeyPatch,
    queue_name: str,
    already_noticed: bool,
) -> None:
    from recall.services import coordinator
    from recall.services.watcher import build_live_watch_runtime

    transcript = Path.home() / ".claude/projects/proj/live.jsonl"
    runtime = build_live_watch_runtime(config=server._config)
    server._watch_runtime = runtime
    queue = getattr(runtime, queue_name)
    coverage_read = threading.Event()
    release_read = threading.Event()
    original = coordinator.reconciliation_status

    def paused_coverage(*args, **kwargs):
        result = original(*args, **kwargs)
        coverage_read.set()
        assert release_read.wait(timeout=5)
        return result

    async def scenario() -> None:
        await complete_one_inventory_walk(server)
        if already_noticed:
            queue.mark(str(transcript))
        monkeypatch.setattr(coordinator, "reconciliation_status", paused_coverage)
        status_task = asyncio.create_task(server._handle_daemon_status({}, None))
        try:
            assert await asyncio.to_thread(coverage_read.wait, 5)
            with transcript.open("a") as handle:
                handle.write(transcript.read_text().splitlines()[-1] + "\n")
            if not already_noticed:
                queue.mark(str(transcript))
            observed_from = time.monotonic()
            await server._observe_requested_source(transcript, server._config, None)
            runtime.settle(str(transcript), observed_from)
        finally:
            release_read.set()
            status = await status_task
        reconciliation = status["reconciliation"]
        assert reconciliation["noticed_pending"] == 0
        assert reconciliation["raw_indexing_ready"] is False
        assert reconciliation["keyword_search_ready"] is False
        assert reconciliation["enrichment_ready"] is False
        monkeypatch.setattr(coordinator, "reconciliation_status", original)
        pending = await server._handle_daemon_status({}, None)
        assert pending["reconciliation"]["pending"] == 1
        await server.index_session_now(transcript)
        assert (await server._handle_daemon_status({}, None))["reconciliation"][
            "raw_indexing_ready"
        ] is True

    asyncio.run(scenario())


def test_status_does_not_mix_embedding_counts_with_later_raw_publication(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.services import daemon

    transcript = Path.home() / ".claude/projects/proj/live.jsonl"
    state = EmbedPhaseState(_config=server._config)
    state.backend = DeterministicBackend(server._config.embedding.dimensions)
    server._embed_state = state
    counts_read = threading.Event()
    release_read = threading.Event()
    original = daemon.count_pending_split

    def paused_counts(*args, **kwargs):
        result = original(*args, **kwargs)
        counts_read.set()
        assert release_read.wait(timeout=5)
        return result

    async def scenario() -> None:
        await complete_one_inventory_walk(server)
        for _ in range(16):
            if await server._run_embed_batch(server._config, state) == 0:
                break
        assert (await server._handle_daemon_status({}, None))["reconciliation"][
            "enrichment_ready"
        ] is True
        with transcript.open("a") as handle:
            handle.write(transcript.read_text().splitlines()[-1] + "\n")
        await server.index_session_now(transcript, await_service=False)
        monkeypatch.setattr(daemon, "count_pending_split", paused_counts)
        status_task = asyncio.create_task(server._handle_daemon_status({}, None))
        try:
            assert await asyncio.to_thread(counts_read.wait, 5)
            await server.index_session_now(transcript)
        finally:
            release_read.set()
            status = await status_task
        assert status["embed_pending_eligible"] + status["embed_pending_held"] == 0
        assert status["reconciliation"]["enrichment_ready"] is False
        monkeypatch.setattr(daemon, "count_pending_split", original)
        published = await server._handle_daemon_status({}, None)
        assert published["embed_pending_eligible"] + published["embed_pending_held"] > 0
        assert published["reconciliation"]["enrichment_ready"] is False

    asyncio.run(scenario())


def _admission_backend(server: RpcServer, monkeypatch: pytest.MonkeyPatch) -> EmbedPhaseState:
    backend = DeterministicBackend(server._config.embedding.dimensions)
    state = EmbedPhaseState(_config=server._config, backend=backend)
    server._embed_state = state
    monkeypatch.setattr(server, "_get_embed_backend", lambda: backend)
    conn = server._get_conn()
    conn.execute(
        "INSERT INTO message_embeddings(message_id, content_embedding) "
        "SELECT m.id, ? FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.source_session_id = 'live-mid-tool' ORDER BY m.idx LIMIT 1",
        [[0.25] * backend.dimensions],
    )
    return state


@pytest.mark.parametrize("phase", ["snapshot", "commit"])
@pytest.mark.parametrize(
    ("mode", "session"), [("vector", None), ("hybrid", None), ("vector", "live-mid-tool")]
)
def test_memory_admission_search_waits_for_enrichment_and_returns(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch, phase: str, mode: str, session: str | None
) -> None:
    """REQ-RPC-020: heavy reads wait while keyword and scalar reads remain available."""
    import importlib

    from recall.services.rpc_server import _READ_REQUEST_TIMEOUT

    search_module = importlib.import_module("recall.services.search")
    state = _admission_backend(server, monkeypatch)
    if session is not None:
        session = (
            server._get_conn()
            .execute("SELECT id FROM sessions WHERE source_session_id = ?", [session])
            .fetchone()[0]
        )
    entered = threading.Event()
    release = threading.Event()
    active = threading.Event()
    original_scan = search_module._vector_search_all
    operation = "prepare_embed_cycle" if phase == "snapshot" else "publish_prepared_embed_cycle"
    original_enrichment = getattr(embed_phase, operation)

    def held_enrichment(*args, **kwargs):
        active.set()
        entered.set()
        try:
            release.wait()
            return original_enrichment(*args, **kwargs)
        finally:
            active.clear()

    def vector_scan(*args, **kwargs):
        if active.is_set():
            raise duckdb.OutOfMemoryException("overlapping enrichment and vector scans")
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(embed_phase, operation, held_enrichment)
    monkeypatch.setattr(search_module, "_vector_search_all", vector_scan)

    async def scenario() -> None:
        enrichment = asyncio.create_task(server._run_embed_batch(server._config, state))
        searches = []
        status = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            # Waiting clients must not fill the ordinary read pool.
            searches = [
                asyncio.create_task(
                    asyncio.wait_for(
                        server._handle_search(
                            {"query": "test suite", "mode": mode, "session": session}, None
                        ),
                        timeout=_READ_REQUEST_TIMEOUT,
                    )
                )
                for _ in range(8)
            ]
            keyword = await asyncio.wait_for(
                server._handle_search({"query": "test suite", "mode": "keyword"}, None), 2
            )
            assert keyword
            assert (await server._handle_stats({}, None))["messages"] > 0
            status = asyncio.create_task(server._handle_daemon_status({}, None))
            assert (await asyncio.wait_for(status, 2))["embed_pending_eligible"] > 0
            admission = server._database_admission
            await wait_until(
                lambda: (
                    len(admission._waiting) == len(searches)
                    or any(search.done() for search in searches)
                )
            )
            assert active.is_set() and admission._writer and admission._scan
            assert not any(search.done() for search in searches), (
                "heavy reads must wait while enrichment owns memory"
            )
            assert len(admission._waiting) == len(searches)
            assert all(
                claim.scan and not claim.writer and not claim.ready.done()
                for claim in admission._waiting
            )
            release.set()
            assert all(await asyncio.gather(*searches))
            assert (await asyncio.wait_for(status, 2))["embed_pending_eligible"] > 0
            assert await asyncio.wait_for(enrichment, 2) > 0
            assert state.last_error is None
        finally:
            release.set()
            await asyncio.gather(
                enrichment, *searches, *([status] if status else []), return_exceptions=True
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("requested", [False, True])
def test_memory_admission_serializes_scans_and_prevents_collision_backoff(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch, requested: bool
) -> None:
    """A concurrent reader cannot cause a snapshot OOM and its 300-second retry delay."""
    import importlib

    search_module = importlib.import_module("recall.services.search")
    state = _admission_backend(server, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    active = threading.Event()
    original_scan = search_module._vector_search_all
    original_prepare = embed_phase.prepare_embed_cycle
    collisions = []

    def vector_scan(*args, **kwargs):
        if active.is_set():
            collisions.append("vector")
            raise duckdb.OutOfMemoryException("overlapping vector scans")
        active.set()
        entered.set()
        try:
            assert release.wait(5)
            return original_scan(*args, **kwargs)
        finally:
            active.clear()

    def prepare(*args, **kwargs):
        if active.is_set():
            collisions.append("snapshot")
            raise duckdb.OutOfMemoryException("reader collision in embed snapshot")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(search_module, "_vector_search_all", vector_scan)
    monkeypatch.setattr(embed_phase, "prepare_embed_cycle", prepare)

    async def scenario() -> None:
        first = asyncio.create_task(
            server._handle_search({"query": "test", "mode": "vector"}, None)
        )
        tasks = [first]
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            second = asyncio.create_task(
                server._handle_search({"query": "test", "mode": "hybrid"}, None)
            )
            tasks.append(second)
            if requested:
                enrichment = asyncio.create_task(
                    server._generate_requested_enrichment(
                        server._config,
                        lambda: prepare(
                            server._config, state, conn=server._get_conn(), max_sessions=1
                        ),
                    )
                )
            else:
                enrichment = asyncio.create_task(server._run_embed_batch(server._config, state))
            tasks.append(enrichment)
            status = asyncio.create_task(server._handle_daemon_status({}, None))
            tasks.append(status)
            assert (await asyncio.wait_for(status, 2))["embed_pending_eligible"] > 0
            # A queued memory owner must not block raw writers or reserve their lock.
            assert await server._writer_call(lambda: 17, "raw writer during admission") == 17
            completed, _ = await asyncio.wait({second, enrichment}, timeout=0.05)
            assert not completed, "heavy operations must wait for the active scan"
            release.set()
            assert await asyncio.wait_for(first, 2)
            assert await asyncio.wait_for(second, 2)
            result = await asyncio.wait_for(enrichment, 2)
            if requested:
                assert isinstance(result, embed_phase.PreparedEmbedCycle)
                assert result.committed_session_ids
            else:
                assert isinstance(result, int)
                assert result > 0
            assert collisions == []
            assert state.last_error is None
            assert state.stalled_until == 0
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())


def test_raw_publication_waits_for_vector_memory_and_preserves_appends(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw commit cannot exhaust memory retained by a concurrent vector scan."""
    import importlib
    import json

    search_module = importlib.import_module("recall.services.search")
    _admission_backend(server, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    memory_errors = []
    original_commit = server._commit_raw_sources

    def scan(conn, statement, parameters):
        with server._corpus_capacity.lease("ordinary"):
            entered.set()
            assert release.wait(5)
            return conn.execute(statement, parameters).fetchall()

    def commit(*args, **kwargs):
        if not release.is_set():
            error = duckdb.OutOfMemoryException("raw allocation overlaps retained vector memory")
            memory_errors.append(error)
            raise error
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(search_module, "execute_scan", scan)
    monkeypatch.setattr(server, "_commit_raw_sources", commit)
    transcript = Path.home() / ".claude/projects/proj/live.jsonl"
    record = json.loads(transcript.read_text().splitlines()[0])
    record.update(
        type="user",
        uuid="raw-under-vector-pressure",
        message={"role": "user", "content": "raw append survives vector memory pressure"},
    )
    with transcript.open("a") as stream:
        stream.write(json.dumps(record) + "\n")

    async def scenario() -> None:
        search = asyncio.create_task(
            server._handle_search({"query": "test", "mode": "vector"}, None)
        )
        publication = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            publication = asyncio.create_task(server.index_session_now(transcript))
            await asyncio.wait({publication}, timeout=0.1)
            release.set()
            assert await asyncio.wait_for(search, 2)
            await asyncio.wait_for(publication, 3)
            contents = await server._run_readonly(
                lambda conn: conn.execute(
                    "SELECT content FROM message_state WHERE content = ?",
                    ["raw append survives vector memory pressure"],
                ).fetchall()
            )
            assert contents == [("raw append survives vector memory pressure",)]
            assert memory_errors == []
        finally:
            release.set()
            await asyncio.gather(
                search, *([publication] if publication else []), return_exceptions=True
            )

    asyncio.run(scenario())


def test_memory_admission_retains_permit_until_cancelled_scan_closes(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    search_module = importlib.import_module("recall.services.search")
    _admission_backend(server, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    active = threading.Event()
    original = search_module._vector_search_all
    collisions = []

    def scan(*args, **kwargs):
        if active.is_set():
            collisions.append("overlap")
        active.set()
        entered.set()
        try:
            assert release.wait(5)
            return original(*args, **kwargs)
        finally:
            active.clear()

    monkeypatch.setattr(search_module, "_vector_search_all", scan)

    async def scenario() -> None:
        first = asyncio.create_task(
            server._handle_search({"query": "test", "mode": "vector"}, None)
        )
        second = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            first.cancel()
            second = asyncio.create_task(
                server._handle_search({"query": "test", "mode": "vector"}, None)
            )
            completed, _ = await asyncio.wait({first, second}, timeout=0.05)
            assert not completed, "cancellation must drain the worker before releasing admission"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert await asyncio.wait_for(second, 2)
            assert collisions == []
        finally:
            release.set()
            await asyncio.gather(first, *([second] if second else []), return_exceptions=True)

    asyncio.run(scenario())


def test_memory_admission_search_resumes_when_background_drain_becomes_idle(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admission_backend(server, monkeypatch)
    server._background_enrichment_idle.clear()

    async def scenario() -> None:
        search = asyncio.create_task(
            server._handle_search({"query": "test", "mode": "vector"}, None)
        )
        try:
            assert await server._handle_search({"query": "test suite", "mode": "keyword"}, None)
            assert (await server._handle_daemon_status({}, None))["embed_pending_eligible"] > 0
            completed, _ = await asyncio.wait({search}, timeout=0.05)
            assert not completed
            server._background_enrichment_idle.set()
            assert await asyncio.wait_for(search, 2)
        finally:
            server._background_enrichment_idle.set()
            await asyncio.gather(search, return_exceptions=True)

    asyncio.run(scenario())


def test_memory_admission_search_yield_expires_without_failing(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admission_backend(server, monkeypatch)
    monkeypatch.setattr("recall.services.rpc_server._SEARCH_ENRICHMENT_WAIT_SECONDS", 0.05)
    server._background_enrichment_idle.clear()

    async def scenario() -> None:
        assert await asyncio.wait_for(
            server._handle_search({"query": "test", "mode": "vector"}, None), 2
        )
        assert not server._background_enrichment_idle.is_set()

    asyncio.run(scenario())


def test_status_reads_while_enrichment_waits_on_unrelated_writer(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _admission_backend(server, monkeypatch)
    entered = threading.Event()
    release = threading.Event()

    def unrelated_write() -> None:
        server._get_conn().execute("SELECT 1").fetchone()
        entered.set()
        assert release.wait(5)

    async def scenario() -> None:
        raw = asyncio.create_task(server._writer_call(unrelated_write, "held raw writer"))
        enrichment = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            enrichment = asyncio.create_task(server._run_embed_batch(server._config, state))
            await wait_until(lambda: state.stage == "snapshot")
            status = await asyncio.wait_for(server._handle_daemon_status({}, None), 1)
            assert status["embed_pending_eligible"] > 0
            assert not enrichment.done()
            release.set()
            await raw
            assert await asyncio.wait_for(enrichment, 2) > 0
        finally:
            release.set()
            await asyncio.gather(raw, *([enrichment] if enrichment else []), return_exceptions=True)

    asyncio.run(scenario())


def test_later_watch_notice_is_catalogued_while_raw_service_is_blocked(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Notice observation must not wait for any earlier transcript to be served."""
    from recall.db.source_files import SourceCatalog

    server._config = replace(
        server._config,
        daemon=replace(server._config.daemon, debounce=0.01, live_discovery_interval=600),
    )
    from recall.services.reconciler import Admission

    # Model preparation back-pressure without replacing observation or persistence.
    monkeypatch.setattr(
        server, "_raw_candidate_admitted", lambda *args, **kwargs: Admission("preparation")
    )
    root = Path.home() / ".claude/projects/proj"
    first = root / "arrival-first.jsonl"
    later = root / "arrival-later.jsonl"
    body = (root / "live.jsonl").read_text()

    async def scenario() -> None:
        await complete_one_inventory_walk(server)
        runtime, drain = await server._start_watch_mode()
        await server._raw_slots.acquire()
        await server._raw_slots.acquire()
        try:
            first.write_text(body)
            runtime.queue.mark(str(first), time.monotonic() - 1)

            async def observed(path: Path) -> bool:
                return await server._run_readonly(
                    lambda conn: (
                        SourceCatalog(conn, clock=time.time).get("claude_code", str(path))
                        is not None
                    )
                )

            async def wait_for_observation(path: Path) -> None:
                async with asyncio.timeout(5):
                    while not await observed(path):
                        await asyncio.sleep(0.01)

            await wait_for_observation(first)
            later.write_text(body)
            runtime.queue.mark(str(later), time.monotonic() - 1)
            await wait_for_observation(later)
            await wait_until(lambda: runtime.noticed_pending() == 0)
            status = (await server._handle_daemon_status({}, None))["reconciliation"]
            assert status["pending"] == 2
            assert status["raw_indexing_ready"] is False
            assert status["keyword_search_ready"] is False
        finally:
            server._shutdown_event.set()
            server._raw_slots.release()
            server._raw_slots.release()
            drain.cancel()
            with suppress(asyncio.CancelledError):
                await drain
            await server._stop_watch_mode(runtime)

    asyncio.run(scenario())


def test_watch_observation_retains_notices_when_its_database_epoch_expires(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.db.source_files import SourceCatalog
    from recall.services import coordinator
    from recall.services.watcher import build_live_watch_runtime

    path = Path.home() / ".claude/projects/proj/epoch-arrival.jsonl"
    path.write_text((path.parent / "live.jsonl").read_text())
    runtime = build_live_watch_runtime(config=server._config)
    runtime.queue.mark(str(path))
    runtime.discovered.mark(str(path))
    entered, release = threading.Event(), threading.Event()
    capture = coordinator.capture_watch_observations

    def held_capture(*args):
        result = capture(*args)
        entered.set()
        assert release.wait(timeout=5)
        return result

    monkeypatch.setattr(coordinator, "capture_watch_observations", held_capture)

    async def scenario() -> None:
        observation = asyncio.create_task(server._observe_watch_paths(runtime, (str(path),)))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            server._database_epoch += 1
        finally:
            release.set()
        assert await observation is False
        assert runtime.noticed_pending() == 1
        assert (
            SourceCatalog(server._get_conn(), clock=time.time).get("claude_code", str(path)) is None
        )
        monkeypatch.setattr(coordinator, "capture_watch_observations", capture)
        assert await server._observe_watch_paths(runtime, (str(path),)) is True
        assert runtime.noticed_pending() == 0
        pending = SourceCatalog(server._get_conn(), clock=time.time).get("claude_code", str(path))
        assert pending is not None and not pending.current


@pytest.mark.parametrize("phase", ["snapshot", "commit"])
def test_status_reads_fresh_counts_during_long_enrichment_writer(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """REQ-RPC-020: a busy enrichment writer cannot delay fresh status counts."""
    state = _admission_backend(server, monkeypatch)
    operation = "prepare_embed_cycle" if phase == "snapshot" else "publish_prepared_embed_cycle"
    original = getattr(embed_phase, operation)
    entered, release = threading.Event(), threading.Event()
    conn = server._get_conn()
    total = conn.execute(
        "SELECT (SELECT COUNT(*) FROM messages) + (SELECT COUNT(*) FROM tool_calls) "
        "- (SELECT COUNT(*) FROM message_embeddings) "
        "- (SELECT COUNT(*) FROM tool_call_embeddings)"
    ).fetchone()[0]
    assert total > 0
    # An idle loop's cached zero must never override fresh database counts.
    state.record_pending(embed_phase.PendingEmbeds(0, 0, []))

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(20)
        return original(*args, **kwargs)

    monkeypatch.setattr(embed_phase, operation, held)

    async def scenario() -> None:
        enrichment = asyncio.create_task(server._run_embed_batch(server._config, state))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            started = time.time()
            status = await asyncio.wait_for(server._handle_daemon_status({}, None), 2)
            assert status["embed_pending_eligible"] == total
            assert status["embed_pending_held"] == 0
            assert started <= status["embed_pending_measured_at"] <= time.time()
            completed, _ = await asyncio.wait({enrichment}, timeout=11)
            assert not completed, "status must complete during the held writer, not after it"
            release.set()
            drained = await asyncio.wait_for(enrichment, 3)
            assert drained > 0
            status = await asyncio.wait_for(server._handle_daemon_status({}, None), 2)
            assert status["embed_pending_eligible"] == total - drained
            assert status["embed_pending_held"] == 0
        finally:
            release.set()
            await asyncio.gather(enrichment, return_exceptions=True)

    asyncio.run(scenario())


def test_vector_and_hybrid_finish_during_checkpoint_with_enrichment_waiting(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RPC-020: checkpoint waits leave scan capacity and useful deadline budget."""
    state = _admission_backend(server, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    server._background_enrichment_idle.clear()

    def checkpoint() -> None:
        server._get_conn().execute("SELECT 1").fetchone()
        entered.set()
        assert release.wait(40)
        server._get_conn().execute("CHECKPOINT")

    async def scenario() -> None:
        writer = asyncio.create_task(server._writer_call(checkpoint, "held checkpoint"))
        tasks = [writer]
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            enrichment = asyncio.create_task(server._run_embed_batch(server._config, state))
            tasks.append(enrichment)
            await wait_until(lambda: state.stage == "snapshot")
            searches = [
                asyncio.create_task(server._handle_search({"query": "test", "mode": mode}, None))
                for mode in ("vector", "hybrid")
            ]
            tasks.extend(searches)
            assert all(await asyncio.wait_for(asyncio.gather(*searches), 30))
            assert not writer.done()
            assert not enrichment.done()
            # The preference itself must leave useful budget, not consume most of it.
            # A second request still completes with 20 s of its deadline reserved.
            assert await asyncio.wait_for(
                server._handle_search({"query": "test", "mode": "vector"}, None), 10
            )
            release.set()
            await writer
            assert await asyncio.wait_for(enrichment, 3) > 0
            assert state.last_error is None
        finally:
            release.set()
            server._background_enrichment_idle.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("checkpoint_contended", [False, True])
def test_inventory_catchup_leaves_active_service_for_captured_appends(
    server: RpcServer, checkpoint_contended: bool
) -> None:
    """Old captured history found by a walk cannot consume append preference."""
    import json

    from recall.db.source_files import SourceCatalog

    conn = server._get_conn()
    paths = conn.execute(
        "SELECT source_path FROM source_files ORDER BY last_serviced_seq"
    ).fetchall()
    history, active = (Path(row[0]) for row in paths)
    # Restoring old bytes changes the observed metadata, so they still need
    # reconciliation, but their old mtime supplies no current activity evidence.
    os.utime(history, (1, 1))
    with active.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "active-after-history-restore",
                    "sessionId": "live-end-turn",
                    "timestamp": "2026-09-07T18:05:00Z",
                    "message": {"role": "user", "content": "appended during catch-up"},
                }
            )
            + "\n"
        )
    server._checkpoint_contention_retries = int(checkpoint_contended)

    async def scenario() -> None:
        await complete_one_inventory_walk(server)
        assert await server._serve_raw_turn()
        catalog = SourceCatalog(conn, clock=time.time)
        appended = catalog.get("claude_code", str(active))
        catching_up = catalog.get("claude_code", str(history))
        assert appended is not None and appended.current
        assert catching_up is not None and not catching_up.current
        assert conn.execute(
            "SELECT count(*) FROM message_state WHERE content = 'appended during catch-up'"
        ).fetchone() == (1,)
        # The same oldest obligation remains serviceable after maintenance.
        server._checkpoint_contention_retries = 0
        assert await server._serve_raw_turn()
        caught_up = catalog.get("claude_code", str(history))
        assert caught_up is not None and caught_up.current

    asyncio.run(scenario())
