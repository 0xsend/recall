"""Reconciliation must isolate preparation from current-session service."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pytest
from lane_harness import Lane, build_lane
from recall.services import coordinator


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Lane]:
    lane = build_lane(tmp_path, monkeypatch)
    yield lane
    asyncio.run(lane.server.stop())


def test_active_source_commits_while_another_source_is_still_preparing(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blocked parser cannot own the scheduler or all preparation capacity."""
    entered = threading.Event()
    release = threading.Event()
    prepare = coordinator.prepare_raw_sources
    lane.append(lane.watched, "slow-source-append", uuid="slow-source-append")
    lane.append(lane.unwatched, "independent-active-append", uuid="independent-active-append")

    def blocked_prepare(selected, parsers, **kwargs):
        if selected[0].source_path == str(lane.watched):
            entered.set()
            assert release.wait(5), "test did not release parser barrier"
        return prepare(selected, parsers, **kwargs)

    monkeypatch.setattr(coordinator, "prepare_raw_sources", blocked_prepare)

    async def scenario() -> None:
        slow = asyncio.create_task(lane.server.index_session_now(lane.watched))
        active = None
        try:
            assert await asyncio.to_thread(entered.wait, 2), "slow source never began preparation"
            active = asyncio.create_task(lane.server.index_session_now(lane.unwatched))
            await asyncio.wait_for(asyncio.shield(active), timeout=1)
            count = await lane.server._run_readonly(
                lambda conn: conn.execute(
                    "SELECT COUNT(*) FROM message_state WHERE content = 'independent-active-append'"
                ).fetchone()[0]
            )
            assert count == 1
            assert not slow.done(), "slow preparation must still be held at the barrier"
        finally:
            release.set()
            await asyncio.wait_for(
                asyncio.gather(*(task for task in (slow, active) if task is not None)), 5
            )

    asyncio.run(scenario())


def test_historical_reservation_preserves_capacity_for_active_preparation(lane: Lane) -> None:
    from recall.db.source_files import SourceCatalog, SourceSignature

    catalog = SourceCatalog(lane.server._get_conn(), clock=lambda: 1)
    history_path = "/synthetic/large-history.jsonl"
    active_path = "/synthetic/active.jsonl"
    catalog.observe("codex", "/synthetic", history_path, SourceSignature(1, 2, 3, 4, 80 * 1024**2))
    catalog.observe("codex", "/synthetic", active_path, SourceSignature(1, 3, 4, 5, 32 * 1024**2))
    history = catalog.get("codex", history_path)
    active = catalog.get("codex", active_path)
    assert history is not None and active is not None
    assert lane.server._raw_candidate_admitted(history, {active_path}, history_busy=False)
    reservation = lane.server._reserve_raw_preparation(history, historical=True)
    try:
        assert lane.server._raw_candidate_admitted(active, {active_path}, history_busy=True)
        assert not lane.server._raw_candidate_admitted(history, {active_path}, history_busy=True)
    finally:
        lane.server._release_raw_preparation(reservation)
    assert lane.server._raw_reserved_bytes == 0


def test_recreate_discards_preparation_claimed_against_the_prior_database(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.parsers.claude_code import ClaudeCodeParser

    server = lane.server
    lane.append(lane.watched, "obsolete-preparation", uuid="obsolete-preparation")
    parser = ClaudeCodeParser()
    coordinator.observe_path(
        parser, coordinator.capture_path(parser, lane.watched), conn=server._get_conn()
    )
    entered = threading.Event()
    release = threading.Event()
    prepare = coordinator.prepare_raw_sources

    def held_prepare(selected, parsers, **kwargs):
        result = prepare(selected, parsers, **kwargs)
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(coordinator, "prepare_raw_sources", held_prepare)

    async def scenario() -> None:
        work = asyncio.create_task(server._serve_raw_turn())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            async with server._raw_turn_lock:
                await asyncio.wait_for(
                    server._writer_call(
                        lambda: server._reset_database(server._config), "owned recreate"
                    ),
                    2,
                )
        finally:
            release.set()
            await asyncio.wait_for(work, 5)
        assert (
            await server._run_readonly(
                lambda conn: conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            )
            == 0
        )
        assert server._raw_claims == {}

    asyncio.run(scenario())


def test_duplicate_refreshes_share_one_source_publication(lane: Lane) -> None:
    lane.append(lane.watched, "coalesced-message", uuid="coalesced-message")

    async def scenario() -> None:
        await asyncio.wait_for(
            asyncio.gather(*(lane.server.index_session_now(lane.watched) for _ in range(24))), 5
        )
        assert (
            await lane.server._run_readonly(
                lambda conn: conn.execute(
                    "SELECT COUNT(*) FROM message_state WHERE content = 'coalesced-message'"
                ).fetchone()[0]
            )
            == 1
        )
        assert lane.server._raw_claims == {}
        assert lane.server._fresh_jobs == {}

    asyncio.run(scenario())


def test_fresh_mtime_imports_do_not_dilute_explicit_activity(lane: Lane) -> None:
    from recall.db.source_files import SourceCatalog, SourceSignature

    catalog = SourceCatalog(lane.server._get_conn(), clock=lambda: 5000)
    recent_import = SourceSignature(1, 2, 5_000_000_000_000, 5_000_000_000_000, 100)
    for index in range(128):
        catalog.observe("codex", "/synthetic", f"/synthetic/import-{index:03}", recent_import)
    active_paths = {f"/synthetic/active-{index:02}" for index in range(12)}
    old_active = SourceSignature(1, 2, 5_000_000_000_000, 1_000_000_000_000, 100)
    for path in sorted(active_paths):
        catalog.observe("codex", "/synthetic", path, old_active)

    active, recent, oldest = catalog.priority_pending(
        active_paths, active_since_ns=4_700_000_000_000
    )

    assert {item.source_path for item in active} == active_paths
    assert all(item.source_path not in active_paths for item in recent)
    assert oldest, "finite history must retain service eligibility"


def test_scheduler_uses_every_event_backed_path_not_the_status_sample(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = lane.watch_runtime([])
    paths = {str(lane.watched.parent / f"event-active-{index:02}.jsonl") for index in range(12)}
    for path in sorted(paths):
        runtime.live_set.promote(Path(path), 1.0, 10.0, bump_event=True)
    lane.server._watch_runtime = runtime
    observed: list[set[str]] = []

    def capture_selection(*_args, active_paths, **_kwargs):
        observed.append(set(active_paths))
        return ()

    monkeypatch.setattr(coordinator, "select_raw_sources", capture_selection)

    assert not asyncio.run(lane.server._serve_raw_turn())
    assert observed == [paths]


def test_continued_source_change_adds_activity_evidence(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    activity_seen = threading.Event()
    note_activity = lane.server._note_source_activity

    def observe(source_path: str) -> None:
        note_activity(source_path)
        if source_path == str(lane.watched):
            activity_seen.set()

    monkeypatch.setattr(lane.server, "_note_source_activity", observe)
    lane.append(lane.watched, "continued-change", uuid="continued-change")

    async def scenario() -> None:
        task = asyncio.create_task(lane.server._run_inventory_loop())
        try:
            assert await asyncio.to_thread(activity_seen.wait, 5)
            assert str(lane.watched) in lane.server._active_source_paths(lane.server._config)
        finally:
            lane.server._shutdown_event.set()
            await task

    asyncio.run(scenario())


def test_later_observed_append_does_not_discard_a_verified_finite_prefix(lane: Lane) -> None:
    """A newer desired generation can remain pending while a captured prefix commits."""
    from recall.db.source_files import SourceCatalog
    from recall.parsers.claude_code import ClaudeCodeParser

    server = lane.server
    conn = server._get_conn()
    parser = ClaudeCodeParser()
    lane.append(lane.watched, "captured-prefix-message", uuid="captured-prefix-message")
    selected = coordinator.observe_path(
        parser, coordinator.capture_path(parser, lane.watched), conn=conn
    )
    prepared = coordinator.prepare_raw_sources((selected,), {selected.source: parser})
    prefix_bytes = lane.watched.stat().st_size
    lane.append(lane.watched, "later-pending-message", uuid="later-pending-message")
    newest = coordinator.observe_path(
        parser, coordinator.capture_path(parser, lane.watched), conn=conn
    )
    assert newest.desired_generation > selected.desired_generation

    coordinator.commit_prepared_raw_sources(prepared, server._config, conn=conn)

    assert conn.execute(
        "SELECT COUNT(*) FROM message_state WHERE content = 'captured-prefix-message'"
    ).fetchone() == (1,)
    assert conn.execute(
        "SELECT COUNT(*) FROM message_state WHERE content = 'later-pending-message'"
    ).fetchone() == (0,)
    catalog = SourceCatalog(conn, clock=lambda: 0)
    pending = catalog.get(selected.source, selected.source_path)
    assert pending is not None
    assert pending.committed_offset == prefix_bytes
    assert pending.signature is not None
    assert pending.signature.size == lane.watched.stat().st_size
    assert not pending.current
    assert pending.last_error is None


def test_status_reports_an_active_read_without_waiting_for_it(lane: Lane) -> None:
    entered = threading.Event()
    release = threading.Event()

    def held_read(conn) -> None:
        conn.execute("SELECT 1").fetchone()
        entered.set()
        assert release.wait(5), "read barrier not released"

    async def scenario() -> None:
        reader = asyncio.create_task(lane.server._run_readonly(held_read))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            status = await asyncio.wait_for(lane.server._handle_daemon_status({}, None), 1)
            reconciliation = status["reconciliation"]
            assert reconciliation["ordinary_read_active"] >= 1
            assert reconciliation["ordinary_read_oldest_age_seconds"] is not None
            assert reconciliation["ordinary_read_oldest_age_seconds"] >= 0
            assert reconciliation["recovery_waiting"] == 0
        finally:
            release.set()
            await asyncio.wait_for(reader, 5)

    asyncio.run(scenario())


def test_model_preparation_does_not_hold_database_lifecycle_access(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    server = lane.server

    class Backend:
        query_prefix = ""

        def embed(self, texts):
            entered.set()
            assert release.wait(5), "model barrier not released"
            return [[0.25] * server._config.embedding.dimensions for _ in texts]

    monkeypatch.setattr(server, "_get_embed_backend", lambda: Backend())

    def lifecycle_read() -> int:
        server._conn_lifecycle_gate.acquire_write()
        try:
            return server._open_conn_unlocked().execute("SELECT 42").fetchone()[0]
        finally:
            server._conn_lifecycle_gate.release_write()

    async def scenario() -> None:
        search = asyncio.create_task(
            server._handle_search({"query": "hello", "mode": "vector"}, None)
        )
        maintenance = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            # A lifecycle owner must not wait for a model that owns no database work.
            maintenance = asyncio.create_task(
                server._writer_call(lifecycle_read, "lifecycle probe")
            )
            assert await asyncio.wait_for(asyncio.shield(maintenance), 1) == 42
        finally:
            release.set()
            await asyncio.wait_for(
                asyncio.gather(*(task for task in (search, maintenance) if task is not None)), 5
            )

    asyncio.run(scenario())


def test_saturated_read_capacity_does_not_queue_an_independent_writer(lane: Lane) -> None:
    server = lane.server
    server._read_executor = ThreadPoolExecutor(max_workers=2)
    server._read_slots = asyncio.BoundedSemaphore(2)
    entered = threading.Barrier(3, timeout=2)
    release = threading.Event()

    def held_read(conn):
        conn.execute("SELECT 1").fetchone()
        entered.wait()
        assert release.wait(5), "read barrier not released"

    async def scenario() -> None:
        readers = [asyncio.create_task(server._run_readonly(held_read)) for _ in range(2)]
        try:
            await asyncio.to_thread(entered.wait)
            writer = asyncio.create_task(
                server._writer_call(
                    lambda: server._get_conn().execute("SELECT 42").fetchone()[0],
                    "writer independence probe",
                )
            )
            assert await asyncio.wait_for(asyncio.shield(writer), 1) == 42
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*readers), 5)

    asyncio.run(scenario())


def test_cancelled_read_interrupts_its_cursor_and_writer_remains_usable(lane: Lane) -> None:
    server = lane.server
    entered = threading.Event()
    release = threading.Event()
    outcome: list[str] = []

    def barrier(value: int) -> int:
        entered.set()
        assert release.wait(5), "query barrier not released"
        return value

    def long_read(conn) -> None:
        conn.create_function(
            "responsive_query_barrier", barrier, ["BIGINT"], "BIGINT", side_effects=True
        )
        try:
            conn.execute("SELECT SUM(responsive_query_barrier(i)) FROM range(1000) t(i)").fetchall()
            outcome.append("completed")
        except duckdb.InterruptException:
            outcome.append("interrupted")
            raise

    async def scenario() -> None:
        task = asyncio.create_task(server._run_readonly(long_read))
        try:
            assert await asyncio.to_thread(entered.wait, 2), "read never reached query barrier"
            task.cancel()
            await asyncio.sleep(0)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            assert outcome == ["interrupted"]
            assert (
                await asyncio.wait_for(
                    server._writer_call(
                        lambda: server._get_conn().execute("SELECT 42").fetchone()[0],
                        "writer after interrupted reader",
                    ),
                    1,
                )
                == 42
            )
        finally:
            release.set()

    asyncio.run(scenario())


def test_automatic_search_keeps_keyword_results_when_model_download_is_unavailable(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = lane.server
    lane.append(lane.watched, "offlinecontrol", uuid="offlinecontrol")
    asyncio.run(server.index_session_now(lane.watched))
    conn = server._get_conn()
    message_id = conn.execute(
        "SELECT message_id FROM message_state WHERE content = 'offlinecontrol'"
    ).fetchone()[0]
    conn.execute(
        "INSERT OR REPLACE INTO message_embeddings(message_id, content_embedding) VALUES (?, ?)",
        [message_id, [0.25] * server._config.embedding.dimensions],
    )

    def unavailable():
        raise RuntimeError("failed to download model: offline")

    monkeypatch.setattr(server, "_get_embed_backend", unavailable)
    results = asyncio.run(server._handle_search({"query": "offlinecontrol"}, None))
    assert any(result.content == "offlinecontrol" for result in results)
    assert all(result.lexical_match for result in results)


def test_explicit_vector_failure_never_returns_keyword_results(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable():
        raise ValueError("embedding backend unavailable")

    monkeypatch.setattr(lane.server, "_get_embed_backend", unavailable)
    with pytest.raises((RuntimeError, ValueError), match="embedding backend unavailable"):
        asyncio.run(lane.server._handle_search({"query": "hello", "mode": "vector"}, None))


def test_checkpoint_contention_defers_without_blocking_independent_readers(lane: Lane) -> None:
    from recall.db.connection import checkpoint_wal_if_due

    server = lane.server
    held = threading.Event()
    release = threading.Event()

    def held_reader(conn):
        conn.execute("BEGIN")
        try:
            conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            held.set()
            assert release.wait(5), "reader barrier not released"
        finally:
            conn.execute("ROLLBACK")

    async def scenario() -> None:
        reader = asyncio.create_task(server._run_readonly(held_reader))
        checkpoint = None
        try:
            assert await asyncio.to_thread(held.wait, 2)
            await server._writer_call(
                lambda: server._get_conn().execute(
                    "CREATE TABLE checkpoint_conflict(value INTEGER)"
                ),
                "persistent DDL after reader snapshot",
            )
            checkpoint = asyncio.create_task(
                server._writer_call(
                    lambda: checkpoint_wal_if_due(
                        server._get_conn(), server._config.db_path, high_water_bytes=1
                    ),
                    "checkpoint under old read snapshot",
                )
            )
            result = await asyncio.wait_for(asyncio.shield(checkpoint), 1)
            assert result.contended
            assert (
                await asyncio.wait_for(
                    server._run_readonly(lambda conn: conn.execute("SELECT 42").fetchone()[0]), 1
                )
                == 42
            )
            assert not reader.done()
        finally:
            release.set()
            await asyncio.wait_for(
                asyncio.gather(
                    *(task for task in (reader, checkpoint) if task is not None),
                    return_exceptions=True,
                ),
                5,
            )
        completed = await server._writer_call(
            lambda: checkpoint_wal_if_due(
                server._get_conn(), server._config.db_path, high_water_bytes=1
            ),
            "checkpoint after reader release",
        )
        assert completed.attempted
        assert not completed.contended

    asyncio.run(scenario())
