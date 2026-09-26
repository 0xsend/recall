"""Bounded background work remains eligible, repairable and observable."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from lane_harness import Lane, build_lane
from recall.core.config import SourceConfig
from recall.db import open_sidecar, sidecar_path
from recall.db.source_files import SourceCatalog, SourceSignature


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Lane]:
    lane = build_lane(tmp_path, monkeypatch)
    yield lane
    asyncio.run(lane.server.stop())


def test_ready_lane_rows_have_one_combined_bound(lane: Lane) -> None:
    catalog = SourceCatalog(lane.server._get_conn(), clock=time.time)
    signature = SourceSignature(1, 2, 3, 4, 5)
    for index in range(600):
        catalog.observe("codex", "/root", f"/root/{index:04}.jsonl", signature)
    lanes = catalog.priority_pending({f"/root/{index:04}.jsonl" for index in range(300)})
    assert sum(map(len, lanes)) <= 256
    assert all(lanes)


def test_disabled_and_unconfigured_roots_never_consume_service_turns(lane: Lane) -> None:
    server = lane.server
    conn = server._get_conn()
    catalog = SourceCatalog(conn, clock=time.time)
    signature = SourceSignature(1, 2, 3, 4, 5)
    for source, root in (
        ("unknown", "/unknown"),
        ("codex", "/disabled"),
        ("claude_code", "/elsewhere"),
    ):
        catalog.observe(source, root, f"{root}/bad.jsonl", signature)
    server._config = replace(server._config, sources={"codex": SourceConfig(roots=())})
    lane.append(lane.watched, "eligible work", uuid="eligible-work")
    # Mark the eligible path pending while keeping the foreign rows older.
    from recall.parsers.claude_code import ClaudeCodeParser
    from recall.services.coordinator import capture_path, observe_path

    parser = ClaudeCodeParser()
    observe_path(parser, capture_path(parser, lane.watched), conn=conn)

    async def scenario() -> None:
        assert await server._serve_raw_turn()
        assert not await server._serve_raw_turn()

    asyncio.run(scenario())
    assert conn.execute(
        "SELECT COUNT(*) FROM message_state WHERE content = 'eligible work'"
    ).fetchone() == (1,)
    assert conn.execute(
        "SELECT SUM(last_serviced_seq) FROM source_files "
        "WHERE root_path IN ('/unknown', '/disabled', '/elsewhere')"
    ).fetchone() == (0,)


def test_poll_repairs_pending_keyword_updates_without_new_raw_work(lane: Lane) -> None:
    server = lane.server
    conn = server._get_conn()
    row = conn.execute(
        "SELECT message_id FROM message_state WHERE content IS NOT NULL LIMIT 1"
    ).fetchone()
    assert row is not None
    message_id = row[0]
    conn.execute(
        "UPDATE message_state SET fts_content = 'repairablekeyword' WHERE message_id = ?",
        [message_id],
    )
    conn.execute(
        "INSERT INTO fts_sidecar_pending(kind, id, op) VALUES ('message', ?, 'upsert')",
        [message_id],
    )

    async def scenario() -> None:
        task = asyncio.create_task(server._run_reconciliation_poll_loop())
        try:
            deadline = asyncio.get_running_loop().time() + 3
            while asyncio.get_running_loop().time() < deadline:
                pending = await server._run_readonly(
                    lambda db: db.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone()[0]
                )
                if pending == 0:
                    break
                await asyncio.sleep(0.02)
            assert pending == 0, "poll never repaired the durable keyword update"
            with closing(open_sidecar(sidecar_path(server._config.data_dir))) as sidecar:
                assert sidecar.execute(
                    "SELECT COUNT(*) FROM message_fts WHERE message_fts MATCH 'repairablekeyword'"
                ).fetchone() == (1,)
        finally:
            server._shutdown_event.set()
            await task

    asyncio.run(scenario())


def test_keyword_repair_advances_past_failures_then_retries_them(lane: Lane) -> None:
    server = lane.server
    conn = server._get_conn()
    conn.executemany(
        "INSERT INTO fts_sidecar_pending(kind, id, op) VALUES ('invalid-kind', ?, 'upsert')",
        [(str(i),) for i in range(256)],
    )
    conn.execute(
        "INSERT INTO fts_sidecar_pending(kind, id, op) "
        "VALUES ('message', 'deleted-message', 'delete')"
    )

    async def scenario() -> None:
        assert await server._repair_keyword_batch() == 0
        assert conn.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone() == (257,)
        assert server._fts_repair_error and "unsupported kind/op" in server._fts_repair_error
        assert await server._repair_keyword_batch() == 1
        assert conn.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone() == (256,)
        # The later successful batch cannot hide the still-failed first batch.
        assert server._fts_repair_error is not None
        conn.execute("UPDATE fts_sidecar_pending SET kind = 'message', op = 'delete'")
        server._next_fts_repair_at = 0
        assert await server._repair_keyword_batch() == 256
        assert conn.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone() == (0,)
        assert server._fts_repair_error is None

    asyncio.run(scenario())


def test_background_enrichment_can_enable_after_starting_disabled(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from recall.services.system_state import EmbedPreconditionResult

    server = lane.server
    generated = threading.Event()

    class Backend:
        dimensions = server._config.embedding.dimensions
        model_id = "maintenance-verifier"
        query_prefix = ""

        def embed(self, texts):
            generated.set()
            return [[0.25] * self.dimensions for _ in texts]

    monkeypatch.setattr("recall.services.embeddings.get_backend", lambda _: Backend())
    monkeypatch.setattr(
        "recall.services.system_state.check_power", lambda: EmbedPreconditionResult(ok=True)
    )
    monkeypatch.setattr(
        "recall.services.system_state.check_load", lambda _: EmbedPreconditionResult(ok=True)
    )
    for path in (lane.watched, lane.unwatched):
        import os

        os.utime(path, (1, 1))
        asyncio.run(server.index_session_now(path))

    async def scenario() -> None:
        task = await server._start_embed_phase()
        assert task is not None, "disabled at startup permanently prevents runtime enrichment"
        try:
            assert not generated.is_set()
            server._config = replace(
                server._config,
                daemon=replace(server._config.daemon, embed=True, embed_idle_session=0),
            )
            assert await asyncio.to_thread(generated.wait, 12)
        finally:
            server._shutdown_event.set()
            await task

    asyncio.run(scenario())


def test_active_observation_pages_cover_large_rosters_and_exclude_foreign_roots(lane: Lane) -> None:
    from recall.services.coordinator import active_observation_page

    conn = lane.server._get_conn()
    now = time.time()
    root = lane.watched.parent
    config = replace(
        lane.server._config,
        sources={
            "codex": SourceConfig(roots=(root,)),
            "claude_code": SourceConfig(roots=()),
            "pi_agent": SourceConfig(roots=()),
            "grok": SourceConfig(roots=()),
            "kimi_code": SourceConfig(roots=()),
        },
    )
    catalog = SourceCatalog(conn, clock=lambda: now)
    signature = SourceSignature(1, 2, 3, int(now * 1e9), 5)
    wanted = {("codex", str(root / f"rollout-{index:04}.jsonl")) for index in range(300)}
    for source, path in sorted(wanted):
        catalog.observe(source, str(root), path, signature)
    catalog.observe("codex", "/foreign", "/foreign/rollout.jsonl", signature)
    catalog.observe("claude_code", str(root), str(root / "disabled.jsonl"), signature)
    seen = []
    cursor = None
    for _ in range(3):
        page = active_observation_page(config, conn=conn, cursor=cursor, now=now)
        assert len(page.targets) <= 128
        seen.extend((target.source, target.source_path) for target in page.targets)
        cursor = page.next_cursor
    assert cursor is None
    assert len(seen) == len(set(seen)) == 300
    assert set(seen) == wanted


def test_active_capture_reports_unreadable_paths_and_keeps_other_observations(lane: Lane) -> None:
    from recall.services.coordinator import (
        ActiveObservationPage,
        ActiveObservationTarget,
        capture_active_observations,
    )

    signature = SourceSignature(1, 2, 3, 4, 5)
    page = ActiveObservationPage(
        (
            ActiveObservationTarget("claude_code", str(lane.watched), signature),
            ActiveObservationTarget(
                "claude_code", str(lane.watched.parent / "missing.jsonl"), signature
            ),
        ),
        None,
    )
    batch, failures = capture_active_observations(lane.server._config, page)
    assert [item.source_path for item in batch.files] == [str(lane.watched.resolve())]
    assert len(failures) == 1 and "FileNotFoundError" in failures[0]
    assert "missing.jsonl" in failures[0]


def count_scheduling_turns(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record when each raw scheduling turn queries the catalog."""
    from recall.services import coordinator

    turns: list[float] = []
    select = coordinator.select_raw_sources

    def counted(*args, **kwargs):
        turns.append(time.monotonic())
        return select(*args, **kwargs)

    monkeypatch.setattr(coordinator, "select_raw_sources", counted)
    return turns


def without_idle_timers(server, monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave an idle scheduler nothing but edges to wake it within a test's run."""
    from recall.services import rpc_server

    monkeypatch.setattr(rpc_server, "_RAW_IDLE_RESCAN_SECONDS", 3600.0)
    server._config = replace(
        server._config, daemon=replace(server._config.daemon, fts_debounce=3600)
    )


async def wait_for(condition, *, seconds: float) -> None:
    async with asyncio.timeout(seconds):
        while not await condition():
            await asyncio.sleep(0.05)


def test_idle_scheduler_waits_for_an_edge(lane: Lane, monkeypatch: pytest.MonkeyPatch) -> None:
    """An idle scheduler takes no turns while nothing changes, and an append
    wakes it: with no timer left to serve it, only the edge can."""
    server = lane.server
    without_idle_timers(server, monkeypatch)
    turns = count_scheduling_turns(monkeypatch)

    def indexed(conn) -> bool:
        return conn.execute(
            "SELECT COUNT(*) FROM message_state WHERE content = 'woken append'"
        ).fetchone() == (1,)

    async def scenario() -> None:
        task = asyncio.create_task(server._run_reconciliation_poll_loop())
        try:
            await wait_for(lambda: asyncio.sleep(0, result=bool(turns)), seconds=10)
            settled = len(turns)
            await asyncio.sleep(2.0)
            assert len(turns) == settled, "an idle scheduler kept taking turns"
            lane.append(lane.watched, "woken append", uuid="woken-append")
            await wait_for(lambda: server._run_readonly(indexed), seconds=30)
        finally:
            server._shutdown_event.set()
            await task

    asyncio.run(scenario())


def test_idle_scheduler_commits_a_walked_import(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transcript only the walk can find is committed once the walk catalogues it."""
    import os

    server = lane.server
    without_idle_timers(server, monkeypatch)
    server._config = replace(server._config, daemon=replace(server._config.daemon, interval=1))
    imported = lane.watched.parent.parent / "imported" / "old-session.jsonl"

    def committed(conn) -> bool:
        source = SourceCatalog(conn, clock=time.time).get("claude_code", str(imported.resolve()))
        return source is not None and source.current

    async def scenario() -> None:
        task = asyncio.create_task(server._run_reconciliation_poll_loop())
        try:
            await wait_for(lambda: asyncio.sleep(0, result=server._inventory_complete), seconds=10)
            transcript = lane.watched.read_text(encoding="utf-8")
            assert '"sessionId":"live-mid-tool"' in transcript
            imported.parent.mkdir()
            imported.write_text(
                transcript.replace('"sessionId":"live-mid-tool"', '"sessionId":"old-session"'),
                encoding="utf-8",
            )
            os.utime(imported, (1, 1))
            await wait_for(lambda: server._run_readonly(committed), seconds=30)
        finally:
            server._shutdown_event.set()
            await task

    asyncio.run(scenario())


def test_idle_scheduler_rescans_for_a_retry_that_comes_due(lane: Lane) -> None:
    """A retry backoff expires with time alone, and no edge announces it."""
    server = lane.server
    server._config = replace(
        server._config, daemon=replace(server._config.daemon, fts_debounce=3600)
    )
    source_path = str(lane.watched.resolve())
    SourceCatalog(server._get_conn(), clock=time.time).fail("claude_code", source_path, "transient")

    def retried(conn) -> bool:
        source = SourceCatalog(conn, clock=time.time).get("claude_code", source_path)
        assert source is not None
        return source.last_error is None

    async def scenario() -> None:
        task = asyncio.create_task(server._run_reconciliation_poll_loop())
        try:
            await wait_for(lambda: server._run_readonly(retried), seconds=20)
        finally:
            server._shutdown_event.set()
            await task

    asyncio.run(scenario())
