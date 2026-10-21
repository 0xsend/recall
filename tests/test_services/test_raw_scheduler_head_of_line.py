"""The raw scheduler must never refuse the same head forever.

A transient capacity refusal retains its lane position while other admissible
work proceeds. A source whose own reservation exceeds the preparation budget
has no release to wait for, so it must be admitted alone rather than refused
forever.

Observed live (daemon pid 52120, 2026-09-18): 1,376 `writer raw scheduling`
calls and zero `writer raw commit` in the same window, `reconciliation.pending`
frozen at 9,607 for 25 minutes, `raw_preparation_reserved_bytes` 0 (nothing
claimed), `wal_pressure_blocked` false, `checkpoint_contention_retries` 0 --
every gate open except admission, with four pending codex rollouts of 143-301
MiB whose reservations (4x size) are 574-1204 MiB against a 512 MiB budget.
Three `recall index` requests were parked on that queue at once.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest
from lane_harness import lane_config
from recall.core.rpc_types import RpcError
from recall.db.schema import ensure_schema
from recall.db.source_files import SourceCatalog, SourceSignature
from recall.services.reconciler import FairScheduler
from recall.services.rpc_server import (
    _RAW_CAPTURE_MULTIPLIER,
    _RAW_PREPARATION_BYTES,
    RpcServer,
)

# Larger than the whole preparation budget once multiplied, so no claim release
# can ever make room for it. The live offender was 300.9 MiB.
OVERSIZED_BYTES = (_RAW_PREPARATION_BYTES // _RAW_CAPTURE_MULTIPLIER) + 1
ORDINARY_BYTES = 4 * 1024 * 1024


def _observe(catalog: SourceCatalog, source_path: str, size: int, *, mtime_ns: int) -> None:
    catalog.observe(
        "codex",
        "/synthetic",
        source_path,
        SourceSignature(1, abs(hash(source_path)) % 10**9, mtime_ns, mtime_ns, size),
    )


@pytest.mark.parametrize("borrow_mib", [150, 200])
def test_three_claims_cannot_perpetuate_an_owed_memory_refusal(
    tmp_path: Path, borrow_mib: int
) -> None:
    """A 300 MiB history claim gets room despite continuous active work."""
    from contextlib import closing
    from dataclasses import replace

    from recall.db.source_files import source_key

    with closing(duckdb.connect(":memory:")) as conn:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        history_path = "/synthetic/history.jsonl"
        _observe(catalog, history_path, 75 * 1024**2, mtime_ns=1)
        active = {f"/synthetic/active-{index}.jsonl" for index in range(6)}
        for path in sorted(active):
            size = borrow_mib * 1024**2 // _RAW_CAPTURE_MULTIPLIER
            _observe(catalog, path, size - 1, mtime_ns=2)
            assert catalog.acknowledge("codex", path, 1, size - 1, "a" * 64)
            _observe(catalog, path, size, mtime_ns=3)
        server = RpcServer(lane_config(tmp_path))
        scheduler = FairScheduler(clock=lambda: 100.0)
        claims = []
        history_served = False
        try:
            for _ in range(60):
                # One release wakes one select, with up to three concurrent
                # preparations. Replenish every completed active source.
                if len(claims) == 3:
                    _, reservation = claims.pop(0)
                    server._release_raw_preparation(reservation)
                selected = scheduler.select(
                    catalog,
                    active,
                    limit=1,
                    excluded_keys=tuple(source_key(s.source, s.source_path) for s, _ in claims),
                    can_admit=lambda item, priority: server._raw_candidate_admitted(
                        item.source, is_active=item.is_active, priority=priority
                    ),
                )
                if not selected:
                    assert claims, "pending work must progress once claims drain"
                    _, reservation = claims.pop(0)
                    server._release_raw_preparation(reservation)
                    continue
                item = selected[0]
                source = item.source
                claims.append(
                    (source, server._reserve_raw_preparation(source, historical=not item.is_active))
                )
                if source.source_path == history_path:
                    history_served = True
                    break
                assert source.signature is not None
                assert catalog.acknowledge(
                    source.source,
                    source.source_path,
                    source.desired_generation,
                    source.signature.size,
                    "a" * 64,
                )
                catalog.observe(
                    source.source,
                    "/synthetic",
                    source.source_path,
                    replace(source.signature, mtime_ns=source.signature.mtime_ns + 1),
                )
            assert history_served, "active borrows kept the owed history source out for 60 turns"
        finally:
            for _, reservation in claims:
                server._release_raw_preparation(reservation)


def test_a_source_larger_than_the_budget_does_not_wedge_the_queue(tmp_path: Path) -> None:
    """REQ-RECON-024: an oversized head is served alone, not refused forever.

    Red before the fix: `select` returned nothing on every call while nine
    ordinary sources sat behind the oversized one, which is how the live daemon
    logged 1,376 scheduling calls and zero commits.
    """
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    catalog = SourceCatalog(conn, clock=time.time)
    # Observed first, so it holds the head of the fair order.
    _observe(catalog, "/synthetic/oversized.jsonl", OVERSIZED_BYTES, mtime_ns=1)
    for index in range(9):
        _observe(catalog, f"/synthetic/ordinary-{index}.jsonl", ORDINARY_BYTES, mtime_ns=2 + index)

    server = RpcServer(lane_config(tmp_path))
    scheduler = FairScheduler(clock=time.time)
    selected = scheduler.select(
        catalog,
        set(),
        limit=1,
        # The shape `_serve_raw_turn` uses: nothing is active, and a backlog
        # older than the idle threshold is nobody's "recent", so the oldest
        # lane is what answers and the oversized source is its head.
        active_since_ns=int(time.time() * 1e9),
        can_admit=lambda item, priority: server._raw_candidate_admitted(
            item.source, is_active=item.is_active, priority=priority
        ),
    )

    assert selected, (
        "the scheduler selected nothing while ten sources were pending; one "
        "oversized head refuses every turn and stops every lane behind it"
    )
    assert selected[0].source.source_path == "/synthetic/oversized.jsonl"


def test_an_oversized_source_yields_to_work_already_in_preparation(tmp_path: Path) -> None:
    """It runs alone: admitted on an idle budget, refused while anything is claimed."""
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    catalog = SourceCatalog(conn, clock=time.time)
    _observe(catalog, "/synthetic/oversized.jsonl", OVERSIZED_BYTES, mtime_ns=1)
    _observe(catalog, "/synthetic/ordinary.jsonl", ORDINARY_BYTES, mtime_ns=2)
    oversized = catalog.get("codex", "/synthetic/oversized.jsonl")
    ordinary = catalog.get("codex", "/synthetic/ordinary.jsonl")
    assert oversized is not None and ordinary is not None

    server = RpcServer(lane_config(tmp_path))
    assert server._raw_candidate_admitted(oversized, is_active=False).admitted

    reservation = server._reserve_raw_preparation(ordinary, historical=True)
    try:
        assert not server._raw_candidate_admitted(oversized, is_active=False).admitted
    finally:
        server._release_raw_preparation(reservation)
    assert server._raw_reserved_bytes == 0

    # And the oversized reservation itself is accepted rather than asserted away.
    solo = server._reserve_raw_preparation(oversized, historical=True)
    server._release_raw_preparation(solo)


def test_an_oversized_claim_leaves_live_sources_their_protected_slice(tmp_path: Path) -> None:
    """REQ-RECON-024: running alone must not also mean running instead of live work.

    Red before the fix: the oversized claim was charged to the shared budget it
    already exceeds, so every live source was refused for the length of its
    preparation -- `recall live` and `show --fresh` refresh through this
    admission and answered stale until it committed.
    """
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    catalog = SourceCatalog(conn, clock=time.time)
    _observe(catalog, "/synthetic/oversized.jsonl", OVERSIZED_BYTES, mtime_ns=1)
    _observe(catalog, "/synthetic/live.jsonl", ORDINARY_BYTES, mtime_ns=2)
    oversized = catalog.get("codex", "/synthetic/oversized.jsonl")
    live = catalog.get("codex", "/synthetic/live.jsonl")
    assert oversized is not None and live is not None

    server = RpcServer(lane_config(tmp_path))
    claim = server._reserve_raw_preparation(oversized, historical=True)
    try:
        assert server._raw_candidate_admitted(live, is_active=True).admitted, (
            "a live source was refused while an oversized historical claim ran alone; "
            "every `--fresh` read then answers stale for the length of its parse"
        )
        # The slice is the live lane's, not a second history lane.
        ordinary = catalog.get("codex", "/synthetic/live.jsonl")
        assert ordinary is not None
        assert not server._raw_candidate_admitted(ordinary, is_active=False).admitted
    finally:
        server._release_raw_preparation(claim)
    assert server._raw_reserved_bytes == 0


def test_a_requested_path_gives_up_with_the_real_reason_when_nothing_is_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RECON-025: an unservable path fails truthfully instead of waiting forever.

    Red before the fix: `_index_requested_source` looped on `_serve_raw_turn`
    with no bound, so the handler never returned, the index turn was never
    released, and every later `recall index` queued behind it.
    """
    from lane_harness import build_lane

    lane = build_lane(tmp_path, monkeypatch)
    monkeypatch.setattr("recall.services.rpc_server._REQUESTED_SERVICE_STALL_SECONDS", 0.5)
    # A scheduler that serves nobody -- exactly the live wedge, without needing
    # an oversized transcript to produce it.
    monkeypatch.setattr("recall.services.coordinator.select_raw_sources", lambda *a, **k: ())
    lane.append(lane.watched, "needs servicing", uuid="stall-155")

    async def scenario() -> None:
        with pytest.raises(RpcError) as err:
            await asyncio.wait_for(lane.server.index_session_now(lane.watched), timeout=30)
        assert "served no source" in err.value.message
        assert str(lane.watched) in err.value.message
        assert "recall daemon status" in err.value.message

    asyncio.run(scenario())
    asyncio.run(lane.server.stop())


class _FakeWriter:
    """Just enough `asyncio.StreamWriter` to hang up on the handler."""

    def __init__(self) -> None:
        self._closing = False

    def write(self, data: bytes) -> None:
        if self._closing:
            raise BrokenPipeError("peer closed")

    async def drain(self) -> None:
        if self._closing:
            raise ConnectionResetError("peer closed")

    def is_closing(self) -> bool:
        return self._closing

    def hang_up(self) -> None:
        self._closing = True


def test_an_abandoned_index_request_ends_and_releases_its_turn(tmp_path: Path) -> None:
    """REQ-RPC-018: a departed client ends the index it asked for.

    Red before the fix: `recall.index` was dispatched with no peer watch at all,
    so the connection loop sat inside the handler, never read EOF, and the index
    turn stayed held -- the live daemon reported three in-flight requests long
    after every client had been killed.
    """
    from recall.services.rpc_server import ClientConnection, ClientDisconnected

    server = RpcServer(lane_config(tmp_path))
    writer = _FakeWriter()
    client = ClientConnection(cast(asyncio.StreamWriter, writer))
    started = asyncio.Event()

    async def never_returns(_params: dict[str, Any], _client: ClientConnection | None) -> Any:
        async with server._index_turn():
            started.set()
            await asyncio.Event().wait()

    async def scenario() -> None:
        request = asyncio.create_task(
            server._run_bounded_read_handler(
                "recall.index",
                never_returns,
                {},
                client,
                None,
                cast(asyncio.StreamWriter, writer),
                timeout=None,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        assert server._index_turns.in_flight == 1

        writer.hang_up()
        with pytest.raises(ClientDisconnected):
            await asyncio.wait_for(request, timeout=5)
        assert server._index_turns.in_flight == 0, (
            "the abandoned request kept its index turn; every later index queues behind it"
        )

    asyncio.run(scenario())


def test_captured_retry_keeps_active_admission_outside_the_active_roster(tmp_path: Path) -> None:
    """A retry becoming due between roster reads keeps captured-live protection."""
    from contextlib import closing
    from dataclasses import replace

    with closing(duckdb.connect(":memory:")) as conn:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        signature = SourceSignature(1, 2, 3, 4, 5)
        retry_path = "/root/retrying.jsonl"
        catalog.observe("codex", "/root", retry_path, signature)
        assert catalog.acknowledge("codex", retry_path, 1, 5, "a" * 64)
        catalog.observe("codex", "/root", retry_path, replace(signature, size=6))
        catalog.fail("codex", retry_path, "temporary read failure")
        live_paths = {f"/root/live-{index:03}.jsonl" for index in range(86)}
        for path in sorted(live_paths):
            catalog.observe("codex", "/root", path, signature)
            assert catalog.acknowledge("codex", path, 1, 5, "a" * 64)
            catalog.observe("codex", "/root", path, replace(signature, size=6))
        active_paths = live_paths | {retry_path}
        scheduler = FairScheduler(clock=lambda: 101.0)
        before_retry = SourceCatalog(conn, clock=lambda: 101.0)
        assert len(scheduler.select(before_retry, active_paths, limit=6)) == 6
        # The active and recent queries run before the retry deadline; the
        # oldest query runs after it. The retry is absent from the active roster.
        query_times = iter((101.0, 101.0, 103.0))
        catalog = SourceCatalog(conn, clock=lambda: next(query_times, 103.0))
        server = RpcServer(lane_config(tmp_path))
        server._checkpoint_contention_retries = 1
        selected = scheduler.select(
            catalog,
            active_paths,
            limit=1,
            can_admit=lambda item, priority: server._raw_candidate_admitted(
                item.source, is_active=item.is_active, priority=priority
            ),
        )
        assert len(selected) == 1, "captured live retry was refused as historical work"
        item = selected[0]
        assert item.source.source_path == retry_path
        assert item.lane == "oldest"
        assert item.is_active
        reservation = server._reserve_raw_preparation(item.source, historical=not item.is_active)
        try:
            assert not reservation.historical
            assert server._raw_historical_reserved_bytes == 0
        finally:
            server._release_raw_preparation(reservation)


@pytest.mark.parametrize("checkpoint_deferred", [False, True])
def test_borrowed_capacity_leaves_room_for_the_owed_history_source(
    tmp_path: Path, checkpoint_deferred: bool
) -> None:
    """Continuous active claims cannot keep an older large source ineligible."""
    from contextlib import closing
    from dataclasses import replace

    from recall.db.source_files import source_key

    with closing(duckdb.connect(":memory:")) as conn:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        history_path = "/synthetic/history.jsonl"
        _observe(catalog, history_path, 80 * 1024**2, mtime_ns=1)
        active = {f"/synthetic/active-{index}.jsonl" for index in range(8)}
        small_path = "/synthetic/small.jsonl"
        active.add(small_path)
        for path in sorted(active):
            size = ORDINARY_BYTES if path == small_path else 60 * 1024**2
            _observe(catalog, path, size, mtime_ns=2)
            assert catalog.acknowledge("codex", path, 1, size, "a" * 64)
            _observe(catalog, path, size + 1, mtime_ns=3)
        scheduler = FairScheduler(clock=lambda: 100.0)
        first = scheduler.select(catalog, active, limit=6)
        assert len(first) == 6
        for item in first:
            source = item.source
            assert source.signature is not None
            assert catalog.acknowledge(
                "codex",
                source.source_path,
                source.desired_generation,
                source.signature.size,
                "a" * 64,
            )
            catalog.observe(
                "codex",
                "/synthetic",
                source.source_path,
                replace(source.signature, size=source.signature.size + 1),
            )
        held = catalog.get("codex", "/synthetic/active-0.jsonl")
        assert held is not None
        server = RpcServer(lane_config(tmp_path))
        held_claim = server._reserve_raw_preparation(held, historical=False)
        server._checkpoint_contention_retries = int(checkpoint_deferred)
        loan = None
        try:
            selected = scheduler.select(
                catalog,
                active,
                limit=1,
                excluded_keys=(source_key(held.source, held.source_path),),
                can_admit=lambda item, priority: server._raw_candidate_admitted(
                    item.source, is_active=item.is_active, priority=priority
                ),
            )
            if checkpoint_deferred:
                assert len(selected) == 1
                assert selected[0].source.source_path in active - {small_path, held.source_path}
                loan = server._reserve_raw_preparation(selected[0].source, historical=False)
            else:
                assert selected == [], (
                    "even a small borrow must wait while existing claims occupy owed headroom"
                )
            server._release_raw_preparation(held_claim)
            held_claim = None
            if checkpoint_deferred:
                assert loan is not None
                server._release_raw_preparation(loan)
                server._checkpoint_contention_retries = 0
            small = catalog.get("codex", small_path)
            assert small is not None
            loan = server._reserve_raw_preparation(small, historical=False)
            # A compatible active claim remains in flight: no idle budget is needed.
            recovered = scheduler.select(
                catalog,
                active,
                limit=1,
                can_admit=lambda item, priority: server._raw_candidate_admitted(
                    item.source, is_active=item.is_active, priority=priority
                ),
            )
            assert len(recovered) == 1
            assert recovered[0].source.source_path == history_path
        finally:
            if held_claim is not None:
                server._release_raw_preparation(held_claim)
            if loan is not None:
                server._release_raw_preparation(loan)


@pytest.mark.parametrize("is_active", [False, True])
def test_oversized_admission_does_not_require_ordinary_claims_to_drain(
    tmp_path: Path, is_active: bool
) -> None:
    """The same oversized/ordinary pair fits in either admission order."""
    from contextlib import closing

    with closing(duckdb.connect(":memory:")) as conn:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        _observe(catalog, "/synthetic/ordinary.jsonl", ORDINARY_BYTES, mtime_ns=1)
        assert catalog.acknowledge(
            "codex", "/synthetic/ordinary.jsonl", 1, ORDINARY_BYTES, "a" * 64
        )
        _observe(catalog, "/synthetic/ordinary.jsonl", ORDINARY_BYTES + 1, mtime_ns=2)
        _observe(catalog, "/synthetic/oversized.jsonl", OVERSIZED_BYTES, mtime_ns=3)
        if is_active:
            assert catalog.acknowledge(
                "codex", "/synthetic/oversized.jsonl", 1, OVERSIZED_BYTES, "a" * 64
            )
            _observe(catalog, "/synthetic/oversized.jsonl", OVERSIZED_BYTES + 1, mtime_ns=4)
        ordinary = catalog.get("codex", "/synthetic/ordinary.jsonl")
        oversized = catalog.get("codex", "/synthetic/oversized.jsonl")
        assert ordinary is not None and oversized is not None
        server = RpcServer(lane_config(tmp_path))
        ordinary_claim = server._reserve_raw_preparation(ordinary, historical=False)
        oversized_claim = None
        try:
            assert server._raw_candidate_admitted(oversized, is_active=is_active).admitted
            oversized_claim = server._reserve_raw_preparation(oversized, historical=not is_active)
            assert not server._raw_candidate_admitted(oversized, is_active=True).admitted, (
                "only one oversized reservation may be held"
            )
        finally:
            if oversized_claim is not None:
                server._release_raw_preparation(oversized_claim)
            server._release_raw_preparation(ordinary_claim)
