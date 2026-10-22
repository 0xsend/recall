"""Service measurements stay fresh while the background monitor probes slowly."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any

import pytest
import recall.services.daemon as daemon
from recall.core.config import AppConfig
from recall.services.rpc_server import RpcServer


@dataclass
class Clock:
    now: float = 1000.0

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.parametrize(
    ("probe_seconds", "fail_once"), [(0.159, False), (5.0, False), (20.0, False), (5.0, True)]
)
def test_background_probe_finishes_within_the_service_age_bound(
    monkeypatch: pytest.MonkeyPatch, probe_seconds: float, fail_once: bool
) -> None:
    clock = Clock()
    server = RpcServer()
    ages: list[float] = []
    measured_at: float | None = None
    probes = 0
    failed = False

    def observe() -> None:
        if measured_at is not None:
            view = server._service_probes.view_for(None)
            assert view.measured_at == measured_at
            ages.append(clock.time() - view.measured_at)

    def probe(_config: AppConfig, record: Any) -> daemon.ServiceView:
        nonlocal measured_at, probes, failed
        # Status continues to read the previous view throughout the slow probe.
        clock.advance(probe_seconds)
        if measured_at is not None:
            observe()
        if fail_once and probes == 1 and not failed:
            failed = True
            raise OSError("service temporarily unavailable")
        measured_at = clock.time()
        probes += 1
        if probes == 4:
            server._shutdown_event.set()
        return daemon.ServiceView(record, None, False, None, measured_at)

    async def wait_for(wait: Awaitable[bool], timeout: float) -> bool:
        # Close the event.wait coroutine without running a real-time timer.
        assert asyncio.iscoroutine(wait)
        wait.close()
        clock.advance(timeout + 0.25)
        observe()
        raise TimeoutError

    async def read_record(_fn: Any) -> None:
        return None

    monkeypatch.setattr(daemon.time, "time", clock.time)
    monkeypatch.setattr(daemon.time, "monotonic", clock.time)
    monkeypatch.setattr(daemon, "probe_service", probe)
    monkeypatch.setattr(server, "_run_readonly", read_record)
    monkeypatch.setattr(asyncio, "wait_for", wait_for)

    asyncio.run(server._service_probe_monitor())

    assert probes == 4
    assert ages
    assert max(ages) <= 30.0, ages


@pytest.mark.parametrize("slow_seconds", [16.0, 20.0])
def test_fast_probes_restore_the_refresh_cadence_after_a_slow_success(
    monkeypatch: pytest.MonkeyPatch, slow_seconds: float
) -> None:
    clock = Clock()
    cache = daemon.ServiceProbeCache(AppConfig.load())
    duration = slow_seconds
    ages: list[float] = []
    previous: daemon.ServiceView | None = None

    def probe(_config: AppConfig, record: Any) -> daemon.ServiceView:
        clock.advance(duration)
        if previous is not None:
            # Reads during the probe keep the actual previous measurement.
            assert cache.view_for(record) is previous
            ages.append(clock.time() - previous.measured_at)
        return daemon.ServiceView(record, None, False, None, clock.time())

    monkeypatch.setattr(daemon.time, "time", clock.time)
    monkeypatch.setattr(daemon.time, "monotonic", clock.time)
    monkeypatch.setattr(daemon, "probe_service", probe)

    previous = cache.refresh(None)
    assert cache.refresh_delay() == 1.0  # The slow probe's deadline is already past.
    clock.advance(cache.refresh_delay())
    duration = 0.125  # Binary-exact times keep the strict age boundary meaningful.
    previous = cache.refresh(None)
    assert cache.refresh_delay() == pytest.approx(15.0)

    clock.advance(14.75)
    assert cache.refresh_delay() == pytest.approx(0.25)
    clock.advance(cache.refresh_delay())
    duration = 15.0  # The full reserve after a fast successful probe.
    previous = cache.refresh(None)
    assert ages == pytest.approx([1.125, 30.0])

    clock.advance(cache.refresh_delay())
    duration = 0.125
    previous = cache.refresh(None)
    assert cache.refresh_delay() == pytest.approx(15.0)
    assert max(ages) <= 30.0
