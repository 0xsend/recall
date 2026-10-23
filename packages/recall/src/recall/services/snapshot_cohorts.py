"""Ephemeral, bounded ownership of snapshots for callers already waiting.

A cohort stops admitting callers before its worker begins the transaction.
Results stay private to the owner and are copied on delivery, so caller-local
projection and overlays cannot mutate another response. Completed work is never
used to satisfy a later arrival.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Hashable
from copy import deepcopy
from typing import Any


class SnapshotCapacityError(Exception):
    """All connection-derived waiter or draining-worker capacity is occupied."""


class _Snapshot:
    __slots__ = ("__value",)

    def __init__(self, value: Any) -> None:
        self.__value = value

    def copy(self) -> Any:
        return deepcopy(self.__value)


class _Cohort:
    def __init__(self, operation: Callable[[Callable[[], None]], Awaitable[Any]]) -> None:
        self._lock = threading.Lock()
        self._started = False
        self._accepting = True
        self._waiters = 0
        self.task = asyncio.create_task(self._run(operation))

    async def _run(self, operation: Callable[[Callable[[], None]], Awaitable[Any]]) -> _Snapshot:
        return _Snapshot(await operation(self._start))

    def _start(self) -> None:
        # The worker and event loop arbitrate admission on the same lock.
        with self._lock:
            self._started = True
            self._accepting = False

    def join(self) -> bool:
        with self._lock:
            if not self._accepting or self.task.done():
                return False
            self._waiters += 1
            return True

    def leave(self) -> None:
        with self._lock:
            assert self._waiters > 0
            self._waiters -= 1
            if self._waiters == 0 and not self._started:
                self._accepting = False
                self.task.cancel()


class SnapshotCohorts:
    def __init__(
        self,
        *,
        max_waiters: int,
        max_draining: int,
        on_error: Callable[[BaseException], object],
    ) -> None:
        self._max_waiters = max_waiters
        self._max_owned = max_waiters + max_draining
        self._on_error = on_error
        self._waiters = 0
        self._queued: dict[Hashable, _Cohort] = {}
        self._owned: set[_Cohort] = set()
        self._closed = False

    async def read(
        self,
        key: Hashable,
        operation: Callable[[Callable[[], None]], Awaitable[Any]],
    ) -> Any:
        if self._closed or self._waiters >= self._max_waiters:
            raise SnapshotCapacityError("responsive snapshot waiter capacity is unavailable")
        cohort = self._queued.get(key)
        if cohort is None or not cohort.join():
            if len(self._owned) >= self._max_owned:
                raise SnapshotCapacityError("responsive snapshot workers are still draining")
            cohort = _Cohort(operation)
            assert cohort.join()
            self._queued[key] = cohort
            self._owned.add(cohort)
            cohort.task.add_done_callback(lambda task: self._finished(key, cohort, task))
        self._waiters += 1
        try:
            snapshot = await asyncio.shield(cohort.task)
            return snapshot.copy()
        finally:
            self._waiters -= 1
            cohort.leave()

    def _finished(self, key: Hashable, cohort: _Cohort, task: asyncio.Task[_Snapshot]) -> None:
        if self._queued.get(key) is cohort:
            del self._queued[key]
        self._owned.remove(cohort)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                self._on_error(error)

    async def close(self) -> None:
        self._closed = True
        tasks = tuple(cohort.task for cohort in self._owned)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
