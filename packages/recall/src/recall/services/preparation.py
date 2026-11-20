"""Bounded parser queues own their workers through observation and drain."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol, Self

from recall.db.fatal import inspect_fatal_graph


class PreparationFactory(Protocol):
    def prepare[K, V](
        self, work: Iterable[tuple[K, Callable[[], V]]], *, workers: int
    ) -> PreparationQueue[K, V]: ...


class PreparationQueue[K, V]:
    """Only materialized parser values leave the queue, never worker futures."""

    def __init__(
        self,
        work: Iterable[tuple[K, Callable[[], V]]],
        *,
        workers: int,
        observe: Callable[[BaseException, str], bool],
        retired: Callable[[PreparationQueue[K, V]], None],
    ) -> None:
        assert workers > 0
        self._observe = observe
        self._retired = retired
        self._lock = threading.Lock()
        self._terminal: BaseException | None = None
        self._closed = False
        self._pending = deque(work)
        self._ready: dict[K, Future[V]] = {}
        # The admitted window bounds useful parallelism even when an explicit
        # worker count exceeds it. Sequential callers keep parsing inline.
        self._window = min(max(workers * 2, 4), 16)
        self._pool = (
            ThreadPoolExecutor(max_workers=min(workers, 16), thread_name_prefix="recall-index")
            if workers > 1
            else None
        )
        self._started = False

    def _start(self) -> None:
        assert not self._closed, "preparation lifetime closed"
        if self._started:
            return
        self._started = True
        if self._pool is not None:
            try:
                for _ in range(min(self._window, len(self._pending))):
                    self._submit_next()
            except BaseException as primary:
                self._failed(primary, "parse_source:submit")
                self.__exit__(type(primary), primary, primary.__traceback__)
                raise

    def _run(self, work: Callable[[], V]) -> V:
        try:
            return work()
        except BaseException as error:
            self._failed(error, "parse_source")
            raise

    def _failed(self, error: BaseException, operation: str) -> None:
        terminal = self._observe(error, operation)
        control = not isinstance(error, (Exception, asyncio.CancelledError))
        if terminal or control:
            with self._lock:
                if self._terminal is None:
                    self._terminal = error

    def _submit_next(self) -> None:
        assert self._pool is not None
        if self._pending:
            key, work = self._pending.popleft()
            assert key not in self._ready, "duplicate preparation key"
            self._ready[key] = self._pool.submit(self._run, work)

    def take(self, key: K) -> V:
        self._start()
        if self._pool is None:
            next_key, work = self._pending.popleft()
            assert key == next_key, "sequential preparation consumed out of order"
            return self._run(work)
        future = self._ready.pop(key)
        try:
            return future.result()
        finally:
            with self._lock:
                terminal = self._terminal
            if terminal is None:
                self._submit_next()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        try:
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
        except BaseException as error:
            self._observe(error, "parse_source:shutdown")
            with self._lock:
                terminal = self._terminal
            if terminal is not None and error is not terminal:
                raise BaseExceptionGroup(
                    "preparation and shutdown failed", [terminal, error]
                ) from None
            raise
        finally:
            self._ready.clear()
            self._retired(self)
        with self._lock:
            terminal = self._terminal
        if terminal is not None:
            raise terminal

    def __enter__(self) -> Self:
        self._start()
        return self

    def __exit__(self, _kind: object, primary: BaseException | None, _trace: object) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if primary is None:
                raise
            if cleanup is not primary:
                raise BaseExceptionGroup(
                    "preparation scope and drain failed", [primary, cleanup]
                ) from None


class StandalonePreparation:
    """Offline parser lifetime observes terminal failures without daemon markers."""

    def prepare[K, V](
        self, work: Iterable[tuple[K, Callable[[], V]]], *, workers: int
    ) -> PreparationQueue[K, V]:
        return PreparationQueue(
            work, workers=workers, observe=self._observe, retired=lambda _: None
        )

    @staticmethod
    def _observe(error: BaseException, operation: str) -> bool:
        graph = inspect_fatal_graph(error)
        return not graph.complete or graph.failure is not None
