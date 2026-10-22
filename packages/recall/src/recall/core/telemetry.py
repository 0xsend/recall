"""Bounded operation accounting; no database reads or transcript payloads."""

from __future__ import annotations

import asyncio
import heapq
import threading
import time
from collections import deque
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar, Token, copy_context
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import wraps
from typing import Any, Literal

SPANS_MAX = 256
# The recent-span ring evicts by age, so a burst after a slow operation would push it out
# before the sample. Each interval also keeps this many of the longest spans per stage.
SLOWEST_SPANS_PER_STAGE = 4
DURATION_BUCKETS_SECONDS = (0.001, 0.01, 0.1, 1.0, 5.0, 30.0)


class Stage(StrEnum):
    EXECUTOR_WAIT = "executor_wait"
    EXECUTOR_RUN = "executor_run"
    WRITER_WAIT = "writer_wait"
    WRITER_RUN = "writer_run"
    READ = "read"
    SEARCH = "search"
    INVENTORY = "inventory"
    RAW_SELECT = "raw_select"
    RAW_PREPARE = "raw_prepare"
    PARSE = "parse"
    RAW_COMMIT = "raw_commit"
    EMBED_CYCLE = "embed_cycle"
    EMBED_SELECT = "embed_select"
    EMBED_SNAPSHOT = "embed_snapshot"
    EMBED_GENERATE = "embed_generate"
    CONTEXT = "context"
    MODEL_LOAD = "model_load"
    TOKENIZE = "tokenize"
    INFERENCE = "inference"
    VALIDATE = "validate"
    EMBED_COMMIT = "embed_commit"
    EMBED_PUBLISH = "embed_publish"
    FTS = "fts"
    CHECKPOINT = "checkpoint"


class Counter(StrEnum):
    BATCHES = "embedding_batches"
    TEXTS = "embedding_texts"
    TOKENS = "embedding_tokens"
    PADDED_TOKENS = "embedding_padded_tokens"
    GENERATED = "generated_targets"
    COMMITTED = "committed_targets"
    REJECTED = "rejected_sessions"
    DRAINED = "drained_targets"
    GENERATION_ERRORS = "generation_errors"


class Gauge(StrEnum):
    PENDING = "eligible_pending_targets"
    PENDING_AT = "pending_sample_at"
    SELECTED_IDLE = "selected_session_idle_seconds_max"


@dataclass
class Aggregate:
    count: int = 0
    errors: int = 0
    cancelled: int = 0
    wall_seconds_sum: float = 0.0
    wall_seconds_max: float = 0.0
    # Inclusive, calling-thread CPU only. Native worker pools are not included.
    thread_cpu_seconds_sum: float | None = None
    thread_cpu_samples: int = 0
    buckets: list[int] = field(default_factory=lambda: [0] * (len(DURATION_BUCKETS_SECONDS) + 1))


@dataclass(frozen=True)
class SpanRecord:
    id: int
    parent_id: int | None
    stage: Stage
    start: float
    end: float
    thread_id: int
    thread_cpu_seconds: float | None
    outcome: Literal["ok", "error", "cancelled"]


@dataclass(frozen=True)
class ActiveSpan:
    id: int
    stage: Stage
    start: float
    parent_id: int | None


@dataclass
class Backend:
    name: str
    device: str
    model: str
    runtime: str
    active_bytes: int | None = None
    cache_bytes: int | None = None
    peak_bytes: int | None = None
    memory_sample_at: float | None = None
    cache_limit_bytes: int | None = None


@dataclass(frozen=True)
class Snapshot:
    stages: dict[Stage, Aggregate]
    counters: dict[Counter, int]
    gauges: dict[Gauge, float]
    spans: list[SpanRecord]
    active: list[ActiveSpan]
    dropped_spans: int
    dropped_active: int
    backend: Backend | None
    # Longest first per stage, for this interval; a span may also appear in `spans`.
    slowest_spans: dict[Stage, list[SpanRecord]] = field(default_factory=dict)


class Recorder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sequence = 0
        self.aggregates = {stage: Aggregate() for stage in Stage}
        self.counters = dict.fromkeys(Counter, 0)
        self.gauges: dict[Gauge, float] = {}
        self.spans: deque[SpanRecord] = deque(maxlen=SPANS_MAX)
        # Min-heaps of (duration, id, record); the root is the first to give way.
        self.slowest: dict[Stage, list[tuple[float, int, SpanRecord]]] = {}
        self.active: dict[int, tuple[Stage, float, int | None]] = {}
        self.dropped_spans = 0
        self.dropped_active = 0
        self.backend: Backend | None = None

    def begin(self, stage: Stage, start: float, parent: int | None) -> int:
        with self.lock:
            self.sequence += 1
            identifier = self.sequence
            if len(self.active) < SPANS_MAX:
                self.active[identifier] = (stage, start, parent)
            else:
                self.dropped_active += 1
            return identifier

    def finish(self, record: SpanRecord) -> None:
        duration = record.end - record.start
        with self.lock:
            self.active.pop(record.id, None)
            aggregate = self.aggregates[record.stage]
            aggregate.count += 1
            aggregate.errors += record.outcome == "error"
            aggregate.cancelled += record.outcome == "cancelled"
            aggregate.wall_seconds_sum += duration
            aggregate.wall_seconds_max = max(aggregate.wall_seconds_max, duration)
            if record.thread_cpu_seconds is not None:
                aggregate.thread_cpu_seconds_sum = (
                    aggregate.thread_cpu_seconds_sum or 0.0
                ) + record.thread_cpu_seconds
                aggregate.thread_cpu_samples += 1
            bucket = next(
                (i for i, upper in enumerate(DURATION_BUCKETS_SECONDS) if duration <= upper),
                len(DURATION_BUCKETS_SECONDS),
            )
            aggregate.buckets[bucket] += 1
            self.dropped_spans += len(self.spans) == SPANS_MAX
            self.spans.append(record)
            slowest = self.slowest.get(record.stage)
            if slowest is None:
                slowest = []
                self.slowest[record.stage] = slowest
            if len(slowest) < SLOWEST_SPANS_PER_STAGE:
                heapq.heappush(slowest, (duration, record.id, record))
            elif duration > slowest[0][0]:
                heapq.heapreplace(slowest, (duration, record.id, record))

    def increment(self, counter: Counter, amount: int) -> None:
        with self.lock:
            self.counters[counter] += amount

    def snapshot(self) -> Snapshot:
        with self.lock:
            result = Snapshot(
                stages={
                    stage: replace(value, buckets=list(value.buckets))
                    for stage, value in self.aggregates.items()
                    if value.count
                },
                counters=dict(self.counters),
                gauges=dict(self.gauges),
                spans=list(self.spans),
                active=[ActiveSpan(key, *value) for key, value in self.active.items()],
                dropped_spans=self.dropped_spans,
                dropped_active=self.dropped_active,
                backend=replace(self.backend) if self.backend else None,
                slowest_spans={
                    stage: [record for *_, record in sorted(heap, reverse=True)]
                    for stage, heap in self.slowest.items()
                },
            )
            self.spans.clear()
            self.slowest.clear()
            return result


current: ContextVar[Recorder | None] = ContextVar("recall_recorder", default=None)
_parent: ContextVar[int | None] = ContextVar("recall_span", default=None)


class Span:
    """A lexical span; async spans deliberately carry no thread CPU measurement."""

    def __init__(self, stage: Stage, *, thread_cpu: bool = True) -> None:
        self.stage = stage
        self.thread_cpu = thread_cpu
        self.recorder = current.get()
        self.token: Token[int | None] | None = None

    def __enter__(self) -> Span:
        if self.recorder is not None:
            self.start = time.monotonic()
            self.cpu_start = time.thread_time() if self.thread_cpu else None
            self.thread_id = threading.get_native_id()
            self.parent = _parent.get()
            self.id = self.recorder.begin(self.stage, self.start, self.parent)
            self.token = _parent.set(self.id)
        return self

    def __exit__(self, _kind: object, error: BaseException | None, _trace: object) -> None:
        if self.recorder is None:
            return
        assert self.token is not None
        _parent.reset(self.token)
        cpu = time.thread_time() - self.cpu_start if self.cpu_start is not None else None
        self.recorder.finish(
            SpanRecord(
                self.id,
                self.parent,
                self.stage,
                self.start,
                time.monotonic(),
                self.thread_id,
                cpu,
                "cancelled"
                if isinstance(error, asyncio.CancelledError)
                else "error"
                if error
                else "ok",
            )
        )


def measured[**P, R](stage: Stage) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def call(*args: P.args, **kwargs: P.kwargs) -> R:
            with Span(stage):
                return fn(*args, **kwargs)

        return call

    return decorate


def measured_async[**P, R](
    stage: Stage,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    def decorate(fn: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(fn)
        async def call(*args: P.args, **kwargs: P.kwargs) -> R:
            with Span(stage, thread_cpu=False):
                return await fn(*args, **kwargs)

        return call

    return decorate


def count(counter: Counter, amount: int) -> None:
    recorder = current.get()
    if recorder is not None:
        recorder.increment(counter, amount)


def embedding_batch(attention_masks: list[list[int]]) -> None:
    if current.get() is not None:
        count(Counter.BATCHES, 1)
        count(Counter.TEXTS, len(attention_masks))
        count(Counter.TOKENS, sum(map(sum, attention_masks)))
        count(Counter.PADDED_TOKENS, sum(map(len, attention_masks)))


def backend_loaded(
    *, backend: str, device: str, model: str, runtime: str, cache_limit_bytes: int | None = None
) -> None:
    recorder = current.get()
    if recorder is not None:
        with recorder.lock:
            recorder.backend = Backend(
                backend, device, model[:256], runtime[:64], cache_limit_bytes=cache_limit_bytes
            )


def mlx_memory(*, active: int, cache: int, peak: int) -> None:
    recorder = current.get()
    if recorder is not None:
        with recorder.lock:
            if recorder.backend is not None:
                recorder.backend.active_bytes = active
                recorder.backend.cache_bytes = cache
                recorder.backend.peak_bytes = peak
                recorder.backend.memory_sample_at = time.time()


class DiagnosticExecutor(ThreadPoolExecutor):
    """Propagate span context and distinguish executor delay from worker execution."""

    def submit[**P, R](self, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        context = copy_context()
        recorder = current.get()
        queued = time.monotonic()
        parent = _parent.get()
        identifier = recorder.begin(Stage.EXECUTOR_WAIT, queued, parent) if recorder else 0

        def finish_wait(outcome: str = "ok") -> None:
            if recorder is not None:
                recorder.finish(
                    SpanRecord(
                        identifier,
                        parent,
                        Stage.EXECUTOR_WAIT,
                        queued,
                        time.monotonic(),
                        threading.get_native_id(),
                        None,
                        "cancelled" if outcome == "cancelled" else "ok",
                    )
                )

        def run() -> R:
            finish_wait()
            with Span(Stage.EXECUTOR_RUN):
                return fn(*args, **kwargs)

        def run_context() -> R:
            return context.run(run)

        try:
            future = super().submit(run_context)
        except RuntimeError:
            finish_wait("cancelled")
            raise
        future.add_done_callback(
            lambda completed: finish_wait("cancelled") if completed.cancelled() else None
        )
        return future


def gauge(name: Gauge, value: float) -> None:
    recorder = current.get()
    if recorder is not None:
        with recorder.lock:
            recorder.gauges[name] = value
