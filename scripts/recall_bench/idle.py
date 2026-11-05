"""Conditioned idle windows and conservative per-occurrence CPU attribution.

Process CPU spans include native workers and unrelated concurrent work. They
are upper bounds for an occurrence, never additive stage accounting. Quiet is
observed through the daemon's raw-commit telemetry and unfiltered active roster.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import psutil
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .evidence import JobIncomplete, Result
from .fs import NewTree
from .samplers import Sample, sample, summarize
from .turns import Sample as DiagnosticSample
from .turns import Span

QUIET_SECONDS = 120.0
WINDOW_SECONDS = 600.0
CPU_PERCENT_MAX = 5.0
OPERATION_CPU_SECONDS_MAX = 1.0
OBSERVATION_GAP_SECONDS_MAX = 2.5
DIAGNOSTICS_AGE_SECONDS_MAX = 30.0
# Synchronous spans own work; async spans deliberately have no process CPU.
ASYNC_STAGES = frozenset({"executor_wait", "writer_wait", "read", "search", "embed_cycle"})


class Record(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)


class Reconciliation(Record):
    rpc_ready: bool
    pending: int = Field(ge=0)
    noticed_pending: int = Field(ge=0)
    raw_indexing_ready: bool
    enrichment_ready: bool
    error: str | None = None
    enrichment_error: str | None = None
    keyword_repair_error: str | None = None


class Status(Record):
    daemon_pid: int = Field(gt=0)
    reconciliation: Reconciliation
    watch_total_failed: int = Field(ge=0)
    embed_pending_eligible: int | None = Field(default=None, ge=0)
    embed_pending_held: int | None = Field(default=None, ge=0)


class Coverage(Record):
    complete: bool


class Live(Record):
    sessions: list[dict[str, object]]
    next_cursor: str | None
    coverage: Coverage


@dataclass(frozen=True)
class Observation:
    at: float
    diagnostics_at: float
    raw_commits: int
    raw_active: bool
    active_sessions: int
    ready: bool
    pending_eligible: int | None = None
    pending_held: int | None = None

    @property
    def quiet(self) -> bool:
        return self.ready and not self.raw_active and self.active_sessions == 0


class Observer:
    """Read-only RPC to an explicit socket; cannot start or signal a daemon."""

    def __init__(self, path: Path, pid: int) -> None:
        from recall.core.config import (
            AppConfig,
            CliConfig,
            DaemonConfig,
            EmbeddingConfig,
            FtsConfig,
        )
        from recall.core.rpc_client import RpcClient

        config = AppConfig(
            data_dir=path,
            db_path=path / "recall.duckdb",
            lock_path=path / "recall.lock",
            config_path=path / "config.toml",
            fts=FtsConfig(),
            embedding=EmbeddingConfig(),
            daemon=DaemonConfig(),
            cli=CliConfig(),
        )
        self.client = RpcClient(config)
        self.pid = pid
        self.seconds_max = 0.0

    def close(self) -> None:
        self.client.close()

    def call(self, method: str, params: dict[str, object] | None = None) -> object:
        from recall.core.rpc_types import (
            RpcAdmissionBusyError,
            RpcCallError,
            RpcConnectionError,
            RpcError,
        )
        from recall.read_budget import ReadBudget

        started = time.monotonic()
        try:
            budget = ReadBudget.start()
            self.client.connect(auto_fork=False, read_budget=budget)
            return self.client.call(method, params, read_budget=budget)
        except (
            RpcError,
            RpcAdmissionBusyError,
            RpcCallError,
            RpcConnectionError,
            OSError,
        ) as error:
            raise JobIncomplete("idle_observation", str(error)) from error
        finally:
            self.seconds_max = max(self.seconds_max, time.monotonic() - started)

    def observe(self, diagnostics: DiagnosticSample) -> Observation:
        try:
            status = Status.model_validate(self.call("recall.daemon_status"))
            # The default live roster is active-only. One row proves presence;
            # an empty page proves absence only with complete coverage and no cursor.
            live = Live.model_validate(self.call("recall.live", {"limit": 1}))
        except ValidationError as error:
            raise JobIncomplete("idle_observation", f"invalid daemon response: {error}") from error
        if status.daemon_pid != self.pid:
            raise JobIncomplete("idle_identity", "the socket changed daemon PID")
        reconciliation = status.reconciliation
        if (
            reconciliation.error
            or reconciliation.enrichment_error
            or reconciliation.keyword_repair_error
            or status.watch_total_failed
        ):
            raise JobIncomplete("idle_observation", "daemon reports reconciliation errors")
        count = diagnostics.stages.get("raw_commit", {}).get("count", 0)
        if type(count) is not int or count < 0:
            raise JobIncomplete("idle_observation", "invalid raw-commit counter")
        now = time.monotonic()
        if not 0 <= now - diagnostics.monotonic <= DIAGNOSTICS_AGE_SECONDS_MAX:
            raise JobIncomplete("idle_observation", "daemon diagnostics are stale")
        ready = (
            reconciliation.rpc_ready
            and reconciliation.raw_indexing_ready
            and reconciliation.pending == 0
            and reconciliation.noticed_pending == 0
            and reconciliation.enrichment_ready
            and live.coverage.complete
            and (bool(live.sessions) or live.next_cursor is None)
        )
        return Observation(
            now,
            diagnostics.monotonic,
            count,
            any(stage == "raw_commit" for stage, _ in diagnostics.active),
            len(live.sessions),
            ready,
            status.embed_pending_eligible,
            status.embed_pending_held,
        )


@dataclass
class QuietGate:
    """The window capability is produced only after a continuously observed quiet interval."""

    quiet_since: float | None = None
    previous: Observation | None = None

    def observe(self, observation: Observation) -> Window | None:
        previous = self.previous
        self.previous = observation
        changed = previous is None or observation.raw_commits != previous.raw_commits
        gap = previous is not None and observation.at - previous.at > OBSERVATION_GAP_SECONDS_MAX
        if changed or gap or not observation.quiet:
            self.quiet_since = observation.at if observation.quiet else None
        elif self.quiet_since is None:
            self.quiet_since = observation.at
        if self.quiet_since is None:
            return None
        # Require diagnostics written after the full quiet interval, not a stale
        # zero counter whose next flush could reveal a commit just before entry.
        if observation.diagnostics_at - self.quiet_since < QUIET_SECONDS:
            return None
        return Window(observation, self.quiet_since)


@dataclass
class Window:
    entry: Observation
    quiet_since: float
    invalid: list[str] = field(default_factory=list)
    last_at: float | None = None

    def observe(self, observation: Observation) -> None:
        if observation.raw_commits != self.entry.raw_commits or observation.raw_active:
            self.invalidate("raw commit during window")
        if observation.active_sessions:
            self.invalidate("active agent session during window")
        if not observation.ready:
            self.invalidate("daemon no longer drained/ready")
        previous = self.last_at if self.last_at is not None else self.entry.at
        if observation.at - previous > OBSERVATION_GAP_SECONDS_MAX:
            self.invalidate("per-second observation gap")
        self.last_at = observation.at

    def invalidate(self, reason: str) -> None:
        if reason not in self.invalid:
            self.invalid.append(reason)

    def verdict(
        self, samples: list[Sample], operations: list[Span], missing: list[str]
    ) -> dict[str, object]:
        totals = summarize(samples)
        seconds = float(str(totals["seconds"]))
        cpu = totals["cpu_percent"]
        assert isinstance(cpu, float) and math.isfinite(cpu)
        maximum = max((op.process_cpu_seconds or 0.0 for op in operations), default=0.0)
        invalid = list(self.invalid)
        if seconds < WINDOW_SECONDS:
            invalid.append(f"window shorter than {WINDOW_SECONDS:g} seconds")
        invalid.extend(missing)
        breaches = []
        if cpu > CPU_PERCENT_MAX:
            breaches.append("mean process CPU exceeds 5% of one core")
        if maximum > OPERATION_CPU_SECONDS_MAX:
            breaches.append("periodic occurrence exceeds 1 CPU-second")
        verdict: Literal["invalid", "failed", "passed"] = (
            "invalid" if invalid else "failed" if breaches else "passed"
        )
        return {
            **totals,
            "verdict": verdict,
            "invalid": invalid,
            "breaches": breaches,
            "quiet_seconds": self.entry.at - self.quiet_since,
            "cpu_percent_limit": CPU_PERCENT_MAX,
            "operation_cpu_seconds_max": maximum,
            "operation_cpu_seconds_limit": OPERATION_CPU_SECONDS_MAX,
            "operations": len(operations),
            "attribution": "inclusive process CPU upper bounds; overlapping spans are not additive",
        }


def attributed(
    diagnostics: list[DiagnosticSample],
    start: float,
    end: float,
) -> tuple[list[Span], list[str]]:
    """Every synchronous occurrence overlapping the window; fail closed on lost spans."""
    before = [item for item in diagnostics if item.monotonic <= start]
    after = [item for item in diagnostics if item.monotonic >= end]
    if not before or not after:
        return [], ["diagnostics do not bracket window"]
    baseline, final = before[-1], after[0]
    interval = [
        item for item in diagnostics if baseline.monotonic < item.monotonic <= final.monotonic
    ]
    spans = {span.id: span for item in interval for span in item.cpu_spans}
    missing = []
    if (
        final.dropped_cpu_spans != baseline.dropped_cpu_spans
        or final.dropped_active != baseline.dropped_active
    ):
        missing.append("diagnostic span capacity exceeded")
    if final.output_failures != baseline.output_failures:
        missing.append("diagnostics output failed")
    if any(at <= end for _, at in final.active):
        missing.append("operation still running at final diagnostics")
    # Counts catch discarded diagnostic intervals even when the ring itself did not overflow.
    expected = sum(
        int(str(stage.get("process_cpu_samples", 0))) for stage in final.stages.values()
    ) - sum(int(str(stage.get("process_cpu_samples", 0))) for stage in baseline.stages.values())
    if len(spans) != expected:
        missing.append("per-occurrence diagnostic records are incomplete")
    operations = [
        span
        for span in spans.values()
        if span.stage not in ASYNC_STAGES and span.end >= start and span.start <= end
    ]
    if not operations or any(span.process_cpu_seconds is None for span in operations):
        missing.append("per-occurrence process CPU instrumentation unavailable")
    # Diagnostics accounts its own calling-thread CPU cumulatively, one write behind.
    # Retain every write as a separate operation; it has no native worker pool.
    previous = baseline
    for item in interval:
        cpu = item.sampling_cpu_seconds - previous.sampling_cpu_seconds
        if item.monotonic >= start and previous.monotonic <= end:
            operations.append(
                Span(
                    -len(operations) - 1,
                    "diagnostics",
                    previous.monotonic,
                    item.monotonic,
                    "ok",
                    cpu,
                    "diagnostics write",
                )
            )
        previous = item
    return sorted(operations, key=lambda span: (span.start, span.id)), missing


def operation_json(span: Span) -> dict[str, object]:
    return asdict(span)


class IdleRun:
    """One observation owner for transition, gate and window, with retained evidence."""

    def __init__(
        self,
        *,
        observer: Observer,
        process: psutil.Process,
        diagnostics: Callable[[], list[DiagnosticSample]],
        wait: Callable[[float], None],
        check: Callable[[], None],
        tree: NewTree,
        phase: Callable[[str], None],
        result: Result,
        window_seconds: float,
        settle_timeout: float,
    ) -> None:
        self.observer = observer
        self.process = process
        self.diagnostics = diagnostics
        self.wait = wait
        self.check = check
        self.tree = tree
        self.phase = phase
        self.result = result
        self.window_seconds = window_seconds
        self.settle_timeout = settle_timeout
        self.observations_fd: int | None = None

    def observe(self, phase: str) -> tuple[Observation, Sample]:
        self.check()
        diagnostics = self.diagnostics()
        if not diagnostics:
            raise JobIncomplete("idle_diagnostics", "daemon wrote no diagnostics")
        observation = self.observer.observe(diagnostics[-1])
        try:
            item = sample(self.process)
        except psutil.Error as error:
            raise JobIncomplete("idle_identity", str(error)) from error
        assert self.observations_fd is not None
        append_json(self.observations_fd, {"phase": phase, **asdict(observation), **item.to_json()})
        return observation, item

    def transition(self) -> None:
        self.phase("transition")
        started = time.monotonic()
        samples = []
        pending_seen = False
        while True:
            observation, item = self.observe("transition")
            samples.append(item)
            pending_seen |= (observation.pending_eligible or 0) + (
                observation.pending_held or 0
            ) > 0
            if pending_seen and observation.ready and len(samples) >= 2:
                self.result.step(
                    "transition",
                    **summarize(samples),
                    pending_seen=True,
                    pending_eligible=observation.pending_eligible,
                    pending_held=observation.pending_held,
                )
                return
            if time.monotonic() - started >= self.settle_timeout:
                raise JobIncomplete(
                    "idle_transition",
                    "pending enrichment was not observed draining",
                    pending_seen=pending_seen,
                )
            self.wait(1.0)

    def run(self, transition: bool = False) -> dict[str, object]:
        import os
        from pathlib import PurePosixPath

        self.observations_fd = self.tree.open_new(
            self.tree.root, PurePosixPath("idle-observations.jsonl")
        )
        try:
            if transition:
                self.transition()
            self.phase("quiet-gate")
            gate = QuietGate()
            started = time.monotonic()
            window = None
            while window is None:
                observation, _ = self.observe("quiet-gate")
                window = gate.observe(observation)
                if time.monotonic() - started >= self.settle_timeout:
                    raise JobIncomplete("idle_gate", "no continuous quiet interval before timeout")
                if window is None:
                    self.wait(1.0)
            self.result.step("quiet-gate", **asdict(window.entry), quiet_since=window.quiet_since)
            return self.measure(window)
        finally:
            os.close(self.observations_fd)
            self.observations_fd = None
            self.observer.close()

    def measure(self, window: Window) -> dict[str, object]:
        import os
        from pathlib import PurePosixPath

        self.phase("window")
        samples = []
        descriptor = self.tree.open_new(self.tree.root, PurePosixPath("samples.jsonl"))
        try:
            while True:
                observation, item = self.observe("window")
                window.observe(observation)
                samples.append(item)
                append_json(descriptor, item.to_json())
                if len(samples) >= 2 and item.at - samples[0].at >= self.window_seconds:
                    break
                self.wait(max(0.001, samples[0].at + len(samples) - time.monotonic()))
        finally:
            os.close(descriptor)
        # Flush the trailing interval so a late raw commit, lost record or operation
        # crossing the endpoint cannot turn a partial trace into a pass.
        self.phase("attribution")
        end = samples[-1].at
        until = time.monotonic() + DIAGNOSTICS_AGE_SECONDS_MAX
        while True:
            self.check()
            diagnostics = self.diagnostics()
            if diagnostics and diagnostics[-1].monotonic > end:
                break
            if time.monotonic() >= until:
                raise JobIncomplete("idle_diagnostics", "no diagnostics after window")
            self.wait(1.0)
        final = self.observer.observe(diagnostics[-1])
        if final.raw_commits != window.entry.raw_commits:
            window.invalidate("raw commit during window or trailing diagnostics flush")
        operations, missing = attributed(diagnostics, samples[0].at, end)
        self.tree.write_new(
            self.tree.root,
            PurePosixPath("periodic-operations.jsonl"),
            b"".join((json.dumps(operation_json(op)) + "\n").encode() for op in operations),
        )
        verdict = window.verdict(samples, operations, missing)
        self.tree.write_new(
            self.tree.root,
            PurePosixPath("idle-verdict.json"),
            (json.dumps(verdict, indent=2) + "\n").encode(),
        )
        self.result.step("window", **verdict)
        return verdict


def append_json(descriptor: int, value: object) -> None:
    import os

    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    while data:
        written = os.write(descriptor, data)
        if written == 0:
            raise OSError("zero-byte evidence write")
        data = data[written:]
