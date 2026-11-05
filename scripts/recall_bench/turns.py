"""Writer and checkpoint turns, from the daemon's own log and diagnostics samples.

The daemon's log records every writer turn by label, with its lock wait and
its run time; checkpoints are writer turns labelled `WAL checkpoint
maintenance`. That is the per-turn record.

The daemon also appends a sample to `diagnostics.jsonl` in its data
directory every ten seconds, rotating at 5 MiB. Each sample carries
cumulative per-stage aggregates (count, maximum, histogram) since the daemon
started, the recent span ring, and the four longest spans of each stage
finished in that sample's interval. So the last sample gives the exact
maximum and count over the daemon's life, and the per-interval longest spans
give the exact maximum over any window of whole intervals. Spans the ring
dropped are counted, never assumed absent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from recall.services.diagnostics import Sample as DaemonSample

from .evidence import JobIncomplete
from .fs import Directory

DIAGNOSTICS_NAME = "diagnostics.jsonl"
DIAGNOSTICS_BACKUPS = 3
DIAGNOSTICS_BYTES_MAX = 64 * 1024 * 1024
TURN_STAGES = ("writer_run", "checkpoint")
ARGUMENT = "--run-dir"


@dataclass(frozen=True)
class Span:
    id: int
    stage: str
    start: float
    end: float
    outcome: str
    process_cpu_seconds: float | None = None
    label: str | None = None

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Sample:
    """One diagnostics line of the owned daemon."""

    timestamp: float
    monotonic: float
    stages: Mapping[str, Mapping[str, object]]
    counters: Mapping[str, int]
    spans: tuple[Span, ...]
    slowest: tuple[Span, ...]
    dropped_spans: int
    active: tuple[tuple[str, float], ...] = ()
    dropped_active: int = 0
    sampling_cpu_seconds: float = 0.0
    output_failures: int = 0
    cpu_spans: tuple[Span, ...] = ()
    dropped_cpu_spans: int = 0

    @classmethod
    def parse(cls, line: Mapping[str, object]) -> Sample:
        sample = DaemonSample.model_validate(line)
        telemetry = sample.telemetry

        def span(item) -> Span:
            return Span(
                item.id,
                str(item.stage),
                item.start,
                item.end,
                item.outcome,
                item.process_cpu_seconds,
                item.label,
            )

        return cls(
            timestamp=sample.timestamp,
            monotonic=sample.monotonic,
            stages={str(stage): asdict(aggregate) for stage, aggregate in telemetry.stages.items()},
            counters={str(counter): value for counter, value in telemetry.counters.items()},
            spans=tuple(span(item) for item in telemetry.spans),
            slowest=tuple(
                span(item) for items in telemetry.slowest_spans.values() for item in items
            ),
            dropped_spans=telemetry.dropped_spans,
            active=tuple((str(item.stage), item.start) for item in telemetry.active),
            dropped_active=telemetry.dropped_active,
            sampling_cpu_seconds=sample.sampling_cpu_seconds,
            output_failures=sample.output_failures,
            cpu_spans=tuple(span(item) for item in telemetry.cpu_spans),
            dropped_cpu_spans=telemetry.dropped_cpu_spans,
        )


def read_samples(root: Directory, pid: int) -> list[Sample]:
    """Every sample the daemon `pid` wrote into the held run directory, oldest first."""
    names = [f"{DIAGNOSTICS_NAME}.{index}" for index in range(DIAGNOSTICS_BACKUPS, 0, -1)]
    samples: list[Sample] = []
    total = 0
    for name in [*names, DIAGNOSTICS_NAME]:
        file = root.open_present_file(ARGUMENT, name)
        if file is None:
            continue
        with file:
            size = file.status.st_size
            total += size
            if total > DIAGNOSTICS_BYTES_MAX:
                raise OSError(f"diagnostics exceed {DIAGNOSTICS_BYTES_MAX} bytes")
            data = file.head(size)
        # A concurrent write may not yet have its final newline. Only complete
        # append records cross the parse boundary; a stuck writer becomes stale.
        for raw in data.split(b"\n")[:-1]:
            try:
                line = json.loads(raw)
                if line.get("pid") == pid:
                    samples.append(Sample.parse(line))
            except (ValueError, AttributeError) as error:
                raise JobIncomplete(
                    "diagnostics", f"invalid diagnostic record in {name}: {error}"
                ) from error
    samples.sort(key=lambda sample: sample.monotonic)
    return samples


def _aggregate(sample: Sample, stage: str) -> Mapping[str, object]:
    value = sample.stages.get(stage)
    return value if isinstance(value, dict) else {}


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def summarize(samples: list[Sample], window_start: float | None = None) -> dict[str, object]:
    """Per stage: every turn the daemon counted, its exact maximum, and the recorded spans.

    With `window_start` (monotonic), the window covers every sample interval
    that ends after it; counts are the difference from the last sample before it.
    """
    assert samples, "no diagnostics sample from the owned daemon"
    last = samples[-1]
    before = [
        sample
        for sample in samples
        if window_start is not None and sample.monotonic <= window_start
    ]
    in_window = [sample for sample in samples if sample not in before]
    baseline = before[-1] if before else None
    stages: dict[str, object] = {}
    for stage in TURN_STAGES:
        final = _aggregate(last, stage)
        base = _aggregate(baseline, stage) if baseline is not None else {}
        recorded = {
            span.id: span
            for sample in in_window
            for span in (*sample.spans, *sample.slowest)
            if span.stage == stage
        }
        longest = [span for sample in in_window for span in sample.slowest if span.stage == stage]
        stages[stage] = {
            "count": int(_number(final.get("count")) - _number(base.get("count"))),
            "errors": int(_number(final.get("errors")) - _number(base.get("errors"))),
            "seconds_max_lifetime": _number(final.get("wall_seconds_max")),
            "seconds_max": max((span.seconds for span in longest), default=0.0)
            if baseline is not None
            else _number(final.get("wall_seconds_max")),
            "buckets": final.get("buckets"),
            "spans_recorded": len(recorded),
            "spans": [
                {"start": span.start, "seconds": span.seconds, "outcome": span.outcome}
                for span in sorted(recorded.values(), key=lambda span: span.start)
            ],
        }
    return {
        "samples": len(samples),
        "window_start": window_start,
        "last_sample_at": last.timestamp,
        "dropped_spans": last.dropped_spans - (baseline.dropped_spans if baseline else 0),
        "generation_errors": int(last.counters.get("generation_errors", 0)),
        "stages": stages,
    }


# `rpc_server._writer_call` logs every writer turn at INFO, checkpoints included.
WRITER_LOG = re.compile(rb"writer (?P<label>[^:\n]+): wait=(?P<wait>[0-9.]+)s op=(?P<op>[0-9.]+)s")
CHECKPOINT_LABEL = "WAL checkpoint maintenance"


@dataclass(frozen=True)
class LoggedTurn:
    label: str
    wait_seconds: float
    op_seconds: float


def logged_turns(log: bytes) -> list[LoggedTurn]:
    """Every writer turn the daemon's log records, in order."""
    return [
        LoggedTurn(
            match["label"].decode(errors="replace"), float(match["wait"]), float(match["op"])
        )
        for match in WRITER_LOG.finditer(log)
    ]


def summarize_logged(turns: list[LoggedTurn]) -> dict[str, object]:
    labels: dict[str, dict[str, float]] = {}
    for turn in turns:
        entry = labels.setdefault(
            turn.label, {"count": 0, "op_seconds_max": 0.0, "wait_seconds_max": 0.0}
        )
        entry["count"] += 1
        entry["op_seconds_max"] = max(entry["op_seconds_max"], turn.op_seconds)
        entry["wait_seconds_max"] = max(entry["wait_seconds_max"], turn.wait_seconds)
    checkpoints = [turn.op_seconds for turn in turns if turn.label == CHECKPOINT_LABEL]
    return {
        "turns": len(turns),
        "op_seconds_max": max((turn.op_seconds for turn in turns), default=0.0),
        "checkpoints": len(checkpoints),
        "checkpoint_seconds_max": max(checkpoints, default=0.0),
        "by_label": dict(sorted(labels.items())),
    }
