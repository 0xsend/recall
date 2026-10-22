"""Diagnostics preserve outcomes, report concurrency honestly, and bound storage."""

import asyncio
import sys
import threading
from pathlib import Path

import pytest
from recall.core.telemetry import (
    SLOWEST_SPANS_PER_STAGE,
    Counter,
    DiagnosticExecutor,
    Recorder,
    Snapshot,
    Span,
    SpanRecord,
    Stage,
    count,
    current,
    embedding_batch,
    measured_async,
)
from recall.services.diagnostics import Diagnostics, Sample


def test_nested_failure_preserves_error_and_reports_both_spans() -> None:
    recorder = Recorder()
    token = current.set(recorder)
    try:
        with (
            pytest.raises(ValueError, match="private input"),
            Span(Stage.RAW_COMMIT),
            Span(Stage.FTS),
        ):
            raise ValueError("private input")
    finally:
        current.reset(token)
    snapshot = recorder.snapshot()
    assert len(snapshot.spans) == 2
    child, parent = snapshot.spans
    assert child.parent_id == parent.id
    assert child.outcome == parent.outcome == "error"
    assert snapshot.active == []
    assert snapshot.stages[Stage.RAW_COMMIT].errors == 1
    assert snapshot.stages[Stage.FTS].errors == 1
    assert "private input" not in str(snapshot)


def test_executor_preserves_parent_and_accounts_cancelled_queue() -> None:
    recorder = Recorder()
    token = current.set(recorder)
    entered = threading.Event()
    release = threading.Event()

    def work() -> None:
        with Span(Stage.INFERENCE):
            entered.set()
            assert release.wait(5)
            count(Counter.TEXTS, 3)

    try:
        with DiagnosticExecutor(max_workers=1) as executor:
            try:
                with Span(Stage.EMBED_CYCLE, thread_cpu=False):
                    future = executor.submit(work)
                    assert entered.wait(5)
                    queued = executor.submit(work)
                    assert queued.cancel()
            finally:
                release.set()
            future.result(timeout=5)
    finally:
        current.reset(token)
    snapshot = recorder.snapshot()
    assert snapshot.counters[Counter.TEXTS] == 3
    assert snapshot.stages[Stage.EXECUTOR_WAIT].count == 2
    assert snapshot.stages[Stage.EXECUTOR_WAIT].cancelled == 1
    assert snapshot.active == []
    cycle = next(s for s in snapshot.spans if s.stage == Stage.EMBED_CYCLE)
    execution = next(s for s in snapshot.spans if s.stage == Stage.EXECUTOR_RUN)
    inference = next(s for s in snapshot.spans if s.stage == Stage.INFERENCE)
    assert execution.parent_id == cycle.id
    assert inference.parent_id == execution.id
    assert inference.thread_id != cycle.thread_id
    assert cycle.thread_cpu_seconds is None


def test_cancelled_async_operation_has_no_thread_cpu() -> None:
    recorder = Recorder()
    token = current.set(recorder)

    @measured_async(Stage.SEARCH)
    async def search() -> None:
        raise asyncio.CancelledError

    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(search())
    finally:
        current.reset(token)
    snapshot = recorder.snapshot()
    assert snapshot.stages[Stage.SEARCH].cancelled == 1
    assert snapshot.stages[Stage.SEARCH].thread_cpu_seconds_sum is None
    assert snapshot.stages[Stage.SEARCH].thread_cpu_samples == 0
    assert snapshot.spans[0].thread_cpu_seconds is None


def test_overflow_drops_details_without_losing_totals() -> None:
    recorder = Recorder()
    token = current.set(recorder)
    try:
        for _ in range(300):
            with Span(Stage.PARSE):
                pass
        embedding_batch([[1, 1, 0], [1, 1, 1]])
    finally:
        current.reset(token)
    snapshot = recorder.snapshot()
    assert len(snapshot.spans) == 256
    assert snapshot.dropped_spans == 44
    assert snapshot.stages[Stage.PARSE].count == 300
    assert sum(snapshot.stages[Stage.PARSE].buckets) == 300
    assert snapshot.counters[Counter.TOKENS] == 5
    assert snapshot.counters[Counter.PADDED_TOKENS] == 6
    assert recorder.snapshot().spans == []
    assert recorder.snapshot().stages[Stage.PARSE].count == 300


def test_samples_are_readable_and_rotation_retains_bounded_history(tmp_path: Path) -> None:
    from recall.services.diagnostics import LOG_BYTES_MAX

    path = tmp_path / "diagnostics.jsonl"
    collector = Diagnostics(path, requested_backend="auto", configured_model="test-model")
    # Exercise the actual 5 MiB threshold with full recent and slow-span samples.
    rotations = 0
    for interval in range(500):
        for stage in Stage:
            for index in range(16):
                _completed(collector.recorder, stage, interval * 20.0 + index, 0.01)
        rotating = path.exists() and path.stat().st_size >= LOG_BYTES_MAX
        collector.write()
        rotations += rotating
        if rotations == 4:
            break
    assert rotations == 4
    files = sorted(tmp_path.iterdir())
    assert len(files) == 4
    for file in files:
        lines = file.read_text().splitlines()
        assert lines
        for line in lines:
            sample = Sample.model_validate_json(line)
            assert set(sample.telemetry.slowest_spans) == set(Stage)
            assert len(sample.telemetry.spans) == 256
            assert sample.telemetry.dropped_spans > 0
        assert file.stat().st_size <= LOG_BYTES_MAX + len(lines[-1].encode()) + 1
    sample = Sample.model_validate_json(path.read_text())
    assert sample.process_cpu_seconds > 0
    assert sample.rss_bytes > 0
    assert sample.threads > 0
    assert sample.telemetry.backend is None
    assert sample.requested_backend == "auto"
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        assert sample.scheduling is not None
    assert path.stat().st_mode & 0o777 == 0o600


def test_output_failure_is_visible_and_recovers(tmp_path: Path, caplog) -> None:
    parent = tmp_path / "occupied"
    parent.write_text("not a directory")
    collector = Diagnostics(
        parent / "diagnostics.jsonl", requested_backend="onnx", configured_model="test"
    )
    collector.write()
    assert collector.failed
    assert "diagnostics unavailable" in caplog.text
    parent.unlink()
    collector.write()
    assert not collector.failed
    assert Sample.model_validate_json(collector.path.read_text()).pid > 0


def _completed(recorder: Recorder, stage: Stage, start: float, seconds: float) -> int:
    identifier = recorder.begin(stage, start, None)
    recorder.finish(SpanRecord(identifier, None, stage, start, start + seconds, 1, None, "ok"))
    return identifier


def _sampled_ids(snapshot: Snapshot) -> set[int]:
    retained = [*snapshot.spans, *(s for spans in snapshot.slowest_spans.values() for s in spans)]
    return {span.id for span in retained}


def test_slow_span_survives_a_burst_in_its_interval() -> None:
    recorder = Recorder()
    slow_commit = _completed(recorder, Stage.RAW_COMMIT, 0.0, 19.3)
    slow_read = _completed(recorder, Stage.READ, 0.0, 6.0)
    for index in range(1_000):
        _completed(recorder, Stage.PARSE, 20.0 + index, 0.001)
        _completed(recorder, Stage.RAW_COMMIT, 20.0 + index, 0.01)
    first = recorder.snapshot()
    assert first.dropped_spans > 0
    assert {slow_commit, slow_read} <= _sampled_ids(first)
    assert slow_commit not in _sampled_ids(recorder.snapshot())


def test_slowest_spans_are_the_longest_per_stage_and_bounded() -> None:
    recorder = Recorder()
    durations = [(index * 7919) % 1_000 / 100 for index in range(1_000)]
    for stage in Stage:
        for index, seconds in enumerate(durations):
            _completed(recorder, stage, float(index), seconds)
    snapshot = recorder.snapshot()
    longest = sorted(durations, reverse=True)[:SLOWEST_SPANS_PER_STAGE]
    assert set(snapshot.slowest_spans) == set(Stage)
    for spans in snapshot.slowest_spans.values():
        assert [round(s.end - s.start, 6) for s in spans] == longest
    assert recorder.snapshot().slowest_spans == {}


def test_slowest_spans_round_trip_through_a_written_sample(tmp_path: Path) -> None:
    path = tmp_path / "diagnostics.jsonl"
    collector = Diagnostics(path, requested_backend="onnx", configured_model="test")
    slow = _completed(collector.recorder, Stage.WRITER_RUN, 0.0, 12.0)
    for index in range(300):
        _completed(collector.recorder, Stage.WRITER_RUN, 13.0 + index, 0.01)
    collector.write()
    sample = Sample.model_validate_json(path.read_text())
    assert slow in _sampled_ids(sample.telemetry)
    assert slow not in {span.id for span in sample.telemetry.spans}


def test_samples_written_before_slowest_spans_still_validate() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "diagnostics" / "sample-0.38.1.jsonl"
    sample = Sample.model_validate_json(fixture.read_text())
    assert sample.version == "0.38.1"
    assert len(sample.telemetry.spans) == 3
    assert sample.telemetry.slowest_spans == {}


@pytest.mark.parametrize("outcome", ["error", "cancelled"])
def test_slow_failed_span_survives_overflow_and_serialization(
    tmp_path: Path, monkeypatch, outcome: str
) -> None:
    from recall.core import telemetry

    path = tmp_path / "diagnostics.jsonl"
    collector = Diagnostics(path, requested_backend="onnx", configured_model="test")
    clock = iter([0.0, 12.0])
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: next(clock))
    error = ValueError if outcome == "error" else asyncio.CancelledError
    token = current.set(collector.recorder)
    try:
        with pytest.raises(error), Span(Stage.SEARCH, thread_cpu=False):
            raise error("private input")
    finally:
        current.reset(token)
    monkeypatch.undo()
    for index in range(300):
        _completed(collector.recorder, Stage.SEARCH, 20.0 + index, 0.001)
    collector.write()
    sample = Sample.model_validate_json(path.read_text())
    longest = sample.telemetry.slowest_spans[Stage.SEARCH][0]
    assert longest.end - longest.start == 12.0
    assert longest.outcome == outcome
    assert longest.thread_cpu_seconds is None
    assert longest.id not in {span.id for span in sample.telemetry.spans}
    assert sample.telemetry.dropped_spans == 45
    aggregate = sample.telemetry.stages[Stage.SEARCH]
    assert aggregate.count == 301
    assert aggregate.errors == (outcome == "error")
    assert aggregate.cancelled == (outcome == "cancelled")
    assert "private input" not in path.read_text()
    assert collector.recorder.snapshot().slowest_spans == {}
