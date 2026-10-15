"""Diagnostics preserve outcomes, report concurrency honestly, and bound storage."""

import asyncio
import threading
from pathlib import Path

import pytest
from recall.core.telemetry import (
    Counter,
    DiagnosticExecutor,
    Recorder,
    Span,
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


def test_samples_are_readable_and_rotation_retains_bounded_history(
    tmp_path: Path, monkeypatch
) -> None:
    from recall.services import diagnostics

    monkeypatch.setattr(diagnostics, "LOG_BYTES_MAX", 1)
    path = tmp_path / "diagnostics.jsonl"
    collector = Diagnostics(path, requested_backend="auto", configured_model="test-model")
    for _ in range(7):
        collector.write()
    files = sorted(tmp_path.iterdir())
    assert len(files) == 4
    sample = Sample.model_validate_json(path.read_text())
    assert sample.process_cpu_seconds > 0
    assert sample.rss_bytes > 0
    assert sample.threads > 0
    assert sample.telemetry.backend is None
    assert sample.requested_backend == "auto"
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
