"""Contracts of the real-model bench; real inference is an author/guest run."""

from __future__ import annotations

import argparse
import math

import pytest
from pydantic import ValidationError

from scripts.recall_bench.daemon import add_profile_arguments
from scripts.recall_bench.embedding import (
    Reference,
    _pending,
    equivalence,
    inference_intervals,
    logged_errors,
    metrics,
    schedule,
    validate_options,
)
from scripts.recall_bench.embedding_trace import Event
from scripts.recall_bench.evidence import JobFailed


def reference(**changes) -> Reference:
    values = {
        "hashes": ["input-a"],
        "vectors": [[1.0] + [0.0] * 383],
        "retrieval": {"vector": ["message-a"], "hybrid": ["message-a"]},
    }
    values.update(changes)
    return Reference.model_validate(values)


def test_schedule_has_mixed_maximum_and_long_batches_without_duplicate_inputs():
    groups = schedule(256)
    assert [group.kind for group in groups] == ["mixed", "maximum", "long-batch"]
    assert [len(group.texts) for group in groups] == [36, 36, 256]
    assert [text.count("database ") for text in groups[0].texts[:4]] == [4, 32, 128, 600]
    assert all(text.count("database ") == 600 for group in groups[1:] for text in group.texts)
    texts = [text for group in groups for text in group.texts]
    assert len(set(texts)) == 328
    assert schedule(256) == groups
    assert len(schedule(36)[2].texts) == 36


@pytest.mark.parametrize("rows", [0, 35, 513, 100000])
def test_schedule_refuses_unbounded_or_non_long_batch(rows):
    with pytest.raises(ValueError):
        schedule(rows)


def test_equivalence_accepts_normalized_vectors_inside_predeclared_tolerance():
    candidate = reference(vectors=[[math.sqrt(1 - 0.001**2), 0.001] + [0.0] * 382])
    result = equivalence(reference(), candidate)
    assert result["absolute_error_max"] == 0.001
    assert result["vectors"] == 1
    assert result["retrieval_equal"] is True


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"hashes": ["another-input"]}, "inputs"),
        ({"vectors": [[0.0, 1.0] + [0.0] * 382]}, "tolerance"),
        ({"vectors": [[0.0] * 384]}, "normalized"),
        ({"vectors": [[1.0] + [0.0] * 383] * 2}, "counts"),
        ({"retrieval": {"vector": ["different"]}}, "retrieval"),
    ],
)
def test_equivalence_fails_closed(changes, reason):
    with pytest.raises(JobFailed, match=reason):
        equivalence(reference(), reference(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"vectors": [[1.0] * 383]},
        {"vectors": [[float("nan")] * 384]},
        {"vectors": [[float("inf")] * 384]},
        {"tolerance": 0.003},
        {"format": "unknown"},
    ],
)
def test_reference_rejects_invalid_boundary_data(changes):
    with pytest.raises(ValidationError):
        reference(**changes)


def event(kind, at, cpu=1, rss=1000, footprint=None, instance=1, thread=2):
    return Event(
        kind=kind,
        at=at,
        cpu_seconds=cpu,
        rss_bytes=rss,
        rss_hwm_bytes=4000,
        footprint_bytes=footprint,
        instance=instance,
        thread=thread,
    )


def test_metrics_keep_cpu_elapsed_peak_retained_and_footprint_separate():
    events = [
        event("load-start", 0, cpu=1),
        event("sample", 1, cpu=2, rss=3000, footprint=5000),
        event("load-end", 2, cpu=3, rss=2000, footprint=4000),
        event("inference-start", 3, cpu=3),
        event("inference-end", 4, cpu=4),
        event("unload-start", 5, cpu=4),
        event("unload-end", 6, cpu=4.5),
    ]
    measured = metrics(events)
    phases = measured["phases"]
    assert isinstance(phases, dict)
    load = phases["load"][0]
    assert load["seconds"] == 2
    assert load["cpu_seconds"] == 2
    assert load["rss_bytes_max"] == 3000
    assert load["rss_bytes_retained"] == 2000
    assert load["rss_hwm_bytes"] == 4000
    assert load["footprint_bytes_max"] == 5000
    assert load["footprint_bytes_retained"] == 4000
    whole = measured["whole_process"]
    assert isinstance(whole, dict)
    assert whole["cpu_seconds"] == 3.5
    assert inference_intervals(events) == [(3, 4)]


def test_inference_intervals_do_not_pair_a_start_with_every_later_end():
    events = [
        event("inference-start", 1),
        event("inference-end", 2),
        event("inference-start", 3),
        event("inference-end", 4),
    ]
    assert inference_intervals(events) == [(1, 2), (3, 4)]
    with pytest.raises(JobFailed, match="did not finish"):
        inference_intervals(events[:-1])


@pytest.mark.parametrize(
    "events",
    [
        [],
        [event("sample", 1)],
        [event("load-start", 1), event("load-end", 2)],
        [event("inference-end", 1)],
        [
            Event(
                kind="error",
                at=1,
                cpu_seconds=1,
                rss_bytes=1,
                rss_hwm_bytes=1,
                error="OutOfMemoryException",
            )
        ],
    ],
)
def test_absent_or_failed_lifecycle_never_passes(events):
    with pytest.raises(JobFailed):
        metrics(events)


def status():
    return {
        "reconciliation": {"pending": 0, "noticed_pending": 0, "enrichment_ready": True},
        "embed_pending_eligible": 0,
        "embed_pending_held": 0,
        "catchup_in_progress": False,
    }


def test_drain_requires_fresh_eligible_and_held_counts():
    assert not _pending(status())
    for key in ("embed_pending_eligible", "embed_pending_held"):
        value = status()
        value[key] = 1
        assert _pending(value)
        del value[key]
        assert _pending(value)
    assert _pending({**status(), "reconciliation": {}})


@pytest.mark.parametrize(
    "changes",
    [
        {"embed_last_error": "OutOfMemoryException: failed to pin block"},
        {"watch_total_failed": 1},
        {"reconciliation": {"enrichment_error": "failed"}},
    ],
)
def test_daemon_errors_are_failures_even_if_a_later_snapshot_could_recover(changes):
    with pytest.raises(JobFailed):
        _pending({**status(), **changes})


def test_profile_options_require_explicit_scratch_cache_and_reference_mode():
    parser = argparse.ArgumentParser()
    add_profile_arguments(parser)
    for options in (
        ["--profile", "embed"],
        ["--profile", "embed", "--model-cache", "/scratch/cache"],
        ["--profile", "idle", "--embed-record-reference"],
        [
            "--profile",
            "embed",
            "--model-cache",
            "/scratch/cache",
            "--embed-record-reference",
            "--writes",
            "1",
        ],
    ):
        with pytest.raises(JobFailed):
            validate_options(parser.parse_args(options))
    validate_options(
        parser.parse_args(
            ["--profile", "embed", "--model-cache", "/scratch/cache", "--embed-record-reference"]
        )
    )


def test_recovered_oom_and_error_logs_disqualify_the_measurement():
    assert logged_errors(
        b"INFO ready\nWARNING OutOfMemoryException: pin failed\nINFO recovered"
    ) == ["WARNING OutOfMemoryException: pin failed"]
    assert logged_errors(b"ERROR embedding failed\n") == ["ERROR embedding failed"]
    assert logged_errors(b"INFO generation_errors=0\n") == []


def test_retrieval_preserves_logical_records_across_fresh_run_roots():
    from pathlib import Path

    from scripts.recall_bench.embedding import _retrieval

    before = {
        "source": "claude_code",
        "source_path": "/scratch/baseline/pool/a.jsonl",
        "kind": "message",
        "role": "user",
        "content": "a distinct result",
        "timestamp": "2026-11-01 00:00:00",
        "session_id": "physical-before",
        "message_id": "message-before",
        "score": 0.8,
    }
    after = {
        **before,
        "source_path": "/scratch/candidate/pool/a.jsonl",
        "session_id": "physical-after",
        "message_id": "message-after",
        "score": 0.800001,
    }
    expected = _retrieval({"results": [before]}, Path("/scratch/baseline"))
    assert _retrieval({"results": [after]}, Path("/scratch/candidate")) == expected
    for changes in (
        {"content": "different result"},
        {"role": "assistant"},
        {"source_path": "/scratch/candidate/pool/b.jsonl"},
    ):
        assert (
            _retrieval({"results": [{**after, **changes}]}, Path("/scratch/candidate")) != expected
        )
    assert _retrieval({"results": [after, after]}, Path("/scratch/candidate")) == expected * 2
    with pytest.raises(JobFailed, match="logical identity"):
        _retrieval({"results": [{"session_id": "no-record"}]}, Path("/scratch/candidate"))


def test_post_inference_idle_excludes_unload_cpu():
    from scripts.recall_bench.embedding import idle_metrics

    events = [
        event("sample", 1, cpu=1),
        event("sample", 2, cpu=1),
        event("unload-start", 3, cpu=1),
        event("unload-end", 4, cpu=3),
    ]
    idle = idle_metrics(events, 1)
    assert idle["seconds"] == 2
    assert idle["cpu_seconds"] == 0
    assert idle["cpu_percent"] == 0
