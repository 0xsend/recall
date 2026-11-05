"""The BRIEF idle contract: quiet entry, invalidation and CPU verdicts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from scripts.recall_bench.idle import Observation, QuietGate, Window, attributed
from scripts.recall_bench.samplers import Sample
from scripts.recall_bench.turns import Sample as Diagnostic
from scripts.recall_bench.turns import Span


def observation(at: float, **changes) -> Observation:
    return replace(Observation(at, at, 3, False, 0, True), **changes)


def test_gate_requires_a_full_quiet_interval_and_a_fresh_diagnostic_flush():
    gate = QuietGate()
    assert gate.observe(observation(0)) is None
    assert gate.observe(observation(1)) is None
    for second in range(2, 120):
        assert gate.observe(observation(second)) is None
    assert gate.observe(observation(120, diagnostics_at=119)) is None
    window = gate.observe(observation(121))
    assert window is not None
    assert window.quiet_since == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"raw_commits": 4},
        {"raw_active": True},
        {"active_sessions": 1},
        {"ready": False},
    ],
)
def test_entry_restarts_when_the_daemon_reports_work(changes):
    gate = QuietGate()
    for second in range(60):
        assert gate.observe(observation(second)) is None
    assert gate.observe(observation(60, **changes)) is None
    # Reset raw counter is another change; it cannot inherit earlier quiet time.
    for second in range(61, 181):
        assert gate.observe(observation(second)) is None
    assert gate.observe(observation(181)) is not None


def test_unobserved_interval_cannot_count_toward_quiet_entry():
    gate = QuietGate()
    gate.observe(observation(0))
    assert gate.observe(observation(120)) is None
    assert gate.quiet_since == 120


def samples(cpu: float, duration: float = 600) -> list[Sample]:
    return [Sample(121, 10, 1000, 2, None), Sample(121 + duration, 10 + cpu, 2000, 2, None)]


def operation(cpu: float) -> Span:
    return Span(1, "executor_run", 130, 131, "ok", cpu, "inventory capture")


@pytest.mark.parametrize(
    ("cpu", "op_cpu", "expected"),
    [
        (30.0, 1.0, "passed"),
        (30.01, 1.0, "failed"),
        (1.0, 1.001, "failed"),
    ],
)
def test_cpu_verdict_uses_one_core_and_inclusive_per_occurrence_limits(cpu, op_cpu, expected):
    window = Window(observation(121), 0)
    verdict = window.verdict(samples(cpu), [operation(op_cpu)], [])
    assert verdict["verdict"] == expected
    assert verdict["cpu_percent"] == pytest.approx(cpu / 6)
    assert verdict["rss_bytes_max"] == 2000


@pytest.mark.parametrize(
    "changes",
    [
        {"raw_commits": 4},
        {"raw_active": True},
        {"active_sessions": 1},
        {"ready": False},
    ],
)
def test_a_broken_window_stays_invalid_even_if_it_becomes_quiet_again(changes):
    window = Window(observation(121), 0)
    window.observe(observation(122, **changes))
    window.observe(observation(123))
    assert window.verdict(samples(1), [operation(0.1)], [])["verdict"] == "invalid"


def test_short_windows_and_missing_attribution_never_pass():
    window = Window(observation(121), 0)
    assert window.verdict(samples(0, 599.99), [operation(0)], [])["verdict"] == "invalid"
    assert window.verdict(samples(0), [], ["lost records"])["verdict"] == "invalid"
    window.observe(observation(125))
    assert "per-second observation gap" in window.invalid


def diagnostic(at, count, spans=(), **changes):
    return replace(
        Diagnostic(
            at,
            at,
            {"executor_run": {"count": count, "process_cpu_samples": count}},
            {},
            spans,
            (),
            0,
            cpu_spans=spans,
        ),
        **changes,
    )


def test_attribution_covers_boundary_spans_and_deduplicates_without_summing_nested_costs():
    parent = Span(1, "executor_run", 90, 110, "ok", 0.9, "inventory")
    child = Span(2, "inventory", 99, 105, "ok", 0.8)
    end = Span(3, "executor_run", 199, 205, "ok", 0.7, "discovery")
    records = [
        diagnostic(95, 0),
        diagnostic(120, 2, (parent, child), slowest=(parent,)),
        diagnostic(210, 3, (end,)),
    ]
    operations, missing = attributed(records, 100, 200)
    assert missing == []
    assert [(op.label, op.process_cpu_seconds) for op in operations if op.id > 0] == [
        ("inventory", 0.9),
        (None, 0.8),
        ("discovery", 0.7),
    ]
    window = Window(observation(121), 0)
    assert window.verdict(samples(1), operations, [])["operation_cpu_seconds_max"] == 0.9


@pytest.mark.parametrize(
    "changes",
    [
        {"dropped_cpu_spans": 1},
        {"dropped_active": 1},
        {"output_failures": 1},
        {"active": (("executor_run", 150),)},
    ],
)
def test_lost_or_unfinished_operations_invalidate_attribution(changes):
    _, missing = attributed(
        [diagnostic(90, 0), diagnostic(210, 1, (operation(0.1),), **changes)], 100, 200
    )
    assert missing


def test_an_older_build_and_discarded_intervals_cannot_qualify():
    _, missing = attributed([diagnostic(90, 0), diagnostic(210, 2, (operation(0.1),))], 100, 200)
    assert "per-occurrence diagnostic records are incomplete" in missing
    old = replace(operation(0.1), process_cpu_seconds=None)
    _, missing = attributed([diagnostic(90, 0), diagnostic(210, 1, (old,))], 100, 200)
    assert "per-occurrence process CPU instrumentation unavailable" in missing


def test_real_daemon_reports_delayed_enrichment_separately(tmp_path):
    # Real raw indexing, pending/held selection, publication, RPC and diagnostics;
    # the model stand-in isolates this test from downloads and inference hardware.
    import json
    import os
    import sys

    from tests.test_scripts.test_bench_daemon import Lane, assert_every_group_gone, bench, prepare
    from tests.test_scripts.test_bench_profiles import slow_writer_build

    root = tmp_path / "lane"
    root.mkdir()
    lane = Lane(root, root / "canary", {**os.environ, "BENCH_WRITE_ROOTS": str(root)})
    template = prepare(lane, "template", "launchd" if sys.platform == "darwin" else "systemd")
    cache = root / "model-cache"
    cache.mkdir()
    (cache / "fixture.txt").write_text("model is replaced only in this test")
    build = slow_writer_build(
        tmp_path / "build",
        """
from recall.services.embed_phase import EmbedPhaseState
from recall.services import system_state
system_state.check_load = lambda threshold: system_state.EmbedPreconditionResult(ok=True)
class Backend:
    model_id = 'BAAI/bge-small-en-v1.5'
    dimensions = 384
    query_prefix = ''
    def embed(self, texts):
        return [[0.1] * 384 for _ in texts]
def load(self):
    self.backend = Backend()
    return self.backend
EmbedPhaseState.load_backend = load
""",
    )
    output, run = root / "output", root / "run"
    status, result = bench(
        lane,
        "daemon-run",
        "--template",
        template,
        "--build",
        build,
        "--output",
        output,
        "--run-dir",
        run,
        "--profile",
        "idle",
        "--window",
        "2",
        "--idle-transition",
        "--idle-model-cache",
        cache,
        "--deadline",
        "120",
        "--settle-timeout",
        "60",
    )
    assert (status, result["outcome"]) == (0, "passed"), json.dumps(result.get("error", result))
    steps = {step["step"]: step for step in result["steps"]}
    assert steps["transition"]["pending_seen"] is True
    assert steps["transition"]["pending_eligible"] == steps["transition"]["pending_held"] == 0
    assert steps["transition"]["samples"] >= 2
    assert steps["window"]["verdict"] == "passed"
    assert_every_group_gone(output)
