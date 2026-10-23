"""Preserve completed runtime observations as transcript-free comparison evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, TypedDict

from measure import LANE
from recall.core.telemetry import Counter, Stage
from recall.services.diagnostics import Sample

GROUPS = {
    "baseline": ("baseline-1", "baseline-2", "baseline-3", "baseline-4"),
    "u128": ("u128-1c", "u128-2", "u128-4", "u128-5"),
    "drop-results": (
        "drop-results-1",
        "drop-results-2",
        "drop-results-4",
    ),
    "watermark-16": ("watermark-16-1", "watermark-16-2", "watermark-16-3"),
    "threads-1": ("threads-1", "threads-2", "threads-3", "threads-6"),
    "uncompressed": ("uncompressed-1", "uncompressed-2", "uncompressed-3"),
    "no-roaring": ("no-roaring-1", "no-roaring-2", "no-roaring-3", "no-roaring-4"),
    "rebuilt": ("rebuilt-1", "rebuilt-2", "rebuilt-3"),
}
EVIDENCE = LANE / "scratch/evidence"
OUTPUT = Path(__file__).with_name("evidence")


class CensoredAttempt(TypedDict):
    result: dict[str, Any]
    qualification: str
    checkpoints: list[dict[str, Any]]


def run_evidence(label: str) -> dict[str, Any]:
    directory = EVIDENCE / label
    result = json.loads((directory / "result.json").read_text())
    assert result["outcome"] == "passed", label
    assert all(code == 0 for code in result["steps"].values()), label
    checkpoints = [
        json.loads(line) for line in (directory / "checkpoints.jsonl").read_text().splitlines()
    ]
    diagnostics = [
        Sample.model_validate_json(line)
        for file in directory.glob("diagnostics.jsonl*")
        for line in file.read_text().splitlines()
    ]
    last = max(diagnostics, key=lambda sample: sample.monotonic)
    stage = last.telemetry.stages[Stage.CHECKPOINT]
    assert len(checkpoints) == stage.count and stage.errors == 0 and stage.cancelled == 0, label
    durations = [checkpoint["duration_seconds"] for checkpoint in checkpoints]
    assert 0 <= stage.wall_seconds_max - max(durations) < 0.01, label
    assert 0 <= stage.wall_seconds_sum - sum(durations) < 0.01 * len(durations), label
    assert all(sample.telemetry.counters[Counter.GENERATION_ERRORS] == 0 for sample in diagnostics)
    samples = json.loads((directory / "samples.json").read_text())
    loads = [result["load_start"][0], result["load_end"][0]]
    loads.extend(sample["load"][0] for sample in samples)
    counts = json.loads((directory / "final-stats.json").read_text())
    pool = json.loads((directory / "pool.json").read_text())
    return {
        "label": label,
        "product_head": result["head"],
        "owned_pid": json.loads((directory / "owned.json").read_text())["pid"],
        "daemon_started_at": last.started_at,
        "steps": result["steps"],
        "seconds": result["seconds"],
        "appends": result["appends"],
        "appended_lines": result["appended_lines"],
        "appended_bytes": result["appended_bytes"],
        "pool_manifest_sha256": hashlib.sha256(
            json.dumps(pool, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "final_counts": {
            name: counts[name] for name in ("sessions", "messages", "tool_calls", "bash_calls")
        },
        "checkpoint_count": len(checkpoints),
        "checkpoint_seconds_max": max(durations),
        "checkpoint_seconds_sum": sum(durations),
        "checkpoint_thread_cpu_seconds_sum": stage.thread_cpu_seconds_sum,
        "checkpoint_bytes_sum": sum(checkpoint["physical_bytes"] for checkpoint in checkpoints),
        "floor_passed": max(durations) <= 5,
        "load1_median": statistics.median(loads),
        "load1_max": max(loads),
        "low_load_cohort_eligible": max(loads) <= 12,
        "sampled_rss_bytes_max": max(sample.rss_bytes for sample in diagnostics),
        "owned_children_kernel_peak_rss_bytes": result.get("owned_children_kernel_peak_rss_bytes"),
        "process_write_bytes_final_diagnostic": last.io_write_bytes,
        "checkpoints": checkpoint_rows(checkpoints),
    }


def checkpoint_rows(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": ordinal,
            "duration_seconds": checkpoint["duration_seconds"],
            "physical_bytes": checkpoint["physical_bytes"],
            "logical_bytes": checkpoint["logical_bytes"],
            "wal_bytes_before": checkpoint["wal_bytes_before"],
            "wal_bytes_after": checkpoint["wal_bytes_after"],
            "load1": checkpoint["load"][0],
        }
        for ordinal, checkpoint in enumerate(checkpoints, 1)
    ]


def group_evidence(labels: tuple[str, ...], runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    maxima = [runs[label]["checkpoint_seconds_max"] for label in labels]
    written = [runs[label]["checkpoint_bytes_sum"] for label in labels]
    checkpoints = [checkpoint for label in labels for checkpoint in runs[label]["checkpoints"]]
    return {
        "processes": len(labels),
        "checkpoints": len(checkpoints),
        "checkpoint_seconds_median": statistics.median(
            checkpoint["duration_seconds"] for checkpoint in checkpoints
        ),
        "run_max_seconds_median": statistics.median(maxima),
        "run_max_seconds_worst": max(maxima),
        "checkpoint_bytes_median": statistics.median(
            checkpoint["physical_bytes"] for checkpoint in checkpoints
        ),
        "checkpoint_bytes_worst": max(checkpoint["physical_bytes"] for checkpoint in checkpoints),
        "checkpoint_bytes_per_run_median": statistics.median(written),
        "checkpoint_bytes_per_run_worst": max(written),
        "checkpoint_load1_median": statistics.median(
            checkpoint["load1"] for checkpoint in checkpoints
        ),
        "checkpoint_load1_min": min(checkpoint["load1"] for checkpoint in checkpoints),
        "checkpoint_load1_max": max(checkpoint["load1"] for checkpoint in checkpoints),
        "every_checkpoint_le_5_seconds": max(maxima) <= 5,
    }


def main() -> int:
    missing = [
        label
        for labels in GROUPS.values()
        for label in labels
        if not (EVIDENCE / label / "result.json").exists()
    ]
    if missing:
        print(json.dumps({"outcome": "incomplete", "missing": missing}))
        return 1
    outcomes = {
        label: json.loads((EVIDENCE / label / "result.json").read_text())
        for labels in GROUPS.values()
        for label in labels
    }
    completed = {
        name: tuple(label for label in labels if outcomes[label]["outcome"] == "passed")
        for name, labels in GROUPS.items()
    }
    runs = {label: run_evidence(label) for labels in completed.values() for label in labels}
    censored: dict[str, CensoredAttempt] = {}
    for label, result in outcomes.items():
        if result["outcome"] == "passed":
            continue
        directory = EVIDENCE / label
        path = directory / "checkpoints.jsonl"
        observations = (
            [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        )
        censored[label] = {
            "result": result,
            "qualification": "incomplete or failed; excluded from complete-run aggregates",
            "checkpoints": checkpoint_rows(observations),
        }
    reference = runs["baseline-1"]["final_counts"]
    assert all(run["final_counts"] == reference for run in runs.values())
    assert all(run["appends"] == 1100 and run["appended_lines"] == 24027 for run in runs.values())
    assert len({run["appended_bytes"] for run in runs.values()}) == 1
    assert len({run["pool_manifest_sha256"] for run in runs.values()}) == 1
    assert len({(run["owned_pid"], run["daemon_started_at"]) for run in runs.values()}) == len(runs)
    groups = {
        name: {
            "attempted_processes": len(GROUPS[name]),
            "complete_labels": labels,
            "censored_labels": tuple(label for label in GROUPS[name] if label in censored),
            **(group_evidence(labels, runs) if labels else {"processes": 0}),
        }
        for name, labels in completed.items()
    }
    low_load_labels = {
        name: tuple(label for label in labels if runs[label]["low_load_cohort_eligible"])
        for name, labels in completed.items()
    }
    low_load = {
        name: {"labels": labels, **group_evidence(labels, runs)}
        for name, labels in low_load_labels.items()
        if labels
    }
    matched = all(len(labels) >= 3 for labels in low_load_labels.values())
    OUTPUT.mkdir(exist_ok=True)
    result = {
        "outcome": "passed" if matched else "incomplete",
        "minimum_three_complete_processes_per_arm": all(
            len(labels) >= 3 for labels in completed.values()
        ),
        "low_load_cohort_limit": 12,
        "matched_load_comparison_complete": matched,
        "low_load_cohort": low_load,
        "meaning": "measurement completeness, not performance acceptance",
        "groups": groups,
        "runs": runs,
        "censored_attempts": censored,
        "cancelled_before_measurement": json.loads(
            (EVIDENCE / "cancelled-queues.json").read_text()
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUTPUT / "checkpoints.csv").open("w", newline="") as stream:
        columns = (
            "label",
            "outcome",
            "ordinal",
            "duration_seconds",
            "physical_bytes",
            "logical_bytes",
            "wal_bytes_before",
            "wal_bytes_after",
            "load1",
        )
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for run in runs.values():
            writer.writerows(
                {"label": run["label"], "outcome": "passed", **checkpoint}
                for checkpoint in run["checkpoints"]
            )
        for label, attempt in censored.items():
            writer.writerows(
                {"label": label, "outcome": attempt["result"]["outcome"], **checkpoint}
                for checkpoint in attempt["checkpoints"]
            )
    print(
        json.dumps({"outcome": result["outcome"], "groups": groups, "low_load_cohort": low_load}),
        flush=True,
    )
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
