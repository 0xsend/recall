"""`summarize`: turn `daemon-run` result directories into the report tables.

Each input is an output directory of `daemon-run`; its `result.json` is read
read-only. The job writes `summary.json` and `summary.md` into a new output
directory: one row per run (every run, whatever its outcome), then per
profile and build the median and the worst of each measurement over the
runs that passed, with fixture identity and host load. A run that did not
pass is listed and counted, never folded into a median.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import PurePosixPath

from .evidence import JobFailed, Result
from .fs import Cancellation, Refused
from .isolation import Policy

Progress = Callable[[dict[str, object]], None]

RESULT_NAME = "result.json"
RESULT_BYTES_MAX = 64 * 1024 * 1024
INPUTS_MAX = 1000
ERROR_CHARS_MAX = 160
# Each measurement is lower-is-better, so the worst run is the largest value.
METRICS = (
    "readiness_seconds",
    "settle_seconds",
    "drain_seconds",
    "freshness_p95_seconds",
    "freshness_max_seconds",
    "writer_max_seconds",
    "checkpoint_max_seconds",
    "call_p95_seconds",
    "call_max_seconds",
    "rss_max_bytes",
    "idle_cpu_percent",
)


@dataclass(frozen=True)
class Run:
    """One result directory's measurements, read from its outcome line."""

    label: str
    profile: str
    outcome: str
    build: str
    tool: str
    fixture: str
    load_start: float | None
    load_end: float | None
    appended_bytes: int | None
    appended_lines: int | None
    metrics: Mapping[str, float | None]
    error: str | None


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True, help="new directory for the tables")
    parser.add_argument("results", nargs="+", help="daemon-run output directories")


def _steps(result: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    steps = result.get("steps")
    found: dict[str, Mapping[str, object]] = {}
    for step in steps if isinstance(steps, list) else ():
        if isinstance(step, dict) and isinstance(step.get("step"), str):
            found.setdefault(step["step"], step)
    return found


def _get(mapping: object, *keys: str) -> object:
    for key in keys:
        if not isinstance(mapping, dict):
            return None
        mapping = mapping.get(key)
    return mapping


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _identity(checkout: object) -> str:
    head = str(_get(checkout, "head") or "unknown")[:12]
    return f"{head}+{str(_get(checkout, 'dirty_hash'))[:8]}" if _get(checkout, "dirty") else head


def parse_run(label: str, result: Mapping[str, object]) -> Run:
    steps = _steps(result)
    scope = result.get("scope")
    final = steps.get("final", {})
    # A failed floor still carries its floors in the error.
    floors = _get(final, "floors") or _get(result, "error", "floors") or {}
    payload = _get(scope, "fixture", "payload")
    fixture = (
        ",".join(
            f"{PurePosixPath(name).name}:{digest[:12]}" for name, digest in sorted(payload.items())
        )
        if isinstance(payload, dict)
        else "unknown"
    )
    if _get(scope, "fixture", "pool_files"):
        fixture += f",pool:{str(_get(scope, 'fixture', 'pool_sha256'))[:12]}"
    error = result.get("error")
    return Run(
        label=label,
        profile=str(_get(scope, "profile") or "unknown"),
        outcome=str(result.get("outcome")),
        build=_identity(_get(scope, "build")),
        tool=_identity(_get(scope, "tool")),
        fixture=fixture,
        load_start=_number(_get(scope, "host_load_start", "load1")),
        load_end=_number(_get(scope, "host_load_end", "load1")),
        appended_bytes=_integer(_get(final, "appended_bytes")),
        appended_lines=_integer(_get(final, "appended_lines")),
        metrics={
            "readiness_seconds": _number(_get(steps.get("readiness"), "seconds")),
            "settle_seconds": _number(_get(steps.get("settle"), "seconds")),
            "drain_seconds": _number(_get(steps.get("drain"), "seconds")),
            "freshness_p95_seconds": _number(_get(floors, "freshness_seconds", "p95")),
            "freshness_max_seconds": _number(_get(floors, "freshness_seconds", "max")),
            "writer_max_seconds": _number(_get(floors, "writer_seconds_max")),
            "checkpoint_max_seconds": _number(_get(floors, "checkpoint_seconds_max")),
            "call_p95_seconds": _number(_get(floors, "call_seconds", "p95")),
            "call_max_seconds": _number(_get(floors, "call_seconds", "max")),
            "rss_max_bytes": _number(
                _get(floors, "rss_bytes_max") or _get(steps.get("window"), "rss_bytes_max")
            ),
            "idle_cpu_percent": _number(_get(steps.get("window"), "cpu_percent")),
        },
        error=_error_text(error) if isinstance(error, dict) else None,
    )


def _error_text(error: Mapping[str, object]) -> str:
    """The error's kind, then its message or, for errors without one, its scalar details."""
    message = error.get("message")
    if message is None:
        message = " ".join(
            f"{key}={value}"
            for key, value in error.items()
            if key != "kind" and isinstance(value, str | int | float)
        )
    return f"{error.get('kind')}: {str(message)[:ERROR_CHARS_MAX]}"


@dataclass(frozen=True)
class Group:
    """The runs of one profile, build and fixture; statistics cover the passed runs."""

    runs: int
    passed: int
    appended: tuple[tuple[int | None, int | None], ...]
    # Per metric: median and worst, or None when no passed run measured it.
    metrics: Mapping[str, Mapping[str, float] | None]

    def to_json(self) -> dict[str, object]:
        return {
            "runs": self.runs,
            "passed": self.passed,
            "appended": [list(item) for item in self.appended],
            **self.metrics,
        }


def groups(runs: Sequence[Run]) -> dict[tuple[str, str, str], Group]:
    """Per profile, build and fixture: run counts, then the median and worst of each metric."""
    grouped: dict[tuple[str, str, str], list[Run]] = {}
    for run in runs:
        grouped.setdefault((run.profile, run.build, run.fixture), []).append(run)
    table: dict[tuple[str, str, str], Group] = {}
    for key, members in sorted(grouped.items()):
        passed = [run for run in members if run.outcome == "passed"]
        metrics: dict[str, Mapping[str, float] | None] = {}
        for metric in METRICS:
            values = [value for run in passed if (value := run.metrics[metric]) is not None]
            metrics[metric] = (
                {"median": statistics.median(values), "worst": max(values)} if values else None
            )
        appended = tuple(
            sorted({(run.appended_bytes, run.appended_lines) for run in passed}, key=str)
        )
        table[key] = Group(len(members), len(passed), appended, metrics)
    return table


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 1e6 else f"{value / 1024**2:.0f} MiB"
    return str(value)


def markdown(runs: Sequence[Run], table: Mapping[tuple[str, str, str], Group]) -> str:
    lines = ["# Bench summary", "", "## Every run", ""]
    header = [
        "run",
        "profile",
        "outcome",
        "build",
        "tool",
        "fixture",
        "load start/end",
        "appended bytes/lines",
    ]
    header += [*METRICS, "error"]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for run in runs:
        cells = [
            run.label,
            run.profile,
            run.outcome,
            run.build,
            run.tool,
            run.fixture,
            f"{_cell(run.load_start)}/{_cell(run.load_end)}",
            f"{_cell(run.appended_bytes)}/{_cell(run.appended_lines)}",
            *(_cell(run.metrics[metric]) for metric in METRICS),
            run.error or "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Median and worst over passed runs", ""]
    header = ["profile", "build", "fixture", "runs", "passed", "appended bytes/lines", *METRICS]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for (profile, build, fixture), group in table.items():
        cells = [profile, build, fixture, _cell(group.runs), _cell(group.passed)]
        cells.append(", ".join(f"{_cell(size)}/{_cell(lines)}" for size, lines in group.appended))
        for metric in METRICS:
            value = group.metrics[metric]
            cells.append(
                "-" if value is None else f"{_cell(value['median'])} / {_cell(value['worst'])}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Cells are median / worst. `passed` means a run's measurements completed and met the",
        "floors its profile evaluates; it is not Path B or installed-host qualification.",
        "",
    ]
    return "\n".join(lines)


def run(
    args: argparse.Namespace, result: Result, cancellation: Cancellation, progress: Progress
) -> None:
    if len(args.results) > INPUTS_MAX:
        raise JobFailed("usage", f"at most {INPUTS_MAX} result directories")
    policy = Policy.for_host()
    runs: list[Run] = []
    with ExitStack() as held:
        roots = held.enter_context(policy.open_write_roots())
        result.leftovers = roots.leftovers
        for raw in args.results:
            cancellation.check()
            with policy.open_tree("result", raw) as directory:
                file = directory.open_file("result", RESULT_NAME)
                with file:
                    if file.status.st_size > RESULT_BYTES_MAX:
                        raise Refused("result", file.path, "is larger than a result can be")
                    data = file.head(file.status.st_size)
            try:
                document = json.loads(data)
            except json.JSONDecodeError as error:
                raise JobFailed("result", f"{raw}/{RESULT_NAME} is not JSON") from error
            if not isinstance(document, dict) or document.get("job") != "daemon-run":
                raise JobFailed("result", f"{raw}/{RESULT_NAME} is not a daemon-run result")
            runs.append(parse_run(directory.path.name, document))
        table = groups(runs)
        target = held.enter_context(roots.new_directory("--output", args.output))
        with target.create() as tree:
            summary = {
                "runs": [
                    {
                        **{key: value for key, value in vars(run).items() if key != "metrics"},
                        **run.metrics,
                    }
                    for run in runs
                ],
                "groups": [
                    {"profile": profile, "build": build, "fixture": fixture, **group.to_json()}
                    for (profile, build, fixture), group in table.items()
                ],
            }
            tree.publish(
                PurePosixPath("summary.json"), (json.dumps(summary, indent=2) + "\n").encode()
            )
            tree.publish(PurePosixPath("summary.md"), markdown(runs, table).encode())
            tree.commit(cancellation)
    result.step("summarize", runs=len(runs), passed=sum(run.outcome == "passed" for run in runs))
    result.scope = {"measurements": [], "output": str(target.path), "inputs": len(runs)}
