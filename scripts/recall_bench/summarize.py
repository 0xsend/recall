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
from pathlib import Path, PurePosixPath

from pydantic import ValidationError
from pydantic_core import from_json

from .documents import Checkout, Document, Floors, Seconds, Step
from .evidence import JobFailed, Result
from .fs import Cancellation, Refused
from .isolation import Policy

Progress = Callable[[dict[str, object]], None]

RESULT_NAME = "result.json"
RESULT_BYTES_MAX = 64 * 1024 * 1024
INPUTS_MAX = 1000
ERROR_CHARS_MAX = 240
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
    parser.add_argument(
        "results",
        nargs="+",
        help="daemon-run output directories, or files holding a daemon-run outcome line "
        "(a refusal before the run starts publishes only that line)",
    )


def _identity(checkout: Checkout | None) -> str:
    if checkout is None:
        return "unknown"
    head = (checkout.head or "unknown")[:12]
    return f"{head}+{(checkout.dirty_hash or 'unknown')[:8]}" if checkout.dirty else head


def parse_run(label: str, result: Document) -> Run:
    steps = {step.step: step for step in reversed(result.steps)}
    scope = result.scope
    final = steps.get("final", Step(step="final"))
    floors = final.floors or (result.error.floors if result.error else None) or Floors()
    committed = steps.get("committed", Step(step="committed"))
    freshness = floors.freshness_seconds or committed.freshness_seconds or Seconds(count=0)
    appended = steps.get("final") or steps.get("workload", Step(step="workload"))
    fixture = "unknown"
    if scope.fixture is not None:
        fixture = ",".join(
            f"{PurePosixPath(name).name}:{digest[:12]}"
            for name, digest in sorted(scope.fixture.payload.items())
        )
        if scope.fixture.pool_files:
            fixture += f",pool:{(scope.fixture.pool_sha256 or 'unknown')[:12]}"
    window = steps.get("window", Step(step="window"))
    calls = floors.call_seconds or Seconds(count=0)
    return Run(
        label=label,
        profile=scope.profile or "unknown",
        outcome=result.outcome,
        build=_identity(scope.build),
        tool=_identity(scope.tool),
        fixture=fixture,
        load_start=scope.host_load_start.load1 if scope.host_load_start else None,
        load_end=scope.host_load_end.load1 if scope.host_load_end else None,
        appended_bytes=appended.appended_bytes,
        appended_lines=appended.appended_lines,
        metrics={
            "readiness_seconds": steps.get("readiness", Step(step="readiness")).seconds,
            "settle_seconds": steps.get("settle", Step(step="settle")).seconds,
            "drain_seconds": steps.get("drain", Step(step="drain")).seconds,
            "freshness_p95_seconds": freshness.p95,
            "freshness_max_seconds": freshness.max,
            "writer_max_seconds": floors.writer_seconds_max,
            "checkpoint_max_seconds": floors.checkpoint_seconds_max,
            "call_p95_seconds": calls.p95,
            "call_max_seconds": floors.status_seconds_max if scope.profile == "idle" else calls.max,
            "rss_max_bytes": next(
                (
                    value
                    for value in (
                        floors.rss_bytes_max,
                        committed.rss_bytes_max,
                        window.rss_bytes_max,
                    )
                    if value is not None
                ),
                None,
            ),
            "idle_cpu_percent": window.cpu_percent,
        },
        error=_error_text(result.error.model_dump(exclude_none=True)) if result.error else None,
    )


def _error_text(error: Mapping[str, object]) -> str:
    """The error's kind, then its message or, for errors without one, its scalar details.

    A failed CLI call's own output follows its message: that is what the build said.
    """
    reason = error.get("reason")
    if isinstance(reason, str):
        return f"{error.get('kind')}: {reason}"
    message = error.get("message")
    if message is None:
        message = " ".join(
            f"{key}={value}"
            for key, value in error.items()
            if key != "kind" and isinstance(value, str | int | float)
        )
    output = next(
        (
            text.strip()
            for key in ("stdout", "stderr")
            if (text := error.get(key)) and isinstance(text, str) and text.strip()
        ),
        None,
    )
    text = f"{message}: {output}" if output else str(message)
    return f"{error.get('kind')}: {text[:ERROR_CHARS_MAX]}"


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
    # Error text quotes the build's output, which may hold table syntax.
    return str(value).replace("|", "\\|").replace("\n", " ")


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
            _cell(run.error or ""),
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
        "`-` marks a measurement the run did not reach.",
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
    # Each run once, however it was given (its directory, a saved outcome line, a copy):
    # two runs never publish the same document, so the document is the run's identity.
    seen: dict[str, str] = {}
    with ExitStack() as held:
        roots = held.enter_context(policy.open_write_roots())
        result.leftovers = roots.leftovers
        for raw in args.results:
            cancellation.check()
            # A run's output directory holds its result; a saved outcome line is the result.
            path = Path(raw)
            is_line = path.is_file()
            tree, name = (path.parent, path.name) if is_line else (path, RESULT_NAME)
            with policy.open_tree("result", tree) as directory:
                file = directory.open_file("result", name)
                with file:
                    if file.status.st_size > RESULT_BYTES_MAX:
                        raise Refused("result", file.path, "is larger than a result can be")
                    data = file.head(file.status.st_size)
            try:
                document = from_json(data, allow_inf_nan=False)
            except ValueError as error:
                raise Refused("result", file.path, f"invalid JSON: {error}") from error
            try:
                validated = Document.model_validate(document)
            except ValidationError as error:
                detail = error.errors(include_url=False, include_input=False)[0]
                raise Refused(
                    "result", file.path, f"invalid daemon-run result: {detail}"
                ) from error
            identity = validated.model_dump_json(exclude_unset=True)
            if identity in seen:
                raise Refused("result", file.path, f"is the same run as {seen[identity]}")
            seen[identity] = raw
            runs.append(parse_run(path.stem if is_line else directory.path.name, validated))
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
