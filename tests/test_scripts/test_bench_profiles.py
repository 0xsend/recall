"""Drive `bench.py daemon-run --profile append|cohort` with real processes.

The build under test is this checkout, against a template prepared from an
empty synthetic backup and a pool of generated Claude Code transcripts. A
stand-in build runs this checkout's recall with one writer turn held past
five seconds, to show that a breached floor fails the run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import duckdb
import psutil
import pytest

from scripts.recall_bench.load import ExitWatch
from scripts.recall_bench.workload import Workload
from tests.test_scripts.test_bench_daemon import (
    NATIVE,
    SERVICE_TOOLS,
    Lane,
    Locked,
    assert_every_group_gone,
    assert_home_untouched,
    assert_lock_released,
    assert_no_service_reach,
    bench,
    synthetic_backup,
)

ROOT = Path(__file__).resolve().parents[2]
POOL_FILES = 3
POOL_LINES = 400
# Two appenders, ten writes each (BRIEF judges freshness over at least 20), two pool copies.
WORKLOAD = ("--appenders", "2", "--writes", "10", "--interval-ms", "100", "--copies", "2")
STEP_FIELDS = ("seed", "workload", "drain", "verify", "final")


@pytest.fixture(scope="module")
def lane(tmp_path_factory: pytest.TempPathFactory) -> Lane:
    """A short write root, and canaries first on the tool's own PATH."""
    root = tmp_path_factory.mktemp("bp")
    canary_bin = root / "canary-bin"
    canary_bin.mkdir()
    log = root / "canary-invocations"
    for name in SERVICE_TOOLS:
        script = canary_bin / name
        script.write_text(f"#!/bin/sh\necho \"$0 $*\" >> '{log}'\nexit 97\n")
        script.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{canary_bin}:{environment['PATH']}"
    environment["BENCH_WRITE_ROOTS"] = str(root)
    return Lane(root, log, environment)


def transcript(lines: int) -> str:
    """A Claude Code session of alternating user and assistant text messages."""
    rows = []
    for index in range(lines):
        role = "user" if index % 2 == 0 else "assistant"
        minute, second = divmod(index, 60)
        rows.append(
            json.dumps(
                {
                    "type": "message",
                    "timestamp": f"2026-10-30T10:{minute:02d}:{second:02d}Z",
                    "message": {
                        "role": role,
                        "content": [{"type": "text", "text": f"{role} line {index} of the pool"}],
                    },
                }
            )
        )
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def template(lane: Lane) -> Path:
    directory = lane.new("pool-src")
    backup = synthetic_backup(directory, NATIVE)
    pool = directory / "pool"
    (pool / "claude_code").mkdir(parents=True)
    for number in range(POOL_FILES):
        name = str(uuid.UUID(int=number + 1, version=4))
        (pool / "claude_code" / f"{name}.jsonl").write_text(transcript(POOL_LINES))
    template = lane.new("tp")
    status, line = bench(
        lane, "prepare", "--backup", backup, "--template", template, "--source-pool", pool
    )
    assert (status, line["outcome"]) == (0, "passed"), line
    return template


def run_args(
    template: Path, output: Path, run: Path, profile: str, build: Path = ROOT
) -> list[str | Path]:
    return [
        "--template", template, "--build", build, "--output", output, "--run-dir", run,
        "--profile", profile, *WORKLOAD,
    ]  # fmt: skip


def independent_counts(run: Path, scratch: Path) -> dict[str, int]:
    """Index the run's final transcripts from scratch with the build, without a daemon."""
    roots = scratch / "roots"
    shutil.copytree(run / "pool", roots)
    data = scratch / "data"
    data.mkdir()
    config = scratch / "config.toml"
    config.write_text(
        "".join(
            f'[sources.{source}]\nroots = ["{roots / source}"]\n'
            for source in ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
        )
    )
    script = textwrap.dedent(
        """
        from recall.services.indexer import index_sessions
        index_sessions(source=None, full=False, recreate=True, verbose=False)
        """
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("RECALL_")}
    environment.update(
        RECALL_DATA_DIR=str(data),
        RECALL_CONFIG_PATH=str(config),
        RECALL_DB_PATH=str(data / "recall.duckdb"),
        RECALL_LOCK_PATH=str(data / "recall.lock"),
    )
    subprocess.run([sys.executable, "-c", script], env=environment, check=True, timeout=300)
    counts: dict[str, int] = {}
    with duckdb.connect(str(data / "recall.duckdb"), read_only=True) as database:
        for table in ("sessions", "messages"):
            row = database.execute(f"SELECT count(*) FROM {table}").fetchone()
            assert row is not None
            counts[table] = row[0]
    return counts


@dataclass(frozen=True)
class Finished:
    output: Path
    run: Path
    status: int
    result: dict


@pytest.fixture(scope="module")
def append_run(lane: Lane, template: Path) -> Finished:
    """One append run on the `with_bench_lock.py -- bench.py` shape."""
    output, run, queue = lane.new("o-append"), lane.new("r-append"), lane.new("queue")
    locked = Locked(lane, queue, *run_args(template, output, run, "append"))
    status, result = locked.finish()
    assert_lock_released(lane, queue)
    return Finished(output, run, status, result)


@pytest.fixture(scope="module")
def slow_run(lane: Lane, template: Path, tmp_path_factory: pytest.TempPathFactory) -> Finished:
    """One append run of a build whose first writer turn takes 5.5 s."""
    build = slow_writer_build(tmp_path_factory.mktemp("slow"))
    output, run = lane.new("o-slow"), lane.new("r-slow")
    status, result = bench(lane, "daemon-run", *run_args(template, output, run, "append", build))
    return Finished(output, run, status, result)


def test_an_append_run_passes_with_exact_counts(
    lane: Lane, append_run: Finished, tmp_path: Path
) -> None:
    output, run, status, result = (
        append_run.output, append_run.run, append_run.status, append_run.result
    )  # fmt: skip
    assert (status, result["outcome"]) == (0, "passed"), result
    steps = {step["step"]: step for step in result["steps"]}
    assert set(STEP_FIELDS) <= steps.keys()
    # Two copies of three transcripts, each seeded with 70% of its lines.
    assert steps["seed"]["sessions"] == 2 * POOL_FILES
    appended = [json.loads(line) for line in (output / "appends.jsonl").read_text().splitlines()]
    assert len(appended) == steps["workload"]["appends"] == 20
    assert sum(item["lines"] for item in appended) == steps["final"]["appended_lines"]
    assert sum(item["bytes"] for item in appended) == steps["final"]["appended_bytes"]
    # Every transcript in the run holds exactly its seed plus its appends.
    seeded = int(POOL_LINES * 0.7)
    files = sorted((run / "pool/claude_code").rglob("*.jsonl"))
    assert len(files) == 2 * POOL_FILES
    lines = sum(len(path.read_bytes().splitlines()) for path in files)
    assert lines == 2 * POOL_FILES * seeded + steps["final"]["appended_lines"]
    # The daemon's final counts equal an independent index of the same transcripts.
    final = steps["final"]["stats"]
    expected = independent_counts(run, tmp_path / "independent")
    assert (final["sessions"], final["messages"]) == (expected["sessions"], expected["messages"])
    assert steps["verify"] == {"step": "verify", "sessions": 2 * POOL_FILES, "mismatched": 0}
    floors = steps["final"]["floors"]
    assert floors["freshness_seconds"]["count"] == 20
    assert steps["committed"]["freshness_seconds"] == floors["freshness_seconds"]
    assert floors["writer_seconds_max"] <= 5
    assert (
        "per-append freshness at the append schedule (profile append)"
        not in result["scope"]["missing"]
    )
    assert_no_service_reach(lane, result, run)
    assert_every_group_gone(output)
    assert_home_untouched(lane)


def schedule(finished: Finished) -> list[tuple[Path, int, int, int]]:
    appended = (finished.output / "appends.jsonl").read_text().splitlines()
    return [
        (
            Path(item["path"]).relative_to(finished.run),
            item["lines"],
            item["bytes"],
            item["end_size"],
        )
        for item in map(json.loads, appended)
    ]


@pytest.mark.parametrize("lines_total", [20, 21, 200, 1000])
def test_every_line_total_of_at_least_one_line_per_write_is_scheduled_exactly(
    lines_total: int,
) -> None:
    workload = Workload(
        copies=2, seed_fraction=0.7, copy_seed=93, seed=59, appenders=2, writes=10,
        interval_seconds=0.1, lines_total=lines_total,
    )  # fmt: skip

    chunks = workload.chunks()

    assert (len(chunks), sum(chunks), min(chunks) >= 1) == (20, lines_total, True)


def test_the_same_seeds_append_the_same_bytes(
    lane: Lane, template: Path, append_run: Finished
) -> None:
    """A second run of one workload writes the identical schedule into its own run."""
    output, run = lane.new("o-repeat"), lane.new("r-repeat")
    status, result = bench(lane, "daemon-run", *run_args(template, output, run, "append"))

    assert (status, result["outcome"]) == (0, "passed"), result
    repeated = schedule(Finished(output, run, status, result))
    assert len(repeated) == 20
    assert repeated == schedule(append_run)
    files = {path.relative_to(run): path.read_bytes() for path in (run / "pool").rglob("*.jsonl")}
    assert files == {
        path.relative_to(append_run.run): path.read_bytes()
        for path in (append_run.run / "pool").rglob("*.jsonl")
    }


def test_scope_missing_names_every_class_the_profile_does_not_cover(
    lane: Lane, template: Path
) -> None:
    output, run = lane.new("o-cohort"), lane.new("r-cohort")
    status, result = bench(
        lane,
        "daemon-run",
        *run_args(template, output, run, "cohort"),
        "--writes", "3", "--status-callers", "2", "--live-callers", "2",
    )  # fmt: skip

    # Whether every call met 2 s depends on the shared host's load; the run must
    # otherwise complete, and a timing breach is the only failure allowed here.
    if status != 0:
        assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", "floor")
        assert result["error"]["message"] == "a live or status call took longer than 2 s", result
        assert result["error"]["failed_calls"] == []
    phases = [step["phase"] for step in result["steps"] if step["step"] == "phase"]
    assert phases == [
        f"{kind}-{page}" for page in ("default", "252", "256") for kind in ("loaded", "quiet")
    ]
    missing = result["scope"]["missing"]
    assert "status/live under concurrent load (profile cohort)" not in missing
    for uncovered in (
        "per-append freshness at the append schedule (profile append)",
        "embed backlog drain and capacity (profile backlog)",
        "Path B acceptance (manual)",
        "installed-host qualification (rollout)",
        "populated scale, MLX and CUDA (manual)",
    ):
        assert uncovered in missing
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
    pages = {tuple(call["argv"]) for call in calls if call["argv"][0] == "live"}
    assert ("live", "--json", "--fresh") in pages
    assert ("live", "--json", "--fresh", "--limit", "256") in pages
    assert calls and all(call["status"] == 0 and call["error"] is None for call in calls)
    assert_every_group_gone(output)


SLOW_WRITER = """
import time

from recall.core.telemetry import Span, Stage
from recall.services import rpc_server


class SlowWriterSpan(Span):
    held = False

    def __exit__(self, kind, error, trace):
        if self.stage == Stage.WRITER_RUN and not SlowWriterSpan.held:
            SlowWriterSpan.held = True
            time.sleep(5.5)
        super().__exit__(kind, error, trace)


rpc_server.Span = SlowWriterSpan
"""


def slow_writer_build(directory: Path) -> Path:
    """This checkout's recall in a virtualenv whose first writer turn takes 5.5 s."""
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(directory / ".venv")],
        check=True,
        timeout=120,
    )
    version_dir = next((directory / ".venv/lib").iterdir()).name
    links = {"python": os.path.realpath(sys.executable), "python3": "python", version_dir: "python"}
    for alias, target in links.items():
        link = directory / ".venv/bin" / alias
        link.unlink(missing_ok=True)
        link.symlink_to(target)
    site = directory / ".venv/lib" / version_dir / "site-packages"
    real_site = next(path for path in sys.path if path.endswith("site-packages"))
    (site / "a_checkout.pth").write_text(f"import site; site.addsitedir({real_site!r})\n")
    (site / "slow_writer.py").write_text(SLOW_WRITER)
    (site / "b_slow_writer.pth").write_text("import slow_writer\n")
    recall = directory / ".venv/bin/recall"
    recall.write_text(f"#!{directory / '.venv/bin/python'}\nfrom recall.cli import main\nmain()\n")
    recall.chmod(0o755)
    return directory


def test_a_writer_turn_over_five_seconds_fails_the_run(slow_run: Finished) -> None:
    status, result = slow_run.status, slow_run.result

    assert (status, result["outcome"]) == (1, "failed"), result
    assert result["error"]["kind"] == "floor"
    assert "a writer or checkpoint turn took longer than 5 s" in result["error"]["message"]
    floors = result["error"]["floors"]
    assert 5.5 <= floors["writer_seconds_max"] < 30
    # Everything else about the run completed: only the floor failed it.
    steps = {step["step"] for step in result["steps"]}
    assert {"workload", "drain", "verify", "final"} <= steps
    assert_every_group_gone(slow_run.output)


def test_summarize_lists_every_run_and_keeps_failures_out_of_the_statistics(
    lane: Lane, append_run: Finished, slow_run: Finished
) -> None:
    output = lane.new("summary")
    status, result = bench(
        lane, "summarize", "--output", output, append_run.output, slow_run.output
    )

    assert (status, result["outcome"]) == (0, "passed"), result
    summary = json.loads((output / "summary.json").read_text())
    runs = {run["label"]: run for run in summary["runs"]}
    assert {name: run["outcome"] for name, run in runs.items()} == {
        "o-append": "passed",
        "o-slow": "failed",
    }
    assert runs["o-slow"]["writer_max_seconds"] >= 5.5
    assert runs["o-slow"]["error"].startswith("floor: ")
    final = next(step for step in append_run.result["steps"] if step["step"] == "final")
    assert (runs["o-append"]["appended_bytes"], runs["o-append"]["appended_lines"]) == (
        final["appended_bytes"],
        final["appended_lines"],
    )
    # Different builds group apart; each group's statistics cover only its passed runs.
    groups = {(group["build"], group["passed"], group["runs"]) for group in summary["groups"]}
    assert len(groups) == 2 and {(passed, runs) for _build, passed, runs in groups} == {
        (1, 1),
        (0, 1),
    }
    slow_group = next(group for group in summary["groups"] if group["passed"] == 0)
    assert slow_group["writer_max_seconds"] is None
    table = (output / "summary.md").read_text()
    assert "| o-append | append | passed |" in table and "| o-slow | append | failed |" in table


@pytest.mark.parametrize(
    ("profile", "extra", "kind", "named"),
    [
        ("idle", ("--appenders", "2"), "usage", "--appenders"),
        ("append", ("--writes", "10", "--append-lines", "5"), "usage", "--append-lines"),
        # More lines than the pool's transcripts hold.
        ("append", ("--writes", "10", "--append-lines", "1000000"), "refused", "pool"),
    ],
)
def test_a_workload_the_arguments_cannot_make_is_refused_before_anything_is_created(
    lane: Lane, template: Path, profile: str, extra: tuple[str, ...], kind: str, named: str
) -> None:
    label = f"{profile}-{kind}"
    output, run = lane.new(f"o-usage-{label}"), lane.new(f"r-usage-{label}")
    status, result = bench(
        lane, "daemon-run", "--template", template, "--build", ROOT, "--output", output,
        "--run-dir", run, "--profile", profile, *extra,
    )  # fmt: skip

    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", kind), result
    assert named in json.dumps(result["error"])
    assert not output.exists() and not run.exists()


def test_summarize_lists_a_refusal_from_its_outcome_line_beside_completed_runs(
    lane: Lane, template: Path, append_run: Finished
) -> None:
    output, run = lane.new("o-pool"), lane.new("r-pool")
    status, refused = bench(
        lane, "daemon-run", *run_args(template, output, run, "append"),
        "--append-lines", "1000000",
    )  # fmt: skip
    assert (status, refused["error"]["kind"]) == (1, "refused"), refused
    # A refusal before the run publishes only its outcome line, saved here as a user would.
    line = lane.new("pool-refused.json")
    line.write_text(json.dumps(refused) + "\n")

    summary = lane.new("summary-refused")
    status, result = bench(lane, "summarize", "--output", summary, line, append_run.output)

    assert (status, result["outcome"]) == (0, "passed"), result
    rows = {row["label"]: row for row in json.loads((summary / "summary.json").read_text())["runs"]}
    assert {label: row["outcome"] for label, row in rows.items()} == {
        "pool-refused": "failed",
        "o-append": "passed",
    }
    error = rows["pool-refused"]["error"]
    assert error.startswith("refused: ") and refused["error"]["reason"] in error


@pytest.fixture(autouse=True)
def _no_leftover_children() -> Iterator[None]:
    """A test leaves none of the processes it started; other modules' helpers are theirs."""
    before = {child.pid for child in psutil.Process().children()}
    yield
    assert [child for child in psutil.Process().children() if child.pid not in before] == []


def test_a_call_ends_when_its_process_exits_not_when_the_loop_notices() -> None:
    """A busy loop that sees an exit late must not lengthen the call it measures."""
    watch = ExitWatch()
    try:
        child = subprocess.Popen(["/bin/sleep", "0.3"], start_new_session=True)
        started = time.monotonic()
        watch.watch(child.pid)
        time.sleep(1.5)  # the loop is busy elsewhere
        ended = watch.exited_at(child.pid)
        # The watcher only observed the exit: the child is still ours to reap.
        assert os.waitpid(child.pid, os.WNOHANG)[0] == child.pid
        child.returncode = 0
    finally:
        watch.close()
    assert 0.25 <= ended - started < 1.0, ended - started
