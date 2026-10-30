"""`daemon-run`: one owned scratch daemon on a fresh clone of a template, measured by profile.

Every profile starts the build's daemon under the guarded launcher, waits for
readiness from the owned PID and settles (catalog complete, nothing pending,
no new failures). Then:

- `idle` samples the daemon's CPU and memory over the window.
- `append` seeds copies of the template's pool before the daemon starts,
  makes the deterministic append schedule (`workload.py`) while search and show
  observe each append's record, drains, confirms every appended
  transcript through `show`, and reads every writer and checkpoint turn from
  the daemon's diagnostics.
- `cohort` runs the responsive cohort qualification: live and status callers
  while appenders write, for default, 252-row and 256-row pages, each followed
  by a quiet control of the same length, then a complete drain.

Each takes the final status and stats. `--hold` keeps the daemon up
afterwards for a guest until it creates the `done` file or the hold times out.

`passed` means only that this profile's measurements completed and met the
floors it evaluates; `scope.missing` names what it does not cover.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import psutil

from . import launch
from .catalog import CatalogNotReady, CatalogWatch, Complete
from .cli import Command, run_cli
from .evidence import (
    BUILD_PATHS,
    JobFailed,
    JobIncomplete,
    Result,
    checkout_identity,
    tool_identity,
)
from .fs import Cancellation, NewTree
from .isolation import (
    BIN_DIR,
    CANARY_DIR,
    PAYLOAD_DIR,
    POOL_DIR,
    SOURCES,
    SYSTEM_PATH,
    Build,
    Policy,
    RunDir,
    Template,
    UsageLog,
)
from .load import (
    Call,
    Driver,
    Freshness,
    once,
    percentile,
    repeat,
    rss_bytes_max,
    seconds_summary,
)
from .samplers import Sample, host_load, sample, summarize
from .supervisor import Child, Completed, Deadline, Stopped, Supervisor
from .turns import TURN_STAGES, logged_turns, read_samples, summarize_logged
from .turns import summarize as summarize_turns
from .visibility import Visibility, Witness
from .visibility import prepare as prepare_visibility
from .workload import COPIES_MAX, WRITES_MAX, Appender, Workload, check_schedule, seed

Progress = Callable[[dict[str, object]], None]

PROFILES = ("idle", "append", "cohort")
CLI_TIMEOUT_SECONDS = 60.0
LAUNCHER_TIMEOUT_SECONDS = 120.0
POLL_SECONDS = 1.0
SAMPLE_SECONDS = 1.0
WINDOW_SECONDS_MAX = 24 * 3600
# BRIEF.md floors the profiles evaluate.
STATUS_SECONDS_MAX = 2.0
RSS_BYTES_MAX = 4 * 1024**3
WRITER_SECONDS_MAX = 5.0
FRESHNESS_SECONDS_P95_MAX = 15.0
FRESHNESS_SECONDS_MAX = 30.0
FRESHNESS_APPENDS_MIN = 20
# Appenders, writes, interval (ms) and exact line total: the `experiments/checkpoint_bytes`
# reconstruction of the io-59 workload (append) and RECONCILIATION_QA.md's responsive
# cohort (cohort, writes per phase).
WORKLOAD_DEFAULTS: dict[str, tuple[int, int, float, int | None]] = {
    "append": (1, 1100, 120_000 / 1100, 24_027),
    "cohort": (12, 40, 250.0, None),
}
COPIES = 4
SEED_FRACTION = 0.7
COPY_SEED = 93
SCHEDULE_SEED = 59
COHORT_CALLERS = 20
# Every option of the workload group: each defaults to None, so a given one is visible.
WORKLOAD_OPTIONS = (
    "appenders",
    "writes",
    "interval_ms",
    "append_lines",
    "seed",
    "copy_seed",
    "copies",
    "status_callers",
    "live_callers",
)
OBSERVERS = 2
LIVE_PAGE_MAX = 256
# Default fresh pages, 252-row maximum pages and full 256-row pages.
COHORT_PAGES: tuple[tuple[str, int | None], ...] = (("default", None), ("252", 252), ("256", 256))
SHOW_CALLERS = 8
DIAGNOSTICS_WAIT_SECONDS = 30.0
# Counts that must not move while nothing is appended.
STABLE_COUNTS = ("sessions", "messages", "tool_calls", "bash_calls")
DONE_NAME = "done"
GUEST_ENV_NAME = "guest.env"
RESULT_NAME = "result.json"
MEASUREMENTS = {
    "idle": ("readiness", "catalog completion", "idle cpu", "idle rss"),
    "append": (
        "readiness",
        "catalog completion",
        "per-append freshness (append to search and show visibility)",
        "writer and checkpoint turns",
        "status/live call latency",
        "rss",
    ),
    "cohort": (
        "readiness",
        "catalog completion",
        "live and status call latency under appends and quiet controls",
        "fresh pages: default, 252 and 256 rows",
        "writer and checkpoint turns",
        "complete drain",
        "rss",
    ),
}
APPEND_CLASS = "per-append freshness at the append schedule (profile append)"
COHORT_CLASS = "status/live under concurrent load (profile cohort)"
MISSING_CLASSES = (
    APPEND_CLASS,
    COHORT_CLASS,
    "embed backlog drain and capacity (profile backlog)",
    "enrichment and vector/hybrid search under load (run configs disable embedding)",
    "Path B acceptance (manual)",
    "service-slot fairness on a populated host (manual)",
    "the ten-task operator charter (manual)",
    "Path A migration runtime and resume (manual)",
    "installed-host qualification (rollout)",
    "populated scale, MLX and CUDA (manual)",
)
GUEST_RULES = """\
# Rules for this hold (verbatim from the campaign's common addenda 4, 5 and 18):
# - Signal only the PIDs or process groups you started, recorded at launch and
#   re-checked before signalling. Never use `pkill`, `killall`,
#   `pgrep | xargs kill`, or any signal aimed at a process-table search.
# - Never run these in a scratch environment: `recall daemon stop`,
#   `recall daemon restart`, `recall daemon install`, or
#   `recall daemon uninstall`, by you or by a guest. Doing so can stop the
#   human's production daemon.
# - Stop scratch daemons by owned PID or process group only.
# - Stop scratch daemons only with `recall daemon stop --soft` or by signalling
#   the PID you recorded; never `daemon stop|restart|install|uninstall` or a
#   non-`--background` `daemon start`.
"""


def _positive(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _count(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def workload_from(args: argparse.Namespace) -> Workload:
    """The profile's workload, its defaults replaced by whatever was given."""
    appenders, writes, interval_ms, lines = WORKLOAD_DEFAULTS[args.profile]
    # The default line total belongs to the default shape.
    shaped = args.appenders is not None or args.writes is not None
    appenders = args.appenders or appenders
    writes = (args.writes or writes) * (len(COHORT_PAGES) if args.profile == "cohort" else 1)
    copies = args.copies or COPIES
    lines_total = args.append_lines or (None if shaped else lines)
    if copies > COPIES_MAX:
        raise JobFailed("usage", f"--copies may be at most {COPIES_MAX}")
    if appenders * writes > WRITES_MAX:
        raise JobFailed("usage", f"--appenders times the writes may be at most {WRITES_MAX}")
    if lines_total is not None and lines_total < appenders * writes:
        raise JobFailed("usage", "--append-lines must give every write at least one line")
    workload = Workload(
        copies=copies,
        seed_fraction=SEED_FRACTION,
        copy_seed=COPY_SEED if args.copy_seed is None else args.copy_seed,
        seed=SCHEDULE_SEED if args.seed is None else args.seed,
        appenders=appenders,
        writes=writes,
        interval_seconds=(args.interval_ms or interval_ms) / 1000,
        lines_total=lines_total,
    )
    # The line counts depend only on the arguments; refuse an impossible total now.
    workload.chunks()
    return workload


def cohort_callers(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """The cohort's live and status caller names."""
    live = args.live_callers or COHORT_CALLERS
    status = args.status_callers or COHORT_CALLERS
    return [f"live-{index}" for index in range(live)], [
        f"status-{index}" for index in range(status)
    ]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--template", required=True, help="template made by `prepare`")
    parser.add_argument("--build", required=True, help="checkout whose .venv runs recall")
    parser.add_argument("--output", required=True, help="new evidence directory")
    parser.add_argument(
        "--run-dir", required=True, help="new, short directory: the daemon's data dir"
    )
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--window", type=_positive, default=60.0, help="idle sampling seconds")
    workload = parser.add_argument_group("append and cohort workload")
    workload.add_argument("--appenders", type=_count, help="concurrent appenders")
    workload.add_argument("--writes", type=_count, help="writes per appender (cohort: per phase)")
    workload.add_argument("--interval-ms", type=_positive, help="milliseconds between ticks")
    workload.add_argument("--append-lines", type=_count, help="exact total of appended lines")
    workload.add_argument("--seed", type=int, help=f"schedule seed (default {SCHEDULE_SEED})")
    workload.add_argument("--copy-seed", type=int, help=f"UUID renaming seed (default {COPY_SEED})")
    workload.add_argument("--copies", type=_count, help=f"copies of the pool (default {COPIES})")
    workload.add_argument(
        "--status-callers", type=_count, help=f"cohort status callers (default {COHORT_CALLERS})"
    )
    workload.add_argument(
        "--live-callers", type=_count, help=f"cohort live callers (default {COHORT_CALLERS})"
    )
    parser.add_argument(
        "--settle-timeout", type=_positive, default=900.0, help="readiness and settle seconds"
    )
    parser.add_argument("--deadline", type=_positive, default=7200.0, help="whole-run seconds")
    parser.add_argument(
        "--usage-log",
        choices=[choice.value for choice in UsageLog],
        default=UsageLog.EMPTY.value,
        help="Grok usage log the daemon harvests: an empty file or the template's copy",
    )
    parser.add_argument("--hold", action="store_true", help="keep the daemon up for a guest")
    parser.add_argument("--hold-timeout", type=_positive, default=3600.0)


def run(
    args: argparse.Namespace, result: Result, cancellation: Cancellation, progress: Progress
) -> None:
    if args.window > WINDOW_SECONDS_MAX:
        raise JobFailed("usage", f"--window may be at most {WINDOW_SECONDS_MAX} s")
    given = [name for name in WORKLOAD_OPTIONS if getattr(args, name) is not None]
    if args.profile == "idle" and given:
        options = ", ".join("--" + name.replace("_", "-") for name in given)
        raise JobFailed("usage", f"{options}: workload options apply only to append and cohort")
    # Before anything is created: a workload the arguments cannot make is refused here.
    workload = workload_from(args) if args.profile != "idle" else None
    deadline = Deadline(args.deadline)
    # Hashing and cloning a large template observe the deadline at every chunk.
    cancellation.expire_at(deadline.at)
    policy = Policy.for_host()
    with ExitStack() as held:
        roots = held.enter_context(policy.open_write_roots())
        result.leftovers = roots.leftovers
        template = held.enter_context(
            Template.open(policy, "--template", args.template, cancellation)
        )
        if workload is not None:
            check_schedule(template.root, workload)
        build = Build.resolve(policy, "--build", args.build)
        output_target = held.enter_context(roots.new_directory("--output", args.output))
        result.step("resolve", template=str(template.path), build=str(build.path))
        run_dir = RunDir.create(
            roots,
            "--run-dir",
            args.run_dir,
            template,
            build,
            UsageLog(args.usage_log),
            cancellation,
        )
        result.step("clone", run_dir=str(run_dir.path), files=len(template.manifest.files))
        covered = {"idle": (), "append": (APPEND_CLASS,), "cohort": (COHORT_CLASS,)}[args.profile]
        result.scope = {
            "profile": args.profile,
            "measurements": list(MEASUREMENTS[args.profile]),
            "missing": [name for name in MISSING_CLASSES if name not in covered],
            "template": str(template.path),
            "fixture": _fixture(template),
            "run_dir": str(run_dir.path),
            "usage_log": str(run_dir.usage_log_path),
            "build": {"path": str(build.path), **checkout_identity(build.path, BUILD_PATHS)},
            "tool": tool_identity(),
            "settings": {
                "window_seconds": args.window,
                "workload": asdict(workload) if workload is not None else None,
                "settle_timeout_seconds": args.settle_timeout,
                "deadline_seconds": args.deadline,
                "hold": args.hold,
                "config": run_dir.config_text(),
            },
            "host_load_start": host_load(),
        }
        tree = held.enter_context(output_target.create())
        # Evidence stays whatever the outcome.
        tree.commit(cancellation)
        try:
            _supervised(args, result, cancellation, progress, deadline, build, run_dir, tree)
        finally:
            # The published result is the outcome line `bench.py` prints.
            result.scope["host_load_end"] = host_load()
            result.end(sys.exc_info()[1])
            tree.publish(PurePosixPath(RESULT_NAME), result.to_json())


def _supervised(
    args: argparse.Namespace,
    result: Result,
    cancellation: Cancellation,
    progress: Progress,
    deadline: Deadline,
    build: Build,
    run_dir: RunDir,
    tree: NewTree,
) -> None:
    ended: BaseException | None = None
    with Supervisor(tree, run_dir, build, cancellation, progress) as supervisor:
        try:
            profile = Profile(args, result, progress, deadline, build, run_dir, tree, supervisor)
            {"idle": profile.idle, "append": profile.append, "cohort": profile.cohort}[
                args.profile
            ]()
        except BaseException as error:
            ended = error
            raise
        finally:
            stopped = supervisor.stop_all()
            result.step("stop", processes=[asdict(item) for item in stopped])
            safety = _safety(result, run_dir, stopped)
            if safety is not None and not isinstance(ended, JobFailed):
                raise safety from ended


def _safety(result: Result, run_dir: RunDir, stopped: list[Stopped]) -> JobFailed | None:
    """Record what reached for a service manager, and fail on anything that got through."""
    attempts = _attempts(run_dir.attempts_path)
    canary = (
        run_dir.canary_log_path.read_text().splitlines() if run_dir.canary_log_path.exists() else []
    )
    # An exec the audit hook stopped; a refused CLI call is the allow-list working.
    refused = [item for item in attempts if item.get("kind") == "refused"]
    refused_calls = [item for item in attempts if item.get("kind") == "refused_call"]
    refused_signals = [item for item in attempts if item.get("kind") == "refused_signal"]
    suppressed: dict[str, int] = {}
    for item in attempts:
        if item.get("kind") == "suppressed":
            seam = str(item.get("seam"))
            suppressed[seam] = suppressed.get(seam, 0) + 1
    survivors = [item.name for item in stopped if not item.group_gone]
    result.scope["service_reach"] = {
        "canary_invocations": canary,
        "refused": refused,
        "refused_calls": [item.get("detail") for item in refused_calls],
        "refused_signals": [item.get("detail") for item in refused_signals],
        "suppressed": suppressed,
    }
    if canary:
        return JobFailed("service_manager_invoked", "a service-manager canary ran", lines=canary)
    if refused:
        return JobFailed("service_reach_refused", "a child tried to reach a service manager")
    if refused_signals:
        return JobFailed("signal_refused", "a child tried to signal a process it does not own")
    if survivors:
        return JobFailed("survivors", "an owned process group outlived cleanup", names=survivors)
    return None


def _attempts(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Profile:
    """The steps of one run, in order; every child goes through the supervisor."""

    def __init__(
        self,
        args: argparse.Namespace,
        result: Result,
        progress: Progress,
        deadline: Deadline,
        build: Build,
        run_dir: RunDir,
        tree: NewTree,
        supervisor: Supervisor,
    ) -> None:
        self.args = args
        self.result = result
        self.progress = progress
        self.deadline = deadline
        self.build = build
        self.run_dir = run_dir
        self.tree = tree
        self.supervisor = supervisor
        self.status_seconds: list[float] = []
        self.witnesses: dict[int, Witness] = {}

    def phase(self, name: str) -> None:
        self.progress({"event": "phase", "phase": name})

    def idle(self) -> None:
        daemon, settled_stats = self.boot()
        self.phase("window")
        window = self.window(daemon.process, daemon.exited)
        self.result.step("window", **window)
        final_status = self.cli_json(Command.daemon_status())
        final_stats = self.cli_json(Command.stats())
        self.tree.write_new(self.tree.root, PurePosixPath("status-final.json"), _dump(final_status))
        moved = {
            name: (settled_stats.get(name), final_stats.get(name))
            for name in STABLE_COUNTS
            if settled_stats.get(name) != final_stats.get(name)
        }
        floors = {
            "status_seconds_max": max(self.status_seconds),
            "status_seconds_limit": STATUS_SECONDS_MAX,
            "rss_bytes_max": window["rss_bytes_max"],
            "rss_bytes_limit": RSS_BYTES_MAX,
        }
        self.result.step("final", stats=final_stats, floors=floors)
        if moved:
            raise JobFailed("counts_moved", "counts changed while idle", moved=moved)
        if max(self.status_seconds) > STATUS_SECONDS_MAX:
            raise JobFailed("floor", "a status call took longer than 2 s", floors=floors)
        if int(str(window["rss_bytes_max"])) > RSS_BYTES_MAX:
            raise JobFailed("floor", "daemon RSS exceeded 4 GiB", floors=floors)
        self.hold_if_asked(daemon)

    def boot(self) -> tuple[Child, dict[str, object]]:
        """Start the owned daemon, wait for readiness and settle; return it and the stats."""
        self.phase("check")
        self.check()
        self.phase("preflight")
        self.preflight()
        self.phase("start")
        daemon = self.supervisor.spawn("daemon", self.run_dir.launcher_argv(self.build, "daemon"))
        started = time.monotonic()
        self.phase("readiness")
        status = self.readiness(daemon.pid, daemon.exited, started)
        self.result.step(
            "readiness",
            seconds=time.monotonic() - started,
            daemon_pid=daemon.pid,
            installed_scheduler=_runtime(status).get("installed_scheduler"),
        )
        self.phase("settle")
        settled, complete = self.settle(status, daemon.exited)
        settled_stats = self.cli_json(Command.stats())
        self.result.step(
            "settle",
            seconds=time.monotonic() - started,
            catalog_seconds=complete.seconds,
            recovered=[asdict(item) for item in complete.recovered],
            pending=_reconciliation(settled).get("pending"),
            stats=settled_stats,
        )
        return daemon, settled_stats

    def hold_if_asked(self, daemon: Child) -> None:
        if self.args.hold:
            self.phase("hold")
            self.hold(daemon.pid, daemon.exited)

    def seeded(self, workload: Workload) -> Appender:
        """The run's pool replaced by the workload's seeded copies, before the daemon starts."""
        self.phase("seed")
        with self.run_dir.held() as root:
            appender = seed(root, workload)
            if self.args.profile == "append":
                try:
                    self.witnesses = prepare_visibility(
                        appender, self.run_dir.path, self.supervisor.check
                    )
                except BaseException:
                    appender.close()
                    raise
        self.result.step(
            "seed",
            sessions=len(appender.sessions),
            seed_bytes=sum(session.seed_size for session in appender.sessions),
            writes=len(appender.writes),
            append_bytes=appender.bytes_total,
            append_lines=appender.lines_total,
        )
        return appender

    def append(self) -> None:
        workload = workload_from(self.args)
        appender = self.seeded(workload)
        try:
            daemon, settled_stats = self.boot()
            driver = self.driver(daemon)
            try:
                freshness = Visibility(driver.appended, self.witnesses, driver.record_visibility)
                observers = {
                    f"observer-{number}": repeat(Command.live(LIVE_PAGE_MAX))
                    for number in range(OBSERVERS)
                } | freshness.callers()
                self.phase("workload")
                started = time.monotonic()
                driver.drive(
                    "workload",
                    observers,
                    appender=appender,
                    ticks=range(workload.writes),
                    interval_seconds=workload.interval_seconds,
                    timeout_seconds=workload.writes * workload.interval_seconds
                    + self.args.settle_timeout,
                    observe=freshness.observe,
                )
                self.result.step(
                    "workload",
                    seconds=time.monotonic() - started,
                    appends=len(driver.appended),
                    appended_bytes=sum(item.write.byte_count for item in driver.appended),
                    appended_lines=sum(item.write.line_count for item in driver.appended),
                )
                self.drained(driver, freshness, observers, daemon, started)
                self.finish(driver, freshness, appender, daemon, settled_stats, started)
            finally:
                driver.close()
        finally:
            appender.close()
        self.hold_if_asked(daemon)

    def cohort(self) -> None:
        workload = workload_from(self.args)
        per_phase = workload.writes // len(COHORT_PAGES)
        appender = self.seeded(workload)
        try:
            daemon, settled_stats = self.boot()
            driver = self.driver(daemon)
            try:
                freshness = Freshness(driver.appended)
                started = time.monotonic()
                for number, (label, limit) in enumerate(COHORT_PAGES):
                    live, status = cohort_callers(self.args)
                    callers = {
                        **{name: repeat(Command.live_fresh(limit)) for name in live},
                        **{name: repeat(Command.daemon_status(limit)) for name in status},
                    }
                    for loaded in (True, False):
                        phase = f"{'loaded' if loaded else 'quiet'}-{label}"
                        self.phase(phase)
                        began = time.monotonic()
                        quiet_until = began + per_phase * workload.interval_seconds
                        calls = driver.drive(
                            phase,
                            callers,
                            appender=appender if loaded else None,
                            ticks=range(number * per_phase, (number + 1) * per_phase)
                            if loaded
                            else range(0),
                            interval_seconds=workload.interval_seconds,
                            until=None if loaded else (lambda now, at=quiet_until: now >= at),
                            timeout_seconds=per_phase * workload.interval_seconds
                            + self.args.settle_timeout,
                            observe=freshness.observe,
                        )
                        self.result.step(
                            "phase",
                            phase=phase,
                            seconds=time.monotonic() - began,
                            appends=sum(
                                item.write.tick // per_phase == number for item in driver.appended
                            )
                            if loaded
                            else 0,
                            calls=_calls_summary(calls),
                        )
                observers = {
                    f"observer-{number}": repeat(Command.live(LIVE_PAGE_MAX))
                    for number in range(OBSERVERS)
                }
                self.drained(driver, freshness, observers, daemon, started)
                self.finish(driver, freshness, appender, daemon, settled_stats, started)
            finally:
                driver.close()
        finally:
            appender.close()
        self.hold_if_asked(daemon)

    def driver(self, daemon: Child) -> Driver:
        def validate(command: Command, document: Mapping[str, object]) -> str | None:
            if command.argv[:2] == ("daemon", "status"):
                if document.get("daemon_pid") != daemon.pid:
                    return "status was not answered by the owned daemon"
            elif command.argv[0] == "live" and document.get("watching") is not True:
                return "live was answered without the daemon's live watcher"
            return None

        return Driver(
            self.supervisor, self.build, self.run_dir, self.tree, self.deadline, daemon, validate
        )

    def drained(
        self,
        driver: Driver,
        freshness: Freshness,
        observers: Mapping[str, Callable[[], Command | None]],
        daemon: Child,
        started: float,
    ) -> None:
        """Observe until every append is committed, then settle again: a complete drain."""
        self.phase("drain")
        began = time.monotonic()
        driver.drive(
            "drain",
            observers,
            until=lambda _now: not freshness.unseen(),
            timeout_seconds=self.args.settle_timeout,
            observe=freshness.observe,
        )
        # Recorded before the settle, so a run that fails from here still reports it.
        self.result.step(
            "committed",
            seconds=time.monotonic() - began,
            **(
                {
                    "freshness_seconds": seconds_summary(freshness.seconds()),
                    "committed_offset_seconds": seconds_summary(freshness.offsets.seconds()),
                }
                if isinstance(freshness, Visibility)
                else {"committed_offset_seconds": seconds_summary(freshness.seconds())}
            ),
            rss_bytes_max=rss_bytes_max(driver.samples),
        )
        status = self.cli_json(Command.daemon_status())
        settled, _complete = self.settle(status, daemon.exited)
        self.result.step(
            "drain",
            seconds=time.monotonic() - began,
            since_workload_start=time.monotonic() - started,
            pending=_reconciliation(settled).get("pending"),
        )

    def finish(
        self,
        driver: Driver,
        freshness: Freshness,
        appender: Appender,
        daemon: Child,
        settled_stats: dict[str, object],
        started: float,
    ) -> None:
        """Confirm every appended transcript through `show`, read the turns, judge the floors."""
        self.phase("verify")
        appended_paths = sorted({item.path for item in driver.appended})
        missing = [path for path in appended_paths if path not in freshness.identifiers]
        if missing:
            raise JobFailed(
                "not_live", "appended transcripts never appeared live", paths=missing[:20]
            )
        expected = {
            str(self.run_dir.path / session.relative): session.size for session in appender.sessions
        }
        shows = driver.drive(
            "verify",
            dict(
                zip(
                    (f"show-{path}" for path in appended_paths),
                    once(
                        [
                            Command.show(freshness.identifiers[path], tail=1)
                            for path in appended_paths
                        ]
                    ),
                    strict=True,
                )
            ),
            timeout_seconds=self.args.settle_timeout,
        )
        shown: dict[str, object] = {}
        for call in shows:
            path = call.caller.removeprefix("show-")
            document = call.document or {}
            shown[path] = _freshness(document) if call.succeeded else call.error
        wrong = {
            path: {"expected": expected[path], "shown": shown.get(path)}
            for path in appended_paths
            if not _shows_current(shown.get(path), expected[path])
        }
        self.result.step("verify", sessions=len(appended_paths), mismatched=len(wrong))
        if wrong:
            raise JobFailed(
                "show_mismatch",
                "show disagrees with the appended bytes",
                sessions=dict(list(wrong.items())[:20]),
            )
        turns = self.turns(daemon, started)
        final_status = self.cli_json(Command.daemon_status())
        final_stats = self.cli_json(Command.stats())
        self.tree.write_new(self.tree.root, PurePosixPath("status-final.json"), _dump(final_status))
        self.tree.write_new(self.tree.root, PurePosixPath("turns.json"), _dump(turns))
        self.judge(driver, freshness, appender, turns, settled_stats, final_stats, final_status)

    def turns(self, daemon: Child, started: float) -> dict[str, object]:
        """Every writer and checkpoint turn, from a diagnostics sample written after the drain."""
        after = time.time()
        until = time.monotonic() + DIAGNOSTICS_WAIT_SECONDS
        while True:
            self.supervisor.check()
            with self.run_dir.held() as root:
                samples = read_samples(root, daemon.pid)
            if samples and samples[-1].timestamp >= after:
                logged = logged_turns(daemon.output()[1])
                self.tree.write_new(
                    self.tree.root,
                    PurePosixPath("writer-turns.jsonl"),
                    b"".join((json.dumps(asdict(turn)) + "\n").encode() for turn in logged),
                )
                return {
                    **summarize_turns(samples, window_start=started),
                    "log": summarize_logged(logged),
                }
            if time.monotonic() >= until:
                raise JobFailed(
                    "diagnostics", "the daemon wrote no diagnostics sample after the drain"
                )
            self.supervisor.wait_quietly(POLL_SECONDS, self.deadline)

    def judge(
        self,
        driver: Driver,
        freshness: Freshness,
        appender: Appender,
        turns: dict[str, object],
        settled_stats: dict[str, object],
        final_stats: dict[str, object],
        final_status: dict[str, object],
    ) -> None:
        """Record every floor, then fail on the first breached or uncovered one."""
        loaded = [
            call
            for call in driver.calls
            if call.phase.startswith(("workload", "loaded-", "quiet-", "drain"))
        ]
        responsive = [
            call
            for call in loaded
            if call.argv[0] in ("live",) or call.argv[:2] == ("daemon", "status")
        ]
        failed_calls = [call.to_json() for call in responsive if not call.succeeded]
        call_seconds = [call.seconds for call in responsive if call.succeeded] + self.status_seconds
        stages = turns["stages"]
        assert isinstance(stages, dict)
        log = turns["log"]
        assert isinstance(log, dict)
        writer_seconds = max(
            float(log["op_seconds_max"]),
            *(float(stages[stage]["seconds_max_lifetime"]) for stage in TURN_STAGES),
        )
        fresh = freshness.seconds()
        sessions_added = {
            name: (settled_stats.get(name), final_stats.get(name)) for name in STABLE_COUNTS
        }
        floors: dict[str, object] = {
            "writer_seconds_max": writer_seconds,
            "writer_seconds_limit": WRITER_SECONDS_MAX,
            "writer_turns_logged": log["turns"],
            "checkpoints_logged": log["checkpoints"],
            "checkpoint_seconds_max": log["checkpoint_seconds_max"],
            "call_seconds": seconds_summary(call_seconds),
            "call_seconds_limit": STATUS_SECONDS_MAX,
            "failed_calls": len(failed_calls),
            "rss_bytes_max": rss_bytes_max(driver.samples),
            "rss_bytes_limit": RSS_BYTES_MAX,
            **(
                {"freshness_seconds": seconds_summary(fresh)}
                if isinstance(freshness, Visibility)
                else {}
            ),
            "freshness_seconds_p95_limit": FRESHNESS_SECONDS_P95_MAX,
            "freshness_seconds_limit": FRESHNESS_SECONDS_MAX,
            "loop_seconds_max": driver.loop_seconds_max,
        }
        self.result.step(
            "final",
            stats=final_stats,
            counts=sessions_added,
            appended_bytes=appender.bytes_total,
            appended_lines=appender.lines_total,
            generation_errors=turns["generation_errors"],
            compaction=final_status.get("compaction"),
            floors=floors,
        )
        breaches: list[str] = []
        if final_stats.get("sessions") != settled_stats.get("sessions"):
            breaches.append("appends changed the session count")
        if turns["generation_errors"]:
            breaches.append("the daemon counted generation errors")
        if failed_calls:
            breaches.append("a live or status call failed")
        if call_seconds and max(call_seconds) > STATUS_SECONDS_MAX:
            breaches.append("a live or status call took longer than 2 s")
        if writer_seconds > WRITER_SECONDS_MAX:
            breaches.append("a writer or checkpoint turn took longer than 5 s")
        if int(str(floors["rss_bytes_max"])) > RSS_BYTES_MAX:
            breaches.append("daemon RSS exceeded 4 GiB")
        if self.args.profile == "append":
            if len(fresh) < FRESHNESS_APPENDS_MIN:
                breaches.append(f"fewer than {FRESHNESS_APPENDS_MIN} appends were observed")
            elif (
                max(fresh) > FRESHNESS_SECONDS_MAX
                or percentile(fresh, 0.95) > FRESHNESS_SECONDS_P95_MAX
            ):
                breaches.append("append freshness exceeded p95 15 s or max 30 s")
        else:
            skipped = self.skipped_callers(driver)
            if skipped:
                breaches.append("a caller completed no call in a phase")
                floors["skipped_callers"] = skipped
        if breaches:
            raise JobFailed(
                "floor", "; ".join(breaches), floors=floors, failed_calls=failed_calls[:20]
            )

    def skipped_callers(self, driver: Driver) -> list[str]:
        """Each `phase/caller` of the cohort that completed no successful call."""
        succeeded = {(call.phase, call.caller) for call in driver.calls if call.succeeded}
        live, status = cohort_callers(self.args)
        callers = live + status
        return [
            f"{kind}-{label}/{caller}"
            for label, _limit in COHORT_PAGES
            for kind in ("loaded", "quiet")
            for caller in callers
            if (f"{kind}-{label}", caller) not in succeeded
        ]

    def check(self) -> None:
        completed = self.launcher("check")
        if completed.status == launch.SEAM_MISSING:
            raise JobIncomplete(
                "seam_missing",
                "the build lacks a seam the launcher replaces",
                report=_last(completed),
            )
        if completed.status == launch.UNSUPPORTED_BUILD:
            raise JobFailed(
                "unsupported_build", "the build is older than the floor", report=_last(completed)
            )
        if completed.status != 0:
            raise JobFailed("check", "the launcher check failed", **_failure(completed))
        self.result.step("check", **_last(completed))

    def preflight(self) -> None:
        completed = self.launcher("preflight")
        if completed.status != 0:
            raise JobFailed(
                "preflight", "the build could not load the run config", **_failure(completed)
            )
        paths = _last(completed).get("paths")
        if not isinstance(paths, dict):
            raise JobFailed("preflight", "preflight reported no paths", **_failure(completed))
        root = self.run_dir.path
        expected_sources = {source: [str(root / POOL_DIR / source)] for source in SOURCES}
        outside = sorted(
            name
            for name in (
                "data_dir",
                "db_path",
                "lock_path",
                "config_path",
                "socket",
                "pid",
                "duckdb_temp_dir",
            )
            if not _within(paths.get(name), root)
        )
        if paths.get("data_dir") != str(root):
            outside.append("data_dir is not the run dir")
        if paths.get("cwd") != os.path.realpath(root):
            outside.append("the child did not start in the run dir")
        if paths.get("sources") != expected_sources:
            outside.append("sources")
        if paths.get("scheduler") != "auto":
            outside.append("scheduler")
        self.result.step("preflight", paths=paths)
        if outside:
            raise JobFailed("preflight", "a resolved path is outside the run dir", fields=outside)

    def launcher(self, mode: str) -> Completed:
        completed = self.supervisor.run(
            mode,
            self.run_dir.launcher_argv(self.build, mode),
            timeout_seconds=LAUNCHER_TIMEOUT_SECONDS,
            deadline=self.deadline,
        )
        if completed.timed_out:
            self.supervisor.check()
            raise JobFailed("timeout", f"launcher {mode} did not finish")
        return completed

    def cli(self, command: Command) -> Completed:
        completed = run_cli(
            self.supervisor,
            self.build,
            self.run_dir,
            command,
            timeout_seconds=CLI_TIMEOUT_SECONDS,
            deadline=self.deadline,
        )
        if completed.timed_out:
            self.supervisor.check()
        return completed

    def cli_json(self, command: Command) -> dict[str, object]:
        completed = self.cli(command)
        if command == Command.daemon_status() and completed.status == 0:
            self.status_seconds.append(completed.seconds)
        if completed.status != 0:
            raise JobFailed(
                "cli", f"`recall {' '.join(command.argv)}` failed", **_failure(completed)
            )
        return _document(completed)

    def readiness(self, pid: int, exited: Callable[[], bool], started: float) -> dict[str, object]:
        """Wait until status answers from the owned daemon itself."""
        while True:
            self.waiting(exited, started, "readiness")
            completed = self.cli(Command.daemon_status())
            if completed.status == 0:
                status = _document(completed)
                ready = _reconciliation(status).get("rpc_ready") is True
                if status.get("daemon_pid") == pid and ready:
                    return status
            self.supervisor.wait_quietly(POLL_SECONDS, self.deadline)

    def waiting(self, exited: Callable[[], bool], started: float, phase: str) -> None:
        if exited():
            raise JobFailed("daemon_exited", f"the daemon exited during {phase}")
        self.supervisor.check()
        if time.monotonic() - started > self.args.settle_timeout:
            raise JobIncomplete("settle_timeout", f"{phase} did not finish within --settle-timeout")

    def settle(
        self, status: dict[str, object], exited: Callable[[], bool]
    ) -> tuple[dict[str, object], Complete]:
        """Wait for the catalog-completion predicate, nothing pending and no new failure."""
        started = time.monotonic()
        catalog = CatalogWatch(clock=time.monotonic)
        failure_at = _runtime(status).get("last_failure_at")
        while True:
            self.waiting(exited, started, "settle")
            reconciliation = _reconciliation(status)
            if _runtime(status).get("last_failure_at") != failure_at:
                raise JobFailed(
                    "generation_error", "the daemon recorded a failure", runtime=_runtime(status)
                )
            errors = {
                name: reconciliation.get(name)
                for name in ("error", "enrichment_error", "keyword_repair_error")
                if reconciliation.get(name) is not None
            }
            if errors or status.get("watch_total_failed"):
                raise JobFailed(
                    "generation_error",
                    "the daemon reports errors",
                    errors=errors,
                    watch_total_failed=status.get("watch_total_failed"),
                )
            try:
                complete = catalog.observe(reconciliation)
            except CatalogNotReady as error:
                raise JobFailed("catalog", str(error)) from error
            drained = (
                reconciliation.get("pending") == 0
                and reconciliation.get("noticed_pending") == 0
                and reconciliation.get("enrichment_ready") is True
                and status.get("catchup_in_progress") is False
            )
            if complete is not None and drained:
                return status, complete
            self.supervisor.wait_quietly(POLL_SECONDS, self.deadline)
            status = self.cli_json(Command.daemon_status())

    def window(self, process: psutil.Process, exited: Callable[[], bool]) -> dict[str, object]:
        samples: list[Sample] = []
        descriptor = self.tree.open_new(self.tree.root, PurePosixPath("samples.jsonl"))
        try:
            until = time.monotonic() + self.args.window
            while True:
                if exited():
                    raise JobFailed("daemon_exited", "the daemon exited during the window")
                self.supervisor.check()
                item = sample(process)
                samples.append(item)
                _append(descriptor, item.to_json())
                if time.monotonic() >= until and len(samples) >= 2:
                    break
                self.supervisor.wait_quietly(SAMPLE_SECONDS, self.deadline)
        finally:
            os.close(descriptor)
        return summarize(samples)

    def hold(self, pid: int, exited: Callable[[], bool]) -> None:
        """Keep the daemon up until the guest creates `done`, the daemon stops, or the timeout."""
        done = self.tree.path / DONE_NAME
        environment = {
            "PATH": f"{self.run_dir.path / BIN_DIR}:{self.run_dir.path / CANARY_DIR}:{SYSTEM_PATH}",
            **self.run_dir.recall_paths(),
            "BENCH_DONE": str(done),
            "BENCH_DAEMON_PID": str(pid),
        }
        text = GUEST_RULES + (
            f"# The daemon (pid {pid}) belongs to the bench supervisor: do not signal it.\n"
            "# `recall daemon stop` is refused here, `--soft` included: it signals\n"
            "# whatever the PID file names. Ending the hold stops the daemon.\n"
            f"# End the hold with: touch {done}\n"
            "# `recall` on this PATH runs the build guarded; it refuses calls\n"
            "# outside the bench allow-list.\n"
        )
        text += "".join(
            f"export {name}={shlex.quote(value)}\n" for name, value in environment.items()
        )
        self.tree.write_new(self.tree.root, PurePosixPath(GUEST_ENV_NAME), text.encode())
        self.progress(
            {"event": "hold", "env": str(self.tree.path / GUEST_ENV_NAME), "done": str(done)}
        )
        until = time.monotonic() + self.args.hold_timeout
        while self.tree.root.lstat(DONE_NAME) is None:
            if exited():
                self.result.step("hold", ended="daemon stopped")
                return
            self.supervisor.check()
            if time.monotonic() >= until:
                self.result.step("hold", ended="timeout")
                raise JobIncomplete(
                    "hold_timeout", "the guest did not finish within --hold-timeout"
                )
            self.supervisor.wait_quietly(POLL_SECONDS, self.deadline)
        self.result.step("hold", ended="done")


def _within(value: object, root: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return path == root or root in path.parents


def _document(completed: Completed) -> dict[str, object]:
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise JobFailed("cli", "a recall call printed no JSON", **_failure(completed)) from error
    if not isinstance(document, dict):
        raise JobFailed("cli", "a recall call printed no JSON object", **_failure(completed))
    return document


def _last(completed: Completed) -> dict[str, object]:
    lines = completed.stdout.decode(errors="replace").strip().splitlines()
    try:
        document = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        return {"stdout": lines[-1][:2000]}
    return document if isinstance(document, dict) else {"stdout": lines[-1][:2000]}


def _failure(completed: Completed) -> dict[str, object]:
    # The CLI prints its error document on stdout and notes on stderr.
    return {
        "status": completed.status,
        "stdout": completed.stdout.decode(errors="replace")[-2000:],
        "stderr": completed.stderr.decode(errors="replace")[-2000:],
    }


def _runtime(status: Mapping[str, object]) -> dict[str, object]:
    runtime = status.get("runtime_status")
    return runtime if isinstance(runtime, dict) else {}


def _reconciliation(status: Mapping[str, object]) -> dict[str, object]:
    reconciliation = status.get("reconciliation")
    return reconciliation if isinstance(reconciliation, dict) else {}


def _dump(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def _append(descriptor: int, line: Mapping[str, object]) -> None:
    os.write(descriptor, (json.dumps(line, sort_keys=True) + "\n").encode())


def _calls_summary(calls: list[Call]) -> dict[str, object]:
    by_kind: dict[str, list[Call]] = {}
    for call in calls:
        by_kind.setdefault(
            " ".join(call.argv[:2]) if call.argv[0] == "daemon" else call.argv[0], []
        ).append(call)
    return {
        kind: {
            "failed": sum(not call.succeeded for call in items),
            "seconds": seconds_summary([call.seconds for call in items if call.succeeded]),
        }
        for kind, items in sorted(by_kind.items())
    }


def _freshness(document: Mapping[str, object]) -> object:
    return document.get("freshness")


def _shows_current(freshness: object, size: int) -> bool:
    """`show` reports the transcript committed exactly to its appended size."""
    return (
        isinstance(freshness, dict)
        and freshness.get("current") is True
        and freshness.get("indexed_size") == size
        and freshness.get("file_size") == size
    )


def _fixture(template: Template) -> dict[str, object]:
    """The template's identity: each payload file's SHA-256, and the pool's size."""
    pool = [record for record in template.manifest.files if record.path.startswith(POOL_DIR + "/")]
    return {
        "payload": {
            record.path: record.sha256
            for record in template.manifest.files
            if record.path.startswith(PAYLOAD_DIR + "/")
        },
        "pool_files": len(pool),
        "pool_bytes": sum(record.bytes for record in pool),
        "pool_sha256": hashlib.sha256(
            "".join(f"{record.path}\0{record.sha256}\n" for record in pool).encode()
        ).hexdigest(),
    }
