"""`daemon-run`: one owned scratch daemon on a fresh clone of a template, measured by profile.

Profile `idle`: start the build's daemon under the guarded launcher, wait for
readiness from the owned PID, settle (catalog complete, nothing pending, no
new failures), sample the daemon's CPU and memory over the window, then take
the final status and stats. `--hold` keeps the daemon up afterwards for a
guest until it creates the `done` file or the hold times out.

`passed` means only that this profile's measurements completed and met the
floors it evaluates; `scope.missing` names what it does not cover.
"""

from __future__ import annotations

import argparse
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
    POOL_DIR,
    SOURCES,
    SYSTEM_PATH,
    Build,
    Policy,
    RunDir,
    Template,
    UsageLog,
)
from .samplers import Sample, host_load, sample, summarize
from .supervisor import Completed, Deadline, Stopped, Supervisor

Progress = Callable[[dict[str, object]], None]

PROFILES = ("idle",)
CLI_TIMEOUT_SECONDS = 60.0
LAUNCHER_TIMEOUT_SECONDS = 120.0
POLL_SECONDS = 1.0
SAMPLE_SECONDS = 1.0
WINDOW_SECONDS_MAX = 24 * 3600
# BRIEF.md floors this profile evaluates.
STATUS_SECONDS_MAX = 2.0
RSS_BYTES_MAX = 4 * 1024**3
# Counts that must not move while nothing is appended.
STABLE_COUNTS = ("sessions", "messages", "tool_calls", "bash_calls")
DONE_NAME = "done"
GUEST_ENV_NAME = "guest.env"
RESULT_NAME = "result.json"
MISSING_CLASSES = (
    "append freshness and writer turns (profile append)",
    "status/live under concurrent load (profile cohort)",
    "embed backlog drain and capacity (profile backlog)",
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
        raise argparse.ArgumentTypeError("must be a finite positive number of seconds")
    return value


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--template", required=True, help="template made by `prepare`")
    parser.add_argument("--build", required=True, help="checkout whose .venv runs recall")
    parser.add_argument("--output", required=True, help="new evidence directory")
    parser.add_argument(
        "--run-dir", required=True, help="new, short directory: the daemon's data dir"
    )
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--window", type=_positive, default=60.0, help="sampling seconds")
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
        result.scope = {
            "profile": args.profile,
            "measurements": ["readiness", "catalog completion", "idle cpu", "idle rss"],
            "missing": list(MISSING_CLASSES),
            "template": str(template.path),
            "run_dir": str(run_dir.path),
            "usage_log": str(run_dir.usage_log_path),
            "build": {"path": str(build.path), **checkout_identity(build.path, BUILD_PATHS)},
            "tool": tool_identity(),
            "settings": {
                "window_seconds": args.window,
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
            Profile(args, result, progress, deadline, build, run_dir, tree, supervisor).idle()
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
        "suppressed": suppressed,
    }
    if canary:
        return JobFailed("service_manager_invoked", "a service-manager canary ran", lines=canary)
    if refused:
        return JobFailed("service_reach_refused", "a child tried to reach a service manager")
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

    def phase(self, name: str) -> None:
        self.progress({"event": "phase", "phase": name})

    def idle(self) -> None:
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
        if self.args.hold:
            self.phase("hold")
            self.hold(daemon.pid, daemon.exited)

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
                raise JobFailed("generation_error", "the daemon reports errors", errors=errors)
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
    return {
        "status": completed.status,
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
