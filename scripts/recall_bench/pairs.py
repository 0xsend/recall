"""`pairs`: run baseline/candidate arms in order, stopping at the first non-passed arm.

Each arm uses daemon-run's supervisor and the maintained FIFO bench lock.
The lock is held per arm, so other measuring lanes can interleave. The batch
keeps every arm's outcome, including an early refusal with no run directory.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath
from typing import Never

from with_bench_lock import ticket, wait_deadline

from . import daemon
from .evidence import EXIT_STATUS, JobFailed, JobIncomplete, Result, tool_identity
from .fs import Cancellation, Directory, Refused
from .isolation import Build, Policy, Template, WriteRoots, scan

Progress = Callable[[dict[str, object]], None]
PAIRS_MAX = 1000
LOCK_WAIT_SECONDS = 10800


class Options(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise JobFailed("usage", message)


def _count(text: str) -> int:
    value = int(text)
    if not 1 <= value <= PAIRS_MAX:
        raise argparse.ArgumentTypeError(f"must be within 1..{PAIRS_MAX}")
    return value


def add_arguments(parser: argparse.ArgumentParser) -> None:
    for name, help_text in (
        ("lock", "shared FIFO queue directory"),
        ("baseline", "baseline build checkout"),
        ("candidate", "candidate build checkout"),
        ("template", "verified template made by prepare"),
        ("run-parent", "existing short directory for fresh per-arm run directories"),
        ("output", "new directory for arm evidence and batch result"),
    ):
        parser.add_argument(f"--{name}", required=True, help=help_text)
    parser.add_argument("--count", type=_count, default=3, help="baseline/candidate pair count")
    parser.add_argument(
        "options", nargs=argparse.REMAINDER, help="-- followed by daemon-run options"
    )


def _queue(policy: Policy, roots: WriteRoots, raw: str, cancellation: Cancellation) -> Directory:
    # New queues are inside the launcher's write roots; an existing shared
    # queue may be outside them, but still passes every read-path refusal.
    try:
        directory = policy.open_tree("--lock", raw)
    except Refused:
        with roots.new_directory("--lock", raw) as target, target.create() as tree:
            tree.commit(cancellation)
        directory = policy.open_tree("--lock", raw)
    try:
        # The maintained lock runner opens its queue files by relative name.
        # Refuse aliases before handing it the held directory as cwd. Queue
        # directories and their mutex must never be repurposed while in use.
        scan("--lock", directory)
    except BaseException:
        directory.close()
        raise
    return directory


@contextmanager
def _locked(directory: Directory, label: str, cancellation: Cancellation) -> Iterator[None]:
    """Acquire the existing lock in its held directory, with cancellable admission."""

    def interrupted(signum: int, _frame: object) -> None:
        cancellation.signum = cancellation.signum or signum
        cancellation.check()

    with ExitStack() as held:
        held.enter_context(directory.as_working_directory())
        previous = {signum: signal.signal(signum, interrupted) for signum in Cancellation.SIGNALS}
        try:
            with wait_deadline(LOCK_WAIT_SECONDS):
                held.enter_context(ticket(Path("."), label))
        except TimeoutError as error:
            raise JobIncomplete("lock-timeout", str(error)) from error
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        cancellation.check()
        yield


def run(
    args: argparse.Namespace, result: Result, cancellation: Cancellation, progress: Progress
) -> None:
    options = Options(prog="pairs daemon-run options", allow_abbrev=False)
    daemon.add_profile_arguments(options)
    raw = args.options[1:] if args.options[:1] == ["--"] else args.options
    workload = options.parse_args(raw)
    if workload.hold:
        raise JobFailed("usage", "pairs cannot --hold; use daemon-run for a guest environment")
    policy = Policy.for_host()
    with ExitStack() as held:
        roots = held.enter_context(policy.open_write_roots())
        result.leftovers = roots.leftovers
        template = held.enter_context(
            Template.open(policy, "--template", args.template, cancellation)
        )
        builds = [
            Build.resolve(policy, f"--{name}", getattr(args, name))
            for name in ("baseline", "candidate")
        ]
        run_parent = held.enter_context(policy.open_tree("--run-parent", args.run_parent))
        policy = (
            policy.with_checkout(builds[0].path)
            .with_checkout(builds[1].path)
            .with_checkout(template.path)
        )
        # All destinations, including the shared queue, refuse both builds.
        roots.policy = policy
        queue = held.enter_context(_queue(policy, roots, args.lock, cancellation))
        target = held.enter_context(roots.new_directory("--output", args.output))
        # Validate every run destination before making batch output or taking a ticket.
        for index in range(args.count):
            for variant in ("baseline", "candidate"):
                with roots.new_directory("--run-dir", run_parent.path / f"{index + 1}-{variant}"):
                    pass
        result.scope = {
            "measurements": [],
            "missing": ["batch success alone is not a benchmark qualification"],
            "tool": tool_identity(),
            "count": args.count,
            "template": str(template.path),
            "lock": str(queue.path),
            "baseline": str(builds[0].path),
            "candidate": str(builds[1].path),
            "options": vars(workload),
        }
        with target.create() as tree:
            tree.commit(cancellation)
            try:
                for index in range(args.count):
                    for variant, build in zip(("baseline", "candidate"), builds, strict=True):
                        label = f"{index + 1}-{variant}"
                        cancellation.expire_at(time.monotonic() + LOCK_WAIT_SECONDS)
                        cancellation.check()
                        arm = Result(job="daemon-run")
                        arm_args = argparse.Namespace(
                            **vars(workload),
                            template=str(template.path),
                            build=str(build.path),
                            run_dir=str(run_parent.path / label),
                            output=str(tree.path / label),
                        )
                        error: Exception | None = None
                        with _locked(queue, label, cancellation):
                            try:
                                daemon.run(arm_args, arm, cancellation, progress)
                            except Exception as ended:
                                error = ended
                            status = (
                                arm.end(error) if error is not None else EXIT_STATUS[arm.outcome]
                            )
                        result.leftovers.extend(arm.leftovers)
                        result.exit_codes[label] = status
                        result.step(
                            "run",
                            pair=index + 1,
                            variant=variant,
                            outcome=arm.outcome,
                            exit_code=status,
                            output=arm_args.output,
                            run_dir=arm_args.run_dir,
                        )
                        tree.publish(PurePosixPath(label + ".json"), arm.to_json())
                        progress(
                            {
                                "event": "arm",
                                "label": label,
                                "outcome": arm.outcome,
                                "exit_code": status,
                            }
                        )
                        if arm.outcome != "passed" or status != 0:
                            if error is not None:
                                raise error
                            failure = JobIncomplete if arm.outcome == "incomplete" else JobFailed
                            raise failure("arm", f"{label} did not pass", exit_code=status)
            finally:
                result.end(sys.exc_info()[1])
                tree.publish(PurePosixPath("result.json"), result.to_json())
                tree.publish(
                    PurePosixPath("pairs.json"), json.dumps(result.line(), indent=2).encode()
                )
