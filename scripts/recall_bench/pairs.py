"""`pairs`: run baseline/candidate arms in order, stopping at the first non-passed arm.

Each arm uses daemon-run's supervisor and the maintained FIFO bench lock.
The lock is held per arm, so other measuring lanes can interleave. The batch
keeps every arm's outcome, including an early refusal with no run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Never

from pydantic import ValidationError
from pydantic_core import from_json
from with_bench_lock import Cancelled as LockCancelled
from with_bench_lock import CommandExecution, FinishedCommand, run_command, ticket, wait_deadline

from . import daemon
from .database import Database
from .documents import Document
from .evidence import TOOL_ROOT, JobFailed, JobIncomplete, Result, tool_identity
from .fs import Cancellation, Cancelled, Directory, NewTree, Refused
from .isolation import Build, NewDirectory, Policy, WriteRoots, scan

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


@dataclass(frozen=True)
class Layout:
    """Three disjoint namespaces: runs, evidence, and the shared queue."""

    runs: Directory
    output: NewDirectory
    queue: Directory | NewDirectory

    def __post_init__(self) -> None:
        paths = (
            ("--run-parent", self.runs.path),
            ("--output", self.output.path),
            ("--lock", self.queue.path),
        )
        # Capabilities already refused unapproved links. Canonical spelling
        # also unifies the filesystem's accepted aliases (macOS /tmp, etc.).
        for index, (argument, path) in enumerate(paths):
            parts = tuple(part.casefold() for part in path.resolve().parts)
            for other_argument, other in paths[index + 1 :]:
                other_parts = tuple(part.casefold() for part in other.resolve().parts)
                common = min(len(parts), len(other_parts))
                if parts[:common] == other_parts[:common]:
                    raise Refused(
                        argument,
                        path,
                        f"overlaps {other_argument} {other}; "
                        "runs, output and queue require disjoint directories",
                    )

    def open_queue(self, cancellation: Cancellation) -> Directory:
        if isinstance(self.queue, Directory):
            directory = Directory(self.queue.path, os.dup(self.queue.descriptor))
        else:
            with self.queue.create() as tree:
                tree.commit(cancellation)
                directory = Directory(tree.path, os.dup(tree.root.descriptor))
        try:
            # Queue names stay stable while any runner might use them.
            scan("--lock", directory)
        except BaseException:
            directory.close()
            raise
        return directory


def _queue_target(policy: Policy, roots: WriteRoots, raw: str) -> Directory | NewDirectory:
    # Existing shared queues may be outside the launcher's private write roots.
    # Resolve both forms without creating anything: Layout owns all mutations.
    try:
        return policy.open_tree("--lock", raw)
    except Refused as error:
        if not isinstance(error.__cause__, FileNotFoundError):
            raise
        return roots.new_directory("--lock", raw)


@contextmanager
def _interruptible(cancellation: Cancellation) -> Iterator[None]:
    """Raise only inside the guarded scope; restore handlers and masks on every exit."""
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, Cancellation.SIGNALS)

    def interrupted(signum: int, _frame: object) -> None:
        cancellation.signum = cancellation.signum or signum
        raise LockCancelled(cancellation.signum)

    previous = {signum: signal.signal(signum, interrupted) for signum in Cancellation.SIGNALS}
    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        cancellation.check()
        yield
    except LockCancelled as error:
        cancellation.signum = cancellation.signum or error.signum
        raise Cancelled(cancellation.signum) from error
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK, Cancellation.SIGNALS)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)


@contextmanager
def _locked(directory: Directory, label: str, cancellation: Cancellation) -> Iterator[IO[str]]:
    """Acquire the maintained lock in its held directory, with cancellable admission."""

    with ExitStack() as held:
        held.enter_context(directory.as_working_directory())
        try:
            with _interruptible(cancellation), wait_deadline(LOCK_WAIT_SECONDS):
                owner = held.enter_context(ticket(Path("."), label))
        except TimeoutError as error:
            raise JobIncomplete("lock-timeout", str(error)) from error
        cancellation.check()
        yield owner


def daemon_command(args: argparse.Namespace) -> list[str]:
    """The fixed daemon-run entrypoint; only profile options cross from the caller."""
    return [
        str(TOOL_ROOT / ".venv/bin/python"),
        str(TOOL_ROOT / "scripts/bench.py"),
        "daemon-run",
        "--template",
        args.template,
        "--build",
        args.build,
        "--run-dir",
        args.run_dir,
        "--output",
        args.output,
        *args.options,
    ]


def _execute(
    command: list[str],
    owner: IO[str],
    tree: NewTree,
    label: str,
    cancellation: Cancellation,
    execution: CommandExecution,
) -> None:
    with ExitStack() as files:
        stdout = tree.open_new(tree.root, PurePosixPath(label + ".stdout"))
        files.callback(os.close, stdout)
        stderr = tree.open_new(tree.root, PurePosixPath(label + ".stderr"))
        files.callback(os.close, stderr)
        with _interruptible(cancellation):
            # The child keeps admission even if the batch owner is killed.
            run_command(command, owner, stdout=stdout, stderr=stderr, execution=execution)


def _record_arm(
    tree: NewTree,
    label: str,
    pair: int,
    variant: str,
    args: argparse.Namespace,
    execution: CommandExecution,
    result: Result,
    ended: BaseException | None,
    progress: Progress,
) -> tuple[Document, int]:
    state = execution.state
    assert state is not None
    raw_status = state.status if isinstance(state, FinishedCommand) else None
    status = (
        (128 - raw_status if raw_status < 0 else raw_status) if raw_status is not None else None
    )
    source = "child"
    try:
        with tree.root.open_file("arm", label + ".stdout") as file:
            if file.status.st_size > 64 * 1024**2:
                raise ValueError("output exceeds 64 MiB")
            lines = file.head(file.status.st_size).splitlines()
        data = lines[-1] + b"\n"
        arm = Document.model_validate(from_json(data, allow_inf_nan=False))
    except (ValueError, IndexError, ValidationError) as error:
        source = "owner"
        fallback = Result(job="daemon-run")
        fallback.end(
            ended
            if isinstance(ended, Cancelled)
            else JobFailed("arm-output", f"{label} did not publish a valid outcome: {error}")
        )
        data = fallback.to_json()
        arm = Document.model_validate(from_json(data, allow_inf_nan=False))
    result.leftovers.extend(item.model_dump() for item in arm.leftovers)
    if status is not None:
        result.exit_codes[label] = status
    result.step(
        "run",
        pair=pair,
        variant=variant,
        outcome=arm.outcome,
        exit_code=status,
        outcome_source=source,
        output=args.output,
        run_dir=args.run_dir,
    )
    tree.publish(PurePosixPath(label + ".json"), data)
    progress({"event": "arm", "label": label, "outcome": arm.outcome, "exit_code": status})
    return arm, status if status is not None else 2


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
        database = held.enter_context(
            Database.open(policy, "--template", args.template, cancellation)
        )
        template = database.template
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
        queue_target = held.enter_context(_queue_target(policy, roots, args.lock))
        target = held.enter_context(roots.new_directory("--output", args.output))
        layout = Layout(run_parent, target, queue_target)
        # Validate every run destination before making batch output or taking a ticket.
        for index in range(args.count):
            for variant in ("baseline", "candidate"):
                with roots.new_directory("--run-dir", run_parent.path / f"{index + 1}-{variant}"):
                    pass
        queue = held.enter_context(layout.open_queue(cancellation))
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
                        arm_args = argparse.Namespace(
                            template=str(template.path),
                            build=str(build.path),
                            run_dir=str(run_parent.path / label),
                            output=str(tree.path / label),
                            options=raw,
                        )
                        execution = CommandExecution()
                        try:
                            with _locked(queue, label, cancellation) as owner:
                                _execute(
                                    daemon_command(arm_args),
                                    owner,
                                    tree,
                                    label,
                                    cancellation,
                                    execution,
                                )
                        finally:
                            # Once a child started, every exit path records it before
                            # cancellation can leave the batch, including a complete
                            # child outcome delivered just before the signal.
                            if execution.state is not None:
                                with Cancellation.deferred():
                                    arm, status = _record_arm(
                                        tree,
                                        label,
                                        index + 1,
                                        variant,
                                        arm_args,
                                        execution,
                                        result,
                                        sys.exc_info()[1],
                                        progress,
                                    )
                        cancellation.check()
                        if arm.outcome != "passed" or status != 0:
                            if status in (130, 143):
                                raise Cancelled(status - 128)
                            failure = JobIncomplete if arm.outcome == "incomplete" else JobFailed
                            raise failure(
                                "arm",
                                f"{label} did not pass",
                                exit_code=status,
                                child_error=arm.error.model_dump() if arm.error else None,
                            )
            finally:
                result.end(sys.exc_info()[1])
                tree.publish(PurePosixPath("result.json"), result.to_json())
                tree.publish(
                    PurePosixPath("pairs.json"), json.dumps(result.line(), indent=2).encode()
                )
