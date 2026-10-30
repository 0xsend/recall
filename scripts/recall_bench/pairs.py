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
from with_bench_lock import run_command, ticket, wait_deadline

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
def _locked(directory: Directory, label: str, cancellation: Cancellation) -> Iterator[IO[str]]:
    """Acquire the maintained lock in its held directory, with cancellable admission."""

    def interrupted(signum: int, _frame: object) -> None:
        cancellation.signum = cancellation.signum or signum
        cancellation.check()

    with ExitStack() as held:
        held.enter_context(directory.as_working_directory())
        previous = {signum: signal.signal(signum, interrupted) for signum in Cancellation.SIGNALS}
        try:
            with wait_deadline(LOCK_WAIT_SECONDS):
                owner = held.enter_context(ticket(Path("."), label))
        except TimeoutError as error:
            raise JobIncomplete("lock-timeout", str(error)) from error
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
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
    command: list[str], owner: IO[str], tree: NewTree, label: str, cancellation: Cancellation
) -> int:
    stdout = tree.open_new(tree.root, PurePosixPath(label + ".stdout"))
    stderr = tree.open_new(tree.root, PurePosixPath(label + ".stderr"))

    def interrupted(signum: int, _frame: object) -> None:
        cancellation.signum = cancellation.signum or signum
        raise LockCancelled(signum)

    previous = {signum: signal.signal(signum, interrupted) for signum in Cancellation.SIGNALS}
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    try:
        # The child inherits the ticket descriptor. If pairs is killed, the
        # daemon-run process keeps the queue locked until its own cleanup ends.
        status = run_command(command, owner, stdout=stdout, stderr=stderr)
        cancellation.check()
        return 128 - status if status < 0 else status
    except LockCancelled as error:
        cancellation.signum = cancellation.signum or error.signum
        cancellation.check()
        raise AssertionError("cancellation must raise") from error
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        # run_command holds TERM/INT through cleanup; pairs continues after it.
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        os.close(stdout)
        os.close(stderr)


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
                        with _locked(queue, label, cancellation) as owner:
                            status = _execute(
                                daemon_command(arm_args), owner, tree, label, cancellation
                            )
                        with tree.root.open_file("arm", label + ".stdout") as file:
                            if file.status.st_size > 64 * 1024**2:
                                raise JobFailed("arm-output", f"{label} output exceeds 64 MiB")
                            data = file.head(file.status.st_size)
                        try:
                            arm = Document.model_validate(
                                from_json(data.splitlines()[-1], allow_inf_nan=False)
                            )
                        except (ValueError, IndexError, ValidationError) as error:
                            raise JobFailed(
                                "arm-output", f"{label} did not publish a valid outcome"
                            ) from error
                        result.leftovers.extend(item.model_dump() for item in arm.leftovers)
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
                        tree.publish(PurePosixPath(label + ".json"), data.splitlines()[-1] + b"\n")
                        progress(
                            {
                                "event": "arm",
                                "label": label,
                                "outcome": arm.outcome,
                                "exit_code": status,
                            }
                        )
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
