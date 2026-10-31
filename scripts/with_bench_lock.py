"""Run one measurement under a crash-safe FIFO bench lock (POSIX hosts)."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import select
import signal
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import psutil
from recall_bench.owned_process_group import FailedStop, OwnedGroup, StopFailed, stop_groups

QUEUE_LIMIT = 1024
CHILD_STOP_SECONDS = 10
CHILD_POLL_SECONDS = 0.01


class Cancelled(Exception):
    def __init__(self, signum: int) -> None:
        self.signum = signum


class CommandWait:
    def __init__(self, notifications: int) -> None:
        self.notifications = notifications
        self.cancelled: int | None = None

    def notify(self, signum: int, _frame: object) -> None:
        # Record cancellation rather than raising inside spawn or cleanup.
        if signum != signal.SIGCHLD and self.cancelled is None:
            self.cancelled = signum

    def wait_unreaped(self, owner: OwnedGroup) -> None:
        while True:
            if self.cancelled is not None:
                raise Cancelled(self.cancelled)
            if owner.exited():
                return
            # Signals write to the pipe even if delivered on a native thread.
            # Registering before spawn keeps the status-to-wait race closed.
            select.select([self.notifications], [], [])
            os.read(self.notifications, 4096)


@contextmanager
def command_wait() -> Iterator[CommandWait]:
    reader, writer = os.pipe()
    handlers = {}
    try:
        os.set_blocking(writer, False)
        waiter = CommandWait(reader)
        # A caught SIGCHLD retains zombies. Darwin discards SIG_DFL events;
        # SIG_IGN would allow auto-reap. Exec restores caught dispositions.
        for signum in (signal.SIGCHLD, signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, waiter.notify)
        previous = signal.set_wakeup_fd(writer, warn_on_full_buffer=False)
        try:
            yield waiter
        finally:
            signal.set_wakeup_fd(previous)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        os.close(writer)
        os.close(reader)


def emit(event: str, **fields: object) -> None:
    # Keep the newline in the same write so cancellation cannot split it
    # from the event body.
    sys.stderr.write(json.dumps({"lock": event, **fields}) + "\n")
    sys.stderr.flush()


@contextmanager
def wait_deadline(seconds: float) -> Iterator[None]:
    def expired(_signum: int, _frame: object) -> None:
        raise TimeoutError("bench lock wait timed out")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def ticket(directory: Path, label: str) -> Iterator[IO[str]]:
    """Register under a mutex, then wait on every earlier live ticket.

    Waiting on all predecessors preserves FIFO even if a middle waiter dies.
    Files are bookkeeping only: ownership is always the kernel-held flock.
    The registration mutex and directory persist; unlinking them would let
    concurrent callers lock different inodes for the same queue.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as handles:
        gate = handles.enter_context((directory / "queue").open("a+"))
        fcntl.flock(gate, fcntl.LOCK_EX)
        predecessors: list[IO[str]] = []
        for path in sorted(directory.glob("*.ticket")):
            handle = handles.enter_context(path.open("r+"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                predecessors.append(handle)
            else:
                path.unlink()
                handle.close()
        if len(predecessors) >= QUEUE_LIMIT:
            raise RuntimeError(f"bench lock queue is full ({QUEUE_LIMIT})")
        number = int(Path(predecessors[-1].name).stem) + 1 if predecessors else 0
        path = directory / f"{number:020d}.ticket"
        owner = handles.enter_context(path.open("x+"))
        fcntl.flock(owner, fcntl.LOCK_EX)
        identity = {
            "pid": os.getpid(),
            "start_time": psutil.Process().create_time(),
            "label": label,
            "ticket": number,
        }
        json.dump(identity, owner)
        owner.flush()
        fcntl.flock(gate, fcntl.LOCK_UN)
        emit("queued", **identity)
        try:
            for predecessor in predecessors:
                fcntl.flock(predecessor, fcntl.LOCK_EX)
                predecessor.close()
            yield owner
        finally:
            # A killed supervisor leaves this entry for the next registration
            # to reclaim, once the command's inherited descriptor also closes.
            owner.close()


@dataclass(frozen=True)
class RunningCommand:
    pid: int


@dataclass(frozen=True)
class FinishedCommand:
    pid: int
    status: int


@dataclass
class CommandExecution:
    """Observed process state, retained even when cancellation unwinds the call."""

    state: RunningCommand | FinishedCommand | None = None


def run_command(
    command: list[str],
    owner: IO[str],
    *,
    stdout: int | None = None,
    stderr: int | None = None,
    execution: CommandExecution | None = None,
) -> int:
    """Run with inherited admission and retain process observations through cancellation."""
    # Do not interrupt between spawning and recording the owned process group.
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGCHLD, signal.SIGTERM, signal.SIGINT})

    def restore_mask() -> None:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)

    if execution is None:
        execution = CommandExecution()
    waiter: CommandWait | None = None
    cleanup_succeeded = False
    try:
        with command_wait() as waiter:
            child: OwnedGroup | None = None
            try:
                try:
                    child = OwnedGroup.start(
                        command,
                        stdout=stdout,
                        stderr=stderr,
                        pass_fds=(owner.fileno(),),
                        preexec_fn=restore_mask,
                    )
                    execution.state = RunningCommand(child.pid)
                finally:
                    restore_mask()
                waiter.wait_unreaped(child)
            finally:
                # Handlers record signals throughout shutdown;
                # repeated cancellation cannot interrupt cleanup on any thread.
                if child is not None:
                    stopped = stop_groups(
                        [child], grace_seconds=CHILD_STOP_SECONDS, proof_seconds=3.0
                    )[0]
                    if stopped.status is not None:
                        execution.state = FinishedCommand(child.pid, stopped.status)
                    if isinstance(stopped, FailedStop):
                        raise StopFailed(stopped)
                cleanup_succeeded = True
    finally:
        # Include exceptional bodies without hiding a process-group shutdown
        # failure, which must remain visible even when cancellation is pending.
        if cleanup_succeeded and waiter is not None and waiter.cancelled is not None:
            raise Cancelled(waiter.cancelled)
    assert child is not None and isinstance(execution.state, FinishedCommand)
    return execution.state.status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock", type=Path, required=True, help="shared absolute queue directory for this host"
    )
    parser.add_argument(
        "--timeout", type=float, default=10800, help="maximum queue wait in seconds"
    )
    parser.add_argument("label")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.lock.is_absolute():
        parser.error(
            "--lock must name the shared absolute queue directory agreed by all lanes on this "
            "host; see RECONCILIATION_QA.md#bench-lock"
        )
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")

    cancellation_requested = False

    def cancel(signum: int, _frame: object) -> None:
        nonlocal cancellation_requested
        if not cancellation_requested:
            cancellation_requested = True
            raise Cancelled(signum)

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    status = 1
    acquired = False
    try:
        try:
            with wait_deadline(args.timeout), ticket(args.lock, args.label) as owner:
                acquired = True
                # The deadline limits admission, never the measurement itself.
                signal.setitimer(signal.ITIMER_REAL, 0)
                emit("acquired", pid=os.getpid(), label=args.label)
                status = run_command(command, owner)
                status = 128 - status if status < 0 else status
        except TimeoutError as error:
            emit("timed_out", label=args.label, error=str(error))
            status = 75
        except (OSError, RuntimeError) as error:
            emit("failed", label=args.label, error=str(error))
            status = 1
        emit("released", label=args.label, child_status=status, acquired=acquired)
        return status
    except Cancelled as error:
        # Include error and release reporting in the cancellation boundary.
        # A cancellation recorded by command_wait also consumes the CLI's
        # one interrupt, so repeated signals cannot interrupt the final report.
        cancellation_requested = True
        status = 128 + error.signum
        emit("released", label=args.label, child_status=status, acquired=acquired)
        return status


if __name__ == "__main__":
    raise SystemExit(main())
