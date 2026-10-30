"""Run one measurement under a crash-safe FIFO bench lock (POSIX hosts)."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import IO

import psutil

QUEUE_LIMIT = 1024
CHILD_STOP_SECONDS = 10
CHILD_POLL_SECONDS = 0.01


class Cancelled(Exception):
    def __init__(self, signum: int) -> None:
        self.signum = signum


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"lock": event, **fields}), file=sys.stderr, flush=True)


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


def wait_unreaped(pid: int, timeout: float | None = None) -> None:
    """Observe exit without consuming the leader's PID reservation.

    Popen.wait/poll reap; os.waitid is unavailable on macOS Python 3.12.
    """
    process = psutil.Process(pid)
    deadline = None if timeout is None else time.monotonic() + timeout
    while process.status() != psutil.STATUS_ZOMBIE:
        delay = CHILD_POLL_SECONDS
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("command did not exit during shutdown grace period")
            delay = min(delay, remaining)
        time.sleep(delay)


def run_command(command: list[str], owner: IO[str]) -> int:
    # Do not interrupt between spawning and recording the owned process group.
    # This standalone runner has no threads; restore the mask before child exec.
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})

    def restore_mask() -> None:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)

    # SIG_IGN would let the kernel reap the child before group teardown.
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    child: subprocess.Popen[bytes] | None = None
    try:
        try:
            child = subprocess.Popen(
                command,
                start_new_session=True,
                pass_fds=(owner.fileno(),),
                preexec_fn=restore_mask,
            )
        finally:
            # Pending TERM/INT can raise immediately when unmasked.
            restore_mask()
        wait_unreaped(child.pid)
    finally:
        try:
            # Cancellation cannot interrupt the stop/reap sequence. This
            # standalone process exits after releasing its ticket, so the mask
            # remains blocked through the rest of shutdown.
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        finally:
            if child is not None:
                # Keep the leader unreaped until every group signal is sent:
                # even an exited leader reserves its PID against PGID reuse.
                try:
                    with suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGTERM)
                    wait_unreaped(child.pid, timeout=CHILD_STOP_SECONDS)
                except TimeoutError:
                    pass  # The group is killed after its shutdown grace period.
                finally:
                    try:
                        with suppress(ProcessLookupError):
                            os.killpg(child.pid, signal.SIGKILL)
                    finally:
                        child.wait()
    assert child is not None and child.returncode is not None
    return child.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True, help="shared host queue directory")
    parser.add_argument(
        "--timeout", type=float, default=10800, help="maximum queue wait in seconds"
    )
    parser.add_argument("label")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
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
    try:
        with wait_deadline(args.timeout), ticket(args.lock, args.label) as owner:
            # The deadline limits admission, never the measurement itself.
            signal.setitimer(signal.ITIMER_REAL, 0)
            emit("acquired", pid=os.getpid(), label=args.label)
            status = run_command(command, owner)
            status = 128 - status if status < 0 else status
    except Cancelled as error:
        status = 128 + error.signum
    except TimeoutError as error:
        emit("timed_out", label=args.label, error=str(error))
        status = 75
    except (OSError, RuntimeError) as error:
        emit("failed", label=args.label, error=str(error))
        status = 1
    emit("released", label=args.label, child_status=status)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
