"""Run one measurement under a crash-safe FIFO bench lock (POSIX hosts)."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import json
import math
import os
import select
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import psutil

QUEUE_LIMIT = 1024
CHILD_STOP_SECONDS = 10
CHILD_POLL_SECONDS = 0.01
GROUP_SNAPSHOT_LIMIT = 65536


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

    def wait_unreaped(self, pid: int) -> None:
        process = psutil.Process(pid)
        while True:
            if self.cancelled is not None:
                raise Cancelled(self.cancelled)
            if process.status() == psutil.STATUS_ZOMBIE:
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


def wait_unreaped(pid: int, timeout: float) -> None:
    """Allow bounded shutdown without consuming the leader's PID reservation.

    Popen.wait/poll reap; os.waitid is unavailable on macOS Python 3.12.
    """
    process = psutil.Process(pid)
    deadline = time.monotonic() + timeout
    while process.status() != psutil.STATUS_ZOMBIE:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("command did not exit during shutdown grace period")
        time.sleep(min(CHILD_POLL_SECONDS, remaining))


def darwin_group_exited(pgid: int) -> bool:
    """Confirm that a still-owned session's group has no live members."""
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    list_group = libproc.proc_listpgrppids
    list_group.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    list_group.restype = ctypes.c_int

    def snapshot() -> set[int]:
        pids = (ctypes.c_int * GROUP_SNAPSHOT_LIMIT)()
        count = list_group(pgid, pids, ctypes.sizeof(pids))
        # The unreaped leader must be present. An empty, failed or possibly
        # truncated query cannot establish that the group has exited.
        if not 0 < count < len(pids):
            return set()
        return set(pids[:count])

    members = snapshot()
    if pgid not in members:
        return False
    zombies: dict[int, psutil.Process] = {}
    for pid in members:
        try:
            process = psutil.Process(pid)
            if process.status() != psutil.STATUS_ZOMBIE:
                return False
            zombies[pid] = process
        except psutil.NoSuchProcess:
            pass  # A reaped descendant cannot produce more group members.
        except psutil.AccessDenied:
            return False
    # A member could have forked before its zombie observation. Reject any
    # such addition. Outsiders cannot join this new session, and zombies cannot
    # fork, so a complete second snapshot closes that race.
    remaining = snapshot()
    if pgid not in remaining or not remaining <= zombies.keys():
        return False
    # Descendants can be reaped by init. Retain their birth identities so a
    # recycled PID cannot stand in for a zombie observed above.
    return all(zombies[pid].is_running() for pid in remaining)


def signal_unreaped_group(pgid: int, signum: signal.Signals) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        # XNU killpg1 filters zombies, then returns EPERM if no eligible member
        # was found. A live permission denial must still fail the run.
        # https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_sig.c
        if error.errno != errno.EPERM or sys.platform != "darwin":
            raise
        # An exiting member can become unsignalable before its zombie status
        # is observable. Wait only to prove completion, never to assume it.
        deadline = time.monotonic() + CHILD_STOP_SECONDS
        while not darwin_group_exited(pgid):
            if time.monotonic() >= deadline:
                raise
            time.sleep(CHILD_POLL_SECONDS)


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
            child: subprocess.Popen[bytes] | None = None
            try:
                try:
                    child = subprocess.Popen(
                        command,
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=True,
                        pass_fds=(owner.fileno(),),
                        preexec_fn=restore_mask,
                    )
                    execution.state = RunningCommand(child.pid)
                finally:
                    restore_mask()
                waiter.wait_unreaped(child.pid)
            finally:
                # Handlers record signals throughout shutdown;
                # repeated cancellation cannot interrupt cleanup on any thread.
                if child is not None:
                    # Keep the leader unreaped until every group signal is sent:
                    # even an exited leader reserves its PID against PGID reuse.
                    try:
                        signal_unreaped_group(child.pid, signal.SIGTERM)
                        wait_unreaped(child.pid, timeout=CHILD_STOP_SECONDS)
                    except TimeoutError:
                        pass  # The group is killed after its shutdown grace period.
                    finally:
                        try:
                            signal_unreaped_group(child.pid, signal.SIGKILL)
                        finally:
                            status = child.wait()
                            execution.state = FinishedCommand(child.pid, status)
                cleanup_succeeded = True
    finally:
        # Include exceptional bodies without hiding a process-group shutdown
        # failure, which must remain visible even when cancellation is pending.
        if cleanup_succeeded and waiter is not None and waiter.cancelled is not None:
            raise Cancelled(waiter.cancelled)
    assert child is not None and child.returncode is not None
    return child.returncode


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
