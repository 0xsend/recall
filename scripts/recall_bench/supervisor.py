"""The one owner of every process a bench job starts.

Each child runs in its own session, so its process group is its own. The
supervisor records the child (PID, group, start time, argv) in the output's
`processes.jsonl` before releasing it: a supervised launcher waits for one
byte on stdin before doing anything. The shared launch owner retains its
unreaped leader through all group signals and completion evidence, then reaps.
No process is ever found for signalling through a process-table search.

Every child is started the one way the isolation core allows: the run
directory is proved to still be this run (`RunDir.held`), the build is
resolved again (`Build.recheck`), and the child starts in the held run
directory with the run's allow-listed environment. `check` re-proves the run
at every supervision point, and waiting for a child re-proves it at least
every RUN_CHECK_SECONDS and once more when the child exits.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import PurePosixPath

import psutil

from .fs import Cancellation, NewTree
from .isolation import Build, RunDir, child_environment
from .owned_process_group import CompletionEvidence, FailedStop, OwnedGroup, StopError, stop_groups

STOP_GRACE_SECONDS = 5.0
GONE_SECONDS = 3.0
POLL_SECONDS = 0.1
RUN_CHECK_SECONDS = 1.0
CHILDREN_MAX = 10_000
OUTPUT_BYTES_MAX = 64 * 1024 * 1024
REGISTRY_NAME = "processes.jsonl"

Progress = Callable[[dict[str, object]], None]


class Deadline:
    """A point in monotonic time after which waiting work is incomplete."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        if not seconds > 0:
            raise ValueError("a deadline must be positive")
        self._clock = clock
        self.at = clock() + seconds

    def remaining(self) -> float:
        return self.at - self._clock()

    def expired(self) -> bool:
        return self.remaining() <= 0


@dataclass(frozen=True)
class Registration:
    name: str
    pid: int
    create_time: float
    argv: tuple[str, ...]

    @property
    def pgid(self) -> int:
        # start_new_session makes every child the leader of its own group.
        return self.pid


@dataclass(frozen=True)
class Stopped:
    """How a child and its process group ended."""

    name: str
    pid: int
    status: int | None
    signals: tuple[str, ...]
    group_gone: bool
    clean: bool
    evidence: CompletionEvidence | None
    errors: tuple[StopError, ...]
    unresolved: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class Completed:
    status: int | None
    stdout: bytes
    stderr: bytes
    seconds: float

    @property
    def timed_out(self) -> bool:
        return self.status is None


class ChildStopFailed(RuntimeError):
    def __init__(self, stopped: Stopped) -> None:
        self.stopped = stopped
        super().__init__(f"cleanup failed for {stopped.name}: {stopped.errors}")


@dataclass
class Child:
    registration: Registration
    owner: OwnedGroup
    stdout: int
    stderr: int
    stopped: Stopped | None = field(default=None)

    @property
    def pid(self) -> int:
        return self.registration.pid

    def exited(self) -> bool:
        """True once the leader has exited; it stays unreaped until `stop`."""
        return self.owner.exited()

    @property
    def process(self) -> psutil.Process:
        """Read-only sampling view; lifecycle belongs to the owner."""
        return psutil.Process(self.pid)

    def output(self) -> tuple[bytes, bytes]:
        return _read_all(self.stdout), _read_all(self.stderr)


def _read_all(descriptor: int) -> bytes:
    size = os.fstat(descriptor).st_size
    return os.pread(descriptor, min(size, OUTPUT_BYTES_MAX), 0)


class Supervisor:
    """Start, record, wait for and stop children; `stop_all` on leaving the block."""

    def __init__(
        self,
        tree: NewTree,
        run: RunDir,
        build: Build,
        cancellation: Cancellation,
        progress: Progress,
    ) -> None:
        self._tree = tree
        self._run = run
        self._build = build
        self._cancellation = cancellation
        self._progress = progress
        self._registry = tree.open_new(tree.root, PurePosixPath(REGISTRY_NAME))
        self.children: list[Child] = []
        self.cleanup_errors: list[StopError] = []
        self._closed = False

    def _record(self, line: Mapping[str, object]) -> None:
        os.write(self._registry, (json.dumps(line, sort_keys=True) + "\n").encode())
        os.fsync(self._registry)

    def spawn(self, name: str, argv: Sequence[str]) -> Child:
        """Start a gated launcher in the held run directory, record it, then let it run."""
        if len(self.children) >= CHILDREN_MAX:
            raise RuntimeError(f"more than {CHILDREN_MAX} children")
        number = len(self.children)
        stdout = self._tree.open_new(self._tree.root, PurePosixPath(f"{number:04d}-{name}.out"))
        stderr = self._tree.open_new(self._tree.root, PurePosixPath(f"{number:04d}-{name}.err"))
        # The signal handler only records, so nothing interrupts the steps
        # between starting the child and recording it.
        # WAIVER(TOCTOU): the child starts in the held run directory, but execve takes
        # the interpreter by name, and the child opens its RECALL_*, TMPDIR and HF_HOME
        # paths by name for its lifetime. Replacing or aliasing one of those names
        # after these checks redirects the child: a link or bind mount at the name, or
        # the run's path re-created as another directory. Operators must not repurpose
        # a live run or build. `check` re-proves the run while the child lives, which
        # bounds how long a redirected child goes unnoticed, not what its names reach.
        # Permanent: closing it needs an OS sandbox (plan section 8, option c).
        with self._run.held() as root:
            self._build.recheck()
            with root.as_working_directory():
                owner = OwnedGroup.start(
                    list(argv),
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                    env=child_environment(self._build, self._run),
                )
        registration = Registration(name, owner.pid, owner.create_time, tuple(argv))
        child = Child(registration, owner, stdout, stderr)
        self.children.append(child)
        self._record(
            {
                "event": "started",
                "name": name,
                "pid": registration.pid,
                "pgid": registration.pgid,
                "create_time": registration.create_time,
                "argv": list(argv),
            }
        )
        owner.release_stdin(b"g")
        self._progress({"event": "started", "name": name, "pid": registration.pid})
        return child

    def wait(self, child: Child, seconds: float, deadline: Deadline) -> bool:
        """Wait up to `seconds` for the child to exit; True if it did.

        A run replaced under a stalled child is refused within
        RUN_CHECK_SECONDS rather than at the child's timeout, and an exited
        child's result is never taken from a run that is no longer this one.
        The child stays recorded, so leaving the supervisor stops its group.
        """
        until = time.monotonic() + min(seconds, max(deadline.remaining(), 0.0))
        checked = time.monotonic()
        while not child.exited():
            self._cancellation.check()
            now = time.monotonic()
            if now >= until:
                return False
            if now - checked >= RUN_CHECK_SECONDS:
                self.check()
                checked = now
            time.sleep(POLL_SECONDS)
        self.check()
        return True

    def check(self) -> None:
        """Raise at a supervision point if the job was cancelled, its deadline
        passed, or the run directory is no longer this run."""
        self._cancellation.check()
        with self._run.held():
            pass

    def wait_quietly(self, seconds: float, deadline: Deadline) -> None:
        """Pause between polls, still answering cancellation and the deadline."""
        until = time.monotonic() + min(seconds, max(deadline.remaining(), 0.0))
        while time.monotonic() < until:
            self._cancellation.check()
            time.sleep(min(POLL_SECONDS, max(until - time.monotonic(), 0.0)))
        self._cancellation.check()

    def run(
        self, name: str, argv: Sequence[str], *, timeout_seconds: float, deadline: Deadline
    ) -> Completed:
        """Run one short child to completion, or stop it at its timeout."""
        started = time.monotonic()
        child = self.spawn(name, argv)
        finished = self.wait(child, timeout_seconds, deadline)
        seconds = time.monotonic() - started
        stopped = self.stop(child)
        if not stopped.clean:
            raise ChildStopFailed(stopped)
        stdout, stderr = child.output()
        return Completed(stopped.status if finished else None, stdout, stderr, seconds)

    def stop(self, child: Child) -> Stopped:
        if child.stopped is not None:
            return child.stopped
        return self._stop([child])[0]

    def stop_all(self) -> list[Stopped]:
        """Stop the registry together and finish fallible terminal bookkeeping."""
        self._stop([child for child in reversed(self.children) if child.stopped is None])
        if not self._closed:
            self._closed = True
            for descriptor in (
                *(fd for child in self.children for fd in (child.stdout, child.stderr)),
                self._registry,
            ):
                try:
                    os.close(descriptor)
                except OSError as error:
                    self.cleanup_errors.append(StopError("close", str(error), error.errno))
        return [child.stopped for child in reversed(self.children) if child.stopped is not None]

    def _stop(self, children: list[Child]) -> list[Stopped]:
        with Cancellation.deferred():
            results = stop_groups(
                [child.owner for child in children],
                grace_seconds=STOP_GRACE_SECONDS,
                proof_seconds=GONE_SECONDS,
            )
            for child, result in zip(children, results, strict=True):
                status = result.status
                child.stopped = Stopped(
                    child.registration.name,
                    child.pid,
                    128 - status if status is not None and status < 0 else status,
                    result.signals,
                    result.group_gone,
                    result.clean,
                    result.evidence,
                    result.errors,
                    result.unresolved if isinstance(result, FailedStop) else (),
                )
                try:
                    self._record({"event": "stopped", **asdict(child.stopped)})
                except OSError as error:
                    failure = StopError("registry", str(error), error.errno)
                    self.cleanup_errors.append(failure)
                    child.stopped = replace(
                        child.stopped, clean=False, errors=(*child.stopped.errors, failure)
                    )
        return [child.stopped for child in children if child.stopped is not None]

    def __enter__(self) -> Supervisor:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop_all()
