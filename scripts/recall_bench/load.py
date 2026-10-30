"""Concurrent recall CLI callers and scheduled appends, driven from one loop.

Every caller is a sequence of allowed commands run one at a time; many
callers run at once. One loop starts each caller's next command when its
previous one has exited, makes every append that is due, and samples the
daemon. A call's duration runs from just before its child is started to the
first poll that sees it exited, so it includes CLI startup and output and is
an upper bound by at most one loop turn; the largest loop turn is reported
with the calls.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .cli import Command, start_cli
from .evidence import JobFailed, JobIncomplete
from .fs import NewTree
from .isolation import Build, RunDir
from .samplers import Sample, sample
from .supervisor import Child, Deadline, Supervisor
from .workload import Appender, Transcript, Write

LOOP_SECONDS = 0.005
CHECK_SECONDS = 1.0
SAMPLE_SECONDS = 1.0
CALL_TIMEOUT_SECONDS = 60.0
OUTPUT_TAIL_BYTES = 2000

# A caller's next command, or None when it has no more.
Next = Callable[[], Command | None]


@dataclass(frozen=True)
class Call:
    caller: str
    phase: str
    argv: tuple[str, ...]
    started: float
    seconds: float
    status: int
    error: str | None
    document: Mapping[str, object] | None = field(default=None, compare=False, repr=False)

    @property
    def succeeded(self) -> bool:
        return self.status == 0 and self.error is None

    def to_json(self) -> dict[str, object]:
        return {
            "caller": self.caller,
            "phase": self.phase,
            "argv": list(self.argv),
            "started": self.started,
            "seconds": self.seconds,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class Appended:
    """One write as made: where it ended in its transcript, and when."""

    write: Write
    path: str
    end_size: int
    at: float


@dataclass
class Running:
    caller: str
    command: Command
    child: Child
    started: float


class Driver:
    """Owns the loop; every child it starts goes through the job's supervisor."""

    def __init__(
        self,
        supervisor: Supervisor,
        build: Build,
        run: RunDir,
        tree: NewTree,
        deadline: Deadline,
        daemon: Child,
        validate: Callable[[Command, Mapping[str, object]], str | None],
    ) -> None:
        self.supervisor = supervisor
        self.build = build
        self.run = run
        self.deadline = deadline
        self.daemon = daemon
        # Judges a parsed reply; a reason makes the call a failure.
        self.validate = validate
        self.calls: list[Call] = []
        self.appended: list[Appended] = []
        self.samples: list[Sample] = []
        self.loop_seconds_max = 0.0
        self._calls = tree.open_new(tree.root, PurePosixPath("calls.jsonl"))
        self._appends = tree.open_new(tree.root, PurePosixPath("appends.jsonl"))
        self._samples = tree.open_new(tree.root, PurePosixPath("samples.jsonl"))
        self._checked = 0.0
        self._sampled = 0.0

    def close(self) -> None:
        for descriptor in (self._calls, self._appends, self._samples):
            os.close(descriptor)

    def drive(
        self,
        phase: str,
        callers: Mapping[str, Next],
        *,
        appender: Appender | None = None,
        ticks: range = range(0),
        interval_seconds: float = 0.0,
        until: Callable[[float], bool] | None = None,
        timeout_seconds: float,
        observe: Callable[[Call], None] | None = None,
    ) -> list[Call]:
        """Run `callers` and the writes of `ticks`; return this phase's calls.

        Tick `ticks[k]` is due `k * interval_seconds` after the phase starts.
        A phase with writes or an `until` condition (given monotonic now) is
        bounded: callers stop starting commands once every write is made and
        `until` holds, and the commands already running finish. An unbounded
        phase ends when every caller has no more commands.
        """
        started = time.monotonic()
        bounded = bool(ticks) or until is not None
        running: dict[str, Running] = {}
        finished: set[str] = set()
        phase_calls: list[Call] = []
        last_turn = started
        while True:
            now = time.monotonic()
            self.loop_seconds_max = max(self.loop_seconds_max, now - last_turn)
            last_turn = now
            if now - started > timeout_seconds:
                raise JobIncomplete(
                    "phase_timeout", f"{phase} did not finish in {timeout_seconds} s"
                )
            self._supervise(now)
            writes_done = self._write_due(appender, ticks, interval_seconds, now - started)
            starting = not bounded or not (writes_done and (until is None or until(now)))
            for name, next_command in callers.items():
                item = running.get(name)
                if item is not None and item.child.exited():
                    call = self._finish(phase, item)
                    phase_calls.append(call)
                    if observe is not None and call.document is not None:
                        observe(call)
                    del running[name]
                    item = None
                if item is None and starting and name not in finished:
                    command = next_command()
                    if command is None:
                        finished.add(name)
                    else:
                        running[name] = self._start(name, command)
                elif item is not None and now - item.started > CALL_TIMEOUT_SECONDS:
                    self.supervisor.stop(item.child)
            callers_done = len(finished) == len(callers) or (bounded and not starting)
            if not running and writes_done and callers_done:
                return phase_calls
            self.supervisor.wait_quietly(LOOP_SECONDS, self.deadline)

    def _write_due(
        self, appender: Appender | None, ticks: range, interval_seconds: float, elapsed: float
    ) -> bool:
        """Make every write due by now; True once this phase's writes are all made."""
        if appender is None or not ticks:
            return True
        assert interval_seconds > 0, interval_seconds
        due_tick = ticks[min(int(elapsed / interval_seconds), len(ticks) - 1)]
        for write in appender.due(due_tick):
            self._append(appender, write)
        return (
            appender.done >= len(appender.writes) or appender.writes[appender.done].tick > ticks[-1]
        )

    def _supervise(self, now: float) -> None:
        if self.daemon.exited():
            raise JobFailed("daemon_exited", "the daemon exited while load was running")
        if now - self._checked >= CHECK_SECONDS:
            self.supervisor.check()
            self._checked = now
        if now - self._sampled >= SAMPLE_SECONDS:
            item = sample(self.daemon.process)
            self.samples.append(item)
            _append_line(self._samples, item.to_json())
            self._sampled = now

    def _append(self, appender: Appender, write: Write) -> None:
        session: Transcript = appender.write(write)
        at = time.monotonic()
        item = Appended(write, str(self.run.path / session.relative), session.size, at)
        self.appended.append(item)
        _append_line(
            self._appends,
            {
                "index": write.index,
                "appender": write.appender,
                "tick": write.tick,
                "path": item.path,
                "lines": write.line_count,
                "bytes": write.byte_count,
                "end_size": item.end_size,
                "at": at,
            },
        )

    def _start(self, name: str, command: Command) -> Running:
        started = time.monotonic()
        child = start_cli(self.supervisor, self.build, self.run, command)
        return Running(name, command, child, started)

    def _finish(self, phase: str, item: Running) -> Call:
        seconds = time.monotonic() - item.started
        stopped, stdout, stderr = self.supervisor.finish(item.child)
        error: str | None = None
        document: Mapping[str, object] | None = None
        if stopped.status != 0:
            # The CLI prints its error document on stdout and notes on stderr.
            output = (stdout.strip() or stderr.strip()).decode(errors="replace")
            error = output[-OUTPUT_TAIL_BYTES:] or "no output"
        else:
            try:
                parsed = json.loads(stdout)
            except json.JSONDecodeError:
                error = "printed no JSON"
            else:
                if isinstance(parsed, dict):
                    document = parsed
                    error = self.validate(item.command, parsed)
                else:
                    error = "printed no JSON object"
        call = Call(
            item.caller,
            phase,
            item.command.argv,
            item.started,
            seconds,
            stopped.status,
            error,
            document,
        )
        self.calls.append(call)
        _append_line(self._calls, call.to_json())
        return call


def _append_line(descriptor: int, line: Mapping[str, object]) -> None:
    os.write(descriptor, (json.dumps(line, sort_keys=True, default=str) + "\n").encode())


def repeat(command: Command) -> Next:
    return lambda: command


def once(commands: Sequence[Command]) -> Iterator[Next]:
    """One caller per command, each running it exactly once."""
    for command in commands:
        remaining = [command]
        yield lambda remaining=remaining: remaining.pop() if remaining else None


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank: the smallest value with at least `fraction` of the values at or below it."""
    assert values and 0 < fraction <= 1, (len(values), fraction)
    ordered = sorted(values)
    rank = max(1, math.ceil(round(fraction * len(ordered), 9)))
    return ordered[rank - 1]


def seconds_summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def rss_bytes_max(samples: Sequence[Sample]) -> int:
    return max((item.rss_bytes for item in samples), default=0)


def live_sizes(document: Mapping[str, object]) -> dict[str, tuple[int, str | None]]:
    """Each live session's path, committed (indexed) size and recall id."""
    sessions = document.get("sessions")
    sizes: dict[str, tuple[int, str | None]] = {}
    if not isinstance(sessions, list):
        return sizes
    for session in sessions:
        if not isinstance(session, dict):
            continue
        freshness = session.get("freshness")
        path, identifier = session.get("path"), session.get("id")
        if isinstance(freshness, dict) and isinstance(path, str):
            indexed = freshness.get("indexed_size")
            if isinstance(indexed, int):
                sizes[path] = (indexed, identifier if isinstance(identifier, str) else None)
    return sizes


class Freshness:
    """When each append was first seen committed by a live page.

    A page whose snapshot shows a transcript's committed size at or past an
    append's end has seen that append; its call's end is an upper bound on
    when the snapshot was taken, so freshness is reported as that bound.
    """

    def __init__(self, appended: list[Appended]) -> None:
        self.appended = appended
        self.seen: dict[int, float] = {}
        self.identifiers: dict[str, str] = {}
        self.committed: dict[str, int] = {}

    def observe(self, call: Call) -> None:
        assert call.document is not None
        ended = call.started + call.seconds
        for path, (size, identifier) in live_sizes(call.document).items():
            if identifier is not None:
                self.identifiers[path] = identifier
            self.committed[path] = max(self.committed.get(path, 0), size)
        for item in self.appended:
            if item.write.index in self.seen:
                continue
            if self.committed.get(item.path, 0) >= item.end_size and ended >= item.at:
                self.seen[item.write.index] = ended - item.at

    def unseen(self) -> list[Appended]:
        return [item for item in self.appended if item.write.index not in self.seen]

    def seconds(self) -> list[float]:
        return [
            self.seen[item.write.index] for item in self.appended if item.write.index in self.seen
        ]
