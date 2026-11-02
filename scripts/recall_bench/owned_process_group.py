"""Launch-owned group signalling, pre-reap completion evidence and bounded reaping.

Only the retained, unreaped session leader authorizes group signals. Membership
observations never authorize signals to enumerated PIDs. Linux inventory is not
an atomic snapshot; its behind-both-cursors birth race requires specialist
acceptance before a Linux build can be qualified for landing.
"""

from __future__ import annotations

import ctypes
import errno
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import psutil

GROUP_SNAPSHOT_LIMIT = 65_536
STAT_BYTES_MAX = 4096
POLL_SECONDS = 0.01
OWNERS_MAX = 10_000


@dataclass(frozen=True)
class StopError:
    phase: str
    message: str
    errno: int | None = None


@dataclass(frozen=True)
class CompletionEvidence:
    kind: Literal["esrch", "darwin_zombies", "linux_inventory"]
    members: tuple[int, ...]


@dataclass(frozen=True)
class CleanStop:
    status: int
    evidence: CompletionEvidence
    signals: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return True

    @property
    def group_gone(self) -> bool:
        return True

    @property
    def errors(self) -> tuple[StopError, ...]:
        return ()


@dataclass(frozen=True)
class FailedStop:
    status: int | None
    evidence: CompletionEvidence | None
    signals: tuple[str, ...]
    errors: tuple[StopError, ...]
    unresolved: tuple[tuple[int, float], ...]

    @property
    def clean(self) -> bool:
        return False

    @property
    def group_gone(self) -> bool:
        return self.evidence is not None


StopResult = CleanStop | FailedStop


class StopFailed(RuntimeError):
    def __init__(self, result: FailedStop) -> None:
        self.result = result
        super().__init__("; ".join(error.message for error in result.errors))


@dataclass(frozen=True)
class ProcStat:
    pid: int
    state: str
    pgid: int
    session: int
    threads: int
    start: int


def _parse_stat(data: bytes) -> ProcStat:
    if not data.endswith(b"\n") or len(data) > STAT_BYTES_MAX:
        raise ValueError("truncated or oversized proc stat")
    opening = data.find(b" (")
    closing = data.rfind(b") ")
    if opening <= 0 or closing <= opening:
        raise ValueError("invalid proc stat delimiters")
    fields = data[closing + 2 :].split()
    if len(fields) != 50:
        raise ValueError("unsupported or incomplete proc stat layout")
    record = ProcStat(
        int(data[:opening]),
        fields[0].decode("ascii"),
        int(fields[2]),
        int(fields[3]),
        int(fields[17]),
        int(fields[19]),
    )
    if record.pid <= 0 or record.pgid < 0 or record.session < 0 or record.threads < 1:
        raise ValueError("invalid proc stat identity")
    if record.state not in {"R", "S", "D", "Z", "T", "t", "X", "x", "K", "W", "P", "I"}:
        raise ValueError("unknown proc stat state")
    return record


def _read_stat(pid: int) -> ProcStat:
    with (Path("/proc") / str(pid) / "stat").open("rb") as source:
        return _parse_stat(source.read(STAT_BYTES_MAX + 1))


def _darwin_snapshot(pgid: int) -> set[int]:
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    list_group = libproc.proc_listpgrppids
    list_group.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    list_group.restype = ctypes.c_int
    pids = (ctypes.c_int * GROUP_SNAPSHOT_LIMIT)()
    count = list_group(pgid, pids, ctypes.sizeof(pids))
    if not 0 < count < len(pids):
        return set()
    return set(pids[:count])


class OwnedGroup:
    """A session created here, with no public raw signalling or reaping handle."""

    _child: subprocess.Popen[bytes]
    _process: psutil.Process
    _birth: float
    _linux_start: int | None
    _identities: dict[int, int]
    _members: dict[int, float]
    _inventory_previous: dict[int, int] | None
    _evidence: CompletionEvidence | None
    _errors: list[StopError]
    _signals: list[str]
    _denial: StopError | None
    _signalling: bool
    _authorized: bool
    _result: StopResult | None

    def __init__(self) -> None:
        raise TypeError("use OwnedGroup.start to obtain a launch-owned capability")

    @classmethod
    def start(
        cls,
        argv: Sequence[str],
        *,
        stdin: int | None = None,
        stdout: int | None = None,
        stderr: int | None = None,
        env: Mapping[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
        preexec_fn: Callable[[], None] | None = None,
    ) -> OwnedGroup:
        owner = object.__new__(cls)
        owner._child = subprocess.Popen(
            list(argv),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=env,
            pass_fds=pass_fds,
            preexec_fn=preexec_fn,
            start_new_session=True,
        )
        owner._process = psutil.Process(owner._child.pid)
        owner._birth = owner._process.create_time()
        owner._linux_start = None
        owner._identities = {}
        owner._members = {owner.pid: owner._birth}
        owner._inventory_previous = None
        owner._evidence = None
        owner._errors = []
        owner._signals = []
        owner._denial = None
        owner._signalling = True
        owner._authorized = True
        owner._result = None
        if sys.platform == "linux":
            try:
                leader = _read_stat(owner.pid)
                if (
                    leader.pid != owner.pid
                    or leader.pgid != owner.pid
                    or leader.session != owner.pid
                ):
                    raise ValueError("launched child is not its session leader")
                owner._linux_start = leader.start
                owner._identities[owner.pid] = leader.start
            except (OSError, ValueError) as error:
                owner._error("launch_inventory", error)
        return owner

    @property
    def pid(self) -> int:
        return self._child.pid

    @property
    def create_time(self) -> float:
        return self._birth

    def release_stdin(self, data: bytes) -> None:
        assert self._child.stdin is not None
        with suppress(BrokenPipeError):
            try:
                self._child.stdin.write(data)
            finally:
                self._child.stdin.close()

    def exited(self) -> bool:
        if self._result is not None and self._result.status is not None:
            return True
        return self._process.status() == psutil.STATUS_ZOMBIE

    def _error(self, phase: str, error: Exception) -> None:
        self._errors.append(StopError(phase, str(error)[:4096], getattr(error, "errno", None)))

    def _identity(self) -> None:
        if psutil.Process(self.pid).create_time() != self._birth:
            raise ProcessLookupError("retained leader birth identity changed")

    def _send(self, signum: signal.Signals) -> None:
        assert self._signalling and self._result is None
        if not self._authorized or self._evidence is not None:
            return
        try:
            self._identity()
        except (OSError, psutil.Error) as error:
            self._authorized = False
            self._error("ownership", error)
            return
        try:
            os.killpg(self.pid, signum)
        except ProcessLookupError:
            self._evidence = CompletionEvidence("esrch", ())
        except PermissionError as error:
            if sys.platform == "darwin" and error.errno == errno.EPERM:
                self._denial = StopError(signum.name, str(error)[:4096], error.errno)
            else:
                self._error(signum.name, error)
        except OSError as error:
            self._error(signum.name, error)
        else:
            self._signals.append(signum.name)

    def _darwin_proof(self, deadline: float) -> CompletionEvidence | None:
        members = _darwin_snapshot(self.pid)
        if self.pid not in members:
            return None
        zombies: dict[int, psutil.Process] = {}
        for pid in members:
            if time.monotonic() >= deadline:
                return None
            try:
                process = psutil.Process(pid)
                birth = process.create_time()
                previous = self._members.setdefault(pid, birth)
                if birth != previous or process.status() != psutil.STATUS_ZOMBIE:
                    return None
                zombies[pid] = process
            except psutil.NoSuchProcess:
                continue
        remaining = _darwin_snapshot(self.pid)
        if self.pid not in remaining or not remaining <= zombies.keys():
            return None
        if not all(zombies[pid].is_running() for pid in remaining):
            return None
        self._identity()
        return CompletionEvidence("darwin_zombies", tuple(sorted(remaining)))

    def _linux_walk(self, deadline: float) -> dict[int, int] | None:
        members: dict[int, int] = {}
        count = 0
        with os.scandir("/proc") as entries:
            for entry in entries:
                if not entry.name.isdecimal():
                    continue
                count += 1
                if count > GROUP_SNAPSHOT_LIMIT or time.monotonic() >= deadline:
                    return None
                try:
                    record = _read_stat(int(entry.name))
                except FileNotFoundError:
                    continue
                if record.pid != int(entry.name):
                    return None
                if record.pgid != self.pid or record.session != self.pid:
                    continue
                previous = self._identities.setdefault(record.pid, record.start)
                if previous != record.start or record.state != "Z" or record.threads != 1:
                    return None
                members[record.pid] = record.start
        if not members or members.get(self.pid) != self._linux_start:
            return None
        return members

    def _linux_inventory(self, deadline: float) -> CompletionEvidence | None:
        members = self._linux_walk(deadline)
        previous = self._inventory_previous
        self._inventory_previous = members
        if members is None or members != previous:
            return None
        for pid, start in members.items():
            if time.monotonic() >= deadline:
                return None
            record = _read_stat(pid)
            if (
                record.start != start
                or record.state != "Z"
                or record.threads != 1
                or record.pgid != self.pid
                or record.session != self.pid
            ):
                return None
        self._identity()
        return CompletionEvidence("linux_inventory", tuple(sorted(members)))

    def _prove(self, deadline: float) -> None:
        if self._evidence is not None or not self._authorized:
            return
        try:
            if sys.platform == "darwin":
                evidence = self._darwin_proof(deadline)
            elif sys.platform == "linux":
                evidence = self._linux_inventory(deadline)
            else:
                raise OSError("unsupported group membership observation platform")
        except (OSError, ValueError, psutil.Error) as error:
            # A failed observation invalidates any consecutive inventory pair.
            self._inventory_previous = None
            if not any(item.phase == "proof" for item in self._errors):
                self._error("proof", error)
            return
        if evidence is not None and time.monotonic() < deadline:
            self._evidence = evidence
            self._denial = None

    def _finish_denial(self) -> None:
        if self._denial is not None:
            self._errors.append(self._denial)
            self._denial = None

    def _finish(self, deadline: float) -> StopResult:
        assert self._result is None and self._signalling
        self._signalling = False
        status: int | None = None
        try:
            if self.exited():
                status = self._child.wait(timeout=max(0, deadline - time.monotonic()))
            else:
                self._errors.append(StopError("reap", "retained leader is still live"))
        except (OSError, psutil.Error, subprocess.TimeoutExpired) as error:
            self._error("reap", error)
        if self._evidence is None:
            self._errors.append(StopError("proof", "group completion remains unproved"))
        if self._evidence is not None and status is not None and not self._errors:
            result: StopResult = CleanStop(status, self._evidence, tuple(self._signals))
        else:
            unresolved = () if self._evidence is not None else tuple(sorted(self._members.items()))
            result = FailedStop(
                status, self._evidence, tuple(self._signals), tuple(self._errors), unresolved
            )
        self._result = result
        return result


def _prove_until(owners: Sequence[OwnedGroup], deadline: float) -> None:
    while time.monotonic() < deadline:
        pending = [owner for owner in owners if owner._evidence is None and owner._authorized]
        if not pending:
            return
        for owner in pending:
            owner._prove(deadline)
        if all(owner._evidence is not None or not owner._authorized for owner in owners):
            return
        time.sleep(min(POLL_SECONDS, max(0, deadline - time.monotonic())))


def stop_groups(
    owners: Sequence[OwnedGroup],
    *,
    grace_seconds: float,
    proof_seconds: float,
) -> list[StopResult]:
    """Stop all owned groups within shared phase budgets; cache operating failures.

    Callers defer cancellation through their recording handlers. No operating
    error is raised here, and a terminal owner is never signalled again.
    """
    assert len(owners) <= OWNERS_MAX and grace_seconds >= 0 and proof_seconds > 0
    assert len({id(owner) for owner in owners}) == len(owners)
    active = [owner for owner in owners if owner._result is None]
    grace_deadline = time.monotonic() + grace_seconds
    for owner in active:
        owner._send(signal.SIGTERM)
    _prove_until(active, grace_deadline)
    for owner in active:
        owner._finish_denial()
    proof_deadline = time.monotonic() + proof_seconds
    for owner in active:
        owner._send(signal.SIGKILL)
    _prove_until(active, proof_deadline)
    for owner in active:
        owner._finish_denial()
        owner._finish(proof_deadline)
    results = [owner._result for owner in owners]
    assert all(result is not None for result in results)
    return [result for result in results if result is not None]
