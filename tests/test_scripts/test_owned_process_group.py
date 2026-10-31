"""The launch owner's contract, with real groups and controlled syscall windows."""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import suppress
from pathlib import Path

import psutil
import pytest

from scripts.recall_bench import owned_process_group as groups

WAIT = "import os; print(os.getpid(), flush=True); os.read(0, 1)"


def until(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "owned child did not reach its gate"
        time.sleep(0.001)


@pytest.fixture
def launched(tmp_path: Path) -> Iterator[Callable[[str], groups.OwnedGroup]]:
    children: list[tuple[groups.OwnedGroup, Path]] = []

    def start(code: str = WAIT) -> groups.OwnedGroup:
        output = tmp_path / f"{len(children)}.out"
        with output.open("wb") as stream:
            owner = groups.OwnedGroup.start(
                [sys.executable, "-I", "-u", "-c", code],
                stdin=subprocess.PIPE,
                stdout=stream.fileno(),
                stderr=stream.fileno(),
            )
        children.append((owner, output))
        until(lambda: output.exists() and output.read_text().endswith("\n"))
        return owner

    yield start
    # Denial tests deliberately leave real survivors. Never signal a terminal
    # PGID: revalidate and clean each individually recorded child identity.
    for owner, output in children:
        identities = [(owner.pid, owner.create_time)]
        for text in output.read_text().splitlines()[1:]:
            if text.startswith("descendant "):
                pid, birth = text.split()[1:]
                identities.append((int(pid), float(birth)))
        for pid, birth in reversed(identities):
            with suppress(psutil.NoSuchProcess):
                process = psutil.Process(pid)
                assert process.create_time() == birth
                if process.status() != psutil.STATUS_ZOMBIE:
                    process.kill()
        owner._child.wait(timeout=5)


def stop(owner: groups.OwnedGroup) -> groups.StopResult:
    return groups.stop_groups([owner], grace_seconds=5.0, proof_seconds=3.0)[0]


def bounded_failure(owner: groups.OwnedGroup) -> groups.StopResult:
    return groups.stop_groups([owner], grace_seconds=0.05, proof_seconds=0.5)[0]


@pytest.mark.parametrize("status", [0, 7])
def test_actual_status_and_terminal_owner_never_signals_again(
    launched,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    owner = launched(WAIT + f"; raise SystemExit({status})")
    owner.release_stdin(b"g")
    until(owner.exited)
    signal_group = os.killpg

    def owned_signal(pgid: int, signum: int) -> None:
        assert pgid == owner.pid
        assert psutil.Process(pgid).create_time() == owner.create_time
        assert owner._child.returncode is None, "signal after reap"
        assert signum != 0
        signal_group(pgid, signum)

    monkeypatch.setattr(os, "killpg", owned_signal)
    result = stop(owner)
    assert isinstance(result, groups.CleanStop) and result.status == status
    assert result.group_gone and result.evidence.kind in {"darwin_zombies", "linux_inventory"}
    assert stop(owner) is result


@pytest.mark.skipif(sys.platform != "darwin", reason="native XNU zombie group EPERM")
def test_native_term_exit_window_keeps_real_eperm_clean(launched, monkeypatch) -> None:
    owner = launched(WAIT)
    send = os.killpg
    errors = []

    def at_term(pgid, signum):
        assert owner._child.returncode is None, "signal after reap"
        if signum == signal.SIGTERM:
            owner.release_stdin(b"g")
            until(owner.exited)
        try:
            send(pgid, signum)
        except PermissionError as error:
            errors.append(error.errno)
            raise

    monkeypatch.setattr(os, "killpg", at_term)
    # Use the supervisor's actual phase budgets: the controlled scheduler
    # must release a live process before the native syscall can return.
    result = groups.stop_groups([owner], grace_seconds=5.0, proof_seconds=3.0)[0]
    assert errors == [errno.EPERM], errors
    assert isinstance(result, groups.CleanStop) and result.status == 0
    assert result.evidence.kind == "darwin_zombies"


@pytest.mark.parametrize("denied_signal", [signal.SIGTERM, signal.SIGKILL])
@pytest.mark.parametrize("exited_leader", [False, True])
def test_live_denial_is_bounded_failed_and_never_claims_gone(
    launched,
    monkeypatch,
    denied_signal,
    exited_leader,
) -> None:
    # The descendant records its own birth identity before the leader exits.
    descendant = (
        "import os, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('descendant', os.getpid(), "
        "__import__('psutil').Process().create_time(), flush=True); "
        "signal.pause()"
    )
    code = (
        (
            "import os, signal, subprocess, sys; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"p = subprocess.Popen([sys.executable, '-I', '-u', '-c', {descendant!r}], "
            "stdout=subprocess.PIPE); "
            "row = p.stdout.readline(); "
            "print(str(os.getpid()) + '\\n' + row.decode().strip(), flush=True); "
            "os.read(0, 1)"
        )
        if exited_leader
        else ("import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); " + WAIT)
    )
    owner = launched(code)
    if exited_leader:
        owner.release_stdin(b"g")
        until(owner.exited)
    send = os.killpg

    def denied(pgid, signum):
        assert owner._child.returncode is None, "signal after reap"
        # Keep a live member for the whole proof window, including KILL.
        if signum in {denied_signal, signal.SIGKILL}:
            raise PermissionError(errno.EPERM, "live member denied")
        send(pgid, signum)

    monkeypatch.setattr(os, "killpg", denied)
    started = time.monotonic()
    result = bounded_failure(owner)
    assert time.monotonic() - started < 2
    assert isinstance(result, groups.FailedStop) and not result.group_gone
    assert any(error.errno == errno.EPERM for error in result.errors)
    assert result.status == (0 if exited_leader else None)
    assert stop(owner) is result


@pytest.mark.parametrize("error_number", [errno.EPERM, errno.EACCES])
def test_denial_remains_failed_after_real_kill_cleans_group(launched, monkeypatch, error_number):
    owner = launched(WAIT)
    send = os.killpg

    def denied_term(pgid, signum):
        if signum == signal.SIGTERM:
            raise PermissionError(error_number, "term denied")
        send(pgid, signum)

    monkeypatch.setattr(os, "killpg", denied_term)
    result = stop(owner)
    assert isinstance(result, groups.FailedStop) and result.group_gone
    assert result.status == -signal.SIGKILL
    assert any(error.errno == error_number for error in result.errors)


def test_partial_delivery_cannot_prove_a_live_descendant_gone(launched, monkeypatch):
    descendant = (
        "import os, signal, psutil; "
        "print('descendant', os.getpid(), psutil.Process().create_time(), flush=True); "
        "signal.pause()"
    )
    owner = launched(
        "import os, subprocess, sys; print(os.getpid(), flush=True); "
        f"subprocess.Popen([sys.executable, '-I', '-u', '-c', {descendant!r}]); os.read(0,1)"
    )
    until(lambda: len(psutil.Process(owner.pid).children()) == 1)
    owner.release_stdin(b"g")
    until(owner.exited)
    send = os.kill

    def partial(pgid, signum):
        assert owner._child.returncode is None
        # Simulate the kernel's partial group delivery: only the retained leader.
        with suppress(PermissionError):
            send(pgid, signum)

    monkeypatch.setattr(os, "killpg", partial)
    result = bounded_failure(owner)
    assert isinstance(result, groups.FailedStop) and not result.group_gone
    assert result.status == 0


def test_batch_attempts_every_group_and_shares_phase_deadlines(launched, monkeypatch):
    owners = [
        launched(WAIT),
        launched(WAIT),
        launched("import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); " + WAIT),
    ]
    send = os.killpg

    def deny_one(pgid, signum):
        if pgid == owners[0].pid:
            raise PermissionError(errno.EPERM, "first denied")
        send(pgid, signum)

    monkeypatch.setattr(os, "killpg", deny_one)
    started = time.monotonic()
    results = groups.stop_groups(owners, grace_seconds=5.0, proof_seconds=3.0)
    assert time.monotonic() - started < 10
    assert isinstance(results[0], groups.FailedStop) and not results[0].group_gone
    assert all(result.clean for result in results[1:])
    assert results[2].status == -signal.SIGKILL


@pytest.mark.parametrize("snapshot", [set(), {999999}])
def test_missing_leader_snapshot_is_unproved(launched, monkeypatch, snapshot):
    owner = launched(WAIT)
    owner.release_stdin(b"g")
    until(owner.exited)
    monkeypatch.setattr(groups, "_darwin_snapshot", lambda pgid: snapshot)
    assert owner._darwin_proof(time.monotonic() + 1) is None
    monkeypatch.undo()
    stop(owner)


def stat_record(*, comm: bytes = b"worker ) name", state: bytes = b"Z", threads: int = 1) -> bytes:
    # proc(5) fields 3..52; only the specified identity fields are nonzero.
    fields = [b"0"] * 50
    fields[0:4] = [state, b"1", b"123", b"123"]
    fields[17] = str(threads).encode()
    fields[19] = b"987"
    return b"123 (" + comm + b") " + b" ".join(fields) + b"\n"


def test_linux_stat_uses_last_parenthesis_and_named_fields():
    result = groups._parse_stat(stat_record())
    assert result == groups.ProcStat(123, "Z", 123, 123, 1, 987)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"123 (short) Z 1\n",
        b"x (bad) Z\n",
        b"123 (" + b"x" * 4096 + b") Z\n",
        b"123 bad Z\n",
        stat_record()[:-1],
    ],
)
def test_linux_stat_rejects_unclassifiable_records(data):
    with pytest.raises(ValueError):
        groups._parse_stat(data)


@pytest.mark.parametrize(
    "case",
    [
        "live",
        "threaded",
        "missing",
        "reused",
        "changed",
        "denied",
        "io",
        "disappeared",
        "overflow",
        "expired",
    ],
)
def test_linux_inventory_fails_closed_at_the_read_boundary(launched, tmp_path, monkeypatch, case):
    owner = launched(WAIT)
    owner.release_stdin(b"g")
    until(owner.exited)
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / str(owner.pid)).mkdir()
    (proc / "123").mkdir()
    native_start, native_identities = owner._linux_start, owner._identities
    owner._linux_start = 987
    owner._identities = {owner.pid: 987}
    scan = os.scandir
    monkeypatch.setattr(os, "scandir", lambda path: scan(proc) if path == "/proc" else scan(path))
    calls = 0

    def read(pid):
        nonlocal calls
        calls += 1
        if pid == owner.pid:
            return groups.ProcStat(pid, "Z", owner.pid, owner.pid, 1, 987)
        if case == "denied":
            raise PermissionError(errno.EACCES, "proc unreadable")
        if case == "io":
            raise OSError(errno.EIO, "proc read failed")
        if case == "disappeared":
            raise FileNotFoundError(errno.ENOENT, "exited after listing")
        return groups.ProcStat(
            pid,
            "S" if case == "live" else "Z",
            owner.pid,
            owner.pid,
            2 if case == "threaded" else 1,
            456,
        )

    monkeypatch.setattr(groups, "_read_stat", read)
    if case == "missing":
        (proc / str(owner.pid)).rmdir()
    if case == "reused":
        owner._identities[123] = 455
    if case == "overflow":
        monkeypatch.setattr(groups, "GROUP_SNAPSHOT_LIMIT", 1)
    deadline = time.monotonic() + (1 if case != "expired" else -1)
    if case in {"denied", "io"}:
        with pytest.raises(OSError):
            owner._linux_inventory(deadline)
    else:
        assert owner._linux_inventory(deadline) is None
        if case == "changed":
            (proc / "123").rmdir()
        second = owner._linux_inventory(deadline)
        if case == "disappeared":
            assert second is not None and second.kind == "linux_inventory"
        else:
            assert second is None
    # Restore the native adapter before the fixture reaps its owned child.
    owner._linux_start, owner._identities = native_start, native_identities
    monkeypatch.undo()
    stop(owner)


@pytest.mark.parametrize("case", ["live", "unseen", "reused", "unreadable"])
def test_darwin_proof_rejects_unproved_members(launched, monkeypatch, case):
    owner = launched(WAIT)
    if case != "live":
        owner.release_stdin(b"g")
        until(owner.exited)
    snapshots = iter([{owner.pid}, {owner.pid, 999999}])
    if case == "unseen":
        monkeypatch.setattr(groups, "_darwin_snapshot", lambda pgid: next(snapshots))
    else:
        monkeypatch.setattr(groups, "_darwin_snapshot", lambda pgid: {owner.pid})
    if case == "reused":
        owner._members[owner.pid] = owner.create_time - 1
    if case == "unreadable":

        def denied(pid):
            raise psutil.AccessDenied(pid)

        monkeypatch.setattr(psutil, "Process", denied)
        with pytest.raises(psutil.AccessDenied):
            owner._darwin_proof(time.monotonic() + 1)
    else:
        assert owner._darwin_proof(time.monotonic() + 1) is None
    monkeypatch.undo()
    # A mismatched retained identity correctly remains a failed stop.
    stop(owner)


@pytest.mark.skipif(sys.platform != "linux", reason="native procfs thread state")
def test_linux_zombie_main_with_live_worker_is_not_quiescent(launched):
    owner = launched(
        "import ctypes, os, signal, threading, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "threading.Thread(target=lambda: time.sleep(60)).start(); "
        "print(os.getpid(), flush=True); ctypes.CDLL(None).pthread_exit(None)"
    )
    until(owner.exited)
    record = groups._read_stat(owner.pid)
    assert record.threads > 1
    assert owner._linux_inventory(time.monotonic() + 1) is None
    assert stop(owner).status == -signal.SIGKILL


def test_proof_io_failure_is_a_result_and_reaps_available_leader(launched, monkeypatch):
    owner = launched(WAIT)
    owner.release_stdin(b"g")
    until(owner.exited)

    def unreadable(*args):
        raise OSError(errno.EIO, "membership query failed")

    if sys.platform == "darwin":
        monkeypatch.setattr(groups, "_darwin_snapshot", unreadable)
    else:
        monkeypatch.setattr(groups, "_read_stat", unreadable)
    result = bounded_failure(owner)
    assert isinstance(result, groups.FailedStop)
    assert result.status == 0 and not result.group_gone
    assert any(error.errno == errno.EIO for error in result.errors)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux EPERM is never discharged")
def test_linux_eperm_on_zombie_stays_failed_despite_inventory(launched, monkeypatch):
    owner = launched(WAIT)
    owner.release_stdin(b"g")
    until(owner.exited)

    def denied(*args):
        raise PermissionError(errno.EPERM, "Linux denial")

    monkeypatch.setattr(os, "killpg", denied)
    result = stop(owner)
    assert isinstance(result, groups.FailedStop) and result.group_gone
    assert result.evidence is not None and result.evidence.kind == "linux_inventory"
    assert result.status == 0 and result.errors[0].errno == errno.EPERM


@pytest.mark.parametrize("count", [-1, 0, 65536, 65537])
def test_darwin_snapshot_rejects_failed_empty_and_truncated_queries(monkeypatch, count):
    class Query:
        def __call__(self, pgid, buffer, size):
            return count

    class Library:
        proc_listpgrppids = Query()

    monkeypatch.setattr(groups.ctypes, "CDLL", lambda *args, **kwargs: Library())
    assert groups._darwin_snapshot(123) == set()


@pytest.mark.parametrize(
    "data",
    [stat_record().replace(b"987", b"bad"), stat_record(threads=-1), stat_record(state=b"?")],
)
def test_linux_stat_rejects_invalid_identity_fields(data):
    with pytest.raises(ValueError):
        groups._parse_stat(data)


@pytest.mark.skipif(sys.platform != "linux", reason="native Linux reparenting and fork inventory")
def test_linux_reparented_forking_descendant_keeps_inventory_unproved(launched):
    descendant = (
        "import os, signal, psutil; "
        "print('descendant', os.getpid(), psutil.Process().create_time(), flush=True)\n"
        "for _ in range(64):\n"
        "    child = os.fork()\n"
        "    if child == 0: os._exit(0)\n"
        "    os.waitpid(child, 0)\n"
        "signal.pause()\n"
    )
    owner = launched(
        "import os, subprocess, sys; "
        f"p = subprocess.Popen([sys.executable, '-I', '-u', '-c', {descendant!r}], "
        "stdout=subprocess.PIPE); row = p.stdout.readline(); "
        "print(str(os.getpid()) + '\\n' + row.decode().strip(), flush=True); os.read(0,1)"
    )
    owner.release_stdin(b"g")
    until(owner.exited)
    assert owner._linux_inventory(time.monotonic() + 1) is None
    assert stop(owner).clean
