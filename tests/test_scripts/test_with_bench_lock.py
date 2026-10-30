"""Drive the bench runner with real processes, kernel locks and controlled work."""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

import psutil
import pytest

RUNNER = Path(__file__).resolve().parents[2] / "scripts/with_bench_lock.py"
WORK = (
    "import json, os, sys; "
    "print(json.dumps({'pid': os.getpid()}), flush=True); "
    "sys.stdin.readline(); "
    "print('finished', flush=True)"
)


class Run:
    def __init__(
        self,
        lock: Path,
        label: str,
        code: str = WORK,
        timeout: float = 10,
        runner: Path = RUNNER,
    ) -> None:
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(runner),
                "--lock",
                str(lock),
                "--timeout",
                str(timeout),
                label,
                sys.executable,
                "-c",
                code,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self.child_pid: int | None = None
        self.child: psutil.Process | None = None

    def line(self, channel: str) -> str:
        pipe = getattr(self.process, channel)
        assert pipe is not None
        deadline = time.monotonic() + 10
        line = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(pipe, selectors.EVENT_READ)
            while not line.endswith(b"\n"):
                assert selector.select(timeout=max(0, deadline - time.monotonic())), (
                    f"no {channel} from {self.process.pid}"
                )
                byte = pipe.read(1)
                assert byte, f"unexpected EOF from {self.process.pid} (exit {self.process.poll()})"
                line.extend(byte)
        return line.decode().strip()

    def event(self, expected: str) -> dict:
        event = json.loads(self.line("stderr"))
        assert event["lock"] == expected, event
        return event

    def ready(self) -> None:
        self.event("acquired")
        self.child_pid = json.loads(self.line("stdout"))["pid"]
        self.child = psutil.Process(self.child_pid)

    def finish(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(b"go\n")
        self.process.stdin.flush()
        assert self.line("stdout") == "finished"
        assert self.process.wait(timeout=10) == 0

    def cleanup(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGCONT)
            self.process.terminate()
            self.process.wait(timeout=15)
        if self.child is not None and self.child.is_running():
            with suppress(psutil.NoSuchProcess):
                self.child.kill()
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe is not None:
                pipe.close()


@pytest.fixture
def runs() -> Iterator[list[Run]]:
    launched: list[Run] = []
    try:
        yield launched
    finally:
        for run in reversed(launched):
            run.cleanup()


def launch(runs: list[Run], lock: Path, label: str, **kwargs) -> Run:
    run = Run(lock, label, **kwargs)
    runs.append(run)
    run.event("queued")
    return run


@pytest.fixture
def owned_group_runner(tmp_path: Path) -> Path:
    """Reject an unsafe group signal before it can reach a reused PGID."""
    runner = tmp_path / "owned_group_runner.py"
    runner.write_text(
        "import errno, os, runpy, psutil\n"
        "killpg = os.killpg\n"
        "def signal_group(pgid, signum):\n"
        "    try:\n"
        "        leader = psutil.Process(pgid)\n"
        "        owned = leader.ppid() == os.getpid()\n"
        "    except psutil.NoSuchProcess:\n"
        "        owned = False\n"
        "    if not owned:\n"
        "        raise PermissionError(errno.EPERM, 'group leader already reaped')\n"
        "    killpg(pgid, signum)\n"
        "os.killpg = signal_group\n"
        f"runpy.run_path({str(RUNNER)!r}, run_name='__main__')\n"
    )
    return runner


def test_success_never_signals_a_reaped_group(
    tmp_path: Path, runs: list[Run], owned_group_runner: Path
) -> None:
    run = launch(
        runs,
        tmp_path / "queue",
        "success",
        code="raise SystemExit(0)",
        runner=owned_group_runner,
    )
    run.event("acquired")
    assert run.event("released")["child_status"] == 0
    assert run.process.wait(timeout=10) == 0


def test_exited_leader_descendants_are_stopped(
    tmp_path: Path, runs: list[Run], owned_group_runner: Path
) -> None:
    descendant_code = (
        "import json, os, signal; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print(json.dumps({'descendant': os.getpid()}), flush=True); "
        "signal.pause()"
    )
    code = (
        "import json, os, subprocess, sys\n"
        "print(json.dumps({'pid': os.getpid()}), flush=True)\n"
        f"subprocess.Popen([sys.executable, '-c', {descendant_code!r}], stdin=subprocess.DEVNULL)\n"
        "sys.stdin.readline()\n"
        "os._exit(0)\n"
    )
    owner = launch(runs, tmp_path / "queue", "descendants", code=code, runner=owned_group_runner)
    owner.ready()
    descendant = psutil.Process(json.loads(owner.line("stdout"))["descendant"])
    try:
        assert owner.process.stdin is not None
        owner.process.stdin.write(b"go\n")
        owner.process.stdin.flush()
        assert owner.event("released")["child_status"] == 0
        assert owner.process.wait(timeout=10) == 0
        deadline = time.monotonic() + 5
        while descendant.is_running() and descendant.status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline, "descendant survived lock release"
            time.sleep(0.01)
    finally:
        with suppress(psutil.NoSuchProcess):
            if descendant.status() != psutil.STATUS_ZOMBIE:
                descendant.kill()


def test_cancellation_kills_term_resistant_work(
    tmp_path: Path, runs: list[Run], owned_group_runner: Path
) -> None:
    owner = launch(
        runs,
        tmp_path / "queue",
        "resistant",
        code="import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); " + WORK,
        runner=owned_group_runner,
    )
    owner.ready()
    owner.process.terminate()
    assert owner.process.wait(timeout=15) == 143
    assert owner.event("released")["child_status"] == 143
    assert owner.child is not None
    assert not owner.child.is_running()


def test_sigkill_orphan_is_reclaimed(tmp_path: Path, runs: list[Run]) -> None:
    lock = tmp_path / "queue"
    owner = launch(runs, lock, "owner")
    owner.ready()
    metadata = json.loads(next(lock.glob("*.ticket")).read_text())
    assert metadata["pid"] == owner.process.pid
    assert metadata["start_time"] == psutil.Process(owner.process.pid).create_time()
    owner.process.kill()
    assert owner.process.wait(timeout=10) == -signal.SIGKILL
    assert owner.child_pid is not None
    os.kill(owner.child_pid, signal.SIGKILL)
    owner.child_pid = None
    successor = launch(runs, lock, "successor")
    successor.ready()
    successor.finish()


def test_fifo_three_waiters(tmp_path: Path, runs: list[Run]) -> None:
    lock = tmp_path / "queue"
    owner = launch(runs, lock, "owner")
    owner.ready()
    waiters = [launch(runs, lock, label) for label in ("first", "second", "third")]
    # Stop the oldest waiter while the owner exits. Later arrivals must still
    # wait; polling implementations allow them to run during this pause.
    waiters[0].process.send_signal(signal.SIGSTOP)
    owner.finish()
    waiters[0].process.send_signal(signal.SIGCONT)
    for waiter in waiters:
        # A follower entering early blocks the expected predecessor here:
        # only that predecessor receives permission to finish its command.
        waiter.ready()
        waiter.finish()


def test_killed_middle_waiter_does_not_break_fifo(tmp_path: Path, runs: list[Run]) -> None:
    lock = tmp_path / "queue"
    owner = launch(runs, lock, "owner")
    owner.ready()
    first = launch(runs, lock, "first")
    middle = launch(runs, lock, "middle")
    last = launch(runs, lock, "last")
    middle.process.kill()
    assert middle.process.wait(timeout=10) == -signal.SIGKILL
    owner.finish()
    first.ready()
    first.finish()
    last.ready()
    last.finish()


def test_supervisor_sigkill_keeps_live_command_exclusive(tmp_path: Path, runs: list[Run]) -> None:
    lock = tmp_path / "queue"
    owner = launch(runs, lock, "owner")
    owner.ready()
    owner.process.kill()
    assert owner.process.wait(timeout=10) == -signal.SIGKILL
    refused = launch(runs, lock, "blocked", timeout=0.2)
    refused.event("timed_out")
    assert refused.process.wait(timeout=10) == 75
    # The surviving workload still owns its inherited descriptor.
    assert owner.process.stdin is not None
    owner.process.stdin.write(b"go\n")
    owner.process.stdin.flush()
    assert owner.line("stdout") == "finished"
    successor = launch(runs, lock, "successor")
    successor.ready()
    successor.finish()


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_cancellation_stops_work_before_release(
    tmp_path: Path, runs: list[Run], signum: signal.Signals
) -> None:
    lock = tmp_path / "queue"
    code = (
        "import os, signal\n"
        "def stop(signum, frame):\n"
        "    print('stopping', flush=True)\n"
        "    os.read(0, 1)\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n" + WORK
    )
    owner = launch(runs, lock, "owner", code=code)
    owner.ready()
    assert owner.child_pid is not None
    child = psutil.Process(owner.child_pid)
    owner.process.send_signal(signum)
    assert owner.line("stdout") == "stopping"
    owner.process.send_signal(signal.SIGTERM)
    owner.process.send_signal(signal.SIGINT)
    blocked = launch(runs, lock, "blocked", timeout=0.2)
    blocked.event("timed_out")
    assert blocked.process.wait(timeout=10) == 75
    assert owner.process.poll() is None  # Still owns cleanup of the held workload.
    assert owner.process.stdin is not None
    owner.process.stdin.write(b"go\n")
    owner.process.stdin.flush()
    assert owner.process.wait(timeout=15) == 128 + signum
    assert not child.is_running()
    owner.child_pid = None
    successor = launch(runs, lock, "successor")
    successor.ready()
    successor.finish()


@pytest.mark.parametrize("code", [0, 7, 124])
def test_command_exit_status(tmp_path: Path, code: int) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--lock",
            str(tmp_path / "queue"),
            "status",
            sys.executable,
            "-c",
            f"raise SystemExit({code})",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == code, result.stderr
    assert json.loads(result.stderr.splitlines()[-1])["child_status"] == code
