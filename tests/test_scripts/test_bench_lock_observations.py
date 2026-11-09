"""Exercise the lock helper as a library with real commands and cancellation.

Self-signals use raise_signal so native worker threads cannot receive an
injection intended for a particular boundary on the Python main thread.
Subprocess completion and pipe EOF synchronize assertions; scheduling delay
is not a correctness deadline.
"""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import psutil
import pytest

from tests.bench_lock_events import parse_events

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def cli_script(tmp_path: Path, code: str) -> Path:
    script = tmp_path / "cli.py"
    script.write_text(
        "import os, signal, sys\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import with_bench_lock as lock\n"
        # Retain a callable when the waiter restores its caller's context.
        # A SIGCHLD queued during restoration must still have a handler when
        # Python dispatches it, including inside the cancellation trace hook.
        "def child_event(signum, frame):\n"
        "    pass\n"
        "signal.signal(signal.SIGCHLD, child_event)\n" + code
    )
    return script


def caller(tmp_path: Path, code: str) -> dict:
    script = tmp_path / "caller.py"
    script.write_text(
        "import json, os, signal, sys, psutil\n"
        "from dataclasses import asdict\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import with_bench_lock as lock\n"
        "from recall_bench import owned_process_group as groups\n"
        "spawn = groups.subprocess.Popen\n"
        "def started(*args, **kwargs):\n"
        "    child = spawn(*args, **kwargs)\n"
        "    with open('commands.jsonl', 'a') as record:\n"
        "        record.write(json.dumps({'pid': child.pid, "
        "'start': psutil.Process(child.pid).create_time()}) + '\\n')\n"
        "    return child\n"
        "groups.subprocess.Popen = started\n"
        "def cancel(signum, frame):\n"
        "    raise lock.Cancelled(signum)\n"
        "def child_event(signum, frame):\n"
        "    pass\n"
        "signal.signal(signal.SIGTERM, cancel)\n"
        "signal.signal(signal.SIGINT, cancel)\n"
        "signal.signal(signal.SIGCHLD, child_event)\n"
        "signals = (signal.SIGCHLD, signal.SIGTERM, signal.SIGINT)\n"
        "handlers = {s: signal.getsignal(s) for s in signals}\n"
        "mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])\n"
        "reader, writer = os.pipe()\n"
        "os.set_blocking(writer, False)\n"
        "previous_wakeup = signal.set_wakeup_fd(writer)\n"
        + code
        + "\nfinished = result.get('completed', [])\n"
        "if 'finished' in result:\n"
        "    finished.append(result['finished'])\n"
        "for command in finished:\n"
        "    try:\n"
        "        os.waitpid(command['pid'], os.WNOHANG)\n"
        "    except ChildProcessError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError('finished command was not reaped')\n"
        + "\nassert {s: signal.getsignal(s) for s in signals} == handlers\n"
        "assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == mask\n"
        "assert signal.set_wakeup_fd(previous_wakeup) == writer\n"
        "os.close(writer)\n"
        "os.close(reader)\n"
        "print(json.dumps(result))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            input="go\n",
            capture_output=True,
            text=True,
        )
    finally:
        commands = tmp_path / "commands.jsonl"
        if commands.exists():
            for line in commands.read_text().splitlines():
                identity = json.loads(line)
                with suppress(psutil.NoSuchProcess):
                    process = psutil.Process(identity["pid"])
                    assert process.create_time() == identity["start"], "command PID was reused"
                    process.kill()
                    process.wait(timeout=5)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("status", [0, 7])
def test_library_restores_callers_signal_context(tmp_path: Path, status: int) -> None:
    result = caller(
        tmp_path,
        "with lock.ticket(Path('queue'), 'caller') as owner:\n"
        "    status = lock.run_command([sys.executable, '-c', "
        f"'raise SystemExit({status})'], owner)\n"
        "result = {'status': status}\n",
    )
    assert result == {"status": status}


@pytest.mark.parametrize("status", [0, 7])
def test_observations_retain_real_status_and_redirected_output(tmp_path: Path, status: int) -> None:
    result = caller(
        tmp_path,
        "execution = lock.CommandExecution()\n"
        "assert execution.state is None\n"
        "running = []\n"
        "wait = lock.CommandWait.wait_unreaped\n"
        "def observe(self, pid):\n"
        "    assert isinstance(execution.state, lock.RunningCommand)\n"
        "    running.append(asdict(execution.state))\n"
        "    wait(self, pid)\n"
        "lock.CommandWait.wait_unreaped = observe\n"
        "with lock.ticket(Path('queue'), 'caller') as owner, "
        "open('stdout', 'wb') as out, open('stderr', 'wb') as err:\n"
        "    status = lock.run_command([sys.executable, '-c', "
        "\"import os, sys; print(os.getpid()); print('diagnostic', file=sys.stderr); "
        f'raise SystemExit({status})"], '
        "owner, stdout=out.fileno(), stderr=err.fileno(), execution=execution)\n"
        "assert isinstance(execution.state, lock.FinishedCommand)\n"
        "result = {'running': running, 'finished': asdict(execution.state), 'status': status, "
        "'stdout': Path('stdout').read_text(), 'stderr': Path('stderr').read_text()}\n",
    )
    pid = int(result["stdout"])
    assert result["running"] == [{"pid": pid}]
    assert result["finished"] == {"pid": pid, "status": status}
    assert result["status"] == status
    assert result["stderr"] == "diagnostic\n"


@pytest.mark.parametrize("signum", [15, 2])
def test_active_cancellation_records_reaped_command(tmp_path: Path, signum: int) -> None:
    result = caller(
        tmp_path,
        "execution = lock.CommandExecution()\n"
        "with lock.ticket(Path('queue'), 'caller') as owner, open('stdout', 'wb') as out:\n"
        "    try:\n"
        "        lock.run_command([sys.executable, '-c', "
        '"import os, signal; print(os.getpid(), flush=True); '
        f'os.kill(os.getppid(), {signum}); signal.pause()"], '
        "owner, stdout=out.fileno(), execution=execution)\n"
        "    except lock.Cancelled as error:\n"
        "        cancelled = error.signum\n"
        "assert isinstance(execution.state, lock.FinishedCommand)\n"
        "result = {'cancelled': cancelled, 'finished': asdict(execution.state), "
        "'pid': int(Path('stdout').read_text())}\n",
    )
    assert result["cancelled"] == signum
    assert result["finished"] == {"pid": result["pid"], "status": -15}


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("phase", ["before_cleanup", "during_cleanup", "after_return"])
def test_completed_command_status_survives_cancellation_at_each_arm(
    tmp_path: Path, position: int, phase: str
) -> None:
    result = caller(
        tmp_path,
        "completed = []\n"
        "wait = lock.CommandWait.wait_unreaped\n"
        "def completed_then_cancel(self, pid):\n"
        "    wait(self, pid)\n"
        f"    if index == {position} and {phase!r} == 'before_cleanup':\n"
        "        signal.raise_signal(signal.SIGTERM)\n"
        "        wait(self, pid)\n"
        "lock.CommandWait.wait_unreaped = completed_then_cancel\n"
        "signal_group = os.killpg\n"
        "def cancel_during_cleanup(pid, signum):\n"
        f"    if index == {position} and {phase!r} == 'during_cleanup' "
        "and signum == signal.SIGTERM:\n"
        "        signal.raise_signal(signal.SIGTERM)\n"
        "    signal_group(pid, signum)\n"
        "os.killpg = cancel_during_cleanup\n"
        "try:\n"
        "    for index in range(3):\n"
        "        execution = lock.CommandExecution()\n"
        "        try:\n"
        "            with lock.ticket(Path('queue'), str(index)) as owner:\n"
        "                lock.run_command([sys.executable, '-c', 'raise SystemExit(0)'], "
        "owner, execution=execution)\n"
        f"            if index == {position} and {phase!r} == 'after_return':\n"
        "                signal.raise_signal(signal.SIGTERM)\n"
        "        finally:\n"
        "            assert isinstance(execution.state, lock.FinishedCommand)\n"
        "            completed.append(asdict(execution.state))\n"
        "except lock.Cancelled as error:\n"
        "    cancelled = error.signum\n"
        "result = {'completed': completed, 'cancelled': cancelled}\n",
    )
    assert result["cancelled"] == 15
    assert len(result["completed"]) == position + 1
    assert [command["status"] for command in result["completed"]] == [0] * (position + 1)
    assert len({command["pid"] for command in result["completed"]}) == position + 1


def test_spawn_failure_restores_signal_context_without_observation(tmp_path: Path) -> None:
    result = caller(
        tmp_path,
        "execution = lock.CommandExecution()\n"
        "with lock.ticket(Path('queue'), 'caller') as owner:\n"
        "    try:\n"
        "        lock.run_command([str(Path('missing').absolute())], owner, execution=execution)\n"
        "    except FileNotFoundError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError('missing command started')\n"
        "result = {'state': execution.state}\n",
    )
    assert result == {"state": None}


@pytest.mark.parametrize("signum", [15, 2])
@pytest.mark.parametrize("status", [0, 7])
def test_cancellation_in_waiter_epilogue_reaches_caller(
    tmp_path: Path, signum: int, status: int
) -> None:
    # Only the scheduling point is replaced: the real wakeup fd is restored,
    # then a real signal arrives before command_wait restores its handlers.
    result = caller(
        tmp_path,
        "execution = lock.CommandExecution()\n"
        "set_wakeup = signal.set_wakeup_fd\n"
        "def restore_then_cancel(fd, **kwargs):\n"
        "    previous = set_wakeup(fd, **kwargs)\n"
        "    if fd == writer:\n"
        f"        signal.raise_signal({signum})\n"
        "    return previous\n"
        "signal.set_wakeup_fd = restore_then_cancel\n"
        "cancelled = None\n"
        "with lock.ticket(Path('queue'), 'caller') as owner:\n"
        "    try:\n"
        "        lock.run_command([sys.executable, '-c', "
        f"'raise SystemExit({status})'], owner, execution=execution)\n"
        "    except lock.Cancelled as error:\n"
        "        cancelled = error.signum\n"
        "signal.set_wakeup_fd = set_wakeup\n"
        "assert isinstance(execution.state, lock.FinishedCommand)\n"
        "result = {'cancelled': cancelled, 'finished': asdict(execution.state)}\n",
    )
    assert result["cancelled"] == signum
    assert result["finished"]["status"] == status


@pytest.mark.parametrize("signum, expected", [(15, 143), (2, 130)])
@pytest.mark.parametrize("status", [0, 7])
@pytest.mark.parametrize("phase", ["write", "flush"])
def test_cli_cancellation_while_reporting_release(
    tmp_path: Path, signum: int, expected: int, status: int, phase: str
) -> None:
    # Real signals at both stream boundaries preserve complete records,
    # including the observed command status and the final cancellation status.
    script = cli_script(
        tmp_path,
        "stream = sys.stderr\n"
        "class InterruptRelease:\n"
        "    interrupted = False\n"
        "    releasing = False\n"
        "    def interrupt(self):\n"
        "        if self.releasing and not self.interrupted:\n"
        "            self.interrupted = True\n"
        f"            signal.raise_signal({signum})\n"
        "    def write(self, text):\n"
        "        self.releasing = '\"released\"' in text\n"
        f"        if {phase!r} == 'write': self.interrupt()\n"
        "        return stream.write(text)\n"
        "    def flush(self):\n"
        f"        if {phase!r} == 'flush': self.interrupt()\n"
        "        stream.flush()\n"
        "sys.stderr = InterruptRelease()\n"
        "raise SystemExit(lock.main())\n",
    )
    queue = tmp_path / "queue"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--lock",
            str(queue),
            "release",
            sys.executable,
            "-c",
            f"print('finished'); raise SystemExit({status})",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected, result.stderr
    assert result.stdout == "finished\n"
    events = parse_events(result)
    assert [event["lock"] for event in events] == ["queued", "acquired", "released", "released"]
    assert events[-2:] == [
        {"lock": "released", "label": "release", "child_status": status, "acquired": True},
        {"lock": "released", "label": "release", "child_status": expected, "acquired": True},
    ]
    # A separate real CLI invocation proves admission was released.
    again = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "with_bench_lock.py"),
            "--lock",
            str(queue),
            "again",
            sys.executable,
            "-c",
            "print('admitted')",
        ],
        capture_output=True,
        text=True,
    )
    assert again.returncode == 0, again.stderr
    assert again.stdout == "admitted\n"


@pytest.mark.parametrize("signum", [15, 2])
def test_spawn_failure_preserves_epilogue_cancellation(tmp_path: Path, signum: int) -> None:
    result = caller(
        tmp_path,
        "execution = lock.CommandExecution()\n"
        "set_wakeup = signal.set_wakeup_fd\n"
        "def restore_then_cancel(fd, **kwargs):\n"
        "    previous = set_wakeup(fd, **kwargs)\n"
        "    if fd == writer:\n"
        f"        signal.raise_signal({signum})\n"
        "    return previous\n"
        "signal.set_wakeup_fd = restore_then_cancel\n"
        "cancelled = None\n"
        "with lock.ticket(Path('queue'), 'caller') as owner:\n"
        "    try:\n"
        "        lock.run_command([str(Path('missing').absolute())], owner, execution=execution)\n"
        "    except lock.Cancelled as error:\n"
        "        cancelled = error.signum\n"
        "    except FileNotFoundError:\n"
        "        pass\n"
        "signal.set_wakeup_fd = set_wakeup\n"
        "result = {'cancelled': cancelled, 'state': execution.state}\n",
    )
    assert result == {"cancelled": signum, "state": None}


@pytest.mark.parametrize("signum, expected", [(15, 143), (2, 130)])
def test_cli_spawn_failure_preserves_epilogue_cancellation(
    tmp_path: Path, signum: int, expected: int
) -> None:
    script = cli_script(
        tmp_path,
        "set_wakeup = signal.set_wakeup_fd\n"
        "def restore_then_cancel(fd, **kwargs):\n"
        "    previous = set_wakeup(fd, **kwargs)\n"
        "    if fd == -1:\n"
        f"        signal.raise_signal({signum})\n"
        "    return previous\n"
        "signal.set_wakeup_fd = restore_then_cancel\n"
        "raise SystemExit(lock.main())\n",
    )
    queue = tmp_path / "queue"
    result = subprocess.run(
        [sys.executable, str(script), "--lock", str(queue), "spawn", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected, result.stderr
    assert result.stdout == ""
    events = parse_events(result)
    assert [e["lock"] for e in events] == ["queued", "acquired", "released"]
    assert events[-1] == {
        "lock": "released",
        "label": "spawn",
        "child_status": expected,
        "acquired": True,
    }
    reuse = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "with_bench_lock.py"),
            "--lock",
            str(queue),
            "reuse",
            sys.executable,
            "-c",
            "print('admitted')",
        ],
        capture_output=True,
        text=True,
    )
    assert reuse.returncode == 0, reuse.stderr
    assert reuse.stdout == "admitted\n"


@pytest.mark.parametrize("signum", [15, 2])
@pytest.mark.parametrize("exhaust_descriptors", [False, True])
def test_cli_waiter_setup_failure_preserves_pending_cancellation(
    tmp_path: Path, signum: int, exhaust_descriptors: bool
) -> None:
    script = cli_script(
        tmp_path,
        "import errno, os, resource, signal, sys, threading\n"
        "pipe = os.pipe\n"
        "held = []\n"
        "limits = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
        "def exhausted_pipe():\n"
        f"    if {exhaust_descriptors!r}:\n"
        "        limit = 32 if limits[0] == resource.RLIM_INFINITY else min(32, limits[0])\n"
        "        resource.setrlimit(resource.RLIMIT_NOFILE, (limit, limits[1]))\n"
        "        for _ in range(32):\n"
        "            try:\n"
        "                held.append(os.open(os.devnull, os.O_RDONLY))\n"
        "            except OSError as error:\n"
        "                assert error.errno == errno.EMFILE\n"
        "                break\n"
        "        else:\n"
        "            raise AssertionError('descriptors were not exhausted')\n"
        f"    signal.raise_signal({signum})\n"
        f"    assert {signum} in signal.sigpending(), 'cancellation must be pending at setup'\n"
        f"    if {exhaust_descriptors!r}:\n"
        "        return pipe()\n"
        "    raise OSError(24, 'pipe descriptors exhausted')\n"
        "os.pipe = exhausted_pipe\n"
        "ready, stop = threading.Event(), threading.Event()\n"
        "def receive_signals():\n"
        "    ready.set()\n"
        "    assert stop.wait(20), 'signal receiver was not stopped'\n"
        "worker = threading.Thread(target=receive_signals)\n"
        "worker.start()\n"
        "try:\n"
        "    assert ready.wait(5), 'signal receiver did not start'\n"
        "    status = lock.main()\n"
        "finally:\n"
        "    stop.set()\n"
        "    worker.join(timeout=5)\n"
        "    for descriptor in held: os.close(descriptor)\n"
        "    resource.setrlimit(resource.RLIMIT_NOFILE, limits)\n"
        "    assert not worker.is_alive(), 'signal receiver did not stop'\n"
        "raise SystemExit(status)\n",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--lock",
            str(tmp_path / "queue"),
            "setup",
            sys.executable,
            "-c",
            "print('command started')",
        ],
        capture_output=True,
        text=True,
    )
    events = parse_events(result)
    assert result.returncode == 128 + signum, result.stderr
    assert result.stdout == ""
    assert [event["lock"] for event in events] == ["queued", "acquired", "failed", "released"]
    assert events[-2]["error_type"] == "OSError"
    if exhaust_descriptors:
        assert events[-2]["error"] == "[Errno 24] Too many open files"
    else:
        assert events[-2]["error"] == "[Errno 24] pipe descriptors exhausted"
    assert events[-1]["child_status"] == 128 + signum


@pytest.mark.parametrize("signum", [15, 2])
def test_cli_cancellation_at_admission_does_not_start_command(tmp_path: Path, signum: int) -> None:
    script = cli_script(
        tmp_path,
        "stream = sys.stderr\n"
        "class InterruptAdmission:\n"
        "    def write(self, text):\n"
        "        if '\"acquired\"' in text:\n"
        f"            signal.raise_signal({signum})\n"
        "        return stream.write(text)\n"
        "    def flush(self):\n"
        "        stream.flush()\n"
        "sys.stderr = InterruptAdmission()\n"
        "raise SystemExit(lock.main())\n",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--lock",
            str(tmp_path / "queue"),
            "admission",
            sys.executable,
            "-c",
            "print('command started')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 128 + signum, result.stderr
    assert result.stdout == ""
    events = parse_events(result)
    assert [event["lock"] for event in events] == ["queued", "acquired", "released"]
    assert events[-1]["child_status"] == 128 + signum


@pytest.mark.parametrize("first, second", [(15, 2), (2, 15)])
def test_cli_repeated_cancellation_while_handling_waiter_cancellation(
    tmp_path: Path, first: int, second: int
) -> None:
    script = cli_script(
        tmp_path,
        "set_wakeup = signal.set_wakeup_fd\n"
        "def restore_then_cancel(fd, **kwargs):\n"
        "    previous = set_wakeup(fd, **kwargs)\n"
        "    if fd == -1:\n"
        f"        signal.raise_signal({first})\n"
        "    return previous\n"
        "signal.set_wakeup_fd = restore_then_cancel\n"
        "interrupted = False\n"
        "def interrupt_handler(frame, event, arg):\n"
        "    global interrupted\n"
        "    if (not interrupted and event == 'line' "
        "and frame.f_code is lock.main.__code__\n"
        "            and isinstance(frame.f_locals.get('error'), lock.Cancelled)):\n"
        "        interrupted = True\n"
        f"        signal.raise_signal({second})\n"
        "    return interrupt_handler\n"
        "sys.settrace(interrupt_handler)\n"
        "status = lock.main()\n"
        "assert interrupted, 'cancellation handler was not exercised'\n"
        "raise SystemExit(status)\n",
    )
    queue = tmp_path / "queue"
    result = subprocess.run(
        [sys.executable, str(script), "--lock", str(queue), "spawn", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
    )
    events = parse_events(result)
    assert result.returncode == 128 + first, result.stderr
    assert result.stdout == ""
    assert [event["lock"] for event in events] == ["queued", "acquired", "released"]
    assert events[-1] == {
        "lock": "released",
        "label": "spawn",
        "child_status": 128 + first,
        "acquired": True,
    }
    reuse = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "with_bench_lock.py"),
            "--lock",
            str(queue),
            "reuse",
            sys.executable,
            "-c",
            "print('admitted')",
        ],
        capture_output=True,
        text=True,
    )
    assert reuse.returncode == 0, reuse.stderr
    assert reuse.stdout == "admitted\n"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux stat boundary in the real CLI")
@pytest.mark.parametrize(
    "mode,signum,expected",
    [
        ("release", 2, 130),
        ("release", 15, 143),
        ("hard", 15, 1),
        ("unproved", 15, 1),
        ("multiple", 15, 1),
        ("report", 15, 1),
    ],
)
def test_cli_reports_proof_state_without_a_diagnostic_wrapper(tmp_path, mode, signum, expected):
    script = cli_script(
        tmp_path,
        "import errno, os, signal, sys\n"
        "from pathlib import Path\n"
        "wait = lock.CommandWait.wait_unreaped\n"
        "native_open = Path.open\n"
        "def waited(self, owner):\n"
        "    wait(self, owner)\n"
        "    target = Path(f'/proc/{owner.pid}/stat')\n"
        "    count = 0\n"
        "    class Reader:\n"
        "        def __enter__(self): return self\n"
        "        def __exit__(self, *args): pass\n"
        "        def read(self, size):\n"
        f"            raise OSError({5 if mode in {'hard', 'multiple', 'report'} else 3}, "
        "'stat replay')\n"
        "    def opened(path, *args, **kwargs):\n"
        "        nonlocal count\n"
        "        if path == target:\n"
        "            count += 1\n"
        f"            if count == 3 or ({mode!r} == 'unproved' and count % 3 == 0):\n"
        "                return Reader()\n"
        "        return native_open(path, *args, **kwargs)\n"
        "    Path.open = opened\n"
        f"    signal.raise_signal({signum})\n"
        "lock.CommandWait.wait_unreaped = waited\n"
        + (
            "send = os.killpg\n"
            "def denied(pgid, signum):\n"
            "    if signum == signal.SIGTERM: raise PermissionError(1, 'term denied')\n"
            "    send(pgid, signum)\n"
            "os.killpg = denied\n"
            if mode == "multiple"
            else ""
        )
        + (
            "stream = sys.stderr\n"
            "class InterruptFailure:\n"
            "    def write(self, text):\n"
            "        if '\"failed\"' in text:\n"
            "            signal.raise_signal(signal.SIGTERM)\n"
            "            signal.raise_signal(signal.SIGINT)\n"
            "        return stream.write(text)\n"
            "    def flush(self):\n"
            "        stream.flush()\n"
            "sys.stderr = InterruptFailure()\n"
            if mode == "report"
            else ""
        )
        + "raise SystemExit(lock.main())\n",
    )
    queue = tmp_path / "queue"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--lock",
            str(queue),
            "trial",
            sys.executable,
            "-c",
            "print('completed'); raise SystemExit(7)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected, result.stderr
    assert result.stdout == "completed\n"
    events = parse_events(result)
    assert events[-1] == {
        "lock": "released",
        "label": "trial",
        "child_status": expected,
        "acquired": True,
    }
    failed = [event for event in events if event["lock"] == "failed"]
    if mode == "release":
        assert failed == []
    else:
        assert len(failed) == 1
        event = failed[0]
        assert event["error_type"] == "StopFailed"
        assert event["phase"] == ("SIGTERM" if mode == "multiple" else "proof")
        assert event["completion_proved"] is (mode != "unproved")
        if mode == "unproved":
            assert event["stop_errors"] == [
                {"phase": "proof", "message": "group completion remains unproved", "errno": None}
            ]
        else:
            errors = [{"phase": "proof", "message": "[Errno 5] stat replay", "errno": 5}]
            if mode == "multiple":
                errors.insert(
                    0, {"phase": "SIGTERM", "message": "[Errno 1] term denied", "errno": 1}
                )
            assert event["stop_errors"] == errors
        assert event["error"] == "; ".join(error["message"] for error in event["stop_errors"])
    follow = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "with_bench_lock.py"),
            "--lock",
            str(queue),
            "follow",
            sys.executable,
            "-c",
            "pass",
        ],
        capture_output=True,
        text=True,
    )
    assert follow.returncode == 0, follow.stderr
