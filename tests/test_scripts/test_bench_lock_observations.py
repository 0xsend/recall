"""Exercise the lock helper as a library with real commands and cancellation."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import psutil
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


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
            timeout=20,
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
        "        os.kill(os.getpid(), signal.SIGTERM)\n"
        "        wait(self, pid)\n"
        "lock.CommandWait.wait_unreaped = completed_then_cancel\n"
        "signal_group = os.killpg\n"
        "def cancel_during_cleanup(pid, signum):\n"
        f"    if index == {position} and {phase!r} == 'during_cleanup' "
        "and signum == signal.SIGTERM:\n"
        "        os.kill(os.getpid(), signal.SIGTERM)\n"
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
        "                os.kill(os.getpid(), signal.SIGTERM)\n"
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
        f"        os.kill(os.getpid(), {signum})\n"
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
def test_cli_cancellation_while_reporting_release(
    tmp_path: Path, signum: int, expected: int, status: int
) -> None:
    # The stream seam forwards writes to the real stderr pipe. It delivers
    # a real signal on the first released write, after real command cleanup.
    script = tmp_path / "cli.py"
    script.write_text(
        "import os, signal, sys\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import with_bench_lock as lock\n"
        "stream = sys.stderr\n"
        "class InterruptRelease:\n"
        "    interrupted = False\n"
        "    def write(self, text):\n"
        "        if '\"released\"' in text and not self.interrupted:\n"
        "            self.interrupted = True\n"
        f"            os.kill(os.getpid(), {signum})\n"
        "        return stream.write(text)\n"
        "    def flush(self):\n"
        "        stream.flush()\n"
        "sys.stderr = InterruptRelease()\n"
        "raise SystemExit(lock.main())\n"
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
        timeout=20,
    )
    assert result.returncode == expected, result.stderr
    assert result.stdout == "finished\n"
    events = [json.loads(line) for line in result.stderr.splitlines()]
    assert [event["lock"] for event in events] == ["queued", "acquired", "released"]
    assert events[-1] == {
        "lock": "released",
        "label": "release",
        "child_status": expected,
        "acquired": True,
    }
    # A separate real CLI invocation proves admission was released.
    again = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "with_bench_lock.py"),
            "--lock",
            str(queue),
            "--timeout",
            "2",
            "again",
            sys.executable,
            "-c",
            "print('admitted')",
        ],
        capture_output=True,
        text=True,
        timeout=10,
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
        f"        os.kill(os.getpid(), {signum})\n"
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
    script = tmp_path / "cli.py"
    script.write_text(
        "import os, signal, sys\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import with_bench_lock as lock\n"
        "set_wakeup = signal.set_wakeup_fd\n"
        "def restore_then_cancel(fd, **kwargs):\n"
        "    previous = set_wakeup(fd, **kwargs)\n"
        "    if fd == -1:\n"
        f"        os.kill(os.getpid(), {signum})\n"
        "    return previous\n"
        "signal.set_wakeup_fd = restore_then_cancel\n"
        "raise SystemExit(lock.main())\n"
    )
    queue = tmp_path / "queue"
    result = subprocess.run(
        [sys.executable, str(script), "--lock", str(queue), "spawn", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == expected, result.stderr
    assert result.stdout == ""
    events = [json.loads(line) for line in result.stderr.splitlines()]
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
            "--timeout",
            "2",
            "reuse",
            sys.executable,
            "-c",
            "print('admitted')",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert reuse.returncode == 0, reuse.stderr
    assert reuse.stdout == "admitted\n"
