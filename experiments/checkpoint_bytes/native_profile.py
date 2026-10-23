"""Own a native profiler's entire descendant tree; diagnose, never qualify timings.

py-spy launches the harness and traces its daemon descendants, so Linux Yama's
normal ancestor rule suffices. No system ptrace/perf settings are changed.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import signal
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from measure import LANE, REPO

EVIDENCE = LANE / "scratch/evidence"


def reap_children(process: subprocess.Popen[bytes] | None) -> list[int]:
    signalled = set()
    group_signalled = False
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        else:
            group_signalled = True
    started = time.monotonic()
    while time.monotonic() - started < 120:
        try:
            while True:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
        except ChildProcessError:
            return sorted(signalled)
        children = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text().split()
        for value in children:
            pid = int(value)
            # SUBREAPER plus this program's sole Popen construction makes these
            # individually attributed descendants. Unreaped children cannot
            # have their PIDs recycled while this loop owns their wait status.
            try:
                already_notified = (
                    group_signalled and process is not None and os.getpgid(pid) == process.pid
                )
                if time.monotonic() - started >= 60:
                    os.kill(pid, signal.SIGKILL)
                elif pid not in signalled and not already_notified:
                    os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            signalled.add(pid)
        time.sleep(0.1)
    raise TimeoutError("owned profiling descendants did not exit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--policy", choices=("baseline", "one-worker"), default="baseline")
    parser.add_argument("--rehearsal", choices=("early", "deadline", "term", "normal"))
    args = parser.parse_args()
    assert re.fullmatch(r"[a-z0-9-]{1,32}", args.run)
    label = args.run
    record: dict[str, Any] = {
        "outcome": "incomplete",
        "steps": {},
        "timings_qualified": False,
        "policy": args.policy,
    }
    process: subprocess.Popen[bytes] | None = None
    code = 1

    def interrupt(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "cannot own profiling descendants")
        worker = [
            str(REPO / ".venv/bin/python"),
            str(Path(__file__).with_name("profile_worker.py")),
            "--run",
            label,
            "--policy",
            args.policy,
        ]
        if args.rehearsal:
            worker += ["--rehearsal", args.rehearsal]
        trace = EVIDENCE / f"{label}.native.raw"
        command = [
            "uv",
            "run",
            "--no-sync",
            "/home/ae/.local/bin/py-spy",
            "record",
            "--subprocesses",
            "--native",
            "--rate",
            "20",
            "--format",
            "raw",
            "--output",
            str(trace),
            "--",
            *worker,
        ]
        with (EVIDENCE / f"{label}.native.log").open("xb") as log:
            process = subprocess.Popen(
                command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            record["profiler_group"] = process.pid
            record["owner_pid"] = os.getpid()
            (EVIDENCE / f"{label}.profile.json").write_text(json.dumps(record, indent=2) + "\n")
            try:
                profiler_code = process.wait(timeout=5400)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError("native profiling watchdog expired") from error
        record["steps"]["profiler"] = profiler_code
        child_code = json.loads((EVIDENCE / label / "profile-worker-exit.json").read_text())[
            "exit_code"
        ]
        observed = json.loads((EVIDENCE / label / "result.json").read_text())["outcome"]
        record["steps"]["harness"] = child_code
        record["harness_outcome"] = observed
        record["trace"] = str(trace)
        if not args.rehearsal:
            assert trace.is_file() and trace.stat().st_size > 0, "profiler produced no trace"
        if profiler_code == 0 and child_code == 0 and observed == "passed":
            record["outcome"] = "passed"
            code = 0
        else:
            record["outcome"] = "incomplete" if observed == "incomplete" else "failed"
            code = child_code or profiler_code or 1
    except BaseException as error:
        record["outcome"] = (
            "incomplete" if isinstance(error, TimeoutError | SystemExit) else "failed"
        )
        record["error"] = "".join(traceback.format_exception(error))
        code = (
            int(error.code or 1)
            if isinstance(error, SystemExit)
            else 124
            if isinstance(error, TimeoutError)
            else 1
        )
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            record["signalled_owned_descendants"] = reap_children(process)
            record["steps"]["cleanup"] = 0
        except TimeoutError as error:
            record["outcome"] = "incomplete"
            record["cleanup_error"] = str(error)
            code = 1
        (EVIDENCE / f"{label}.profile.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
