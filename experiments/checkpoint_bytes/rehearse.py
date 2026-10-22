"""Exercise the exact lock → uv → harness → process-group launch shape."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO.parent / "scratch/evidence"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--harness", type=Path, default=Path(__file__).with_name("measure.py"))
    args = parser.parse_args()
    results = []
    for case, expected in (("early", 1), ("deadline", 124), ("term", 143), ("normal", 0)):
        label = f"{args.prefix}-{case}"
        lock = REPO.parent / "scratch" / f"{args.prefix}.lock"
        env = dict(os.environ, WITH_LOCK_PATH=str(lock))
        with (EVIDENCE / f"{label}.log").open("x") as log:
            process = subprocess.Popen(
                [
                    "bash",
                    str(Path(__file__).with_name("with_lock.sh")),
                    label,
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    str(args.harness),
                    "--run",
                    label,
                    "--rehearsal",
                    case,
                ],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                if case == "term":
                    deadline = time.monotonic() + 30
                    while not (EVIDENCE / label / "owned.json").exists():
                        assert process.poll() is None
                        if time.monotonic() >= deadline:
                            raise TimeoutError("rehearsal did not launch its owned child")
                        time.sleep(0.1)
                    process.send_signal(signal.SIGTERM)
                code = process.wait(timeout=75)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=75)
        result = json.loads((EVIDENCE / label / "result.json").read_text())
        pid = json.loads((EVIDENCE / label / "owned.json").read_text())["pid"]
        assert not Path(f"/proc/{pid}").exists(), (case, pid)
        assert not lock.exists(), case
        results.append(
            {
                "case": case,
                "exit_code": code,
                "expected": expected,
                "outcome": result["outcome"],
                "child_gone": True,
                "lock_released": True,
            }
        )
        (EVIDENCE / f"{args.prefix}.json").write_text(json.dumps(results, indent=2) + "\n")
        assert code == expected, results[-1]
        print(json.dumps(results[-1]), flush=True)
    print(json.dumps({"outcome": "passed", "steps": results}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
