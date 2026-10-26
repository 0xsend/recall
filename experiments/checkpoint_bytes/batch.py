"""Run the remaining comparison blocks, releasing the lock between variants."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO.parent / "scratch/evidence"
TOOLS = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", required=True)
    args = parser.parse_args()
    record: dict[str, Any] = {"outcome": "incomplete", "steps": []}
    path = EVIDENCE / "batch.json"

    def interrupt(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    try:
        predecessor = EVIDENCE / args.after / "result.json"
        deadline = time.monotonic() + 2700
        while not predecessor.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("predecessor did not finish")
            time.sleep(2)
        assert json.loads(predecessor.read_text())["outcome"] == "passed", args.after
        specs = [("convert-drop-results", "convert")]
        specs += [(f"{variant}-1", variant) for variant in ("drop-results", "watermark-16")]
        specs += [
            (f"{variant}-{repeat}", variant)
            for repeat in (2, 3)
            for variant in ("baseline", "u128", "drop-results", "watermark-16")
        ]
        for label, variant in specs:
            if variant == "convert":
                command = ["python", str(TOOLS / "variants.py"), "drop-results"]
                result_path = EVIDENCE / "conversion-drop-results.json"
            else:
                command = [
                    "python",
                    str(TOOLS / "measure.py"),
                    "--run",
                    label,
                    "--variant",
                    variant,
                ]
                result_path = EVIDENCE / label / "result.json"
            with (EVIDENCE / f"{label}.outer.log").open("x") as log:
                process = subprocess.Popen(
                    [
                        "bash",
                        str(TOOLS / "with_lock.sh"),
                        label,
                        "uv",
                        "run",
                        "--no-sync",
                        *command,
                    ],
                    cwd=REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                record["running"] = {"label": label, "runner_pid": process.pid}
                path.write_text(json.dumps(record, indent=2) + "\n")
                try:
                    code = process.wait(timeout=14400)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=120)
            observed = json.loads(result_path.read_text()) if result_path.exists() else {}
            record["steps"].append(
                {"label": label, "exit_code": code, "outcome": observed.get("outcome", "missing")}
            )
            record.pop("running")
            path.write_text(json.dumps(record, indent=2) + "\n")
            if code != 0 or observed.get("outcome") != "passed":
                raise RuntimeError(f"{label} did not complete successfully: {code}")
            print(json.dumps(record["steps"][-1]), flush=True)
        record["outcome"] = "passed"
        return 0
    except BaseException as error:
        record["error"] = "".join(traceback.format_exception(error))
        record["outcome"] = (
            "incomplete" if isinstance(error, TimeoutError | SystemExit) else "failed"
        )
        return int(error.code or 1) if isinstance(error, SystemExit) else 1
    finally:
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
