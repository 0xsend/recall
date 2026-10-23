"""Finish read-only audits of completed cases, retaining existing results."""

from __future__ import annotations

import json
import signal
from pathlib import Path

from measure import LANE
from payload_compare import compare
from storage_audit import audit

STORAGE_RUNS = (
    "baseline-1",
    "baseline-2",
    "u128-1c",
    "drop-results-1",
    "threads-1",
    "no-roaring-1",
    "rebuilt-2",
    "uncompressed-3",
)
PAYLOAD_RUNS = ("baseline-2", "no-roaring-1", "rebuilt-2", "uncompressed-3")


def write_result(path: Path, result: dict[str, object]) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


def main() -> None:
    def expired(_signum: int, _frame: object) -> None:
        raise TimeoutError("completed-case audits exceeded their 600-second budget")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(600)
    try:
        for label in STORAGE_RUNS:
            path = LANE / "scratch/evidence" / f"{label}.storage.json"
            if path.exists():
                saved = json.loads(path.read_text())
                assert saved["label"] == label and saved["storage_version"]
            else:
                write_result(path, audit(label))
            print(f"{label}: persisted storage audit complete", flush=True)
        for label in PAYLOAD_RUNS:
            path = LANE / "scratch/evidence" / f"{label}.payload-equality.json"
            if path.exists():
                saved = json.loads(path.read_text())
                assert saved["outcome"] == "passed" and saved["candidate"] == label
            else:
                write_result(path, compare("baseline-1", label))
            print(f"{label}: exact pool payload comparison complete", flush=True)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
