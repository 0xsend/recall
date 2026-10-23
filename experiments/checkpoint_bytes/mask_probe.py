"""Synthetic array-validity control; forced checkpoints, not daemon qualification."""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb
from measure import LANE, io_sample
from recall.core.telemetry import Recorder, Stage, current
from recall.db.connection import checkpoint_wal_if_due, storage_version

ROWS = 110000
APPEND = 128


def insert_sql(start: int, end: int) -> str:
    return (
        "INSERT INTO masks SELECT md5('mask-'||i::VARCHAR), v,v,v FROM ("
        "SELECT i, list_transform(range(384), lambda j: "
        "CASE WHEN i%4=0 THEN NULL::FLOAT ELSE 0.25::FLOAT END)::FLOAT[384] v "
        f"FROM range({start},{end}) t(i))"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("auto", "no-roaring"))
    parser.add_argument("--repeat", required=True, type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    label = f"mask-{args.arm}-{args.repeat}"
    evidence = LANE / "scratch/evidence" / label
    evidence.mkdir(exist_ok=False)
    work = Path(os.environ["TMPDIR"]) / label
    work.mkdir(exist_ok=False)
    database = work / "masks.duckdb"
    result: dict[str, Any] = {
        "outcome": "incomplete",
        "scope": "synthetic component control; forced high-water 1; not daemon threshold timings",
        "arm": args.arm,
        "repeat": args.repeat,
        "pid": os.getpid(),
        "checkpoints": [],
    }

    def interrupt(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    started = time.monotonic()
    recorder = Recorder()
    token = current.set(recorder)
    code = 1
    try:
        with duckdb.connect(
            str(database),
            config={
                "storage_compatibility_version": "v1.2.0",
                "threads": "1",
                "memory_limit": "2GB",
                "temp_directory": str(work / "spill"),
            },
        ) as conn:
            conn.execute("SET wal_autocheckpoint='10GB'")
            if args.arm == "no-roaring":
                conn.execute("SET disabled_compression_methods='roaring'")
            conn.execute(
                "CREATE TABLE masks(id VARCHAR PRIMARY KEY, a FLOAT[384], "
                "b FLOAT[384], c FLOAT[384])"
            )
            for phase, begin, end in (("seed", 0, ROWS), ("append", ROWS, ROWS + APPEND)):
                conn.execute(insert_sql(begin, end))
                before = io_sample(os.getpid())
                checkpoint = checkpoint_wal_if_due(conn, database, high_water_bytes=1)
                after = io_sample(os.getpid())
                assert checkpoint.attempted and not checkpoint.contended
                result["checkpoints"].append(
                    {
                        "phase": phase,
                        **asdict(checkpoint),
                        "physical_bytes": after["write_bytes"] - before["write_bytes"],
                        "logical_bytes": after["wchar"] - before["wchar"],
                        "load": os.getloadavg(),
                    }
                )
                result[f"{phase}_codecs"] = conn.execute(
                    "SELECT segment_type,compression,count(*) "
                    "FROM pragma_storage_info('masks') GROUP BY 1,2 ORDER BY 1,2"
                ).fetchall()
                has_roaring = any(
                    codec.lower() == "roaring" for _, codec, _ in result[f"{phase}_codecs"]
                )
                assert has_roaring == (args.arm == "auto"), "control did not exercise its codec"
            row = conn.execute("SELECT count(*) FROM masks").fetchone()
            assert row == (ROWS + APPEND,), row
            result["storage_version"] = storage_version(conn)
            result["stage"] = asdict(recorder.snapshot().stages[Stage.CHECKPOINT])
            result["database_bytes"] = database.stat().st_size
        result["outcome"] = "passed"
        code = 0
    except BaseException as error:
        result["outcome"] = "incomplete" if isinstance(error, SystemExit) else "failed"
        result["error"] = "".join(traceback.format_exception(error))
        code = int(error.code or 1) if isinstance(error, SystemExit) else 1
    finally:
        current.reset(token)
        result["seconds"] = time.monotonic() - started
        result["kernel_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
