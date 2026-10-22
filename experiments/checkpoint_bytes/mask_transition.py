"""Test whether excluding future encodings removes existing validity decode cost."""

from __future__ import annotations

import json
import os
import resource
import signal
import time
from pathlib import Path
from typing import Any

import duckdb
from mask_probe import APPEND, ROWS, insert_sql
from measure import LANE, io_sample
from recall.db.connection import checkpoint_wal_if_due


def main() -> None:
    evidence = LANE / "scratch/evidence/mask-transition.json"
    work = Path(os.environ["TMPDIR"]) / "mask-transition"
    work.mkdir(exist_ok=False)
    assert not evidence.exists()
    result: dict[str, Any] = {"outcome": "incomplete", "checkpoints": []}
    started = time.monotonic()

    def expired(_signum: int, _frame: object) -> None:
        raise TimeoutError("synthetic transition exceeded its 120-second budget")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(120)
    try:
        with duckdb.connect(
            str(work / "masks.duckdb"),
            config={
                "storage_compatibility_version": "v1.2.0",
                "threads": "1",
                "memory_limit": "2GB",
                "temp_directory": str(work / "spill"),
            },
        ) as conn:
            conn.execute("SET wal_autocheckpoint='10GB'")
            conn.execute(
                "CREATE TABLE masks(id VARCHAR PRIMARY KEY, "
                "a FLOAT[384], b FLOAT[384], c FLOAT[384])"
            )
            for phase, begin, end in (
                ("seed-auto", 0, ROWS),
                ("append-exclude", ROWS, ROWS + APPEND),
            ):
                if phase == "append-exclude":
                    conn.execute("SET disabled_compression_methods='roaring'")
                conn.execute(insert_sql(begin, end))
                before = io_sample(os.getpid())
                checkpoint = checkpoint_wal_if_due(conn, work / "masks.duckdb", high_water_bytes=1)
                after = io_sample(os.getpid())
                assert checkpoint.attempted and not checkpoint.contended
                codecs = conn.execute(
                    "SELECT segment_type,compression,count(*) FROM pragma_storage_info('masks') "
                    "GROUP BY 1,2 ORDER BY 1,2"
                ).fetchall()
                has_roaring = any(codec.lower() == "roaring" for _, codec, _ in codecs)
                assert has_roaring == (phase == "seed-auto"), codecs
                result["checkpoints"].append(
                    {
                        "phase": phase,
                        "duration_seconds": checkpoint.duration_seconds,
                        "physical_bytes": after["write_bytes"] - before["write_bytes"],
                        "logical_bytes": after["wchar"] - before["wchar"],
                        "codecs": codecs,
                        "load": os.getloadavg(),
                    }
                )
            assert conn.execute("SELECT count(*) FROM masks").fetchone() == (ROWS + APPEND,)
        result["outcome"] = "passed"
    finally:
        signal.alarm(0)
        result["seconds"] = time.monotonic() - started
        result["kernel_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        evidence.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
