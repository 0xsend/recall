"""Matched rebuilt key-only indexes: encoding effect without payload compaction.

These forced checkpoints are mechanism evidence, not daemon threshold results.
Every --arm invocation is a fresh process and an independent prepared-file copy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb
from measure import LANE, io_sample
from variants import HASH_COLUMNS, integer_id

INSERTS = 1000
PREPARED = LANE / "scratch/key-fixture"


def settings(conn: duckdb.DuckDBPyConnection, spill: Path) -> None:
    conn.execute("SET threads=1")
    conn.execute("SET memory_limit='2GB'")
    conn.execute(f"SET temp_directory='{spill}'")
    conn.execute("SET wal_autocheckpoint='256MiB'")


def prepare() -> dict[str, Any]:
    PREPARED.mkdir(exist_ok=False)
    result: dict[str, Any] = {}
    for arm in ("varchar", "u128"):
        path = PREPARED / f"{arm}.duckdb"
        conn = duckdb.connect(str(path), config={"storage_compatibility_version": "v1.2.0"})
        try:
            settings(conn, PREPARED / "spill")
            conn.execute(
                f"ATTACH '{LANE / 'scratch/fixture/recall.duckdb'}' AS population (READ_ONLY)"
            )
            counts = {}
            for table, columns in HASH_COLUMNS.items():
                column = columns[0]
                key_type = "UHUGEINT" if arm == "u128" else "VARCHAR"
                conn.execute(f"CREATE TABLE {table}(id {key_type} PRIMARY KEY)")
                projection = integer_id(column) if arm == "u128" else column
                conn.execute(f"INSERT INTO {table} SELECT {projection} FROM population.{table}")
                count_row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
                assert count_row is not None
                counts[table] = count_row[0]
                conn.execute("CHECKPOINT")
            result[arm] = counts
        finally:
            conn.close()
    assert result["varchar"] == result["u128"]
    return result


def measure(arm: str, repeat: int) -> dict[str, Any]:
    from recall.core.telemetry import Recorder, Stage, current
    from recall.db.connection import checkpoint_wal_if_due

    prepared = json.loads((LANE / "scratch/evidence/key-prepare/result.json").read_text())
    assert prepared["outcome"] == "passed"
    work = Path(os.environ["TMPDIR"]) / f"key-{arm}-{repeat}"
    work.mkdir(exist_ok=False)
    database = work / "keys.duckdb"
    shutil.copy2(PREPARED / f"{arm}.duckdb", database)
    recorder = Recorder()
    token = current.set(recorder)
    conn = duckdb.connect(str(database))
    try:
        settings(conn, work / "spill")
        projection = integer_id("hex_id") if arm == "u128" else "hex_id"
        for table in HASH_COLUMNS:
            conn.execute(
                f"INSERT INTO {table} SELECT {projection} FROM "
                f"(SELECT md5('checkpoint-keys-93-'||CAST(range AS VARCHAR)) AS hex_id "
                f"FROM range({INSERTS}))"
            )
        before = io_sample(os.getpid())
        started = time.monotonic()
        checkpoint = checkpoint_wal_if_due(conn, database, high_water_bytes=1)
        after = io_sample(os.getpid())
        assert checkpoint.attempted and not checkpoint.contended
        stage = recorder.snapshot().stages[Stage.CHECKPOINT]
        for table, count in prepared["results"][arm].items():
            count_row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
            assert count_row is not None and count_row[0] == count + INSERTS
        return {
            "arm": arm,
            "repeat": repeat,
            "checkpoint": asdict(checkpoint),
            "stage": asdict(stage),
            "start": started,
            "physical_bytes": after["write_bytes"] - before["write_bytes"],
            "logical_bytes": after["wchar"] - before["wchar"],
            "load": os.getloadavg(),
            "inserts_per_table": INSERTS,
            "table_count": len(HASH_COLUMNS),
        }
    finally:
        current.reset(token)
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--arm", choices=("varchar", "u128"))
    parser.add_argument("--repeat", type=int)
    args = parser.parse_args()
    label = "key-prepare" if args.prepare else f"key-{args.arm}-{args.repeat}"
    evidence = LANE / "scratch/evidence" / label
    evidence.mkdir(exist_ok=False)
    result: dict[str, Any] = {"outcome": "incomplete", "steps": {}}

    def interrupt(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        if args.prepare:
            result["results"] = prepare()
        else:
            assert args.arm and args.repeat in (1, 2, 3)
            result["results"] = measure(args.arm, args.repeat)
        result["steps"]["work"] = 0
        result["outcome"] = "passed"
        return 0
    except BaseException as error:
        result["outcome"] = "incomplete" if isinstance(error, SystemExit) else "failed"
        result["error"] = "".join(traceback.format_exception(error))
        return int(error.code or 1) if isinstance(error, SystemExit) else 1
    finally:
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
