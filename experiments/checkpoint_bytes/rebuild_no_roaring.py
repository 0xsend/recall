"""Build a separate physical copy without Roaring; never activate it.

Logical schema and values stay unchanged. Aggregate signatures are prototype
evidence, not exact migration qualification or approval for a product rebuild.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import signal
import threading
import time
import traceback
from typing import Any, NotRequired, TypedDict

import duckdb
from measure import FIXTURE_SHA, LANE, write_json
from recall.db.connection import storage_version
from variants import signature


class TableCopy(TypedDict):
    before: tuple[object, ...]
    after: tuple[object, ...]
    persisted_codecs: NotRequired[list[tuple[Any, ...]]]


def quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def main() -> int:
    destination = LANE / "scratch/fixture-no-roaring-rebuilt"
    destination.mkdir(exist_ok=False)
    source = LANE / "scratch/fixture"
    evidence = LANE / "scratch/evidence/rebuild-no-roaring.json"
    report: dict[str, Any] = {
        "outcome": "incomplete",
        "source_fixture_sha256": FIXTURE_SHA,
        "pid": os.getpid(),
        "scope": "scratch physical copy only; no activation or product migration",
        "steps": {},
    }
    started = time.monotonic()
    expired = threading.Event()

    def interrupt(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    write_json(evidence, report)
    code = 1
    try:
        shutil.copy2(source / "recall.fts.sqlite", destination / "recall.fts.sqlite")
        report["steps"]["sidecar_copy"] = 0
        with duckdb.connect(
            str(destination / "recall.duckdb"),
            config={
                "storage_compatibility_version": "v1.2.0",
                "threads": "1",
                "memory_limit": "2GB",
                "temp_directory": str(destination / "spill"),
            },
        ) as conn:

            def deadline() -> None:
                expired.set()
                conn.interrupt()

            timer = threading.Timer(600, deadline)
            timer.daemon = True
            timer.start()
            try:
                conn.execute("SET disabled_compression_methods='roaring'")
                source_path = str(source / "recall.duckdb").replace("'", "''")
                conn.execute(f"ATTACH '{source_path}' AS original (READ_ONLY)")
                tables = conn.execute(
                    "SELECT schema_name, table_name FROM duckdb_tables() "
                    "WHERE database_name='original' AND NOT internal ORDER BY 1,2"
                ).fetchall()
                conn.execute("COPY FROM DATABASE original TO recall")
                report["steps"]["logical_copy"] = 0
                table_results: dict[str, TableCopy] = {}
                for schema, table in tables:
                    name = f"{quote(schema)}.{quote(table)}"
                    before = signature(conn, "*", f"original.{name}")
                    after = signature(conn, "*", f"recall.{name}")
                    assert before == after, (schema, table)
                    table_results[f"{schema}.{table}"] = {"before": before, "after": after}
                report["tables"] = table_results
                report["steps"]["aggregate_comparison"] = 0
                conn.execute("CHECKPOINT")
                report["steps"]["checkpoint"] = 0
            finally:
                timer.cancel()
                timer.join(timeout=1)
        with duckdb.connect(
            str(destination / "recall.duckdb"),
            read_only=True,
            config={
                "threads": "1",
                "memory_limit": "2GB",
                "temp_directory": str(destination / "spill"),
            },
        ) as conn:
            report["storage_version"] = storage_version(conn)
            for schema, table in tables:
                name = f"{quote(schema)}.{quote(table)}"
                assert signature(conn, "*", name) == tuple(
                    report["tables"][f"{schema}.{table}"]["after"]
                ), (schema, table)
                codecs = conn.execute(
                    "SELECT compression, count(*) FROM pragma_storage_info(?) GROUP BY 1", [name]
                ).fetchall()
                assert all(codec.lower() != "roaring" for codec, _ in codecs), (schema, table)
                report["tables"][f"{schema}.{table}"]["persisted_codecs"] = codecs
            report["steps"]["reopen"] = 0
        report["duckdb_bytes"] = (destination / "recall.duckdb").stat().st_size
        with (destination / "recall.duckdb").open("rb") as stream:
            report["prepared_duckdb_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        report["outcome"] = "passed"
        code = 0
    except BaseException as error:
        report["outcome"] = (
            "incomplete" if expired.is_set() or isinstance(error, SystemExit) else "failed"
        )
        report["error"] = "".join(traceback.format_exception(error))
        code = (
            int(error.code or 1)
            if isinstance(error, SystemExit)
            else 124
            if expired.is_set()
            else 1
        )
    finally:
        report["seconds"] = time.monotonic() - started
        report["kernel_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["memory_floor_passed"] = report["kernel_peak_rss_bytes"] <= 4 * 1024**3
        write_json(evidence, report)
        print(
            json.dumps({key: value for key, value in report.items() if key != "tables"}), flush=True
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
