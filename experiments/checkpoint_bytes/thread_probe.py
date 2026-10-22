"""Measure a one-worker DuckDB policy with the unchanged populated harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import duckdb
import measure
from recall.core.config import AppConfig
from recall.db import connection


def main() -> int:
    arguments = sys.argv.copy()
    evidence = Path(arguments[arguments.index("--daemon") + 1]) if "--daemon" in arguments else None
    original = connection._apply_runtime_pragmas

    def apply(conn: duckdb.DuckDBPyConnection, config: AppConfig) -> None:
        original(conn, config)
        row = conn.execute("SELECT current_setting('threads')").fetchone()
        assert row == (1,), row
        if evidence is not None:
            with (evidence / "thread-policy.jsonl").open("a") as stream:
                stream.write(json.dumps({"threads": row[0], "scope": "all DuckDB work"}) + "\n")

    # The harness starts this same entry point in its owned daemon process.
    # Its workload, deadlines, diagnostics and cleanup remain shared.
    with (
        patch.object(measure, "__file__", str(Path(__file__).resolve())),
        patch.object(connection, "_MAX_THREADS", 1),
        patch.object(connection, "_apply_runtime_pragmas", apply),
    ):
        return measure.main()


if __name__ == "__main__":
    raise SystemExit(main())
