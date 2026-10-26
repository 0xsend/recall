"""Measure uncompressed new column segments with the populated baseline harness."""

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
        conn.execute("SET force_compression='uncompressed'")
        row = conn.execute(
            "SELECT current_setting('force_compression'), current_setting('threads')"
        ).fetchone()
        assert row is not None and str(row[0]).lower() == "uncompressed" and row[1] == 4, row
        if evidence is not None:
            with (evidence / "compression-policy.jsonl").open("a") as stream:
                stream.write(json.dumps({"force_compression": row[0], "threads": row[1]}) + "\n")

    # Only newly written column segments change encoding. Existing segments,
    # logical schema, constraints, WAL policy and commit durability stay shared.
    with (
        patch.object(measure, "__file__", str(Path(__file__).resolve())),
        patch.object(connection, "_apply_runtime_pragmas", apply),
    ):
        return measure.main()


if __name__ == "__main__":
    raise SystemExit(main())
