"""Measure disabling new Roaring segments while retaining automatic compression."""

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
        conn.execute("SET disabled_compression_methods='roaring'")
        row = conn.execute(
            "SELECT current_setting('disabled_compression_methods'), "
            "current_setting('force_compression'), current_setting('threads')"
        ).fetchone()
        assert row is not None and str(row[0]).lower() == "roaring" and row[1:] == ("auto", 4), row
        if evidence is not None:
            with (evidence / "compression-policy.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"disabled": row[0], "force": row[1], "threads": row[2]}) + "\n"
                )

    # Existing Roaring segments remain readable. Only codec selection for new
    # segments changes, preserving every logical constraint and durability policy.
    with (
        patch.object(measure, "__file__", str(Path(__file__).resolve())),
        patch.object(connection, "_apply_runtime_pragmas", apply),
    ):
        return measure.main()


if __name__ == "__main__":
    raise SystemExit(main())
