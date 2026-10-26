"""Record persisted encodings and file sizes after an owned measurement exits."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import duckdb
from measure import LANE
from recall.db.connection import storage_version
from variants import HASH_COLUMNS


def audit(label: str) -> dict[str, object]:
    assert re.fullmatch(r"[a-z0-9-]{1,32}", label)
    evidence = LANE / "scratch/evidence" / label
    result = json.loads((evidence / "result.json").read_text())
    assert result["outcome"] == "passed", label
    owned = json.loads((evidence / "owned.json").read_text())
    data = Path(owned["data"])
    assert data == Path(os.environ["TMPDIR"]) / label / "data", data
    database = data / "recall.duckdb"
    with duckdb.connect(
        str(database),
        read_only=True,
        config={"threads": "1", "memory_limit": "2GB", "temp_directory": str(data / "audit-spill")},
    ) as conn:
        tables = {}
        for table in HASH_COLUMNS:
            # Exclude statistics: string min/max could contain transcript text.
            rows = conn.execute(
                "SELECT row_group_id, column_name, segment_type, compression, count(*) segments, "
                'sum("count") values_count FROM pragma_storage_info(?) '
                "GROUP BY ALL ORDER BY row_group_id, column_name, segment_type, compression",
                [table],
            ).fetchall()
            last_group = max(row[0] for row in rows)
            tables[table] = {
                "row_group_count": len({row[0] for row in rows}),
                "last_row_group": last_group,
                "last_group_encodings": [
                    {
                        "column": row[1],
                        "segment_type": row[2],
                        "compression": row[3],
                        "segments": row[4],
                        "values_count": row[5],
                    }
                    for row in rows
                    if row[0] == last_group
                ],
            }
        return {
            "label": label,
            "storage_version": storage_version(conn),
            "duckdb_bytes": database.stat().st_size,
            "sqlite_bytes": (data / "recall.fts.sqlite").stat().st_size,
            "tables": tables,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    assert 1 <= len(args.runs) <= 32
    for label in args.runs:
        result = audit(label)
        output = LANE / "scratch/evidence" / f"{label}.storage.json"
        with output.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"label": label, "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
