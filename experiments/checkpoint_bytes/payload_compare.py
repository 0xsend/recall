"""Compare the fixed pool's persisted content, vectors and tail facts exactly.

Run under the benchmark lock after both measurement processes exit. Natural
source/position keys account for run-specific absolute paths and hash IDs.
No transcript text or vector values are emitted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
from measure import LANE


@dataclass(frozen=True)
class Relation:
    name: str
    projection: str
    joins: str
    expected_rows: int
    tool: bool = False
    vector_columns: tuple[int, ...] = ()

    def sql(self) -> str:
        if self.tool:
            keys = "COALESCE(m.idx,-1), tc.idx"
            parent = (
                "JOIN tool_calls tc ON tc.session_id=s.id "
                "LEFT JOIN messages m ON m.id=tc.message_id "
            )
            order = "1,2,3,4"
        else:
            keys = "m.idx"
            parent = "JOIN messages m ON m.session_id=s.id "
            order = "1,2,3"
        return (
            "SELECT s.source, split_part(s.source_path,'/roots/',2) source_file, "
            f"{keys}, {self.projection} FROM sessions s {parent}{self.joins} "
            f"WHERE starts_with(s.source_path, ?) ORDER BY {order}"
        )


RELATIONS = (
    Relation(
        "message_state",
        "r.* EXCLUDE(message_id)",
        "JOIN message_state r ON r.message_id=m.id",
        67076,
    ),
    Relation(
        "message_embeddings",
        "r.* EXCLUDE(message_id)",
        "JOIN message_embeddings r ON r.message_id=m.id",
        67076,
        vector_columns=(3, 4),
    ),
    Relation(
        "tool_calls",
        "tc.* EXCLUDE(id, session_id, message_id, idx, agent_id)",
        "",
        25876,
        tool=True,
    ),
    Relation(
        "tool_call_embeddings",
        "r.* EXCLUDE(tool_call_id)",
        "JOIN tool_call_embeddings r ON r.tool_call_id=tc.id",
        25876,
        tool=True,
        vector_columns=(4,),
    ),
    Relation(
        "tool_results",
        "r.* EXCLUDE(tool_call_id)",
        "JOIN tool_results r ON r.tool_call_id=tc.id",
        23398,
        tool=True,
    ),
    Relation(
        "tool_use_ids",
        "r.* EXCLUDE(tool_call_id, session_id)",
        "JOIN tool_use_ids r ON r.tool_call_id=tc.id",
        23408,
        tool=True,
    ),
)


def exact_row(row: tuple[Any, ...], vector_columns: tuple[int, ...]) -> tuple[Any, ...]:
    values = list(row)
    for column in vector_columns:
        vector = values[column]
        if vector is not None:
            assert isinstance(vector, (list, tuple)) and len(vector) == 384
            assert all(math.isfinite(value) for value in vector)
            values[column] = struct.pack("<384f", *vector)
    return tuple(values)


def owned_data(label: str) -> Path:
    assert re.fullmatch(r"[a-z0-9-]{1,32}", label)
    directory = LANE / "scratch/evidence" / label
    result = json.loads((directory / "result.json").read_text())
    assert result["outcome"] == "passed" and result["steps"]["cleanup"] == 0, label
    data = Path(json.loads((directory / "owned.json").read_text())["data"])
    assert data == Path(os.environ["TMPDIR"]) / label / "data", data
    return data


def compare(reference: str, candidate: str) -> dict[str, object]:
    with ExitStack() as stack:
        connections = []
        prefixes = []
        for label in (reference, candidate):
            data = owned_data(label)
            connections.append(
                stack.enter_context(
                    duckdb.connect(
                        str(data / "recall.duckdb"),
                        read_only=True,
                        config={
                            "threads": "1",
                            "memory_limit": "1GB",
                            "temp_directory": str(data / "payload-spill"),
                        },
                    )
                )
            )
            prefixes.append(str(data.parent / "roots") + "/")
        observations = {}
        for relation in RELATIONS:
            cursors = [
                conn.execute(relation.sql(), [prefix])
                for conn, prefix in zip(connections, prefixes, strict=True)
            ]
            count = 0
            while count <= relation.expected_rows:
                batches = [cursor.fetchmany(256) for cursor in cursors]
                if len(batches[0]) != len(batches[1]):
                    raise ValueError(f"{relation.name}: different row counts after {count}")
                if not batches[0]:
                    break
                for left, right in zip(*batches, strict=True):
                    if exact_row(left, relation.vector_columns) != exact_row(
                        right, relation.vector_columns
                    ):
                        raise ValueError(f"{relation.name}: different payload at ordinal {count}")
                    count += 1
            assert count == relation.expected_rows, (relation.name, count)
            observations[relation.name] = {"rows": count, "exact_equal": True}
        return {
            "outcome": "passed",
            "reference": reference,
            "candidate": candidate,
            "scope": (
                "fixed-pool rows only; derived IDs/agent aliases and unrelated history excluded"
            ),
            "vector_comparison": (
                "finite float32 values packed to exact bytes, including signed zero"
            ),
            "relations": observations,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="baseline-1")
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    output = LANE / "scratch/evidence" / f"{args.candidate}.payload-equality.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
