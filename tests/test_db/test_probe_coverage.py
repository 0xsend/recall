"""Equal counts qualify coverage only with attributed analyzed index access."""

import copy
import json
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest
from recall.db.maintenance import probe_index_divergence


@pytest.mark.parametrize(
    "shape", ["compound", "duplicate", "constraint", "expression", "sequential"]
)
def test_equal_counts_do_not_overstate_named_index_coverage(tmp_path: Path, shape: str):
    database = tmp_path / "coverage.duckdb"
    with duckdb.connect(str(database)) as conn:
        constraint = " UNIQUE" if shape == "constraint" else ""
        conn.execute(f"CREATE TABLE probe_keys(k VARCHAR{constraint},other VARCHAR)")
        conn.execute(
            "INSERT INTO probe_keys SELECT 'key-'||i,'group-'||(i%4) FROM range(10000) t(i)"
        )
        if shape == "compound":
            conn.execute("CREATE INDEX intended ON probe_keys(k,other)")
        else:
            conn.execute("CREATE INDEX intended ON probe_keys(k)")
        if shape == "duplicate":
            conn.execute("CREATE INDEX competing ON probe_keys(k)")
        elif shape == "expression":
            conn.execute("CREATE INDEX transformed ON probe_keys(lower(k))")
        elif shape == "sequential":
            conn.execute("SET disabled_optimizers='filter_pushdown'")
        report = probe_index_divergence(conn, sample=2, budget_seconds=30)
        assert report.complete and report.diverged == ()
        assert report.samples_checked == 0
        assert report.samples_unverifiable == (4 if shape == "duplicate" else 2)


def test_one_attributed_native_index_qualifies_sampled_coverage_readonly(tmp_path: Path):
    database = tmp_path / "coverage.duckdb"
    with duckdb.connect(str(database)) as conn:
        conn.execute("CREATE TABLE probe_keys(k VARCHAR)")
        conn.execute("INSERT INTO probe_keys SELECT 'key-'||i FROM range(10000) t(i)")
        conn.execute("CREATE INDEX intended ON probe_keys(k)")
        conn.execute("CHECKPOINT")
    before = database.read_bytes()
    with duckdb.connect(str(database), read_only=True) as conn:
        report = probe_index_divergence(conn, sample=2, budget_seconds=30)
        assert report.complete and report.diverged == ()
        assert report.indexes_probed == 1
        assert report.samples_checked == 2
        assert report.samples_unverifiable == 0
    assert database.read_bytes() == before


@pytest.mark.parametrize(
    "missing",
    [
        "plan",
        "access",
        "table",
        "predicate",
        "cardinality",
        "ambiguous",
        "malformed",
        "oversized",
        "depth",
    ],
)
def test_unrecognized_analyzed_access_never_counts_checked(tmp_path: Path, missing: str):
    with duckdb.connect(str(tmp_path / "coverage.duckdb")) as conn:
        conn.execute("CREATE TABLE probe_keys(k VARCHAR)")
        conn.execute("INSERT INTO probe_keys SELECT 'key-'||i FROM range(10000) t(i)")
        conn.execute("CREATE INDEX intended ON probe_keys(k)")
        calls = 0

        class Result:
            def __init__(self, payload):
                self.payload = payload

            def fetchone(self):
                return "analyzed_plan", self.payload

        class AnalyzedConnection:
            def execute(self, sql, params=None):
                nonlocal calls
                result = conn.execute(sql, params) if params is not None else conn.execute(sql)
                if not sql.startswith("EXPLAIN (ANALYZE,FORMAT JSON)"):
                    return result
                calls += 1
                row = result.fetchone()
                assert row is not None
                plan = json.loads(row[1])
                stack = [plan]
                scans = []
                while stack:
                    node = stack.pop()
                    if node.get("operator_type") == "TABLE_SCAN":
                        scans.append(node)
                    stack.extend(node.get("children", []))
                assert len(scans) == 1
                scan = scans[0]
                if missing == "plan":
                    plan = {}
                elif missing == "access":
                    scan["extra_info"]["Type"] = "Unrecognized Scan"
                elif missing == "table":
                    scan["extra_info"]["Table"] = "memory.main.unrelated"
                elif missing == "predicate":
                    scan["extra_info"]["Filters"] = "k='another-key'"
                elif missing == "cardinality":
                    scan["operator_cardinality"] = 0
                elif missing == "ambiguous":
                    plan["children"].append(copy.deepcopy(scan))
                elif missing == "depth":
                    for _ in range(34):
                        plan = {"children": [plan]}
                payload = json.dumps(plan)
                if missing == "malformed":
                    payload = "{not-json"
                elif missing == "oversized":
                    payload = " " * (2 * 1024 * 1024) + payload
                return Result(payload)

        report = probe_index_divergence(
            cast(Any, AnalyzedConnection()), sample=2, budget_seconds=30
        )
        assert report.complete and report.diverged == ()
        assert calls == 2
        assert report.samples_checked == 0
        assert report.samples_unverifiable == 2


def test_oversized_recent_key_remains_unverifiable_without_fetching_it_whole(tmp_path: Path):
    with duckdb.connect(str(tmp_path / "coverage.duckdb")) as conn:
        conn.execute("CREATE TABLE probe_keys(k VARCHAR)")
        conn.execute("CREATE INDEX intended ON probe_keys(k)")
        conn.execute("INSERT INTO probe_keys VALUES (?)", ["x" * (70 * 1024)])
        report = probe_index_divergence(conn, sample=2, budget_seconds=30)
        assert report.complete and report.diverged == ()
        assert report.samples_checked == 0 and report.samples_unverifiable == 1
