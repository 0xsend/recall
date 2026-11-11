"""Complete watched rosters cross the read boundary without scalar-path binding."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.services.live import live_metadata_coverage, live_roster_query


@pytest.mark.parametrize(
    "paths",
    [
        (),
        ("quote'百分%.jsonl",),
        ("quote'百分%.jsonl",) * 2,
        tuple(f"source-{n}.jsonl" for n in range(4096)),
    ],
)
def test_live_roster_keeps_complete_unknown_membership(
    tmp_path: Path, paths: tuple[str, ...]
) -> None:
    watched = [str(tmp_path / path) for path in paths]
    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=8)
        query = live_roster_query(
            watched_paths=watched,
            include_idle=False,
            now=datetime(2026, 11, 10),
            idle_window_seconds=86400,
        )
        sql, params = query
        actual = conn.execute(
            sql + "SELECT source_path FROM classified WHERE matches ORDER BY source_path", params
        ).fetchall()
        assert actual == [(path,) for path in sorted(set(watched))]
        coverage = live_metadata_coverage(
            conn, query=query, watched_count=len(watched), catalog_scan_complete=True
        )
        assert coverage.observed_paths == len(set(watched))
        assert coverage.indexed_paths == 0
        assert coverage.unindexed_paths == len(set(watched))
        assert coverage.unknown_count == 0
        assert coverage.complete


def test_live_roster_binding_work_does_not_scale_with_members(tmp_path: Path) -> None:
    script = """
import json, sys
from datetime import datetime
import duckdb
from recall.db.schema import ensure_schema
from recall.services.live import live_metadata_coverage, live_roster_query
conn=duckdb.connect(':memory:')
ensure_schema(conn, embed_dim=8)
imports=[0]
def audit(event,args):
    if event=='import' and args[0]=='pandas': imports[0]+=1
sys.addaudithook(audit)
results=[]
for count in (13,4096):
    paths=[sys.argv[1]+'/'+str(n) for n in range(count)]
    query=live_roster_query(watched_paths=paths,include_idle=False,now=datetime(2026,11,10),idle_window_seconds=86400)
    before=imports[0]
    coverage=live_metadata_coverage(conn,query=query,watched_count=count,catalog_scan_complete=True)
    results.append({'members':coverage.observed_paths,'scalar_probes':imports[0]-before})
conn.close()
print(json.dumps(results))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    small, large = json.loads(completed.stdout)
    assert small["members"] == 13
    assert large["members"] == 4096
    # DuckDB may probe optional scalar types, but may not probe once per path.
    assert large["scalar_probes"] <= small["scalar_probes"] + 4


@pytest.mark.parametrize(
    "names",
    [
        (),
        ("quote'百分%.jsonl",),
        ("quote'百分%.jsonl",) * 2,
        tuple(f"source-{n}.jsonl" for n in range(4096)),
    ],
)
def test_indexed_roster_join_keeps_every_member(tmp_path: Path, names: tuple[str, ...]) -> None:
    from recall.services.live import live_session_rows
    from recall.services.live_session_set import LiveMember

    paths = [tmp_path / name for name in names]
    indexed = sorted(set(paths))[::2]
    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=8)
        for number, path in enumerate(indexed):
            conn.execute(
                "INSERT INTO sessions VALUES (?, 'codex', ?, NULL)",
                [f"session-{number}", str(path)],
            )
        expected_ids = {str(path): f"session-{number}" for number, path in enumerate(indexed)}
        rows = live_session_rows([LiveMember(p, 100, 50, str(p.parent)) for p in paths], conn=conn)
        assert [(r.path, r.session_id) for r in rows] == [
            (str(p), expected_ids.get(str(p))) for p in paths
        ]
        assert all(r.mtime == 100 and r.last_event_at == 50 for r in rows)
