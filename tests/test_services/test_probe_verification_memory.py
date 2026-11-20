"""Traversing unsupported samples never clears the durable verification request."""

from pathlib import Path

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.services.manual_repair import ManualRepair
from recall.services.owned_jobs import DaemonJobs
from recall.services.repair_events import read_events
from recall.services.repair_protocol import RepairProtocol
from recall.services.runtime_state import (
    load_runtime_status_from_conn,
    set_needs_index_verification,
)

from tests.test_services.test_repair_protocol import actionable
from tests.test_services.test_run_completion import _config


@pytest.mark.parametrize("flow", ["normal", "automatic", "manual"])
def test_unverifiable_coverage_retains_verification_memory(tmp_path: Path, flow: str):
    config = _config(tmp_path, None)
    with duckdb.connect(str(config.db_path)) as conn:
        ensure_schema(conn)
        conn.execute("CREATE TABLE acknowledged(k VARCHAR,other VARCHAR)")
        conn.execute("INSERT INTO acknowledged SELECT 'key-'||i,'other' FROM range(8) t(i)")
        conn.execute("CREATE INDEX compound_witness ON acknowledged(k,other)")
        set_needs_index_verification(conn, True)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    protocol = RepairProtocol(config.data_dir)
    if flow == "manual":
        protocol = ManualRepair(config)
        protocol.execute()
    elif flow == "automatic":
        actionable(config.data_dir)
        with duckdb.connect(str(config.db_path)) as conn, jobs.lease(conn) as lease:
            protocol.prepare(lease, read_events(config.data_dir))
            rebuilt = protocol.rebuild(lease, protocol.arm(lease))
        with duckdb.connect(str(config.db_path)) as fresh, jobs.lease(fresh) as lease:
            protocol.verify(lease, rebuilt)
    else:
        with duckdb.connect(str(config.db_path)) as conn, jobs.lease(conn) as lease:
            protocol.begin_start(lease)
    assert protocol.last_probe is not None
    assert protocol.last_probe.complete and protocol.last_probe.samples_unverifiable == 2
    with duckdb.connect(str(config.db_path), read_only=True) as conn:
        assert load_runtime_status_from_conn(conn).needs_index_verification
