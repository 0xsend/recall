"""Full durable queues hold even when every record is an ordinary diagnostic."""

import json
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest
from recall.core.config import AppConfig
from recall.db import connect
from recall.services.self_repair import DaemonStartupRefused, run_startup_self_repair


def _other(event_id: int) -> bytes:
    return json.dumps(
        {
            "version": 2,
            "event_id": str(UUID(int=event_id)),
            "failure": {
                "site": "writer",
                "exception_type": "FatalException",
                "message": "ordinary fatal",
                "signature": "ordinary",
                "failure_class": "other",
                "at": "2026-11-01T00:00:00",
            },
            "needs_index_verification": False,
        }
    ).encode()


@pytest.mark.parametrize("count", [64, 65])
def test_full_record_count_holds_before_diagnostic_retirement(tmp_path: Path, count: int) -> None:
    config = replace(AppConfig.load(), data_dir=tmp_path, db_path=tmp_path / "recall.duckdb")
    with connect(config) as conn:
        conn.execute("CREATE TABLE witness(value VARCHAR)")
        conn.execute("INSERT INTO witness VALUES ('acknowledged')")
    spool = tmp_path / "fatal-events"
    spool.mkdir()
    for event_id in range(1, count + 1):
        (spool / f"{UUID(int=event_id)}.json").write_bytes(_other(event_id))
    before = {path.name: path.read_bytes() for path in spool.iterdir()}
    with pytest.raises(DaemonStartupRefused, match="matched restore"):
        run_startup_self_repair(config)
    assert {path.name: path.read_bytes() for path in spool.iterdir()} == before
    with connect(config) as conn:
        assert conn.execute("SELECT * FROM witness").fetchall() == [("acknowledged",)]
        assert conn.execute("SELECT phase, attempt_no FROM daemon_index_repair").fetchone() == (
            "idle",
            0,
        )


@pytest.mark.parametrize("size", [1024 * 1024, 1024 * 1024 + 1])
def test_full_byte_count_holds_before_diagnostic_retirement(tmp_path: Path, size: int) -> None:
    config = replace(AppConfig.load(), data_dir=tmp_path, db_path=tmp_path / "recall.duckdb")
    with connect(config):
        pass
    spool = tmp_path / "fatal-events"
    spool.mkdir()
    record = _other(1)
    path = spool / f"{UUID(int=1)}.json"
    path.write_bytes(record + b" " * (size - len(record)))
    before = path.read_bytes()
    with pytest.raises(DaemonStartupRefused, match="matched restore"):
        run_startup_self_repair(config)
    assert path.read_bytes() == before
