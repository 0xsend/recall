"""A scan's originating instruction precedes cursor and connection cleanup."""

import errno
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import cast

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.db import repair_oracle as oracle
from recall.db.fts_sidecar import open_sidecar, upsert_message_fts
from recall.db.schema import ensure_schema
from recall.services.recovery_admission import file_jobs
from recall.services.repair_events import read_events


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("disk", [False, True])
def test_scan_observes_original_before_independent_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: bool, disk: bool
):
    config = replace(AppConfig.load(), data_dir=tmp_path, db_path=tmp_path / "recall.duckdb")
    jobs, admission = file_jobs(config)
    admission.observe_schema(41)
    primary = (
        OSError(errno.ENOSPC, "scan disk capacity")
        if disk
        else duckdb.FatalException("Failed to delete all rows from index in oracle scan")
    )
    cleanup = OSError(errno.EIO, "scan close unavailable")
    closes = 0

    def before_close():
        nonlocal closes
        closes += 1
        batch = read_events(tmp_path)
        if disk:
            assert any(
                event.failure is None and event.needs_index_verification
                for event in batch.identified()
            )
            assert batch.primary is None
        else:
            assert batch.primary is not None and batch.primary.failure is not None
            assert batch.primary.failure.message == str(primary)

    def fail_scan(*args):
        raise primary

    monkeypatch.setattr(oracle, "_sorted_rows", fail_scan)
    with duckdb.connect(str(config.db_path)) as conn:
        ensure_schema(conn)
        if native:
            conn.execute("CREATE TABLE witness(id INTEGER)")
            table = next(table for table in oracle._catalog(conn) if table.name == "witness")

            class Cursor:
                def __init__(self):
                    self.raw = conn.cursor()

                def execute(self, *args):
                    return self.raw.execute(*args)

                def close(self):
                    before_close()
                    self.raw.close()
                    raise cleanup

            class NativeConnection:
                def cursor(self):
                    return Cursor()

            with pytest.raises(BaseExceptionGroup) as caught:
                jobs.call_db(
                    "oracle_test",
                    lambda: oracle._capture_table(
                        cast(duckdb.DuckDBPyConnection, NativeConnection()),
                        table,
                        tmp_path,
                        128,
                        observer=jobs,
                    ),
                )
        else:
            sidecar = tmp_path / "recall.fts.sqlite"
            with closing(open_sidecar(sidecar)) as keyword, keyword:
                upsert_message_fts(keyword, "a", "acknowledged", "")
            connect = sqlite3.connect

            class KeywordConnection(sqlite3.Connection):
                def close(self):
                    before_close()
                    super().close()
                    raise cleanup

            monkeypatch.setattr(
                sqlite3,
                "connect",
                lambda *args, **kwargs: connect(*args, factory=KeywordConnection, **kwargs),
            )
            with pytest.raises(BaseExceptionGroup) as caught:
                jobs.call_db(
                    "oracle_test",
                    lambda: oracle._capture_sidecar(sidecar, tmp_path, 128, 0, observer=jobs),
                )
    assert caught.value.exceptions == (primary, cleanup)
    assert closes == 1
    if not disk:
        assert jobs.first_fatal is not None and jobs.first_fatal.graph.failure is primary
