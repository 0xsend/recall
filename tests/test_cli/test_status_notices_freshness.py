"""Freshness/staleness notice must be timezone-correct.

The daemon writes runtime_state timestamps and the CLI computes their age to
decide whether to warn ``daemon appears stale``. Each timestamp makes a full
round-trip through a DuckDB naive ``TIMESTAMP`` column: an *aware* UTC datetime
is silently converted to the process-local zone (and tzinfo dropped) on insert.
If the reader then re-attaches UTC, the computed age is inflated by the host's
UTC offset, firing a false staleness warning seconds after a successful run.

Existing notice tests hand-build *aware* ISO strings and never touch the DB, so
they cannot see this. This test forces a non-UTC process tz and drives the real
write -> read -> notice path.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest
from recall.cli.status_notices import (
    _freshness_staleness_notice,
    _informational_notice,
)
from recall.core.types import RunKind
from recall.db.schema import ensure_schema
from recall.services.runtime_state import (
    IndexRunCounts,
    load_runtime_status_from_conn,
    record_run_success,
)


@pytest.fixture
def behind_utc_tz(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force a process tz behind UTC (UTC-3, no DST) and restore libc state on teardown.

    DuckDB converts an aware datetime to the *process-local* zone (libc), not its
    ``SET TimeZone`` session var, so the bug is driven with ``TZ`` + ``tzset()``.
    Teardown must call ``tzset()`` again or the restored ``TZ`` never takes and
    later tests inherit America/Sao_Paulo.
    """
    monkeypatch.setenv("TZ", "America/Sao_Paulo")
    time.tzset()
    try:
        yield
    finally:
        monkeypatch.undo()
        time.tzset()


def _status_from_conn(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Build the status dict the CLI notices consume, mirroring the RPC serializer.

    The live status path serializes ``last_successful_at`` via
    ``datetime.isoformat()``; on a naive DuckDB value that yields a naive ISO
    string with no offset (the real-world ``"2026-07-07T09:39:16.015024"``).
    """
    runtime = load_runtime_status_from_conn(conn)
    assert runtime.last_successful_at is not None
    assert runtime.last_run_kind is not None
    return {
        "installed": True,
        "resolved_mode": "watch",
        "runtime_status": {
            "last_successful_at": runtime.last_successful_at.isoformat(),
            "last_run_kind": runtime.last_run_kind.value,
            "last_index_summary": {
                "total": 1,
                "indexed": 1,
                "skipped": 0,
                "failed": 0,
                "changed": 1,
            },
        },
    }


def test_fresh_run_not_reported_stale_in_non_utc_tz(tmp_path: Path, behind_utc_tz: None) -> None:
    conn = duckdb.connect(str(tmp_path / "recall.duckdb"))
    try:
        ensure_schema(conn)
        record_run_success(
            conn,
            run_kind=RunKind.DAEMON_WATCH,
            index_summary=IndexRunCounts(total=1, indexed=1, skipped=0, failed=0, changed=1),
        )
        status = _status_from_conn(conn)
    finally:
        conn.close()

    # A run that completed ~now must not be flagged stale, regardless of host tz.
    assert _freshness_staleness_notice(status, interval_seconds=300) is None

    # ...and the informational line must read "0m", not the host's UTC offset.
    info = _informational_notice(status["runtime_status"])
    assert info is not None
    assert "last updated 0m ago" in info, info
