"""The runtime_state failure stamp always carries a message (REQ-RESIL-026).

A restart that interrupted an index request left `last_failure_at` set with
`last_failure_message: ""` on a live host, because the exception it recorded
stringified empty. A dated failure with no reason is unreadable.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import duckdb
import pytest
from recall.core.types import RunKind
from recall.db.schema import ensure_schema
from recall.services.runtime_state import (
    load_runtime_status_from_conn,
    record_run_failure,
)


@pytest.fixture
def conn() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    ensure_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


def test_record_run_failure_keeps_the_reported_message(conn: duckdb.DuckDBPyConnection) -> None:
    record_run_failure(conn, run_kind=RunKind.INDEX, message="catalog write-write conflict")

    status = load_runtime_status_from_conn(conn)
    assert status.last_failure_message == "catalog write-write conflict"
    assert status.last_failure_at is not None


def test_runtime_status_reads_fatal_memory_without_the_0029_column(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """REQ-RESIL-014: a read-only open cannot migrate, so a database one
    generation behind still reports the fatal columns it does have."""
    signature = "fresh_index:FatalException:database has been invalidated"
    conn.execute("ALTER TABLE runtime_state DROP COLUMN last_fatal_at")
    conn.execute(
        "UPDATE runtime_state SET last_fatal_signature = ?, fatal_repeat_count = 2 "
        "WHERE singleton = TRUE",
        [signature],
    )

    status = load_runtime_status_from_conn(conn)

    assert status.last_fatal_signature == signature
    assert status.fatal_repeat_count == 2
    assert status.last_fatal_at is None


def test_runtime_status_reads_the_fatal_columns_in_one_round_trip(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Status is loaded on every daemon-status call, so its reads must not multiply.

    `last_fatal_at` sits in the same singleton row as the rest of the fatal
    memory; selecting it separately bought nothing but a third query.
    """
    recording = _RecordingConnection(conn)

    status = load_runtime_status_from_conn(cast(duckdb.DuckDBPyConnection, recording))

    assert status.last_fatal_at is None
    # One read for the base columns, one for the fatal-memory columns.
    assert len(recording.statements) == 2
    assert "last_fatal_at" in recording.statements[1]


class _RecordingConnection:
    """A connection that counts the statements a read costs."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn
        self.statements: list[str] = []

    def execute(self, statement: str, *args: object, **kwargs: object) -> object:
        self.statements.append(statement)
        return self._conn.execute(statement, *args, **kwargs)


@pytest.mark.parametrize("reported", ["", "   \n"])
def test_record_run_failure_names_the_run_when_the_message_is_blank(
    conn: duckdb.DuckDBPyConnection, reported: str
) -> None:
    record_run_failure(conn, run_kind=RunKind.INDEX, message=reported)

    status = load_runtime_status_from_conn(conn)
    assert status.last_failure_at is not None
    assert status.last_failure_message == "unknown failure during the index run"
