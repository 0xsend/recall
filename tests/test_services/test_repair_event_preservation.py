"""A later error must not erase an instruction needed by the next process."""

from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from recall.services.self_repair import marker_path, remember_disk_full, remember_fatal_failure


@pytest.mark.parametrize(
    "later", ["database has been invalidated", "Failed to delete all rows from index"]
)
def test_unfolded_instruction_survives_later_fatal(tmp_path: Path, later: str) -> None:
    remember_fatal_failure(
        tmp_path,
        duckdb.FatalException("Failed to delete all rows from index"),
        site="startup_sidecar",
        now=datetime(2026, 11, 1),
    )
    original = marker_path(tmp_path).read_bytes()
    remember_fatal_failure(
        tmp_path,
        duckdb.FatalException(later),
        site="writer_open",
        now=datetime(2026, 11, 1) + timedelta(seconds=1),
    )
    assert marker_path(tmp_path).read_bytes() == original
    records = list((tmp_path / "fatal-events").glob("*.json"))
    assert len(records) == 1
    assert b"writer_open" in records[0].read_bytes()


def test_disk_full_verification_is_a_separate_instruction(tmp_path: Path) -> None:
    remember_fatal_failure(
        tmp_path,
        duckdb.FatalException("Failed to delete all rows from index"),
        site="startup_sidecar",
        now=datetime(2026, 11, 1),
    )
    original = marker_path(tmp_path).read_bytes()
    assert remember_disk_full(tmp_path)
    assert marker_path(tmp_path).read_bytes() == original
    records = list((tmp_path / "fatal-events").glob("*.json"))
    assert len(records) == 1
    assert b'"needs_index_verification":true' in records[0].read_bytes()
