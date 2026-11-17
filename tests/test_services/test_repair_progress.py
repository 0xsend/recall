"""The progress file is written by one thread, bounded, strict, and gone when the repair ends."""

import json
import os
import stat
from pathlib import Path

import pytest
from recall.core.repair_progress import (
    PROGRESS_BYTES_MAX,
    PROGRESS_FILENAME,
    PROGRESS_TEMP_FILENAME,
    read_progress,
)
from recall.db.oracle_sort import TableSortStats
from recall.services.repair_progress import ProgressPublisher

TABLE = TableSortStats(runs=1, merge_levels=0, bytes_merged=0, stream_bytes=1000)


def _publisher(tmp_path: Path, now: list[float]) -> ProgressPublisher:
    return ProgressPublisher(tmp_path, clock=lambda: now[0])


def test_the_file_exists_when_the_repair_begins_and_is_private(tmp_path: Path) -> None:
    publisher = ProgressPublisher(tmp_path)
    try:
        publisher.begin("index", 1)
        record = read_progress(tmp_path)
        assert record is not None
        assert (record.owner_pid, record.attempt_no, record.trigger_kind) == (
            os.getpid(),
            1,
            "index",
        )
        assert record.phase == "admitting"
        assert stat.S_IMODE((tmp_path / PROGRESS_FILENAME).stat().st_mode) == 0o600
        assert (tmp_path / PROGRESS_FILENAME).stat().st_size <= PROGRESS_BYTES_MAX
    finally:
        publisher.end()


def test_a_normal_start_never_creates_the_file(tmp_path: Path) -> None:
    publisher = ProgressPublisher(tmp_path)
    publisher.phase("capturing_reference")
    publisher.table_finished("main.t", TABLE)
    publisher.end()
    assert list(tmp_path.iterdir()) == []


def test_unit_counters_reset_with_the_phase_and_never_decrease_within_it(tmp_path: Path) -> None:
    now = [100.0]
    publisher = _publisher(tmp_path, now)
    try:
        publisher.begin("index", 1)
        publisher.phase("capturing_reference")
        publisher.tables_planned(4)
        publisher.table_finished("main.a", TABLE)
        publisher.table_finished("main.b", TABLE)
        line = publisher.last_line()
        assert line is not None and "2 of 4 tables done" in line
        publisher.tables_planned(2)
        line = publisher.last_line()
        assert line is not None and "2 of 6 tables done" in line
        publisher.phase("verifying_reference")
        line = publisher.last_line()
        assert line is not None and "tables done" not in line
    finally:
        publisher.end()


def test_ending_removes_the_file_and_its_temporary_and_a_late_write_cannot_return_it(
    tmp_path: Path,
) -> None:
    publisher = ProgressPublisher(tmp_path)
    publisher.begin("index", 1)
    publisher.end()
    assert list(tmp_path.iterdir()) == []
    # A write that was in flight when the repair ended finishes afterwards and is abandoned.
    record = _record()
    publisher._write(record)
    assert list(tmp_path.iterdir()) == [tmp_path / PROGRESS_TEMP_FILENAME]
    assert not (tmp_path / PROGRESS_FILENAME).exists()


def _record():
    from recall.core.repair_progress import ProgressRecord

    return ProgressRecord(
        schema=1,
        owner_pid=1,
        attempt_no=1,
        trigger_kind="index",
        phase="admitting",
        started_at=1.0,
        phase_started_at=1.0,
        updated_at=1.0,
    )


def test_the_next_owner_removes_what_a_dead_owner_left(tmp_path: Path) -> None:
    (tmp_path / PROGRESS_FILENAME).write_text("{}")
    (tmp_path / PROGRESS_TEMP_FILENAME).write_text("{")
    ProgressPublisher(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_unwritable_data_directory_does_not_stop_the_repair(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    publisher = ProgressPublisher(data)
    data.chmod(0o500)
    try:
        publisher.begin("index", 1)
        publisher.phase("capturing_reference")
        publisher.tables_planned(1)
        publisher.table_finished("main.a", TABLE)
        assert read_progress(data) is None
    finally:
        publisher.end()
        data.chmod(0o700)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not json",
        b"[]",
        b'{"schema": 2}',
        json.dumps(
            {
                "schema": 1,
                "owner_pid": 1,
                "attempt_no": 1,
                "trigger_kind": "index",
                "phase": "admitting",
                "started_at": 1.0,
                "phase_started_at": 1.0,
                "updated_at": 1.0,
                "unknown": True,
            }
        ).encode(),
        b"x" * (PROGRESS_BYTES_MAX + 1),
    ],
    ids=["empty", "text", "array", "schema", "unknown-key", "oversized"],
)
def test_an_invalid_file_reads_as_unknown(tmp_path: Path, payload: bytes) -> None:
    (tmp_path / PROGRESS_FILENAME).write_bytes(payload)
    assert read_progress(tmp_path) is None
