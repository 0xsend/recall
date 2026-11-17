"""The external sorter writes the sorted multiset and merges each byte once per level."""

import hashlib
import random
from pathlib import Path

import pytest
from recall.db import oracle_sort
from recall.db.oracle_sort import (
    FRAME,
    SORT_DIRECTORY,
    SORT_FILES_MAX,
    SortContext,
    SortedStream,
    SortShape,
    sort_rows,
)

RECORD_BYTES = 100


def _records(count: int, seed: int = 7) -> list[bytes]:
    generator = random.Random(seed)
    # Duplicates and prefixes among random records; every record has a distinct length class.
    base = [generator.randbytes(RECORD_BYTES) for _ in range(count)]
    return base + base[: count // 10] + [b"", b"\x00", base[0][:50]]


def _sort(records: list[bytes], directory: Path, shape: SortShape) -> SortedStream:
    pending = iter(records)

    def write_next(sink) -> bool:
        record = next(pending, None)
        if record is None:
            return False
        sink.write(record)
        return True

    return sort_rows(write_next, directory, "table-0.rows", SortContext(shape))


def _stream_records(path: Path) -> list[bytes]:
    data = path.read_bytes()
    records, offset = [], 0
    while offset < len(data):
        (length,) = FRAME.unpack_from(data, offset)
        records.append(data[offset + FRAME.size : offset + FRAME.size + length])
        offset += FRAME.size + length
    return records


@pytest.mark.parametrize(
    "shape",
    [
        SortShape(),
        SortShape(run_bytes=1000, fan_in=2),
        SortShape(run_bytes=1000, fan_in=3),
        SortShape(run_bytes=1000, fan_in=64),
        SortShape(run_bytes=1000, fan_in=4, file_backed_bytes=60),
        SortShape(run_bytes=1000, fan_in=2, file_backed_bytes=1),
    ],
    ids=["defaults", "fan-in 2", "fan-in 3", "fan-in 64", "file-backed 60", "all file-backed"],
)
def test_the_stream_is_the_sorted_multiset_with_its_own_hash(
    tmp_path: Path, shape: SortShape
) -> None:
    records = _records(400)
    result = _sort(records, tmp_path, shape)
    stream = tmp_path / "table-0.rows"
    assert _stream_records(stream) == sorted(records)
    assert result.rows == len(records)
    assert result.byte_count == stream.stat().st_size
    assert result.sha256 == hashlib.sha256(stream.read_bytes()).hexdigest()
    assert not (tmp_path / SORT_DIRECTORY).exists()
    assert [path.name for path in tmp_path.iterdir()] == ["table-0.rows"]


def test_an_empty_scan_writes_an_empty_stream(tmp_path: Path) -> None:
    result = _sort([], tmp_path, SortShape())
    assert (tmp_path / "table-0.rows").read_bytes() == b""
    assert (result.rows, result.byte_count, result.stats.runs) == (0, 0, 0)
    assert result.sha256 == hashlib.sha256(b"").hexdigest()


def test_the_sorter_directory_stays_within_the_validator_bound(tmp_path: Path) -> None:
    peak = 0

    def observe(stage: str) -> None:
        nonlocal peak
        if (tmp_path / SORT_DIRECTORY).exists():
            peak = max(peak, len(list((tmp_path / SORT_DIRECTORY).iterdir())))

    records = _records(3000)
    pending = iter(records)

    def write_next(sink) -> bool:
        record = next(pending, None)
        if record is None:
            return False
        sink.write(record)
        return True

    shape = SortShape(run_bytes=200, fan_in=2)
    result = sort_rows(write_next, tmp_path, "table-0.rows", SortContext(shape, observe))
    assert result.stats.runs > 1000
    assert 0 < peak < SORT_FILES_MAX


@pytest.mark.parametrize(
    ("runs", "levels"), [(4, 1), (16, 2), (64, 3)], ids=["4 runs", "16 runs", "64 runs"]
)
def test_each_byte_is_merged_once_per_level(tmp_path: Path, runs: int, levels: int) -> None:
    # Ten 100-byte records fill a 1,000-byte run; fan-in 4.
    generator = random.Random(runs)
    records = [generator.randbytes(RECORD_BYTES) for _ in range(runs * 10)]
    shape = SortShape(run_bytes=1000, fan_in=4)
    result = _sort(records, tmp_path, shape)
    stats = result.stats
    assert (stats.runs, stats.merge_levels) == (runs, levels)
    assert stats.stream_bytes == result.byte_count
    assert stats.bytes_merged <= stats.merge_levels * stats.stream_bytes


def test_a_sorter_that_accumulates_into_one_growing_run_breaks_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def accumulate(self, level: int, run) -> None:
        self.levels[0].append(run)
        if len(self.levels[0]) >= self.context.shape.fan_in:
            group, self.levels[0] = self.levels[0], []
            self.levels[0].append(self.merge(group, self.next_path("accumulated")))

    monkeypatch.setattr(oracle_sort._Levels, "add", accumulate)
    generator = random.Random(64)
    records = [generator.randbytes(RECORD_BYTES) for _ in range(640)]
    result = _sort(records, tmp_path, SortShape(run_bytes=1000, fan_in=4))
    stats = result.stats
    assert stats.runs == 64
    assert stats.bytes_merged > stats.merge_levels * stats.stream_bytes
