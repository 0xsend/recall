"""The external sorter that turns one table scan into one canonical row stream.

A stream is the table's records, each a length-prefixed byte string, in byte
order with duplicates kept. Records are totally ordered and equal records are
identical bytes, so the stream is a function of the multiset of records alone:
run size, fan-in and level count never change it.

Records are collected into in-memory runs, each sorted with the native byte
comparison. Runs merge in balanced levels of at most `fan_in` files, so each
byte is copied once per level and never accumulates into a growing file. A
record larger than `file_backed_bytes` is never held in memory: it is spilled
to its own run file and compared through chunked reads.

Every file the sorter creates besides the final stream lives in the `sort`
directory of the caller's working container.
"""

from __future__ import annotations

import hashlib
import heapq
import io
import os
import struct
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

SORT_DIRECTORY = "sort"
# Fail-closed ceiling on what the validator accepts in that directory. The
# sorter's own bound (levels times fan-in) is far below it and is asserted.
SORT_FILES_MAX = 1024

SORT_BYTES_MAX = 64 * 1024 * 1024
RUN_BYTES_DEFAULT = 32 * 1024 * 1024
# A service on macOS inherits a soft limit of 256 descriptors; a merge holds
# at most this many run files open plus its output.
MERGE_FAN_IN = 64
FILE_BACKED_BYTES_DEFAULT = 16 * 1024 * 1024
CHUNK_BYTES = 64 * 1024
FRAME = struct.Struct(">Q")
_OUTPUT_BATCH_RECORDS = 8192
_COPY_BYTES = 1024 * 1024

Barrier = Callable[[str], None]


def _no_barrier(stage: str) -> None:
    pass


class ByteSink(Protocol):
    """What a record writer needs: bytes appended in order."""

    def write(self, data: bytes, /) -> object: ...


@dataclass(frozen=True)
class SortShape:
    """How a sort is cut into runs and merges; it never changes the stream."""

    run_bytes: int = RUN_BYTES_DEFAULT
    fan_in: int = MERGE_FAN_IN
    file_backed_bytes: int = FILE_BACKED_BYTES_DEFAULT

    def __post_init__(self) -> None:
        if not 1 <= self.run_bytes <= SORT_BYTES_MAX:
            raise ValueError("sort run must fit within 64 MiB")
        if not 2 <= self.fan_in <= MERGE_FAN_IN:
            raise ValueError("merge fan-in must be between 2 and 64")
        if self.file_backed_bytes < 1:
            raise ValueError("file-backed record threshold must be positive")


DEFAULT_SHAPE = SortShape()


@dataclass(frozen=True)
class TableSortStats:
    """What sorting one table cost: runs made, levels needed, bytes merged."""

    runs: int
    merge_levels: int
    bytes_merged: int
    stream_bytes: int


@dataclass(frozen=True)
class SortedStream:
    rows: int
    byte_count: int
    sha256: str
    stats: TableSortStats


class ScanReport(Protocol):
    """Receives the unit progress of a whole-database scan."""

    def tables_planned(self, count: int) -> None: ...

    def table_started(self, name: str) -> None: ...

    def table_finished(self, name: str, stats: TableSortStats) -> None: ...


class _NoReport:
    def tables_planned(self, count: int) -> None:
        pass

    def table_started(self, name: str) -> None:
        pass

    def table_finished(self, name: str, stats: TableSortStats) -> None:
        pass


NO_REPORT: ScanReport = _NoReport()


@dataclass(frozen=True)
class SortContext:
    shape: SortShape = DEFAULT_SHAPE
    barrier: Barrier = _no_barrier
    report: ScanReport = NO_REPORT


def merge_levels(runs: int, fan_in: int) -> int:
    """Levels a table of `runs` runs needs: 0 for at most one run, else ceil(log_fan_in(runs))."""
    levels, capacity = 0, 1
    while capacity < runs:
        capacity *= fan_in
        levels += 1
    return levels


@dataclass(frozen=True)
class _Run:
    path: Path
    size: int
    # None when the run was spilled: its header is patched after it is hashed.
    sha256: str | None


class _BigRecord:
    """A record held in a run file, compared and copied through chunked reads."""

    def __init__(self, path: Path, offset: int, length: int) -> None:
        self.path = path
        self.offset = offset
        self.length = length

    def __len__(self) -> int:
        return self.length

    def read(self, offset: int, size: int) -> bytes:
        with self.path.open("rb") as stream:
            stream.seek(self.offset + offset)
            block = stream.read(size)
        assert len(block) == size, "truncated content record"
        return block

    def copy_to(self, output: BinaryIO, digest: hashlib._Hash) -> None:
        for offset in range(0, self.length, _COPY_BYTES):
            block = self.read(offset, min(_COPY_BYTES, self.length - offset))
            digest.update(block)
            output.write(block)

    def _order(self, other: object) -> int:
        assert isinstance(other, (bytes, _BigRecord))
        shared = min(self.length, len(other))
        for offset in range(0, shared, CHUNK_BYTES):
            size = min(CHUNK_BYTES, shared - offset)
            mine = self.read(offset, size)
            theirs = (
                other[offset : offset + size]
                if isinstance(other, bytes)
                else other.read(offset, size)
            )
            if mine != theirs:
                return -1 if mine < theirs else 1
        return (self.length > len(other)) - (self.length < len(other))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (bytes, _BigRecord)):
            return NotImplemented
        return self._order(other) == 0

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, (bytes, _BigRecord)):
            return NotImplemented
        return self._order(other) < 0

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, (bytes, _BigRecord)):
            return NotImplemented
        return self._order(other) > 0


_Record = bytes | _BigRecord


def _read_run(path: Path, file_backed_bytes: int) -> Generator[_Record, None, None]:
    with path.open("rb", buffering=_COPY_BYTES) as stream:
        offset = 0
        while True:
            header = stream.read(FRAME.size)
            if not header:
                return
            assert len(header) == FRAME.size, "truncated content frame"
            (length,) = FRAME.unpack(header)
            offset += FRAME.size
            if length > file_backed_bytes:
                yield _BigRecord(path, offset, length)
                stream.seek(length, os.SEEK_CUR)
            else:
                record = stream.read(length)
                assert len(record) == length, "truncated content record"
                yield record
            offset += length


class _Spill:
    """One record being written; it moves to its own run file once it is too large."""

    def __init__(
        self, buffer: io.BytesIO, start: int, limit: int, path: Callable[[], Path]
    ) -> None:
        self._buffer = buffer
        self._start = start
        self._limit = limit
        self._allocate = path
        self._size = 0
        self.path: Path | None = None
        self._file: BinaryIO | None = None

    def write(self, data: bytes, /) -> int:
        self._size += len(data)
        if self._file is None and self._size <= self._limit:
            return self._buffer.write(data)
        if self._file is None:
            self.path = self._allocate()
            self._file = self.path.open("wb")
            self._file.write(FRAME.pack(0))
            view = self._buffer.getbuffer()
            self._file.write(view[self._start :])
            view.release()
            self._buffer.truncate(self._start)
            self._buffer.seek(self._start)
        return self._file.write(data)

    def close(self) -> int | None:
        """The size of the spilled run file, or None when the record stayed in memory."""
        if self._file is None:
            return None
        self._file.seek(0)
        self._file.write(FRAME.pack(self._size))
        self._file.close()
        assert self.path is not None
        return FRAME.size + self._size


class _Levels:
    """Runs awaiting a merge, grouped so a level merges when it holds `fan_in` runs."""

    def __init__(self, directory: Path, context: SortContext) -> None:
        self.directory = directory
        self.context = context
        self.levels: list[list[_Run]] = [[]]
        self.serial = 0
        self.bytes_merged = 0

    def next_path(self, stem: str) -> Path:
        self.serial += 1
        return self.directory / f"{stem}-{self.serial}.rows"

    def count(self) -> int:
        return sum(len(level) for level in self.levels)

    def add(self, level: int, run: _Run) -> None:
        while len(self.levels) <= level:
            self.levels.append([])
        self.levels[level].append(run)
        assert self.count() <= SORT_FILES_MAX, "sorter exceeded its file bound"
        if len(self.levels[level]) >= self.context.shape.fan_in:
            group, self.levels[level] = self.levels[level], []
            self.add(level + 1, self.merge(group, self.next_path(f"m{level + 1}")))
            self.context.barrier("sort_level_merged")

    def merge(self, runs: list[_Run], destination: Path) -> _Run:
        self.bytes_merged += sum(run.size for run in runs)
        return merge_runs(runs, destination, self.context.shape.file_backed_bytes)

    def consolidate(self, ceiling: int) -> None:
        """Merge until at most `ceiling` runs remain, lowest levels first.

        A level with several runs merges whole into the next level, which keeps
        every byte's merge count within the level formula. Only when every
        level holds a single run (a small fan-in) do the lowest runs of
        different levels merge together.
        """
        while self.count() > ceiling:
            crowded = next((index for index, runs in enumerate(self.levels) if len(runs) > 1), None)
            if crowded is not None:
                group, self.levels[crowded] = self.levels[crowded], []
                target = crowded + 1
            else:
                taken = [index for index, runs in enumerate(self.levels) if runs][
                    : self.context.shape.fan_in
                ]
                group = [self.levels[index].pop() for index in taken]
                target = taken[-1] + 1
            self.add(target, self.merge(group, self.next_path(f"m{target}")))
            self.context.barrier("sort_level_merged")

    def remaining(self) -> list[_Run]:
        return [run for level in self.levels for run in level]


def merge_runs(runs: list[_Run], destination: Path, file_backed_bytes: int) -> _Run:
    """Merge sorted runs into `destination`, hashing what is written, and delete the inputs."""
    digest = hashlib.sha256()
    size = 0
    with destination.open("wb", buffering=_COPY_BYTES) as output:
        batch: list[bytes] = []

        def flush() -> None:
            nonlocal size
            block = b"".join(batch)
            batch.clear()
            digest.update(block)
            output.write(block)
            size += len(block)

        readers = [_read_run(run.path, file_backed_bytes) for run in runs]
        try:
            for record in heapq.merge(*readers):
                if isinstance(record, _BigRecord):
                    flush()
                    header = FRAME.pack(record.length)
                    digest.update(header)
                    output.write(header)
                    record.copy_to(output, digest)
                    size += FRAME.size + record.length
                    continue
                batch.append(FRAME.pack(len(record)))
                batch.append(record)
                if len(batch) >= 2 * _OUTPUT_BATCH_RECORDS:
                    flush()
            flush()
        finally:
            for reader in readers:
                reader.close()
    for run in runs:
        run.path.unlink()
    return _Run(destination, size, digest.hexdigest())


def _write_run(path: Path, records: list[bytes]) -> _Run:
    digest = hashlib.sha256()
    size = 0
    with path.open("wb", buffering=_COPY_BYTES) as output:
        for start in range(0, len(records), _OUTPUT_BATCH_RECORDS):
            block = b"".join(
                FRAME.pack(len(record)) + record
                for record in records[start : start + _OUTPUT_BATCH_RECORDS]
            )
            digest.update(block)
            output.write(block)
            size += len(block)
    return _Run(path, size, digest.hexdigest())


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


STREAM_DIGESTS_MAX = 16 * 1024


class StreamDigests:
    """SHA-256 of finished stream files, remembered by file state within one process.

    A stream is hashed once while it is written and once per later reader; a
    reader in the same process that finds the same path, size, modification
    time and inode reuses the answer. A new process always hashes afresh.
    """

    def __init__(self) -> None:
        self._known: dict[tuple[str, int, int, int], str] = {}

    def of(self, path: Path) -> str:
        status = path.stat()
        key = (str(path), status.st_size, status.st_mtime_ns, status.st_ino)
        digest = self._known.get(key)
        if digest is None:
            digest = file_sha256(path)
            if len(self._known) >= STREAM_DIGESTS_MAX:
                self._known.clear()
            self._known[key] = digest
        return digest


def sort_rows(
    write_next: Callable[[ByteSink], bool],
    directory: Path,
    filename: str,
    context: SortContext,
) -> SortedStream:
    """Write `directory/filename` from every record `write_next` produces.

    `write_next` appends one record to the sink and returns False, writing
    nothing, once the scan is exhausted.
    """
    shape = context.shape
    work = directory / SORT_DIRECTORY
    work.mkdir()
    levels = _Levels(work, context)
    rows = 0
    runs_made = 0
    buffer = io.BytesIO()
    spans: list[tuple[int, int]] = []

    def close_memory_run() -> None:
        nonlocal runs_made
        if not spans:
            return
        data = buffer.getvalue()
        records = [data[start:end] for start, end in spans]
        records.sort()
        spans.clear()
        buffer.seek(0)
        buffer.truncate()
        levels.add(0, _write_run(levels.next_path("r0"), records))
        runs_made += 1
        context.barrier("sort_run_written")

    finished = False
    while not finished:
        start = buffer.tell()
        record = _Spill(buffer, start, shape.file_backed_bytes, lambda: levels.next_path("r0"))
        if not write_next(record):
            assert record.close() is None, "an exhausted scan wrote a partial record"
            buffer.truncate(start)
            finished = True
            break
        rows += 1
        spilled = record.close()
        if spilled is not None:
            assert record.path is not None
            levels.add(0, _Run(record.path, spilled, None))
            runs_made += 1
            context.barrier("sort_run_written")
            continue
        spans.append((start, buffer.tell()))
        if buffer.tell() >= shape.run_bytes:
            close_memory_run()
    close_memory_run()

    levels.consolidate(shape.fan_in)
    remaining = levels.remaining()
    destination = directory / filename
    if not remaining:
        destination.write_bytes(b"")
        final = _Run(destination, 0, hashlib.sha256().hexdigest())
    elif len(remaining) == 1:
        # One run is already the sorted stream.
        remaining[0].path.rename(destination)
        final = _Run(
            destination, remaining[0].size, remaining[0].sha256 or file_sha256(destination)
        )
    else:
        final = levels.merge(remaining, destination)
    work.rmdir()
    context.barrier("table_sorted")
    assert final.sha256 is not None
    return SortedStream(
        rows=rows,
        byte_count=final.size,
        sha256=final.sha256,
        stats=TableSortStats(
            runs=runs_made,
            merge_levels=merge_levels(runs_made, shape.fan_in),
            bytes_merged=levels.bytes_merged,
            stream_bytes=final.size,
        ),
    )
