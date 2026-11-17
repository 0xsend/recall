"""The pre-change capture comparator, kept verbatim as the oracle for stream identity.

Source: `packages/recall/src/recall/db/repair_oracle.py` at `20327fdcd782b25922a43e0df7ee65bb1412f695`; SHA-256 of the extracted functions below: c9c271e71f699f2a0e08bf2b42079db9d2a537eda6569e15f56822f44e92ea5c.
These are the functions that turned one scan into one sorted row stream before
the external sorter was replaced. The product must produce byte-identical
streams for the same inputs; nothing here may follow the product's changes.
"""

# ruff: noqa
from __future__ import annotations

import hashlib
import heapq
import sqlite3
import struct
from collections.abc import Callable
from dataclasses import dataclass
from functools import cmp_to_key
from pathlib import Path
from typing import BinaryIO

import duckdb

from recall import _native
from recall.core.failure_observation import FailureObserver, observe_before_cleanup
from recall.db.repair_oracle import (
    CATALOG_ITEMS_MAX,
    OracleColumn,
    OracleTable,
    OracleUnavailable,
    SidecarOracle,
    _catalog,
    _quote,
)

SORT_BYTES_MAX = 64 * 1024 * 1024
MERGE_WAYS_MAX = 32
RUN_ROWS_MAX = 8192
CHUNK_BYTES = 64 * 1024
FRAME = struct.Struct(">Q")


def _field(stream: BinaryIO, value: bytes | None) -> None:
    if value is None:
        stream.write(b"\x00")
    else:
        stream.write(b"\x01" + FRAME.pack(len(value)))
        stream.write(value)


def _array(value: list[float | None], data_type: str) -> bytes:
    code = ">f" if data_type.startswith("FLOAT") else ">d"
    return FRAME.pack(len(value)) + b"".join(
        b"\x00" if item is None else b"\x01" + struct.pack(code, item) for item in value
    )


def _write_row(
    conn: duckdb.DuckDBPyConnection,
    stream: BinaryIO,
    table: OracleTable,
    rowid: int,
    values: tuple[object, ...],
) -> None:
    qualified = f"{_quote(table.schema_name)}.{_quote(table.name)}"
    for column, value in zip(table.columns, values, strict=True):
        _field(stream, column.name.encode())
        _field(stream, column.data_type.encode())
        if value is None:
            _field(stream, None)
        elif column.encoding in {"text", "blob"}:
            assert isinstance(value, tuple) and len(value) == 2
            length, inline = value
            if length is None:
                _field(stream, None)
                continue
            assert isinstance(length, int)
            if inline is not None:
                assert isinstance(inline, bytes) and len(inline) == length
                _field(stream, inline)
                continue
            # Oversized cells stay in DuckDB and cross in bounded byte chunks.
            stream.write(b"\x01" + FRAME.pack(length))
            expression = _quote(column.name)
            if column.encoding == "text":
                expression = f"encode(CAST({expression} AS VARCHAR))"
            written = 0
            while written < length:
                part = conn.execute(
                    f"SELECT list_slice({expression}, ?, ?) FROM {qualified} WHERE rowid = ?",
                    [written + 1, min(written + CHUNK_BYTES, length), rowid],
                ).fetchone()
                if part is None or not isinstance(part[0], bytes) or not part[0]:
                    raise OracleUnavailable("incomplete sequential value capture")
                stream.write(part[0])
                written += len(part[0])
        elif column.encoding in {"float", "double"}:
            assert isinstance(value, float)
            _field(stream, struct.pack(">f" if column.encoding == "float" else ">d", value))
        elif column.encoding == "array":
            assert isinstance(value, (list, tuple))
            _field(stream, _array(list(value), column.data_type))
        else:
            assert isinstance(value, str)
            _field(stream, value.encode())


@dataclass(frozen=True)
class _Record:
    stream: BinaryIO
    offset: int
    length: int

    def __lt__(self, other: _Record) -> bool:
        return _compare(self, other) < 0


def _compare(left: _Record, right: _Record) -> int:
    shared = min(left.length, right.length)
    for offset in range(0, shared, CHUNK_BYTES):
        size = min(CHUNK_BYTES, shared - offset)
        left.stream.seek(left.offset + offset)
        a = left.stream.read(size)
        right.stream.seek(right.offset + offset)
        b = right.stream.read(size)
        if len(a) != size or len(b) != size:
            raise OracleUnavailable("truncated content stream")
        if a != b:
            return -1 if a < b else 1
    return (left.length > right.length) - (left.length < right.length)


def _copy_record(record: _Record, output: BinaryIO) -> None:
    output.write(FRAME.pack(record.length))
    record.stream.seek(record.offset)
    remaining = record.length
    while remaining:
        block = record.stream.read(min(CHUNK_BYTES, remaining))
        if not block:
            raise OracleUnavailable("truncated content record")
        output.write(block)
        remaining -= len(block)


def _next_record(stream: BinaryIO, offset: int) -> _Record | None:
    stream.seek(offset)
    header = stream.read(FRAME.size)
    if not header:
        return None
    if len(header) != FRAME.size:
        raise OracleUnavailable("truncated content frame")
    return _Record(stream, offset + FRAME.size, FRAME.unpack(header)[0])


def _merge(paths: list[Path], destination: Path) -> None:
    assert 1 <= len(paths) <= MERGE_WAYS_MAX
    streams: list[BinaryIO] = []
    try:
        heap: list[tuple[_Record, int]] = []
        for ordinal, path in enumerate(paths):
            stream = path.open("rb")
            streams.append(stream)
            record = _next_record(stream, 0)
            if record is not None:
                heapq.heappush(heap, (record, ordinal))
        with destination.open("wb") as output:
            while heap:
                record, ordinal = heapq.heappop(heap)
                _copy_record(record, output)
                following = _next_record(streams[ordinal], record.offset + record.length)
                if following is not None:
                    heapq.heappush(heap, (following, ordinal))
    finally:
        for stream in streams:
            stream.close()
    for path in paths:
        path.unlink()


def _sorted_rows(
    write_next: Callable[[BinaryIO], bool],
    directory: Path,
    filename: str,
    sort_bytes: int,
) -> int:
    paths: list[Path] = []
    rows = 0
    finished = False
    while not finished:
        source = directory / "unsorted.rows"
        records: list[_Record] = []
        with source.open("w+b") as stream:
            while len(records) < RUN_ROWS_MAX and stream.tell() < sort_bytes:
                start = stream.tell()
                stream.write(FRAME.pack(0))
                if not write_next(stream):
                    stream.seek(start)
                    stream.truncate()
                    finished = True
                    break
                end = stream.tell()
                stream.seek(start)
                stream.write(FRAME.pack(end - start - FRAME.size))
                stream.seek(end)
                records.append(_Record(stream, start + FRAME.size, end - start - FRAME.size))
                rows += 1
            records.sort(key=cmp_to_key(_compare))
            path = directory / f"run-{len(paths)}.rows"
            with path.open("wb") as output:
                for record in records:
                    _copy_record(record, output)
            paths.append(path)
        source.unlink()
        if len(paths) == MERGE_WAYS_MAX:
            combined = directory / "combined.rows"
            _merge(paths, combined)
            target = directory / "run-0.rows"
            combined.rename(target)
            paths = [target]
    _merge(paths, directory / filename)
    return rows


def _capture_table(
    conn: duckdb.DuckDBPyConnection,
    table: OracleTable,
    directory: Path,
    sort_bytes: int,
    *,
    observer: FailureObserver,
) -> OracleTable:
    # A separate cursor keeps the sequential scan alive while oversized cells
    # are fetched in bounded chunks on the parent connection.
    cursor = conn.cursor()
    with observe_before_cleanup(observer, "oracle:table_scan", cursor.close):
        expressions = ["rowid"]
        for column in table.columns:
            quoted = _quote(column.name)
            if column.encoding in {"text", "blob"}:
                raw = f"encode(CAST({quoted} AS VARCHAR))" if column.encoding == "text" else quoted
                expressions.append(
                    f"row(octet_length({raw}), CASE WHEN octet_length({raw}) <= 4096 "
                    f"THEN {raw} END)"
                )
            elif column.encoding in {"float", "double", "array"}:
                expressions.append(quoted)
            else:
                expressions.append(f"CAST({quoted} AS VARCHAR)")
        cursor.execute(
            "SELECT "
            + ", ".join(expressions)
            + f" FROM {_quote(table.schema_name)}.{_quote(table.name)}"
        )

        def write_next(stream: BinaryIO) -> bool:
            row = cursor.fetchone()
            if row is None:
                return False
            _write_row(conn, stream, table, int(row[0]), row[1:])
            return True

        rows = _sorted_rows(write_next, directory, table.filename, sort_bytes)
        destination = directory / table.filename
    digest = _digest(destination)
    return table.model_copy(
        update={"rows": rows, "byte_count": destination.stat().st_size, "sha256": digest}
    )


def _write_sqlite_row(
    conn: sqlite3.Connection,
    stream: BinaryIO,
    table: OracleTable,
    location: tuple[tuple[str, object], ...],
    values: tuple[object, ...],
) -> None:
    for ordinal, column in enumerate(table.columns):
        kind, length, value = values[ordinal * 3 : ordinal * 3 + 3]
        assert isinstance(kind, str)
        _field(stream, column.name.encode())
        _field(stream, kind.encode())
        if kind == "null":
            _field(stream, None)
        elif kind == "integer":
            assert isinstance(value, int)
            _field(stream, str(value).encode())
        elif kind == "real":
            assert isinstance(value, float)
            _field(stream, struct.pack(">d", value))
        else:
            assert kind in {"text", "blob"} and isinstance(length, int)
            if value is not None:
                assert isinstance(value, (str, bytes))
                _field(stream, value.encode() if isinstance(value, str) else value)
                continue
            stream.write(b"\x01" + FRAME.pack(length))
            offset = 0
            while offset < length:
                part = conn.execute(
                    f"SELECT substr(CAST({_quote(column.name)} AS BLOB), ?, ?) "
                    f"FROM {_quote(table.name)} WHERE "
                    + " AND ".join(f"{_quote(key)} IS ?" for key, _ in location),
                    [
                        offset + 1,
                        min(CHUNK_BYTES, length - offset),
                        *(value for _, value in location),
                    ],
                ).fetchone()
                if part is None or not isinstance(part[0], bytes) or not part[0]:
                    raise OracleUnavailable("incomplete keyword content capture")
                stream.write(part[0])
                offset += len(part[0])


def _capture_sidecar(
    path: Path,
    directory: Path,
    sort_bytes: int,
    first_file: int,
    consume: Callable[[OracleTable], None] | None = None,
    *,
    observer: FailureObserver,
) -> SidecarOracle:
    try:
        path.stat()
    except FileNotFoundError:
        return SidecarOracle(present=False, tables=())
    conn = _native.sqlite_connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    tables: list[OracleTable] = []
    with observe_before_cleanup(observer, "oracle:keyword_scan", conn.close):
        conn.execute("BEGIN")
        catalog = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchmany(CATALOG_ITEMS_MAX + 1)
        if len(catalog) > CATALOG_ITEMS_MAX:
            raise OracleUnavailable("keyword catalog exceeds oracle bound")
        for name, ddl in catalog:
            if ddl is None:
                raise OracleUnavailable("keyword table lacks reconstructible DDL")
            info = conn.execute(f"PRAGMA table_info({_quote(name)})").fetchmany(
                CATALOG_ITEMS_MAX + 1
            )
            if len(info) > CATALOG_ITEMS_MAX:
                raise OracleUnavailable("keyword column catalog exceeds oracle bound")
            columns = tuple(
                OracleColumn(name=row[1], data_type=row[2], encoding="scalar") for row in info
            )
            if ddl.upper().startswith("CREATE VIRTUAL TABLE"):
                # Contentless FTS values are NULL; its actual membership is
                # the virtual table's rowids plus the separately captured maps.
                columns = (
                    OracleColumn(name="rowid", data_type="INTEGER", encoding="scalar"),
                    *columns,
                )
            table = OracleTable(
                schema_name="main",
                name=name,
                ddl=ddl,
                columns=columns,
                rows=0,
                byte_count=0,
                sha256="0" * 64,
                filename=f"table-{first_file + len(tables)}.rows",
            )
            keys = (
                tuple(row[1] for row in info if row[5])
                if "WITHOUT ROWID" in ddl.upper()
                else ("rowid",)
            )
            for key in keys:
                maximum = conn.execute(
                    f"SELECT max(length(CAST({_quote(key)} AS BLOB))) FROM {_quote(name)}"
                ).fetchone()
                if maximum is not None and maximum[0] is not None and maximum[0] > CHUNK_BYTES:
                    raise OracleUnavailable("keyword scan locator exceeds 64 KiB")
            expressions = [_quote(key) for key in keys]
            for column in columns:
                quoted = _quote(column.name)
                kind = f"typeof({quoted})"
                length = f"length(CAST({quoted} AS BLOB))"
                expressions.extend(
                    (
                        kind,
                        length,
                        f"CASE WHEN {kind} NOT IN ('text','blob') "
                        f"OR {length} <= 4096 THEN {quoted} END",
                    )
                )
            cursor = conn.execute("SELECT " + ", ".join(expressions) + f" FROM {_quote(name)}")

            def write_next(stream: BinaryIO, cursor=cursor, table=table, keys=keys) -> bool:
                row = cursor.fetchone()
                if row is None:
                    return False
                location = tuple(zip(keys, row[: len(keys)], strict=True))
                _write_sqlite_row(conn, stream, table, location, row[len(keys) :])
                return True

            with observe_before_cleanup(observer, "oracle:keyword_cursor", cursor.close):
                rows = _sorted_rows(write_next, directory, table.filename, sort_bytes)
            output = directory / table.filename
            tables.append(
                table.model_copy(
                    update={
                        "rows": rows,
                        "byte_count": output.stat().st_size,
                        "sha256": _digest(output),
                    }
                )
            )
            if consume is not None:
                consume(tables[-1])
    return SidecarOracle(present=True, tables=tuple(tables))


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture_streams(
    conn: duckdb.DuckDBPyConnection,
    directory: Path,
    sidecar: Path | None,
    observer: FailureObserver,
) -> dict[str, tuple[int, int, str]]:
    """Capture every table the way the pre-change `capture_oracle` did; map file to (rows, bytes, sha256)."""
    tables = tuple(
        _capture_table(conn, table, directory, SORT_BYTES_MAX, observer=observer)
        for table in _catalog(conn)
    )
    keywords = (
        None
        if sidecar is None
        else _capture_sidecar(sidecar, directory, SORT_BYTES_MAX, len(tables), observer=observer)
    )
    everything = tables + (() if keywords is None else keywords.tables)
    return {table.filename: (table.rows, table.byte_count, table.sha256) for table in everything}
