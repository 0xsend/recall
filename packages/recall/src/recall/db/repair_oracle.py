"""Durable exact logical multisets for an exclusively owned repair attempt.

The owner excludes writers and sidecar publications throughout capture/compare.
Rows are typed length-prefixed streams; the on-disk merge compares bytes, keeps
multiplicity, and never uses physical rowids as a cross-database identity.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import shutil
import sqlite3
import struct
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cmp_to_key
from pathlib import Path
from typing import Annotated, BinaryIO, Literal, Self
from uuid import UUID

import duckdb
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from recall.core.durability import sync_directory
from recall.core.failure_observation import FailureObserver, observe_before_cleanup
from recall.db.repair import TargetManifest, capture_target_manifest

SORT_BYTES_MAX = 64 * 1024 * 1024
MERGE_WAYS_MAX = 32
RUN_ROWS_MAX = 8192
CATALOG_ITEMS_MAX = 4096
MANIFEST_BYTES_MAX = 4 * 1024 * 1024
CHUNK_BYTES = 64 * 1024
FRAME = struct.Struct(">Q")

# Only these diagnostics can legitimately change between capture and postcheck.
# All other runtime fields, every other table, and non-main schemas are exact.
RUNTIME_REPAIR_COLUMNS = frozenset(
    {
        "last_failure_message",
        "last_failure_at",
        "last_fatal_signature",
        "fatal_repeat_count",
        "last_fatal_at",
        "last_index_repair_at",
        "last_index_repair_signature",
        "needs_index_verification",
    }
)


class OracleUnavailable(RuntimeError):
    """No complete, compatible reference can authorize repair."""


def _version_one(value: object) -> Literal[1]:
    if type(value) is not int or value != 1:
        raise ValueError("unsupported oracle version")
    return 1


OracleVersion = Annotated[Literal[1], BeforeValidator(_version_one)]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class OracleReference(_Model):
    trigger_id: UUID
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class OracleColumn(_Model):
    name: str
    data_type: str
    encoding: Literal["text", "blob", "float", "double", "array", "scalar"]


class OracleTable(_Model):
    schema_name: str
    name: str
    ddl: str
    columns: tuple[OracleColumn, ...] = Field(max_length=CATALOG_ITEMS_MAX)
    excluded_columns: tuple[str, ...] = ()
    baseline_diagnostics: tuple[tuple[str, str | None], ...] = ()
    rows: int = Field(ge=0, strict=True)
    byte_count: int = Field(ge=0, strict=True)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    filename: str = Field(pattern=r"^table-[0-9]+\.rows$")


class SidecarOracle(_Model):
    present: bool
    tables: tuple[OracleTable, ...] = Field(max_length=CATALOG_ITEMS_MAX)

    @property
    def generation(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class OracleManifest(_Model):
    version: OracleVersion = 1
    sidecar: SidecarOracle | None = None
    trigger_id: UUID
    targets: TargetManifest
    tables: tuple[OracleTable, ...] = Field(max_length=CATALOG_ITEMS_MAX)
    # The repair singleton is independently parsed and preserved by its owner.
    excluded_tables: tuple[Literal["main.daemon_index_repair"]] = ("main.daemon_index_repair",)

    @model_validator(mode="after")
    def validate_stream_identity(self) -> Self:
        tables = self.tables + (() if self.sidecar is None else self.sidecar.tables)
        if tuple(table.filename for table in tables) != tuple(
            f"table-{ordinal}.rows" for ordinal in range(len(tables))
        ):
            raise ValueError("oracle stream identities must be complete and unique")
        for group in (self.tables, () if self.sidecar is None else self.sidecar.tables):
            if len({(table.schema_name, table.name) for table in group}) != len(group):
                raise ValueError("duplicate table identity in oracle")
        return self


NATIVE_KEYWORD_SCHEMAS = frozenset({"fts_main_message_state", "fts_main_tool_calls"})
NATIVE_KEYWORD_COLUMNS = {
    "dict": (("termid", "BIGINT"), ("term", "VARCHAR"), ("df", "BIGINT")),
    "docs": (("docid", "BIGINT"), ("name", "VARCHAR"), ("len", "BIGINT")),
    "fields": (("fieldid", "BIGINT"), ("field", "VARCHAR")),
    "stats": (("num_docs", "BIGINT"), ("avgdl", "DOUBLE")),
    "stopwords": (("sw", "VARCHAR"),),
    "terms": (("docid", "BIGINT"), ("fieldid", "BIGINT"), ("termid", "BIGINT")),
}
NATIVE_KEYWORD_TABLES_MAX = 2 * len(NATIVE_KEYWORD_COLUMNS)


def managed_native_keyword_table(table: OracleTable) -> bool:
    columns = NATIVE_KEYWORD_COLUMNS.get(table.name)
    if table.schema_name not in NATIVE_KEYWORD_SCHEMAS or columns is None:
        return False
    definitions = ", ".join(
        ('"name"' if name == "name" else name) + " " + data_type for name, data_type in columns
    )
    expected_ddl = f"CREATE TABLE {table.schema_name}.{table.name}({definitions});"
    return (
        table.ddl == expected_ddl
        and tuple((column.name, column.data_type) for column in table.columns) == columns
        and not table.excluded_columns
    )


class KeywordScope(_Model):
    backend: Literal["sqlite_sidecar", "duckdb"]
    fields: tuple[Literal["bash", "content", "thinking"], ...]

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if tuple(sorted(set(self.fields))) != self.fields:
            raise ValueError("keyword publication field scope must be canonical and unique")
        return self


class NativeKeywordMacro(_Model):
    schema_name: Literal["fts_main_message_state", "fts_main_tool_calls"]
    name: Literal["match_bm25", "tokenize"]
    parameters: tuple[str, ...] = Field(max_length=16)
    definition: str = Field(max_length=64 * 1024)


def native_keyword_macros(conn: duckdb.DuckDBPyConnection) -> tuple[NativeKeywordMacro, ...]:
    rows = conn.execute(
        "SELECT schema_name,function_name,parameters,macro_definition FROM duckdb_functions() "
        "WHERE schema_name IN ('fts_main_message_state','fts_main_tool_calls') "
        "ORDER BY schema_name,function_name"
    ).fetchmany(5)
    if len(rows) > 4:
        raise OracleUnavailable("native keyword macro catalog exceeds its bound")
    return tuple(
        NativeKeywordMacro.model_validate(
            {
                "schema_name": schema,
                "name": name,
                "parameters": tuple(parameters),
                "definition": ddl,
            }
        )
        for schema, name, parameters, ddl in rows
    )


class PublicationOracle(KeywordScope):
    version: OracleVersion = 1
    trigger_id: UUID
    attempt_no: int = Field(gt=0, lt=2**64 - 1, strict=True)
    oracle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    pending: OracleTable
    sidecar: SidecarOracle
    native_keywords: tuple[OracleTable, ...] = Field(
        default=(), max_length=NATIVE_KEYWORD_TABLES_MAX
    )
    native_macros: tuple[NativeKeywordMacro, ...] = Field(default=(), max_length=4)

    @property
    def keyword_generation(self) -> str:
        value = {
            "backend": self.backend,
            "fields": self.fields,
            "sidecar": self.sidecar.model_dump(mode="json"),
            "native_keywords": tuple(
                table.model_dump(mode="json") for table in self.native_keywords
            ),
            "native_macros": tuple(macro.model_dump(mode="json") for macro in self.native_macros),
        }
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @model_validator(mode="after")
    def validate_streams(self) -> Self:
        if (self.pending.schema_name, self.pending.name) != ("main", "fts_sidecar_pending"):
            raise ValueError("publication reference must describe the pending intent table")
        if self.pending.rows != 0 or self.pending.byte_count != 0:
            raise ValueError("full manual publication retires the captured pending membership")
        if self.backend == "sqlite_sidecar" and (self.native_keywords or self.native_macros):
            raise ValueError("a SQLite publication cannot change canonical native tables")
        if any(not managed_native_keyword_table(table) for table in self.native_keywords):
            raise ValueError("unsupported native keyword publication table")
        tables = (self.pending, *self.sidecar.tables, *self.native_keywords)
        if len({table.filename for table in tables}) != len(tables):
            raise ValueError("publication stream identities overlap")
        return self


Barrier = Callable[[str], None]


def _no_barrier(stage: str) -> None:
    pass


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _encoding(data_type: str) -> Literal["text", "blob", "float", "double", "array", "scalar"]:
    if data_type in {"VARCHAR", "JSON"}:
        return "text"
    if data_type == "BLOB":
        return "blob"
    if data_type in {"FLOAT", "DOUBLE"}:
        return "float" if data_type == "FLOAT" else "double"
    if (
        re.fullmatch(r"(FLOAT|DOUBLE)\[[0-9]+\]", data_type)
        and int(data_type.split("[")[1][:-1]) <= 4096
    ):
        return "array"
    if data_type in {
        "BOOLEAN",
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "UHUGEINT",
        "DATE",
        "TIME",
        "TIME WITH TIME ZONE",
        "TIMESTAMP",
        "TIMESTAMP_S",
        "TIMESTAMP_MS",
        "TIMESTAMP_NS",
        "TIMESTAMP WITH TIME ZONE",
        "INTERVAL",
        "UUID",
    } or re.fullmatch(r"DECIMAL\([0-9]+,[0-9]+\)", data_type):
        return "scalar"
    raise OracleUnavailable(f"unsupported exact-content type: {data_type}")


def _catalog(conn: duckdb.DuckDBPyConnection) -> tuple[OracleTable, ...]:
    rows = conn.execute(
        "SELECT schema_name, table_name, sql FROM duckdb_tables() "
        "WHERE NOT internal AND NOT temporary ORDER BY schema_name, table_name"
    ).fetchmany(CATALOG_ITEMS_MAX + 1)
    if len(rows) > CATALOG_ITEMS_MAX:
        raise OracleUnavailable("table catalog exceeds oracle bound")
    tables: list[OracleTable] = []
    for schema, name, ddl in rows:
        if (schema, name) == ("main", "daemon_index_repair"):
            continue
        columns = conn.execute(
            "SELECT column_name, data_type FROM duckdb_columns() "
            "WHERE schema_name = ? AND table_name = ? AND NOT internal ORDER BY column_index",
            [schema, name],
        ).fetchmany(CATALOG_ITEMS_MAX + 1)
        if len(columns) > CATALOG_ITEMS_MAX or any(n.lower() == "rowid" for n, _ in columns):
            raise OracleUnavailable("column catalog unsupported for bounded sequential capture")
        excluded = tuple(
            n
            for n, _ in columns
            if (schema, name) == ("main", "runtime_state") and n in RUNTIME_REPAIR_COLUMNS
        )
        diagnostics: tuple[tuple[str, str | None], ...] = ()
        if excluded:
            values = conn.execute(
                "SELECT "
                + ", ".join(f"CAST({_quote(n)} AS VARCHAR)" for n in excluded)
                + f" FROM {_quote(schema)}.{_quote(name)}"
            ).fetchmany(2)
            if len(values) != 1:
                raise OracleUnavailable("runtime diagnostic projection requires one singleton")
            diagnostics = tuple(zip(excluded, values[0], strict=True))
        tables.append(
            OracleTable(
                schema_name=schema,
                name=name,
                ddl=ddl,
                columns=tuple(
                    OracleColumn(name=n, data_type=t, encoding=_encoding(t))
                    for n, t in columns
                    if n not in excluded
                ),
                excluded_columns=excluded,
                baseline_diagnostics=diagnostics,
                rows=0,
                byte_count=0,
                sha256="0" * 64,
                filename=f"table-{len(tables)}.rows",
            )
        )
    return tuple(tables)


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
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
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


def _all_tables(manifest: OracleManifest) -> tuple[OracleTable, ...]:
    return manifest.tables + (() if manifest.sidecar is None else manifest.sidecar.tables)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def _capture_directory(path: Path) -> Iterator[None]:
    path.mkdir()
    try:
        yield
    except BaseException as primary:
        try:
            if path.exists():
                shutil.rmtree(path)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "capture and workspace cleanup failed", [primary, cleanup]
            ) from None
        raise
    else:
        if path.exists():
            shutil.rmtree(path)


def capture_oracle(
    conn: duckdb.DuckDBPyConnection,
    directory: Path,
    trigger_id: UUID,
    *,
    sort_bytes: int = SORT_BYTES_MAX,
    barrier: Barrier = _no_barrier,
    sidecar: Path | None = None,
    observer: FailureObserver,
) -> OracleReference:
    """Publish one complete independent reference; never replace a saved oracle."""
    if not 1 <= sort_bytes <= SORT_BYTES_MAX:
        raise ValueError("sort run must fit within 64 MiB")
    parent_exists = directory.parent.exists()
    directory.parent.mkdir(parents=True, exist_ok=True)
    if not parent_exists:
        sync_directory(directory.parent.parent)
    if directory.exists():
        raise OracleUnavailable("a saved oracle cannot be overwritten")
    temporary = directory.parent / f".capture-{trigger_id}"
    # The exclusive owner reconciles an abandoned workspace before capture.
    # mkdir must succeed before this invocation owns its failure cleanup.
    try:
        with _capture_directory(temporary):
            # The scan has no predicate on an indexed key: no ART access can
            # substitute for its complete sequential table traversal.
            tables = tuple(
                _capture_table(conn, table, temporary, sort_bytes, observer=observer)
                for table in _catalog(conn)
            )
            keywords = (
                None
                if sidecar is None
                else _capture_sidecar(
                    sidecar, temporary, sort_bytes, len(tables), observer=observer
                )
            )
            manifest = OracleManifest(
                trigger_id=trigger_id,
                targets=capture_target_manifest(conn),
                tables=tables,
                sidecar=keywords,
            )
            payload = manifest.model_dump_json().encode()
            if len(payload) > MANIFEST_BYTES_MAX:
                raise OracleUnavailable("oracle manifest exceeds 4 MiB")
            required = sum(table.byte_count for table in _all_tables(manifest))
            if shutil.disk_usage(temporary).free < required:
                raise OracleUnavailable("insufficient free space for a complete repair working set")
            (temporary / "manifest.json").write_bytes(payload)
            for path in temporary.iterdir():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            sync_directory(temporary)
            barrier("capture_synced")
            temporary.rename(directory)
            barrier("capture_published")
            sync_directory(directory.parent)
            barrier("capture_directory_synced")
            return OracleReference(
                trigger_id=trigger_id, manifest_hash=hashlib.sha256(payload).hexdigest()
            )
    except (OSError, ValueError, duckdb.Error, sqlite3.Error) as err:
        raise OracleUnavailable(f"content capture failed: {err}") from err


def read_oracle(directory: Path, reference: OracleReference) -> OracleManifest:
    manifest = read_oracle_manifest(directory, reference)
    try:
        for table in _all_tables(manifest):
            path = directory / table.filename
            if path.stat().st_size != table.byte_count or _digest(path) != table.sha256:
                raise OracleUnavailable("saved content stream is incomplete or corrupt")
        return manifest
    except OSError as err:
        raise OracleUnavailable(f"saved content reference unavailable: {err}") from err


def read_oracle_manifest(directory: Path, reference: OracleReference) -> OracleManifest:
    """Read bounded identity metadata without scanning the saved row streams."""
    try:
        with (directory / "manifest.json").open("rb") as stream:
            payload = stream.read(MANIFEST_BYTES_MAX + 1)
        if (
            len(payload) > MANIFEST_BYTES_MAX
            or hashlib.sha256(payload).hexdigest() != reference.manifest_hash
        ):
            raise OracleUnavailable("saved content manifest hash differs")
        manifest = OracleManifest.model_validate_json(payload)
        if manifest.trigger_id != reference.trigger_id:
            raise OracleUnavailable("saved content manifest belongs to a different trigger")
        return manifest
    except (OSError, ValidationError) as err:
        raise OracleUnavailable(f"saved content reference unavailable: {err}") from err


def verify_oracle(
    conn: duckdb.DuckDBPyConnection,
    directory: Path,
    reference: OracleReference,
    *,
    sidecar: Path | None = None,
    observer: FailureObserver,
) -> None:
    """Compare independent fresh scans byte-for-byte, including duplicates."""
    manifest = read_oracle(directory, reference)
    if manifest.sidecar is not None and sidecar is None:
        raise OracleUnavailable("keyword reference requires its actual source namespace")
    with tempfile.TemporaryDirectory(prefix=".compare-", dir=directory.parent) as name:
        candidate = Path(name)
        if manifest.targets != capture_target_manifest(conn):
            raise OracleUnavailable("index catalog differs from saved target manifest")
        _verify_content(conn, manifest, directory, candidate, observer=observer)
        if sidecar is not None:
            expected = {
                table.filename: table
                for table in (() if manifest.sidecar is None else manifest.sidecar.tables)
            }

            def consume(table: OracleTable) -> None:
                before = expected.get(table.filename)
                if before is None:
                    raise OracleUnavailable("unexpected keyword table")
                _compare_table(before, table, directory, candidate)

            actual_sidecar = _capture_sidecar(
                sidecar, candidate, SORT_BYTES_MAX, len(manifest.tables), consume, observer=observer
            )
            if manifest.sidecar != actual_sidecar:
                raise OracleUnavailable(
                    "keyword generation or membership differs from saved reference"
                )


def _verify_content(
    conn: duckdb.DuckDBPyConnection,
    manifest: OracleManifest,
    directory: Path,
    working: Path,
    publication: PublicationOracle | None = None,
    *,
    retired: bool = True,
    publication_directory: Path | None = None,
    observer: FailureObserver,
) -> None:
    catalog = _catalog(conn)
    expected = {(table.schema_name, table.name): table for table in manifest.tables}
    actual = {(table.schema_name, table.name): table for table in catalog}
    if publication is not None and publication.backend == "duckdb":
        if native_keyword_macros(conn) != publication.native_macros:
            raise OracleUnavailable("native keyword macros differ from their publication reference")
        expected = {
            key: table for key, table in expected.items() if not managed_native_keyword_table(table)
        }
        actual = {
            key: table for key, table in actual.items() if not managed_native_keyword_table(table)
        }
    if expected.keys() != actual.keys():
        raise OracleUnavailable("table catalog differs from saved content reference")
    for key, before in expected.items():
        saved = directory
        if publication is not None and retired and key == ("main", "fts_sidecar_pending"):
            before = publication.pending
            assert publication_directory is not None
            saved = publication_directory
        table = actual[key].model_copy(update={"filename": before.filename})
        after = _capture_table(conn, table, working, SORT_BYTES_MAX, observer=observer)
        _compare_table(before, after, saved, working)
    if publication is not None and publication.backend == "duckdb":
        native = {
            (table.schema_name, table.name): table
            for table in catalog
            if managed_native_keyword_table(table)
        }
        expected_native = {
            (table.schema_name, table.name): table for table in publication.native_keywords
        }
        if native.keys() != expected_native.keys():
            raise OracleUnavailable("native keyword catalog differs from its publication reference")
        for key, before in expected_native.items():
            table = native[key].model_copy(update={"filename": before.filename})
            after = _capture_table(conn, table, working, SORT_BYTES_MAX, observer=observer)
            assert publication_directory is not None
            _compare_table(before, after, publication_directory, working)


def verify_manual_content(
    conn: duckdb.DuckDBPyConnection,
    directory: Path,
    reference: OracleReference,
    *,
    publication: PublicationOracle | None = None,
    observer: FailureObserver,
) -> None:
    """Compare canonical content independently of interrupted index/keyword work."""
    manifest = read_oracle(directory, reference)
    if publication is not None and (
        publication.trigger_id != reference.trigger_id
        or publication.oracle_hash != reference.manifest_hash
    ):
        raise OracleUnavailable("publication belongs to a different pre-attempt reference")
    with tempfile.TemporaryDirectory(prefix=".compare-", dir=directory.parent) as name:
        _verify_content(
            conn,
            manifest,
            directory,
            Path(name),
            publication,
            publication_directory=directory / "publication" if publication is not None else None,
            observer=observer,
        )


def verify_published_canonical(
    conn: duckdb.DuckDBPyConnection,
    directory: Path,
    reference: OracleReference,
    publication: PublicationOracle,
    *,
    observer: FailureObserver,
    publication_directory: Path,
) -> None:
    """Check an acknowledged keyword publication before intent retirement."""
    manifest = read_oracle(directory, reference)
    if (
        publication.trigger_id != reference.trigger_id
        or publication.oracle_hash != reference.manifest_hash
    ):
        raise OracleUnavailable("publication belongs to a different pre-attempt reference")
    with tempfile.TemporaryDirectory(prefix=".compare-", dir=directory.parent) as name:
        _verify_content(
            conn,
            manifest,
            directory,
            Path(name),
            publication,
            retired=False,
            publication_directory=publication_directory,
            observer=observer,
        )


def capture_publication(
    conn: duckdb.DuckDBPyConnection,
    directory: Path,
    reference: OracleReference,
    attempt_no: int,
    *,
    sidecar: Path,
    backend: str = "sqlite_sidecar",
    fields: tuple[str, ...] = ("bash", "content", "thinking"),
    barrier: Barrier = _no_barrier,
    observer: FailureObserver,
) -> str:
    """Publish permitted intent retirement and the actual verified keyword generation."""
    scope = KeywordScope.model_validate({"backend": backend, "fields": tuple(sorted(set(fields)))})
    before = read_oracle(directory, reference)
    pending = next(
        table
        for table in before.tables
        if (table.schema_name, table.name) == ("main", "fts_sidecar_pending")
    )
    destination = directory / "publication"
    working = directory / "publication-working"
    if destination.exists():
        raise OracleUnavailable("a saved publication reference cannot be overwritten")
    with _capture_directory(working):
        (working / pending.filename).write_bytes(b"")
        pending = pending.model_copy(
            update={"rows": 0, "byte_count": 0, "sha256": hashlib.sha256(b"").hexdigest()}
        )
        keywords = _capture_sidecar(
            sidecar, working, SORT_BYTES_MAX, len(before.tables), observer=observer
        )
        native: list[OracleTable] = []
        if scope.backend == "duckdb":
            for table in _catalog(conn):
                if managed_native_keyword_table(table):
                    ordinal = len(before.tables) + len(keywords.tables) + len(native)
                    table = table.model_copy(update={"filename": f"table-{ordinal}.rows"})
                    native.append(
                        _capture_table(conn, table, working, SORT_BYTES_MAX, observer=observer)
                    )
        publication = PublicationOracle(
            backend=scope.backend,
            fields=scope.fields,
            native_keywords=tuple(native),
            native_macros=native_keyword_macros(conn) if scope.backend == "duckdb" else (),
            trigger_id=reference.trigger_id,
            attempt_no=attempt_no,
            oracle_hash=reference.manifest_hash,
            pending=pending,
            sidecar=keywords,
        )
        payload = publication.model_dump_json().encode()
        if len(payload) > MANIFEST_BYTES_MAX:
            raise OracleUnavailable("publication manifest exceeds 4 MiB")
        (working / "manifest.json").write_bytes(payload)
        for path in working.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        sync_directory(working)
        barrier("publication_reference_synced")
        working.rename(destination)
        barrier("publication_reference_published")
        sync_directory(directory)
        barrier("publication_reference_directory_synced")
        return hashlib.sha256(payload).hexdigest()


def read_publication(directory: Path, digest: str) -> PublicationOracle:
    """Read one bounded, complete reference within its pinned pre-attempt oracle."""
    return read_publication_at(directory / "publication", digest)


def read_publication_at(publication: Path, digest: str) -> PublicationOracle:
    """Read a complete publication from its bound oracle or archived manual workspace."""
    with (publication / "manifest.json").open("rb") as stream:
        payload = stream.read(MANIFEST_BYTES_MAX + 1)
    if len(payload) > MANIFEST_BYTES_MAX or hashlib.sha256(payload).hexdigest() != digest:
        raise OracleUnavailable("publication manifest hash differs")
    manifest = PublicationOracle.model_validate_json(payload)
    for table in (manifest.pending, *manifest.sidecar.tables, *manifest.native_keywords):
        path = publication / table.filename
        if path.stat().st_size != table.byte_count or _digest(path) != table.sha256:
            raise OracleUnavailable("publication stream is incomplete or corrupt")
    return manifest


def verify_publication_sidecar(
    directory: Path, publication: PublicationOracle, sidecar: Path, *, observer: FailureObserver
) -> None:
    """A release retry must still name the exact acknowledged keyword generation."""
    expected = {table.filename: table for table in publication.sidecar.tables}
    with tempfile.TemporaryDirectory(prefix=".compare-", dir=directory.parent) as name:
        working = Path(name)

        def consume(table: OracleTable) -> None:
            before = expected.get(table.filename)
            if before is None:
                raise OracleUnavailable("unexpected post-publication keyword table")
            _compare_table(before, table, directory / "publication", working)

        manifest = read_oracle_manifest(
            directory,
            OracleReference(
                trigger_id=publication.trigger_id, manifest_hash=publication.oracle_hash
            ),
        )
        actual = _capture_sidecar(
            sidecar, working, SORT_BYTES_MAX, len(manifest.tables), consume, observer=observer
        )
        if actual != publication.sidecar:
            raise OracleUnavailable("post-publication keyword generation differs")


def _compare_table(before: OracleTable, after: OracleTable, saved: Path, working: Path) -> None:
    # Projected diagnostics are recorded but may change during repair.
    if before.model_dump(exclude={"baseline_diagnostics"}) != after.model_dump(
        exclude={"baseline_diagnostics"}
    ):
        raise OracleUnavailable(
            f"exact content/catalog differs: {before.schema_name}.{before.name}"
        )
    with (
        (saved / before.filename).open("rb") as left,
        (working / after.filename).open("rb") as right,
    ):
        while True:
            first, second = left.read(CHUNK_BYTES), right.read(CHUNK_BYTES)
            if first != second:
                raise OracleUnavailable("exact row stream differs from saved reference")
            if not first:
                break
    # No full candidate oracle accumulates, including SQLite shadow tables.
    (working / after.filename).unlink()
