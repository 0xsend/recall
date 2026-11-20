"""The sorter's streams equal the pre-change comparator's, byte for byte, at every boundary."""

import filecmp
import random
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import duckdb
import pytest
from recall.db import oracle_sort
from recall.db.lease import StandaloneLifetime
from recall.db.oracle_sort import SortShape
from recall.db.repair_oracle import capture_oracle, read_oracle_manifest

from tests.test_db import reference_oracle

TRIGGER = UUID("00000000-0000-4000-8000-000000000366")
MIB = 1024 * 1024

# Every length where an inline limit, a 64 KiB chunk, the 1 MiB inline limit or
# the 16 MiB file-backed limit changes how a cell reaches its record.
CELL_LENGTHS = (
    0,
    1,
    2,
    4095,
    4096,
    4097,
    65535,
    65536,
    65537,
    131071,
    131072,
    131073,
    MIB - 1,
    MIB,
    MIB + 1,
    16 * MIB - 1,
    16 * MIB,
    16 * MIB + 1,
)
SIDECAR_LENGTHS = (0, 1, 4095, 4096, 4097, 65535, 65536, 65537, MIB + 1)


@dataclass(frozen=True)
class Fixture:
    directory: Path
    sidecar: Path
    reference_dir: Path
    reference: dict[str, tuple[int, int, str]]


def _build_duckdb(conn: duckdb.DuckDBPyConnection) -> None:
    random_source = random.Random(366)
    conn.execute("CREATE TABLE cells (id INTEGER, t VARCHAR, b BLOB, n BIGINT)")
    rows: list[tuple[int, str | None, bytes | None, int | None]] = []
    for ordinal, length in enumerate(CELL_LENGTHS):
        # Multiplicity matters below 1 MiB; the large sizes stay single to bound the fixture.
        for _ in range(3 if length < MIB else 1):
            body = random_source.randbytes(min(length, 256))
            blob = (b"\x01" + body + b"a" * length)[:length] if length else b""
            rows.append((ordinal, "x" * length, blob, ordinal))
        rows.append((ordinal, None, None, None))
    conn.executemany("INSERT INTO cells VALUES (?, ?, ?, ?)", rows)
    # Equal for more than a chunk, then differing late; prefixes against longer values.
    conn.executemany(
        "INSERT INTO cells VALUES (?, ?, ?, ?)",
        [
            (100, "a" * 70000 + "b", b"p" * 70000 + b"a", 1),
            (100, "a" * 70000 + "c", b"p" * 70000 + b"b", 1),
            (100, "a" * 70000, b"p" * 70000, 1),
            (100, "a" * 65536 + "b", b"p" * 65536 + b"a", 1),
            (100, "a" * 65536, b"p" * 65536, 1),
            (100, "", b"", 1),
            (100, None, b"", 1),
            (100, "", None, 1),
        ],
    )
    conn.execute(
        "CREATE TABLE numerics (id INTEGER, f FLOAT, d DOUBLE, v FLOAT[4], w DOUBLE[3], "
        "h HUGEINT, ts TIMESTAMP, u UUID, dec DECIMAL(18,3), x JSON)"
    )
    conn.execute(
        """INSERT INTO numerics VALUES
        (1, 0.0, 0.0, [0.0,1,2,3], [0.0,0.0,0.0], 0, '2026-01-01 00:00:00', uuid(), 1.5, '{"a":1}'),
        (1, -0.0, -0.0, [-0.0,1,2,3], [-0.0,0.0,0.0], 0, '2026-01-01 00:00:00', uuid(), -1.5,
         '{"a":1}'),
        (1, -0.0, -0.0, [-0.0,1,2,3], [-0.0,0.0,0.0], 0, '2026-01-01 00:00:00', NULL, -1.5,
         '{"a":1}'),
        (2, 'NaN'::FLOAT, 'inf'::DOUBLE, [NULL,1,2,3], NULL, NULL, NULL, NULL, NULL, NULL),
        (2, 'NaN'::FLOAT, '-inf'::DOUBLE, [1.5,1,2,3], [1.0,2.0,3.0],
         170141183460469231731687303715884105727, '1970-01-01', NULL, 0, NULL)"""
    )
    conn.execute("CREATE TABLE duplicates (a INTEGER, b VARCHAR)")
    conn.executemany(
        "INSERT INTO duplicates VALUES (?, ?)", [(i % 7, f"k{i % 5}") for i in range(120)]
    )
    conn.execute("CREATE TABLE empty_table (a INTEGER, b VARCHAR)")


def _build_sidecar(path: Path) -> None:
    sidecar = sqlite3.connect(path)
    sidecar.execute("CREATE TABLE plain (k INTEGER PRIMARY KEY, a, b, c)")
    rows: list[tuple[int, object, object, object]] = []
    key = 1
    for length in SIDECAR_LENGTHS:
        for kind in ("text", "blob"):
            value = "x" * length if kind == "text" else bytes([7]) * length
            rows.append((key, value, None if length % 2 else 17, -0.0 if length % 3 else 0.0))
            key += 1
    rows += [
        (key, None, None, None),
        (key + 1, 5, 5.5, "t"),
        (key + 2, -0.0, 0.0, b"\x00"),
        (key + 3, 9223372036854775807, -9223372036854775808, 1e308),
        (key + 4, "", b"", ""),
        (key + 5, 5, 5.5, "t"),
    ]
    sidecar.executemany("INSERT INTO plain VALUES (?, ?, ?, ?)", rows)
    sidecar.execute("CREATE TABLE unkeyed (a, b)")
    sidecar.executemany(
        "INSERT INTO unkeyed VALUES (?, ?)", [(i % 4, f"d{i % 3}") for i in range(60)]
    )
    sidecar.execute(
        "CREATE TABLE keyed_without_rowid (a INTEGER, b TEXT, PRIMARY KEY (a, b)) WITHOUT ROWID"
    )
    sidecar.executemany(
        "INSERT INTO keyed_without_rowid VALUES (?, ?)", [(i, "w" * (i % 9)) for i in range(50)]
    )
    sidecar.execute("CREATE TABLE empty_sidecar (a, b)")
    sidecar.execute("CREATE VIRTUAL TABLE contentless USING fts5(body, content='')")
    sidecar.executemany(
        "INSERT INTO contentless(rowid, body) VALUES (?, ?)",
        [(i, f"alpha beta {i}") for i in range(1, 30)],
    )
    sidecar.commit()
    sidecar.close()


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Fixture]:
    directory = tmp_path_factory.mktemp("identity")
    sidecar = directory / "keywords.sqlite"
    _build_sidecar(sidecar)
    reference_dir = directory / "reference"
    reference_dir.mkdir()
    with duckdb.connect(str(directory / "database.duckdb")) as conn:
        _build_duckdb(conn)
        conn.execute("CHECKPOINT")
        reference = reference_oracle.capture_streams(
            conn, reference_dir, sidecar, StandaloneLifetime()
        )
    yield Fixture(directory, sidecar, reference_dir, reference)


def _capture(fixture: Fixture, name: str, shape: SortShape) -> Path:
    with duckdb.connect(str(fixture.directory / "database.duckdb"), read_only=True) as conn:
        capture_oracle(
            conn,
            fixture.directory / name,
            TRIGGER,
            shape=shape,
            sidecar=fixture.sidecar,
            observer=StandaloneLifetime(),
        )
    return fixture.directory / name


def _product_streams(directory: Path) -> dict[str, tuple[int, int, str]]:
    manifest = read_oracle_manifest(directory, _reference_of(directory))
    tables = manifest.tables + (() if manifest.sidecar is None else manifest.sidecar.tables)
    return {table.filename: (table.rows, table.byte_count, table.sha256) for table in tables}


def _reference_of(directory: Path):
    import hashlib

    from recall.db.repair_oracle import OracleReference

    digest = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    return OracleReference(trigger_id=TRIGGER, manifest_hash=digest)


def _differences(fixture: Fixture, product: Path) -> list[str]:
    """Names of the streams whose counts, hash or bytes differ from the comparator's."""
    streams = _product_streams(product)
    differing = [
        name
        for name in sorted(set(streams) | set(fixture.reference))
        if streams.get(name) != fixture.reference.get(name)
    ]
    differing += [
        name
        for name in sorted(streams.keys() & fixture.reference.keys())
        if not filecmp.cmp(product / name, fixture.reference_dir / name, shallow=False)
    ]
    return differing


SHAPES = {
    "defaults": SortShape(),
    "small runs, fan-in 3": SortShape(run_bytes=MIB, fan_in=3),
    "file-backed records": SortShape(run_bytes=MIB, fan_in=3, file_backed_bytes=2048),
}


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_capture_streams_equal_head_comparator(
    fixture: Fixture, shape: SortShape, tmp_path: Path
) -> None:
    product = _capture(fixture, f"product-{abs(hash(shape))}", shape)
    assert len(fixture.reference) >= 13
    assert _differences(fixture, product) == []


def test_the_equality_check_fails_on_a_merge_that_concatenates_runs(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def concatenate(runs, destination, file_backed_bytes):
        import hashlib

        digest = hashlib.sha256()
        size = 0
        with destination.open("wb") as output:
            for run in runs:
                data = run.path.read_bytes()
                output.write(data)
                digest.update(data)
                size += len(data)
                run.path.unlink()
        return oracle_sort._Run(destination, size, digest.hexdigest())

    monkeypatch.setattr(oracle_sort, "merge_runs", concatenate)
    product = _capture(fixture, "product-wrong-merge", SHAPES["small runs, fan-in 3"])
    assert _differences(fixture, product) != []
