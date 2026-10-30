"""`compare`: compare DuckDB table schemas and row multisets on verified copies.

Inputs are prepared templates. DuckDB receives only output-owned database and
WAL clones, attached READ_ONLY; its memory and spill are bounded. Equality is
logical table equality, not byte equality or index/sidecar equivalence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

import duckdb

from .evidence import JobFailed, Result, tool_identity
from .fs import Cancellation, NewTree, Refused
from .isolation import FileRecord, Policy, Template, open_entry, verify_tree

Progress = Callable[[dict[str, object]], None]
DATABASE = "recall.duckdb"
WAL = DATABASE + ".wal"
MEMORY_BYTES_DEFAULT = 1024**3
MEMORY_BYTES_MAX = 4 * 1024**3
SPILL_BYTES_MAX = 64 * 1024**3
DEADLINE_SECONDS = 4 * 3600
TABLES_MAX = 10_000
COLUMNS_MAX = 4096


@dataclass(frozen=True)
class Table:
    schema: str
    name: str
    columns: tuple[tuple[str, str], ...]

    def sql(self, catalog: str) -> str:
        return ".".join(_identifier(part) for part in (catalog, self.schema, self.name))


@dataclass(frozen=True)
class Difference:
    schema: str
    table: str
    equal: bool
    before_columns: tuple[tuple[str, str], ...] | None
    after_columns: tuple[tuple[str, str], ...] | None
    before_only: int | None = None
    after_only: int | None = None


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _memory(text: str) -> int:
    value = int(text)
    if not 1024**2 <= value <= MEMORY_BYTES_MAX:
        raise argparse.ArgumentTypeError(f"bytes must be within {1024**2}..{MEMORY_BYTES_MAX}")
    return value


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--before", required=True, help="verified template made by prepare")
    parser.add_argument("--after", required=True, help="verified template made by prepare")
    parser.add_argument(
        "--output", required=True, help="new directory for results, copies and spill"
    )
    parser.add_argument("--memory-limit", type=_memory, default=MEMORY_BYTES_DEFAULT, metavar="B")


class Queries:
    """One DuckDB worker; the caller remains able to cancel an executing query."""

    def __init__(self, connection: duckdb.DuckDBPyConnection, cancellation: Cancellation) -> None:
        self.connection = connection
        self.cancellation = cancellation
        self.worker = ThreadPoolExecutor(max_workers=1)

    def execute(self, sql: str) -> list[tuple[Any, ...]]:
        self.cancellation.check()
        future = self.worker.submit(lambda: self.connection.execute(sql).fetchall())
        try:
            while True:
                try:
                    rows = future.result(timeout=0.1)
                    self.cancellation.check()
                    return rows
                except TimeoutError:
                    self.cancellation.check()
        except BaseException:
            self.connection.interrupt()
            # The worker must finish before either its connection or files close.
            self.worker.shutdown(wait=True, cancel_futures=True)
            raise

    def close(self) -> None:
        self.worker.shutdown(wait=True)


def _clone(template: Template, label: str, tree: NewTree, cancellation: Cancellation) -> str:
    records = {
        PurePosixPath(record.path).name: record
        for record in template.manifest.files
        if record.path in {f"payload/{DATABASE}", f"payload/{WAL}"}
    }
    if DATABASE not in records:
        raise Refused(label, template.path, f"manifest has no payload/{DATABASE}")
    with tree.make_directory(tree.root, PurePosixPath(label)) as directory:
        for name, record in sorted(records.items()):
            with open_entry(template.root, PurePosixPath(record.path), label) as source:
                tree.clone(directory, PurePosixPath(label, name), source, cancellation).close()
        verify_tree(
            label,
            directory,
            set(),
            {
                name: FileRecord(name, record.bytes, record.sha256)
                for name, record in records.items()
            },
            cancellation,
        )
    return str(tree.path / label / DATABASE)


def _tables(queries: Queries, catalog: str) -> dict[tuple[str, str], Table]:
    rows = queries.execute(
        "SELECT table_schema, table_name FROM information_schema.tables "
        f"WHERE table_catalog = {_literal(catalog)} AND table_type = 'BASE TABLE' "
        f"ORDER BY table_schema, table_name LIMIT {TABLES_MAX + 1}"
    )
    if len(rows) > TABLES_MAX:
        raise JobFailed("limit", f"at most {TABLES_MAX} tables per database")
    tables = {}
    for schema, name in rows:
        columns = queries.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_catalog = {_literal(catalog)} AND table_schema = {_literal(schema)} "
            f"AND table_name = {_literal(name)} ORDER BY ordinal_position LIMIT {COLUMNS_MAX + 1}"
        )
        if len(columns) > COLUMNS_MAX:
            raise JobFailed("limit", f"at most {COLUMNS_MAX} columns per table")
        tables[schema, name] = Table(schema, name, tuple(columns))
    return tables


def _compare(queries: Queries, progress: Progress) -> list[Difference]:
    before, after = _tables(queries, "before"), _tables(queries, "after")
    differences = []
    for schema, name in sorted(before.keys() | after.keys()):
        left, right = before.get((schema, name)), after.get((schema, name))
        if left is None or right is None or left.columns != right.columns:
            item = Difference(
                schema,
                name,
                False,
                left.columns if left else None,
                right.columns if right else None,
            )
        else:
            a, b = left.sql("before"), right.sql("after")
            counts = tuple(
                queries.execute(
                    f"SELECT count(*) FROM (SELECT * FROM {first} "
                    f"EXCEPT ALL SELECT * FROM {second})"
                )[0][0]
                for first, second in ((a, b), (b, a))
            )
            item = Difference(schema, name, counts == (0, 0), left.columns, right.columns, *counts)
        differences.append(item)
        progress({"event": "table", **asdict(item)})
    return differences


def run(
    args: argparse.Namespace, result: Result, cancellation: Cancellation, progress: Progress
) -> None:
    cancellation.expire_at(time.monotonic() + DEADLINE_SECONDS)
    policy = Policy.for_host()
    with ExitStack() as held:
        roots = held.enter_context(policy.open_write_roots())
        result.leftovers = roots.leftovers
        progress({"event": "verify", "argument": "--before"})
        before = held.enter_context(Template.open(policy, "--before", args.before, cancellation))
        progress({"event": "verify", "argument": "--after"})
        after = held.enter_context(Template.open(policy, "--after", args.after, cancellation))
        roots.policy = policy.with_checkout(before.path).with_checkout(after.path)
        target = held.enter_context(roots.new_directory("--output", args.output))
        result.scope = {
            "measurements": ["DuckDB base-table schemas and row multisets"],
            "missing": ["views, indexes and FTS sidecar equivalence", "runtime qualification"],
            "before": {"path": str(before.path), "manifest": json.loads(before.manifest.to_json())},
            "after": {"path": str(after.path), "manifest": json.loads(after.manifest.to_json())},
            "tool": tool_identity(),
            "settings": {
                "memory_limit_bytes": args.memory_limit,
                "spill_limit_bytes": SPILL_BYTES_MAX,
                "deadline_seconds": DEADLINE_SECONDS,
                "threads": 2,
                "read_only": True,
            },
        }
        with target.create() as tree:
            tree.commit(cancellation)  # Keep partial evidence, including failed database opens.
            try:
                paths = [
                    _clone(template, name, tree, cancellation)
                    for template, name in ((before, "before"), (after, "after"))
                ]
                with tree.make_directory(tree.root, PurePosixPath("spill")):
                    pass
                with duckdb.connect(
                    ":memory:",
                    config={
                        "threads": 2,
                        "memory_limit": f"{args.memory_limit}B",
                        "temp_directory": str(tree.path / "spill"),
                        "max_temp_directory_size": f"{SPILL_BYTES_MAX}B",
                        "autoload_known_extensions": False,
                        "autoinstall_known_extensions": False,
                    },
                ) as connection:
                    queries = Queries(connection, cancellation)
                    try:
                        for catalog, path in zip(("before", "after"), paths, strict=True):
                            queries.execute(f"ATTACH {_literal(path)} AS {catalog} (READ_ONLY)")
                        queries.execute("SET enable_external_access = false")
                        differences = _compare(queries, progress)
                    finally:
                        queries.close()
                # A changing input cannot be reported as an equality observation.
                for label, template in (("--before", before), ("--after", after)):
                    current = Template.from_root(label, template.root, cancellation)
                    if current.manifest != template.manifest:
                        raise Refused(label, template.path, "manifest changed during comparison")
                equal = all(item.equal for item in differences)
                result.step("compare", equal=equal, tables=[asdict(item) for item in differences])
                tree.publish(
                    PurePosixPath("comparison.json"),
                    json.dumps(
                        {"equal": equal, "tables": [asdict(item) for item in differences]}, indent=2
                    ).encode()
                    + b"\n",
                )
                if not equal:
                    raise JobFailed("different", "DuckDB table schemas or row multisets differ")
            except duckdb.Error as error:
                raise JobFailed("duckdb", str(error)) from error
            finally:
                result.end(sys.exc_info()[1])
                tree.publish(PurePosixPath("result.json"), result.to_json())
