"""Scratch-only storage prototypes; these are not product migrations.

The numeric prototype renders integer IDs as decimal strings inside Python and
SQLite to exercise the real daemon with compact ART keys. That representation
is intentionally not the proposed public identity contract. Shipping requires
the typed hexadecimal boundary described in the approval-gated plan.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import signal
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from itertools import batched
from pathlib import Path
from typing import Any
from unittest.mock import patch

import duckdb

REPO = Path(__file__).resolve().parents[2]
LANE = REPO.parent
HASH_COLUMNS = {
    "messages": ("id",),
    "message_state": ("message_id",),
    "message_embeddings": ("message_id",),
    "tool_calls": ("id", "message_id"),
    "tool_call_embeddings": ("tool_call_id",),
    "tool_use_ids": ("tool_call_id",),
    "tool_results": ("tool_call_id",),
}


class NumericFetches(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr in (
            "fetchall",
            "fetchmany",
            "fetchone",
        ):
            self.changed = True
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id=f"_checkpoint_{node.func.attr}", ctx=ast.Load()),
                    args=[node.func.value, *node.args],
                    keywords=node.keywords,
                ),
                node,
            )
        return node


def normalize_source_fetches(exported: Path) -> None:
    # This transforms the boundary by return type, rather than guessing which
    # readers remember to stringify IDs. SQL execution stays in its original
    # frame, retaining DuckDB's Arrow replacement-scan ownership.
    for source in (exported / "recall").rglob("*.py"):
        tree = ast.parse(source.read_text())
        transformer = NumericFetches()
        tree = transformer.visit(tree)
        if not transformer.changed:
            continue
        index = 1 if tree.body and isinstance(tree.body[0], ast.Expr) else 0
        while index < len(tree.body):
            statement = tree.body[index]
            if not isinstance(statement, ast.ImportFrom) or statement.module != "__future__":
                break
            index += 1
        tree.body.insert(
            index,
            ast.ImportFrom(
                module="numeric_reads",
                names=[
                    ast.alias(name=name, asname=f"_checkpoint_{name}")
                    for name in ("fetchall", "fetchmany", "fetchone")
                ],
                level=0,
            ),
        )
        source.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")


def install_numeric_ids() -> None:
    # A process-local source export gives numeric ID sets explicit SQL types;
    # native harness IDs and catalog/source IDs keep their VARCHAR bindings.
    # Product sources in the checkout are never edited by the prototype.
    exported = Path(os.environ["RECALL_DATA_DIR"]).parent / "numeric-source"
    source = REPO / "packages/recall/src/recall"
    shutil.copytree(source, exported / "recall", ignore=shutil.ignore_patterns("__pycache__"))
    queries_path = exported / "recall/db/queries.py"
    sql = queries_path.read_text()
    for column in ("message_id", "tool_call_id"):
        sql = sql.replace(
            f"{column} IN (SELECT id FROM _recall_bound_ids)",
            f"{column} IN (SELECT CAST(id AS UHUGEINT) FROM _recall_bound_ids)",
        )
    sql = sql.replace(
        "DELETE FROM messages WHERE id IN (SELECT id FROM _recall_bound_ids)",
        "DELETE FROM messages WHERE id IN (SELECT CAST(id AS UHUGEINT) FROM _recall_bound_ids)",
    )
    queries_path.write_text(sql)
    embed_path = exported / "recall/services/embed_phase.py"
    embed_path.write_text(embed_path.read_text().replace("?::TEXT[]", "?::UHUGEINT[]"))
    inputs_path = exported / "recall/services/enrichment_inputs.py"
    inputs_path.write_text(
        inputs_path.read_text().replace(
            "(kind, idx, id) > (?, ?, ?)", "(kind, idx, id) > (?, ?, ?::UHUGEINT)"
        )
    )
    normalize_source_fetches(exported)
    sys.path.insert(0, str(exported))
    from recall.core import ids

    original_message = ids.message_id
    original_tool = ids.tool_call_id

    def message_id(session_id_value: str, idx: int) -> str:
        return str(int(original_message(session_id_value, idx), 16))

    def tool_call_id(
        message_id_value: str | None, idx: int, session_id_value: str | None = None
    ) -> str:
        hexadecimal = f"{int(message_id_value):032x}" if message_id_value is not None else None
        return str(int(original_tool(hexadecimal, idx, session_id_value), 16))

    patch.object(ids, "message_id", message_id).start()
    patch.object(ids, "tool_call_id", tool_call_id).start()

    from recall.services import fts_sidecar_reconcile as reconcile

    original_match = reconcile._memberships_match

    def memberships_match(
        duckdb_conn: duckdb.DuckDBPyConnection,
        sidecar_conn: sqlite3.Connection,
        primary_sql: str,
        sidecar_sql: str,
        *,
        batch_size: int,
    ) -> bool:
        # Both streams must use the same lexical decimal-string ordering.
        for column in ("message_id", "id"):
            primary_sql = primary_sql.replace(
                f"SELECT {column} FROM",
                f"SELECT CAST({column} AS VARCHAR) AS {column} FROM",
            )
        return original_match(
            duckdb_conn, sidecar_conn, primary_sql, sidecar_sql, batch_size=batch_size
        )

    def message_ids_after(
        duckdb_conn: duckdb.DuckDBPyConnection, last_id: str, batch_size: int
    ) -> list[str]:
        return [
            row[0]
            for row in duckdb_conn.execute(
                "SELECT CAST(message_id AS VARCHAR) AS key FROM message_state "
                "WHERE CAST(message_id AS VARCHAR) > ? ORDER BY key LIMIT ?",
                [last_id, batch_size],
            ).fetchall()
        ]

    def tool_ids_after(
        duckdb_conn: duckdb.DuckDBPyConnection,
        last_id: str,
        batch_size: int,
        *,
        include_bash: bool,
    ) -> list[str]:
        if not include_bash:
            return []
        return [
            row[0]
            for row in duckdb_conn.execute(
                "SELECT CAST(id AS VARCHAR) AS key FROM tool_calls "
                "WHERE bash_command IS NOT NULL AND CAST(id AS VARCHAR) > ? "
                "ORDER BY key LIMIT ?",
                [last_id, batch_size],
            ).fetchall()
        ]

    # Keep the original call signatures while scoping the experiment to this process.
    patch.object(reconcile, "_memberships_match", memberships_match).start()
    patch.object(reconcile, "_fetch_message_ids_after", message_ids_after).start()
    patch.object(reconcile, "_fetch_tool_call_ids_after", tool_ids_after).start()


def install_result_writer() -> None:
    from collections.abc import Iterable

    import pyarrow as pa
    from recall.db import queries

    def insert_results(
        conn: duckdb.DuckDBPyConnection,
        rows: Iterable[tuple[str, str, bool, datetime | None]],
    ) -> None:
        # The daemon already owns the sole writer transaction here. A bounded
        # batch contributes its first row per identity, then an anti-join admits
        # only absent identities in that same transaction. Replay is a no-op.
        schema = pa.schema(
            [
                ("tool_call_id", pa.string()),
                ("result_summary", pa.string()),
                ("is_error", pa.bool_()),
                ("completed_at", pa.timestamp("us")),
                ("completed_at_utc", pa.timestamp("us", tz="UTC")),
            ]
        )
        for batch in batched(rows, 256):
            unique: dict[str, tuple[object, ...]] = {}
            for identity, summary, is_error, stamp in batch:
                aware = stamp is not None and stamp.utcoffset() is not None
                unique.setdefault(
                    identity,
                    (
                        identity,
                        summary,
                        is_error,
                        None if aware else stamp,
                        stamp if aware else None,
                    ),
                )
            table = pa.Table.from_pylist(
                [dict(zip(schema.names, row, strict=True)) for row in unique.values()],
                schema=schema,
            )
            conn.register("_checkpoint_results", table)
            try:
                conn.execute(
                    "INSERT INTO tool_results "
                    "SELECT incoming.tool_call_id, incoming.result_summary, incoming.is_error, "
                    "COALESCE(CAST(incoming.completed_at_utc AS TIMESTAMP), incoming.completed_at) "
                    "FROM _checkpoint_results incoming ANTI JOIN tool_results present "
                    "ON incoming.tool_call_id = present.tool_call_id"
                )
            finally:
                conn.unregister("_checkpoint_results")

    patch.object(queries, "insert_tool_results", insert_results).start()


def install(variant: str) -> None:
    if variant == "u128":
        install_numeric_ids()
    elif variant == "drop-results":
        install_result_writer()
    elif variant in ("baseline", "watermark-16"):
        return
    else:
        raise ValueError(variant)


def signature(conn: duckdb.DuckDBPyConnection, projection: str, table: str) -> tuple[object, ...]:
    row = conn.execute(
        f"SELECT count(*), sum(hash(COLUMNS(*))) FROM (SELECT {projection} FROM {table})"
    ).fetchone()
    assert row is not None
    return row


def integer_id(column: str) -> str:
    # DuckDB 1.5.5 accepts hexadecimal UBIGINT strings, but not hexadecimal
    # UHUGEINT strings. Promote before shifting so the high limb stays exact.
    return (
        f"((CAST(CAST('0x'||substr({column},1,16) AS UBIGINT) AS UHUGEINT) << 64) "
        f"+ CAST('0x'||substr({column},17,16) AS UBIGINT))"
    )


def convert(variant: str) -> None:
    destination = LANE / "scratch" / f"fixture-{variant}"
    destination.mkdir(exist_ok=False)
    report: dict[str, Any] = {"variant": variant, "outcome": "incomplete", "steps": {}}
    started = time.monotonic()

    def interrupt(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for name in ("recall.duckdb", "recall.fts.sqlite"):
            shutil.copy2(LANE / "scratch/fixture" / name, destination / name)
        report["steps"]["copy"] = 0
        with duckdb.connect(str(destination / "recall.duckdb")) as conn:
            conn.execute("SET threads=4")
            conn.execute("SET memory_limit='2GB'")
            conn.execute(f"SET temp_directory='{destination / 'spill'}'")
            tables = HASH_COLUMNS if variant == "u128" else {"tool_results": ()}
            conn.execute("BEGIN")
            for table, columns in tables.items():
                schema_row = conn.execute(
                    "SELECT sql FROM duckdb_tables() WHERE table_name=?", [table]
                ).fetchone()
                assert schema_row is not None
                schema = schema_row[0]
                indexes = conn.execute(
                    "SELECT sql FROM duckdb_indexes() WHERE table_name=?", [table]
                ).fetchall()
                names = [row[1] for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()]
                projection = ", ".join(
                    f"{integer_id(name)} AS {name}" if name in columns else name for name in names
                )
                before = signature(conn, projection, table)
                shadow = f"checkpoint_{table}"
                schema = schema.replace(f"CREATE TABLE {table}(", f"CREATE TABLE {shadow}(")
                if variant == "u128":
                    for column in columns:
                        schema = re.sub(rf"\b{column} VARCHAR\b", f"{column} UHUGEINT", schema)
                else:
                    schema = schema.replace(
                        "tool_call_id VARCHAR PRIMARY KEY", "tool_call_id VARCHAR NOT NULL"
                    )
                conn.execute(schema)
                conn.execute(f"INSERT INTO {shadow} SELECT {projection} FROM {table}")
                after = signature(conn, "*", shadow)
                assert before == after, table
                conn.execute(f"DROP TABLE {table}")
                conn.execute(f"ALTER TABLE {shadow} RENAME TO {table}")
                for (statement,) in indexes:
                    conn.execute(statement)
                report[table] = {"before": before, "after": after}
            if variant == "u128":
                conn.execute(
                    f"UPDATE fts_sidecar_pending SET id=CAST({integer_id('id')} AS VARCHAR)"
                )
            conn.execute("COMMIT")
            conn.execute("CHECKPOINT")
            report["steps"]["duckdb"] = 0
        # Reopen proves the persisted encoding and aggregate payload signatures.
        # This deliberately does not claim exact migration qualification.
        with duckdb.connect(str(destination / "recall.duckdb"), read_only=True) as conn:
            for table in tables:
                assert signature(conn, "*", table) == tuple(report[table]["after"]), table
        report["steps"]["reopen"] = 0
        if variant == "u128":
            with sqlite3.connect(destination / "recall.fts.sqlite") as sidecar:
                sidecar.create_function("decimal_id", 1, lambda value: str(int(value, 16)))
                for table, column in (
                    ("message_fts_rowid", "message_id"),
                    ("tool_calls_fts_rowid", "tool_call_id"),
                ):
                    sidecar.execute(f"UPDATE {table} SET {column}=decimal_id({column})")
                sidecar.execute(
                    "UPDATE bootstrap_progress SET last_id=decimal_id(last_id) "
                    "WHERE last_id IS NOT NULL"
                )
            report["steps"]["sidecar"] = 0
        report["outcome"] = "passed"
    except BaseException as error:
        report["outcome"] = "incomplete" if isinstance(error, SystemExit) else "failed"
        report["error"] = "".join(traceback.format_exception(error))
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        path = LANE / "scratch/evidence" / f"conversion-{variant}.json"
        path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps(report, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("u128", "drop-results"))
    convert(parser.parse_args().variant)
