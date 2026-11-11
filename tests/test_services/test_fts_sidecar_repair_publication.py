"""Keyword repair publishes each pending page as one sidecar commit.

The commit count comes from the SQLite file change counter (header bytes 24-27),
which rollback-journal mode increments once per committed write transaction.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import duckdb
import pytest
from recall import _native
from recall.db.fts_sidecar import open_sidecar
from recall.db.schema import ensure_schema
from recall.services.fts_sidecar_reconcile import (
    reconcile_sidecar,
    repair_pending_batch,
)

_PAGE_ROWS = 6


def _repair_page(conn: duckdb.DuckDBPyConnection, sidecar: sqlite3.Connection) -> tuple[int, int]:
    result = repair_pending_batch(conn, sidecar, limit=_PAGE_ROWS)
    return result.applied, result.remaining


def _drain(conn: duckdb.DuckDBPyConnection, sidecar: sqlite3.Connection) -> tuple[int, int]:
    stats = reconcile_sidecar(conn, sidecar, batch_size=_PAGE_ROWS)
    return sum(stats.pending_drained.values()), sum(stats.pending_remaining.values())


# Each repair entry point returns (rows applied, rows still pending).
_REPAIRS: dict[str, Callable[[duckdb.DuckDBPyConnection, sqlite3.Connection], tuple[int, int]]] = {
    "repair_pending_batch": _repair_page,
    "reconcile_sidecar": _drain,
}


@pytest.mark.parametrize("repair", sorted(_REPAIRS))
def test_pending_page_publishes_in_one_commit(tmp_path: Path, repair: str) -> None:
    path = tmp_path / "recall.fts.sqlite"
    with _native.duckdb_connect(":memory:") as conn:
        ensure_schema(conn)
        sidecar = open_sidecar(path)
        try:
            _seed_page(conn, sidecar)
            before = _change_counter(path)

            applied, remaining = _REPAIRS[repair](conn, sidecar)

            assert _change_counter(path) - before == 1
            assert (applied, remaining) == (_PAGE_ROWS, 0)
            assert _keyword_state(sidecar) == {
                "messages": ["msg-a", "msg-b", "msg-c"],
                "tool_calls": ["tc-a"],
                "alpha": ["msg-a"],
                "stale": [],
            }
        finally:
            sidecar.close()


@pytest.mark.parametrize("repair", sorted(_REPAIRS))
def test_poisoned_row_stays_pending_while_its_page_retires(tmp_path: Path, repair: str) -> None:
    path = tmp_path / "recall.fts.sqlite"
    with _native.duckdb_connect(":memory:") as conn:
        ensure_schema(conn)
        sidecar = open_sidecar(path)
        try:
            _seed_page(conn, sidecar)
            sidecar.execute("""CREATE TRIGGER reject_b BEFORE INSERT ON message_fts_rowid
                WHEN NEW.message_id = 'msg-b' BEGIN
                SELECT RAISE(ABORT, 'poisoned row'); END""")

            applied, remaining = _REPAIRS[repair](conn, sidecar)

            assert (applied, remaining) == (_PAGE_ROWS - 1, 1)
            assert _pending(conn) == [("message", "msg-b", "upsert")]
            assert _keyword_state(sidecar) == {
                "messages": ["msg-a", "msg-c"],
                "tool_calls": ["tc-a"],
                "alpha": ["msg-a"],
                "stale": [],
            }

            sidecar.execute("DROP TRIGGER reject_b")
            retried = repair_pending_batch(conn, sidecar, limit=_PAGE_ROWS)

            assert (retried.applied, retried.remaining) == (1, 0)
            assert _keyword_state(sidecar)["messages"] == ["msg-a", "msg-b", "msg-c"]
        finally:
            sidecar.close()


@pytest.mark.parametrize("repair", sorted(_REPAIRS))
def test_page_retires_nothing_when_its_publication_is_lost(tmp_path: Path, repair: str) -> None:
    path = tmp_path / "recall.fts.sqlite"
    with _native.duckdb_connect(":memory:") as conn:
        ensure_schema(conn)
        sidecar = open_sidecar(path)
        try:
            _seed_page(conn, sidecar)
            sidecar.execute("""CREATE TRIGGER abort_page BEFORE INSERT ON message_fts_rowid
                WHEN NEW.message_id = 'msg-b' BEGIN
                SELECT RAISE(ROLLBACK, 'transaction lost'); END""")

            applied, remaining = _REPAIRS[repair](conn, sidecar)

            # No queued upsert publishes. (reconcile_sidecar's ghost pass still runs.)
            state = _keyword_state(sidecar)
            assert (applied, remaining) == (0, _PAGE_ROWS)
            assert {"msg-a", "msg-b", "msg-c", "tc-a"}.isdisjoint(
                state["messages"] + state["tool_calls"]
            )

            sidecar.execute("DROP TRIGGER abort_page")
            retried = repair_pending_batch(conn, sidecar, limit=_PAGE_ROWS)

            assert (retried.applied, retried.remaining) == (_PAGE_ROWS, 0)
        finally:
            sidecar.close()


def _seed_page(conn: duckdb.DuckDBPyConnection, sidecar: sqlite3.Connection) -> None:
    """Seed six pending rows: three message upserts, a tool upsert, and two deletes."""
    for key, content in (("msg-a", "alpha"), ("msg-b", "bravo"), ("msg-c", "charlie")):
        conn.execute(
            "INSERT INTO message_state(message_id, role, fts_content, fts_thinking) "
            "VALUES (?, 'assistant', ?, '')",
            [key, content],
        )
    conn.execute(
        "INSERT INTO tool_calls(id, session_id, idx, tool_name, bash_command, is_compound) "
        "VALUES ('tc-a', 'session-1', 0, 'bash', 'echo tango', false)"
    )
    with sidecar:
        sidecar.execute(
            "INSERT INTO message_fts_rowid(message_id, content_fingerprint) VALUES ('gone', 'x')"
        )
        sidecar.execute(
            "INSERT INTO message_fts(rowid, fts_content, fts_thinking) "
            "VALUES (last_insert_rowid(), 'stale', '')"
        )
        sidecar.execute(
            "INSERT INTO tool_calls_fts_rowid(tool_call_id, content_fingerprint) "
            "VALUES ('tc-gone', 'x')"
        )
        sidecar.execute(
            "INSERT INTO tool_calls_fts(rowid, bash_command) VALUES (last_insert_rowid(), 'stale')"
        )
    for kind, key, op in (
        ("message", "msg-a", "upsert"),
        ("message", "msg-b", "upsert"),
        ("message", "gone", "delete"),
        ("tool_call", "tc-a", "upsert"),
        ("message", "msg-c", "upsert"),
        ("tool_call", "tc-gone", "delete"),
    ):
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind, id, op) VALUES (?, ?, ?)", [kind, key, op]
        )


def _change_counter(path: Path) -> int:
    with path.open("rb") as handle:
        handle.seek(24)
        return int.from_bytes(handle.read(4), "big")


def _pending(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str]]:
    return [
        (str(kind), str(key), str(op))
        for kind, key, op in conn.execute(
            "SELECT kind, id, op FROM fts_sidecar_pending ORDER BY queued_at, rowid"
        ).fetchall()
    ]


def _keyword_state(sidecar: sqlite3.Connection) -> dict[str, list[str]]:
    def ids(sql: str, *parameters: str) -> list[str]:
        return [str(row[0]) for row in sidecar.execute(sql, parameters)]

    return {
        "messages": ids("SELECT message_id FROM message_fts_rowid ORDER BY 1"),
        "tool_calls": ids("SELECT tool_call_id FROM tool_calls_fts_rowid ORDER BY 1"),
        "alpha": ids(
            "SELECT m.message_id FROM message_fts f JOIN message_fts_rowid m "
            "ON m.rowid = f.rowid WHERE message_fts MATCH ? ORDER BY 1",
            "alpha",
        ),
        "stale": ids(
            "SELECT t.tool_call_id FROM tool_calls_fts f JOIN tool_calls_fts_rowid t "
            "ON t.rowid = f.rowid WHERE tool_calls_fts MATCH ? ORDER BY 1",
            "stale",
        ),
    }
