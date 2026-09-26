from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from recall.db.fts_sidecar import (
    FtsSidecarUnavailableError,
    open_sidecar,
    probe_sqlite_fts5_support,
)


def test_open_sidecar_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "recall.fts.sqlite"
    first = open_sidecar(path)
    first.close()

    second = open_sidecar(path)
    second.close()


def test_probe_raises_when_fts5_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 42, 0))
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.42.0")

    with pytest.raises(FtsSidecarUnavailableError) as exc_info:
        probe_sqlite_fts5_support()

    message = str(exc_info.value)
    assert "3.43" in message
    assert 'backend = "duckdb"' in message


def test_rowid_mapping_tables_have_correct_shape(tmp_path: Path) -> None:
    conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        assert _table_info(conn, "message_fts_rowid") == [
            ("rowid", "INTEGER", False, True),
            ("message_id", "TEXT", True, False),
        ]
        assert _table_info(conn, "tool_calls_fts_rowid") == [
            ("rowid", "INTEGER", False, True),
            ("tool_call_id", "TEXT", True, False),
        ]
        assert _unique_indexes(conn, "message_fts_rowid") == {
            "sqlite_autoindex_message_fts_rowid_1"
        }
        assert _unique_indexes(conn, "tool_calls_fts_rowid") == {
            "sqlite_autoindex_tool_calls_fts_rowid_1"
        }
    finally:
        conn.close()


def _table_info(
    conn: sqlite3.Connection,
    table_name: str,
) -> list[tuple[str, str, bool, bool]]:
    return [
        (row[1], row[2], bool(row[3]), bool(row[5]))
        for row in conn.execute(f"PRAGMA table_info({table_name})")
    ]


def _unique_indexes(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA index_list({table_name})") if bool(row[2])}
