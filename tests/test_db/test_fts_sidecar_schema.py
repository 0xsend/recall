from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from recall.db.fts_sidecar import (
    FtsSidecarUnavailableError,
    open_sidecar,
    probe_sqlite_fts5_support,
    upsert_message_fts,
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


def test_fingerprint_schema_rejects_prior_writers_even_for_existing_ids(tmp_path: Path) -> None:
    conn = open_sidecar(tmp_path / "keywords.sqlite")
    try:
        upsert_message_fts(conn, "m", "needle", "")
        before = conn.execute("SELECT hex(block) FROM message_fts_data WHERE id=1").fetchone()
        for key in ("m", "another"):
            with pytest.raises(sqlite3.IntegrityError, match="fingerprint-aware"), conn:
                conn.execute(
                    "INSERT OR IGNORE INTO message_fts_rowid(message_id) VALUES (?)", [key]
                )
        assert (
            conn.execute("SELECT hex(block) FROM message_fts_data WHERE id=1").fetchone() == before
        )
    finally:
        conn.close()


def test_legacy_open_requires_explicit_initialization_without_mutation(tmp_path: Path) -> None:
    path = tmp_path / "keywords.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE message_fts_rowid(rowid INTEGER PRIMARY KEY, message_id TEXT UNIQUE)"
        )
    before = path.read_bytes()
    with pytest.raises(FtsSidecarUnavailableError, match="explicit canonical initialization"):
        open_sidecar(path)
    assert path.read_bytes() == before
