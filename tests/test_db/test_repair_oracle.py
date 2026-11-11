"""The pre-attempt reference proves logical rows across rebuilds and replacement."""

from pathlib import Path
from uuid import UUID

import duckdb
import pytest
from recall.db.repair_oracle import OracleUnavailable, capture_oracle, verify_oracle

TRIGGER = UUID("00000000-0000-4000-8000-000000000001")


def populate(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE SCHEMA companions")
    conn.execute("CREATE TABLE canonical (id VARCHAR PRIMARY KEY, text VARCHAR, n DECIMAL(30,9))")
    conn.execute("INSERT INTO canonical VALUES ('a', NULL, 123.000000001), ('b', 'βeta', NULL)")
    conn.execute("CREATE TABLE companions.vectors (id VARCHAR, v FLOAT[3], raw BLOB)")
    conn.execute(
        "INSERT INTO companions.vectors VALUES ('a', [1.0, -0.0, NULL], '\\x00\\xFF'::BLOB)"
    )
    conn.execute("CREATE TABLE intents (kind VARCHAR, id VARCHAR)")
    conn.execute("INSERT INTO intents VALUES ('message', 'a'), ('message', 'a'), ('session', 'b')")
    conn.execute("CREATE INDEX witness ON canonical(text)")


def test_complete_reference_survives_reorder_and_fresh_process_open(tmp_path: Path) -> None:
    db = tmp_path / "original.duckdb"
    with duckdb.connect(str(db)) as conn:
        populate(conn)
        reference = capture_oracle(conn, tmp_path / "oracle", TRIGGER, sort_bytes=128)
        conn.execute("CREATE TABLE reverse_intents AS SELECT * FROM intents ORDER BY kind DESC")
        conn.execute("DELETE FROM intents")
        conn.execute("INSERT INTO intents SELECT * FROM reverse_intents")
        conn.execute("DROP TABLE reverse_intents")
    with duckdb.connect(str(db)) as conn:
        verify_oracle(conn, tmp_path / "oracle", reference)


@pytest.mark.parametrize(
    "change",
    [
        "UPDATE canonical SET text = 'changed' WHERE id = 'a'",
        "DELETE FROM intents WHERE rowid = 0",
        "INSERT INTO intents VALUES ('message', 'a')",
        "UPDATE companions.vectors SET v = [1.0, 0.5, NULL]",
        "ALTER TABLE canonical ADD COLUMN unexpected INTEGER",
        "DROP TABLE companions.vectors",
    ],
)
def test_reference_rejects_content_or_catalog_change(tmp_path: Path, change: str) -> None:
    with duckdb.connect() as conn:
        populate(conn)
        reference = capture_oracle(conn, tmp_path / "oracle", TRIGGER, sort_bytes=128)
        conn.execute(change)
        with pytest.raises(OracleUnavailable):
            verify_oracle(conn, tmp_path / "oracle", reference)


def test_reference_rejects_incomplete_or_tampered_capture(tmp_path: Path) -> None:
    with duckdb.connect() as conn:
        populate(conn)
        reference = capture_oracle(conn, tmp_path / "oracle", TRIGGER)
        (tmp_path / "oracle" / "manifest.json").write_bytes(b"{}")
        with pytest.raises(OracleUnavailable):
            verify_oracle(conn, tmp_path / "oracle", reference)


def test_oversized_text_and_blob_are_exact_with_small_sort_runs(tmp_path: Path) -> None:
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE payload (text VARCHAR, raw BLOB)")
        conn.execute("INSERT INTO payload VALUES (?, ?)", ["λ" * 100_000, bytes(range(256)) * 1000])
        reference = capture_oracle(conn, tmp_path / "oracle", TRIGGER, sort_bytes=128)
        verify_oracle(conn, tmp_path / "oracle", reference)
        conn.execute("UPDATE payload SET raw = '\\x00'::BLOB")
        with pytest.raises(OracleUnavailable):
            verify_oracle(conn, tmp_path / "oracle", reference)


def test_fresh_comparison_keeps_only_one_transient_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import closing

    from recall.db import repair_oracle as oracle
    from recall.db.fts_sidecar import open_sidecar, upsert_message_fts

    keywords = tmp_path / "recall.fts.sqlite"
    with closing(open_sidecar(keywords)) as sidecar:
        upsert_message_fts(sidecar, "a", "exact keyword content", "")
        sidecar.commit()
    with duckdb.connect() as conn:
        populate(conn)
        reference = capture_oracle(conn, tmp_path / "oracle", TRIGGER, sidecar=keywords)
        original = oracle._sorted_rows
        observations: list[tuple[str, ...]] = []

        def sorted_rows(write_next, directory, filename, sort_bytes):
            observations.append(tuple(path.name for path in directory.glob("table-*.rows")))
            return original(write_next, directory, filename, sort_bytes)

        monkeypatch.setattr(oracle, "_sorted_rows", sorted_rows)
        verify_oracle(conn, tmp_path / "oracle", reference, sidecar=keywords)
        assert len(observations) >= 4
        assert all(not files for files in observations), observations
