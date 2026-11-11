"""Catalog-only witnesses remain reconstructible after an armed DROP."""

import duckdb
from recall.db.repair import capture_target_manifest


def test_complete_manifest_keeps_catalog_only_and_non_main_targets() -> None:
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE canonical (id VARCHAR, text VARCHAR)")
        conn.execute("CREATE INDEX witness ON canonical(text)")
        conn.execute("CREATE SCHEMA companion")
        conn.execute("CREATE TABLE companion.rows (id VARCHAR)")
        conn.execute("CREATE INDEX other_witness ON companion.rows(id)")
        manifest = capture_target_manifest(conn)
        identities = {(t.schema_name, t.index_name, t.table_name) for t in manifest.targets}
        assert ("main", "witness", "canonical") in identities
        assert ("companion", "other_witness", "rows") in identities
        conn.execute("DROP INDEX companion.other_witness")
        for target in manifest.targets:
            if target.index_name == "other_witness":
                conn.execute(target.sql)
        assert conn.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE schema_name = 'companion'"
        ).fetchall() == [("other_witness",)]


def test_missing_schema_target_manifest_is_stable_after_healing() -> None:
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE source_files (source VARCHAR, source_path VARCHAR)")
        before = capture_target_manifest(conn)
        assert {(t.schema_name, t.index_name) for t in before.targets} == {
            ("main", "idx_source_files_source_path")
        }
        for target in before.targets:
            conn.execute(target.sql)
        assert capture_target_manifest(conn) == before
