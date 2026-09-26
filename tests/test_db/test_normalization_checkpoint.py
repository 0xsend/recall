"""Durable normalization checkpoint storage contract (REQ-INDEX-025).

A resumable checkpoint is only trustworthy if it was written by the same
transaction that acknowledged the catalog generation it describes.  These tests
pin the storage half of that rule: the column is additive and nullable so every
pre-existing row reads as "no checkpoint, take the full path", and the value
moves only with a successful acknowledgement.

The parser-side envelope lives in ``tests/test_parsers/test_normalization_checkpoint_adapters.py``
and the resume/fallback decision in
``tests/test_services/test_checkpoint_resume_fallbacks.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest
from recall.db.schema import SCHEMA_VERSION, _get_schema_version, ensure_schema
from recall.db.source_files import SourceCatalog, SourceSignature, prefix_sha256, source_key

CHECKPOINT_COLUMN = "normalization_checkpoint"
LEGACY_PATH = "/sessions/legacy.jsonl"

# Two distinct well-formed envelopes; their bodies are opaque to storage, which
# must round-trip whatever the parser layer encoded.
FIRST_CHECKPOINT = '{"version":1,"parser_revision":"rev-a","offset":64,"adapter_state":{}}'
SECOND_CHECKPOINT = '{"version":1,"parser_revision":"rev-a","offset":128,"adapter_state":{}}'


def _columns(conn: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ?",
        [table],
    ).fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def _signature(size: int) -> SourceSignature:
    return SourceSignature(
        dev=1,
        inode=2,
        ctime_ns=10,
        mtime_ns=20,
        size=size,
        sidecar_mtime_ns=0,
        parser_revision="rev-a",
        sidecar_signature="",
    )


@pytest.fixture
def catalog_conn() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def _seed(
    conn: duckdb.DuckDBPyConnection, *, size: int = 64, path: str = "/sessions/rollout.jsonl"
) -> tuple[SourceCatalog, int]:
    catalog = SourceCatalog(conn, clock=lambda: 100.0)
    generation = catalog.observe("codex", "/sessions", path, _signature(size))
    return catalog, generation


def _stored_checkpoint(conn: duckdb.DuckDBPyConnection, path: str) -> str | None:
    row = conn.execute(
        f"SELECT {CHECKPOINT_COLUMN} FROM source_files WHERE source_key = ?",
        [source_key("codex", path)],
    ).fetchone()
    assert row is not None
    return None if row[0] is None else str(row[0])


def _to_schema_v30(conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Reshape a current-schema database back to schema version 30.

    Version 31 adds ``source_files.normalization_checkpoint``.  DuckDB refuses
    ``DROP COLUMN`` while a secondary index is present, so the catalog's only
    index is dropped and recreated the way migration 0030 does it.
    """
    conn.execute("DROP INDEX IF EXISTS idx_source_files_source_path")
    conn.execute(f"ALTER TABLE source_files DROP COLUMN IF EXISTS {CHECKPOINT_COLUMN}")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_files_source_path "
        "ON source_files(source, source_path)"
    )
    conn.execute("DELETE FROM schema_version WHERE version > 30")
    conn.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (30)")
    conn.execute("DELETE FROM schema_migrations WHERE migration_id LIKE '0031%'")
    return conn


class TestSchemaShape:
    def test_fresh_schema_carries_a_nullable_checkpoint_column(
        self, catalog_conn: duckdb.DuckDBPyConnection
    ) -> None:
        columns = _columns(catalog_conn, "source_files")
        assert columns.get(CHECKPOINT_COLUMN) == "VARCHAR"

        catalog, _ = _seed(catalog_conn)
        assert catalog.get("codex", "/sessions/rollout.jsonl") is not None
        # A catalogued but never-acknowledged source has no checkpoint at all;
        # NULL is the only value that means "no resume proof exists".
        assert _stored_checkpoint(catalog_conn, "/sessions/rollout.jsonl") is None


class TestVersion30Migration:
    """A version-30 row is seeded with raw DDL-shaped SQL on purpose.

    The catalog API is written against the current schema; using it here would
    exercise the post-migration column set on a pre-migration database and
    report a failure that has nothing to do with the migration.
    """

    def _seed_v30_row(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute(
            """INSERT INTO source_files
                   (source_key, source, source_path, root_path, ctime_ns, mtime_ns, size,
                    parser_revision, desired_generation, committed_generation, committed_offset,
                    committed_prefix_sha256, observed_at)
               VALUES (?, 'codex', ?, '/sessions', 10, 20, 64, 'rev-a', 1, 1, 64, ?, 100.0)""",
            [
                source_key("codex", LEGACY_PATH),
                LEGACY_PATH,
                prefix_sha256(b"x" * 64),
            ],
        )

    def _catalog_snapshot(self, conn: duckdb.DuckDBPyConnection) -> list[tuple[object, ...]]:
        return conn.execute(
            "SELECT source_key, source, source_path, root_path, desired_generation, "
            "committed_generation, committed_offset, committed_prefix_sha256, content_epoch, "
            "size FROM source_files ORDER BY source_key"
        ).fetchall()

    def test_version_30_database_gains_the_column_additively(self, tmp_path: Path) -> None:
        conn = duckdb.connect(str(tmp_path / "v30.duckdb"))
        try:
            ensure_schema(conn)
            _to_schema_v30(conn)
            assert CHECKPOINT_COLUMN not in _columns(conn, "source_files")
            self._seed_v30_row(conn)
            before = self._catalog_snapshot(conn)

            from recall.db.migrations import run_pending_migrations

            run_pending_migrations(conn, 30)

            assert _get_schema_version(conn) == SCHEMA_VERSION
            assert SCHEMA_VERSION >= 31
            assert _columns(conn, "source_files").get(CHECKPOINT_COLUMN) == "VARCHAR"
            assert self._catalog_snapshot(conn) == before
            # A row acknowledged before the upgrade has no resume proof, so it
            # must read NULL rather than inherit a fabricated checkpoint.
            assert _stored_checkpoint(conn, LEGACY_PATH) is None
        finally:
            conn.close()

    def test_migration_is_idempotent_when_replayed(self, tmp_path: Path) -> None:
        conn = duckdb.connect(str(tmp_path / "v30-replay.duckdb"))
        try:
            ensure_schema(conn)
            _to_schema_v30(conn)
            self._seed_v30_row(conn)

            from recall.db.migrations import run_pending_migrations

            run_pending_migrations(conn, 30)
            catalog = SourceCatalog(conn, clock=lambda: 100.0)
            generation = catalog.observe("codex", "/sessions", LEGACY_PATH, _signature(128))
            assert catalog.acknowledge(
                "codex",
                LEGACY_PATH,
                generation,
                128,
                prefix_sha256(b"x" * 128),
                checkpoint=FIRST_CHECKPOINT,
            )
            run_pending_migrations(conn, 30)

            assert _get_schema_version(conn) == SCHEMA_VERSION
            assert conn.execute("SELECT COUNT(*) FROM source_files").fetchone() == (1,)
            # Replay must not reset a checkpoint written after the first run.
            assert _stored_checkpoint(conn, LEGACY_PATH) == FIRST_CHECKPOINT
        finally:
            conn.close()


class TestAcknowledgementBoundary:
    def test_checkpoint_and_offset_advance_in_the_same_statement(
        self, catalog_conn: duckdb.DuckDBPyConnection
    ) -> None:
        catalog, generation = _seed(catalog_conn)
        assert catalog.acknowledge(
            "codex",
            "/sessions/rollout.jsonl",
            generation,
            64,
            prefix_sha256(b"x" * 64),
            checkpoint=FIRST_CHECKPOINT,
        )

        item = catalog.get("codex", "/sessions/rollout.jsonl")
        assert item is not None
        assert item.committed_offset == 64
        assert item.normalization_checkpoint == FIRST_CHECKPOINT

    def test_stale_generation_leaves_the_prior_checkpoint_intact(
        self, catalog_conn: duckdb.DuckDBPyConnection
    ) -> None:
        catalog, first_generation = _seed(catalog_conn)
        assert catalog.acknowledge(
            "codex",
            "/sessions/rollout.jsonl",
            first_generation,
            64,
            prefix_sha256(b"x" * 64),
            checkpoint=FIRST_CHECKPOINT,
        )
        second_generation = catalog.observe(
            "codex", "/sessions", "/sessions/rollout.jsonl", _signature(128)
        )
        assert second_generation > first_generation
        assert catalog.acknowledge(
            "codex",
            "/sessions/rollout.jsonl",
            second_generation,
            128,
            prefix_sha256(b"x" * 128),
            checkpoint=SECOND_CHECKPOINT,
        )

        # A worker that prepared against the older generation must not land its
        # stale checkpoint on top of the newer acknowledgement.
        assert not catalog.acknowledge(
            "codex",
            "/sessions/rollout.jsonl",
            first_generation,
            64,
            prefix_sha256(b"x" * 64),
            checkpoint=FIRST_CHECKPOINT,
        )
        item = catalog.get("codex", "/sessions/rollout.jsonl")
        assert item is not None
        assert item.committed_offset == 128
        assert item.normalization_checkpoint == SECOND_CHECKPOINT

    def test_declining_to_resume_clears_the_previous_checkpoint(
        self, catalog_conn: duckdb.DuckDBPyConnection
    ) -> None:
        catalog, first_generation = _seed(catalog_conn)
        assert catalog.acknowledge(
            "codex",
            "/sessions/rollout.jsonl",
            first_generation,
            64,
            prefix_sha256(b"x" * 64),
            checkpoint=FIRST_CHECKPOINT,
        )
        second_generation = catalog.observe(
            "codex", "/sessions", "/sessions/rollout.jsonl", _signature(128)
        )
        # An adapter that reaches an open boundary acknowledges the new content
        # with no checkpoint. Keeping the old one would leave a proof pointing
        # at offset 64 beside a committed offset of 128 -- a resume that skips
        # rows. The checkpoint is cleared instead.
        assert catalog.acknowledge(
            "codex",
            "/sessions/rollout.jsonl",
            second_generation,
            128,
            prefix_sha256(b"x" * 128),
            checkpoint=None,
        )

        item = catalog.get("codex", "/sessions/rollout.jsonl")
        assert item is not None
        assert item.committed_offset == 128
        assert item.normalization_checkpoint is None

    def test_rolled_back_acknowledgement_restores_the_prior_checkpoint(
        self, catalog_conn: duckdb.DuckDBPyConnection
    ) -> None:
        catalog, first_generation = _seed(catalog_conn)
        assert catalog.acknowledge(
            "codex",
            "/sessions/rollout.jsonl",
            first_generation,
            64,
            prefix_sha256(b"x" * 64),
            checkpoint=FIRST_CHECKPOINT,
        )
        second_generation = catalog.observe(
            "codex", "/sessions", "/sessions/rollout.jsonl", _signature(128)
        )

        catalog_conn.execute("BEGIN TRANSACTION")
        assert catalog.acknowledge(
            "codex",
            "/sessions/rollout.jsonl",
            second_generation,
            128,
            prefix_sha256(b"x" * 128),
            checkpoint=SECOND_CHECKPOINT,
        )
        catalog_conn.execute("ROLLBACK")

        item = catalog.get("codex", "/sessions/rollout.jsonl")
        assert item is not None
        # The rows the checkpoint describes rolled back with it; both halves of
        # the envelope survive or neither does.
        assert item.committed_offset == 64
        assert item.normalization_checkpoint == FIRST_CHECKPOINT
