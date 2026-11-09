"""Schema-36 upgrade and transfer constraint outcomes."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from recall.db.migrations import get_migrations
from recall.db.schema import ensure_schema, get_schema_version, set_schema_version_to


def migration():
    found = [item for item in get_migrations() if item.id == "0038_duplicate_transfers"]
    assert len(found) == 1, "schema 38 must register its additive transfer migration"
    return found[0]


def test_upgrade_preserves_populated_rows_and_replay(tmp_path: Path):
    database = tmp_path / "upgrade.duckdb"
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
        conn.execute("DROP TABLE IF EXISTS duplicate_transfers")
        conn.execute("DELETE FROM schema_version")
        set_schema_version_to(conn, 36)
        conn.execute("INSERT INTO sessions VALUES ('s', 'grok', '/owned/session.jsonl', 'native')")
        before = conn.execute("SELECT * FROM sessions").fetchall()
        assert migration().upgrade(conn)
        assert get_schema_version(conn) == 38
        assert conn.execute("SELECT * FROM sessions").fetchall() == before
        assert migration().upgrade(conn)
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE migration_id = '0038_duplicate_transfers'"
        ).fetchone() == (1,)
        assert conn.execute("SELECT * FROM duplicate_transfers").fetchall() == []


def test_failure_rolls_back_table_version_and_audit(monkeypatch: pytest.MonkeyPatch):
    from recall.db import schema

    with duckdb.connect() as conn:
        ensure_schema(conn)
        conn.execute("DROP TABLE IF EXISTS duplicate_transfers")
        conn.execute("DELETE FROM schema_version")
        set_schema_version_to(conn, 36)

        def fail(*args):
            raise RuntimeError("injected version write failure")

        monkeypatch.setattr(schema, "set_schema_version_to", fail)
        with pytest.raises(RuntimeError, match="injected"):
            migration().upgrade(conn)
        assert get_schema_version(conn) == 36
        assert (
            conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name = 'duplicate_transfers'"
            ).fetchall()
            == []
        )
        assert (
            conn.execute(
                "SELECT * FROM schema_migrations WHERE migration_id = '0038_duplicate_transfers'"
            ).fetchall()
            == []
        )


def test_fresh_and_upgrade_columns_and_constraints_match():
    with duckdb.connect() as fresh, duckdb.connect() as upgrade:
        ensure_schema(fresh)
        ensure_schema(upgrade)
        upgrade.execute("DROP TABLE IF EXISTS duplicate_transfers")
        upgrade.execute("DELETE FROM schema_version")
        set_schema_version_to(upgrade, 36)
        migration().upgrade(upgrade)
        for query in (
            """SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns WHERE table_name = 'duplicate_transfers'
                ORDER BY ordinal_position""",
            """SELECT constraint_type, constraint_text FROM duckdb_constraints()
                WHERE table_name = 'duplicate_transfers'
                ORDER BY constraint_type, constraint_text""",
        ):
            assert fresh.execute(query).fetchall() == upgrade.execute(query).fetchall()


@pytest.mark.parametrize(
    "assignment",
    [
        "predecessor_publish_seq=-1",
        "predecessor_refusal_kind='stopping_diagnostic'",
        "predecessor_refusal_observation='{}'",
        "context_message_bound=0",
        "survivor_orphan_count=1",
        "coverage_proven_at=now()",
        "coverage_proven_at=now(), predecessor_content_epoch=1, survivor_orphan_count=0",
        "predecessor_refusal_kind='stopping_diagnostic', "
        "predecessor_refusal_observation='\"' || repeat('x', 1024) || '\"'",
        "coverage_proven_at=now(), predecessor_content_epoch=1, "
        "witness_epoch=1, survivor_orphan_count=2",
        "coverage_proven_at=now(), predecessor_content_epoch=1, witness_epoch=1, "
        "survivor_orphan_count=0, stuck_reason='normalized_mismatch'",
        "coverage_proven_at=now(), predecessor_content_epoch=1, witness_epoch=1, "
        "survivor_orphan_count=0, predecessor_refusal_kind='stopping_diagnostic', "
        "predecessor_refusal_observation='{}'",
    ],
)
def test_constraints_reject_invalid_transfer(assignment):
    with duckdb.connect() as conn:
        ensure_schema(conn)
        migration()
        conn.execute(
            """INSERT INTO duplicate_transfers(survivor_id, predecessor_id, source,
                original_path, moved_path, survivor_offset, survivor_prefix_sha256,
                survivor_message_count, survivor_tool_count, context_message_bound, phase)
                VALUES ('s', 'p', 'grok', '/old', '/new', 1, ?, 1, 1, 1, 'catch_up')""",
            ["a" * 64],
        )
        with pytest.raises(duckdb.ConstraintException):
            conn.execute(f"UPDATE duplicate_transfers SET {assignment}")


def test_counter_default_update_and_reopen(tmp_path):
    database = tmp_path / "counter.duckdb"
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
        conn.execute(
            """INSERT INTO duplicate_transfers(survivor_id,predecessor_id,source,
                original_path,moved_path,survivor_offset,survivor_prefix_sha256,
                survivor_message_count,survivor_tool_count,context_message_bound,phase)
                VALUES ('s','p','grok','/old','/new',1,?,0,0,0,'catch_up')""",
            ["a" * 64],
        )
        assert conn.execute(
            "SELECT predecessor_publish_seq FROM duplicate_transfers"
        ).fetchone() == (0,)
        conn.execute(
            "UPDATE duplicate_transfers SET predecessor_publish_seq=predecessor_publish_seq+1"
        )
    with duckdb.connect(str(database)) as conn:
        assert conn.execute(
            "SELECT predecessor_publish_seq,survivor_offset,survivor_prefix_sha256 "
            "FROM duplicate_transfers"
        ).fetchone() == (1, 1, "a" * 64)


def test_upgrade_below36_is_not_applied():
    with duckdb.connect() as conn:
        ensure_schema(conn)
        conn.execute("DROP TABLE duplicate_transfers")
        conn.execute("DELETE FROM schema_version")
        set_schema_version_to(conn, 35)
        assert not migration().upgrade(conn)
        assert get_schema_version(conn) == 35
        assert (
            conn.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name='duplicate_transfers'"
            ).fetchone()
            is None
        )
