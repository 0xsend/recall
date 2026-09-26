from __future__ import annotations

import duckdb
import pytest
from recall.db.schema import SCHEMA_VERSION, ensure_schema, recreate_embedding_tables


def _table_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    return [str(row[1]) for row in rows]


def _table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()
    return bool(row and row[0])


def _create_v13_db_with_blocked_0014() -> duckdb.DuckDBPyConnection:
    """Build a v13-shaped DB where 0014 fails while relaxing message_state.

    The pre-created message_state_new table is fault injection for the DDL
    recipe used by 0014. Dropping it models the operator removing the blocker
    before retrying ensure_schema.
    """
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("INSERT INTO schema_version (version) VALUES (13)")
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL CHECK (source IN ('claude_code', 'codex', 'pi_agent')),
            source_path TEXT UNIQUE NOT NULL,
            source_session_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE message_state (
            message_id TEXT PRIMARY KEY,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT,
            thinking TEXT,
            timestamp TIMESTAMP,
            has_thinking BOOLEAN DEFAULT FALSE
        )
        """
    )
    conn.execute("INSERT INTO sessions VALUES ('s1', 'claude_code', '/tmp/s1', NULL)")
    conn.execute(
        "INSERT INTO message_state (message_id, role, content) VALUES ('m1', 'user', 'hello')"
    )
    conn.execute(
        """
        CREATE TABLE runtime_state (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
            embedding_dimensions INTEGER
        )
        """
    )
    conn.execute("INSERT OR IGNORE INTO runtime_state (singleton) VALUES (TRUE)")
    conn.execute(
        """
        CREATE TABLE embedding_cache (
            cache_key TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('content', 'thinking', 'bash')),
            raw_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            embedding FLOAT[384] NOT NULL,
            normalization_version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE message_state_new (
            message_id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT,
            thinking TEXT,
            timestamp TIMESTAMP,
            has_thinking BOOLEAN DEFAULT FALSE
        )
        """
    )
    return conn


def test_fresh_schema_creates_all_tables() -> None:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)

    expected_tables = [
        "schema_version",
        "sessions",
        "messages",
        "session_state",
        "message_state",
        "tool_calls",
        "message_embeddings",
        "tool_call_embeddings",
        "embedding_cache",
        "runtime_state",
    ]
    for table in expected_tables:
        assert _table_exists(conn, table), f"table {table} not created"


def test_schema_version_is_set() -> None:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)

    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    assert row is not None
    assert row[0] == SCHEMA_VERSION


def test_fresh_schema_uses_bigint_for_token_counters() -> None:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)

    expected = {
        "session_state": ("input_tokens", "output_tokens", "cached_input_tokens"),
        "usage_events": (
            "prompt_tokens",
            "cached_prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
        ),
        "runtime_state": ("last_context_input_tokens", "last_context_output_tokens"),
    }
    for table, columns in expected.items():
        actual = {
            str(row[1]): str(row[2]).upper()
            for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        }
        assert all(actual[column] == "BIGINT" for column in columns)


def test_schema_version_mismatch_raises() -> None:
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("INSERT INTO schema_version (version) VALUES (7)")

    with pytest.raises(RuntimeError, match="schema version mismatch"):
        ensure_schema(conn)


def test_no_fk_constraints() -> None:
    """Verify no REFERENCES clauses exist in any table definition."""
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)

    # DuckDB's information_schema.referential_constraints shows FK relationships
    rows = conn.execute(
        "SELECT COUNT(*) FROM information_schema.referential_constraints"
    ).fetchone()
    assert rows is not None
    assert rows[0] == 0, "FK constraints found in schema"


def test_embedding_tables_separate_from_content() -> None:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)

    # message_state should NOT have embedding columns
    ms_cols = _table_columns(conn, "message_state")
    assert "content_embedding" not in ms_cols
    assert "thinking_embedding" not in ms_cols

    # tool_calls should NOT have bash_embedding
    tc_cols = _table_columns(conn, "tool_calls")
    assert "bash_embedding" not in tc_cols

    # Embedding tables should have the right columns
    me_cols = _table_columns(conn, "message_embeddings")
    assert "message_id" in me_cols
    assert "content_embedding" in me_cols
    assert "thinking_embedding" in me_cols

    tce_cols = _table_columns(conn, "tool_call_embeddings")
    assert "tool_call_id" in tce_cols
    assert "bash_embedding" in tce_cols


def test_recreate_embedding_tables() -> None:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn, embed_dim=384)

    # Insert some data
    conn.execute(
        "INSERT INTO message_embeddings (message_id) VALUES (?)",
        ["msg-1"],
    )
    conn.execute(
        "INSERT INTO tool_call_embeddings (tool_call_id) VALUES (?)",
        ["tc-1"],
    )
    conn.execute(
        """
        INSERT INTO embedding_cache (cache_key, kind, raw_text, normalized_text,
            embedding, normalization_version)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ["key-1", "content", "raw", "norm", [1.0] * 384, 1],
    )

    # Recreate with different dimension
    recreate_embedding_tables(conn, 256)

    # Old data should be gone
    assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM tool_call_embeddings").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone() == (0,)

    # New tables should have correct column types
    me_cols = _table_columns(conn, "message_embeddings")
    assert "content_embedding" in me_cols

    # Verify embedding_dimensions updated in runtime_state
    row = conn.execute("SELECT embedding_dimensions FROM runtime_state WHERE singleton").fetchone()
    assert row is not None
    assert row[0] == 256


def test_dimension_change_recreates_embedding_tables_automatically() -> None:
    """REQ-SCHEMA-005: dimension changes use recreate_embedding_tables path."""
    conn = duckdb.connect(":memory:")
    ensure_schema(conn, embed_dim=384)

    # Insert embedding data at 384 dims
    conn.execute(
        "INSERT INTO message_embeddings (message_id, content_embedding) VALUES (?, ?)",
        ["msg-1", [1.0] * 384],
    )
    assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (1,)

    # Re-ensure with different dims — should NOT raise, should recreate
    ensure_schema(conn, embed_dim=256)

    # Old data gone, tables recreated at new dimension
    assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (0,)
    row = conn.execute("SELECT embedding_dimensions FROM runtime_state WHERE singleton").fetchone()
    assert row is not None
    assert row[0] == 256


def test_sessions_and_messages_tables_keep_only_identity_columns() -> None:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)

    assert _table_columns(conn, "sessions") == [
        "id",
        "source",
        "source_path",
        "source_session_id",
    ]
    assert _table_columns(conn, "messages") == [
        "id",
        "session_id",
        "idx",
        "agent_id",
    ]


# (Old direct _relax test removed; superseded by new ensure-based framework tests below
# that exercise the full v13→v14 path via the migration runner and cover the same
# assertions plus schema_migrations table + very-old v11 behavior.)


# --- New framework tests (TDD: written failing first; will pass after migrations impl) ---


def test_fresh_ensure_creates_schema_migrations_table() -> None:
    """Fresh DBs get the schema_migrations table (REQ-MIG-003).

    Baseline has 0 rows (pending logic only runs for current < target).
    """
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)

    assert _table_exists(conn, "schema_migrations"), (
        "schema_migrations table must exist after fresh ensure"
    )
    row = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()
    count = row[0] if row is not None else 0
    assert count == 0, (
        "fresh DB should have 0 migration records (pending logic only runs for < target)"
    )


def test_ensure_on_v13_simulated_db_runs_migration_and_creates_migs_table() -> None:
    """v13 DB with old CHECKs auto-migrates via framework.

    Verifies: migs table created + records for replayed migrations + version bump
    to current schema + grok inserts allowed.
    """
    conn = duckdb.connect(":memory:")

    # Simulate v13 DB (same setup as old test, but using public ensure path)
    conn.execute(
        """
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("INSERT INTO schema_version (version) VALUES (13)")

    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL CHECK (source IN ('claude_code', 'codex', 'pi_agent')),
            source_path TEXT UNIQUE NOT NULL,
            source_session_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE message_state (
            message_id TEXT PRIMARY KEY,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT,
            thinking TEXT,
            timestamp TIMESTAMP,
            has_thinking BOOLEAN DEFAULT FALSE
        )
        """
    )
    conn.execute("INSERT INTO sessions VALUES ('s1', 'claude_code', '/tmp/s1', NULL)")
    conn.execute(
        "INSERT INTO message_state (message_id, role, content) VALUES ('m1', 'user', 'hello')"
    )

    # Minimal supporting tables so that after migration the ensure path can reach
    # _check_embedding_dimensions (runtime_state) and not explode on missing catalog objects.
    # (A real v13 DB would have had these from its original _apply_schema.)
    conn.execute(
        """
        CREATE TABLE runtime_state (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
            embedding_dimensions INTEGER
        )
        """
    )
    conn.execute("INSERT OR IGNORE INTO runtime_state (singleton) VALUES (TRUE)")

    conn.execute(
        """
        CREATE TABLE message_embeddings (
            message_id TEXT PRIMARY KEY,
            content_embedding FLOAT[384],
            thinking_embedding FLOAT[384]
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE tool_call_embeddings (
            tool_call_id TEXT PRIMARY KEY,
            bash_embedding FLOAT[384]
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE embedding_cache (
            cache_key TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('content', 'thinking', 'bash')),
            raw_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            embedding FLOAT[384] NOT NULL,
            normalization_version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Before: grok insert fails
    with pytest.raises(Exception, match="CHECK constraint failed"):
        conn.execute("INSERT INTO sessions VALUES ('s2', 'grok', '/tmp/s2', NULL)")

    # Call public ensure (triggers framework + 0014 migration)
    ensure_schema(conn)

    # Post-migration assertions
    from recall.db.schema import _get_schema_version

    assert _get_schema_version(conn) == SCHEMA_VERSION

    assert _table_exists(conn, "schema_migrations")
    migs = conn.execute("SELECT migration_id FROM schema_migrations").fetchall()
    migration_ids = {row[0] for row in migs}
    assert "0014_relax_open_enum_constraints" in migration_ids
    assert "0015_contextual_retrieval" in migration_ids

    # Now grok + system role work, old data preserved
    conn.execute("INSERT INTO sessions VALUES ('s2', 'grok', '/tmp/s2', 'grok-uuid-123')")
    row = conn.execute("SELECT COUNT(*) FROM sessions WHERE source = 'grok'").fetchone()
    assert (row[0] if row is not None else 0) == 1
    conn.execute(
        "INSERT INTO message_state (message_id, role, content) "
        "VALUES ('m2', 'system', 'tool output')"
    )
    row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    assert (row[0] if row is not None else 0) == 2
    row = conn.execute("SELECT COUNT(*) FROM message_state").fetchone()
    assert (row[0] if row is not None else 0) == 2


def test_v11_simulated_db_still_requires_recreate_after_mig_attempt() -> None:
    """v11 DB records 0014 but does not bump; still raises mismatch (recreate required)."""
    conn = duckdb.connect(":memory:")

    conn.execute(
        """
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("INSERT INTO schema_version (version) VALUES (11)")

    # Minimal v11-ish tables (no grok in CHECK)
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL CHECK (source IN ('claude_code', 'codex', 'pi_agent')),
            source_path TEXT UNIQUE NOT NULL,
            source_session_id TEXT
        )
        """
    )
    conn.execute("INSERT INTO sessions VALUES ('s1', 'claude_code', '/tmp/s1', NULL)")

    with pytest.raises(RuntimeError, match="schema version mismatch"):
        ensure_schema(conn)

    # Even after the attempt, version is still 11 (mig recorded but no bump for <13)
    from recall.db.schema import _get_schema_version

    assert _get_schema_version(conn) == 11
    # migs table was created by runner + 0014 recorded (idempotent attempt)
    assert _table_exists(conn, "schema_migrations")


# --- Regression tests for Critical 1 (no-op and partial failure paths) ---


def test_v13_with_pre_relaxed_checks_completes_to_current_schema() -> None:
    """Regression test for the no-op case.

    A v13 DB where the CHECK constraints are already absent (e.g. user had
    previously run the old _relax hack) must still advance cleanly
    and record the migration.
    """
    conn = duckdb.connect(":memory:")

    conn.execute(
        """
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("INSERT INTO schema_version (version) VALUES (13)")

    # Create tables WITHOUT the old tight CHECKs (already relaxed state)
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            source_path TEXT UNIQUE NOT NULL,
            source_session_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE message_state (
            message_id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT,
            thinking TEXT,
            timestamp TIMESTAMP,
            has_thinking BOOLEAN DEFAULT FALSE
        )
        """
    )
    conn.execute("INSERT INTO sessions VALUES ('s1', 'claude_code', '/tmp/s1', NULL)")

    # Minimal supporting tables so ensure_schema can complete
    conn.execute(
        """
        CREATE TABLE runtime_state (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
            embedding_dimensions INTEGER
        )
        """
    )
    conn.execute("INSERT OR IGNORE INTO runtime_state (singleton) VALUES (TRUE)")
    conn.execute(
        """
        CREATE TABLE message_embeddings (
            message_id TEXT PRIMARY KEY,
            content_embedding FLOAT[384],
            thinking_embedding FLOAT[384]
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE tool_call_embeddings (
            tool_call_id TEXT PRIMARY KEY,
            bash_embedding FLOAT[384]
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE embedding_cache (
            cache_key TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('content', 'thinking', 'bash')),
            raw_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            embedding FLOAT[384] NOT NULL,
            normalization_version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    ensure_schema(conn)

    from recall.db.schema import _get_schema_version

    assert _get_schema_version(conn) == SCHEMA_VERSION

    migs = conn.execute("SELECT migration_id FROM schema_migrations").fetchall()
    assert any(row[0] == "0014_relax_open_enum_constraints" for row in migs)


def test_v13_with_one_table_already_relaxed_completes_to_current_schema() -> None:
    """Regression test for the mixed state case.

    A v13 DB where one table already has a relaxed CHECK (no needs_*) and
    the other still has the old CHECK should still complete the migration
    successfully.
    """
    conn = duckdb.connect(":memory:")

    conn.execute(
        """
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("INSERT INTO schema_version (version) VALUES (13)")

    # sessions has old CHECK, message_state does not (to force partial path)
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL CHECK (source IN ('claude_code', 'codex', 'pi_agent')),
            source_path TEXT UNIQUE NOT NULL,
            source_session_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE message_state (
            message_id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT,
            thinking TEXT,
            timestamp TIMESTAMP,
            has_thinking BOOLEAN DEFAULT FALSE
        )
        """
    )
    conn.execute("INSERT INTO sessions VALUES ('s1', 'claude_code', '/tmp/s1', NULL)")

    # Minimal supporting tables
    conn.execute(
        """
        CREATE TABLE runtime_state (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
            embedding_dimensions INTEGER
        )
        """
    )
    conn.execute("INSERT OR IGNORE INTO runtime_state (singleton) VALUES (TRUE)")
    conn.execute(
        """
        CREATE TABLE message_embeddings (
            message_id TEXT PRIMARY KEY,
            content_embedding FLOAT[384],
            thinking_embedding FLOAT[384]
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE tool_call_embeddings (
            tool_call_id TEXT PRIMARY KEY,
            bash_embedding FLOAT[384]
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE embedding_cache (
            cache_key TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('content', 'thinking', 'bash')),
            raw_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            embedding FLOAT[384] NOT NULL,
            normalization_version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # We can't easily force a DDL failure in a unit test without mocking,
    # so we instead verify the contract: if the migration returns False,
    # the runner should not have recorded it.
    # For now we just ensure the v13 sim with mixed state still works on success path.
    # (A true partial failure test would require fault injection.)

    ensure_schema(conn)  # Should succeed in this setup

    from recall.db.schema import _get_schema_version

    assert _get_schema_version(conn) == SCHEMA_VERSION
    assert _table_exists(conn, "schema_migrations")


def test_partial_failure_during_migration_does_not_record_as_applied() -> None:
    """Real partial-failure regression test.

    If the migration successfully relaxes one table but fails on the second
    (simulated by pre-creating the _new table so CREATE TABLE fails), the
    migration must return False, the version must stay at 13, and 0014 must
    *not* be recorded in schema_migrations.
    """
    conn = duckdb.connect(":memory:")

    conn.execute(
        """
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("INSERT INTO schema_version (version) VALUES (13)")

    # Both tables have the old tight CHECKs
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL CHECK (source IN ('claude_code', 'codex', 'pi_agent')),
            source_path TEXT UNIQUE NOT NULL,
            source_session_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE message_state (
            message_id TEXT PRIMARY KEY,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT,
            thinking TEXT,
            timestamp TIMESTAMP,
            has_thinking BOOLEAN DEFAULT FALSE
        )
        """
    )
    conn.execute("INSERT INTO sessions VALUES ('s1', 'claude_code', '/tmp/s1', NULL)")

    # Pre-create message_state_new so the migration's CREATE TABLE will fail.
    # This simulates a real DDL failure on the second table after the first
    # one succeeded.
    conn.execute(
        """
        CREATE TABLE message_state_new (
            message_id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT,
            thinking TEXT,
            timestamp TIMESTAMP,
            has_thinking BOOLEAN DEFAULT FALSE
        )
        """
    )

    # Minimal supporting tables for ensure_schema to reach the migration
    conn.execute(
        """
        CREATE TABLE runtime_state (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
            embedding_dimensions INTEGER
        )
        """
    )
    conn.execute("INSERT OR IGNORE INTO runtime_state (singleton) VALUES (TRUE)")

    # Run the migration via the runner
    from recall.db.migrations import run_pending_migrations
    from recall.db.schema import _get_schema_version

    run_pending_migrations(conn, 13)

    # After partial failure:
    # - Version should still be 13
    # - 0014 should NOT be recorded
    assert _get_schema_version(conn) == 13
    migs = conn.execute("SELECT migration_id FROM schema_migrations").fetchall()
    assert not any("0014" in (row[0] or "") for row in migs)


def test_failed_0014_halts_before_later_migrations() -> None:
    """A failed prerequisite migration must stop the current migration pass."""
    conn = _create_v13_db_with_blocked_0014()

    with pytest.raises(RuntimeError, match="schema version mismatch"):
        ensure_schema(conn)

    migration_ids = {
        row[0] for row in conn.execute("SELECT migration_id FROM schema_migrations").fetchall()
    }
    assert "0014_relax_open_enum_constraints" not in migration_ids
    assert "0015_contextual_retrieval" not in migration_ids


def test_failed_0014_retry_applies_0014_and_0015_after_recovery() -> None:
    """A later ensure_schema pass must retry both the failed migration and successors."""
    conn = _create_v13_db_with_blocked_0014()

    with pytest.raises(RuntimeError, match="schema version mismatch"):
        ensure_schema(conn)

    conn.execute("DROP TABLE message_state_new")
    ensure_schema(conn)

    from recall.db.schema import _get_schema_version

    migration_ids = {
        row[0] for row in conn.execute("SELECT migration_id FROM schema_migrations").fetchall()
    }
    assert "0014_relax_open_enum_constraints" in migration_ids
    assert "0015_contextual_retrieval" in migration_ids
    assert _get_schema_version(conn) == SCHEMA_VERSION
    assert "context_version" in _table_columns(conn, "embedding_cache")


def test_fresh_schema_omits_retired_catalog_index_and_column() -> None:
    """REQ-MIG-010: a fresh DB carries neither retired catalog object."""
    conn = duckdb.connect(":memory:")
    ensure_schema(conn, embed_dim=8)

    assert SCHEMA_VERSION == 31
    assert "inventory_generation" not in _table_columns(conn, "source_files")
    indexes = {
        str(row[0])
        for row in conn.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'source_files'"
        ).fetchall()
    }
    assert "idx_source_files_pending" not in indexes
    assert "idx_source_files_source_path" in indexes


def test_fresh_catalog_still_serves_pending_work_without_the_retired_index() -> None:
    """The pending scan is a filter over column comparisons, not an index lookup."""
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    ensure_schema(conn, embed_dim=8)
    catalog = SourceCatalog(conn, clock=lambda: 100.0)
    signature = SourceSignature(dev=1, inode=2, ctime_ns=3, mtime_ns=4, size=5)
    catalog.observe("codex", "/root", "/root/a.jsonl", signature)

    assert [row.source_path for row in catalog.pending(limit=8)] == ["/root/a.jsonl"]
    assert catalog.acknowledge("codex", "/root/a.jsonl", 1, 5, "a" * 64)
    assert catalog.pending(limit=8) == []
