from __future__ import annotations

import hashlib
from collections.abc import Iterator

import duckdb
from recall.core.types import EmbedKind
from recall.db.migrations import get_migrations, run_pending_migrations
from recall.db.schema import (
    SCHEMA_VERSION,
    ensure_schema,
    get_schema_version,
    recreate_embedding_tables,
)
from recall.services.embeddings import (
    NORMALIZATION_VERSION,
    _prepare_text,
    normalize_bash_command,
)


def _connect_memory_db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect(":memory:")
    try:
        yield conn
    finally:
        conn.close()


def _create_v14_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE schema_migrations (
            migration_id TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    conn.execute(
        """
        CREATE TABLE embedding_cache (
            cache_key TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('content', 'thinking', 'bash')),
            raw_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            embedding FLOAT[3] NOT NULL,
            normalization_version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE runtime_state (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
            last_attempted_at TIMESTAMP,
            last_successful_at TIMESTAMP,
            last_run_kind TEXT CHECK (
                last_run_kind IS NULL
                OR last_run_kind IN (
                    'index', 'embed', 'daemon-once', 'daemon-scheduled', 'daemon-watch'
                )
            ),
            last_index_total INTEGER,
            last_index_indexed INTEGER,
            last_index_skipped INTEGER,
            last_index_failed INTEGER,
            last_index_changed INTEGER,
            last_index_total_seconds DOUBLE,
            last_embed_messages INTEGER,
            last_embed_thinking INTEGER,
            last_embed_bash INTEGER,
            last_failure_message TEXT,
            last_failure_at TIMESTAMP,
            installed_scheduler TEXT CHECK (
                installed_scheduler IS NULL
                OR installed_scheduler IN ('launchd', 'systemd', 'cron')
            ),
            embedding_dimensions INTEGER
        )
        """
    )
    conn.execute("INSERT INTO schema_version (version) VALUES (14)")
    conn.execute("INSERT INTO runtime_state (singleton, embedding_dimensions) VALUES (TRUE, 3)")
    conn.execute(
        """
        INSERT INTO message_state (
            message_id, role, content, thinking, timestamp, has_thinking
        ) VALUES ('m1', 'assistant', 'content', 'thinking', NULL, TRUE)
        """
    )
    conn.execute(
        """
        INSERT INTO embedding_cache (
            cache_key, kind, raw_text, normalized_text, embedding, normalization_version
        ) VALUES ('k1', 'bash', 'echo 12345', 'echo <num>', [0.1, 0.2, 0.3], 1)
        """
    )


def _create_v11_schema_with_v15_prereq_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the v11 shape needed to verify 0015 refuses unsafe replay."""
    conn.execute(
        """
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    conn.execute(
        """
        CREATE TABLE embedding_cache (
            cache_key TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('content', 'thinking', 'bash')),
            raw_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            embedding FLOAT[3] NOT NULL,
            normalization_version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("INSERT INTO schema_version (version) VALUES (11)")


def _column_info(
    conn: duckdb.DuckDBPyConnection, table_name: str
) -> dict[str, tuple[str, bool, str]]:
    rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    return {str(row[1]): (str(row[2]).upper(), bool(row[3]), str(row[4])) for row in rows}


def _assert_contextual_retrieval_columns(conn: duckdb.DuckDBPyConnection) -> None:
    message_columns = _column_info(conn, "message_state")
    assert message_columns["context_text"][0] == "VARCHAR"
    assert message_columns["context_mode"][0] == "VARCHAR"
    assert "''" in message_columns["context_text"][2]
    assert "'off'" in message_columns["context_mode"][2]
    assert message_columns["fts_content"][0] == "VARCHAR"
    assert message_columns["fts_thinking"][0] == "VARCHAR"
    assert "''" in message_columns["fts_content"][2]
    assert "''" in message_columns["fts_thinking"][2]

    cache_columns = _column_info(conn, "embedding_cache")
    assert cache_columns["context_version"][0] == "INTEGER"
    assert cache_columns["context_version"][1] is True
    assert cache_columns["context_version"][2] == "0"

    runtime_columns = _column_info(conn, "runtime_state")
    assert runtime_columns["last_context_messages"][0] == "INTEGER"
    assert runtime_columns["last_context_mode"][0] == "VARCHAR"
    assert runtime_columns["last_context_input_tokens"][0] == "BIGINT"
    assert runtime_columns["last_context_output_tokens"][0] == "BIGINT"
    assert runtime_columns["last_context_model"][0] == "VARCHAR"


def _has_context_mode_check(conn: duckdb.DuckDBPyConnection) -> bool:
    rows = conn.execute(
        """
        SELECT expression FROM duckdb_constraints()
        WHERE table_name = 'message_state' AND constraint_type = 'CHECK'
        """
    ).fetchall()
    return any(
        "context_mode" in str(row[0])
        and all(mode in str(row[0]) for mode in ("off", "template", "llm-local", "llm-remote"))
        for row in rows
    )


def test_v14_to_v15_contextual_retrieval_migration_is_idempotent() -> None:
    for conn in _connect_memory_db():
        _create_v14_schema(conn)

        ensure_schema(conn, embed_dim=3)

        assert get_schema_version(conn) == SCHEMA_VERSION
        assert SCHEMA_VERSION >= 18
        _assert_contextual_retrieval_columns(conn)
        assert conn.execute(
            """
            SELECT context_text, context_mode, fts_content, fts_thinking
            FROM message_state WHERE message_id = 'm1'
            """
        ).fetchone() == ("", "off", "content", "thinking")
        assert conn.execute(
            "SELECT context_version FROM embedding_cache WHERE cache_key = 'k1'"
        ).fetchone() == (0,)
        assert conn.execute(
            """
            SELECT
                last_context_messages,
                last_context_mode,
                last_context_input_tokens,
                last_context_output_tokens,
                last_context_model
            FROM runtime_state
            WHERE singleton
            """
        ).fetchone() == (None, None, None, None, None)

        migration = next(m for m in get_migrations() if m.id == "0015_contextual_retrieval")
        assert migration.upgrade(conn) is True
        _assert_contextual_retrieval_columns(conn)

        conn.execute(
            """
            INSERT INTO message_state (message_id, role)
            VALUES ('m2', 'user')
            """
        )
        assert conn.execute(
            "SELECT context_text, context_mode FROM message_state WHERE message_id = 'm2'"
        ).fetchone() == ("", "off")

        try:
            conn.execute(
                """
                INSERT INTO message_state (message_id, role, context_mode)
                VALUES ('bad-mode', 'user', 'invalid')
                """
            )
        except duckdb.ConstraintException:
            pass
        else:
            raise AssertionError("message_state.context_mode accepted an invalid mode")


def test_v15_migration_repairs_partial_embedding_cache_context_version() -> None:
    for conn in _connect_memory_db():
        _create_v14_schema(conn)
        conn.execute("ALTER TABLE embedding_cache ADD COLUMN context_version INTEGER DEFAULT 0")

        assert _column_info(conn, "embedding_cache")["context_version"][1] is False

        ensure_schema(conn, embed_dim=3)

        cache_columns = _column_info(conn, "embedding_cache")
        assert get_schema_version(conn) == SCHEMA_VERSION
        assert cache_columns["context_version"] == ("INTEGER", True, "0")
        assert conn.execute(
            "SELECT context_version FROM embedding_cache WHERE cache_key = 'k1'"
        ).fetchone() == (0,)


def test_v15_migration_repairs_context_version_without_default() -> None:
    for conn in _connect_memory_db():
        _create_v14_schema(conn)
        conn.execute("ALTER TABLE embedding_cache ADD COLUMN context_version INTEGER")
        conn.execute("UPDATE embedding_cache SET context_version = 0")
        conn.execute("ALTER TABLE embedding_cache ALTER COLUMN context_version SET NOT NULL")

        assert _column_info(conn, "embedding_cache")["context_version"] == (
            "INTEGER",
            True,
            "None",
        )

        ensure_schema(conn, embed_dim=3)

        cache_columns = _column_info(conn, "embedding_cache")
        assert get_schema_version(conn) == SCHEMA_VERSION
        assert cache_columns["context_version"] == ("INTEGER", True, "0")
        conn.execute(
            """
            INSERT INTO embedding_cache (
                cache_key, kind, raw_text, normalized_text, embedding
            ) VALUES ('defaulted', 'bash', 'pwd', 'pwd', [0.0, 0.0, 0.0])
            """
        )
        assert conn.execute(
            "SELECT context_version FROM embedding_cache WHERE cache_key = 'defaulted'"
        ).fetchone() == (0,)


def test_v15_migration_repairs_partial_message_state_context_columns() -> None:
    for conn in _connect_memory_db():
        _create_v14_schema(conn)
        conn.execute("ALTER TABLE message_state ADD COLUMN context_text TEXT")
        conn.execute("ALTER TABLE message_state ADD COLUMN context_mode TEXT DEFAULT 'off'")

        assert _has_context_mode_check(conn) is False

        ensure_schema(conn, embed_dim=3)

        assert get_schema_version(conn) == SCHEMA_VERSION
        _assert_contextual_retrieval_columns(conn)
        assert _has_context_mode_check(conn) is True
        assert conn.execute(
            """
            SELECT context_text, context_mode, fts_content, fts_thinking
            FROM message_state WHERE message_id = 'm1'
            """
        ).fetchone() == ("", "off", "content", "thinking")

        try:
            conn.execute(
                """
                INSERT INTO message_state (message_id, role, context_mode)
                VALUES ('bad-partial-mode', 'user', 'invalid')
                """
            )
        except duckdb.ConstraintException:
            pass
        else:
            raise AssertionError("repaired message_state.context_mode accepted an invalid mode")


def test_v16_migration_repairs_partial_message_state_fts_columns() -> None:
    for conn in _connect_memory_db():
        _create_v14_schema(conn)
        migration_0015 = next(m for m in get_migrations() if m.id == "0015_contextual_retrieval")
        assert migration_0015.upgrade(conn) is True
        conn.execute("ALTER TABLE message_state ADD COLUMN fts_content INTEGER DEFAULT 0")

        ensure_schema(conn, embed_dim=3)

        assert get_schema_version(conn) == SCHEMA_VERSION
        _assert_contextual_retrieval_columns(conn)
        assert conn.execute(
            """
            SELECT fts_content, fts_thinking
            FROM message_state WHERE message_id = 'm1'
            """
        ).fetchone() == ("content", "thinking")


def test_failed_v15_migration_stays_pending_and_retries() -> None:
    for conn in _connect_memory_db():
        _create_v14_schema(conn)
        conn.execute("DROP TABLE embedding_cache")

        try:
            ensure_schema(conn, embed_dim=3)
        except RuntimeError as err:
            assert "schema version mismatch" in str(err)
        else:
            raise AssertionError("ensure_schema succeeded despite incomplete v15 migration")

        assert get_schema_version(conn) == 14
        assert conn.execute(
            """
            SELECT COUNT(*) FROM schema_migrations
            WHERE migration_id = '0015_contextual_retrieval'
            """
        ).fetchone() == (0,)

        conn.execute(
            """
            CREATE TABLE embedding_cache (
                cache_key TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK (kind IN ('content', 'thinking', 'bash')),
                raw_text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                embedding FLOAT[3] NOT NULL,
                normalization_version INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        ensure_schema(conn, embed_dim=3)

        assert get_schema_version(conn) == SCHEMA_VERSION
        assert conn.execute(
            """
            SELECT COUNT(*) FROM schema_migrations
            WHERE migration_id = '0015_contextual_retrieval'
            """
        ).fetchone() == (1,)
        _assert_contextual_retrieval_columns(conn)


def test_v15_migration_halts_without_recording_on_pre_v14_schema() -> None:
    for conn in _connect_memory_db():
        _create_v11_schema_with_v15_prereq_tables(conn)

        run_pending_migrations(conn, current_version=11)

        assert get_schema_version(conn) == 11
        assert conn.execute(
            """
            SELECT COUNT(*) FROM schema_migrations
            WHERE migration_id = '0015_contextual_retrieval'
            """
        ).fetchone() == (0,)
        assert "context_version" not in _column_info(conn, "embedding_cache")

        try:
            ensure_schema(conn, embed_dim=3)
        except RuntimeError as err:
            assert "schema version mismatch" in str(err)
            assert "recall index --recreate --yes" in str(err)
        else:
            raise AssertionError("ensure_schema succeeded despite pre-v14 schema")


def test_fresh_schema_and_recreated_embedding_cache_have_v15_shape() -> None:
    for conn in _connect_memory_db():
        ensure_schema(conn, embed_dim=3)

        assert get_schema_version(conn) == SCHEMA_VERSION
        _assert_contextual_retrieval_columns(conn)

        recreate_embedding_tables(conn, embed_dim=5)

        cache_columns = _column_info(conn, "embedding_cache")
        assert cache_columns["context_version"] == ("INTEGER", True, "0")
        conn.execute(
            """
            INSERT INTO embedding_cache (
                cache_key, kind, raw_text, normalized_text, embedding
            ) VALUES ('fresh', 'content', 'raw', 'raw', [0, 0, 0, 0, 0])
            """
        )
        assert conn.execute(
            "SELECT context_version FROM embedding_cache WHERE cache_key = 'fresh'"
        ).fetchone() == (0,)


def test_v17_migration_widens_context_mode_check_to_include_llm_codex() -> None:
    """0017 must let new rows store context_mode='llm-codex' without losing old data."""
    for conn in _connect_memory_db():
        _create_v14_schema(conn)

        ensure_schema(conn, embed_dim=3)

        assert get_schema_version(conn) == SCHEMA_VERSION
        # Original row from _create_v14_schema must survive the table rebuild.
        assert conn.execute(
            "SELECT context_text, context_mode FROM message_state WHERE message_id = 'm1'"
        ).fetchone() == ("", "off")

        # The new mode must now be accepted.
        conn.execute(
            """
            INSERT INTO message_state (message_id, role, context_mode)
            VALUES ('codex-row', 'assistant', 'llm-codex')
            """
        )
        assert conn.execute(
            "SELECT context_mode FROM message_state WHERE message_id = 'codex-row'"
        ).fetchone() == ("llm-codex",)

        # And the CHECK must still reject completely-unknown modes.
        try:
            conn.execute(
                """
                INSERT INTO message_state (message_id, role, context_mode)
                VALUES ('bad', 'user', 'fictional-mode')
                """
            )
        except duckdb.ConstraintException:
            pass
        else:
            raise AssertionError("widened CHECK accepted an unknown mode")


def test_v17_migration_is_idempotent_on_already_widened_check() -> None:
    """Re-running 0017 against a DB that already includes 'llm-codex' must no-op cleanly."""
    for conn in _connect_memory_db():
        _create_v14_schema(conn)
        ensure_schema(conn, embed_dim=3)
        # Drop the recorded migration row so the runner replays 0017.
        conn.execute("DELETE FROM schema_migrations WHERE migration_id = '0017_add_llm_codex_mode'")

        migration = next(m for m in get_migrations() if m.id == "0017_add_llm_codex_mode")
        assert migration.upgrade(conn) is True

        # Row count and CHECK semantics unchanged.
        rows = conn.execute(
            "SELECT context_mode FROM message_state WHERE message_id = 'm1'"
        ).fetchone()
        assert rows == ("", "off") or rows == ("off",)


def test_bash_embedding_cache_key_is_unchanged_by_v15_migration() -> None:
    cache_namespace = "tests.FixedBackend:BAAI/bge-small-en-v1.5"
    raw_command = "kubectl logs api-abcdef123-abcde --since=123456"

    before = _prepare_text(EmbedKind.BASH, raw_command, cache_namespace)

    for conn in _connect_memory_db():
        _create_v14_schema(conn)
        ensure_schema(conn, embed_dim=3)

    after = _prepare_text(EmbedKind.BASH, raw_command, cache_namespace)
    expected_source = normalize_bash_command(raw_command)
    expected_key = hashlib.sha256(
        f"{cache_namespace}:bash:{NORMALIZATION_VERSION}:{expected_source}".encode()
    ).hexdigest()

    assert NORMALIZATION_VERSION == 1
    assert before.normalized_text == after.normalized_text == expected_source
    assert before.cache_key == after.cache_key == expected_key
