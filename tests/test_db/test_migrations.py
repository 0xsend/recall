from __future__ import annotations

from typing import cast

import duckdb


def test_migration_0018_creates_pending_table(tmp_path) -> None:
    conn = _v17_db(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        run_pending_migrations(conn, 17)

        # Pending chain from 17 includes 0018 and later migrations (0019+).
        assert _get_schema_version(conn) == SCHEMA_VERSION
        assert _table_columns(conn, "fts_sidecar_pending") == {
            "kind": "VARCHAR",
            "id": "VARCHAR",
            "op": "VARCHAR",
            "queued_at": "TIMESTAMP",
        }
        assert "idx_fts_sidecar_pending_kind_id" in _table_indexes(
            conn,
            "fts_sidecar_pending",
        )
    finally:
        conn.close()


def test_migration_0018_is_idempotent(tmp_path) -> None:
    conn = _v17_db(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        run_pending_migrations(conn, 17)
        run_pending_migrations(conn, 17)

        assert _get_schema_version(conn) == SCHEMA_VERSION
        assert conn.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone() == (0,)
    finally:
        conn.close()


def test_migration_0019_usage_ledger(tmp_path) -> None:
    conn = _v18_db_with_session_state(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        run_pending_migrations(conn, 18)

        assert _get_schema_version(conn) == SCHEMA_VERSION
        cols = _table_columns(conn, "usage_events")
        assert "id" in cols
        assert "source_session_id" in cols
        assert "cached_prompt_tokens" in cols
        assert _table_columns(conn, "usage_log_cursors")
        ss = _table_columns(conn, "session_state")
        assert "cached_input_tokens" in ss
        assert "host" in ss
    finally:
        conn.close()


def test_migration_0020_adds_sidecar_mtime(tmp_path) -> None:
    """The column must land NULL, not 0, so legacy rows re-index once."""
    conn = _v18_db_with_session_state(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        run_pending_migrations(conn, 18)

        assert _get_schema_version(conn) == SCHEMA_VERSION
        assert SCHEMA_VERSION == 31
        assert "sidecar_mtime" in _table_columns(conn, "session_state")
    finally:
        conn.close()


def test_migration_0020_is_idempotent(tmp_path) -> None:
    conn = _v18_db_with_session_state(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        run_pending_migrations(conn, 18)
        run_pending_migrations(conn, _get_schema_version(conn))

        assert _get_schema_version(conn) == SCHEMA_VERSION
        columns = [c for c in _table_columns(conn, "session_state") if c == "sidecar_mtime"]
        assert len(columns) == 1
    finally:
        conn.close()


def test_migration_0019_is_idempotent(tmp_path) -> None:
    conn = _v18_db_with_session_state(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        run_pending_migrations(conn, 18)
        run_pending_migrations(conn, 18)
        assert _get_schema_version(conn) == SCHEMA_VERSION
    finally:
        conn.close()


def test_migration_0021_widens_token_counters_preserves_values_and_indexes(tmp_path) -> None:
    conn = _v20_db_with_token_counters(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import _get_schema_version

        expected_values = {
            "session_state": {
                "input_tokens": 2_146_844_021,
                "output_tokens": 2_100_000_001,
                "cached_input_tokens": 2_000_000_002,
            },
            "usage_events": {
                "prompt_tokens": 2_146_844_003,
                "cached_prompt_tokens": 2_146_844_004,
                "completion_tokens": 2_146_844_005,
                "reasoning_tokens": 2_146_844_006,
            },
            "runtime_state": {
                "last_context_input_tokens": 2_146_844_007,
                "last_context_output_tokens": 2_146_844_008,
            },
        }
        before_indexes = _table_index_sql(conn, "session_state") | _table_index_sql(
            conn, "usage_events"
        )

        run_pending_migrations(conn, 20)

        assert _get_schema_version(conn) == 21
        target_columns = {
            "session_state": ("input_tokens", "output_tokens", "cached_input_tokens"),
            "usage_events": (
                "prompt_tokens",
                "cached_prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
            ),
            "runtime_state": ("last_context_input_tokens", "last_context_output_tokens"),
        }
        for table, columns in target_columns.items():
            actual_columns = _table_columns(conn, table)
            assert all(actual_columns[column] == "BIGINT" for column in columns)
        assert conn.execute(
            "SELECT input_tokens, output_tokens, cached_input_tokens FROM session_state"
        ).fetchone() == tuple(expected_values["session_state"].values())
        assert conn.execute(
            "SELECT prompt_tokens, cached_prompt_tokens, completion_tokens, reasoning_tokens "
            "FROM usage_events"
        ).fetchone() == tuple(expected_values["usage_events"].values())
        assert conn.execute(
            "SELECT last_context_input_tokens, last_context_output_tokens FROM runtime_state"
        ).fetchone() == tuple(expected_values["runtime_state"].values())
        assert (
            _table_index_sql(conn, "session_state") | _table_index_sql(conn, "usage_events")
            == before_indexes
        )
    finally:
        conn.close()


def test_migration_0021_is_idempotent(tmp_path) -> None:
    conn = _v20_db_with_token_counters(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import _get_schema_version

        run_pending_migrations(conn, 20)
        values = conn.execute("SELECT input_tokens FROM session_state").fetchone()
        indexes = _table_index_sql(conn, "session_state") | _table_index_sql(conn, "usage_events")

        run_pending_migrations(conn, _get_schema_version(conn))

        assert _get_schema_version(conn) == 21
        assert conn.execute("SELECT input_tokens FROM session_state").fetchone() == values
        assert (
            _table_index_sql(conn, "session_state") | _table_index_sql(conn, "usage_events")
            == indexes
        )
    finally:
        conn.close()


def test_migration_0021_restores_indexes_and_version_on_failure(tmp_path, monkeypatch) -> None:
    conn = _v20_db_with_token_counters(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import _get_schema_version

        before_indexes = _table_index_sql(conn, "session_state") | _table_index_sql(
            conn, "usage_events"
        )

        def fail_set_schema_version_to(_conn, _version) -> None:
            raise RuntimeError("forced migration failure")

        monkeypatch.setattr("recall.db.schema.set_schema_version_to", fail_set_schema_version_to)
        run_pending_migrations(conn, 20)

        assert _get_schema_version(conn) == 20
        assert _table_columns(conn, "session_state")["input_tokens"] == "INTEGER"
        assert (
            _table_index_sql(conn, "session_state") | _table_index_sql(conn, "usage_events")
            == before_indexes
        )
        assert (
            conn.execute(
                "SELECT migration_id FROM schema_migrations "
                "WHERE migration_id = '0021_widen_token_counters'"
            ).fetchone()
            is None
        )
    finally:
        conn.close()


def test_migration_0021_allows_cumulative_counter_past_int32(tmp_path) -> None:
    conn = _v20_db_with_token_counters(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations

        run_pending_migrations(conn, 20)

        conn.execute(
            "UPDATE session_state SET input_tokens = input_tokens + ?",
            [1_721_435_718],
        )

        assert conn.execute("SELECT input_tokens FROM session_state").fetchone() == (3_868_279_739,)
    finally:
        conn.close()


def _v17_db(tmp_path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "recall.duckdb"))
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
    conn.execute("INSERT INTO schema_version (version) VALUES (17)")
    return conn


def _v18_db_with_session_state(tmp_path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "recall19.duckdb"))
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
    conn.execute("INSERT INTO schema_version (version) VALUES (18)")
    # Minimal session_state so ALTER ADD COLUMN can run.
    conn.execute(
        """
        CREATE TABLE session_state (
            session_id TEXT PRIMARY KEY,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            duration_seconds INTEGER,
            model TEXT,
            cwd TEXT,
            git_repo TEXT,
            git_branch TEXT,
            message_count INTEGER DEFAULT 0,
            tool_count INTEGER DEFAULT 0,
            input_tokens INTEGER,
            output_tokens INTEGER,
            is_complete BOOLEAN DEFAULT TRUE,
            file_mtime DOUBLE NOT NULL,
            file_size BIGINT NOT NULL,
            last_byte_offset BIGINT DEFAULT 0,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn


def test_migration_0022_backfills_unattributed_host(tmp_path, monkeypatch) -> None:
    """The sentinel is replaced by this machine's short hostname; rows that already
    carry a real label -- including one that merely looks like the sentinel --
    are untouched (REQ-MIG-008)."""
    monkeypatch.setattr("recall.core.types.default_session_host", lambda: "devbox")
    conn = _v21_db_with_unattributed_hosts(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        run_pending_migrations(conn, 21)

        assert _get_schema_version(conn) == SCHEMA_VERSION
        hosts = dict(conn.execute("SELECT session_id, host FROM session_state").fetchall())
        assert hosts == {
            "s-unattributed-1": "devbox",
            "s-unattributed-2": "devbox",
            "s-already-named": "WorkstationOne",
            "s-localhost": "localhost",
        }
    finally:
        conn.close()


def test_migration_0022_records_undo_pre_image(tmp_path, monkeypatch) -> None:
    """Every mutated row's prior value is recorded before the UPDATE, so the change
    is reversible without a whole-database copy (REQ-MIG-008)."""
    monkeypatch.setattr("recall.core.types.default_session_host", lambda: "devbox")
    conn = _v21_db_with_unattributed_hosts(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations

        run_pending_migrations(conn, 21)

        undo = conn.execute(
            """
            SELECT table_name, row_key, column_name, old_value
            FROM schema_migration_undo
            WHERE migration_id = '0022_backfill_session_host'
            ORDER BY row_key
            """
        ).fetchall()
        assert undo == [
            ("session_state", "s-unattributed-1", "host", "local"),
            ("session_state", "s-unattributed-2", "host", "local"),
        ]

        # The recorded pre-image is sufficient to reverse the migration.
        conn.execute(
            """
            UPDATE session_state SET host = u.old_value
            FROM schema_migration_undo u
            WHERE u.migration_id = '0022_backfill_session_host'
              AND u.table_name = 'session_state'
              AND u.row_key = session_state.session_id
            """
        )
        restored = dict(conn.execute("SELECT session_id, host FROM session_state").fetchall())
        assert restored["s-unattributed-1"] == "local"
        assert restored["s-already-named"] == "WorkstationOne"
    finally:
        conn.close()


def test_migration_0022_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("recall.core.types.default_session_host", lambda: "devbox")
    conn = _v21_db_with_unattributed_hosts(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        run_pending_migrations(conn, 21)
        hosts = dict(conn.execute("SELECT session_id, host FROM session_state").fetchall())
        undo_count = conn.execute("SELECT COUNT(*) FROM schema_migration_undo").fetchone()

        run_pending_migrations(conn, _get_schema_version(conn))

        assert _get_schema_version(conn) == SCHEMA_VERSION
        assert dict(conn.execute("SELECT session_id, host FROM session_state").fetchall()) == hosts
        assert conn.execute("SELECT COUNT(*) FROM schema_migration_undo").fetchone() == undo_count
    finally:
        conn.close()


def test_migration_0022_no_op_when_hostname_unobtainable(tmp_path, monkeypatch) -> None:
    """If this machine cannot name itself there is nothing better than the sentinel
    to write, so the migration advances the version without touching data."""
    monkeypatch.setattr("recall.core.types.default_session_host", lambda: "local")
    conn = _v21_db_with_unattributed_hosts(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        run_pending_migrations(conn, 21)

        assert _get_schema_version(conn) == SCHEMA_VERSION
        hosts = dict(conn.execute("SELECT session_id, host FROM session_state").fetchall())
        assert hosts["s-unattributed-1"] == "local"
        assert conn.execute("SELECT COUNT(*) FROM schema_migration_undo").fetchone() == (0,)
    finally:
        conn.close()


def _v20_db_with_token_counters(tmp_path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "recall20_tokens.duckdb"))
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
    conn.execute("INSERT INTO schema_version (version) VALUES (20)")
    conn.execute(
        """
        CREATE TABLE session_state (
            session_id TEXT PRIMARY KEY,
            cwd TEXT,
            git_repo TEXT,
            started_at TIMESTAMP,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cached_input_tokens INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE usage_events (
            id TEXT PRIMARY KEY,
            prompt_tokens INTEGER,
            cached_prompt_tokens INTEGER,
            completion_tokens INTEGER,
            reasoning_tokens INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE runtime_state (
            singleton BOOLEAN PRIMARY KEY,
            last_context_input_tokens INTEGER,
            last_context_output_tokens INTEGER
        )
        """
    )
    conn.execute(
        """
        INSERT INTO session_state (
            session_id, cwd, git_repo, started_at,
            input_tokens, output_tokens, cached_input_tokens
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "session-1",
            "/work",
            "repo-1",
            "2026-08-03 00:00:00",
            2_146_844_021,
            2_100_000_001,
            2_000_000_002,
        ],
    )
    conn.execute(
        """
        INSERT INTO usage_events VALUES (?, ?, ?, ?, ?)
        """,
        [
            "event-1",
            2_146_844_003,
            2_146_844_004,
            2_146_844_005,
            2_146_844_006,
        ],
    )
    conn.execute(
        """
        INSERT INTO runtime_state VALUES (TRUE, ?, ?)
        """,
        [2_146_844_007, 2_146_844_008],
    )
    conn.execute("CREATE INDEX idx_session_state_cwd ON session_state(cwd)")
    conn.execute("CREATE INDEX idx_session_state_git_repo ON session_state(git_repo)")
    conn.execute("CREATE INDEX idx_session_state_started ON session_state(started_at DESC)")
    conn.execute("CREATE INDEX idx_usage_events_source_sid ON usage_events(prompt_tokens)")
    conn.execute("CREATE INDEX idx_usage_events_session ON usage_events(cached_prompt_tokens)")
    conn.execute("CREATE INDEX idx_usage_events_ts ON usage_events(completion_tokens)")
    return conn


def _table_columns(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
) -> dict[str, str]:
    return {
        str(row[1]): str(row[2]).upper()
        for row in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    }


def _table_indexes(conn: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            """
            SELECT index_name
            FROM duckdb_indexes()
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchall()
    }


def _table_index_sql(conn: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, str]:
    return {
        str(row[0]): str(row[1])
        for row in conn.execute(
            """
            SELECT index_name, sql
            FROM duckdb_indexes()
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchall()
    }


def _v21_db_with_unattributed_hosts(tmp_path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "recall21_hosts.duckdb"))
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
    conn.execute("INSERT INTO schema_version (version) VALUES (21)")
    conn.execute(
        """
        CREATE TABLE session_state (
            session_id TEXT PRIMARY KEY,
            cwd TEXT,
            host TEXT NOT NULL DEFAULT 'local',
            file_mtime DOUBLE NOT NULL DEFAULT 0,
            file_size BIGINT NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        INSERT INTO session_state (session_id, cwd, host) VALUES
            ('s-unattributed-1', '/home/dev/a', 'local'),
            ('s-unattributed-2', '/home/dev/b', 'local'),
            ('s-already-named',  '/home/dev/c', 'WorkstationOne'),
            ('s-localhost',      '/home/dev/d', 'localhost')
        """
    )
    return conn


# --- 0023 + 0029: failure-signature memory columns on runtime_state (REQ-RESIL-014..019) ---

_FATAL_MEMORY_COLUMNS = {
    "last_fatal_signature": "VARCHAR",
    "fatal_repeat_count": "INTEGER",
    "last_index_repair_at": "TIMESTAMP",
    "last_index_repair_signature": "VARCHAR",
    "needs_index_verification": "BOOLEAN",
    # 0029: the signature without it reads as live however old it is.
    "last_fatal_at": "TIMESTAMP",
}


def _v22_db_with_runtime_state(tmp_path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "recall22.duckdb"))
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
    conn.execute("INSERT INTO schema_version (version) VALUES (22)")
    conn.execute(
        """
        CREATE TABLE runtime_state (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
            last_failure_message TEXT,
            last_failure_at TIMESTAMP
        )
        """
    )
    conn.execute(
        "INSERT INTO runtime_state (singleton, last_failure_message) VALUES (TRUE, 'kept')"
    )
    return conn


def test_migration_0023_adds_fatal_memory_columns_with_defaults(tmp_path) -> None:
    conn = _v22_db_with_runtime_state(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        run_pending_migrations(conn, 22)

        assert _get_schema_version(conn) == SCHEMA_VERSION
        columns = _table_columns(conn, "runtime_state")
        for name, sql_type in _FATAL_MEMORY_COLUMNS.items():
            assert columns[name] == sql_type, name
        row = conn.execute(
            """
            SELECT last_failure_message, fatal_repeat_count, needs_index_verification,
                   last_fatal_signature
            FROM runtime_state
            """
        ).fetchone()
        assert row == ("kept", 0, False, None)
    finally:
        conn.close()


def test_migration_0023_is_idempotent(tmp_path) -> None:
    conn = _v22_db_with_runtime_state(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        run_pending_migrations(conn, 22)
        run_pending_migrations(conn, 22)

        assert _get_schema_version(conn) == SCHEMA_VERSION
        assert set(_FATAL_MEMORY_COLUMNS) <= set(_table_columns(conn, "runtime_state"))
    finally:
        conn.close()


def test_migration_0023_tolerates_missing_runtime_state(tmp_path) -> None:
    conn = duckdb.connect(str(tmp_path / "recall22-bare.duckdb"))
    try:
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE schema_migrations (migration_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO schema_version (version) VALUES (22)")
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        run_pending_migrations(conn, 22)

        assert _get_schema_version(conn) == SCHEMA_VERSION
    finally:
        conn.close()


def test_fresh_schema_matches_migrated_runtime_state_columns(tmp_path) -> None:
    """schema.sql (fresh DBs) and 0023 (existing DBs) must agree on the new columns."""
    from recall.db.schema import ensure_schema

    conn = duckdb.connect(str(tmp_path / "fresh23.duckdb"))
    try:
        ensure_schema(conn)
        columns = _table_columns(conn, "runtime_state")
        for name, sql_type in _FATAL_MEMORY_COLUMNS.items():
            assert columns[name] == sql_type, name
    finally:
        conn.close()


_LIVE_TABLE_COLUMNS = {
    "tool_results": {
        "tool_call_id": "VARCHAR",
        "result_summary": "VARCHAR",
        "is_error": "BOOLEAN",
        "completed_at": "TIMESTAMP",
    },
    "tool_use_ids": {
        "tool_call_id": "VARCHAR",
        "session_id": "VARCHAR",
        "tool_use_id": "VARCHAR",
    },
    "session_stop_markers": {
        "session_id": "VARCHAR",
        "message_idx": "INTEGER",
        "reason": "VARCHAR",
        "ends_turn": "BOOLEAN",
    },
    "live_marks": {
        "source": "VARCHAR",
        "source_session_id": "VARCHAR",
        "host": "VARCHAR",
        "pid": "BIGINT",
        "surface_key": "VARCHAR",
        "marked_at": "TIMESTAMP",
    },
}


def _v23_db_with_tool_calls(tmp_path) -> duckdb.DuckDBPyConnection:
    """A schema-23 DB holding one tool_call and its embedding, as a live DB would."""
    conn = duckdb.connect(str(tmp_path / "recall23.duckdb"))
    conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY,"
        " applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE schema_migrations (migration_id TEXT PRIMARY KEY,"
        " applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute("INSERT INTO schema_version (version) VALUES (23)")
    conn.execute(
        """
        CREATE TABLE tool_calls (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            message_id TEXT,
            idx INTEGER NOT NULL,
            tool_name TEXT NOT NULL,
            tool_input JSON,
            bash_command TEXT,
            bash_base TEXT,
            bash_sub TEXT,
            is_compound BOOLEAN DEFAULT FALSE,
            agent_id TEXT,
            subagent_type TEXT,
            subagent_description TEXT,
            subagent_model TEXT,
            skill_name TEXT
        )
        """
    )
    conn.execute(
        "CREATE TABLE tool_call_embeddings (tool_call_id TEXT PRIMARY KEY, bash_embedding FLOAT[8])"
    )
    conn.execute(
        "INSERT INTO tool_calls (id, session_id, message_id, idx, tool_name)"
        " VALUES ('tc1', 'sess1', 'msg1', 0, 'Bash')"
    )
    conn.execute("INSERT INTO tool_call_embeddings VALUES ('tc1', ?)", [[1.0] * 8])
    return conn


def test_migration_0024_adds_live_tables_without_touching_tool_calls(tmp_path) -> None:
    """REQ-LIVE-006: the migration is additive — no existing row is rewritten."""
    conn = _v23_db_with_tool_calls(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        before = conn.execute("SELECT id, tool_name FROM tool_calls").fetchall()
        before_embeddings = conn.execute("SELECT tool_call_id FROM tool_call_embeddings").fetchall()

        run_pending_migrations(conn, 23)

        assert _get_schema_version(conn) == SCHEMA_VERSION
        for table, columns in _LIVE_TABLE_COLUMNS.items():
            assert _table_columns(conn, table) == columns, table
        assert conn.execute("SELECT id, tool_name FROM tool_calls").fetchall() == before
        assert (
            conn.execute("SELECT tool_call_id FROM tool_call_embeddings").fetchall()
            == before_embeddings
        )
        assert _table_columns(conn, "tool_calls").keys() == {
            "id",
            "session_id",
            "message_id",
            "idx",
            "tool_name",
            "tool_input",
            "bash_command",
            "bash_base",
            "bash_sub",
            "is_compound",
            "agent_id",
            "subagent_type",
            "subagent_description",
            "subagent_model",
            "skill_name",
        }, "tool_calls must not gain a column: a full re-parse would UPDATE every row"
    finally:
        conn.close()


def test_migration_0024_is_idempotent(tmp_path) -> None:
    conn = _v23_db_with_tool_calls(tmp_path)
    try:
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        run_pending_migrations(conn, 23)
        run_pending_migrations(conn, 23)

        assert _get_schema_version(conn) == SCHEMA_VERSION
        assert conn.execute("SELECT COUNT(*) FROM tool_results").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM tool_use_ids").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM session_stop_markers").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM live_marks").fetchone() == (0,)
    finally:
        conn.close()


def test_fresh_schema_matches_migrated_live_tables(tmp_path) -> None:
    """schema.sql (fresh DBs) and 0024 (existing DBs) must agree."""
    from recall.db.schema import ensure_schema

    conn = duckdb.connect(str(tmp_path / "fresh24.duckdb"))
    try:
        ensure_schema(conn, embed_dim=8)
        for table, columns in _LIVE_TABLE_COLUMNS.items():
            assert _table_columns(conn, table) == columns, table
    finally:
        conn.close()


def test_migration_0025_adds_durable_source_catalog_and_is_repeatable(tmp_path) -> None:
    conn = duckdb.connect(str(tmp_path / "recall24.duckdb"))
    try:
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE schema_migrations (migration_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO schema_version VALUES (24)")
        from recall.db.migrations import run_pending_migrations
        from recall.db.schema import SCHEMA_VERSION, _get_schema_version

        run_pending_migrations(conn, 24)
        run_pending_migrations(conn, 25)

        assert _get_schema_version(conn) == SCHEMA_VERSION == 31
        columns = _table_columns(conn, "source_files")
        assert columns["desired_generation"] == "BIGINT"
        assert columns["committed_prefix_sha256"] == "VARCHAR"
        assert columns["content_epoch"] == "BIGINT"
        root_columns = _table_columns(conn, "reconciliation_roots")
        assert root_columns["failure_count"] == "BIGINT"
        assert root_columns["scan_generation"] == "BIGINT"
    finally:
        conn.close()


def test_migration_0026_preserves_pending_and_committed_source_state(tmp_path) -> None:
    from recall.db.migrations import get_migrations, run_pending_migrations
    from recall.db.schema import _get_schema_version

    conn = duckdb.connect(str(tmp_path / "admission.duckdb"))
    try:
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO schema_version VALUES (24)")
        migration25 = next(m for m in get_migrations() if m.target_version == 25)
        assert migration25.upgrade(conn)
        conn.execute(
            """INSERT INTO source_files (
                source_key, source, source_path, root_path, ctime_ns, mtime_ns, size,
                desired_generation, committed_generation, committed_offset,
                committed_prefix_sha256, content_epoch, first_pending_at,
                last_serviced_seq, retry_count, next_retry_at, last_error, observed_at)
                VALUES ('codex/path', 'codex', '/path', '/', 3, 4, 20,
                        9, 8, 10, ?, 2, 50, 17, 3, 80, 'retry me', 60)""",
            ["a" * 64],
        )
        # inventory_generation is excluded: 0030 retires it further down the chain.
        columns = tuple(
            c for c in _table_columns(conn, "source_files") if c != "inventory_generation"
        )
        before = conn.execute(f"SELECT {', '.join(columns)} FROM source_files").fetchall()
        run_pending_migrations(conn, 25)
        run_pending_migrations(conn, 25)
        from recall.db.schema import SCHEMA_VERSION

        assert _get_schema_version(conn) == SCHEMA_VERSION
        assert conn.execute(f"SELECT {', '.join(columns)} FROM source_files").fetchall() == before
        assert conn.execute("SELECT first_pending_seq FROM source_files").fetchone() == (0,)
        assert "idx_source_files_pending" not in _table_indexes(conn, "source_files")
        assert "idx_source_files_source_path" in _table_indexes(conn, "source_files")
        column = conn.execute(
            "SELECT is_nullable, column_default FROM information_schema.columns "
            "WHERE table_name = 'source_files' AND column_name = 'first_pending_seq'"
        ).fetchone()
        assert column == ("NO", "0")
    finally:
        conn.close()


def test_migration_0027_adds_index_migration_state_and_keeps_existing_unapplied(
    tmp_path,
) -> None:
    from recall.db.migrations import run_pending_migrations
    from recall.db.schema import INDEX_MIGRATION_VERSION, SCHEMA_VERSION, _get_schema_version

    conn = duckdb.connect(str(tmp_path / "recall26.duckdb"))
    try:
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE schema_migrations (migration_id TEXT PRIMARY KEY)")
        conn.execute(
            """CREATE TABLE runtime_state (
                singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton)
            )"""
        )
        conn.execute("INSERT INTO schema_version VALUES (26)")
        conn.execute("INSERT INTO runtime_state (singleton) VALUES (TRUE)")
        run_pending_migrations(conn, 26)
        run_pending_migrations(conn, 26)
        assert _get_schema_version(conn) == SCHEMA_VERSION == 31
        assert conn.execute(
            "SELECT index_migration_version FROM runtime_state WHERE singleton"
        ).fetchone() == (0,)
        assert INDEX_MIGRATION_VERSION == 1
        assert conn.execute(
            "SELECT phase, target_version FROM index_migration_jobs WHERE singleton"
        ).fetchone() == ("idle", 0)
        assert conn.execute("SELECT COUNT(*) FROM index_migration_scope").fetchone() == (0,)
    finally:
        conn.close()


def test_migration_0028_preserves_existing_job_scope_and_index_version(tmp_path) -> None:
    from recall.db.schema import ensure_schema

    with duckdb.connect(str(tmp_path / "maintenance27.duckdb")) as conn:
        ensure_schema(conn)
        conn.execute("ALTER TABLE index_migration_jobs DROP COLUMN storage_target")
        conn.execute("ALTER TABLE index_migration_jobs DROP COLUMN storage_attempt")
        conn.execute("DELETE FROM schema_version WHERE version > 27")
        conn.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (27)")
        conn.execute("UPDATE runtime_state SET index_migration_version=0 WHERE singleton")
        conn.execute("""UPDATE index_migration_jobs SET phase='failed',
            backup_path='/owned/original', captured_count=2, completed_count=1,
            started_at='2026-09-01', error='interrupted' WHERE singleton""")
        conn.execute("INSERT INTO index_migration_scope VALUES ('a','codex','/owned/a',TRUE)")
        conn.execute("INSERT INTO index_migration_scope VALUES ('b','codex','/owned/b',FALSE)")
        job = conn.execute("SELECT * FROM index_migration_jobs").fetchall()
        scope = conn.execute("SELECT * FROM index_migration_scope ORDER BY source_key").fetchall()
        ensure_schema(conn)
        ensure_schema(conn)
        assert (
            conn.execute(
                "SELECT * EXCLUDE(storage_target, storage_attempt) FROM index_migration_jobs"
            ).fetchall()
            == job
        )
        assert conn.execute(
            "SELECT storage_target, storage_attempt FROM index_migration_jobs"
        ).fetchone() == (None, 0)
        assert (
            conn.execute("SELECT * FROM index_migration_scope ORDER BY source_key").fetchall()
            == scope
        )
        assert conn.execute("SELECT index_migration_version FROM runtime_state").fetchone() == (0,)


def _v29_catalog_db(tmp_path, to_schema_v29) -> duckdb.DuckDBPyConnection:
    """A schema-29 DB whose source_files still carries the retired index and column.

    Built by reversing 0030 on a fresh database, the same way the 0028 test
    reverses its own migration: a live v29 host has exactly this shape.
    """
    from recall.db.schema import ensure_schema

    conn = duckdb.connect(str(tmp_path / "catalog29.duckdb"))
    ensure_schema(conn, embed_dim=8)
    return to_schema_v29(conn)


_V29_CATALOG_ROWS = (
    # A committed, current source and a pending, failing one: the two states the
    # pending scan distinguishes, plus a source a complete walk found missing.
    (
        "codex/committed",
        "codex",
        "/root/committed.jsonl",
        "/root",
        7,
        7,
        20,
        None,
        0,
        0.0,
        False,
        11,
    ),
    (
        "codex/pending",
        "codex",
        "/root/pending.jsonl",
        "/root",
        9,
        8,
        10,
        "retry me",
        3,
        80.0,
        False,
        12,
    ),
    ("codex/gone", "codex", "/root/gone.jsonl", "/root", 2, 2, 5, None, 0, 0.0, True, 13),
)


def _seed_v29_catalog(conn: duckdb.DuckDBPyConnection) -> None:
    for row in _V29_CATALOG_ROWS:
        conn.execute(
            """INSERT INTO source_files (
                source_key, source, source_path, root_path, ctime_ns, mtime_ns, size,
                desired_generation, committed_generation, committed_offset,
                last_error, retry_count, next_retry_at, missing, inventory_generation,
                observed_at)
                VALUES (?, ?, ?, ?, 3, 4, 20, ?, ?, ?, ?, ?, ?, ?, ?, 60)""",
            list(row),
        )


def _retained_catalog(conn: duckdb.DuckDBPyConnection) -> list[tuple[object, ...]]:
    columns = [c for c in _table_columns(conn, "source_files") if c != "inventory_generation"]
    return conn.execute(
        f"SELECT {', '.join(columns)} FROM source_files ORDER BY source_key"
    ).fetchall()


def test_migration_0030_removes_retired_catalog_objects_and_keeps_every_row(
    tmp_path, to_schema_v29
) -> None:
    """REQ-MIG-010: version 30 drops shape only — no logical row changes."""
    from recall.db.schema import SCHEMA_VERSION, _get_schema_version, ensure_schema

    conn = _v29_catalog_db(tmp_path, to_schema_v29)
    try:
        _seed_v29_catalog(conn)
        before = _retained_catalog(conn)

        ensure_schema(conn, embed_dim=8)

        assert _get_schema_version(conn) == SCHEMA_VERSION == 31
        assert "inventory_generation" not in _table_columns(conn, "source_files")
        assert "idx_source_files_pending" not in _table_indexes(conn, "source_files")
        assert "idx_source_files_source_path" in _table_indexes(conn, "source_files")
        assert _retained_catalog(conn) == before
        assert conn.execute("SELECT COUNT(*) FROM source_files").fetchone() == (
            len(_V29_CATALOG_ROWS),
        )
    finally:
        conn.close()


def test_migration_0030_is_idempotent_when_replayed(tmp_path, to_schema_v29) -> None:
    """Replaying the upgrade on an already-migrated DB changes nothing."""
    from recall.db.migrations import get_migrations, run_pending_migrations
    from recall.db.schema import _get_schema_version, ensure_schema

    conn = _v29_catalog_db(tmp_path, to_schema_v29)
    try:
        _seed_v29_catalog(conn)
        ensure_schema(conn, embed_dim=8)
        after_first = _retained_catalog(conn)

        migration = next(m for m in get_migrations() if m.target_version == 30)
        assert migration.upgrade(conn)
        run_pending_migrations(conn, 29)
        ensure_schema(conn, embed_dim=8)

        assert _get_schema_version(conn) == 31
        assert _retained_catalog(conn) == after_first
        assert "inventory_generation" not in _table_columns(conn, "source_files")
        assert "idx_source_files_pending" not in _table_indexes(conn, "source_files")
        assert "idx_source_files_source_path" in _table_indexes(conn, "source_files")
    finally:
        conn.close()


class _FailingOnDropColumn:
    """A connection proxy that fails the one DDL statement 0030 cannot compensate.

    DuckDB rejects `ALTER TABLE ... DROP COLUMN` while any secondary index is
    present, so the migration must drop its indexes first; this injects a
    failure in exactly that window to prove the indexes come back.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def execute(self, sql: str, *args: object, **kwargs: object):
        if "DROP COLUMN" in sql:
            raise duckdb.Error("injected catalog DDL failure")
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)


def test_migration_0030_leaves_v29_intact_when_the_ddl_fails(tmp_path, to_schema_v29) -> None:
    """A failed DDL must restore the indexes and leave version 29 unrecorded."""
    from recall.db.migrations import get_migrations, run_pending_migrations
    from recall.db.schema import _get_schema_version

    conn = _v29_catalog_db(tmp_path, to_schema_v29)
    try:
        assert any(m.target_version == 30 for m in get_migrations())
        _seed_v29_catalog(conn)
        before = _retained_catalog(conn)

        run_pending_migrations(cast(duckdb.DuckDBPyConnection, _FailingOnDropColumn(conn)), 29)

        assert _get_schema_version(conn) == 29
        migration_ids = {
            str(row[0])
            for row in conn.execute("SELECT migration_id FROM schema_migrations").fetchall()
        }
        assert not any(mid.startswith("0030") for mid in migration_ids)
        assert "inventory_generation" in _table_columns(conn, "source_files")
        assert "idx_source_files_pending" in _table_indexes(conn, "source_files")
        assert "idx_source_files_source_path" in _table_indexes(conn, "source_files")
        assert _retained_catalog(conn) == before
    finally:
        conn.close()
