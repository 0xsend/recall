"""Schema 43 migration (0043_utc_timestamps), undo reversal, defaults, and rollback checks."""

from __future__ import annotations

import importlib
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from recall.core.schema_version import RecoveryStateUnavailableError
from recall.db.lease import StandaloneLifetime
from recall.db.migrations import get_migrations, run_pending_migrations
from recall.db.queries import enqueue_sidecar_pending
from recall.db.schema import (
    INDEX_MIGRATION_VERSION,
    SCHEMA_VERSION,
    ensure_schema,
    ensure_schema_lenient,
    get_schema_version,
)
from recall.services.index_migration import (
    applied_version,
    begin_storage_migration,
    is_eligible,
    rollback_migration,
)

from tests.test_services.test_run_completion import _config


def _set_tz(monkeypatch: pytest.MonkeyPatch, tz_name: str) -> None:
    monkeypatch.setenv("TZ", tz_name)
    if hasattr(time, "tzset"):
        time.tzset()


def _seed_schema_42_db(
    path: Path,
    *,
    tz_name: str = "America/Chicago",
    version: int = 42,
    reject_version: bool = False,
) -> None:
    """Create a schema-41 or schema-42 DB with legacy local-time defaults and operational rows."""
    with duckdb.connect(str(path)) as conn:
        conn.execute(f"SET TimeZone = '{tz_name}'")
        ensure_schema(conn)
        # Restore pre-0043 local-time catalog defaults to faithfully mirror upgraded production DBs.
        conn.execute(
            "ALTER TABLE schema_version ALTER COLUMN applied_at SET DEFAULT CURRENT_TIMESTAMP"
        )
        conn.execute(
            "ALTER TABLE schema_migrations ALTER COLUMN applied_at SET DEFAULT CURRENT_TIMESTAMP"
        )
        conn.execute(
            "ALTER TABLE schema_migration_undo "
            "ALTER COLUMN recorded_at SET DEFAULT CURRENT_TIMESTAMP"
        )
        conn.execute(
            "ALTER TABLE session_state ALTER COLUMN indexed_at SET DEFAULT CURRENT_TIMESTAMP"
        )
        conn.execute(
            "ALTER TABLE embedding_cache ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP"
        )
        conn.execute("ALTER TABLE fts_sidecar_pending ALTER COLUMN queued_at SET DEFAULT now()")

        conn.execute("DELETE FROM schema_migrations WHERE migration_id = '0043_utc_timestamps'")
        if version == 41:
            conn.execute("DROP TABLE claude_usage_scopes")
            conn.execute("DROP TABLE claude_response_usage")
            conn.execute("ALTER TABLE runtime_state DROP COLUMN claude_usage_incarnation")
            conn.execute(
                "DELETE FROM schema_migrations WHERE migration_id = '0042_claude_response_usage'"
            )
        conn.execute("DELETE FROM schema_version WHERE version > ?", [version])
        if reject_version:
            conn.execute("DROP TABLE schema_version")
            conn.execute(
                f"CREATE TABLE schema_version("
                f"version INTEGER PRIMARY KEY CHECK(version <= {version}), "
                f"applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
        conn.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (?)", [version])
        conn.execute("UPDATE runtime_state SET index_migration_version = 1 WHERE singleton")

        # Historical session / message / tool_result rows (must remain untouched by 0043).
        conn.execute(
            "INSERT INTO sessions(id, source, source_path, source_session_id) "
            "VALUES ('hist', 'claude_code', '/tmp/hist.jsonl', 'native')"
        )
        conn.execute(
            "INSERT INTO session_state(session_id, started_at, ended_at, indexed_at, "
            "file_mtime, file_size) VALUES ('hist', TIMESTAMP '2026-07-15 13:00:00', "
            "TIMESTAMP '2026-07-15 13:05:00', TIMESTAMP '2026-07-15 13:06:00', 1.0, 100)"
        )
        conn.execute("INSERT INTO messages(id, session_id, idx) VALUES ('m1', 'hist', 0)")
        conn.execute(
            "INSERT INTO message_state(message_id, role, content, timestamp) "
            "VALUES ('m1', 'user', 'hi', TIMESTAMP '2026-07-15 13:00:00')"
        )
        conn.execute(
            "INSERT INTO tool_calls(id, session_id, message_id, idx, tool_name) "
            "VALUES ('tc1', 'hist', 'm1', 0, 'Bash')"
        )
        conn.execute(
            "INSERT INTO tool_results(tool_call_id, result_summary, is_error, completed_at) "
            "VALUES ('tc1', 'ok', FALSE, TIMESTAMP '2026-07-15 13:01:00')"
        )

        # Operational tables converted by 0043: CDT (+5h), CST (+6h), NULLs, special PKs.
        conn.execute(
            """
            INSERT INTO usage_events(
                id, source, source_session_id, session_id, ts,
                prompt_tokens, cached_prompt_tokens, completion_tokens, reasoning_tokens,
                host, harvested_at
            ) VALUES
                ('ev_cdt', 'grok', 's1', 'hist', TIMESTAMP '2026-07-15 13:00:00.123456',
                 10, 0, 5, 0, 'aem5', TIMESTAMP '2026-07-15 13:02:00.111111'),
                ('ev_cst', 'grok', 's2', NULL, TIMESTAMP '2026-11-22 12:00:00.654321',
                 20, 0, 10, 0, 'aem5', TIMESTAMP '2026-11-22 12:05:00.222222'),
                ('ev_null', 'grok', 's3', NULL, NULL,
                 30, 0, 15, 0, NULL, TIMESTAMP '2026-11-22 12:10:00')
            """
        )
        conn.execute(
            """
            INSERT INTO usage_log_cursors(
                path, byte_offset, file_size, consumed_tail_sha256, updated_at
            ) VALUES ('/tmp/unified.jsonl', 128, 128, 'abc', TIMESTAMP '2026-07-15 13:10:00.500000')
            """
        )
        conn.execute(
            """
            INSERT INTO live_marks(source, source_session_id, host, pid, marked_at) VALUES
                ('claude_code', 'sid:with/slash"quote', 'host:1', 1234,
                 TIMESTAMP '2026-07-15 13:15:00.789012'),
                ('codex', 'sid_cst', 'aem5', 5678, TIMESTAMP '2026-11-22 12:20:00'),
                ('codex', 'sid_null', 'aem5', 9999, NULL)
            """
        )


def test_migration_0043_converts_operational_tables_records_undo_and_reverses_under_chicago(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_tz(monkeypatch, "America/Chicago")
    path = tmp_path / "v41_chicago.duckdb"
    _seed_schema_42_db(path, tz_name="America/Chicago", version=41)
    before_utc = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=5)

    with duckdb.connect(str(path)) as conn:
        conn.execute("SET TimeZone = 'America/Chicago'")
        ensure_schema(conn)
        after_utc = datetime.now(UTC).replace(tzinfo=None) + timedelta(seconds=5)

        assert get_schema_version(conn) == SCHEMA_VERSION == 43
        assert INDEX_MIGRATION_VERSION == 2
        assert applied_version(conn) == 1
        assert is_eligible(conn) is True

        # Both 0042 and 0043 audit rows AND schema_version 42/43 rows landed in UTC.
        for mid in ("0042_claude_response_usage", "0043_utc_timestamps"):
            row = conn.execute(
                "SELECT applied_at FROM schema_migrations WHERE migration_id = ?", [mid]
            ).fetchone()
            assert row is not None
            assert before_utc <= row[0] <= after_utc, f"{mid} applied_at not UTC: {row[0]}"
        for ver in (42, 43):
            row = conn.execute(
                "SELECT applied_at FROM schema_version WHERE version = ?", [ver]
            ).fetchone()
            assert row is not None
            assert before_utc <= row[0] <= after_utc, f"schema_version {ver} not UTC: {row[0]}"

        # Verify operational table UTC conversions (CDT +5h, CST +6h, microseconds, NULLs).
        events = {
            r[0]: (r[1], r[2])
            for r in conn.execute("SELECT id, ts, harvested_at FROM usage_events").fetchall()
        }
        assert events["ev_cdt"] == (
            datetime(2026, 7, 15, 18, 0, 0, 123456),
            datetime(2026, 7, 15, 18, 2, 0, 111111),
        )
        assert events["ev_cst"] == (
            datetime(2026, 11, 22, 18, 0, 0, 654321),
            datetime(2026, 11, 22, 18, 5, 0, 222222),
        )
        assert events["ev_null"] == (None, datetime(2026, 11, 22, 18, 10, 0))
        usage_indexes = {
            r[0]
            for r in conn.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'usage_events'"
            ).fetchall()
        }
        assert {
            "idx_usage_events_source_sid",
            "idx_usage_events_session",
            "idx_usage_events_ts",
        } <= usage_indexes

        cursor_row = conn.execute(
            "SELECT updated_at FROM usage_log_cursors WHERE path = '/tmp/unified.jsonl'"
        ).fetchone()
        assert cursor_row == (datetime(2026, 7, 15, 18, 10, 0, 500000),)

        marks = {
            (r[0], r[1], r[2]): r[3]
            for r in conn.execute(
                "SELECT source, source_session_id, host, marked_at FROM live_marks"
            ).fetchall()
        }
        assert marks[("claude_code", 'sid:with/slash"quote', "host:1")] == datetime(
            2026, 7, 15, 18, 15, 0, 789012
        )
        assert marks[("codex", "sid_cst", "aem5")] == datetime(2026, 11, 22, 18, 20, 0)
        assert marks[("codex", "sid_null", "aem5")] is None

        # Historical tables untouched by startup migration.
        assert conn.execute(
            "SELECT started_at, ended_at, indexed_at FROM session_state WHERE session_id = 'hist'"
        ).fetchone() == (
            datetime(2026, 7, 15, 13, 0, 0),
            datetime(2026, 7, 15, 13, 5, 0),
            datetime(2026, 7, 15, 13, 6, 0),
        )
        assert conn.execute(
            "SELECT timestamp FROM message_state WHERE message_id = 'm1'"
        ).fetchone() == (datetime(2026, 7, 15, 13, 0, 0),)
        assert conn.execute(
            "SELECT completed_at FROM tool_results WHERE tool_call_id = 'tc1'"
        ).fetchone() == (datetime(2026, 7, 15, 13, 1, 0),)

        # Undo pre-images recorded with UTC recorded_at (5 usage_events + 1 cursor + 2 live_marks).
        undo_rows = conn.execute(
            "SELECT table_name, row_key, column_name, old_value, recorded_at "
            "FROM schema_migration_undo WHERE migration_id = '0043_utc_timestamps'"
        ).fetchall()
        assert len(undo_rows) == 8
        for _, _, _, _, rec_at in undo_rows:
            assert before_utc <= rec_at <= after_utc

        # Post-upgrade defaults & writers on upgraded DB land in naive UTC.
        enqueue_sidecar_pending(conn, "message", ["m1"], "upsert")
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind, id, op) VALUES ('message', 'm_def', 'upsert')"
        )
        for (q_at,) in conn.execute("SELECT queued_at FROM fts_sidecar_pending").fetchall():
            assert (
                before_utc <= q_at <= datetime.now(UTC).replace(tzinfo=None) + timedelta(seconds=5)
            )

        conn.execute(
            "INSERT INTO embedding_cache("
            "cache_key, kind, raw_text, normalized_text, embedding"
            ") VALUES ('k1', 'content', 'a', 'a', ?::FLOAT[384])",
            [[0.0] * 384],
        )
        (ec_at,) = conn.execute(
            "SELECT created_at FROM embedding_cache WHERE cache_key = 'k1'"
        ).fetchone() or (None,)
        assert ec_at is not None
        assert before_utc <= ec_at <= datetime.now(UTC).replace(tzinfo=None) + timedelta(seconds=5)

        # Idempotency: re-running ensure_schema or direct 0043._upgrade never double-shifts.
        ensure_schema(conn)
        migration = next(m for m in get_migrations() if m.id == "0043_utc_timestamps")
        with StandaloneLifetime().lease(conn) as lease:
            assert migration.upgrade(lease)
        assert conn.execute(
            "SELECT ts, harvested_at FROM usage_events WHERE id = 'ev_cdt'"
        ).fetchone() == (
            datetime(2026, 7, 15, 18, 0, 0, 123456),
            datetime(2026, 7, 15, 18, 2, 0, 111111),
        )
        assert conn.execute(
            "SELECT marked_at FROM live_marks WHERE source_session_id = 'sid_cst'"
        ).fetchone() == (datetime(2026, 11, 22, 18, 20, 0),)

        # Documented reverse SQL restores exact pre-upgrade values bit-for-bit.
        conn.execute(
            """
            BEGIN;
            DROP TABLE IF EXISTS usage_events_restore;
            CREATE TABLE usage_events_restore (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_session_id TEXT NOT NULL,
                session_id TEXT,
                ts TIMESTAMP,
                prompt_tokens BIGINT,
                cached_prompt_tokens BIGINT,
                completion_tokens BIGINT,
                reasoning_tokens BIGINT,
                host TEXT,
                harvested_at TIMESTAMP NOT NULL
            );
            INSERT INTO usage_events_restore (
                id, source, source_session_id, session_id, ts,
                prompt_tokens, cached_prompt_tokens, completion_tokens,
                reasoning_tokens, host, harvested_at
            )
            SELECT
                e.id, e.source, e.source_session_id, e.session_id,
                COALESCE(CAST(u_ts.old_value AS TIMESTAMP), e.ts),
                e.prompt_tokens, e.cached_prompt_tokens, e.completion_tokens,
                e.reasoning_tokens, e.host,
                COALESCE(CAST(u_h.old_value AS TIMESTAMP), e.harvested_at)
            FROM usage_events e
            LEFT JOIN schema_migration_undo u_ts
              ON u_ts.migration_id = '0043_utc_timestamps'
             AND u_ts.table_name = 'usage_events'
             AND u_ts.column_name = 'ts'
             AND u_ts.row_key = e.id
            LEFT JOIN schema_migration_undo u_h
              ON u_h.migration_id = '0043_utc_timestamps'
             AND u_h.table_name = 'usage_events'
             AND u_h.column_name = 'harvested_at'
             AND u_h.row_key = e.id;
            DROP TABLE usage_events;
            ALTER TABLE usage_events_restore RENAME TO usage_events;
            CREATE INDEX idx_usage_events_source_sid
                ON usage_events(source, source_session_id);
            CREATE INDEX idx_usage_events_session ON usage_events(session_id);
            CREATE INDEX idx_usage_events_ts ON usage_events(ts);

            UPDATE usage_log_cursors
            SET updated_at = CAST(u.old_value AS TIMESTAMP)
            FROM schema_migration_undo u
            WHERE u.migration_id = '0043_utc_timestamps'
              AND u.table_name = 'usage_log_cursors'
              AND u.column_name = 'updated_at'
              AND u.row_key = usage_log_cursors.path;

            UPDATE live_marks
            SET marked_at = CAST(u.old_value AS TIMESTAMP)
            FROM schema_migration_undo u
            WHERE u.migration_id = '0043_utc_timestamps'
              AND u.table_name = 'live_marks'
              AND u.column_name = 'marked_at'
              AND json_extract_string(u.row_key, '$[0]') = live_marks.source
              AND json_extract_string(u.row_key, '$[1]') = live_marks.source_session_id
              AND json_extract_string(u.row_key, '$[2]') = live_marks.host;
            COMMIT;
            """
        )
        assert conn.execute(
            "SELECT ts, harvested_at FROM usage_events WHERE id = 'ev_cdt'"
        ).fetchone() == (
            datetime(2026, 7, 15, 13, 0, 0, 123456),
            datetime(2026, 7, 15, 13, 2, 0, 111111),
        )
        assert conn.execute(
            "SELECT updated_at FROM usage_log_cursors WHERE path = '/tmp/unified.jsonl'"
        ).fetchone() == (datetime(2026, 7, 15, 13, 10, 0, 500000),)
        assert conn.execute(
            "SELECT marked_at FROM live_marks WHERE source_session_id = ?",
            ['sid:with/slash"quote'],
        ).fetchone() == (datetime(2026, 7, 15, 13, 15, 0, 789012),)


def test_migration_0043_is_zero_row_noop_on_utc_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_tz(monkeypatch, "UTC")
    path = tmp_path / "v42_utc.duckdb"
    _seed_schema_42_db(path, tz_name="UTC", version=42)

    with duckdb.connect(str(path)) as conn:
        conn.execute("SET TimeZone = 'UTC'")
        ensure_schema(conn)
        assert get_schema_version(conn) == SCHEMA_VERSION == 43
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_migration_undo WHERE migration_id = '0043_utc_timestamps'"
        ).fetchone() == (0,)
        assert conn.execute("SELECT ts FROM usage_events WHERE id = 'ev_cdt'").fetchone() == (
            datetime(2026, 7, 15, 13, 0, 0, 123456),
        )


def test_migration_0043_rolls_back_atomically_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_tz(monkeypatch, "America/Chicago")
    path = tmp_path / "v42_fail.duckdb"
    _seed_schema_42_db(path, tz_name="America/Chicago", version=42, reject_version=True)

    with duckdb.connect(str(path)) as conn:
        conn.execute("SET TimeZone = 'America/Chicago'")
        run_pending_migrations(conn, 42)
        assert get_schema_version(conn) == 42
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_migration_undo WHERE migration_id = '0043_utc_timestamps'"
        ).fetchone() == (0,)
        assert conn.execute("SELECT ts FROM usage_events WHERE id = 'ev_cdt'").fetchone() == (
            datetime(2026, 7, 15, 13, 0, 0, 123456),
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE migration_id = '0043_utc_timestamps'"
        ).fetchone() == (0,)


def test_migration_0043_refuses_pre_42_and_damaged_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_tz(monkeypatch, "America/Chicago")
    v41 = tmp_path / "v41_only.duckdb"
    _seed_schema_42_db(v41, tz_name="America/Chicago", version=41)
    with duckdb.connect(str(v41)) as conn:
        conn.execute("SET TimeZone = 'America/Chicago'")
        migration = importlib.import_module("recall.db.migrations.0043_utc_timestamps")
        with StandaloneLifetime().lease(conn) as lease:
            assert migration._upgrade(lease) is False
        assert get_schema_version(conn) == 41

    damaged = tmp_path / "v42_damaged.duckdb"
    _seed_schema_42_db(damaged, tz_name="America/Chicago", version=42)
    with duckdb.connect(str(damaged)) as conn:
        conn.execute("SET TimeZone = 'America/Chicago'")
        conn.execute("DROP TABLE daemon_index_repair")
        with pytest.raises(RecoveryStateUnavailableError):
            ensure_schema(conn)
        with pytest.raises(RecoveryStateUnavailableError):
            ensure_schema_lenient(conn)
        assert get_schema_version(conn) == 42
        assert conn.execute("SELECT ts FROM usage_events WHERE id = 'ev_cdt'").fetchone() == (
            datetime(2026, 7, 15, 13, 0, 0, 123456),
        )


def test_rollback_migration_accepts_schema_43_and_refuses_schema_42(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    config = _config(data, None)
    with duckdb.connect(
        str(config.db_path), config={"storage_compatibility_version": "v1.0.0"}
    ) as conn:
        ensure_schema(conn)
        job = begin_storage_migration(conn, config, observer=StandaloneLifetime())
    with closing(sqlite3.connect(data / "recall.fts.sqlite")) as sidecar:
        sidecar.execute("CREATE TABLE witness(v TEXT)")
        sidecar.commit()
    assert job.backup_path is not None

    # A schema-42 backup cannot be installed into a schema-43 DB (no logical downgrade).
    with duckdb.connect(str(Path(job.backup_path) / "recall.duckdb")) as backup:
        backup.execute("DELETE FROM schema_version WHERE version > 42")
        backup.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (42)")
    before = config.db_path.read_bytes()
    with pytest.raises(RuntimeError, match="cannot lower the logical schema"):
        rollback_migration(config, preserve=tmp_path / "preserve_bad")
    assert config.db_path.read_bytes() == before

    # Restoring the backup to schema 43 succeeds cleanly.
    with duckdb.connect(str(Path(job.backup_path) / "recall.duckdb")) as backup:
        backup.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (43)")
    restored = rollback_migration(config, preserve=tmp_path / "preserve_ok")
    assert restored.exists()
    assert (config.data_dir / "daemon-refused").exists()


def test_migration_0043_converts_pending_sidecar_queue_preserving_utc_plus_9_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-upgrade local delete at 19:00 JST must sort before post-upgrade 10:05Z upsert."""
    from recall.db.fts_sidecar import open_sidecar
    from recall.services.fts_sidecar_reconcile import repair_pending_batch_on

    _set_tz(monkeypatch, "Asia/Tokyo")
    path = tmp_path / "v42_tokyo.duckdb"
    _seed_schema_42_db(path, tz_name="Asia/Tokyo", version=42)

    with duckdb.connect(str(path)) as conn:
        conn.execute("SET TimeZone = 'Asia/Tokyo'")
        # Pre-upgrade delete queued at 19:00 local JST (= 10:00 UTC).
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind, id, op, queued_at) "
            "VALUES ('message', 'm1', 'delete', TIMESTAMP '2026-07-15 19:00:00')"
        )
        ensure_schema(conn)
        assert conn.execute(
            "SELECT queued_at FROM fts_sidecar_pending WHERE id = 'm1' AND op = 'delete'"
        ).fetchone() == (datetime(2026, 7, 15, 10, 0, 0),)

        # Post-upgrade recreation upsert queued at 10:05 UTC.
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind, id, op, queued_at) "
            "VALUES ('message', 'm1', 'upsert', TIMESTAMP '2026-07-15 10:05:00')"
        )
        sidecar = open_sidecar(Path(":memory:"))
        try:
            with StandaloneLifetime().lease(conn) as lease:
                result = repair_pending_batch_on(lease, sidecar)
            assert result.applied == 2 and result.remaining == 0
            assert sidecar.execute("SELECT COUNT(*) FROM message_fts").fetchone() == (1,)
        finally:
            sidecar.close()


def test_pending_sidecar_repair_converges_across_chicago_dst_fallback_hour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ambiguous Chicago fall-back delete (01:30 -> 07:30Z) cannot erase live 07:05Z upsert."""
    from recall.db.fts_sidecar import open_sidecar
    from recall.services.fts_sidecar_reconcile import repair_pending_batch_on

    _set_tz(monkeypatch, "America/Chicago")
    path = tmp_path / "v42_fallback.duckdb"
    _seed_schema_42_db(path, tz_name="America/Chicago", version=42)

    with duckdb.connect(str(path)) as conn:
        conn.execute("SET TimeZone = 'America/Chicago'")
        # Pre-upgrade delete at 01:30 CDT on Nov 1, 2026 (06:30Z); 0043 resolves ambiguous
        # 01:30 to 07:30Z, which sorts AFTER a post-upgrade 07:05Z upsert.
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind, id, op, queued_at) "
            "VALUES ('message', 'm1', 'delete', TIMESTAMP '2026-11-01 01:30:00')"
        )
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind, id, op, queued_at) "
            "VALUES ('message', 'm1', 'upsert', TIMESTAMP '2026-11-01 07:05:00')"
        )
        sidecar = open_sidecar(Path(":memory:"))
        try:
            with StandaloneLifetime().lease(conn) as lease:
                result = repair_pending_batch_on(lease, sidecar)
            assert result.applied == 2 and result.remaining == 0
            assert sidecar.execute("SELECT COUNT(*) FROM message_fts").fetchone() == (1,)
        finally:
            sidecar.close()
