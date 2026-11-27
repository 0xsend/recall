"""Schema 44 adds subagent lineage to session identity rows (REQ-SCHEMA-007)."""

from __future__ import annotations

from pathlib import Path

from recall import _native
from recall.db.lease import StandaloneLifetime
from recall.db.migrations import get_migrations, run_pending_migrations
from recall.db.schema import SCHEMA_VERSION, ensure_schema, get_schema_version

ROOT = "34221246-b285-48af-9b2a-fe4c083f3208"
PROJECT = "/Users/me/.claude/projects/-Users-me-repo"
ROWS = [
    ("sub", "claude_code", f"{PROJECT}/{ROOT}/subagents/agent-a1b2c3.jsonl", ROOT),
    ("root", "claude_code", f"{PROJECT}/{ROOT}.jsonl", ROOT),
    ("meta-only", "claude_code", f"{PROJECT}/{ROOT}/subagents/notes.jsonl", None),
    ("codex", "codex", f"/x/{ROOT}/subagents/agent-a1b2c3.jsonl", "c"),
]
LINEAGE = "SELECT id, root_source_session_id, agent_id, agent_type, agent_description FROM sessions"


def _schema_43_db(path: Path):
    conn = _native.duckdb_connect(str(path))
    ensure_schema(conn, embed_dim=8)
    conn.execute("DROP TABLE sessions")
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_session_id TEXT,
            UNIQUE (source, source_path)
        )
        """
    )
    conn.execute(
        "INSERT INTO sessions (id, source, source_path, source_session_id) VALUES (?, ?, ?, ?)",
        ROWS[0],
    )
    conn.executemany(
        "INSERT INTO sessions (id, source, source_path, source_session_id) VALUES (?, ?, ?, ?)",
        ROWS[1:],
    )
    conn.execute("DELETE FROM schema_migrations WHERE migration_id = '0044_session_lineage'")
    conn.execute("DELETE FROM schema_version WHERE version > 43")
    conn.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (43)")
    return conn


def test_migration_0044_backfills_claude_subagent_paths_and_keeps_identity(tmp_path) -> None:
    conn = _schema_43_db(tmp_path / "recall.duckdb")
    try:
        run_pending_migrations(conn, 43)

        assert get_schema_version(conn) == SCHEMA_VERSION == 44
        assert conn.execute(f"{LINEAGE} ORDER BY id").fetchall() == [
            ("codex", None, None, None, None),
            ("meta-only", None, None, None, None),
            ("root", None, None, None, None),
            ("sub", ROOT, "a1b2c3", None, None),
        ]
        assert conn.execute(
            "SELECT id, source, source_path, source_session_id FROM sessions ORDER BY id"
        ).fetchall() == sorted(ROWS)
    finally:
        conn.close()


def test_migration_0044_rerun_keeps_lineage_the_indexer_already_wrote(tmp_path) -> None:
    conn = _schema_43_db(tmp_path / "recall.duckdb")
    try:
        run_pending_migrations(conn, 43)
        conn.execute(
            "UPDATE sessions SET agent_type = 'Explore', agent_description = 'Find it'"
            " WHERE id = 'sub'"
        )
        before = conn.execute(f"{LINEAGE} ORDER BY id").fetchall()

        migration = next(m for m in get_migrations() if m.target_version == 44)
        with StandaloneLifetime().lease(conn) as lease:
            assert migration.upgrade(lease)
        run_pending_migrations(conn, 43)

        assert conn.execute(f"{LINEAGE} ORDER BY id").fetchall() == before
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE migration_id = '0044_session_lineage'"
        ).fetchone() == (1,)
    finally:
        conn.close()
