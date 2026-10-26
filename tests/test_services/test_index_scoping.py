"""REQ-INDEX-022: scoping a full re-parse to a slice of the corpus.

A parser change only reaches already-indexed sessions through `--full`, and on
a large corpus that is one 73-157 minute all-or-nothing run.  `--since` and
`--project` let it be done in chunks.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.parsers.claude_code import ClaudeCodeParser
from recall.services.indexer import DiscoveredSessionPath, _scope_paths


def _path(name: str, age_seconds: float) -> DiscoveredSessionPath:
    return DiscoveredSessionPath(
        parser=ClaudeCodeParser(),
        path=Path(name),
        resolved_path=name,
        file_mtime=datetime.now(UTC).timestamp() - age_seconds,
        file_size=1,
    )


def _paths() -> list[DiscoveredSessionPath]:
    return [_path("/s/recent.jsonl", 3600), _path("/s/old.jsonl", 86400 * 90)]


def _conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn, embed_dim=8)
    conn.execute(
        "INSERT INTO sessions VALUES "
        "('old', 'claude_code', '/s/old.jsonl', NULL), "
        "('recent', 'claude_code', '/s/recent.jsonl', NULL), "
        "('other-source', 'codex', '/s/recent.jsonl', NULL)"
    )
    conn.execute(
        "INSERT INTO session_state(session_id, file_mtime, file_size, git_repo) VALUES "
        "('old', 0, 0, '/myrepo'), ('recent', 0, 0, '/other'), "
        "('other-source', 0, 0, '/myrepo')"
    )
    return conn


def test_no_scope_keeps_everything() -> None:
    paths = _paths()
    assert _scope_paths(None, paths, since=None, project=None) == paths


def test_since_keeps_only_transcripts_touched_after_the_cutoff() -> None:
    cutoff = datetime.now(UTC) - timedelta(days=30)
    kept = _scope_paths(None, _paths(), since=cutoff, project=None)
    assert [p.resolved_path for p in kept] == ["/s/recent.jsonl"]


def test_project_keeps_only_sessions_whose_git_repo_matches() -> None:
    """Matching mirrors `recall list --project` (git_repo ILIKE %p%) so the
    flag means one thing across the CLI."""
    with _conn() as conn:
        kept = _scope_paths(conn, _paths(), since=None, project="myrepo")
    assert [p.resolved_path for p in kept] == ["/s/old.jsonl"]


def test_scopes_compose() -> None:
    cutoff = datetime.now(UTC) - timedelta(days=30)
    with _conn() as conn:
        assert _scope_paths(conn, _paths(), since=cutoff, project="myrepo") == []


def test_project_without_a_connection_is_refused() -> None:
    """A project scope needs the DB to resolve; silently ignoring it would
    re-parse the whole corpus when the user asked for one repo."""
    with pytest.raises(ValueError, match="project"):
        _scope_paths(None, _paths(), since=None, project="myrepo")
