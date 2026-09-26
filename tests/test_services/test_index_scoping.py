"""REQ-INDEX-022: scoping a full re-parse to a slice of the corpus.

A parser change only reaches already-indexed sessions through `--full`, and on
a large corpus that is one 73-157 minute all-or-nothing run.  `--since` and
`--project` let it be done in chunks.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from recall.parsers.claude_code import ClaudeCodeParser
from recall.services.indexer import DiscoveredSessionPath, _scope_paths

if TYPE_CHECKING:
    import duckdb


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
    """The project lookup is monkeypatched, so the connection is never used."""
    return cast("duckdb.DuckDBPyConnection", object())


def test_no_scope_keeps_everything() -> None:
    paths = _paths()
    assert _scope_paths(None, paths, since=None, project=None) == paths


def test_since_keeps_only_transcripts_touched_after_the_cutoff() -> None:
    cutoff = datetime.now(UTC) - timedelta(days=30)
    kept = _scope_paths(None, _paths(), since=cutoff, project=None)
    assert [p.resolved_path for p in kept] == ["/s/recent.jsonl"]


def test_project_keeps_only_sessions_whose_git_repo_matches(monkeypatch) -> None:
    """Matching mirrors `recall list --project` (git_repo ILIKE %p%) so the
    flag means one thing across the CLI."""
    monkeypatch.setattr(
        "recall.services.indexer._source_paths_for_project",
        lambda conn, project: {"/s/old.jsonl"},
    )
    kept = _scope_paths(_conn(), _paths(), since=None, project="myrepo")
    assert [p.resolved_path for p in kept] == ["/s/old.jsonl"]


def test_scopes_compose(monkeypatch) -> None:
    monkeypatch.setattr(
        "recall.services.indexer._source_paths_for_project",
        lambda conn, project: {"/s/old.jsonl"},
    )
    cutoff = datetime.now(UTC) - timedelta(days=30)
    assert _scope_paths(_conn(), _paths(), since=cutoff, project="myrepo") == []


def test_project_without_a_connection_is_refused() -> None:
    """A project scope needs the DB to resolve; silently ignoring it would
    re-parse the whole corpus when the user asked for one repo."""
    with pytest.raises(ValueError, match="project"):
        _scope_paths(None, _paths(), since=None, project="myrepo")
