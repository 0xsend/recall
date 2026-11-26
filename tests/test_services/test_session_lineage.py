"""Subagent sessions file under their root session (REQ-SCHEMA-007, REQ-CLI-031)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from recall.core.config import AppConfig
from recall.db.lease import StandaloneLifetime
from recall.services.analytics import usage_by_source
from recall.services.session_ids import resolve_session_identifier
from recall.services.sessions import list_sessions, resolve_session_header, session_tree_on

from tests.test_services.test_checkpoint_resume_fallbacks import Harness, _first_wins_harness

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/claude_code/session_v2.jsonl"
ROOT_SID = "34221246-b285-48af-9b2a-fe4c083f3208"
ORPHAN_SID = "99999999-0000-4000-8000-000000000000"
PROJECT = "-Users-me-repo"


def transcript(sid: str, label: str, minute: int) -> bytes:
    entry = json.loads(FIXTURE.read_text().splitlines()[1])
    entry["sessionId"] = sid
    entry["uuid"] = label
    entry["timestamp"] = f"2026-11-20T10:{minute:02d}:00Z"
    entry["message"]["id"] = f"msg_{label}"
    entry["message"]["content"] = [{"type": "text", "text": label}]
    return (json.dumps(entry) + "\n").encode()


@pytest.fixture
def indexed(tmp_path: Path):
    root = tmp_path / "roots"
    first = _first_wins_harness(
        "claude-code", f"{PROJECT}/{ROOT_SID}.jsonl", AppConfig.load(), root
    )
    files = {
        f"{PROJECT}/{ROOT_SID}.jsonl": (transcript(ROOT_SID, "root", 0), None),
        f"{PROJECT}/{ROOT_SID}/subagents/agent-explore1.jsonl": (
            transcript(ROOT_SID, "explore", 1),
            {"agentType": "Explore", "description": "Find the parser"},
        ),
        f"{PROJECT}/{ROOT_SID}/subagents/agent-plan1.jsonl": (
            transcript(ROOT_SID, "plan", 2),
            None,
        ),
        f"{PROJECT}/{ORPHAN_SID}/subagents/agent-lost1.jsonl": (
            transcript(ORPHAN_SID, "lost", 3),
            None,
        ),
    }
    for relative, (content, meta) in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if meta is not None:
            path.with_name(path.name.replace(".jsonl", ".meta.json")).write_text(json.dumps(meta))
        Harness(first.config, first.conn, first.parser, path).reconcile()
    yield first.conn, first.config
    first.conn.close()


def _ids(conn) -> dict[str, str]:
    rows = conn.execute("SELECT source_path, id FROM sessions").fetchall()
    return {Path(path).stem: session_id for path, session_id in rows}


def test_list_files_linked_sessions_under_an_indexed_root(indexed) -> None:
    conn, config = indexed
    ids = _ids(conn)

    default = list_sessions(source=None, since=None, project=None, config=config, conn=conn)
    assert {row.id: (row.subagent_count, row.root_session_id) for row in default} == {
        ids[ROOT_SID]: (2, None),
        ids["agent-lost1"]: (0, None),
    }

    every = list_sessions(
        source=None, since=None, project=None, subagents=True, config=config, conn=conn
    )
    assert {row.id: row.root_session_id for row in every} == {
        ids[ROOT_SID]: None,
        ids["agent-explore1"]: ids[ROOT_SID],
        ids["agent-plan1"]: ids[ROOT_SID],
        ids["agent-lost1"]: None,
    }
    explore = next(row for row in every if row.id == ids["agent-explore1"])
    assert explore.lineage is not None
    assert (explore.lineage.agent_type, explore.lineage.agent_description) == (
        "Explore",
        "Find the parser",
    )


def test_source_id_selects_the_root_and_show_lists_its_subagents(indexed) -> None:
    conn, _ = indexed
    ids = _ids(conn)

    assert resolve_session_identifier(conn, ROOT_SID).id == ids[ROOT_SID]
    with StandaloneLifetime().lease(conn) as lease:
        root = resolve_session_header(ROOT_SID, conn=conn)
        root_id, count, subagents = session_tree_on(lease, root)
        assert (root_id, count) == (None, 2)
        assert [(s.id, s.agent_type) for s in subagents] == [
            (ids["agent-explore1"], "Explore"),
            (ids["agent-plan1"], None),
        ]

        child = resolve_session_header(ids["agent-plan1"], conn=conn)
        assert session_tree_on(lease, child) == (ids[ROOT_SID], 0, [])
        orphan = resolve_session_header(ids["agent-lost1"], conn=conn)
        assert session_tree_on(lease, orphan) == (None, 0, [])


def test_usage_counts_one_session_per_spawn_tree(indexed) -> None:
    conn, config = indexed
    stats = usage_by_source(config=config, conn=conn)
    assert sum(stat.session_count for stat in stats) == 2


def test_a_database_without_lineage_columns_reads_as_all_roots(indexed) -> None:
    """Read-only opens may precede the daemon's upgrade; reads degrade, not fail."""
    conn, config = indexed
    ids = _ids(conn)
    conn.execute(
        "CREATE TABLE identity AS SELECT id, source, source_path, source_session_id FROM sessions"
    )
    conn.execute("DROP TABLE sessions")
    conn.execute("ALTER TABLE identity RENAME TO sessions")

    rows = list_sessions(source=None, since=None, project=None, config=config, conn=conn)
    assert len(rows) == 4
    assert all(row.lineage is None and row.subagent_count == 0 for row in rows)
    with StandaloneLifetime().lease(conn) as lease:
        child = resolve_session_header(ids["agent-plan1"], conn=conn)
        assert child.lineage is None
        assert session_tree_on(lease, child) == (None, 0, [])
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_session_identifier(conn, ROOT_SID)
    # Claude subagents carry their root's source session id, so trees still count once.
    assert sum(stat.session_count for stat in usage_by_source(config=config, conn=conn)) == 2
