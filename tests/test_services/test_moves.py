"""A transcript indexed again after it moved (REQ-INDEX-027)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from recall.cli.app import app
from recall.core.config import AppConfig
from recall.core.models import TailFacts
from recall.core.transcripts import plain_transcript_path
from recall.db import connect
from recall.db.source_files import SourceCatalog
from recall.parsers import SessionParser, get_parser
from recall.parsers.claude_code import ClaudeCodeParser
from recall.services.coordinator import PreparedRawCycle, persist_prepared_inventory
from recall.services.indexer import (
    _apply_stored_contexts,
    _load_session_write_snapshot,
    _write_session,
)
from recall.services.reconciler import iter_inventory_batches, stat_signature
from typer.testing import CliRunner

FIXTURE = Path(__file__).parents[1] / "fixtures" / "claude_code" / "live_end_turn.jsonl"
LINES = FIXTURE.read_text(encoding="utf-8").splitlines(keepends=True)


def _other_lines(n: int) -> list[str]:
    """The fixture conversation as another transcript: the same shape, started on another day."""
    return [line.replace("2026-09-07T", f"2026-08-{n + 1:02d}T") for line in LINES]


def _transcript(home: Path, project: str, lines: list[str]) -> Path:
    path = home / ".claude" / "projects" / project / "3f1c.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _index(
    conn: duckdb.DuckDBPyConnection,
    path: Path,
    *,
    root: Path | None = None,
    parser: SessionParser | None = None,
) -> str:
    """Index a transcript and commit its catalog prefix, as reconciliation does."""
    path = path.resolve()
    data = path.read_bytes()
    parser = parser or ClaudeCodeParser()
    capture = parser.parse(path)
    session = capture.session
    snapshot = _load_session_write_snapshot(conn, session, host=None, capture=capture)
    _apply_stored_contexts(conn, session, snapshot=snapshot)
    _write_session(
        conn, session, last_byte_offset=len(data), tail_facts=TailFacts(), snapshot=snapshot
    )
    catalog = SourceCatalog(conn, clock=lambda: 0.0)
    generation = catalog.observe(
        parser.source.value, str(root or path.parent), str(path), stat_signature(path)
    )
    catalog.acknowledge(
        parser.source.value, str(path), generation, len(data), hashlib.sha256(data).hexdigest()
    )
    return session.id


def _walk(
    conn: duckdb.DuckDBPyConnection,
    root: Path,
    on_enrichment_change: Callable[[], None] | None = None,
    *,
    parser: SessionParser | None = None,
) -> None:
    """Run one complete inventory walk of `root`, as the daemon's reconciliation does."""
    parser = parser or ClaudeCodeParser()
    generation, _ = persist_prepared_inventory(
        PreparedRawCycle(parser, root, None),
        None,
        conn=conn,
        on_enrichment_change=on_enrichment_change,
    )
    for event in iter_inventory_batches(parser, root, clock=lambda: 0.0):
        generation, _ = persist_prepared_inventory(
            PreparedRawCycle(parser, root, event),
            generation,
            conn=conn,
            on_enrichment_change=on_enrichment_change,
        )


def _session_ids(conn: duckdb.DuckDBPyConnection) -> list[str]:
    return [str(row[0]) for row in conn.execute("SELECT id FROM sessions ORDER BY id").fetchall()]


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path.resolve() / "data"))
    (tmp_path / "data").mkdir()
    connection = connect(AppConfig.load())
    yield connection
    connection.close()


def test_a_moved_and_extended_transcript_supersedes_its_old_row(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    old_path = _transcript(home, "-work-old", LINES[:3])
    old_id = _index(conn, old_path)
    conn.execute(
        "INSERT INTO usage_events (id, source, source_session_id, session_id, harvested_at) "
        "VALUES ('u1', 'claude_code', 'live-end-turn', ?, now())",
        [old_id],
    )

    new_path = _transcript(home, "-work-new", LINES)
    old_path.unlink()
    new_id = _index(conn, new_path)

    assert _session_ids(conn) == [new_id]
    assert conn.execute("SELECT session_id FROM usage_events").fetchone() == (new_id,)


def test_a_move_is_found_among_many_transcripts_with_its_name(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    others = [
        _index(conn, _transcript(home, f"-work-other-{n}", _other_lines(n))) for n in range(20)
    ]
    old_path = _transcript(home, "-work-old", LINES[:3])
    _index(conn, old_path)

    new_path = _transcript(home, "-work-new", LINES)
    old_path.unlink()
    new_id = _index(conn, new_path)

    assert _session_ids(conn) == sorted([*others, new_id])


def test_a_move_keeps_existing_keyword_identities(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    old_path = _transcript(home, "-work-old", LINES[:3])
    old_id = _index(conn, old_path)
    old_messages = {
        str(row[0])
        for row in conn.execute("SELECT id FROM messages WHERE session_id = ?", [old_id]).fetchall()
    }

    new_path = _transcript(home, "-work-new", LINES)
    old_path.unlink()
    _index(conn, new_path)

    queued = {
        str(row[0])
        for row in conn.execute(
            "SELECT id FROM fts_sidecar_pending WHERE kind = 'message' AND op = 'delete'"
        ).fetchall()
    }
    assert old_messages
    assert queued == set()
    assert old_messages <= {row[0] for row in conn.execute("SELECT id FROM messages").fetchall()}


def test_a_successor_indexed_short_of_the_old_history_keeps_the_old_row(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    old_path = _transcript(home, "-work-old", LINES)
    old_id = _index(conn, old_path)

    new_path = _transcript(home, "-work-new", LINES[:1])
    partial = ClaudeCodeParser().parse(new_path).session
    _transcript(home, "-work-new", LINES)
    old_path.unlink()
    _write_session(
        conn,
        partial,
        last_byte_offset=len(LINES[0].encode()),
        tail_facts=TailFacts(),
    )

    assert _session_ids(conn) == sorted([old_id, partial.id])


def test_a_moved_transcript_keeps_its_stored_summaries(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    old_path = _transcript(home, "-work-old", LINES[:3])
    old_id = _index(conn, old_path)
    conn.execute(
        "UPDATE message_state SET context_mode = 'llm-local', context_text = 'summary'"
        " WHERE message_id IN (SELECT id FROM messages WHERE session_id = ?)",
        [old_id],
    )

    new_path = _transcript(home, "-work-new", LINES)
    old_path.unlink()
    new_id = _index(conn, new_path)

    carried = conn.execute(
        "SELECT m.idx, ms.context_mode, ms.context_text FROM messages m"
        " JOIN message_state ms ON ms.message_id = m.id"
        " WHERE m.session_id = ? AND ms.context_mode = 'llm-local' ORDER BY m.idx",
        [new_id],
    ).fetchall()
    message_count = len(ClaudeCodeParser().parse(new_path).session.messages)
    assert carried
    assert len(carried) < message_count
    assert all(text == "summary" for _, _, text in carried)


@pytest.mark.parametrize("predecessor_context", [None, "proven summary"])
def test_a_byte_different_departed_sibling_cannot_donate_context(
    tmp_path: Path,
    conn: duckdb.DuckDBPyConnection,
    predecessor_context: str | None,
) -> None:
    root = tmp_path.resolve() / "custom"
    # Whitespace changes the committed bytes without changing parsed messages.
    sibling = _custom_transcript(root, "sibling/a.jsonl", [" " + line for line in LINES[:3]])
    sibling_id = _index(conn, sibling, root=root)
    conn.execute(
        "UPDATE message_state SET context_mode = 'llm-local', context_text = 'sibling summary'"
        " WHERE message_id IN (SELECT id FROM messages WHERE session_id = ?)",
        [sibling_id],
    )
    original = _custom_transcript(root, "original/a.jsonl", LINES[:3])
    original_id = _index(conn, original, root=root)
    if predecessor_context is not None:
        conn.execute(
            "UPDATE message_state SET context_mode = 'llm-local', context_text = ?"
            " WHERE message_id IN (SELECT id FROM messages WHERE session_id = ?)",
            [predecessor_context, original_id],
        )
    sibling.unlink()
    original.unlink()
    successor = _index(conn, _custom_transcript(root, "moved/a.jsonl", LINES), root=root)

    assert _session_ids(conn) == sorted([sibling_id, successor])
    contexts = conn.execute(
        "SELECT DISTINCT ms.context_text FROM message_state ms"
        " JOIN messages m ON m.id = ms.message_id"
        " WHERE m.session_id = ? AND ms.context_mode = 'llm-local'",
        [successor],
    ).fetchall()
    assert contexts == ([] if predecessor_context is None else [("proven summary",)])
    assert conn.execute(
        "SELECT DISTINCT ms.context_text FROM message_state ms"
        " JOIN messages m ON m.id = ms.message_id WHERE m.session_id = ?",
        [sibling_id],
    ).fetchall() == [("sibling summary",)]


def test_a_copy_whose_original_remains_keeps_both_rows(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    old_path = _transcript(home, "-work-old", LINES)
    old_id = _index(conn, old_path)

    new_id = _index(conn, _transcript(home, "-work-new", LINES))

    assert _session_ids(conn) == sorted([old_id, new_id])


def test_a_divergent_transcript_keeps_both_rows(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    old_path = _transcript(home, "-work-old", LINES[:3])
    old_id = _index(conn, old_path)

    new_path = _transcript(home, "-work-new", [LINES[0], *LINES[2:]])
    old_path.unlink()
    new_id = _index(conn, new_path)

    assert _session_ids(conn) == sorted([old_id, new_id])


def test_a_copy_indexed_before_its_original_is_deleted_supersedes_it_when_the_delete_is_seen(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    projects = home / ".claude" / "projects"
    old_path = _transcript(home, "-work-old", LINES[:3])
    old_id = _index(conn, old_path, root=projects)
    conn.execute(
        "UPDATE message_state SET context_mode = 'llm-local', context_text = 'summary'"
        " WHERE message_id IN (SELECT id FROM messages WHERE session_id = ?)",
        [old_id],
    )
    conn.execute(
        "INSERT INTO usage_events (id, source, source_session_id, session_id, harvested_at) "
        "VALUES ('u1', 'claude_code', 'live-end-turn', ?, now())",
        [old_id],
    )
    old_messages = {
        str(row[0])
        for row in conn.execute("SELECT id FROM messages WHERE session_id = ?", [old_id]).fetchall()
    }
    new_path = _transcript(home, "-work-new", LINES)
    new_id = _index(conn, new_path, root=projects)
    assert _session_ids(conn) == sorted([old_id, new_id])

    old_path.unlink()
    _walk(conn, projects)

    assert _session_ids(conn) == [old_id]
    assert new_id not in _session_ids(conn)
    assert conn.execute("SELECT session_id FROM usage_events").fetchone() == (old_id,)
    deleted = {
        str(row[0])
        for row in conn.execute(
            "SELECT id FROM fts_sidecar_pending WHERE kind = 'message' AND op = 'delete'"
        ).fetchall()
    }
    assert old_messages <= deleted
    carried = conn.execute(
        "SELECT COUNT(*) FROM messages m JOIN message_state ms ON ms.message_id = m.id"
        " WHERE m.session_id = ? AND ms.context_text = 'summary'",
        [old_id],
    ).fetchone()
    assert carried is not None and carried[0] > 0


def _missing_paths(conn: duckdb.DuckDBPyConnection) -> set[str]:
    rows = conn.execute("SELECT source_path FROM source_files WHERE missing").fetchall()
    return {str(row[0]) for row in rows}


def test_a_failed_supersession_is_given_up_while_the_walk_finishes(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    import recall.services.moves as moves_module
    from recall.services.rpc_server import RpcServer

    config = AppConfig.load()
    server = RpcServer(replace(config, fts=replace(config.fts, backend="duckdb")))
    server._conn = conn

    home = tmp_path.resolve()
    projects = home / ".claude" / "projects"
    gone = [
        _transcript(home, f"-work-gone-{n}", _other_lines(n)[:2]).rename(
            projects / f"-work-gone-{n}" / f"gone-{n}.jsonl"
        )
        for n in range(3)
    ]
    for path in gone:
        _index(conn, path, root=projects)
    for n in range(4):
        kept = _transcript(home, f"-work-kept-{n}", _other_lines(10 + n))
        _index(conn, kept.rename(kept.with_name(f"kept-{n}.jsonl")), root=projects)
    old_path = _transcript(home, "-work-old", LINES)
    old_id = _index(conn, old_path, root=projects)
    new_id = _index(conn, _transcript(home, "-work-new", LINES), root=projects)
    for path in [*gone, old_path]:
        path.unlink()
    attempts: list[tuple[str, ...]] = []

    def fail(*_args: object, predecessor_ids: tuple[str, ...], **_kwargs: object) -> None:
        attempts.append(predecessor_ids)
        raise RuntimeError("supersede failed")

    monkeypatch.setattr(moves_module, "supersede", fail)

    async def walk() -> None:
        async with server._write_lock:
            _walk(conn, projects, server._invalidate_moved_history)
            _walk(conn, projects, server._invalidate_moved_history)

    asyncio.run(walk())

    assert attempts == [(old_id,)]
    assert _missing_paths(conn) == {str(path) for path in [*gone, old_path]}
    assert conn.execute("SELECT scan_complete FROM reconciliation_roots").fetchone() == (True,)
    assert old_id in _session_ids(conn) and new_id in _session_ids(conn)
    assert server._keyword_dirty is False


@pytest.mark.parametrize(
    ("bound", "value", "expected"),
    [
        ("_MOVE_PROOFS_PER_WALK_MAX", 2, [8, 6, 5]),
        # The byte budget is spent by the first proof, which still proceeds.
        ("_MOVE_PROOF_BYTES_PER_WALK_MAX", 1, [9, 8, 7]),
    ],
)
def test_a_walk_supersedes_a_bounded_number_and_leaves_the_rest_to_the_next(
    tmp_path: Path,
    conn: duckdb.DuckDBPyConnection,
    monkeypatch: pytest.MonkeyPatch,
    bound: str,
    value: int,
    expected: list[int],
) -> None:
    import recall.services.coordinator as coordinator_module

    monkeypatch.setattr(coordinator_module, bound, value)
    home = tmp_path.resolve()
    projects = home / ".claude" / "projects"
    olds: list[Path] = []
    for n in range(5):
        old_path = _transcript(home, "-work-old", _other_lines(n))
        olds.append(old_path.rename(old_path.with_name(f"s{n}.jsonl")))
        _index(conn, olds[-1], root=projects)
        new_path = _transcript(home, "-work-new", _other_lines(n))
        _index(conn, new_path.rename(new_path.with_name(f"s{n}.jsonl")), root=projects)
    for path in olds:
        path.unlink()

    counts = []
    for _ in range(3):
        _walk(conn, projects)
        counts.append(len(_session_ids(conn)))

    assert counts == expected
    assert _missing_paths(conn) == {str(path) for path in olds[: 10 - expected[-1]]}


def test_a_walk_that_finds_nothing_supersedes_nothing(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    projects = home / ".claude" / "projects"
    old_path = _transcript(home, "-work-old", LINES)
    old_id = _index(conn, old_path, root=projects)
    elsewhere = home / "volume" / ".claude" / "projects" / "-work-old" / "3f1c.jsonl"
    elsewhere.parent.mkdir(parents=True)
    shutil.copyfile(old_path, elsewhere)
    new_id = _index(conn, elsewhere, root=elsewhere.parents[1])

    old_path.unlink()
    old_path.parent.rmdir()
    _walk(conn, projects)

    assert _session_ids(conn) == sorted([old_id, new_id])
    assert _missing_paths(conn) == {str(old_path)}


def test_a_walk_that_loses_most_of_its_root_supersedes_nothing(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    projects = home / ".claude" / "projects"
    kept = _transcript(home, "-work-kept", LINES)
    _index(conn, kept.rename(kept.with_name("kept.jsonl")), root=projects)
    old_path = _transcript(home, "-work-old", LINES)
    old_id = _index(conn, old_path, root=projects)
    other = _transcript(home, "-work-other", LINES[:2])
    other = other.rename(other.with_name("other.jsonl"))
    _index(conn, other, root=projects)
    elsewhere = home / "volume" / ".claude" / "projects" / "-work-old" / "3f1c.jsonl"
    elsewhere.parent.mkdir(parents=True)
    shutil.copyfile(old_path, elsewhere)
    new_id = _index(conn, elsewhere, root=elsewhere.parents[1])

    old_path.unlink()
    other.unlink()
    _walk(conn, projects)

    assert old_id in _session_ids(conn) and new_id in _session_ids(conn)
    assert _missing_paths(conn) == {str(old_path), str(other)}


def test_a_delete_seen_before_the_copy_is_indexed_supersedes_on_insert(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    projects = home / ".claude" / "projects"
    old_path = _transcript(home, "-work-old", LINES[:3])
    _index(conn, old_path, root=projects)
    new_path = _transcript(home, "-work-new", LINES)
    old_path.unlink()
    _walk(conn, projects)

    new_id = _index(conn, new_path, root=projects)

    assert _session_ids(conn) == [new_id]


def test_a_seen_delete_keeps_the_row_a_lagging_copy_does_not_hold(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    projects = home / ".claude" / "projects"
    old_path = _transcript(home, "-work-old", LINES)
    old_id = _index(conn, old_path, root=projects)
    new_path = _transcript(home, "-work-new", LINES[:1])
    new_id = _index(conn, new_path, root=projects)
    _transcript(home, "-work-new", LINES)

    old_path.unlink()
    _walk(conn, projects)

    assert _session_ids(conn) == sorted([old_id, new_id])


def test_a_seen_delete_with_two_surviving_copies_keeps_every_row(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    projects = home / ".claude" / "projects"
    old_path = _transcript(home, "-work-old", LINES)
    ids = [_index(conn, old_path, root=projects)]
    ids += [_index(conn, _transcript(home, f"-work-{n}", LINES), root=projects) for n in range(2)]

    old_path.unlink()
    _walk(conn, projects)

    assert _session_ids(conn) == sorted(ids)


def test_supersede_moved_reports_then_removes_existing_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path.resolve()
    monkeypatch.setenv("RECALL_DATA_DIR", str(home / "data"))
    (home / "data").mkdir()
    old_path = _transcript(home, "-work-old", LINES)
    new_path = home / ".claude" / "projects" / "-work-new" / "3f1c.jsonl"
    new_path.parent.mkdir(parents=True)
    shutil.copyfile(old_path, new_path)
    conn = connect(AppConfig.load())
    try:
        old_id = _index(conn, old_path)
        _index(conn, new_path)
    finally:
        conn.close()
    old_path.unlink()

    report = CliRunner().invoke(app, ["db", "supersede-moved", "--json"])
    applied = CliRunner().invoke(app, ["db", "supersede-moved", "--apply", "--json"])
    again = CliRunner().invoke(app, ["db", "supersede-moved", "--json"])

    assert report.exit_code == 0, report.output
    assert json.loads(report.stdout)["moves"] == [
        {"predecessor_path": str(old_path), "successor_path": str(new_path)}
    ]
    assert json.loads(applied.stdout)["superseded"] == 1
    assert json.loads(applied.stdout)["census"]["phase"] == "pre_apply"
    assert json.loads(applied.stdout)["census"]["classes"] == {"proven_move": 1}
    assert json.loads(again.stdout)["census"]["phase"] == "report"
    assert json.loads(again.stdout)["census"]["classes"] == {}
    assert json.loads(again.stdout)["moves"] == []
    conn = connect(AppConfig.load())
    try:
        assert _session_ids(conn) == [old_id]
        assert conn.execute(
            "SELECT COUNT(*) FROM fts_sidecar_pending WHERE op = 'delete'"
        ).fetchone() == (0,)
    finally:
        conn.close()


def test_supersede_moved_keeps_the_stored_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path.resolve()
    monkeypatch.setenv("RECALL_DATA_DIR", str(home / "data"))
    (home / "data").mkdir()
    old_path = _transcript(home, "-work-old", LINES[:3])
    new_path = _transcript(home, "-work-new", LINES)
    conn = connect(AppConfig.load())
    try:
        old_id = _index(conn, old_path)
        _index(conn, new_path)
        conn.execute(
            "UPDATE message_state SET context_mode = 'llm-local', context_text = 'summary'"
            " WHERE message_id IN (SELECT id FROM messages WHERE session_id = ?)",
            [old_id],
        )
    finally:
        conn.close()
    old_path.unlink()

    applied = CliRunner().invoke(app, ["db", "supersede-moved", "--apply", "--json"])

    assert json.loads(applied.stdout)["superseded"] == 1
    conn = connect(AppConfig.load())
    try:
        carried = conn.execute(
            "SELECT m.id, ms.fts_content FROM messages m"
            " JOIN message_state ms ON ms.message_id = m.id"
            " WHERE m.session_id = ? AND ms.context_mode = 'llm-local'"
            " AND ms.context_text = 'summary'",
            [old_id],
        ).fetchall()
        queued = conn.execute(
            "SELECT id FROM fts_sidecar_pending WHERE kind = 'message' AND op = 'upsert'"
        ).fetchall()
        assert carried
        assert all(text.startswith("summary") for _, text in carried)
        assert {message_id for message_id, _ in carried} <= {row[0] for row in queued}
    finally:
        conn.close()


def test_supersede_moved_keeps_a_row_whose_survivor_lags_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path.resolve()
    monkeypatch.setenv("RECALL_DATA_DIR", str(home / "data"))
    (home / "data").mkdir()
    old_path = _transcript(home, "-work-old", LINES)
    new_path = _transcript(home, "-work-new", LINES[:1])
    conn = connect(AppConfig.load())
    try:
        _index(conn, old_path)
        _index(conn, new_path)
    finally:
        conn.close()
    _transcript(home, "-work-new", LINES)
    old_path.unlink()

    report = CliRunner().invoke(app, ["db", "supersede-moved", "--json"])

    assert report.exit_code == 0, report.output
    assert json.loads(report.stdout)["moves"] == []


def test_supersede_moved_rejects_unknown_fields_before_applying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path.resolve()
    monkeypatch.setenv("RECALL_DATA_DIR", str(home / "data"))
    (home / "data").mkdir()
    old_path = _transcript(home, "-work-old", LINES)
    new_path = _transcript(home, "-work-new", LINES)
    conn = connect(AppConfig.load())
    try:
        old_id = _index(conn, old_path)
        new_id = _index(conn, new_path)
    finally:
        conn.close()
    old_path.unlink()

    result = CliRunner().invoke(
        app, ["db", "supersede-moved", "--apply", "--json", "--fields", "bogus"]
    )

    assert result.exit_code == 2, result.output
    conn = connect(AppConfig.load())
    try:
        assert _session_ids(conn) == sorted([old_id, new_id])
    finally:
        conn.close()


def test_a_failed_successor_insert_keeps_the_old_row(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    import recall.services.indexer as indexer_module

    home = tmp_path.resolve()
    old_path = _transcript(home, "-work-old", LINES[:3])
    old_id = _index(conn, old_path)
    new_path = _transcript(home, "-work-new", LINES)
    old_path.unlink()

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("insert failed")

    monkeypatch.setattr(indexer_module, "insert_tool_calls", fail)
    with pytest.raises(RuntimeError):
        _write_session(conn, ClaudeCodeParser().parse(new_path).session, tail_facts=TailFacts())

    assert _session_ids(conn) == [old_id]
    old_messages = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?", [old_id]
    ).fetchone()
    assert old_messages is not None and old_messages[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone() == (0,)


def test_inventory_supersession_wakes_an_idle_embed_snapshot(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    import asyncio

    from recall.services.embed_phase import EmbedPhaseState, prepare_embed_cycle
    from recall.services.rpc_server import RpcServer

    projects = tmp_path / ".claude" / "projects"
    old_path = _transcript(tmp_path, "-work-old", LINES[:3])
    old_id = _index(conn, old_path, root=projects)
    conn.execute(
        "UPDATE message_state SET context_mode='llm-local', context_text='summary'"
        " WHERE message_id IN (SELECT id FROM messages WHERE session_id=?)",
        [old_id],
    )
    new_path = _transcript(tmp_path, "-work-new", LINES)
    new_id = _index(conn, new_path, root=projects)
    conn.execute(
        "INSERT INTO message_embeddings(message_id) SELECT id FROM messages WHERE session_id=?",
        [new_id],
    )
    conn.execute(
        "INSERT INTO tool_call_embeddings(tool_call_id) "
        "SELECT id FROM tool_calls WHERE session_id=?",
        [new_id],
    )
    conn.execute(
        "UPDATE session_state SET file_mtime = CASE WHEN session_id=? THEN 1000 ELSE 0 END",
        [old_id],
    )
    config = AppConfig.load()
    state = EmbedPhaseState(_config=config)
    server = RpcServer(config)
    server._conn = conn

    async def snapshot():
        return await server._writer_call(
            lambda: prepare_embed_cycle(
                config,
                state,
                conn=conn,
                now=1200,
                work_generation=server._write_lock.enrichment_generation,
            ),
            "embed snapshot",
            affects_enrichment=False,
        )

    assert asyncio.run(snapshot()).pending.total == 0
    old_path.unlink()
    asyncio.run(
        server._writer_call(
            lambda: _walk(conn, projects, server._write_lock.invalidate_enrichment),
            "inventory batch",
            affects_enrichment=False,
        )
    )
    pending = asyncio.run(snapshot()).pending
    assert pending.total > 0
    assert pending.session_ids == [old_id]
    assert _session_ids(conn) == [old_id]


def _custom_transcript(root: Path, relative: str, lines: list[str]) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_a_move_under_a_custom_root_supersedes_its_old_row(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    root = tmp_path.resolve() / "lanes"
    old_path = _custom_transcript(root, "old/3f1c.jsonl", LINES[:3])
    _index(conn, old_path, root=root)

    new_path = root / "new" / "3f1c.jsonl"
    new_path.parent.mkdir()
    old_path.rename(new_path)
    _custom_transcript(root, "new/3f1c.jsonl", LINES)
    new_id = _index(conn, new_path, root=root)

    assert _session_ids(conn) == [new_id]


def test_a_copy_under_a_custom_root_supersedes_its_original_when_the_delete_is_seen(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    root = tmp_path.resolve() / "lanes"
    old_path = _custom_transcript(root, "old/3f1c.jsonl", LINES[:3])
    old_id = _index(conn, old_path, root=root)
    new_path = _custom_transcript(root, "new/3f1c.jsonl", LINES)
    _index(conn, new_path, root=root)

    old_path.unlink()
    _walk(conn, root)

    assert _session_ids(conn) == [old_id]


def test_a_renamed_transcript_supersedes_its_old_row(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    old_path = _transcript(home, "-work", LINES)
    _index(conn, old_path)

    new_path = old_path.with_name("9a2e.jsonl")
    old_path.rename(new_path)
    new_id = _index(conn, new_path)

    assert _session_ids(conn) == [new_id]


def test_unrelated_transcripts_with_one_name_under_a_custom_root_keep_their_rows(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    root = tmp_path.resolve() / "lanes"
    first = _custom_transcript(root, "a/3f1c.jsonl", LINES[:3])
    first_id = _index(conn, first, root=root)
    second = _custom_transcript(root, "b/3f1c.jsonl", [LINES[0], *LINES[2:]])

    first.unlink()
    second_id = _index(conn, second, root=root)
    _walk(conn, root)

    assert _session_ids(conn) == sorted([first_id, second_id])


def test_a_root_replaced_by_a_symlink_to_its_moved_content_keeps_one_row(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    home = tmp_path.resolve()
    root = home / "roots" / "claude"
    old_path = _custom_transcript(root, "-work/3f1c.jsonl", LINES)
    _index(conn, old_path, root=root)

    moved = home / "outside" / "claude"
    moved.parent.mkdir()
    root.rename(moved)
    root.symlink_to(moved, target_is_directory=True)
    new_id = _index(conn, root / "-work" / "3f1c.jsonl", root=moved)
    _walk(conn, moved)

    assert _session_ids(conn) == [new_id]
    assert conn.execute("SELECT source_path FROM sessions").fetchone() == (
        str(moved / "-work" / "3f1c.jsonl"),
    )


def test_a_copy_among_transcripts_started_together_supersedes_only_its_original(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    root = tmp_path.resolve() / "lanes"
    sibling_lines = [LINES[0], LINES[1].replace("Running it now.", "Running it later."), *LINES[2:]]
    original = _custom_transcript(root, "day/a.jsonl", LINES)
    original_id = _index(conn, original, root=root)
    sibling_id = _index(conn, _custom_transcript(root, "day/b.jsonl", sibling_lines), root=root)
    _index(conn, _custom_transcript(root, "elsewhere/a.jsonl", LINES), root=root)

    original.unlink()
    _walk(conn, root)

    assert _session_ids(conn) == sorted([sibling_id, original_id])


def test_supersede_moved_collapses_renamed_and_symlinked_duplicates_left_by_older_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path.resolve()
    monkeypatch.setenv("RECALL_DATA_DIR", str(home / "data"))
    (home / "data").mkdir()
    lanes = home / "lanes"
    renamed_from = _custom_transcript(lanes, "a/3f1c.jsonl", LINES)
    renamed_to = _custom_transcript(lanes, "a/9a2e.jsonl", LINES)
    root = home / "roots" / "claude"
    aliased = _custom_transcript(root, "-work/3f1c.jsonl", _other_lines(0))
    moved = home / "outside" / "claude"
    _custom_transcript(moved, "-work/3f1c.jsonl", _other_lines(0))
    conn = connect(AppConfig.load())
    try:
        # Each copy is indexed while its original still exists, as a build that
        # found candidates by path left them.
        renamed_id = _index(conn, renamed_from, root=lanes)
        _index(conn, renamed_to, root=lanes)
        moved_id = _index(conn, aliased, root=root)
        _index(conn, moved / "-work" / "3f1c.jsonl", root=moved)
        assert len(_session_ids(conn)) == 4
    finally:
        conn.close()
    renamed_from.unlink()
    shutil.rmtree(root)
    root.symlink_to(moved, target_is_directory=True)

    report = CliRunner().invoke(app, ["db", "supersede-moved", "--json"])
    applied = CliRunner().invoke(app, ["db", "supersede-moved", "--apply", "--json"])

    assert report.exit_code == 0, report.output
    assert sorted(json.loads(report.stdout)["moves"], key=lambda move: move["successor_path"]) == [
        {"predecessor_path": str(renamed_from), "successor_path": str(renamed_to)},
        {"predecessor_path": str(aliased), "successor_path": str(moved / "-work" / "3f1c.jsonl")},
    ]
    assert json.loads(applied.stdout)["superseded"] == 2
    conn = connect(AppConfig.load())
    try:
        assert _session_ids(conn) == sorted([renamed_id, moved_id])
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("source", "fixture", "relative"),
    [
        ("claude_code", "claude_code/live_end_turn.jsonl", "project/a.jsonl"),
        ("codex", "codex/session1/rollout.jsonl", "day/rollout.jsonl"),
        (
            "kimi_code",
            "kimi_code/session1/agents/main/wire.jsonl",
            "project/a/agents/main/wire.jsonl",
        ),
    ],
)
@pytest.mark.parametrize("symlinked", [False, True], ids=["custom-root", "initial-symlink"])
def test_configured_root_move_preserves_one_session(
    tmp_path: Path,
    conn: duckdb.DuckDBPyConnection,
    source: str,
    fixture: str,
    relative: str,
    symlinked: bool,
) -> None:
    from recall.core.types import parse_source

    root = tmp_path.resolve() / "custom"
    root.mkdir()
    configured = tmp_path / "configured"
    if symlinked:
        configured.symlink_to(root, target_is_directory=True)
    else:
        configured = root
    original = root / relative
    original.parent.mkdir(parents=True)
    original.write_bytes((FIXTURE.parents[1] / fixture).read_bytes())
    parser = get_parser(parse_source(source))
    parser.roots = (root,)
    _index(conn, configured / relative, root=root, parser=parser)

    moved = root / "moved" / relative
    moved.parent.mkdir(parents=True)
    original.rename(moved)
    successor = _index(conn, moved, root=root, parser=parser)

    assert _session_ids(conn) == [successor]
    assert conn.execute("SELECT source_path FROM sessions").fetchone() == (str(moved),)


def test_moving_one_subagent_preserves_its_parent_and_sibling(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    root = tmp_path.resolve() / "custom"
    parent = _custom_transcript(root, "project/parent.jsonl", LINES)
    sibling_lines = [line.replace("run the test suite", "inspect auth") for line in LINES]
    sibling = _custom_transcript(root, "project/parent/subagents/agent-b.jsonl", sibling_lines)
    child_lines = [line.replace("run the test suite", "inspect payments") for line in LINES]
    child = _custom_transcript(root, "project/parent/subagents/agent-a.jsonl", child_lines)
    kept = [_index(conn, path, root=root) for path in (parent, sibling)]
    _index(conn, child, root=root)

    moved = child.with_name("agent-renamed.jsonl")
    child.rename(moved)
    successor = _index(conn, moved, root=root)
    _walk(conn, root)

    assert _session_ids(conn) == sorted([*kept, successor])
    assert conn.execute("SELECT COUNT(DISTINCT source_session_id) FROM sessions").fetchone() == (1,)


def test_a_rename_between_preparation_and_commit_preserves_history_until_reobserved(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    from recall.services.coordinator import (
        capture_path,
        commit_prepared_raw_sources,
        observe_path,
        prepare_raw_sources,
    )

    root = tmp_path.resolve() / "custom"
    original = _custom_transcript(root, "project/a.jsonl", LINES)
    predecessor = _index(conn, original, root=root)
    staged = original.with_name("staged.jsonl")
    original.rename(staged)
    parser = ClaudeCodeParser(roots=(root,))
    item = observe_path(parser, capture_path(parser, staged), conn=conn)
    prepared = prepare_raw_sources((item,), {"claude_code": parser})
    assert prepared[0].result is not None
    final = staged.with_name("final.jsonl")
    staged.rename(final)

    assert commit_prepared_raw_sources(prepared, AppConfig.load(), conn=conn).changed == 0
    assert _session_ids(conn) == [predecessor]
    final_item = observe_path(parser, capture_path(parser, final), conn=conn)
    prepared = prepare_raw_sources((final_item,), {"claude_code": parser})
    assert commit_prepared_raw_sources(prepared, AppConfig.load(), conn=conn).changed == 1
    _walk(conn, root)
    assert _session_ids(conn) == [predecessor]


def test_inserting_a_second_holder_keeps_an_ambiguous_predecessor(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    root = tmp_path.resolve() / "custom"
    original = _custom_transcript(root, "project/a.jsonl", LINES)
    predecessor = _index(conn, original, root=root)
    copy = _custom_transcript(root, "copy/a.jsonl", LINES)
    copy_id = _index(conn, copy, root=root)
    moved = original.with_name("renamed.jsonl")
    original.rename(moved)
    moved_id = _index(conn, moved, root=root)
    _walk(conn, root)

    assert _session_ids(conn) == sorted([predecessor, copy_id, moved_id])


def test_walk_defers_a_timestamp_collision_group_that_would_exceed_its_proof_count(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    from recall.core.types import default_session_host
    from recall.services.moves import plan_vanished_moves

    root = tmp_path.resolve() / "custom"
    originals = []
    for n in range(3):
        lines = [line.replace("run the test suite", f"inspect project {n}") for line in LINES]
        original = _custom_transcript(root, f"old/{n}.jsonl", lines)
        _index(conn, original, root=root)
        copy = _custom_transcript(root, f"new/{n}.jsonl", lines)
        _index(conn, copy, root=root)
        originals.append(original)
    for original in originals:
        original.unlink()

    result = plan_vanished_moves(
        conn,
        source="claude_code",
        vanished_paths=[str(path) for path in originals],
        host=default_session_host(),
        proofs_max=4,
        proof_bytes_max=16 * 1024 * 1024,
    )

    assert result.proofs <= 4
    assert len(result.plans) == 1
    assert len(result.deferred_paths) == 2


@pytest.mark.parametrize(
    ("source", "fixture", "relative"),
    [
        ("claude_code", "claude_code/live_end_turn.jsonl", "project/a.jsonl"),
        ("codex", "codex/session1/rollout.jsonl", "day/rollout.jsonl"),
        ("pi_agent", "pi_agent/session1.jsonl", "project/a.jsonl"),
        ("grok", "grok/session1.jsonl", "project/session/chat_history.jsonl"),
        (
            "kimi_code",
            "kimi_code/session1/agents/main/wire.jsonl",
            "project/a/agents/main/wire.jsonl",
        ),
        ("codex", "codex/session1/rollout.jsonl", "day/rollout.jsonl.zst"),
    ],
)
@pytest.mark.parametrize("copy_then_delete", [False, True], ids=["rename", "copy-delete"])
def test_move_retains_session_and_child_ids_through_reparse(
    tmp_path: Path,
    conn: duckdb.DuckDBPyConnection,
    source: str,
    fixture: str,
    relative: str,
    copy_then_delete: bool,
) -> None:
    from recall.core.types import parse_source
    from recall.parsers.compressed import zstd
    from recall.services.coordinator import (
        capture_path,
        commit_prepared_raw_sources,
        observe_path,
        prepare_raw_sources,
    )

    compressed = relative.endswith(".zst")
    root = tmp_path.resolve() / "transcripts"
    original = root / "before" / relative
    original.parent.mkdir(parents=True)
    raw = (FIXTURE.parents[1] / fixture).read_bytes()
    original.write_bytes(zstd.compress(raw) if compressed else raw)
    parser = get_parser(parse_source(source))
    parser.roots = (root,)
    config = AppConfig.load()

    def publish(path: Path, *, changed: int = 1) -> None:
        item = observe_path(parser, capture_path(parser, path), conn=conn)
        prepared = prepare_raw_sources((item,), {source: parser})
        assert commit_prepared_raw_sources(prepared, config, conn=conn).changed == changed, (
            conn.execute("SELECT source_path, last_error FROM source_files").fetchall()
        )
        current = SourceCatalog(conn, clock=lambda: 0.0).get(
            source, str(plain_transcript_path(path))
        )
        assert current is not None and current.current

    publish(original)
    session_id = _session_ids(conn)[0]
    children = {
        table: conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
        for table in ("messages", "tool_calls")
    }
    moved = root / "after" / relative
    moved.parent.mkdir(parents=True)
    if copy_then_delete:
        shutil.copyfile(original, moved)
        original.unlink()
    else:
        original.rename(moved)
    publish(moved)

    assert _session_ids(conn) == [session_id]
    assert conn.execute(
        "SELECT source_path FROM sessions WHERE id = ?", [session_id]
    ).fetchone() == (str(plain_transcript_path(moved)),)
    for table, before in children.items():
        assert conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall() == before
    assert conn.execute(
        "SELECT session_id FROM source_files WHERE source_path = ?",
        [str(plain_transcript_path(moved))],
    ).fetchone() == (session_id,)

    # Explicit normalization must reuse the binding even without a resume proof.
    catalog = SourceCatalog(conn, clock=lambda: 0.0)
    catalog.force_reconcile_batch([(source, str(plain_transcript_path(moved)))])
    publish(moved, changed=0)
    assert _session_ids(conn) == [session_id]
    for table, before in children.items():
        assert conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall() == before


def test_reusing_a_moved_transcripts_original_path_allocates_a_separate_identity(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    root = tmp_path.resolve() / "custom"
    original = _custom_transcript(root, "old/a.jsonl", LINES)
    old_id = _index(conn, original, root=root)
    moved = root / "new/a.jsonl"
    moved.parent.mkdir()
    original.rename(moved)
    assert _index(conn, moved, root=root) == old_id

    replacement = _custom_transcript(root, "old/a.jsonl", _other_lines(1))
    replacement_id = _index(conn, replacement, root=root)
    assert replacement_id != old_id
    assert _session_ids(conn) == sorted([old_id, replacement_id])
    assert conn.execute("SELECT source_path FROM sessions WHERE id = ?", [old_id]).fetchone() == (
        str(moved),
    )
    assert _index(conn, replacement, root=root) == replacement_id


def _publish_codex(conn: duckdb.DuckDBPyConnection, path: Path) -> str:
    from recall.parsers.codex import CodexParser
    from recall.services.coordinator import (
        capture_path,
        commit_prepared_raw_sources,
        observe_path,
        prepare_raw_sources,
    )

    parser = CodexParser(roots=(path.parent,))
    item = observe_path(parser, capture_path(parser, path), conn=conn)
    prepared = prepare_raw_sources((item,), {"codex": parser})
    result = commit_prepared_raw_sources(prepared, AppConfig.load(), conn=conn)
    assert result.changed == 1, conn.execute(
        "SELECT source_path, last_error FROM source_files"
    ).fetchall()
    row = SourceCatalog(conn, clock=lambda: 0.0).get("codex", str(plain_transcript_path(path)))
    assert row is not None and row.current
    binding = conn.execute(
        "SELECT session_id FROM source_files WHERE source = 'codex' AND source_path = ?",
        [str(plain_transcript_path(path))],
    ).fetchone()
    assert binding is not None and binding[0] is not None
    return str(binding[0])


@pytest.mark.parametrize("surface", ["inventory", "maintenance"])
def test_compressed_copy_remains_independent_until_original_departs(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection, surface: str
) -> None:
    from recall.core.types import default_session_host
    from recall.parsers.codex import CodexParser
    from recall.parsers.compressed import zstd
    from recall.services.move_census import census_moved_duplicates
    from recall.services.moves import apply_moved_duplicates

    root = tmp_path.resolve() / "rollouts"
    root.mkdir()
    original = root / "rollout-old.jsonl.zst"
    original.write_bytes(
        zstd.compress((FIXTURE.parents[1] / "codex/session1/rollout.jsonl").read_bytes())
    )
    old_id = _publish_codex(conn, original)
    children = conn.execute(
        "SELECT id FROM messages WHERE session_id = ? ORDER BY idx", [old_id]
    ).fetchall()
    copy = root / "rollout-copy.jsonl.zst"
    shutil.copyfile(original, copy)
    copy_id = _publish_codex(conn, copy)

    assert old_id != copy_id
    assert _session_ids(conn) == sorted([old_id, copy_id])
    plans, census = census_moved_duplicates(conn, host=default_session_host())
    assert plans == []
    assert census.classes == {"surviving_copies": 1}

    original.unlink()
    if surface == "inventory":
        _walk(conn, root, parser=CodexParser(roots=(root,)))
    else:
        plans, census = census_moved_duplicates(conn, host=default_session_host())
        assert census.classes == {"proven_move": 1}
        assert len(plans) == 1
        apply_moved_duplicates(conn, plans, queue_sidecar_deletes=False)
    assert _session_ids(conn) == [old_id]
    assert conn.execute("SELECT source_path FROM sessions").fetchone() == (
        str(copy.with_suffix("")),
    )
    assert conn.execute("SELECT id FROM messages ORDER BY idx").fetchall() == children


def test_reusing_a_moved_path_for_compressed_content_allocates_a_distinct_identity(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    from recall.parsers.compressed import zstd

    raw = (FIXTURE.parents[1] / "codex/session1/rollout.jsonl").read_bytes()
    original = tmp_path.resolve() / "rollout-original.jsonl"
    original.write_bytes(raw)
    old_id = _publish_codex(conn, original)
    moved = original.with_name("rollout-moved.jsonl")
    original.rename(moved)
    assert _publish_codex(conn, moved) == old_id

    replacement = original.with_suffix(".jsonl.zst")
    replacement.write_bytes(zstd.compress(raw.replace(b"2024-01-16", b"2024-02-17")))
    replacement_id = _publish_codex(conn, replacement)
    assert replacement_id != old_id
    assert _session_ids(conn) == sorted([old_id, replacement_id])
    assert conn.execute("SELECT source_path FROM sessions WHERE id = ?", [old_id]).fetchone() == (
        str(moved),
    )


def test_unreadable_compressed_original_cannot_authorize_supersession(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    from recall.core.types import default_session_host
    from recall.parsers.compressed import zstd
    from recall.services.move_census import census_moved_duplicates

    original = tmp_path.resolve() / "rollout-original.jsonl.zst"
    original.write_bytes(
        zstd.compress((FIXTURE.parents[1] / "codex/session1/rollout.jsonl").read_bytes())
    )
    original_id = _publish_codex(conn, original)
    copy = original.with_name("rollout-copy.jsonl.zst")
    shutil.copyfile(original, copy)
    original.unlink()
    original.symlink_to(original)
    copy_id = _publish_codex(conn, copy)

    assert original_id != copy_id
    plans, census = census_moved_duplicates(conn, host=default_session_host())
    assert plans == []
    assert census.classes == {"surviving_copies": 1}
    assert _session_ids(conn) == sorted([original_id, copy_id])


@pytest.mark.parametrize("overflow", [False, True])
def test_failed_move_acknowledgement_rolls_back_identity_and_catalog_ownership(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection, overflow: bool
) -> None:
    root = tmp_path.resolve() / "custom"
    if overflow:
        original, old_id = _timestamp_collisions(conn, root)[0]
    else:
        original = _custom_transcript(root, "old/a.jsonl", LINES)
        old_id = _index(conn, original, root=root)
    SourceCatalog(conn, clock=lambda: 0.0).bind_session("claude_code", str(original), old_id)
    before = {
        table: conn.execute(f"SELECT * FROM {table}").fetchall()
        for table in ("sessions", "messages", "tool_calls", "source_files")
    }
    moved = root / "new/a.jsonl"
    moved.parent.mkdir()
    original.rename(moved)

    def reject() -> None:
        raise RuntimeError("acknowledgement interrupted")

    with pytest.raises(RuntimeError, match="acknowledgement interrupted"):
        _write_session(
            conn,
            ClaudeCodeParser().parse(moved).session,
            last_byte_offset=moved.stat().st_size,
            tail_facts=TailFacts(),
            on_commit=reject,
        )
    for table, rows in before.items():
        assert conn.execute(f"SELECT * FROM {table}").fetchall() == rows
    assert _index(conn, moved, root=root) == old_id


def test_identical_departed_copies_do_not_choose_an_identity_by_walk_order(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    from recall.core.types import default_session_host
    from recall.services.moves import plan_moved_duplicates

    root = tmp_path.resolve() / "custom"
    originals = [_custom_transcript(root, f"copy-{n}/a.jsonl", LINES) for n in range(2)]
    ids = [_index(conn, path, root=root) for path in originals]
    for path in originals:
        path.unlink()
    moved = _custom_transcript(root, "moved/a.jsonl", LINES)
    moved_id = _index(conn, moved, root=root)

    assert _session_ids(conn) == sorted([*ids, moved_id])
    assert plan_moved_duplicates(conn, host=default_session_host()) == []


@pytest.mark.parametrize("queue_sidecar_deletes", [False, True], ids=["offline", "daemon"])
@pytest.mark.parametrize(
    ("copy_tokens", "expected_tokens"),
    [((None, None, None), (99, 7, 3)), ((111, None, 0), (111, 7, 0))],
    ids=["missing-usage", "partial-usage"],
)
def test_grok_copy_indexed_before_delete_keeps_original_identity_and_tool_results(
    tmp_path: Path,
    conn: duckdb.DuckDBPyConnection,
    queue_sidecar_deletes: bool,
    copy_tokens: tuple[int | None, ...],
    expected_tokens: tuple[int, ...],
) -> None:
    from recall.core.types import Source, default_session_host
    from recall.services.moves import apply_moved_duplicates, plan_moved_duplicates

    root = tmp_path.resolve() / "custom"
    parser = get_parser(Source.GROK)
    original = root / "old/session/chat_history.jsonl"
    original.parent.mkdir(parents=True)
    original.write_bytes((FIXTURE.parents[1] / "grok/session1.jsonl").read_bytes())
    old_id = _index(conn, original, root=root, parser=parser)
    tool = conn.execute("SELECT id FROM tool_calls WHERE session_id = ?", [old_id]).fetchone()
    assert tool is not None
    tool_id = tool[0]
    vector = (1.0,) * 384
    conn.execute("INSERT INTO tool_call_embeddings VALUES (?, ?)", [tool_id, vector])
    message = conn.execute(
        "SELECT id FROM messages WHERE session_id = ? ORDER BY idx LIMIT 1", [old_id]
    ).fetchone()
    assert message is not None
    conn.execute("INSERT INTO message_embeddings VALUES (?, ?, NULL)", [message[0], vector])
    moved = root / "new/session/chat_history.jsonl"
    moved.parent.mkdir(parents=True)
    shutil.copyfile(original, moved)
    copy_id = _index(conn, moved, root=root, parser=parser)
    conn.execute(
        "UPDATE session_state SET input_tokens = 99, output_tokens = 7, "
        "cached_input_tokens = 3 WHERE session_id = ?",
        [old_id],
    )
    conn.execute(
        "UPDATE session_state SET input_tokens = ?, output_tokens = ?, "
        "cached_input_tokens = ? WHERE session_id = ?",
        [*copy_tokens, copy_id],
    )
    conn.execute(
        "INSERT INTO usage_events (id, source, source_session_id, session_id, "
        "prompt_tokens, completion_tokens, cached_prompt_tokens, harvested_at) "
        "VALUES ('harvested', 'grok', 'session', ?, 99, 7, 3, now())",
        [old_id],
    )
    tool = conn.execute("SELECT id FROM tool_calls WHERE session_id = ?", [copy_id]).fetchone()
    assert tool is not None
    copy_tool = tool[0]
    conn.execute(
        "INSERT INTO tool_results VALUES (?, 'complete listing', false, NULL)", [copy_tool]
    )
    original.unlink()

    plans = plan_moved_duplicates(conn, host=default_session_host())
    assert len(plans) == 1
    assert apply_moved_duplicates(conn, plans, queue_sidecar_deletes=queue_sidecar_deletes) == 1
    assert (
        conn.execute(
            "SELECT input_tokens, output_tokens, cached_input_tokens FROM session_state "
            "WHERE session_id = ?",
            [old_id],
        ).fetchone()
        == expected_tokens
    )
    assert conn.execute(
        "SELECT session_id, prompt_tokens, completion_tokens, cached_prompt_tokens "
        "FROM usage_events WHERE id = 'harvested'"
    ).fetchone() == (old_id, 99, 7, 3)
    assert _session_ids(conn) == [old_id]
    assert conn.execute("SELECT id FROM tool_calls").fetchall() == [(tool_id,)]
    assert conn.execute(
        "SELECT tool_call_id, bash_embedding FROM tool_call_embeddings"
    ).fetchall() == [(tool_id, vector)]
    assert conn.execute(
        "SELECT content_embedding FROM message_embeddings WHERE message_id = ?", [message[0]]
    ).fetchone() == (vector,)
    assert conn.execute("SELECT tool_call_id, session_id FROM tool_use_ids").fetchall() == [
        (tool_id, old_id)
    ]
    assert conn.execute("SELECT tool_call_id, result_summary FROM tool_results").fetchall() == [
        (tool_id, "complete listing")
    ]


@pytest.mark.parametrize("backend", ["duckdb", "sqlite_sidecar"])
def test_path_only_publication_does_not_invalidate_keyword_search(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection, backend: str
) -> None:
    from recall.services.coordinator import capture_path, observe_path, prepare_raw_sources
    from recall.services.rpc_server import RpcServer

    root = tmp_path.resolve() / "transcripts"
    original = _custom_transcript(root, "old/a.jsonl", LINES)
    parser = ClaudeCodeParser(roots=(root,))
    config = AppConfig.load()
    config = replace(config, fts=replace(config.fts, backend=backend))
    server = RpcServer(config=config)
    server._conn = conn

    def publish(path: Path) -> int:
        item = observe_path(parser, capture_path(parser, path), conn=conn)
        prepared = prepare_raw_sources((item,), {"claude_code": parser})
        return server._commit_raw_sources(prepared, config)

    assert publish(original) == 1
    assert server._keyword_dirty is (backend == "duckdb")
    server._keyword_dirty = False
    moved = root / "new/a.jsonl"
    moved.parent.mkdir()
    original.rename(moved)
    assert publish(moved) == 1
    assert server._keyword_dirty is False

    moved.write_text(moved.read_text().replace("run the test suite", "inspect payments"))
    assert publish(moved) == 1
    assert server._keyword_dirty is (backend == "duckdb")


def test_report_preserves_an_older_database_without_migrating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    config = AppConfig.load()
    with connect(config) as connection:
        connection.execute("DELETE FROM schema_version WHERE version > 31")
        connection.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (31)")
    before = config.db_path.read_bytes()

    report = CliRunner().invoke(app, ["db", "supersede-moved", "--json"])

    assert report.exit_code == 0, report.output
    assert json.loads(report.stdout)["pairs"] == []
    assert config.db_path.read_bytes() == before


def test_report_does_not_create_a_missing_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "absent"))
    config = AppConfig.load()
    result = CliRunner().invoke(app, ["db", "supersede-moved", "--json"])
    assert result.exit_code == 2, result.output
    assert "database does not exist" in result.stdout
    assert not config.db_path.exists()


@pytest.mark.parametrize(
    ("case", "expected_class", "pair_count"),
    [
        ("proven", "proven_move", 1),
        ("copies", "surviving_copies", 0),
        ("missing", "no_survivor", 0),
        ("timestamp", "missing_timestamp", 1),
        ("proof", "missing_catalog_proof", 0),
        ("limit", "prefix_limit_exceeded", 0),
        ("limit_lagging", "prefix_limit_exceeded", 0),
        ("lagging", "lagging_survivor", 0),
        ("unreadable", "unreadable_survivor", 0),
        ("divergent", "divergent_content", 0),
        ("ambiguous", "multiple_proven_survivors", 0),
        ("departed_copies", "ambiguous_predecessors", 0),
        ("overflow", "candidate_overflow", 0),
    ],
)
def test_census_classifies_without_changing_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_class: str,
    pair_count: int,
) -> None:
    home = tmp_path.resolve()
    monkeypatch.setenv("RECALL_DATA_DIR", str(home / "data"))
    config = AppConfig.load()
    old = _transcript(home, "-old", LINES)
    new = _transcript(home, "-new", LINES[:1] if case == "lagging" else LINES)
    with connect(config) as connection:
        old_id = _index(connection, old)
        new_id = _index(connection, new)
        if case == "timestamp":
            connection.execute("UPDATE session_state SET started_at = NULL")
        elif case == "proof":
            connection.execute(
                "UPDATE source_files SET committed_prefix_sha256 = NULL WHERE source_path = ?",
                [str(old)],
            )
        elif case == "limit":
            connection.execute(
                "UPDATE source_files SET committed_offset = 536870913 WHERE source_path = ?",
                [str(new)],
            )
        elif case == "limit_lagging":
            connection.execute(
                "UPDATE source_files SET committed_offset = 536870913 WHERE source_path = ?",
                [str(old)],
            )
        elif case == "ambiguous":
            _index(connection, _transcript(home, "-copy", LINES))
        elif case == "departed_copies":
            departed_copy = _transcript(home, "-copy", LINES)
            _index(connection, departed_copy)
            departed_copy.unlink()
        elif case == "overflow":
            others = [_transcript(home, f"-old-{i}", LINES) for i in range(16)]
            for path in others:
                _index(connection, path)
            for path in others:
                path.unlink()
    if case != "copies":
        old.unlink()
    if case == "lagging":
        new.write_text("".join(LINES), encoding="utf-8")
    elif case == "unreadable":
        new.unlink()
        new.mkdir()
    elif case == "divergent":
        new.write_text("X" + "".join(LINES)[1:], encoding="utf-8")
    elif case == "missing":
        new.unlink()
    before = config.db_path.read_bytes()

    report = CliRunner().invoke(app, ["db", "supersede-moved", "--json"])

    assert report.exit_code == 0, report.output
    payload = json.loads(report.stdout)
    expected_classes = {expected_class: 1}
    if case == "timestamp":
        expected_classes["proven_move"] = 1
    if case == "limit_lagging":
        expected_classes["lagging_survivor"] = 1
    assert payload["census"]["classes"] == expected_classes
    assert len(payload["pairs"]) == pair_count
    assert payload["superseded"] == 0
    assert config.db_path.read_bytes() == before
    if pair_count:
        pair = payload["pairs"][0]
        assert (pair["predecessor_id"], pair["successor_id"]) == (old_id, new_id)
        assert pair["predecessor_rows_affected"]["sessions"] == 1
        assert pair["predecessor_rows_affected"]["messages"] == 4
        assert (
            pair["predecessor"]["committed_prefix_sha256"]
            == hashlib.sha256("".join(LINES).encode()).hexdigest()
        )


@pytest.mark.parametrize("data_dir", [None, ""])
@pytest.mark.parametrize("flags", [["--apply"], ["--params", '{"apply":true}']])
def test_apply_requires_an_explicit_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data_dir: str | None, flags: list[str]
) -> None:
    # The suite supplies an isolated home; its default index is still not an
    # explicit maintenance target, even when DB/lock overrides name it.
    config = AppConfig.load()
    old = _transcript(tmp_path, "-old", LINES)
    new = _transcript(tmp_path, "-new", LINES)
    with connect(config) as connection:
        _index(connection, old)
        _index(connection, new)
    old.unlink()
    before = config.db_path.read_bytes()
    monkeypatch.setenv("RECALL_DB_PATH", str(config.db_path))
    monkeypatch.setenv("RECALL_LOCK_PATH", str(config.lock_path))
    if data_dir is not None:
        monkeypatch.setenv("RECALL_DATA_DIR", data_dir)

    result = CliRunner().invoke(app, ["db", "supersede-moved", "--json", *flags])

    assert result.exit_code == 2, result.output
    assert "RECALL_DATA_DIR" in result.stdout
    assert config.db_path.read_bytes() == before


@pytest.mark.parametrize("foreign_only", [False, True])
def test_census_exposes_host_scope_and_excluded_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign_only: bool
) -> None:
    from recall.core.types import default_session_host

    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    config = AppConfig.load()
    host = default_session_host()
    hosts = ("foreign.example",) if foreign_only else (host, "local", "foreign.example")
    with connect(config) as connection:
        for i, row_host in enumerate(hosts):
            identity = _index(connection, _transcript(tmp_path, f"-host-{i}", _other_lines(i)))
            connection.execute(
                "UPDATE session_state SET host = ? WHERE session_id = ?", [row_host, identity]
            )
    before = config.db_path.read_bytes()

    report = CliRunner().invoke(app, ["db", "supersede-moved", "--json"])

    assert report.exit_code == 0, report.output
    census = json.loads(report.stdout)["census"]
    assert census["total_sessions"] == len(hosts)
    assert census["sessions"] == (0 if foreign_only else 2)
    assert census["excluded_sessions"] == 1
    assert census["host_filter"] == {"hosts": [host, "local"], "include_null": True}
    assert config.db_path.read_bytes() == before


def test_changed_timestamp_is_only_a_report_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    config = AppConfig.load()
    old = _transcript(tmp_path, "-old", LINES)
    new = _transcript(tmp_path, "-new", LINES)
    with connect(config) as connection:
        old_id = _index(connection, old)
        new_id = _index(connection, new)
        connection.execute(
            "UPDATE session_state SET started_at = started_at + INTERVAL 1 DAY"
            " WHERE session_id = ?",
            [new_id],
        )
    old.unlink()
    result = CliRunner().invoke(app, ["db", "supersede-moved", "--apply", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["pairs"] == []
    assert len(payload["census"]["changed_timestamp_candidates"]) == 1
    with connect(config) as connection:
        assert _session_ids(connection) == sorted([old_id, new_id])


@pytest.mark.parametrize("target", ["missing", "populated", "processed"])
def test_historical_repair_preserves_equivalent_donor_only_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    config = AppConfig.load()
    old = _transcript(tmp_path, "-old", LINES)
    new = _transcript(tmp_path, "-new", LINES)
    donor_vector = [1.0] + [0.0] * 383
    target_vector = [0.0, 1.0] + [0.0] * 382
    with connect(config) as connection:
        predecessor = _index(connection, old)
        successor = _index(connection, new)
        message_ids = [
            connection.execute(
                "SELECT id FROM messages WHERE session_id = ? AND idx = 0", [identity]
            ).fetchall()[0][0]
            for identity in (predecessor, successor)
        ]
        tool_ids = [
            connection.execute(
                "SELECT id FROM tool_calls WHERE session_id = ?", [identity]
            ).fetchall()[0][0]
            for identity in (predecessor, successor)
        ]
        connection.execute(
            "INSERT INTO message_embeddings VALUES (?, ?, ?)",
            [message_ids[0], donor_vector, donor_vector],
        )
        connection.execute(
            "INSERT INTO tool_call_embeddings VALUES (?, ?)", [tool_ids[0], donor_vector]
        )
        if target != "missing":
            vector = target_vector if target == "populated" else None
            connection.execute(
                "INSERT INTO message_embeddings VALUES (?, ?, ?)", [message_ids[1], vector, vector]
            )
            connection.execute(
                "INSERT INTO tool_call_embeddings VALUES (?, ?)", [tool_ids[1], vector]
            )
    old.unlink()

    result = CliRunner().invoke(app, ["db", "supersede-moved", "--apply", "--json"])

    assert result.exit_code == 0, result.output
    expected = {
        "missing": tuple(donor_vector),
        "populated": tuple(target_vector),
        "processed": None,
    }[target]
    with connect(config) as connection:
        assert connection.execute(
            "SELECT content_embedding, thinking_embedding FROM message_embeddings"
            " WHERE message_id=?",
            [message_ids[0]],
        ).fetchone() == (expected, expected)
        assert connection.execute(
            "SELECT bash_embedding FROM tool_call_embeddings WHERE tool_call_id=?", [tool_ids[0]]
        ).fetchone() == (expected,)
        assert _session_ids(connection) == [predecessor]


def test_historical_repair_does_not_reuse_vectors_for_different_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    config = AppConfig.load()
    old = _transcript(tmp_path, "-old", LINES)
    new = _transcript(tmp_path, "-new", LINES)
    vector = [1.0] + [0.0] * 383
    with connect(config) as connection:
        predecessor = _index(connection, old)
        successor = _index(connection, new)
        message = connection.execute(
            "SELECT id FROM messages WHERE session_id=? AND idx=0", [predecessor]
        ).fetchall()[0][0]
        tool = connection.execute(
            "SELECT id FROM tool_calls WHERE session_id=?", [predecessor]
        ).fetchall()[0][0]
        connection.execute(
            "INSERT INTO message_embeddings VALUES (?, ?, ?)", [message, vector, vector]
        )
        connection.execute("INSERT INTO tool_call_embeddings VALUES (?, ?)", [tool, vector])
        connection.execute(
            "UPDATE message_state SET context_mode='llm-local', context_text='different inputs'"
            " WHERE message_id IN (SELECT id FROM messages WHERE session_id=?)",
            [successor],
        )
        connection.execute(
            "UPDATE tool_calls SET bash_command='different command' WHERE session_id=?", [successor]
        )
    old.unlink()
    result = CliRunner().invoke(app, ["db", "supersede-moved", "--apply", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["pairs"][0]["successor_vector_message_ids"] == []
    assert payload["pairs"][0]["successor_vector_tool_call_ids"] == []
    with connect(config) as connection:
        assert connection.execute("SELECT count(*) FROM message_embeddings").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM tool_call_embeddings").fetchone() == (0,)


def _timestamp_collisions(
    conn: duckdb.DuckDBPyConnection, root: Path, *, count: int = 17
) -> list[tuple[Path, str]]:
    rows = []
    for n in range(count):
        lines = [
            line.replace(
                "run the test suite",
                f"inspect project {n:02d}" if count > 17 else f"inspect project {n}",
            )
            for line in LINES
        ]
        path = _custom_transcript(root, f"project/{n}.jsonl", lines)
        rows.append((path, _index(conn, path, root=root)))
    return rows


@pytest.mark.parametrize("count", [17, 40])
def test_exact_move_in_overflow_timestamp_group_retains_all_identities(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection, count: int
) -> None:
    root = tmp_path.resolve() / "custom"
    originals = _timestamp_collisions(conn, root, count=count)
    before = {
        table: conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
        for table in ("messages", "tool_calls")
    }
    original, original_id = originals[0]
    moved = original.with_name("renamed.jsonl")
    original.rename(moved)

    assert _index(conn, moved, root=root) == original_id
    _walk(conn, root)
    assert _session_ids(conn) == sorted(identity for _, identity in originals)
    assert conn.execute(
        "SELECT source_path FROM sessions WHERE id = ?", [original_id]
    ).fetchone() == (str(moved),)
    assert _index(conn, moved, root=root) == original_id
    for table, rows in before.items():
        assert conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall() == rows


@pytest.mark.parametrize(
    "case",
    [
        "late_divergence",
        "longer_holder",
        "shorter_departure",
        "identical_departures",
        "missing_digest",
        "foreign_host",
        "surviving_original",
        "appended_move",
        "copy_then_delete",
    ],
)
def test_overflow_move_retains_independent_or_ambiguous_histories(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection, case: str
) -> None:
    from recall.core.types import default_session_host
    from recall.services.moves import plan_moved_duplicates

    root = tmp_path.resolve() / "custom"
    originals = _timestamp_collisions(conn, root)
    original, original_id = originals[0]
    lines = original.read_text().splitlines(keepends=True)
    kept = [identity for _, identity in originals]
    if case in ("longer_holder", "shorter_departure", "identical_departures"):
        competitor_lines = (
            lines + [line.replace("2026-09-07T", "2026-09-08T") for line in LINES]
            if case == "longer_holder"
            else lines
        )
        if case == "shorter_departure":
            competitor_lines = lines[:3]
        competitor = _custom_transcript(root, "competitor/a.jsonl", competitor_lines)
        kept.append(_index(conn, competitor, root=root))
        if case != "longer_holder":
            competitor.unlink()
    elif case == "missing_digest":
        conn.execute(
            "UPDATE source_files SET committed_prefix_sha256 = NULL WHERE source_path = ?",
            [str(original)],
        )
    elif case == "foreign_host":
        conn.execute(
            "UPDATE session_state SET host = 'foreign.example' WHERE session_id = ?", [original_id]
        )

    moved = original.with_name("renamed.jsonl")
    if case in ("surviving_original", "copy_then_delete"):
        shutil.copyfile(original, moved)
    else:
        original.rename(moved)
    if case == "late_divergence":
        moved.write_text("".join(lines).replace("All 312 tests pass.", "All 311 tests pass."))
    elif case == "appended_move":
        moved.write_text(
            "".join(lines + [line.replace("2026-09-07T", "2026-09-08T") for line in LINES])
        )
    successor_id = _index(conn, moved, root=root)
    if case == "copy_then_delete":
        original.unlink()
    _walk(conn, root)

    assert successor_id not in kept
    assert _session_ids(conn) == sorted([*kept, successor_id])
    assert plan_moved_duplicates(conn, host=default_session_host()) == []


@pytest.mark.parametrize("side", ["departed", "surviving", "unknown", "unequal"])
def test_overflow_narrowed_set_must_still_be_complete_and_bounded(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection, side: str
) -> None:
    root = tmp_path.resolve() / "custom"
    # 40 rows exercise the 33-row enumeration sentinel as well as per-side caps.
    originals = []
    for n in range(40):
        lines = (
            LINES
            if side in ("departed", "surviving", "unknown")
            else LINES + [line.replace("2026-09-07T", "2026-09-08T") for line in LINES]
        )
        path = _custom_transcript(root, f"copy/{n}.jsonl", lines)
        originals.append((path, _index(conn, path, root=root)))
    if side == "unknown":
        conn.execute("UPDATE source_files SET committed_prefix_sha256 = NULL")
    if side == "departed":
        for path, _ in originals:
            path.unlink()
    moved = _custom_transcript(root, "moved/a.jsonl", LINES)
    successor_id = _index(conn, moved, root=root)
    assert _session_ids(conn) == sorted([*(identity for _, identity in originals), successor_id])
    assert successor_id not in {identity for _, identity in originals}


@pytest.mark.parametrize("case", ["short", "unreadable", "prefix_cap"])
def test_overflow_successor_requires_a_readable_bounded_prefix(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection, case: str
) -> None:
    from recall.core.types import default_session_host
    from recall.services.moves import find_moved_predecessors

    root = tmp_path.resolve() / "custom"
    originals = _timestamp_collisions(conn, root)
    original, _ = originals[0]
    capture = ClaudeCodeParser().parse(original)
    boundary = original.stat().st_size
    moved = original.with_name("renamed.jsonl")
    original.rename(moved)
    if case == "short":
        moved.write_bytes(moved.read_bytes()[:-1])
    elif case == "unreadable":
        moved.unlink()
        moved.mkdir()
    else:
        boundary = 512 * 1024 * 1024 + 1
        conn.execute("UPDATE source_files SET committed_offset = ?", [boundary])
    assert (
        find_moved_predecessors(
            conn,
            replace(
                capture,
                session=capture.session.model_copy(update={"source_path": str(moved)}),
                next_byte_offset=boundary,
            ),
            host=default_session_host(),
        )
        == ()
    )


def test_overflow_fallback_cannot_adopt_a_sole_shorter_departure(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    root = tmp_path.resolve() / "custom"
    originals = _timestamp_collisions(conn, root)
    original, original_id = originals[0]
    complete = original.read_text()
    original.write_text("".join(complete.splitlines(keepends=True)[:3]))
    assert _index(conn, original, root=root) == original_id
    moved = original.with_name("renamed.jsonl")
    original.rename(moved)
    moved.write_text(complete)

    successor_id = _index(conn, moved, root=root)
    assert successor_id != original_id
    assert _session_ids(conn) == sorted([*(identity for _, identity in originals), successor_id])


def test_move_proof_cannot_adopt_bytes_outside_the_committed_capture(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.core.types import default_session_host
    from recall.services import watcher

    root = tmp_path.resolve() / "custom"
    originals = _timestamp_collisions(conn, root)
    original, predecessor_id = originals[0]
    predecessor_rows = {
        table: conn.execute(
            f"SELECT * FROM {table} WHERE session_id = ? ORDER BY id", [predecessor_id]
        ).fetchall()
        for table in ("messages", "tool_calls")
    }
    moved = original.with_name("renamed.jsonl")
    original.rename(moved)
    bytes_b = moved.read_bytes()
    bytes_a = bytes_b.replace(b"inspect project 0", b"inspect project X")
    assert len(bytes_a) == len(bytes_b) and bytes_a != bytes_b
    moved.write_bytes(bytes_a)
    parser = ClaudeCodeParser()
    prepared = parser.parse(moved)
    catalog = SourceCatalog(conn, clock=lambda: 0.0)
    generation = catalog.observe(parser.source.value, str(root), str(moved), stat_signature(moved))

    def acknowledge(session, result, rewrite):
        catalog.bind_session(session.source.value, session.source_path, session.id)
        checkpoint = result.normalization_checkpoint
        assert catalog.acknowledge(
            session.source.value,
            session.source_path,
            generation,
            result.next_byte_offset,
            result.committed_prefix_sha256,
            semantic_rewrite=rewrite,
            checkpoint=checkpoint.encode() if checkpoint is not None else None,
            full_parse=result.is_full_parse,
        )

    load_snapshot = watcher._load_session_write_snapshot

    def replace_during_proof(*args, **kwargs):
        moved.write_bytes(bytes_b)
        try:
            return load_snapshot(*args, **kwargs)
        finally:
            moved.write_bytes(bytes_a)

    monkeypatch.setattr(watcher, "_load_session_write_snapshot", replace_during_proof)
    watcher.index_single_session(
        moved,
        parser,
        AppConfig.load(),
        conn=conn,
        prepared_result=prepared,
        host=default_session_host(),
        lightweight_context=True,
        on_commit=acknowledge,
    )

    successor = conn.execute(
        "SELECT id FROM sessions WHERE source_path = ?", [str(moved)]
    ).fetchone()
    assert successor is not None
    successor_id = successor[0]
    assert successor_id != predecessor_id
    assert len(_session_ids(conn)) == 18
    assert conn.execute(
        "SELECT committed_prefix_sha256 FROM source_files WHERE source_path = ?", [str(moved)]
    ).fetchone() == (hashlib.sha256(bytes_a).hexdigest(),)
    assert conn.execute(
        "SELECT count(*) FROM messages m JOIN message_state ms ON ms.message_id = m.id"
        " WHERE m.session_id = ? AND ms.content LIKE '%inspect project X%'",
        [successor_id],
    ).fetchone() == (1,)
    assert conn.execute(
        "SELECT source_path FROM sessions WHERE id = ?", [predecessor_id]
    ).fetchone() == (str(original),)
    for table, rows in predecessor_rows.items():
        assert (
            conn.execute(
                f"SELECT * FROM {table} WHERE session_id = ? ORDER BY id", [predecessor_id]
            ).fetchall()
            == rows
        )
