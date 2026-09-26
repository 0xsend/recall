"""A transcript indexed again after its project directory moved (REQ-INDEX-027)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import duckdb
import pytest
from recall.cli.app import app
from recall.core.config import AppConfig
from recall.core.models import TailFacts
from recall.db import connect
from recall.db.source_files import SourceCatalog
from recall.parsers.claude_code import ClaudeCodeParser
from recall.services.coordinator import PreparedRawCycle, persist_prepared_inventory
from recall.services.indexer import _apply_stored_contexts, _write_session
from recall.services.reconciler import iter_inventory_batches, stat_signature
from typer.testing import CliRunner

FIXTURE = Path(__file__).parents[1] / "fixtures" / "claude_code" / "live_end_turn.jsonl"
LINES = FIXTURE.read_text(encoding="utf-8").splitlines(keepends=True)


def _transcript(home: Path, project: str, lines: list[str]) -> Path:
    path = home / ".claude" / "projects" / project / "3f1c.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _index(conn: duckdb.DuckDBPyConnection, path: Path, *, root: Path | None = None) -> str:
    """Index a transcript and commit its catalog prefix, as reconciliation does."""
    data = path.read_bytes()
    session = ClaudeCodeParser().parse(path).session
    _apply_stored_contexts(conn, session)
    _write_session(conn, session, last_byte_offset=len(data), tail_facts=TailFacts())
    catalog = SourceCatalog(conn, clock=lambda: 0.0)
    generation = catalog.observe(
        "claude_code", str(root or path.parent), str(path), stat_signature(path)
    )
    catalog.acknowledge(
        "claude_code", str(path), generation, len(data), hashlib.sha256(data).hexdigest()
    )
    return session.id


def _walk(conn: duckdb.DuckDBPyConnection, root: Path) -> None:
    """Run one complete inventory walk of `root`, as the daemon's reconciliation does."""
    parser = ClaudeCodeParser()
    generation, _ = persist_prepared_inventory(
        PreparedRawCycle(parser, root, None), None, conn=conn
    )
    for event in iter_inventory_batches(parser, root, clock=lambda: 0.0):
        generation, _ = persist_prepared_inventory(
            PreparedRawCycle(parser, root, event), generation, conn=conn
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
    others = [_index(conn, _transcript(home, f"-work-other-{n}", LINES)) for n in range(20)]
    old_path = _transcript(home, "-work-old", LINES[:3])
    _index(conn, old_path)

    new_path = _transcript(home, "-work-new", LINES)
    old_path.unlink()
    new_id = _index(conn, new_path)

    assert _session_ids(conn) == sorted([*others, new_id])


def test_superseded_rows_leave_keyword_search_with_the_transaction(
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
    assert queued == old_messages


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

    assert _session_ids(conn) == [new_id]
    assert conn.execute("SELECT session_id FROM usage_events").fetchone() == (new_id,)
    deleted = {
        str(row[0])
        for row in conn.execute(
            "SELECT id FROM fts_sidecar_pending WHERE kind = 'message' AND op = 'delete'"
        ).fetchall()
    }
    assert deleted == old_messages
    carried = conn.execute(
        "SELECT COUNT(*) FROM messages m JOIN message_state ms ON ms.message_id = m.id"
        " WHERE m.session_id = ? AND ms.context_text = 'summary'",
        [new_id],
    ).fetchone()
    assert carried is not None and carried[0] > 0


def _missing_paths(conn: duckdb.DuckDBPyConnection) -> set[str]:
    rows = conn.execute("SELECT source_path FROM source_files WHERE missing").fetchall()
    return {str(row[0]) for row in rows}


def test_a_failed_supersession_is_given_up_while_the_walk_finishes(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    import recall.services.moves as moves_module

    home = tmp_path.resolve()
    projects = home / ".claude" / "projects"
    gone = [
        _transcript(home, f"-work-gone-{n}", LINES[:2]).rename(
            projects / f"-work-gone-{n}" / f"gone-{n}.jsonl"
        )
        for n in range(3)
    ]
    for path in gone:
        _index(conn, path, root=projects)
    for n in range(4):
        kept = _transcript(home, f"-work-kept-{n}", LINES)
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
    _walk(conn, projects)
    _walk(conn, projects)

    assert attempts == [(old_id,)]
    assert _missing_paths(conn) == {str(path) for path in [*gone, old_path]}
    assert conn.execute("SELECT scan_complete FROM reconciliation_roots").fetchone() == (True,)
    assert old_id in _session_ids(conn) and new_id in _session_ids(conn)


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
        old_path = _transcript(home, "-work-old", LINES)
        olds.append(old_path.rename(old_path.with_name(f"s{n}.jsonl")))
        _index(conn, olds[-1], root=projects)
        new_path = _transcript(home, "-work-new", LINES)
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
        _index(conn, old_path)
        new_id = _index(conn, new_path)
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
    assert json.loads(again.stdout)["moves"] == []
    conn = connect(AppConfig.load())
    try:
        assert _session_ids(conn) == [new_id]
        # The stopped daemon reconciles keyword search at startup; a bulk queue
        # would only delay it and cannot survive an interrupted insert.
        assert conn.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone() == (0,)
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
        new_id = _index(conn, new_path)
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
            [new_id],
        ).fetchall()
        queued = conn.execute(
            "SELECT id FROM fts_sidecar_pending WHERE kind = 'message' AND op = 'upsert'"
        ).fetchall()
        assert carried
        assert all(text.startswith("summary") for _, text in carried)
        assert {row[0] for row in queued} == {message_id for message_id, _ in carried}
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
