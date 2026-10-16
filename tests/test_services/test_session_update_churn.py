"""Session progress updates preserve indexed metadata without row replacement."""

from datetime import datetime
from pathlib import Path

import duckdb
import pytest
from recall.core.models import Session, TailFacts
from recall.core.types import Source
from recall.db.schema import ensure_schema
from recall.services.indexer import _incremental_write_session, _write_session


@pytest.mark.parametrize("incremental", [False, True])
def test_progress_preserves_row_and_changed_indexed_metadata_is_searchable(
    tmp_path: Path, incremental: bool
) -> None:
    with duckdb.connect(str(tmp_path / "sessions.duckdb")) as conn:
        ensure_schema(conn)
        session = Session(
            id="session",
            source=Source.CODEX,
            source_path="/owned/session",
            started_at=datetime(2026, 1, 1),
            cwd="/old",
            git_repo="old",
            file_mtime=1,
            file_size=10,
        )
        _write_session(conn, session, tail_facts=TailFacts())
        before = conn.execute("SELECT rowid FROM session_state").fetchone()
        write = _incremental_write_session if incremental else _write_session
        progressed = session.model_copy(update={"file_mtime": 2, "file_size": 20})
        write(conn, progressed, last_byte_offset=20, tail_facts=TailFacts())
        assert conn.execute("SELECT rowid FROM session_state").fetchone() == before
        changed = progressed.model_copy(update={"cwd": "/new", "git_repo": "new"})
        write(conn, changed, last_byte_offset=20, tail_facts=TailFacts())
        assert conn.execute(
            "SELECT session_id FROM session_state WHERE cwd='/new' AND git_repo='new'"
        ).fetchall() == [("session",)]
        assert (
            conn.execute("SELECT session_id FROM session_state WHERE cwd='/old'").fetchall() == []
        )
