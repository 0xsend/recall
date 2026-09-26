"""A `message_state` row whose `messages` row is gone must not abort the process.

`_load_persisted_session_rows` reads prior messages with an INNER JOIN onto
`message_state`, so a row present in one table and absent from the other is
invisible to the writer.  The session still looks "already indexed" (its
`sessions` row is there), so `_sync_existing_session` runs, classifies every
parsed message as new, and inserts into `message_state` again.  DuckDB answers
the duplicate primary key with a FatalException -- an uncaught C++ exception
that aborts the whole process, so the daemon dies rather than returning an
error, and the launchd/systemd restart makes it look like a lost connection.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
from recall.core.ids import message_id, session_id
from recall.core.models import Message, Role, Session, TailFacts
from recall.db.schema import ensure_schema
from recall.services.indexer import _write_session

SOURCE = "codex"
PATH = "/tmp/rollout-orphan.jsonl"


def _session() -> Session:
    sid = session_id(SOURCE, PATH)
    return Session(
        id=sid,
        source=SOURCE,
        source_path=PATH,
        source_session_id="orphan-1",
        started_at=datetime.now(UTC),
        ended_at=datetime.now(UTC),
        message_count=1,
        file_mtime=1.0,
        file_size=1,
        messages=[
            Message(
                id=message_id(sid, 0),
                session_id=sid,
                idx=0,
                role=Role.USER,
                content="hello",
                timestamp=datetime.now(UTC),
            )
        ],
    )


def _count(conn: duckdb.DuckDBPyConnection, sql: str, params: list[str] | None = None) -> int:
    row = conn.execute(sql, params or []).fetchone()
    assert row is not None
    return int(row[0])


def _conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "recall.duckdb"))
    ensure_schema(conn)
    return conn


def test_reparse_survives_a_message_state_row_left_without_its_message(
    tmp_path: Path,
) -> None:
    conn = _conn(tmp_path)
    session = _session()
    _write_session(conn, session, tail_facts=TailFacts())

    # The exact inconsistency seen in the wild: the message row is gone while
    # its state row survives, because every delete path reaches `message_state`
    # through `messages`.
    conn.execute("DELETE FROM messages WHERE session_id = ?", [session.id])
    assert _count(conn, "SELECT count(*) FROM message_state") == 1

    _write_session(conn, session, tail_facts=TailFacts())

    assert _count(conn, "SELECT count(*) FROM messages") == 1
    assert _count(conn, "SELECT count(*) FROM message_state") == 1


def test_reparse_survives_a_message_row_left_without_its_state(
    tmp_path: Path,
) -> None:
    conn = _conn(tmp_path)
    session = _session()
    _write_session(conn, session, tail_facts=TailFacts())

    conn.execute("DELETE FROM message_state WHERE message_id = ?", [message_id(session.id, 0)])

    _write_session(conn, session, tail_facts=TailFacts())

    assert _count(conn, "SELECT count(*) FROM messages") == 1
    assert _count(conn, "SELECT count(*) FROM message_state") == 1
