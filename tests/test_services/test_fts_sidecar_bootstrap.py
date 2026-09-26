from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
from recall.db.fts_sidecar import open_sidecar
from recall.db.schema import ensure_schema
from recall.services.fts_sidecar_bootstrap import BootstrapProgress, bootstrap_sidecar


def test_bootstrap_populates_sidecar(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _seed_messages(duckdb_conn, 50)
        _seed_tool_calls(duckdb_conn, 25, null_bash_every=5)

        progress = bootstrap_sidecar(duckdb_conn, sidecar_conn, batch_size=10)

        assert progress == BootstrapProgress(
            messages_processed=50,
            tool_calls_processed=20,
            messages_done=True,
            tool_calls_done=True,
        )
        assert _count(sidecar_conn, "message_fts_rowid") == 50
        assert _count(sidecar_conn, "tool_calls_fts_rowid") == 20
        assert _completed(sidecar_conn, "message")
        assert _completed(sidecar_conn, "tool_call")
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_bootstrap_is_resumable(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_path = tmp_path / "recall.fts.sqlite"
    sidecar_conn = open_sidecar(sidecar_path)
    try:
        _seed_messages(duckdb_conn, 100)

        def raise_after_three_batches(progress: BootstrapProgress) -> None:
            if progress.messages_processed == 30:
                raise RuntimeError("stop after third committed batch")

        try:
            bootstrap_sidecar(
                duckdb_conn,
                sidecar_conn,
                batch_size=10,
                progress_callback=raise_after_three_batches,
            )
        except RuntimeError as err:
            assert str(err) == "stop after third committed batch"

        row = _progress_row(sidecar_conn, "message")
        assert row == ("msg-029", None)
        assert _count(sidecar_conn, "message_fts_rowid") == 30

        progress = bootstrap_sidecar(duckdb_conn, sidecar_conn, batch_size=10)

        assert progress == BootstrapProgress(
            messages_processed=70,
            tool_calls_processed=0,
            messages_done=True,
            tool_calls_done=True,
        )
        assert _count(sidecar_conn, "message_fts_rowid") == 100
        assert _completed(sidecar_conn, "message")
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_bootstrap_noop_on_fully_bootstrapped_sidecar(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _seed_messages(duckdb_conn, 10)
        _seed_tool_calls(duckdb_conn, 10)
        bootstrap_sidecar(duckdb_conn, sidecar_conn, batch_size=4)
        before_counts = (
            _count(sidecar_conn, "message_fts_rowid"),
            _count(sidecar_conn, "tool_calls_fts_rowid"),
        )

        progress = bootstrap_sidecar(duckdb_conn, sidecar_conn, batch_size=4)

        assert progress == BootstrapProgress(
            messages_processed=0,
            tool_calls_processed=0,
            messages_done=True,
            tool_calls_done=True,
        )
        assert (
            _count(sidecar_conn, "message_fts_rowid"),
            _count(sidecar_conn, "tool_calls_fts_rowid"),
        ) == before_counts
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_bootstrap_respects_batch_size(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _seed_messages(duckdb_conn, 25)
        _seed_tool_calls(duckdb_conn, 25)
        callbacks: list[BootstrapProgress] = []

        bootstrap_sidecar(
            duckdb_conn,
            sidecar_conn,
            batch_size=10,
            progress_callback=callbacks.append,
        )

        assert len(callbacks) == 6
        message_callbacks = [
            progress.messages_processed
            for progress in callbacks
            if progress.tool_calls_processed == 0
        ]
        tool_call_callbacks = [
            progress.tool_calls_processed
            for progress in callbacks
            if progress.tool_calls_processed > 0
        ]
        assert message_callbacks == [10, 20, 25]
        assert tool_call_callbacks == [10, 20, 25]
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_bootstrap_skips_existing_mapping_rows(tmp_path: Path) -> None:
    duckdb_conn = _duckdb_with_schema()
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _seed_message(duckdb_conn, "msg-1", fts_content="existing mapping")
        _seed_message(duckdb_conn, "msg-2", fts_content="new mapping")
        sidecar_conn.execute(
            "INSERT INTO message_fts_rowid(rowid, message_id) VALUES (?, ?)",
            [999, "msg-1"],
        )
        sidecar_conn.commit()

        bootstrap_sidecar(duckdb_conn, sidecar_conn)

        assert dict(
            sidecar_conn.execute(
                "SELECT message_id, rowid FROM message_fts_rowid ORDER BY message_id"
            )
        ) == {"msg-1": 999, "msg-2": 1000}
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def _duckdb_with_schema() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    return conn


def _seed_messages(conn: duckdb.DuckDBPyConnection, count: int) -> None:
    for index in range(count):
        fts_content = "" if index % 3 == 0 else f"content {index}"
        _seed_message(conn, f"msg-{index:03d}", fts_content=fts_content)


def _seed_message(
    conn: duckdb.DuckDBPyConnection,
    message_id: str,
    *,
    fts_content: str,
) -> None:
    conn.execute(
        """
        INSERT INTO message_state (
            message_id, role, content, thinking, has_thinking, fts_content, fts_thinking
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            message_id,
            "assistant",
            f"raw {message_id}",
            f"thinking {message_id}",
            True,
            fts_content,
            f"thoughts {message_id}",
        ],
    )


def _seed_tool_calls(
    conn: duckdb.DuckDBPyConnection,
    count: int,
    *,
    null_bash_every: int | None = None,
) -> None:
    for index in range(count):
        bash_command = None
        if null_bash_every is None or index % null_bash_every != 0:
            bash_command = f"echo {index}"
        conn.execute(
            """
            INSERT INTO tool_calls (
                id, session_id, message_id, idx, tool_name, bash_command, is_compound
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f"tc-{index:03d}",
                "session-1",
                None,
                index,
                "bash",
                bash_command,
                False,
            ],
        )


def _count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def _completed(conn: sqlite3.Connection, kind: str) -> bool:
    row = conn.execute(
        "SELECT completed_at IS NOT NULL FROM bootstrap_progress WHERE kind = ?",
        [kind],
    ).fetchone()
    assert row is not None
    return bool(row[0])


def _progress_row(conn: sqlite3.Connection, kind: str) -> tuple[str | None, str | None]:
    row = conn.execute(
        "SELECT last_id, completed_at FROM bootstrap_progress WHERE kind = ?",
        [kind],
    ).fetchone()
    assert row is not None
    return row[0], row[1]
