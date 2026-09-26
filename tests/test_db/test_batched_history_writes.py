"""Row batches preserve storage semantics, including timestamps and duplicate facts."""

import multiprocessing
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest
from recall.core.models import Message, StopMarker, ToolCall
from recall.core.types import Role
from recall.db.connection import wal_size_bytes
from recall.db.queries import (
    fetch_tool_use_ids,
    insert_messages,
    insert_tool_calls,
    insert_tool_results,
    insert_tool_use_ids,
    resolve_tool_call_ids,
    upsert_stop_markers,
)
from recall.db.schema import ensure_schema


def test_bounded_identity_lookups_preserve_session_scope_and_literal_ids() -> None:
    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=2)
        insert_tool_calls(
            conn,
            [
                ToolCall(id="first", session_id="local", message_id=None, idx=0, tool_name="Bash"),
                ToolCall(id="last'λ", session_id="local", message_id=None, idx=1, tool_name="Bash"),
                ToolCall(
                    id="foreign", session_id="remote", message_id=None, idx=0, tool_name="Bash"
                ),
            ],
        )
        insert_tool_use_ids(
            conn,
            [
                ("first", "local", "shared"),
                ("last'λ", "local", "literal'λ"),
                ("foreign", "remote", "shared"),
            ],
        )
        assert resolve_tool_call_ids(
            conn, "local", ["shared"] * 501 + ["literal'λ", "missing"]
        ) == {"shared": "first", "literal'λ": "last'λ"}
        assert fetch_tool_use_ids(conn, ["first"] * 501 + ["last'λ", "missing"]) == {
            "first": "shared",
            "last'λ": "literal'λ",
        }
        assert resolve_tool_call_ids(conn, "remote", ["shared", "literal'λ"]) == {
            "shared": "foreign"
        }
        assert resolve_tool_call_ids(conn, "local", []) == {}
        assert fetch_tool_use_ids(conn, []) == {}


def test_explicit_common_context_overrides_message_context() -> None:
    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=2)
        insert_messages(
            conn,
            [
                Message(
                    id="a",
                    session_id="s",
                    idx=0,
                    role=Role.USER,
                    content="λ",
                    context_text="model-specific ",
                    context_mode="llm-local",
                ),
                Message(id="b", session_id="s", idx=1, role=Role.ASSISTANT, thinking="thought"),
            ],
            context_text="shared ",
            context_mode="template",
        )
        assert conn.execute(
            "SELECT context_text, context_mode, fts_content, fts_thinking "
            "FROM message_state ORDER BY message_id"
        ).fetchall() == [
            ("shared ", "template", "shared λ", "shared "),
            ("shared ", "template", "shared ", "shared thought"),
        ]


@pytest.mark.parametrize("presentations", [1, 3])
def test_representing_stop_facts_across_batches_does_not_grow_wal(
    tmp_path: Path, presentations: int
) -> None:
    path = tmp_path / "markers.duckdb"
    final = [StopMarker(idx=i, reason="completed", ends_turn=True) for i in range(256)]
    history = [
        *final[:255],
        StopMarker(idx=255, reason="working", ends_turn=False),
        StopMarker(idx=255, reason="completed", ends_turn=True),
    ] * presentations
    with duckdb.connect(str(path)) as conn:
        ensure_schema(conn, embed_dim=2)
        conn.execute("BEGIN")
        upsert_stop_markers(conn, "s", history)
        conn.execute("COMMIT")
        conn.execute("CHECKPOINT")
        assert wal_size_bytes(path) == 0

        # Control: a no-op transaction with unique final facts emits no WAL.
        conn.execute("BEGIN")
        upsert_stop_markers(conn, "s", final)
        conn.execute("COMMIT")
        assert wal_size_bytes(path) == 0

        conn.execute("BEGIN")
        upsert_stop_markers(conn, "s", history)
        conn.execute("COMMIT")
        assert conn.execute(
            "SELECT reason, ends_turn FROM session_stop_markers WHERE message_idx=255"
        ).fetchone() == ("completed", True)
        assert wal_size_bytes(path) == 0


def test_failed_stop_fact_stream_preserves_caller_rollback_and_connection() -> None:
    def interrupted():
        for idx in range(300):
            yield StopMarker(idx=idx, reason="working", ends_turn=False)
        raise ValueError("source interrupted")

    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=2)
        upsert_stop_markers(conn, "prior", [StopMarker(idx=0, reason="completed", ends_turn=True)])
        conn.execute("BEGIN")
        with pytest.raises(duckdb.Error, match="source interrupted"):
            upsert_stop_markers(conn, "s", interrupted())
        conn.execute("ROLLBACK")
        assert conn.execute("SELECT * FROM session_stop_markers").fetchall() == [
            ("prior", 0, "completed", True)
        ]
        upsert_stop_markers(conn, "s", iter(()))
        upsert_stop_markers(conn, "s", [StopMarker(idx=1, reason="stop", ends_turn=True)])
        assert conn.execute(
            "SELECT * FROM session_stop_markers ORDER BY session_id"
        ).fetchall() == [
            ("prior", 0, "completed", True),
            ("s", 1, "stop", True),
        ]


def test_marker_stream_allows_concurrent_cursor_reads() -> None:
    process = multiprocessing.get_context("spawn").Process(target=_read_during_marker_stream)
    process.start()
    try:
        process.join(timeout=15)
        assert process.exitcode == 0, "concurrent marker stream stalled or failed"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        process.close()


def _read_during_marker_stream() -> None:
    entered = threading.Event()
    release = threading.Event()

    def markers() -> Iterator[StopMarker]:
        for idx in range(256):
            yield StopMarker(idx=idx, reason="completed", ends_turn=True)
        entered.set()
        assert release.wait(5), "stream was not released"
        yield StopMarker(idx=256, reason="completed", ends_turn=True)

    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            writer = pool.submit(upsert_stop_markers, conn, "s", markers())
            assert entered.wait(5), "stream was not entered"

            def read() -> tuple[int] | None:
                with conn.cursor() as reader:
                    return reader.execute("SELECT COUNT(*) FROM session_stop_markers").fetchone()

            try:
                assert pool.submit(read).result(timeout=2) == (0,)
            finally:
                release.set()
            writer.result(timeout=5)
            assert conn.execute("SELECT COUNT(*) FROM session_stop_markers").fetchone() == (257,)


@pytest.mark.parametrize("existing", [False, True])
def test_last_stop_fact_wins_with_duplicate_positions_across_batches(existing: bool) -> None:
    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=2)
        if existing:
            upsert_stop_markers(conn, "s", [StopMarker(idx=0, reason="prior", ends_turn=True)])
        markers = [StopMarker(idx=0, reason="working", ends_turn=False) for _ in range(257)]
        markers.append(StopMarker(idx=0, reason="completed", ends_turn=True))
        upsert_stop_markers(conn, "s", iter(markers))
        assert conn.execute("SELECT * FROM session_stop_markers").fetchall() == [
            ("s", 0, "completed", True)
        ]
        upsert_stop_markers(conn, "s", iter(markers))
        assert conn.execute("SELECT * FROM session_stop_markers").fetchall() == [
            ("s", 0, "completed", True)
        ]


def test_large_tool_batches_preserve_fields_fact_ownership_and_atomic_rollback() -> None:
    with duckdb.connect(":memory:") as conn:
        conn.execute("SET TimeZone = 'Europe/Berlin'")
        ensure_schema(conn, embed_dim=2)
        calls = [
            ToolCall(
                id=f"t{i}",
                session_id="s",
                message_id=None if i % 2 else "m",
                idx=i,
                tool_name="Bash",
                tool_input={"nested": ["λ", None, 7]},
                bash_command="git status",
                bash_base="git",
                bash_sub="status",
                is_compound=True,
                agent_id="agent",
                subagent_type="explore",
                subagent_description="look",
                subagent_model="model",
                skill_name="inspect",
            )
            for i in range(514)
        ]
        use_ids = [(call.id, "s", f"h{call.idx}") for call in calls]
        results: list[tuple[str, str, bool, datetime | None]] = [
            (call.id, "first λ", False, datetime(2026, 2, 3, 9, 10, tzinfo=UTC)) for call in calls
        ]
        # A duplicate inside one input batch still preserves the first fact.
        use_ids.insert(1, ("t0", "s", "ignored"))
        results.insert(1, ("t0", "ignored", True, None))
        markers = [StopMarker(idx=i, reason="stop", ends_turn=True) for i in range(514)]
        markers.insert(1, StopMarker(idx=0, reason="latest", ends_turn=False))
        conn.execute("BEGIN")
        insert_tool_calls(conn, calls)
        insert_tool_use_ids(conn, iter(use_ids))
        insert_tool_results(conn, iter(results))
        upsert_stop_markers(conn, "s", iter(markers))
        conn.execute("COMMIT")
        row = conn.execute("SELECT * FROM tool_calls WHERE id = 't513'").fetchone()
        assert row == (
            "t513",
            "s",
            None,
            513,
            "Bash",
            '{"nested": ["\\u03bb", null, 7]}',
            "git status",
            "git",
            "status",
            True,
            "agent",
            "explore",
            "look",
            "model",
            "inspect",
        )
        assert conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone() == (514,)
        assert conn.execute(
            "SELECT tool_use_id FROM tool_use_ids WHERE tool_call_id='t0'"
        ).fetchone() == ("h0",)
        assert conn.execute(
            "SELECT result_summary, is_error, completed_at FROM tool_results "
            "WHERE tool_call_id='t0'"
        ).fetchone() == ("first λ", False, datetime(2026, 2, 3, 10, 10))
        assert conn.execute(
            "SELECT reason, ends_turn FROM session_stop_markers WHERE message_idx=0"
        ).fetchone() == ("latest", False)
        insert_tool_results(
            conn,
            [
                ("t0", "replacement", True, None),
                ("naive", "naive", True, datetime(2026, 2, 3, 9, 10)),
                ("undated", "undated", False, None),
            ],
        )
        assert conn.execute(
            "SELECT result_summary, completed_at FROM tool_results "
            "WHERE tool_call_id IN ('t0', 'naive', 'undated') ORDER BY tool_call_id"
        ).fetchall() == [
            ("naive", datetime(2026, 2, 3, 9, 10)),
            ("first λ", datetime(2026, 2, 3, 10, 10)),
            ("undated", None),
        ]
        # A duplicate at the end of a multi-batch insert aborts the enclosing
        # transaction, leaving no partial identities or changes to other rows.
        fresh = [call.model_copy(update={"id": f"new{call.idx}"}) for call in calls]
        conn.execute("BEGIN")
        with pytest.raises(duckdb.ConstraintException):
            insert_tool_calls(conn, [*fresh, calls[0]])
        conn.execute("ROLLBACK")
        assert conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone() == (514,)
        insert_tool_calls(conn, [fresh[0]])
        assert conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone() == (515,)
