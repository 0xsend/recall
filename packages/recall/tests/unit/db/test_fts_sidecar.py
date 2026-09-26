from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest
from recall.db.fts_sidecar import (
    _message_match_query,
    open_sidecar,
    search_messages_fts,
    search_tool_calls_fts,
    upsert_message_fts,
    upsert_message_fts_batch,
    upsert_tool_call_fts,
    upsert_tool_call_fts_batch,
)


class _CountingConnection(sqlite3.Connection):
    transaction_entries: int

    def __enter__(self) -> _CountingConnection:
        self.transaction_entries += 1
        return cast("_CountingConnection", super().__enter__())


def _open_counting_sidecar(path: Path) -> _CountingConnection:
    setup_conn = open_sidecar(path)
    setup_conn.close()
    conn = sqlite3.connect(path, factory=_CountingConnection)
    counting_conn = conn
    counting_conn.transaction_entries = 0
    return counting_conn


def test_message_search_restricts_user_column_filter_to_configured_fields(
    tmp_path: Path,
) -> None:
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        upsert_message_fts(
            sidecar_conn,
            "msg-query-scope",
            "ordinary content",
            "thinkingonlyterm",
        )

        hits = search_messages_fts(
            sidecar_conn,
            "nope OR fts_thinking:thinkingonlyterm",
            ["content"],
            limit=10,
        )

        assert hits == []
    finally:
        sidecar_conn.close()


def test_message_search_default_fields_still_match_thinking(tmp_path: Path) -> None:
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        upsert_message_fts(
            sidecar_conn,
            "msg-default-scope",
            "ordinary content",
            "defaultthinkingterm",
        )

        hits = search_messages_fts(
            sidecar_conn,
            "defaultthinkingterm",
            ["content", "thinking"],
            limit=10,
        )

        assert hits == [("msg-default-scope", hits[0][1])]
    finally:
        sidecar_conn.close()


def test_batch_upserts_use_bounded_transactions(tmp_path: Path) -> None:
    sidecar_conn = _open_counting_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        message_rows = [
            (f"msg-batch-{idx}", f"batchcontentterm{idx}", f"batchthinkingterm{idx}")
            for idx in range(5)
        ]
        tool_call_rows = [(f"tool-batch-{idx}", f"echo batchbashterm{idx}") for idx in range(5)]

        upsert_message_fts_batch(
            sidecar_conn,
            message_rows,
            fields=("content", "thinking"),
            batch_size=2,
        )
        assert sidecar_conn.transaction_entries == 3

        upsert_tool_call_fts_batch(
            sidecar_conn,
            tool_call_rows,
            fields=("bash",),
            batch_size=2,
        )
        assert sidecar_conn.transaction_entries == 6
    finally:
        sidecar_conn.close()


def test_batch_upserts_match_single_row_sidecar_behavior(tmp_path: Path) -> None:
    message_rows = [
        ("msg-content", "visiblecontentterm", "hiddenthinkingterm"),
        ("msg-thinking", "ordinary content", "visiblethinkingterm"),
    ]
    tool_call_rows = [
        ("tool-bash", "echo visiblebashterm"),
        ("tool-none", None),
    ]
    single_conn = open_sidecar(tmp_path / "single.fts.sqlite")
    batch_conn = open_sidecar(tmp_path / "batch.fts.sqlite")
    try:
        for message_id, fts_content, fts_thinking in message_rows:
            upsert_message_fts(
                single_conn,
                message_id,
                fts_content,
                fts_thinking,
                fields=("content",),
            )
        for tool_call_id, bash_command in tool_call_rows:
            upsert_tool_call_fts(
                single_conn,
                tool_call_id,
                bash_command,
                fields=("content", "bash"),
            )

        upsert_message_fts_batch(
            batch_conn,
            message_rows,
            fields=("content",),
        )
        upsert_tool_call_fts_batch(
            batch_conn,
            tool_call_rows,
            fields=("content", "bash"),
        )

        assert single_conn.execute("SELECT count(*) FROM message_fts").fetchone() == (
            batch_conn.execute("SELECT count(*) FROM message_fts").fetchone()
        )
        assert single_conn.execute("SELECT count(*) FROM tool_calls_fts").fetchone() == (
            batch_conn.execute("SELECT count(*) FROM tool_calls_fts").fetchone()
        )
        assert (
            single_conn.execute(
                "SELECT rowid, message_id FROM message_fts_rowid ORDER BY rowid"
            ).fetchall()
            == batch_conn.execute(
                "SELECT rowid, message_id FROM message_fts_rowid ORDER BY rowid"
            ).fetchall()
        )
        assert (
            single_conn.execute(
                "SELECT rowid, tool_call_id FROM tool_calls_fts_rowid ORDER BY rowid"
            ).fetchall()
            == batch_conn.execute(
                "SELECT rowid, tool_call_id FROM tool_calls_fts_rowid ORDER BY rowid"
            ).fetchall()
        )
        content_hits = search_messages_fts(batch_conn, "visiblecontentterm", ["content"], 10)
        hidden_thinking_hits = search_messages_fts(
            batch_conn,
            "hiddenthinkingterm",
            ["content", "thinking"],
            10,
        )
        bash_hits = search_tool_calls_fts(batch_conn, "visiblebashterm", 10)

        assert content_hits == [("msg-content", content_hits[0][1])]
        assert hidden_thinking_hits == []
        assert bash_hits == [("tool-bash", bash_hits[0][1])]
    finally:
        single_conn.close()
        batch_conn.close()


# --- default/content+thinking/tool paths must escape queries ---


def test_message_match_query_escapes_operator_chars_on_default_path() -> None:
    # Default (no --field) and content+thinking previously returned the raw query,
    # letting FTS5 interpret -, :, (, ) as operators and leak raw errors.
    assert _message_match_query("REQ-BRIDGE", []) == '"REQ" "BRIDGE"'
    assert _message_match_query("foo:bar", []) == '"foo" "bar"'
    assert _message_match_query("foo(bar)", []) == '"foo" "bar"'
    assert _message_match_query("REQ-BRIDGE", ["content", "thinking"]) == '"REQ" "BRIDGE"'


def test_message_match_query_preserves_prefix_and_phrase_operators() -> None:
    # Escaping must not regress the operators users actually rely on.
    assert _message_match_query("foo*", []) == '"foo"*'
    assert _message_match_query('"quoted phrase"', []) == '"quoted phrase"'
    assert _message_match_query("plain terms", []) == '"plain" "terms"'


def test_message_match_query_scopes_single_field_after_escaping() -> None:
    assert _message_match_query("REQ-BRIDGE", ["content"]) == '{fts_content}: ("REQ" "BRIDGE")'
    assert _message_match_query("REQ-BRIDGE", ["thinking"]) == '{fts_thinking}: ("REQ" "BRIDGE")'


def test_default_search_with_operator_chars_does_not_leak_fts5_errors(tmp_path: Path) -> None:
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        upsert_message_fts(sidecar_conn, "msg-ident", "the REQ-BRIDGE page-bridge handler", "")

        # Previously these raised raw sqlite3 FTS5 errors; now they run cleanly.
        for query in ["REQ-BRIDGE", "page-bridge", "foo:bar", "foo(bar)"]:
            search_messages_fts(sidecar_conn, query, [], limit=10)

        # A plain identifier is treated as terms and matches the indexed message.
        hits = search_messages_fts(sidecar_conn, "REQ-BRIDGE", [], limit=10)
        assert hits == [("msg-ident", hits[0][1])]
    finally:
        sidecar_conn.close()


def test_message_batch_preserves_existing_rowid_and_last_write_wins(tmp_path: Path) -> None:
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        upsert_message_fts(
            sidecar_conn,
            "msg-stable",
            "firstcontentterm",
            "firstthinkingterm",
        )
        existing_rowid = sidecar_conn.execute(
            "SELECT rowid FROM message_fts_rowid WHERE message_id = ?",
            ["msg-stable"],
        ).fetchone()
        assert existing_rowid == (1,)

        upsert_message_fts_batch(
            sidecar_conn,
            [
                ("msg-stable", "secondcontentterm", "secondthinkingterm"),
                ("msg-stable", "finalcontentterm", "finalthinkingterm"),
            ],
        )

        assert sidecar_conn.execute(
            "SELECT rowid FROM message_fts_rowid WHERE message_id = ?",
            ["msg-stable"],
        ).fetchone() == (1,)
        final_hits = search_messages_fts(sidecar_conn, "finalcontentterm", ["content"], 10)
        assert final_hits == [("msg-stable", final_hits[0][1])]
        assert search_messages_fts(sidecar_conn, "firstcontentterm", ["content"], 10) == []
        assert search_messages_fts(sidecar_conn, "secondcontentterm", ["content"], 10) == []
        thinking_hits = search_messages_fts(
            sidecar_conn,
            "finalthinkingterm",
            ["thinking"],
            10,
        )
        assert thinking_hits == [("msg-stable", thinking_hits[0][1])]
    finally:
        sidecar_conn.close()


def test_message_batch_duplicate_across_boundaries_keeps_rowid_and_last_terms(
    tmp_path: Path,
) -> None:
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        upsert_message_fts_batch(
            sidecar_conn,
            [
                ("msg-dup", "boundaryfirstterm", "boundaryfirstthink"),
                ("msg-other", "othervisibleterm", ""),
                ("msg-dup", "boundaryfinalterm", "boundaryfinalthink"),
            ],
            batch_size=2,
        )

        assert sidecar_conn.execute(
            "SELECT rowid, message_id FROM message_fts_rowid ORDER BY rowid"
        ).fetchall() == [(1, "msg-dup"), (2, "msg-other")]
        final_hits = search_messages_fts(sidecar_conn, "boundaryfinalterm", ["content"], 10)
        other_hits = search_messages_fts(sidecar_conn, "othervisibleterm", ["content"], 10)
        assert final_hits == [("msg-dup", final_hits[0][1])]
        assert other_hits == [("msg-other", other_hits[0][1])]
        assert search_messages_fts(sidecar_conn, "boundaryfirstterm", ["content"], 10) == []
        think_hits = search_messages_fts(
            sidecar_conn,
            "boundaryfinalthink",
            ["thinking"],
            10,
        )
        assert think_hits == [("msg-dup", think_hits[0][1])]
    finally:
        sidecar_conn.close()


def test_message_batch_field_scoping_omits_unselected_columns(tmp_path: Path) -> None:
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        upsert_message_fts_batch(
            sidecar_conn,
            [("msg-scope", "scopedcontentterm", "scopedthinkingterm")],
            fields=("content",),
        )

        content_hits = search_messages_fts(sidecar_conn, "scopedcontentterm", ["content"], 10)
        assert content_hits == [("msg-scope", content_hits[0][1])]
        assert (
            search_messages_fts(
                sidecar_conn,
                "scopedthinkingterm",
                ["content", "thinking"],
                10,
            )
            == []
        )
    finally:
        sidecar_conn.close()


def test_message_batch_trigger_rolls_back_current_batch_and_keeps_prior(
    tmp_path: Path,
) -> None:
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        sidecar_conn.execute(
            """
            CREATE TRIGGER reject_msg_reject
            BEFORE INSERT ON message_fts_rowid
            WHEN new.message_id = 'msg-reject'
            BEGIN
                SELECT RAISE(ABORT, 'rejected mapping row');
            END
            """
        )
        sidecar_conn.commit()

        with pytest.raises(sqlite3.DatabaseError, match="rejected mapping row"):
            upsert_message_fts_batch(
                sidecar_conn,
                [
                    ("msg-keep-a", "keepaterm", ""),
                    ("msg-keep-b", "keepbterm", ""),
                    ("msg-rolled", "rolledterm", ""),
                    ("msg-reject", "rejectterm", ""),
                ],
                batch_size=2,
            )

        assert sidecar_conn.execute(
            "SELECT rowid, message_id FROM message_fts_rowid ORDER BY rowid"
        ).fetchall() == [(1, "msg-keep-a"), (2, "msg-keep-b")]
        keep_a = search_messages_fts(sidecar_conn, "keepaterm", ["content"], 10)
        keep_b = search_messages_fts(sidecar_conn, "keepbterm", ["content"], 10)
        assert keep_a == [("msg-keep-a", keep_a[0][1])]
        assert keep_b == [("msg-keep-b", keep_b[0][1])]
        assert search_messages_fts(sidecar_conn, "rolledterm", ["content"], 10) == []
        assert search_messages_fts(sidecar_conn, "rejectterm", ["content"], 10) == []
    finally:
        sidecar_conn.close()


def test_message_batch_resolves_rowids_across_lookup_chunks(tmp_path: Path) -> None:
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        rows = [(f"msg-chunk-{idx}", f"chunkcontentterm{idx}", "") for idx in range(501)]
        upsert_message_fts_batch(sidecar_conn, rows)

        assert sidecar_conn.execute("SELECT count(*) FROM message_fts_rowid").fetchone() == (501,)
        first_hits = search_messages_fts(sidecar_conn, "chunkcontentterm0", ["content"], 10)
        boundary_hits = search_messages_fts(
            sidecar_conn,
            "chunkcontentterm500",
            ["content"],
            10,
        )
        assert first_hits == [("msg-chunk-0", first_hits[0][1])]
        assert boundary_hits == [("msg-chunk-500", boundary_hits[0][1])]
        assert sidecar_conn.execute(
            "SELECT rowid FROM message_fts_rowid WHERE message_id = ?",
            ["msg-chunk-500"],
        ).fetchone() == (501,)
    finally:
        sidecar_conn.close()


def test_tool_call_search_with_operator_chars_does_not_leak_fts5_errors(tmp_path: Path) -> None:
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        upsert_tool_call_fts(sidecar_conn, "tool-ident", "grep REQ-BRIDGE src")

        for query in ["REQ-BRIDGE", "foo:bar", "foo(bar)"]:
            search_tool_calls_fts(sidecar_conn, query, limit=10)

        hits = search_tool_calls_fts(sidecar_conn, "REQ-BRIDGE", limit=10)
        assert hits == [("tool-ident", hits[0][1])]
    finally:
        sidecar_conn.close()


def test_tool_batch_interleaved_null_reinsert_and_duplicates_across_boundaries(
    tmp_path: Path,
) -> None:
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        upsert_tool_call_fts(sidecar_conn, "tool-reinsert", "echo beforedeleteterm")
        assert sidecar_conn.execute(
            "SELECT rowid FROM tool_calls_fts_rowid WHERE tool_call_id = ?",
            ["tool-reinsert"],
        ).fetchone() == (1,)

        upsert_tool_call_fts_batch(
            sidecar_conn,
            [
                ("tool-dup", "echo firstdupterm"),
                ("tool-other", "echo othervisibleterm"),
                ("tool-reinsert", None),
                ("tool-reinsert", "echo afterreinsertterm"),
                ("tool-dup", "echo finaldupterm"),
            ],
            batch_size=2,
        )

        assert sidecar_conn.execute(
            "SELECT rowid, tool_call_id FROM tool_calls_fts_rowid ORDER BY rowid"
        ).fetchall() == [
            (2, "tool-dup"),
            (3, "tool-other"),
            (4, "tool-reinsert"),
        ]
        dup_hits = search_tool_calls_fts(sidecar_conn, "finaldupterm", 10)
        other_hits = search_tool_calls_fts(sidecar_conn, "othervisibleterm", 10)
        reinsert_hits = search_tool_calls_fts(sidecar_conn, "afterreinsertterm", 10)
        assert dup_hits == [("tool-dup", dup_hits[0][1])]
        assert other_hits == [("tool-other", other_hits[0][1])]
        assert reinsert_hits == [("tool-reinsert", reinsert_hits[0][1])]
        assert search_tool_calls_fts(sidecar_conn, "firstdupterm", 10) == []
        assert search_tool_calls_fts(sidecar_conn, "beforedeleteterm", 10) == []
    finally:
        sidecar_conn.close()


def test_tool_batch_disabled_bash_removes_existing_entries(tmp_path: Path) -> None:
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        upsert_tool_call_fts(sidecar_conn, "tool-disabled", "echo visiblebashterm")
        upsert_tool_call_fts_batch(
            sidecar_conn,
            [("tool-disabled", "echo stillbashterm"), ("tool-new", "echo newbashterm")],
            fields=("content",),
        )

        assert (
            sidecar_conn.execute("SELECT tool_call_id FROM tool_calls_fts_rowid").fetchall() == []
        )
        assert search_tool_calls_fts(sidecar_conn, "visiblebashterm", 10) == []
        assert search_tool_calls_fts(sidecar_conn, "stillbashterm", 10) == []
        assert search_tool_calls_fts(sidecar_conn, "newbashterm", 10) == []
    finally:
        sidecar_conn.close()


def test_tool_batch_trigger_rolls_back_delete_and_keeps_prior_batch(tmp_path: Path) -> None:
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        upsert_tool_call_fts_batch(
            sidecar_conn,
            [
                ("tool-keep-a", "echo keepaterm"),
                ("tool-keep-b", "echo keepbterm"),
                ("tool-restore", "echo restoreterm"),
            ],
            batch_size=2,
        )
        sidecar_conn.execute(
            """
            CREATE TRIGGER reject_tool_reject
            BEFORE INSERT ON tool_calls_fts_rowid
            WHEN new.tool_call_id = 'tool-reject'
            BEGIN
                SELECT RAISE(ABORT, 'rejected tool mapping row');
            END
            """
        )
        sidecar_conn.commit()

        with pytest.raises(sqlite3.DatabaseError, match="rejected tool mapping row"):
            upsert_tool_call_fts_batch(
                sidecar_conn,
                [
                    ("tool-restore", None),
                    ("tool-reject", "echo rejectterm"),
                ],
                batch_size=2,
            )

        assert sidecar_conn.execute(
            "SELECT rowid, tool_call_id FROM tool_calls_fts_rowid ORDER BY rowid"
        ).fetchall() == [
            (1, "tool-keep-a"),
            (2, "tool-keep-b"),
            (3, "tool-restore"),
        ]
        keep_a = search_tool_calls_fts(sidecar_conn, "keepaterm", 10)
        keep_b = search_tool_calls_fts(sidecar_conn, "keepbterm", 10)
        restored = search_tool_calls_fts(sidecar_conn, "restoreterm", 10)
        assert keep_a == [("tool-keep-a", keep_a[0][1])]
        assert keep_b == [("tool-keep-b", keep_b[0][1])]
        assert restored == [("tool-restore", restored[0][1])]
        assert search_tool_calls_fts(sidecar_conn, "rejectterm", 10) == []
    finally:
        sidecar_conn.close()


@pytest.mark.parametrize("reject_reinsert", [False, True])
def test_large_tool_deletion_run_preserves_reinsert_order_and_batch_rollback(
    tmp_path: Path, reject_reinsert: bool
) -> None:
    with closing(open_sidecar(tmp_path / "large-deletion.sqlite")) as sidecar:
        upsert_tool_call_fts_batch(sidecar, [(f"tool-{i}", "oldterm") for i in range(600)])
        before = sidecar.execute(
            "SELECT rowid, tool_call_id FROM tool_calls_fts_rowid ORDER BY rowid"
        ).fetchall()
        if reject_reinsert:
            sidecar.execute("""
                CREATE TRIGGER reject_reinsert BEFORE INSERT ON tool_calls_fts_rowid
                WHEN new.tool_call_id = 'tool-0'
                BEGIN SELECT RAISE(ABORT, 'reinsert rejected'); END
            """)
            sidecar.commit()
        rows: list[tuple[str, str | None]] = [(f"tool-{i}", None) for i in range(501)]
        rows.extend([("tool-0", None), ("absent", None), ("tool-0", "newterm")])
        if reject_reinsert:
            with pytest.raises(sqlite3.DatabaseError, match="reinsert rejected"):
                upsert_tool_call_fts_batch(sidecar, rows)
            expected = before
            old_count = 600
        else:
            upsert_tool_call_fts_batch(sidecar, rows)
            expected = [*before[501:], (601, "tool-0")]
            old_count = 99
        assert (
            sidecar.execute(
                "SELECT rowid, tool_call_id FROM tool_calls_fts_rowid ORDER BY rowid"
            ).fetchall()
            == expected
        )
        assert sidecar.execute(
            "SELECT count(*) FROM tool_calls_fts WHERE tool_calls_fts MATCH 'oldterm'"
        ).fetchone() == (old_count,)
        assert sidecar.execute(
            "SELECT count(*) FROM tool_calls_fts WHERE tool_calls_fts MATCH 'newterm'"
        ).fetchone() == (0 if reject_reinsert else 1,)


@pytest.mark.parametrize("batch_size", [1, 3, 5000])
def test_absent_tool_deletions_preserve_effective_order_and_rowid_reuse(
    tmp_path: Path, batch_size: int
) -> None:
    with closing(open_sidecar(tmp_path / "ordered.sqlite")) as sidecar:
        upsert_tool_call_fts_batch(sidecar, [("keep", "keepterm"), ("retire", "oldterm")])
        upsert_tool_call_fts_batch(
            sidecar,
            [
                ("a", "firstterm"),
                ("absent", None),
                ("b", "bterm"),
                ("retire", None),
                ("c", "temporaryterm"),
                ("c", None),
                ("absent", None),
                ("d", "dterm"),
                ("c", "cterm"),
                ("a", None),
                ("a", "aterm"),
            ],
            batch_size=batch_size,
        )
        assert sidecar.execute(
            "SELECT rowid, tool_call_id FROM tool_calls_fts_rowid ORDER BY rowid"
        ).fetchall() == [(1, "keep"), (4, "b"), (5, "d"), (6, "c"), (7, "a")]
        for key in ("keep", "a", "b", "c", "d"):
            assert [hit[0] for hit in search_tool_calls_fts(sidecar, key + "term", 10)] == [key]
        for term in ("firstterm", "oldterm", "temporaryterm"):
            assert search_tool_calls_fts(sidecar, term, 10) == []


def test_absent_tool_deletion_planning_rolls_back_on_late_mapping_failure(tmp_path: Path) -> None:
    with closing(open_sidecar(tmp_path / "rollback.sqlite")) as sidecar:
        upsert_tool_call_fts_batch(sidecar, [("keep", "keepterm"), ("retire", "oldterm")])
        sidecar.execute("""
            CREATE TRIGGER reject_late BEFORE INSERT ON tool_calls_fts_rowid
            WHEN new.tool_call_id = 'reject'
            BEGIN SELECT RAISE(ABORT, 'late mapping failure'); END
        """)
        sidecar.commit()
        with pytest.raises(sqlite3.DatabaseError, match="late mapping failure"):
            upsert_tool_call_fts_batch(
                sidecar,
                [("absent", None), ("a", "aterm"), ("retire", None), ("reject", "badterm")],
            )
        assert sidecar.execute(
            "SELECT rowid, tool_call_id FROM tool_calls_fts_rowid ORDER BY rowid"
        ).fetchall() == [(1, "keep"), (2, "retire")]
        assert [hit[0] for hit in search_tool_calls_fts(sidecar, "oldterm", 10)] == ["retire"]
        assert search_tool_calls_fts(sidecar, "aterm", 10) == []
        assert search_tool_calls_fts(sidecar, "badterm", 10) == []
