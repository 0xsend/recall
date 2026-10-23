"""Live pages preserve each session's own freshness, epoch and surviving stop marker."""

import hashlib
from datetime import datetime
from pathlib import Path

import duckdb
from recall.db.schema import ensure_schema
from recall.services.live import TurnPhase, decode_cursor, live_view, live_view_page


def test_live_rows_keep_catalog_bindings_and_surviving_stop_markers(tmp_path: Path) -> None:
    now = datetime.now()
    names = ("working", "awaiting", "empty", "unbound")
    paths = {name: tmp_path / f"{name}'雪.jsonl" for name in names}
    unseen = tmp_path / "unseen.jsonl"
    unseen.write_text("unindexed\n")
    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn)
        for name, path in paths.items():
            path.write_text("transcript\n")
            stat = path.stat()
            conn.execute("INSERT INTO sessions VALUES (?, 'codex', ?, ?)", [name, str(path), name])
            conn.execute(
                "INSERT INTO session_state (session_id, file_mtime, file_size) VALUES (?, ?, ?)",
                [name, stat.st_mtime, stat.st_size],
            )
            conn.execute(
                """INSERT INTO source_files
                (source_key, source, source_path, root_path, session_id, dev, inode,
                 ctime_ns, mtime_ns, size, sidecar_signature, desired_generation,
                 committed_generation, committed_offset, content_epoch, observed_at)
                VALUES (?, 'codex', ?, ?, ?, ?, ?, ?, ?, ?, ?, 3, 3, ?, 7, ?)""",
                [
                    name,
                    str(path),
                    str(tmp_path),
                    "another-session" if name == "unbound" else name,
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_ctime_ns,
                    stat.st_mtime_ns,
                    stat.st_size,
                    hashlib.sha256(b"").hexdigest(),
                    stat.st_size,
                    now.timestamp(),
                ],
            )
            if name != "empty":
                for idx, role in enumerate(("user", "assistant")):
                    message_id = f"{name}-{idx}"
                    conn.execute(
                        "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
                        [message_id, name, idx],
                    )
                    conn.execute(
                        "INSERT INTO message_state (message_id, role, content) VALUES (?, ?, ?)",
                        [message_id, role, message_id],
                    )
            conn.execute("INSERT INTO session_stop_markers VALUES (?, 99, 'orphan', TRUE)", [name])
            if name != "empty":
                conn.execute(
                    "INSERT INTO session_stop_markers VALUES (?, 1, ?, ?)",
                    [name, "end_turn" if name == "awaiting" else "max_tokens", name == "awaiting"],
                )
        # The same path under another source is not this session's catalog boundary.
        conn.execute(
            """INSERT INTO source_files
            (source_key, source, source_path, root_path, session_id, ctime_ns,
             mtime_ns, size, content_epoch, observed_at)
            VALUES ('alias', 'claude_code', ?, ?, 'wrong-session', 0, 0, 0, 99, 0)""",
            [str(paths["working"]), str(tmp_path)],
        )
        watched = [*(str(path) for path in paths.values()), str(unseen)]
        views = [
            live_view(
                conn=conn,
                watched_paths=watched,
                include_idle=True,
                now=now,
                idle_window_seconds=86400,
                local_host="fixture-host",
                limit=256,
            ),
            live_view_page(
                conn=conn,
                watched_paths=watched,
                include_idle=True,
                now=now,
                idle_window_seconds=86400,
                local_host="fixture-host",
                limit=256,
                cursor=None,
            ).sessions,
        ]
        for sessions in views:
            by_id = {row.id: row for row in sessions}
            assert len(sessions) == 5
            assert by_id["working"].turn.state is TurnPhase.WORKING
            assert by_id["working"].turn.stop_reason == "max_tokens"
            assert by_id["awaiting"].turn.state is TurnPhase.AWAITING_INPUT
            assert by_id["awaiting"].turn.stop_reason == "end_turn"
            assert by_id["empty"].turn.state is TurnPhase.UNKNOWN
            assert by_id["empty"].turn.stop_reason is None
            assert by_id["unbound"].freshness.limitations == ("catalog_progress_unavailable",)
            assert by_id["unbound"].freshness.content_epoch is None
            assert by_id[None].path == str(unseen)
            assert by_id[None].cursor is None
            for name in ("working", "awaiting", "empty"):
                assert by_id[name].freshness.current is True
                assert by_id[name].freshness.content_epoch == 7
                encoded_cursor = by_id[name].cursor
                assert encoded_cursor is not None
                cursor = decode_cursor(encoded_cursor)
                assert cursor.content_epoch == 7


def test_live_tail_keeps_main_thread_text_when_newer_message_state_is_missing(
    tmp_path: Path,
) -> None:
    now = datetime.now()
    contents = {idx: f"{idx}𝄞é" * 200 for idx in range(9)}
    path = tmp_path / "tail.jsonl"
    path.write_text("transcript\n")
    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn)
        conn.execute("INSERT INTO sessions VALUES ('tail', 'codex', ?, 'tail')", [str(path)])
        conn.execute(
            "INSERT INTO session_state (session_id, file_mtime, file_size) VALUES ('tail', ?, ?)",
            [path.stat().st_mtime, path.stat().st_size],
        )
        for idx in range(19):
            conn.execute(
                "INSERT INTO messages (id, session_id, idx, agent_id) VALUES (?, 'tail', ?, ?)",
                [f"tail-{idx}", idx, "child" if idx == 8 else None],
            )
            if idx < 9:
                conn.execute(
                    """INSERT INTO message_state (message_id, role, content, timestamp)
                    VALUES (?, ?, ?, ?)""",
                    [f"tail-{idx}", "assistant" if idx % 2 == 0 else "user", contents[idx], now],
                )
        sessions = live_view_page(
            conn=conn,
            watched_paths=[str(path)],
            include_idle=True,
            now=now,
            idle_window_seconds=86400,
            limit=256,
            cursor=None,
            local_host="fixture-host",
        ).sessions
        assert len(sessions) == 1
        turn = sessions[0].turn
        assert turn.last_user_text == contents[7][:400]
        assert turn.last_assistant_text == contents[6][:400]
        assert turn.last_user_at == now
        assert turn.last_assistant_at == now
