"""Multi-TZ UTC storage, wire contract, fleet merge, display, and version-2 migration tests."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest
from recall.cli.contract import serialize_data
from recall.cli.utils import format_datetime, json_default
from recall.core.host_text import HostValue
from recall.core.rpc_types import serialize_rpc_value
from recall.core.time import format_utc_iso, to_naive_utc, to_utc
from recall.core.types import Source
from recall.db.fts_sidecar import ensure_sidecar
from recall.db.fts_sidecar_stage import OwnedKeywordTurn
from recall.db.lease import StandaloneLifetime
from recall.db.queries import insert_tool_results
from recall.db.schema import INDEX_MIGRATION_VERSION, ensure_schema
from recall.db.source_files import SourceCatalog, source_key
from recall.parsers.codex import CodexParser
from recall.services.coordinator import (
    capture_path,
    commit_prepared_raw_sources,
    observe_path,
    prepare_raw_sources,
)
from recall.services.fleet_endpoints import EndpointSuccess, FleetExecutor
from recall.services.fleet_query import (
    fleet_list,
    fleet_live,
    fleet_search,
    fleet_show,
)
from recall.services.index_migration import (
    applied_version,
    begin_migration,
    is_eligible,
)
from recall.services.indexer import _load_context_recompute_groups
from recall.services.live import Liveness, derive_liveness, live_roster_query
from recall.services.live_marks import upsert_live_mark
from recall.services.live_session_set import LiveRoster
from recall.services.moves import find_moved_predecessors
from recall.services.sessions import list_sessions
from recall.services.usage_harvest import LogCursor, _insert_new_events, _save_cursor, _UsageEvent
from recall.services.watcher import index_single_session_on

from tests.test_services.test_run_completion import _config


def _set_tz(monkeypatch: pytest.MonkeyPatch, tz_name: str) -> None:
    monkeypatch.setenv("TZ", tz_name)
    if hasattr(time, "tzset"):
        time.tzset()


def _codex_lines(
    *, session_id: str, include_second: bool = False, second_dup_result: bool = False
) -> str:
    records: list[dict[str, Any]] = [
        {
            "timestamp": "2026-07-15T18:00:00.000Z",
            "type": "session_meta",
            "payload": {"id": session_id, "cwd": "/tmp/proj", "cli_version": "0.1.0"},
        },
        {
            "timestamp": "2026-07-15T18:00:01.000Z",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "check utc storage"},
        },
        {
            "timestamp": "2026-07-15T18:00:02.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": "echo hi"}),
                "call_id": "call_1",
            },
        },
        {
            "timestamp": "2026-07-15T18:00:03.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "first result",
            },
        },
    ]
    if include_second:
        records.append(
            {
                "timestamp": "2026-07-15T18:05:00.000Z",
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "second turn"},
            }
        )
    if second_dup_result:
        records.append(
            {
                "timestamp": "2026-07-15T18:06:00.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "second duplicate result",
                },
            }
        )
    return "\n".join(json.dumps(r) for r in records) + "\n"


def _ingest_snapshot(
    tmp_path: Path, tz_name: str, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    _set_tz(monkeypatch, tz_name)
    root = tmp_path / tz_name.replace("/", "_")
    root.mkdir(parents=True, exist_ok=True)
    transcript = root / "rollout-2026-07-15-s1.jsonl"
    transcript.write_text(_codex_lines(session_id="native_s1"), encoding="utf-8")
    os.utime(transcript, (1784138403.0, 1784138403.0))

    config = _config(root / "data", root)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    parser = CodexParser(roots=(root,))

    with duckdb.connect(str(config.db_path)) as conn:
        conn.execute(f"SET TimeZone = '{tz_name}'")
        ensure_schema(conn)
        ensure_sidecar(config)
        with StandaloneLifetime().lease(conn) as lease:
            turn = OwnedKeywordTurn.for_config(lease, config)
            index_single_session_on(lease, transcript, parser, config, turn=turn, host="h1")

            # Incremental suffix append + duplicate tool result (first-wins across batches!).
            transcript.write_text(
                _codex_lines(session_id="native_s1", include_second=True, second_dup_result=True),
                encoding="utf-8",
            )
            os.utime(transcript, (1784138760.0, 1784138760.0))
            index_single_session_on(lease, transcript, parser, config, turn=turn, host="h1")

        # Also exercise operational writers (live_marks, usage_events, usage_log_cursors).
        upsert_live_mark(
            conn,
            source="codex",
            source_session_id="native_s1",
            host="h1",
            pid=42,
            marked_at=datetime(2026, 7, 15, 18, 7, 0, tzinfo=UTC),
        )
        _insert_new_events(
            conn,
            [
                _UsageEvent(
                    id="ue1",
                    source_session_id="native_s1",
                    ts=datetime(2026, 7, 15, 18, 8, 0, tzinfo=UTC),
                    prompt_tokens=10,
                    cached_prompt_tokens=0,
                    completion_tokens=5,
                    reasoning_tokens=0,
                )
            ],
            host="h1",
            harvested_at=datetime(2026, 7, 15, 18, 9, 0, tzinfo=UTC),
        )
        _save_cursor(conn, "/tmp/u.jsonl", LogCursor(byte_offset=10, file_size=10))

        # Verify cross-device move lookup (new inode, different path, aware started_at).
        moved = root / "moved-rollout.jsonl"
        moved.write_text(transcript.read_text(encoding="utf-8"), encoding="utf-8")
        transcript.unlink()
        # Seed committed catalog proof for departed source so move proof succeeds.
        orig_row = conn.execute("SELECT id FROM sessions").fetchone()
        assert orig_row is not None
        orig_sid = orig_row[0]
        full_capture = parser.parse(moved)
        assert full_capture is not None
        conn.execute(
            """INSERT INTO source_files(
                source_key, source, source_path, root_path, session_id,
                dev, inode, ctime_ns, mtime_ns, size,
                committed_offset, committed_prefix_sha256, observed_at
            ) VALUES (?, 'codex', ?, ?, ?, 0, 0, 1, 1, ?, ?, ?, 1.0)""",
            [
                source_key("codex", str(transcript)),
                str(transcript),
                str(root),
                orig_sid,
                full_capture.next_byte_offset,
                full_capture.next_byte_offset,
                full_capture.committed_prefix_sha256,
            ],
        )
        moved_predecessors = find_moved_predecessors(conn, full_capture, host="h1")

        # Verify list_sessions --since and live_roster_query under this TZ.
        listed_in = list_sessions(
            source=Source.CODEX,
            since=datetime(2026, 7, 15, 18, 4, 0, tzinfo=UTC),
            project=None,
            config=config,
            conn=conn,
        )
        listed_out = list_sessions(
            source=Source.CODEX,
            since=datetime(2026, 7, 15, 18, 6, 1, tzinfo=UTC),
            project=None,
            config=config,
            conn=conn,
        )
        recompute_in = _load_context_recompute_groups(
            conn, since=datetime(2026, 7, 15, 18, 4, 0, tzinfo=UTC), only_mode=None
        )
        recompute_out = _load_context_recompute_groups(
            conn, since=datetime(2026, 7, 15, 18, 6, 1, tzinfo=UTC), only_mode=None
        )

        return {
            "session_ts": conn.execute("SELECT started_at, ended_at FROM session_state").fetchall(),
            "message_ts": conn.execute(
                "SELECT timestamp FROM message_state ORDER BY timestamp"
            ).fetchall(),
            "tool_results": conn.execute(
                "SELECT result_summary, is_error, completed_at FROM tool_results"
            ).fetchall(),
            "live_mark": conn.execute("SELECT marked_at FROM live_marks").fetchall(),
            "usage_event": conn.execute("SELECT ts, harvested_at FROM usage_events").fetchall(),
            "moved_found": moved_predecessors == (orig_sid,),
            "listed_in_count": len(listed_in),
            "listed_out_count": len(listed_out),
            "listed_last_activity_json": serialize_data(listed_in[0].last_activity_at),
            "recompute_in_count": len(recompute_in),
            "recompute_out_count": len(recompute_out),
        }


def test_chicago_and_utc_produce_identical_duckdb_rows_moves_and_wire_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chicago = _ingest_snapshot(tmp_path, "America/Chicago", monkeypatch)
    utc = _ingest_snapshot(tmp_path, "UTC", monkeypatch)

    assert chicago == utc
    assert chicago["session_ts"] == [
        (datetime(2026, 7, 15, 18, 0, 0), datetime(2026, 7, 15, 18, 6, 0))
    ]
    assert chicago["tool_results"] == [("first result", False, datetime(2026, 7, 15, 18, 0, 3))]
    assert chicago["live_mark"] == [(datetime(2026, 7, 15, 18, 7, 0),)]
    assert chicago["usage_event"] == [
        (datetime(2026, 7, 15, 18, 8, 0), datetime(2026, 7, 15, 18, 9, 0))
    ]
    assert chicago["moved_found"] is True
    assert chicago["listed_in_count"] == 1
    assert chicago["listed_out_count"] == 0
    assert chicago["listed_last_activity_json"] == "2026-07-15T18:06:00Z"
    assert chicago["recompute_in_count"] == 1
    assert chicago["recompute_out_count"] == 0


def test_duplicate_tool_results_first_wins_and_v2_updates_only_completed_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_tz(monkeypatch, "America/Chicago")
    db_path = tmp_path / "dup.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("SET TimeZone = 'America/Chicago'")
        ensure_schema(conn)
        # Seed legacy local-naive completed_at on tc1.
        insert_tool_results(
            conn,
            [("tc1", "first summary", False, datetime(2026, 7, 15, 13, 0, 3))],
            update_completed_at=False,
        )
        # Ordinary suffix append with duplicate result for tc1 must keep first-wins (no overwrite).
        insert_tool_results(
            conn,
            [("tc1", "later overwrite", True, datetime(2026, 7, 15, 18, 9, 9, tzinfo=UTC))],
            update_completed_at=False,
        )
        assert conn.execute(
            "SELECT result_summary, is_error, completed_at "
            "FROM tool_results WHERE tool_call_id = 'tc1'"
        ).fetchone() == ("first summary", False, datetime(2026, 7, 15, 13, 0, 3))

        # Full reparse (v2 UTC migration) with two results for tc1 deduplicates first-wins and
        # updates ONLY completed_at to UTC, keeping result_summary/is_error untouched.
        insert_tool_results(
            conn,
            [
                ("tc1", "different summary", True, datetime(2026, 7, 15, 18, 0, 3, tzinfo=UTC)),
                ("tc1", "second dup", True, datetime(2026, 7, 15, 18, 9, 9, tzinfo=UTC)),
            ],
            update_completed_at=True,
        )
        assert conn.execute(
            "SELECT result_summary, is_error, completed_at "
            "FROM tool_results WHERE tool_call_id = 'tc1'"
        ).fetchone() == ("first summary", False, datetime(2026, 7, 15, 18, 0, 3))


def test_wire_serializers_emit_z_and_format_datetime_localizes_text_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summer_naive_utc = datetime(2026, 7, 15, 18, 30, 0)
    winter_aware_utc = datetime(2026, 11, 22, 18, 30, 0, tzinfo=UTC)

    assert format_utc_iso(summer_naive_utc) == "2026-07-15T18:30:00Z"
    assert serialize_rpc_value(summer_naive_utc) == "2026-07-15T18:30:00Z"
    assert serialize_data(summer_naive_utc) == "2026-07-15T18:30:00Z"
    assert json_default(summer_naive_utc) == "2026-07-15T18:30:00Z"
    assert serialize_data(winter_aware_utc) == "2026-11-22T18:30:00Z"

    _set_tz(monkeypatch, "America/Chicago")
    assert format_datetime(summer_naive_utc) == "2026-07-15 13:30"
    assert format_datetime("2026-07-15T18:30:00Z") == "2026-07-15 13:30"
    assert format_datetime("2026-11-22T18:30:00Z") == "2026-11-22 12:30"

    _set_tz(monkeypatch, "UTC")
    assert format_datetime(summer_naive_utc) == "2026-07-15 18:30"
    assert format_datetime("2026-07-15T18:30:00Z") == "2026-07-15 18:30"


def test_mixed_version_fleet_entries_normalize_all_top_and_nested_timestamps() -> None:
    class FakeExecutor:
        def __init__(self, payloads: list[tuple[str, Any]]) -> None:
            self._payloads = payloads
            self.scope = type(
                "Scope",
                (),
                {"endpoints": tuple(HostValue(n) for n, _ in payloads), "selector": None},
            )()

        def execute(self, operation: Any, *, concurrency: int = 4) -> list[EndpointSuccess]:
            return [
                EndpointSuccess(name=HostValue(name), value=operation.decode(raw))
                for name, raw in self._payloads
            ]

    def _exec(payloads: list[tuple[str, Any]]) -> FleetExecutor:
        return cast(FleetExecutor, FakeExecutor(payloads))

    # fleet_list with mixed v0.48.0 naive and v0.49+ Z rows.
    list_res = fleet_list(
        _exec(
            [
                (
                    "old-peer",
                    [
                        {
                            "id": "s_old",
                            "source": "codex",
                            "started_at": "2026-07-15T17:00:00",
                            "ended_at": "2026-07-15T17:30:00",
                            "indexed_at": "2026-07-15T17:31:00",
                            "last_activity_at": "2026-07-15T17:30:00",
                            "host": "old-peer",
                        }
                    ],
                ),
                (
                    "new-peer",
                    [
                        {
                            "id": "s_new",
                            "source": "codex",
                            "started_at": "2026-07-15T18:00:00Z",
                            "ended_at": "2026-07-15T18:30:00Z",
                            "indexed_at": "2026-07-15T18:31:00Z",
                            "last_activity_at": "2026-07-15T18:30:00Z",
                            "host": "new-peer",
                        }
                    ],
                ),
            ]
        ),
        limit=10,
    )
    assert [r["id"] for r in list_res.rows] == ["s_new", "s_old"]
    assert list_res.rows[1]["started_at"] == "2026-07-15T17:00:00Z"
    assert list_res.rows[1]["ended_at"] == "2026-07-15T17:30:00Z"
    assert list_res.rows[1]["indexed_at"] == "2026-07-15T17:31:00Z"
    assert list_res.rows[1]["last_activity_at"] == "2026-07-15T17:30:00Z"

    # fleet_live with nested turn.* and running_tool.started_at timestamps.
    live_res = fleet_live(
        _exec(
            [
                (
                    "old-peer",
                    {
                        "schema_version": 2,
                        "watching": True,
                        "next_cursor": None,
                        "coverage": {"complete": True},
                        "sessions": [
                            {
                                "id": "live_old",
                                "liveness": "active",
                                "last_activity_at": "2026-07-15T17:59:00",
                                "host": "old-peer",
                                "turn": {
                                    "state": "tool",
                                    "last_user_at": "2026-07-15T17:58:00",
                                    "last_assistant_at": "2026-07-15T17:58:30",
                                    "running_tool": {
                                        "name": "Bash",
                                        "summary": "ls",
                                        "started_at": "2026-07-15T17:58:45",
                                    },
                                },
                            }
                        ],
                    },
                ),
                (
                    "new-peer",
                    {
                        "schema_version": 2,
                        "watching": True,
                        "next_cursor": None,
                        "coverage": {"complete": True},
                        "sessions": [
                            {
                                "id": "live_new",
                                "liveness": "active",
                                "last_activity_at": "2026-07-15T18:00:00Z",
                                "host": "new-peer",
                                "turn": {"state": "idle", "last_user_at": "2026-07-15T18:00:00Z"},
                            }
                        ],
                    },
                ),
            ]
        ),
        limit=10,
    )
    assert [r["id"] for r in live_res.rows] == ["live_new", "live_old"]
    old_live = live_res.rows[1]
    assert old_live["last_activity_at"] == "2026-07-15T17:59:00Z"
    assert old_live["turn"]["last_user_at"] == "2026-07-15T17:58:00Z"
    assert old_live["turn"]["last_assistant_at"] == "2026-07-15T17:58:30Z"
    assert old_live["turn"]["running_tool"]["started_at"] == "2026-07-15T17:58:45Z"

    # fleet_search + fleet_show with nested message timestamps.
    search_res = fleet_search(
        _exec(
            [
                (
                    "old-peer",
                    [
                        {
                            "kind": "message",
                            "session_id": "s_old",
                            "score": 1.0,
                            "timestamp": "2026-07-15 17:00:00",
                            "host": "old-peer",
                        }
                    ],
                )
            ]
        ),
        query="test",
        limit=10,
    )
    assert search_res.rows[0]["timestamp"] == "2026-07-15T17:00:00Z"

    show_res = fleet_show(
        _exec(
            [
                (
                    "old-peer",
                    {
                        "id": "s_old",
                        "started_at": "2026-07-15T17:00:00",
                        "ended_at": "2026-07-15T17:05:00",
                        "indexed_at": "2026-07-15T17:06:00",
                        "host": "old-peer",
                        "messages": [
                            {
                                "id": "m1",
                                "timestamp": "2026-07-15T17:00:01",
                                "tool_calls": [
                                    {"id": "tc1", "completed_at": "2026-07-15T17:00:02"}
                                ],
                            }
                        ],
                    },
                )
            ]
        ),
        session_id="s_old",
    )
    assert show_res.session is not None
    assert show_res.session["started_at"] == "2026-07-15T17:00:00Z"
    assert show_res.session["messages"][0]["timestamp"] == "2026-07-15T17:00:01Z"
    assert (
        show_res.session["messages"][0]["tool_calls"][0]["completed_at"] == "2026-07-15T17:00:02Z"
    )


def test_coordinator_v2_migration_repairs_indexed_at_atomically_and_keeps_v1_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_tz(monkeypatch, "America/Chicago")
    root = tmp_path / "coord"
    root.mkdir()
    transcript = root / "rollout-notime.jsonl"
    # Valid codex session with no timestamps on messages/meta so started_at and ended_at are NULL.
    transcript.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "timeless prompt"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = _config(root / "data", root)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    parser = CodexParser(roots=(root,))
    from recall.db import connect

    with connect(config) as conn:
        conn.execute("SET TimeZone = 'America/Chicago'")
        ensure_sidecar(config)
        catalog = SourceCatalog(conn, clock=time.time)
        observe_path(parser, capture_path(parser, transcript), conn=conn)
        item = catalog.get("codex", str(transcript))
        assert item is not None
        prepared = prepare_raw_sources((item,), {"codex": parser})
        commit_prepared_raw_sources(prepared, config, conn=conn)

        # Simulate pre-v2 DB state with local-naive indexed_at and applied_version = 0.
        legacy_indexed_at = datetime(2026, 7, 15, 13, 0, 0)
        conn.execute("UPDATE session_state SET indexed_at = ?", [legacy_indexed_at])
        conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")

        # 1. Start an in-flight v1 job and finalize under schema 43:
        #    must credit version 1 only, keeping is_eligible(conn) == True for version 2!
        observer = StandaloneLifetime()
        begin_migration(conn, config, observer=observer)
        conn.execute("UPDATE index_migration_jobs SET target_version = 1 WHERE singleton")
        item = catalog.get("codex", str(transcript))
        assert item is not None
        prepared = prepare_raw_sources((item,), {"codex": parser})
        commit_prepared_raw_sources(prepared, config, conn=conn)
        assert applied_version(conn) == 1
        assert is_eligible(conn) is True

        # 2. Reset indexed_at to legacy local-naive and run version-2 migration:
        #    inject failure inside mark_captured_complete_in_tx -> entire acknowledgement rolls back
        #    so source remains pending and retries cleanly on next turn!
        conn.execute("UPDATE session_state SET indexed_at = ?", [legacy_indexed_at])
        begin_migration(conn, config, observer=observer)
        from recall.services import index_migration as im_mod

        real_mark = im_mod.mark_captured_complete_in_tx

        def boom(c: duckdb.DuckDBPyConnection, sk: str) -> None:
            raise RuntimeError("simulated crash at completion")

        monkeypatch.setattr(im_mod, "mark_captured_complete_in_tx", boom)
        item = catalog.get("codex", str(transcript))
        assert item is not None
        prepared = prepare_raw_sources((item,), {"codex": parser})
        commit_prepared_raw_sources(prepared, config, conn=conn)
        # Crash left source incomplete & unacknowledged (not stranded!).
        assert conn.execute("SELECT completed FROM index_migration_scope").fetchone() == (False,)
        assert conn.execute("SELECT indexed_at FROM session_state").fetchone() == (
            legacy_indexed_at,
        )

        # Retry after recovery commits acknowledgement, UTC indexed_at repair, and finalizes v2!
        monkeypatch.setattr(im_mod, "mark_captured_complete_in_tx", real_mark)
        conn.execute("UPDATE source_files SET last_error = NULL, next_retry_at = 0")
        before_utc = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=5)
        item = catalog.get("codex", str(transcript))
        assert item is not None
        prepared = prepare_raw_sources((item,), {"codex": parser})
        commit_prepared_raw_sources(prepared, config, conn=conn)
        after_utc = datetime.now(UTC).replace(tzinfo=None) + timedelta(seconds=5)

        assert conn.execute(
            "SELECT phase, completed_count FROM index_migration_jobs"
        ).fetchone() == (
            "idle",
            1,
        )
        (repaired_indexed_at,) = conn.execute(
            "SELECT indexed_at FROM session_state"
        ).fetchone() or (None,)
        assert repaired_indexed_at is not None
        assert before_utc <= repaired_indexed_at <= after_utc
        assert applied_version(conn) == INDEX_MIGRATION_VERSION == 2
        assert is_eligible(conn) is False


def test_live_roster_and_liveness_use_utc_cutoffs_under_chicago(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_tz(monkeypatch, "America/Chicago")
    now_utc = datetime(2026, 7, 15, 18, 10, 0, tzinfo=UTC)
    recent_naive_utc = datetime(2026, 7, 15, 18, 9, 0)
    old_naive_utc = datetime(2026, 7, 15, 17, 0, 0)

    assert (
        derive_liveness(
            watched=False,
            session_ended=False,
            marked_pid_alive=None,
            last_activity_at=recent_naive_utc,
            now=now_utc,
            idle_window_seconds=300.0,
        )
        is Liveness.IDLE
    )
    assert (
        derive_liveness(
            watched=False,
            session_ended=False,
            marked_pid_alive=None,
            last_activity_at=old_naive_utc,
            now=now_utc,
            idle_window_seconds=300.0,
        )
        is Liveness.UNKNOWN
    )

    with duckdb.connect(str(tmp_path / "live.duckdb")) as conn:
        conn.execute("SET TimeZone = 'America/Chicago'")
        ensure_schema(conn)
        mtime = now_utc.timestamp() - 60.0
        conn.execute(
            "INSERT INTO sessions(id, source, source_path, source_session_id) "
            "VALUES ('s1', 'codex', '/tmp/s1.jsonl', 'n1')"
        )
        conn.execute(
            "INSERT INTO session_state(session_id, ended_at, file_mtime, file_size, host) "
            "VALUES ('s1', ?, ?, 10, 'aem5')",
            [to_naive_utc(now_utc - timedelta(seconds=60)), mtime],
        )
        sql, params = live_roster_query(
            watched_paths=LiveRoster.from_paths(()),
            include_idle=True,
            now=now_utc,
            idle_window_seconds=300.0,
            active_window_seconds=30.0,
        )
        rows = conn.execute(sql + " SELECT * FROM classified WHERE matches", params).fetchall()
        assert len(rows) == 1
        assert to_utc(rows[0][11]) == datetime.fromtimestamp(mtime, tz=UTC).replace(microsecond=0)


def test_cross_device_move_across_schema_42_to_43_transition_retains_session_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-v2 legacy local-naive started_at must still adopt moved transcript across devices."""
    _set_tz(monkeypatch, "America/Chicago")
    root = tmp_path / "move_cutover"
    root.mkdir()
    old_path = root / "old.jsonl"
    old_path.write_text(_codex_lines(session_id="native_cutover"), encoding="utf-8")
    parser = CodexParser(roots=(root,))
    orig_capture = parser.parse(old_path)
    assert orig_capture is not None

    # Move across devices: new path, new inode, original departed.
    new_path = root / "new.jsonl"
    new_path.write_text(old_path.read_text(encoding="utf-8"), encoding="utf-8")
    old_path.unlink()
    moved_capture = parser.parse(new_path)
    assert moved_capture is not None

    with duckdb.connect(str(tmp_path / "move_cutover.duckdb")) as conn:
        conn.execute("SET TimeZone = 'America/Chicago'")
        ensure_schema(conn)
        # Seed legacy schema-42 row with local-naive Chicago started_at (13:00 for 18:00Z).
        for sid, native_id in (("sess_with_native", "native_cutover"), ("sess_no_native", None)):
            conn.execute("DELETE FROM source_files")
            conn.execute("DELETE FROM session_state")
            conn.execute("DELETE FROM sessions")
            conn.execute(
                "INSERT INTO sessions(id, source, source_path, source_session_id) "
                "VALUES (?, 'codex', ?, ?)",
                [sid, str(old_path), native_id],
            )
            conn.execute(
                "INSERT INTO session_state(session_id, started_at, file_mtime, file_size, host) "
                "VALUES (?, TIMESTAMP '2026-07-15 13:00:00', 1.0, 100, 'h1')",
                [sid],
            )
            conn.execute(
                """INSERT INTO source_files(
                    source_key, source, source_path, root_path, session_id,
                    dev, inode, ctime_ns, mtime_ns, size,
                    committed_offset, committed_prefix_sha256, observed_at
                ) VALUES (?, 'codex', ?, ?, ?, 0, 0, 1, 1, ?, ?, ?, 1.0)""",
                [
                    source_key("codex", str(old_path)),
                    str(old_path),
                    str(root),
                    sid,
                    orig_capture.next_byte_offset,
                    orig_capture.next_byte_offset,
                    orig_capture.committed_prefix_sha256,
                ],
            )
            assert find_moved_predecessors(conn, moved_capture, host="h1") == (sid,)

        # Regression: 20 unrelated departed transcripts sharing the same native ID 1..20 min later
        # must NOT displace a proven cross-device continuation (for either UTC or legacy-local row).
        for stored_ts in (datetime(2026, 7, 15, 18, 0, 0), datetime(2026, 7, 15, 13, 0, 0)):
            conn.execute("DELETE FROM source_files")
            conn.execute("DELETE FROM session_state")
            conn.execute("DELETE FROM sessions")
            conn.execute(
                "INSERT INTO sessions(id, source, source_path, source_session_id) "
                "VALUES ('true_pred', 'codex', ?, 'native_cutover')",
                [str(old_path)],
            )
            conn.execute(
                "INSERT INTO session_state(session_id, started_at, file_mtime, file_size, host) "
                "VALUES ('true_pred', ?, 1.0, 100, 'h1')",
                [stored_ts],
            )
            conn.execute(
                """INSERT INTO source_files(
                    source_key, source, source_path, root_path, session_id,
                    dev, inode, ctime_ns, mtime_ns, size,
                    committed_offset, committed_prefix_sha256, observed_at
                ) VALUES (?, 'codex', ?, ?, 'true_pred', 0, 0, 1, 1, ?, ?, ?, 1.0)""",
                [
                    source_key("codex", str(old_path)),
                    str(old_path),
                    str(root),
                    orig_capture.next_byte_offset,
                    orig_capture.next_byte_offset,
                    orig_capture.committed_prefix_sha256,
                ],
            )
            for idx in range(1, 21):
                other_path = str(root / f"other_{idx}.jsonl")
                conn.execute(
                    "INSERT INTO sessions(id, source, source_path, source_session_id) "
                    "VALUES (?, 'codex', ?, 'native_cutover')",
                    [f"other_{idx}", other_path],
                )
                conn.execute(
                    "INSERT INTO session_state"
                    "(session_id, started_at, file_mtime, file_size, host) "
                    "VALUES (?, ?, 1.0, 100, 'h1')",
                    [f"other_{idx}", stored_ts + timedelta(minutes=idx)],
                )
                conn.execute(
                    """INSERT INTO source_files(
                        source_key, source, source_path, root_path, session_id,
                        dev, inode, ctime_ns, mtime_ns, size,
                        committed_offset, committed_prefix_sha256, observed_at
                    ) VALUES (?, 'codex', ?, ?, ?, 0, 0, 1, 1, 50, 50, 'deadbeef', 1.0)""",
                    [source_key("codex", other_path), other_path, str(root), f"other_{idx}"],
                )
            assert find_moved_predecessors(conn, moved_capture, host="h1") == ("true_pred",)
