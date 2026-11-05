"""Generated arrival schedules, keyword witnesses and observed scheduler obligations."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, replace

import pytest

from scripts.recall_bench.arrivals import Hits, Plan, Writer, Written, schedule
from scripts.recall_bench.daemon import add_profile_arguments
from scripts.recall_bench.evidence import JobFailed
from scripts.recall_bench.load import Call
from scripts.recall_bench.pressure import plan_from
from scripts.recall_bench.scheduling_trace import Cost, Decision, parse, summarize


def test_arrival_schedule_is_repeatable_and_delays_both_sources() -> None:
    items = schedule(Plan())
    assert items == schedule(Plan())
    assert len(items) == 304
    assert len({item.relative for item in items}) == 304
    assert len({item.token for item in items}) == 304
    assert [(item.due, item.source) for item in items if item.kind == "probe"] == [
        (3, "claude_code"),
        (3, "codex"),
        (33, "claude_code"),
        (33, "codex"),
    ]
    assert [
        sum(item.kind == "burst" and item.round == round for item in items) for round in range(2)
    ] == [150, 150]
    assert items[0].body(4, "2026-01-01T00:00:00Z") == schedule(Plan())[0].body(
        4, "2026-01-01T00:00:00Z"
    )


def test_history_is_old_work_not_a_post_settle_arrival() -> None:
    items = schedule(Plan(backlog=3), history=True)
    assert [(item.kind, item.source, item.due) for item in items] == [
        ("history", "claude_code", 0),
        ("history", "codex", 0),
        ("history", "claude_code", 0),
    ]


@pytest.mark.parametrize(
    "plan",
    [
        Plan(burst=149),
        Plan(rounds=0),
        Plan(backlog=0),
        Plan(rounds=1000),
        Plan(lines=10001),
        Plan(delay=30),
        Plan(delay=0),
        Plan(age=float("nan")),
    ],
)
def test_invalid_schedules_fail_closed(plan: Plan) -> None:
    with pytest.raises(JobFailed, match=r"."):
        schedule(plan)


def test_shared_pair_profile_parser_accepts_pressure_options() -> None:
    parser = argparse.ArgumentParser()
    add_profile_arguments(parser)
    args = parser.parse_args(
        [
            "--profile",
            "backlog",
            "--with-arrivals",
            "--scheduling-trace",
            "--backlog-files",
            "25000",
        ]
    )
    assert plan_from(args).backlog == 25000
    assert args.with_arrivals and args.scheduling_trace


@pytest.mark.parametrize(
    "options",
    [
        ["--profile", "idle", "--arrival-count", "200"],
        ["--profile", "arrival", "--writes", "20"],
        ["--profile", "arrival", "--with-arrivals"],
        ["--profile", "backlog", "--with-appends", "--live-callers", "2"],
    ],
)
def test_ignored_workload_options_are_refused_before_filesystem_access(options) -> None:
    from scripts.recall_bench.daemon import run
    from scripts.recall_bench.evidence import Result
    from scripts.recall_bench.fs import Cancellation

    parser = argparse.ArgumentParser()
    add_profile_arguments(parser)
    with pytest.raises(JobFailed, match=r".") as caught:
        run(parser.parse_args(options), Result("daemon-run"), Cancellation(), lambda _row: None)
    assert caught.value.kind == "usage"


def witness() -> tuple[Hits, list[dict]]:
    items = tuple(item for item in schedule(Plan(rounds=1)) if item.kind == "probe")
    # Writer's writes are recorded externally; this unit drives the public observer boundary.
    writer = object.__new__(Writer)
    writer.items = items
    writer.written = [Written(item, 10.0, 100, "digest") for item in items]
    records = []
    return Hits(writer, records.append), records


def reply(
    caller: str, text: str, *, lexical: bool = True, started: float = 11, seconds: float = 2
) -> Call:
    return Call(
        caller,
        "workload",
        ("search",),
        started,
        seconds,
        0,
        None,
        {"results": [{"content": text, "lexical_match": lexical}]},
    )


def test_keyword_hit_measures_every_probe_with_nearest_rank_summary() -> None:
    hits, records = witness()
    callers = hits.callers()
    callers["probe-0"]()
    callers["probe-1"]()
    hits.observe(reply("probe-0", hits.writer.items[0].token))
    hits.observe(reply("probe-1", hits.writer.items[1].token, started=14, seconds=4))
    summary = hits.summary("probe")
    assert summary["missing"] == 0
    assert summary["seconds"] == {"count": 2, "median": 5.5, "p95": 8, "max": 8}
    assert [row["seconds"] for row in records] == [3, 8]


def test_nonlexical_wrong_and_prewrite_hits_remain_missing() -> None:
    hits, _records = witness()
    hits.callers()["probe-0"]()
    token = hits.writer.items[0].token
    for call in [
        reply("probe-0", "wrong"),
        reply("probe-0", token, lexical=False),
        reply("probe-0", token, started=9),
    ]:
        hits.observe(call)
    assert hits.summary("probe")["missing"] == 2
    with pytest.raises(JobFailed, match="observer failed"):
        hits.observe(replace(reply("probe-0", token), status=1))
    with pytest.raises(JobFailed, match="no results list"):
        hits.observe(replace(reply("probe-0", token), document={}))


def test_batched_keyword_reply_retains_each_individual_visibility() -> None:
    writer = object.__new__(Writer)
    writer.items = schedule(Plan(rounds=1))[:150]
    writer.written = [Written(item, 10, 100, "digest") for item in writer.items]
    hits = Hits(writer, lambda _row: None)
    hits.callers()["burst-0"]()
    call = replace(
        reply("burst-0", "unused"),
        document={
            "results": [{"content": item.token, "lexical_match": True} for item in writer.items]
        },
    )
    hits.observe(call)
    assert hits.summary("burst")["seconds"] == {"count": 150, "median": 3, "p95": 3, "max": 3}
    assert hits.summary("burst")["missing"] == 0


def decision(seq: int, selected: str, oldest: tuple[str, ...] = ("old",)) -> Decision:
    lane = ("active", "active", "active", "active", "recent", "recent", "oldest")[seq % 7]
    return Decision(
        "decision", seq + 1, seq, seq + 1, ("a",), ("r",), oldest, False, ((selected, lane),), ()
    )


def test_costs_are_inclusive_and_not_added_together() -> None:
    data = summarize(
        [
            Cost("cost", "selection", 1, 2, 1),
            Cost("cost", "query", 1.1, 1.5, 0.8),
            Cost("cost", "query", 5, 100, 10),
        ],
        0,
        4,
    )
    assert data["costs"] == {
        "selection": {"count": 1, "wall": 2, "cpu": 1, "wall_max": 2},
        "query": {"count": 1, "wall": 1.5, "cpu": 0.8, "wall_max": 1.5},
    }


def test_seven_slot_service_and_oldest_reservation() -> None:
    rows = [decision(seq, "a" if seq < 4 else "r" if seq < 6 else "old") for seq in range(7)]
    result = summarize(rows, 0, 10)
    assert result["breaches"] == []
    assert result["lane_counts"] == {"active": 4, "recent": 2, "oldest": 1}
    assert result["oldest_reservations"] == 1
    assert result["bounded_services"] == 1
    assert result["open_obligations"] == 0
    broken = [*rows[:-1], decision(6, "other")]
    breaches = summarize(broken, 0, 10)["breaches"]
    assert isinstance(breaches, list)
    assert "oldest slot did not serve the oldest eligible key" in breaches
    assert "pending observed source exhausted 7*N slots" in breaches


def test_trace_refuses_truncated_unknown_and_invalid_events() -> None:
    with pytest.raises(TypeError):
        parse({"event": "decision"})
    with pytest.raises(ValueError, match="unknown"):
        parse({"event": "other"})
    with pytest.raises(ValueError, match="invalid scheduler"):
        parse(asdict(replace(decision(0, "a"), after=10)))
    assert parse(asdict(Cost("cost", "query", 1, 2, 3))) == Cost("cost", "query", 1, 2, 3)
    with pytest.raises(ValueError, match="clock"):
        parse(asdict(Cost("cost", "query", 1, float("nan"), 0)))


def test_backlog_drain_requires_every_terminal_condition() -> None:
    from scripts.recall_bench.pressure import drained

    good = {
        "reconciliation": {
            "pending": 0,
            "noticed_pending": 0,
            "catalog_scan_complete": True,
            "enrichment_ready": True,
        },
        "catchup_in_progress": False,
    }
    assert drained(good)
    assert not drained({})
    for field, value in (
        ("pending", 1),
        ("noticed_pending", 1),
        ("catalog_scan_complete", False),
        ("enrichment_ready", False),
    ):
        assert not drained({**good, "reconciliation": {**good["reconciliation"], field: value}})
    assert not drained({**good, "catchup_in_progress": True})


def test_daemon_local_trace_records_real_selection_query_and_fetch(tmp_path, monkeypatch) -> None:
    import json
    import subprocess
    import sys
    import textwrap

    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "db"))
    script = textwrap.dedent("""
        import sys, time
        from pathlib import Path
        from recall.core.config import AppConfig
        from recall.db.connection import connect
        from recall.services.coordinator import select_raw_sources, prepare_raw_sources
        from recall.services.reconciler import FairScheduler
        from scripts.recall_bench.scheduling_trace import install
        conn = connect(AppConfig.load())
        install(Path(sys.argv[1]))
        result = select_raw_sources({}, conn=conn, scheduler=FairScheduler(clock=time.time),
                                    active_paths=set())
        assert result == ()
        assert prepare_raw_sources((), {}) == ()
        conn.close()
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
    rows = [
        parse(json.loads(line))
        for line in (tmp_path / "bench-scheduling.jsonl").read_text().splitlines()
    ]
    costs = [row for row in rows if isinstance(row, Cost)]
    assert sum(row.stage == "query_binding" for row in costs) == 3
    assert sum(row.stage == "materialization" for row in costs) == 3
    assert {(row.lane, row.stage) for row in costs if row.stage == "materialization"} == {
        ("active", "materialization"),
        ("recent", "materialization"),
        ("oldest", "materialization"),
    }
    assert {row.stage for row in costs} >= {"selection", "preparation", "roster", "pending"}
    assert len([row for row in rows if isinstance(row, Decision)]) == 1


def test_probe_delay_begins_after_slow_burst_writes(tmp_path) -> None:
    from scripts.recall_bench.fs import Directory

    for source in ("claude_code", "codex"):
        (tmp_path / "pool" / source).mkdir(parents=True)
    now = [100.0]
    records = []

    def record(item) -> None:
        records.append(item)
        now[0] += 0.1

    with Directory(tmp_path, os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)) as root:
        writer = Writer(root, schedule(Plan(rounds=1)), Plan(rounds=1), record, lambda: now[0])
        writer.due(0)
        assert len(writer.written) == 150
        writer.due(16)
        assert len(writer.written) == 150
        writer.due(19)
        assert writer.done
    assert [item["source"] for item in records[-2:]] == ["claude_code", "codex"]
    assert all(item["bytes"] > 0 for item in records)


def test_only_successful_history_commits_inside_the_window_prove_progress() -> None:
    from scripts.recall_bench.pressure import historical_commits

    status = {
        "watch_recent_events": [
            {"path": "history", "timestamp": 11, "success": True},
            {"path": "arrival", "timestamp": 11, "success": True},
            {"path": "failed", "timestamp": 11, "success": False},
            {"path": "early", "timestamp": 9, "success": True},
        ]
    }
    assert historical_commits(status, {"history", "failed", "early"}, 10) == {"history"}
    assert historical_commits({}, {"history"}, 10) == set()
