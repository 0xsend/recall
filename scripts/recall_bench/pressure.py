"""Arrival and actively draining historical-backlog daemon profiles."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from .arrivals import Hits, Plan, Writer, schedule
from .cli import Command
from .evidence import JobFailed, JobIncomplete
from .isolation import POOL_DIR, scan
from .load import Call, repeat, rss_bytes_max, seconds_summary
from .samplers import sample, summarize
from .scheduling_trace import BYTES_MAX, NAME, RECORDS_MAX, STOP_NAME, parse
from .scheduling_trace import summarize as trace_summary
from .turns import TURN_STAGES, read_samples
from .visibility import Visibility
from .workload import Appender

if TYPE_CHECKING:
    from .daemon import Profile


def plan_from(args) -> Plan:
    defaults = Plan()
    plan = Plan(
        getattr(args, "arrival_count", defaults.burst),
        getattr(args, "arrival_rounds", defaults.rounds),
        getattr(args, "probe_delay", defaults.delay),
        getattr(args, "arrival_interval", defaults.interval),
        getattr(args, "backlog_files", defaults.backlog),
        getattr(args, "backlog_age", defaults.age),
        getattr(args, "generated_lines", defaults.lines),
        101 if args.seed is None else args.seed,
    )
    plan.validate()
    return plan


def run(profile: Profile) -> None:
    # Import at runtime to avoid a module cycle and retain the functional test harness's floors.
    from . import daemon as floors

    args = profile.args
    plan = plan_from(args)
    backlog = args.profile == "backlog"
    appender: Appender | None = None
    if args.with_appends:
        appender = profile.seeded(floors.workload_from(args))
    descriptor = profile.tree.open_new(profile.tree.root, PurePosixPath("arrivals.jsonl"))

    def record(item: Mapping[str, object]) -> None:
        data = (json.dumps(item, sort_keys=True) + "\n").encode()
        if os.write(descriptor, data) != len(data):
            raise OSError("short arrival evidence write")

    try:
        with profile.run_dir.held() as root:
            with root.descend("--run-dir", PurePosixPath(POOL_DIR)) as pool:
                nonhistory = len(scan("--run-dir", pool).files)
            history = Writer(root, schedule(plan, history=True) if backlog else (), plan, record)
            history.due(0)
            history_paths = {str(root.path / item.relative) for item in history.items}
            writer = Writer(
                root, schedule(plan) if not backlog or args.with_arrivals else (), plan, record
            )
            hits = Hits(writer, record)
            daemon, _stats = profile.boot(wait_for_settle=not backlog)
            driver = profile.driver(daemon)
            try:
                append_hits = Visibility(
                    driver.appended, profile.witnesses, driver.record_visibility
                )
                started = time.monotonic()
                window_wall_started = time.time()
                initial_cpu = sample(daemon.process)
                initial_status = profile.cli_json(Command.daemon_status())
                pending_samples: list[dict[str, object]] = []
                latest_status: dict[str, object] = dict(initial_status)
                history_commits: set[str] = set()

                def remember_status(call: Call) -> None:
                    if call.argv[:2] == ("daemon", "status") and call.document is not None:
                        latest_status.clear()
                        latest_status.update(call.document)

                def observe(call: Call) -> None:
                    remember_status(call)
                    hits.observe(call)
                    if appender is not None:
                        append_hits.observe(call)
                    if call.argv[:2] != ("daemon", "status"):
                        return
                    if call.document is not None:
                        observe_pending(call.document, call.started + call.seconds)

                def observe_pending(document: Mapping[str, object], at: float) -> None:
                    history_commits.update(
                        historical_commits(document, history_paths, window_wall_started)
                    )
                    reconciliation = floors._reconciliation(document)
                    pending = reconciliation.get("pending")
                    if type(pending) is not int:
                        raise JobFailed("backlog_status", "status reports no pending count")
                    lower = pending - nonhistory - len(writer.written)
                    pending_samples.append(
                        {"at": at, "pending": pending, "historical_pending_lower_bound": lower}
                    )
                    if backlog and lower <= 0:
                        raise JobIncomplete(
                            "backlog_drained",
                            "historical backlog not proved active "
                            "throughout sampling; increase --backlog-files",
                            samples=pending_samples,
                        )

                if backlog:
                    # Readiness need not imply the first historical inventory batch is published.
                    wait_started = time.monotonic()
                    while True:
                        pending = floors._reconciliation(initial_status).get("pending")
                        if type(pending) is int and pending > nonhistory:
                            break
                        profile.waiting(daemon.exited, wait_started, "backlog admission")
                        profile.supervisor.wait_quietly(floors.POLL_SECONDS, profile.deadline)
                        initial_status = profile.cli_json(Command.daemon_status())
                    started = time.monotonic()
                    window_wall_started = time.time()
                    initial_cpu = sample(daemon.process)
                    observe_pending(initial_status, started)
                observers = hits.callers() if writer.items else {}
                if appender is not None:
                    observers.update(append_hits.callers())
                observers["pressure-status"] = repeat(Command.daemon_status())
                duration = args.window if backlog else max(item.due for item in writer.items)
                if appender is not None:
                    duration = max(
                        duration,
                        (appender.writes[-1].tick + 1)
                        * floors.workload_from(args).interval_seconds,
                    )
                profile.phase("backlog-window" if backlog else "arrival-window")
                try:
                    driver.drive(
                        "workload",
                        observers,
                        appender=appender,
                        ticks=range(floors.workload_from(args).writes) if appender else range(0),
                        interval_seconds=floors.workload_from(args).interval_seconds
                        if appender
                        else 0,
                        on_tick=writer.due,
                        until=lambda now: now >= started + duration and writer.done,
                        timeout_seconds=duration + args.settle_timeout,
                        observe=observe,
                    )
                    # An explicit last read closes the backlog evidence window.
                    observe_pending(profile.cli_json(Command.daemon_status()), time.monotonic())
                    if backlog and not history_commits:
                        raise JobIncomplete(
                            "backlog_progress", "no historical commit observed during sampling"
                        )
                finally:
                    window_ended = time.monotonic()
                    if args.scheduling_trace:
                        stop = os.open(
                            STOP_NAME,
                            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                            0o600,
                            dir_fd=root.descriptor,
                        )
                        os.close(stop)
                    window = summarize([initial_cpu, *driver.samples, sample(daemon.process)])
                    profile.result.step(
                        "window",
                        **window,
                        pending_samples=pending_samples,
                        historical_commits=len(history_commits),
                    )
                profile.phase("drain")
                drain_started = time.monotonic()

                def observe_drain(call: Call) -> None:
                    remember_status(call)
                    hits.observe(call)
                    if appender is not None:
                        append_hits.observe(call)

                driver.drive(
                    "visibility-drain",
                    observers,
                    until=lambda _now: (
                        len(hits.seen) == len(writer.items) and not append_hits.unseen()
                    ),
                    timeout_seconds=args.settle_timeout,
                    observe=observe_drain,
                )
                if not drained(latest_status):
                    driver.drive(
                        "drain",
                        {"pressure-status": repeat(Command.daemon_status())},
                        until=lambda _now: drained(latest_status),
                        timeout_seconds=args.settle_timeout - (time.monotonic() - drain_started),
                        observe=observe_drain,
                        caller_interval_seconds=floors.POLL_SECONDS,
                    )
                status = profile.cli_json(Command.daemon_status())
                settled, complete = profile.settle(status, daemon.exited)
                # A tiny fixture can drain without an observer iteration; take an owned sample now.
                driver.samples.append(sample(daemon.process))
                profile.result.step(
                    "settle",
                    seconds=complete.seconds,
                    catalog_seconds=complete.seconds,
                    pending=floors._reconciliation(settled).get("pending"),
                )
                profile.result.step(
                    "drain",
                    seconds=time.monotonic() - drain_started,
                    since_workload_start=time.monotonic() - started,
                    pending=0,
                )
                probes, burst = hits.summary("probe"), hits.summary("burst")
                freshness = probes["seconds"]
                assert isinstance(freshness, dict)
                profile.result.step(
                    "committed",
                    freshness_seconds=freshness,
                    probes=probes,
                    burst=burst,
                    rss_bytes_max=rss_bytes_max(driver.samples),
                )
                turns = profile.turns(daemon, started)
                profile.tree.write_new(
                    profile.tree.root, PurePosixPath("turns.json"), floors._dump(turns)
                )
                with profile.run_dir.held() as held:
                    diagnostics = read_samples(held, daemon.pid)
                    stages = diagnostics[-1].stages
                    profile.result.step(
                        "scheduling",
                        stages={name: stages.get(name) for name in ("raw_select", "raw_prepare")},
                        scope="Inclusive calling-thread CPU; native pool CPU excluded",
                    )
                    if args.scheduling_trace:
                        with held.open_file("--run-dir", NAME) as file:
                            if file.status.st_size > BYTES_MAX:
                                raise JobFailed(
                                    "trace_bound", "scheduling trace exceeds byte limit"
                                )
                            lines = file.head(file.status.st_size).splitlines()
                        if len(lines) > RECORDS_MAX:
                            raise JobFailed("trace_bound", "scheduling trace exceeds record limit")
                        records = [parse(json.loads(line)) for line in lines]
                        tracing = trace_summary(records, started, time.monotonic())
                        profile.result.step(
                            "fairness",
                            **tracing,
                            sampling=trace_summary(records, started, window_ended),
                        )
                        if (
                            not tracing["decisions"]
                            or tracing["breaches"]
                            or tracing["stopped_at"] is None
                        ):
                            raise JobFailed(
                                "fairness",
                                "scheduler trace missing or violated obligations",
                                trace=tracing,
                            )
                final_status = profile.cli_json(Command.daemon_status())
                final_stats = profile.cli_json(Command.stats())
                profile.tree.write_new(
                    profile.tree.root,
                    PurePosixPath("status-final.json"),
                    floors._dump(final_status),
                )
                coverage = floors._reconciliation(final_status).get("coverage")
                if not isinstance(coverage, list):
                    raise JobFailed("coverage", "final status has no coverage")
                discovered = sum(row.get("discovered", 0) for row in coverage)
                expected = nonhistory + len(history.items) + len(writer.items)
                if discovered != expected:
                    raise JobFailed(
                        "discovery",
                        "final discovered count differs from generated pool",
                        discovered=discovered,
                        expected=expected,
                    )
                profile.result.step(
                    "verify",
                    discovered=discovered,
                    expected=expected,
                    history_files=len(history.items),
                    arrivals=len(hits.seen),
                )
                stage_errors = {
                    name: value.get("errors")
                    for name, value in stages.items()
                    if name in ("raw_select", "raw_prepare") and value.get("errors")
                }
                if stage_errors:
                    raise JobFailed(
                        "scheduling_errors",
                        "scheduling/preparation recorded errors",
                        stages=stage_errors,
                    )
                log = turns["log"]
                assert isinstance(log, dict)
                turn_stages = turns["stages"]
                assert isinstance(turn_stages, dict)
                writer_max = max(
                    float(log["op_seconds_max"]),
                    *(float(turn_stages[stage]["seconds_max_lifetime"]) for stage in TURN_STAGES),
                )
                call_seconds = [
                    call.seconds for call in driver.calls if call.argv[:2] == ("daemon", "status")
                ] + profile.status_seconds
                measurements = {
                    "freshness_seconds": freshness,
                    "writer_seconds_max": writer_max,
                    "checkpoint_seconds_max": log["checkpoint_seconds_max"],
                    "call_seconds": seconds_summary(call_seconds),
                    "rss_bytes_max": rss_bytes_max(driver.samples),
                    "scheduling_instrumented": args.scheduling_trace,
                    "append_freshness_seconds": seconds_summary(append_hits.seconds()),
                }
                profile.result.step(
                    "final",
                    stats=final_stats,
                    floors=measurements,
                    appended_bytes=appender.bytes_total if appender else 0,
                    appended_lines=appender.lines_total if appender else 0,
                    probes=probes,
                    burst=burst,
                    generation_errors=turns["generation_errors"],
                )
                if (
                    probes["missing"]
                    or burst["missing"]
                    or turns["generation_errors"]
                    or (
                        freshness.get("count")
                        and (
                            float(freshness["p95"]) > floors.FRESHNESS_SECONDS_P95_MAX
                            or float(freshness["max"]) > floors.FRESHNESS_SECONDS_MAX
                        )
                    )
                    or (
                        append_hits.seconds()
                        and (
                            max(append_hits.seconds()) > floors.FRESHNESS_SECONDS_MAX
                            or floors.percentile(append_hits.seconds(), 0.95)
                            > floors.FRESHNESS_SECONDS_P95_MAX
                        )
                    )
                    or writer_max > floors.WRITER_SECONDS_MAX
                    or max(call_seconds) > floors.STATUS_SECONDS_MAX
                    or rss_bytes_max(driver.samples) > floors.RSS_BYTES_MAX
                ):
                    raise JobFailed(
                        "floor", "pressure profile breached a floor", floors=measurements
                    )
            finally:
                driver.close()
        profile.hold_if_asked(daemon)
    finally:
        os.close(descriptor)
        if appender is not None:
            appender.close()


def historical_commits(status: Mapping[str, object], paths: set[str], since: float) -> set[str]:
    """Only actual successful watch events in this window prove historical progress."""
    events = status.get("watch_recent_events")
    if not isinstance(events, list):
        return set()
    return {
        event["path"]
        for event in events
        if isinstance(event, dict)
        and event.get("path") in paths
        and event.get("success") is True
        and isinstance(event.get("timestamp"), int | float)
        and event["timestamp"] >= since
    }


def drained(status: Mapping[str, object]) -> bool:
    reconciliation = status.get("reconciliation")
    if not isinstance(reconciliation, dict):
        return False
    return (
        reconciliation.get("pending") == 0
        and reconciliation.get("noticed_pending") == 0
        and reconciliation.get("catalog_scan_complete") is True
        and reconciliation.get("enrichment_ready") is True
        and status.get("catchup_in_progress") is False
    )
