"""Optional bounded daemon-local profiling. Loaded by path in the guarded launcher.

No product function is replaced. Python profile return events retain the
scheduler's actual lane rosters and choices; C events split execute/fetchall
inside its pending query. Clocks are inclusive calling-thread CPU and wall
costs, never native worker CPU. Profiling adds overhead: use a separate run
without this observer for performance acceptance.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Literal

NAME = "bench-scheduling.jsonl"
STOP_NAME = "bench-scheduling-stop"
BYTES_MAX = 64 * 1024 * 1024
RECORDS_MAX = 200_000


@dataclass(frozen=True)
class Cost:
    event: Literal["cost"]
    stage: str
    at: float
    wall: float
    cpu: float
    lane: Literal["active", "recent", "oldest"] | None = None


@dataclass(frozen=True)
class Decision:
    event: Literal["decision"]
    at: float
    before: int
    after: int
    active: tuple[str, ...]
    recent: tuple[str, ...]
    oldest: tuple[str, ...]
    oldest_active: bool
    selected: tuple[tuple[str, str], ...]
    refused: tuple[str, ...]
    active_paths: int = 0


@dataclass(frozen=True)
class Stopped:
    event: Literal["stopped"]
    at: float


Trace = Cost | Decision | Stopped


def install(root: Path) -> None:
    """Observe only recognized build seams; absence fails before starting the daemon."""
    coordinator = importlib.import_module("recall.services.coordinator")
    catalog = importlib.import_module("recall.db.source_files").SourceCatalog
    scheduler = importlib.import_module("recall.services.reconciler").FairScheduler
    targets = {
        inspect.unwrap(coordinator.select_raw_sources).__code__: "selection",
        inspect.unwrap(coordinator.prepare_raw_sources).__code__: "preparation",
        catalog.priority_pending.__code__: "roster",
        catalog._pending_rows.__code__: "pending",
        scheduler.select.__code__: "scheduler",
    }
    descriptor = os.open(root / NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    lock = threading.Lock()
    local = threading.local()
    size = 0
    count = 0
    enabled = True

    def pending_lane(frame: FrameType) -> Literal["active", "recent", "oldest"] | None:
        query = frame.f_back
        roster = query.f_back if query is not None else None
        if roster is None or roster.f_code is not catalog.priority_pending.__code__:
            return None
        if "active" not in roster.f_locals:
            return "active"
        return "recent" if "recent_rows" not in roster.f_locals else "oldest"

    def record(item: Trace) -> None:
        nonlocal size, count
        data = (json.dumps(asdict(item), separators=(",", ":")) + "\n").encode()
        with lock:
            if not enabled:
                return
            size += len(data)
            count += 1
            if size > BYTES_MAX or count > RECORDS_MAX:
                raise RuntimeError("scheduling trace exhausted its bound")
            if os.write(descriptor, data) != len(data):
                raise OSError("short scheduling trace write")

    def profile(frame: FrameType, event: str, arg: Any) -> None:
        nonlocal enabled
        stage = targets.get(frame.f_code)
        if stage is None or not enabled:
            return
        if stage == "scheduler" and event == "call" and (root / STOP_NAME).exists():
            record(Stopped("stopped", time.monotonic()))
            with lock:
                enabled = False
                for code in targets:
                    monitor.set_local_events(tool, code, 0)
                monitor.set_events(tool, 0)
                os.close(descriptor)
            return
        if not hasattr(local, "starts"):
            local.starts = {}
        starts = local.starts
        key = (id(frame), "python")
        if event == "call":
            before = frame.f_locals["self"]._sequence if stage == "scheduler" else 0
            starts[key] = (time.monotonic(), time.thread_time(), before)
            starts[(id(frame), "boundary")] = starts[key]
        elif event == "return":
            began, cpu, before = starts.pop(key)
            starts.pop((id(frame), "boundary"), None)
            ended, cpu_end = time.monotonic(), time.thread_time()
            record(Cost("cost", stage, began, ended - began, cpu_end - cpu, pending_lane(frame)))
            if stage == "scheduler" and arg is not None:
                values = frame.f_locals
                active, recent, oldest = (values[name] for name in ("active", "recent", "oldest"))
                record(
                    Decision(
                        "decision",
                        ended,
                        before,
                        values["self"]._sequence,
                        tuple(item.key for item in active),
                        tuple(item.key for item in recent),
                        tuple(item.key for item in oldest),
                        bool(oldest and oldest[0].is_active),
                        tuple(
                            (item.source.source + "\x1f" + item.source.source_path, item.lane)
                            for item in arg
                        ),
                        tuple(values["refused"]),
                        len(values["active_paths"]),
                    )
                )
        elif stage == "pending" and event in ("c_call", "c_return", "c_exception"):
            name = getattr(arg, "__name__", "")
            if name not in ("time", "execute", "fetchall"):
                return
            key = (id(frame), name)
            if event == "c_call":
                starts[key] = (time.monotonic(), time.thread_time(), 0)
            else:
                ended, cpu_end = time.monotonic(), time.thread_time()
                # pybind methods on Python 3.12 can emit only c_return. Their
                # preceding boundary is the argument clock or execute return.
                began, cpu, _ = starts.pop(key, starts[(id(frame), "boundary")])
                label = {
                    "time": "pending_setup",
                    "execute": "query_binding",
                    "fetchall": "materialization",
                }[name]
                record(
                    Cost("cost", label, began, ended - began, cpu_end - cpu, pending_lane(frame))
                )
                starts[(id(frame), "boundary")] = (ended, cpu_end, 0)

    # Local monitoring observes only these code objects, including executor
    # threads. A global profile hook would perturb every parser and writer call.
    monitor = sys.monitoring
    tool = monitor.PROFILER_ID
    monitor.use_tool_id(tool, "recall-bench")

    def started(_code, _offset) -> None:
        profile(sys._getframe(1), "call", None)

    def returned(_code, _offset, value) -> None:
        profile(sys._getframe(1), "return", value)

    def called(_code, _offset, target, _arg) -> None:
        profile(sys._getframe(1), "c_call", target)

    def c_returned(_code, _offset, target, _arg) -> None:
        profile(sys._getframe(1), "c_return", target)

    monitor.register_callback(tool, monitor.events.PY_START, started)
    monitor.register_callback(tool, monitor.events.PY_RETURN, returned)

    def unwound(_code, _offset, _error) -> None:
        profile(sys._getframe(1), "return", None)

    monitor.register_callback(tool, monitor.events.PY_UNWIND, unwound)
    monitor.set_events(tool, monitor.events.PY_UNWIND)
    monitor.register_callback(tool, monitor.events.CALL, called)
    monitor.register_callback(tool, monitor.events.C_RETURN, c_returned)
    monitor.register_callback(tool, monitor.events.C_RAISE, c_returned)
    for code, stage in targets.items():
        events = monitor.events.PY_START | monitor.events.PY_RETURN
        if stage == "pending":
            events |= monitor.events.CALL
        monitor.set_local_events(tool, code, events)


def parse(line: Mapping[str, Any]) -> Trace:
    """Parse saved records at the evidence boundary, refusing unknown/truncated data."""
    if line.get("event") == "cost":
        item = Cost(**line)
        if item.stage not in {
            "selection",
            "preparation",
            "roster",
            "pending",
            "scheduler",
            "query",
            "query_binding",
            "pending_setup",
            "materialization",
        }:
            raise ValueError("unknown trace stage")
        if not all(
            type(value) in (int, float) and math.isfinite(value) and value >= 0
            for value in (item.at, item.wall, item.cpu)
        ):
            raise ValueError("invalid trace clock")
        if item.lane not in (None, "active", "recent", "oldest"):
            raise ValueError("invalid query lane")
        return item
    if line.get("event") == "stopped":
        stopped = Stopped(**line)
        if type(stopped.at) not in (int, float) or not math.isfinite(stopped.at) or stopped.at < 0:
            raise ValueError("invalid trace stop clock")
        return stopped
    if line.get("event") != "decision":
        raise ValueError("unknown trace event")
    item = Decision(**line)
    if (
        type(item.before) is not int
        or type(item.after) is not int
        or not 0 <= item.before <= item.after
        or item.after - item.before > 1
        or len(item.selected) > 1
        or max(map(len, (item.active, item.recent, item.oldest))) > 85
        or type(item.active_paths) is not int
        or item.active_paths < 0
        or not math.isfinite(item.at)
        or item.at < 0
    ):
        raise ValueError("invalid scheduler trace")
    if not all(
        isinstance(key, str)
        for lane in (item.active, item.recent, item.oldest, item.refused)
        for key in lane
    ):
        raise ValueError("invalid roster identities")
    if any(
        len(row) != 2 or not all(isinstance(part, str) for part in row) for row in item.selected
    ):
        raise ValueError("invalid selected identity")
    return item


def summarize(records: Sequence[Trace], start: float, end: float) -> dict[str, object]:
    """Retain costs separately and check observed continuous eligibility obligations."""
    costs: dict[str, dict[str, float]] = {}
    decisions: list[Decision] = []
    cutoff = None
    for item in records:
        if isinstance(item, Stopped):
            cutoff = item.at
            continue
        if not start <= item.at <= end:
            continue
        if isinstance(item, Decision):
            decisions.append(item)
        else:
            name = item.stage if item.lane is None else f"{item.lane}.{item.stage}"
            entry = costs.setdefault(name, {"count": 0, "wall": 0, "cpu": 0, "wall_max": 0})
            entry["count"] += 1
            entry["wall"] += item.wall
            entry["cpu"] += item.cpu
            entry["wall_max"] = max(entry["wall_max"], item.wall)
    lanes = ("active", "active", "active", "active", "recent", "recent", "oldest")
    waiting: dict[str, tuple[int, int]] = {}
    slots = completed = censored = reservations = 0
    all_lanes = 0
    breaches: list[str] = []
    counts = {name: 0 for name in set(lanes)}
    for item in decisions:
        current = set(item.oldest)
        for key in list(waiting):
            if key not in current:
                del waiting[key]
                censored += 1
        for rank, key in enumerate(item.oldest, 1):
            waiting.setdefault(key, (slots, 7 * rank))
        if item.refused:
            censored += len(waiting)
            waiting.clear()
        if not item.selected:
            continue
        key, lane = item.selected[0]
        counts[lane] = counts.get(lane, 0) + 1
        if lane != lanes[item.before % 7]:
            breaches.append("selected lane differs from 4:2:1 obligation")
        if item.active and item.recent and item.oldest:
            all_lanes += 1
        if lane == "oldest" and item.oldest and not item.refused:
            reservations += 1
            if key != item.oldest[0]:
                breaches.append("oldest slot did not serve the oldest eligible key")
        slots += 1
        if key in waiting:
            began, bound = waiting.pop(key)
            completed += 1
            if slots - began > bound:
                breaches.append("observed source exceeded 7*N slots")
        if any(slots - began >= bound for began, bound in waiting.values()):
            breaches.append("pending observed source exhausted 7*N slots")
    return {
        "instrumented": True,
        "stopped_at": cutoff,
        "costs": costs,
        "decisions": len(decisions),
        "active_paths_max": max((item.active_paths for item in decisions), default=0),
        "service_slots": slots,
        "lane_counts": counts,
        "all_lanes_slots": all_lanes,
        "oldest_reservations": reservations,
        "bounded_services": completed,
        "censored_obligations": censored,
        "open_obligations": len(waiting),
        "breaches": breaches,
        "scope": "Detailed trace ends after the sampling window; open obligations are unqualified. "
        "Observed roster ranks only; disappeared/refused eligibility is censored. "
        "Full drain and exact keyword visibility separately check finite workload starvation. "
        "Inclusive clocks overlap; never add parent/child costs. Native worker CPU unavailable.",
    }
