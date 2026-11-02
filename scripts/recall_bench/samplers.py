"""CPU, memory and I/O of one owned process, sampled with psutil on every platform."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class Sample:
    at: float
    cpu_seconds: float
    rss_bytes: int
    threads: int
    # Linux only: physical bytes and logical bytes (chars); None where psutil has none.
    io: dict[str, int] | None

    def to_json(self) -> dict[str, object]:
        return {
            "at": self.at,
            "cpu_seconds": self.cpu_seconds,
            "rss_bytes": self.rss_bytes,
            "threads": self.threads,
            "io": self.io if self.io is not None else "unavailable",
        }


def sample(process: psutil.Process) -> Sample:
    with process.oneshot():
        times = process.cpu_times()
        rss = process.memory_info().rss
        threads = process.num_threads()
        io = None
        if sys.platform == "linux":
            counters = process.io_counters()
            io = {
                "read_bytes": counters.read_bytes,
                "write_bytes": counters.write_bytes,
                "read_chars": counters.read_chars,
                "write_chars": counters.write_chars,
            }
    return Sample(time.monotonic(), times.user + times.system, rss, threads, io)


def summarize(samples: list[Sample]) -> dict[str, object]:
    """Window totals; I/O deltas are reported as unavailable, never as zero, off Linux."""
    assert len(samples) >= 2, "a window needs a first and a last sample"
    first, last = samples[0], samples[-1]
    seconds = last.at - first.at
    cpu = last.cpu_seconds - first.cpu_seconds
    io: dict[str, int] | str = "unavailable"
    if first.io is not None and last.io is not None:
        io = {name: last.io[name] - first.io[name] for name in first.io}
    return {
        "samples": len(samples),
        "seconds": seconds,
        "cpu_seconds": cpu,
        "cpu_percent": 100 * cpu / seconds if seconds > 0 else None,
        "rss_bytes_max": max(item.rss_bytes for item in samples),
        "rss_bytes_mean": sum(item.rss_bytes for item in samples) / len(samples),
        "threads_max": max(item.threads for item in samples),
        "io": io,
    }


def host_load() -> dict[str, float]:
    one, five, fifteen = os.getloadavg()
    return {"load1": one, "load5": five, "load15": fifteen, "cpus": float(os.cpu_count() or 0)}
