"""Local, bounded daemon diagnostics independent of the database and writer."""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import psutil
from pydantic import BaseModel

from recall.core.telemetry import (
    DURATION_BUCKETS_SECONDS,
    Recorder,
    Snapshot,
    current,
)

logger = logging.getLogger("recall.diagnostics")
SAMPLE_SECONDS = 10.0
LOG_BYTES_MAX = 5 * 1024 * 1024
LOG_BACKUPS = 3


class Sample(BaseModel):
    schema_version: int = 1
    pid: int
    started_at: float
    timestamp: float
    monotonic: float
    version: str
    platform: str
    requested_backend: str
    configured_model: str
    duration_buckets_seconds: tuple[float, ...] = DURATION_BUCKETS_SECONDS
    sampling_cpu_seconds: float
    output_failures: int
    process_cpu_seconds: float
    rss_bytes: int
    threads: int
    io_read_bytes: int | None
    io_write_bytes: int | None
    telemetry: Snapshot


class Diagnostics:
    def __init__(self, path: Path, *, requested_backend: str, configured_model: str) -> None:
        self.path = path
        self.recorder = Recorder()
        self.process = psutil.Process()
        self.started_at = time.time()
        self.requested_backend = requested_backend
        self.configured_model = configured_model
        self.version = version("recall")
        self.platform = platform.system()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name="recall-diagnostics", daemon=True)
        self.failed = False
        self.sampling_cpu_seconds = 0.0
        self.output_failures = 0

    def sample(self) -> Sample:
        with self.process.oneshot():
            cpu = self.process.cpu_times()
            memory = self.process.memory_info()
            threads = self.process.num_threads()
            io = self.process.io_counters() if hasattr(self.process, "io_counters") else None
        return Sample(
            pid=os.getpid(),
            started_at=self.started_at,
            timestamp=time.time(),
            monotonic=time.monotonic(),
            version=self.version,
            platform=self.platform,
            requested_backend=self.requested_backend,
            configured_model=self.configured_model,
            sampling_cpu_seconds=self.sampling_cpu_seconds,
            output_failures=self.output_failures,
            process_cpu_seconds=cpu.user + cpu.system,
            rss_bytes=memory.rss,
            threads=threads,
            io_read_bytes=io.read_bytes if io else None,
            io_write_bytes=io.write_bytes if io else None,
            telemetry=self.recorder.snapshot(),
        )

    def write(self) -> None:
        started = time.thread_time()
        try:
            sample = self.sample()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size >= LOG_BYTES_MAX:
                for index in range(LOG_BACKUPS, 0, -1):
                    source = self.path if index == 1 else Path(f"{self.path}.{index - 1}")
                    if source.exists():
                        source.replace(f"{self.path}.{index}")
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                stream.write(sample.model_dump_json() + "\n")
            if self.failed:
                logger.info("daemon diagnostics recovered")
            self.failed = False
        except (OSError, psutil.Error) as error:
            if not self.failed:
                logger.warning("daemon diagnostics unavailable: %s", error)
            self.failed = True
            self.output_failures += 1
        finally:
            self.sampling_cpu_seconds += time.thread_time() - started

    def run(self) -> None:
        self.write()
        while not self.stop.wait(SAMPLE_SECONDS):
            self.write()
        self.write()


@contextmanager
def collect(path: Path, *, requested_backend: str, configured_model: str) -> Iterator[Diagnostics]:
    diagnostics = Diagnostics(
        path, requested_backend=requested_backend, configured_model=configured_model
    )
    token = current.set(diagnostics.recorder)
    diagnostics.thread.start()
    try:
        yield diagnostics
    finally:
        current.reset(token)
        diagnostics.stop.set()
        diagnostics.thread.join(timeout=2)
