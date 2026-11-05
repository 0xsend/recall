"""`idle-observe`: opt-in, read-only observation of an already-running daemon.

Never starts, stops or probes a service manager. The caller explicitly names
an installed daemon PID and data directory; output still obeys bench write roots.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import psutil

from .daemon import RSS_BYTES_MAX, STATUS_SECONDS_MAX, _count, _positive
from .evidence import JobFailed, JobIncomplete, Result, tool_identity
from .fs import Cancellation, Directory
from .idle import IdleRun, Observer
from .isolation import Policy
from .samplers import host_load
from .turns import read_samples


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--installed",
        action="store_true",
        required=True,
        help="explicitly authorize read-only installed-daemon observation",
    )
    parser.add_argument("--data-dir", required=True, help="explicit existing daemon data directory")
    parser.add_argument(
        "--config", required=True, help="selected daemon config path (defaults if absent)"
    )
    parser.add_argument("--pid", required=True, type=_count, help="expected serving PID")
    parser.add_argument("--output", required=True, help="new directory under BENCH_WRITE_ROOTS")
    parser.add_argument("--window", type=_positive, default=600.0)
    parser.add_argument("--settle-timeout", type=_positive, default=900.0)
    parser.add_argument("--deadline", type=_positive, default=7200.0)
    parser.add_argument(
        "--idle-transition",
        action="store_true",
        help="require observed pending enrichment to finish before quiet entry",
    )


def selected_config(raw: str) -> str:
    config_path = Path(raw)
    if not config_path.is_absolute() or ".." in config_path.parts:
        raise JobFailed("usage", "--config must be an absolute path without ..")
    try:
        with config_path.open("rb") as config_file:
            data = config_file.read(1024 * 1024 + 1)
    except FileNotFoundError:
        data = b""
    if len(data) > 1024 * 1024:
        raise JobFailed("usage", "selected config exceeds 1 MiB")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise JobIncomplete("idle_config", "selected config is not UTF-8") from error


def run(
    args: argparse.Namespace,
    result: Result,
    cancellation: Cancellation,
    progress: Callable[[dict[str, object]], None],
) -> None:
    data = Path(args.data_dir)
    if not data.is_absolute() or ".." in data.parts:
        raise JobFailed("usage", "--data-dir must be an absolute path without ..")
    if args.window > 86400 or args.deadline > 86400:
        raise JobFailed("usage", "window and deadline are bounded to 86400 seconds")
    config_text = selected_config(args.config)
    try:
        process = psutil.Process(args.pid)
        birth = process.create_time()
    except psutil.Error as error:
        raise JobIncomplete(
            "idle_identity", f"cannot observe the requested PID: {error}"
        ) from error
    observer = Observer(data, args.pid, config_text)
    until = time.monotonic() + args.deadline
    cancellation.expire_at(until)
    # The only exception to scratch-only input policy: explicitly authorized,
    # read-only diagnostics. The directory capability has no mutation calls here.
    with (
        Directory.walk("--data-dir", data, lambda path, status: None) as root,
        Policy.for_host().open_write_roots() as roots,
    ):
        # Output must remain disjoint from the observed tree by filesystem
        # identity too (including case aliases on macOS).
        policy = roots.policy
        roots.policy = Policy.build(
            production=(*policy.production, data),
            homes=policy.homes,
            checkouts=policy.checkouts,
            mounts=policy.mounts,
        )
        with roots.new_directory("--output", args.output) as target, target.create() as tree:
            tree.commit(cancellation)
            result.scope = {
                "profile": "idle",
                "mode": "installed-observation",
                "pid": args.pid,
                "process_created_at": birth,
                "data_dir": str(data),
                "tool": tool_identity(),
                "host_load_start": host_load(),
                "missing": [
                    "installed lifecycle, memory/backlog/migration and retrieval qualification"
                ],
            }

            def check() -> None:
                cancellation.check()
                try:
                    unchanged = process.is_running() and process.create_time() == birth
                except psutil.Error as error:
                    raise JobIncomplete("idle_identity", str(error)) from error
                if not unchanged:
                    raise JobIncomplete(
                        "idle_identity", "observed daemon exited or changed identity"
                    )

            def wait(seconds: float) -> None:
                time.sleep(min(seconds, max(0.0, until - time.monotonic())))
                check()

            try:
                verdict = IdleRun(
                    observer=observer,
                    process=process,
                    diagnostics=lambda: read_samples(root, args.pid),
                    wait=wait,
                    check=check,
                    tree=tree,
                    phase=lambda name: progress({"event": "phase", "phase": name}),
                    result=result,
                    window_seconds=args.window,
                    settle_timeout=args.settle_timeout,
                ).run(transition=args.idle_transition)
                result.step(
                    "final",
                    floors={
                        "status_seconds_max": observer.seconds_max,
                        "rss_bytes_max": verdict["rss_bytes_max"],
                    },
                )
                if (
                    observer.seconds_max > STATUS_SECONDS_MAX
                    or int(str(verdict["rss_bytes_max"])) > RSS_BYTES_MAX
                ):
                    raise JobFailed("floor", "read latency or RSS floor exceeded")
                if verdict["verdict"] == "invalid":
                    raise JobIncomplete("idle_invalid", "idle window is invalid", verdict=verdict)
                if verdict["verdict"] == "failed":
                    raise JobFailed("floor", "idle CPU floor failed", verdict=verdict)
            finally:
                observer.close()
                result.scope["host_load_end"] = host_load()
                result.end(sys.exc_info()[1])
                tree.publish(PurePosixPath("result.json"), result.to_json())
