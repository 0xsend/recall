"""Disposable populated-daemon checkpoint measurement; no production defaults.

Run from the repository through uv, under with_lock.sh. The last output line is
an outcome, after the owned daemon process group has been stopped and reaped.
Raw transcripts and populated databases are local evidence, never git inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import duckdb

REPO = Path(__file__).resolve().parents[2]
LANE = REPO.parent
SOURCES = ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
FIXTURE_SHA = "2849121c0f56043fed5eac2cd7ffac3f1e83b2d64b28b3bdcb0b8806241088b4"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
APPENDS = 1100
APPEND_LINES = 24027
WORKLOAD_SECONDS = 120
DEADLINE_SECONDS = 2400


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def io_sample(pid: int) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (
            line.split(": ") for line in Path(f"/proc/{pid}/io").read_text().splitlines()
        )
    }


def daemon(evidence: Path, variant: str) -> None:
    """Inject only observation and an empty usage-log path into the real CLI."""
    from variants import install

    install(variant)
    from recall import db
    from recall.db import connection
    from recall.services import usage_harvest

    empty_log = evidence / "empty-unified.jsonl"
    empty_log.touch()

    def empty_usage_path(*, home: Path | None = None) -> Path:
        return empty_log

    original_checkpoint = connection.checkpoint_wal_if_due
    if variant == "watermark-16":
        patch.object(connection, "WAL_CHECKPOINT_HIGH_WATER_BYTES", 16 * 1024 * 1024).start()

    def observed_checkpoint(
        conn: duckdb.DuckDBPyConnection,
        db_path: Path,
        *,
        high_water_bytes: int = connection.WAL_CHECKPOINT_HIGH_WATER_BYTES,
    ) -> connection.WalCheckpointResult:
        before = io_sample(os.getpid())
        started = time.monotonic()
        result = original_checkpoint(conn, db_path, high_water_bytes=high_water_bytes)
        after = io_sample(os.getpid())
        if result.attempted:
            with (evidence / "checkpoints.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            **asdict(result),
                            "start": started,
                            "end": time.monotonic(),
                            "io_before": before,
                            "io_after": after,
                            "physical_bytes": after["write_bytes"] - before["write_bytes"],
                            "logical_bytes": after["wchar"] - before["wchar"],
                            "load": os.getloadavg(),
                        }
                    )
                    + "\n"
                )
        return result

    connection.checkpoint_wal_if_due = observed_checkpoint
    db.checkpoint_wal_if_due = observed_checkpoint
    from recall.cli.app import app

    sys.argv = [str(REPO / ".venv/bin/recall"), "daemon", "--mode", "watch"]
    with patch.object(usage_harvest, "default_grok_unified_log", empty_usage_path):
        app()


@dataclass
class Transcript:
    path: Path
    source: str
    lines: list[bytes]
    position: int


def prepare_pool(work: Path) -> tuple[list[Transcript], list[dict[str, object]]]:
    rng = random.Random(93)
    sessions: list[Transcript] = []
    manifest: list[dict[str, object]] = []
    for repeat in range(4):
        for kind, source in (("claude", "claude_code"), ("codex", "codex")):
            for original in sorted((LANE / "scratch/pool" / kind).glob("*.jsonl")):
                raw = original.read_bytes()
                text = raw.decode("utf-8", errors="surrogateescape")
                name = original.name
                matches = UUID.findall(name)
                assert matches, original
                for old in matches:
                    replacement = str(uuid.UUID(int=rng.getrandbits(128), version=4))
                    text = text.replace(old, replacement)
                    name = name.replace(old, replacement)
                path = work / "roots" / source / f"copy-{repeat}" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                lines = text.encode("utf-8", errors="surrogateescape").splitlines(keepends=True)
                position = max(1, int(len(lines) * 0.7))
                path.write_bytes(b"".join(lines[:position]))
                sessions.append(Transcript(path, source, lines, position))
                manifest.append(
                    {
                        "original": str(original.relative_to(LANE)),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "copy": repeat,
                        "seed_lines": position,
                        "seed_bytes": path.stat().st_size,
                        "lines": len(lines),
                    }
                )
    assert len(sessions) == 96, len(sessions)
    return sessions, manifest


def run(label: str, rehearsal: str | None, variant: str) -> int:
    evidence = LANE / "scratch/evidence" / label
    evidence.mkdir(parents=True, exist_ok=False)
    work = Path(os.environ["TMPDIR"]) / label
    work.mkdir(parents=True, exist_ok=False)
    data = work / "data"
    data.mkdir()
    assert len(str(data / "recall.sock")) < 104
    process: subprocess.Popen[bytes] | None = None
    result: dict[str, Any] = {
        "outcome": "incomplete",
        "label": label,
        "variant": variant,
        "measurement_only": True,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "steps": {},
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "fixture_sha256": FIXTURE_SHA,
        "load_start": os.getloadavg(),
    }
    steps = result["steps"]
    samples: list[dict[str, Any]] = []
    status: dict[str, Any] = {}
    started = time.monotonic()

    def interrupt(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def record_sample() -> None:
        assert process is not None
        if process.poll() is not None:
            raise RuntimeError(f"owned daemon exited {process.returncode}")
        samples.append(
            {
                "monotonic": time.monotonic(),
                "load": os.getloadavg(),
                "io": io_sample(process.pid),
                "wal_bytes": (data / "recall.duckdb.wal").stat().st_size
                if (data / "recall.duckdb.wal").exists()
                else 0,
            }
        )

    try:
        if rehearsal:
            process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
            )
            write_json(evidence / "owned.json", {"pid": process.pid, "rehearsal": rehearsal})
            steps["prepare"] = 0
            if rehearsal == "early":
                raise RuntimeError("injected preparation failure")
            if rehearsal == "deadline":
                raise TimeoutError("injected deadline expiry")
            if rehearsal == "term":
                process.wait(timeout=90)
            result["outcome"] = "passed"
            return_code = 0
        else:
            from recall.core.config import AppConfig
            from recall.core.ids import session_id
            from recall.core.rpc_client import RpcClient
            from recall.core.rpc_types import RpcCallError, RpcConnectionError

            for name in ("recall.duckdb", "recall.fts.sqlite"):
                fixture = f"fixture-{variant}" if variant in ("u128", "drop-results") else "fixture"
                subprocess.run(
                    [
                        "cp",
                        "--reflink=auto",
                        str(LANE / "scratch" / fixture / name),
                        str(data / name),
                    ],
                    check=True,
                    timeout=120,
                )
            sessions, manifest = prepare_pool(work)
            write_json(evidence / "pool.json", manifest)
            config_path = work / "config.toml"
            roots = {source: work / "roots" / source for source in SOURCES}
            for root in roots.values():
                root.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                '[embedding]\nbackend = "onnx"\nmodel = "BAAI/bge-small-en-v1.5"\n'
                'batch_size = 64\n[embedding.context]\nmode = "template"\nfallback = "template"\n'
                "[compaction]\nauto_trigger = false\n"
                "[daemon]\nembed_idle_session = 30\nembed_interval = 10\n"
                + "\n".join(
                    f'[sources.{source}]\nroots = ["{root}"]' for source, root in roots.items()
                )
                + "\n"
            )
            # Preserve HOME/model residency, remove ambient recall overrides.
            env = {key: value for key, value in os.environ.items() if not key.startswith("RECALL_")}
            env.update(
                PATH=f"{REPO / '.venv/bin'}:{env['PATH']}",
                RECALL_DATA_DIR=str(data),
                RECALL_CONFIG_PATH=str(config_path),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
            )
            os.environ.update(
                {key: value for key, value in env.items() if key.startswith("RECALL_")}
            )
            config = AppConfig.load()
            assert config.data_dir == data
            assert all(config.sources[s].roots == (roots[s],) for s in SOURCES)
            steps["prepare"] = 0
            with (evidence / "daemon.log").open("wb") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--daemon",
                        str(evidence),
                        "--variant",
                        variant,
                    ],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            write_json(evidence / "owned.json", {"pid": process.pid, "data": str(data)})

            def call(method: str, params: dict[str, Any] | None = None) -> Any:
                client = RpcClient(config=config)
                client.connect(auto_fork=False)
                try:
                    return client.call(method, params or {}, idle_timeout=30)
                finally:
                    client.close()

            def wait_state(phase: str, deadline: float, *, ready_only: bool = False) -> None:
                nonlocal status
                began = time.monotonic()
                while time.monotonic() < deadline:
                    record_sample()
                    try:
                        status = call("recall.daemon_status", {"limit": 1})
                    except (OSError, RpcCallError, RpcConnectionError) as error:
                        result["last_readiness_error"] = repr(error)
                        time.sleep(1)
                        continue
                    assert process is not None
                    assert status["daemon_pid"] == process.pid
                    assert int((data / "recall.pid").read_text().split()[0]) == process.pid
                    recon = status["reconciliation"]
                    write_json(evidence / "latest-status.json", status)
                    if recon.get("enrichment_error"):
                        raise RuntimeError(recon["enrichment_error"])
                    current = 0
                    if ready_only:
                        satisfied = True
                    elif (
                        recon["catalog_scan_complete"]
                        and recon["pending"] == 0
                        and recon["enrichment_ready"]
                    ):
                        for session in sessions:
                            shown = call(
                                "recall.show",
                                {
                                    "session_id": session_id(session.source, str(session.path)),
                                    "tools": False,
                                    "tail": 1,
                                },
                            )
                            freshness = shown.get("freshness") or {}
                            current += (
                                bool(freshness.get("current"))
                                and freshness.get("file_size") == session.path.stat().st_size
                            )
                        satisfied = current == len(sessions)
                    else:
                        satisfied = False
                    if satisfied:
                        steps[phase] = 0
                        result[f"{phase}_seconds"] = time.monotonic() - began
                        write_json(evidence / f"{phase}-status.json", status)
                        return
                    time.sleep(2)
                raise TimeoutError(f"{phase}: required observable state not reached")

            wait_state("ready", started + 120, ready_only=True)
            print(f"{label}: ready pid={process.pid}", file=sys.stderr, flush=True)
            wait_state("settle", started + DEADLINE_SECONDS)
            print(f"{label}: settled", file=sys.stderr, flush=True)
            # Fixed append count/line count, rather than scheduler-speed-dependent work.
            rng = random.Random(59)
            chunks = [rng.randint(5, 40) for _ in range(APPENDS)]
            difference = APPEND_LINES - sum(chunks)
            for index in range(abs(difference)):
                chunks[index % APPENDS] += 1 if difference > 0 else -1
            assert sum(chunks) == APPEND_LINES and min(chunks) > 0
            appended_bytes = 0
            workload_start = time.monotonic()
            selected = 0
            for index, count in enumerate(chunks):
                for _ in sessions:
                    session = sessions[selected % len(sessions)]
                    selected += 1
                    if len(session.lines) - session.position >= count:
                        break
                else:
                    raise RuntimeError("transcript pool cannot supply the declared append schedule")
                chunk = session.lines[session.position : session.position + count]
                assert len(chunk) == count, (index, session.path)
                body = b"".join(chunk)
                with session.path.open("ab") as stream:
                    stream.write(body)
                session.position += count
                appended_bytes += len(body)
                if index % 10 == 0:
                    record_sample()
                delay = workload_start + (index + 1) * WORKLOAD_SECONDS / APPENDS - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            steps["append"] = 0
            result["appended_bytes"] = appended_bytes
            result["appended_lines"] = sum(chunks)
            result["appends"] = len(chunks)
            wait_state("drain", time.monotonic() + DEADLINE_SECONDS)
            final_stats = call("recall.stats")
            write_json(evidence / "final-stats.json", final_stats)
            reference = LANE / "scratch/evidence/baseline-1/final-stats.json"
            if reference.exists():
                baseline_stats = json.loads(reference.read_text())
                for key in ("sessions", "messages", "tool_calls", "bash_calls"):
                    assert final_stats[key] == baseline_stats[key], (key, final_stats[key])
            result["final_counts"] = {
                key: final_stats[key]
                for key in ("sessions", "messages", "tool_calls", "bash_calls")
            }
            steps["final_stats"] = 0
            checkpoints = [
                json.loads(line)
                for line in (evidence / "checkpoints.jsonl").read_text().splitlines()
            ]
            assert checkpoints and all(not row["contended"] for row in checkpoints)
            diagnostics = [
                json.loads(line)
                for path in data.glob("diagnostics.jsonl*")
                for line in path.read_text().splitlines()
            ]
            errors = max(row["telemetry"]["counters"]["generation_errors"] for row in diagnostics)
            assert errors == 0, errors
            result["checkpoint_count"] = len(checkpoints)
            result["checkpoint_seconds_max"] = max(row["duration_seconds"] for row in checkpoints)
            result["five_second_floor_passed"] = result["checkpoint_seconds_max"] <= 5
            result["generation_errors"] = errors
            result["outcome"] = "passed"
            return_code = 0
    except BaseException as error:
        result["error"] = "".join(traceback.format_exception(error))
        result["outcome"] = (
            "incomplete" if isinstance(error, TimeoutError | SystemExit) else "failed"
        )
        return_code = (
            int(error.code or 1)
            if isinstance(error, SystemExit)
            else 124
            if isinstance(error, TimeoutError)
            else 1
        )
    finally:
        if process is not None:
            if process.poll() is None:
                assert os.getpgid(process.pid) == process.pid
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
                result["outcome"] = "incomplete"
                result["cleanup_error"] = "owned daemon required SIGKILL"
                return_code = 1
            steps["child_exit"] = process.returncode
            steps["cleanup"] = 0
            if not rehearsal and process.returncode != 0:
                result["outcome"] = "failed"
                result["cleanup_error"] = f"daemon exited {process.returncode}"
                return_code = 1
        result["load_end"] = os.getloadavg()
        result["seconds"] = time.monotonic() - started
        write_json(evidence / "samples.json", samples)
        if not rehearsal:
            import shutil

            for path in data.glob("diagnostics.jsonl*"):
                shutil.copy2(path, evidence / path.name)
        write_json(evidence / "result.json", result)
        print(json.dumps(result, default=str), flush=True)
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", type=Path)
    parser.add_argument("--run")
    parser.add_argument(
        "--variant",
        choices=("baseline", "u128", "drop-results", "watermark-16"),
        default="baseline",
    )
    parser.add_argument("--rehearsal", choices=("early", "deadline", "term", "normal"))
    args = parser.parse_args()
    if args.daemon:
        daemon(args.daemon, args.variant)
        return 0
    assert args.run and re.fullmatch(r"[a-z0-9-]{1,32}", args.run)
    return run(args.run, args.rehearsal, args.variant)


if __name__ == "__main__":
    raise SystemExit(main())
