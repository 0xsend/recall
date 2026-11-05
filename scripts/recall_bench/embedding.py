"""Real-model workload and evidence for the maintained daemon benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .cli import Command
from .embedding_trace import TRACE_BYTES_MAX, TRACE_NAME, Event, text_hash
from .evidence import JobFailed
from .isolation import Policy
from .load import Call, once, repeat, seconds_summary
from .samplers import Sample, summarize
from .visibility import Visibility, prepare
from .workload import APPEND_FLAGS, Appender, Transcript, Write

if TYPE_CHECKING:
    from .daemon import Profile

ROWS_MIN = 36
ROWS_MAX = 512
RAW_WRITES = 20
REFERENCE_BYTES_MAX = 16 * 1024**2
VECTOR_ABS_TOLERANCE = 0.002
MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class Inputs:
    kind: str
    texts: tuple[str, ...]


def schedule(rows: int) -> tuple[Inputs, ...]:
    """Three distinct sessions; unique prefixes survive truncation and caching."""
    if not ROWS_MIN <= rows <= ROWS_MAX:
        raise ValueError(f"embedding rows must be within {ROWS_MIN}..{ROWS_MAX}")
    widths = (4, 32, 128, 600)
    return tuple(
        Inputs(
            kind,
            tuple(
                f"recallbench {kind} input {index} "
                + "database " * (widths[index % len(widths)] if kind == "mixed" else 600)
                for index in range(rows if kind == "long-batch" else 36)
            ),
        )
        for kind in ("mixed", "maximum", "long-batch")
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--duckdb-memory-limit", choices=("2GB", "4GB"), default="2GB")
    group = parser.add_argument_group("real embedding profile")
    group.add_argument(
        "--model-cache", help="existing scratch HF_HOME, pre-populated; children stay offline"
    )
    group.add_argument(
        "--embed-rows",
        type=int,
        default=256,
        help="long-batch rows and backend batch size (36..512)",
    )
    group.add_argument(
        "--embed-model-timeout",
        type=int,
        default=10,
        help="natural model idle-unload seconds (1..600)",
    )
    reference = group.add_mutually_exclusive_group()
    reference.add_argument(
        "--embed-reference", help="stored reference.json to compare (absolute tolerance 0.002)"
    )
    reference.add_argument(
        "--embed-record-reference",
        action="store_true",
        help="record an explicit baseline; pairs compare candidate with that arm",
    )


def validate_options(args: argparse.Namespace) -> None:
    if args.profile != "embed":
        if (
            args.model_cache
            or args.embed_reference
            or args.embed_record_reference
            or args.embed_rows != 256
            or args.embed_model_timeout != 10
        ):
            raise JobFailed("usage", "embedding options require --profile embed")
        return
    if not args.model_cache or not (args.embed_reference or args.embed_record_reference):
        raise JobFailed(
            "usage",
            "embed requires --model-cache and --embed-reference or --embed-record-reference",
        )
    if not ROWS_MIN <= args.embed_rows <= ROWS_MAX or not 1 <= args.embed_model_timeout <= 600:
        raise JobFailed("usage", "embedding rows must be 36..512 and model timeout 1..600")
    from .daemon import WORKLOAD_OPTIONS

    if any(getattr(args, name) is not None for name in WORKLOAD_OPTIONS):
        raise JobFailed("usage", "embed supplies its own deterministic raw append schedule")


Vector = Annotated[
    list[Annotated[float, Field(allow_inf_nan=False)]], Field(min_length=384, max_length=384)
]


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    format: Literal["recall-bench-vectors/1"] = "recall-bench-vectors/1"
    model: Literal["BAAI/bge-small-en-v1.5"] = MODEL
    tolerance: Annotated[float, Field(ge=0.002, le=0.002)] = VECTOR_ABS_TOLERANCE
    hashes: list[str] = Field(min_length=1, max_length=1100)
    vectors: list[Vector] = Field(min_length=1, max_length=1100)
    retrieval: dict[str, list[str]]
    backend: Literal["onnx", "mlx"] = "onnx"
    artifacts: dict[str, str] = Field(default_factory=dict)
    build: dict[str, object] = Field(default_factory=dict)


def equivalence(reference: Reference, candidate: Reference) -> dict[str, object]:
    if reference.backend != candidate.backend or reference.artifacts != candidate.artifacts:
        raise JobFailed("reference", "backend/model artifacts differ from the stored reference")
    if (
        reference.hashes != candidate.hashes
        or len(reference.vectors) != len(reference.hashes)
        or len(candidate.vectors) != len(candidate.hashes)
    ):
        raise JobFailed("vectors", "reference inputs/vector counts differ")
    largest = 0.0
    validate_vectors(reference)
    validate_vectors(candidate)
    for before, after in zip(reference.vectors, candidate.vectors, strict=True):
        largest = max(largest, max(abs(a - b) for a, b in zip(before, after, strict=True)))
    if largest > VECTOR_ABS_TOLERANCE:
        raise JobFailed(
            "vectors",
            "vector equivalence exceeded the declared tolerance",
            absolute_error_max=largest,
            tolerance=VECTOR_ABS_TOLERANCE,
        )
    if reference.retrieval != candidate.retrieval:
        raise JobFailed("retrieval", "final vector/hybrid retrieval differs from the reference")
    return {
        "vectors": len(candidate.vectors),
        "absolute_error_max": largest,
        "tolerance": VECTOR_ABS_TOLERANCE,
        "retrieval_equal": True,
    }


def validate_vectors(reference: Reference) -> None:
    if len(reference.vectors) != len(reference.hashes):
        raise JobFailed("vectors", "input/vector counts differ")
    if any(abs(sum(value * value for value in vector) - 1) > 0.002 for vector in reference.vectors):
        raise JobFailed("vectors", "a stored vector is not normalized")


def model_artifacts(profile: Profile) -> dict[str, str]:
    """Resolve the offline cache revision and hash exactly the model inputs."""
    cache = profile.run_dir.model_cache
    assert cache is not None
    repository = cache / "hub/models--BAAI--bge-small-en-v1.5"
    policy = Policy.for_host()
    try:
        with policy.open_file("--model-cache revision", repository / "refs/main") as file:
            revision = file.head(41).decode().strip()
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise JobFailed("model_cache", "cached main ref is not a model revision")
        snapshot = repository / "snapshots" / revision
        weights = "model.safetensors" if sys.platform == "darwin" else "onnx/model.onnx"
        artifacts = {"revision": revision}
        for name in ("config.json", "tokenizer.json", weights):
            target = (snapshot / name).resolve()
            if not target.is_relative_to(cache):
                raise JobFailed("model_cache", "model artifact resolves outside scratch cache")
            with policy.open_file("--model-cache artifact", target) as file:
                artifacts[name] = file.sha256(profile.supervisor._cancellation)
        return artifacts
    except (OSError, UnicodeError) as error:
        raise JobFailed("model_cache", f"pre-populate the scratch model cache: {error}") from error


def logged_errors(data: bytes) -> list[str]:
    """Recovered engine errors still disqualify a run."""
    return [
        line[:2000]
        for line in data.decode(errors="replace").splitlines()
        if re.search(r"\b(ERROR|CRITICAL)\b|out.?of.?memory", line, re.IGNORECASE)
    ]


def _line(text: str, index: int) -> bytes:
    return (
        json.dumps(
            {
                "type": "message",
                "timestamp": "2026-11-01T00:00:00Z",
                "message": {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": [{"type": "text", "text": text}],
                },
            }
        )
        + "\n"
    ).encode()


def seed(profile: Profile, inputs: Sequence[Inputs]) -> Appender:
    raw = tuple(
        _line(f"recallbench raw freshness witness {index} unique word{index}", index)
        for index in range(RAW_WRITES + 1)
    )
    relative = PurePosixPath("pool/claude_code/recallbench-raw.jsonl")
    session = Transcript("claude_code", relative, raw, 1, len(raw[0]), len(raw[0]))
    appender = Appender(
        [session], [Write(i, 0, i, 0, i + 1, 1, len(raw[i + 1])) for i in range(RAW_WRITES)]
    )
    try:
        with profile.run_dir.held() as root, root.descend("embed", relative.parent) as directory:
            for group in inputs:
                data = b"".join(_line(text, i) for i, text in enumerate(group.texts))
                fd = os.open(
                    f"recallbench-{group.kind}.jsonl",
                    APPEND_FLAGS,
                    0o600,
                    dir_fd=directory.descriptor,
                )
                try:
                    if os.write(fd, data) != len(data):
                        raise OSError("short embedding transcript write")
                finally:
                    os.close(fd)
            session.descriptor = os.open(
                relative.name, APPEND_FLAGS, 0o600, dir_fd=directory.descriptor
            )
            if os.write(session.descriptor, raw[0]) != len(raw[0]):
                raise OSError("short raw seed write")
        return appender
    except BaseException:
        appender.close()
        raise


def trace(profile: Profile) -> list[Event]:
    try:
        with profile.run_dir.held() as root, root.open_file("embedding trace", TRACE_NAME) as file:
            if file.status.st_size > TRACE_BYTES_MAX:
                raise JobFailed("trace", "embedding trace exceeded 128 MiB")
            data = file.head(file.status.st_size)
        events = [Event.model_validate_json(line) for line in data.split(b"\n")[:-1] if line]
        return sorted(events, key=lambda event: event.at)
    except (ValueError, ValidationError) as error:
        raise JobFailed("trace", f"invalid embedding observation: {error}") from error


def inference_intervals(events: Sequence[Event]) -> list[tuple[float, float]]:
    opened: dict[tuple[int | None, int | None], float] = {}
    spans = []
    for event in events:
        key = (event.instance, event.thread)
        if event.kind == "inference-start":
            if key in opened:
                raise JobFailed("trace", "nested inference interval")
            opened[key] = event.at
        elif event.kind == "inference-end":
            if key not in opened:
                raise JobFailed("trace", "inference end lacks its start")
            spans.append((opened.pop(key), event.at))
    if opened:
        raise JobFailed("trace", "inference interval did not finish")
    return spans


def metrics(events: Sequence[Event]) -> dict[str, object]:
    errors = [event.error for event in events if event.kind == "error"]
    if errors:
        raise JobFailed("embedding_error", "real model or observer failed", errors=errors)
    phases: dict[str, list[dict[str, object]]] = {
        name: [] for name in ("load", "inference", "unload")
    }
    opened: dict[tuple[str, int | None, int | None], int] = {}
    for index, event in enumerate(events):
        if event.kind.endswith("-start"):
            name = event.kind.removesuffix("-start")
            key = (name, event.instance, event.thread)
            if key in opened:
                raise JobFailed("trace", "nested lifecycle interval")
            opened[key] = index
        elif event.kind.endswith("-end"):
            name = event.kind.removesuffix("-end")
            first = opened.pop((name, event.instance, event.thread), None)
            if first is None:
                raise JobFailed("trace", "lifecycle end lacks its start")
            window = events[first : index + 1]
            phases[name].append(
                {**window_metrics(window), "instance": event.instance, "role": event.role}
            )
    missing = [name for name, values in phases.items() if not values]
    if opened or missing:
        raise JobFailed("trace", "model lifecycle is incomplete", missing=missing)
    return {"phases": phases, "whole_process": window_metrics(events)}


def idle_metrics(events: Sequence[Event], started: float) -> dict[str, object]:
    unloads = [event.at for event in events if event.kind == "unload-start" and event.at >= started]
    if not unloads:
        raise JobFailed("trace", "no post-inference idle window preceded unload")
    return window_metrics([event for event in events if started <= event.at <= min(unloads)])


def window_metrics(events: Sequence[Event]) -> dict[str, object]:
    if len(events) < 2:
        raise JobFailed("trace", "a lifecycle window needs two observations")
    first, last = events[0], events[-1]
    footprint = [item.footprint_bytes for item in events if item.footprint_bytes is not None]
    samples = [Sample(item.at, item.cpu_seconds, item.rss_bytes, 0, None) for item in events]
    return {
        **summarize(samples),
        "rss_bytes_retained": last.rss_bytes,
        "rss_hwm_bytes": max(max(item.rss_hwm_bytes, item.rss_bytes) for item in events),
        "footprint_bytes_max": max(footprint) if footprint else None,
        "footprint_bytes_retained": last.footprint_bytes,
        "start": first.at,
        "end": last.at,
    }


def _pending(status: Mapping[str, object]) -> bool:
    reconciliation = status.get("reconciliation")
    if not isinstance(reconciliation, dict):
        raise JobFailed("drain", "status has no reconciliation state")
    errors = [
        status.get("embed_last_error"),
        reconciliation.get("error"),
        reconciliation.get("enrichment_error"),
        reconciliation.get("keyword_repair_error"),
    ]
    if any(errors) or status.get("watch_total_failed"):
        raise JobFailed("embedding_error", "daemon reported an embedding/raw error", errors=errors)
    return not (
        reconciliation.get("pending") == 0
        and reconciliation.get("noticed_pending") == 0
        and reconciliation.get("enrichment_ready") is True
        and status.get("embed_pending_eligible") == 0
        and status.get("embed_pending_held") == 0
        and status.get("catchup_in_progress") is False
    )


class RetrievedRecord(BaseModel):
    """Logical search result, independent of a fresh clone's absolute path IDs."""

    model_config = ConfigDict(frozen=True, strict=True)
    source: str
    source_path: str
    kind: str
    role: str | None = None
    content: str | None = None
    thinking: str | None = None
    timestamp: str | None = None
    tool_name: str | None = None
    bash_command: str | None = None


def _retrieval(document: Mapping[str, object], run_dir: Path) -> list[str]:
    rows = document.get("results")
    if not isinstance(rows, list) or not rows:
        raise JobFailed("retrieval", "semantic search returned no results")
    identifiers = []
    for row in rows:
        try:
            record = RetrievedRecord.model_validate(row)
        except ValidationError as error:
            raise JobFailed(
                "retrieval", f"semantic result has no logical identity: {error}"
            ) from error
        # Pure lexical normalization: never resolve or probe a stored source path.
        path = PurePosixPath(record.source_path)
        try:
            relative = path.relative_to(run_dir)
        except ValueError:
            canonical = record.source_path
        else:
            canonical = f"run/{relative}"
        record = record.model_copy(update={"source_path": canonical})
        identifiers.append(text_hash(record.model_dump_json()))
    return identifiers


def run(profile: Profile) -> None:
    inputs = schedule(profile.args.embed_rows)
    hashes = [text_hash(text) for group in inputs for text in group.texts]
    artifacts = model_artifacts(profile)
    profile.result.scope["embedding"] = {
        "model": MODEL,
        "backend": "mlx" if sys.platform == "darwin" else "onnx",
        "artifacts": artifacts,
        "rows": profile.args.embed_rows,
        "duckdb_memory_limit": profile.args.duckdb_memory_limit,
        "tolerance": VECTOR_ABS_TOLERANCE,
        "model_timeout": profile.args.embed_model_timeout,
    }
    prior = None
    if profile.args.embed_reference:
        with Policy.for_host().open_file("--embed-reference", profile.args.embed_reference) as file:
            if file.status.st_size > REFERENCE_BYTES_MAX:
                raise JobFailed("reference", "stored reference exceeds 16 MiB")
            try:
                prior = Reference.model_validate_json(file.head(file.status.st_size))
            except ValidationError as error:
                raise JobFailed("reference", str(error)) from error
        if prior.hashes != hashes:
            raise JobFailed("reference", "stored reference is for a different workload")
    appender = seed(profile, inputs)
    try:
        witnesses = prepare(appender, profile.run_dir.path, profile.supervisor.check)
        daemon, _stats = profile.boot(wait_for_settle=False)
        driver = profile.driver(daemon)
        started = time.monotonic()
        status: dict[str, object] = {}
        evidence: dict[str, object] | None = None
        try:
            freshness = Visibility(driver.appended, witnesses, driver.record_visibility)

            def observe(call: Call) -> None:
                freshness.observe(call)
                if call.caller == "status":
                    if not call.succeeded:
                        raise JobFailed(
                            "cli", "status failed during embedding", call=call.to_json()
                        )
                    status.clear()
                    status.update(call.document or {})
                    _pending(status)

            callers = {
                "status": repeat(Command.daemon_status()),
                "live": repeat(Command.live_fresh()),
                "vector": repeat(Command.semantic("database performance", "vector")),
                "hybrid": repeat(Command.semantic("database performance", "hybrid")),
                **freshness.callers(),
            }
            last_trace_check = 0.0
            inference_seen = False

            def model_work_started() -> bool:
                nonlocal last_trace_check, inference_seen
                now = time.monotonic()
                if not inference_seen and now - last_trace_check >= 0.1:
                    last_trace_check = now
                    inference_seen = any(
                        item.kind == "inference-start"
                        and item.role == "enrichment"
                        and item.width == 512
                        for item in trace(profile)
                    )
                return inference_seen

            profile.phase("embedding")
            driver.drive(
                "embedding",
                callers,
                appender=appender,
                ticks=range(RAW_WRITES),
                interval_seconds=0.25,
                start_when=model_work_started,
                until=lambda _now: bool(status) and not _pending(status) and not freshness.unseen(),
                timeout_seconds=profile.args.settle_timeout,
                observe=observe,
            )
            profile.result.step(
                "committed",
                seconds=time.monotonic() - started,
                freshness_seconds=seconds_summary(freshness.seconds()),
            )
            final_status, complete = profile.settle(
                profile.cli_json(Command.daemon_status()), daemon.exited
            )
            if _pending(final_status):
                raise JobFailed("drain", "embedding work remains after settlement")
            profile.result.step(
                "settle", seconds=time.monotonic() - started, catalog_seconds=complete.seconds
            )
            profile.result.step(
                "drain",
                seconds=time.monotonic() - started,
                eligible=final_status.get("embed_pending_eligible"),
                held=final_status.get("embed_pending_held"),
            )
            # Stop semantic callers so no new query inference obscures the idle/unload window.
            profile.phase("embedding-idle")
            idle_started = time.monotonic()

            def unloaded(_now: float) -> bool:
                events = trace(profile)
                return any(
                    item.kind == "unload-end" and item.at >= idle_started for item in events
                ) or (
                    final_status.get("embed_model_loaded") is False
                    and any(item.kind == "unload-end" for item in events)
                )

            driver.drive(
                "embedding-idle",
                {"status": repeat(Command.daemon_status()), "live": repeat(Command.live_fresh())},
                until=unloaded,
                timeout_seconds=profile.args.embed_model_timeout + profile.args.settle_timeout,
                observe=observe,
            )
            events = trace(profile)
            loaded = [item for item in events if item.kind == "load-end"]
            expected_backend = "mlx" if sys.platform == "darwin" else "onnx"
            if any(
                item.backend != expected_backend
                or item.model != MODEL
                or not item.runtime
                or not item.device
                for item in loaded
            ):
                raise JobFailed(
                    "backend", "real-model observations lack backend/device/runtime identity"
                )
            if expected_backend == "onnx" and any(
                item.device != "CPUExecutionProvider" for item in loaded
            ):
                raise JobFailed("backend", "ONNX run did not use CPUExecutionProvider exclusively")
            measured = metrics(events)
            measured["post_inference_idle"] = idle_metrics(events, idle_started)
            overlap: dict[str, int] = {}
            evidence = {
                "metrics": measured,
                "overlap": overlap,
                "comparison": None,
                "verification_state": "observed",
                "reference_mode": "compare" if prior else "record",
                "workload_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
                "workload": [{"kind": group.kind, "rows": len(group.texts)} for group in inputs],
            }

            profile.result.step("embedding", metrics=measured, overlap=overlap)
            vectors = {
                digest: vector
                for item in events
                if item.hashes is not None and item.vectors is not None
                for digest, vector in zip(item.hashes, item.vectors, strict=True)
            }
            missing = [digest for digest in hashes if digest not in vectors]
            if missing:
                raise JobFailed(
                    "vectors", "workload inputs were not embedded", missing=len(missing)
                )
            lengths = {
                digest: length
                for item in events
                if item.hashes and item.token_lengths
                for digest, length in zip(item.hashes, item.token_lengths, strict=True)
            }
            if any(
                lengths.get(text_hash(text)) != 512
                for group in inputs
                if group.kind != "mixed"
                for text in group.texts
            ):
                raise JobFailed("workload", "maximum/long-batch inputs did not reach 512 tokens")
            spans = inference_intervals([item for item in events if item.role == "enrichment"])
            overlap.update(
                {
                    name: sum(
                        call.caller == name
                        and call.succeeded
                        and any(
                            a <= call.started + call.seconds and b >= call.started for a, b in spans
                        )
                        for call in driver.calls
                    )
                    for name in ("status", "live", "vector", "hybrid")
                }
            )
            if not all(overlap.values()):
                raise JobFailed(
                    "concurrency",
                    "a required reader did not overlap real inference",
                    overlap=overlap,
                )
            if not any(a <= item.at <= b for item in driver.appended for a, b in spans):
                raise JobFailed("concurrency", "no raw append overlapped real inference")
            retrieval = {}
            for mode in ("vector", "hybrid"):
                calls = driver.drive(
                    "retrieval",
                    dict(
                        zip(
                            [mode],
                            once([Command.semantic("database performance", mode)]),
                            strict=True,
                        )
                    ),
                    timeout_seconds=profile.args.settle_timeout,
                )
                if len(calls) != 1 or not calls[0].succeeded:
                    raise JobFailed("retrieval", "final semantic read failed")
                retrieval[mode] = _retrieval(calls[0].document or {}, profile.run_dir.path)
            build_identity = profile.result.scope["build"]
            assert isinstance(build_identity, dict)
            candidate = Reference(
                hashes=hashes,
                vectors=[vectors[digest] for digest in hashes],
                retrieval=retrieval,
                backend="mlx" if sys.platform == "darwin" else "onnx",
                artifacts=artifacts,
                build=build_identity,
            )
            validate_vectors(candidate)
            comparison = equivalence(prior, candidate) if prior else None
            evidence.update(
                comparison=comparison,
                verification_state="passed" if prior else "reference_recorded",
            )
            profile.tree.write_new(
                profile.tree.root,
                PurePosixPath("reference.json"),
                candidate.model_dump_json().encode(),
            )
            failed = [call.to_json() for call in driver.calls if not call.succeeded]
            responsive = [
                call.seconds
                for call in driver.calls
                if call.argv[0] == "live" or call.argv[:2] == ("daemon", "status")
            ]
            fresh = freshness.seconds()
            from .daemon import (
                FRESHNESS_SECONDS_MAX,
                FRESHNESS_SECONDS_P95_MAX,
                RSS_BYTES_MAX,
                STATUS_SECONDS_MAX,
            )
            from .load import percentile

            if (
                failed
                or len(fresh) != RAW_WRITES
                or max(fresh) > FRESHNESS_SECONDS_MAX
                or percentile(fresh, 0.95) > FRESHNESS_SECONDS_P95_MAX
                or max(responsive) > STATUS_SECONDS_MAX
                or max(max(item.rss_hwm_bytes, item.rss_bytes) for item in events) > RSS_BYTES_MAX
            ):
                raise JobFailed(
                    "floor",
                    "embedding profile breached a read, freshness, memory or completion floor",
                    failed_calls=failed[:20],
                    freshness=seconds_summary(fresh),
                    responsiveness=seconds_summary(responsive),
                )
            turns = profile.turns(daemon, started)
            errors = logged_errors(daemon.output()[1])
            if errors or turns["generation_errors"]:
                raise JobFailed(
                    "daemon_error",
                    "daemon reported errors, including recovered errors",
                    errors=errors[:20],
                    generation_errors=turns["generation_errors"],
                )
            stages = turns["stages"]
            log = turns["log"]
            assert isinstance(stages, dict) and isinstance(log, dict)
            writer_seconds = max(
                float(log["op_seconds_max"]),
                *(float(value["seconds_max_lifetime"]) for value in stages.values()),
            )
            from .daemon import WRITER_SECONDS_MAX

            if writer_seconds > WRITER_SECONDS_MAX:
                raise JobFailed(
                    "floor",
                    "a writer/checkpoint turn exceeded five seconds",
                    writer_seconds_max=writer_seconds,
                )
            profile.tree.write_new(
                profile.tree.root, PurePosixPath("turns.json"), json.dumps(turns).encode()
            )
            profile.tree.write_new(
                profile.tree.root,
                PurePosixPath("status-final.json"),
                json.dumps(profile.cli_json(Command.daemon_status())).encode(),
            )
            responsive.extend(profile.status_seconds)
            if max(responsive) > STATUS_SECONDS_MAX:
                raise JobFailed("floor", "a status call took longer than two seconds")
            profile.result.step(
                "verify",
                vectors=len(hashes),
                comparison=comparison,
                reference_mode="compare" if prior else "record",
            )
            profile.result.step(
                "final",
                stats=profile.cli_json(Command.stats()),
                appended_bytes=appender.bytes_total,
                appended_lines=appender.lines_total,
                floors={
                    "writer_seconds_max": writer_seconds,
                    "freshness_seconds": seconds_summary(fresh),
                    "call_seconds": seconds_summary(responsive),
                    "rss_bytes_max": max(
                        max(item.rss_hwm_bytes, item.rss_bytes) for item in events
                    ),
                },
            )
            profile.hold_if_asked(daemon)
        except JobFailed as error:
            if evidence is not None:
                evidence.update(verification_state="failed", verification_error=error.describe())
            raise
        finally:
            try:
                if evidence is not None:
                    profile.tree.publish(
                        PurePosixPath("embedding.json"),
                        json.dumps(evidence, allow_nan=False).encode(),
                    )
            finally:
                driver.close()
    finally:
        appender.close()
