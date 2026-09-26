#!/usr/bin/env python3
"""Reusable small/populated reconciliation performance command (REQ-RECON-020).

Diagnostic baseline/performance gate only. Final Path B acceptance is still required.
Never accepts by fastest sample. Hard BRIEF writer/checkpoint/RSS floors cannot be
relaxed. Does not convert storage format or move storage aliases.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Never

if TYPE_CHECKING:
    from recall.core.models import Session
    from recall.services.coordinator import PreparedRawSource
    from recall.services.indexer import PersistedSessionRows

REPO = Path(__file__).resolve().parents[1]
VERIFIER_VERSION = 2
WORKER_TIMEOUT_S = 300
SAMPLES_MIN = 3
SAMPLES_MAX = 20
WRITER_CEILING_S = 5.0
CHECKPOINT_CEILING_S = 5.0
READ_CEILING_S = 2.0
RSS_CEILING_BYTES = 4 * 1024**3
BYTE_SLACK = 16 * 1024 * 1024
MESSAGE_COUNT = 128
TOOL_COUNT = 256
OWNED_V1 = "RECALL_BENCH_OWNED_USER_v1"
OWNED_V2 = "RECALL_BENCH_OWNED_USER_v2"
APPEND_MARKER = "RECALL_BENCH_APPEND_MARKER_v1"
OWNED_NAME = "rollout-2026-01-01T00-00-00-benchmark.jsonl"
LATENCY_KEYS = (
    "first_writer_s",
    "first_total_s",
    "append_writer_s",
    "append_total_s",
    "rewrite_writer_s",
    "rewrite_total_s",
    "noop_s",
    "checkpoint_s",
    "keyword_search_s",
    "vector_search_s",
    "overlapping_read_max_s",
    "live_roster_max_s",
)
BYTE_KEYS = ("peak_rss_bytes", "db_bytes_after", "sidecar_bytes_after", "wal_bytes_precheckpoint")
GROWTH_KEYS = ("database_growth_bytes", "sidecar_growth_bytes")
_KEEP_ENV = (
    "PATH",
    "VIRTUAL_ENV",
    "UV_PROJECT",
    "UV_PYTHON",
    "UV_SYSTEM_PYTHON",
    "PYTHONUTF8",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TERM",
    "__PYVENV_LAUNCHER__",
)

Status = Literal["ok", "failed", "invalid_comparison"]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    st = path.stat()
    return {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "ctime_ns": st.st_ctime_ns,
        "sha256": _sha256_file(path),
    }


def _is_wal_journal(path: Path) -> bool:
    name = path.name
    return name.endswith((".wal", "-wal", "-shm", "-journal", ".journal"))


def _require_checkpointed(directory: Path) -> None:
    for name in (
        "recall.duckdb.wal",
        "recall.fts.sqlite-wal",
        "recall.fts.sqlite-shm",
        "recall.fts.sqlite-journal",
    ):
        if (directory / name).exists():
            raise RuntimeError(f"population is not checkpointed: {name}")
    if not (directory / "recall.duckdb").is_file():
        raise RuntimeError("population is missing recall.duckdb")


def _payload_metadata(directory: Path) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
        for name in ("recall.duckdb", "recall.fts.sqlite")
        if (directory / name).is_file()
        for stat in [(directory / name).stat()]
    ]


def _payload_identity(directory: Path) -> dict[str, object]:
    return {
        "files": [
            {"name": name, **_file_identity(directory / name)}
            for name in ("recall.duckdb", "recall.fts.sqlite")
            if (directory / name).is_file()
        ]
    }


def _clone_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        subprocess.run(
            ["cp", "-c", str(src), str(dst)], check=True, capture_output=True, timeout=30
        )
    else:
        shutil.copy2(src, dst)


def _clone_population(src: Path, dst: Path) -> None:
    _require_checkpointed(src)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("recall.duckdb", "recall.fts.sqlite"):
        if (src / name).is_file():
            _clone_or_copy_file(src / name, dst / name)
    _require_checkpointed(src)


def _peak_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _median(values: list[float]) -> float:
    assert len(values) >= 1
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _json_norm(value: object) -> str:
    parsed: object = value
    if value is None:
        return "null"
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_small_fixture() -> str:
    started = datetime(2026, 1, 1, tzinfo=UTC)
    lines = [
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": "recall-bench-codex",
                    "timestamp": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "cwd": "/repo",
                    "git": {"branch": "main", "root": "/repo"},
                },
            },
            separators=(",", ":"),
        )
    ]
    pairs = MESSAGE_COUNT // 2
    for index in range(pairs):
        stamp = started + timedelta(seconds=index)
        user = OWNED_V1 if index == 0 else f"bench-msg-user-{index}"
        lines.append(
            json.dumps(
                {
                    "type": "response_item",
                    "timestamp": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": user}],
                    },
                },
                separators=(",", ":"),
            )
        )
        lines.append(
            json.dumps(
                {
                    "type": "response_item",
                    "timestamp": (stamp + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": f"bench-msg-assistant-{index}"}
                        ],
                    },
                },
                separators=(",", ":"),
            )
        )
    for index in range(TOOL_COUNT):
        stamp = started + timedelta(seconds=1000 + index)
        call_id = f"call_bench_{index}"
        lines.append(
            json.dumps(
                {
                    "type": "response_item",
                    "timestamp": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "payload": {
                        "type": "function_call",
                        "name": "exec_command",
                        "call_id": call_id,
                        "arguments": json.dumps({"cmd": f"bench-tool-{index}"}),
                    },
                },
                separators=(",", ":"),
            )
        )
        lines.append(
            json.dumps(
                {
                    "type": "response_item",
                    "timestamp": (stamp + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "payload": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": f"ok-{index}",
                    },
                },
                separators=(",", ":"),
            )
        )
    return "\n".join(lines) + "\n"


def _append_marker(path: Path) -> None:
    record = {
        "type": "event_msg",
        "timestamp": "2026-09-10T00:00:00Z",
        "payload": {"type": "user_message", "message": APPEND_MARKER},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _rewrite_first_user(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if OWNED_V1 in text:
        path.write_text(text.replace(OWNED_V1, OWNED_V2, 1), encoding="utf-8")
        return
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        raw = line.strip()
        if not raw:
            continue
        record = json.loads(raw)
        payload = record.get("payload") or {}
        changed = False
        if record.get("type") == "event_msg" and payload.get("type") == "user_message":
            payload["message"] = OWNED_V2
            changed = True
        elif (
            record.get("type") == "response_item"
            and payload.get("type") == "message"
            and payload.get("role") == "user"
        ):
            content = payload.get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict):
                content[0]["text"] = OWNED_V2
            else:
                payload["content"] = [{"type": "input_text", "text": OWNED_V2}]
            changed = True
        if changed:
            record["payload"] = payload
            suffix = "\n" if line.endswith("\n") else ""
            lines[index] = json.dumps(record, separators=(",", ":")) + suffix
            path.write_text("".join(lines), encoding="utf-8")
            return
    raise RuntimeError("no supported user record to rewrite")


def _no_model() -> Never:
    raise AssertionError("embedding model load is prohibited")


def _environment() -> dict[str, object]:
    processor = platform.processor() or platform.machine()
    return {
        "python": platform.python_version(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "cpu": processor,
        "cpu_count": os.cpu_count(),
        "sqlite": sqlite3.sqlite_version,
        "platform": platform.platform(),
        "packages": {
            "recall": _pkg_version("recall"),
            "duckdb": _pkg_version("duckdb"),
            "pyarrow": _pkg_version("pyarrow"),
            "numpy": _pkg_version("numpy"),
        },
    }


def _git_state() -> dict[str, object]:
    paths = ["packages/recall/src", "packages/recall/pyproject.toml", "uv.lock"]
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", *paths], cwd=REPO)
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *paths).splitlines()
    digest = hashlib.sha256(diff)
    for name in sorted(untracked):
        digest.update(name.encode())
        digest.update((REPO / name).read_bytes())
    return {
        "head": _git("rev-parse", "HEAD"),
        "dirty_relevant": bool(diff or untracked),
        "dirty_hash": digest.hexdigest(),
    }


def _isolated_env(worker_dir: Path) -> dict[str, str]:
    home = worker_dir / "home"
    data = worker_dir / "data"
    env = {key: os.environ[key] for key in _KEEP_ENV if key in os.environ}
    env.update(
        {
            "HOME": str(home),
            "TMPDIR": str(worker_dir / "tmp"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
            "RECALL_DATA_DIR": str(data),
            "RECALL_CONFIG_PATH": str(worker_dir / "config" / "config.toml"),
            "RECALL_DB_PATH": str(data / "recall.duckdb"),
            "RECALL_LOCK_PATH": str(data / "recall.lock"),
            "RECALL_DAEMON_EMBED": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HOME": str(worker_dir / "hf"),
        }
    )
    return env


def _write_config(worker_dir: Path, owned_dir: Path) -> None:
    config_path = worker_dir / "config" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "[embedding.context]",
                'mode = "template"',
                'fallback = "off"',
                "",
                "[daemon]",
                "embed = false",
                "",
                "[fts]",
                'backend = "sqlite_sidecar"',
                "[compaction]",
                "auto_trigger = false",
                "[sources.codex]",
                f"roots = [{json.dumps(str(owned_dir))}]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _store_sizes(db_path: Path, data_dir: Path) -> dict[str, int]:
    from recall.db.connection import wal_size_bytes
    from recall.db.fts_sidecar import sidecar_path

    sidecar = sidecar_path(data_dir)
    sidecar_wal = sidecar.with_name(sidecar.name + "-wal")
    wal_bytes = wal_size_bytes(db_path)
    return {
        "db_bytes": db_path.stat().st_size if db_path.exists() else 0,
        "sidecar_bytes": sidecar.stat().st_size if sidecar.exists() else 0,
        "wal_bytes": int(wal_bytes),
        "sidecar_wal_bytes": sidecar_wal.stat().st_size if sidecar_wal.exists() else 0,
    }


def _tool_order(row: tuple[object, ...]) -> tuple[bool, int, int]:
    message_idx, tool_idx = row[:2]
    assert message_idx is None or isinstance(message_idx, int)
    assert isinstance(tool_idx, int)
    return message_idx is None, message_idx or 0, tool_idx


def _canonical(
    session: Session, rows: PersistedSessionRows
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    messages = tuple(
        (m.idx, m.role.value, m.content, m.thinking, m.agent_id) for m in session.messages
    )
    tools: list[tuple[object, ...]] = []
    for message in session.messages:
        for tool in message.tool_calls:
            tools.append(
                (
                    message.idx,
                    tool.idx,
                    tool.tool_name,
                    _json_norm(tool.tool_input),
                    tool.bash_command,
                )
            )
    for tool in session.orphan_tool_calls:
        tools.append(
            (None, tool.idx, tool.tool_name, _json_norm(tool.tool_input), tool.bash_command)
        )
    parsed_tools = tuple(sorted(tools, key=_tool_order))
    stored_messages = tuple(
        (row[2], str(row[3]), row[4], row[5], row[8]) for row in rows.message_rows
    )
    id_to_idx = {str(row[0]): row[2] for row in rows.message_rows}
    stored_tools = tuple(
        (
            id_to_idx.get(str(row[2])) if row[2] is not None else None,
            row[3],
            str(row[4]),
            _json_norm(row[5]),
            row[6],
        )
        for row in rows.tool_call_rows
    )
    if messages != stored_messages:
        raise RuntimeError(
            f"canonical message mismatch parsed={len(messages)} stored={len(stored_messages)}"
        )
    if parsed_tools != tuple(sorted(stored_tools, key=_tool_order)):
        raise RuntimeError(
            f"canonical tool mismatch parsed={len(parsed_tools)} stored={len(stored_tools)}"
        )
    return messages, parsed_tools


def _verify_keyword_membership(data: Path, rows: PersistedSessionRows) -> None:
    """Check exact owned membership and document presence, including NULL commands."""
    with sqlite3.connect((data / "recall.fts.sqlite").as_uri() + "?mode=ro", uri=True) as sidecar:
        for mapping, document, key, ids, expected in (
            (
                "message_fts_rowid",
                "message_fts",
                "message_id",
                [str(row[0]) for row in rows.message_rows],
                {str(row[0]) for row in rows.message_rows},
            ),
            (
                "tool_calls_fts_rowid",
                "tool_calls_fts",
                "tool_call_id",
                [str(row[0]) for row in rows.tool_call_rows],
                {str(row[0]) for row in rows.tool_call_rows if row[6] is not None},
            ),
        ):
            found: set[str] = set()
            for chunk in batched(ids, 500):
                placeholders = ",".join("?" for _ in chunk)
                actual = sidecar.execute(
                    f"SELECT m.{key}, d.rowid FROM {mapping} m "
                    f"LEFT JOIN {document} d ON d.rowid=m.rowid "
                    f"WHERE m.{key} IN ({placeholders})",
                    chunk,
                ).fetchall()
                for identifier, rowid in actual:
                    assert rowid is not None, f"missing {document} document"
                    found.add(identifier)
            assert found == expected, f"incorrect {document} membership"


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True, default=str).encode()
    ).hexdigest()


def _floors_breached(sample: dict[str, object]) -> list[str]:
    breaches: list[str] = []
    for key in LATENCY_KEYS + BYTE_KEYS:
        value = sample.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            breaches.append(f"invalid_{key}")
    for key in ("first_writer_s", "append_writer_s", "rewrite_writer_s"):
        value = sample.get(key)
        if isinstance(value, (int, float)) and float(value) > WRITER_CEILING_S:
            breaches.append(key)
    checkpoint = sample.get("checkpoint_s")
    if isinstance(checkpoint, (int, float)) and float(checkpoint) > CHECKPOINT_CEILING_S:
        breaches.append("checkpoint_s")
    rss = sample.get("peak_rss_bytes")
    if isinstance(rss, (int, float)) and rss > RSS_CEILING_BYTES:
        breaches.append("peak_rss_bytes")
    overlapping = sample.get("overlapping_reads", 0)
    assert isinstance(overlapping, int)
    if overlapping <= 0:
        breaches.append("no_overlapping_read")
    for key in ("overlapping_read_max_s", "live_roster_max_s"):
        read_max = sample.get(key)
        if isinstance(read_max, (int, float)) and float(read_max) > READ_CEILING_S:
            breaches.append(key)
    if sample.get("overlapping_reads_truthful") is not True:
        breaches.append("unverified_reads")
    if sample.get("wal_bytes_after") != 0:
        breaches.append("uncheckpointed_wal")
    if sample.get("noop_wal_bytes_before") != sample.get("noop_wal_bytes_after"):
        breaches.append("noop_changed_wal")
    return breaches


def run_worker(worker_dir: Path, progress: dict[str, object]) -> dict[str, object]:
    import duckdb
    from recall.core.config import AppConfig
    from recall.core.types import SearchMode
    from recall.db import connect
    from recall.db.connection import wal_size_bytes
    from recall.db.queries import insert_message_embeddings
    from recall.db.source_files import SourceCatalog
    from recall.parsers.codex import CodexParser
    from recall.services.coordinator import (
        PreparedRawSource,
        capture_path,
        commit_prepared_raw_sources,
        observe_path,
        prepare_raw_sources,
    )
    from recall.services.indexer import _load_persisted_session_rows
    from recall.services.live import live_view_page
    from recall.services.search import search

    spec = json.loads((worker_dir / "spec.json").read_text(encoding="utf-8"))
    kind = str(spec["kind"])
    owned = Path(spec["owned_source"])
    home = Path(os.environ["HOME"]).resolve()
    data = Path(os.environ["RECALL_DATA_DIR"]).resolve()
    if home != (worker_dir / "home").resolve() or data != (worker_dir / "data").resolve():
        raise RuntimeError("worker isolation leak")
    cfg = AppConfig.load()
    if cfg.db_path.exists():
        raw = duckdb.connect(str(cfg.db_path), read_only=True)
        try:
            row = raw.execute(
                "SELECT embedding_dimensions FROM runtime_state WHERE singleton"
            ).fetchone()
        except duckdb.Error:
            row = None
        finally:
            raw.close()
        if row is not None and row[0] is not None:
            cfg = replace(cfg, embedding=replace(cfg.embedding, dimensions=int(row[0])))
    conn = connect(cfg)
    parser = CodexParser(roots=(owned.parent,))
    sizes_before = _store_sizes(cfg.db_path, cfg.data_dir)
    storage = conn.execute(
        "SELECT tags['storage_version'] FROM duckdb_databases() "
        "WHERE database_name=current_database()"
    ).fetchone()
    assert storage is not None

    def reconcile() -> tuple[int, tuple[PreparedRawSource, ...], float, float]:
        t0 = time.perf_counter()
        captured = capture_path(parser, owned)
        item = observe_path(parser, captured, conn=conn)
        prepared = prepare_raw_sources((item,), {"codex": parser})
        tw = time.perf_counter()
        committed = commit_prepared_raw_sources(prepared, cfg, conn=conn)
        t1 = time.perf_counter()
        return committed, prepared, t1 - t0, t1 - tw

    def assert_owned(
        prepared: tuple[PreparedRawSource, ...],
        *,
        expect_v1: bool,
        expect_v2: bool,
        expect_append: bool,
    ) -> tuple[str, str, int, int]:
        capture = prepared[0]
        if capture.result is None or capture.result.diagnostics:
            raise RuntimeError(capture.error or "prepare produced no result")
        session = capture.result.session
        if not session.messages:
            raise RuntimeError("parser produced no messages")
        if (
            kind == "small"
            and expect_v1
            and not expect_append
            and (
                len(session.messages) != MESSAGE_COUNT
                or len(session.orphan_tool_calls) != TOOL_COUNT
            )
        ):
            raise RuntimeError("small fixture did not parse expected counts")
        if expect_v1 and not any(message.content == OWNED_V1 for message in session.messages):
            raise RuntimeError("parser did not index owned user content")
        if expect_v2 and not any(
            OWNED_V2 in (message.content or "") for message in session.messages
        ):
            raise RuntimeError("rewrite did not change expected user projection")
        if expect_append and not any(
            message.content == APPEND_MARKER for message in session.messages
        ):
            raise RuntimeError("append marker missing from parsed projection")
        rows = _load_persisted_session_rows(conn, session.id)
        if rows is None:
            raise RuntimeError("owned session rows missing")
        parsed, tools = _canonical(session, rows)
        _verify_keyword_membership(cfg.data_dir, rows)
        return str(session.id), _digest([parsed, tools]), len(parsed), len(tools)

    print("phase first", flush=True)
    committed, prepared, first_total, first_writer = reconcile()
    progress.update(first_total_s=first_total, first_writer_s=first_writer)
    if committed != 1:
        raise RuntimeError(f"first commit returned {committed}")
    sid, first_digest, message_count, tool_count = assert_owned(
        prepared, expect_v1=kind == "small", expect_v2=False, expect_append=False
    )

    print("phase append", flush=True)
    _append_marker(owned)
    committed, prepared, append_total, append_writer = reconcile()
    progress.update(append_total_s=append_total, append_writer_s=append_writer)
    if committed != 1:
        raise RuntimeError(f"append commit returned {committed}")
    sid, append_digest, message_count, tool_count = assert_owned(
        prepared, expect_v1=kind == "small", expect_v2=False, expect_append=True
    )
    if append_digest == first_digest:
        raise RuntimeError("append did not change owned projection")

    print("phase rewrite", flush=True)
    _rewrite_first_user(owned)
    committed, prepared, rewrite_total, rewrite_writer = reconcile()
    progress.update(rewrite_total_s=rewrite_total, rewrite_writer_s=rewrite_writer)
    if committed != 1:
        raise RuntimeError(f"rewrite commit returned {committed}")
    sid, rewrite_digest, message_count, tool_count = assert_owned(
        prepared, expect_v1=False, expect_v2=True, expect_append=True
    )
    if rewrite_digest == append_digest:
        raise RuntimeError("rewrite did not change owned projection")

    print("phase noop", flush=True)
    last = prepared[0]
    # The rewrite commit advanced the source past the generation `last` was
    # captured at, and committing a stale capture is a recorded rejection, not a
    # no-op. Replay the same content against the catalog's current row.
    current = SourceCatalog(conn, clock=time.time).get(last.item.source, last.item.source_path)
    if current is None or not current.current:
        raise RuntimeError("rewrite left the owned source pending")
    noop_prepared = PreparedRawSource(current, last.parser, last.result, observed=None)
    wal_before = wal_size_bytes(cfg.db_path)
    t0 = time.perf_counter()
    noop_committed = commit_prepared_raw_sources((noop_prepared,), cfg, conn=conn)
    noop_s = time.perf_counter() - t0
    wal_after = wal_size_bytes(cfg.db_path)
    progress.update(noop_s=noop_s, noop_wal_bytes_before=wal_before, noop_wal_bytes_after=wal_after)
    if noop_committed != 0:
        raise RuntimeError(f"no-op commit returned {noop_committed}")
    if wal_after != wal_before:
        raise RuntimeError("no-op grew WAL")

    print("phase search", flush=True)
    rows = _load_persisted_session_rows(conn, sid)
    assert rows is not None
    seed_id = str(rows.message_rows[0][0])
    dimensions = cfg.embedding.dimensions
    if dimensions < 1:
        raise RuntimeError("embedding dimensions must be positive")
    vector = [1.0] + [0.0] * (dimensions - 1)
    insert_message_embeddings(conn, [(seed_id, vector, None)])
    t0 = time.perf_counter()
    keyword_hits = search(
        query=APPEND_MARKER,
        source=None,
        tool=None,
        session=None,
        mode=SearchMode.KEYWORD,
        config=cfg,
        conn=conn,
        embed_backend_factory=_no_model,
    )
    keyword_s = time.perf_counter() - t0
    progress["keyword_search_s"] = keyword_s
    search_errors: dict[str, str] = {}
    t0 = time.perf_counter()
    try:
        vector_hits = search(
            query=APPEND_MARKER,
            source=None,
            tool=None,
            session=None,
            mode=SearchMode.VECTOR,
            config=cfg,
            conn=conn,
            query_embedding=vector,
            embed_backend_factory=_no_model,
        )
    except duckdb.OutOfMemoryException as err:
        # A measured failed query is retained; it never becomes a valid latency
        # comparison. Continue only after proving the owned handle is usable.
        search_errors["vector"] = type(err).__name__
        print(f"vector search failed: {type(err).__name__}", flush=True)
        assert conn.execute("SELECT 1").fetchone() == (1,)
        vector_hits = []
    vector_s = time.perf_counter() - t0
    progress["vector_search_s"] = vector_s
    if not keyword_hits:
        raise RuntimeError("keyword search returned no owned hit")
    if not vector_hits and not search_errors:
        raise RuntimeError("vector search returned no owned hit")
    keyword_ids = [
        {"kind": hit.kind, "session_id": hit.session_id, "message_id": hit.message_id}
        for hit in keyword_hits[:3]
    ]
    vector_ids = [
        {"kind": hit.kind, "session_id": hit.session_id, "message_id": hit.message_id}
        for hit in vector_hits[:3]
    ]
    if not any(hit.session_id == sid for hit in keyword_hits):
        raise RuntimeError("keyword hit identity mismatch")
    if not search_errors and not any(hit.message_id == seed_id for hit in vector_hits):
        raise RuntimeError("vector hit identity mismatch")

    print("phase live roster", flush=True)
    roster_samples: list[dict[str, float | int]] = []
    roster_cursor = None
    # Match the daemon's naive local clock for stored activity timestamps.
    roster_now = datetime.now()
    for _ in range(3):
        t0 = time.perf_counter()
        page = live_view_page(
            conn=conn,
            watched_paths=(),
            include_idle=True,
            now=roster_now,
            idle_window_seconds=365 * 86400,
            limit=50,
            cursor=roster_cursor,
            local_host=platform.node(),
        )
        roster_samples.append({"seconds": time.perf_counter() - t0, "rows": len(page.sessions)})
        roster_cursor = page.next_cursor
        if roster_cursor is None:
            break
    if not any(sample["rows"] for sample in roster_samples):
        raise RuntimeError("live roster returned no owned or populated rows")
    live_roster_max = max(float(sample["seconds"]) for sample in roster_samples)
    progress.update(live_roster_max_s=live_roster_max, live_roster_samples=roster_samples)

    print("phase checkpoint", flush=True)
    ready = threading.Event()
    done = threading.Event()
    reads: list[tuple[float, float, int, bool]] = []
    reader_error: str | None = None

    def reader() -> None:
        nonlocal reader_error
        cursor = None
        try:
            cursor = conn.cursor()
            catalog = SourceCatalog(cursor, clock=time.time)
            for _ in range(10_000):
                if done.is_set() and reads:
                    break
                start = time.perf_counter()
                page = catalog.status_page(limit=100)
                end = time.perf_counter()
                truthful = all(
                    bool(row.source_path)
                    and row.desired_generation >= 0
                    and row.committed_generation >= 0
                    for row in page.rows
                )
                reads.append((start, end, len(page.rows), truthful))
                ready.set()
                if done.wait(0.001):
                    break
        except Exception as err:
            reader_error = type(err).__name__
            ready.set()
        finally:
            if cursor is not None:
                cursor.close()

    thread = threading.Thread(target=reader, name="bench-checkpoint-reader", daemon=True)
    thread.start()
    if not ready.wait(2):
        raise RuntimeError("checkpoint reader was not ready")
    wal_precheckpoint = wal_size_bytes(cfg.db_path)
    assert wal_precheckpoint > 0, "checkpoint has no committed WAL"
    started = time.perf_counter()
    try:
        conn.execute("CHECKPOINT")
    finally:
        checkpoint_s = time.perf_counter() - started
        done.set()
        thread.join(5)
    if thread.is_alive():
        raise RuntimeError("checkpoint reader did not release connection")
    overlapping = [item for item in reads if started < item[0] < item[1] < started + checkpoint_s]
    read_latencies = [item[1] - item[0] for item in overlapping]
    sizes_after = _store_sizes(cfg.db_path, cfg.data_dir)
    if sizes_after["wal_bytes"] != 0:
        raise RuntimeError("checkpoint left pending WAL")
    conn.close()
    sample: dict[str, object] = {
        "python_executable": sys.executable,
        "environment": _environment(),
        "status": "ok",
        "kind": kind,
        "storage_format": storage[0],
        "session_id": sid,
        "owned_message_count": message_count,
        "owned_tool_count": tool_count,
        "canonical_digest": rewrite_digest,
        "live_roster_max_s": live_roster_max,
        "live_roster_samples": roster_samples,
        "first_writer_s": first_writer,
        "first_total_s": first_total,
        "append_writer_s": append_writer,
        "append_total_s": append_total,
        "rewrite_writer_s": rewrite_writer,
        "rewrite_total_s": rewrite_total,
        "noop_s": noop_s,
        "noop_wal_bytes_before": wal_before,
        "noop_wal_bytes_after": wal_after,
        "checkpoint_s": checkpoint_s,
        "overlapping_reads": len(overlapping),
        "overlapping_read_max_s": max(read_latencies) if read_latencies else 0.0,
        "overlapping_reads_truthful": all(item[3] for item in overlapping) if overlapping else True,
        "checkpoint_reader_error": reader_error,
        "search_errors": search_errors,
        "keyword_search_s": keyword_s,
        "vector_search_s": vector_s,
        "keyword_hit_ids": keyword_ids,
        "vector_hit_ids": vector_ids,
        "peak_rss_bytes": _peak_rss_bytes(),
        "db_bytes_before": sizes_before["db_bytes"],
        "sidecar_bytes_before": sizes_before["sidecar_bytes"],
        "wal_bytes_before": sizes_before["wal_bytes"],
        "db_bytes_after": sizes_after["db_bytes"],
        "sidecar_bytes_after": sizes_after["sidecar_bytes"],
        "wal_bytes_after": sizes_after["wal_bytes"],
        "wal_bytes_precheckpoint": wal_precheckpoint,
        "database_growth_bytes": sizes_after["db_bytes"] - sizes_before["db_bytes"],
        "sidecar_growth_bytes": sizes_after["sidecar_bytes"] - sizes_before["sidecar_bytes"],
    }
    if search_errors or reader_error or not sample["overlapping_reads_truthful"]:
        sample["status"] = "failed"
        sample["error_type"] = (
            search_errors.get("vector") or reader_error or "untruthful_status_page"
        )
    breaches = _floors_breached(sample)
    sample["floors_breached"] = breaches
    if breaches:
        sample["status"] = "failed"
    return sample


def _failed_sample(error_type: str, detail: str) -> dict[str, object]:
    return {
        "status": "failed",
        "error_type": error_type,
        "error": detail[:300],
        "floors_breached": [],
    }


def _latency_allowed(baseline: float, candidate: float) -> bool:
    slack = 0.010 if baseline < 1.0 else 0.050
    return Decimal(str(candidate)) <= Decimal(str(baseline)) * Decimal("1.20") + Decimal(str(slack))


def _bytes_allowed(baseline: float, candidate: float) -> bool:
    return candidate <= baseline * 1.10 + BYTE_SLACK


def _input_digests(payload: dict[str, object]) -> dict[str, object]:
    identity = payload.get("input")
    if not isinstance(identity, dict):
        return {}
    if identity.get("kind") == "small":
        return {"kind": "small", "fixture_sha256": identity.get("fixture_sha256")}
    files = identity.get("population_files")
    names = []
    if isinstance(files, list):
        for item in files:
            if isinstance(item, dict):
                names.append((item.get("name"), item.get("sha256"), item.get("size")))
    source = identity.get("source")
    source_digest = source.get("sha256") if isinstance(source, dict) else None
    return {"kind": "populated", "population": names, "source_sha256": source_digest}


def _env_key(payload: dict[str, object]) -> tuple[object, ...]:
    env = payload.get("environment")
    if not isinstance(env, dict):
        return ()
    packages = env.get("packages", {})
    assert isinstance(packages, dict)
    return (
        env.get("python"),
        env.get("os"),
        env.get("machine"),
        env.get("cpu"),
        env.get("cpu_count"),
        env.get("sqlite"),
        packages.get("pyarrow"),
        packages.get("numpy"),
    )


def compare_baseline(current: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    mismatches: list[str] = []
    if baseline.get("verifier_version") != current.get("verifier_version"):
        mismatches.append("verifier_version")
    if baseline.get("script_sha256") != current.get("script_sha256"):
        mismatches.append("script_sha256")
    if _input_digests(baseline) != _input_digests(current):
        mismatches.append("input")
    if _env_key(baseline) != _env_key(current):
        mismatches.append("environment")
    for payload in (current, baseline):
        samples = payload.get("samples")
        if not isinstance(samples, list) or len(samples) < SAMPLES_MIN:
            mismatches.append("samples")
        elif any(not isinstance(sample, dict) or sample.get("search_errors") for sample in samples):
            mismatches.append("failed_search_measurement")
    if mismatches:
        return {"status": "invalid_comparison", "mismatches": mismatches, "regressions": []}
    regressions: list[str] = []
    current_agg = current.get("aggregate")
    baseline_agg = baseline.get("aggregate")
    if not isinstance(current_agg, dict) or not isinstance(baseline_agg, dict):
        return {"status": "invalid_comparison", "mismatches": ["aggregate"], "regressions": []}
    for key in LATENCY_KEYS:
        left = baseline_agg.get(key)
        right = current_agg.get(key)
        if not isinstance(left, dict) or not isinstance(right, dict):
            mismatches.append(key)
            continue
        for stat_name in ("median", "worst"):
            b_val = left.get(stat_name)
            c_val = right.get(stat_name)
            if not isinstance(b_val, (int, float)) or not isinstance(c_val, (int, float)):
                mismatches.append(f"{key}.{stat_name}")
                continue
            if not math.isfinite(b_val) or not math.isfinite(c_val) or min(b_val, c_val) < 0:
                mismatches.append(f"{key}.{stat_name}")
                continue
            if not _latency_allowed(float(b_val), float(c_val)):
                regressions.append(f"{key}.{stat_name}")
    for key in BYTE_KEYS + GROWTH_KEYS:
        left = baseline_agg.get(key)
        right = current_agg.get(key)
        if not isinstance(left, dict) or not isinstance(right, dict):
            mismatches.append(key)
            continue
        for stat_name in ("median", "worst"):
            b_val = left.get(stat_name)
            c_val = right.get(stat_name)
            if not isinstance(b_val, (int, float)) or not isinstance(c_val, (int, float)):
                mismatches.append(f"{key}.{stat_name}")
                continue
            if not math.isfinite(b_val) or not math.isfinite(c_val):
                mismatches.append(f"{key}.{stat_name}")
                continue
            if not _bytes_allowed(float(b_val), float(c_val)):
                regressions.append(f"{key}.{stat_name}")
    if mismatches:
        return {
            "status": "invalid_comparison",
            "mismatches": mismatches,
            "regressions": regressions,
        }
    return {
        "status": "failed" if regressions else "ok",
        "mismatches": [],
        "regressions": regressions,
    }


def _aggregate(samples: list[dict[str, object]]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for key in LATENCY_KEYS + BYTE_KEYS + GROWTH_KEYS:
        values = [
            float(value) for sample in samples if isinstance(value := sample.get(key), (int, float))
        ]
        if len(values) < SAMPLES_MIN:
            aggregate[key] = {"samples": values, "median": None, "worst": None, "n": len(values)}
            continue
        aggregate[key] = {
            "samples": values,
            "median": _median(values),
            "worst": max(values),
            "n": len(values),
        }
    return aggregate


@dataclass(frozen=True)
class ParentArgs:
    output: Path
    population: Path | None
    source: Path | None
    samples: int
    baseline: Path | None


def _prepare_worker(
    worker_dir: Path, args: ParentArgs, fixture: str | None, source: Path | None
) -> dict[str, object]:
    home = worker_dir / "home"
    data = worker_dir / "data"
    owned_dir = worker_dir / "owned"
    tmp = worker_dir / "tmp"
    for path in (home, data, owned_dir, tmp, worker_dir / "hf"):
        path.mkdir(parents=True, exist_ok=True)
    owned = owned_dir / OWNED_NAME
    if fixture is not None:
        owned.write_text(fixture, encoding="utf-8")
        kind = "small"
    else:
        assert source is not None and args.population is not None
        _clone_population(args.population, data)
        _clone_or_copy_file(source, owned)
        kind = "populated"
    _write_config(worker_dir, owned_dir)
    spec = {"kind": kind, "owned_source": str(owned)}
    (worker_dir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec


def run_parent(args: ParentArgs) -> int:
    if args.output.exists():
        print(f"output directory already exists: {args.output}", file=sys.stderr)
        return 2
    if (args.population is None) != (args.source is None):
        print("--population and --source must be provided together", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True)
    fixture = build_small_fixture() if args.population is None else None
    source = args.source.resolve() if args.source is not None else None
    population = args.population.resolve() if args.population is not None else None
    if population is not None:
        _require_checkpointed(population)
        input_before = {
            "kind": "populated",
            "population_files": _payload_identity(population)["files"],
            "source": _file_identity(source) if source is not None else None,
        }
    else:
        assert fixture is not None
        input_before = {
            "kind": "small",
            "messages": MESSAGE_COUNT,
            "tools": TOOL_COUNT,
            "fixture_sha256": hashlib.sha256(fixture.encode()).hexdigest(),
        }
    envelope: dict[str, object] = {
        "verifier_version": VERIFIER_VERSION,
        "acceptance": "diagnostic_baseline_performance_gate_only",
        "path_b_acceptance": False,
        "floors": {
            "writer_s": WRITER_CEILING_S,
            "checkpoint_s": CHECKPOINT_CEILING_S,
            "rss_bytes": RSS_CEILING_BYTES,
            "status_read_s": READ_CEILING_S,
        },
        "script_sha256": _sha256_file(Path(__file__)),
        "git": _git_state(),
        "environment": _environment(),
        "input": input_before,
        "input_paths": {
            "population": str(population) if population else None,
            "source": str(source) if source else None,
        },
        "samples": [],
    }
    samples: list[dict[str, object]] = []
    initial_metadata = _payload_metadata(population) if population else None
    script = str(Path(__file__).resolve())
    for index in range(1, args.samples + 1):
        label = f"{index:02d}"
        worker_dir = args.output / "samples" / label
        worker_dir.mkdir(parents=True)
        print(f"sample {index}/{args.samples} starting", flush=True)
        try:
            _prepare_worker(worker_dir, args, fixture, source)
            if population is not None and source is not None:
                _require_checkpointed(population)
                if (
                    _payload_metadata(population) != initial_metadata
                    or _file_identity(source) != input_before["source"]
                ):
                    raise RuntimeError("input identity changed while preparing worker")
            log_path = worker_dir / "worker.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [sys.executable, script, "--worker-dir", str(worker_dir)],
                    env=_isolated_env(worker_dir),
                    cwd=str(REPO),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=WORKER_TIMEOUT_S,
                    check=False,
                )
            result_path = worker_dir / "result.json"
            if result_path.is_file():
                sample = json.loads(result_path.read_text(encoding="utf-8"))
            elif completed.returncode != 0:
                sample = _failed_sample("worker_exit", f"exit {completed.returncode}")
            else:
                sample = _failed_sample("missing_result", "worker wrote no result.json")
            sample["sample_index"] = index
            sample["worker_dir"] = str(worker_dir)
            sample["returncode"] = completed.returncode
        except subprocess.TimeoutExpired:
            sample = _failed_sample("timeout", f"worker exceeded {WORKER_TIMEOUT_S}s")
            sample["sample_index"] = index
            sample["worker_dir"] = str(worker_dir)
            (worker_dir / "result.json").write_text(json.dumps(sample, indent=2) + "\n")
        except Exception as err:
            sample = _failed_sample(type(err).__name__, str(err))
            sample["sample_index"] = index
            sample["worker_dir"] = str(worker_dir)
            (worker_dir / "result.json").write_text(json.dumps(sample, indent=2) + "\n")
        if (
            population is not None
            and source is not None
            and (
                _payload_metadata(population) != initial_metadata
                or _file_identity(source) != input_before["source"]
            )
        ):
            sample["status"] = "failed"
            sample["error_type"] = "input_mutated"
        samples.append(sample)
        (worker_dir / "result.json").write_text(json.dumps(sample, indent=2) + "\n")
        print(f"sample {index}/{args.samples} {sample.get('status')}", flush=True)

    if population is not None and source is not None:
        _require_checkpointed(population)
        if (
            _payload_identity(population)["files"] != input_before["population_files"]
            or _file_identity(source) != input_before["source"]
        ):
            samples.append(_failed_sample("input_mutated", "final input digest differs"))
    envelope["samples"] = samples
    envelope["aggregate"] = _aggregate(samples)
    floors = [item for sample in samples for item in _floors_breached(sample)]
    any_failed = any(sample.get("status") != "ok" for sample in samples)
    envelope["floors_breached"] = floors
    comparison = None
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        comparison = compare_baseline(envelope, baseline)
        envelope["baseline_comparison"] = comparison
    status: Status = "ok"
    exit_code = 0
    if comparison is not None and comparison["status"] == "invalid_comparison":
        status = "invalid_comparison"
        exit_code = 3
    elif floors or any_failed or (comparison is not None and comparison["status"] == "failed"):
        status = "failed"
        exit_code = 1
    envelope["status"] = status
    (args.output / "result.json").write_text(
        json.dumps(envelope, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "output": str(args.output / "result.json")}), flush=True)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Recall reconciliation performance benchmark")
    parser.add_argument("--output", type=Path, help="new output directory")
    parser.add_argument("--population", type=Path, help="checkpointed data directory")
    parser.add_argument("--source", type=Path, help="Codex transcript")
    parser.add_argument("--samples", type=int, default=SAMPLES_MIN)
    parser.add_argument("--baseline", type=Path, help="prior result.json")
    parser.add_argument("--worker-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_dir is not None:
        worker_dir = args.worker_dir.resolve()
        result_path = worker_dir / "result.json"
        progress: dict[str, object] = {}
        try:
            sample = run_worker(worker_dir, progress)
            result_path.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
            return 0 if sample.get("status") == "ok" else 1
        except Exception as err:
            sample = {
                **progress,
                **_failed_sample(type(err).__name__, str(err)),
                "peak_rss_bytes": _peak_rss_bytes(),
            }
            result_path.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
            return 1
    if args.output is None:
        parser.error("--output is required")
    if not SAMPLES_MIN <= args.samples <= SAMPLES_MAX:
        parser.error(f"samples must be between {SAMPLES_MIN} and {SAMPLES_MAX}")
    return run_parent(
        ParentArgs(
            output=args.output,
            population=args.population,
            source=args.source,
            samples=args.samples,
            baseline=args.baseline,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
