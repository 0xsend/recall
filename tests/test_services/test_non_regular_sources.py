"""Real coordinator captures refuse special files without disturbing stored history."""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.core.types import Source
from recall.db.schema import ensure_schema
from recall.db.source_files import SourceCatalog
from recall.parsers import get_parser
from recall.parsers.compressed import zstd
from recall.services.coordinator import (
    capture_path,
    commit_prepared_raw_sources,
    observe_path,
    prepare_raw_sources,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"


def release_pipe(path: Path, future: Future) -> None:
    """Release every fallback open of the owned pipe before joining a red worker."""
    completed = Event()
    future.add_done_callback(lambda _: completed.set())
    deadline = time.monotonic() + 5
    while not completed.is_set() and time.monotonic() < deadline:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError:
            pass
        else:
            os.close(descriptor)
        completed.wait(0.01)
    assert completed.is_set(), "FIFO worker did not finish after writer release"


@pytest.mark.parametrize(
    "source,fixture,name,sidecar",
    [
        ("claude_code", "claude_code/session1.jsonl", "session.jsonl", None),
        ("codex", "codex/session1/rollout.jsonl", "rollout-test.jsonl", None),
        ("codex", "codex/session1/rollout.jsonl", "rollout-test.jsonl.zst", None),
        ("pi_agent", "pi_agent/session1.jsonl", "session.jsonl", None),
        ("grok", "grok/session1.jsonl", "chat_history.jsonl", None),
        ("grok", "grok/session1.jsonl", "chat_history.jsonl", "summary.json"),
        ("grok", "grok/session1.jsonl", "chat_history.jsonl", "signals.json"),
        ("kimi_code", "kimi_code/session1/agents/main/wire.jsonl", "wire.jsonl", None),
        ("kimi_code", "kimi_code/session1/agents/main/wire.jsonl", "wire.jsonl", "state.json"),
    ],
)
def test_coordinator_preserves_history_when_a_source_becomes_a_pipe(
    tmp_path: Path, source: str, fixture: str, name: str, sidecar: str | None
) -> None:
    root = tmp_path / "transcripts"
    root.mkdir()
    path = root / "session/agents/main" / name
    path.parent.mkdir(parents=True)
    data = (FIXTURES / fixture).read_bytes()
    path.write_bytes(zstd.compress(data) if name.endswith(".zst") else data)
    special = (
        path
        if sidecar is None
        else (path.parents[2] / sidecar if source == "kimi_code" else path.parent / sidecar)
    )
    if sidecar:
        special.write_text('{"cwd":"/project","info":{"cwd":"/project"}}')
    parser = get_parser(Source(source))
    parser.roots = (root,)
    config = AppConfig.load()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(":memory:")
    ensure_schema(connection, embed_dim=config.embedding.dimensions)
    catalog = SourceCatalog(connection, clock=lambda: 100.0)
    logical = path.with_suffix("") if name.endswith(".zst") else path

    def reconcile() -> None:
        item = observe_path(parser, capture_path(parser, logical), conn=connection)
        prepared = prepare_raw_sources((item,), {source: parser})
        commit_prepared_raw_sources(prepared, config, conn=connection)

    try:
        reconcile()
        before = connection.execute("SELECT * FROM messages ORDER BY id").fetchall()
        assert before
        stored_session = connection.execute("SELECT * FROM sessions").fetchall()
        stored_state = connection.execute("SELECT * FROM session_state").fetchall()
        item = catalog.get(source, str(logical))
        assert item is not None
        saved = special.with_name(special.name + ".saved")
        shutil.move(special, saved)
        os.mkfifo(special)
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(prepare_raw_sources, (item,), {source: parser}, full=True)
                prepared = future.result(timeout=2)
                commit_prepared_raw_sources(prepared, config, conn=connection)
                assert connection.execute("SELECT * FROM messages ORDER BY id").fetchall() == before
                assert connection.execute("SELECT * FROM sessions").fetchall() == stored_session
                assert connection.execute("SELECT * FROM session_state").fetchall() == stored_state
                failed = catalog.get(source, str(logical))
                assert failed is not None and failed.last_error is not None
                assert "regular file" in failed.last_error
            finally:
                release_pipe(special, future)
        special.unlink()
        saved.rename(special)
        reconcile()
        restored = catalog.get(source, str(logical))
        assert restored is not None and restored.current
        assert connection.execute("SELECT * FROM messages ORDER BY id").fetchall() == before
    finally:
        connection.close()


def test_publication_rejects_a_pipe_replacing_a_prepared_transcript(tmp_path: Path) -> None:
    from recall.parsers.claude_code import ClaudeCodeParser

    path = tmp_path / "session.jsonl"
    path.write_bytes((FIXTURES / "claude_code/session1.jsonl").read_bytes())
    parser = ClaudeCodeParser(roots=(tmp_path,))
    config = AppConfig.load()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(":memory:")
    ensure_schema(connection, embed_dim=config.embedding.dimensions)
    try:
        item = observe_path(parser, capture_path(parser, path), conn=connection)
        prepared = prepare_raw_sources((item,), {"claude_code": parser})
        path.unlink()
        os.mkfifo(path)
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(
                    commit_prepared_raw_sources, prepared, config, conn=connection
                )
                outcome = future.result(timeout=2)
                assert outcome.changed == 0
                assert connection.execute("SELECT COUNT(*) FROM messages").fetchone() == (0,)
                failed = SourceCatalog(connection, clock=lambda: 100).get("claude_code", str(path))
                assert failed is not None and failed.last_error is not None
            finally:
                release_pipe(path, future)
    finally:
        connection.close()


def test_resumed_preparation_refuses_a_pipe_after_metadata_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import recall.services.coordinator as coordinator
    from recall.parsers.claude_code import ClaudeCodeParser

    path = tmp_path / "session.jsonl"
    data = (FIXTURES / "claude_code/session1.jsonl").read_bytes()
    path.write_bytes(data)
    parser = ClaudeCodeParser(roots=(tmp_path,))
    config = AppConfig.load()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(":memory:")
    ensure_schema(connection, embed_dim=config.embedding.dimensions)
    try:
        item = observe_path(parser, capture_path(parser, path), conn=connection)
        prepared = prepare_raw_sources((item,), {"claude_code": parser})
        commit_prepared_raw_sources(prepared, config, conn=connection)
        before = connection.execute("SELECT * FROM messages ORDER BY id").fetchall()
        with path.open("ab") as stream:
            stream.write(b'{"type":"user","message":{"role":"user","content":"later"}}\n')
        item = observe_path(parser, capture_path(parser, path), conn=connection)
        assert item.normalization_checkpoint is not None
        original = coordinator.capture_path

        def capture_then_replace(*args, **kwargs):
            observed = original(*args, **kwargs)
            path.unlink()
            os.mkfifo(path)
            return observed

        # Force the real filesystem replacement at the boundary between the
        # coordinator's metadata capture and its verified-prefix byte read.
        monkeypatch.setattr(coordinator, "capture_path", capture_then_replace)
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(prepare_raw_sources, (item,), {"claude_code": parser})
                refused = future.result(timeout=2)
                commit_prepared_raw_sources(refused, config, conn=connection)
                assert connection.execute("SELECT * FROM messages ORDER BY id").fetchall() == before
                failed = SourceCatalog(connection, clock=lambda: 100).get("claude_code", str(path))
                assert failed is not None and failed.last_error is not None
                assert "regular file" in failed.last_error
            finally:
                release_pipe(path, future)
    finally:
        connection.close()


@pytest.mark.parametrize("kind", ["pipe", "directory", "socket"])
def test_inventory_reports_non_regular_transcript_entries(tmp_path: Path, kind: str) -> None:
    import socket
    from dataclasses import replace

    from recall.core.config import SourceConfig
    from recall.services.coordinator import prepare_raw_cycle
    from recall.services.reconciler import InventoryResult

    path = tmp_path / "session.jsonl"
    listener = socket.socket(socket.AF_UNIX)
    try:
        if kind == "pipe":
            os.mkfifo(path)
        elif kind == "directory":
            path.mkdir()
        else:
            listener.bind(str(path))
        config = replace(
            AppConfig.load(),
            sources={
                source.value: SourceConfig(
                    roots=(tmp_path,) if source == Source.CLAUDE_CODE else ()
                )
                for source in Source
            },
        )
        results = [
            prepared.event
            for prepared in prepare_raw_cycle(config)
            if isinstance(prepared.event, InventoryResult)
        ]
        assert any("regular file" in failure for result in results for failure in result.failures)
    finally:
        listener.close()
        if kind == "pipe":
            descriptor = os.open(path, os.O_RDWR | os.O_NONBLOCK)
            os.close(descriptor)
