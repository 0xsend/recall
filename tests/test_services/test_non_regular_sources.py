"""Real coordinator captures refuse special files without disturbing stored history."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

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

from tests.fifo_process import run_fifo_scenario

FIXTURES = Path(__file__).parents[1] / "fixtures"


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
    run_fifo_scenario(
        _coordinator_preserves_history_when_a_source_becomes_a_pipe,
        tmp_path,
        source,
        fixture,
        name,
        sidecar,
    )


def _coordinator_preserves_history_when_a_source_becomes_a_pipe(
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
        try:
            prepared = prepare_raw_sources((item,), {source: parser}, full=True)
            commit_prepared_raw_sources(prepared, config, conn=connection)
            assert connection.execute("SELECT * FROM messages ORDER BY id").fetchall() == before
            assert connection.execute("SELECT * FROM sessions").fetchall() == stored_session
            assert connection.execute("SELECT * FROM session_state").fetchall() == stored_state
            failed = catalog.get(source, str(logical))
            assert failed is not None and failed.last_error is not None
            assert "regular file" in failed.last_error
        finally:
            descriptor = os.open(special, os.O_RDWR | os.O_NONBLOCK)
            os.close(descriptor)
        special.unlink()
        saved.rename(special)
        reconcile()
        restored = catalog.get(source, str(logical))
        assert restored is not None and restored.current
        assert connection.execute("SELECT * FROM messages ORDER BY id").fetchall() == before
    finally:
        connection.close()


def test_publication_rejects_a_pipe_replacing_a_prepared_transcript(tmp_path: Path) -> None:
    run_fifo_scenario(_publication_rejects_a_pipe_replacing_a_prepared_transcript, tmp_path)


def _publication_rejects_a_pipe_replacing_a_prepared_transcript(tmp_path: Path) -> None:
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
        try:
            outcome = commit_prepared_raw_sources(prepared, config, conn=connection)
            assert outcome.changed == 0
            assert connection.execute("SELECT COUNT(*) FROM messages").fetchone() == (0,)
            failed = SourceCatalog(connection, clock=lambda: 100).get("claude_code", str(path))
            assert failed is not None and failed.last_error is not None
        finally:
            descriptor = os.open(path, os.O_RDWR | os.O_NONBLOCK)
            os.close(descriptor)
    finally:
        connection.close()


def test_resumed_preparation_refuses_a_pipe_after_metadata_capture(tmp_path: Path) -> None:
    run_fifo_scenario(_resumed_preparation_refuses_a_pipe_after_metadata_capture, tmp_path)


def _resumed_preparation_refuses_a_pipe_after_metadata_capture(tmp_path: Path) -> None:
    monkeypatch = pytest.MonkeyPatch()
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
        try:
            refused = prepare_raw_sources((item,), {"claude_code": parser})
            commit_prepared_raw_sources(refused, config, conn=connection)
            assert connection.execute("SELECT * FROM messages ORDER BY id").fetchall() == before
            failed = SourceCatalog(connection, clock=lambda: 100).get("claude_code", str(path))
            assert failed is not None and failed.last_error is not None
            assert "regular file" in failed.last_error
        finally:
            descriptor = os.open(path, os.O_RDWR | os.O_NONBLOCK)
            os.close(descriptor)
    finally:
        connection.close()


@pytest.mark.parametrize("kind", ["pipe", "directory", "socket"])
def test_inventory_ignores_never_seen_non_regular_entries(
    tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
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
            monkeypatch.chdir(path.parent)
            listener.bind(path.name)
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
        assert results
        assert all(result.complete and not result.failures for result in results)
    finally:
        listener.close()
        if kind == "pipe":
            descriptor = os.open(path, os.O_RDWR | os.O_NONBLOCK)
            os.close(descriptor)


@pytest.mark.parametrize(
    "source,sidecar",
    [("kimi_code", "state.json"), ("grok", "summary.json"), ("grok", "signals.json")],
)
@pytest.mark.parametrize("stage", ["admission", "read", "publication"])
def test_legacy_index_preserves_state_when_sidecar_is_refused(
    tmp_path: Path, source: str, sidecar: str, stage: str
) -> None:
    run_fifo_scenario(
        _legacy_index_preserves_state_when_sidecar_is_refused, tmp_path, source, sidecar, stage
    )


def _legacy_index_preserves_state_when_sidecar_is_refused(
    tmp_path: Path, source: str, sidecar: str, stage: str
) -> None:
    from dataclasses import replace

    from recall.core.config import SourceConfig
    from recall.services.indexer import index_sessions

    root = tmp_path / "roots"
    path = (
        root
        / "session/agents/main"
        / ("wire.jsonl" if source == "kimi_code" else "chat_history.jsonl")
    )
    path.parent.mkdir(parents=True)
    fixture = (
        "kimi_code/session1/agents/main/wire.jsonl"
        if source == "kimi_code"
        else "grok/session1.jsonl"
    )
    path.write_bytes((FIXTURES / fixture).read_bytes())
    sibling = path.parents[2] / sidecar if source == "kimi_code" else path.parent / sidecar
    sibling.write_text(
        '{"workDir":"/project","info":{"cwd":"/project"},"sessionDurationSeconds":123}'
    )
    config = replace(
        AppConfig.load(),
        sources={s.value: SourceConfig(roots=(root,) if s.value == source else ()) for s in Source},
    )
    config.data_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(":memory:")
    ensure_schema(connection, embed_dim=config.embedding.dimensions)

    def index():
        return index_sessions(
            source=Source(source),
            full=True,
            recreate=False,
            verbose=False,
            workers=1,
            config=config,
            conn=connection,
        )

    try:
        assert index().indexed == 1
        before = connection.execute("SELECT * FROM session_state").fetchall()
        sessions = connection.execute("SELECT * FROM sessions").fetchall()

        def replace_sibling():
            sibling.unlink()
            os.mkfifo(sibling)

        patch = pytest.MonkeyPatch()
        if stage == "admission":
            replace_sibling()
        else:
            parser_type = type(get_parser(Source(source)))
            original = parser_type.parse

            def replace_during_parse(self, *args, **kwargs):
                if stage == "read":
                    replace_sibling()
                result = original(self, *args, **kwargs)
                if stage == "publication":
                    replace_sibling()
                return result

            patch.setattr(parser_type, "parse", replace_during_parse)
        outcome = index()
        assert outcome.indexed == 0
        assert outcome.failed == 1
        patch.undo()
        assert connection.execute("SELECT * FROM session_state").fetchall() == before
        assert connection.execute("SELECT * FROM sessions").fetchall() == sessions
    finally:
        descriptor = os.open(sibling, os.O_RDWR | os.O_NONBLOCK)
        os.close(descriptor)
        connection.close()


@pytest.mark.parametrize(
    "source,sidecar",
    [("kimi_code", "state.json"), ("grok", "summary.json"), ("grok", "signals.json")],
)
@pytest.mark.parametrize("observation", ["inventory", "notification"])
@pytest.mark.parametrize("previously_missing", [False, True], ids=["new", "returned"])
def test_new_regular_source_with_refused_sidecar_is_catalogued(
    tmp_path: Path, source: str, sidecar: str, observation: str, previously_missing: bool
) -> None:
    run_fifo_scenario(
        _new_regular_source_with_refused_sidecar,
        tmp_path,
        source,
        sidecar,
        observation,
        previously_missing,
    )


def _new_regular_source_with_refused_sidecar(
    tmp_path: Path, source: str, sidecar: str, observation: str, previously_missing: bool
) -> None:
    from dataclasses import replace

    from recall.core.config import SourceConfig
    from recall.services.coordinator import (
        capture_watch_observations,
        persist_prepared_inventory,
        prepare_raw_cycle,
        reconciliation_status,
    )
    from recall.services.reconciler import persist_inventory_batch

    root = tmp_path / "roots"
    path = (
        root
        / "session/agents/main"
        / ("wire.jsonl" if source == "kimi_code" else "chat_history.jsonl")
    )
    path.parent.mkdir(parents=True)
    fixture = (
        "kimi_code/session1/agents/main/wire.jsonl"
        if source == "kimi_code"
        else "grok/session1.jsonl"
    )
    path.write_bytes((FIXTURES / fixture).read_bytes())
    sibling = path.parents[2] / sidecar if source == "kimi_code" else path.parent / sidecar
    os.mkfifo(sibling)
    config = replace(
        AppConfig.load(),
        sources={s.value: SourceConfig(roots=(root,) if s.value == source else ()) for s in Source},
    )
    config.data_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(":memory:")
    ensure_schema(connection, embed_dim=config.embedding.dimensions)
    catalog = SourceCatalog(connection, clock=lambda: 100.0)

    def inventory():
        generation = None
        for prepared in prepare_raw_cycle(config):
            generation, _ = persist_prepared_inventory(prepared, generation, conn=connection)

    if previously_missing:
        from recall.db.source_files import source_key
        from recall.services.reconciler import stat_signature

        catalog.observe(source, str(root), str(path), stat_signature(path))
        catalog.mark_missing([source_key(source, str(path))])
    try:
        if observation == "inventory":
            inventory()
        else:
            captured = capture_watch_observations(config, (str(path),))
            persist_inventory_batch(catalog, captured.inventory, notification=True)
        refused = catalog.get(source, str(path))
        assert refused is not None
        assert not refused.missing and not refused.current
        assert refused.last_error is not None and "regular file" in refused.last_error
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone() == (0,)
        inventory()
        status = reconciliation_status(config, conn=connection)
        assert status["catalog_scan_complete"]
        assert status["pending"] == 1 and not status["raw_indexing_ready"]
        sibling.unlink()
        sibling.write_text('{"workDir":"/project","info":{"cwd":"/project"}}')
        inventory()
        item = catalog.get(source, str(path))
        assert item is not None
        parser = get_parser(Source(source), config.sources)
        prepared = prepare_raw_sources((item,), {source: parser})
        commit_prepared_raw_sources(prepared, config, conn=connection)
        restored = catalog.get(source, str(path))
        assert restored is not None and restored.current
        count = connection.execute("SELECT COUNT(*) FROM messages").fetchone()
        assert count is not None and count[0] > 0
    finally:
        descriptor = os.open(sibling, os.O_RDWR | os.O_NONBLOCK)
        os.close(descriptor)
        connection.close()
