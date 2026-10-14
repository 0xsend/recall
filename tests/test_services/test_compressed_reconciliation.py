"""Canonical compressed sources reconcile without replacing equivalent history."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    ContextConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.models import NormalizationCheckpoint
from recall.core.source_checkpoint import SourceCheckpoint
from recall.core.types import Source
from recall.db.connection import connect
from recall.db.source_files import SourceCatalog
from recall.parsers.codex import CodexParser
from recall.parsers.compressed import zstd
from recall.services.coordinator import commit_prepared_raw_sources, prepare_raw_sources
from recall.services.indexer import _capture_still_matches
from recall.services.live import catalog_progress_for_session, derive_freshness
from recall.services.reconciler import inventory_root


def event(text: str) -> bytes:
    return (
        json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": text}})
        + "\n"
    ).encode()


@pytest.mark.parametrize("target_name", ["capture.jsonl", "capture.jsonl.zst"])
def test_plaintext_alias_retains_its_persisted_identity(tmp_path: Path, target_name: str) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    target = tmp_path / target_name
    raw = event("legacy alias history")
    target.write_bytes(raw)
    alias = root / "rollout.jsonl"
    alias.symlink_to(target)
    parser = CodexParser(roots=(root,))
    config = AppConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data/recall.duckdb",
        lock_path=tmp_path / "data/recall.lock",
        config_path=tmp_path / "config.toml",
        cli=CliConfig(),
        embedding=EmbeddingConfig(context=ContextConfig(mode="off")),
        fts=FtsConfig(fields=()),
        daemon=DaemonConfig(embed=False),
    )
    conn = connect(config)
    try:
        catalog = SourceCatalog(conn, clock=time.time)

        def reconcile() -> None:
            inventory_root(catalog, parser, root, clock=time.time)
            prepared = prepare_raw_sources(tuple(catalog.pending()), {"codex": parser})
            commit_prepared_raw_sources(prepared, config, conn=conn)

        reconcile()
        before = conn.execute("SELECT id, source_path FROM sessions").fetchall()
        messages = conn.execute("SELECT * FROM messages").fetchall()
        assert before[0][1] == str(target)
        alias.unlink()
        physical = (
            target if target.name.endswith(".zst") else target.with_name(target.name + ".zst")
        )
        alias.with_name(alias.name + ".zst").symlink_to(physical)
        physical.write_bytes(zstd.compress(raw))
        if physical != target:
            target.unlink()
        reconcile()
        assert conn.execute("SELECT id, source_path FROM sessions").fetchall() == before
        assert conn.execute("SELECT * FROM messages").fetchall() == messages
        item = catalog.get("codex", str(target))
        assert item is not None and item.current and not item.missing and item.last_error is None
        conn.close()
        conn = connect(config)
        catalog = SourceCatalog(conn, clock=time.time)
        reconcile()
        assert not catalog.pending()
        assert conn.execute("SELECT id, source_path FROM sessions").fetchall() == before
    finally:
        conn.close()


@pytest.mark.parametrize("original_compressed", [False, True])
def test_move_proof_uses_decoded_history_across_compression_layouts(
    tmp_path: Path, original_compressed: bool
) -> None:
    old_root, new_root = tmp_path / "sessions/old", tmp_path / "sessions/new"
    old_root.mkdir(parents=True)
    new_root.mkdir(parents=True)
    first, second = event("moved history"), event("continued history")
    raw = first + second
    old = old_root / ("rollout-move.jsonl.zst" if original_compressed else "rollout-move.jsonl")
    old.write_bytes(zstd.compress(raw) if original_compressed else raw)
    parser = CodexParser(roots=(old_root, new_root))
    config = AppConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data/recall.duckdb",
        lock_path=tmp_path / "data/recall.lock",
        config_path=tmp_path / "config.toml",
        cli=CliConfig(),
        embedding=EmbeddingConfig(context=ContextConfig(mode="off")),
        fts=FtsConfig(fields=()),
        daemon=DaemonConfig(embed=False),
    )
    conn = connect(config)
    try:
        catalog = SourceCatalog(conn, clock=time.time)
        inventory_root(catalog, parser, old_root, clock=time.time)
        prepared = prepare_raw_sources(tuple(catalog.pending()), {"codex": parser})
        assert commit_prepared_raw_sources(prepared, config, conn=conn) == 1
        conn.execute("UPDATE message_state SET context_text='preserved', context_mode='llm-local'")
        old.unlink()
        successor = new_root / "rollout-move.jsonl.zst"
        successor.write_bytes(zstd.compress(first) + zstd.compress(second))
        inventory_root(catalog, parser, new_root, clock=time.time)
        prepared = prepare_raw_sources(tuple(catalog.pending()), {"codex": parser})
        assert commit_prepared_raw_sources(prepared, config, conn=conn) == 1
        assert conn.execute("SELECT source_path FROM sessions").fetchall() == [
            (str(successor.with_suffix("")),)
        ]
        assert conn.execute("SELECT context_text FROM message_state").fetchall() == [
            ("preserved",),
            ("preserved",),
        ]
    finally:
        conn.close()


def test_representation_transitions_restart_and_append_preserve_enrichment(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    path = root / "rollout.jsonl"
    raw = event("compression sentinel") + event("unchanged history")
    path.write_bytes(raw)
    parser = CodexParser(roots=(root,))
    config = AppConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data/recall.duckdb",
        lock_path=tmp_path / "data/recall.lock",
        config_path=tmp_path / "config.toml",
        embedding=EmbeddingConfig(context=ContextConfig(mode="off")),
        fts=FtsConfig(fields=()),
        daemon=DaemonConfig(embed=False),
        cli=CliConfig(),
    )
    conn = connect(config)

    def reconcile() -> int:
        catalog = SourceCatalog(conn, clock=time.time)
        assert inventory_root(catalog, parser, root, clock=time.time) == 1
        prepared = prepare_raw_sources(tuple(catalog.pending()), {"codex": parser})
        committed = commit_prepared_raw_sources(prepared, config, conn=conn)
        item = catalog.get("codex", str(path))
        assert item is not None and item.last_error is None, item
        return committed

    def rows() -> list[tuple[object, ...]]:
        return conn.execute(
            """SELECT m.id, m.idx, s.content, s.context_text, s.context_mode, e.content_embedding
               FROM messages m JOIN message_state s ON s.message_id=m.id
               LEFT JOIN message_embeddings e ON e.message_id=m.id ORDER BY m.idx"""
        ).fetchall()

    try:
        assert reconcile() == 1
        before_ids = [r[0] for r in rows()]
        session_row = conn.execute("SELECT id FROM sessions").fetchone()
        assert session_row is not None
        sid = session_row[0]
        conn.execute("UPDATE message_state SET context_text='retained', context_mode='llm-local'")
        for mid in before_ids:
            conn.execute(
                "INSERT INTO message_embeddings(message_id,content_embedding) VALUES (?, ?)",
                [mid, [0.25] * 384],
            )
        enriched = rows()
        initial_item = SourceCatalog(conn, clock=time.time).get("codex", str(path))
        assert initial_item is not None
        epoch = initial_item.content_epoch
        packed = path.with_name(path.name + ".zst")
        packed.write_bytes(zstd.compress(raw))
        path.unlink()
        assert reconcile() == 0
        assert rows() == enriched
        item = SourceCatalog(conn, clock=time.time).get("codex", str(path))
        assert item is not None and item.current and not item.missing
        assert NormalizationCheckpoint.decode(item.normalization_checkpoint) is None
        assert SourceCheckpoint.decode(item.normalization_checkpoint).decoded is not None
        assert item.content_epoch == epoch
        progress = catalog_progress_for_session(conn, session_id=sid, source_path=str(path))
        assert derive_freshness(
            str(path), indexed_mtime=0, indexed_size=0, catalog=progress
        ).current
        assert reconcile() == 0
        conn.close()
        conn = connect(config)
        assert rows() == enriched
        assert reconcile() == 0
        path.write_bytes(raw + event("appended after decompression"))
        packed.unlink()
        assert reconcile() == 1
        assert rows()[:2] == enriched
        assert len(rows()) == 3
        assert rows()[2][2] == "appended after decompression"
        item = SourceCatalog(conn, clock=time.time).get("codex", str(path))
        assert item is not None and item.current and item.content_epoch == epoch
        assert item.normalization_checkpoint is not None
        with path.open("ab") as output:
            output.write(event("incremental append"))
        assert reconcile() == 1
        assert len(rows()) == 4 and rows()[:2] == enriched
    finally:
        conn.close()


@pytest.mark.parametrize(
    "damage", ["malformed", "truncated", "corrupt", "replacement", "representation"]
)
def test_failed_compressed_capture_preserves_prior_rows_and_stays_noncurrent(
    tmp_path: Path, damage: str
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    path = root / "rollout.jsonl"
    raw = event("prior history") + event("must survive")
    path.write_bytes(raw)
    parser = CodexParser(roots=(root,))
    config = AppConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data/recall.duckdb",
        lock_path=tmp_path / "data/recall.lock",
        config_path=tmp_path / "config.toml",
        embedding=EmbeddingConfig(context=ContextConfig(mode="off")),
        fts=FtsConfig(fields=()),
        daemon=DaemonConfig(embed=False),
        cli=CliConfig(),
    )
    conn = connect(config)
    catalog = SourceCatalog(conn, clock=time.time)
    try:
        inventory_root(catalog, parser, root, clock=time.time)
        initial = prepare_raw_sources(tuple(catalog.pending()), {"codex": parser})
        assert commit_prepared_raw_sources(initial, config, conn=conn) == 1
        before = conn.execute("SELECT * FROM messages ORDER BY idx").fetchall()
        data = zstd.compress(
            raw + (b"{bad}\n" if damage == "malformed" else b""),
            options={zstd.CompressionParameter.checksum_flag: 1},
        )
        if damage == "truncated":
            data = data[:-1]
        elif damage == "corrupt":
            data = data[:-1] + bytes([data[-1] ^ 1])
        packed = path.with_name(path.name + ".zst")
        packed.write_bytes(data)
        path.unlink()
        inventory_root(catalog, parser, root, clock=time.time)
        prepared = prepare_raw_sources(tuple(catalog.pending()), {"codex": parser})
        if damage == "replacement":
            replacement = root / "replacement.tmp"
            replacement.write_bytes(data)
            replacement.replace(packed)
        elif damage == "representation":
            path.write_bytes(raw)
        if damage in {"replacement", "representation"}:
            assert prepared[0].result is not None
            assert not _capture_still_matches(path, prepared[0].result)
        assert commit_prepared_raw_sources(prepared, config, conn=conn) == 0
        assert conn.execute("SELECT * FROM messages ORDER BY idx").fetchall() == before
        item = catalog.get(Source.CODEX.value, str(path))
        assert item is not None and not item.current and item.last_error
    finally:
        conn.close()


def test_watch_events_resolve_canonical_sources_and_ignore_temporary_files(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from recall.services.watcher import DebouncedIndexQueue, SessionFileHandler

    path = tmp_path / "rollout.jsonl"
    path.write_bytes(event("prior history"))
    packed = path.with_name(path.name + ".zst")
    packed.write_bytes(zstd.compress(path.read_bytes()))
    path.unlink()
    queue = DebouncedIndexQueue(debounce=0)
    handler = SessionFileHandler(queue, [CodexParser(roots=(tmp_path,))])
    handler.on_created(SimpleNamespace(is_directory=False, src_path=str(packed) + ".tmp"))
    assert queue.ready() == []
    handler.on_moved(
        SimpleNamespace(is_directory=False, src_path=str(packed) + ".tmp", dest_path=str(packed))
    )
    assert queue.ready() == [str(path)]
    path.write_bytes(event("preferred plain"))
    handler.on_deleted(SimpleNamespace(is_directory=False, src_path=str(packed)))
    assert queue.ready() == [str(path)]


def test_linux_delete_self_file_event_survives_directory_classification(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from recall.services.watcher import DebouncedIndexQueue, SessionFileHandler

    path = tmp_path / "rollout.jsonl.zst"
    path.write_bytes(zstd.compress(event("replacement history")))
    queue = DebouncedIndexQueue(debounce=0)
    handler = SessionFileHandler(queue, [CodexParser(roots=(tmp_path,))])
    handler.on_deleted(SimpleNamespace(is_directory=True, event_type="deleted", src_path=str(path)))
    assert queue.ready() == [str(path.with_suffix(""))]


def test_compressed_mutation_during_verification_rejects_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import BinaryIO

    path = tmp_path / "rollout.jsonl.zst"
    path.write_bytes(zstd.compress(event("captured history")))
    result = CodexParser().parse(path)
    original_open = Path.open

    class MutatingReader:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            self.handle.close()

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int) -> bytes:
            block = self.handle.read(size)
            with original_open(path, "ab") as append:
                append.write(b"x")
            return block

    def changing_open(candidate: Path, mode: str = "r", **kwargs: object):
        assert candidate == path and mode == "rb"
        return MutatingReader(original_open(candidate, "rb"))

    monkeypatch.setattr(Path, "open", changing_open)
    assert not _capture_still_matches(path.with_suffix(""), result)


def test_file_subscriptions_follow_physical_representation_with_canonical_queue(
    tmp_path: Path,
) -> None:
    import threading

    from recall.services.live_session_set import LiveSessionSet
    from recall.services.watcher import (
        DebouncedIndexQueue,
        SessionFileHandler,
        _reconcile_observer_subscriptions,
    )

    path = tmp_path / "rollout.jsonl"
    path.write_bytes(event("watch history"))
    parser = CodexParser(roots=(tmp_path,))
    live_set = LiveSessionSet(max_subscriptions=8, idle_threshold=300, is_macos=False)
    live_set.promote(path, path.stat().st_mtime, time.monotonic(), bump_event=False)
    queue = DebouncedIndexQueue(debounce=0)
    handler = SessionFileHandler(queue, [parser], live_set=live_set)
    scheduled: dict[str, object] = {}

    class Observer:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            pass

        def schedule(self, handler: object, path: str, recursive: bool = False) -> object:
            self.paths.append(path)
            return path

        def unschedule(self, watch: object) -> None:
            self.paths.remove(str(watch))

    observer = Observer()

    def reconcile() -> None:
        _reconcile_observer_subscriptions(
            observer=observer,
            handler=handler,
            live_set=live_set,
            scheduled_watches=scheduled,
            poll_fallback=set(),
            reconcile_lock=threading.Lock(),
        )

    reconcile()
    assert observer.paths == [str(path)]
    packed = path.with_name(path.name + ".zst")
    packed.write_bytes(zstd.compress(path.read_bytes()))
    path.unlink()
    reconcile()
    assert observer.paths == [str(packed)]
    path.write_bytes(event("plain restored"))
    reconcile()
    assert observer.paths == [str(path)]


def test_watch_source_io_failure_does_not_stop_subsequent_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.services.watcher import DebouncedIndexQueue, SessionFileHandler
    from watchdog.events import FileCreatedEvent

    broken = tmp_path / "rollout-broken.jsonl"
    healthy = tmp_path / "rollout-healthy.jsonl"
    broken.write_bytes(event("unreadable"))
    healthy.write_bytes(event("healthy"))
    open_path = Path.open

    def unreadable(path: Path, *args: object, **kwargs: object):
        if path == broken:
            raise PermissionError("source temporarily unreadable")
        return open_path(path, "rb")

    monkeypatch.setattr(Path, "open", unreadable)
    queue = DebouncedIndexQueue(debounce=0)
    handler = SessionFileHandler(queue, [CodexParser(roots=(tmp_path,))])
    handler.dispatch(FileCreatedEvent(str(broken)))
    handler.dispatch(FileCreatedEvent(str(healthy)))
    assert queue.ready() == [str(broken), str(healthy)]
