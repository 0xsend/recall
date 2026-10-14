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
        assert item.normalization_checkpoint is None and item.content_epoch == epoch
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
