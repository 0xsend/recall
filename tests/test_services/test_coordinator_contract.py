from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.db.schema import ensure_schema
from recall.db.source_files import SourceCatalog, SourceSignature, source_key
from recall.parsers.codex import CodexParser
from recall.services.coordinator import (
    capture_path,
    commit_prepared_raw_sources,
    index_only_sessions,
    observe_path,
    prepare_raw_cycle,
    prepare_raw_sources,
    reconciliation_status,
)
from recall.services.reconciler import InventoryBatch


def test_inventory_returns_a_lazy_bounded_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".codex/sessions"
    root.mkdir(parents=True)
    for index in range(300):
        (root / f"rollout-{index}.jsonl").write_text('{"type":"session_meta","payload":{}}\n')
    stream = prepare_raw_cycle(AppConfig.load())
    assert isinstance(stream, Iterator)
    counts = [len(item.event.files) for item in stream if isinstance(item.event, InventoryBatch)]
    assert sum(counts) == 300
    assert max(counts) <= 128


def test_reconciliation_status_ages_only_current_pending_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / ".codex/sessions"
    root.mkdir(parents=True)
    config = AppConfig.load()
    conn = duckdb.connect(":memory:")
    ensure_schema(conn, embed_dim=config.embedding.dimensions)
    now = [100.0]
    catalog = SourceCatalog(conn, clock=lambda: now[0])
    signature = SourceSignature(dev=1, inode=2, ctime_ns=3, mtime_ns=4, size=5)
    missing_path = str(root / "missing.jsonl")
    current_path = str(root / "current.jsonl")

    try:
        catalog.observe("codex", str(root), missing_path, signature)
        catalog.mark_missing([source_key("codex", missing_path)])
        now[0] = 190.0
        generation = catalog.observe("codex", str(root), current_path, signature)
        monkeypatch.setattr("recall.services.coordinator.time.time", lambda: 200.0)

        pending = reconciliation_status(config, conn=conn)
        pending_coverage = cast(list[dict[str, object]], pending["coverage"])
        coverage = next(row for row in pending_coverage if row["source"] == "codex")
        assert pending["pending"] == 1
        assert coverage["pending"] == 1
        assert coverage["missing"] == 1
        assert coverage["oldest_pending_age"] == 10.0

        assert catalog.acknowledge("codex", current_path, generation, 5, "a" * 64)
        drained = reconciliation_status(config, conn=conn)
        drained_coverage = cast(list[dict[str, object]], drained["coverage"])
        coverage = next(row for row in drained_coverage if row["source"] == "codex")
        assert drained["pending"] == 0
        assert coverage["pending"] == 0
        assert coverage["oldest_pending_age"] is None
    finally:
        conn.close()


def test_index_only_sessions_counts_this_hosts_sessions_without_a_surviving_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("recall.services.coordinator.default_session_host", lambda: "laptop")
    config = AppConfig.load()
    conn = duckdb.connect(":memory:")
    ensure_schema(conn, embed_dim=config.embedding.dimensions)
    catalog = SourceCatalog(conn, clock=lambda: 100.0)
    signature = SourceSignature(dev=1, inode=2, ctime_ns=3, mtime_ns=4, size=5)
    root = str(tmp_path / "root")

    def add_session(source: str, path: str, identity: str, host: str) -> None:
        session_id = f"{identity}-{len(path)}-{host}"
        conn.execute(
            "INSERT INTO sessions (id, source, source_path, source_session_id) VALUES (?, ?, ?, ?)",
            [session_id, source, path, identity],
        )
        conn.execute(
            "INSERT INTO session_state (session_id, host, file_mtime, file_size) "
            "VALUES (?, ?, 0, 0)",
            [session_id, host],
        )

    def catalog_path(source: str, path: str, *, present: bool) -> None:
        catalog.observe(source, root, path, signature)
        if not present:
            catalog.mark_missing([source_key(source, path)])

    try:
        add_session("claude_code", f"{root}/present.jsonl", "present", "laptop")
        catalog_path("claude_code", f"{root}/present.jsonl", present=True)
        add_session("claude_code", f"{root}/deleted.jsonl", "deleted", "laptop")
        catalog_path("claude_code", f"{root}/deleted.jsonl", present=False)
        add_session("claude_code", f"{root}/uncataloged.jsonl", "uncataloged", "laptop")
        # A renamed home directory: the old path is gone, the same session lives on.
        add_session("claude_code", f"{root}/old/moved.jsonl", "moved", "laptop")
        catalog_path("claude_code", f"{root}/old/moved.jsonl", present=False)
        add_session("claude_code", f"{root}/new-home/moved.jsonl", "moved", "laptop")
        catalog_path("claude_code", f"{root}/new-home/moved.jsonl", present=True)
        add_session("grok", f"{root}/unattributed.jsonl", "unattributed", "local")
        add_session("grok", f"{root}/imported.jsonl", "imported", "devbox")

        assert index_only_sessions(conn) == {"claude_code": 2, "grok": 1}
        status = reconciliation_status(config, conn=conn)
        assert status["index_only_sessions"] == {"claude_code": 2, "grok": 1}
    finally:
        conn.close()


def test_reconciliation_preserves_append_epoch_and_hashes_only_committed_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "off")
    config = AppConfig.load()
    path = tmp_path / "rollout-history.jsonl"
    parser = CodexParser(roots=(tmp_path,))

    def message(text: str) -> bytes:
        return (
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": text,
                    },
                }
            )
            + "\n"
        ).encode()

    conn = duckdb.connect(":memory:")
    ensure_schema(conn, embed_dim=config.embedding.dimensions)
    catalog = SourceCatalog(conn, clock=lambda: 100.0)

    def reconcile() -> None:
        item = observe_path(parser, capture_path(parser, path), conn=conn)
        prepared = prepare_raw_sources((item,), {parser.source.value: parser})
        commit_prepared_raw_sources(prepared, config, conn=conn)

    try:
        path.write_bytes(message("first"))
        reconcile()
        initial = catalog.get("codex", str(path))
        assert initial is not None and initial.content_epoch == 0
        complete = message("first") + message("second")
        path.write_bytes(complete + b'{"type":')
        reconcile()
        appended = catalog.get("codex", str(path))
        assert appended is not None and appended.content_epoch == 0
        assert appended.committed_offset == len(complete)
        assert appended.committed_prefix_sha256 == hashlib.sha256(complete).hexdigest()
        assert appended.last_error is not None
        path.write_bytes(message("edited") + message("second"))
        reconcile()
        rewritten = catalog.get("codex", str(path))
        assert rewritten is not None and rewritten.content_epoch == 1
        assert rewritten.last_error is None
    finally:
        conn.close()
