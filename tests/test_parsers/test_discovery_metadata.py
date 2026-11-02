"""Metadata IO floors for periodic inventories and live discovery."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest
from recall.core.types import Source
from recall.parsers import get_parser
from recall.parsers.codex import CodexParser
from recall.parsers.discovery import DiscoveryScope
from recall.services.reconciler import InventoryBatch, InventoryResult, iter_inventory_batches


def test_inventory_reuses_directory_entry_metadata(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "rollout-history.jsonl"
    path.write_text("{}\n")
    parser = CodexParser(roots=(tmp_path,))
    stat = Path.stat
    reads = []

    def count(path: Path, **kwargs):
        if path.name == "rollout-history.jsonl":
            reads.append(path)
        return stat(path, **kwargs)

    monkeypatch.setattr(Path, "stat", count)
    batches = list(iter_inventory_batches(parser, tmp_path, clock=lambda: 1.0))
    captures = [
        item for batch in batches if isinstance(batch, InventoryBatch) for item in batch.files
    ]
    assert [(item.source_path, item.signature.size) for item in captures] == [(str(path), 3)]
    # The directory entry supplies the first observation; one fresh stat after
    # sidecar capture proves it stayed stable. Plain Codex selection needs no IO.
    assert len(reads) <= 1


def test_live_sweep_reuses_metadata_without_restatting_history(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "rollout-history.jsonl"
    path.write_text("{}\n")
    os.utime(path, (1, 1))
    parser = CodexParser(roots=(tmp_path,))
    stat = Path.stat
    reads = []

    def count(path: Path, **kwargs):
        if path.name == "rollout-history.jsonl":
            reads.append(path)
        return stat(path, **kwargs)

    monkeypatch.setattr(Path, "stat", count)
    assert parser.live_candidates(now=datetime.fromtimestamp(1000), idle_threshold=100) == []
    assert reads == []


def test_plain_representation_appearing_during_linked_capture_is_not_acknowledged(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "outside" / "rollout-linked.jsonl"
    target.parent.mkdir()
    compressed = target.with_suffix(".jsonl.zst")
    compressed.write_bytes(b"compressed representation")
    (root / "rollout-entrance.jsonl").symlink_to(compressed)

    class Parser(CodexParser):
        def sidecar_paths(self, path: Path) -> list[Path]:
            # Force plain-first selection to change outside the directory scope
            # while metadata is captured, without modifying the link entrance.
            target.write_bytes(b"authoritative plain representation")
            return []

    events = list(iter_inventory_batches(Parser(roots=(root,)), root, clock=lambda: 1.0))
    assert not [event for event in events if isinstance(event, InventoryBatch)]
    result = next(event for event in events if isinstance(event, InventoryResult))
    assert not result.complete
    assert any("source changed during capture" in failure for failure in result.failures)


@pytest.mark.parametrize("operation", ["append", "rewrite", "create", "move", "delete", "link"])
@pytest.mark.parametrize("source", list(Source))
def test_periodic_inventory_finds_changes_without_notifications(
    tmp_path: Path, operation: str, source: Source
):
    root = tmp_path / "root"
    root.mkdir()
    name = {
        Source.CODEX: "rollout-history.jsonl",
        Source.GROK: "chat_history.jsonl",
        Source.KIMI_CODE: "wire.jsonl",
    }.get(source, "history.jsonl")
    path = root / name
    path.write_text("{}\n")
    parser = get_parser(source)
    parser.roots = (root,)

    def inventory():
        batches = list(iter_inventory_batches(parser, root, clock=lambda: 1.0))
        return {
            item.source_path: item.signature.size
            for batch in batches
            if isinstance(batch, InventoryBatch)
            for item in batch.files
        }

    assert inventory() == {str(path): 3}
    directory_mtime = root.stat().st_mtime_ns
    if operation == "append":
        with path.open("a") as stream:
            stream.write("{}\n")
        assert root.stat().st_mtime_ns == directory_mtime
        expected = {str(path): 6}
    elif operation == "rewrite":
        path.write_text("[]\n[]\n")
        assert root.stat().st_mtime_ns == directory_mtime
        expected = {str(path): 6}
    elif operation == "create":
        (root / "new").mkdir()
        created = root / "new" / name
        created.write_text("[]\n")
        expected = {str(path): 3, str(created): 3}
    elif operation == "move":
        (root / "moved").mkdir()
        moved = root / "moved" / name
        path.rename(moved)
        expected = {str(moved): 3}
    elif operation == "delete":
        path.unlink()
        expected = {}
    else:
        target = tmp_path / "outside"
        target.mkdir()
        linked = target / name
        linked.write_text("[]\n")
        (root / "delegated").symlink_to(target, target_is_directory=True)
        expected = {str(path): 3, str(linked): 3}
    assert inventory() == expected
    scope = DiscoveryScope.build(parser)
    assert scope.revalidate()
