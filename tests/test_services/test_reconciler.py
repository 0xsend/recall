from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import duckdb
from recall.services import reconciler
from recall.services.reconciler import (
    Coalescer,
    FairScheduler,
    inventory_root,
    inventory_root_scopes,
)


def test_catalog_persists_pending_and_does_not_ack_newer_generation() -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        first = SourceSignature(dev=1, inode=2, ctime_ns=3, mtime_ns=4, size=5)
        second = SourceSignature(dev=1, inode=2, ctime_ns=3, mtime_ns=6, size=7)
        first_generation = catalog.observe("codex", "/root", "/root/a.jsonl", first)
        second_generation = catalog.observe("codex", "/root", "/root/a.jsonl", second)

        assert second_generation == first_generation + 1
        assert catalog.acknowledge("codex", "/root/a.jsonl", first_generation, 5, "a" * 64)
        pending = catalog.pending(limit=1)
        assert len(pending) == 1
        assert pending[0].desired_generation == second_generation
        assert pending[0].committed_generation == first_generation
        assert pending[0].first_pending_at == 100.0
    finally:
        conn.close()


def test_scheduler_reserves_oldest_work_under_active_pressure() -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    now = [100.0]
    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: now[0])
        signature = SourceSignature(dev=1, inode=2, ctime_ns=3, mtime_ns=4, size=5)
        catalog.observe("codex", "/root", "/root/old.jsonl", signature)
        for index in range(8):
            now[0] += 1
            catalog.observe("codex", "/root", f"/root/live-{index}.jsonl", signature)

        scheduled = FairScheduler(clock=lambda: now[0]).select(
            catalog, {f"/root/live-{index}.jsonl" for index in range(8)}, limit=7
        )
        assert [item.source.source_path for item in scheduled][-1] == "/root/old.jsonl"
        assert [item.lane for item in scheduled] == [
            "active",
            "active",
            "active",
            "active",
            "recent",
            "recent",
            "oldest",
        ]
    finally:
        conn.close()


def test_failed_work_retries_after_bounded_backoff_and_survives_restart(tmp_path) -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    database = tmp_path / "catalog.duckdb"
    now = [100.0]
    signature = SourceSignature(dev=1, inode=2, ctime_ns=3, mtime_ns=4, size=5)
    conn = duckdb.connect(str(database))
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: now[0])
        catalog.observe("codex", "/root", "/root/a.jsonl", signature)
        catalog.fail("codex", "/root/a.jsonl", "parse failed", {"record": 4})
        assert catalog.pending() == []
    finally:
        conn.close()

    now[0] = 102.0
    conn = duckdb.connect(str(database))
    try:
        catalog = SourceCatalog(conn, clock=lambda: now[0])
        pending = catalog.pending()
        assert [(item.source_path, item.retry_count) for item in pending] == [("/root/a.jsonl", 1)]
    finally:
        conn.close()


def test_acknowledgment_rolls_back_with_content_transaction() -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        generation = catalog.observe(
            "codex", "/root", "/root/a.jsonl", SourceSignature(1, 2, 3, 4, 5)
        )
        conn.execute("BEGIN")
        assert catalog.acknowledge("codex", "/root/a.jsonl", generation, 5, "b" * 64)
        conn.execute("ROLLBACK")
        assert catalog.pending()[0].committed_generation == 0
    finally:
        conn.close()


def test_status_page_is_bounded_and_uses_a_stable_cursor() -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        signature = SourceSignature(1, 2, 3, 4, 5)
        catalog.observe("codex", "/root", "/root/a.jsonl", signature)
        catalog.observe("codex", "/root", "/root/b.jsonl", signature)
        first = catalog.status_page(limit=1)
        second = catalog.status_page(limit=1, cursor=first.next_cursor)
        assert [item.source_path for item in first.rows] == ["/root/a.jsonl"]
        assert first.next_cursor == "codex\x1f/root/a.jsonl"
        assert [item.source_path for item in second.rows] == ["/root/b.jsonl"]
        assert second.next_cursor is None
    finally:
        conn.close()


def test_scheduler_is_work_conserving_for_each_single_lane() -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        signature = SourceSignature(1, 2, 3, 4, 5)
        catalog.observe("codex", "/root", "/root/history.jsonl", signature)
        selected = FairScheduler(clock=lambda: 100.0).select(catalog, set(), limit=1)
        assert [item.source.source_path for item in selected] == ["/root/history.jsonl"]
        assert FairScheduler(clock=lambda: 100.0).select(catalog, {"/root/history.jsonl"}, limit=1)
    finally:
        conn.close()


def test_active_path_after_deep_history_is_selected_without_losing_oldest() -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    now = [0.0]
    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: now[0])
        signature = SourceSignature(1, 2, 3, 4, 5)
        for index in range(1000):
            now[0] += 1
            catalog.observe("codex", "/root", f"/root/old-{index:04}.jsonl", signature)
        catalog.observe("codex", "/root", "/root/active.jsonl", signature)
        scheduled = FairScheduler(clock=lambda: now[0]).select(
            catalog, {"/root/active.jsonl"}, limit=7
        )
        paths = [item.source.source_path for item in scheduled]
        assert paths[0] == "/root/active.jsonl"
        assert "/root/old-0000.jsonl" in paths
    finally:
        conn.close()


def test_append_does_not_change_epoch_but_semantic_ack_does() -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        first = SourceSignature(1, 2, 3, 4, 5)
        append = SourceSignature(1, 2, 3, 5, 8)
        generation = catalog.observe("codex", "/root", "/root/a.jsonl", first)
        catalog.acknowledge("codex", "/root/a.jsonl", generation, 5, "a" * 64)
        generation = catalog.observe("codex", "/root", "/root/a.jsonl", append)
        assert catalog.pending()[0].content_epoch == 0
        catalog.acknowledge(
            "codex", "/root/a.jsonl", generation, 8, "b" * 64, semantic_rewrite=True
        )
        assert catalog.status_page().rows[0].content_epoch == 1
    finally:
        conn.close()


def test_coalescer_obeys_quiet_and_maximum_windows_under_steady_marks() -> None:
    now = [0.0]
    coalescer = Coalescer(clock=lambda: now[0])
    assert coalescer.mark("/root/a.jsonl")
    for tick in (4.0, 8.0):
        now[0] = tick
        assert coalescer.mark("/root/a.jsonl")
        assert not coalescer.ready("/root/a.jsonl")
    now[0] = 10.0
    assert coalescer.ready("/root/a.jsonl")
    coalescer.consume("/root/a.jsonl")
    assert coalescer.ready("/root/a.jsonl")


def test_two_sources_account_for_the_same_canonical_path_independently() -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 1.0)
        signature = SourceSignature(1, 2, 3, 4, 5)
        catalog.observe("codex", "/root", "/root/shared.jsonl", signature)
        catalog.observe("grok", "/root", "/root/shared.jsonl", signature)
        rows = catalog.status_page().rows
        assert [(row.source, row.source_path) for row in rows] == [
            ("codex", "/root/shared.jsonl"),
            ("grok", "/root/shared.jsonl"),
        ]
    finally:
        conn.close()


def test_inventory_streams_old_import_and_keeps_prior_rows_on_incomplete_scan(tmp_path) -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog

    class Parser:
        source = "codex"
        file_pattern = "*.jsonl"

        @staticmethod
        def sidecar_paths(path: Path) -> list[Path]:
            return [path.with_suffix(".meta")]

    root = tmp_path / "configured"
    root.mkdir()
    old = root / "old.jsonl"
    old.write_text("{}\n")
    old.with_suffix(".meta").write_text("metadata")
    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        assert inventory_root(catalog, Parser(), root, clock=lambda: 100.0, chunk_size=1) == 1
        old.unlink()
        old.with_suffix(".meta").unlink()
        root.rmdir()

        class BrokenWalk:
            source = "codex"
            file_pattern = "*.jsonl"

            @staticmethod
            def sidecar_paths(path: Path) -> list[Path]:
                return []

        # A missing configured root is an explicit incomplete scan and must not erase history.
        assert inventory_root(catalog, BrokenWalk(), root, clock=lambda: 101.0) == 0
        row = catalog.status_page().rows[0]
        assert not row.missing
        scan = conn.execute(
            "SELECT scan_complete, failure_count FROM reconciliation_roots"
        ).fetchone()
        assert scan == (False, 1)
    finally:
        conn.close()


class _JsonlParser:
    source = "codex"
    file_pattern = "*.jsonl"

    @staticmethod
    def sidecar_paths(path: Path) -> list[Path]:
        return []


def _catalog_rows(conn: duckdb.DuckDBPyConnection) -> list[tuple[object, ...]]:
    return conn.execute("SELECT * FROM source_files ORDER BY source_key").fetchall()


def test_walk_of_unchanged_tree_leaves_catalog_rows_untouched(tmp_path) -> None:
    """REQ-RECON-019: the daemon walks every root on a timer, so a walk that finds
    nothing new must not rewrite the catalog."""
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog

    root = tmp_path / "sessions"
    root.mkdir()
    for index in range(300):
        (root / f"session-{index:03}.jsonl").write_text("{}\n", encoding="utf-8")
    conn = duckdb.connect(str(tmp_path / "recall.duckdb"))
    try:
        ensure_schema(conn, embed_dim=8)
        now = [100.0]
        catalog = SourceCatalog(conn, clock=lambda: now[0])
        assert inventory_root(catalog, _JsonlParser(), root, clock=lambda: now[0]) == 300
        before = _catalog_rows(conn)

        now[0] = 200.0
        assert inventory_root(catalog, _JsonlParser(), root, clock=lambda: now[0]) == 300

        assert _catalog_rows(conn) == before
    finally:
        conn.close()


def test_complete_walk_marks_removed_files_missing_and_restores_returning_ones(tmp_path) -> None:
    """REQ-RECON-001: absence is established by a complete walk, and a file that
    returns is present again."""
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog

    root = tmp_path / "sessions"
    root.mkdir()
    for name in ("kept", "removed"):
        (root / f"{name}.jsonl").write_text("{}\n", encoding="utf-8")
    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)

        def walk() -> dict[str, bool]:
            inventory_root(catalog, _JsonlParser(), root, clock=lambda: 100.0)
            return {Path(row.source_path).stem: row.missing for row in catalog.status_page().rows}

        assert walk() == {"kept": False, "removed": False}
        (root / "removed.jsonl").unlink()
        assert walk() == {"kept": False, "removed": True}
        (root / "removed.jsonl").write_text("{}\n", encoding="utf-8")
        assert walk() == {"kept": False, "removed": False}
    finally:
        conn.close()


def test_walk_hands_the_writer_only_sources_the_catalog_does_not_hold(tmp_path) -> None:
    """REQ-RECON-019: a walk over an unchanged root sends the writer nothing, and every
    source whose observation would change the catalog still reaches it."""
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog
    from recall.services.reconciler import InventoryBatch, iter_inventory_batches, unobserved

    root = tmp_path / "sessions"
    root.mkdir()
    for name in ("same", "appended", "returning"):
        (root / f"{name}.jsonl").write_text("{}\n", encoding="utf-8")
    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)

        def walk() -> None:
            inventory_root(catalog, _JsonlParser(), root, clock=lambda: 100.0)

        def unobserved_names(parser_revision: str = "") -> set[str]:
            present = catalog.present_states("codex", str(root.resolve()))
            return {
                Path(item.source_path).stem
                for event in iter_inventory_batches(
                    _JsonlParser(), root, clock=lambda: 100.0, parser_revision=parser_revision
                )
                if isinstance(event, InventoryBatch)
                for item in unobserved(event, present).files
            }

        assert unobserved_names() == {"same", "appended", "returning"}
        walk()
        assert unobserved_names() == set()
        assert unobserved_names(parser_revision="next") == {"same", "appended", "returning"}

        # A file that leaves and comes back unmodified keeps its identity, but its
        # row now says missing.
        (root / "returning.jsonl").rename(tmp_path / "away.jsonl")
        walk()
        (tmp_path / "away.jsonl").rename(root / "returning.jsonl")
        with (root / "appended.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        (root / "new.jsonl").write_text("{}\n", encoding="utf-8")

        assert unobserved_names() == {"appended", "returning", "new"}
    finally:
        conn.close()


def test_capture_catalogues_a_symlinked_transcript_under_its_target(tmp_path) -> None:
    """A symlink is catalogued at the path it resolves to and counted as an alias, so
    absence detection does not trust the discovered count; a symlinked root is not."""
    from recall.services.reconciler import InventoryBatch, InventoryResult, iter_inventory_batches

    root = tmp_path / "sessions"
    (root / "day").mkdir(parents=True)
    target = root / "day" / "real.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    (root / "link.jsonl").symlink_to(target)
    (root / "notes.txt").write_text("not a transcript", encoding="utf-8")
    (tmp_path / "linked-root").symlink_to(root)

    for walked in (root, tmp_path / "linked-root"):
        events = list(iter_inventory_batches(_JsonlParser(), walked, clock=lambda: 100.0))
        captured = [
            (item.root_path, item.source_path)
            for event in events
            if isinstance(event, InventoryBatch)
            for item in event.files
        ]
        result = events[-1]
        assert isinstance(result, InventoryResult)
        assert captured == [(str(root.resolve()), str(target.resolve()))] * 2
        assert (result.discovered_count, result.aliased_count) == (2, 1)


def test_inventory_walk_error_does_not_mark_unseen_rows_missing(tmp_path, monkeypatch) -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    class Parser:
        source = "codex"
        file_pattern = "*.jsonl"

        @staticmethod
        def sidecar_paths(path: Path) -> list[Path]:
            return []

    def broken_walk(root: Path, *, onerror: Callable[[OSError], None]) -> object:
        onerror(PermissionError("denied"))
        return iter(())

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        catalog.observe(
            "codex", str(tmp_path), str(tmp_path / "prior.jsonl"), SourceSignature(1, 2, 3, 4, 5)
        )
        monkeypatch.setattr(reconciler.os, "walk", broken_walk)
        assert inventory_root(catalog, Parser(), tmp_path, clock=lambda: 101.0) == 0
        assert not catalog.status_page().rows[0].missing
        assert conn.execute(
            "SELECT scan_complete, failure_count FROM reconciliation_roots"
        ).fetchone() == (
            False,
            1,
        )
    finally:
        conn.close()


def test_error_and_retry_are_scoped_to_source_for_shared_paths() -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        signature = SourceSignature(1, 2, 3, 4, 5)
        catalog.observe("codex", "/root", "/root/shared.jsonl", signature)
        catalog.observe("grok", "/root", "/root/shared.jsonl", signature)
        catalog.fail("codex", "/root/shared.jsonl", "parse failed")
        catalog.retry_now("grok", "/root/shared.jsonl")
        assert [(row.source, row.retry_count) for row in catalog.status_page().rows] == [
            ("codex", 1),
            ("grok", 0),
        ]
    finally:
        conn.close()


def test_inventory_root_scopes_distinguish_default_disabled_and_missing(tmp_path) -> None:
    class Parser:
        source = "codex"

        def __init__(self, roots: tuple[Path, ...] | None) -> None:
            self.roots = roots

        def default_roots(self) -> list[Path]:
            return [tmp_path / "default"]

    default = inventory_root_scopes(Parser(None))
    disabled = inventory_root_scopes(Parser(()))
    configured = inventory_root_scopes(Parser((tmp_path / "missing",)))
    assert (default[0].configuration, default[0].available) == ("default", False)
    assert (disabled[0].configuration, disabled[0].root_path) == ("disabled", None)
    assert (configured[0].configuration, configured[0].available) == ("configured", False)


def test_service_sequence_rotates_oldest_work_across_scheduler_restart(tmp_path) -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    database = tmp_path / "rotation.duckdb"
    conn = duckdb.connect(str(database))
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 1.0)
        signature = SourceSignature(1, 2, 3, 4, 5)
        for index in range(8):
            catalog.observe("codex", "/root", f"/root/{index}.jsonl", signature)
        first = FairScheduler(clock=lambda: 1.0).select(catalog, set(), limit=1)[0].source
        assert catalog.acknowledge("codex", first.source_path, 1, 5, "a" * 64)
        catalog.observe("codex", "/root", first.source_path, SourceSignature(1, 2, 3, 5, 6))
    finally:
        conn.close()

    conn = duckdb.connect(str(database))
    try:
        second = (
            FairScheduler(clock=lambda: 1.0)
            .select(SourceCatalog(conn, clock=lambda: 1.0), set(), limit=1)[0]
            .source
        )
        assert second.source_path != first.source_path
        assert second.last_serviced_seq == 0
    finally:
        conn.close()


def test_scheduler_reserves_one_historical_service_in_every_seven_under_active_churn() -> None:
    """New active writes cannot keep an old pending source out of a seven-slot window."""
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    now = [0.0]
    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: now[0])
        old_paths = [f"/root/history-{index}.jsonl" for index in range(4)]
        active_paths = {f"/root/active-{index}.jsonl" for index in range(4)}
        signature = SourceSignature(1, 2, 3, 4, 5)
        for path in old_paths:
            catalog.observe("codex", "/root", path, signature)
        for path in active_paths:
            catalog.observe("codex", "/root", path, signature)

        scheduler = FairScheduler(clock=lambda: now[0])
        selected: list[str] = []
        for slot in range(28):
            item = scheduler.select(catalog, active_paths, limit=1)[0].source
            selected.append(item.source_path)
            assert catalog.acknowledge(
                "codex", item.source_path, item.desired_generation, 5, "a" * 64
            )
            now[0] += 1
            catalog.observe(
                "codex",
                "/root",
                item.source_path,
                SourceSignature(1, 2, 3, 4 + slot, 6 + slot),
            )
            newest = f"/root/active-new-{slot}.jsonl"
            active_paths.add(newest)
            catalog.observe("codex", "/root", newest, SourceSignature(1, 2, 3, 100 + slot, 5))

        for start in range(0, 28, 7):
            assert any(path in old_paths for path in selected[start : start + 7])
    finally:
        conn.close()


def test_scheduler_does_not_merge_same_canonical_path_from_two_sources() -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 1.0)
        signature = SourceSignature(1, 2, 3, 4, 5)
        catalog.observe("codex", "/root", "/root/shared.jsonl", signature)
        catalog.observe("grok", "/root", "/root/shared.jsonl", signature)

        selected = FairScheduler(clock=lambda: 1.0).select(catalog, set(), limit=2)
        assert [(item.source.source, item.source.source_path) for item in selected] == [
            ("codex", "/root/shared.jsonl"),
            ("grok", "/root/shared.jsonl"),
        ]
    finally:
        conn.close()


def test_rewritten_active_source_rotates_behind_unserved_active_source() -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 1.0)
        signature = SourceSignature(1, 2, 3, 4, 5)
        active_paths = {"/root/a.jsonl", "/root/b.jsonl"}
        for path in active_paths:
            catalog.observe("codex", "/root", path, signature)

        scheduler = FairScheduler(clock=lambda: 1.0)
        first = scheduler.select(catalog, active_paths, limit=1)[0].source
        assert catalog.acknowledge(
            "codex", first.source_path, first.desired_generation, 5, "a" * 64
        )
        catalog.observe("codex", "/root", first.source_path, SourceSignature(1, 2, 3, 5, 6))
        second = scheduler.select(catalog, active_paths, limit=1)[0].source
        assert second.source_path != first.source_path
        assert second.last_serviced_seq == 0
    finally:
        conn.close()


def test_stale_completion_cannot_replace_a_newer_committed_generation() -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 1.0)
        first = catalog.observe("codex", "/root", "/root/a.jsonl", SourceSignature(1, 2, 3, 4, 5))
        second = catalog.observe("codex", "/root", "/root/a.jsonl", SourceSignature(1, 2, 3, 5, 6))
        assert catalog.acknowledge("codex", "/root/a.jsonl", second, 6, "b" * 64)
        assert not catalog.acknowledge("codex", "/root/a.jsonl", first, 5, "a" * 64)
        row = catalog.status_page().rows[0]
        assert (row.committed_generation, row.committed_offset, row.committed_prefix_sha256) == (
            second,
            6,
            "b" * 64,
        )
    finally:
        conn.close()


def test_committed_content_and_ack_survive_restart_but_uncommitted_pair_does_not(tmp_path) -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    database = tmp_path / "atomic-content.duckdb"
    conn = duckdb.connect(str(database))
    try:
        ensure_schema(conn, embed_dim=8)
        conn.execute("CREATE TABLE owned_content (generation BIGINT PRIMARY KEY, body TEXT)")
        catalog = SourceCatalog(conn, clock=lambda: 1.0)
        generation = catalog.observe(
            "codex", "/root", "/root/a.jsonl", SourceSignature(1, 2, 3, 4, 5)
        )
        conn.execute("BEGIN")
        conn.execute("INSERT INTO owned_content VALUES (?, ?)", [generation, "before-crash"])
        assert catalog.acknowledge("codex", "/root/a.jsonl", generation, 5, "a" * 64)
        conn.execute("ROLLBACK")
        assert conn.execute("SELECT * FROM owned_content").fetchall() == []
        assert catalog.pending()[0].committed_generation == 0

        conn.execute("BEGIN")
        conn.execute("INSERT INTO owned_content VALUES (?, ?)", [generation, "after-commit"])
        assert catalog.acknowledge("codex", "/root/a.jsonl", generation, 5, "a" * 64)
        conn.execute("COMMIT")
    finally:
        conn.close()

    conn = duckdb.connect(str(database))
    try:
        catalog = SourceCatalog(conn, clock=lambda: 2.0)
        assert conn.execute("SELECT body FROM owned_content").fetchall() == [("after-commit",)]
        assert catalog.pending() == []
    finally:
        conn.close()


def test_interrupted_inventory_records_exact_failures_and_bounded_details(
    tmp_path, monkeypatch
) -> None:
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    class Parser:
        source = "codex"
        file_pattern = "*.jsonl"

        @staticmethod
        def sidecar_paths(path: Path) -> list[Path]:
            return []

    def interrupted_walk(root: Path, *, onerror: Callable[[OSError], None]) -> object:
        for index in range(40):
            onerror(PermissionError(f"denied-{index}"))
        return iter(())

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 1.0)
        catalog.observe(
            "codex", str(tmp_path), str(tmp_path / "prior.jsonl"), SourceSignature(1, 2, 3, 4, 5)
        )
        monkeypatch.setattr(reconciler.os, "walk", interrupted_walk)
        assert inventory_root(catalog, Parser(), tmp_path, clock=lambda: 2.0) == 0
        scan = conn.execute(
            "SELECT scan_complete, failure_count, failures FROM reconciliation_roots"
        ).fetchone()
        assert scan is not None
        scan_complete, failure_count, failures = scan
        assert (scan_complete, failure_count) == (False, 40)
        assert len(json.loads(failures)) == 32
        assert not catalog.status_page().rows[0].missing
    finally:
        conn.close()


def test_every_original_pending_source_is_serviced_within_seven_times_population() -> None:
    """A continually dirty oldest path yields to every other original pending path."""
    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    now = [1.0]
    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: now[0])
        originals = {f"/root/original-{index}.jsonl" for index in range(8)}
        for path in sorted(originals):
            catalog.observe("codex", "/root", path, SourceSignature(1, 2, 3, 4, 5))
            now[0] += 1
        scheduler = FairScheduler(clock=lambda: now[0])
        serviced: set[str] = set()
        for slot in range(56):
            item = scheduler.select(catalog, set(), limit=1)[0].source
            # A write arrives during preparation, so acknowledgment never clears
            # this source's original pending age.
            catalog.observe(
                "codex",
                "/root",
                item.source_path,
                SourceSignature(1, 2, 100 + slot, 100 + slot, 6 + slot),
            )
            assert catalog.acknowledge(
                "codex", item.source_path, item.desired_generation, 5, "a" * 64
            )
            serviced.add(item.source_path)
            now[0] += 1
            catalog.observe(
                "codex", "/root", f"/root/new-{slot}.jsonl", SourceSignature(1, 2, 3, 4, 5)
            )
        assert originals <= serviced
    finally:
        conn.close()


def test_process_death_preserves_atomic_content_and_catalog_checkpoint(tmp_path: Path) -> None:
    """Kill at the actual transaction boundary, then recover through DuckDB WAL."""
    import selectors
    import subprocess
    import sys

    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    program = """
import sys
import duckdb
from recall.db.source_files import SourceCatalog
conn = duckdb.connect(sys.argv[1])
conn.execute('BEGIN')
conn.execute("INSERT INTO owned_content VALUES ('captured-generation')")
assert SourceCatalog(conn, clock=lambda: 1.0).acknowledge('codex', '/root/a', 1, 5, 'a' * 64)
if sys.argv[2] == 'commit':
    conn.execute('COMMIT')
print('boundary', flush=True)
sys.stdin.readline()
"""
    for committed in (False, True):
        database = tmp_path / f"death-{committed}.duckdb"
        conn = duckdb.connect(str(database))
        ensure_schema(conn, embed_dim=8)
        conn.execute("CREATE TABLE owned_content (body TEXT)")
        SourceCatalog(conn, clock=lambda: 1.0).observe(
            "codex", "/root", "/root/a", SourceSignature(1, 2, 3, 4, 5)
        )
        conn.close()
        process = subprocess.Popen(
            [sys.executable, "-c", program, str(database), "commit" if committed else "open"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            assert process.stdout is not None
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                assert selector.select(timeout=10), "child did not reach the transaction boundary"
                assert process.stdout.readline() == b"boundary\n"
            process.kill()
            process.communicate(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
        conn = duckdb.connect(str(database))
        try:
            expected = [("captured-generation",)] if committed else []
            assert conn.execute("SELECT * FROM owned_content").fetchall() == expected
            row = SourceCatalog(conn, clock=lambda: 2.0).status_page().rows[0]
            assert row.committed_generation == (1 if committed else 0)
            assert row.desired_generation == 1
        finally:
            conn.close()


def test_sidecar_same_size_same_mtime_rewrite_creates_pending_generation(tmp_path: Path) -> None:
    import os

    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog

    sidecar = tmp_path / "metadata.json"
    sidecar.write_text('{"title":"old"}')
    (tmp_path / "rollout.jsonl").write_text("{}\n")

    class Parser:
        source = "codex"
        file_pattern = "*.jsonl"

        @staticmethod
        def sidecar_paths(path: Path) -> list[Path]:
            return [sidecar]

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 1.0)
        inventory_root(catalog, Parser(), tmp_path, clock=lambda: 1.0)
        first = catalog.status_page().rows[0]
        assert catalog.acknowledge("codex", first.source_path, 1, 3, "a" * 64)
        before = sidecar.stat()
        sidecar.write_text('{"title":"new"}')
        os.utime(sidecar, ns=(before.st_atime_ns, before.st_mtime_ns))
        inventory_root(catalog, Parser(), tmp_path, clock=lambda: 2.0)
        assert catalog.status_page().rows[0].desired_generation == 2
    finally:
        conn.close()


def test_batched_observations_preserve_generation_retry_and_membership() -> None:
    from dataclasses import replace

    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        now = [100.0]
        catalog = SourceCatalog(conn, clock=lambda: now[0])
        signature = SourceSignature(None, None, 3, 4, 5, 0, "parser", "sidecars")
        observations = [
            ("codex", "/root", f"/root/{name}", signature) for name in ("same", "changed")
        ]
        catalog.observe_batch(observations)
        catalog.acknowledge("codex", "/root/same", 1, 5, "a" * 64)
        catalog.fail("codex", "/root/changed", "blocked")
        now[0] = 110.0
        catalog.observe_batch(observations)
        same = catalog.get("codex", "/root/same")
        changed = catalog.get("codex", "/root/changed")
        assert same is not None and same.current and same.first_pending_at is None
        assert changed is not None and changed.desired_generation == 1
        assert changed.retry_count == 1 and changed.next_retry_at == 102.0
        assert changed.first_pending_at == 100.0
        catalog.observe_batch(
            [("codex", "/new-root", "/root/changed", replace(signature, sidecar_signature="new"))],
        )
        changed = catalog.get("codex", "/root/changed")
        assert changed is not None and changed.desired_generation == 2
        assert changed.root_path == "/new-root"
        assert changed.next_retry_at == 0 and changed.first_pending_at == 100.0
        assert changed.last_error == "blocked" and changed.committed_generation == 0
    finally:
        conn.close()


def test_inventory_refresh_preserves_indexed_rows_for_unchanged_signatures() -> None:
    """REQ-RECON-019: refreshing scan membership must not rewrite unchanged indexes."""
    from dataclasses import replace

    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        now = [100.0]
        catalog = SourceCatalog(conn, clock=lambda: now[0])
        signature = SourceSignature(None, None, 3, 4, 5, 0, "parser", "sidecars")
        catalog.observe_batch(
            [("codex", "/root", f"/root/{name}", signature) for name in ("same", "changed")],
        )
        assert catalog.acknowledge("codex", "/root/same", 1, 5, "a" * 64)
        catalog.fail("codex", "/root/changed", "blocked")
        before = dict(conn.execute("SELECT source_path, rowid FROM source_files").fetchall())
        now[0] = 110.0
        catalog.observe_batch(
            [
                ("codex", "/new-root", "/root/same", signature),
                ("codex", "/root", "/root/changed", replace(signature, size=6)),
            ],
        )
        same = catalog.get("codex", "/root/same")
        changed = catalog.get("codex", "/root/changed")
        assert same is not None and same.current and same.root_path == "/new-root"
        assert same.first_pending_at is None
        assert changed is not None and changed.desired_generation == 2
        assert changed.retry_count == 1 and changed.next_retry_at == 0
        assert changed.last_error == "blocked" and changed.first_pending_at == 100.0
        assert conn.execute(
            "SELECT observed_at FROM source_files ORDER BY source_path"
        ).fetchall() == [(110.0,), (110.0,)]
        # Neither row moves. DuckDB rewrites a row whose UPDATE names an indexed
        # column, so before REQ-MIG-010 retired idx_source_files_pending the
        # changed row was physically reinserted on every refresh; dropping that
        # index makes the same logical update an in-place write.
        after = dict(conn.execute("SELECT source_path, rowid FROM source_files").fetchall())
        assert after["/root/same"] == before["/root/same"]
        assert after["/root/changed"] == before["/root/changed"]
    finally:
        conn.close()


def test_mixed_inventory_refresh_respects_caller_rollback() -> None:
    from dataclasses import replace

    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        now = [100.0]
        catalog = SourceCatalog(conn, clock=lambda: now[0])
        signature = SourceSignature(1, 2, 3, 4, 5)
        catalog.observe_batch(
            [("codex", "/root", f"/root/{name}", signature) for name in ("same", "changed")],
        )
        assert catalog.acknowledge("codex", "/root/changed", 1, 5, "a" * 64)
        before = conn.execute("SELECT * FROM source_files ORDER BY source_key").fetchall()
        conn.execute("BEGIN TRANSACTION")
        now[0] = 110.0
        catalog.observe_batch(
            [
                ("codex", "/new-root", "/root/same", signature),
                ("codex", "/root", "/root/changed", replace(signature, size=6)),
                ("codex", "/root", "/root/new", signature),
            ],
        )
        assert conn.execute(
            "SELECT source_path, root_path, desired_generation, first_pending_at "
            "FROM source_files ORDER BY source_key"
        ).fetchall() == [
            ("/root/changed", "/root", 2, 110.0),
            ("/root/new", "/root", 1, 110.0),
            ("/root/same", "/new-root", 1, 100.0),
        ]
        conn.execute("ROLLBACK")
        assert conn.execute("SELECT * FROM source_files ORDER BY source_key").fetchall() == before
        # The handle remains usable, and an unchanged re-observation writes nothing.
        catalog.observe_batch([("codex", "/root", "/root/changed", signature)])
        assert conn.execute("SELECT * FROM source_files ORDER BY source_key").fetchall() == before
    finally:
        conn.close()


def test_batched_aliases_preserve_each_observed_change() -> None:
    from dataclasses import replace

    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        first = SourceSignature(1, 2, 3, 4, 5)
        catalog.observe_batch(
            [
                ("codex", "/root", "/root/same", first),
                ("codex", "/root", "/root/same", replace(first, size=6)),
                ("codex", "/root", "/root/same", replace(first, size=7)),
            ]
        )
        result = catalog.get("codex", "/root/same")
        assert result is not None and result.desired_generation == 3
        assert result.signature is not None and result.signature.size == 7
        assert result.first_pending_at == 100.0
    finally:
        conn.close()


def test_batched_signatures_retain_integer_precision_above_float_range() -> None:
    from dataclasses import replace

    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        first = SourceSignature(None, None, 2**60 + 1, 2**60 + 3, 5, 2**60 + 7, "p", "s")
        catalog.observe_batch([("codex", "/root", "/root/session", first)])
        row = catalog.get("codex", "/root/session")
        assert row is not None and row.signature == first
        second = replace(first, mtime_ns=first.mtime_ns + 1)
        catalog.observe_batch([("codex", "/root", "/root/session", second)])
        row = catalog.get("codex", "/root/session")
        assert row is not None and row.signature == second and row.desired_generation == 2
    finally:
        conn.close()


def test_inventory_refresh_cannot_outrank_a_pending_source_write() -> None:
    from dataclasses import replace

    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        now = [1000.0]
        catalog = SourceCatalog(conn, clock=lambda: now[0])
        signature = SourceSignature(1, 2, 3, 1000 * 10**9, 5)
        catalog.observe("codex", "/root", "/root/writing", signature)
        catalog.acknowledge("codex", "/root/writing", 1, 5, "a" * 64)
        now[0] = 1001.0
        catalog.observe("codex", "/root", "/root/writing", replace(signature, size=6))
        history = [
            ("codex", "/root", f"/root/old-{index}", replace(signature, mtime_ns=1))
            for index in range(100)
        ]
        now[0] = 1002.0
        catalog.observe_batch(history)
        now[0] = 1003.0
        catalog.observe_batch(history)
        selected = FairScheduler(clock=lambda: now[0]).select(catalog, set(), limit=7)
        assert "/root/writing" in {item.source.source_path for item in selected}
        assert any(item.source.source_path.startswith("/root/old-") for item in selected)
    finally:
        conn.close()


def test_already_serviced_pending_work_keeps_its_bound_under_new_arrivals() -> None:
    from dataclasses import replace

    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        now = [1000.0]
        catalog = SourceCatalog(conn, clock=lambda: now[0])
        signature = SourceSignature(1, 2, 3, 1000 * 10**9, 5)
        target = "/root/resumed"
        catalog.observe("codex", "/root", target, signature)
        catalog.acknowledge("codex", target, 1, 5, "a" * 64)
        now[0] += 1
        catalog.observe("codex", "/root", target, replace(signature, size=6))
        active = {target}
        scheduler = FairScheduler(clock=lambda: now[0])
        serviced = []
        for number in range(7):
            now[0] += 1
            incoming = f"/root/arrival-{number}"
            catalog.observe("codex", "/root", incoming, signature)
            active.add(incoming)
            item = scheduler.select(catalog, active, limit=1)[0].source
            serviced.append(item.source_path)
            assert item.signature is not None
            catalog.acknowledge(
                "codex", item.source_path, item.desired_generation, item.signature.size, "a" * 64
            )
        assert target in serviced, "new arrivals starved work pending before their arrival"
    finally:
        conn.close()


def test_recent_mtime_uses_recent_lane_without_activity_evidence() -> None:
    from dataclasses import replace

    from recall.db.schema import ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 1000.0)
        signature = SourceSignature(1, 2, 3, 990 * 10**9, 5)
        catalog.observe_batch(
            [("codex", "/root", f"/root/active-{index}", signature) for index in range(100)]
        )
        catalog.observe_batch(
            [
                ("codex", "/root", f"/root/old-{index}", replace(signature, mtime_ns=1))
                for index in range(100)
            ]
        )
        active, recent, oldest = catalog.priority_pending(set(), active_since_ns=900 * 10**9)
        assert active == []
        assert recent
        assert all(item.source_path.startswith("/root/active-") for item in recent)
        assert oldest
    finally:
        conn.close()


def _v29_catalog_conn(to_schema_v29) -> duckdb.DuckDBPyConnection:
    """An in-memory catalog still shaped like schema version 29 (REQ-MIG-010)."""
    from recall.db.schema import ensure_schema

    conn = duckdb.connect(":memory:")
    ensure_schema(conn, embed_dim=8)
    return to_schema_v29(conn)


def test_migrated_catalog_keeps_pending_ack_and_missing_convergence(
    tmp_path, to_schema_v29
) -> None:
    """REQ-MIG-010: version 30 preserves every reconciliation invariant."""
    from dataclasses import replace

    from recall.db.schema import SCHEMA_VERSION, _get_schema_version, ensure_schema
    from recall.db.source_files import SourceCatalog, SourceSignature

    conn = _v29_catalog_conn(to_schema_v29)
    try:
        now = [100.0]
        catalog = SourceCatalog(conn, clock=lambda: now[0])
        signature = SourceSignature(None, None, 3, 4, 5, 0, "parser", "sidecars")
        present = tmp_path / "present.jsonl"
        present.write_text("{}\n", encoding="utf-8")
        gone = tmp_path / "gone.jsonl"
        root = str(tmp_path)
        catalog.observe_batch(
            [
                ("codex", root, str(present), signature),
                ("codex", root, str(gone), signature),
            ]
        )
        assert catalog.acknowledge("codex", str(present), 1, 5, "a" * 64)
        catalog.fail("codex", str(gone), "blocked")
        now[0] = 110.0
        catalog.observe_batch([("codex", root, str(present), replace(signature, size=6))])
        before_pending = [row.source_path for row in catalog.pending(limit=8)]
        assert before_pending == [str(gone), str(present)]

        ensure_schema(conn, embed_dim=8)

        # The invariants below only mean anything once the migration has run.
        assert _get_schema_version(conn) == SCHEMA_VERSION == 31
        assert "inventory_generation" not in {
            str(row[1]) for row in conn.execute("PRAGMA table_info('source_files')").fetchall()
        }

        migrated = SourceCatalog(conn, clock=lambda: now[0])
        assert [row.source_path for row in migrated.pending(limit=8)] == before_pending
        assert migrated.acknowledge("codex", str(present), 2, 6, "b" * 64)
        assert [row.source_path for row in migrated.pending(limit=8)] == [str(gone)]
        current = migrated.get("codex", str(present))
        assert current is not None and current.current
        failed = migrated.get("codex", str(gone))
        assert failed is not None and failed.last_error == "blocked"

        # Missing-source convergence still runs off the catalog, not the walk.
        result = reconciler.InventoryResult(
            source="codex",
            root_path=root,
            started_at=110.0,
            finished_at=111.0,
            discovered_count=1,
            failures=(),
            failure_count=0,
        )
        reconciler.mark_vanished_sources(migrated, result)
        vanished = migrated.get("codex", str(gone))
        assert vanished is not None and vanished.missing
        assert migrated.present_count("codex", root) == 1
    finally:
        conn.close()
