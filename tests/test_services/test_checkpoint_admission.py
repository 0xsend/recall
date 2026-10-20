"""Checkpoint refusal leaves admissible lanes runnable and retains oldest service."""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from lane_harness import lane_config
from recall.db.connection import checkpoint_wal_if_due
from recall.db.schema import ensure_schema
from recall.db.source_files import SourceCatalog, SourceSignature
from recall.services.reconciler import FairScheduler, ScheduledSource
from recall.services.rpc_server import RpcServer


@pytest.mark.parametrize("limit", [1, 4])
def test_contended_checkpoint_preserves_active_service_and_oldest_claim(
    tmp_path: Path, limit: int
) -> None:
    database = tmp_path / "checkpoint.duckdb"
    with closing(duckdb.connect(str(database))) as conn, closing(conn.cursor()) as reader:
        ensure_schema(conn, embed_dim=8)
        conn.execute("SET wal_autocheckpoint = '256MiB'")
        conn.execute("CREATE TABLE checkpoint_probe(value INTEGER)")
        conn.execute("INSERT INTO checkpoint_probe VALUES (0)")
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        history = [f"/root/history-{index}.jsonl" for index in range(4)]
        active = {f"/root/active-{index}.jsonl" for index in range(12)}
        signature = SourceSignature(1, 2, 3, 4, 5)
        for path in history:
            catalog.observe("codex", "/root", path, signature)
        for path in sorted(active):
            catalog.observe("codex", "/root", path, signature)
            assert catalog.acknowledge("codex", path, 1, 5, "a" * 64)
            catalog.observe("codex", "/root", path, replace(signature, size=6))
        scheduler = FairScheduler(clock=lambda: 100.0)
        server = RpcServer(lane_config(tmp_path))

        def select(count: int) -> list[ScheduledSource]:
            return scheduler.select(
                catalog,
                active,
                limit=count,
                active_since_ns=50_000_000_000,
                can_admit=lambda item: server._raw_candidate_admitted(
                    item.source, is_active=item.is_active, history_busy=False
                ),
            )

        def serve(items: list[ScheduledSource]) -> None:
            for item in items:
                source = item.source
                assert source.signature is not None
                assert catalog.acknowledge(
                    source.source,
                    source.source_path,
                    source.desired_generation,
                    source.signature.size,
                    "a" * 64,
                )
                if item.is_active:
                    catalog.observe(
                        source.source,
                        "/root",
                        source.source_path,
                        replace(source.signature, size=source.signature.size + 1),
                    )

        # Each contention window falls on the reserved oldest slot. Repeated
        # active appends and later arrivals cannot consume that obligation.
        for index, oldest in enumerate(history):
            normal = select(6)
            assert len(normal) == 6
            serve(normal)
            reader.execute("BEGIN")
            try:
                reader.execute("SELECT * FROM checkpoint_probe").fetchall()
                conn.execute("UPDATE checkpoint_probe SET value = value + 1")
                for _ in range(3):
                    result = checkpoint_wal_if_due(conn, database, high_water_bytes=1)
                    assert result.contended
                    server._checkpoint_contention_retries += 1
                    server._wal_pressure_blocked = True
                    assert select(limit) == [], "WAL pressure must still block every lane"
                    server._wal_pressure_blocked = False
                    selected = select(limit)
                    assert len(selected) == limit, "refused history suppressed active appends"
                    assert all(item.is_active for item in selected)
                    assert len({item.source.source_path for item in selected}) == limit
                    serve(selected)
                    catalog.observe("codex", "/root", f"/root/arrival-{index}.jsonl", signature)
            finally:
                reader.execute("ROLLBACK")
            completed = checkpoint_wal_if_due(conn, database, high_water_bytes=1)
            assert completed.attempted and not completed.contended
            server._checkpoint_contention_retries = 0
            recovered = select(1)
            assert len(recovered) == 1
            assert recovered[0].source.source_path == oldest
            serve(recovered)
        # All four original history sources receive service within 7*N ordinary
        # slots, despite twelve refused attempts and continuously pending active work.
        for path in history:
            source = catalog.get("codex", path)
            assert source is not None and source.committed_generation == 1
