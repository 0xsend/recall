"""Pending lanes preserve exact source membership independently of roster size."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pytest
from recall.core.source_path import SourcePath
from recall.core.types import Source
from recall.db.schema import ensure_schema
from recall.db.source_files import PendingSource, SourceCatalog, SourceSignature, source_key


@dataclass(frozen=True)
class Candidate:
    source: str
    path: str
    committed: int
    mtime: int
    sequence: int


@pytest.mark.parametrize(
    "roster",
    [
        (),
        ("/root/quote'百分%.jsonl",),
        ("/root/quote'百分%.jsonl",) * 2,
        tuple(f"/root/{n}" for n in range(4096)),
    ],
)
@pytest.mark.parametrize("scoped", [False, True])
def test_pending_lanes_match_reference(roster: tuple[str, ...], scoped: bool) -> None:
    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 100)
        paths = ["/root/quote'百分%.jsonl", "/root/雪.jsonl", *[f"/root/{n}" for n in range(160)]]
        candidates = [
            Candidate(source, path, int(n % 3 != 2), n % 2 * 1000, n + 1)
            for n, path in enumerate(paths)
            for source in ("codex", "claude_code")
        ]
        for offset in range(0, len(candidates), 256):
            catalog.observe_batch(
                [
                    (c.source, "/root", c.path, SourceSignature(1, 2, 1, c.mtime, 1))
                    for c in candidates[offset : offset + 256]
                ]
            )
        # Distinct service/admission ordering, delayed retries, missing and settled rows.
        conn.execute("UPDATE source_files SET committed_generation = 1, desired_generation = 2")
        for c in candidates:
            conn.execute(
                "UPDATE source_files SET committed_generation=?, first_pending_seq=?, "
                "first_pending_at=?, last_serviced_seq=?, next_retry_at=?, missing=?, "
                "desired_generation=?, committed_offset=? WHERE source_key=?",
                [
                    c.committed,
                    c.sequence,
                    c.sequence,
                    c.sequence // 4,
                    200 if c.sequence % 17 == 0 else 0,
                    c.sequence % 19 == 0,
                    c.committed if c.sequence % 23 == 0 else 2,
                    1 if c.sequence % 23 == 0 else 0,
                    source_key(c.source, c.path),
                ],
            )
        requested = (SourcePath(Source.CLAUDE_CODE, "/root/雪.jsonl"),) if scoped else ()
        excluded = (source_key("codex", "/root/4"),)
        active_paths = set(roster)

        def is_active(c: Candidate) -> bool:
            return c.committed > 0 and (
                c.path in active_paths
                or c.mtime >= 900
                or any(k.source == c.source and k.path == c.path for k in requested)
            )

        eligible = [
            c
            for c in candidates
            if c.sequence % 17
            and c.sequence % 19
            and source_key(c.source, c.path) not in excluded
            and c.sequence % 23
            and (
                not scoped
                or c.source == "codex"
                or any(k.source == c.source and k.path == c.path for k in requested)
            )
        ]

        def fair(c: Candidate) -> tuple[int, int, str, str]:
            return c.sequence, c.sequence, c.source, c.path

        def project(rows: list[Candidate]) -> list[PendingSource]:
            return [PendingSource(c.source, c.path, is_active(c)) for c in rows[:85]]

        expected = (
            project(sorted([c for c in eligible if is_active(c)], key=fair)),
            project(
                sorted(
                    [c for c in eligible if not is_active(c) and c.mtime >= 900],
                    key=lambda c: (-c.mtime, -c.sequence, c.source, c.path),
                )
            ),
            project(sorted(eligible, key=fair)),
        )
        actual = catalog.priority_pending(
            active_paths,
            active_since_ns=900,
            eligible_roots=(("codex", "/root"),) if scoped else None,
            requested_paths=requested,
            excluded_keys=excluded,
        )
        assert actual == expected
