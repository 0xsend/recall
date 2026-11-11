"""Admitted watched identities are reusable without filesystem resolution."""

import os
from datetime import datetime
from pathlib import Path

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.services.live import live_roster_query
from recall.services.live_session_set import LiveSessionSet


@pytest.mark.parametrize("count", [13, 4096])
def test_admitted_roster_reads_do_not_resolve_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    live_set = LiveSessionSet(max_subscriptions=1, idle_threshold=30, is_macos=True)
    live_set.seed(((tmp_path / f"member-{n}", 100) for n in range(count)), monotonic_now=1)
    resolutions = []
    original = os.path.realpath

    def resolve(path, *args, **kwargs):
        resolutions.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", resolve)
    # Exercise the read boundary that serves probe, page and coverage.
    for _ in range(3):
        paths = live_set.roster()
        query = live_roster_query(
            watched_paths=paths,
            include_idle=False,
            now=datetime(2026, 11, 10),
            idle_window_seconds=86400,
        )
        with duckdb.connect(":memory:") as conn:
            ensure_schema(conn, embed_dim=8)
            sql, params = query
            rows = conn.execute(
                sql + "SELECT source_path FROM classified ORDER BY source_path", params
            ).fetchall()
        assert rows == [
            (str(tmp_path / f"member-{n}"),)
            for n in sorted(range(count), key=lambda n: f"member-{n}")
        ]
    assert resolutions == []


def test_roster_identity_tracks_alias_refresh_demotion_and_reseed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "quote'百分%.jsonl"
    target.write_text("")
    other = tmp_path / "other.jsonl"
    other.write_text("")
    link = tmp_path / "alias.jsonl"
    link.symlink_to(target)
    live_set = LiveSessionSet(max_subscriptions=1, idle_threshold=30, is_macos=True)
    live_set.seed([(Path("alias.jsonl"), 100), (target, 100)], monotonic_now=1)
    before = live_set.roster()
    expected_target = str(target.resolve())
    assert before.paths == (expected_target,)
    assert before.member_count == 2
    link.unlink()
    link.symlink_to(other)
    live_set.promote(Path("alias.jsonl"), 200, 2, bump_event=True)
    assert live_set.roster().paths == tuple(sorted((expected_target, str(other.resolve()))))
    assert before.paths == (expected_target,)
    live_set.demote_idle(wall_now=250, monotonic_now=100)
    assert live_set.roster().paths == ()
    assert live_set.roster().member_count == 0
    compressed = tmp_path / "rollout-compressed.jsonl.zst"
    compressed.write_bytes(b"")
    live_set.seed([(compressed, 300)], monotonic_now=101)
    assert live_set.roster().paths == (str(compressed.resolve().with_suffix("")),)
    assert live_set.roster().member_count == 1
