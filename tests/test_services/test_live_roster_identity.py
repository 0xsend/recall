"""Read-time watched identities are shared by every query in a snapshot."""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
    SourceConfig,
)
from recall.db.schema import ensure_schema
from recall.parsers.codex import CodexParser
from recall.services.live import live_roster_query
from recall.services.live_session_set import LiveSessionSet
from recall.services.rpc_server import RpcServer
from recall.services.watcher import DebouncedIndexQueue, SessionFileHandler


@pytest.mark.parametrize("count", [13, 4096])
def test_roster_resolves_once_per_snapshot_for_multiple_queries(
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
        for _query in range(3):
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
    assert len(resolutions) == 3 * count * 2


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


@pytest.mark.parametrize("topology", ["leaf", "ancestor", "chain"])
def test_handler_modification_resolves_replaced_member_at_read(
    tmp_path: Path, topology: str
) -> None:
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    a = original / "rollout-a.jsonl"
    b = replacement / ("rollout-a.jsonl" if topology == "ancestor" else "rollout-b.jsonl")
    a.write_text("{}\n")
    b.write_text("{}\n")
    live_set = LiveSessionSet(max_subscriptions=1, idle_threshold=300, is_macos=True)
    live_set.seed([(a, 1), (b, 1)], monotonic_now=1)
    a.unlink()
    if topology == "ancestor":
        original.rmdir()
        original.symlink_to(replacement, target_is_directory=True)
    elif topology == "chain":
        middle = tmp_path / "middle.jsonl"
        middle.symlink_to(b)
        a.symlink_to(middle)
    else:
        a.symlink_to(b)
    handler = SessionFileHandler(
        DebouncedIndexQueue(), [CodexParser(roots=(tmp_path,))], live_set=live_set
    )
    handler.on_modified(SimpleNamespace(src_path=str(a), is_directory=False))
    roster = live_set.roster()
    sql, params = live_roster_query(
        watched_paths=roster,
        include_idle=False,
        now=datetime(2026, 11, 10),
        idle_window_seconds=86400,
    )
    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=8)
        rows = conn.execute(
            sql + "SELECT source_path FROM classified ORDER BY source_path", params
        ).fetchall()
    assert rows == [(str(b),)]
    assert roster.member_count == 2


@pytest.mark.parametrize("fresh", [False, True])
def test_live_resolves_topology_once_per_database_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh: bool
) -> None:
    a = tmp_path / "rollout-a.jsonl"
    b = tmp_path / "rollout-b.jsonl"
    link = tmp_path / "rollout-link.jsonl"
    a.write_text("{}\n")
    b.write_text("{}\n")
    link.symlink_to(a)
    live_set = LiveSessionSet(max_subscriptions=1, idle_threshold=300, is_macos=True)
    live_set.seed([(link, 1)], monotonic_now=1)
    config = AppConfig(
        data_dir=tmp_path,
        db_path=tmp_path / "index.duckdb",
        lock_path=tmp_path / "index.lock",
        config_path=tmp_path / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
        sources={
            name: SourceConfig(roots=())
            for name in ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
        },
    )
    server = RpcServer(config)
    monkeypatch.setattr(server, "_watch_runtime", SimpleNamespace(live_set=live_set))
    phases = []
    resolutions = []
    original = os.path.realpath

    def resolve(path, *args, **kwargs):
        if Path(path) == link:
            resolutions.append(path)
        return original(path, *args, **kwargs)

    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=8)

        async def snapshot(key, epoch, read):
            response = read(conn)
            phases.append([row.path for row in response.page.sessions])
            if fresh and len(phases) == 1:
                link.unlink()
                link.symlink_to(b)
            return response

        # Scheduling is controlled to place a topology change between actual
        # probe/final callbacks. Their SQL and result hydration remain real.
        monkeypatch.setattr(server, "_cohort_snapshot", snapshot)
        monkeypatch.setattr(os.path, "realpath", resolve)
        response = asyncio.run(server._read_live({"fresh": fresh, "all": True}, None))
    assert phases == ([[str(a)], [str(b)]] if fresh else [[str(a)]])
    assert [row.path for row in response.page.sessions] == [str(b if fresh else a)]
    assert len(resolutions) == (2 if fresh else 1)
