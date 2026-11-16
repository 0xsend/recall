"""Live observation preserves visibility and read validation order."""

import asyncio
import os
import threading
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest
from recall.core.config import AppConfig, SourceConfig
from recall.core.types import Source
from recall.db.schema import ensure_schema
from recall.db.source_files import SourceCatalog, SourceSignature
from recall.services import rpc_server, watcher
from recall.services.live_session_set import LiveSessionSet
from recall.services.rpc_server import RpcServer
from recall.services.session_ids import UnknownSessionError

from tests.test_services.test_watcher_live import FakeClock, FakeObserver, FakeParser, _touch


def test_live_status_sees_events_while_root_watch_installation_is_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded = _touch(tmp_path / "seeded.jsonl", mtime=100)
    arriving = _touch(tmp_path / "arriving.jsonl", mtime=101)
    parser = FakeParser(tmp_path, Source.CODEX, [seeded])
    monkeypatch.setattr(watcher, "all_parsers", lambda _sources: [parser])
    installing = threading.Event()
    finish = threading.Event()
    errors: list[BaseException] = []

    def install_roots(runtime) -> None:
        installing.set()
        assert finish.wait(5), "test did not release root installation"

    monkeypatch.setattr(watcher, "_watch_source_roots", install_roots)
    runtime = watcher.build_live_watch_runtime(
        config=AppConfig.load(),
        clock=FakeClock(wall_now=101, monotonic_now=1),
        observer_factory=FakeObserver,
    )

    def start() -> None:
        try:
            watcher.start_live_watch_runtime(runtime)
        except BaseException as error:
            errors.append(error)

    watcher.reset_live_snapshot()
    thread = threading.Thread(target=start)
    try:
        thread.start()
        assert installing.wait(5), "startup did not reach root installation"
        runtime.handler.on_modified(SimpleNamespace(src_path=str(arriving), is_directory=False))
        snapshot = watcher.get_live_snapshot()
        assert snapshot.live_session_count == 2
        assert snapshot.live_session_paths == [str(arriving), str(seeded)]
        assert snapshot.watcher_subscription_count == len(runtime.scheduled_watches)
        assert runtime.queue.pending_count() == 1
        assert thread.is_alive(), "status must be checked before startup completes"
    finally:
        finish.set()
        thread.join(5)
        watcher.stop_live_watch_runtime(runtime)
        watcher.reset_live_snapshot()
    assert not thread.is_alive()
    assert not errors


@pytest.mark.parametrize("invalid_session", [False, True])
def test_live_read_cutoff_and_session_validation_precede_path_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_session: bool
) -> None:
    watched = tmp_path / "rollout-watched.jsonl"
    watched.write_text("{}\n")
    recent = tmp_path / "rollout-recent.jsonl"
    recent.write_text("{}\n")
    live_set = LiveSessionSet(max_subscriptions=1, idle_threshold=300, is_macos=True)
    live_set.seed([(watched, 1)], monotonic_now=1)
    config = replace(
        AppConfig.load(),
        sources={
            name: SourceConfig(roots=())
            for name in ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
        },
    )
    server = RpcServer(config)
    monkeypatch.setattr(server, "_watch_runtime", SimpleNamespace(live_set=live_set))
    instant = datetime(2026, 11, 10, 12)
    elapsed = timedelta(0)
    resolutions: list[str] = []
    original = os.path.realpath

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant + elapsed

    def resolve(path, *args, **kwargs):
        nonlocal elapsed
        if Path(path) == watched:
            resolutions.append(str(path))
            elapsed = timedelta(seconds=0.1)
        return original(path, *args, **kwargs)

    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: instant.timestamp())
        mtime_ns = int((instant.timestamp() - 299.95) * 1e9)
        catalog.observe_batch(
            [("codex", str(tmp_path), str(recent), SourceSignature(1, 2, 1, mtime_ns, 3))]
        )

        async def snapshot(key, epoch, read):
            return read(conn)

        # Only scheduling and time are controlled; validation, SQL and hydration run.
        monkeypatch.setattr(server, "_cohort_snapshot", snapshot)
        monkeypatch.setattr(rpc_server, "datetime", Clock)
        monkeypatch.setattr(os.path, "realpath", resolve)
        if invalid_session:
            with pytest.raises(UnknownSessionError):
                asyncio.run(server._read_live({"session": "does-not-exist"}, None))
            assert resolutions == []
        else:
            response = asyncio.run(server._read_live({}, None))
            assert sorted(row.path for row in response.page.sessions) == sorted(
                [str(watched), str(recent)]
            )
            assert resolutions
