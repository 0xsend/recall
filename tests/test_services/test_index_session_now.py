"""The daemon primitive behind `--fresh` and `--follow` (REQ-LIVE-011).

`index_session_now` bypasses the debounce queue for one transcript, runs one
incremental parse under the write lock, and then announces it. The risk that
matters is ordering: a subscriber woken before the write is visible reads the
state it was trying to get past.
"""

from __future__ import annotations

import asyncio
import shutil
import threading
from pathlib import Path

import pytest
from recall.core.config import AppConfig
from recall.services.live_events import SessionIndexed
from recall.services.rpc_server import RpcServer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _install_transcript(tmp_path: Path) -> Path:
    projects = tmp_path / ".claude" / "projects" / "proj"
    projects.mkdir(parents=True, exist_ok=True)
    dest = projects / "live.jsonl"
    shutil.copy(FIXTURES / "claude_code" / "live_end_turn.jsonl", dest)
    return dest


def _server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RpcServer:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    return RpcServer(config=AppConfig.load())


def test_the_event_arrives_only_once_its_rows_are_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The named concurrency risk: waking a subscriber before the write commits.

    `live_end_turn.jsonl` is four records, so a subscriber that can see them all
    the moment it wakes reads idx 0..3.
    """
    transcript = _install_transcript(tmp_path)
    server = _server(tmp_path, monkeypatch)

    async def scenario() -> tuple[int | None, int]:
        with server._session_indexed.subscribe(str(transcript)) as subscription:
            await server.index_session_now(transcript)
            event = await subscription.wait(timeout=5.0)
            assert event is not None
            visible = await server._run_readonly(
                lambda conn: conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            )
            return event.high_water_idx, int(visible)

    high_water_idx, visible = asyncio.run(scenario())

    assert high_water_idx == 3
    assert visible == 4


def test_the_write_lock_is_held_while_the_session_is_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = _install_transcript(tmp_path)
    server = _server(tmp_path, monkeypatch)

    from recall.services import coordinator

    original = coordinator.index_single_session
    entered = threading.Event()
    release = threading.Event()

    def blocked_commit(*args, **kwargs):
        entered.set()
        assert release.wait(5), "test did not release the writer"
        return original(*args, **kwargs)

    monkeypatch.setattr(coordinator, "index_single_session", blocked_commit)

    async def scenario() -> None:
        task = asyncio.create_task(server.index_session_now(transcript))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert server._write_lock.locked()
            # A separate read stays available while the writer is occupied.
            visible = await server._run_readonly(
                lambda conn: conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            )
            assert visible == 0
        finally:
            release.set()
            await task
        assert not server._write_lock.locked()
        assert (
            await server._run_readonly(
                lambda conn: conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            )
            == 4
        )

    asyncio.run(scenario())


def test_the_path_is_taken_off_the_debounce_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the drain loop indexes the same transcript again a moment later."""
    from recall.services.watcher import build_live_watch_runtime

    transcript = _install_transcript(tmp_path)
    server = _server(tmp_path, monkeypatch)
    runtime = build_live_watch_runtime(config=server._config)
    runtime.queue.mark(str(transcript), now=100.0)
    server._watch_runtime = runtime

    asyncio.run(server.index_session_now(transcript))

    assert runtime.queue.pending_count() == 0


def test_a_path_no_parser_claims_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, monkeypatch)
    stray = tmp_path / "not-a-transcript.txt"
    stray.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no parser"):
        asyncio.run(server.index_session_now(stray))


def _symlinked_home_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[RpcServer, Path]:
    """A home reached through a symlink, which is the common real arrangement.

    dotfiles-managed `~/.claude`, Silverblue's `/home -> /var/home`, and macOS
    `/tmp -> /private/tmp` all make the harness's spelling of a transcript path
    differ from the resolved one the index stores.
    """
    real = tmp_path / "real"
    (real / ".claude" / "projects" / "proj").mkdir(parents=True)
    shutil.copy(
        FIXTURES / "claude_code" / "live_end_turn.jsonl",
        real / ".claude" / "projects" / "proj" / "live.jsonl",
    )
    link = tmp_path / "link"
    link.symlink_to(real)
    monkeypatch.setenv("HOME", str(link))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    return RpcServer(config=AppConfig.load()), link / ".claude" / "projects" / "proj" / "live.jsonl"


def test_the_event_finds_the_session_when_home_is_reached_through_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reporting "no row yet" for a session just indexed is what `--fresh` exists to avoid."""
    server, transcript = _symlinked_home_server(tmp_path, monkeypatch)

    async def scenario() -> SessionIndexed | None:
        with server._session_indexed.subscribe(str(transcript)) as subscription:
            await server.index_session_now(transcript)
            return await subscription.wait(timeout=5.0)

    event = asyncio.run(scenario())

    assert event is not None
    assert event.session_id is not None
    assert event.high_water_idx == 3


def test_a_caller_may_pass_the_resolved_path_the_index_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--fresh` reads the path out of `sessions.source_path`, which is resolved.

    Refusing it because `watch_roots()` is built from an unresolved `$HOME`
    would fail every fresh read on a host whose home is a symlink.
    """
    server, transcript = _symlinked_home_server(tmp_path, monkeypatch)
    resolved = transcript.resolve()
    assert resolved != transcript

    async def scenario() -> SessionIndexed | None:
        with server._session_indexed.subscribe(str(resolved)) as subscription:
            await server.index_session_now(resolved)
            return await subscription.wait(timeout=5.0)

    event = asyncio.run(scenario())

    assert event is not None
    assert event.high_water_idx == 3
