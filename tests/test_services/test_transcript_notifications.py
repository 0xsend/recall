"""Unreadable Codex representations do not break notification bookkeeping."""

from __future__ import annotations

import asyncio
import errno
from dataclasses import replace
from pathlib import Path

import pytest
from recall.parsers.codex import CodexParser
from recall.services.coordinator import capture_watch_observations
from recall.services.live_events import SessionIndexed, SessionIndexedNotifier
from recall.services.watcher import (
    DebouncedIndexQueue,
    SessionFileHandler,
    SourceRootHandler,
    _resolve_parser_for_path,
)
from watchdog.events import FileCreatedEvent, FileModifiedEvent


@pytest.fixture(params=[errno.EACCES, errno.ELOOP, errno.EIO])
def denied_plain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request):
    plain = tmp_path / "rollout-denied.jsonl"
    plain.write_text("{}\n")
    stat = Path.stat

    def denied(path: Path, *args, **kwargs):
        if path == plain:
            raise OSError(request.param, "plain sibling unavailable", str(path))
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    return plain


def test_event_handlers_continue_after_plain_stat_failure(denied_plain: Path) -> None:
    parser = CodexParser(roots=(denied_plain.parent,))
    queue = DebouncedIndexQueue(debounce=0)
    content = SessionFileHandler(queue, [parser])
    root = SourceRootHandler(
        queue,
        str(denied_plain.parent),
        parser.file_pattern,
        content=content,
        clock=content._clock,
    )
    content.on_modified(FileModifiedEvent(str(denied_plain)))
    assert queue.pending_count() == 0
    root.on_created(FileCreatedEvent(str(denied_plain)))
    assert queue.ready() == [str(denied_plain)]
    neighbor = denied_plain.parent / "rollout-neighbor.jsonl"
    neighbor.write_text("{}\n")
    content.on_modified(FileModifiedEvent(str(neighbor)))
    assert queue.ready() == [str(denied_plain), str(neighbor)]


def test_notice_can_settle_after_plain_access_is_lost(tmp_path: Path) -> None:
    path = tmp_path / "rollout-marked.jsonl"
    path.write_text("{}\n")
    queue = DebouncedIndexQueue(debounce=0)
    queue.mark(str(path), now=1)
    path.unlink()
    path.symlink_to(path)
    assert queue.settle(str(path), observed_from=2)
    assert queue.pending_count() == 0
    queue.mark(str(path), now=3)
    assert queue.settle(str(path), observed_from=2) is False
    assert queue.pending_count() == 1
    assert queue.settle(str(path), observed_from=4)


def test_unreadable_plain_and_compressed_names_share_the_notification_channel(
    denied_plain: Path,
) -> None:
    async def scenario() -> None:
        notifier = SessionIndexedNotifier()
        with notifier.subscribe(str(denied_plain)) as subscription:
            event = SessionIndexed(str(denied_plain) + ".zst", "retained-session", 7)
            assert notifier.has_subscribers(event.path)
            notifier.publish(event)
            assert await subscription.wait(timeout=1) == event
        assert notifier.total_subscriptions == 0

    asyncio.run(scenario())


def test_parser_lookup_and_capture_isolate_unreadable_plain(
    denied_plain: Path, tmp_path: Path
) -> None:
    from lane_harness import lane_config
    from recall.core.config import SourceConfig

    config = replace(lane_config(tmp_path), sources={"codex": SourceConfig(roots=(tmp_path,))})
    parser = CodexParser(roots=(tmp_path,))
    assert _resolve_parser_for_path(str(denied_plain), [parser]) is parser
    neighbor = tmp_path / "rollout-neighbor.jsonl"
    neighbor.write_text("{}\n")
    batch, failures = capture_watch_observations(config, (str(denied_plain), str(neighbor)))
    assert [item.source_path for item in batch.files] == [str(neighbor)]
    assert len(failures) == 1
    assert failures[0].path == str(denied_plain)
    assert "plain sibling unavailable" in failures[0].detail


def test_compressed_symlink_cycle_is_a_capture_failure(tmp_path: Path) -> None:
    from lane_harness import lane_config
    from recall.core.config import SourceConfig

    path = tmp_path / "rollout-cycle.jsonl.zst"
    path.symlink_to(path)
    config = replace(lane_config(tmp_path), sources={"codex": SourceConfig(roots=(tmp_path,))})
    batch, failures = capture_watch_observations(config, (str(path),))
    assert not batch.files
    assert len(failures) == 1
    assert "OSError" in failures[0].detail
