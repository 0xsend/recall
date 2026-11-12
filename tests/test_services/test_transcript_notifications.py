"""Unreadable Codex representations do not break notification bookkeeping."""

from __future__ import annotations

import asyncio
import errno
import os
from dataclasses import replace
from pathlib import Path

import pytest
from recall.parsers.claude_code import ClaudeCodeParser
from recall.parsers.codex import CodexParser
from recall.parsers.discovery import DiscoveryScope
from recall.services.coordinator import capture_watch_observations
from recall.services.live_events import SessionIndexed, SessionIndexedNotifier
from recall.services.watcher import (
    DebouncedIndexQueue,
    SessionFileHandler,
    SourceRootHandler,
    TopologyChanges,
    _resolve_parser_for_path,
)
from watchdog.events import (
    DirDeletedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)


@pytest.mark.parametrize("kind", ["created", "modified", "deleted", "moved"])
@pytest.mark.parametrize("filename", ["activity.json", "not-a-rollout.jsonl"])
def test_non_transcript_events_do_not_scan_or_request_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, filename: str
) -> None:
    root_path = tmp_path / "sessions"
    root_path.mkdir()
    auxiliary = root_path / filename
    auxiliary.write_text("{}")
    parser = CodexParser(roots=(root_path,))
    queue = DebouncedIndexQueue(debounce=0)
    foreign = tmp_path / "other-source"
    foreign.mkdir()
    content = SessionFileHandler(queue, [parser, ClaudeCodeParser(roots=(foreign,))])
    topology = TopologyChanges()
    handler = SourceRootHandler(
        queue,
        str(root_path),
        parser.file_pattern,
        content=content,
        clock=content._clock,
        topology=topology,
    )
    scanned: list[object] = []
    scandir = os.scandir

    def record_scan(path):
        scanned.append(path)
        return scandir(path)

    monkeypatch.setattr(os, "scandir", record_scan)
    match kind:
        case "created":
            handler.on_created(FileCreatedEvent(str(auxiliary)))
        case "modified":
            handler.on_modified(FileModifiedEvent(str(auxiliary)))
        case "deleted":
            auxiliary.unlink()
            handler.on_deleted(FileDeletedEvent(str(auxiliary)))
        case "moved":
            destination = auxiliary.with_suffix(".old")
            auxiliary.rename(destination)
            handler.on_moved(FileMovedEvent(str(auxiliary), str(destination)))
    assert queue.pending_count() == 0
    assert not topology.pending()
    assert scanned == []


def test_a_new_transcript_directory_does_not_scan_other_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "codex"
    foreign = tmp_path / "claude"
    root.mkdir()
    foreign.mkdir()
    queue = DebouncedIndexQueue(debounce=0)
    content = SessionFileHandler(
        queue, [CodexParser(roots=(root,)), ClaudeCodeParser(roots=(foreign,))]
    )
    nested = root / "today"
    nested.mkdir()
    transcript = nested / "rollout-new.jsonl"
    transcript.write_text("{}\n")
    scanned: list[object] = []
    scandir = os.scandir

    def record_scan(path):
        scanned.append(path)
        return scandir(path)

    monkeypatch.setattr(os, "scandir", record_scan)
    content.on_created(FileCreatedEvent(str(transcript)))
    assert queue.ready() == [str(transcript)]
    assert root in scanned
    assert foreign not in scanned


@pytest.mark.parametrize("kind", ["transcript", "directory", "directory_link", "file_link"])
def test_removed_membership_routes_still_request_inventory(tmp_path: Path, kind: str) -> None:
    root_path = tmp_path / "sessions"
    root_path.mkdir()
    match kind:
        case "transcript":
            removed = root_path / "rollout-deleted.jsonl"
            removed.write_text("{}\n")
        case "directory":
            removed = root_path / "nested"
            removed.mkdir()
        case "directory_link":
            target = tmp_path / "target"
            target.mkdir()
            removed = root_path / "opaque-link"
            removed.symlink_to(target, target_is_directory=True)
        case "file_link":
            removed = root_path / "rollout-link.jsonl"
            target = tmp_path / "opaque.data"
            target.write_text("{}\n")
            removed.symlink_to(target)
        case _:
            raise AssertionError(kind)
    parser = CodexParser(roots=(root_path,))
    queue = DebouncedIndexQueue(debounce=0)
    content = SessionFileHandler(queue, [parser])
    topology = TopologyChanges()
    handler = SourceRootHandler(
        queue,
        str(root_path),
        parser.file_pattern,
        content=content,
        clock=content._clock,
        topology=topology,
    )
    if kind == "directory":
        removed.rmdir()
        handler.on_deleted(DirDeletedEvent(str(removed)))
    else:
        removed.unlink()
        handler.on_deleted(FileDeletedEvent(str(removed)))
    assert topology.pending()


def test_renaming_an_auxiliary_file_into_a_transcript_notices_its_content(tmp_path: Path) -> None:
    auxiliary = tmp_path / "activity.json"
    auxiliary.write_text("{}\n")
    parser = CodexParser(roots=(tmp_path,))
    queue = DebouncedIndexQueue(debounce=0)
    content = SessionFileHandler(queue, [parser])
    topology = TopologyChanges()
    handler = SourceRootHandler(
        queue,
        str(tmp_path),
        parser.file_pattern,
        content=content,
        clock=content._clock,
        topology=topology,
    )
    transcript = tmp_path / "rollout-promoted.jsonl"
    auxiliary.rename(transcript)
    handler.on_moved(FileMovedEvent(str(auxiliary), str(transcript)))
    assert topology.pending()
    assert queue.ready() == [str(transcript)]


@pytest.mark.parametrize("kind", ["deleted", "moved"])
@pytest.mark.parametrize("present_at_start", [True, False])
@pytest.mark.parametrize("link_count", [1, 2])
def test_link_removal_after_scope_refresh_still_requests_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    present_at_start: bool,
    link_count: int,
) -> None:
    monkeypatch.setattr("recall.services.watcher.FILE_LINK_LIMIT", 1)
    root = tmp_path / "sessions"
    root.mkdir()
    target = tmp_path / "external"
    target.mkdir()
    route = root / "opaque-link"
    if link_count == 2:
        other_target = tmp_path / "other-external"
        other_target.mkdir()
        (root / "other-link").symlink_to(other_target, target_is_directory=True)
    if present_at_start:
        route.symlink_to(target, target_is_directory=True)
    parser = CodexParser(roots=(root,))
    queue = DebouncedIndexQueue(debounce=0)
    content = SessionFileHandler(queue, [parser])
    topology = TopologyChanges()
    handler = SourceRootHandler(
        queue,
        str(root),
        parser.file_pattern,
        content=content,
        clock=content._clock,
        topology=topology,
    )
    if not present_at_start:
        route.symlink_to(target, target_is_directory=True)
        content.refresh_scopes((DiscoveryScope.build(parser),))
    # Discovery can finish after the filesystem change but before watchdog's
    # callback, so the current scope no longer contains this entrance.
    destination = tmp_path / "moved-out"
    if kind == "deleted":
        route.unlink()
    else:
        route.rename(destination)
        # The destination is outside the watch and may disappear before the
        # queued move callback; the source entrance must suffice on its own.
        destination.unlink()
    content.refresh_scopes((DiscoveryScope.build(parser),))
    if kind == "deleted":
        handler.on_deleted(FileDeletedEvent(str(route)))
    else:
        handler.on_moved(FileMovedEvent(str(route), str(destination)))
    assert topology.pending()
    assert queue.pending_count() == 0


@pytest.mark.parametrize("link_count, scan_on_aux_removal", [(1, False), (2, True)])
def test_a_link_deleted_before_watch_refresh_requests_another_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, link_count: int, scan_on_aux_removal: bool
) -> None:
    monkeypatch.setattr("recall.services.watcher.FILE_LINK_LIMIT", 1)
    root_path = tmp_path / "sessions"
    root_path.mkdir()
    target = tmp_path / "external"
    target.mkdir()
    parser = CodexParser(roots=(root_path,))
    queue = DebouncedIndexQueue(debounce=0)
    content = SessionFileHandler(queue, [parser])
    topology = TopologyChanges()
    handler = SourceRootHandler(
        queue,
        str(root_path),
        parser.file_pattern,
        content=content,
        clock=content._clock,
        topology=topology,
    )
    for index in range(link_count):
        route = root_path / f"opaque-link-{index}"
        route.symlink_to(target, target_is_directory=True)
        handler.on_created(FileCreatedEvent(str(route)))
        handler.on_created(FileCreatedEvent(str(route)))
    assert topology.pending()
    topology.settle(topology.generation())
    auxiliary = root_path / "activity.json"
    auxiliary.write_text("{}")
    auxiliary.unlink()
    handler.on_deleted(FileDeletedEvent(str(auxiliary)))
    assert topology.pending() is scan_on_aux_removal
    topology.settle(topology.generation())
    route.unlink()
    handler.on_deleted(FileDeletedEvent(str(route)))
    assert topology.pending()


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
    observations = capture_watch_observations(config, (str(denied_plain), str(neighbor)))
    assert [item.source_path for item in observations.inventory.files] == [str(neighbor)]
    assert len(observations.failures) == 1
    assert observations.failures[0].path == str(denied_plain)
    assert "plain sibling unavailable" in observations.failures[0].detail


def test_compressed_symlink_cycle_is_a_capture_failure(tmp_path: Path) -> None:
    from lane_harness import lane_config
    from recall.core.config import SourceConfig

    path = tmp_path / "rollout-cycle.jsonl.zst"
    path.symlink_to(path)
    config = replace(lane_config(tmp_path), sources={"codex": SourceConfig(roots=(tmp_path,))})
    observations = capture_watch_observations(config, (str(path),))
    assert not observations.inventory.files
    assert len(observations.failures) == 1
    assert "OSError" in observations.failures[0].detail


def test_index_discovery_skips_identity_failure_without_losing_neighbors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.core.config import SourceConfig
    from recall.core.types import Source
    from recall.services import indexer

    denied = tmp_path / "rollout-denied.jsonl"
    neighbor = tmp_path / "rollout-neighbor.jsonl"
    denied.write_text("{}\n")
    neighbor.write_text("{}\n")
    identity = indexer.transcript_identity

    def unavailable(path: Path) -> Path:
        if path == denied:
            raise PermissionError("identity unavailable after discovery")
        return identity(path)

    monkeypatch.setattr(indexer, "transcript_identity", unavailable)
    paths = indexer._discover_paths(
        Source.CODEX, sources={"codex": SourceConfig(roots=(tmp_path,))}
    )
    assert [path.resolved_path for path in paths] == [str(neighbor)]
