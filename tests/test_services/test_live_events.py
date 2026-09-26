"""The daemon's per-session indexed event (REQ-LIVE-011).

One event source serves both `--fresh` (await a single firing) and `--follow`
(subscribe until a deadline), so what matters here is the channel's discipline:
who wakes, what a subscriber sees when it falls behind, and that nothing is
retained once it leaves.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from recall.services.live_events import (
    SessionIndexed,
    SessionIndexedNotifier,
    TooManySubscriptionsError,
)

PATH = "/home/dev/.claude/projects/proj/live.jsonl"
OTHER_PATH = "/home/dev/.claude/projects/proj/other.jsonl"


def _event(path: str = PATH, *, high_water_idx: int = 7) -> SessionIndexed:
    return SessionIndexed(
        path=path,
        session_id="11223344556677889900aabbccddeeff",
        high_water_idx=high_water_idx,
    )


def test_a_subscriber_receives_the_event_for_its_own_path() -> None:
    async def scenario() -> SessionIndexed | None:
        notifier = SessionIndexedNotifier()
        with notifier.subscribe(PATH) as subscription:
            notifier.publish(_event())
            return await subscription.wait(timeout=1.0)

    event = asyncio.run(scenario())

    assert event is not None
    assert event.path == PATH
    assert event.high_water_idx == 7


def test_an_event_for_another_path_leaves_a_subscriber_waiting() -> None:
    async def scenario() -> SessionIndexed | None:
        notifier = SessionIndexedNotifier()
        with notifier.subscribe(PATH) as subscription:
            notifier.publish(_event(OTHER_PATH))
            return await subscription.wait(timeout=0.05)

    assert asyncio.run(scenario()) is None


def test_the_newest_event_supersedes_one_the_subscriber_has_not_drained() -> None:
    """Level-triggered, edge-woken: the event says when to look, the cursor says what.

    Queueing undrained events behind a slow client would grow without bound and
    buy nothing — the later high-water idx already covers the earlier one.
    """

    async def scenario() -> tuple[SessionIndexed | None, SessionIndexed | None]:
        notifier = SessionIndexedNotifier()
        with notifier.subscribe(PATH) as subscription:
            notifier.publish(_event(high_water_idx=7))
            notifier.publish(_event(high_water_idx=9))
            first = await subscription.wait(timeout=1.0)
            second = await subscription.wait(timeout=0.05)
            return first, second

    first, second = asyncio.run(scenario())

    assert first is not None
    assert first.high_water_idx == 9
    assert second is None


def test_every_subscriber_on_a_path_is_woken() -> None:
    async def scenario() -> list[int | None]:
        notifier = SessionIndexedNotifier()
        with notifier.subscribe(PATH) as first, notifier.subscribe(PATH) as second:
            notifier.publish(_event(high_water_idx=12))
            events = [await first.wait(timeout=1.0), await second.wait(timeout=1.0)]
        return [None if event is None else event.high_water_idx for event in events]

    assert asyncio.run(scenario()) == [12, 12]


def test_leaving_the_subscription_retains_nothing() -> None:
    """A disconnected `--follow` client must not leave the daemon holding state."""
    notifier = SessionIndexedNotifier()

    with notifier.subscribe(PATH):
        assert notifier.subscriber_count(PATH) == 1

    assert notifier.subscriber_count(PATH) == 0
    assert notifier.has_subscribers(PATH) is False
    notifier.publish(_event())
    assert notifier.subscriber_count(PATH) == 0


def test_subscribing_past_the_cap_is_refused_rather_than_unbounded() -> None:
    notifier = SessionIndexedNotifier(max_subscriptions=2)

    with (
        notifier.subscribe(PATH),
        notifier.subscribe(OTHER_PATH),
        pytest.raises(TooManySubscriptionsError),
        notifier.subscribe(PATH),
    ):
        pass

    assert notifier.subscriber_count(PATH) == 0


def test_a_symlinked_prefix_is_the_same_channel(tmp_path: Path) -> None:
    """On macOS `/tmp` is a symlink to `/private/tmp`.

    The watcher sees the path the harness writes and a caller sees the path the
    index stored; if those two spellings were different channels, `--fresh`
    would wait out its whole timeout on a session that had just been indexed.
    """
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    transcript = real_dir / "live.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir)

    async def scenario() -> SessionIndexed | None:
        notifier = SessionIndexedNotifier()
        with notifier.subscribe(str(link_dir / "live.jsonl")) as subscription:
            notifier.publish(_event(str(transcript)))
            return await subscription.wait(timeout=1.0)

    assert asyncio.run(scenario()) is not None
