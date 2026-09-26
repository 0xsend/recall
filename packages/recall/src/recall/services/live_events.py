"""The daemon's per-session "indexed" event (REQ-LIVE-011).

One event source serves both `--fresh` (await a single firing) and `--follow`
(subscribe until a deadline), so there is exactly one place that decides "this
transcript's new bytes are committed and readable".

The channel is level-triggered and edge-woken: the event says *when* to look,
and the subscriber's own cursor read decides *what* changed. That is why an
event a subscriber has not drained is replaced rather than queued — a slow or
disconnected client can never make the daemon retain events, and the newer
high-water idx supersedes the older one anyway.

Everything here is loop-affine. `publish` runs on the event loop thread after
the write lock is released, never from the executor.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("recall.live_events")

# A `--follow` client that vanishes without closing its connection would
# otherwise leave a subscription behind, so the total is bounded and the
# refusal is explicit rather than a slow leak.
_DEFAULT_MAX_SUBSCRIPTIONS = 256


class TooManySubscriptionsError(RuntimeError):
    """Raised when the notifier is already holding its maximum subscriptions."""


@dataclass(frozen=True)
class SessionIndexed:
    """One transcript's new bytes are committed and visible to a read cursor.

    `session_id` and `high_water_idx` are None when the index has no row for
    the path yet — the daemon indexes a brand-new transcript and the write can
    still leave nothing readable (an empty or unparseable file).

    `high_water_idx` is a hint, never a read bound. Two publishers (the drain
    loop and `index_session_now`) read it on separate executor threads after
    releasing the write lock, so a lower idx can be published after a higher
    one and depth-one supersession keeps whichever arrived last. A subscriber
    reads from its own cursor and lets the rows decide what is new; one that
    treated this field as the end of its window would occasionally miss
    messages.
    """

    path: str
    session_id: str | None
    high_water_idx: int | None


def live_path_key(path: str) -> str:
    """Canonical spelling of a transcript path.

    Deliberately identical to what every parser writes into
    `sessions.source_path` (`str(path.expanduser().resolve())`), because this
    is both the channel key and the key the index is queried by. The watcher
    sees the path the harness writes, a caller reads the path the index stored,
    and on a host whose `$HOME` or `~/.claude` is a symlink — or on macOS,
    where `/tmp` is one — those are different strings for the same file.
    Without one spelling, `--fresh` waits out its whole timeout on a session
    that was just indexed.
    """
    return str(Path(path).expanduser().resolve())


class SessionIndexedSubscription:
    """One listener's slot on a path's channel. Created by `subscribe`."""

    def __init__(self, key: str) -> None:
        self._key = key
        # Depth one on purpose: see the module docstring on level-triggering.
        self._slot: asyncio.Queue[SessionIndexed] = asyncio.Queue(maxsize=1)

    async def wait(self, *, timeout: float) -> SessionIndexed | None:
        """Await the next event for this path, or None once `timeout` elapses."""
        try:
            return await asyncio.wait_for(self._slot.get(), timeout=timeout)
        except TimeoutError:
            return None

    def _offer(self, event: SessionIndexed) -> None:
        with suppress(asyncio.QueueEmpty):
            self._slot.get_nowait()
        with suppress(asyncio.QueueFull):
            self._slot.put_nowait(event)


class SessionIndexedNotifier:
    """Fan-out of session-indexed events to the clients waiting on each path."""

    def __init__(self, *, max_subscriptions: int = _DEFAULT_MAX_SUBSCRIPTIONS) -> None:
        self._max_subscriptions = max_subscriptions
        self._subscriptions: dict[str, list[SessionIndexedSubscription]] = {}
        self._count = 0

    @contextmanager
    def subscribe(self, path: str) -> Iterator[SessionIndexedSubscription]:
        """Listen for events on `path` for the duration of the block.

        Leaving the block is the cancel-on-disconnect path: the slot is dropped
        whether the client finished, timed out, or the connection died.
        """
        if self._count >= self._max_subscriptions:
            raise TooManySubscriptionsError(
                f"already holding {self._count} session-indexed subscriptions"
            )
        key = live_path_key(path)
        subscription = SessionIndexedSubscription(key)
        self._subscriptions.setdefault(key, []).append(subscription)
        self._count += 1
        try:
            yield subscription
        finally:
            listeners = self._subscriptions.get(key)
            if listeners is not None and subscription in listeners:
                listeners.remove(subscription)
                self._count -= 1
                if not listeners:
                    del self._subscriptions[key]

    def publish(self, event: SessionIndexed) -> None:
        """Wake every subscriber on the event's path. Never blocks."""
        for subscription in self._subscriptions.get(live_path_key(event.path), ()):
            subscription._offer(event)

    def has_subscribers(self, path: str) -> bool:
        return bool(self._subscriptions.get(live_path_key(path)))

    def subscriber_count(self, path: str) -> int:
        return len(self._subscriptions.get(live_path_key(path), ()))

    @property
    def total_subscriptions(self) -> int:
        return self._count
