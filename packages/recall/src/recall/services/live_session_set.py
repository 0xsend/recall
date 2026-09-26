from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class SubscriptionKind(StrEnum):
    FILE = "file"
    DIR = "dir"


@dataclass(frozen=True)
class LiveMember:
    path: Path
    mtime: float
    #: Monotonic instant of the last fsevent seen for this path, or None when no
    #: event has ever fired for it. None is not "an event just now": a member
    #: promoted from an mtime alone carries no event evidence, and stamping the
    #: promotion instant here would fabricate some.
    last_event_at: float | None
    subscription_key: str


@dataclass(frozen=True)
class LiveSetStats:
    count: int
    subscription_count: int
    top_paths: tuple[Path, ...]


class LiveSessionSet:
    """Thread-safe active-session set for REQ-DAEMON-043/044/046/047/052.

    The daemon's watcher layer decides when to promote or demote sessions. This
    class owns the in-memory membership rules and the platform-specific
    subscription surface the watcher must schedule.
    """

    def __init__(self, *, max_subscriptions: int, idle_threshold: float, is_macos: bool) -> None:
        self._max_subscriptions = max_subscriptions
        self._idle_threshold = idle_threshold
        self._is_macos = is_macos
        self._members: dict[Path, LiveMember] = {}
        self._lock = threading.Lock()

    def seed(
        self,
        candidates: Iterable[tuple[Path, float]],
        *,
        monotonic_now: float,
    ) -> None:
        """Replace the live set from (path, wall_mtime) candidates.

        Seeding is an inventory of what is on disk, not an observation of
        activity, so no seeded member gets event evidence: each ages out on its
        own mtime. Stamping the daemon's start instant here instead made every
        transcript on disk un-demotable for a full `idle_threshold` after any
        restart. REQ-DAEMON-043/046 require mtime and last_event_at to live on
        independent timebases (wall clock vs monotonic clock), and seeding
        respects that split.
        """
        with self._lock:
            self._members.clear()
            for path, mtime in candidates:
                self._promote_unlocked(
                    path=path,
                    mtime=mtime,
                    monotonic_now=monotonic_now,
                    bump_event=False,
                    log_promote=False,
                )

    def promote(
        self,
        path: Path,
        mtime: float,
        monotonic_now: float,
        *,
        bump_event: bool,
    ) -> LiveMember | None:
        """Promote a path into the live set or refresh an existing member.

        `bump_event=True` is for fsevent-driven promotions — the caller has
        observed real activity, so `last_event_at` moves forward. `bump_event=
        False` is for discovery-loop re-sweeps: the mtime is refreshed but
        `last_event_at` is preserved for existing members, so a quiet file
        still ages out of the live set per REQ-DAEMON-046.
        """
        with self._lock:
            return self._promote_unlocked(
                path=path,
                mtime=mtime,
                monotonic_now=monotonic_now,
                bump_event=bump_event,
                log_promote=True,
            )

    def demote_idle(
        self,
        *,
        wall_now: float,
        monotonic_now: float,
    ) -> list[LiveMember]:
        """Remove members whose mtime and last_event_at are both idle.

        REQ-DAEMON-046: a live member is demoted when its wall-clock mtime has
        been unchanged for `idle_threshold` seconds AND no fsevent has fired
        against it in the same monotonic window. The two cutoffs live on
        independent clocks so they MUST be compared separately. A member that
        has never had an event carries no evidence on the monotonic axis, so
        its mtime decides alone.
        """
        with self._lock:
            mtime_cutoff = wall_now - self._idle_threshold
            event_cutoff = monotonic_now - self._idle_threshold
            stale_members = sorted(
                (
                    member
                    for member in self._members.values()
                    if member.mtime < mtime_cutoff
                    and (member.last_event_at is None or member.last_event_at < event_cutoff)
                ),
                key=lambda member: (member.mtime, str(member.path)),
            )
            for member in stale_members:
                del self._members[member.path]
                logger.info(
                    "demoted live session path=%s reason=idle mtime=%s last_event_at=%s "
                    "mtime_cutoff=%s event_cutoff=%s",
                    member.path,
                    member.mtime,
                    member.last_event_at,
                    mtime_cutoff,
                    event_cutoff,
                )
            return list(stale_members)

    def members(self) -> tuple[LiveMember, ...]:
        """Snapshot every live member, most recently modified first.

        `stats()` caps its path list at 10 because it feeds a status line.
        Deriving liveness (REQ-LIVE-001) needs the whole set, and needs a
        snapshot rather than a view — the watcher promotes and demotes from
        other threads while a caller reads.
        """
        with self._lock:
            return tuple(self._members_sorted_unlocked())

    def iter_subscriptions(self) -> list[tuple[str, SubscriptionKind]]:
        with self._lock:
            subscriptions = self._subscription_items_unlocked()
            return list(subscriptions)

    def stats(self) -> LiveSetStats:
        with self._lock:
            top_paths = tuple(member.path for member in self._members_sorted_unlocked()[:10])
            return LiveSetStats(
                count=len(self._members),
                subscription_count=len(self._subscription_items_unlocked()),
                top_paths=top_paths,
            )

    def _members_sorted_unlocked(self) -> list[LiveMember]:
        """Members most recently modified first; ties broken by path."""
        return sorted(
            self._members.values(),
            key=lambda member: (-member.mtime, str(member.path)),
        )

    def _promote_unlocked(
        self,
        *,
        path: Path,
        mtime: float,
        monotonic_now: float,
        bump_event: bool,
        log_promote: bool,
    ) -> LiveMember | None:
        existing = self._members.get(path)
        if existing is not None:
            # Discovery re-sweeps must NOT reset last_event_at for an already-live
            # member — otherwise a quiet file can never age out. Only fsevent-
            # driven promotions (bump_event=True) move the monotonic stamp
            # forward. See REQ-DAEMON-046.
            new_last_event_at = monotonic_now if bump_event else existing.last_event_at
            refreshed = LiveMember(
                path=path,
                mtime=mtime,
                last_event_at=new_last_event_at,
                subscription_key=existing.subscription_key,
            )
            self._members[path] = refreshed
            if log_promote:
                logger.info(
                    "refreshed live session path=%s mtime=%s bump_event=%s",
                    path,
                    mtime,
                    bump_event,
                )
            return refreshed

        candidate = LiveMember(
            path=path,
            mtime=mtime,
            last_event_at=monotonic_now if bump_event else None,
            subscription_key=self._subscription_key_for_path(path),
        )

        self._members[path] = candidate
        if log_promote:
            logger.info("promoted live session path=%s reason=active mtime=%s", path, mtime)
        return candidate

    def _subscription_items_unlocked(self) -> list[tuple[str, SubscriptionKind]]:
        kind = SubscriptionKind.DIR if self._is_macos else SubscriptionKind.FILE
        # Subscription capacity bounds only the event backend.  It cannot
        # define the roster: a dropped path would be absent from `recall live`
        # and freshness even though inventory can still reconcile it.  Prefer
        # the newest key deterministically; discovery supplies level-triggered
        # coverage for every retained member outside this bounded edge set.
        newest_by_key: dict[str, LiveMember] = {}
        for member in self._members.values():
            previous = newest_by_key.get(member.subscription_key)
            if previous is None or (member.mtime, str(member.path)) > (
                previous.mtime,
                str(previous.path),
            ):
                newest_by_key[member.subscription_key] = member
        selected_keys = [
            key
            for key, _member in sorted(
                newest_by_key.items(), key=lambda item: (-item[1].mtime, item[0])
            )[: self._max_subscriptions]
        ]
        return [(key, kind) for key in sorted(selected_keys)]

    def _subscription_key_for_path(self, path: Path) -> str:
        if self._is_macos:
            return str(path.parent)
        return str(path)
