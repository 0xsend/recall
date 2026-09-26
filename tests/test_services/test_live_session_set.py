from __future__ import annotations

from pathlib import Path

from recall.services.live_session_set import LiveSessionSet, SubscriptionKind


def test_seed_empty_iterable_leaves_no_members_or_subscriptions() -> None:
    live_set = LiveSessionSet(max_subscriptions=3, idle_threshold=30.0, is_macos=False)

    live_set.seed([], monotonic_now=0.0)

    assert live_set.stats().count == 0
    assert live_set.stats().subscription_count == 0
    assert live_set.iter_subscriptions() == []


def test_promote_grows_set_until_capacity_without_rejection(tmp_path: Path) -> None:
    live_set = LiveSessionSet(max_subscriptions=3, idle_threshold=30.0, is_macos=False)

    members = [
        live_set.promote(
            tmp_path / f"session-{index}.jsonl",
            mtime=100.0 + index,
            monotonic_now=200.0 + index,
            bump_event=True,
        )
        for index in range(3)
    ]

    assert all(member is not None for member in members)
    assert live_set.stats().count == 3
    assert [path for path, kind in live_set.iter_subscriptions()] == sorted(
        str(tmp_path / f"session-{index}.jsonl") for index in range(3)
    )
    assert {kind for _, kind in live_set.iter_subscriptions()} == {SubscriptionKind.FILE}


def test_promote_keeps_roster_members_when_subscription_capacity_is_reached(tmp_path: Path) -> None:
    live_set = LiveSessionSet(max_subscriptions=2, idle_threshold=30.0, is_macos=False)
    oldest_path = tmp_path / "oldest.jsonl"
    newer_path = tmp_path / "newer.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"

    live_set.promote(oldest_path, mtime=10.0, monotonic_now=50.0, bump_event=True)
    live_set.promote(newer_path, mtime=20.0, monotonic_now=60.0, bump_event=True)

    promoted = live_set.promote(candidate_path, mtime=30.0, monotonic_now=70.0, bump_event=True)

    assert promoted is not None
    assert promoted.path == candidate_path
    assert oldest_path in live_set._members
    assert candidate_path in live_set._members
    assert [member.path for member in live_set.members()] == [
        candidate_path,
        newer_path,
        oldest_path,
    ]
    assert live_set.iter_subscriptions() == [
        (str(candidate_path), SubscriptionKind.FILE),
        (str(newer_path), SubscriptionKind.FILE),
    ]


def test_promote_keeps_an_older_candidate_outside_the_subscription_schedule(tmp_path: Path) -> None:
    live_set = LiveSessionSet(max_subscriptions=2, idle_threshold=30.0, is_macos=False)
    newer_paths = [tmp_path / "newer-a.jsonl", tmp_path / "newer-b.jsonl"]
    rejected_path = tmp_path / "rejected.jsonl"

    live_set.promote(newer_paths[0], mtime=50.0, monotonic_now=100.0, bump_event=True)
    live_set.promote(newer_paths[1], mtime=60.0, monotonic_now=110.0, bump_event=True)
    promoted = live_set.promote(rejected_path, mtime=40.0, monotonic_now=120.0, bump_event=True)

    assert promoted is not None
    assert rejected_path in live_set._members
    assert live_set.iter_subscriptions() == [
        (str(newer_paths[0]), SubscriptionKind.FILE),
        (str(newer_paths[1]), SubscriptionKind.FILE),
    ]


def test_promote_discovery_refresh_preserves_last_event_at(tmp_path: Path) -> None:
    """REQ-DAEMON-046: discovery re-sweeps must not bump last_event_at for
    existing members, otherwise a quiet file can never age out."""
    live_set = LiveSessionSet(max_subscriptions=2, idle_threshold=30.0, is_macos=False)
    path = tmp_path / "quiet.jsonl"

    live_set.promote(path, mtime=10.0, monotonic_now=100.0, bump_event=True)
    # Discovery sweep later: mtime unchanged, fake clock has moved forward,
    # but bump_event=False means the live member must retain its original
    # last_event_at so demote_idle can still age it out.
    live_set.promote(path, mtime=10.0, monotonic_now=500.0, bump_event=False)

    member = live_set._members[path]
    assert member.last_event_at == 100.0
    assert member.mtime == 10.0


def test_promote_fsevent_bumps_last_event_at(tmp_path: Path) -> None:
    live_set = LiveSessionSet(max_subscriptions=2, idle_threshold=30.0, is_macos=False)
    path = tmp_path / "active.jsonl"

    live_set.promote(path, mtime=10.0, monotonic_now=100.0, bump_event=True)
    live_set.promote(path, mtime=20.0, monotonic_now=500.0, bump_event=True)

    member = live_set._members[path]
    assert member.last_event_at == 500.0
    assert member.mtime == 20.0


def test_demote_idle_requires_both_stale_mtime_and_last_event_at(tmp_path: Path) -> None:
    """REQ-DAEMON-046: a member is only demoted when both the wall-clock
    mtime AND the monotonic last_event_at are past their respective cutoffs.

    The two stamps live on independent clocks (wall vs monotonic), so the
    test uses distinct wall_now and monotonic_now values to make sure the
    implementation does not conflate them.
    """
    live_set = LiveSessionSet(max_subscriptions=3, idle_threshold=50.0, is_macos=False)
    stale_path = tmp_path / "stale.jsonl"
    touched_path = tmp_path / "touched.jsonl"
    fresh_path = tmp_path / "fresh.jsonl"

    # Wall mtimes (epoch seconds) for three files.
    wall_now = 10_000.0
    # Monotonic stamps (arbitrary timebase) for last_event_at.
    live_set.promote(stale_path, mtime=9_000.0, monotonic_now=100.0, bump_event=True)
    live_set.promote(touched_path, mtime=9_000.0, monotonic_now=9_999.0, bump_event=True)
    live_set.promote(fresh_path, mtime=9_990.0, monotonic_now=9_999.0, bump_event=True)

    demoted = live_set.demote_idle(wall_now=wall_now, monotonic_now=10_000.0)

    assert [member.path for member in demoted] == [stale_path]
    assert stale_path not in live_set._members
    assert touched_path in live_set._members
    assert fresh_path in live_set._members


def test_iter_subscriptions_linux_returns_file_targets(tmp_path: Path) -> None:
    live_set = LiveSessionSet(max_subscriptions=3, idle_threshold=30.0, is_macos=False)
    first = tmp_path / "a" / "one.jsonl"
    second = tmp_path / "b" / "two.jsonl"

    live_set.promote(first, mtime=10.0, monotonic_now=10.0, bump_event=True)
    live_set.promote(second, mtime=20.0, monotonic_now=20.0, bump_event=True)

    assert live_set.iter_subscriptions() == [
        (str(first), SubscriptionKind.FILE),
        (str(second), SubscriptionKind.FILE),
    ]


def test_iter_subscriptions_macos_collapses_to_distinct_parent_dirs(tmp_path: Path) -> None:
    live_set = LiveSessionSet(max_subscriptions=3, idle_threshold=50.0, is_macos=True)
    first = tmp_path / "same" / "one.jsonl"
    second = tmp_path / "same" / "two.jsonl"

    # `first` becomes idle well before `second`, but the parent-dir
    # subscription must remain scheduled until the LAST live child demotes.
    live_set.promote(first, mtime=100.0, monotonic_now=100.0, bump_event=True)
    live_set.promote(second, mtime=300.0, monotonic_now=300.0, bump_event=True)

    assert live_set.iter_subscriptions() == [
        (str(first.parent), SubscriptionKind.DIR),
    ]

    # First file is idle by both clocks; second is still fresh.
    demoted_first = live_set.demote_idle(wall_now=170.0, monotonic_now=170.0)
    assert [member.path for member in demoted_first] == [first]
    assert live_set.iter_subscriptions() == [
        (str(first.parent), SubscriptionKind.DIR),
    ]

    # Now the second file ages out too.
    demoted_second = live_set.demote_idle(wall_now=400.0, monotonic_now=400.0)
    assert [member.path for member in demoted_second] == [second]
    assert live_set.iter_subscriptions() == []


def test_stats_returns_sorted_top_paths_and_subscription_count(tmp_path: Path) -> None:
    live_set = LiveSessionSet(max_subscriptions=20, idle_threshold=30.0, is_macos=False)
    expected_paths: list[Path] = []
    for index in range(12):
        path = tmp_path / f"session-{index:02d}.jsonl"
        live_set.promote(path, mtime=float(index), monotonic_now=100.0 + index, bump_event=True)
        expected_paths.append(path)

    stats = live_set.stats()

    assert stats.count == 12
    assert stats.subscription_count == 12
    assert stats.top_paths == tuple(reversed(expected_paths[-10:]))


def test_members_returns_every_live_path_most_recently_modified_first(tmp_path: Path) -> None:
    """`stats()` caps its path list at 10 for a status line; liveness needs all of them."""
    live_set = LiveSessionSet(max_subscriptions=20, idle_threshold=30.0, is_macos=False)
    for index in range(12):
        live_set.promote(
            tmp_path / f"session-{index:02d}.jsonl",
            mtime=100.0 + index,
            monotonic_now=200.0 + index,
            bump_event=True,
        )

    members = live_set.members()

    assert len(members) == 12
    assert members[0].path == tmp_path / "session-11.jsonl"
    assert members[0].mtime == 111.0
    assert members[-1].path == tmp_path / "session-00.jsonl"
    assert len(live_set.stats().top_paths) == 10


def test_roster_is_not_limited_by_the_subscription_budget(tmp_path: Path) -> None:
    """A bounded watcher schedule must not make older active sessions invisible."""
    live_set = LiveSessionSet(max_subscriptions=2, idle_threshold=30.0, is_macos=False)
    for index in range(65):
        live_set.promote(
            tmp_path / f"session-{index:02d}.jsonl",
            mtime=float(index),
            monotonic_now=float(index),
            bump_event=True,
        )

    assert len(live_set.members()) == 65
    assert len(live_set.iter_subscriptions()) == 2
    assert [member.path for member in live_set.members()[-2:]] == [
        tmp_path / "session-01.jsonl",
        tmp_path / "session-00.jsonl",
    ]


def test_members_snapshot_is_unaffected_by_a_later_promotion(tmp_path: Path) -> None:
    live_set = LiveSessionSet(max_subscriptions=5, idle_threshold=30.0, is_macos=False)
    live_set.promote(tmp_path / "first.jsonl", mtime=10.0, monotonic_now=50.0, bump_event=True)

    snapshot = live_set.members()
    live_set.promote(tmp_path / "second.jsonl", mtime=20.0, monotonic_now=60.0, bump_event=True)

    assert [member.path for member in snapshot] == [tmp_path / "first.jsonl"]
    assert len(live_set.members()) == 2


def test_a_seeded_member_ages_out_on_the_files_clock_not_the_daemons(tmp_path: Path) -> None:
    """A daemon restart must not reset how long a transcript has been quiet.

    Found by the U26 bug bash. Seeding stamped every member's `last_event_at`
    with the daemon's own start instant, so `demote_idle`'s AND could not be
    satisfied for a full `idle_threshold` no matter how stale the file was. A
    session whose last write was 297 s before the daemon started was seeded
    (inside the 300 s window), then held `active` by plain `recall live` for
    468 s of real quiet, demoting only 300 s after startup. `recall daemon
    restart` is a routine post-upgrade step, so this is a normal-operations
    path.
    """
    path = tmp_path / "quiet-since-before-the-restart.jsonl"
    live_set = LiveSessionSet(max_subscriptions=4, idle_threshold=300.0, is_macos=False)
    daemon_start_wall = 10_000.0

    live_set.seed([(path, daemon_start_wall - 297.0)], monotonic_now=500.0)
    demoted = live_set.demote_idle(wall_now=daemon_start_wall + 10.0, monotonic_now=510.0)

    assert [member.path for member in demoted] == [path]


def test_a_discovery_promotion_ages_out_on_the_files_clock_too(tmp_path: Path) -> None:
    """Same hole without a restart: a sweep promotes a file already near the edge."""
    path = tmp_path / "promoted-near-the-edge.jsonl"
    live_set = LiveSessionSet(max_subscriptions=4, idle_threshold=300.0, is_macos=False)

    live_set.promote(path, mtime=10_000.0 - 299.0, monotonic_now=500.0, bump_event=False)
    demoted = live_set.demote_idle(wall_now=10_000.0 + 2.0, monotonic_now=502.0)

    assert [member.path for member in demoted] == [path]
