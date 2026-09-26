"""Performance evidence must fail closed on missing measurements and noisy regressions."""

from __future__ import annotations

import pytest

from scripts.benchmark_reconciliation import (
    BYTE_KEYS,
    LATENCY_KEYS,
    _floors_breached,
    _latency_allowed,
)


def valid_sample() -> dict[str, object]:
    return {
        **dict.fromkeys(LATENCY_KEYS, 0.1),
        **dict.fromkeys(BYTE_KEYS, 1024),
        "overlapping_reads": 1,
        "overlapping_reads_truthful": True,
        "wal_bytes_after": 0,
        "noop_wal_bytes_before": 100,
        "noop_wal_bytes_after": 100,
    }


def test_benchmark_requires_complete_finite_measurements() -> None:
    assert _floors_breached(valid_sample()) == []
    for key in LATENCY_KEYS + BYTE_KEYS:
        for invalid in (None, float("nan"), float("inf"), -1, True):
            sample = valid_sample()
            sample[key] = invalid
            assert _floors_breached(sample), (key, invalid)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("first_writer_s", 5.01),
        ("checkpoint_s", 5.01),
        ("peak_rss_bytes", 4 * 1024**3 + 1),
        ("overlapping_reads", 0),
        ("overlapping_read_max_s", 2.01),
        ("live_roster_max_s", 2.01),
        ("overlapping_reads_truthful", False),
        ("wal_bytes_after", 1),
        ("noop_wal_bytes_after", 101),
    ],
)
def test_benchmark_rejects_hard_floor_breaches(key: str, value: object) -> None:
    sample = valid_sample()
    sample[key] = value
    assert _floors_breached(sample)


def test_timing_tolerance_is_declared_slack_not_fastest_sample() -> None:
    assert _latency_allowed(0.1, 0.13)
    assert not _latency_allowed(0.1, 0.131)
    assert _latency_allowed(2.0, 2.45)
    assert not _latency_allowed(2.0, 2.451)
