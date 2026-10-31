"""The catalog-completion predicate, driven with an injected clock."""

from __future__ import annotations

import pytest

from scripts.recall_bench.catalog import CatalogNotReady, CatalogWatch


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def row(source: str, complete: bool, started: float | None, finished: float | None) -> dict:
    return {
        "source": source,
        "root_path": f"/run/pool/{source}",
        "scan_complete": complete,
        "scan_started_at": started,
        "scan_finished_at": finished,
    }


def sample(complete: bool, *rows: dict) -> dict:
    return {"catalog_scan_complete": complete, "coverage": list(rows)}


def test_a_complete_catalog_is_complete_at_once() -> None:
    clock = Clock()
    watch = CatalogWatch(clock)
    clock.now += 2
    complete = watch.observe(sample(True, row("codex", True, 10.0, 11.0)))
    assert complete is not None
    assert (complete.seconds, complete.recovered) == (2.0, ())


def test_an_incomplete_root_must_finish_a_scan_started_after_it_was_seen() -> None:
    clock = Clock()
    watch = CatalogWatch(clock)
    assert watch.observe(sample(False, row("codex", False, 50.0, 40.0))) is None
    clock.now += 5
    complete = watch.observe(sample(True, row("codex", True, 50.0, 60.0)))
    assert complete is not None
    assert [
        (item.source, item.scan_started_at, item.scan_finished_at) for item in complete.recovered
    ] == [("codex", 50.0, 60.0)]


def test_a_complete_flag_without_a_later_finish_is_refused() -> None:
    watch = CatalogWatch(Clock())
    watch.observe(sample(False, row("codex", False, 50.0, 40.0)))
    with pytest.raises(CatalogNotReady, match="no scan finish after its scan start"):
        watch.observe(sample(True, row("codex", True, 50.0, 40.0)))


def test_the_latest_scan_start_seen_is_the_one_that_must_finish() -> None:
    watch = CatalogWatch(Clock())
    watch.observe(sample(False, row("codex", False, 50.0, None)))
    watch.observe(sample(False, row("codex", False, 70.0, 60.0)))
    with pytest.raises(CatalogNotReady, match="no scan finish after its scan start"):
        watch.observe(sample(True, row("codex", True, 70.0, 65.0)))


def test_empty_coverage_is_refused() -> None:
    with pytest.raises(CatalogNotReady, match="coverage is empty"):
        CatalogWatch(Clock()).observe(sample(True))


def test_an_incomplete_catalog_must_name_an_incomplete_root() -> None:
    with pytest.raises(CatalogNotReady, match="names no incomplete root"):
        CatalogWatch(Clock()).observe(sample(False, row("codex", True, 1.0, 2.0)))


def test_a_root_seen_before_its_first_scan_must_then_finish_one() -> None:
    watch = CatalogWatch(Clock())
    assert watch.observe(sample(False, row("codex", False, None, None))) is None
    complete = watch.observe(sample(True, row("codex", True, 80.0, 90.0)))
    assert complete is not None
    assert [(item.scan_started_at, item.scan_finished_at) for item in complete.recovered] == [
        (None, 90.0)
    ]


def test_a_root_seen_before_its_first_scan_without_a_finish_is_refused() -> None:
    watch = CatalogWatch(Clock())
    watch.observe(sample(False, row("codex", False, None, None)))
    with pytest.raises(CatalogNotReady, match="has no finished scan"):
        watch.observe(sample(True, row("codex", True, None, None)))


def test_a_dip_that_outlasts_the_recovery_bound_fails() -> None:
    clock = Clock()
    watch = CatalogWatch(clock, recovery_seconds=300)
    assert watch.observe(sample(False, row("codex", False, 50.0, None))) is None
    clock.now += 300.5
    with pytest.raises(CatalogNotReady, match="did not recover within 300 s"):
        watch.observe(sample(False, row("codex", False, 50.0, None)))
