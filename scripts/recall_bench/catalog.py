"""When a daemon's catalog counts as complete, judged from successive status samples.

A sample is the `reconciliation` object of `recall daemon status --json`. The
catalog is complete when the daemon reports a complete scan and every root
that was seen incomplete has since finished a scan that started no earlier
than the one it was seen in. A root seen before its first scan started (a
fresh daemon on a newly configured root) must have finished a scan. A dip
that does not recover within the recovery bound is a failure, never a wait
without end.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

RECOVERY_SECONDS = 300.0
Root = tuple[str, str]


class CatalogNotReady(Exception):
    """The catalog cannot become complete from what was observed."""


@dataclass(frozen=True)
class RootScan:
    """One coverage row: a source root and its latest scan."""

    source: str
    root_path: str
    scan_complete: bool
    scan_started_at: float | None
    scan_finished_at: float | None

    @property
    def key(self) -> Root:
        return self.source, self.root_path

    @classmethod
    def parse(cls, row: object) -> RootScan:
        if not isinstance(row, dict):
            raise CatalogNotReady("a catalog coverage row is not an object")
        source, root_path = row.get("source"), row.get("root_path")
        if not isinstance(source, str) or not isinstance(root_path, str):
            raise CatalogNotReady(f"a catalog coverage row names no root: {row}")
        return cls(
            source,
            root_path,
            row.get("scan_complete") is True,
            _time(row, "scan_started_at"),
            _time(row, "scan_finished_at"),
        )


def _time(row: Mapping[str, object], name: str) -> float | None:
    value = row.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CatalogNotReady(f"catalog coverage {name} is not a number: {value!r}")
    return float(value)


@dataclass(frozen=True)
class Recovered:
    source: str
    root_path: str
    scan_started_at: float | None
    scan_finished_at: float


@dataclass(frozen=True)
class Complete:
    seconds: float
    recovered: tuple[Recovered, ...]


@dataclass
class CatalogWatch:
    """Feed it every sample in order; it says complete, keep waiting, or raises."""

    clock: Callable[[], float]
    recovery_seconds: float = RECOVERY_SECONDS
    began: float = field(init=False)
    # The latest scan start seen for each root while it was incomplete; None
    # while the root had not started a scan.
    _pending: dict[Root, float | None] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.began = self.clock()

    def observe(self, reconciliation: Mapping[str, object]) -> Complete | None:
        elapsed = self.clock() - self.began
        rows = _rows(reconciliation)
        if not rows:
            raise CatalogNotReady("catalog coverage is empty")
        if elapsed > self.recovery_seconds:
            raise CatalogNotReady(f"catalog did not recover within {self.recovery_seconds:g} s")
        if reconciliation.get("catalog_scan_complete") is True:
            return Complete(elapsed, tuple(self._recovered(rows)))
        incomplete = [row for row in rows.values() if not row.scan_complete]
        if not incomplete:
            raise CatalogNotReady("an incomplete catalog names no incomplete root")
        for row in incomplete:
            starts = [self._pending.get(row.key), row.scan_started_at]
            known = [start for start in starts if start is not None]
            self._pending[row.key] = max(known) if known else None
        return None

    def _recovered(self, rows: Mapping[Root, RootScan]) -> list[Recovered]:
        recovered: list[Recovered] = []
        for key, started in self._pending.items():
            row = rows.get(key)
            finished = row.scan_finished_at if row is not None else None
            if row is None or not row.scan_complete or finished is None:
                raise CatalogNotReady(f"root {key} has no finished scan")
            if started is not None and finished <= started:
                raise CatalogNotReady(f"root {key} has no scan finish after its scan start")
            recovered.append(Recovered(key[0], key[1], started, finished))
        return recovered


def _rows(reconciliation: Mapping[str, object]) -> dict[Root, RootScan]:
    coverage = reconciliation.get("coverage")
    if not isinstance(coverage, list):
        raise CatalogNotReady("status has no catalog coverage list")
    rows = [RootScan.parse(row) for row in coverage]
    return {row.key: row for row in rows}
