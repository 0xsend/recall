"""Shared validation and normalization for filtered stats parameters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from recall.core.time import format_utc_iso as format_utc_iso
from recall.core.time import parse_since
from recall.core.types import Source, parse_source


@dataclass(frozen=True)
class StatsFilters:
    """Validated source/since scope shared across CLI, RPC, poll reads, and fleet fan-out."""

    source: Source | None = None
    since: datetime | None = None
    since_iso: str | None = None

    @property
    def active(self) -> bool:
        return self.source is not None or self.since is not None


def normalize_stats_filters(
    *,
    source: str | None = None,
    since: str | None = None,
    now: datetime | None = None,
) -> StatsFilters:
    """Validate and canonicalize stats filters identically across CLI, RPC, and poll reads."""
    parsed_source: Source | None = None
    if source is not None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must not be empty")
        parsed_source = parse_source(source.strip())

    parsed_since: datetime | None = None
    since_iso: str | None = None
    if since is not None:
        if not isinstance(since, str) or not since.strip():
            raise ValueError("since must not be empty")
        parsed_since = parse_since(since.strip(), now=now)
        since_iso = format_utc_iso(parsed_since)

    return StatsFilters(source=parsed_source, since=parsed_since, since_iso=since_iso)
