from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

_DURATION_RE = re.compile(r"^(\d+)([smhdw])$")


def parse_since(value: str, now: datetime | None = None) -> datetime:
    if not value:
        raise ValueError("since value is empty")
    match = _DURATION_RE.match(value)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        delta = _duration_delta(amount, unit)
        base = now or datetime.now(UTC)
        return base - delta

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as err:
        raise ValueError(f"invalid since value: {value}") from err

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _duration_delta(amount: int, unit: str) -> timedelta:
    match unit:
        case "s":
            return timedelta(seconds=amount)
        case "m":
            return timedelta(minutes=amount)
        case "h":
            return timedelta(hours=amount)
        case "d":
            return timedelta(days=amount)
        case "w":
            return timedelta(weeks=amount)
        case _:
            raise ValueError(f"unsupported duration unit: {unit}")


def utcnow_naive() -> datetime:
    """Current UTC wall-clock as a *naive* datetime, for DuckDB TIMESTAMP columns.

    DuckDB's naive ``TIMESTAMP`` columns down-convert an *aware* datetime to the
    process-local zone and drop tzinfo on insert, so an aware ``datetime.now(UTC)``
    lands on disk as local wall-clock. Readers that re-attach UTC (e.g.
    ``cli.status_notices._attach_utc_to_naive``) then over-report age by the host's
    UTC offset — a false "daemon appears stale" on any non-UTC host. Storing naive
    UTC keeps the on-disk value unambiguous and matches the reader's UTC assumption
    everywhere.
    """
    return datetime.now(UTC).replace(tzinfo=None)


def to_naive_utc(value: datetime) -> datetime:
    """Normalize a datetime to naive UTC for DuckDB TIMESTAMP storage.

    Aware inputs are converted to UTC; naive inputs are assumed already-UTC. Either
    way tzinfo is stripped so the value stores verbatim — see ``utcnow_naive`` for
    why aware datetimes must never reach a naive TIMESTAMP column.
    """
    if value.tzinfo is not None:
        value = value.astimezone(UTC)
    return value.replace(tzinfo=None)
