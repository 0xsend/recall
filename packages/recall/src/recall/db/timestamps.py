"""Equality keys for DuckDB's existing naive timestamp storage."""

from __future__ import annotations

from datetime import datetime
from typing import overload


@overload
def timestamp_for_compare(value: datetime | None) -> datetime | None: ...


@overload
def timestamp_for_compare(value: object) -> object: ...


def timestamp_for_compare(value: object) -> object:
    """Compare parsed and stored values in the default connection's local wall time.

    Aware bindings round-trip through DuckDB's naive TIMESTAMP columns as local
    wall time. Naive values already read from those columns stay unchanged.
    This is an equality projection only; writers retain their storage contract.
    """
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone().replace(tzinfo=None)
    return value
