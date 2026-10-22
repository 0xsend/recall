"""Normalize the scratch numeric storage domain at every DB fetch boundary."""

from __future__ import annotations

from typing import Any


def rows(cursor: Any, values: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    numeric = {
        index
        for index, column in enumerate(cursor.description or ())
        if str(column[1]) == "UHUGEINT"
    }
    if not numeric:
        return values
    return [
        tuple(
            str(value) if index in numeric and value is not None else value
            for index, value in enumerate(row)
        )
        for row in values
    ]


def fetchall(cursor: Any) -> list[tuple[Any, ...]]:
    return rows(cursor, cursor.fetchall())


def fetchmany(cursor: Any, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
    return rows(cursor, cursor.fetchmany(*args, **kwargs))


def fetchone(cursor: Any) -> tuple[Any, ...] | None:
    value = cursor.fetchone()
    return rows(cursor, [value])[0] if value is not None else None
