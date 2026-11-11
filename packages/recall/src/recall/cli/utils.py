from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import typer
from pydantic import BaseModel


def json_default(value: Any):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def print_json(data: Any) -> None:
    typer.echo(json.dumps(data, default=json_default, indent=2))


def format_datetime(value: datetime | str | None) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return value
    return value.strftime("%Y-%m-%d %H:%M")


def source_unreadable_note(session: dict[str, Any]) -> str | None:
    """Explain the source limitation carried by the same payload as JSON."""
    limitations = (session.get("freshness") or {}).get("limitations") or ()
    if "source_unreadable" not in limitations:
        return None
    return (
        "transcript file could not be read; showing stored content."
        " Check that the transcript file exists and is readable."
    )


def stale_after_fresh_note(rows: list[dict[str, Any]]) -> str | None:
    """Name the rows a `--fresh` read failed to catch up (REQ-LIVE-010).

    Derived from the freshness each row already carries rather than from a
    separate signal off the daemon: the answer itself is what the caller acts
    on, so the answer is what the caveat must describe. `None` when nothing is
    behind — the note exists to contradict the flag's promise, not to narrate
    a kept one. Never-indexed rows were not eligible for `live --fresh` catch-up,
    so they are named without claiming a timeout.
    """
    behind = [row for row in rows if not (row.get("freshness") or {}).get("current")]
    if not behind:
        return None
    # Each row has one category containing all of its actionable explanations.
    # Unreadability takes precedence over never-indexed, then ordinary lag.
    categories = Counter(
        (
            source_unreadable_note(row),
            "not_yet_indexed" in ((row.get("freshness") or {}).get("limitations") or ()),
        )
        for row in behind
    )
    first_index_guidance = (
        " live --fresh does not first-index — wait for the coordinator"
        " or use show --fresh once the session is known"
    )
    parts: list[str] = []
    for (unreadable_note, not_yet), count in sorted(
        categories.items(), key=lambda item: (item[0][1], item[0][0] is not None)
    ):
        prefix = f"{count} of {len(rows)} session(s)"
        if unreadable_note is not None:
            explanation = f": {unreadable_note}"
            if not_yet:
                explanation += " These sessions are not yet indexed;" + first_index_guidance
        elif not_yet:
            explanation = " are not yet indexed;" + first_index_guidance
        else:
            explanation = (
                " are still behind the transcript;"
                " inspect freshness for remaining lag or source limitations"
            )
        parts.append(prefix + explanation)
    return "note: " + " ".join(part.removesuffix(".") + "." for part in parts)
