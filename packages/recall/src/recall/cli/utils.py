from __future__ import annotations

import json
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
    not_yet = [
        row
        for row in behind
        if "not_yet_indexed" in tuple((row.get("freshness") or {}).get("limitations") or ())
    ]
    timed_out = [row for row in behind if row not in not_yet]
    parts: list[str] = []
    if timed_out:
        parts.append(
            f"{len(timed_out)} of {len(rows)} session(s) are still behind the transcript;"
            " the daemon ran out of its live.fresh_timeout budget before catching them up"
        )
    if not_yet:
        parts.append(
            f"{len(not_yet)} of {len(rows)} session(s) are not yet indexed;"
            " live --fresh does not first-index — wait for the coordinator"
            " or use show --fresh once the session is known"
        )
    return "note: " + " ".join(parts)
