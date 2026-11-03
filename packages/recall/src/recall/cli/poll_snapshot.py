"""Invocation-scoped poll provenance and coverage, separate from stdout data."""

from __future__ import annotations

import shlex
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import typer

from recall.core.poll_reads import PollSnapshot


@dataclass
class PollInvocation:
    notice_emitted: bool = False
    coverage: dict[str, Any] | None = None


_invocation: ContextVar[PollInvocation | None] = ContextVar("poll_invocation", default=None)


def begin_invocation() -> Token[PollInvocation | None]:
    return _invocation.set(PollInvocation())


def end_invocation(token: Token[PollInvocation | None]) -> None:
    _invocation.reset(token)


def coverage_reading() -> dict[str, Any] | None:
    invocation = _invocation.get()
    return invocation.coverage if invocation is not None else None


def record_snapshot(snapshot: PollSnapshot, data_dir: Path) -> None:
    from recall.cli.status_notices import _format_age

    invocation = _invocation.get()
    if invocation is None:
        invocation = PollInvocation()
    if snapshot.coverage is not None:
        invocation.coverage = {"reconciliation": snapshot.coverage}
    if invocation.notice_emitted:
        return
    timestamp = snapshot.last_successful_at
    if timestamp is None:
        provenance = "last successful index time: unknown"
    else:
        timestamp = (
            timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp.astimezone(UTC)
        )
        provenance = (
            f"last successful index: {timestamp.isoformat()} ({_format_age(timestamp)} ago)"
        )
    typer.echo(
        f"Poll mode: reading the last indexed snapshot; {provenance}; no refresh ran. "
        f"Update with: RECALL_DATA_DIR={shlex.quote(str(data_dir))} recall daemon --once.",
        err=True,
    )
    invocation.notice_emitted = True
