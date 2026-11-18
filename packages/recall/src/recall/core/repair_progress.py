"""The progress file an owning daemon leaves for readers while it repairs indexes.

`repair-progress.json` is non-authoritative: no durable state, admission or
recovery decision reads it. Its only reader is an operator-facing surface
(`daemon status`, a CLI refusal, the start guards) that already knows, from the
database ownership gate, whether anything still owns the database. The file
carries data only; the readers render the text, so wording changes never change
the file.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger("recall.repair_progress")

PROGRESS_FILENAME = "repair-progress.json"
PROGRESS_TEMP_FILENAME = "repair-progress.json.tmp"
PROGRESS_PARKED_FILENAME = "repair-progress.json.parked"
PROGRESS_BYTES_MAX = 16 * 1024
# The publisher refreshes `updated_at` every 2 seconds; a record older than this
# was written by an owner that is hung, busy or already gone.
HEARTBEAT_SECONDS = 2.0
FRESH_SECONDS = 10.0

Phase = Literal[
    "admitting",
    "capturing_reference",
    "verifying_reference",
    "arming",
    "rebuilding_indexes",
    "checkpointing",
    "verifying_content",
    "probing",
    "committing",
    "retiring",
]
TriggerKind = Literal["index", "probe", "resume"]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ProgressUnit(_Model):
    """Tables handled so far in the current phase; `total` can grow once, `done` never shrinks."""

    unit: Literal["table"] = "table"
    done: int = Field(ge=0)
    total: int = Field(ge=0)
    current: str | None = None
    bytes_done: int = Field(ge=0)


class ProgressRecord(_Model):
    schema_version: Literal[1] = Field(alias="schema")
    owner_pid: int = Field(gt=0)
    attempt_no: int = Field(ge=0)
    trigger_kind: TriggerKind
    phase: Phase
    started_at: float
    phase_started_at: float
    updated_at: float
    progress: ProgressUnit | None = None

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, populate_by_name=True)

    def render(self) -> bytes:
        return json.dumps(self.model_dump(by_alias=True), separators=(",", ":")).encode()


def read_progress(data_dir: Path) -> ProgressRecord | None:
    """The record, or None when the file is missing, unreadable, oversized or invalid."""
    path = data_dir / PROGRESS_FILENAME
    try:
        with path.open("rb") as stream:
            payload = stream.read(PROGRESS_BYTES_MAX + 1)
    except OSError:
        return None
    if len(payload) > PROGRESS_BYTES_MAX:
        return None
    try:
        return ProgressRecord.model_validate_json(payload)
    except ValidationError:
        return None


def remove_progress(data_dir: Path) -> None:
    for name in (PROGRESS_FILENAME, PROGRESS_TEMP_FILENAME, PROGRESS_PARKED_FILENAME):
        try:
            (data_dir / name).unlink()
        except FileNotFoundError:
            continue
        except OSError as error:
            logger.warning("could not remove %s: %s", name, error)


@contextmanager
def progress_parked(data_dir: Path) -> Iterator[None]:
    """Keep a dead repair's record out of readers' sight while an offline writer owns the database.

    A record found under an owner that holds the ownership gate belongs to a repair that is
    gone, and it would read as a live repair for as long as the owner works. The record is
    renamed aside, so nothing the owner does can leave it visible and nothing it refuses or
    fails at can lose it: it is discarded when the body finishes and renamed back when the
    body raises. The caller holds the gate across the whole scope, so no other writer can
    publish a record meanwhile. A process killed inside the scope leaves the parked name,
    which readers ignore and the next owner removes.
    """
    live = data_dir / PROGRESS_FILENAME
    parked = data_dir / PROGRESS_PARKED_FILENAME
    try:
        os.replace(live, parked)
        was_parked = True
    except FileNotFoundError:
        was_parked = False
    except OSError as error:
        logger.warning("could not park %s: %s", PROGRESS_FILENAME, error)
        was_parked = False
    try:
        yield
    except BaseException:
        if was_parked:
            try:
                os.replace(parked, live)
            except OSError as error:
                logger.warning("could not restore %s: %s", PROGRESS_FILENAME, error)
        raise
    remove_progress(data_dir)


PHASE_LABELS: dict[str, str] = {
    "admitting": "reading recovery state",
    "capturing_reference": "capturing the reference",
    "verifying_reference": "verifying the reference",
    "arming": "arming the repair",
    "rebuilding_indexes": "rebuilding indexes",
    "checkpointing": "checkpointing",
    "verifying_content": "verifying content",
    "probing": "probing indexes",
    "committing": "committing the result",
    "retiring": "retiring recovery files",
}


def describe_progress(record: ProgressRecord, *, now: float) -> str:
    """`phase X, table N of M (name), E s elapsed`, for any surface that quotes the repair."""
    parts = [f"phase {PHASE_LABELS[record.phase]}"]
    unit = record.progress
    if unit is not None and unit.total:
        parts.append(f"{unit.done} of {unit.total} tables done")
        if unit.current is not None:
            parts.append(f"now {unit.current}")
    parts.append(f"{max(0, round(now - record.started_at))} s elapsed")
    return ", ".join(parts)
