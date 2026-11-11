"""Immutable, bounded failure instructions outside an invalidated database.

Publication is best effort under EIO/ENOSPC. Only fresh durable reads establish
what the next process can know; logging and exit status are not persistence.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Self, assert_never
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from recall.core.durability import sync_directory
from recall.core.repair_state import Marker, PublicationRole
from recall.core.time import to_naive_utc
from recall.db.repair import FENCE_VERSION, SchemaFence

RECORD_LIMIT = 64
BYTE_LIMIT = 1024 * 1024
DIRECTORY_ENTRIES_MAX = 2 * RECORD_LIMIT + 1
_publication_lock = threading.Lock()
PRIMARY_FILENAME = "daemon-failure.json"
SPOOL_DIRECTORY = "fatal-events"


class EventFailure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    site: str
    exception_type: str
    message: str = Field(max_length=500)
    signature: str
    failure_class: Literal["index-divergence", "disk-full", "other"]
    at: datetime

    @field_validator("at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return to_naive_utc(value)


class LegacyEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    failure: EventFailure | None
    needs_index_verification: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_instruction(self) -> Self:
        if self.failure is None and not self.needs_index_verification:
            raise ValueError("empty failure instruction")
        return self


class RecoveryEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[2] = 2
    event_id: UUID
    failure: EventFailure | None
    needs_index_verification: bool = Field(strict=True)
    probe: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def validate_instruction(self) -> Self:
        if self.failure is None and not (self.needs_index_verification or self.probe):
            raise ValueError("empty failure instruction")
        if self.probe and self.failure is not None:
            raise ValueError("positive probe is not a fabricated fatal")
        return self

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    @property
    def marker(self) -> Marker:
        if self.failure is not None:
            match self.failure.failure_class:
                case "index-divergence":
                    return Marker.INDEX
                case "disk-full":
                    return Marker.DISK
                case "other":
                    return Marker.OTHER
        return Marker.PROBE if self.probe else Marker.DISK


class EventStorageError(RuntimeError):
    """Unreadable, malformed, conflicting or full durable event storage."""


class EventIdentityConflict(EventStorageError):
    """One identity names conflicting semantic instructions."""


@dataclass(frozen=True)
class EventBatch:
    primary: RecoveryEvent | LegacyEvent | None
    secondary: tuple[tuple[Path, RecoveryEvent | LegacyEvent], ...]
    secondary_bytes: int

    @property
    def full(self) -> bool:
        return len(self.secondary) >= RECORD_LIMIT or self.secondary_bytes >= BYTE_LIMIT

    def classify(self, acknowledged: dict[UUID, tuple[str, bool]]) -> Marker:
        # Fullness has precedence even over a completely valid ordinary batch.
        if self.full:
            return Marker.INVALID
        try:
            records = self.identified()
        except EventIdentityConflict:
            return Marker.MIXED
        unresolved: list[Marker] = []
        for record in records:
            content_hash = record.content_hash
            disposition = acknowledged.get(record.event_id)
            if disposition is not None:
                expected_hash, final = disposition
                if expected_hash != content_hash:
                    return Marker.MIXED
                if final:
                    continue
            unresolved.append(record.marker)
        if not unresolved:
            return Marker.REPLAY if records else Marker.NONE
        for marker in (Marker.INDEX, Marker.PROBE, Marker.DISK, Marker.OTHER):
            if marker in unresolved:
                return marker
        raise AssertionError("unclassified event batch")

    def identified(self) -> tuple[RecoveryEvent, ...]:
        """One typed semantic instruction per id, regardless of retained files."""
        records = tuple(record for _, record in self.secondary)
        if self.primary is not None:
            records = (self.primary, *records)
        identities: dict[UUID, RecoveryEvent] = {}
        for record in records:
            if isinstance(record, LegacyEvent):
                raise EventStorageError("legacy instructions require fenced identity conversion")
            previous = identities.setdefault(record.event_id, record)
            if previous.content_hash != record.content_hash:
                raise EventIdentityConflict("one event id has conflicting content")
        return tuple(identities.values())


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise EventStorageError("duplicate field in failure instruction")
        result[name] = value
    return result


def _read_record(path: Path) -> RecoveryEvent | LegacyEvent:
    try:
        with path.open("rb") as stream:
            raw = stream.read(BYTE_LIMIT + 1)
        if len(raw) > BYTE_LIMIT:
            raise EventStorageError("event exceeds storage bound")
        payload = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(payload, dict):
            raise EventStorageError("event must be an object")
        if type(payload.get("version")) is not int:
            raise EventStorageError("event version must be an integer")
        if payload.get("version") == 1:
            return LegacyEvent.model_validate(payload)
        return RecoveryEvent.model_validate(payload)
    except (OSError, ValueError, ValidationError) as err:
        raise EventStorageError(f"could not read event {path.name}: {err}") from err


def read_events(data_dir: Path) -> EventBatch:
    primary_path = data_dir / PRIMARY_FILENAME
    try:
        primary = _read_record(primary_path)
    except EventStorageError as err:
        if isinstance(err.__cause__, FileNotFoundError):
            primary = None
        else:
            raise
    spool = data_dir / SPOOL_DIRECTORY
    secondary: list[tuple[Path, RecoveryEvent | LegacyEvent]] = []
    size = 0
    try:
        entries = os.scandir(spool)
    except FileNotFoundError:
        return EventBatch(primary, (), 0)
    except OSError as err:
        raise EventStorageError("could not enumerate failure events") from err
    with entries:
        for ordinal, entry in enumerate(entries):
            if ordinal >= DIRECTORY_ENTRIES_MAX:
                raise EventStorageError("too many unpublished event fragments; invalid spool")
            path = Path(entry.path)
            # Unpublished temporary records are never promoted by a reader.
            if path.name.startswith(".event-") and path.suffix == ".tmp":
                continue
            try:
                UUID(path.stem)
                if path.suffix != ".json":
                    raise ValueError("event filename must end in .json")
                size += entry.stat().st_size
            except (OSError, ValueError) as err:
                raise EventStorageError("invalid failure-event entry") from err
            if len(secondary) >= RECORD_LIMIT or size >= BYTE_LIMIT:
                raise EventStorageError("failure-event spool is full; use matched restore")
            record = _read_record(path)
            if isinstance(record, RecoveryEvent) and str(record.event_id) != path.stem:
                raise EventStorageError("failure-event identity differs from filename")
            secondary.append((path, record))
    return EventBatch(primary, tuple(sorted(secondary, key=lambda pair: pair[0].name)), size)


PublicationBarrier = Callable[[str], None]


def _no_barrier(stage: str) -> None:
    pass


def publish_exclusive(
    path: Path, payload: bytes, *, barrier: PublicationBarrier = _no_barrier
) -> bool:
    """Publish once, without replacing an existing instruction.

    The barrier is an injected I/O boundary for owned callers and crash tests.
    A successful exclusive link followed by failed directory sync is an unknown
    durability outcome; callers must freshly read instead of replaying writes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".event-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            barrier("before_file_sync")
            os.fsync(stream.fileno())
            barrier("after_file_sync")
        barrier("before_link")
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        barrier("after_link")
        barrier("before_directory_sync")
        sync_directory(path.parent)
        barrier("after_directory_sync")
        return True
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def publish_event(
    data_dir: Path,
    event: RecoveryEvent | LegacyEvent,
    *,
    new_id: Callable[[], UUID] = uuid4,
    barrier: PublicationBarrier = _no_barrier,
    role: PublicationRole = PublicationRole.PRIMARY,
) -> Path:
    with _publication_lock:
        return _publish_event_locked(data_dir, event, new_id=new_id, barrier=barrier, role=role)


_IMPORT_NAMESPACE = UUID("d948e233-6d80-43d4-985e-ed39130e6cbf")


def import_legacy_events(
    data_dir: Path,
    fence: SchemaFence,
    *,
    barrier: PublicationBarrier = _no_barrier,
) -> EventBatch:
    """Assign stable identities under the caller's exclusive lifecycle lock.

    This is the sole permitted replacement of an unfolded primary: its entire
    semantic instruction is retained. A crash before publication repeats the
    same identity; a crash after publication reads the assigned identity.
    """
    assert fence.version >= FENCE_VERSION
    with _publication_lock:
        batch = read_events(data_dir)
        records = batch.secondary
        if batch.primary is not None:
            records = ((data_dir / PRIMARY_FILENAME, batch.primary), *records)
        for path, record in records:
            if not isinstance(record, LegacyEvent):
                continue
            identity = (
                uuid5(_IMPORT_NAMESPACE, path.name + ":" + record.model_dump_json())
                if path.name == PRIMARY_FILENAME
                else UUID(path.stem)
            )
            converted = RecoveryEvent(
                event_id=identity,
                failure=record.failure,
                needs_index_verification=record.needs_index_verification,
            )
            descriptor, temporary = tempfile.mkstemp(
                prefix=".event-", suffix=".tmp", dir=path.parent
            )
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(converted.model_dump_json().encode())
                    stream.flush()
                    barrier("before_file_sync")
                    os.fsync(stream.fileno())
                    barrier("after_file_sync")
                barrier("before_replace")
                os.replace(temporary, path)
                barrier("after_replace")
                barrier("before_directory_sync")
                sync_directory(path.parent)
                barrier("after_directory_sync")
            finally:
                with suppress(FileNotFoundError):
                    os.unlink(temporary)
        return read_events(data_dir)


def _publish_event_locked(
    data_dir: Path,
    event: RecoveryEvent | LegacyEvent,
    *,
    new_id: Callable[[], UUID],
    barrier: PublicationBarrier,
    role: PublicationRole,
) -> Path:
    batch = read_events(data_dir)
    payload = event.model_dump_json().encode()
    primary = data_dir / PRIMARY_FILENAME
    match role:
        case PublicationRole.PRIMARY:
            if batch.primary is None and publish_exclusive(primary, payload, barrier=barrier):
                return primary
        case PublicationRole.SECONDARY:
            pass
        case _:
            assert_never(role)
    if batch.full or batch.secondary_bytes + len(payload) > BYTE_LIMIT:
        raise EventStorageError("failure-event spool is full; use matched restore")
    event_id = event.event_id if isinstance(event, RecoveryEvent) else new_id()
    secondary = data_dir / SPOOL_DIRECTORY / f"{event_id}.json"
    if not secondary.parent.exists():
        secondary.parent.mkdir(parents=True, exist_ok=True)
        sync_directory(data_dir)
    if (
        not publish_exclusive(secondary, payload, barrier=barrier)
        and secondary.read_bytes() != payload
    ):
        raise EventStorageError("existing event identity has different content")
    return secondary
