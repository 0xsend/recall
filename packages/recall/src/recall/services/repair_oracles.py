"""Bounded oracle slots reconciled only under exclusive lifecycle ownership."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from uuid import UUID

from recall.core.durability import sync_directory
from recall.db.repair import RepairRow
from recall.db.repair_oracle import (
    CATALOG_ITEMS_MAX,
    MANIFEST_BYTES_MAX,
    MERGE_WAYS_MAX,
    OracleManifest,
    OracleReference,
    OracleUnavailable,
    read_oracle_manifest,
)

SLOTS_MAX = 4  # Pinned, replacement, capture workspace, comparison workspace.
FILES_MAX = 2 * CATALOG_ITEMS_MAX + MERGE_WAYS_MAX + 3


@dataclass(frozen=True)
class _Slot:
    path: Path
    working: bool
    trigger_id: UUID | None


def _slot(path: Path) -> _Slot:
    name = path.name
    if not path.is_dir():
        raise OracleUnavailable("oracle namespace contains a non-directory entry")
    if re.fullmatch(r"\.compare-[a-z0-9_]{8}", name):
        return _Slot(path, True, None)
    working = name.startswith(".capture-")
    identity = name.removeprefix(".capture-") if working else name
    try:
        trigger_id = UUID(identity)
    except ValueError as error:
        raise OracleUnavailable("oracle namespace contains an unknown entry") from error
    if str(trigger_id) != identity:
        raise OracleUnavailable("oracle slot identity is not canonical")
    return _Slot(path, working, trigger_id)


def _files(slot: _Slot) -> tuple[Path, ...]:
    entries = tuple(islice(slot.path.iterdir(), FILES_MAX + 3))
    if len(entries) > FILES_MAX + 2:
        raise OracleUnavailable("oracle workspace exceeds its file bound")
    files: list[Path] = []
    for path in entries:
        if path.name in {"publication", "publication-working"}:
            if not path.is_dir() or slot.working:
                raise OracleUnavailable("publication workspace has an invalid owner")
            children = tuple(islice(path.iterdir(), FILES_MAX + 1))
            if len(children) > FILES_MAX:
                raise OracleUnavailable("publication workspace exceeds its file bound")
            for child in children:
                _require_stream(child, working=path.name == "publication-working")
            files.extend(children)
        else:
            _require_stream(path, working=slot.working)
            files.append(path)
    return tuple(files)


def _require_stream(path: Path, *, working: bool) -> None:
    match = re.fullmatch(r"(table|run)-([0-9]+)\.rows", path.name)
    allowed = path.name == "manifest.json"
    if match is not None:
        bound = 2 * CATALOG_ITEMS_MAX if match[1] == "table" else MERGE_WAYS_MAX
        allowed = int(match[2]) < bound
    if working and path.name in {"unsorted.rows", "combined.rows"}:
        allowed = True
    if not allowed or not path.is_file():
        raise OracleUnavailable("oracle slot contains an unknown file")


def _manifest(slot: _Slot) -> OracleManifest | None:
    path = slot.path / "manifest.json"
    try:
        with path.open("rb") as stream:
            payload = stream.read(MANIFEST_BYTES_MAX + 1)
    except FileNotFoundError:
        return None
    if len(payload) > MANIFEST_BYTES_MAX:
        raise OracleUnavailable("oracle manifest exceeds its bound")
    manifest = OracleManifest.model_validate_json(payload)
    if manifest.trigger_id != slot.trigger_id:
        raise OracleUnavailable("oracle slot and manifest identities differ")
    return manifest


class OracleStore:
    """Own the reserved namespace, with the durable row as the sole pin source.

    Callers hold exclusive lifecycle/writer ownership. Reconciliation is
    restart recovery, not a failure handler: only then can a workspace from
    a dead owner be reclaimed. Capture failure cleanup owns only its own slot.
    Operator rollback archives are outside this namespace.
    """

    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "repair-oracles"

    def reconcile(self, row: RepairRow, candidate: UUID | None = None) -> None:
        pins: dict[str, OracleReference] = {}
        if row.oracle_ref is not None:
            assert row.trigger_id is not None and row.oracle_hash is not None
            pins[Path(row.oracle_ref).name] = OracleReference(
                trigger_id=row.trigger_id, manifest_hash=row.oracle_hash
            )
        retained = len(pins) + (candidate is not None and str(candidate) not in pins)
        if retained > 2:
            raise OracleUnavailable("no free replacement slot without deleting a pinned oracle")
        if not self.root.exists():
            if pins:
                raise OracleUnavailable("the pinned oracle namespace is missing")
            return
        paths = tuple(islice(self.root.iterdir(), SLOTS_MAX + 1))
        if len(paths) > SLOTS_MAX:
            raise OracleUnavailable("oracle namespace exceeds its slot bound")
        slots = tuple(_slot(path) for path in paths)
        for name, reference in pins.items():
            read_oracle_manifest(self.root / name, reference)
        removals: list[tuple[_Slot, tuple[Path, ...]]] = []
        for slot in slots:
            files = _files(slot)
            if not slot.working and slot.path.name in pins:
                continue
            # A pre-REQUIRED complete capture is reused only by its matching
            # trigger, after the caller compares it to the unchanged source.
            if not slot.working and slot.trigger_id == candidate and _manifest(slot) is not None:
                continue
            if not slot.working:
                manifest = _manifest(slot)
                if manifest is None and files:
                    raise OracleUnavailable("unreferenced complete slot lacks its manifest")
                if manifest is not None:
                    tables = manifest.tables + (
                        () if manifest.sidecar is None else manifest.sidecar.tables
                    )
                    allowed = {table.filename for table in tables} | {"manifest.json"}
                    if any(path.parent == slot.path and path.name not in allowed for path in files):
                        raise OracleUnavailable("unreferenced slot contains an unrecorded stream")
            removals.append((slot, files))
        # Validate the whole bounded namespace before the first deletion.
        for slot, files in removals:
            self._remove(slot, files)

    def _remove(self, slot: _Slot, files: tuple[Path, ...]) -> None:
        for path in files:
            if path.name != "manifest.json":
                path.unlink()
        for name in ("publication", "publication-working"):
            child = slot.path / name
            if child.exists():
                sync_directory(child)
                (child / "manifest.json").unlink(missing_ok=True)
                sync_directory(child)
                child.rmdir()
        sync_directory(slot.path)
        manifest = slot.path / "manifest.json"
        manifest.unlink(missing_ok=True)
        sync_directory(slot.path)
        slot.path.rmdir()
        sync_directory(self.root)
