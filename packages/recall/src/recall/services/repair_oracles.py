"""Bounded oracle slots reconciled only under exclusive lifecycle ownership."""

from __future__ import annotations

import re
import stat
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from uuid import UUID

from recall.core.durability import sync_directory
from recall.db.oracle_sort import SORT_DIRECTORY, SORT_FILES_MAX
from recall.db.repair import RepairRow
from recall.db.repair_oracle import (
    CATALOG_ITEMS_MAX,
    MANIFEST_BYTES_MAX,
    NATIVE_KEYWORD_TABLES_MAX,
    OracleManifest,
    OracleReference,
    OracleUnavailable,
    PublicationOracle,
    read_oracle_manifest,
)

SLOTS_MAX = 4  # Pinned, replacement, capture workspace, comparison workspace.
# Table streams, the manifest and the sorter's single `sort` directory.
FILES_MAX = 2 * CATALOG_ITEMS_MAX + NATIVE_KEYWORD_TABLES_MAX + 2


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
    entries = tuple(islice(slot.path.iterdir(), FILES_MAX + 1))
    if len(entries) > FILES_MAX:
        raise OracleUnavailable("oracle workspace exceeds its file bound")
    files: list[Path] = []
    for path in entries:
        if path.name in {"publication", "publication-working"}:
            if not path.is_dir() or slot.working:
                raise OracleUnavailable("publication workspace has an invalid owner")
            files.extend(_container_files(path, working=path.name == "publication-working"))
        else:
            files.extend(_container_entry(path, working=slot.working))
    return tuple(files)


def _container_files(container: Path, *, working: bool) -> tuple[Path, ...]:
    children = tuple(islice(container.iterdir(), FILES_MAX + 1))
    if len(children) > FILES_MAX:
        raise OracleUnavailable("publication workspace exceeds its file bound")
    return tuple(file for child in children for file in _container_entry(child, working=working))


def _container_entry(path: Path, *, working: bool) -> tuple[Path, ...]:
    """The files one entry of a container stands for, validated as a whole.

    Only a working container (its owner is dead and nothing published it) may
    hold the sorter's `sort` directory; there it is a single entry.
    """
    if working and path.name == SORT_DIRECTORY:
        return _sort_files(path)
    _require_stream(path)
    return (path,)


def _sort_files(directory: Path) -> tuple[Path, ...]:
    if directory.is_symlink() or not directory.is_dir():
        raise OracleUnavailable("oracle sort workspace is not a directory")
    children = tuple(islice(directory.iterdir(), SORT_FILES_MAX + 1))
    if len(children) > SORT_FILES_MAX:
        raise OracleUnavailable("oracle sort workspace exceeds its file bound")
    if any(not stat.S_ISREG(child.lstat().st_mode) for child in children):
        raise OracleUnavailable("oracle sort workspace contains an unknown entry")
    return children


def _require_stream(path: Path) -> None:
    allowed = path.name == "manifest.json"
    match = re.fullmatch(r"table-([0-9]+)\.rows", path.name)
    if match is not None:
        allowed = int(match[1]) < 2 * CATALOG_ITEMS_MAX + NATIVE_KEYWORD_TABLES_MAX
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

    def retire_unreferenced_publication(self, row: RepairRow) -> None:
        """Reconcile a dead manual publisher only after its canonical precheck.

        A committed certificate pins its publication. Without that certificate,
        the atomic completion could not have retired any acknowledged intents.
        """
        if row.release_certificate is not None:
            raise OracleUnavailable("a committed publication reference remains pinned")
        assert row.oracle_ref is not None and row.oracle_hash is not None
        slot = _slot(self.root.parent / row.oracle_ref)
        files = _files(slot)
        published = slot.path / "publication"
        work = None if row.trigger_json is None else row.trigger_json.manual_work
        if (
            work is not None
            and work.publication is not None
            and self.root.parent / work.publication.path == published
        ):
            raise OracleUnavailable("a running source publication remains pinned")
        metadata = published / "manifest.json"
        if metadata.exists():
            with metadata.open("rb") as stream:
                payload = stream.read(MANIFEST_BYTES_MAX + 1)
            if len(payload) > MANIFEST_BYTES_MAX:
                raise OracleUnavailable("publication manifest exceeds its bound")
            # A dead publisher's streams may already have been removed. Its
            # bounded instruction still identifies which uncommitted work is
            # disposable; it cannot authorize content or release.
            publication = PublicationOracle.model_validate_json(payload)
            if (
                publication.trigger_id != row.trigger_id
                or publication.oracle_hash != row.oracle_hash
                or publication.attempt_no > row.attempt_no
            ):
                raise OracleUnavailable("unreferenced publication belongs to another attempt")
        elif any(path.parent == published for path in files):
            raise OracleUnavailable("unreferenced publication lacks its identity manifest")
        for name in ("publication", "publication-working"):
            child = slot.path / name
            if child.exists():
                self._remove_child(child, files)

    def _remove_child(self, child: Path, files: tuple[Path, ...]) -> None:
        self._remove_sort(child, files)
        for path in files:
            if path.parent == child and path.name != "manifest.json":
                path.unlink()
        sync_directory(child)
        (child / "manifest.json").unlink(missing_ok=True)
        sync_directory(child)
        child.rmdir()
        sync_directory(child.parent)

    def _remove(self, slot: _Slot, files: tuple[Path, ...]) -> None:
        for name in ("publication", "publication-working"):
            child = slot.path / name
            if child.exists():
                self._remove_child(child, files)
        self._remove_sort(slot.path, files)
        for path in files:
            if path.parent == slot.path and path.name != "manifest.json":
                path.unlink()
        sync_directory(slot.path)
        manifest = slot.path / "manifest.json"
        manifest.unlink(missing_ok=True)
        sync_directory(slot.path)
        slot.path.rmdir()
        sync_directory(self.root)

    def _remove_sort(self, container: Path, files: tuple[Path, ...]) -> None:
        """Delete the sorter's directory first: nothing in it is ever referenced."""
        directory = container / SORT_DIRECTORY
        if not directory.exists():
            return
        for path in files:
            if path.parent == directory:
                path.unlink()
        sync_directory(directory)
        directory.rmdir()
        sync_directory(container)
