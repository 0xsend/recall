"""Migration backup capabilities anchored in the selected installation.

Stored backup strings are metadata. Only this boundary translates an admitted
string into descriptor-relative operations; callers never reopen a checked path.
"""

from __future__ import annotations

import errno
import json
import os
import stat
from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from recall.core.config import AppConfig

_PATH_BYTES_MAX = 4096
_PATH_COMPONENTS_MAX = 256
_MANIFEST_BYTES_MAX = 1024 * 1024
_FILES = frozenset({"recall.duckdb", "recall.fts.sqlite"})
_INCOMPLETE = "original migration backup is incomplete or missing; cannot resume or restore"


@dataclass(frozen=True)
class NoBackup:
    """No backup identity was recorded."""


@dataclass(frozen=True)
class UnqualifiedBackup:
    """Metadata that grants no filesystem access."""


@dataclass(frozen=True)
class BackupInputs:
    """Validated regular inputs, kept open by the capability's owning scope."""

    database: BinaryIO
    sidecar: BinaryIO | None


@dataclass(frozen=True)
class LocalBackup:
    """An anchored presence observation, with a directory only when acquired."""

    present: bool
    _directory: int | None

    @contextmanager
    def inputs(self) -> Iterator[BackupInputs]:
        if self._directory is None:
            raise RuntimeError(_INCOMPLETE)
        with ExitStack() as stack:
            manifest = stack.enter_context(_regular_input(self._directory, "manifest.json"))
            encoded = manifest.read(_MANIFEST_BYTES_MAX + 1)
            if len(encoded) > _MANIFEST_BYTES_MAX:
                raise RuntimeError(f"{_INCOMPLETE}; manifest exceeds 1 MiB")
            try:
                payload = json.loads(encoded)
            except (ValueError, UnicodeError) as err:
                raise RuntimeError(_INCOMPLETE) from err
            if not isinstance(payload, dict):
                raise RuntimeError(_INCOMPLETE)
            files = payload.get("files")
            if (
                not payload.get("complete")
                or payload.get("kind") not in ("index-migration", "storage-migration")
                or not isinstance(files, list)
                or not all(isinstance(name, str) and name in _FILES for name in files)
                or "recall.duckdb" not in files
                or len(files) != len(set(files))
            ):
                raise RuntimeError(_INCOMPLETE)
            database = stack.enter_context(_regular_input(self._directory, "recall.duckdb"))
            sidecar = (
                stack.enter_context(_regular_input(self._directory, "recall.fts.sqlite"))
                if "recall.fts.sqlite" in files
                else None
            )
            yield BackupInputs(database, sidecar)


@contextmanager
def _regular_input(directory: int, name: str) -> Iterator[BinaryIO]:
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    except (FileNotFoundError, NotADirectoryError) as err:
        raise RuntimeError(_INCOMPLETE) from err
    except OSError as err:
        if err.errno == errno.ELOOP:
            raise RuntimeError(f"{_INCOMPLETE}; linked backup input") from err
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"{_INCOMPLETE}; backup input is not a regular file")
        handle = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with handle:
        yield handle


def _components(recorded: str, root: Path) -> tuple[str, ...] | None:
    if len(recorded) > _PATH_BYTES_MAX:
        return None
    try:
        encoded = os.fsencode(recorded)
    except UnicodeError:
        return None
    parts = recorded.split("/")
    if b"\0" in encoded or len(encoded) > _PATH_BYTES_MAX or len(parts) > _PATH_COMPONENTS_MAX:
        return None
    absolute = recorded.startswith("/")
    normalized: list[str] = []
    for part in parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if normalized:
                normalized.pop()
            elif not absolute:
                return None
        else:
            normalized.append(part)
    if absolute:
        prefix = root.parts[1:]
        if tuple(normalized[: len(prefix)]) != prefix:
            return None
        return tuple(normalized[len(prefix) :])
    return tuple(normalized)


def _open_entry(parent: int, name: str) -> LocalBackup | UnqualifiedBackup:
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    except FileNotFoundError:
        return LocalBackup(False, None)
    except OSError as err:
        if err.errno not in {errno.ENOTDIR, errno.ELOOP}:
            raise
        # Darwin may report ENOTDIR for a no-follow directory open on a link.
        try:
            entry = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return LocalBackup(False, None)
        if stat.S_ISLNK(entry.st_mode):
            return UnqualifiedBackup()
        if stat.S_ISDIR(entry.st_mode):
            raise
        return LocalBackup(True, None)
    return LocalBackup(True, descriptor)


def access_backup(
    config: AppConfig, recorded: str | None
) -> AbstractContextManager[NoBackup | UnqualifiedBackup | LocalBackup]:
    """Capture the selected root against the calling operation's cwd."""
    return _access_backup(Path.cwd() / config.data_dir, recorded)


@contextmanager
def _access_backup(
    selected: Path, recorded: str | None
) -> Iterator[NoBackup | UnqualifiedBackup | LocalBackup]:
    """Bind only the selected root, then inspect admitted descendants without links.

    Every operation rebinds its root and requalifies metadata. The root itself
    may contain links; all descendant links, including internal ones, refuse
    qualification. Unexpected I/O and invalid roots propagate as errors.
    """
    root = selected.resolve(strict=True)
    if not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW")) or (
        os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
    ):
        raise RuntimeError("migration backup access requires no-follow descriptor operations")
    with ExitStack() as stack:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, directory)
        if not recorded:
            yield NoBackup()
            return
        components = _components(recorded, root)
        if components is None:
            yield UnqualifiedBackup()
            return
        for index, component in enumerate(components):
            entry = _open_entry(directory, component)
            if isinstance(entry, UnqualifiedBackup):
                yield entry
                return
            if entry._directory is None:
                # An ordinary final file is present but cannot hold inputs;
                # no child can exist below a non-directory intermediate entry.
                yield entry if index == len(components) - 1 else LocalBackup(False, None)
                return
            directory = entry._directory
            stack.callback(os.close, directory)
        yield LocalBackup(True, directory)


def backup_presence(config: AppConfig, recorded: str | None) -> bool | None:
    with access_backup(config, recorded) as access:
        return access.present if isinstance(access, LocalBackup) else None


@contextmanager
def original_backup(config: AppConfig, recorded: str | None) -> Iterator[BackupInputs]:
    """Require a complete local backup before resume, inspection or restore."""
    with access_backup(config, recorded) as access:
        if isinstance(access, UnqualifiedBackup):
            raise RuntimeError(
                "foreign or unqualified migration backup is uninspected; cannot resume or restore"
            )
        if isinstance(access, NoBackup):
            raise RuntimeError(_INCOMPLETE)
        with access.inputs() as inputs:
            yield inputs
