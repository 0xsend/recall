"""Directories and files held by descriptor, from resolution through use.

Every path is opened one component at a time from `/` with `O_NOFOLLOW`, and
everything after that is relative to a held descriptor. A name is never
resolved twice, so replacing a checked directory with a symlink, or renaming
it, cannot redirect a later read or write.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import os
import secrets
import signal
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW
NEW_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
COPY_FILE_FLAGS = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
CHUNK_BYTES = 8 * 1024 * 1024
ENTRY_LIMIT = 100_000
# Linux `FICLONE` (_IOW(0x94, 9, int)); a reflink where the filesystem supports it.
FICLONE = 0x40049409
# macOS `renameatx_np` flag: fail rather than replace an existing name.
RENAME_EXCL = 0x00000004
# macOS links these top-level names into /private; nothing else may be a link.
DARWIN_PRIVATE = frozenset({"tmp", "var", "etc"})
UNCLONEABLE = frozenset({errno.EOPNOTSUPP, errno.ENOTSUP, errno.EXDEV, errno.EINVAL, errno.ENOTTY})

Identity = tuple[int, int]


class Refused(Exception):
    """An argument the isolation policy will not accept; nothing was started."""

    def __init__(self, argument: str, path: object, reason: str) -> None:
        super().__init__(f"{argument} {path}: {reason}")
        self.argument = argument
        self.path = str(path)
        self.reason = reason

    def describe(self) -> dict[str, object]:
        return {
            "kind": "refused",
            "argument": self.argument,
            "path": self.path,
            "reason": self.reason,
        }


class Cancelled(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(f"cancelled by signal {signum}")
        self.signum = signum


class Cancellation:
    """The first TERM or INT, observed by long work at its own safe points.

    The handler only records the signal, so no exception ever lands inside a
    system call or between a creation and its record.
    """

    SIGNALS = (signal.SIGTERM, signal.SIGINT)

    def __init__(self, notify: Callable[[dict[str, object]], None] | None = None) -> None:
        self.signum: int | None = None
        self._notify = notify
        self._raised = False

    def install(self) -> None:
        for signum in self.SIGNALS:
            signal.signal(signum, self._record)

    def _record(self, signum: int, _frame: object) -> None:
        if self.signum is None:
            self.signum = signum

    def check(self) -> None:
        if self.signum is None:
            return
        if not self._raised and self._notify is not None:
            # Observable before cleanup starts, so a caller can time a second signal.
            self._notify({"event": "cancelled", "signal": self.signum})
        self._raised = True
        raise Cancelled(self.signum)

    @classmethod
    @contextmanager
    def deferred(cls) -> Iterator[None]:
        """Hold TERM and INT until the block ends, so cleanup runs to completion."""
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, cls.SIGNALS)
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def check_name(name: str) -> str:
    """Accept one path component; anything else could leave the directory."""
    if name in {"", ".", ".."} or "/" in name or "\0" in name:
        raise ValueError(f"not a single path component: {name!r}")
    return name


def identity(status: os.stat_result) -> Identity:
    return status.st_dev, status.st_ino


def _kind(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISLNK(mode):
        return "link"
    return "other"


def mount_id(descriptor: int) -> int | None:
    """The Linux mount a descriptor lives on; binds on one device share st_dev, not this."""
    if sys.platform != "linux":
        return None
    with open(f"/proc/self/fdinfo/{descriptor}") as fdinfo:
        for line in fdinfo:
            if line.startswith("mnt_id:"):
                return int(line.split()[1])
    raise OSError(errno.ENOTSUP, "fdinfo has no mnt_id", str(descriptor))


# Called with each component's display path and status as a walk opens it.
Inspect = Callable[[Path, os.stat_result], None]


class File:
    """A regular file held open by descriptor."""

    def __init__(self, path: Path, descriptor: int) -> None:
        self.path = path
        self.descriptor = descriptor
        self.status = os.fstat(descriptor)
        if not stat.S_ISREG(self.status.st_mode):
            os.close(descriptor)
            raise OSError(errno.EINVAL, "not a regular file", str(path))
        self.mount = mount_id(descriptor)

    def sha256(self, cancellation: Cancellation) -> str:
        digest = hashlib.sha256()
        offset = 0
        while chunk := os.pread(self.descriptor, CHUNK_BYTES, offset):
            cancellation.check()
            digest.update(chunk)
            offset += len(chunk)
        return digest.hexdigest()

    def refresh(self) -> os.stat_result:
        self.status = os.fstat(self.descriptor)
        return self.status

    def head(self, size: int) -> bytes:
        return os.pread(self.descriptor, size, 0)

    def close(self) -> None:
        os.close(self.descriptor)

    def __enter__(self) -> File:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class Listing:
    """One directory's entries, classified without following anything."""

    directories: tuple[str, ...]
    files: tuple[str, ...]
    symlinks: tuple[str, ...]
    others: tuple[str, ...]


class Directory:
    """A directory held open; every operation is relative to its descriptor."""

    def __init__(self, path: Path, descriptor: int) -> None:
        self.path = path
        self.descriptor = descriptor
        self.status = os.fstat(descriptor)
        self.mount = mount_id(descriptor)

    @property
    def identity(self) -> Identity:
        return identity(self.status)

    @classmethod
    def walk(cls, argument: str, path: Path, inspect: Inspect) -> Directory:
        """Open `path` from `/`, refusing a symlink at any component."""
        assert path.is_absolute() and ".." not in path.parts, path
        directory = cls(Path("/"), os.open("/", DIRECTORY_FLAGS))
        inspect(directory.path, directory.status)
        try:
            for part in path.parts[1:]:
                child = directory._open_walked(argument, part)
                directory.close()
                directory = child
                inspect(directory.path, directory.status)
        except BaseException:
            directory.close()
            raise
        return directory

    def _open_walked(self, argument: str, name: str) -> Directory:
        if sys.platform == "darwin" and self.path == Path("/") and name in DARWIN_PRIVATE:
            status = self.lstat(name)
            linked = status is not None and stat.S_ISLNK(status.st_mode)
            target = os.readlink(name, dir_fd=self.descriptor) if linked else None
            if target in {f"private/{name}", f"/private/{name}"}:
                with self.open_directory(argument, "private") as private:
                    resolved = private.open_directory(argument, name)
                return Directory(self.path / name, resolved._release())
        return self.open_directory(argument, name)

    def _release(self) -> int:
        descriptor, self.descriptor = self.descriptor, -1
        return descriptor

    def _refuse_entry(self, argument: str, name: str, error: OSError) -> Refused:
        try:
            mode = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False).st_mode
        except FileNotFoundError:
            return Refused(argument, self.path / name, "does not exist")
        if stat.S_ISLNK(mode):
            return Refused(argument, self.path / name, f"symlinked component {self.path / name}")
        if error.errno in {errno.ENOTDIR, errno.ELOOP}:
            return Refused(argument, self.path / name, "not a directory")
        return Refused(argument, self.path / name, os.strerror(error.errno or 0))

    def open_directory(self, argument: str, name: str) -> Directory:
        try:
            descriptor = os.open(check_name(name), DIRECTORY_FLAGS, dir_fd=self.descriptor)
        except OSError as error:
            raise self._refuse_entry(argument, name, error) from error
        return Directory(self.path / name, descriptor)

    def open_file(self, argument: str, name: str) -> File:
        try:
            descriptor = os.open(
                check_name(name), FILE_FLAGS | os.O_NONBLOCK, dir_fd=self.descriptor
            )
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise Refused(argument, self.path / name, "symlinked payload file") from error
            if error.errno == errno.ENOENT:
                raise Refused(argument, self.path / name, "does not exist") from error
            raise
        try:
            return File(self.path / name, descriptor)
        except OSError as error:
            raise Refused(argument, self.path / name, "not a regular file") from error

    def lstat(self, name: str) -> os.stat_result | None:
        try:
            return os.stat(check_name(name), dir_fd=self.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return None

    def list(self, argument: str) -> Listing:
        names = sorted(os.listdir(self.descriptor))
        if len(names) > ENTRY_LIMIT:
            raise Refused(argument, self.path, f"more than {ENTRY_LIMIT} entries")
        kinds: dict[str, list[str]] = {"dir": [], "file": [], "link": [], "other": []}
        for name in names:
            try:
                name.encode()
            except UnicodeEncodeError:
                raise Refused(argument, self.path / name, "name is not valid UTF-8") from None
            status = self.lstat(name)
            if status is None:
                raise Refused(argument, self.path / name, "changed while listing")
            kind = _kind(status.st_mode)
            kinds[kind].append(name)
        return Listing(
            tuple(kinds["dir"]), tuple(kinds["file"]), tuple(kinds["link"]), tuple(kinds["other"])
        )

    def descend(self, argument: str, relative: PurePosixPath) -> Directory:
        """Open a subdirectory one component at a time; the caller closes it."""
        directory = self.open_directory(argument, relative.parts[0])
        for part in relative.parts[1:]:
            child = directory.open_directory(argument, part)
            directory.close()
            directory = child
        return directory

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)

    def __enter__(self) -> Directory:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class NewTree:
    """A directory this job creates, and everything it creates inside.

    Creation is exclusive and relative to held descriptors. Unless `keep` is
    called, leaving the block removes exactly what was recorded, deepest
    first, with TERM and INT held; anything someone else added stays and is
    reported.
    """

    def __init__(self, parent: Directory, name: str) -> None:
        os.mkdir(check_name(name), 0o755, dir_fd=parent.descriptor)
        self._parent = parent
        self._name = name
        self.root = _open_new_directory(parent, name)
        self._created: list[tuple[PurePosixPath, bool]] = []
        self._kept = False
        self.leftovers: list[str] = []

    @property
    def path(self) -> Path:
        return self.root.path

    def make_directory(self, parent: Directory, relative: PurePosixPath) -> Directory:
        os.mkdir(check_name(relative.name), 0o755, dir_fd=parent.descriptor)
        self._created.append((relative, True))
        return _open_new_directory(parent, relative.name)

    def write_new(self, directory: Directory, relative: PurePosixPath, data: bytes) -> None:
        descriptor = os.open(
            check_name(relative.name), NEW_FILE_FLAGS, 0o644, dir_fd=directory.descriptor
        )
        self._created.append((relative, False))
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    def publish(self, relative: PurePosixPath, data: bytes) -> None:
        """Write a top-level file whole or not at all, never replacing anything.

        `linkat` fails on an existing name, file or symlink alike, so a second
        publish leaves the first file untouched.
        """
        assert len(relative.parts) == 1, relative
        partial = PurePosixPath(f".{relative.name}.partial")
        self.write_new(self.root, partial, data)
        try:
            os.link(
                partial.name,
                relative.name,
                src_dir_fd=self.root.descriptor,
                dst_dir_fd=self.root.descriptor,
                follow_symlinks=False,
            )
        except OSError:
            os.unlink(partial.name, dir_fd=self.root.descriptor)
            self._created.remove((partial, False))
            raise
        self._created.append((relative, False))
        os.unlink(partial.name, dir_fd=self.root.descriptor)
        self._created.remove((partial, False))

    def clone(
        self,
        directory: Directory,
        relative: PurePosixPath,
        source: File,
        cancellation: Cancellation,
    ) -> File:
        """Copy `source` to a new file, copy-on-write where the volume allows.

        The copy is returned open: on Linux it is the exclusive descriptor the
        bytes were written through; on macOS the moved clone, whose identity
        must equal the one checked before the move.
        """
        name = check_name(relative.name)
        cancellation.check()
        cloned = _clone_darwin(source, directory, name)
        if cloned is None:
            descriptor = os.open(name, COPY_FILE_FLAGS, 0o644, dir_fd=directory.descriptor)
            self._created.append((relative, False))
            copy = File(directory.path / name, descriptor)
            try:
                _copy_contents(source, copy, cancellation)
            except BaseException:
                copy.close()
                raise
            copy.refresh()
        else:
            self._created.append((relative, False))
            copy = directory.open_file("clone", name)
            if identity(copy.status) != cloned:
                copy.close()
                raise OSError(errno.EIO, "the clone was replaced", str(directory.path / name))
        if copy.status.st_size != source.status.st_size or copy.status.st_nlink != 1:
            copy.close()
            raise OSError(errno.EIO, "clone does not match its source", str(source.path))
        return copy

    def keep(self) -> None:
        self._kept = True

    def _discard(self) -> None:
        for relative, is_directory in reversed(self._created):
            try:
                self._remove(relative, is_directory)
            except (OSError, Refused):
                self.leftovers.append(str(relative))
        try:
            os.rmdir(self._name, dir_fd=self._parent.descriptor)
        except OSError:
            self.leftovers.append(".")

    def _remove(self, relative: PurePosixPath, is_directory: bool) -> None:
        remove = os.rmdir if is_directory else os.unlink
        if not relative.parent.parts:
            remove(relative.name, dir_fd=self.root.descriptor)
            return
        with self.root.descend("cleanup", relative.parent) as parent:
            remove(relative.name, dir_fd=parent.descriptor)

    def __enter__(self) -> NewTree:
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            if not self._kept:
                with Cancellation.deferred():
                    self._discard()
        finally:
            self.root.close()


def _open_new_directory(parent: Directory, name: str) -> Directory:
    directory = parent.open_directory("create", name)
    fresh = directory.status.st_dev == parent.status.st_dev and not os.listdir(directory.descriptor)
    if not fresh:
        directory.close()
        raise OSError(errno.EEXIST, "a created directory was replaced", str(parent.path / name))
    return directory


def _clone_darwin(source: File, directory: Directory, name: str) -> Identity | None:
    """An APFS clone, moved into place without replacing or following anything.

    `fclonefileat` follows a symlink at its destination name, so it writes
    only to an unguessable name in this job's own directory, and
    `renameatx_np(RENAME_EXCL)` then gives the clone its real name.
    """
    if sys.platform != "darwin":
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    clone, rename = libc.fclonefileat, libc.renameatx_np
    clone.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    staging = f".clone-{secrets.token_hex(16)}"
    if clone(source.descriptor, directory.descriptor, os.fsencode(staging), 0) != 0:
        code = ctypes.get_errno()
        if code in UNCLONEABLE:
            return None
        raise OSError(code, os.strerror(code), str(directory.path / name))
    try:
        status = os.stat(staging, dir_fd=directory.descriptor, follow_symlinks=False)
        placed = stat.S_ISREG(status.st_mode) and status.st_nlink == 1
        if not placed or status.st_dev != directory.status.st_dev:
            raise OSError(
                errno.EIO, "clone is not a new regular file", str(directory.path / staging)
            )
        fd = directory.descriptor
        if rename(fd, os.fsencode(staging), fd, os.fsencode(name), RENAME_EXCL) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(directory.path / name))
    except BaseException:
        os.unlink(staging, dir_fd=directory.descriptor)
        raise
    return identity(status)


def _copy_contents(source: File, copy: File, cancellation: Cancellation) -> None:
    destination = copy.descriptor
    if sys.platform == "linux":
        try:
            fcntl.ioctl(destination, FICLONE, source.descriptor)
            return
        except OSError as error:
            if error.errno not in UNCLONEABLE:
                raise
    offset = 0
    while True:
        cancellation.check()
        copied = _copy_chunk(source.descriptor, destination, offset)
        if copied == 0:
            return
        offset += copied


def _copy_chunk(source: int, destination: int, offset: int) -> int:
    if sys.platform == "linux":
        try:
            return os.copy_file_range(source, destination, CHUNK_BYTES, offset, offset)
        except OSError as error:
            if error.errno not in UNCLONEABLE | {errno.ENOSYS}:
                raise
    data = os.pread(source, CHUNK_BYTES, offset)
    written = 0
    while written < len(data):
        written += os.pwrite(destination, data[written:], offset + written)
    return len(data)
