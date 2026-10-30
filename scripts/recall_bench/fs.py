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
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
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
# Linux `FS_IOC_GETVERSION` (_IOR('v', 1, long)); the kernel writes the inode's
# 32-bit generation.
FS_IOC_GETVERSION = 0x80087601
NO_GENERATION = frozenset({errno.ENOTTY, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL})
# macOS `renameatx_np` flag: fail rather than replace an existing name.
RENAME_EXCL = 0x00000004
# macOS links these top-level names into /private; nothing else may be a link.
DARWIN_PRIVATE = frozenset({"tmp", "var", "etc"})
UNCLONEABLE = frozenset({errno.EOPNOTSUPP, errno.ENOTSUP, errno.EXDEV, errno.EINVAL, errno.ENOTTY})
ALIAS_REASON = "multiply linked file; it may alias another file"

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


class DeadlineExpired(Exception):
    """The job's whole-run deadline passed; the work is incomplete."""


class Cancellation:
    """The first TERM or INT, or the run deadline, observed by long work at its own safe points.

    The handler only records the signal, so no exception ever lands inside a
    system call or between a creation and its record. A signal outranks the
    deadline.
    """

    SIGNALS = (signal.SIGTERM, signal.SIGINT)

    def __init__(self, notify: Callable[[dict[str, object]], None] | None = None) -> None:
        self.signum: int | None = None
        self._notify = notify
        self._raised = False
        self._expires_at: float | None = None

    def expire_at(self, monotonic_at: float) -> None:
        """Make every later safe point past `monotonic_at` end the job as incomplete."""
        self._expires_at = monotonic_at

    def install(self) -> None:
        for signum in self.SIGNALS:
            signal.signal(signum, self._record)

    def _record(self, signum: int, _frame: object) -> None:
        if self.signum is None:
            self.signum = signum

    def check_held(self) -> None:
        """`check`, counting a TERM or INT held pending by `deferred` as received."""
        pending = signal.sigpending()
        for signum in self.SIGNALS:
            if self.signum is None and signum in pending:
                self.signum = signum
        self.check()

    def check(self) -> None:
        if self.signum is None:
            if self._expires_at is not None and time.monotonic() >= self._expires_at:
                raise DeadlineExpired("the run deadline expired")
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


def generation(descriptor: int) -> int | None:
    """The filesystem's number for this use of the inode, or None where it has none."""
    # WAIVER(identity reuse): without a generation an entry is compared by
    # inode alone. macOS volumes give none to an unprivileged process; APFS
    # numbers inodes from a per-volume counter (`apfs_next_obj_id`, Apple File
    # System Reference) and does not reuse them. Linux tmpfs answers ENOTTY and
    # numbers inodes from a counter as well. Debt: a Linux filesystem that both
    # reuses inode numbers and has no generation stays unchecked; removed by
    # retaining a descriptor for every created entry, which a large pool's
    # directory count does not fit under default descriptor limits.
    if sys.platform != "linux":
        return None
    buffer = bytearray(8)
    try:
        fcntl.ioctl(descriptor, FS_IOC_GETVERSION, buffer)
    except OSError as error:
        if error.errno in NO_GENERATION:
            return None
        raise
    return int.from_bytes(buffer[:4], sys.byteorder)


@dataclass(frozen=True)
class Creation:
    """One file or directory as it was created or first seen, read from its descriptor.

    ext4 gives a removed directory's inode number to the next directory made
    beside it, so `(st_dev, st_ino)` alone would accept the replacement. The
    generation differs for each use of an inode number, so a later entry at
    the same number is a different `Creation`.
    """

    inode: Identity
    generation: int | None

    @classmethod
    def of(cls, descriptor: int) -> Creation:
        return cls(identity(os.fstat(descriptor)), generation(descriptor))


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

    @property
    def creation(self) -> Creation:
        return Creation(identity(self.status), generation(self.descriptor))

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

    @property
    def creation(self) -> Creation:
        return Creation(self.identity, generation(self.descriptor))

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
        """A regular file in this directory with no other name and on this mount.

        The checks run on the new descriptor before it is returned, so no byte
        of a hardlinked or bind-mounted alias is ever read or cloned.
        """
        file = self.open_present_file(argument, name)
        if file is None:
            raise Refused(argument, self.path / name, "does not exist")
        return file

    def open_present_file(self, argument: str, name: str) -> File | None:
        """`open_file`, or None when nothing has the name any more."""
        try:
            descriptor = os.open(
                check_name(name), FILE_FLAGS | os.O_NONBLOCK, dir_fd=self.descriptor
            )
        except FileNotFoundError:
            return None
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise Refused(argument, self.path / name, "symlinked payload file") from error
            raise
        try:
            file = File(self.path / name, descriptor)
        except OSError as error:
            raise Refused(argument, self.path / name, "not a regular file") from error
        if file.status.st_nlink != 1:
            file.close()
            raise Refused(argument, file.path, ALIAS_REASON)
        if (file.status.st_dev, file.mount) != (self.status.st_dev, self.mount):
            file.close()
            raise Refused(argument, file.path, "crosses a mount point")
        return file

    def check_place(self, argument: str, name: str) -> None:
        """Refuse an entry of any kind that is not on this directory's device and mount.

        On Linux the entry is held by an `O_PATH` descriptor, which opens a
        socket, FIFO or directory without side effects and carries its mount.
        """
        if sys.platform == "linux":
            try:
                descriptor = os.open(
                    check_name(name), os.O_PATH | os.O_NOFOLLOW, dir_fd=self.descriptor
                )
            except FileNotFoundError:
                return
            try:
                place = (os.fstat(descriptor).st_dev, mount_id(descriptor))
            finally:
                os.close(descriptor)
        else:
            status = self.lstat(name)
            if status is None:
                return
            place = (status.st_dev, None)
        if place != (self.status.st_dev, self.mount):
            raise Refused(argument, self.path / name, "crosses a mount point")

    def creation_of(self, name: str) -> Creation | None:
        """The `Creation` of the entry at `name`, without following it; None if absent.

        Raises OSError for an entry that cannot be opened, such as a symlink or
        a socket, which is never one this job created.
        """
        try:
            descriptor = os.open(
                check_name(name), FILE_FLAGS | os.O_NONBLOCK, dir_fd=self.descriptor
            )
        except FileNotFoundError:
            return None
        try:
            return Creation.of(descriptor)
        finally:
            os.close(descriptor)

    def lstat(self, name: str) -> os.stat_result | None:
        try:
            return os.stat(check_name(name), dir_fd=self.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return None

    def statuses(self, argument: str) -> Iterator[tuple[str, os.stat_result]]:
        """Each entry and its status, skipping entries removed while listing."""
        names = sorted(os.listdir(self.descriptor))
        if len(names) > ENTRY_LIMIT:
            raise Refused(argument, self.path, f"more than {ENTRY_LIMIT} entries")
        for name in names:
            status = self.lstat(name)
            if status is not None:
                yield name, status

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

    @contextmanager
    def as_working_directory(self) -> Iterator[None]:
        """Make this directory the process's working directory for the block.

        A child spawned inside the block starts in this directory by
        descriptor, wherever its name now points. Jobs resolve no relative
        names, so the process-wide change is invisible to them.
        """
        previous = os.open(".", os.O_RDONLY)
        try:
            os.fchdir(self.descriptor)
            yield
        finally:
            os.fchdir(previous)
            os.close(previous)

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)

    def __enter__(self) -> Directory:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class Created:
    """An entry a job created, and the directory it was created in, by identity."""

    relative: PurePosixPath
    is_directory: bool
    identity: Creation
    parent: Creation


# Entries a job could not remove, as {"path", "reason"}; the job reports them.
Leftovers = list[dict[str, str]]


class NewTree:
    """A directory this job creates, and everything it creates inside.

    Creation is exclusive and relative to held descriptors. Unless `commit`
    succeeds, leaving the block removes, deepest first and with TERM and INT
    held, each recorded entry whose `Creation` and parent's `Creation` are
    still the ones this job made, so an entry someone removed and replaced
    stays even when the replacement reuses its inode number. Anything else
    stays and goes to `leftovers`.
    """

    def __init__(self, parent: Directory, name: str, leftovers: Leftovers) -> None:
        os.mkdir(check_name(name), 0o755, dir_fd=parent.descriptor)
        self._parent = parent
        self._name = name
        self.root = _open_new_directory(parent, name)
        self._created: list[Created] = []
        self._kept = False
        self._leftovers = leftovers

    @property
    def path(self) -> Path:
        return self.root.path

    def _record(
        self, directory: Directory, relative: PurePosixPath, is_directory: bool, made: Creation
    ) -> Created:
        entry = Created(relative, is_directory, made, directory.creation)
        self._created.append(entry)
        return entry

    def make_directory(self, parent: Directory, relative: PurePosixPath) -> Directory:
        os.mkdir(check_name(relative.name), 0o755, dir_fd=parent.descriptor)
        directory = _open_new_directory(parent, relative.name)
        self._record(parent, relative, True, directory.creation)
        return directory

    def write_new(
        self, directory: Directory, relative: PurePosixPath, data: bytes, mode: int = 0o644
    ) -> Creation:
        return self._write_new(directory, relative, data, mode).identity

    def _write_new(
        self, directory: Directory, relative: PurePosixPath, data: bytes, mode: int = 0o644
    ) -> Created:
        descriptor = os.open(
            check_name(relative.name), NEW_FILE_FLAGS, mode, dir_fd=directory.descriptor
        )
        # The umask must not decide whether a canary or wrapper can run.
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            entry = self._record(directory, relative, False, Creation.of(descriptor))
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return entry

    def open_new(self, directory: Directory, relative: PurePosixPath) -> int:
        """Create a file exclusively; its descriptor reads anywhere and appends."""
        descriptor = os.open(
            check_name(relative.name),
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_APPEND,
            0o644,
            dir_fd=directory.descriptor,
        )
        self._record(directory, relative, False, Creation.of(descriptor))
        return descriptor

    def publish(self, relative: PurePosixPath, data: bytes) -> None:
        """Write a top-level file whole or not at all, never replacing anything.

        `linkat` fails on an existing name, file or symlink alike, so a second
        publish leaves the first file untouched.
        """
        assert len(relative.parts) == 1, relative
        partial = self._write_new(self.root, PurePosixPath(f".{relative.name}.partial"), data)
        try:
            os.link(
                partial.relative.name,
                relative.name,
                src_dir_fd=self.root.descriptor,
                dst_dir_fd=self.root.descriptor,
                follow_symlinks=False,
            )
        except OSError:
            os.unlink(partial.relative.name, dir_fd=self.root.descriptor)
            self._created.remove(partial)
            raise
        self._record(self.root, relative, False, partial.identity)
        os.unlink(partial.relative.name, dir_fd=self.root.descriptor)
        self._created.remove(partial)

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
            copy = File(directory.path / name, descriptor)
            self._record(directory, relative, False, copy.creation)
            try:
                _copy_contents(source, copy, cancellation)
            except BaseException:
                copy.close()
                raise
            copy.refresh()
        else:
            self._record(directory, relative, False, cloned)
            copy = directory.open_file("clone", name)
            if copy.creation != cloned:
                copy.close()
                raise OSError(errno.EIO, "the clone was replaced", str(directory.path / name))
        if copy.status.st_size != source.status.st_size or copy.status.st_nlink != 1:
            copy.close()
            raise OSError(errno.EIO, "clone does not match its source", str(source.path))
        return copy

    def directories(self) -> dict[PurePosixPath, Creation]:
        """Every directory this job created inside the tree."""
        return {entry.relative: entry.identity for entry in self._created if entry.is_directory}

    def commit(self, cancellation: Cancellation) -> None:
        """Keep the tree; the job's last fallible step.

        TERM and INT are held across the decision. A signal recorded before
        it, even after the last check inside a read, or held pending during
        it, removes the tree. One arriving later is delivered after the tree
        is kept and does not undo it; the job's next cancellation check
        still stops the job.
        """
        with Cancellation.deferred():
            cancellation.check_held()
            self._kept = True

    def _leave(self, path: Path, reason: str) -> None:
        self._leftovers.append({"path": str(path), "reason": reason})

    def _discard(self) -> None:
        for entry in reversed(self._created):
            reason = self._remove(entry)
            if reason is not None:
                self._leave(self.path / entry.relative, reason)
        try:
            current = self._parent.creation_of(self._name)
        except OSError:
            self._leave(self.path, "replaced since this job created it; left in place")
            return
        if current is None:
            return
        if current != self.root.creation:
            self._leave(self.path, "replaced since this job created it; left in place")
            return
        # WAIVER(TOCTOU): POSIX has no rmdir by descriptor, so the name is removed
        # after its identity is checked. Permanent: the window is two system calls.
        try:
            os.rmdir(self._name, dir_fd=self._parent.descriptor)
        except OSError as error:
            self._leave(self.path, os.strerror(error.errno or 0))

    def _remove(self, entry: Created) -> str | None:
        """Remove one created entry if it is still ours; otherwise say why not."""
        with ExitStack() as held:
            parent = self.root
            if entry.relative.parent.parts:
                try:
                    parent = held.enter_context(self.root.descend("cleanup", entry.relative.parent))
                except (OSError, Refused):
                    return "its directory is gone or no longer a directory"
            if parent.creation != entry.parent:
                return "its directory was replaced since this job created it"
            try:
                current = parent.creation_of(entry.relative.name)
            except OSError:
                return "replaced since this job created it; left in place"
            if current is None:
                return None
            if current != entry.identity:
                return "replaced since this job created it; left in place"
            # WAIVER(TOCTOU): POSIX has no unlink by descriptor, so the name is removed
            # after its identity is checked. Permanent: the window is two system calls.
            remove = os.rmdir if entry.is_directory else os.unlink
            try:
                remove(entry.relative.name, dir_fd=parent.descriptor)
            except OSError as error:
                return os.strerror(error.errno or 0)
        return None

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


def _clone_darwin(source: File, directory: Directory, name: str) -> Creation | None:
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
    # macOS gives no generation (see `generation`).
    return Creation(identity(status), None)


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
