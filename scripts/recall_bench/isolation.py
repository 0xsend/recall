"""Path capabilities, verified templates, run directories and child environments.

Threat model: the caller is a careless agent or one holding stale context. It
passes a wrong path, copies another lane's command, or runs an older build,
while other lanes create, clean up and replace their own files concurrently.
The assets are the host's production recall data, config, tool install and
service units, plus other lanes' files. A deliberately escaping build and a
hostile local account are out of scope.

Jobs obtain every path through `Policy` and `WriteRoots`, which return held
descriptors (`fs.Directory`, `fs.File`) checked by identity, so a production
location cannot be expressed as an argument and a checked name cannot be
swapped before use. Fixtures open only through `Template` and `RunDir`, so a
backup original can never become a data directory.
"""

from __future__ import annotations

import json
import os
import pwd
import stat
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .evidence import TOOL_ROOT
from .fs import (
    ALIAS_REASON,
    Cancellation,
    Directory,
    File,
    Identity,
    Leftovers,
    NewTree,
    Refused,
    identity,
)

__all__ = ["Refused"]

# Recall's per-account state, the stable tool install, and service definitions.
ACCOUNT_TREES = (
    ".local/share/recall",
    ".config/recall",
    ".local/state/recall",
    ".cache/recall",
    ".local/share/uv/tools/recall",
    ".local/bin",
    "Library/LaunchAgents",
    ".config/systemd/user",
)
XDG_HOMES = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME")
SOURCES = ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
WRITE_ROOTS_VARIABLE = "BENCH_WRITE_ROOTS"
WRITE_ROOTS_MAX = 4

MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = "recall-bench-template/2"
PAYLOAD_DIR = "payload"
POOL_DIR = "pool"
USAGE_LOG_NAME = "usage.jsonl"
CONFIG_NAME = "config.toml"
SOCKET_NAME = "recall.sock"
RUN_DIRECTORIES = ("tmp", "spill", "hf")
# Payload files land at the top of a run directory, beside these.
RESERVED_PAYLOAD_NAMES = frozenset(
    {POOL_DIR, USAGE_LOG_NAME, CONFIG_NAME, SOCKET_NAME, "fleet.toml", *RUN_DIRECTORIES}
)

# sizeof(sockaddr_un.sun_path), which includes the terminating NUL.
SOCKET_PATH_BYTES = {"darwin": 104, "linux": 108}
FILE_LIMIT = 200_000
DIRECTORY_LIMIT = 20_000
DEPTH_LIMIT = 32
SHEBANG_BYTES = 4096
INTERPRETER_HOPS_MAX = 16
MOUNTINFO = Path("/proc/self/mountinfo")
SYSTEM_PATH = "/usr/bin:/bin"
INHERITED_ENVIRONMENT = ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE")


def _plain_absolute(argument: str, raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise Refused(argument, raw, "path must be absolute")
    if ".." in path.parts:
        raise Refused(argument, raw, "path must not contain '..'")
    return path


def _folded_within(inner: Path, outer: Path) -> bool:
    folded_inner, folded_outer = str(inner).casefold(), str(outer).casefold()
    return folded_inner == folded_outer or folded_inner.startswith(folded_outer.rstrip("/") + "/")


# A location this account cannot stat is compared by path, like one that does not exist.
UNREACHABLE = (FileNotFoundError, NotADirectoryError, PermissionError)


def _host_identities(path: Path) -> Iterator[Identity]:
    """Identities of `path` and each reachable ancestor, as the host resolves them."""
    for candidate in (path, *path.parents):
        try:
            yield identity(os.stat(candidate))
        except UNREACHABLE:
            continue


@dataclass(frozen=True)
class Mounts:
    """Each Linux mount's root within its filesystem; a bind mount's root is not `/`.

    Descriptors carry their mount id (`fs.mount_id`), so this is looked up by
    what a held descriptor is on, never by matching path strings.
    """

    roots: Mapping[int, str]

    @classmethod
    def for_host(cls) -> Mounts:
        if sys.platform != "linux":
            # No bind mounts on macOS; mount crossings change st_dev and are refused there.
            return cls({})
        roots: dict[int, str] = {}
        for line in MOUNTINFO.read_text().splitlines():
            fields = line.split(" ")
            roots[int(fields[0])] = _unescape(fields[3])
        return cls(roots)

    def check(self, argument: str, path: Path, mount: int | None) -> None:
        if mount is None:
            return
        root = self.roots.get(mount)
        if root is None:
            raise Refused(argument, path, "is on a mount that appeared after the job started")
        if root != "/":
            raise Refused(argument, path, f"is inside a bind mount of {root}")


def _unescape(field_text: str) -> str:
    # mountinfo escapes space, tab, newline and backslash as three octal digits.
    return field_text.encode().decode("unicode_escape").encode("latin-1").decode()


@dataclass(frozen=True)
class Policy:
    """Where arguments may point, decided by file identity.

    No argument may be inside, equal to, or contain a protected location, and
    none may be or contain an account home. Locations that do not exist yet
    are compared by case-folded path instead.
    """

    production: tuple[Path, ...]
    homes: tuple[Path, ...]
    checkouts: tuple[Path, ...]
    mounts: Mounts
    _inside: Mapping[Identity, Path] = field(repr=False)
    _containing: Mapping[Identity, Path] = field(repr=False)
    _home_containing: Mapping[Identity, Path] = field(repr=False)

    @classmethod
    def build(
        cls,
        production: tuple[Path, ...],
        homes: tuple[Path, ...],
        checkouts: tuple[Path, ...],
        mounts: Mounts,
    ) -> Policy:
        inside: dict[Identity, Path] = {}
        containing: dict[Identity, Path] = {}
        for location in (*production, *checkouts):
            with suppress(*UNREACHABLE):
                inside.setdefault(identity(os.stat(location)), location)
            for found in _host_identities(location):
                containing.setdefault(found, location)
        home_containing: dict[Identity, Path] = {}
        for home in homes:
            for found in _host_identities(home):
                home_containing.setdefault(found, home)
        return cls(production, homes, checkouts, mounts, inside, containing, home_containing)

    @classmethod
    def for_host(cls, environ: Mapping[str, str] = os.environ) -> Policy:
        homes = {Path(pwd.getpwuid(os.getuid()).pw_dir)}
        if environ.get("HOME"):
            homes.add(Path(environ["HOME"]))
        trees = {home / relative for home in homes for relative in ACCOUNT_TREES}
        trees.update(Path(environ[name]) / "recall" for name in XDG_HOMES if environ.get(name))
        trees.update(_service_and_tool_locations(environ))
        return cls.build(
            production=tuple(sorted(Path(os.path.realpath(tree)) for tree in trees)),
            homes=tuple(sorted(Path(os.path.realpath(home)) for home in homes)),
            checkouts=(TOOL_ROOT,),
            mounts=Mounts.for_host(),
        )

    def with_checkout(self, path: Path) -> Policy:
        return Policy.build(self.production, self.homes, (*self.checkouts, path), self.mounts)

    def without_checkouts(self) -> Policy:
        return Policy.build(self.production, self.homes, (), self.mounts)

    def _missing(self) -> Iterator[Path]:
        for location in (*self.production, *self.checkouts):
            if not os.path.lexists(location):
                yield location

    def check_spelling(self, argument: str, path: Path) -> None:
        """Refuse by path where the protected location does not exist yet."""
        for location in self._missing():
            if _folded_within(path, location) or _folded_within(location, path):
                raise Refused(argument, path, f"overlaps refused location {location}")

    def inspector(self, argument: str) -> Callable[[Path, os.stat_result], None]:
        """Refuse a walked component that is a protected location."""

        def inspect(path: Path, status: os.stat_result) -> None:
            location = self._inside.get(identity(status))
            if location is not None:
                raise Refused(argument, path, f"overlaps refused location {location}")

        return inspect

    def check_target(self, argument: str, path: Path, status: os.stat_result) -> None:
        """Refuse a resolved target that is or contains a home or protected location."""
        home = self._home_containing.get(identity(status))
        if home is not None:
            raise Refused(argument, path, f"is or contains the account home {home}")
        location = self._containing.get(identity(status))
        if location is not None:
            raise Refused(argument, path, f"overlaps refused location {location}")

    def open_tree(self, argument: str, raw: str | Path) -> Directory:
        """A read-only directory, held by descriptor."""
        path = _plain_absolute(argument, raw)
        directory = Directory.walk(argument, path, self.inspector(argument))
        try:
            self.check_target(argument, path, directory.status)
            self.check_spelling(argument, path)
            self.mounts.check(argument, path, directory.mount)
        except BaseException:
            directory.close()
            raise
        return directory

    def open_file(self, argument: str, raw: str | Path) -> File:
        """A read-only regular file with no other name, held by descriptor."""
        path = _plain_absolute(argument, raw)
        with self.walk_parent(argument, path) as parent:
            status = parent.lstat(path.name)
            if status is not None and stat.S_ISDIR(status.st_mode):
                self.check_target(argument, path, status)
            self.check_spelling(argument, path)
            if status is None:
                raise Refused(argument, raw, "does not exist")
            if stat.S_ISLNK(status.st_mode):
                raise Refused(argument, raw, f"symlinked component {path}")
            file = parent.open_file(argument, path.name)
        try:
            self.mounts.check(argument, path, file.mount)
        except BaseException:
            file.close()
            raise
        return file

    def walk_parent(self, argument: str, path: Path) -> Directory:
        """Open the directory holding `path`, preferring an overlap as the reason to refuse."""
        try:
            return Directory.walk(argument, path.parent, self.inspector(argument))
        except Refused:
            self.check_spelling(argument, path)
            raise

    def open_write_roots(self, environ: Mapping[str, str] = os.environ) -> WriteRoots:
        value = environ.get(WRITE_ROOTS_VARIABLE, "")
        paths = [Path(part) for part in value.split(":") if part]
        if not paths:
            raise Refused(
                WRITE_ROOTS_VARIABLE,
                "",
                "is not set; the lane launcher names the directories bench jobs may write in",
            )
        if len(paths) > WRITE_ROOTS_MAX:
            raise Refused(WRITE_ROOTS_VARIABLE, value, f"names more than {WRITE_ROOTS_MAX} roots")
        roots = WriteRoots(self, [])
        try:
            for path in paths:
                roots.held.append(self._open_write_root(path))
        except BaseException:
            roots.close()
            raise
        return roots

    def _open_write_root(self, raw: Path) -> Directory:
        argument = WRITE_ROOTS_VARIABLE
        path = _plain_absolute(argument, raw)
        directory = Directory.walk(argument, path, self.inspector(argument))
        try:
            # A lane root may hold checkouts; it may not hold production state or a home.
            self.without_checkouts().check_target(argument, path, directory.status)
            self.without_checkouts().check_spelling(argument, path)
            self.mounts.check(argument, path, directory.mount)
        except BaseException:
            directory.close()
            raise
        return directory


def _service_and_tool_locations(environ: Mapping[str, str]) -> Iterator[Path]:
    """Effective service-unit and uv tool locations under the caller's environment."""
    if environ.get("XDG_CONFIG_HOME"):
        yield Path(environ["XDG_CONFIG_HOME"]) / "systemd/user"
    if environ.get("UV_TOOL_DIR"):
        yield Path(environ["UV_TOOL_DIR"]) / "recall"
    if environ.get("XDG_DATA_HOME"):
        data = Path(environ["XDG_DATA_HOME"])
        yield data / "uv/tools/recall"
        yield data.parent / "bin"
    for name in ("UV_TOOL_BIN_DIR", "XDG_BIN_HOME"):
        if environ.get(name):
            yield Path(environ[name])


@dataclass
class WriteRoots:
    """The directories this job may create in, held open for the whole job."""

    policy: Policy
    held: list[Directory]
    # What failed jobs could not remove; the job's outcome line reports it.
    leftovers: Leftovers = field(default_factory=list)

    def new_directory(self, argument: str, raw: str | Path) -> NewDirectory:
        path = _plain_absolute(argument, raw)
        policy = self.policy
        with policy.walk_parent(argument, path) as walked:
            existing = walked.lstat(path.name)
            if existing is not None and not stat.S_ISLNK(existing.st_mode):
                policy.check_target(argument, path, existing)
            policy.check_spelling(argument, path)
            if existing is not None:
                raise Refused(argument, raw, "already exists; every run uses a new directory")
            parent_identity = walked.identity
        root = next(
            (
                root
                for root in self.held
                if len(path.parts) > len(root.path.parts)
                and path.parts[: len(root.path.parts)] == root.path.parts
            ),
            None,
        )
        if root is None:
            roots = ":".join(str(root.path) for root in self.held)
            raise Refused(argument, raw, f"is outside every write root ({roots})")
        relative = PurePosixPath(*path.parts[len(root.path.parts) : -1])
        parent = root.descend(argument, relative) if relative.parts else _reopen(root)
        try:
            same_mount = (parent.status.st_dev, parent.mount) == (root.status.st_dev, root.mount)
            if parent.identity != parent_identity or not same_mount:
                raise Refused(argument, raw, "parent directory changed or crosses a mount point")
        except BaseException:
            parent.close()
            raise
        return NewDirectory(parent, path.name, self.leftovers)

    def close(self) -> None:
        for directory in self.held:
            directory.close()
        self.held.clear()

    def __enter__(self) -> WriteRoots:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _reopen(directory: Directory) -> Directory:
    return Directory(directory.path, os.dup(directory.descriptor))


@dataclass
class NewDirectory:
    """A name that did not exist inside a held parent within a write root."""

    parent: Directory
    name: str
    leftovers: Leftovers

    @property
    def path(self) -> Path:
        return self.parent.path / self.name

    def create(self) -> NewTree:
        return NewTree(self.parent, self.name, self.leftovers)

    def close(self) -> None:
        self.parent.close()

    def __enter__(self) -> NewDirectory:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class Signature:
    """What changes when a file is written, renamed over, or relinked."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    links: int

    @classmethod
    def of(cls, status: os.stat_result) -> Signature:
        return cls(
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
            status.st_nlink,
        )


@dataclass(frozen=True)
class Inventory:
    """Every directory and file below a tree root, by relative path."""

    directories: Mapping[PurePosixPath, Identity]
    files: Mapping[PurePosixPath, Signature]


def scan(argument: str, root: Directory) -> Inventory:
    """List a tree without following anything, refusing what could alias other data.

    Symlinks, special files, multiply linked files and mount crossings are
    refused: each can make a path under `root` name bytes that live elsewhere.
    """
    directories: dict[PurePosixPath, Identity] = {}
    files: dict[PurePosixPath, Signature] = {}

    def visit(directory: Directory, relative: PurePosixPath, depth: int) -> None:
        if depth > DEPTH_LIMIT:
            raise Refused(argument, directory.path, f"deeper than {DEPTH_LIMIT} directories")
        listing = directory.list(argument)
        if listing.symlinks:
            raise Refused(argument, directory.path / listing.symlinks[0], "symlinked payload file")
        if listing.others:
            raise Refused(argument, directory.path / listing.others[0], "not a regular file")
        for name in listing.files:
            status = directory.lstat(name)
            if status is None:
                raise Refused(argument, directory.path / name, "changed while listing")
            if status.st_dev != root.status.st_dev:
                raise Refused(argument, directory.path / name, "crosses a mount point")
            if status.st_nlink != 1:
                raise Refused(argument, directory.path / name, ALIAS_REASON)
            files[relative / name] = Signature.of(status)
            if len(files) > FILE_LIMIT:
                raise Refused(argument, root.path, f"more than {FILE_LIMIT} files")
        for name in listing.directories:
            with directory.open_directory(argument, name) as child:
                if (child.status.st_dev, child.mount) != (root.status.st_dev, root.mount):
                    raise Refused(argument, child.path, "crosses a mount point")
                directories[relative / name] = child.identity
                if len(directories) > DIRECTORY_LIMIT:
                    raise Refused(argument, root.path, f"more than {DIRECTORY_LIMIT} directories")
                visit(child, relative / name, depth + 1)

    visit(root, PurePosixPath(), 0)
    return Inventory(directories, files)


@contextmanager
def open_parent(root: Directory, relative: PurePosixPath, argument: str) -> Iterator[Directory]:
    """The held directory containing `relative`, opened from `root` without following."""
    if relative.parent.parts:
        with root.descend(argument, relative.parent) as parent:
            yield parent
    else:
        yield root


def open_entry(root: Directory, relative: PurePosixPath, argument: str) -> File:
    with open_parent(root, relative, argument) as parent:
        file = parent.open_file(argument, relative.name)
    if (file.status.st_dev, file.mount) != (root.status.st_dev, root.mount):
        file.close()
        raise Refused(argument, file.path, "crosses a mount point")
    return file


@dataclass(frozen=True)
class FileRecord:
    path: str
    bytes: int
    sha256: str


def _relative(text: object, what: str) -> PurePosixPath:
    if not isinstance(text, str) or not text or text.startswith("/"):
        raise ValueError(f"manifest {what} is not a relative path: {text!r}")
    path = PurePosixPath(text)
    if str(path) != text or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"manifest {what} is not a normalized relative path: {text!r}")
    return path


def _layout_error(path: PurePosixPath, is_directory: bool) -> str | None:
    """Why a template path is outside the layout `prepare` writes, or None."""
    parts = path.parts
    if parts[0] == PAYLOAD_DIR:
        if is_directory:
            return None if len(parts) == 1 else "payload holds no directories"
        if len(parts) != 2:
            return "payload files are flat"
        return "name is reserved in a run directory" if parts[1] in RESERVED_PAYLOAD_NAMES else None
    if parts[0] == POOL_DIR:
        if len(parts) == 1:
            return None if is_directory else "pool is a directory"
        return (
            None
            if parts[1] in SOURCES and (is_directory or len(parts) > 2)
            else (f"pool entries must be under one of {', '.join(SOURCES)}")
        )
    if parts == (USAGE_LOG_NAME,) and not is_directory:
        return None
    return "not part of the template layout"


@dataclass(frozen=True)
class Manifest:
    """Every directory and file of a template, with each file's size and SHA-256.

    Parsing enforces the template layout, so a parsed manifest maps onto a
    run directory without two entries colliding.
    """

    directories: tuple[str, ...]
    files: tuple[FileRecord, ...]
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        directories = {PurePosixPath(name) for name in self.directories}
        for name in self.directories:
            path = PurePosixPath(name)
            if (reason := _layout_error(path, True)) is not None:
                raise ValueError(f"manifest directory {name}: {reason}")
            if path.parent.parts and path.parent not in directories:
                raise ValueError(f"manifest directory {name} has no listed parent")
        for record in self.files:
            path = PurePosixPath(record.path)
            if (reason := _layout_error(path, False)) is not None:
                raise ValueError(f"manifest file {record.path}: {reason}")
            if path.parent.parts and path.parent not in directories:
                raise ValueError(f"manifest file {record.path} has no listed directory")
        if PurePosixPath(PAYLOAD_DIR) not in directories:
            raise ValueError("manifest has no payload directory")
        names = [*self.directories, *(record.path for record in self.files)]
        if len(set(names)) != len(names):
            raise ValueError("manifest lists a path twice")
        mapped = [run_relative(name) for name in names if name != PAYLOAD_DIR]
        assert len(set(mapped)) == len(mapped), "the layout maps entries onto one run path"

    def to_json(self) -> bytes:
        document = {
            "format": MANIFEST_FORMAT,
            "directories": list(self.directories),
            "files": [
                {"path": record.path, "bytes": record.bytes, "sha256": record.sha256}
                for record in self.files
            ],
            "provenance": dict(self.provenance),
        }
        return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()

    @classmethod
    def parse(cls, data: bytes) -> Manifest:
        document = json.loads(data)
        if not isinstance(document, dict) or document.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"not a {MANIFEST_FORMAT} manifest")
        directories, files = document.get("directories"), document.get("files")
        if not isinstance(directories, list) or not isinstance(files, list):
            raise ValueError("manifest directories and files must be lists")
        records: list[FileRecord] = []
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("manifest file entry must be an object")
            path = str(_relative(item.get("path"), "file"))
            size, digest = item.get("bytes"), item.get("sha256")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError(f"manifest size for {path} is invalid")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"manifest sha256 for {path} is invalid")
            records.append(FileRecord(path, size, digest))
        provenance = document.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("manifest provenance must be an object")
        names = tuple(str(_relative(name, "directory")) for name in directories)
        return cls(names, tuple(records), provenance)


def run_relative(template_path: str) -> str:
    """Where a template entry lives in a run directory: payload files at the top."""
    path = PurePosixPath(template_path)
    if path.parts[0] == PAYLOAD_DIR:
        return str(path.relative_to(PAYLOAD_DIR))
    return template_path


def verify_tree(
    argument: str,
    root: Directory,
    directories: set[str],
    files: Mapping[str, FileRecord],
    cancellation: Cancellation,
) -> None:
    """Refuse unless `root` holds exactly these directories and files, byte for byte."""
    inventory = scan(argument, root)
    found_directories = {str(path) for path in inventory.directories}
    if found_directories != directories:
        missing = sorted(directories - found_directories)
        extra = sorted(found_directories - directories)
        raise Refused(
            argument,
            root.path,
            f"manifest mismatch: missing directories {missing}, unexpected {extra}",
        )
    found = {str(path) for path in inventory.files}
    missing, extra = sorted(files.keys() - found), sorted(found - files.keys())
    if missing or extra:
        raise Refused(
            argument, root.path, f"manifest mismatch: missing {missing}, unexpected {extra}"
        )
    for name, record in sorted(files.items()):
        with open_entry(root, PurePosixPath(name), argument) as file:
            if file.status.st_size != record.bytes:
                raise Refused(argument, file.path, "manifest mismatch: size differs")
            if file.sha256(cancellation) != record.sha256:
                raise Refused(argument, file.path, "manifest mismatch: sha256 differs")


class Template:
    """A prepared template, held open, whose contents matched its manifest when opened."""

    def __init__(self, root: Directory, manifest: Manifest) -> None:
        self.root = root
        self.manifest = manifest

    @property
    def path(self) -> Path:
        return self.root.path

    @classmethod
    def open(
        cls, policy: Policy, argument: str, raw: str | Path, cancellation: Cancellation
    ) -> Template:
        root = policy.open_tree(argument, raw)
        try:
            return cls.from_root(argument, root, cancellation)
        except BaseException:
            root.close()
            raise

    @classmethod
    def from_root(cls, argument: str, root: Directory, cancellation: Cancellation) -> Template:
        """Verify a held template directory; the template then owns `root`."""
        with root.open_file(argument, MANIFEST_NAME) as manifest_file:
            data = manifest_file.head(manifest_file.status.st_size)
            try:
                manifest = Manifest.parse(data)
            except ValueError as error:
                raise Refused(argument, manifest_file.path, f"invalid manifest: {error}") from error
            files = {record.path: record for record in manifest.files}
            if MANIFEST_NAME in files:
                raise Refused(argument, manifest_file.path, "manifest lists itself")
            # The manifest is the one file it cannot describe; it must still be the only extra.
            files[MANIFEST_NAME] = FileRecord(
                MANIFEST_NAME, len(data), manifest_file.sha256(cancellation)
            )
        verify_tree(argument, root, set(manifest.directories), files, cancellation)
        return cls(root, manifest)

    def close(self) -> None:
        self.root.close()

    def __enter__(self) -> Template:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def check_socket_path(argument: str, data_dir: Path) -> None:
    """Refuse a data directory whose socket path cannot be bound."""
    limit = SOCKET_PATH_BYTES.get(sys.platform)
    if limit is None:
        raise Refused(argument, data_dir, f"unsupported platform {sys.platform}")
    socket_path = data_dir / SOCKET_NAME
    length = len(os.fsencode(socket_path)) + 1
    if length > limit:
        raise Refused(
            argument,
            data_dir,
            f"socket path {socket_path} needs {length} bytes with NUL; the limit is {limit}",
        )


def _toml_string(value: Path) -> str:
    # JSON string escapes are a subset of TOML basic-string escapes.
    return json.dumps(str(value))


@dataclass(frozen=True)
class RunDir:
    """A fresh, verified clone of a template; it is the daemon's data directory.

    Recall processes receive its path by name, so the run keeps the identity
    of its root, of every directory it created and of its config, and
    `held()` proves the name still means this run before a child is given it.
    """

    path: Path
    identity: Identity
    directories: tuple[tuple[str, Identity], ...]
    config_identity: Identity
    policy: Policy = field(repr=False, compare=False)

    @classmethod
    def create(
        cls,
        roots: WriteRoots,
        argument: str,
        raw: str | Path,
        template: Template,
        cancellation: Cancellation,
    ) -> RunDir:
        with roots.new_directory(argument, raw) as target:
            check_socket_path(argument, target.path)
            records = {run_relative(record.path): record for record in template.manifest.files}
            directories = {
                run_relative(name) for name in template.manifest.directories if name != PAYLOAD_DIR
            }
            with target.create() as tree:
                for name in sorted(directories, key=lambda name: PurePosixPath(name).parts):
                    relative = PurePosixPath(name)
                    with open_parent(tree.root, relative, argument) as parent:
                        tree.make_directory(parent, relative).close()
                for record in template.manifest.files:
                    relative = PurePosixPath(run_relative(record.path))
                    with (
                        open_entry(template.root, PurePosixPath(record.path), argument) as source,
                        open_parent(tree.root, relative, argument) as parent,
                    ):
                        tree.clone(parent, relative, source, cancellation).close()
                verify_tree(
                    argument,
                    tree.root,
                    directories,
                    {
                        name: FileRecord(name, record.bytes, record.sha256)
                        for name, record in records.items()
                    },
                    cancellation,
                )
                for name in RUN_DIRECTORIES:
                    tree.make_directory(tree.root, PurePosixPath(name)).close()
                # Every source gets a root, empty when the template's pool has none.
                pool = PurePosixPath(POOL_DIR)
                with ExitStack() as held:
                    pool_directory = (
                        held.enter_context(tree.root.open_directory(argument, POOL_DIR))
                        if POOL_DIR in directories
                        else held.enter_context(tree.make_directory(tree.root, pool))
                    )
                    for source in SOURCES:
                        if f"{POOL_DIR}/{source}" not in directories:
                            tree.make_directory(pool_directory, pool / source).close()
                config = tree.write_new(
                    tree.root, PurePosixPath(CONFIG_NAME), _config_text(target.path).encode()
                )
                tree.commit(cancellation)
                return cls(
                    target.path,
                    tree.root.identity,
                    tuple(sorted((str(path), made) for path, made in tree.directories().items())),
                    config,
                    roots.policy,
                )

    @property
    def config_path(self) -> Path:
        return self.path / CONFIG_NAME

    @property
    def socket_path(self) -> Path:
        return self.path / SOCKET_NAME

    def config_text(self) -> str:
        return _config_text(self.path)

    @contextmanager
    def held(self) -> Iterator[Directory]:
        """The run directory as its name resolves now, refused unless it is still this run.

        Its top level may gain product files (the database is replaced by
        rename during compaction), but no link, other name or mount crossing.
        """
        argument = "--run-dir"
        with self.policy.open_tree(argument, self.path) as root:
            if root.identity != self.identity:
                raise Refused(
                    argument, self.path, "is no longer the directory created for this run"
                )
            for name, status in root.statuses(argument):
                if stat.S_ISLNK(status.st_mode):
                    raise Refused(
                        argument, root.path / name, "symlinked entry in the run directory"
                    )
                if status.st_dev != root.status.st_dev:
                    raise Refused(argument, root.path / name, "crosses a mount point")
                if stat.S_ISREG(status.st_mode) and status.st_nlink != 1:
                    raise Refused(argument, root.path / name, ALIAS_REASON)
            for name, expected in self.directories:
                with root.descend(argument, PurePosixPath(name)) as directory:
                    place = (directory.status.st_dev, directory.mount)
                    if directory.identity != expected or place != (root.status.st_dev, root.mount):
                        raise Refused(
                            argument, directory.path, "was replaced since the run was created"
                        )
            expected_config = self.config_text().encode()
            with root.open_file(argument, CONFIG_NAME) as config:
                unchanged = identity(config.status) == self.config_identity
                if not unchanged or config.head(len(expected_config) + 1) != expected_config:
                    raise Refused(argument, config.path, "changed since the run was created")
            yield root

    def recall_paths(self) -> dict[str, str]:
        return {
            "RECALL_DATA_DIR": str(self.path),
            "RECALL_CONFIG_PATH": str(self.config_path),
            "RECALL_DB_PATH": str(self.path / "recall.duckdb"),
            "RECALL_LOCK_PATH": str(self.path / "recall.lock"),
            "RECALL_FLEET_PATH": str(self.path / "fleet.toml"),
            "RECALL_DUCKDB_TEMP_DIR": str(self.path / "spill"),
        }


def _config_text(path: Path) -> str:
    """Pin every source root inside the run and keep the scheduler off services.

    `pid` is a fail-closed placeholder: builds up to at least 0.40.5
    reject it, so no recall child can load this config until the
    supervisor's scheduler-probe stubs replace it (recall#186 step 2).
    """
    lines = ["[daemon]", 'scheduler = "pid"', ""]
    for source in SOURCES:
        root = _toml_string(path / POOL_DIR / source)
        lines += [f"[sources.{source}]", f"roots = [{root}]", ""]
    return "\n".join(lines)


@dataclass(frozen=True)
class Build:
    """The checkout under test; its own virtualenv runs every recall process.

    The environment directories and the `recall` entry point must be real
    entries inside the checkout, and the entry point must name the
    environment's own interpreter. The interpreter itself may link to a
    shared Python install, but no link on its way may pass through a
    protected location.
    """

    path: Path
    recall_identity: Identity
    interpreter: Path
    interpreter_identity: Identity
    policy: Policy = field(repr=False, compare=False)

    @classmethod
    def resolve(cls, policy: Policy, argument: str, raw: str | Path) -> Build:
        # The build may be the tool checkout itself, never a production install.
        production = policy.without_checkouts()
        with production.open_tree(argument, raw) as checkout, ExitStack() as held:
            path = checkout.path
            python = path / ".venv/bin/python"
            venv = held.enter_context(checkout.open_directory(argument, ".venv"))
            bin_directory = held.enter_context(venv.open_directory(argument, "bin"))
            lib = held.enter_context(venv.open_directory(argument, "lib"))
            versions = [
                name for name in lib.list(argument).directories if name.startswith("python")
            ]
            if len(versions) != 1:
                raise Refused(argument, raw, f"{lib.path} must hold exactly one python directory")
            site = held.enter_context(
                lib.descend(argument, PurePosixPath(versions[0], "site-packages"))
            )
            places = {
                (d.status.st_dev, d.mount) for d in (checkout, venv, bin_directory, lib, site)
            }
            if len(places) != 1:
                raise Refused(argument, raw, "the virtualenv crosses a mount point")
            # PATH starts at .venv/bin, so its other interpreter names must alias `python`.
            aliases = {
                (bin_directory, "python3"): "python",
                (bin_directory, versions[0]): "python",
                (venv, "lib64"): "lib",
            }
            for (directory, name), target in aliases.items():
                _check_alias(argument, directory, name, target)
            status = bin_directory.lstat("recall")
            if status is not None and stat.S_ISLNK(status.st_mode):
                raise Refused(argument, raw, f"symlinked component {bin_directory.path / 'recall'}")
            with bin_directory.open_file(argument, "recall") as recall:
                if not recall.status.st_mode & 0o111:
                    raise Refused(argument, raw, f"{recall.path} is not an executable file")
                first_line = recall.head(SHEBANG_BYTES).split(b"\n", 1)[0]
                if first_line != b"#!" + os.fsencode(python):
                    raise Refused(argument, raw, f"{recall.path} does not run {python}")
                recall_identity = identity(recall.status)
            with venv.open_file(argument, "pyvenv.cfg") as config:
                home = _pyvenv_home(config.head(SHEBANG_BYTES))
            if home is not None:
                _check_outside(production, argument, Path(home), f"pyvenv.cfg home {home}")
        interpreter, interpreter_identity = _interpreter(production, argument, python)
        return cls(path, recall_identity, interpreter, interpreter_identity, policy)

    def recheck(self) -> None:
        """Resolve the build again and refuse unless it is the one resolved before."""
        again = Build.resolve(self.policy, "--build", self.path)
        if again.recall_identity != self.recall_identity:
            raise Refused("--build", self.recall, "the recall entry point changed since resolution")
        if (again.interpreter, again.interpreter_identity) != (
            self.interpreter,
            self.interpreter_identity,
        ):
            raise Refused("--build", self.python, "the interpreter changed since resolution")

    @property
    def bin(self) -> Path:
        return self.path / ".venv" / "bin"

    @property
    def python(self) -> Path:
        return self.bin / "python"

    @property
    def recall(self) -> Path:
        return self.bin / "recall"


def _check_alias(argument: str, directory: Directory, name: str, target: str) -> None:
    status = directory.lstat(name)
    if status is None:
        return
    linked = stat.S_ISLNK(status.st_mode) and os.readlink(name, dir_fd=directory.descriptor)
    if linked != target:
        raise Refused(argument, directory.path / name, f"must be absent or a link to {target}")


def _pyvenv_home(data: bytes) -> str | None:
    for line in data.decode(errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "home":
            return value.strip()
    return None


def _check_outside(policy: Policy, argument: str, path: Path, subject: str) -> Path:
    """Refuse if `path`, as the host resolves it, lies in a protected location."""
    real = Path(os.path.realpath(path))
    for directory in (real, *real.parents):
        try:
            policy.inspector(argument)(directory, os.stat(directory))
        except FileNotFoundError:
            raise Refused(argument, path, f"{subject} does not exist") from None
    return real


def _interpreter(policy: Policy, argument: str, python: Path) -> tuple[Path, Identity]:
    """Follow the interpreter's links, refusing any hop inside a protected location."""
    hop = python
    for _ in range(INTERPRETER_HOPS_MAX):
        parent = _check_outside(policy, argument, hop.parent, f"interpreter link {hop}")
        try:
            status = os.lstat(parent / hop.name)
        except FileNotFoundError:
            raise Refused(argument, python, f"interpreter link {hop} is dangling") from None
        if stat.S_ISLNK(status.st_mode):
            hop = Path(os.path.normpath(parent / os.readlink(parent / hop.name)))
            continue
        if not (stat.S_ISREG(status.st_mode) and status.st_mode & 0o111):
            raise Refused(argument, python, f"{hop} is not an executable file")
        return parent / hop.name, identity(status)
    raise Refused(argument, python, f"more than {INTERPRETER_HOPS_MAX} interpreter links")


def child_environment(
    build: Build, run: RunDir, environ: Mapping[str, str] = os.environ
) -> dict[str, str]:
    """An allow-listed environment: nothing ambient reaches a recall child.

    Every `RECALL_*` path, `TMPDIR` and `HF_HOME` are inside the run
    directory. HOME stays the real account home because the build's
    resolvers require one, and PATH is the build's virtualenv before the
    fixed system path. The service-manager variables are absent, so
    `systemctl --user` cannot reach the account's manager.
    """
    if not environ.get("HOME"):
        raise Refused("HOME", "", "a recall child needs HOME")
    env = {name: environ[name] for name in INHERITED_ENVIRONMENT if environ.get(name)}
    env.update(run.recall_paths())
    env.update(
        {
            "PATH": f"{build.bin}:{SYSTEM_PATH}",
            "TMPDIR": str(run.path / "tmp"),
            "HF_HOME": str(run.path / "hf"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
        }
    )
    return env
