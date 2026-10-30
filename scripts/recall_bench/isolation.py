"""Path capabilities, verified templates, run directories and child environments.

Threat model: the caller is a careless agent or one holding stale context. It
passes a wrong path, copies another lane's command, or runs an older build. The
assets are the host's production recall data, config, tool install and service
units, plus other lanes' files. A deliberately escaping build and a hostile
local account are out of scope.

Jobs obtain every path through `Policy`, so a production location cannot be
expressed as an argument, and they open fixtures only through `Template` and
`RunDir`, so a backup original can never become a data directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import stat
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

from .evidence import TOOL_ROOT, Directory, check_name

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

MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = "recall-bench-template/1"
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
DEPTH_LIMIT = 32
CP = "/bin/cp"
CLONE_FLAGS = {"darwin": ("-c",), "linux": ("--reflink=auto",)}
CLONE_TIMEOUT_SECONDS = 3600
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
INHERITED_ENVIRONMENT = ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE")


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


def _identity(path: Path) -> tuple[int, int] | None:
    try:
        status = os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    return status.st_dev, status.st_ino


def _ancestors_or_self(path: Path) -> Iterator[Path]:
    yield path
    yield from path.parents


def _is_within(inner: Path, outer: Path) -> bool:
    """Whether `inner` is `outer` or below it, by file identity where both exist.

    Identity also holds on case-insensitive volumes, where two spellings name
    the same directory.
    """
    outer_identity = _identity(outer)
    if outer_identity is None:
        folded_inner, folded_outer = str(inner).casefold(), str(outer).casefold()
        return folded_inner == folded_outer or folded_inner.startswith(folded_outer + "/")
    return any(_identity(ancestor) == outer_identity for ancestor in _ancestors_or_self(inner))


def _plain_absolute(argument: str, raw: str | Path) -> Path:
    """An absolute path with no symlink at any existing component."""
    path = Path(raw)
    if not path.is_absolute():
        raise Refused(argument, raw, "path must be absolute")
    if ".." in path.parts:
        raise Refused(argument, raw, "path must not contain '..'")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except (FileNotFoundError, NotADirectoryError):
            break
        if stat.S_ISLNK(mode):
            raise Refused(argument, raw, f"symlinked component {current}")
    return path


@dataclass(frozen=True)
class ReadOnlyFile:
    path: Path


@dataclass(frozen=True)
class ReadOnlyTree:
    path: Path


@dataclass(frozen=True)
class NewDirectory:
    """A directory that did not exist when its argument was resolved."""

    path: Path

    def create(self) -> Directory:
        return Directory.create(self.path)


@dataclass(frozen=True)
class Policy:
    """Where arguments may point.

    No argument may be inside, equal to, or contain a production tree or a
    checkout, and none may be or contain an account home.
    """

    production: tuple[Path, ...]
    homes: tuple[Path, ...]
    checkouts: tuple[Path, ...] = ()

    @classmethod
    def for_host(cls, environ: Mapping[str, str] = os.environ) -> Policy:
        homes = {Path(pwd.getpwuid(os.getuid()).pw_dir)}
        if environ.get("HOME"):
            homes.add(Path(environ["HOME"]))
        trees = {home / relative for home in homes for relative in ACCOUNT_TREES}
        trees.update(Path(environ[name]) / "recall" for name in XDG_HOMES if environ.get(name))
        if environ.get("UV_TOOL_DIR"):
            trees.add(Path(environ["UV_TOOL_DIR"]) / "recall")
        return cls(
            production=tuple(sorted(Path(os.path.realpath(tree)) for tree in trees)),
            homes=tuple(sorted(Path(os.path.realpath(home)) for home in homes)),
            checkouts=(TOOL_ROOT,),
        )

    def with_checkout(self, path: Path) -> Policy:
        return replace(self, checkouts=(*self.checkouts, path))

    def check(self, argument: str, raw: str | Path) -> Path:
        path = _plain_absolute(argument, raw)
        for home in self.homes:
            if _is_within(home, path):
                raise Refused(argument, raw, f"is or contains the account home {home}")
        for tree in (*self.production, *self.checkouts):
            if _is_within(path, tree) or _is_within(tree, path):
                raise Refused(argument, raw, f"overlaps refused location {tree}")
        return path

    def read_file(self, argument: str, raw: str | Path) -> ReadOnlyFile:
        path = self.check(argument, raw)
        if not stat.S_ISREG(_lstat_mode(argument, path)):
            raise Refused(argument, raw, "not a regular file")
        return ReadOnlyFile(path)

    def read_tree(self, argument: str, raw: str | Path) -> ReadOnlyTree:
        path = self.check(argument, raw)
        if not stat.S_ISDIR(_lstat_mode(argument, path)):
            raise Refused(argument, raw, "not a directory")
        return ReadOnlyTree(path)

    def new_directory(self, argument: str, raw: str | Path) -> NewDirectory:
        path = self.check(argument, raw)
        if os.path.lexists(path):
            raise Refused(argument, raw, "already exists; every run uses a new directory")
        if not path.parent.is_dir():
            raise Refused(argument, raw, f"parent directory {path.parent} does not exist")
        return NewDirectory(path)


def _lstat_mode(argument: str, path: Path) -> int:
    try:
        return os.lstat(path).st_mode
    except FileNotFoundError:
        raise Refused(argument, path, "does not exist") from None


@dataclass(frozen=True)
class Entry:
    """A regular file found under a tree, named relative to that tree."""

    relative: PurePosixPath
    path: Path


def regular_files(argument: str, root: Path) -> list[Entry]:
    """Every file under `root`, refusing symlinks and anything not a file or directory."""
    entries: list[Entry] = []
    pending: list[tuple[PurePosixPath, int]] = [(PurePosixPath(), 0)]
    while pending:
        relative, depth = pending.pop()
        if depth > DEPTH_LIMIT:
            raise Refused(argument, root / relative, f"deeper than {DEPTH_LIMIT} directories")
        with os.scandir(root / relative) as listing:
            for item in listing:
                child = relative / item.name
                mode = item.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    pending.append((child, depth + 1))
                elif stat.S_ISREG(mode):
                    entries.append(Entry(child, root / child))
                elif stat.S_ISLNK(mode):
                    raise Refused(argument, root / child, "symlinked payload file")
                else:
                    raise Refused(argument, root / child, "not a regular file")
                if len(entries) > FILE_LIMIT:
                    raise Refused(argument, root, f"more than {FILE_LIMIT} files")
    return sorted(entries, key=lambda entry: entry.relative)


def sha256(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True)
class FileRecord:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Manifest:
    """Every file of a template, with its size and SHA-256."""

    files: tuple[FileRecord, ...]
    provenance: Mapping[str, object] = field(default_factory=dict)

    def to_json(self) -> bytes:
        document = {
            "format": MANIFEST_FORMAT,
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
        files = document.get("files")
        if not isinstance(files, list):
            raise ValueError("manifest files must be a list")
        records: list[FileRecord] = []
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("manifest file entry must be an object")
            path, size, digest = item.get("path"), item.get("bytes"), item.get("sha256")
            if not isinstance(path, str) or not _is_relative_file(path):
                raise ValueError(f"manifest path is not a relative file path: {path!r}")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError(f"manifest size for {path} is invalid")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"manifest sha256 for {path} is invalid")
            records.append(FileRecord(path, size, digest))
        if len({record.path for record in records}) != len(records):
            raise ValueError("manifest lists a path twice")
        provenance = document.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("manifest provenance must be an object")
        return cls(tuple(records), provenance)


def _is_relative_file(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (
        bool(parts) and not path.startswith("/") and all(part not in {".", ".."} for part in parts)
    )


def verify_files(argument: str, root: Path, expected: Mapping[str, FileRecord]) -> None:
    """Refuse unless `root` holds exactly the expected files, byte for byte."""
    found = {str(entry.relative): entry for entry in regular_files(argument, root)}
    missing = sorted(expected.keys() - found.keys())
    extra = sorted(found.keys() - expected.keys())
    if missing or extra:
        raise Refused(argument, root, f"manifest mismatch: missing {missing}, unexpected {extra}")
    for name, record in sorted(expected.items()):
        path = found[name].path
        if path.stat(follow_symlinks=False).st_size != record.bytes:
            raise Refused(argument, path, "manifest mismatch: size differs")
        if sha256(path) != record.sha256:
            raise Refused(argument, path, "manifest mismatch: sha256 differs")


@dataclass(frozen=True)
class Template:
    """A prepared template whose files matched its manifest when opened."""

    path: Path
    manifest: Manifest

    @classmethod
    def open(cls, policy: Policy, argument: str, raw: str | Path) -> Template:
        root = policy.read_tree(argument, raw).path
        manifest_path = root / MANIFEST_NAME
        if not stat.S_ISREG(_lstat_mode(argument, manifest_path)):
            raise Refused(argument, manifest_path, "manifest is not a regular file")
        try:
            manifest = Manifest.parse(manifest_path.read_bytes())
        except ValueError as error:
            raise Refused(argument, manifest_path, f"invalid manifest: {error}") from error
        expected = {record.path: record for record in manifest.files}
        if MANIFEST_NAME in expected:
            raise Refused(argument, manifest_path, "manifest lists itself")
        # The manifest is the one file it cannot describe; it must still be the only extra.
        expected[MANIFEST_NAME] = FileRecord(
            MANIFEST_NAME, manifest_path.stat().st_size, sha256(manifest_path)
        )
        verify_files(argument, root, expected)
        return cls(root, manifest)


def clone_file(source: Path, directory: Directory, name: str) -> int:
    """Copy-on-write clone where the volume supports it; returns cp's status."""
    flags = CLONE_FLAGS.get(sys.platform)
    if flags is None:
        raise OSError(f"no clone command for platform {sys.platform}")
    completed = subprocess.run(
        [CP, *flags, "--", str(source), str(directory.path / check_name(name))],
        capture_output=True,
        timeout=CLONE_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        raise OSError(
            f"cp exited {completed.returncode} cloning {source}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    if not directory.is_regular(name):
        raise OSError(f"clone of {source} is not a regular file")
    return completed.returncode


def clone_tree(files: Mapping[str, Path], root: Directory) -> None:
    """Clone each source file to its relative destination under `root`."""
    directories: dict[PurePosixPath, Directory] = {PurePosixPath(): root}
    try:
        for relative, source in sorted(files.items()):
            destination = PurePosixPath(relative)
            parent = PurePosixPath()
            for part in destination.parent.parts:
                child = parent / part
                if child not in directories:
                    directories[child] = directories[parent].subdirectory(part)
                parent = child
            clone_file(source, directories[parent], destination.name)
    finally:
        for relative, directory in directories.items():
            if relative != PurePosixPath():
                directory.close()


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


def run_relative(template_path: str) -> str:
    """Where a template file lives in a run directory: payload files at the top."""
    path = PurePosixPath(template_path)
    if path.parts[0] == PAYLOAD_DIR:
        return str(path.relative_to(PAYLOAD_DIR))
    return template_path


def _toml_string(value: Path) -> str:
    # JSON string escapes are a subset of TOML basic-string escapes.
    return json.dumps(str(value))


@dataclass(frozen=True)
class RunDir:
    """A fresh, verified clone of a template; it is the daemon's data directory."""

    path: Path

    @classmethod
    def create(cls, policy: Policy, argument: str, raw: str | Path, template: Template) -> RunDir:
        target = policy.new_directory(argument, raw)
        check_socket_path(argument, target.path)
        files = {
            run_relative(record.path): template.path / record.path
            for record in template.manifest.files
        }
        expected = {
            run_relative(record.path): replace(record, path=run_relative(record.path))
            for record in template.manifest.files
        }
        with target.create().held() as root:
            clone_tree(files, root)
            verify_files(argument, target.path, expected)
            run = cls(target.path)
            for name in RUN_DIRECTORIES:
                root.subdirectory(name).close()
            # Every source gets a root, empty when the template's pool has none.
            for directory in (POOL_DIR, *(f"{POOL_DIR}/{source}" for source in SOURCES)):
                (run.path / directory).mkdir(exist_ok=True)
            root.write_new(CONFIG_NAME, run.config_text().encode())
        return run

    @property
    def config_path(self) -> Path:
        return self.path / CONFIG_NAME

    @property
    def socket_path(self) -> Path:
        return self.path / SOCKET_NAME

    def config_text(self) -> str:
        """Pin every source root inside the run and keep the scheduler off services.

        Builds up to at least 0.40.5 reject `pid` and refuse to load this
        config, so every recall child fails closed until the supervisor's
        scheduler-probe stubs decide the value (recall#186 step 2).
        """
        lines = ["[daemon]", 'scheduler = "pid"', ""]
        for source in SOURCES:
            root = _toml_string(self.path / POOL_DIR / source)
            lines += [f"[sources.{source}]", f"roots = [{root}]", ""]
        return "\n".join(lines)

    def recall_paths(self) -> dict[str, str]:
        return {
            "RECALL_DATA_DIR": str(self.path),
            "RECALL_CONFIG_PATH": str(self.config_path),
            "RECALL_DB_PATH": str(self.path / "recall.duckdb"),
            "RECALL_LOCK_PATH": str(self.path / "recall.lock"),
            "RECALL_FLEET_PATH": str(self.path / "fleet.toml"),
            "RECALL_DUCKDB_TEMP_DIR": str(self.path / "spill"),
        }


@dataclass(frozen=True)
class Build:
    """The checkout under test; its virtualenv runs every recall process."""

    path: Path

    @classmethod
    def resolve(cls, policy: Policy, argument: str, raw: str | Path) -> Build:
        # The build may be the tool checkout itself, never a production install.
        production = replace(policy, checkouts=())
        build = cls(production.read_tree(argument, raw).path)
        for executable in (build.python, build.recall):
            if not (executable.is_file() and os.access(executable, os.X_OK)):
                raise Refused(argument, raw, f"{executable} is not an executable file")
        return build

    @property
    def bin(self) -> Path:
        return self.path / ".venv" / "bin"

    @property
    def python(self) -> Path:
        return self.bin / "python"

    @property
    def recall(self) -> Path:
        return self.bin / "recall"


def child_environment(
    build: Build, run: RunDir, environ: Mapping[str, str] = os.environ
) -> dict[str, str]:
    """An allow-listed environment: nothing ambient reaches a recall child.

    Every `RECALL_*` path is inside the run directory. The service-manager
    variables are absent, so `systemctl --user` cannot reach the account's
    manager. HOME stays real because the build's resolvers require one.
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
