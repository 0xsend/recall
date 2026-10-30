"""The identity values a systemd user service starts or runs with.

Environment files and process environments routinely hold credentials. Each
read copies the source into one bounded buffer that is zeroed before the reader
returns. Only identity values leave it, and errors name only a path, a line, an
identity key and a fixed cause: never another key, value or count.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path

ENVIRONMENT_FILE_BYTES_MAX = 1024 * 1024
# Above Linux's default 2 MiB ARG_MAX, which bounds an exec-time environment.
PROCESS_ENVIRONMENT_BYTES_MAX = 4 * 1024 * 1024
# systemd's own limit depends on ARG_MAX; identity values are paths.
IDENTITY_VALUE_BYTES_MAX = 4096
PROC_ROOT = Path("/proc")


class IdentityKey(StrEnum):
    """The variables that decide which data dir and files a daemon uses."""

    HOME = "HOME"
    RECALL_DATA_DIR = "RECALL_DATA_DIR"
    RECALL_CONFIG_PATH = "RECALL_CONFIG_PATH"
    RECALL_DB_PATH = "RECALL_DB_PATH"
    RECALL_LOCK_PATH = "RECALL_LOCK_PATH"


_IDENTITY_BY_NAME = {key.value.encode(): key for key in IdentityKey}
# Key bytes held while parsing: enough to tell any identity key from a longer name.
_KEY_BYTES_HELD = max(len(name) for name in _IDENTITY_BY_NAME) + 1

Assignment = tuple[IdentityKey, str]


@dataclass(frozen=True)
class EnvironmentFile:
    """One `EnvironmentFiles` entry as the manager reports it."""

    path: Path
    ignore_errors: bool


_ENTRY = re.compile(r"(/[^\n]*) \(ignore_errors=(yes|no)\)")


def parse_environment_file_entry(line: str) -> EnvironmentFile:
    """Read one `EnvironmentFiles=` value printed by `systemctl show`."""
    match = _ENTRY.fullmatch(line)
    if match is None:
        raise ValueError("unreadable systemd EnvironmentFiles entry")
    path = match[1]
    # systemd expands each entry with glob(3); recall reads only literal paths.
    if any(character in path for character in "*?[\\"):
        raise ValueError(f"EnvironmentFile {path} uses a glob pattern recall does not expand")
    return EnvironmentFile(Path(path), match[2] == "yes")


# Well-formed UTF-8 (RFC 3629): no overlongs, surrogates or code points past U+10FFFF.
_UTF8 = re.compile(
    rb"(?:[\x00-\x7f]"
    rb"|[\xc2-\xdf][\x80-\xbf]"
    rb"|\xe0[\xa0-\xbf][\x80-\xbf]"
    rb"|[\xe1-\xec\xee\xef][\x80-\xbf]{2}"
    rb"|\xed[\x80-\x9f][\x80-\xbf]"
    rb"|\xf0[\x90-\xbf][\x80-\xbf]{2}"
    rb"|[\xf1-\xf3][\x80-\xbf]{3}"
    rb"|\xf4[\x80-\x8f][\x80-\xbf]{2})*+"
)


def read_environment_file(entry: EnvironmentFile) -> tuple[Assignment, ...]:
    """The identity assignments systemd would load from `entry`, in file order.

    A missing optional file contributes nothing, as systemd skips it. Every
    other doubt refuses: a missing required file, anything that is not a
    readable regular file, NUL bytes, invalid UTF-8 (systemd drops the whole
    file), and identity values outside the plain single-line forms.
    """
    path = entry.path
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        if entry.ignore_errors:
            return ()
        raise ValueError(
            f"EnvironmentFile {path} is missing and required; the service cannot start"
        ) from None
    except OSError as err:
        raise ValueError(_unreadable(path, err)) from None
    buffer = bytearray(ENVIRONMENT_FILE_BYTES_MAX + 1)
    try:
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"EnvironmentFile {path} is not a regular file")
            size = _read_into(descriptor, buffer)
        except OSError as err:
            raise ValueError(_unreadable(path, err)) from None
        finally:
            os.close(descriptor)
        if size > ENVIRONMENT_FILE_BYTES_MAX:
            raise ValueError(f"EnvironmentFile {path} exceeds 1 MiB")
        if buffer.find(0, 0, size) != -1:
            raise ValueError(f"EnvironmentFile {path} contains a NUL byte")
        if buffer.startswith(b"\xef\xbb\xbf"):
            raise ValueError(f"EnvironmentFile {path} starts with a byte-order mark")
        if _UTF8.fullmatch(buffer, 0, size) is None:
            raise ValueError(f"EnvironmentFile {path} is not valid UTF-8")
        return _EnvironmentFileParser(buffer, size, path).assignments()
    finally:
        _wipe(buffer)


def _unreadable(path: Path, err: OSError) -> str:
    cause = os.strerror(err.errno) if err.errno is not None else type(err).__name__
    return f"EnvironmentFile {path} is unreadable by recall ({cause})"


def _read_into(descriptor: int, buffer: bytearray) -> int:
    """Fill `buffer` from `descriptor` until EOF or full; return the byte count."""
    size = 0
    with memoryview(buffer) as view:
        while size < len(buffer):
            count = os.readv(descriptor, [view[size:]])
            if count == 0:
                break
            size += count
    return size


def _wipe(buffer: bytearray) -> None:
    buffer[:] = bytes(len(buffer))


class _State(IntEnum):
    """The states of systemd v255 `parse_env_file_internal` (src/basic/env-file.c)."""

    PRE_KEY = 0
    KEY = 1
    PRE_VALUE = 2
    VALUE = 3
    VALUE_ESCAPE = 4
    SINGLE_QUOTE_VALUE = 5
    DOUBLE_QUOTE_VALUE = 6
    DOUBLE_QUOTE_VALUE_ESCAPE = 7
    COMMENT = 8
    COMMENT_ESCAPE = 9


_WHITESPACE = frozenset(b" \t\n\r")
_NEWLINE = frozenset(b"\n\r")
_COMMENTS = frozenset(b"#;")
_BACKSLASH = ord("\\")
_SINGLE_QUOTE = ord("'")
_DOUBLE_QUOTE = ord('"')
_EQUALS = ord("=")
_LINE_FEED = ord("\n")
_QUOTED = (_State.SINGLE_QUOTE_VALUE, _State.DOUBLE_QUOTE_VALUE, _State.DOUBLE_QUOTE_VALUE_ESCAPE)


class _EnvironmentFileParser:
    """systemd's byte state machine, holding only identity keys' values.

    Every byte advances the state, so assignment boundaries match systemd even
    inside multi-line quoted values. Key bytes are held only up to the longest
    identity key plus one; value bytes only for an identity key.
    """

    def __init__(self, buffer: bytearray, size: int, path: Path) -> None:
        self._buffer = buffer
        self._size = size
        self._path = path
        self._assignments: list[Assignment] = []
        self._line = 1
        self._key = bytearray()
        self._key_length = 0
        self._key_whitespace_at: int | None = None
        self._key_line = 1
        self._identity: IdentityKey | None = None
        self._value = bytearray()
        self._value_whitespace_at: int | None = None
        self._value_too_long = False
        self._unsupported = False
        self._quoted_segments = 0
        self._unquoted_text = False

    def assignments(self) -> tuple[Assignment, ...]:
        try:
            state = _State.PRE_KEY
            for index in range(self._size):
                byte = self._buffer[index]
                state = self._advance(state, byte)
                if byte == _LINE_FEED:
                    self._line += 1
            if state in _QUOTED:
                self._unsupported = True
            if state == _State.VALUE:
                self._strip_value()
            if state in (_State.PRE_VALUE, _State.VALUE, _State.VALUE_ESCAPE, *_QUOTED):
                self._push()
            return tuple(self._assignments)
        finally:
            _wipe(self._key)
            _wipe(self._value)

    def _advance(self, state: _State, byte: int) -> _State:
        match state:
            case _State.PRE_KEY:
                if byte in _COMMENTS:
                    return _State.COMMENT
                if byte not in _WHITESPACE:
                    self._start_key(byte)
                    return _State.KEY
                return state
            case _State.KEY:
                if byte in _NEWLINE:
                    return _State.PRE_KEY
                if byte == _EQUALS:
                    self._start_value()
                    return _State.PRE_VALUE
                self._add_key(byte)
                return state
            case _State.PRE_VALUE:
                if byte in _NEWLINE:
                    self._push()
                    return _State.PRE_KEY
                if byte in (_SINGLE_QUOTE, _DOUBLE_QUOTE):
                    # Only one quoted segment, alone, is a plain value.
                    if self._quoted_segments or self._unquoted_text:
                        self._unsupported = True
                    return (
                        _State.SINGLE_QUOTE_VALUE
                        if byte == _SINGLE_QUOTE
                        else _State.DOUBLE_QUOTE_VALUE
                    )
                if byte == _BACKSLASH:
                    self._unsupported = True
                    return _State.VALUE_ESCAPE
                if byte not in _WHITESPACE:
                    if self._quoted_segments:
                        self._unsupported = True
                    self._unquoted_text = True
                    self._value_whitespace_at = None
                    self._add_value(byte)
                    return _State.VALUE
                return state
            case _State.VALUE:
                if byte in _NEWLINE:
                    self._strip_value()
                    self._push()
                    return _State.PRE_KEY
                if byte == _BACKSLASH:
                    self._unsupported = True
                    self._value_whitespace_at = None
                    return _State.VALUE_ESCAPE
                if byte in (_SINGLE_QUOTE, _DOUBLE_QUOTE):
                    # Literal here, but a quote mid-value reads differently in a shell.
                    self._unsupported = True
                if byte not in _WHITESPACE:
                    self._value_whitespace_at = None
                elif self._value_whitespace_at is None:
                    self._value_whitespace_at = len(self._value)
                self._add_value(byte)
                return state
            case _State.VALUE_ESCAPE:
                if byte not in _NEWLINE:
                    self._add_value(byte)
                return _State.VALUE
            case _State.SINGLE_QUOTE_VALUE | _State.DOUBLE_QUOTE_VALUE:
                closing = _SINGLE_QUOTE if state == _State.SINGLE_QUOTE_VALUE else _DOUBLE_QUOTE
                if byte == closing:
                    self._quoted_segments += 1
                    return _State.PRE_VALUE
                if state == _State.DOUBLE_QUOTE_VALUE and byte == _BACKSLASH:
                    self._unsupported = True
                    return _State.DOUBLE_QUOTE_VALUE_ESCAPE
                if byte in _NEWLINE or byte == _BACKSLASH:
                    self._unsupported = True
                self._add_value(byte)
                return state
            case _State.DOUBLE_QUOTE_VALUE_ESCAPE:
                return _State.DOUBLE_QUOTE_VALUE
            case _State.COMMENT:
                if byte == _BACKSLASH:
                    return _State.COMMENT_ESCAPE
                if byte in _NEWLINE:
                    return _State.PRE_KEY
                return state
            case _State.COMMENT_ESCAPE:
                if byte in _NEWLINE:
                    # Since v254 the next line is an assignment; before, part of the comment.
                    raise ValueError(
                        f"EnvironmentFile {self._path} comment line {self._line} ends with a "
                        "backslash, which systemd versions read differently"
                    )
                return _State.COMMENT

    def _start_key(self, byte: int) -> None:
        _wipe(self._key)
        self._key.clear()
        self._key_length = 0
        self._key_whitespace_at = None
        self._key_line = self._line
        self._add_key(byte)

    def _add_key(self, byte: int) -> None:
        # systemd strips a key's trailing whitespace before using it.
        if byte not in _WHITESPACE:
            self._key_whitespace_at = None
        elif self._key_whitespace_at is None:
            self._key_whitespace_at = self._key_length
        if len(self._key) < _KEY_BYTES_HELD:
            self._key.append(byte)
        self._key_length += 1

    def _start_value(self) -> None:
        length = (
            self._key_whitespace_at if self._key_whitespace_at is not None else self._key_length
        )
        self._identity = (
            _IDENTITY_BY_NAME.get(bytes(self._key[:length])) if length < _KEY_BYTES_HELD else None
        )
        _wipe(self._key)
        self._key.clear()
        self._value.clear()
        self._value_whitespace_at = None
        self._value_too_long = False
        self._unsupported = False
        self._quoted_segments = 0
        self._unquoted_text = False

    def _add_value(self, byte: int) -> None:
        if self._identity is None:
            return
        if len(self._value) < IDENTITY_VALUE_BYTES_MAX:
            self._value.append(byte)
        else:
            self._value_too_long = True

    def _strip_value(self) -> None:
        if self._value_whitespace_at is not None:
            del self._value[self._value_whitespace_at :]

    def _push(self) -> None:
        key = self._identity
        self._identity = None
        if key is None:
            return
        where = f"{key} in EnvironmentFile {self._path} line {self._key_line}"
        if self._unsupported:
            raise ValueError(f"{where} uses quoting or escapes recall does not interpret")
        if self._value_too_long:
            raise ValueError(f"{where} is longer than {IDENTITY_VALUE_BYTES_MAX} bytes")
        value = _identity_value(bytes(self._value), where)
        _wipe(self._value)
        self._value.clear()
        self._assignments.append((key, value))


def _identity_value(raw: bytes, where: str) -> str:
    if not raw:
        raise ValueError(f"{where} is empty")
    # systemd drops values with control characters; recall refuses instead of guessing.
    if any(byte < 0x20 or byte == 0x7F for byte in raw):
        raise ValueError(f"{where} contains a control character")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{where} is not valid UTF-8") from None


@dataclass(frozen=True)
class ProcessIdentity:
    """A running process's exec-time identity values and where to read its cwd."""

    pid: int
    environment: dict[str, str]

    def working_directory(self) -> Path:
        try:
            target = os.readlink(PROC_ROOT / str(self.pid) / "cwd")
        except OSError as err:
            raise ValueError(_process_unreadable(self.pid, err)) from None
        if target.endswith(" (deleted)") or not Path(target).is_absolute():
            raise ValueError(
                f"recall-daemon.service process {self.pid} has no usable working directory"
            )
        return Path(target)


def read_process_identity(pid: int) -> ProcessIdentity:
    """The identity values in `pid`'s exec-time environment block.

    The kernel block is exactly what `execve` received, so an identity name
    that appears twice or without `=` means it was not systemd's; every such
    doubt refuses rather than falling back to a default.
    """
    try:
        descriptor = os.open(PROC_ROOT / str(pid) / "environ", os.O_RDONLY | os.O_CLOEXEC)
    except OSError as err:
        raise ValueError(_process_unreadable(pid, err)) from None
    buffer = bytearray(PROCESS_ENVIRONMENT_BYTES_MAX + 1)
    try:
        try:
            size = _read_into(descriptor, buffer)
        except OSError as err:
            raise ValueError(_process_unreadable(pid, err)) from None
        finally:
            os.close(descriptor)
        return ProcessIdentity(pid, _process_identity_values(buffer, size, pid))
    finally:
        _wipe(buffer)


def _process_unreadable(pid: int, err: OSError) -> str:
    cause = os.strerror(err.errno) if err.errno is not None else type(err).__name__
    return (
        f"recall-daemon.service's running process {pid} changed or cannot be read during "
        f"identification ({cause}); retry"
    )


def _process_identity_values(buffer: bytearray, size: int, pid: int) -> dict[str, str]:
    where = f"recall-daemon.service process {pid} environment"
    if size == 0:
        raise ValueError(f"{where} is empty (the process exited or is not a service process)")
    if size > PROCESS_ENVIRONMENT_BYTES_MAX:
        raise ValueError(f"{where} exceeds 4 MiB")
    if buffer[size - 1] != 0:
        raise ValueError(f"{where} is not NUL-terminated")
    values: dict[str, str] = {}
    start = 0
    while start < size:
        end = buffer.find(0, start, size)
        assert end != -1
        for name, key in _IDENTITY_BY_NAME.items():
            if end - start == len(name) and buffer.startswith(name, start):
                raise ValueError(f"{where} has {key} without a value")
            if buffer.startswith(name + b"=", start, end):
                if key.value in values:
                    raise ValueError(f"{where} sets {key} twice")
                begin = start + len(name) + 1
                if end - begin > IDENTITY_VALUE_BYTES_MAX:
                    raise ValueError(
                        f"{where} {key} is longer than {IDENTITY_VALUE_BYTES_MAX} bytes"
                    )
                try:
                    values[key.value] = bytes(buffer[begin:end]).decode("utf-8")
                except UnicodeDecodeError:
                    raise ValueError(f"{where} {key} is not valid UTF-8") from None
                break
        start = end + 1
    return values
