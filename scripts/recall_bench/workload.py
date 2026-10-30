"""The deterministic append workload: seeded transcript copies and a fixed write schedule.

This is the `experiments/checkpoint_bytes/measure.py` reconstruction of the
io-59 workload, taken from the template's pool instead of a lane directory:

- Every pool transcript is copied `copies` times. Each copy renames the UUIDs
  in its file name, in the name and the text, from one `Random(copy_seed)`
  stream, so the copies are distinct sessions of the same bytes length.
- Each copy starts as its first `seed_fraction` of lines; the rest is what
  the schedule appends.
- `Random(seed)` draws each write's line count from CHUNK_LINES. When
  `lines_total` is set, the counts are adjusted round-robin to that exact
  total, as `measure.py` does.
- Write `k * appenders + a` belongs to appender `a` at tick `k`; tick `k` is
  `k * interval` after the workload starts. Writes take sessions in a fixed
  cycle, skipping one without enough lines left. With one appender this is
  exactly `measure.py`'s schedule.

The same inputs give the same writes, byte for byte. Copies are written into
the run's own source roots through descriptors held from the run directory,
and the template's originals are removed from the run first, so the daemon
indexes only the copies.
"""

from __future__ import annotations

import os
import random
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .fs import Directory, Refused
from .isolation import POOL_DIR, SOURCES, open_entry, open_parent, scan

CHUNK_LINES = (5, 40)
COPIES_MAX = 64
WRITES_MAX = 100_000
POOL_BYTES_MAX = 2 * 1024**3
TRANSCRIPT_LINES_MAX = 10_000_000
UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
APPEND_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_APPEND
ARGUMENT = "--run-dir"


@dataclass(frozen=True)
class Workload:
    """Everything that decides the appended bytes; the result records it."""

    copies: int
    seed_fraction: float
    copy_seed: int
    seed: int
    appenders: int
    writes: int
    interval_seconds: float
    lines_total: int | None

    def __post_init__(self) -> None:
        assert 1 <= self.copies <= COPIES_MAX, self.copies
        assert 0 < self.seed_fraction < 1, self.seed_fraction
        assert self.appenders >= 1 and self.writes >= 1, (self.appenders, self.writes)
        assert self.appenders * self.writes <= WRITES_MAX, (self.appenders, self.writes)
        assert self.interval_seconds > 0, self.interval_seconds
        assert self.lines_total is None or self.lines_total >= self.count, self.lines_total

    @property
    def count(self) -> int:
        return self.appenders * self.writes

    def chunks(self) -> list[int]:
        """Each write's line count, in write order."""
        rng = random.Random(self.seed)
        chunks = [rng.randint(*CHUNK_LINES) for _ in range(self.count)]
        if self.lines_total is not None:
            difference = self.lines_total - sum(chunks)
            for index in range(abs(difference)):
                chunks[index % self.count] += 1 if difference > 0 else -1
            assert sum(chunks) == self.lines_total and min(chunks) > 0, "unreachable line total"
        return chunks


@dataclass
class Transcript:
    """One seeded copy: its run-relative path, every line, and how many are written."""

    source: str
    relative: PurePosixPath
    lines: tuple[bytes, ...]
    position: int
    size: int
    seed_size: int
    descriptor: int = field(default=-1, repr=False)

    @property
    def remaining(self) -> int:
        return len(self.lines) - self.position


@dataclass(frozen=True)
class Write:
    """One scheduled append: which session, which lines, and when."""

    index: int
    appender: int
    tick: int
    session: int
    first_line: int
    line_count: int
    byte_count: int


def schedule(workload: Workload, sessions: Sequence[Transcript]) -> list[Write]:
    """Assign every write a session and its lines; pure, given the seeded sessions."""
    if not sessions:
        raise Refused("--template", "pool", "the template's pool holds no transcripts")
    positions = [session.position for session in sessions]
    writes: list[Write] = []
    selected = 0
    for index, count in enumerate(workload.chunks()):
        for _ in sessions:
            candidate = selected % len(sessions)
            selected += 1
            if len(sessions[candidate].lines) - positions[candidate] >= count:
                break
        else:
            raise Refused(
                "--template", "pool", "the pool cannot supply the declared append schedule"
            )
        first = positions[candidate]
        lines = sessions[candidate].lines[first : first + count]
        positions[candidate] += count
        writes.append(
            Write(
                index=index,
                appender=index % workload.appenders,
                tick=index // workload.appenders,
                session=candidate,
                first_line=first,
                line_count=count,
                byte_count=sum(len(line) for line in lines),
            )
        )
    return writes


def _read(root: Directory, relative: PurePosixPath) -> bytes:
    with open_entry(root, relative, ARGUMENT) as file:
        size = file.status.st_size
        data = bytearray()
        while len(data) < size:
            chunk = os.pread(file.descriptor, size - len(data), len(data))
            if not chunk:
                raise Refused(ARGUMENT, file.path, "shrank while it was read")
            data += chunk
        return bytes(data)


def _originals(root: Directory) -> list[tuple[str, PurePosixPath, bytes]]:
    """Every pool transcript, in source order then path order, with its bytes."""
    originals: list[tuple[str, PurePosixPath, bytes]] = []
    total = 0
    for source in SOURCES:
        base = PurePosixPath(POOL_DIR, source)
        with root.descend(ARGUMENT, base) as directory:
            inventory = scan(ARGUMENT, directory)
        for relative in sorted(inventory.files, key=str):
            data = _read(root, base / relative)
            total += len(data)
            if total > POOL_BYTES_MAX:
                raise Refused("--template", "pool", f"holds more than {POOL_BYTES_MAX} bytes")
            originals.append((source, base / relative, data))
    return originals


def _rename(name: str, text: str, rng: random.Random) -> tuple[str, str]:
    for old in UUID_PATTERN.findall(name):
        replacement = str(uuid.UUID(int=rng.getrandbits(128), version=4))
        text = text.replace(old, replacement)
        name = name.replace(old, replacement)
    return name, text


def _copies(
    workload: Workload, originals: Sequence[tuple[str, PurePosixPath, bytes]]
) -> list[Transcript]:
    """The seeded copies, in `measure.py`'s order: copy number, then source, then path."""
    rng = random.Random(workload.copy_seed)
    sessions: list[Transcript] = []
    for copy in range(workload.copies):
        for source in SOURCES:
            for original_source, relative, raw in originals:
                if original_source != source:
                    continue
                text = raw.decode("utf-8", errors="surrogateescape")
                name, text = _rename(relative.name, text, rng)
                lines = tuple(
                    text.encode("utf-8", errors="surrogateescape").splitlines(keepends=True)
                )
                if len(lines) > TRANSCRIPT_LINES_MAX:
                    raise Refused("--template", relative, "has too many lines")
                position = max(1, int(len(lines) * workload.seed_fraction))
                seed_size = sum(len(line) for line in lines[:position])
                sessions.append(
                    Transcript(
                        source=source,
                        relative=PurePosixPath(POOL_DIR, source, f"copy-{copy}", name),
                        lines=lines,
                        position=position,
                        size=seed_size,
                        seed_size=seed_size,
                    )
                )
    names = [session.relative for session in sessions]
    if len(set(names)) != len(names):
        raise Refused("--template", "pool", "two pool transcripts share a file name")
    return sessions


class Appender:
    """Writes the schedule into the seeded copies through descriptors it holds."""

    def __init__(self, sessions: list[Transcript], writes: list[Write]) -> None:
        self.sessions = sessions
        self.writes = writes
        self.done = 0

    @property
    def bytes_total(self) -> int:
        return sum(write.byte_count for write in self.writes)

    @property
    def lines_total(self) -> int:
        return sum(write.line_count for write in self.writes)

    def due(self, tick: int) -> list[Write]:
        """The writes not yet made whose tick is at most `tick`, in order."""
        pending: list[Write] = []
        for write in self.writes[self.done :]:
            if write.tick > tick:
                break
            pending.append(write)
        return pending

    def write(self, item: Write) -> Transcript:
        """Append one scheduled write; writes must be made in schedule order."""
        assert item.index == self.done, (item.index, self.done)
        session = self.sessions[item.session]
        assert item.first_line == session.position, (item, session.position)
        body = b"".join(session.lines[session.position : session.position + item.line_count])
        assert len(body) == item.byte_count, item
        written = os.write(session.descriptor, body)
        if written != len(body):
            raise OSError(f"short append to {session.relative}: {written} of {len(body)}")
        session.position += item.line_count
        session.size += len(body)
        self.done += 1
        return session

    def close(self) -> None:
        for session in self.sessions:
            if session.descriptor >= 0:
                os.close(session.descriptor)
                session.descriptor = -1


def seed(root: Directory, workload: Workload) -> Appender:
    """Replace the run's pool with the seeded copies, held open for appending.

    `root` must be the held run directory, before the daemon starts. Nothing
    else writes the pool then, so removing the originals by name changes only
    the run's own state. The caller closes the returned appender.
    """
    originals = _originals(root)
    sessions = _copies(workload, originals)
    writes = schedule(workload, sessions)
    for _source, relative, _data in originals:
        with open_parent(root, relative, ARGUMENT) as parent:
            os.unlink(relative.name, dir_fd=parent.descriptor)
    appender = Appender(sessions, writes)
    try:
        for session in sessions:
            copy_dir = session.relative.parent
            with root.descend(ARGUMENT, copy_dir.parent) as source_dir:
                if source_dir.lstat(copy_dir.name) is None:
                    os.mkdir(copy_dir.name, 0o755, dir_fd=source_dir.descriptor)
            with root.descend(ARGUMENT, copy_dir) as directory:
                session.descriptor = os.open(
                    session.relative.name, APPEND_FLAGS, 0o644, dir_fd=directory.descriptor
                )
            body = b"".join(session.lines[: session.position])
            if os.write(session.descriptor, body) != len(body):
                raise OSError(f"short seed write to {session.relative}")
    except BaseException:
        appender.close()
        raise
    return appender
