"""Generated transcripts and deterministic post-boot arrivals; never reads an agent home."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from .cli import Command
from .evidence import JobFailed
from .fs import Directory
from .isolation import POOL_DIR
from .load import Call, Next, seconds_summary
from .workload import APPEND_FLAGS

FILES_MAX = 100_000
LINES_MAX = 10_000


@dataclass(frozen=True)
class Plan:
    burst: int = 150
    rounds: int = 2
    delay: float = 3.0
    interval: float = 30.0
    backlog: int = 1000
    age: float = 86400.0
    lines: int = 4
    seed: int = 101

    def validate(self) -> None:
        if self.burst < 150 or self.rounds < 1 or self.backlog < 1:
            raise JobFailed(
                "usage", "arrival count must be >=150; rounds and backlog must be positive"
            )
        if self.rounds * (self.burst + 2) + self.backlog > FILES_MAX:
            raise JobFailed("usage", f"generated file count exceeds {FILES_MAX}")
        if not 1 <= self.lines <= LINES_MAX:
            raise JobFailed("usage", f"generated lines must be within 1..{LINES_MAX}")
        if not all(
            math.isfinite(value) and value > 0 for value in (self.delay, self.interval, self.age)
        ):
            raise JobFailed("usage", "arrival clocks and backlog age must be finite and positive")
        if self.interval <= self.delay:
            raise JobFailed("usage", "arrival interval must exceed probe delay")


@dataclass(frozen=True)
class Emission:
    index: int
    due: float
    source: Literal["claude_code", "codex"]
    kind: Literal["burst", "probe", "history"]
    round: int
    identifier: str
    token: str

    @property
    def relative(self) -> PurePosixPath:
        name = f"{self.identifier}.jsonl"
        if self.source == "codex":
            name = f"rollout-{self.identifier}.jsonl"
        project = f"bench-{self.kind}-{self.round}-{self.index % 10}"
        return PurePosixPath(POOL_DIR, self.source, project, name)

    def body(self, lines: int, timestamp: str) -> bytes:
        if self.source == "claude_code":
            records = [
                {
                    "type": "user",
                    "uuid": str(uuid.uuid5(uuid.UUID(self.identifier), str(index))),
                    "sessionId": self.identifier,
                    "timestamp": timestamp,
                    "message": {"role": "user", "content": self.token},
                }
                for index in range(lines)
            ]
        else:
            records = [
                {
                    "type": "session_meta",
                    "timestamp": timestamp,
                    "payload": {"id": self.identifier, "timestamp": timestamp, "cwd": "/bench"},
                },
                *(
                    {
                        "type": "response_item",
                        "timestamp": timestamp,
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": self.token}],
                        },
                    }
                    for _ in range(lines)
                ),
            ]
        return ("\n".join(json.dumps(row) for row in records) + "\n").encode()


def schedule(plan: Plan, *, history: bool = False) -> tuple[Emission, ...]:
    plan.validate()
    items: list[Emission] = []

    def add(
        due: float,
        source: Literal["claude_code", "codex"],
        kind: Literal["burst", "probe", "history"],
        round: int,
    ) -> None:
        index = len(items)
        identifier = str(uuid.uuid5(uuid.NAMESPACE_OID, f"recall-bench/{plan.seed}/{kind}/{index}"))
        token = "recallbench" + hashlib.sha256(identifier.encode()).hexdigest()[:24]
        if kind != "probe":
            batch = index // max(1, 10_000 // plan.lines)
            token = f"recallbatch{plan.seed}r{round}b{batch} " + token
        items.append(Emission(index, due, source, kind, round, identifier, token))

    if history:
        for index in range(plan.backlog):
            add(0, "claude_code" if index % 2 == 0 else "codex", "history", 0)
    else:
        for round in range(plan.rounds):
            for _ in range(plan.burst):
                add(round * plan.interval, "claude_code", "burst", round)
            for source in ("claude_code", "codex"):
                add(round * plan.interval + plan.delay, source, "probe", round)
    return tuple(items)


@dataclass(frozen=True)
class Written:
    item: Emission
    at: float
    bytes: int
    sha256: str


class Writer:
    """Owns a schedule and records every successful write before exposing its witness."""

    def __init__(
        self,
        root: Directory,
        items: tuple[Emission, ...],
        plan: Plan,
        record: Callable[[Mapping[str, object]], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.root, self.items, self.plan, self.record = root, items, plan, record
        self.clock = clock
        self._probe_due = 0.0
        self.written: list[Written] = []

    @property
    def done(self) -> bool:
        return len(self.written) == len(self.items)

    def due(self, elapsed: float) -> None:
        began = self.clock()
        for item in self.items[len(self.written) :]:
            if item.due > elapsed or (item.kind == "probe" and elapsed < self._probe_due):
                break
            self.write(item)
            if item.kind == "burst":
                self._probe_due = elapsed + self.clock() - began + self.plan.delay

    def write(self, item: Emission) -> None:
        if len(self.written) != item.index:
            raise ValueError("emissions must be written in order")
        timestamp = "2026-01-01T00:00:00Z"
        body = item.body(self.plan.lines, timestamp)
        parent = item.relative.parent
        with self.root.descend("--run-dir", parent.parent) as source:
            if source.lstat(parent.name) is None:
                os.mkdir(parent.name, 0o755, dir_fd=source.descriptor)
        at = self.clock()
        with self.root.descend("--run-dir", parent) as directory:
            descriptor = os.open(
                item.relative.name, APPEND_FLAGS, 0o644, dir_fd=directory.descriptor
            )
            try:
                if os.write(descriptor, body) != len(body):
                    raise OSError(f"short emission to {item.relative}")
                if item.kind == "history":
                    old = time.time() - self.plan.age
                    os.utime(descriptor, (old, old))
            finally:
                os.close(descriptor)
        written = Written(item, at, len(body), hashlib.sha256(body).hexdigest())
        self.record(
            {
                "event": "write",
                "index": item.index,
                "path": str(item.relative),
                "kind": item.kind,
                "source": item.source,
                "round": item.round,
                "at": at,
                "bytes": len(body),
                "sha256": written.sha256,
                "token": item.token,
            }
        )
        self.written.append(written)


class Hits:
    """First exact keyword hit after each write; probes have their own callers."""

    def __init__(self, writer: Writer, record: Callable[[Mapping[str, object]], None]) -> None:
        self.writer, self.record = writer, record
        self.seen: dict[int, float] = {}
        self.active: dict[str, Written] = {}

    def callers(self) -> dict[str, Next]:
        def next_for(worker: str, probes: bool) -> Next:
            def next_command() -> Command:
                if worker not in self.active:
                    owned = {item.item.index for item in self.active.values()}
                    pending = next(
                        (
                            item
                            for item in self.writer.written
                            if (item.item.kind == "probe") == probes
                            and item.item.index not in self.seen
                            and item.item.index not in owned
                        ),
                        None,
                    )
                    if pending is None:
                        return Command.daemon_status()
                    self.active[worker] = pending
                item = self.active[worker].item
                return Command.search(item.token.split()[0], 10_000)

            return next_command

        return {
            name: next_for(name, probes)
            for probes, count in ((True, 2), (False, 2))
            for name in (f"{'probe' if probes else 'burst'}-{i}" for i in range(count))
        }

    def observe(self, call: Call) -> None:
        if not call.succeeded:
            raise JobFailed("arrival_call", "an arrival observer failed", call=call.to_json())
        item = self.active.get(call.caller)
        if item is None or call.argv[0] != "search" or call.document is None:
            return
        rows = call.document.get("results")
        if not isinstance(rows, list):
            raise JobFailed("arrival_reply", "search returned no results list")
        visible = {
            row.get("content")
            for row in rows
            if isinstance(row, dict) and row.get("lexical_match") is True
        }
        for written in self.writer.written:
            if (
                written.item.index not in self.seen
                and call.started >= written.at
                and written.item.token in visible
            ):
                seconds = call.started + call.seconds - written.at
                self.seen[written.item.index] = seconds
                self.record(
                    {
                        "event": "hit",
                        "index": written.item.index,
                        "kind": written.item.kind,
                        "source": written.item.source,
                        "round": written.item.round,
                        "written_at": written.at,
                        "hit_at": call.started + call.seconds,
                        "seconds": seconds,
                    }
                )
        for worker in list(self.active):
            if self.active[worker].item.index in self.seen:
                del self.active[worker]

    def summary(self, kind: str) -> dict[str, object]:
        expected = [item for item in self.writer.items if item.kind == kind]
        values = [self.seen[item.index] for item in expected if item.index in self.seen]
        return {
            "expected": len(expected),
            "missing": len(expected) - len(values),
            "seconds": seconds_summary(values),
            "samples": [
                {
                    "index": item.index,
                    "source": item.source,
                    "round": item.round,
                    "seconds": self.seen.get(item.index),
                }
                for item in expected
            ],
        }
