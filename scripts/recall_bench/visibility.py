"""Per-append witnesses and search/show observations; committed offsets are a separate signal."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from recall.core.types import Source
from recall.parsers import get_parser

from .cli import Command
from .evidence import JobFailed
from .load import Appended, Call, Freshness
from .workload import Appender

WORKERS = 8
SEARCH_LIMIT = 10_000
QUERY_WORDS = 12


@dataclass(frozen=True)
class Witness:
    session: str
    message: str
    content: str
    query: str


def prepare(appender: Appender, root: Path, check: Callable[[], None]) -> dict[int, Witness]:
    """Normalize each scheduled prefix before boot, then restore every seed byte.

    The product's adapters own normalization and identity. Replaying the prefixes
    before boot keeps that work outside the timed workload and preserves its bytes.
    A metadata-only write has no searchable witness and cannot qualify freshness.
    """
    parsers = [get_parser(Source(session.source)) for session in appender.sessions]
    previous = [
        {
            message.id: message.content
            for message in parser.parse(root / session.relative).session.messages
        }
        for session, parser in zip(appender.sessions, parsers, strict=True)
    ]
    witnesses: dict[int, Witness] = {}
    try:
        for write in appender.writes:
            check()
            session = appender.sessions[write.session]
            body = b"".join(session.lines[write.first_line : write.first_line + write.line_count])
            if os.write(session.descriptor, body) != len(body):
                raise OSError(f"short witness write to {session.relative}")
            parsed = parsers[write.session].parse(root / session.relative)
            messages = parsed.session.messages
            changed = [
                message
                for message in messages
                if message.content
                and previous[write.session].get(message.id) != message.content
                and re.search(r"\w", message.content)
            ]
            if not changed:
                raise JobFailed(
                    "freshness_witness",
                    "an append introduces no searchable message",
                    append=write.index,
                    path=str(session.relative),
                )
            message = changed[-1]
            assert message.content is not None
            words = re.findall(r"\w+", message.content)[:QUERY_WORDS]
            query = '"' + " ".join(words) + '"'
            witnesses[write.index] = Witness(parsed.session.id, message.id, message.content, query)
            previous[write.session] = {message.id: message.content for message in messages}
    finally:
        for session in appender.sessions:
            os.ftruncate(session.descriptor, session.seed_size)
    return witnesses


class Visibility(Freshness):
    """Freshness ends only after both reads return the append's normalized witness."""

    def __init__(
        self,
        appended: list[Appended],
        witnesses: Mapping[int, Witness],
        record: Callable[[Mapping[str, object]], None],
    ) -> None:
        super().__init__(appended)
        self.witnesses = witnesses
        self.record = record
        self.offsets = Freshness(appended)
        self.active: dict[str, Appended] = {}
        self.searched: dict[int, float] = {}
        self.shown: dict[int, float] = {}

    def callers(self) -> dict[str, Callable[[], Command | None]]:
        return {
            f"visibility-{worker}": self.next_for(f"visibility-{worker}")
            for worker in range(WORKERS)
        }

    def next_for(self, worker: str) -> Callable[[], Command]:
        def next_command() -> Command:
            if worker not in self.active:
                owned = {item.write.index for item in self.active.values()}
                item = next((item for item in self.unseen() if item.write.index not in owned), None)
                if item is None:
                    return Command.live(256)
                self.active[worker] = item
            item = self.active[worker]
            witness = self.witnesses[item.write.index]
            if item.write.index not in self.searched:
                return Command.search(witness.query, SEARCH_LIMIT)
            return Command.show(witness.session)

        return next_command

    def observe(self, call: Call) -> None:
        self.offsets.observe(call)
        self.identifiers.update(self.offsets.identifiers)
        item = self.active.get(call.caller)
        if item is None or not call.succeeded or call.document is None:
            return
        index = item.write.index
        witness = self.witnesses[index]
        ended = call.started + call.seconds
        if call.started < item.at:
            return
        if call.argv[0] == "search":
            rows = call.document.get("results")
            if isinstance(rows, list) and any(
                isinstance(row, dict)
                and row.get("session_id") == witness.session
                and row.get("message_id") == witness.message
                and row.get("content") == witness.content
                and row.get("lexical_match") is True
                for row in rows
            ):
                self.searched[index] = ended
        elif call.argv[0] == "show":
            rows = call.document.get("messages")
            freshness = call.document.get("freshness")
            indexed = freshness.get("indexed_size") if isinstance(freshness, dict) else None
            if (
                isinstance(indexed, int)
                and indexed >= item.end_size
                and isinstance(rows, list)
                and any(
                    isinstance(row, dict)
                    and row.get("id") == witness.message
                    and row.get("content") == witness.content
                    for row in rows
                )
            ):
                self.shown[index] = ended
                self.seen[index] = max(self.searched[index], ended) - item.at
                self.record(
                    {
                        "index": index,
                        "path": item.path,
                        "session_id": witness.session,
                        "message_id": witness.message,
                        "appended_at": item.at,
                        "search_at": self.searched[index],
                        "show_at": ended,
                        "seconds": self.seen[index],
                    }
                )
                del self.active[call.caller]
