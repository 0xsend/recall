"""Per-append witnesses and search/show observations; committed offsets are a separate signal."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from recall.core.models import Session
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
class Searchable:
    session: str
    message: str | None
    tool: str | None
    field: str
    text: str

    @property
    def key(self) -> tuple[str | None, str | None, str]:
        return self.message, self.tool, self.field

    def matches(self, row: Mapping[str, object]) -> bool:
        return (
            row.get("session_id") == self.session
            and row.get("message_id") == self.message
            and row.get("tool_call_id") == self.tool
            and row.get(self.field) == self.text
        )


def searchable(session: Session) -> list[Searchable]:
    rows = []
    for message in session.messages:
        for field in ("content", "thinking"):
            text = getattr(message, field)
            if text and re.search(r"\w", text):
                rows.append(Searchable(session.id, message.id, None, field, text))
    tools = [tool for message in session.messages for tool in message.tool_calls]
    for tool in [*tools, *session.orphan_tool_calls]:
        if tool.bash_command and re.search(r"\w", tool.bash_command):
            rows.append(
                Searchable(session.id, tool.message_id, tool.id, "bash_command", tool.bash_command)
            )
    return rows


@dataclass(frozen=True)
class Witness:
    search: Searchable
    shown_message: str
    shown_content: str | None
    shown_thinking: str | None
    shown_tools: tuple[str, ...]
    searchable_changed: bool

    @property
    def query(self) -> str:
        return '"' + " ".join(re.findall(r"\w+", self.search.text)[:QUERY_WORDS]) + '"'

    def shown(self, row: Mapping[str, object]) -> bool:
        tools = row.get("tool_calls")
        identifiers = (
            {tool.get("id") for tool in tools if isinstance(tool, dict)}
            if isinstance(tools, list)
            else set()
        )
        return (
            row.get("id") == self.shown_message
            and row.get("content") == self.shown_content
            and row.get("thinking") == self.shown_thinking
            and set(self.shown_tools) <= identifiers
        )


def prepare(appender: Appender, root: Path, check: Callable[[], None]) -> dict[int, Witness]:
    """Normalize each scheduled prefix before boot, then restore every seed byte.

    The product's adapters own normalization and identity. Replaying the prefixes
    before boot keeps that work outside the timed workload and preserves its bytes.
    If a write changes no searchable value (metadata or non-searchable tools),
    search must still return existing membership and show must expose the latest
    normalized record at the new byte boundary. Evidence labels that no-op;
    it never substitutes an offset for either read.
    """
    parsers = [get_parser(Source(session.source)) for session in appender.sessions]
    previous = [
        {row.key: row.text for row in searchable(parser.parse(root / session.relative).session)}
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
            session_result = parsed.session
            rows = searchable(session_result)
            changed = [row for row in rows if previous[write.session].get(row.key) != row.text]
            if not rows or not session_result.messages:
                raise JobFailed(
                    "freshness_witness",
                    "the appended session has no searchable or show-visible record",
                    append=write.index,
                    path=str(session.relative),
                )
            search = (changed or rows)[-1]
            shown = session_result.messages[-1]
            witnesses[write.index] = Witness(
                search,
                shown.id,
                shown.content,
                shown.thinking,
                tuple(tool.id for tool in shown.tool_calls),
                bool(changed),
            )
            previous[write.session] = {row.key: row.text for row in rows}
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
            return Command.show_visible(witness.search.session)

        return next_command

    def observe(self, call: Call) -> None:
        self.offsets.observe(call)
        self.identifiers.update(self.offsets.identifiers)
        item = self.active.get(call.caller)
        if item is None:
            return
        if not call.succeeded:
            raise JobFailed(
                "visibility_call",
                "a search or show visibility call failed",
                append=item.write.index,
                call=call.to_json(),
                stdout=call.error,
            )
        if call.document is None:
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
                and witness.search.matches(row)
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
                and any(isinstance(row, dict) and witness.shown(row) for row in rows)
            ):
                self.shown[index] = ended
                self.seen[index] = max(self.searched[index], ended) - item.at
                self.record(
                    {
                        "index": index,
                        "path": item.path,
                        "session_id": witness.search.session,
                        "search_message_id": witness.search.message,
                        "search_tool_id": witness.search.tool,
                        "search_field": witness.search.field,
                        "searchable_changed": witness.searchable_changed,
                        "show_message_id": witness.shown_message,
                        "appended_at": item.at,
                        "search_at": self.searched[index],
                        "show_at": ended,
                        "seconds": self.seen[index],
                    }
                )
                del self.active[call.caller]
