"""The normalized conversation dependency projection for REQ-INDEX-021."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import NamedTuple

from recall.core.models import Message, Session, ToolCall


class ToolContextInput(NamedTuple):
    idx: int
    name: str
    input_json: str | None
    bash_command: str | None

    @classmethod
    def project(
        cls, idx: int, name: str, tool_input: object, bash_command: str | None
    ) -> ToolContextInput:
        parsed = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
        return cls(
            idx,
            name,
            json.dumps(parsed, sort_keys=True) if tool_input is not None else None,
            bash_command,
        )

    @classmethod
    def from_call(cls, call: ToolCall) -> ToolContextInput:
        return cls.project(call.idx, call.tool_name, call.tool_input, call.bash_command)


class MessageContextInput(NamedTuple):
    idx: int
    role: str
    content: str | None
    thinking: str | None
    agent_id: str | None
    tools: tuple[ToolContextInput, ...]

    @classmethod
    def from_message(cls, message: Message) -> MessageContextInput:
        return cls(
            message.idx,
            message.role,
            message.content,
            message.thinking,
            message.agent_id,
            tuple(ToolContextInput.from_call(call) for call in message.tool_calls),
        )


class ContextInputs(NamedTuple):
    source: str
    messages: tuple[MessageContextInput, ...]
    orphans: tuple[ToolContextInput, ...]

    @classmethod
    def project(
        cls, source: str, messages: Iterable[Message], orphans: Iterable[ToolCall] = ()
    ) -> ContextInputs:
        return cls(
            source,
            tuple(MessageContextInput.from_message(message) for message in messages),
            tuple(ToolContextInput.from_call(call) for call in orphans),
        )

    @classmethod
    def from_session(cls, session: Session) -> ContextInputs:
        return cls.project(session.source, session.messages, session.orphan_tool_calls)
