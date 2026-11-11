"""Typed host fragments in human diagnostics and terminal output."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict


def format_host(host: str) -> str:
    """Encode scalar values and isolated surrogates as one recoverable ASCII value."""
    return host if re.fullmatch(r"[A-Za-z0-9._-]+", host) else json.dumps(host, ensure_ascii=True)


class HostText(BaseModel):
    """A host fragment whose string conversion always uses the terminal encoding."""

    model_config = ConfigDict(frozen=True)
    value: str

    def __str__(self) -> str:
        return format_host(self.value)


class HostMessage(BaseModel):
    """Keep host fragments distinct until a structured or terminal view is selected."""

    model_config = ConfigDict(frozen=True)
    parts: tuple[str | HostText, ...]

    def raw(self) -> str:
        return "".join(part.value if isinstance(part, HostText) else part for part in self.parts)

    def text(self) -> str:
        return "".join(str(part) for part in self.parts)


class HostMessageError(ValueError):
    """A refusal with raw compatibility text and separately typed host fragments."""

    def __init__(self, message: HostMessage) -> None:
        self.host_message = message
        super().__init__(message.raw())
