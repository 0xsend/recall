"""Typed host fragments in human diagnostics and terminal output."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, InstanceOf


def format_host(host: str) -> str:
    """Encode scalar values and isolated surrogates as one recoverable ASCII value."""
    return host if re.fullmatch(r"[A-Za-z0-9._-]+", host) else json.dumps(host, ensure_ascii=True)


class HostText(BaseModel):
    """A host fragment whose string conversion always uses the terminal encoding."""

    model_config = ConfigDict(frozen=True)
    value: str
    quoted: bool = False

    def raw(self) -> str:
        return repr(self.value) if self.quoted else self.value

    def __str__(self) -> str:
        encoded = format_host(self.value)
        return repr(self.value) if self.quoted and encoded == self.value else encoded


class HostValue(str):
    """Inventory identity with safe text conversion and explicit raw access.

    The underlying string stays raw for JSON encoders and string equality.
    String operations can discard the type; retain it until a data boundary.
    """

    def __new__(cls, value: str):
        return super().__new__(cls, str.__str__(value))

    def raw(self) -> str:
        return str.__str__(self)

    def __str__(self) -> str:
        return format_host(self.raw())

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)

    def quoted(self) -> HostText:
        """Preserve historical quotes around plain names in inventory errors."""
        return HostText(value=self.raw(), quoted=True)


class HostMessage(BaseModel):
    """Keep host fragments distinct until a structured or terminal view is selected."""

    model_config = ConfigDict(frozen=True)
    # Retain string subtypes so composed refusals keep their typed text view.
    parts: tuple[InstanceOf[str] | HostText, ...]

    def raw(self) -> str:
        return "".join(part.raw() if isinstance(part, HostText) else part for part in self.parts)

    def text(self) -> str:
        return "".join(str(part) for part in self.parts)


class HostMessageError(ValueError):
    """A refusal with raw compatibility text and separately typed host fragments."""

    def __init__(self, message: HostMessage) -> None:
        self.host_message = message
        super().__init__(message.raw())


class HostDiagnostic(str):
    """A raw structured error string whose terminal conversion retains host typing.

    JSON encodes the underlying string without calling __str__. Ordinary text
    conversion uses the encoded view, including the existing excerpt policy.
    """

    _message: HostMessage
    _excerpt: Callable[[str], str]

    def __new__(cls, message: HostMessage, excerpt: Callable[[str], str]):
        value = super().__new__(cls, excerpt(message.raw()))
        value._message = message
        value._excerpt = excerpt
        return value

    def __str__(self) -> str:
        return self._excerpt(self._message.text())

    def raw(self) -> str:
        return str.__str__(self)

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)

    def __getnewargs_ex__(
        self,
    ) -> tuple[tuple[HostMessage, Callable[[str], str]], dict[str, object]]:
        return (self._message, self._excerpt), {}
