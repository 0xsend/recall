from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from recall.core.types import Source
from recall.parsers.claude_code import ClaudeCodeParser
from recall.parsers.codex import CodexParser
from recall.parsers.grok import GrokParser
from recall.parsers.kimi_code import KimiCodeParser
from recall.parsers.pi_agent import PiAgentParser
from recall.parsers.protocol import SessionParser

if TYPE_CHECKING:
    from recall.core.config import SourceConfig

# Concrete types, not `type[SessionParser]`: only the concrete dataclasses
# carry the `roots` constructor argument `_build` passes.
ParserType = (
    type[ClaudeCodeParser]
    | type[CodexParser]
    | type[PiAgentParser]
    | type[GrokParser]
    | type[KimiCodeParser]
)

_PARSER_TYPES: tuple[ParserType, ...] = (
    ClaudeCodeParser,
    CodexParser,
    PiAgentParser,
    GrokParser,
    KimiCodeParser,
)


def all_parsers(sources: Mapping[str, SourceConfig] | None = None) -> list[SessionParser]:
    """Build every parser, carrying the discovery roots configured for it.

    `sources` is `AppConfig.sources`. A source with no entry keeps its built-in
    location, so the default is byte-identical to having no configuration at
    all (REQ-LIVE-012).
    """
    return [_build(parser_type, sources) for parser_type in _PARSER_TYPES]


def get_parser(source: Source, sources: Mapping[str, SourceConfig] | None = None) -> SessionParser:
    for parser_type in _PARSER_TYPES:
        if parser_type.source == source:
            return _build(parser_type, sources)
    raise ValueError(f"no parser registered for source {source}")


def _build(parser_type: ParserType, sources: Mapping[str, SourceConfig] | None) -> SessionParser:
    configured = (sources or {}).get(parser_type.source.value)
    if configured is None or configured.roots is None:
        return parser_type()
    return parser_type(roots=configured.roots)
