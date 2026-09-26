from __future__ import annotations

import socket
from enum import StrEnum


class Source(StrEnum):
    CLAUDE_CODE = "claude_code"
    CODEX = "codex"
    PI_AGENT = "pi_agent"
    GROK = "grok"
    KIMI_CODE = "kimi_code"


ABSOLUTE_TOKEN_SOURCES = frozenset({Source.CODEX})


# The host label a surface emits for a session it cannot attribute to a named
# machine (REQ-HOST-API-004): the schema default, the pre-migration backfill, and
# the fallback when the hostname is unobtainable all land here. It is a sentinel
# meaning "unattributed, in this database's own frame of reference" -- never a
# machine identity. Anything merging rows across machines MUST treat it as unset
# and substitute a label that is meaningful to the reader (REQ-FLEET-MERGE-002).
UNATTRIBUTED_HOST = "local"


def default_session_host() -> str:
    """Short hostname for session_state.host; the unattributed sentinel if unobtainable.

    Lives in core (not parsers) because it is the producer of UNATTRIBUTED_HOST and
    has no parser semantics: the schema layer needs it too, and `db/` may not import
    `parsers/`.
    """
    try:
        name = socket.gethostname().strip()
        if not name:
            return UNATTRIBUTED_HOST
        return name.split(".", 1)[0] or UNATTRIBUTED_HOST
    except OSError:
        return UNATTRIBUTED_HOST


def is_attributed_host(value: object) -> bool:
    """True when *value* names a real machine rather than the unattributed sentinel.

    Matches the sentinel exactly (after stripping) so real hostnames that merely
    start with or contain "local" -- `localhost`, `local-dev` -- stay attributed.
    """
    return isinstance(value, str) and value.strip() not in ("", UNATTRIBUTED_HOST)


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SearchMode(StrEnum):
    AUTO = "auto"
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


class SchedulerKind(StrEnum):
    AUTO = "auto"
    LAUNCHD = "launchd"
    SYSTEMD = "systemd"
    CRON = "cron"


class EmbedKind(StrEnum):
    ALL = "all"
    CONTENT = "content"
    THINKING = "thinking"
    BASH = "bash"


class DaemonMode(StrEnum):
    AUTO = "auto"
    WATCH = "watch"
    POLL = "poll"


class RunKind(StrEnum):
    INDEX = "index"
    DAEMON_ONCE = "daemon-once"
    DAEMON_SCHEDULED = "daemon-scheduled"
    DAEMON_WATCH = "daemon-watch"


def parse_search_mode(value: str) -> SearchMode:
    normalized = value.strip().lower()
    match normalized:
        case "auto":
            return SearchMode.AUTO
        case "keyword" | "kw" | "fts":
            return SearchMode.KEYWORD
        case "vector" | "vec" | "semantic":
            return SearchMode.VECTOR
        case "hybrid" | "rrf":
            return SearchMode.HYBRID
        case _:
            raise ValueError(f"unsupported search mode: {value}")


def parse_source(value: str) -> Source:
    normalized = value.strip().lower()
    match normalized:
        case "claude-code" | "claude_code":
            return Source.CLAUDE_CODE
        case "codex":
            return Source.CODEX
        case "pi" | "pi-agent" | "pi_agent":
            return Source.PI_AGENT
        case "grok" | "grok-build" | "grok_build" | "grokbuild":
            return Source.GROK
        case "kimi" | "kimi-code" | "kimi_code":
            return Source.KIMI_CODE
        case _:
            raise ValueError(f"unsupported source: {value}")


def parse_scheduler_kind(value: str) -> SchedulerKind:
    normalized = value.strip().lower()
    match normalized:
        case "auto":
            return SchedulerKind.AUTO
        case "launchd":
            return SchedulerKind.LAUNCHD
        case "systemd":
            return SchedulerKind.SYSTEMD
        case "cron":
            return SchedulerKind.CRON
        case _:
            raise ValueError(f"unsupported scheduler: {value}")


def parse_daemon_mode(value: str) -> DaemonMode:
    normalized = value.strip().lower()
    match normalized:
        case "auto":
            return DaemonMode.AUTO
        case "watch":
            return DaemonMode.WATCH
        case "poll":
            return DaemonMode.POLL
        case _:
            raise ValueError(f"unsupported daemon mode: {value}")


def parse_embed_kind(value: str) -> EmbedKind:
    normalized = value.strip().lower()
    match normalized:
        case "all":
            return EmbedKind.ALL
        case "content":
            return EmbedKind.CONTENT
        case "thinking":
            return EmbedKind.THINKING
        case "bash" | "tool" | "tool-calls":
            return EmbedKind.BASH
        case _:
            raise ValueError(f"unsupported embed kind: {value}")
