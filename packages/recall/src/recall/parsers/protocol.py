from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from recall.core.models import ParseResult
from recall.core.source import SessionSource, matches_session_file
from recall.core.types import Source

logger = logging.getLogger(__name__)


class SessionParser(Protocol):
    source: Source

    # Configured discovery roots (REQ-LIVE-012). None means `default_roots()`;
    # an empty tuple means this source is configured to scan nothing.
    roots: tuple[Path, ...] | None

    def default_roots(self) -> list[Path]: ...

    def discover(self) -> list[Path]: ...

    def sidecar_paths(self, path: Path) -> list[Path]:
        """Sibling files whose content this parser folds into session metadata.

        The indexer stats these alongside the session file, so editing one
        invalidates the cached row (REQ-INDEX-018). A parser that enriches from
        a sidecar but omits it here goes stale silently: the session file never
        changes again, so the row is skipped forever and keeps whatever
        metadata the parser produced the first time.

        Paths need not exist; missing entries contribute nothing.
        """
        ...

    def parse(
        self,
        path: Path,
        *,
        offset: int = 0,
        message_idx_base: int = 0,
        orphan_tool_call_idx_base: int = 0,
        resume_state: Mapping[str, Any] | None = None,
    ) -> ParseResult:
        """Normalize the captured prefix, optionally resuming from `offset`.

        `resume_state` is the JSON-decoded `adapter_state` of the checkpoint
        this parser issued for the boundary at `offset` (REQ-INDEX-026). A
        shape this build cannot honor raises `UnsupportedResumeState` rather
        than normalizing the suffix against state it had to guess at.
        """
        ...

    def watch_roots(self) -> list[Path]: ...

    @property
    def file_pattern(self) -> str: ...

    def live_candidates(self, *, now: datetime, idle_threshold: float) -> list[Path]: ...


def default_watch_roots(parser: SessionParser) -> list[Path]:
    """Configured roots when set, else the parser's built-in ones; existing only.

    A configured root that does not exist is skipped rather than raised on: an
    operator lists the lane directories a host *may* have, and a host that has
    not run that harness yet is not a misconfiguration. An explicitly empty
    list is not "unset": it turns the source off, so a scratch or
    single-harness configuration does not silently sweep the whole host.

    Every surviving root is **resolved**, because these paths are what the
    watcher hands the filesystem observer, and the kernel reports events under
    the real path. Watching a spelling it never uses is silent: the session is
    discovered, promoted into the live set, and then simply never indexed again
    while `daemon status` keeps reporting it live. It is the same canonical
    spelling the parsers write into `sessions.source_path` and `live_path_key`
    reads back, so the observer, the live set, the event channel and the index
    all hold one name for one file.
    """
    roots = parser.default_roots() if parser.roots is None else list(parser.roots)
    return [root.resolve() for root in roots if root.exists()]


def default_discover(parser: SessionParser) -> list[Path]:
    """Every file matching the parser's pattern under any of its watch roots.

    Resolved and deduplicated on that spelling: nested roots are an easy thing
    to configure, and indexing one transcript as two sessions is not a small
    bug. `watch_roots` is already resolved, so this only catches a symlinked
    file *inside* a real root.
    """
    return [source.canonical_path for source in default_discover_sources(parser)]


def default_discover_sources(
    parser: SessionParser, resolver: Callable[[SessionSource], SessionSource] | None = None
) -> list[SessionSource]:
    seen: dict[str, SessionSource] = {}
    for root in parser.watch_roots():
        for path in root.rglob(parser.file_pattern):
            if not matches_session_file(parser.source.value, path.name, parser.file_pattern):
                continue
            try:
                descriptor = replace(
                    SessionSource.resolve(parser.source.value, path), root_path=root
                )
                if resolver is not None:
                    descriptor = resolver(descriptor)
            except OSError as err:
                logger.debug("stat failed for %s: %s", path, err)
                continue
            key = str(descriptor.canonical_path)
            previous = seen.get(key)
            if previous is None or (previous.encoding == "zstd" and descriptor.encoding == "jsonl"):
                seen[key] = descriptor
    return [seen[key] for key in sorted(seen)]


def default_live_candidates(
    parser: SessionParser, *, now: datetime, idle_threshold: float
) -> list[Path]:
    """Return parser-owned files whose mtime is inside the live window.

    REQ-DAEMON-042 requires the parser layer to derive the active session set
    from filesystem mtimes alone. REQ-DAEMON-054 forbids lsof, fd scans, and
    subprocess-based liveness checks, so this helper only scans files and stats
    them directly.

    Shares `default_discover`'s walk so the live set and the index can never
    disagree about which files belong to a source.
    """

    cutoff_ts = (now - timedelta(seconds=idle_threshold)).timestamp()
    live_paths: list[Path] = []
    for descriptor in default_discover_sources(parser):
        try:
            if descriptor.physical_path.stat().st_mtime >= cutoff_ts:
                live_paths.append(descriptor.canonical_path)
        except OSError as err:
            logger.debug("stat failed for %s: %s", descriptor.physical_path, err)
            continue

    return live_paths


def live_sources(
    parser: SessionParser,
    *,
    now: datetime,
    idle_threshold: float,
    resolver: Callable[[SessionSource], SessionSource] | None = None,
) -> list[SessionSource]:
    from recall.parsers.codex import CodexParser

    if not isinstance(parser, CodexParser):
        return [
            SessionSource.for_session(parser.source.value, path)
            for path in parser.live_candidates(now=now, idle_threshold=idle_threshold)
        ]
    cutoff = (now - timedelta(seconds=idle_threshold)).timestamp()
    candidates = []
    for descriptor in default_discover_sources(parser, resolver):
        try:
            if descriptor.physical_path.stat().st_mtime >= cutoff:
                candidates.append(descriptor)
        except OSError as err:
            logger.debug("stat failed for %s: %s", descriptor.physical_path, err)
    return candidates


def parse_source(parser: SessionParser, source: SessionSource, **kwargs: Any) -> ParseResult:
    from recall.parsers.codex import CodexParser

    if isinstance(parser, CodexParser):
        return parser.parse(source, **kwargs)
    return parser.parse(source.canonical_path, **kwargs)
