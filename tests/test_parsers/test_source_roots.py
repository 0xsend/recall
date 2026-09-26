"""Per-source discovery roots (REQ-LIVE-012).

Claude Code's root was hard-coded to `~/.claude/projects`, so a lane launched
with `CLAUDE_CONFIG_DIR` pointing elsewhere was invisible unless the operator
symlinked `projects` into place. Roots are configuration now, and the default
has to stay byte-identical to the hard-coded behavior.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from recall.core.config import SourceConfig
from recall.core.types import Source
from recall.parsers import ClaudeCodeParser, all_parsers, get_parser


def _transcript(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text("{}\n", encoding="utf-8")
    return path


def test_the_default_root_is_the_one_the_parser_always_used(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = _transcript(tmp_path / ".claude" / "projects" / "proj", "a.jsonl")

    parser = ClaudeCodeParser()

    assert parser.watch_roots() == [tmp_path / ".claude" / "projects"]
    assert parser.discover() == [expected]


def test_configured_roots_replace_the_default_rather_than_extend_it(tmp_path) -> None:
    """An operator who moved a harness's home should stop scanning the old one."""
    lane = _transcript(tmp_path / "lanes" / "projects" / "proj", "lane.jsonl")
    _transcript(tmp_path / ".claude" / "projects" / "proj", "old.jsonl")

    parser = ClaudeCodeParser(roots=(tmp_path / "lanes" / "projects",))

    assert parser.discover() == [lane]


def test_every_configured_root_is_discovered(tmp_path) -> None:
    first = _transcript(tmp_path / "one" / "proj", "first.jsonl")
    second = _transcript(tmp_path / "two" / "proj", "second.jsonl")

    parser = ClaudeCodeParser(roots=(tmp_path / "one", tmp_path / "two"))

    assert parser.discover() == sorted([first, second])


def test_a_root_that_does_not_exist_is_skipped_not_an_error(tmp_path) -> None:
    present = _transcript(tmp_path / "here" / "proj", "here.jsonl")

    parser = ClaudeCodeParser(roots=(tmp_path / "gone", tmp_path / "here"))

    assert parser.watch_roots() == [tmp_path / "here"]
    assert parser.discover() == [present]


def test_one_file_reachable_through_two_roots_is_discovered_once(tmp_path) -> None:
    """Nested roots are an easy config mistake; a double-indexed session is not."""
    transcript = _transcript(tmp_path / "outer" / "inner" / "proj", "dup.jsonl")

    parser = ClaudeCodeParser(roots=(tmp_path / "outer", tmp_path / "outer" / "inner"))

    assert parser.discover() == [transcript]


def test_live_candidates_span_every_root(tmp_path) -> None:
    now = datetime(2026, 9, 8, 12, 0, 0)
    fresh = _transcript(tmp_path / "one" / "proj", "fresh.jsonl")
    stale = _transcript(tmp_path / "two" / "proj", "stale.jsonl")
    os.utime(fresh, (now.timestamp() - 10, now.timestamp() - 10))
    os.utime(stale, (now.timestamp() - 9999, now.timestamp() - 9999))

    parser = ClaudeCodeParser(roots=(tmp_path / "one", tmp_path / "two"))

    assert parser.live_candidates(now=now, idle_threshold=300.0) == [fresh]
    assert stale.exists()


def test_the_registry_builds_parsers_with_their_configured_roots(tmp_path) -> None:
    sources = {"claude_code": SourceConfig(roots=(tmp_path / "lanes",))}

    configured = get_parser(Source.CLAUDE_CODE, sources=sources)
    every = all_parsers(sources=sources)

    assert configured.roots == (tmp_path / "lanes",)
    assert [parser.roots for parser in every if parser.source is Source.CODEX] == [None]


def test_the_registry_without_config_matches_the_bare_parsers() -> None:
    assert [parser.roots for parser in all_parsers()] == [None] * 5


def test_live_window_boundary_is_inclusive(tmp_path) -> None:
    now = datetime(2026, 9, 8, 12, 0, 0)
    edge = _transcript(tmp_path / "one" / "proj", "edge.jsonl")
    cutoff = (now - timedelta(seconds=300.0)).timestamp()
    os.utime(edge, (cutoff, cutoff))

    parser = ClaudeCodeParser(roots=(tmp_path / "one",))

    assert parser.live_candidates(now=now, idle_threshold=300.0) == [edge]


def test_an_explicitly_empty_root_list_turns_a_source_off(tmp_path, monkeypatch) -> None:
    """`roots = []` means "scan nothing", not "fall back to ~/.claude/projects".

    Without this, a scratch or single-harness configuration silently sweeps the
    operator's whole corpus for every source it did not name.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    _transcript(tmp_path / ".claude" / "projects" / "proj", "a.jsonl")

    parser = ClaudeCodeParser(roots=())

    assert parser.watch_roots() == []
    assert parser.discover() == []
    assert parser.live_candidates(now=datetime(2026, 9, 8, 12, 0, 0), idle_threshold=300.0) == []


def test_an_absent_section_still_means_the_built_in_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = _transcript(tmp_path / ".claude" / "projects" / "proj", "a.jsonl")

    assert ClaudeCodeParser().discover() == [expected]
    assert get_parser(Source.CLAUDE_CODE, sources={}).discover() == [expected]


def test_the_registry_carries_an_empty_root_list_through(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _transcript(tmp_path / ".claude" / "projects" / "proj", "a.jsonl")
    sources = {"claude_code": SourceConfig(roots=())}

    assert get_parser(Source.CLAUDE_CODE, sources=sources).discover() == []


def test_a_root_reached_through_a_symlink_is_reported_resolved(tmp_path) -> None:
    """The watcher schedules these paths; the filesystem reports events under the real one.

    Found by dogfooding on macOS, where `/tmp` is a symlink to `/private/tmp`: a
    lane configured under `/tmp` was discovered and promoted into the live set,
    and then never received a single watch event, because the observer was
    watching a spelling the kernel never uses. The session sat unindexed
    indefinitely while `daemon status` reported it live.
    """
    real = tmp_path / "real"
    _transcript(real / "proj", "a.jsonl")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    parser = ClaudeCodeParser(roots=(link,))

    assert parser.watch_roots() == [real]


def test_discovery_through_a_symlink_yields_the_path_the_index_stores(tmp_path) -> None:
    """`sessions.source_path` is `str(path.expanduser().resolve())`.

    Discovery has to agree with it, or the live set, the event channel and the
    index each hold a different name for one file.
    """
    real = tmp_path / "real"
    transcript = _transcript(real / "proj", "a.jsonl")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    parser = ClaudeCodeParser(roots=(link,))

    assert parser.discover() == [transcript.resolve()]


def test_live_candidates_through_a_symlink_are_resolved(tmp_path) -> None:
    """The live set is keyed by these; an unresolved key never matches an event."""
    real = tmp_path / "real"
    transcript = _transcript(real / "proj", "a.jsonl")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    parser = ClaudeCodeParser(roots=(link,))
    candidates = parser.live_candidates(now=datetime.now(), idle_threshold=3600.0)

    assert candidates == [transcript.resolve()]


def test_a_home_relative_default_root_is_also_resolved(tmp_path, monkeypatch) -> None:
    """A symlinked $HOME is the same defect on a real host, not just under /tmp."""
    real_home = tmp_path / "real_home"
    _transcript(real_home / ".claude" / "projects" / "proj", "a.jsonl")
    linked_home = tmp_path / "linked_home"
    linked_home.symlink_to(real_home, target_is_directory=True)
    monkeypatch.setenv("HOME", str(linked_home))

    roots = ClaudeCodeParser().watch_roots()

    assert roots == [real_home / ".claude" / "projects"]
