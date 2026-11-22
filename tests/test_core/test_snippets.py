"""Tests for Porter-aligned match-centered search snippets and sanitization."""

from __future__ import annotations

from recall.core.snippets import SNIPPET_MAX_CHARS, build_search_snippet, sanitize_snippet


def test_snippet_centers_on_late_match_in_long_message() -> None:
    prefix = "filler text " * 120
    body = f"{prefix}the exact target token xylophone appears right here in context."
    snippet = build_search_snippet(
        query="xylophone",
        kind="message",
        content=body,
        thinking=None,
        role="assistant",
    )
    assert "xylophone" in snippet.lower()
    assert len(snippet) <= SNIPPET_MAX_CHARS


def test_snippet_matches_porter_stem_prefix_and_diacritics() -> None:
    stem_snippet = build_search_snippet(
        query="studies*",
        kind="message",
        content=("noise " * 90) + "We will study this migration plan carefully.",
        thinking=None,
        role="assistant",
    )
    assert "study" in stem_snippet.lower()

    diacritic_snippet = build_search_snippet(
        query="café*",
        kind="message",
        content=("noise " * 90) + "Meet at the cafe tomorrow afternoon.",
        thinking=None,
        role="assistant",
    )
    assert "cafe" in diacritic_snippet.lower()

    phrase_snippet = build_search_snippet(
        query='"duckdb migration"',
        kind="message",
        content=("noise " * 90) + "Verified the DuckDB migration rollback script.",
        thinking=None,
        role="assistant",
    )
    assert "duckdb migration" in phrase_snippet.lower()


def test_snippet_uses_thinking_when_match_is_only_in_thinking() -> None:
    snippet = build_search_snippet(
        query="quarantine",
        kind="message",
        content="Sure, I updated the file.",
        thinking=("reasoning " * 40) + "Need to avoid quarantine side effects here.",
        role="assistant",
        fields=("content", "thinking"),
    )
    assert snippet.startswith("[thinking] ")
    assert "quarantine" in snippet.lower()
    assert len(snippet) <= SNIPPET_MAX_CHARS


def test_snippet_strips_system_timestamp_preamble() -> None:
    content = (
        "Created At: 2026-11-22T01:02:03Z\n"
        "Completed At: 2026-11-22T01:02:04Z\n"
        "recall daemon status returned healthy."
    )
    snippet = build_search_snippet(
        query="unrelated",
        kind="message",
        content=content,
        thinking=None,
        role="system",
    )
    assert not snippet.startswith("Created At:")
    assert snippet.startswith("recall daemon status")


def test_snippet_sanitizes_ansi_osc_and_control_bytes() -> None:
    dirty = (
        "\x1b[31mred\x1b[0m\x1b]8;;https://evil.test\x07link\x1b]8;;\x07"
        "before\x9d0;hidden\x9cafter\r\n\x00\x9b31mnext"
    )
    cleaned = sanitize_snippet(dirty)
    assert "\x1b" not in cleaned
    assert "\x00" not in cleaned
    assert "\x9b" not in cleaned
    assert "\x9d" not in cleaned
    assert "\x9c" not in cleaned
    assert "hidden" not in cleaned
    assert "\n" not in cleaned
    assert "\r" not in cleaned
    assert "redlinkbeforeafter" in cleaned
    assert "next" in cleaned


def test_snippet_centers_on_cross_field_conjunction_match() -> None:
    snippet = build_search_snippet(
        query="alpha beta",
        kind="message",
        content=("filler " * 80) + "target alpha appears late in content.",
        thinking="reasoning mentions beta here.",
        role="assistant",
        fields=("content", "thinking"),
    )
    assert "alpha" in snippet.lower()
    assert len(snippet) <= SNIPPET_MAX_CHARS


def test_snippet_handles_lone_surrogates_late_cross_field_and_long_osc_in_tool() -> None:
    # Lone surrogate must not raise UnicodeEncodeError in SQLite FTS5
    surrogate_snip = build_search_snippet(
        query="needle",
        kind="message",
        content="intro\ud800 target needle here",
        thinking=None,
        role="assistant",
    )
    assert "needle" in surrogate_snip

    # Late cross-field match beyond 133KB in both fields
    late = build_search_snippet(
        query="alpha beta",
        kind="message",
        content=("filler " * 20_000) + "late alpha token",
        thinking=("filler " * 20_000) + "late beta token",
        role="assistant",
        fields=("content", "thinking"),
    )
    assert "alpha" in late.lower()
    assert len(late) <= SNIPPET_MAX_CHARS

    # Long OSC title in bash_command must be stripped BEFORE truncation so trailing text survives,
    # and ESC-introduced OSC terminated by C1 ST (\x9c) must not swallow visible suffix.
    tool_snip = build_search_snippet(
        query="echo",
        kind="tool_call",
        content=None,
        thinking=None,
        role=None,
        tool_name="bash",
        bash_command="echo \x1b]0;" + ("t" * 300) + "\x07VISIBLE \x1b]0;hidden\x9cTAIL",
    )
    assert "VISIBLE" in tool_snip
    assert "TAIL" in tool_snip
    assert "hidden" not in tool_snip
    assert "\x1b" not in tool_snip
    assert "\x9c" not in tool_snip


def test_snippet_handles_documents_longer_than_chunk_window() -> None:
    body = ("word " * 30_000) + "late_needle_token_999"
    snippet = build_search_snippet(
        query="late_needle_token_999",
        kind="message",
        content=body,
        thinking=None,
        role="assistant",
    )
    assert "late_needle_token_999" in snippet
    assert len(snippet) <= SNIPPET_MAX_CHARS


def test_snippet_formats_tool_call() -> None:
    snippet = build_search_snippet(
        query="schema",
        kind="tool_call",
        content=None,
        thinking=None,
        role=None,
        tool_name="read_file",
        bash_command=None,
        tool_input={"Path": "/repo/packages/recall/src/recall/db/schema.sql"},
    )
    assert "[read_file]" in snippet
    assert "schema.sql" in snippet
