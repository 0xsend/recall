"""Tests for shared FTS/embedding non-blank derivation and DuckDB whitespace parity."""

from __future__ import annotations

import sys

import duckdb
import pytest
from recall.core.fts_text import (
    derive_fts_text,
    escape_fts5_query,
    has_non_whitespace,
    sql_derive_fts,
    sql_is_non_blank,
)


def test_derive_fts_text_returns_empty_for_blank_body_even_with_context() -> None:
    assert derive_fts_text("[0xbigboss/recall main] ", None) == ""
    assert derive_fts_text("[0xbigboss/recall main] ", "") == ""
    assert derive_fts_text("[0xbigboss/recall main] ", "   \t\r\n\u2003\u00a0  ") == ""


def test_derive_fts_text_prepends_context_when_body_non_blank() -> None:
    assert derive_fts_text("[repo main] ", "hello") == "[repo main] hello"
    assert derive_fts_text("", "hello") == "hello"


def test_duckdb_sql_non_blank_matches_python_whitespace_for_every_unicode_whitespace_char() -> None:
    whitespace_chars = [chr(cp) for cp in range(sys.maxunicode + 1) if chr(cp).isspace()]
    assert len(whitespace_chars) >= 25
    samples: list[tuple[str | None, bool]] = [
        ("", False),
        (None, False),
        ("a", True),
        (" \t\n\r\x0b\x0c", False),
        ("\u00a0\u1680\u2000\u2003\u2028\u2029\u202f\u205f\u3000", False),
        ("\u2003recall\u2003", True),
        *[(ch, False) for ch in whitespace_chars],
        *[(f"{ch}x", True) for ch in whitespace_chars[:10]],
    ]
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("CREATE TABLE t (idx INTEGER, body TEXT, ctx TEXT)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?)",
            [(i, body, "[ctx] ") for i, (body, _expected) in enumerate(samples)],
        )
        rows = conn.execute(
            f"SELECT idx, {sql_is_non_blank('body')}, {sql_derive_fts('ctx', 'body')} "
            "FROM t ORDER BY idx"
        ).fetchall()
    finally:
        conn.close()

    for (idx, sql_non_blank, sql_derived), (body, expected) in zip(rows, samples, strict=True):
        assert has_non_whitespace(body) is expected, f"Python mismatch at {idx}: {body!r}"
        assert bool(sql_non_blank) is expected, f"DuckDB mismatch at {idx}: {body!r}"
        assert sql_derived == derive_fts_text("[ctx] ", body), f"derive mismatch at {idx}"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("recall", '"recall"'),
        ("reca*", '"reca"*'),
        ("studies*", '"studies"*'),
        ("foo-bar baz", '"foo" "bar" "baz"'),
        ('"exact phrase" tail*', '"exact phrase" "tail"*'),
        ("   ***  ", ""),
    ],
)
def test_escape_fts5_query_preserves_phrases_and_prefix_wildcards(raw: str, expected: str) -> None:
    assert escape_fts5_query(raw) == expected
