"""Shared non-blankness, FTS derivation, and FTS5 query escaping across Python and DuckDB."""

from __future__ import annotations

import re

# Matches every non-whitespace Unicode codepoint identically to Python `str.strip()`.
# Verified across all 29 Unicode whitespace characters in Python 3.12 (0x000000..0x10FFFF).
SQL_NON_BLANK_PATTERN = r"[^\x09-\x0d\x1c-\x1f\x20\x85\p{Z}]"


def has_non_whitespace(text: str | None) -> bool:
    """True iff *text* contains at least one non-whitespace character."""
    return bool(text and text.strip())


def derive_fts_text(context_text: str, body: str | None) -> str:
    """Prepend *context_text* only when *body* carries non-whitespace text."""
    if not has_non_whitespace(body):
        return ""
    assert body is not None
    return f"{context_text}{body}"


def sql_is_non_blank(col_expr: str) -> str:
    """DuckDB SQL boolean predicate matching :func:`has_non_whitespace`."""
    return f"regexp_matches(COALESCE({col_expr}, ''), '{SQL_NON_BLANK_PATTERN}')"


def sql_derive_fts(ctx_expr: str, body_expr: str) -> str:
    """DuckDB SQL expression matching :func:`derive_fts_text`."""
    return (
        f"CASE WHEN {sql_is_non_blank(body_expr)} "
        f"THEN COALESCE({ctx_expr}, '') || {body_expr} ELSE '' END"
    )


def escape_fts5_query(query: str) -> str:
    """Convert raw user search text into a safe SQLite FTS5 MATCH expression.

    Preserves quoted phrases (``"foo bar"``) and prefix wildcards (``reca*``)
    while reducing punctuation like ``-``, ``:``, ``(``, ``)`` to ordinary terms
    so FTS5 syntax errors never leak on identifiers like ``REQ-BRIDGE``.
    """
    terms: list[str] = []
    position = 0
    for match in re.finditer(r'"([^"]*)"', query):
        terms.extend(
            _quote_fts5_term(term) for term in _bare_fts5_terms(query[position : match.start()])
        )
        phrase = match.group(1).strip()
        if phrase:
            terms.append(_quote_fts5_term(phrase))
        position = match.end()
    terms.extend(_quote_fts5_term(term) for term in _bare_fts5_terms(query[position:]))
    return " ".join(terms)


def _bare_fts5_terms(query: str) -> list[str]:
    return re.findall(r"\w+\*?", query, flags=re.UNICODE)


def _quote_fts5_term(term: str) -> str:
    prefix = term.endswith("*") and len(term) > 1
    term_value = term[:-1] if prefix else term
    escaped = term_value.replace('"', '""')
    suffix = "*" if prefix else ""
    return f'"{escaped}"{suffix}'
