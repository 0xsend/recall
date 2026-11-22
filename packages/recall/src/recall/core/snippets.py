"""Bounded, Porter-aligned match-centered snippet extraction for search results."""

from __future__ import annotations

import contextlib
import re
import sqlite3
import threading
from collections.abc import Mapping
from typing import Any

from recall.core.fts_text import escape_fts5_query, has_non_whitespace
from recall.core.tool_format import format_tool_call_line

SNIPPET_MAX_CHARS = 240
_SNIPPET_TOKEN_WINDOW = 28
_MAX_FTS_DOC_CHARS = 2 * 1024 * 1024
_MATCH_OPEN = "\x01"
_MATCH_CLOSE = "\x02"

# Strip ESC-prefixed CSI/OSC/2-byte escapes AND 8-bit C1 CSI (\x9b) / OSC (\x9d) sequences,
# honoring BEL (\x07), C1 ST (\x9c), and 7-bit ST (\x1b\\) for both OSC introducers.
_TERMINAL_ESCAPE_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x9c\x1b]*(?:\x07|\x9c|\x1b\\)?|[@-Z\\-_])"
    r"|\x9b[0-?]*[ -/]*[@-~]"
    r"|\x9d[^\x07\x9c\x1b]*(?:\x9c|\x07|\x1b\\)?"
)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_CONTROL_CHAR_KEEP_MARKERS_RE = re.compile(r"[\x00\x03-\x08\x0b-\x1f\x7f-\x9f]")
_SYSTEM_TIMESTAMP_LINE_RE = re.compile(
    r"^(?:Created At|Completed At):\s*\d{4}-\d{2}-\d{2}T[^\n]*\n?",
    re.MULTILINE,
)

_THREAD_LOCAL = threading.local()


def _safe_utf8(text: str) -> str:
    """Replace lone UTF-16 surrogates so SQLite bindings and terminal output never fail."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _strip_terminal_controls(text: str) -> str:
    cleaned = _TERMINAL_ESCAPE_RE.sub("", _safe_utf8(text))
    cleaned = cleaned.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    cleaned = _CONTROL_CHAR_RE.sub("", cleaned)
    return " ".join(cleaned.split())


def sanitize_snippet(text: str, *, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """Strip terminal escape/control sequences, collapse whitespace, and bound length."""
    assert max_chars >= 4, f"max_chars must be >= 4, got {max_chars}"
    cleaned = _strip_terminal_controls(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _sanitize_tool_input(value: Any, *, depth: int = 0) -> Any:
    if isinstance(value, str):
        return _strip_terminal_controls(value)
    if depth >= 2:
        return None
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= 64:
                break
            sanitized[str(k)] = _sanitize_tool_input(v, depth=depth + 1)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_tool_input(v, depth=depth + 1) for v in value[:32]]
    return value


def build_search_snippet(
    *,
    query: str,
    kind: str,
    content: str | None,
    thinking: str | None,
    role: str | None,
    tool_name: str | None = None,
    bash_command: str | None = None,
    tool_input: Mapping[str, Any] | None = None,
    fields: tuple[str, ...] = ("content", "thinking"),
    max_chars: int = SNIPPET_MAX_CHARS,
) -> str:
    """Build a bounded preview centered on the best lexical match across searched fields.

    Uses an in-memory two-column SQLite FTS5 `porter unicode61` table with the
    exact same query escaping and column scoping as `db/fts_sidecar.py`, so
    phrases, Porter stems, diacritics, `*` prefix operators, and cross-field
    conjunctions match identically to the keyword index.
    """
    assert max_chars >= 4, f"max_chars must be >= 4, got {max_chars}"
    if kind == "tool_call":
        # Strip terminal sequences BEFORE format_tool_call_line truncates so long
        # OSC titles cannot consume the preview budget and hide trailing text.
        safe_tool_name = _strip_terminal_controls(tool_name or "tool") or "tool"
        safe_bash = _strip_terminal_controls(bash_command) if bash_command is not None else None
        safe_input = _sanitize_tool_input(tool_input) if tool_input is not None else None
        line = format_tool_call_line(
            {
                "tool_name": safe_tool_name,
                "bash_command": safe_bash,
                "tool_input": safe_input,
            },
            max_chars=max_chars,
        )
        return sanitize_snippet(line, max_chars=max_chars)

    safe_query = _safe_utf8(query)
    safe_content = _safe_utf8(content) if content is not None else None
    safe_thinking = _safe_utf8(thinking) if thinking is not None else None

    allowed = tuple(field for field in ("content", "thinking") if field in fields) or (
        "content",
        "thinking",
    )
    content_body = (
        _strip_system_preamble(safe_content, role)
        if ("content" in allowed and has_non_whitespace(safe_content) and safe_content is not None)
        else ""
    )
    thinking_body = (
        safe_thinking
        if (
            "thinking" in allowed
            and has_non_whitespace(safe_thinking)
            and safe_thinking is not None
        )
        else ""
    )

    if not content_body and not thinking_body:
        if has_non_whitespace(safe_content) and safe_content is not None:
            content_body = _strip_system_preamble(safe_content, role)
            allowed = ("content",)
        elif has_non_whitespace(safe_thinking) and safe_thinking is not None:
            thinking_body = safe_thinking
            allowed = ("thinking",)
        else:
            return ""

    match_query = escape_fts5_query(safe_query)
    matched: tuple[str, str] | None = None
    if match_query:
        matched = _fts5_match_snippet(
            content_body=content_body,
            thinking_body=thinking_body,
            match_query=match_query,
            allowed=allowed,
        )

    if matched is not None:
        chosen_field, raw_snippet = matched
    elif content_body:
        chosen_field, raw_snippet = "content", content_body
    else:
        chosen_field, raw_snippet = "thinking", thinking_body

    prefix = "[thinking] " if chosen_field == "thinking" else ""
    body_budget = max(4, max_chars - len(prefix))
    rendered = _center_and_sanitize_snippet(raw_snippet, max_chars=body_budget)
    return f"{prefix}{rendered}" if rendered else ""


def _center_and_sanitize_snippet(text: str, *, max_chars: int) -> str:
    """Clean control/ANSI/C1 sequences and center long excerpts on the FTS5 match span."""
    assert max_chars >= 4, f"max_chars must be >= 4, got {max_chars}"
    cleaned = _TERMINAL_ESCAPE_RE.sub("", _safe_utf8(text))
    cleaned = cleaned.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    cleaned = _CONTROL_CHAR_KEEP_MARKERS_RE.sub("", cleaned)
    cleaned = " ".join(cleaned.split())
    open_idx = cleaned.find(_MATCH_OPEN)
    close_idx = cleaned.find(_MATCH_CLOSE, open_idx + 1) if open_idx >= 0 else -1
    plain = cleaned.replace(_MATCH_OPEN, "").replace(_MATCH_CLOSE, "")
    if len(plain) <= max_chars:
        return plain
    if open_idx < 0:
        return plain[: max_chars - 3].rstrip() + "..."
    match_start = len(cleaned[:open_idx].replace(_MATCH_OPEN, "").replace(_MATCH_CLOSE, ""))
    if close_idx > open_idx:
        match_end = len(cleaned[:close_idx].replace(_MATCH_OPEN, "").replace(_MATCH_CLOSE, ""))
    else:
        match_end = min(len(plain), match_start + 16)
    match_center = (match_start + match_end) // 2
    inner_budget = max(4, max_chars - 6)
    start = max(0, match_center - inner_budget // 2)
    end = min(len(plain), start + inner_budget)
    if end == len(plain):
        start = max(0, end - inner_budget)
    lead = "..." if start > 0 else ""
    tail = "..." if end < len(plain) else ""
    window = plain[start:end].strip()
    candidate = f"{lead}{window}{tail}"
    if len(candidate) <= max_chars:
        return candidate
    return candidate[: max_chars - 3].rstrip() + "..."


def _strip_system_preamble(text: str, role: str | None) -> str:
    """Drop leading Created At / Completed At timestamp headers on system tool outputs."""
    if role != "system":
        return text
    stripped = text.lstrip()
    while True:
        match = _SYSTEM_TIMESTAMP_LINE_RE.match(stripped)
        if match is None:
            break
        stripped = stripped[match.end() :].lstrip("\r\n")
    return stripped if has_non_whitespace(stripped) else text


def _snippet_conn() -> sqlite3.Connection | None:
    cached: sqlite3.Connection | bool | None = getattr(_THREAD_LOCAL, "fts5_conn", None)
    if cached is False:
        return None
    if isinstance(cached, sqlite3.Connection):
        return cached
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE VIRTUAL TABLE _snippet_fts USING fts5("
            "fts_content, fts_thinking, tokenize='porter unicode61')"
        )
    except sqlite3.Error:
        _THREAD_LOCAL.fts5_conn = False
        return None
    _THREAD_LOCAL.fts5_conn = conn
    return conn


def _scope_match_query(match_query: str, allowed: tuple[str, ...]) -> str:
    if allowed == ("content",):
        return f"{{fts_content}}: ({match_query})"
    if allowed == ("thinking",):
        return f"{{fts_thinking}}: ({match_query})"
    return match_query


def _iter_chunks(text: str, *, overlap: int = SNIPPET_MAX_CHARS) -> list[tuple[int, str]]:
    if not text:
        return [(0, "")]
    step = _MAX_FTS_DOC_CHARS
    bounded_overlap = max(SNIPPET_MAX_CHARS, min(overlap, step // 4))
    chunks: list[tuple[int, str]] = []
    offset = 0
    total = len(text)
    while offset < total:
        chunks.append((offset, text[offset : offset + step]))
        if offset + step >= total:
            break
        offset += step - bounded_overlap
    return chunks


def _run_fts5_window(
    conn: sqlite3.Connection,
    c_chunk: str,
    t_chunk: str,
    scoped: str,
) -> tuple[float, str, str] | None:
    try:
        conn.execute("DELETE FROM _snippet_fts")
        conn.execute(
            "INSERT INTO _snippet_fts(rowid, fts_content, fts_thinking) VALUES (1, ?, ?)",
            [c_chunk, t_chunk],
        )
        row = conn.execute(
            "SELECT -bm25(_snippet_fts), "
            f"snippet(_snippet_fts, 0, ?, ?, '...', {_SNIPPET_TOKEN_WINDOW}), "
            f"snippet(_snippet_fts, 1, ?, ?, '...', {_SNIPPET_TOKEN_WINDOW}) "
            "FROM _snippet_fts WHERE _snippet_fts MATCH ?",
            [_MATCH_OPEN, _MATCH_CLOSE, _MATCH_OPEN, _MATCH_CLOSE, scoped],
        ).fetchone()
    except (sqlite3.Error, UnicodeError):
        return None
    finally:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("DELETE FROM _snippet_fts")
    if row is None:
        return None
    return (
        float(row[0]),
        row[1] if isinstance(row[1], str) else "",
        row[2] if isinstance(row[2], str) else "",
    )


def _split_fts5_clauses(match_query: str) -> list[str]:
    return [m.group(0) for m in re.finditer(r'"[^"]+"\*?', match_query)]


def _fts5_match_snippet(
    *,
    content_body: str,
    thinking_body: str,
    match_query: str,
    allowed: tuple[str, ...],
) -> tuple[str, str] | None:
    """Score and extract a match-centered snippet via two-column SQLite FTS5."""
    conn = _snippet_conn()
    if conn is None:
        return _fallback_substring_snippet(content_body, thinking_body, match_query, allowed)

    scoped = _scope_match_query(match_query, allowed)
    clean_content = content_body.replace(_MATCH_OPEN, "").replace(_MATCH_CLOSE, "")
    clean_thinking = thinking_body.replace(_MATCH_OPEN, "").replace(_MATCH_CLOSE, "")

    chunk_overlap = max(SNIPPET_MAX_CHARS, len(match_query) * 2 + 64)
    content_chunks = _iter_chunks(clean_content, overlap=chunk_overlap)
    thinking_chunks = _iter_chunks(clean_thinking, overlap=chunk_overlap)

    pairs: list[tuple[int, str, int, str]] = []
    if len(content_chunks) == 1 and len(thinking_chunks) == 1:
        pairs.append((0, content_chunks[0][1], 0, thinking_chunks[0][1]))
    else:
        for c_off, c_chunk in content_chunks:
            pairs.append((c_off, c_chunk, 0, thinking_chunks[0][1]))
        for t_off, t_chunk in thinking_chunks[1:]:
            pairs.append((0, content_chunks[0][1], t_off, t_chunk))

    best: tuple[float, int, str, str] | None = None
    for c_off, c_chunk, t_off, t_chunk in pairs:
        hit = _run_fts5_window(conn, c_chunk, t_chunk, scoped)
        if hit is None:
            continue
        score, c_snip, t_snip = hit
        if "content" in allowed and _MATCH_OPEN in c_snip:
            excerpt = f"...{c_snip}" if c_off > 0 and not c_snip.startswith("...") else c_snip
            candidate = (score, 1, "content", excerpt)
        elif "thinking" in allowed and _MATCH_OPEN in t_snip:
            excerpt = f"...{t_snip}" if t_off > 0 and not t_snip.startswith("...") else t_snip
            candidate = (score, 0, "thinking", excerpt)
        else:
            continue
        if best is None or (candidate[0], candidate[1]) > (best[0], best[1]):
            best = candidate

    if best is not None:
        return best[2], best[3]

    # Multi-chunk cross-field fallback: if a >2MB document splits conjuncts across
    # distant chunks of content and thinking, confirm every conjunct matches somewhere
    # across the allowed chunks and return the first matching span.
    if len(content_chunks) > 1 or len(thinking_chunks) > 1:
        clauses = _split_fts5_clauses(match_query)
        if len(clauses) > 1:
            first_excerpt: tuple[str, str] | None = None
            for clause in clauses:
                clause_scoped = _scope_match_query(clause, allowed)
                clause_hit: tuple[str, str] | None = None
                for c_off, c_chunk in content_chunks:
                    res = _run_fts5_window(conn, c_chunk, "", clause_scoped)
                    if res is not None and "content" in allowed and _MATCH_OPEN in res[1]:
                        ex = (
                            f"...{res[1]}" if c_off > 0 and not res[1].startswith("...") else res[1]
                        )
                        clause_hit = ("content", ex)
                        break
                if clause_hit is None:
                    for t_off, t_chunk in thinking_chunks:
                        res = _run_fts5_window(conn, "", t_chunk, clause_scoped)
                        if res is not None and "thinking" in allowed and _MATCH_OPEN in res[2]:
                            ex = (
                                f"...{res[2]}"
                                if t_off > 0 and not res[2].startswith("...")
                                else res[2]
                            )
                            clause_hit = ("thinking", ex)
                            break
                if clause_hit is None:
                    return None
                if first_excerpt is None or (
                    first_excerpt[0] == "thinking" and clause_hit[0] == "content"
                ):
                    first_excerpt = clause_hit
            return first_excerpt

    return None


def _fallback_substring_snippet(
    content_body: str,
    thinking_body: str,
    match_query: str,
    allowed: tuple[str, ...],
) -> tuple[str, str] | None:
    terms = [
        token.strip('"').rstrip("*").lower()
        for token in match_query.split()
        if token.strip('"').rstrip("*")
    ]
    if not terms:
        return None
    for field_name, body in (("content", content_body), ("thinking", thinking_body)):
        if field_name not in allowed or not body:
            continue
        lower = body.lower()
        positions = [pos for term in terms if (pos := lower.find(term)) >= 0]
        if positions:
            start = max(0, min(positions) - 60)
            end = min(len(body), start + SNIPPET_MAX_CHARS)
            excerpt = body[start:end]
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(body):
                excerpt = excerpt + "..."
            return field_name, excerpt
    return None
