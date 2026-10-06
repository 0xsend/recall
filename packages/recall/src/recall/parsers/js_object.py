"""Tolerant reader for the JS object literals Codex embeds in ``exec`` calls.

Codex ships tool arguments as JavaScript source, not JSON: keys are usually
bare identifiers, strings may be single-quoted, trailing commas are common,
and values are sometimes shorthand references to consts declared earlier in
the same program.  ``json.loads`` rejects roughly 60% of real call sites, so
the wrapper needs a reader that speaks the literal subset those programs
actually use.

Only literals are evaluated.  Anything that needs a JS runtime (interpolated
templates, expressions, function calls) reads as :data:`UNRESOLVED`, which
callers drop rather than guess at.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

__all__ = [
    "UNRESOLVED",
    "collect_string_consts",
    "iter_tool_calls",
    "parse_exec_wrapper",
    "parse_js_object_literal",
    "prune_unresolved",
]


class _Unresolved:
    """Sentinel for a value that only a JS runtime could produce."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNRESOLVED"


UNRESOLVED: Final = _Unresolved()

_QUOTES: Final = frozenset({'"', "'", "`"})
_IDENT_START: Final = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_$")
_IDENT_BODY: Final = _IDENT_START | frozenset("0123456789")
_NUMBER_BODY: Final = frozenset("0123456789+-.eExXabcdefABCDEF")
_ESCAPES: Final = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    "v": "\v",
    "0": "\0",
    "\n": "",  # line continuation
}


class JsLiteralError(ValueError):
    """Raised when the text is not a literal this reader can evaluate."""


def parse_js_object_literal(
    text: str,
    pos: int = 0,
    *,
    consts: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Read one ``{...}`` literal starting at or after ``pos``.

    Returns the decoded mapping and the index just past the closing brace.
    Shorthand properties and identifier values resolve against ``consts``;
    unresolvable ones come back as :data:`UNRESOLVED` so the caller decides
    whether to drop the key or keep the raw source instead.

    Raises :class:`JsLiteralError` when the text is not a readable literal.
    """
    pos = _skip_ws(text, pos)
    value, end = _read_value(text, pos, consts or {}, depth=0)
    if not isinstance(value, dict):
        raise JsLiteralError("expected an object literal")
    return value, end


def collect_string_consts(source: str) -> dict[str, str]:
    """Map ``const NAME = "literal"`` declarations to their string values.

    Codex hoists long commands and patch bodies into a const and passes them
    by shorthand (``tools.exec_command({cmd, workdir})``), so without this the
    command text is unreachable.  Declarations whose value is not a plain
    string literal are skipped, as are interpolated templates.

    Only declarations in executable code count.  A commented-out or quoted
    ``const cmd = ...`` never bound anything, and taking it as the binding
    would persist a command that never ran.
    """
    consts: dict[str, str] = {}
    pos = 0
    length = len(source)
    while pos < length:
        skipped = _skip_non_code(source, pos)
        if skipped != pos:
            pos = skipped
            continue
        if not source.startswith(_CONST_KEYWORD, pos) or _is_ident_char(source, pos - 1):
            pos += 1
            continue
        cursor = pos + len(_CONST_KEYWORD)
        if cursor < length and source[cursor] in _IDENT_BODY:
            pos += 1
            continue
        cursor = _skip_ws(source, cursor)
        name, cursor = _read_identifier(source, cursor)
        if not name:
            pos += len(_CONST_KEYWORD)
            continue
        cursor = _skip_ws(source, cursor)
        if cursor >= length or source[cursor] != "=":
            pos += len(_CONST_KEYWORD)
            continue
        cursor = _skip_ws(source, cursor + 1)
        if cursor >= length or source[cursor] not in _QUOTES:
            pos += len(_CONST_KEYWORD)
            continue
        try:
            value, cursor = _read_string(source, cursor)
        except JsLiteralError:
            pos += len(_CONST_KEYWORD)
            continue
        if isinstance(value, str):
            consts.setdefault(name, value)
        pos = cursor
    return consts


def prune_unresolved(value: Any) -> tuple[Any, bool]:
    """Strip :data:`UNRESOLVED` from a decoded literal, at any depth.

    Returns the cleaned value and whether anything was removed.  A sentinel
    that survives into a ``tool_input`` is not a parse detail: ``json.dumps``
    raises on it when the row is written, failing the whole session's index.

    Unresolved mapping keys are dropped; unresolved list items become
    ``None`` so the surrounding positions still line up.
    """
    if value is UNRESOLVED:
        return None, True
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        dropped = False
        for key, item in value.items():
            if item is UNRESOLVED:
                dropped = True
                continue
            pruned, item_dropped = prune_unresolved(item)
            cleaned[key] = pruned
            dropped = dropped or item_dropped
        return cleaned, dropped
    if isinstance(value, list):
        items: list[Any] = []
        dropped = False
        for item in value:
            pruned, item_dropped = prune_unresolved(item)
            items.append(pruned)
            dropped = dropped or item_dropped
        return items, dropped
    return value, False


_TOOLS_PREFIX: Final = "tools."
_CONST_KEYWORD: Final = "const"


def iter_tool_calls(source: str) -> Iterator[tuple[str, int]]:
    """Yield ``(tool_name, offset)`` for each ``tools.<name>(`` in a program.

    ``offset`` points just past the opening paren, where the argument starts.

    String literals, comments, and regex literals are skipped, because their
    contents did not run: a patch body, an echoed snippet, or a search
    pattern routinely carries the text of a tool call, and counting it
    invents a call.  Template interpolations are treated as opaque string
    content too, so a call built inside ``${...}`` is missed rather than
    double-counted.

    What this yields is call *sites*, not executions.  Control flow is not
    evaluated, so a call guarded by a branch that never runs is yielded
    anyway, and a call inside a loop is yielded once however many times it
    ran.  Resolving either needs a JS engine, and the transcript cannot
    settle it independently: the wrapper's ``custom_tool_call_output`` is one
    combined blob for the whole program, never per-call results.
    """
    pos = 0
    length = len(source)
    while pos < length:
        skipped = _skip_non_code(source, pos)
        if skipped != pos:
            pos = skipped
            continue
        if source.startswith(_TOOLS_PREFIX, pos) and not _is_ident_char(source, pos - 1):
            cursor = pos + len(_TOOLS_PREFIX)
            name, cursor = _read_identifier(source, cursor)
            after = _skip_ws(source, cursor)
            if name and after < length and source[after] == "(":
                yield name, after + 1
                pos = after + 1
                continue
        pos += 1


def _skip_non_code(source: str, pos: int) -> int:
    """Return the index just past any literal or comment starting at ``pos``.

    Returns ``pos`` unchanged when the position is executable code.
    """
    char = source[pos]
    if char in _QUOTES:
        return _skip_string(source, pos)
    if source.startswith("//", pos):
        newline = source.find("\n", pos)
        return len(source) if newline < 0 else newline + 1
    if source.startswith("/*", pos):
        close = source.find("*/", pos + 2)
        return len(source) if close < 0 else close + 2
    if char == "/" and _starts_regex(source, pos):
        return _skip_regex(source, pos)
    return pos


# A `/` opens a regex only where a value may begin.  After a value — an
# identifier, literal, or closing bracket — it is division, and treating that
# as a regex would swallow the rest of the program.
_REGEX_PRECEDERS: Final = frozenset("(,=:[!&|?{};+-*%~^<>")


def _starts_regex(source: str, pos: int) -> bool:
    cursor = pos - 1
    while cursor >= 0 and source[cursor].isspace():
        cursor -= 1
    if cursor < 0:
        return True
    if source[cursor] in _REGEX_PRECEDERS:
        return True
    # `return /re/`, `typeof /re/`, and friends: a keyword can precede one.
    word_end = cursor + 1
    while cursor >= 0 and source[cursor] in _IDENT_BODY:
        cursor -= 1
    return source[cursor + 1 : word_end] in _REGEX_KEYWORDS


_REGEX_KEYWORDS: Final = frozenset({"return", "typeof", "case", "in", "of", "new", "delete", "do"})


def _skip_regex(source: str, pos: int) -> int:
    """Return the index just past the regex literal starting at ``pos``."""
    cursor = pos + 1
    length = len(source)
    in_class = False
    while cursor < length:
        char = source[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == "\n":
            # An unterminated regex is not a regex; leave the `/` as code.
            return pos + 1
        if char == "[":
            in_class = True
        elif char == "]":
            in_class = False
        elif char == "/" and not in_class:
            cursor += 1
            while cursor < length and source[cursor] in _IDENT_BODY:
                cursor += 1
            return cursor
        cursor += 1
    return pos + 1


def _is_ident_char(source: str, pos: int) -> bool:
    return 0 <= pos < len(source) and source[pos] in _IDENT_BODY


def _skip_string(source: str, pos: int) -> int:
    """Return the index just past the string literal starting at ``pos``."""
    quote = source[pos]
    cursor = pos + 1
    length = len(source)
    while cursor < length:
        char = source[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == quote:
            return cursor + 1
        cursor += 1
    return length


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

_MAX_DEPTH: Final = 32


def _skip_ws(text: str, pos: int) -> int:
    length = len(text)
    while pos < length:
        char = text[pos]
        if char.isspace():
            pos += 1
        elif text.startswith("//", pos):
            newline = text.find("\n", pos)
            pos = length if newline < 0 else newline + 1
        elif text.startswith("/*", pos):
            close = text.find("*/", pos + 2)
            pos = length if close < 0 else close + 2
        else:
            break
    return pos


def _read_value(text: str, pos: int, consts: dict[str, Any], *, depth: int) -> tuple[Any, int]:
    if depth > _MAX_DEPTH:
        raise JsLiteralError("literal nested too deeply")
    if pos >= len(text):
        raise JsLiteralError("unexpected end of literal")
    char = text[pos]
    if char == "{":
        return _read_object(text, pos, consts, depth=depth)
    if char == "[":
        return _read_array(text, pos, consts, depth=depth)
    if char in _QUOTES:
        return _read_string(text, pos)
    if char in _IDENT_START:
        name, end = _read_identifier(text, pos)
        if name == "true":
            return True, end
        if name == "false":
            return False, end
        if name in ("null", "undefined"):
            return None, end
        return consts.get(name, UNRESOLVED), end
    if char in "0123456789+-.":
        return _read_number(text, pos)
    raise JsLiteralError(f"unreadable value at {pos}")


def _read_object(text: str, pos: int, consts: dict[str, Any], *, depth: int) -> tuple[Any, int]:
    result: dict[str, Any] = {}
    pos = _skip_ws(text, pos + 1)
    while pos < len(text) and text[pos] != "}":
        key, pos = _read_key(text, pos)
        pos = _skip_ws(text, pos)
        if pos < len(text) and text[pos] == ":":
            value, pos = _read_value(text, _skip_ws(text, pos + 1), consts, depth=depth + 1)
        else:
            # Shorthand property: the key names a binding in the program.
            value = consts.get(key, UNRESOLVED)
        result[key] = value
        pos = _skip_ws(text, pos)
        if pos < len(text) and text[pos] == ",":
            pos = _skip_ws(text, pos + 1)
    if pos >= len(text):
        raise JsLiteralError("unterminated object literal")
    return result, pos + 1


def _read_array(text: str, pos: int, consts: dict[str, Any], *, depth: int) -> tuple[Any, int]:
    items: list[Any] = []
    pos = _skip_ws(text, pos + 1)
    while pos < len(text) and text[pos] != "]":
        value, pos = _read_value(text, pos, consts, depth=depth + 1)
        items.append(value)
        pos = _skip_ws(text, pos)
        if pos < len(text) and text[pos] == ",":
            pos = _skip_ws(text, pos + 1)
    if pos >= len(text):
        raise JsLiteralError("unterminated array literal")
    return items, pos + 1


def _read_key(text: str, pos: int) -> tuple[str, int]:
    char = text[pos]
    if char in _QUOTES:
        key, end = _read_string(text, pos)
        if not isinstance(key, str):
            raise JsLiteralError("computed key")
        return key, end
    name, end = _read_identifier(text, pos)
    if not name:
        raise JsLiteralError(f"unreadable key at {pos}")
    return name, end


def _read_identifier(text: str, pos: int) -> tuple[str, int]:
    if pos >= len(text) or text[pos] not in _IDENT_START:
        return "", pos
    end = pos + 1
    while end < len(text) and text[end] in _IDENT_BODY:
        end += 1
    return text[pos:end], end


def _read_number(text: str, pos: int) -> tuple[Any, int]:
    end = pos + 1
    while end < len(text) and text[end] in _NUMBER_BODY:
        end += 1
    raw = text[pos:end]
    try:
        return (float(raw) if any(c in raw for c in ".eE") else int(raw, 0)), end
    except ValueError as err:
        raise JsLiteralError(f"unreadable number {raw!r}") from err


def _read_string(text: str, pos: int) -> tuple[Any, int]:
    quote = text[pos]
    parts: list[str] = []
    cursor = pos + 1
    length = len(text)
    while cursor < length:
        char = text[cursor]
        if char == "\\":
            if cursor + 1 >= length:
                break
            escape = text[cursor + 1]
            if escape == "u":
                parts.append(_read_unicode_escape(text, cursor + 2))
                cursor += 6
                continue
            parts.append(_ESCAPES.get(escape, escape))
            cursor += 2
            continue
        if char == quote:
            if quote == "`" and "${" in "".join(parts):
                # Interpolated template: the value depends on runtime state.
                return UNRESOLVED, cursor + 1
            return "".join(parts), cursor + 1
        parts.append(char)
        cursor += 1
    raise JsLiteralError("unterminated string literal")


def _read_unicode_escape(text: str, pos: int) -> str:
    digits = text[pos : pos + 4]
    try:
        return chr(int(digits, 16))
    except ValueError as err:
        raise JsLiteralError(f"bad unicode escape {digits!r}") from err


def parse_exec_wrapper(source: str) -> list[tuple[str, dict[str, Any] | None]]:
    """Recover the inner tool calls from an ``exec`` wrapper program.

    Returns ``(tool_name, arguments)`` in source order.  ``arguments`` is
    ``None`` when the call site passes something other than an object literal
    (``tools.apply_patch(patch)`` passes a variable), and values that only a
    JS runtime could produce are dropped at any depth, so callers can fall
    back to keeping the raw source instead of storing a guess.
    """
    consts = collect_string_consts(source)
    calls: list[tuple[str, dict[str, Any] | None]] = []
    for name, offset in iter_tool_calls(source):
        start = offset
        while start < len(source) and source[start].isspace():
            start += 1
        arguments: dict[str, Any] | None = None
        if start < len(source) and source[start] == "{":
            try:
                decoded, _ = parse_js_object_literal(source, start, consts=consts)
            except JsLiteralError:
                decoded = None
            if decoded is not None:
                arguments, dropped = prune_unresolved(decoded)
                if dropped:
                    # Something needed a runtime.  Keep what resolved and the
                    # program itself, so no text is lost to the omission.
                    arguments["source"] = source
        else:
            # `tools.apply_patch(patch)` passes a variable.  When it is a
            # plain string const the reader already collected, the payload is
            # exact, and naming it `input` matches the direct custom_tool_call
            # shape so both forms aggregate (REQ-PARSE-018).  Otherwise leave
            # it None so the caller keeps the whole program.
            identifier = _read_argument_identifier(source, start)
            if identifier is not None and identifier in consts:
                arguments = {"input": consts[identifier]}
        calls.append((name, arguments))
    return calls


def _read_argument_identifier(source: str, pos: int) -> str | None:
    """Read a bare JS identifier at ``pos``, or ``None`` if one is not there."""
    end = pos
    while end < len(source) and (source[end].isalnum() or source[end] in "_$"):
        end += 1
    if end == pos or source[pos].isdigit():
        return None
    # Anything but a plain close-paren means the argument is an expression
    # (a call, a member access, a concatenation), which is not recoverable.
    rest = end
    while rest < len(source) and source[rest].isspace():
        rest += 1
    if rest >= len(source) or source[rest] != ")":
        return None
    return source[pos:end]
