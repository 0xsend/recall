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

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any, Final

__all__ = [
    "STATIC_PROGRAM_CHARS_MAX",
    "UNRESOLVED",
    "StaticBindings",
    "collect_string_consts",
    "iter_tool_calls",
    "parse_exec_wrapper",
    "parse_js_object_literal",
    "prune_unresolved",
    "resolved_exec_commands",
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


def resolved_exec_commands(source: str) -> frozenset[str]:
    """Shell commands an ``exec`` wrapper's inner calls pass as resolved literals.

    Each becomes its own tool call row, so a scan of the wrapper program for
    other command strings must not count these again.
    """
    return frozenset(
        value
        for _, arguments in parse_exec_wrapper(source)
        if arguments is not None
        for key in ("cmd", "command")
        if isinstance(value := arguments.get(key), str)
    )


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


# ---------------------------------------------------------------------------
# Static template resolution
# ---------------------------------------------------------------------------

# Codex programs often build a command per path: `for (const p of paths)` or
# `paths.map(p => ...)` over an array literal, or one `const path = "..."`
# interpolated into a template.  Such a template is resolvable without a JS
# runtime when every interpolated name is bound exactly once in the whole
# program — by a `const` string or as the variable of a loop over a literal
# array nothing else touches.  Anything else stays unresolved, because a
# guessed command is worse than a missing one.
#
# The analysis is one linear scan of the program, capped in size: it runs at
# index time for every call a program carries.

# Real programs naming a skill stay under 7 KB; one past this is not resolved.
STATIC_PROGRAM_CHARS_MAX: Final = 16_384
_TEMPLATES_MAX: Final = 256
_ARRAY_ITEMS_MAX: Final = 64
_TEMPLATE_EXPANSIONS_MAX: Final = 64
# How many enclosing brackets a destructuring pattern may nest through.
_PATTERN_DEPTH_MAX: Final = 8

_IDENT: Final = r"[A-Za-z_$][\w$]*"
_TOKEN: Final = re.compile(r"(?<![\w$])[A-Za-z_$][\w$]*|[()\[\]{}`]")
_PATTERN: Final = re.compile(rf"\s*(?:{_IDENT}|\[\s*{_IDENT}(?:\s*,\s*{_IDENT})*\s*\])")
_IDENT_IN_PATTERN: Final = re.compile(_IDENT)
# `${name}` or `${JSON.stringify(name)}`, the two forms programs quote paths with.
_INTERPOLATION: Final = re.compile(
    rf"\$\{{\s*(JSON\s*\.\s*stringify\s*\(\s*)?({_IDENT})(?(1)\s*\))\s*\}}"
)
# A template is only a command when it is the `cmd` of the object literal
# passed straight to `tools.exec_command(...)`.
_CMD_KEY_BEFORE: Final = re.compile(r"""[{,]\s*(?:cmd|"cmd"|'cmd')\s*:\s*$""")
# A later `cmd` key or spread in the same object overrides the template.
_LATER_CMD_KEY: Final = re.compile(r"""[{,]\s*(?:cmd|"cmd"|'cmd')\s*:""")
_EXEC_TOOL: Final = "exec_command"

_DECLARING_KEYWORDS: Final = frozenset({"const", "let", "var", "function", "class"})
_DESTRUCTURING_KEYWORDS: Final = frozenset({"const", "let", "var"})
_CONTROL_KEYWORDS: Final = frozenset({"if", "while", "for", "switch", "with"})
# Words after which `[` opens an array or pattern rather than a subscript.
_EXPRESSION_KEYWORDS: Final = frozenset(
    {
        "const",
        "let",
        "var",
        "of",
        "in",
        "return",
        "typeof",
        "case",
        "await",
        "yield",
        "new",
        "delete",
        "void",
        "throw",
        "else",
        "do",
    }
)
_ITERATOR_METHODS: Final = frozenset({"map", "forEach", "flatMap"})
# An identifier longer than this reads as one opaque token when looking back.
_TOKEN_CHARS_MAX: Final = 64
_ASSIGNS_AFTER: Final = re.compile(
    r"\s*(?:=>|=(?!=)|\*\*=|<<=|>>>?=|&&=|\|\|=|\?\?=|[-+*/%&|^]=|\+\+|--)"
)
_PATTERN_ENDS: Final = re.compile(r"\s*(?:=(?![=>])|of\b|in\b)")
_PARAMS_END: Final = re.compile(r"\s*(?:=>|\{)")
_INITIALIZER: Final = re.compile(r"\s*=\s*")
# A declaration is over when the next thing cannot continue its expression.
_DECLARATION_ENDS: Final = re.compile(r"[ \t]*(?:[;,}]|$)|\s*\n\s*(?:[A-Za-z_$}]|$)")
_OF: Final = re.compile(r"\s+of\s+")
_CLOSE_PAREN: Final = re.compile(r"\s*\)")
_ARROW: Final = re.compile(r"\s*=>")
_ASYNC: Final = re.compile(r"\s*(?:async\b)?\s*")
_EXTRA_PARAM: Final = re.compile(rf"\s*(?:,\s*{_IDENT}\s*)?")
# An iterator call whose callback cannot reach the array: at most an element
# and an index parameter, never the array itself as a third.
_READING_ITERATOR_CALL: Final = re.compile(
    rf"\s*\.\s*(?:map|forEach|flatMap)\s*\(\s*(?:async\b\s*)?"
    rf"(?:{_IDENT}|\(\s*(?:{_IDENT}|\[\s*{_IDENT}(?:\s*,\s*{_IDENT})*\s*\])"
    rf"(?:\s*,\s*{_IDENT})?\s*\))\s*=>"
)
_LENGTH_AFTER: Final = re.compile(r"\s*\.\s*length\b")
_DESTRUCTURING_ITERATOR_CALL: Final = re.compile(
    rf"\s*\.\s*(?:map|forEach|flatMap)\s*\(\s*(?:async\b\s*)?"
    rf"\(\s*\[\s*{_IDENT}(?:\s*,\s*{_IDENT})*\s*\](?:\s*,\s*{_IDENT})?\s*\)\s*=>"
)
_DESTRUCTURING_CONST: Final = re.compile(r"\s*const\s*\[")
_INDEX_AFTER: Final = re.compile(r"\s*\[")
# What may follow a read of an array element or length: anything that neither
# reaches further into it (`[`, `.`, `?.`, a call or tagged template) nor
# assigns to it.
_REACHES_FURTHER: Final = re.compile(r"\s*(?:[\[.(`]|\?\.)")

_OPENERS: Final = frozenset("([{")


@dataclass(frozen=True)
class _Iteration:
    """A loop or callback walking a literal array; one row per element."""

    rows: tuple[tuple[str, ...], ...]
    scope_start: int
    scope_end: int


@dataclass(frozen=True)
class _Const:
    value: str


@dataclass(frozen=True)
class _Column:
    """One name a loop pattern binds to a column of each iteration row."""

    iteration: _Iteration
    column: int


_Binding = _Const | _Column


@dataclass(frozen=True)
class _Occurrence:
    """An identifier in executable code, not a member name."""

    start: int
    end: int
    enclosing: int  # innermost open bracket, or -1 at top level


@dataclass(frozen=True)
class _Scan:
    source: str
    code: str  # strings, comments and regexes blanked; offsets match source
    templates: tuple[tuple[int, int, int], ...]  # start, end, enclosing bracket
    close_of: dict[int, int]
    parent_of: dict[int, int]
    occurrences: dict[str, tuple[_Occurrence, ...]]
    previous_code: tuple[int, ...]  # [i]: last non-space index before i, or -1

    def tokens_before(self, pos: int, count: int) -> list[str]:
        """Up to ``count`` tokens ending before ``pos``, oldest first.

        Whitespace is skipped however long it runs; a punctuator is one token.
        """
        tokens: list[str] = []
        cursor = pos
        while len(tokens) < count:
            last = self.previous_code[cursor]
            if last < 0:
                break
            if self.code[last] in _IDENT_BODY:
                first = last
                while (
                    first > 0
                    and self.code[first - 1] in _IDENT_BODY
                    and last - first < _TOKEN_CHARS_MAX
                ):
                    first -= 1
                tokens.append(self.code[first : last + 1])
                cursor = first
            else:
                tokens.append(self.code[last])
                cursor = last
        tokens.reverse()
        return tokens

    def last_token(self, pos: int) -> str:
        tokens = self.tokens_before(pos, 1)
        return tokens[0] if tokens else ""

    def close(self, open_pos: int) -> int:
        """Where a bracket closes; an unclosed one runs to the end."""
        return self.close_of.get(open_pos, len(self.code))


class StaticBindings:
    """The names in one program whose values are known without running it."""

    def __init__(self, source: str) -> None:
        assert len(source) <= STATIC_PROGRAM_CHARS_MAX, "caller must bound the program"
        self._scan = _scan(source)
        self._names: dict[str, _Binding | None] = {}
        self._arrays: dict[str, tuple[Any, ...] | None] = {}
        self._iterations: dict[int, _Iteration | None] = {}

    def exec_commands(self) -> Iterator[str]:
        """Every command a template passed as an exec call's ``cmd`` evaluates to."""
        scan = self._scan
        for start, end, enclosing in scan.templates[:_TEMPLATES_MAX]:
            if _is_exec_cmd(scan, start, end, enclosing):
                yield from self._expand(scan.source[start + 1 : end - 1], start)

    def _expand(self, template: str, offset: int) -> tuple[str, ...]:
        if "\\$" in template:
            return ()
        chunks = _INTERPOLATION.split(template)
        statics, stringified, names = chunks[::3], chunks[1::3], chunks[2::3]
        if any("${" in chunk for chunk in statics):
            return ()
        fixed: dict[str, str] = {}
        columns: dict[str, _Column] = {}
        for name in names:
            binding = self._binding(name)
            if binding is None:
                return ()
            if isinstance(binding, _Const):
                fixed[name] = binding.value
                continue
            iteration = binding.iteration
            if not iteration.scope_start <= offset < iteration.scope_end:
                return ()
            columns[name] = binding
        iterations = list(dict.fromkeys(column.iteration for column in columns.values()))
        combinations = 1
        for iteration in iterations:
            combinations *= len(iteration.rows)
        if combinations > _TEMPLATE_EXPANSIONS_MAX:
            return ()
        try:
            texts = [_template_text(chunk) for chunk in statics]
        except JsLiteralError:
            return ()
        commands: dict[str, None] = {}
        for rows in product(*(iteration.rows for iteration in iterations)):
            row_of = dict(zip(iterations, rows, strict=True))
            values = dict(fixed)
            for name, column in columns.items():
                values[name] = row_of[column.iteration][column.column]
            parts = [texts[0]]
            for name, as_json, text in zip(names, stringified, texts[1:], strict=True):
                value = values[name]
                parts.extend((json.dumps(value, ensure_ascii=False) if as_json else value, text))
            commands["".join(parts)] = None
        return tuple(commands)

    def _binding(self, name: str) -> _Binding | None:
        if name not in self._names:
            self._names[name] = self._resolve(name)
        return self._names[name]

    def _resolve(self, name: str) -> _Binding | None:
        """The single binding of ``name``, if it is one this reader evaluates."""
        scan = self._scan
        bindings = [site for site in scan.occurrences.get(name, ()) if _binds(scan, site)]
        if len(bindings) != 1:
            return None
        site = bindings[0]
        if scan.last_token(site.start) == "const" and _INITIALIZER.match(scan.code, site.end):
            value = _declared_literal(scan, site)
            return _Const(value) if isinstance(value, str) else None
        return self._loop_column(site)

    def _loop_column(self, site: _Occurrence) -> _Column | None:
        """The loop column ``site`` binds, for a ``const`` loop or callback param."""
        scan = self._scan
        pattern_open = site.enclosing
        if pattern_open >= 0 and scan.code[pattern_open] == "[":
            enclosing = scan.parent_of[pattern_open]
            pattern_start = pattern_open
        else:
            enclosing = pattern_open
            pattern_start = site.start
        if enclosing < 0 or scan.code[enclosing] != "(":
            return None
        if enclosing not in self._iterations:
            self._iterations[enclosing] = self._iteration(enclosing, pattern_start, site)
        iteration = self._iterations[enclosing]
        if iteration is None:
            return None
        pattern = _PATTERN.match(scan.code, pattern_start)
        assert pattern is not None, "a binding site starts a pattern"
        names = [token.start() for token in _IDENT_IN_PATTERN.finditer(pattern.group(0))]
        column = names.index(site.start - pattern.start())
        return _Column(iteration, column)

    def _iteration(self, paren: int, pattern_start: int, site: _Occurrence) -> _Iteration | None:
        scan = self._scan
        code = scan.code
        pattern = _PATTERN.match(code, pattern_start)
        if pattern is None or not pattern.start() <= site.start < pattern.end():
            return None
        width = None
        if pattern.group(0).lstrip().startswith("["):
            width = len(_IDENT_IN_PATTERN.findall(pattern.group(0)))
        header = code[paren + 1 : pattern_start]
        if _is_for_header(scan, paren):
            # `for (const <pattern> of <array>) <body>`
            if not re.fullmatch(r"\s*const\s+", header):
                return None
            of = _OF.match(code, pattern.end())
            if of is None:
                return None
            items, source_end = self._iterable(of.end())
            closing = _CLOSE_PAREN.match(code, source_end)
            if items is None or closing is None or closing.end() - 1 != scan.close(paren):
                return None
            body = _skip_ws(code, closing.end())
            if body < len(code) and code[body] == "{":
                scope_end = scan.close(body)
            else:
                semicolon = code.find(";", body)
                scope_end = len(code) if semicolon < 0 else semicolon
            return _iteration_of(items, width, body, scope_end)
        # `<array>.map(<name> => ...)` or `<array>.map((<pattern>, i) => ...)`
        if pattern_start == site.start and _ARROW.match(code, pattern.end()):
            call = paren
            if _ASYNC.fullmatch(code, call + 1, pattern_start) is None:
                return None
        elif header.strip() == "" and _ARROW.match(code, scan.close(paren) + 1):
            extra = _EXTRA_PARAM.match(code, pattern.end())
            assert extra is not None, "the extra parameter is optional"
            if extra.end() != scan.close(paren):
                return None
            call = scan.parent_of[paren]
            if call < 0 or code[call] != "(" or _ASYNC.fullmatch(code, call + 1, paren) is None:
                return None
        else:
            return None
        receiver = _iterator_receiver(scan, call)
        if receiver is None:
            return None
        items = self._array(receiver)
        return _iteration_of(items, width, call, scan.close(call))

    def _iterable(self, pos: int) -> tuple[tuple[Any, ...] | None, int]:
        """The literal array a ``for ... of`` walks, and where it ends."""
        scan = self._scan
        if pos < len(scan.code) and scan.code[pos] == "[":
            try:
                value, end = _read_value(scan.source, pos, {}, depth=0)
            except JsLiteralError:
                return None, pos
            return (tuple(value) if isinstance(value, list) else None), end
        name, end = _read_identifier(scan.code, pos)
        return (self._array(name) if name else None), end

    def _array(self, name: str) -> tuple[Any, ...] | None:
        if name not in self._arrays:
            self._arrays[name] = _unchanged_array(self._scan, name)
        return self._arrays[name]


def _unchanged_array(scan: _Scan, name: str) -> tuple[Any, ...] | None:
    """A ``const`` array literal only ever iterated, indexed or measured.

    Any other use — a mutating method, an alias, an argument, a spread — could
    change what a loop over it sees, so the array is not known.
    """
    declaration: tuple[Any, ...] | None = None
    uses: list[_Occurrence] = []
    for site in scan.occurrences.get(name, ()):
        if not _binds(scan, site):
            uses.append(site)
            continue
        if declaration is not None or scan.last_token(site.start) != "const":
            return None
        value = _declared_literal(scan, site)
        if not isinstance(value, list):
            return None
        declaration = tuple(value)
    if declaration is None:
        return None
    # A row reached by any name but a destructuring pattern can be changed
    # through that name, so rows are only ever taken apart by the loop itself.
    rows_are_arrays = any(isinstance(item, list) for item in declaration)
    reads = _reads_rows if rows_are_arrays else _reads_array
    if not all(reads(scan, site) for site in uses):
        return None
    return declaration


def _reads_rows(scan: _Scan, site: _Occurrence) -> bool:
    """Whether this use of an array of rows destructures each row in place.

    Measuring it is also safe: ``.length`` reaches no row.
    """
    code = scan.code
    if _DESTRUCTURING_ITERATOR_CALL.match(code, site.end):
        return True
    length = _LENGTH_AFTER.match(code, site.end)
    if length is not None:
        return _only_read(code, length.end())
    header = site.enclosing
    return (
        scan.last_token(site.start) == "of"
        and _CLOSE_PAREN.match(code, site.end) is not None
        and _is_for_header(scan, header)
        and _DESTRUCTURING_CONST.match(code, header + 1) is not None
    )


def _reads_array(scan: _Scan, site: _Occurrence) -> bool:
    """Whether this use of an array can only observe it, never change it."""
    code = scan.code
    before = scan.tokens_before(site.start, 2)
    if before[-1:] == ["delete"] or before in (["+", "+"], ["-", "-"]):
        return False
    if _READING_ITERATOR_CALL.match(code, site.end):
        return True
    if before[-1:] == ["of"] and _CLOSE_PAREN.match(code, site.end):
        return True
    length = _LENGTH_AFTER.match(code, site.end)
    if length is not None:
        return _only_read(code, length.end())
    index = _INDEX_AFTER.match(code, site.end)
    if index is not None:
        return _only_read(code, scan.close(index.end() - 1) + 1)
    return False


def _only_read(code: str, pos: int) -> bool:
    return not (_REACHES_FURTHER.match(code, pos) or _ASSIGNS_AFTER.match(code, pos))


def _declared_literal(scan: _Scan, site: _Occurrence) -> Any:
    """The literal a ``const`` declares, or ``None`` if more follows it."""
    initializer = _INITIALIZER.match(scan.code, site.end)
    if initializer is None or initializer.end() >= len(scan.code):
        return None
    try:
        value, end = _read_value(scan.source, initializer.end(), {}, depth=0)
    except JsLiteralError:
        return None
    if value is UNRESOLVED or _DECLARATION_ENDS.match(scan.code, end) is None:
        return None
    return value


def _binds(scan: _Scan, site: _Occurrence) -> bool:
    """Whether this occurrence (re)binds its name rather than reading it.

    Deliberately broad — a false positive only leaves a name unresolved.
    """
    before = scan.tokens_before(site.start, 2)
    if (before and before[-1] in _DECLARING_KEYWORDS) or before == ["function", "*"]:
        return True
    if before in (["+", "+"], ["-", "-"]) or _ASSIGNS_AFTER.match(scan.code, site.end):
        return True
    # `for (name of ...)` assigns an existing binding on every iteration.
    if before[-1:] == ["("] and _is_for_header(scan, scan.previous_code[site.start]):
        return True
    return _in_pattern(scan, site.enclosing)


def _in_pattern(scan: _Scan, open_pos: int) -> bool:
    """Whether a bracket is a parameter list or a destructuring target."""
    code = scan.code
    for _ in range(_PATTERN_DEPTH_MAX):
        if open_pos < 0:
            return False
        if code[open_pos] == "(":
            if open_pos not in scan.close_of:
                return True
            return bool(_PARAMS_END.match(code, scan.close(open_pos) + 1)) and (
                _control_keyword(scan, open_pos) is None
            )
        previous = scan.last_token(open_pos)
        if previous in _DESTRUCTURING_KEYWORDS:
            return True
        if code[open_pos] == "[" and _is_subscript_after(previous):
            # `out[name] = ...` assigns an element; it binds nothing.
            return False
        if _PATTERN_ENDS.match(code, scan.close(open_pos) + 1):
            return True
        open_pos = scan.parent_of[open_pos]
    return True


def _scan(source: str) -> _Scan:
    """Blank non-code, pair brackets and index identifiers in one pass each."""
    chars = list(source)
    template_ends: dict[int, int] = {}
    pos = 0
    while pos < len(source):
        end = _skip_non_code(source, pos)
        if end == pos:
            pos += 1
            continue
        quoted = source[pos] in _QUOTES
        if source[pos] == "`" and end - pos >= 2 and source[end - 1] == "`":
            template_ends[pos] = end
        for index in range(pos + quoted, end - quoted):
            if chars[index] != "\n":
                chars[index] = " "
        pos = end
    code = "".join(chars)
    previous_code = [-1] * (len(code) + 1)
    for index, char in enumerate(code):
        previous_code[index + 1] = previous_code[index] if char.isspace() else index

    close_of: dict[int, int] = {}
    parent_of: dict[int, int] = {}
    occurrences: dict[str, list[_Occurrence]] = {}
    templates: list[tuple[int, int, int]] = []
    stack: list[int] = []
    for token in _TOKEN.finditer(code):
        text, start = token.group(0), token.start()
        if text == "`":
            if start in template_ends:
                templates.append((start, template_ends[start], stack[-1] if stack else -1))
        elif text in _OPENERS:
            parent_of[start] = stack[-1] if stack else -1
            stack.append(start)
        elif text in ")]}":
            if stack:
                close_of[stack.pop()] = start
        elif not _is_member(code, start):
            site = _Occurrence(start, token.end(), stack[-1] if stack else -1)
            occurrences.setdefault(text, []).append(site)
    return _Scan(
        source=source,
        code=code,
        templates=tuple(templates),
        close_of=close_of,
        parent_of=parent_of,
        occurrences={name: tuple(sites) for name, sites in occurrences.items()},
        previous_code=tuple(previous_code),
    )


def _is_exec_cmd(scan: _Scan, template: int, template_end: int, enclosing: int) -> bool:
    """Whether a template is the ``cmd`` of ``tools.exec_command({...})``."""
    code = scan.code
    if enclosing < 0 or code[enclosing] != "{":
        return False
    key_window = max(0, template - _TOKEN_CHARS_MAX)
    if _CMD_KEY_BEFORE.search(scan.source, key_window, template) is None:
        return False
    rest = (template_end, scan.close(enclosing))
    if _LATER_CMD_KEY.search(scan.source, *rest) or "..." in code[rest[0] : rest[1]]:
        return False
    call = scan.parent_of[enclosing]
    if call < 0 or code[call] != "(" or code[call + 1 : enclosing].strip():
        return False
    closing = _CLOSE_PAREN.match(code, scan.close(enclosing) + 1)
    if closing is None or closing.end() - 1 != scan.close(call):
        return False
    callee = scan.tokens_before(call, 4)
    return callee[-3:] == ["tools", ".", _EXEC_TOOL] and callee[:-3] != ["."]


def _is_for_header(scan: _Scan, paren: int) -> bool:
    """Whether the bracket at ``paren`` opens a ``for`` or ``for await`` header."""
    return paren >= 0 and scan.code[paren] == "(" and _control_keyword(scan, paren) == "for"


def _control_keyword(scan: _Scan, paren: int) -> str | None:
    """The statement keyword a parenthesis belongs to, if any."""
    before = scan.tokens_before(paren, 2)
    if before[-1:] == ["await"]:
        before = before[:-1]
    if before and before[-1] in _CONTROL_KEYWORDS:
        return before[-1]
    return None


def _iterator_receiver(scan: _Scan, call: int) -> str | None:
    """The array ``name`` in ``name.map(`` whose argument opens at ``call``."""
    before = scan.tokens_before(call, 4)
    if len(before) < 3 or before[-1] not in _ITERATOR_METHODS or before[-2] != ".":
        return None
    name = before[-3]
    if not name or name[0] not in _IDENT_START or before[:-3] == ["."]:
        return None
    return name


def _is_subscript_after(previous: str) -> bool:
    """Whether ``[`` after this token indexes a value rather than opening a literal."""
    if previous in (")", "]"):
        return True
    return bool(previous) and previous[0] in _IDENT_START and previous not in _EXPRESSION_KEYWORDS


def _is_member(code: str, pos: int) -> bool:
    """``obj.name`` names a property; ``...name`` is still the variable."""
    cursor = pos - 1
    while cursor >= 0 and code[cursor].isspace():
        cursor -= 1
    if cursor < 0 or code[cursor] != ".":
        return False
    return not code.endswith("...", 0, cursor + 1)


def _iteration_of(
    items: tuple[Any, ...] | None, width: int | None, scope_start: int, scope_end: int
) -> _Iteration | None:
    if items is None:
        return None
    rows = _iteration_rows(items, width)
    if rows is None:
        return None
    return _Iteration(rows=rows, scope_start=scope_start, scope_end=scope_end)


def _iteration_rows(
    items: tuple[Any, ...], width: int | None
) -> tuple[tuple[str, ...], ...] | None:
    """Each element as a row of strings, or ``None`` if any element is not one.

    ``width`` is the destructured element length; ``None`` binds whole elements.
    """
    if not items or len(items) > _ARRAY_ITEMS_MAX:
        return None
    rows: list[tuple[str, ...]] = []
    for item in items:
        if width is None:
            fields = [item]
        elif isinstance(item, list) and len(item) >= width:
            fields = item[:width]
        else:
            return None
        row: list[str] = []
        for field in fields:
            text = _scalar_text(field)
            if text is None:
                return None
            row.append(text)
        rows.append(tuple(row))
    return tuple(rows)


def _scalar_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _template_text(chunk: str) -> str:
    """Decode the escapes in one static stretch of a template literal."""
    value, _ = _read_string("`" + chunk + "`", 0)
    if not isinstance(value, str):
        raise JsLiteralError("template chunk holds an interpolation")
    return value
