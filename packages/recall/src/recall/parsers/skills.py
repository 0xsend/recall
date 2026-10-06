"""Attribute a tool call to the skill it loaded.

Only Claude Code has a typed ``Skill`` tool; every other harness loads a skill
by reading its ``SKILL.md`` with an ordinary read or shell call.  Those loads
are recorded in full, but nothing ties them to a skill, so cross-harness skill
analytics reads as a structural zero.

Reading a ``SKILL.md`` is not the same act as working on one.  Two signals
separate them, and both are required:

* the path sits under an **installed** skills root (a plugin cache or a
  harness skills dir) or a checkout laid out like one (a repository's own
  ``.agents/skills``, the dotfiles tree a harness root links to) — a plain
  ``skills/foo/SKILL.md`` is authoring;
* the access is **read-shaped** — ``sed``/``cat``/``head`` or a read tool, not
  ``git diff`` (authoring) and not ``find``/``rg`` (discovery).

Names come back in Claude Code's own ``plugin:skill`` form when the path
carries a plugin, so counts from different harnesses aggregate.  One call can
load several skills (``cat a/SKILL.md b/SKILL.md``, ``cat skills/{a,b}/SKILL.md``);
each is reported once.
"""

from __future__ import annotations

import re
import shlex
from posixpath import join, normpath
from typing import Any, Final

__all__ = ["derive_skill_name", "derive_skill_names", "is_skill_candidate"]

_SKILL_FILE: Final = "SKILL.md"

# Roots a harness installs skills into.  A path outside all of them is a
# working copy, and reading it says nothing about which skill fired.
_INSTALLED_ROOTS: Final = (
    ".agents/skills/",
    ".claude/plugins/",
    ".claude/skills/",
    ".codex/plugins/",
    ".codex/skills/",
    ".grok/bundled/skills/",
    ".pi/agent/git/",
    ".pi/agent/plugins/",
    ".pi/agent/skills/",
)

# An installed root only counts directly under a home directory; anywhere else
# the path is a checkout, attributed only through `_CHECKOUT_SKILL_LAYOUT` or
# the agent-profile source layout.
_HOME_PREFIX: Final = re.compile(r"^(?:~|\$HOME|/root|/var/root|/(?:Users|home)/[^/]+)/")

# Harness skill layouts that also occur outside a home directory: a
# repository's own project skills, or a dotfiles checkout that
# `~/.codex/skills` links to and an agent reads through the resolved link.
_CHECKOUT_SKILL_LAYOUT: Final = re.compile(r"/\.(?:agents|claude|codex)/skills/")

_SHELL_EXPANSION: Final = re.compile(r"[$*?\[\]]")

# Codex code mode runs a JavaScript program.  An inner call whose arguments
# needed a runtime is stored with the whole program as its input; the program
# is not analyzed, so such a call loads nothing and is no candidate.  Inner
# calls the wrapper resolved are stored, and attributed, as their own rows.
_CODE_MODE_TOOL: Final = "exec_command"

# Shell brace expansion multiplies one token into many paths.  Bounded because
# the token is agent-written text, not a trusted list.
_BRACE_GROUP: Final = re.compile(r"\{([^{}]*,[^{}]*)\}")
_BRACE_EXPANSIONS_MAX: Final = 64

# Flags that turn a read verb into a write.  `sed -i` edits the skill.
_IN_PLACE_FLAGS: Final = ("--in-place", "-i")

# Verbs that read a file whole.  `rg`, `find`, `ls`, and `wc` are deliberately
# absent: they search, enumerate, or measure, which is discovery, not a load.
_READ_VERBS: Final = frozenset({"bat", "cat", "head", "less", "more", "nl", "sed", "tail", "view"})

# Tools whose entire job is reading one file.
_READ_TOOLS: Final = frozenset({"read", "read_file", "view", "readfile"})

_PATH_KEYS: Final = ("path", "file_path", "filePath", "absolute_path", "target_file")

# Shell control operators, as `shlex(punctuation_chars=True)` groups them.
_OPERATOR_CHARS: Final = frozenset("&|;<>()")

_AGENT_PROFILE_SOURCE: Final = re.compile(
    r"(?:^|/)agent-profile/plugins/([^/]+)/skills/([^/]+)/SKILL\.md$"
)

_MAX_CANDIDATE_VALUES: Final = 256


def derive_skill_name(
    tool_name: str,
    tool_input: Any,
    bash_command: str | None,
    *,
    cwd: str | None = None,
) -> str | None:
    """Return the first skill a tool call loaded, or ``None`` if it loaded none.

    The stored ``tool_calls.skill_name`` column holds one name; counting uses
    ``derive_skill_names`` so a multi-skill read is not undercounted.
    """
    names = derive_skill_names(tool_name, tool_input, bash_command, cwd=cwd)
    return names[0] if names else None


def derive_skill_names(
    tool_name: str,
    tool_input: Any,
    bash_command: str | None,
    *,
    cwd: str | None = None,
) -> tuple[str, ...]:
    """Return every skill a tool call loaded, in order, each once.

    ``bash_command`` is passed in already-extracted so this stays a pure
    function of what the caller has parsed.
    """
    if tool_name.lower() == "skill" and isinstance(tool_input, dict):
        skill = tool_input.get("skill")
        if isinstance(skill, str) and skill.strip():
            return (skill.strip(),)
    if bash_command:
        return _from_command(bash_command, cwd=cwd)
    if tool_name.lower() in _READ_TOOLS and isinstance(tool_input, dict):
        for key in _PATH_KEYS:
            value = tool_input.get(key)
            if isinstance(value, str):
                name = _from_path(value, cwd=cwd)
                if name is not None:
                    return (name,)
    return ()


def is_skill_candidate(
    tool_name: str,
    tool_input: Any,
    bash_command: str | None,
) -> bool:
    """Whether a stored tool call warrants skill-attribution diagnostics."""
    if tool_name.lower() == "skill":
        return True
    if bash_command and _SKILL_FILE in bash_command:
        return True
    if _carries_program(tool_name, tool_input, bash_command):
        return False

    pending: list[Any] = [tool_input]
    visited = 0
    while pending and visited < _MAX_CANDIDATE_VALUES:
        value = pending.pop()
        visited += 1
        if isinstance(value, str):
            if _SKILL_FILE in value:
                return True
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    return False


def _from_command(
    command: str,
    *,
    cwd: str | None,
) -> tuple[str, ...]:
    """Find skill loads in a shell command, one pipeline segment at a time.

    Segment scope matters: ``git diff --stat && sed -n 1,5p <installed>`` is
    one authoring act and one load, and only the segment carrying the path
    decides.
    """
    names: dict[str, None] = {}
    for tokens in _segments(command):
        if _verb(tokens) not in _READ_VERBS or _edits_in_place(tokens):
            continue
        for token in tokens[1:]:
            for path in _expand_braces(token):
                name = _from_path(path, cwd=cwd)
                if name is not None:
                    names[name] = None
    return tuple(names)


def _carries_program(tool_name: str, tool_input: Any, bash_command: str | None) -> bool:
    """Whether the call is a code-mode inner call stored with its whole program."""
    return (
        not bash_command
        and tool_name == _CODE_MODE_TOOL
        and isinstance(tool_input, dict)
        and isinstance(tool_input.get("source"), str)
    )


def _expand_braces(token: str) -> list[str]:
    """Expand ``a/{b,c}/d`` the way the shell would, up to a fixed count.

    A token that would expand past the bound is returned unexpanded, and an
    unexpanded brace never names a skill, so an oversized pattern is left
    unattributed rather than partly counted.
    """
    expanded = [token]
    while any(_BRACE_GROUP.search(item) for item in expanded):
        next_round: list[str] = []
        for item in expanded:
            match = _BRACE_GROUP.search(item)
            if match is None:
                next_round.append(item)
                continue
            head, tail = item[: match.start()], item[match.end() :]
            next_round.extend(head + choice + tail for choice in match.group(1).split(","))
        if len(next_round) > _BRACE_EXPANSIONS_MAX:
            return [token]
        expanded = next_round
    return expanded


def _edits_in_place(tokens: list[str]) -> bool:
    """Whether the segment rewrites its target instead of reading it."""
    return any(
        token == "--in-place" or (token.startswith("-i") and not token.startswith("--"))
        for token in tokens[1:]
    )


def _segments(command: str) -> list[list[str]]:
    """Split a command into tokenized segments, respecting shell quoting.

    Quoting is load-bearing here: ``printf 'a | sed -n 1p <installed>'`` runs
    no ``sed`` at all, so a naive split on ``|`` invents a read that never
    happened.  A line whose quoting does not parse is skipped rather than
    guessed at — what ran in it cannot be known, and a wrong attribution is
    worse than a missing one.
    """
    segments: list[list[str]] = []
    for tokens in _token_lines(command):
        current: list[str] = []
        for token in tokens:
            if _is_operator(token):
                if current:
                    segments.append(current)
                    current = []
                continue
            current.append(token)
        if current:
            segments.append(current)
    return segments


def _token_lines(command: str) -> list[list[str]]:
    """Tokenize executable lines while treating heredoc bodies as literal data."""
    token_lines: list[list[str]] = []
    heredocs: list[tuple[str, bool]] = []
    current: list[str] = []
    quote: str | None = None
    at_word_start = True
    for physical_line in command.split("\n"):
        if heredocs:
            delimiter, strip_tabs = heredocs[0]
            candidate = physical_line.lstrip("\t") if strip_tabs else physical_line
            if candidate == delimiter:
                heredocs.pop(0)
            continue

        quote, continued, at_word_start = _append_physical_line(
            physical_line,
            current,
            quote=quote,
            at_word_start=at_word_start,
        )
        if continued:
            continue
        if quote is not None:
            current.append("\n")
            continue
        try:
            tokens = _lex("".join(current))
        except ValueError:
            current = []
            at_word_start = True
            continue
        current = []
        at_word_start = True
        heredocs.extend(_heredoc_delimiters(tokens))
        token_lines.append(tokens)
    return token_lines


def _append_physical_line(
    line: str,
    current: list[str],
    *,
    quote: str | None,
    at_word_start: bool,
) -> tuple[str | None, bool, bool]:
    """Append shell syntax from one physical line, excluding comments/newline."""
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and quote != "'":
            if index + 1 == len(line):
                return quote, True, at_word_start
            escaped = line[index + 1]
            current.extend((char, escaped))
            if quote is None:
                at_word_start = False
            index += 2
            continue
        if char == "#" and quote is None and at_word_start:
            break
        if char in {"'", '"'}:
            if quote is None:
                quote = char
                at_word_start = False
            elif quote == char:
                quote = None
        elif quote is None:
            at_word_start = char.isspace() or char in _OPERATOR_CHARS
        current.append(char)
        index += 1
    return quote, False, at_word_start


def _heredoc_delimiters(tokens: list[str]) -> list[tuple[str, bool]]:
    """Return declared heredocs so their literal bodies are never executed."""
    delimiters: list[tuple[str, bool]] = []
    for index, token in enumerate(tokens[:-1]):
        if token != "<<":
            continue
        delimiter = tokens[index + 1]
        strip_tabs = delimiter.startswith("-")
        if strip_tabs:
            delimiter = delimiter[1:]
        if delimiter:
            delimiters.append((delimiter, strip_tabs))
    return delimiters


def _lex(line: str) -> list[str]:
    """Tokenize one line, keeping shell operators as their own tokens."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _is_operator(token: str) -> bool:
    return bool(token) and set(token) <= _OPERATOR_CHARS


def _verb(tokens: list[str]) -> str | None:
    """The command word, past any ``FOO=bar`` prefix assignments."""
    for token in tokens:
        if "=" in token.split("/")[-1] and not token.startswith("-"):
            continue
        return token.rsplit("/", 1)[-1]
    return None


def _from_path(
    raw: str,
    *,
    cwd: str | None,
) -> str | None:
    path = raw.strip().strip("'\"")
    if not path.endswith("/" + _SKILL_FILE) or "{" in path:
        return None
    home = _HOME_PREFIX.match(path)
    if home is not None and any(path[home.end() :].startswith(root) for root in _INSTALLED_ROOTS):
        name = _name_from_installed_path(path)
    else:
        name = _from_working_copy(path, cwd=cwd)
    # A variable or glob the shell would expand names no skill of its own.
    if name is None or _SHELL_EXPANSION.search(name):
        return None
    return name


def _name_from_installed_path(path: str) -> str | None:
    parts = path.split("/")
    try:
        skills_idx = len(parts) - 1 - parts[::-1].index("skills")
    except ValueError:
        return None
    name_idx = skills_idx + 1
    # Codex keeps its built-ins under `skills/.system/<name>/`.
    if name_idx < len(parts) and parts[name_idx].startswith("."):
        name_idx += 1
    plugin = _plugin(parts, skills_idx)
    # `~/.agents/skills/<plugin>/<name>/` groups a plugin's skills one level down.
    if plugin is None and parts[skills_idx - 1] == ".agents" and name_idx == len(parts) - 3:
        plugin = parts[name_idx] or None
        name_idx += 1
    if name_idx != len(parts) - 2:
        return None
    name = parts[name_idx]
    if not name:
        return None
    return f"{plugin}:{name}" if plugin else name


def _from_working_copy(path: str, *, cwd: str | None) -> str | None:
    """Attribute a skill read from a checkout rather than an installed root.

    A harness can run skills straight from a checkout: Grok from the
    agent-profile source, Codex through a `~/.codex/skills` link into dotfiles,
    and any harness from a repository's own hidden skills directory.  A relative
    path is resolved against the session cwd; without one it cannot be placed.
    """
    if not path.startswith("/"):
        if not cwd or not cwd.startswith("/"):
            return None
        path = join(cwd, path)
    normalized_path = normpath(path)
    match = _AGENT_PROFILE_SOURCE.search(normalized_path)
    if match is not None:
        plugin, skill = match.groups()
        return f"{plugin}:{skill}"
    if _CHECKOUT_SKILL_LAYOUT.search(normalized_path):
        return _name_from_installed_path(normalized_path)
    return None


def _plugin(parts: list[str], skills_idx: int) -> str | None:
    """Recover the plugin that owns the skill, matching Claude's namespacing.

    Three installed layouts carry one:
    ``plugins/cache/<marketplace>/<plugin>/<version>/skills/<name>`` (Claude
    Code and Codex plugin caches), ``plugins/<plugin>/skills/<name>`` (a
    multi-plugin repo), and ``git/<host>/<org>/<repo>/skills/<name>``, where
    Pi Agent clones a single-plugin repo and the repo is the plugin.
    """
    head = parts[:skills_idx]
    if "cache" in head:
        cache_idx = len(head) - 1 - head[::-1].index("cache")
        if skills_idx - cache_idx == 4:
            return parts[skills_idx - 2] or None
    if skills_idx >= 2 and parts[skills_idx - 2] == "plugins":
        return parts[skills_idx - 1] or None
    if skills_idx >= 5 and parts[skills_idx - 4] == "git" and parts[skills_idx - 5] == "agent":
        return parts[skills_idx - 1] or None
    return None
