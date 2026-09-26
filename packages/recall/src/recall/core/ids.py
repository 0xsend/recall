from __future__ import annotations

import hashlib


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def session_id(source: str, absolute_path: str) -> str:
    return _sha256_hex(f"{source}:{absolute_path}")[:32]


def message_id(session_id_value: str, idx: int) -> str:
    return _sha256_hex(f"{session_id_value}:{idx}")[:32]


def tool_call_id(
    message_id_value: str | None, idx: int, session_id_value: str | None = None
) -> str:
    if message_id_value is not None:
        return _sha256_hex(f"{message_id_value}:{idx}")[:32]
    if session_id_value is None:
        raise ValueError("session_id is required for orphan tool calls")
    return _sha256_hex(f"{session_id_value}:orphan:{idx}")[:32]


_TRANSCRIPT_DIRS = ("projects", "sessions")


def transcript_key(source_path: str) -> str:
    """The part of a transcript path that survives renaming its project directory.

    Harnesses file a transcript under a directory named for the session's
    project or cwd (`projects/<dir>/`, `sessions/<dir>/`); the path below it
    names the transcript itself.  Renaming the project renames only `<dir>`.
    A path without such a directory is its own key.
    """
    parts = source_path.split("/")
    for index in range(len(parts) - 3, -1, -1):
        if parts[index] in _TRANSCRIPT_DIRS:
            return "/".join(parts[index + 2 :])
    return source_path
