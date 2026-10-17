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
