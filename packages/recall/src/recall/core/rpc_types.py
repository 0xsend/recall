"""Shared JSON-RPC 2.0 types and error codes used by both client and server."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from recall.core.config import AppConfig

# JSON-RPC 2.0 standard error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Application error codes
APP_LOCKED = -32000
APP_CONFIRMATION_REQUIRED = -32001
APP_NOT_FOUND = -32002


@dataclass(frozen=True)
class RpcError(Exception):
    code: int
    message: str
    data: Any = None

    # A dataclass exception never populates `BaseException.args`, so the
    # inherited `__str__` renders as "". A caller that logs or records `str(err)`
    # would report the failure with no reason at all (REQ-RESIL-026).
    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class RpcConnectionError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class RpcCallError(Exception):
    code: int
    message: str
    data: Any = None

    def __str__(self) -> str:
        return self.message


def config_fingerprint(config: AppConfig) -> str:
    """Hash of embedding config fields relevant to model staleness (REQ-RPC-010)."""
    context = config.embedding.context
    # base_url participates because routing through a different endpoint (e.g. LiteLLM
    # in front of a local Llama vs. Anthropic-direct Haiku) produces different prefixes;
    # api_key participates for the same reason: against the same base_url, a different
    # key can land on a different LiteLLM tenant/model_group, so prefixes diverge.
    # instruction_prefix changes the generation directly (e.g. '/no_think' suppresses
    # Qwen3's reasoning preamble), so it participates too.
    # reasoning_effort changes the llm-codex backend's invocation and the model's
    # output behavior (low vs. high produce different prefixes); executable
    # participates because pointing the backend at a different codex binary can mean
    # a different CLI version with a different bundled system prompt — same model
    # name, different generated text. Both are stored as plain strings.
    # timeout does not affect output content so it stays out of the fingerprint.
    # Note: api_key is hashed (sha256-truncated), never persisted as plaintext here.
    key = (
        f"{config.embedding.backend}:{config.embedding.model}:"
        f"{config.embedding.dimensions}:{context.mode}:"
        f"{context.fallback}:{context.model}:{context.max_tokens}:"
        f"{context.batch_size}:{context.min_chars}:{context.base_url or ''}:"
        f"{context.api_key or ''}:{context.instruction_prefix or ''}:"
        f"{context.reasoning_effort or ''}:{context.executable}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def serialize_rpc_value(value: Any) -> Any:
    """Serialize a Python value for JSON-RPC transport."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    try:
        from pydantic import BaseModel

        if isinstance(value, BaseModel):
            return serialize_rpc_value(value.model_dump(mode="python"))
    except ImportError:
        pass
    if is_dataclass(value) and not isinstance(value, type):
        return serialize_rpc_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): serialize_rpc_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_rpc_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)
