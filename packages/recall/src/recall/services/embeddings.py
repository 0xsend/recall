"""Embedding generation service."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass

import duckdb

from recall.core.config import EmbeddingConfig
from recall.core.embeddings import BackendDescriptor, EmbeddingBackend, set_backend_probe
from recall.core.models import Message, Session, ToolCall
from recall.core.types import EmbedKind

logger = logging.getLogger("recall.embeddings")

NORMALIZATION_VERSION = 1
CONTEXT_VERSION_BY_MODE = {
    "off": 0,
    "template": 0,
    "llm-local": 1,
    "llm-remote": 2,
    "llm-codex": 3,
}
_GIT_SHA_RE = re.compile(r"\b[0-9a-f]{40}\b")
_URL_NUMERIC_SEGMENT_RE = re.compile(r"(?<=/)\d+(?=(?:/|$))")
_NUMERIC_TOKEN_RE = re.compile(r"\b\d{5,}\b")
_TMP_PATH_RE = re.compile(r"/(?:private/)?tmp(?:/[^\s]+)+")
_VAR_FOLDERS_TMP_RE = re.compile(r"/var/folders/(?:[^\s/]+/)+([^\s/]+)")
_K8S_NAME_RE = re.compile(
    r"(?<!<)\b[a-z0-9](?:[-a-z0-9]*[a-z0-9])?-[a-z0-9]{8,10}-[a-z0-9]{5}\b(?!>)"
)

# ---- Backend registry ----

_REGISTRY: dict[str, BackendDescriptor] = {}


def _register_builtins() -> None:
    def _mlx_available() -> bool:
        from recall.services.mlx_embeddings import MLXBackend

        return MLXBackend.is_available()

    def _mlx_factory(model_id: str) -> EmbeddingBackend:
        from recall.services.mlx_embeddings import MLXBackend

        return MLXBackend(model_id=model_id)

    _REGISTRY["mlx"] = BackendDescriptor(
        name="mlx",
        is_available=_mlx_available,
        factory=_mlx_factory,
        priority=10,
    )

    def _onnx_available() -> bool:
        from recall.services.onnx_embeddings import ONNXBackend

        return ONNXBackend.is_available()

    def _onnx_factory(model_id: str) -> EmbeddingBackend:
        from recall.services.onnx_embeddings import ONNXBackend

        return ONNXBackend(model_id=model_id)

    _REGISTRY["onnx"] = BackendDescriptor(
        name="onnx",
        is_available=_onnx_available,
        factory=_onnx_factory,
        priority=20,
    )


_register_builtins()


def available_backends() -> list[str]:
    """Return names of available backends, sorted by priority (lowest first)."""
    return [
        desc.name
        for desc in sorted(_REGISTRY.values(), key=lambda d: d.priority)
        if desc.is_available()
    ]


def any_backend_available() -> bool:
    """Return whether at least one embedding backend is available."""
    return any(desc.is_available() for desc in _REGISTRY.values())


def resolve_backend_name(configured: str) -> str:
    """Resolve a backend name, expanding ``"auto"`` to the best available.

    Raises ValueError if no backend is available or the name is unknown.
    """
    if configured == "auto":
        backends = available_backends()
        if not backends:
            raise ValueError("no embedding backend available on this host")
        return backends[0]
    if configured not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unsupported embedding backend: {configured}. Available: {valid}, auto")
    return configured


def _probe(backend: str) -> bool:
    if backend == "auto":
        return any_backend_available()
    desc = _REGISTRY.get(backend)
    return desc.is_available() if desc else False


set_backend_probe(_probe)


@dataclass(frozen=True)
class PreparedText:
    kind: EmbedKind
    raw_text: str
    normalized_text: str
    context_text: str
    context_version: int
    cache_key: str


def get_backend(config: EmbeddingConfig) -> EmbeddingBackend:
    """Load the configured embedding backend."""
    name = resolve_backend_name(config.backend)
    desc = _REGISTRY[name]
    return desc.factory(config.model)


def normalize_bash_command(command: str) -> str:
    normalized = command.strip()
    normalized = _GIT_SHA_RE.sub("<git-sha>", normalized)
    normalized = _URL_NUMERIC_SEGMENT_RE.sub("<num>", normalized)
    normalized = _TMP_PATH_RE.sub(_tmp_path_replacement, normalized)
    normalized = _VAR_FOLDERS_TMP_RE.sub(_tmp_path_replacement, normalized)
    normalized = _K8S_NAME_RE.sub("<k8s-name>", normalized)
    normalized = _NUMERIC_TOKEN_RE.sub("<num>", normalized)
    return normalized


def _replace_tmp_path(path: str) -> str:
    basename = path.rsplit("/", 1)[-1]
    if basename and "." in basename:
        return f"<tmp-path>/{basename}"
    return "<tmp-path>"


def _tmp_path_replacement(match: re.Match[str]) -> str:
    return _replace_tmp_path(match.group(0))


def embed_session(
    session: Session,
    backend: EmbeddingBackend,
    batch_size: int = 64,
    conn: duckdb.DuckDBPyConnection | None = None,
    resolved_cache: dict[str, list[float]] | None = None,
    context_text: str = "",
    context_version: int = 0,
) -> None:
    """Populate embedding fields on a session's messages and tool_calls in-place."""
    cache_namespace = _backend_cache_namespace(backend)
    content_entries: list[PreparedText] = []
    content_indices: list[int] = []
    thinking_entries: list[PreparedText] = []
    thinking_indices: list[int] = []
    bash_entries: list[PreparedText] = []
    bash_refs: list[ToolCall] = []

    for i, message in enumerate(session.messages):
        message_context_text, message_context_version = _effective_message_context(
            message,
            fallback_context_text=context_text,
            fallback_context_version=context_version,
        )
        if message.content:
            content_entries.append(
                _prepare_text(
                    EmbedKind.CONTENT,
                    message.content,
                    cache_namespace,
                    context_text=message_context_text,
                    context_version=message_context_version,
                )
            )
            content_indices.append(i)
        if message.thinking:
            thinking_entries.append(
                _prepare_text(
                    EmbedKind.THINKING,
                    message.thinking,
                    cache_namespace,
                    context_text=message_context_text,
                    context_version=message_context_version,
                )
            )
            thinking_indices.append(i)
        for tc in message.tool_calls:
            if tc.bash_command:
                bash_entries.append(_prepare_text(EmbedKind.BASH, tc.bash_command, cache_namespace))
                bash_refs.append(tc)

    for tc in session.orphan_tool_calls:
        if tc.bash_command:
            bash_entries.append(_prepare_text(EmbedKind.BASH, tc.bash_command, cache_namespace))
            bash_refs.append(tc)

    content_embeddings = (
        _resolve_embeddings(backend, content_entries, batch_size, conn, resolved_cache)
        if content_entries
        else []
    )
    thinking_embeddings = (
        _resolve_embeddings(backend, thinking_entries, batch_size, conn, resolved_cache)
        if thinking_entries
        else []
    )
    bash_embeddings = (
        _resolve_embeddings(backend, bash_entries, batch_size, conn, resolved_cache)
        if bash_entries
        else []
    )

    for idx, emb in zip(content_indices, content_embeddings, strict=True):
        session.messages[idx].content_embedding = emb

    for idx, emb in zip(thinking_indices, thinking_embeddings, strict=True):
        session.messages[idx].thinking_embedding = emb

    for tc, emb in zip(bash_refs, bash_embeddings, strict=True):
        tc.bash_embedding = emb


def _effective_message_context(
    message: Message,
    *,
    fallback_context_text: str,
    fallback_context_version: int,
) -> tuple[str, int]:
    if message.context_text or message.context_mode != "off":
        return message.context_text, context_version_for_mode(message.context_mode)
    if fallback_context_text:
        return fallback_context_text, fallback_context_version
    return "", 0


def context_version_for_mode(mode: str) -> int:
    try:
        return CONTEXT_VERSION_BY_MODE[mode]
    except KeyError as err:
        valid = ", ".join(sorted(CONTEXT_VERSION_BY_MODE))
        raise ValueError(f"unsupported context mode: {mode}. Expected one of: {valid}") from err


def _batch_embed(backend: EmbeddingBackend, texts: list[str], batch_size: int) -> list[list[float]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    expected_dims = backend.dimensions
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = backend.embed(batch)
        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                f"embedding backend returned {len(batch_embeddings)} vectors for {len(batch)} texts"
            )
        for embedding in batch_embeddings:
            if len(embedding) != expected_dims:
                raise RuntimeError(
                    f"embedding vector has {len(embedding)} dimensions; expected {expected_dims}"
                )
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def _backend_cache_namespace(backend: EmbeddingBackend) -> str:
    model_id = getattr(backend, "model_id", "")
    backend_type = f"{backend.__class__.__module__}.{backend.__class__.__qualname__}"
    return f"{backend_type}:{model_id}"


def _prepare_text(
    kind: EmbedKind,
    raw_text: str,
    cache_namespace: str,
    context_text: str = "",
    context_version: int = 0,
) -> PreparedText:
    if context_version < 0:
        raise ValueError("context_version must be non-negative")
    if kind == EmbedKind.BASH and context_version != 0:
        raise ValueError("bash embeddings do not support context_version")

    normalized_text = raw_text if kind != EmbedKind.BASH else normalize_bash_command(raw_text)
    cache_key = _embedding_cache_key(
        kind,
        raw_text,
        normalized_text,
        cache_namespace,
        context_text if kind != EmbedKind.BASH else "",
        context_version if kind != EmbedKind.BASH else 0,
    )
    return PreparedText(
        kind=kind,
        raw_text=raw_text,
        normalized_text=normalized_text,
        context_text=context_text if kind != EmbedKind.BASH else "",
        context_version=context_version if kind != EmbedKind.BASH else 0,
        cache_key=cache_key,
    )


def _embedding_cache_key(
    kind: EmbedKind,
    raw_text: str,
    normalized_text: str,
    cache_namespace: str,
    context_text: str,
    context_version: int,
) -> str:
    """Build cache keys without invalidating unchanged off-mode entries.

    Bash keeps the v0.12 key format. CONTENT and THINKING keep that legacy
    format only when context is disabled with version 0; contextual entries add
    a `ctx` component containing the context revision and effective text.
    """
    if kind == EmbedKind.BASH or (not context_text and context_version == 0):
        key_source = normalized_text if kind == EmbedKind.BASH else raw_text
        return hashlib.sha256(
            f"{cache_namespace}:{kind.value}:{NORMALIZATION_VERSION}:{key_source}".encode()
        ).hexdigest()
    embedded_text = f"{context_text}{raw_text}"
    return hashlib.sha256(
        (
            f"{cache_namespace}:{kind.value}:{NORMALIZATION_VERSION}:"
            f"ctx:{context_version}:{embedded_text}"
        ).encode()
    ).hexdigest()


def _resolve_embeddings(
    backend: EmbeddingBackend,
    entries: list[PreparedText],
    batch_size: int,
    conn: duckdb.DuckDBPyConnection | None,
    resolved_cache: dict[str, list[float]] | None = None,
) -> list[list[float]]:
    dimensions = backend.dimensions
    unique_entries = list(OrderedDict((entry.cache_key, entry) for entry in entries).values())
    embeddings_by_key: dict[str, list[float]] = {}
    if resolved_cache is not None:
        for entry in unique_entries:
            embedding = resolved_cache.get(entry.cache_key)
            if embedding is not None:
                embeddings_by_key[entry.cache_key] = embedding
    if conn is not None:
        cached_rows = _fetch_cached_embeddings(
            conn,
            [entry for entry in unique_entries if entry.cache_key not in embeddings_by_key],
            dimensions,
        )
        embeddings_by_key.update(cached_rows)
        if resolved_cache is not None:
            resolved_cache.update(cached_rows)

    missing_entries = [
        entry for entry in unique_entries if entry.cache_key not in embeddings_by_key
    ]
    if missing_entries:
        new_embeddings = _batch_embed(
            backend,
            [
                entry.normalized_text
                if entry.kind == EmbedKind.BASH
                else f"{entry.context_text}{entry.raw_text}"
                for entry in missing_entries
            ],
            batch_size,
        )
        for entry, embedding in zip(missing_entries, new_embeddings, strict=True):
            embeddings_by_key[entry.cache_key] = embedding
            if resolved_cache is not None:
                resolved_cache[entry.cache_key] = embedding
        if conn is not None:
            _store_cached_embeddings(conn, missing_entries, embeddings_by_key)

    return [embeddings_by_key[entry.cache_key] for entry in entries]


def _fetch_cached_embeddings(
    conn: duckdb.DuckDBPyConnection, entries: list[PreparedText], dimensions: int
) -> dict[str, list[float]]:
    if not entries:
        return {}
    placeholders = ",".join(["?"] * len(entries))
    rows = conn.execute(
        f"""
        SELECT cache_key, embedding
        FROM embedding_cache
        WHERE cache_key IN ({placeholders})
        """,
        [entry.cache_key for entry in entries],
    ).fetchall()
    return {str(row[0]): list(row[1]) for row in rows}


def _store_cached_embeddings(
    conn: duckdb.DuckDBPyConnection,
    entries: list[PreparedText],
    embeddings_by_key: dict[str, list[float]],
) -> None:
    if not entries:
        return
    rows = [
        (
            entry.cache_key,
            entry.kind.value,
            entry.raw_text,
            entry.normalized_text,
            embeddings_by_key[entry.cache_key],
            NORMALIZATION_VERSION,
            entry.context_version,
        )
        for entry in entries
    ]
    conn.executemany(
        """
        INSERT OR IGNORE INTO embedding_cache (
            cache_key, kind, raw_text, normalized_text, embedding,
            normalization_version, context_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
