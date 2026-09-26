from __future__ import annotations

import os
import shutil
import sys
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from recall.core.embeddings import embedding_backend_available
from recall.core.types import (
    DaemonMode,
    SchedulerKind,
    Source,
    parse_daemon_mode,
    parse_scheduler_kind,
    parse_source,
)

DEFAULT_FTS_FIELDS = ("content", "thinking", "bash")
VALID_FTS_FIELDS = {"content", "thinking", "bash"}
DEFAULT_FTS_BACKEND = "sqlite_sidecar"
VALID_FTS_BACKENDS = {"duckdb", "sqlite_sidecar"}

DEFAULT_EMBED_BACKEND = "auto"
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CONTEXT_MODE = "off"
DEFAULT_CONTEXT_FALLBACK = "template"
DEFAULT_CONTEXT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_CODEX_CONTEXT_MODEL = "gpt-5.4-mini"
DEFAULT_CONTEXT_MAX_TOKENS = 120
DEFAULT_CONTEXT_BATCH_SIZE = 8
DEFAULT_CONTEXT_CONCURRENCY = 4
DEFAULT_CONTEXT_MIN_CHARS = 50
DEFAULT_CONTEXT_MAX_DOCUMENT_CHARS = 400000
VALID_CONTEXT_MODES = {"off", "template", "llm-local", "llm-remote", "llm-codex"}
VALID_CONTEXT_FALLBACKS = {"template", "off", "error"}
# Codex models accept different subsets of these values. Keep the client-side
# superset model-agnostic and let codex exec surface model-specific errors.
VALID_CONTEXT_REASONING_EFFORTS = {
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
}
DEFAULT_CONTEXT_EXECUTABLE = "codex"
_EMBED_BATCH_FLOOR = 64
_EMBED_BATCH_CEILING = 1024
_EMBED_BATCH_PER_GB = 8


def _default_embed_batch_size() -> int:
    """Return an embedding batch size scaled to the host's RAM.

    Uses 8 per GB of RAM, clamped to [64, 1024].
    """
    try:
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        total_gb = total_bytes / (1024**3)
    except (ValueError, OSError):
        return _EMBED_BATCH_FLOOR
    return int(max(_EMBED_BATCH_FLOOR, min(total_gb * _EMBED_BATCH_PER_GB, _EMBED_BATCH_CEILING)))


DEFAULT_EMBED_BATCH_SIZE = _default_embed_batch_size()

KNOWN_MODEL_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "nomic-ai/nomic-embed-text-v1.5": 768,
}

DEFAULT_DAEMON_INTERVAL = 300
DEFAULT_DAEMON_EMBED = False
DEFAULT_DAEMON_SCHEDULER = SchedulerKind.AUTO
DEFAULT_DAEMON_MODE = DaemonMode.AUTO
DEFAULT_DAEMON_DEBOUNCE = 5
DEFAULT_DAEMON_FTS_DEBOUNCE = 10
DEFAULT_DAEMON_IDLE_TIMEOUT = 1800
DEFAULT_DAEMON_EMBED_INTERVAL = 120
DEFAULT_DAEMON_EMBED_BACKOFF = 300
DEFAULT_DAEMON_EMBED_IDLE_SESSION = 600
DEFAULT_DAEMON_EMBED_MODEL_TIMEOUT = 600
DEFAULT_DAEMON_LIVE_IDLE_THRESHOLD = 300
DEFAULT_DAEMON_LIVE_DISCOVERY_INTERVAL = 30
DEFAULT_DAEMON_LIVE_MAX_SUBSCRIPTIONS = 64
DEFAULT_DAEMON_LOAD_THRESHOLD = 0.7
DEFAULT_DAEMON_BATTERY_THRESHOLD = 0.3
DEFAULT_DAEMON_LOG_MAX_BYTES = 52_428_800
# The daemon's own logger level. INFO because the embed/index phase observability
# lines the SPEC promises (`embed check`, `embed skipped`, `embed batch`) are INFO
# records, and a long-lived daemon whose only diagnostic artifact is its log file
# must emit them without an operator having to restart it in a verbose mode.
DEFAULT_DAEMON_LOG_LEVEL = "info"
DAEMON_LOG_LEVELS = ("debug", "info", "warning", "error", "critical")
DEFAULT_CLI_STATUS_NOTICES = True
DEFAULT_COMPACTION_AUTO_TRIGGER = True
# Compaction is preventive: a database that has reached 2x is already failing.
# One 35k-session host could not complete `index --full` at all while reading
# 1.906 against the former 2.0 default, so auto-compaction never fired; a
# compacted copy of the same data ran clean. 1.5 sits clear of that failure
# point and well above a freshly compacted database (~1.15).
DEFAULT_COMPACTION_BLOAT_THRESHOLD = 1.5
DEFAULT_COMPACTION_CHECK_INTERVAL_HOURS = 6
# Below this size the ratio is dominated by block granularity, not by dead
# rows: a freshly built database of a few MB reads ~1.65 against a 256 KiB
# block size purely because partially-filled blocks round up. Compacting that
# reclaims nothing. Real churn only becomes measurable once the file is large
# enough for block overhead to wash out.
DEFAULT_COMPACTION_MIN_BYTES = 256 * 1024 * 1024
MIN_COMPACTION_CHECK_INTERVAL_HOURS = 1


def _to_int(value: object) -> int:
    """Coerce a TOML config value to int with type narrowing for ty."""
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str)):
        return int(value)
    raise TypeError(f"cannot convert {type(value).__name__} to int")


def _to_float(value: object) -> float:
    """Coerce a TOML config value to float with type narrowing for ty."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"cannot convert {type(value).__name__} to float")


@dataclass(frozen=True)
class FtsConfig:
    fields: tuple[str, ...] = DEFAULT_FTS_FIELDS
    backend: str = DEFAULT_FTS_BACKEND

    @classmethod
    def from_values(cls, values: Iterable[str] | None) -> FtsConfig:
        return cls(fields=_parse_fts_fields(values))

    @classmethod
    def from_toml(cls, values: dict[str, object] | None) -> FtsConfig:
        values = values or {}
        fields_raw = values.get("fields")
        fields = (
            _parse_fts_fields([str(field) for field in fields_raw])
            if isinstance(fields_raw, list)
            else DEFAULT_FTS_FIELDS
        )
        backend = str(values.get("backend", DEFAULT_FTS_BACKEND)).strip().lower()
        if backend not in VALID_FTS_BACKENDS:
            raise ValueError("fts backend must be one of: duckdb, sqlite_sidecar")
        return cls(fields=fields, backend=backend)


def _parse_fts_fields(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return DEFAULT_FTS_FIELDS
    normalized = tuple(field.strip() for field in values if field.strip())
    if not normalized:
        return ()
    invalid = [field for field in normalized if field not in VALID_FTS_FIELDS]
    if invalid:
        raise ValueError(f"invalid FTS fields: {', '.join(invalid)}")
    return normalized


@dataclass(frozen=True)
class ContextConfig:
    mode: str = DEFAULT_CONTEXT_MODE
    # fallback governs only *transient per-message generation failures* for an
    # otherwise-available llm-* backend (template | off | error). A configured
    # llm-* mode whose backend is unsupported on this host (missing extra/key,
    # model load failure) is ALWAYS fatal at startup regardless of fallback — it
    # is a misconfiguration, not a degrade-able runtime error (REQ-CTX-020).
    fallback: str = DEFAULT_CONTEXT_FALLBACK
    model: str = DEFAULT_CONTEXT_MODEL
    max_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS
    batch_size: int = DEFAULT_CONTEXT_BATCH_SIZE
    # concurrency bounds only concurrent LLM/subprocess context generation.
    # DuckDB writes remain serialized by the daemon write lock.
    concurrency: int = DEFAULT_CONTEXT_CONCURRENCY
    min_chars: int = DEFAULT_CONTEXT_MIN_CHARS
    # base_url overrides the Anthropic SDK's default endpoint so `llm-remote`
    # can target Anthropic-compatible proxies (LiteLLM, llama.cpp, etc.).
    # When None, the SDK reads ANTHROPIC_BASE_URL or its built-in default.
    base_url: str | None = None
    # Request timeout (seconds) forwarded to the SDK. When None, the SDK default
    # (600s) applies. Useful to shorten when targeting a local endpoint.
    timeout: float | None = None
    # api_key overrides the Anthropic SDK's env-based key resolution. When None,
    # the SDK reads ANTHROPIC_API_KEY from the environment. Useful when the daemon
    # runs under launchd/systemd where shell env vars do not propagate; the user
    # accepts the trade-off that the value is persisted in config.toml.
    api_key: str | None = None
    # instruction_prefix is prepended to the per-chunk user message sent to the LLM.
    # Designed for hints like Qwen3's `/no_think` directive that disable a model's
    # thinking-mode preamble. Empty/None means no prefix, which matches historical
    # behaviour. Independent from any thinking-tag stripping the backend may apply.
    instruction_prefix: str | None = None
    # max_document_chars caps the rendered session document before it is sent to
    # an LLM backend. None preserves historical unbounded prompting.
    max_document_chars: int | None = DEFAULT_CONTEXT_MAX_DOCUMENT_CHARS
    # reasoning_effort controls how many reasoning tokens a thinking-capable model
    # spends per call. Currently consumed by the llm-codex backend only: forwarded
    # as `-c model_reasoning_effort=...`. Unset means "let the CLI pick its default".
    # Support varies by model: older models may accept minimal, while GPT-5.6 adds
    # max and selected variants add ultra. We only validate the known value shape;
    # codex exec surfaces model-specific incompatibility at runtime.
    reasoning_effort: str | None = None
    # executable overrides the codex CLI binary for the llm-codex backend. Defaults
    # to plain "codex" (resolved via PATH). Tests use a fixture script here. Other
    # backends ignore it.
    executable: str = DEFAULT_CONTEXT_EXECUTABLE

    @classmethod
    def from_values(cls, values: dict[str, object] | None) -> ContextConfig:
        values = values or {}
        mode = str(values.get("mode", DEFAULT_CONTEXT_MODE)).strip()
        if mode not in VALID_CONTEXT_MODES:
            valid = ", ".join(sorted(VALID_CONTEXT_MODES))
            raise ValueError(f"invalid context mode: {mode}. Expected one of: {valid}")
        fallback = str(values.get("fallback", DEFAULT_CONTEXT_FALLBACK)).strip()
        if fallback not in VALID_CONTEXT_FALLBACKS:
            valid = ", ".join(sorted(VALID_CONTEXT_FALLBACKS))
            raise ValueError(f"invalid context fallback: {fallback}. Expected one of: {valid}")
        model = str(values.get("model", DEFAULT_CONTEXT_MODEL)).strip()
        if not model:
            raise ValueError("context model must not be empty")
        max_tokens = _to_int(values.get("max_tokens", DEFAULT_CONTEXT_MAX_TOKENS))
        if max_tokens <= 0:
            raise ValueError("context max_tokens must be positive")
        batch_size = _to_int(values.get("batch_size", DEFAULT_CONTEXT_BATCH_SIZE))
        if batch_size <= 0:
            raise ValueError("context batch_size must be positive")
        concurrency = _to_int(values.get("concurrency", DEFAULT_CONTEXT_CONCURRENCY))
        if concurrency <= 0:
            raise ValueError("context concurrency must be positive")
        min_chars = _to_int(values.get("min_chars", DEFAULT_CONTEXT_MIN_CHARS))
        if min_chars < 0:
            raise ValueError("context min_chars must be non-negative")
        base_url_raw = values.get("base_url")
        base_url: str | None = None
        if base_url_raw is not None:
            # An explicit empty string in config is almost certainly a mistake;
            # reject loudly rather than silently fall back to the SDK default.
            candidate = str(base_url_raw).strip()
            if not candidate:
                raise ValueError("context base_url must not be empty when set")
            if not candidate.startswith(("http://", "https://")):
                raise ValueError(
                    f"context base_url must start with http:// or https://, got {candidate!r}"
                )
            base_url = candidate
        timeout_raw = values.get("timeout")
        timeout: float | None = None
        if timeout_raw is not None:
            timeout = _to_float(timeout_raw)
            if timeout <= 0:
                raise ValueError("context timeout must be positive")
        api_key_raw = values.get("api_key")
        api_key: str | None = None
        if api_key_raw is not None:
            # Reject whitespace-only strings the same way base_url does — an empty
            # value in config is almost certainly a typo, not "intentionally unset".
            candidate = str(api_key_raw).strip()
            if not candidate:
                raise ValueError("context api_key must not be empty when set")
            api_key = candidate
        instruction_prefix_raw = values.get("instruction_prefix")
        instruction_prefix: str | None = None
        if instruction_prefix_raw is not None:
            # Preserve trailing whitespace (callers commonly want '/no_think\n\n'),
            # but reject an entirely-blank value — almost certainly a typo.
            candidate = str(instruction_prefix_raw)
            if not candidate.strip():
                raise ValueError("context instruction_prefix must not be empty when set")
            instruction_prefix = candidate
        missing = object()
        max_document_chars_raw = values.get("max_document_chars", missing)
        max_document_chars: int | None = DEFAULT_CONTEXT_MAX_DOCUMENT_CHARS
        if max_document_chars_raw is None:
            max_document_chars = None
        elif max_document_chars_raw is not missing:
            if isinstance(max_document_chars_raw, bool):
                raise ValueError("context max_document_chars must be a positive integer or null")
            if isinstance(max_document_chars_raw, int):
                max_document_chars = max_document_chars_raw
            elif isinstance(max_document_chars_raw, str):
                try:
                    max_document_chars = int(max_document_chars_raw)
                except ValueError as err:
                    raise ValueError(
                        "context max_document_chars must be a positive integer or null"
                    ) from err
            else:
                raise ValueError("context max_document_chars must be a positive integer or null")
            if max_document_chars <= 0:
                raise ValueError("context max_document_chars must be a positive integer or null")
        reasoning_effort_raw = values.get("reasoning_effort")
        reasoning_effort: str | None = None
        if reasoning_effort_raw is not None:
            candidate = str(reasoning_effort_raw).strip().lower()
            if not candidate:
                raise ValueError("context reasoning_effort must not be empty when set")
            if candidate not in VALID_CONTEXT_REASONING_EFFORTS:
                valid = ", ".join(sorted(VALID_CONTEXT_REASONING_EFFORTS))
                raise ValueError(
                    f"invalid context reasoning_effort: {candidate}. Expected one of: {valid}"
                )
            reasoning_effort = candidate
        # executable defaults to "codex"; reject blank explicitly so misconfig surfaces
        # at config-load time, not later when subprocess execvpe fails opaquely.
        executable_raw = values.get("executable", DEFAULT_CONTEXT_EXECUTABLE)
        executable_candidate = str(executable_raw).strip()
        if not executable_candidate:
            raise ValueError("context executable must not be empty")
        executable = executable_candidate
        return cls(
            mode=mode,
            fallback=fallback,
            model=model,
            max_tokens=max_tokens,
            batch_size=batch_size,
            concurrency=concurrency,
            min_chars=min_chars,
            base_url=base_url,
            timeout=timeout,
            api_key=api_key,
            instruction_prefix=instruction_prefix,
            max_document_chars=max_document_chars,
            reasoning_effort=reasoning_effort,
            executable=executable,
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str = DEFAULT_EMBED_BACKEND
    model: str = DEFAULT_EMBED_MODEL
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    dimensions: int = KNOWN_MODEL_DIMENSIONS[DEFAULT_EMBED_MODEL]
    context: ContextConfig = field(default_factory=ContextConfig)

    @classmethod
    def from_values(cls, values: dict[str, object] | None) -> EmbeddingConfig:
        if values is None:
            return cls()
        context_raw = values.get("context")
        context_values = cast(
            "dict[str, object] | None",
            context_raw if isinstance(context_raw, dict) else None,
        )
        context = ContextConfig.from_values(context_values)
        backend = str(values.get("backend", DEFAULT_EMBED_BACKEND)).strip().lower()
        if not backend:
            raise ValueError("embedding backend must not be empty")
        model = str(values.get("model", DEFAULT_EMBED_MODEL)).strip()
        if not model:
            raise ValueError("embedding model must not be empty")
        batch_size = _to_int(values.get("batch_size", DEFAULT_EMBED_BATCH_SIZE))
        if batch_size <= 0:
            raise ValueError("embedding batch_size must be positive")
        dimensions_raw = values.get("dimensions")
        if dimensions_raw is not None:
            dimensions = _to_int(dimensions_raw)
            if dimensions <= 0:
                raise ValueError("embedding dimensions must be positive")
        else:
            resolved = KNOWN_MODEL_DIMENSIONS.get(model)
            if resolved is None:
                raise ValueError(
                    f"unknown embedding dimensions for model '{model}'. "
                    "Set dimensions explicitly in [embedding] config."
                )
            dimensions = resolved
        return cls(
            backend=backend,
            model=model,
            batch_size=batch_size,
            dimensions=dimensions,
            context=context,
        )


@dataclass(frozen=True)
class CompactionConfig:
    auto_trigger: bool = DEFAULT_COMPACTION_AUTO_TRIGGER
    bloat_ratio_threshold: float = DEFAULT_COMPACTION_BLOAT_THRESHOLD
    check_interval_hours: int = DEFAULT_COMPACTION_CHECK_INTERVAL_HOURS
    min_bytes: int = DEFAULT_COMPACTION_MIN_BYTES

    @classmethod
    def from_values(cls, values: dict[str, object] | None) -> CompactionConfig:
        if values is None:
            return cls()
        auto_value = _parse_optional_bool(values.get("auto_trigger"), "compaction auto_trigger")
        auto_trigger = DEFAULT_COMPACTION_AUTO_TRIGGER if auto_value is None else auto_value
        bloat_ratio_threshold = _to_float(
            values.get("bloat_ratio_threshold", DEFAULT_COMPACTION_BLOAT_THRESHOLD)
        )
        if bloat_ratio_threshold < 0:
            raise ValueError("compaction bloat_ratio_threshold must be non-negative")
        check_interval_hours = _to_int(
            values.get("check_interval_hours", DEFAULT_COMPACTION_CHECK_INTERVAL_HOURS)
        )
        if check_interval_hours < MIN_COMPACTION_CHECK_INTERVAL_HOURS:
            raise ValueError(
                "compaction check_interval_hours must be at least "
                f"{MIN_COMPACTION_CHECK_INTERVAL_HOURS}"
            )
        min_bytes = _to_int(values.get("min_bytes", DEFAULT_COMPACTION_MIN_BYTES))
        if min_bytes < 0:
            raise ValueError("compaction min_bytes must be non-negative")
        return cls(
            auto_trigger=auto_trigger,
            bloat_ratio_threshold=bloat_ratio_threshold,
            check_interval_hours=check_interval_hours,
            min_bytes=min_bytes,
        )


@dataclass(frozen=True)
class DuckDBConfig:
    memory_limit: str | None = None
    temp_directory: str | None = None

    @classmethod
    def from_values(cls, values: dict[str, object] | None) -> DuckDBConfig:
        if values is None:
            return cls()
        memory_limit_raw = values.get("memory_limit")
        memory_limit = str(memory_limit_raw).strip() if memory_limit_raw is not None else None
        temp_directory_raw = values.get("temp_directory")
        temp_directory: str | None = None
        if temp_directory_raw is not None:
            temp_directory = str(temp_directory_raw).strip()
            if not temp_directory:
                raise ValueError("[duckdb] temp_directory must not be empty")
        return cls(memory_limit=memory_limit, temp_directory=temp_directory)


@dataclass(frozen=True)
class DaemonConfig:
    interval: int = DEFAULT_DAEMON_INTERVAL
    embed: bool = DEFAULT_DAEMON_EMBED
    source: Source | None = None
    scheduler: SchedulerKind = DEFAULT_DAEMON_SCHEDULER
    mode: DaemonMode = DEFAULT_DAEMON_MODE
    debounce: int = DEFAULT_DAEMON_DEBOUNCE
    fts_debounce: int = DEFAULT_DAEMON_FTS_DEBOUNCE
    idle_timeout: int = DEFAULT_DAEMON_IDLE_TIMEOUT
    embed_interval: int = DEFAULT_DAEMON_EMBED_INTERVAL
    embed_backoff: int = DEFAULT_DAEMON_EMBED_BACKOFF
    embed_idle_session: int = DEFAULT_DAEMON_EMBED_IDLE_SESSION
    embed_model_timeout: int = DEFAULT_DAEMON_EMBED_MODEL_TIMEOUT
    live_idle_threshold: int = DEFAULT_DAEMON_LIVE_IDLE_THRESHOLD
    live_discovery_interval: int = DEFAULT_DAEMON_LIVE_DISCOVERY_INTERVAL
    live_max_subscriptions: int = DEFAULT_DAEMON_LIVE_MAX_SUBSCRIPTIONS
    load_threshold: float = DEFAULT_DAEMON_LOAD_THRESHOLD
    battery_threshold: float = DEFAULT_DAEMON_BATTERY_THRESHOLD
    log_max_bytes: int = DEFAULT_DAEMON_LOG_MAX_BYTES
    log_level: str = DEFAULT_DAEMON_LOG_LEVEL

    @classmethod
    def from_values(
        cls,
        values: dict[str, object] | None,
        *,
        default_embed: bool = DEFAULT_DAEMON_EMBED,
    ) -> DaemonConfig:
        if values is None:
            return cls(embed=default_embed)
        interval = _to_int(values.get("interval", DEFAULT_DAEMON_INTERVAL))
        if interval <= 0:
            raise ValueError("daemon interval must be positive")
        embed_value = _parse_optional_bool(values.get("embed"), "daemon embed")
        embed = default_embed if embed_value is None else embed_value
        source_value = values.get("source")
        source = parse_source(str(source_value)) if source_value is not None else None
        scheduler_value = values.get("scheduler")
        scheduler = (
            parse_scheduler_kind(str(scheduler_value))
            if scheduler_value is not None
            else DEFAULT_DAEMON_SCHEDULER
        )
        mode_value = values.get("mode")
        mode = parse_daemon_mode(str(mode_value)) if mode_value is not None else DEFAULT_DAEMON_MODE
        debounce = _to_int(values.get("debounce", DEFAULT_DAEMON_DEBOUNCE))
        if debounce <= 0:
            raise ValueError("daemon debounce must be positive")
        fts_debounce = _to_int(values.get("fts_debounce", DEFAULT_DAEMON_FTS_DEBOUNCE))
        if fts_debounce <= 0:
            raise ValueError("daemon fts_debounce must be positive")
        idle_timeout = _to_int(values.get("idle_timeout", DEFAULT_DAEMON_IDLE_TIMEOUT))
        if idle_timeout <= 0:
            raise ValueError("daemon idle_timeout must be positive")
        embed_interval = _to_int(values.get("embed_interval", DEFAULT_DAEMON_EMBED_INTERVAL))
        if embed_interval <= 0:
            raise ValueError("embed_interval must be positive")
        embed_backoff = _to_int(values.get("embed_backoff", DEFAULT_DAEMON_EMBED_BACKOFF))
        if embed_backoff <= 0:
            raise ValueError("embed_backoff must be positive")
        embed_idle_session = _to_int(
            values.get("embed_idle_session", DEFAULT_DAEMON_EMBED_IDLE_SESSION)
        )
        if embed_idle_session <= 0:
            raise ValueError("embed_idle_session must be positive")
        embed_model_timeout = _to_int(
            values.get("embed_model_timeout", DEFAULT_DAEMON_EMBED_MODEL_TIMEOUT)
        )
        if embed_model_timeout <= 0:
            raise ValueError("embed_model_timeout must be positive")
        live_idle_threshold = _to_int(
            values.get("live_idle_threshold", DEFAULT_DAEMON_LIVE_IDLE_THRESHOLD)
        )
        if live_idle_threshold <= 0:
            raise ValueError("daemon live_idle_threshold must be positive")
        live_discovery_interval = _to_int(
            values.get("live_discovery_interval", DEFAULT_DAEMON_LIVE_DISCOVERY_INTERVAL)
        )
        if live_discovery_interval <= 0:
            raise ValueError("daemon live_discovery_interval must be positive")
        live_max_subscriptions = _to_int(
            values.get("live_max_subscriptions", DEFAULT_DAEMON_LIVE_MAX_SUBSCRIPTIONS)
        )
        if live_max_subscriptions <= 0:
            raise ValueError("daemon live_max_subscriptions must be positive")
        load_threshold = _to_float(values.get("load_threshold", DEFAULT_DAEMON_LOAD_THRESHOLD))
        if load_threshold <= 0:
            raise ValueError("daemon load_threshold must be positive")
        battery_threshold = _to_float(
            values.get("battery_threshold", DEFAULT_DAEMON_BATTERY_THRESHOLD)
        )
        if not (0 < battery_threshold <= 1):
            raise ValueError("battery_threshold must be between 0 and 1 (exclusive/inclusive)")
        log_max_bytes = _to_int(values.get("log_max_bytes", DEFAULT_DAEMON_LOG_MAX_BYTES))
        if log_max_bytes < 0:
            raise ValueError("daemon log_max_bytes must be non-negative")
        log_level = str(values.get("log_level", DEFAULT_DAEMON_LOG_LEVEL)).strip().lower()
        if log_level not in DAEMON_LOG_LEVELS:
            raise ValueError(f"daemon log_level must be one of {', '.join(DAEMON_LOG_LEVELS)}")
        return cls(
            interval=interval,
            embed=embed,
            source=source,
            scheduler=scheduler,
            mode=mode,
            debounce=debounce,
            fts_debounce=fts_debounce,
            idle_timeout=idle_timeout,
            embed_interval=embed_interval,
            embed_backoff=embed_backoff,
            embed_idle_session=embed_idle_session,
            embed_model_timeout=embed_model_timeout,
            live_idle_threshold=live_idle_threshold,
            live_discovery_interval=live_discovery_interval,
            live_max_subscriptions=live_max_subscriptions,
            load_threshold=load_threshold,
            battery_threshold=battery_threshold,
            log_max_bytes=log_max_bytes,
            log_level=log_level,
        )


@dataclass(frozen=True)
class CliConfig:
    status_notices: bool = DEFAULT_CLI_STATUS_NOTICES

    @classmethod
    def from_values(cls, values: dict[str, object] | None) -> CliConfig:
        if values is None:
            return cls()
        status_notices = _parse_bool(
            values.get("status_notices", DEFAULT_CLI_STATUS_NOTICES),
            "cli status_notices",
        )
        return cls(status_notices=status_notices)


# 24 h. `recall live --all` calls a session idle inside this window and
# `unknown` outside it — long enough that yesterday's session is still
# recognizable, short enough that a year of history is not "recently active".
DEFAULT_LIVE_IDLE_WINDOW = 86400
DEFAULT_LIVE_FRESH_TIMEOUT = 10.0
# Matches BRIEF responsive-read floor for `live` (including `live --fresh`).
# `show --fresh` uses DEFAULT_LIVE_FRESH_TIMEOUT instead.
DEFAULT_LIVE_ROSTER_FRESH_BUDGET = 2.0


@dataclass(frozen=True)
class LiveConfig:
    """`[live]` — the two windows the live surface reads against.

    ``idle_window`` is how far back a session still counts as idle
    (REQ-LIVE-001). ``fresh_timeout`` bounds how long ``show --fresh`` may
    hold an answer while the daemon catches that one transcript up
    (REQ-LIVE-010); past it the answer returns anyway, reporting itself stale
    rather than hanging. ``live --fresh`` uses the lesser of this value and
    ``DEFAULT_LIVE_ROSTER_FRESH_BUDGET``.
    """

    idle_window: int = DEFAULT_LIVE_IDLE_WINDOW
    fresh_timeout: float = DEFAULT_LIVE_FRESH_TIMEOUT

    @classmethod
    def from_values(cls, values: dict[str, object] | None) -> LiveConfig:
        if not values:
            return cls()
        idle_window = _to_int(values.get("idle_window", DEFAULT_LIVE_IDLE_WINDOW))
        if idle_window <= 0:
            raise ValueError("live idle_window must be positive")
        fresh_timeout = _to_float(values.get("fresh_timeout", DEFAULT_LIVE_FRESH_TIMEOUT))
        if fresh_timeout <= 0:
            raise ValueError("live fresh_timeout must be positive")
        return cls(idle_window=idle_window, fresh_timeout=fresh_timeout)


@dataclass(frozen=True)
class SourceConfig:
    """Where one harness's transcripts live (REQ-LIVE-012).

    A configured list *replaces* the parser's built-in location rather than
    extending it, so an operator who moves a harness's home stops paying to
    scan the old one. `None` means unconfigured — use the built-in location;
    an explicitly empty list means this host scans nothing for this source,
    which is the only way to say "recall, ignore this harness".
    """

    roots: tuple[Path, ...] | None = None

    @classmethod
    def from_values(cls, source: str, values: dict[str, object] | None) -> SourceConfig:
        if not values or "roots" not in values:
            return cls()
        raw = values["roots"]
        if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
            raise ValueError(f"[sources.{source}] roots must be a list of paths")
        return cls(roots=tuple(Path(str(entry)).expanduser() for entry in raw))


def _load_source_configs(data: dict[str, object]) -> dict[str, SourceConfig]:
    """Read `[sources.<source>]` sections, refusing a name no parser answers to.

    A typo here would otherwise read as "that harness has no sessions", which
    is exactly the silent blindness this configuration exists to remove.
    """
    raw = data.get("sources", {})
    if not isinstance(raw, dict):
        raise ValueError("[sources] must be a table of per-source sections")
    known = {source.value for source in Source}
    sources: dict[str, SourceConfig] = {}
    for raw_name, section in raw.items():
        name = str(raw_name)
        if name not in known:
            raise ValueError(
                f"unknown source '{name}' in [sources]; expected one of {sorted(known)}"
            )
        if not isinstance(section, dict):
            raise ValueError(f"[sources.{name}] must be a table")
        values = {str(key): value for key, value in section.items()}
        sources[name] = SourceConfig.from_values(name, values)
    return sources


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    db_path: Path
    lock_path: Path
    config_path: Path
    fts: FtsConfig
    embedding: EmbeddingConfig
    daemon: DaemonConfig
    cli: CliConfig
    compaction: CompactionConfig = CompactionConfig()
    live: LiveConfig = LiveConfig()
    duckdb: DuckDBConfig = field(default_factory=DuckDBConfig)
    # Per-source discovery roots; a source absent here uses its parser default.
    sources: Mapping[str, SourceConfig] = field(default_factory=dict)

    @classmethod
    def load(cls) -> AppConfig:
        home = Path.home()
        default_data_dir = Path(os.environ.get("RECALL_DATA_DIR", home / ".local/share/recall"))
        default_config_path = Path(
            os.environ.get("RECALL_CONFIG_PATH", home / ".config/recall/config.toml")
        )
        db_path = Path(os.environ.get("RECALL_DB_PATH", default_data_dir / "recall.duckdb"))
        lock_path = Path(os.environ.get("RECALL_LOCK_PATH", default_data_dir / "recall.lock"))

        fts_section: dict[str, object] | None = None
        embed_section: dict[str, object] | None = None
        daemon_section: dict[str, object] | None = None
        cli_section: dict[str, object] | None = None
        compaction_section: dict[str, object] | None = None
        duckdb_section: dict[str, object] | None = None
        live_section: dict[str, object] | None = None
        sources: dict[str, SourceConfig] = {}
        if default_config_path.exists():
            raw = default_config_path.read_text(encoding="utf-8")
            data = tomllib.loads(raw) if raw.strip() else {}
            fts_raw = data.get("fts", {}) if isinstance(data, dict) else {}
            if isinstance(fts_raw, dict):
                fts_section = fts_raw
            embed_raw = data.get("embedding", {}) if isinstance(data, dict) else {}
            if isinstance(embed_raw, dict):
                embed_section = embed_raw
            daemon_raw = data.get("daemon", {}) if isinstance(data, dict) else {}
            if isinstance(daemon_raw, dict):
                daemon_section = daemon_raw
            cli_raw = data.get("cli", {}) if isinstance(data, dict) else {}
            if isinstance(cli_raw, dict):
                cli_section = cli_raw
            compaction_raw = data.get("compaction", {}) if isinstance(data, dict) else {}
            if isinstance(compaction_raw, dict):
                compaction_section = compaction_raw
            duckdb_raw = data.get("duckdb", {}) if isinstance(data, dict) else {}
            if isinstance(duckdb_raw, dict):
                duckdb_section = duckdb_raw
            live_raw = data.get("live", {}) if isinstance(data, dict) else {}
            if isinstance(live_raw, dict):
                live_section = live_raw
            if isinstance(data, dict):
                sources = _load_source_configs(data)

        fts_values = dict(fts_section) if fts_section else {}
        env_fields = os.environ.get("RECALL_FTS_FIELDS")
        if env_fields is not None:
            fts_values["fields"] = [field.strip() for field in env_fields.split(",")]
        env_fts_backend = os.environ.get("RECALL_FTS_BACKEND")
        if env_fts_backend is not None:
            fts_values["backend"] = env_fts_backend

        fts = FtsConfig.from_toml(fts_values or None)

        # Embedding config: env vars override config file
        embed_values = dict(embed_section) if embed_section else {}
        env_backend = os.environ.get("RECALL_EMBED_BACKEND")
        if env_backend is not None:
            embed_values["backend"] = env_backend
        env_model = os.environ.get("RECALL_EMBED_MODEL")
        if env_model is not None:
            embed_values["model"] = env_model
        env_batch = os.environ.get("RECALL_EMBED_BATCH_SIZE")
        if env_batch is not None:
            embed_values["batch_size"] = env_batch
        context_values = embed_values.get("context")
        context_values = {} if not isinstance(context_values, dict) else dict(context_values)
        env_context_mode = os.environ.get("RECALL_CONTEXT_MODE")
        if env_context_mode is not None:
            context_values["mode"] = env_context_mode
        env_context_fallback = os.environ.get("RECALL_CONTEXT_FALLBACK")
        if env_context_fallback is not None:
            context_values["fallback"] = env_context_fallback
        env_context_model = os.environ.get("RECALL_CONTEXT_MODEL")
        if env_context_model is not None:
            context_values["model"] = env_context_model
        env_context_batch_size = os.environ.get("RECALL_CONTEXT_BATCH_SIZE")
        if env_context_batch_size is not None:
            context_values["batch_size"] = env_context_batch_size
        env_context_concurrency = os.environ.get("RECALL_CONTEXT_CONCURRENCY")
        if env_context_concurrency is not None:
            context_values["concurrency"] = env_context_concurrency
        env_context_base_url = os.environ.get("RECALL_CONTEXT_BASE_URL")
        if env_context_base_url is not None:
            context_values["base_url"] = env_context_base_url
        env_context_timeout = os.environ.get("RECALL_CONTEXT_TIMEOUT")
        if env_context_timeout is not None:
            context_values["timeout"] = env_context_timeout
        env_context_api_key = os.environ.get("RECALL_CONTEXT_API_KEY")
        if env_context_api_key is not None:
            context_values["api_key"] = env_context_api_key
        env_context_instruction_prefix = os.environ.get("RECALL_CONTEXT_INSTRUCTION_PREFIX")
        if env_context_instruction_prefix is not None:
            context_values["instruction_prefix"] = env_context_instruction_prefix
        env_context_max_document_chars = os.environ.get("RECALL_CONTEXT_MAX_DOCUMENT_CHARS")
        if env_context_max_document_chars is not None:
            context_values["max_document_chars"] = env_context_max_document_chars
        env_context_reasoning_effort = os.environ.get("RECALL_CONTEXT_REASONING_EFFORT")
        if env_context_reasoning_effort is not None:
            context_values["reasoning_effort"] = env_context_reasoning_effort
        env_context_executable = os.environ.get("RECALL_CONTEXT_EXECUTABLE")
        if env_context_executable is not None:
            context_values["executable"] = env_context_executable
        if context_values:
            embed_values["context"] = context_values
        embedding = EmbeddingConfig.from_values(embed_values or None)
        default_daemon_embed = embedding_backend_available(embedding.backend)

        daemon_values = dict(daemon_section) if daemon_section else {}
        env_daemon_interval = os.environ.get("RECALL_DAEMON_INTERVAL")
        if env_daemon_interval is not None:
            daemon_values["interval"] = env_daemon_interval
        env_daemon_embed = os.environ.get("RECALL_DAEMON_EMBED")
        if env_daemon_embed is not None:
            daemon_values["embed"] = env_daemon_embed
        env_daemon_source = os.environ.get("RECALL_DAEMON_SOURCE")
        if env_daemon_source is not None:
            daemon_values["source"] = env_daemon_source
        env_daemon_scheduler = os.environ.get("RECALL_DAEMON_SCHEDULER")
        if env_daemon_scheduler is not None:
            daemon_values["scheduler"] = env_daemon_scheduler
        env_daemon_mode = os.environ.get("RECALL_DAEMON_MODE")
        if env_daemon_mode is not None:
            daemon_values["mode"] = env_daemon_mode
        env_daemon_debounce = os.environ.get("RECALL_DAEMON_DEBOUNCE")
        if env_daemon_debounce is not None:
            daemon_values["debounce"] = env_daemon_debounce
        env_daemon_fts_debounce = os.environ.get("RECALL_DAEMON_FTS_DEBOUNCE")
        if env_daemon_fts_debounce is not None:
            daemon_values["fts_debounce"] = env_daemon_fts_debounce
        env_daemon_idle_timeout = os.environ.get("RECALL_DAEMON_IDLE_TIMEOUT")
        if env_daemon_idle_timeout is not None:
            daemon_values["idle_timeout"] = env_daemon_idle_timeout
        env_embed_interval = os.environ.get("RECALL_DAEMON_EMBED_INTERVAL")
        if env_embed_interval is not None:
            daemon_values["embed_interval"] = env_embed_interval
        env_embed_backoff = os.environ.get("RECALL_DAEMON_EMBED_BACKOFF")
        if env_embed_backoff is not None:
            daemon_values["embed_backoff"] = env_embed_backoff
        env_embed_idle_session = os.environ.get("RECALL_DAEMON_EMBED_IDLE_SESSION")
        if env_embed_idle_session is not None:
            daemon_values["embed_idle_session"] = env_embed_idle_session
        env_embed_model_timeout = os.environ.get("RECALL_DAEMON_EMBED_MODEL_TIMEOUT")
        if env_embed_model_timeout is not None:
            daemon_values["embed_model_timeout"] = env_embed_model_timeout
        env_live_idle_threshold = os.environ.get("RECALL_DAEMON_LIVE_IDLE_THRESHOLD")
        if env_live_idle_threshold is not None:
            daemon_values["live_idle_threshold"] = env_live_idle_threshold
        env_live_discovery_interval = os.environ.get("RECALL_DAEMON_LIVE_DISCOVERY_INTERVAL")
        if env_live_discovery_interval is not None:
            daemon_values["live_discovery_interval"] = env_live_discovery_interval
        env_live_max_subscriptions = os.environ.get("RECALL_DAEMON_LIVE_MAX_SUBSCRIPTIONS")
        if env_live_max_subscriptions is not None:
            daemon_values["live_max_subscriptions"] = env_live_max_subscriptions
        env_load_threshold = os.environ.get("RECALL_DAEMON_LOAD_THRESHOLD")
        if env_load_threshold is not None:
            daemon_values["load_threshold"] = env_load_threshold
        env_battery_threshold = os.environ.get("RECALL_DAEMON_BATTERY_THRESHOLD")
        if env_battery_threshold is not None:
            daemon_values["battery_threshold"] = env_battery_threshold
        env_log_max_bytes = os.environ.get("RECALL_DAEMON_LOG_MAX_BYTES")
        if env_log_max_bytes is not None:
            daemon_values["log_max_bytes"] = env_log_max_bytes
        env_log_level = os.environ.get("RECALL_DAEMON_LOG_LEVEL")
        if env_log_level is not None:
            daemon_values["log_level"] = env_log_level
        daemon = DaemonConfig.from_values(
            daemon_values or None,
            default_embed=default_daemon_embed,
        )

        cli_values = dict(cli_section) if cli_section else {}
        env_cli_status_notices = os.environ.get("RECALL_CLI_STATUS_NOTICES")
        if env_cli_status_notices is not None:
            cli_values["status_notices"] = env_cli_status_notices
        cli = CliConfig.from_values(cli_values or None)

        compaction_values = dict(compaction_section) if compaction_section else {}
        env_compaction_auto = os.environ.get("RECALL_COMPACTION_AUTO")
        if env_compaction_auto is not None:
            compaction_values["auto_trigger"] = env_compaction_auto
        env_compaction_threshold = os.environ.get("RECALL_COMPACTION_THRESHOLD")
        if env_compaction_threshold is not None:
            compaction_values["bloat_ratio_threshold"] = env_compaction_threshold
        env_compaction_interval = os.environ.get("RECALL_COMPACTION_INTERVAL_HOURS")
        if env_compaction_interval is not None:
            compaction_values["check_interval_hours"] = env_compaction_interval
        compaction = CompactionConfig.from_values(compaction_values or None)
        duckdb_config = DuckDBConfig.from_values(duckdb_section or None)

        live_values = dict(live_section) if live_section else {}
        env_live_idle_window = os.environ.get("RECALL_LIVE_IDLE_WINDOW")
        if env_live_idle_window is not None:
            live_values["idle_window"] = env_live_idle_window
        env_live_fresh_timeout = os.environ.get("RECALL_LIVE_FRESH_TIMEOUT")
        if env_live_fresh_timeout is not None:
            live_values["fresh_timeout"] = env_live_fresh_timeout
        live = LiveConfig.from_values(live_values or None)

        return cls(
            data_dir=default_data_dir,
            db_path=db_path,
            lock_path=lock_path,
            config_path=default_config_path,
            fts=fts,
            embedding=embedding,
            daemon=daemon,
            cli=cli,
            compaction=compaction,
            duckdb=duckdb_config,
            live=live,
            sources=sources,
        )


def create_private_dir(path: Path) -> None:
    """Create *path* owner-only (0700) if it is missing; an existing directory keeps its mode.

    Missing parents are created with default permissions, as `mkdir -p` does.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(mode=0o700, exist_ok=True)


def resolve_recall_binary() -> str:
    """Find the recall CLI binary path."""
    candidate = shutil.which("recall")
    if candidate:
        return candidate
    argv0 = Path(sys.argv[0]).expanduser()
    if argv0.name != "pytest" and argv0.exists():
        return str(argv0.resolve())
    sibling = Path(sys.executable).with_name("recall")
    if sibling.exists():
        return str(sibling.resolve())
    raise RuntimeError("could not resolve the installed recall CLI entrypoint")


def _parse_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _parse_optional_bool(value: object | None, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true, false, or auto")
