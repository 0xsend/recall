from __future__ import annotations

from pathlib import Path

import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    ContextConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.rpc_types import config_fingerprint
from recall.core.types import SchedulerKind, SearchMode, Source, parse_search_mode


def test_embedding_config_normalizes_backend_name() -> None:
    config = EmbeddingConfig.from_values({"backend": "  MLX  "})
    assert config.backend == "mlx"


def test_embedding_config_defaults_to_auto() -> None:
    config = EmbeddingConfig.from_values(None)
    assert config.backend == "auto"


def test_embedding_config_accepts_auto_backend() -> None:
    config = EmbeddingConfig.from_values({"backend": "auto"})
    assert config.backend == "auto"


def test_embedding_config_parses_context_mode() -> None:
    config = EmbeddingConfig.from_values(
        {
            "context": {
                "mode": "template",
                "fallback": "off",
                "model": "context-model",
                "max_tokens": "80",
                "batch_size": "4",
                "concurrency": "2",
                "min_chars": "12",
            }
        }
    )

    assert config.context == ContextConfig(
        mode="template",
        fallback="off",
        model="context-model",
        max_tokens=80,
        batch_size=4,
        concurrency=2,
        min_chars=12,
    )


@pytest.mark.parametrize(
    ("context", "message"),
    [
        ({"mode": "semantic"}, "invalid context mode"),
        ({"fallback": "retry"}, "invalid context fallback"),
        ({"model": "   "}, "context model must not be empty"),
        ({"max_tokens": "0"}, "context max_tokens must be positive"),
        ({"batch_size": "-1"}, "context batch_size must be positive"),
        ({"concurrency": "0"}, "context concurrency must be positive"),
        ({"min_chars": "-1"}, "context min_chars must be non-negative"),
        ({"base_url": "not-a-url"}, "context base_url must start with http"),
        ({"base_url": "   "}, "context base_url must not be empty"),
        ({"timeout": "0"}, "context timeout must be positive"),
        ({"timeout": "-5"}, "context timeout must be positive"),
        ({"api_key": "   "}, "context api_key must not be empty"),
        ({"instruction_prefix": "   "}, "context instruction_prefix must not be empty"),
        ({"instruction_prefix": ""}, "context instruction_prefix must not be empty"),
    ],
)
def test_embedding_config_rejects_invalid_context_values(
    context: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        EmbeddingConfig.from_values({"context": context})


def test_embedding_config_parses_context_base_url_and_timeout() -> None:
    config = EmbeddingConfig.from_values(
        {
            "context": {
                "mode": "llm-remote",
                "base_url": "http://litellm.local:4000",
                "timeout": "45",
            }
        }
    )

    assert config.context.base_url == "http://litellm.local:4000"
    assert config.context.timeout == 45.0


def test_embedding_config_optional_context_settings_default_to_none() -> None:
    config = EmbeddingConfig.from_values({"context": {"mode": "llm-remote"}})

    assert config.context.base_url is None
    assert config.context.timeout is None
    assert config.context.api_key is None
    assert config.context.instruction_prefix is None


def test_embedding_config_parses_context_api_key() -> None:
    config = EmbeddingConfig.from_values(
        {"context": {"mode": "llm-remote", "api_key": "sk-recall-local"}}
    )

    assert config.context.api_key == "sk-recall-local"


def test_embedding_config_context_api_key_strips_whitespace() -> None:
    # Mirrors base_url behaviour: leading/trailing whitespace is stripped, not preserved.
    config = EmbeddingConfig.from_values(
        {"context": {"mode": "llm-remote", "api_key": "  sk-padded  "}}
    )

    assert config.context.api_key == "sk-padded"


def test_embedding_config_parses_context_instruction_prefix() -> None:
    config = EmbeddingConfig.from_values(
        {"context": {"mode": "llm-remote", "instruction_prefix": "/no_think\n\n"}}
    )

    # Trailing whitespace MUST be preserved: '/no_think\n\n' is the recommended
    # form for Qwen3, and stripping the newlines breaks the model's parsing.
    assert config.context.instruction_prefix == "/no_think\n\n"


def test_embedding_config_context_max_document_chars_defaults_to_400k() -> None:
    # As of v0.16.0 the in-code default switched from None (unbounded) to
    # DEFAULT_CONTEXT_MAX_DOCUMENT_CHARS so out-of-the-box installs get smart
    # truncation. Explicit `null` in the user's config still produces None
    # (covered by test_embedding_config_context_max_document_chars_null_disables_cap).
    from recall.core.config import DEFAULT_CONTEXT_MAX_DOCUMENT_CHARS

    config = EmbeddingConfig.from_values({"context": {"mode": "llm-codex"}})

    assert config.context.max_document_chars == DEFAULT_CONTEXT_MAX_DOCUMENT_CHARS
    assert DEFAULT_CONTEXT_MAX_DOCUMENT_CHARS == 400_000


def test_embedding_config_context_max_document_chars_null_disables_cap() -> None:
    config = EmbeddingConfig.from_values(
        {"context": {"mode": "llm-codex", "max_document_chars": None}}
    )

    assert config.context.max_document_chars is None


def test_embedding_config_parses_context_max_document_chars() -> None:
    config = EmbeddingConfig.from_values(
        {"context": {"mode": "llm-codex", "max_document_chars": 1000}}
    )

    assert config.context.max_document_chars == 1000


@pytest.mark.parametrize("value", [0, -1])
def test_embedding_config_rejects_non_positive_context_max_document_chars(value: int) -> None:
    with pytest.raises(
        ValueError,
        match="context max_document_chars must be a positive integer or null",
    ):
        EmbeddingConfig.from_values({"context": {"mode": "llm-codex", "max_document_chars": value}})


def test_embedding_config_rejects_non_int_context_max_document_chars() -> None:
    with pytest.raises(
        ValueError,
        match="context max_document_chars must be a positive integer or null",
    ):
        EmbeddingConfig.from_values(
            {"context": {"mode": "llm-codex", "max_document_chars": "not-an-int"}}
        )


def test_embedding_config_accepts_llm_codex_mode() -> None:
    config = EmbeddingConfig.from_values(
        {"context": {"mode": "llm-codex", "model": "gpt-5.3-codex-spark"}}
    )

    assert config.context.mode == "llm-codex"
    assert config.context.model == "gpt-5.3-codex-spark"
    # Defaults for the codex-specific knobs.
    assert config.context.reasoning_effort is None
    assert config.context.executable == "codex"


def test_embedding_config_accepts_reasoning_effort() -> None:
    config = EmbeddingConfig.from_values(
        {"context": {"mode": "llm-codex", "reasoning_effort": " LOW "}}
    )

    # Normalized lower-case + stripped.
    assert config.context.reasoning_effort == "low"


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max", "ultra"])
def test_embedding_config_accepts_each_reasoning_effort_value(effort: str) -> None:
    config = EmbeddingConfig.from_values(
        {"context": {"mode": "llm-codex", "reasoning_effort": effort}}
    )
    assert config.context.reasoning_effort == effort


def test_embedding_config_rejects_unknown_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="invalid context reasoning_effort"):
        EmbeddingConfig.from_values(
            {"context": {"mode": "llm-codex", "reasoning_effort": "extreme"}}
        )


def test_embedding_config_rejects_blank_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        EmbeddingConfig.from_values({"context": {"mode": "llm-codex", "reasoning_effort": "  "}})


def test_embedding_config_rejects_blank_executable() -> None:
    with pytest.raises(ValueError, match="executable must not be empty"):
        EmbeddingConfig.from_values({"context": {"mode": "llm-codex", "executable": "  "}})


def test_embedding_config_accepts_custom_executable() -> None:
    config = EmbeddingConfig.from_values(
        {"context": {"mode": "llm-codex", "executable": "/usr/local/bin/codex"}}
    )
    assert config.context.executable == "/usr/local/bin/codex"


def _fingerprint_for(context: ContextConfig) -> str:
    return config_fingerprint(
        AppConfig(
            data_dir=Path("/tmp/recall-test/data"),
            db_path=Path("/tmp/recall-test/data/recall.duckdb"),
            lock_path=Path("/tmp/recall-test/data/recall.lock"),
            config_path=Path("/tmp/recall-test/config.toml"),
            fts=FtsConfig(),
            embedding=EmbeddingConfig(context=context),
            daemon=DaemonConfig(),
            cli=CliConfig(),
        )
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            ContextConfig(mode="llm-remote"),
            ContextConfig(mode="llm-remote", base_url="http://litellm.local:4000"),
            id="base_url",
        ),
        # A different key against the same proxy can land on a different
        # tenant/model_group and yield divergent prefixes, so rotation is stale.
        pytest.param(
            ContextConfig(mode="llm-remote", api_key="sk-old"),
            ContextConfig(mode="llm-remote", api_key="sk-new"),
            id="api_key",
        ),
        # '/no_think' suppresses Qwen3's reasoning preamble, so prefixes diverge.
        pytest.param(
            ContextConfig(mode="llm-remote"),
            ContextConfig(mode="llm-remote", instruction_prefix="/no_think\n\n"),
            id="instruction_prefix",
        ),
        # low vs. high alter the reasoning tokens and therefore the prefix.
        pytest.param(
            ContextConfig(mode="llm-codex", reasoning_effort="low"),
            ContextConfig(mode="llm-codex", reasoning_effort="high"),
            id="reasoning_effort",
        ),
        # Different codex binaries can ship different bundled system prompts.
        pytest.param(
            ContextConfig(mode="llm-codex", executable="/opt/codex-a/bin/codex"),
            ContextConfig(mode="llm-codex", executable="/opt/codex-b/bin/codex"),
            id="executable",
        ),
        pytest.param(ContextConfig(mode="off"), ContextConfig(mode="template"), id="mode"),
        pytest.param(
            ContextConfig(mode="template"),
            ContextConfig(mode="template", max_tokens=64),
            id="max_tokens",
        ),
    ],
)
def test_config_fingerprint_changes_with_generation_affecting_context(
    before: ContextConfig, after: ContextConfig
) -> None:
    assert _fingerprint_for(before) != _fingerprint_for(after)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        # Regression guard: timeout is documented as not participating.
        pytest.param(
            ContextConfig(mode="llm-remote", timeout=30.0),
            ContextConfig(mode="llm-remote", timeout=120.0),
            id="timeout",
        ),
        pytest.param(
            ContextConfig(mode="llm-codex", concurrency=1),
            ContextConfig(mode="llm-codex", concurrency=8),
            id="concurrency",
        ),
    ],
)
def test_config_fingerprint_ignores_throughput_only_context(
    before: ContextConfig, after: ContextConfig
) -> None:
    assert _fingerprint_for(before) == _fingerprint_for(after)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"backend": "   "}, "embedding backend must not be empty"),
        ({"batch_size": "0"}, "embedding batch_size must be positive"),
        ({"batch_size": "-5"}, "embedding batch_size must be positive"),
        ({"model": "   "}, "embedding model must not be empty"),
    ],
)
def test_embedding_config_rejects_invalid_values(values: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EmbeddingConfig.from_values(values)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("keyword", SearchMode.KEYWORD),
        (" KW ", SearchMode.KEYWORD),
        ("semantic", SearchMode.VECTOR),
        (" RRF ", SearchMode.HYBRID),
    ],
)
def test_parse_search_mode_normalizes_aliases(value: str, expected: SearchMode) -> None:
    assert parse_search_mode(value) == expected


def test_parse_search_mode_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported search mode"):
        parse_search_mode("lexical")


def test_daemon_config_normalizes_source_and_embed_flag() -> None:
    config = DaemonConfig.from_values(
        {
            "interval": "300",
            "embed": "true",
            "source": " PI-AGENT ",
            "scheduler": " SYSTEMD ",
        },
        default_embed=False,
    )
    assert config.interval == 300
    assert config.embed is True
    assert config.source == Source.PI_AGENT
    assert config.scheduler == SchedulerKind.SYSTEMD


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"interval": "0"}, "daemon interval must be positive"),
        ({"interval": "-5"}, "daemon interval must be positive"),
        ({"embed": "maybe"}, "daemon embed must be true, false, or auto"),
        ({"source": "unknown"}, "unsupported source"),
        ({"scheduler": "launchagent"}, "unsupported scheduler"),
    ],
)
def test_daemon_config_rejects_invalid_values(values: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DaemonConfig.from_values(values, default_embed=False)


def test_daemon_config_accepts_auto_embed_and_uses_default() -> None:
    config = DaemonConfig.from_values({"embed": "auto"}, default_embed=True)

    assert config.embed is True


def test_cli_config_parses_status_notices_flag() -> None:
    assert CliConfig.from_values({"status_notices": "true"}).status_notices is True
    assert CliConfig.from_values({"status_notices": "false"}).status_notices is False


def test_app_config_load_reads_cli_and_daemon_scheduler_settings(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [embedding.context]
        mode = "template"
        fallback = "off"
        model = "configured-context-model"
        max_tokens = 99
        batch_size = 3
        concurrency = 2
        min_chars = 12

        [daemon]
        scheduler = "cron"

        [cli]
        status_notices = false
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))

    config = AppConfig.load()

    assert config.daemon.scheduler == SchedulerKind.CRON
    assert config.cli.status_notices is False
    assert config.embedding.context == ContextConfig(
        mode="template",
        fallback="off",
        model="configured-context-model",
        max_tokens=99,
        batch_size=3,
        concurrency=2,
        min_chars=12,
    )


def test_app_config_env_overrides_cli_and_daemon_scheduler_settings(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [daemon]
        scheduler = "cron"

        [cli]
        status_notices = false
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("RECALL_DAEMON_SCHEDULER", "launchd")
    monkeypatch.setenv("RECALL_CLI_STATUS_NOTICES", "true")
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "llm-local")
    monkeypatch.setenv("RECALL_CONTEXT_FALLBACK", "error")
    monkeypatch.setenv("RECALL_CONTEXT_MODEL", "env-context-model")
    monkeypatch.setenv("RECALL_CONTEXT_BASE_URL", "http://litellm.local:4000")
    monkeypatch.setenv("RECALL_CONTEXT_TIMEOUT", "60")
    monkeypatch.setenv("RECALL_CONTEXT_API_KEY", "sk-from-env")
    monkeypatch.setenv("RECALL_CONTEXT_INSTRUCTION_PREFIX", "/no_think\n\n")
    monkeypatch.setenv("RECALL_CONTEXT_MAX_DOCUMENT_CHARS", "200000")
    monkeypatch.setenv("RECALL_CONTEXT_REASONING_EFFORT", "low")
    monkeypatch.setenv("RECALL_CONTEXT_EXECUTABLE", "/opt/codex/bin/codex")
    monkeypatch.setenv("RECALL_CONTEXT_CONCURRENCY", "6")

    config = AppConfig.load()

    assert config.daemon.scheduler == SchedulerKind.LAUNCHD
    assert config.cli.status_notices is True
    assert config.embedding.context.mode == "llm-local"
    assert config.embedding.context.fallback == "error"
    assert config.embedding.context.model == "env-context-model"
    assert config.embedding.context.base_url == "http://litellm.local:4000"
    assert config.embedding.context.timeout == 60.0
    assert config.embedding.context.api_key == "sk-from-env"
    assert config.embedding.context.instruction_prefix == "/no_think\n\n"
    assert config.embedding.context.max_document_chars == 200000
    assert config.embedding.context.reasoning_effort == "low"
    assert config.embedding.context.executable == "/opt/codex/bin/codex"
    assert config.embedding.context.concurrency == 6


@pytest.mark.parametrize(("supported", "expected"), [(True, True), (False, False)])
def test_app_config_defaults_daemon_embed_from_backend_support(
    tmp_path, monkeypatch, supported: bool, expected: bool
) -> None:
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "missing.toml"))
    monkeypatch.setattr(
        "recall.core.config.embedding_backend_available",
        lambda _backend: supported,
    )

    config = AppConfig.load()

    assert config.daemon.embed is expected


def test_app_config_env_can_request_auto_daemon_embed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("RECALL_DAEMON_EMBED", "auto")
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: True)

    config = AppConfig.load()

    assert config.daemon.embed is True
