from __future__ import annotations

from pathlib import Path

import pytest
from recall.core.config import AppConfig, CompactionConfig, ContextConfig, DaemonConfig, FtsConfig


def test_fts_config_defaults_to_sqlite_sidecar() -> None:
    assert FtsConfig().backend == "sqlite_sidecar"


def test_fts_config_accepts_duckdb() -> None:
    assert FtsConfig.from_toml({"backend": "duckdb"}).backend == "duckdb"


def test_fts_config_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="fts backend must be one of: duckdb, sqlite_sidecar"):
        FtsConfig.from_toml({"backend": "elasticsearch"})


def test_recall_fts_backend_env_var_overrides_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [fts]
        backend = "sqlite_sidecar"
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("RECALL_FTS_BACKEND", "duckdb")
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)

    cfg = AppConfig.load()

    assert cfg.fts.backend == "duckdb"


def test_context_batch_size_defaults_to_eight() -> None:
    assert ContextConfig().batch_size == 8


def test_recall_context_batch_size_env_var_overrides_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [embedding.context]
        batch_size = 3
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("RECALL_CONTEXT_BATCH_SIZE", "5")
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)

    cfg = AppConfig.load()

    assert cfg.embedding.context.batch_size == 5


def test_context_batch_size_rejects_values_below_one() -> None:
    with pytest.raises(ValueError, match="context batch_size must be positive"):
        ContextConfig.from_values({"batch_size": "0"})


def test_daemon_config_adaptive_defaults() -> None:
    cfg = DaemonConfig()
    assert cfg.embed_interval == 120
    assert cfg.embed_backoff == 300
    assert cfg.embed_idle_session == 600
    assert cfg.embed_model_timeout == 600
    assert cfg.live_idle_threshold == 300
    assert cfg.live_discovery_interval == 30
    assert cfg.live_max_subscriptions == 64
    assert cfg.load_threshold == 0.7
    assert cfg.battery_threshold == 0.3


def test_daemon_config_from_values_adaptive_fields() -> None:
    cfg = DaemonConfig.from_values(
        {
            "embed_interval": "60",
            "embed_backoff": "120",
            "embed_idle_session": "300",
            "embed_model_timeout": "900",
            "live_idle_threshold": "180",
            "live_discovery_interval": "15",
            "live_max_subscriptions": "32",
            "load_threshold": "0.5",
            "battery_threshold": "0.2",
        }
    )
    assert cfg.embed_interval == 60
    assert cfg.embed_backoff == 120
    assert cfg.embed_idle_session == 300
    assert cfg.embed_model_timeout == 900
    assert cfg.live_idle_threshold == 180
    assert cfg.live_discovery_interval == 15
    assert cfg.live_max_subscriptions == 32
    assert cfg.load_threshold == 0.5
    assert cfg.battery_threshold == 0.2


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"embed_interval": "0"}, "embed_interval must be positive"),
        ({"embed_interval": "-1"}, "embed_interval must be positive"),
        ({"embed_backoff": "0"}, "embed_backoff must be positive"),
        ({"embed_backoff": "-5"}, "embed_backoff must be positive"),
        ({"embed_idle_session": "0"}, "embed_idle_session must be positive"),
        ({"embed_idle_session": "-1"}, "embed_idle_session must be positive"),
        ({"embed_model_timeout": "0"}, "embed_model_timeout must be positive"),
        ({"embed_model_timeout": "-1"}, "embed_model_timeout must be positive"),
        ({"live_idle_threshold": "0"}, "daemon live_idle_threshold must be positive"),
        ({"live_idle_threshold": "-1"}, "daemon live_idle_threshold must be positive"),
        (
            {"live_discovery_interval": "0"},
            "daemon live_discovery_interval must be positive",
        ),
        (
            {"live_discovery_interval": "-1"},
            "daemon live_discovery_interval must be positive",
        ),
        (
            {"live_max_subscriptions": "0"},
            "daemon live_max_subscriptions must be positive",
        ),
        (
            {"live_max_subscriptions": "-1"},
            "daemon live_max_subscriptions must be positive",
        ),
        ({"load_threshold": "0"}, "daemon load_threshold must be positive"),
        ({"load_threshold": "-0.1"}, "daemon load_threshold must be positive"),
        ({"battery_threshold": "0"}, "battery_threshold must be between 0 and 1"),
        ({"battery_threshold": "1.5"}, "battery_threshold must be between 0 and 1"),
        ({"battery_threshold": "-0.1"}, "battery_threshold must be between 0 and 1"),
    ],
)
def test_daemon_config_from_values_invalid_adaptive_fields(
    values: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DaemonConfig.from_values(values)


def test_daemon_config_from_values_threshold_boundary_at_one() -> None:
    """Battery threshold caps at 1.0; load threshold may exceed it."""
    cfg = DaemonConfig.from_values({"load_threshold": "1.0", "battery_threshold": "1.0"})
    assert cfg.load_threshold == 1.0
    assert cfg.battery_threshold == 1.0
    # load_threshold above 1.0 opts into embedding under oversubscription.
    assert DaemonConfig.from_values({"load_threshold": "4.0"}).load_threshold == 4.0


def test_compaction_config_defaults() -> None:
    cfg = CompactionConfig()
    assert cfg.auto_trigger is True
    # Lowered from 2.0: a host reading 1.906 could not complete `index --full`
    # at all, so a 2.0 trigger fired only once the database was already broken.
    assert cfg.bloat_ratio_threshold == 1.5
    assert cfg.check_interval_hours == 6


def test_compaction_config_from_values() -> None:
    cfg = CompactionConfig.from_values(
        {
            "auto_trigger": "false",
            "bloat_ratio_threshold": "2.5",
            "check_interval_hours": "12",
        }
    )
    assert cfg.auto_trigger is False
    assert cfg.bloat_ratio_threshold == 2.5
    assert cfg.check_interval_hours == 12


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"auto_trigger": "maybe"}, "compaction auto_trigger must be true, false, or auto"),
        (
            {"bloat_ratio_threshold": "-0.1"},
            "compaction bloat_ratio_threshold must be non-negative",
        ),
        (
            {"check_interval_hours": "0"},
            "compaction check_interval_hours must be at least 1",
        ),
    ],
)
def test_compaction_config_from_values_invalid(values: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CompactionConfig.from_values(values)


def test_app_config_load_uses_live_daemon_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RECALL_CONFIG_PATH", raising=False)
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)

    cfg = AppConfig.load()

    assert cfg.daemon.live_idle_threshold == 300
    assert cfg.daemon.live_discovery_interval == 30
    assert cfg.daemon.live_max_subscriptions == 64


def test_log_max_bytes_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "missing.toml"))
    monkeypatch.delenv("RECALL_DAEMON_LOG_MAX_BYTES", raising=False)
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)

    cfg = AppConfig.load()

    assert cfg.daemon.log_max_bytes == 52_428_800


def test_log_max_bytes_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("RECALL_DAEMON_LOG_MAX_BYTES", "104857600")
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)

    cfg = AppConfig.load()

    assert cfg.daemon.log_max_bytes == 104_857_600


def test_log_max_bytes_zero_disables(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("RECALL_DAEMON_LOG_MAX_BYTES", "0")
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)

    cfg = AppConfig.load()

    assert cfg.daemon.log_max_bytes == 0


def test_log_max_bytes_from_config_file(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [daemon]
        log_max_bytes = 41943040
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("RECALL_DAEMON_LOG_MAX_BYTES", raising=False)
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)

    cfg = AppConfig.load()

    assert cfg.daemon.log_max_bytes == 41_943_040


def test_app_config_load_reads_live_daemon_fields_from_config(tmp_path, monkeypatch) -> None:
    config_dir = tmp_path / ".config" / "recall"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        """
        [daemon]
        live_idle_threshold = 420
        live_discovery_interval = 45
        live_max_subscriptions = 96
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RECALL_CONFIG_PATH", raising=False)
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)

    cfg = AppConfig.load()

    assert cfg.daemon.live_idle_threshold == 420
    assert cfg.daemon.live_discovery_interval == 45
    assert cfg.daemon.live_max_subscriptions == 96


@pytest.mark.parametrize(
    ("env_var", "config_key", "config_value", "env_value"),
    [
        (
            "RECALL_DAEMON_LIVE_IDLE_THRESHOLD",
            "live_idle_threshold",
            420,
            "480",
        ),
        (
            "RECALL_DAEMON_LIVE_DISCOVERY_INTERVAL",
            "live_discovery_interval",
            45,
            "60",
        ),
        (
            "RECALL_DAEMON_LIVE_MAX_SUBSCRIPTIONS",
            "live_max_subscriptions",
            96,
            "128",
        ),
    ],
)
def test_app_config_env_overrides_live_daemon_fields(
    tmp_path,
    monkeypatch,
    env_var: str,
    config_key: str,
    config_value: int,
    env_value: str,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
        [daemon]
        {config_key} = {config_value}
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv(env_var, env_value)
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)

    cfg = AppConfig.load()

    assert getattr(cfg.daemon, config_key) == int(env_value)


def test_app_config_load_reads_compaction_fields_from_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [compaction]
        auto_trigger = false
        bloat_ratio_threshold = 3.5
        check_interval_hours = 8
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)

    cfg = AppConfig.load()

    assert cfg.compaction.auto_trigger is False
    assert cfg.compaction.bloat_ratio_threshold == 3.5
    assert cfg.compaction.check_interval_hours == 8


def test_app_config_env_overrides_compaction_fields(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [compaction]
        auto_trigger = false
        bloat_ratio_threshold = 3.5
        check_interval_hours = 8
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("RECALL_COMPACTION_AUTO", "1")
    monkeypatch.setenv("RECALL_COMPACTION_THRESHOLD", "9.5")
    monkeypatch.setenv("RECALL_COMPACTION_INTERVAL_HOURS", "2")
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)

    cfg = AppConfig.load()

    assert cfg.compaction.auto_trigger is True
    assert cfg.compaction.bloat_ratio_threshold == 9.5
    assert cfg.compaction.check_interval_hours == 2


def test_source_roots_default_to_unset_meaning_the_parsers_own_location() -> None:
    from recall.core.config import SourceConfig

    assert SourceConfig().roots is None


def test_source_roots_are_read_per_source_and_expand_home(tmp_path, monkeypatch) -> None:
    """REQ-LIVE-012: a lane started with CLAUDE_CONFIG_DIR elsewhere must be discoverable."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [sources.claude_code]
        roots = ["~/.claude-work/projects", "/srv/lanes/projects"]
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)

    cfg = AppConfig.load()

    assert cfg.sources["claude_code"].roots == (
        tmp_path / ".claude-work" / "projects",
        Path("/srv/lanes/projects"),
    )
    assert "codex" not in cfg.sources


def test_an_unknown_source_section_is_rejected(tmp_path, monkeypatch) -> None:
    """A typo that silently indexed nothing is the failure this config exists to fix."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [sources.claude-code]
        roots = ["/srv/lanes"]
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)

    with pytest.raises(ValueError, match="unknown source"):
        AppConfig.load()


def test_source_roots_must_be_a_list_of_paths(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [sources.codex]
        roots = "/srv/lanes"
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)

    with pytest.raises(ValueError, match="roots must be a list"):
        AppConfig.load()


def test_an_absent_source_section_leaves_roots_unset() -> None:
    """Unset and empty are different answers: one defaults, the other scans nothing."""
    from recall.core.config import SourceConfig

    assert SourceConfig.from_values("claude_code", None).roots is None
    assert SourceConfig.from_values("claude_code", {}).roots is None


def test_an_explicitly_empty_root_list_is_kept_as_empty(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[sources.codex]\nroots = []\n", encoding="utf-8")
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))

    config = AppConfig.load()

    assert config.sources["codex"].roots == ()


def test_live_config_defaults() -> None:
    from recall.core.config import LiveConfig

    assert LiveConfig().idle_window == 86400
    assert LiveConfig().fresh_timeout == 10.0


def test_live_fresh_timeout_is_read_from_config(tmp_path, monkeypatch) -> None:
    """REQ-LIVE-010: how long `--fresh` may hold an answer is the operator's call."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("[live]\nfresh_timeout = 2.5\n", encoding="utf-8")
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))

    assert AppConfig.load().live.fresh_timeout == 2.5


def test_live_fresh_timeout_env_override(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[live]\nfresh_timeout = 2.5\n", encoding="utf-8")
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_LIVE_FRESH_TIMEOUT", "0.5")

    assert AppConfig.load().live.fresh_timeout == 0.5


def test_a_non_positive_fresh_timeout_is_rejected() -> None:
    """A zero budget would make every `--fresh` a guaranteed timeout."""
    from recall.core.config import LiveConfig

    with pytest.raises(ValueError, match="live fresh_timeout must be positive"):
        LiveConfig.from_values({"fresh_timeout": 0})
