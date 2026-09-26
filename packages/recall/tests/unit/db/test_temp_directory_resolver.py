from __future__ import annotations

from pathlib import Path

import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    DuckDBConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.db.connection import resolve_temp_directory


def _config(tmp_path: Path, *, temp_directory: str | None = None) -> AppConfig:
    data_dir = tmp_path / "data"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
        duckdb=DuckDBConfig(temp_directory=temp_directory),
    )


def test_temp_directory_default_under_data_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RECALL_DUCKDB_TEMP_DIR", raising=False)
    config = _config(tmp_path)

    assert resolve_temp_directory(config) == config.data_dir / "duckdb_spill"


def test_temp_directory_env_overrides_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / "env-spill"
    config_path = tmp_path / "config-spill"
    monkeypatch.setenv("RECALL_DUCKDB_TEMP_DIR", str(env_path))

    assert resolve_temp_directory(_config(tmp_path, temp_directory=str(config_path))) == env_path


def test_temp_directory_config_over_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config-spill"
    monkeypatch.delenv("RECALL_DUCKDB_TEMP_DIR", raising=False)

    config = _config(tmp_path, temp_directory=str(config_path))

    assert resolve_temp_directory(config) == config_path


def test_temp_directory_rejects_empty_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RECALL_DUCKDB_TEMP_DIR", "   ")

    with pytest.raises(ValueError, match="RECALL_DUCKDB_TEMP_DIR must not be empty"):
        resolve_temp_directory(_config(tmp_path))


def test_temp_directory_rejects_empty_config() -> None:
    with pytest.raises(ValueError, match=r"\[duckdb\] temp_directory must not be empty"):
        DuckDBConfig.from_values({"temp_directory": ""})
