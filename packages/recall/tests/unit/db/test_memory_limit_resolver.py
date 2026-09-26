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
from recall.db.connection import resolve_memory_limit


def _config(
    tmp_path: Path,
    *,
    memory_limit: str | None = None,
    temp_directory: str | None = None,
) -> AppConfig:
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
        duckdb=DuckDBConfig(memory_limit=memory_limit, temp_directory=temp_directory),
    )


def _patch_host_ram(monkeypatch: pytest.MonkeyPatch, *, gib: int) -> None:
    page_size = 4096
    pages = gib * 1024**3 // page_size

    def sysconf(name: str) -> int:
        if name == "SC_PAGE_SIZE":
            return page_size
        if name == "SC_PHYS_PAGES":
            return pages
        raise ValueError(name)

    monkeypatch.setattr("recall.db.connection.os.sysconf", sysconf)


@pytest.mark.parametrize("host_gib", [1, 64, 128, 1024])
def test_default_memory_limit_leaves_room_for_process_allocations(
    monkeypatch: pytest.MonkeyPatch, host_gib: int, tmp_path: Path
) -> None:
    _patch_host_ram(monkeypatch, gib=host_gib)

    monkeypatch.delenv("RECALL_DUCKDB_MEMORY_LIMIT", raising=False)
    assert resolve_memory_limit(_config(tmp_path)) == "2GB"


def test_resolve_memory_limit_prefers_env_over_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RECALL_DUCKDB_MEMORY_LIMIT", "3GB")

    assert resolve_memory_limit(_config(tmp_path, memory_limit="4GB")) == "3GiB"


def test_resolve_memory_limit_prefers_config_over_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RECALL_DUCKDB_MEMORY_LIMIT", raising=False)

    assert resolve_memory_limit(_config(tmp_path, memory_limit="5GB")) == "5GiB"


def test_resolve_memory_limit_raises_on_unparseable_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RECALL_DUCKDB_MEMORY_LIMIT", "banana")

    with pytest.raises(ValueError, match=r"RECALL_DUCKDB_MEMORY_LIMIT.*DuckDB memory size"):
        resolve_memory_limit(_config(tmp_path))


@pytest.mark.parametrize("value", ["0GB", "-5GB"])
def test_resolve_memory_limit_raises_on_non_positive_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    monkeypatch.setenv("RECALL_DUCKDB_MEMORY_LIMIT", value)

    with pytest.raises(ValueError, match="RECALL_DUCKDB_MEMORY_LIMIT must be positive"):
        resolve_memory_limit(_config(tmp_path))
