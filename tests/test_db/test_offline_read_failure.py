"""An offline read keeps its originating native error when closing the handle also fails.

The injected close failure stands in for a native handle that cannot close after
a fatal statement; what stays unverified is the engine's own close behaviour.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb
import pytest
from recall import _native
from recall.core.config import AppConfig
from recall.db.ownership import read_existing_snapshot, read_offline
from recall.db.schema import ensure_schema

INDEX_DELETE = "Failed to delete all rows from index"


class _UnclosableConn:
    def __init__(self, inner: duckdb.DuckDBPyConnection) -> None:
        self._inner = inner

    def close(self) -> None:
        self._inner.close()
        raise duckdb.FatalException("close aborted")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    configured = replace(
        AppConfig.load(),
        data_dir=tmp_path,
        db_path=tmp_path / "recall.duckdb",
        lock_path=tmp_path / "recall.lock",
    )
    with duckdb.connect(str(configured.db_path)) as conn:
        ensure_schema(conn)
    real = _native.duckdb_connect

    def unclosable(*args: Any, **kwargs: Any) -> Any:
        return _UnclosableConn(real(*args, **kwargs))

    monkeypatch.setattr(_native, "duckdb_connect", unclosable)
    return configured


def _fatal_read(_conn: object) -> None:
    raise duckdb.FatalException(INDEX_DELETE)


@pytest.mark.parametrize("opener", [read_offline, read_existing_snapshot])
def test_close_failure_does_not_replace_the_native_read_error(
    config: AppConfig, opener: Any
) -> None:
    with pytest.raises(BaseExceptionGroup) as caught:
        opener(config, _fatal_read)

    assert {str(error) for error in caught.value.exceptions} == {"close aborted", INDEX_DELETE}
