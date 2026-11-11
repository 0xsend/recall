"""FTS cleanup must not outrun observation of the originating native failure."""

from typing import Any, cast

import duckdb
import pytest
from recall.core.config import FtsConfig
from recall.db.lease import StandaloneLifetime
from recall.db.queries import FtsSettingsRestoreError, create_fts_indexes


def test_native_rebuild_failure_is_observed_before_settings_cleanup():
    observed = []
    cleanup_observed = []
    primary = duckdb.FatalException("Failed to delete all rows from index")

    class Observer(StandaloneLifetime):
        def observe(self, error, operation):
            observed.append(error)
            return super().observe(error, operation)

    with duckdb.connect(":memory:") as native:

        class Connection:
            failed = False

            def execute(self, sql, params=None):
                if sql.startswith("INSTALL fts"):
                    self.failed = True
                    raise primary
                if self.failed and sql.startswith("SET "):
                    cleanup_observed.extend(observed)
                    raise OSError("independent cleanup failure")
                return native.execute(sql) if params is None else native.execute(sql, params)

        observer = Observer()
        with pytest.raises(FtsSettingsRestoreError):
            observer.call_db(
                "fts",
                lambda: create_fts_indexes(
                    cast(Any, Connection()), FtsConfig(backend="duckdb"), observer=observer
                ),
            )
    assert cleanup_observed and cleanup_observed[0] is primary
    assert observed[0] is primary
