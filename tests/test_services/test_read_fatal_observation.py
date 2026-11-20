"""Read fallback policy cannot hide a terminal database failure."""

import asyncio
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest
from recall.db.connection import connect
from recall.services.self_repair import read_failure_marker

from tests.test_services.test_rpc_owned_boundary import server_at


class FailingContextCursor:
    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def execute(self, query: str, *args: Any) -> duckdb.DuckDBPyConnection:
        if "last_context_messages" in query:
            raise duckdb.FatalException("Failed to delete all rows from index in context stats")
        return self._connection.execute(query, *args)

    def cursor(self) -> Any:
        return FailingContextCursor(self._connection.cursor())

    def close(self) -> None:
        self._connection.close()

    def interrupt(self) -> None:
        self._connection.interrupt()


def test_stats_fallback_observes_and_propagates_origin_before_returning_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = server_at(tmp_path)
    monkeypatch.setattr(
        server._database,
        "_opener",
        lambda: cast(duckdb.DuckDBPyConnection, FailingContextCursor(connect(server._config))),
    )

    async def scenario() -> None:
        try:
            with pytest.raises(duckdb.FatalException, match="in context stats"):
                await server._handle_stats({}, None)
            marker = read_failure_marker(tmp_path)
            assert marker is not None and marker.failure is not None
            assert marker.failure.message == (
                "Failed to delete all rows from index in context stats"
            )
            assert server._shutdown_event.is_set()
        finally:
            await server.stop()

    asyncio.run(scenario())
