"""A per-source fail-soft catch cannot turn an index fatal into a completed raw turn."""

import asyncio
from pathlib import Path
from typing import Any

import duckdb
import pytest
from recall.parsers.codex import CodexParser
from recall.services import indexer
from recall.services.coordinator import capture_path, observe_path, prepare_raw_sources
from recall.services.self_repair import read_failure_marker

from tests.test_services.test_rpc_owned_boundary import server_at


def test_raw_commit_observes_origin_before_rollback_and_catalog_failure_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = server_at(tmp_path)
    path = tmp_path / "rollout-source.jsonl"
    path.write_text(
        '{"type":"session_meta","payload":{"id":"source"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user","content":"hello"}}\n'
    )
    parser = CodexParser(roots=(tmp_path,))
    item = server._database.call(
        "fixture:observe", lambda conn: observe_path(parser, capture_path(parser, path), conn=conn)
    )
    prepared = prepare_raw_sources((item,), {"codex": parser})
    assert prepared[0].error is None
    original = duckdb.FatalException("Failed to delete all rows from index in source insert")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise original

    monkeypatch.setattr(indexer, "insert_session", fail)

    async def scenario() -> None:
        try:
            with pytest.raises(duckdb.FatalException, match="in source insert"):
                await server._writer_call(
                    lambda: server._commit_raw_sources(prepared, server._config), "source_commit"
                )
            marker = read_failure_marker(tmp_path)
            assert marker is not None and marker.failure is not None
            assert marker.failure.message == str(original)
            assert server._shutdown_event.is_set()
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(server._config.db_path), read_only=True) as fresh:
        assert fresh.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


def test_library_index_cannot_report_partial_success_after_originating_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.core.types import Source
    from recall.services.indexer import DiscoveredSessionPath, index_sessions

    server = server_at(tmp_path)
    path = tmp_path / "rollout-library.jsonl"
    path.write_text('{"type":"session_meta","payload":{"id":"library"}}\n')
    parser = CodexParser(roots=(tmp_path,))
    stat = path.stat()
    discovered = DiscoveredSessionPath(
        parser=parser,
        path=path,
        resolved_path=str(path),
        file_mtime=stat.st_mtime,
        file_size=stat.st_size,
    )
    monkeypatch.setattr(indexer, "_discover_paths", lambda *args, **kwargs: [discovered])

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise duckdb.FatalException("Failed to delete all rows from index in library insert")

    monkeypatch.setattr(indexer, "insert_session", fail)
    with pytest.raises(duckdb.FatalException, match="in library insert"):
        index_sessions(
            source=Source.CODEX,
            full=False,
            recreate=False,
            verbose=False,
            config=server._config,
        )
    asyncio.run(server.stop())
