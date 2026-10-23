"""The public session selector has one meaning across RPC methods."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from lane_harness import build_lane
from recall.core.rpc_types import APP_NOT_FOUND, INVALID_PARAMS, RpcError
from recall.core.types import default_session_host


@pytest.mark.parametrize(
    "method", ["recall.list", "recall.live", "recall.search", "recall.live_mark"]
)
@pytest.mark.parametrize("identifier", ["source", "canonical", "prefix"])
def test_rpc_session_selector_names_only_the_requested_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, identifier: str
) -> None:
    lane = build_lane(tmp_path, monkeypatch)

    async def scenario() -> None:
        try:
            server = lane.server
            shown = await server._methods["recall.show"](
                {"session_id": "live-mid-tool", "tail": 1}, None
            )
            canonical = shown["id"]
            selector = {"source": "live-mid-tool", "canonical": canonical, "prefix": canonical[:8]}[
                identifier
            ]
            params = {"session": selector}
            if method == "recall.live":
                params["all"] = True
            elif method == "recall.search":
                params.update(query="suite", mode="keyword")
            elif method == "recall.live_mark":
                params["pid"] = os.getpid()
            result = await server._methods[method](params, None)
            if method == "recall.live_mark":
                assert result["source_session_id"] == "live-mid-tool"
                marks = (
                    server._get_conn()
                    .execute("SELECT source_session_id, host, pid FROM live_marks")
                    .fetchall()
                )
                assert marks == [("live-mid-tool", default_session_host(), os.getpid())]
            elif method == "recall.search":
                assert result
                assert {row.session_id for row in result} == {canonical}
            else:
                rows = result["sessions"] if method == "recall.live" else result
                assert [row.id for row in rows] == [canonical]
        finally:
            await lane.server.stop()

    asyncio.run(scenario())


def test_mark_excluded_hex_source_id_does_not_mark_a_local_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = build_lane(tmp_path, monkeypatch)

    async def scenario() -> None:
        try:
            conn = lane.server._get_conn()
            foreign = "aabbccddeeff00112233445566778899"
            local = "aabbccddffffffffffffffffffffffff"
            for sid, source_id, host in (
                (foreign, "aabbccdd", "foreign"),
                (local, "real-source", default_session_host()),
            ):
                conn.execute(
                    "INSERT INTO sessions (id, source, source_path, source_session_id) "
                    "VALUES (?, 'claude_code', ?, ?)",
                    [sid, str(tmp_path / f"{sid}.jsonl"), source_id],
                )
                conn.execute(
                    "INSERT INTO session_state "
                    "(session_id, host, file_mtime, file_size, message_count, tool_count) "
                    "VALUES (?, ?, 0, 0, 0, 0)",
                    [sid, host],
                )
            before = conn.execute("SELECT * FROM live_marks").fetchall()
            with pytest.raises(ValueError, match="outside the requested source/host scope"):
                await lane.server._methods["recall.live_mark"](
                    {"session": "aabbccdd", "pid": os.getpid()}, None
                )
            assert conn.execute("SELECT * FROM live_marks").fetchall() == before
        finally:
            await lane.server.stop()

    asyncio.run(scenario())


def test_show_ambiguity_names_candidate_sources_and_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = build_lane(tmp_path, monkeypatch)

    async def scenario() -> None:
        try:
            conn = lane.server._get_conn()
            conn.execute("UPDATE sessions SET source_session_id = 'shared'")
            conn.execute(
                "UPDATE session_state SET host = CASE WHEN session_id = "
                "(SELECT id FROM sessions WHERE source_path = ?) THEN 'host-a' ELSE 'host-b' END",
                [str(lane.watched)],
            )
            with pytest.raises(ValueError, match="ambiguous") as error:
                await lane.server._methods["recall.show"]({"session_id": "shared"}, None)
            assert "claude_code" in str(error.value)
            assert "host-a" in str(error.value)
            assert "host-b" in str(error.value)
        finally:
            await lane.server.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["recall.show", "recall.search", "recall.list", "recall.live"])
def test_unknown_indexed_selectors_are_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    lane = build_lane(tmp_path, monkeypatch)

    async def scenario() -> None:
        try:
            params = {"session_id" if method == "recall.show" else "session": "unknown-session"}
            if method == "recall.search":
                params.update(query="suite", mode="keyword")
            with pytest.raises(RpcError, match="session not found in indexed metadata") as error:
                await lane.server._methods[method](params, None)
            assert error.value.code == APP_NOT_FOUND
        finally:
            await lane.server.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "method", ["recall.list", "recall.live", "recall.search", "recall.live_mark"]
)
def test_source_and_host_scopes_disambiguate_and_reject_conflicting_canonical_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    lane = build_lane(tmp_path, monkeypatch)

    async def scenario() -> None:
        try:
            server = lane.server
            conn = server._get_conn()
            canonical = conn.execute(
                "SELECT id FROM sessions WHERE source_path = ?", [str(lane.watched)]
            ).fetchone()[0]
            conn.execute("UPDATE sessions SET source_session_id = 'shared'")
            conn.execute("UPDATE sessions SET source = 'codex' WHERE id != ?", [canonical])
            params = {"session": "shared"}
            if method == "recall.live":
                params["all"] = True
            elif method == "recall.search":
                params.update(query="suite", mode="keyword")
            elif method == "recall.live_mark":
                params["pid"] = os.getpid()
            with pytest.raises(ValueError, match="ambiguous"):
                await server._methods[method](params, None)
            if method == "recall.live_mark":
                assert conn.execute("SELECT count(*) FROM live_marks").fetchone()[0] == 0
            result = await server._methods[method]({**params, "source": "claude-code"}, None)
            if method == "recall.live_mark":
                assert result["source"] == "claude_code"
                assert result["source_session_id"] == "shared"
                inferred = await server._methods[method](
                    {
                        **params,
                        "session": conn.execute(
                            "SELECT id FROM sessions WHERE source='codex'"
                        ).fetchone()[0],
                    },
                    None,
                )
                assert inferred["source"] == "codex"
            elif method == "recall.search":
                assert result
                assert {row.session_id for row in result} == {canonical}
            else:
                rows = result["sessions"] if method == "recall.live" else result
                assert [row.id for row in rows] == [canonical]
            with pytest.raises(ValueError, match="outside the requested source/host scope"):
                await server._methods[method](
                    {**params, "session": canonical, "source": "codex"}, None
                )
            if method in {"recall.list", "recall.live"}:
                conn.execute(
                    "UPDATE session_state SET host='foreign' WHERE session_id != ?", [canonical]
                )
                result = await server._methods[method](
                    {**params, "host": default_session_host()}, None
                )
                rows = result["sessions"] if method == "recall.live" else result
                assert [row.id for row in rows] == [canonical]
        finally:
            await lane.server.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["recall.list_by_session", "recall.live_by_session"])
@pytest.mark.parametrize("params", [{}, {"session": None}, {"session": 42}])
def test_filtered_roster_requires_a_session_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, params: dict
) -> None:
    lane = build_lane(tmp_path, monkeypatch)

    async def scenario() -> None:
        try:
            with pytest.raises(RpcError, match="session") as error:
                await lane.server._methods[method](params, None)
            assert error.value.code == INVALID_PARAMS
        finally:
            await lane.server.stop()

    asyncio.run(scenario())
