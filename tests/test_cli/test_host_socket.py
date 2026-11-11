"""Resolver refusals cross the real socket in follow and mark consumers."""

from __future__ import annotations

import asyncio
import os

import pytest
from recall.cli import rpc
from recall.cli.app import app
from typer.testing import CliRunner

CASES = [
    ("edge-1", "edge-1"),
    ("edge\nforged", '"edge\\nforged"'),
    ("edge\rforged", '"edge\\rforged"'),
    ("edge\x1b[31m", '"edge\\u001b[31m"'),
    ("edge host=trusted", '"edge host=trusted"'),
]


@pytest.mark.parametrize("origin,encoded", CASES)
@pytest.mark.parametrize("consumer", ["follow", "mark"])
def test_host_refusal_over_socket(socket_dir, monkeypatch, origin, encoded, consumer):
    from recall.core.config import AppConfig
    from recall.services.rpc_server import RpcServer

    monkeypatch.setenv("RECALL_DATA_DIR", str(socket_dir))
    server = RpcServer(config=AppConfig.load())
    monkeypatch.setattr(rpc, "_in_process_server", None)
    conn = server._get_conn()
    for sid, host in [("first", origin), ("second", "trusted")]:
        conn.execute(
            "INSERT INTO sessions (id,source,source_path,source_session_id) "
            "VALUES (?, 'codex', ?, 'shared')",
            [sid, sid + ".fixture"],
        )
        conn.execute(
            "INSERT INTO session_state (session_id,host,file_mtime,file_size) VALUES (?, ?, 0, 0)",
            [sid, host],
        )
    args = (
        ["show", "shared", "--follow"]
        if consumer == "follow"
        else ["live", "mark", "--session", "first", "--pid", str(os.getpid())]
    )
    prefix = (
        "ambiguous source_session_id: shared matches multiple sessions: first (source=codex, host="
        if consumer == "follow"
        else "session first (source=codex, host="
    )
    suffix = (
        "), second (source=codex, host=trusted); "
        "use an exact recall session id or narrow source/host filters"
        if consumer == "follow"
        else ") is outside the requested source/host scope"
    )

    async def scenario():
        loop = asyncio.get_running_loop()
        listener = await loop.create_unix_server(
            server._connection_protocol, path=str(socket_dir / "recall.sock")
        )
        try:
            text = await asyncio.to_thread(CliRunner().invoke, app, [*args, "--format", "text"])
            structured = await asyncio.to_thread(CliRunner().invoke, app, [*args, "--json"])
            return text, structured
        finally:
            listener.close()
            await listener.wait_closed()
            await server.stop()

    text, structured = asyncio.run(scenario())
    assert text.exit_code == structured.exit_code == 2, (text.output, structured.output)
    assert text.stdout_bytes == b""
    assert text.stderr_bytes == ("error: " + prefix + encoded + suffix + "\n").encode()
    assert structured.stderr_bytes == b""
    import json

    assert json.loads(structured.stdout) == {
        "error": {"code": "VALIDATION", "message": prefix + origin + suffix}
    }
    # Open again after the request and shutdown; refusals cannot leave a mark.
    from recall.db.connection import connect

    with connect(server._config) as read:
        assert read.execute("SELECT count(*) FROM live_marks").fetchone() == (0,)
