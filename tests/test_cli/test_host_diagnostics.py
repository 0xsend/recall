"""Host values stay raw in structured refusals and encoded in terminal text."""

from __future__ import annotations

import json

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
@pytest.mark.parametrize(
    "command,scope",
    [
        (command, scope)
        for scope in (False, True)
        for command in ("show", "list", "search", "live")
        if not (scope and command == "show")
    ],
)
def test_session_diagnostic_host_text(origin, encoded, command, scope):
    server = rpc._in_process_server._real
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
    selector = "first" if scope else "shared"
    args = (
        ["show", selector]
        if command == "show"
        else ["search", "needle", "--mode", "keyword", "--session", selector]
        if command == "search"
        else [command, "--session", selector]
    )
    if scope:
        args += ["--source", "claude_code"] if command == "search" else ["--host", "other"]
    prefix = (
        "session first (source=codex, host="
        if scope
        else (
            "ambiguous source_session_id: shared matches multiple sessions: "
            "first (source=codex, host="
        )
    )
    suffix = (
        ") is outside the requested source/host scope"
        if scope
        else (
            "), second (source=codex, host=trusted); "
            "use an exact recall session id or narrow source/host filters"
        )
    )
    text = CliRunner().invoke(app, [*args, "--format", "text"])
    structured = CliRunner().invoke(app, [*args, "--json"])
    assert text.exit_code == structured.exit_code == 2
    assert text.stdout_bytes == b""
    assert text.stderr_bytes == ("error: " + prefix + encoded + suffix + "\n").encode()
    assert text.stderr_bytes.count(b"\n") == 1
    assert structured.stderr_bytes == b""
    assert json.loads(structured.stdout) == {
        "error": {"code": "VALIDATION", "message": prefix + origin + suffix}
    }
