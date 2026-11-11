"""Optional typed host metadata never replaces a daemon's legacy error."""

from __future__ import annotations

import json
import socket
import threading

import pytest
from recall.cli.app import app
from recall.core.rpc_client import RpcCallError, RpcClient
from typer.testing import CliRunner


@pytest.mark.parametrize("kind", ["valid", "malformed", "mismatching", "absent", "legacy"])
@pytest.mark.parametrize("surface", ["rpc", "text", "json"])
def test_optional_host_error_metadata(socket_dir, monkeypatch, kind, surface):
    monkeypatch.setenv("RECALL_DATA_DIR", str(socket_dir))
    monkeypatch.setattr("recall.cli.rpc._in_process_server", None)
    raw = "session host=edge\nforged"
    parts = {"parts": ["session host=", {"value": "edge\nforged"}]}
    failures = []
    client = RpcClient()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(client.socket_path))
        listener.listen(1)
        listener.settimeout(3)

        def reply():
            try:
                peer, _ = listener.accept()
                with peer, peer.makefile("rb") as reader:
                    peer.settimeout(3)
                    request = json.loads(reader.readline())
                    response = {"id": request["id"], "error": {"code": -32602, "message": raw}}
                    if kind != "legacy":
                        response["jsonrpc"] = "2.0"
                    if kind == "valid":
                        response["_host_message"] = parts
                    elif kind == "malformed":
                        response["_host_message"] = {"parts": [123]}
                    elif kind == "mismatching":
                        response["_host_message"] = {"parts": ["different"]}
                    peer.sendall(json.dumps(response).encode() + b"\n")
                    if surface == "rpc":
                        # A metadata error must leave the connection reusable.
                        request = json.loads(reader.readline())
                        peer.sendall(
                            json.dumps({"id": request["id"], "result": "alive"}).encode() + b"\n"
                        )
            except Exception as err:
                failures.append(err)

        worker = threading.Thread(target=reply)
        worker.start()
        try:
            if surface == "rpc":
                client.connect(auto_fork=False)
                with pytest.raises(RpcCallError) as error:
                    client.call("recall.show", {"session_id": "shared"})
                assert error.value.code == -32602
                assert error.value.message == raw
                model = error.value.host_message
                assert (model is not None) == (kind == "valid")
                if model is not None:
                    assert model.text() == 'session host="edge\\nforged"'
                assert client.call("recall.stats") == "alive"
            else:
                result = CliRunner().invoke(app, ["show", "shared", "--format", surface])
                assert result.exit_code == 2
                if surface == "json":
                    assert result.stderr_bytes == b""
                    assert json.loads(result.stdout) == {
                        "error": {"code": "VALIDATION", "message": raw}
                    }
                else:
                    expected = 'session host="edge\\nforged"' if kind == "valid" else raw
                    assert result.stdout_bytes == b""
                    assert result.stderr_bytes == ("error: " + expected + "\n").encode()
        finally:
            client.close()
            worker.join(timeout=4)
            assert not worker.is_alive()
            assert failures == []
