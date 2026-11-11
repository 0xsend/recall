"""Optional typed host metadata never replaces a daemon's legacy error."""

from __future__ import annotations

import json
import socket
import threading

import pytest
from recall.cli.app import app
from recall.core.rpc_client import RpcCallError, RpcClient
from typer.testing import CliRunner

RAW = "session host=edge\nforged"
PARTS = ["session host=", {"value": "edge\nforged"}]
LARGE_HOST = "x" * 32768 + "\nforged"


@pytest.mark.parametrize(
    ("kind", "raw", "metadata", "expected_text"),
    [
        pytest.param("valid", RAW, {"parts": PARTS}, 'session host="edge\\nforged"', id="valid"),
        pytest.param("malformed", RAW, {"parts": [123]}, RAW, id="malformed"),
        pytest.param("mismatching", RAW, {"parts": ["different"]}, RAW, id="mismatching"),
        pytest.param("absent", RAW, None, RAW, id="absent"),
        pytest.param("legacy", RAW, None, RAW, id="legacy"),
        *[
            pytest.param("invalid", RAW, value, RAW, id=name)
            for name, value in [
                ("top-null", None),
                ("top-string", "metadata"),
                ("top-array", PARTS),
                ("top-number", 1),
                ("top-bool", True),
                ("missing-parts", {}),
                ("parts-null", {"parts": None}),
                ("parts-object", {"parts": {"value": "edge\nforged"}}),
                ("parts-string", {"parts": RAW}),
                ("part-null", {"parts": [None]}),
                ("part-bool", {"parts": [True]}),
                ("part-array", {"parts": [PARTS]}),
                ("part-missing-value", {"parts": ["session host=", {}]}),
                ("top-extra", {"parts": PARTS, "undeclared": True}),
                (
                    "part-extra",
                    {"parts": ["session host=", {"value": "edge\nforged", "undeclared": True}]},
                ),
                (
                    "quoted-string",
                    {"parts": ["session host=", {"value": "edge\nforged", "quoted": "false"}]},
                ),
                (
                    "quoted-zero",
                    {"parts": ["session host=", {"value": "edge\nforged", "quoted": 0}]},
                ),
                (
                    "quoted-null",
                    {"parts": ["session host=", {"value": "edge\nforged", "quoted": None}]},
                ),
                (
                    "quoted-object",
                    {
                        "parts": [
                            "session host=",
                            {"value": "edge\nforged", "quoted": {"nested": False}},
                        ]
                    },
                ),
                (
                    "nested-value",
                    {"parts": ["session host=", {"value": {"value": "edge\nforged"}}]},
                ),
                (
                    "nested-extra",
                    {
                        "parts": [
                            "session host=",
                            {"value": "edge\nforged", "nested": {"unexpected": True}},
                        ],
                        "undeclared": True,
                    },
                ),
            ]
        ],
        pytest.param(
            "valid",
            "session host='edge'",
            {"parts": ["session host=", {"value": "edge", "quoted": True}]},
            "session host='edge'",
            id="quoted-true",
        ),
        pytest.param(
            "valid",
            RAW,
            {"parts": ["session host=", {"value": "edge\nforged", "quoted": False}]},
            'session host="edge\\nforged"',
            id="quoted-false",
        ),
        pytest.param(
            "invalid",
            "session host='edge'",
            {"parts": ["session host=", {"value": "edge", "quoted": 1}]},
            "session host='edge'",
            id="quoted-one",
        ),
        pytest.param(
            "invalid",
            "session host='edge'",
            {"parts": ["session host=", {"value": "edge", "quoted": "true"}]},
            "session host='edge'",
            id="quoted-string-true",
        ),
        pytest.param(
            "valid",
            "session host=" + LARGE_HOST,
            {"parts": ["session host=", {"value": LARGE_HOST}]},
            'session host="' + "x" * 32768 + '\\nforged"',
            id="large-matching",
        ),
        pytest.param(
            "mismatching",
            RAW,
            {"parts": ["session host=", {"value": LARGE_HOST}]},
            RAW,
            id="large-mismatching",
        ),
        pytest.param(
            "mismatching",
            "session host=" + LARGE_HOST,
            {"parts": PARTS},
            "session host=" + LARGE_HOST,
            id="large-error-mismatching",
        ),
    ],
)
@pytest.mark.parametrize("surface", ["rpc", "text", "json"])
def test_optional_host_error_metadata(
    socket_dir, monkeypatch, kind, raw, metadata, expected_text, surface
):
    monkeypatch.setenv("RECALL_DATA_DIR", str(socket_dir))
    monkeypatch.setattr("recall.cli.rpc._in_process_server", None)
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
                    response = {
                        "id": request["id"],
                        "error": {"code": -32602, "message": raw, "data": {"reason": "scope"}},
                    }
                    if kind != "legacy":
                        response["jsonrpc"] = "2.0"
                    if kind not in {"absent", "legacy"}:
                        response["_host_message"] = metadata
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
                assert error.value.data == {"reason": "scope"}
                model = error.value.host_message
                assert (model is not None) == (kind == "valid")
                if model is not None:
                    assert model.text() == expected_text
                assert client.call("recall.stats") == "alive"
            else:
                result = CliRunner().invoke(app, ["show", "shared", "--format", surface])
                assert result.exit_code == 2
                if surface == "json":
                    assert result.stderr_bytes == b""
                    assert json.loads(result.stdout) == {
                        "error": {
                            "code": "VALIDATION",
                            "message": raw,
                            "details": {"reason": "scope"},
                        }
                    }
                else:
                    assert result.stdout_bytes == b""
                    assert result.stderr_bytes == ("error: " + expected_text + "\n").encode()
        finally:
            client.close()
            worker.join(timeout=4)
            assert not worker.is_alive()
            assert failures == []
