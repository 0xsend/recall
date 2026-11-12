"""Startup proves local prerequisites, while selected work proves model weights."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pytest

from tests.test_cli.test_context_availability import IDLE_HOST_CLI, _config, _incoming, _rpc


def _environment(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    for key, value in {
        "RECALL_CONFIG_PATH": str(root / "config.toml"),
        "RECALL_DATA_DIR": str(root),
        "RECALL_DB_PATH": str(root / "recall.duckdb"),
        "RECALL_LOCK_PATH": str(root / "recall.lock"),
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(root / "modules") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }.items():
        monkeypatch.setenv(key, value)
    for key in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    return dict(os.environ)


def _dependency(root: Path, failure: str) -> None:
    modules = root / "modules"
    modules.mkdir()
    # The real import/load boundaries are exercised; native MLX itself is a double.
    (modules / "mlx_lm.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        f"root = Path({str(root)!r})\n"
        "(root/'imported').write_text(str(os.getpid()))\n"
        + ("while True: time.sleep(1)\n" if failure == "import-stall" else "")
        + "def load(model):\n"
        "    with (root/'loads').open('a') as f: f.write(str(time.monotonic())+'\\n')\n"
        "    if (root/'repaired').exists(): return object(), object()\n"
        + (
            "    while True: time.sleep(1)\n"
            if failure == "load-stall"
            else f"    raise RuntimeError({failure!r})\n"
        )
        + "def generate(*args, **kwargs): return 'Recovered local context'\n"
    )


@contextmanager
def _process(root: Path, env: dict[str, str]) -> Iterator[subprocess.Popen[str]]:
    with (root / "stdout").open("w") as out, (root / "stderr").open("w") as err:
        child = subprocess.Popen(
            [str(Path(sys.executable).parent / "recall"), "daemon", "--mode", "poll"],
            env=env,
            stdout=out,
            stderr=err,
            text=True,
            start_new_session=True,
        )
        try:
            yield child
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)


def _ready(root: Path, child: subprocess.Popen[str]) -> dict:
    deadline = time.monotonic() + 8
    while child.poll() is None and time.monotonic() < deadline:
        try:
            status = _rpc(root, "recall.daemon_status")["result"]
            if status["daemon_pid"] == child.pid and status["reconciliation"]["raw_indexing_ready"]:
                return status
        except (OSError, KeyError, ValueError):
            pass
        time.sleep(0.025)
    pytest.fail((root / "stderr").read_text())


@pytest.mark.parametrize("failure", ["absent weights", "corrupt weights", "load-stall"])
def test_local_startup_and_status_do_not_load_weights(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    root = socket_dir
    _dependency(root, failure)
    (root / "config.toml").write_text(_config("llm-local"))
    with _process(root, _environment(root, monkeypatch)) as child:
        status = _ready(root, child)
        assert status["startup_refusal"] is None
        assert not (root / "loads").exists()
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=3) == 0


def test_stalled_dependency_import_refuses_within_startup_bound(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = socket_dir
    _dependency(root, "import-stall")
    (root / "config.toml").write_text(_config("llm-local"))
    with _process(root, _environment(root, monkeypatch)) as child:
        child.wait(timeout=8)
        assert child.returncode == 3
        error = (root / "stderr").read_text()
        assert "llm-local" in error and "5" in error and "timed out" in error
        assert not (root / "recall.duckdb").exists()
        assert not (root / "recall.sock").exists()
        imported_pid = int((root / "imported").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(imported_pid, 0)


def test_stalled_startup_import_cancellation_reaps_probe(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = socket_dir
    _dependency(root, "import-stall")
    (root / "config.toml").write_text(_config("llm-local"))
    with _process(root, _environment(root, monkeypatch)) as child:
        deadline = time.monotonic() + 5
        while not (root / "imported").exists():
            assert child.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.025)
        imported_pid = int((root / "imported").read_text())
        child.send_signal(signal.SIGINT)
        child.wait(timeout=2)
        assert not (root / "recall.duckdb").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(imported_pid, 0)


@pytest.mark.parametrize("structured", [False, True])
def test_local_weight_failure_refuses_selected_work_and_retry_recovers(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, structured: bool
) -> None:
    root = socket_dir
    _dependency(root, "corrupt weights")
    (root / "config.toml").write_text(_config("llm-local"))
    env = _environment(root, monkeypatch)
    alternate = _incoming(root)
    args = ["index", "--no-embed", "--root", str(alternate), "--source", "claude-code"]
    output = ["--json"] if structured else ["--format", "text"]
    binary = str(Path(sys.executable).parent / "recall")
    with _process(root, env) as child:
        _ready(root, child)
        for _ in range(2):
            refusal = subprocess.run(
                [binary, *args, *output], env=env, capture_output=True, text=True, timeout=10
            )
            assert refusal.returncode == 1
            message = refusal.stderr
            if structured:
                error = json.loads(refusal.stdout)["error"]
                assert error["code"] == "RUNTIME"
                assert error["details"]["raw_history_committed"] is True
                assert error["details"]["enrichment"] == "refused"
                assert error["details"]["retryable"] is True
                message = error["message"]
            assert "corrupt weights" in message and "retry" in message
            assert "restart the daemon" not in message
        (root / "repaired").touch()
        recovered = subprocess.run(
            [binary, *args, "--json"], env=env, capture_output=True, text=True, timeout=10
        )
        assert recovered.returncode == 0, recovered.stdout + recovered.stderr
        assert json.loads(recovered.stdout)["context_messages"] == 1
    with duckdb.connect(str(root / "recall.duckdb"), read_only=True) as conn:
        assert conn.execute("SELECT context_mode,context_text FROM message_state").fetchall() == [
            ("llm-local", "[Recovered local context] ")
        ]


@pytest.mark.parametrize("structured", [False, True])
def test_once_weight_failure_reports_raw_progress(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, structured: bool
) -> None:
    root = socket_dir
    _dependency(root, "absent weights")
    incoming = _incoming(root) / ".claude/projects/example/incoming.jsonl"
    os.utime(incoming, (1, 1))
    config = (
        _config("llm-local")
        .replace(
            "[sources.claude_code]\nroots = []",
            f'[sources.claude_code]\nroots = ["{root / "import-home/.claude/projects"}"]',
        )
        .replace("embed = false", "embed = false\nembed_idle_session = 1")
    )
    (root / "config.toml").write_text(config)
    result = subprocess.run(
        [
            *IDLE_HOST_CLI,
            "daemon",
            "--once",
            "--embed",
            *(["--json"] if structured else ["--format", "text"]),
        ],
        env=_environment(root, monkeypatch),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1
    message = result.stderr
    if structured:
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "RUNTIME"
        assert error["details"]["raw_history_committed"] is True
        assert error["details"]["retryable"] is True
        assert error["details"]["enrichment"] == "refused"
        assert error["details"]["index_summary"]["indexed"] == 1
        message = error["message"]
    assert "absent weights" in message and "Raw history" in message and "retry" in message
    assert not (root / "recall.sock").exists()


def test_background_weight_failure_is_visible_and_paced(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = socket_dir
    _dependency(root, "corrupt weights")
    incoming = _incoming(root) / ".claude/projects/example/incoming.jsonl"
    os.utime(incoming, (time.time() - 60, time.time() - 60))
    config = (
        _config("llm-local", embed=True)
        .replace(
            "embed = true",
            "embed = true\nembed_idle_session = 1\nembed_backoff = 2\nload_threshold = 99999",
        )
        .replace(
            "[sources.claude_code]\nroots = []",
            f'[sources.claude_code]\nroots = ["{root / "import-home/.claude/projects"}"]',
        )
    )
    (root / "config.toml").write_text(config)
    with _process(root, _environment(root, monkeypatch)) as child:
        _ready(root, child)
        deadline = time.monotonic() + 20
        while True:
            status = _rpc(root, "recall.daemon_status")["result"]
            loads = (root / "loads").read_text().splitlines() if (root / "loads").exists() else []
            if len(loads) >= 2 and status.get("embed_last_error"):
                break
            assert time.monotonic() < deadline, status
            time.sleep(0.05)
        assert "corrupt weights" in status["embed_last_error"]
        assert status["embed_loop_last_outcome"] == "error"
        assert status["embed_loop_next_interval"] == 2
        assert float(loads[1]) - float(loads[0]) >= 2
    with duckdb.connect(str(root / "recall.duckdb"), read_only=True) as conn:
        assert conn.execute("SELECT context_mode FROM message_state").fetchall() == [("off",)]
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (0,)


@pytest.mark.parametrize("failure", ["absent weights", "corrupt weights"])
def test_weight_refusal_retains_raw_template_without_llm_context_or_vectors(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    root = socket_dir
    _dependency(root, failure)
    (root / "config.toml").write_text(_config("llm-local"))
    alternate = _incoming(root)
    incoming = alternate / ".claude/projects/example/incoming.jsonl"
    entry = json.loads(incoming.read_text())
    entry.update(cwd=str(root / "project"), git_root="fixture-project", gitBranch="refusal-fixture")
    incoming.write_text(json.dumps(entry) + "\n")
    env = _environment(root, monkeypatch)
    with _process(root, env) as child:
        _ready(root, child)
        result = subprocess.run(
            [
                str(Path(sys.executable).parent / "recall"),
                "index",
                "--no-embed",
                "--root",
                str(alternate),
                "--source",
                "claude-code",
                "--json",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1
        assert failure in json.loads(result.stdout)["error"]["message"]
    with duckdb.connect(str(root / "recall.duckdb"), read_only=True) as conn:
        assert conn.execute("SELECT context_mode,context_text FROM message_state").fetchall() == [
            ("template", "[fixture-project refusal-fixture] ")
        ]
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (0,)
        assert conn.execute(
            "SELECT COUNT(*) FROM message_state WHERE context_mode='llm-local'"
        ).fetchone() == (0,)
