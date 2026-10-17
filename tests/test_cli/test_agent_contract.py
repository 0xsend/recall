from __future__ import annotations

import json
import shutil
from importlib.metadata import version as package_version
from pathlib import Path

import pytest
from conftest import _can_acquire_duckdb_lock, set_in_process_server
from recall.cli.app import app
from recall.services.coordinator import IndexRequestScope
from typer.testing import CliRunner

_requires_duckdb_lock = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)


def _install_fixtures(tmp_path: Path) -> None:
    claude_target = tmp_path / ".claude" / "projects" / "proj1"
    codex_target = tmp_path / ".codex" / "sessions" / "s1"
    claude_target.mkdir(parents=True)
    codex_target.mkdir(parents=True)

    root = Path(__file__).resolve().parents[2] / "fixtures"
    shutil.copy(root / "claude_code" / "session1.jsonl", claude_target / "session1.jsonl")
    shutil.copy(root / "codex" / "session1" / "rollout.jsonl", codex_target / "rollout.jsonl")


def _init_index(tmp_path: Path, monkeypatch) -> CliRunner:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    _install_fixtures(tmp_path)

    # The autouse RPC fixture creates its server lazily after these environment
    # patches and owns teardown. Replacing it here leaks the explicit server's
    # DuckDB connection across tests, which can race later Arrow ingestion.

    runner = CliRunner()
    # Output-contract fixtures require history, not a downloaded embedding model.
    result = runner.invoke(app, ["index", "--full", "--no-embed", "--yes"])
    assert result.exit_code == 0, f"index failed: {result.output}"
    return runner


@_requires_duckdb_lock
def test_non_tty_list_defaults_to_toon(tmp_path, monkeypatch) -> None:
    """REQ-CLI-013: non-TTY defaults to TOON output."""
    from recall.cli import rpc as rpc_module

    fixture_server = rpc_module._in_process_server
    runner = _init_index(tmp_path, monkeypatch)

    # The autouse fixture owns the server lifecycle and must retain teardown.
    assert rpc_module._in_process_server is fixture_server

    result = runner.invoke(app, ["list", "--limit", "2"])

    assert result.exit_code == 0
    # TOON output is not JSON — verify it's parseable as TOON
    from toon_format import decode as toon_decode

    payload = toon_decode(result.stdout)
    assert isinstance(payload, list)
    assert len(payload) == 2


def test_cli_exposes_llms_manifest() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--llms"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["name"] == "recall"
    assert payload["commands"]["index"]["safety"]["mutates"] is True
    assert payload["commands"]["index"]["safety"]["destructive"] is False


def test_cli_exposes_version() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == package_version("recall")


def test_cli_version_takes_precedence_over_llms() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--version", "--llms"])

    assert result.exit_code == 0
    assert result.stdout.strip() == package_version("recall")


def test_cli_exposes_targeted_command_schema() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["schema", "daemon", "install"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "daemon install"
    assert payload["safety"]["mutates"] is True
    assert payload["safety"]["idempotent"] is True
    assert any(option["name"] == "scheduler" for option in payload["options"])


def test_cli_exposes_skill_census_schema() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["schema", "stats", "skills"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "stats skills"
    assert payload["safety"] == {
        "mutates": False,
        "destructive": False,
        "idempotent": True,
    }
    assert payload["output"]["fields"] == ["rows", "coverage"]
    assert any(
        option["name"] == "source" and option["type"] == "string[]" for option in payload["options"]
    )


def test_schema_daemon_start_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["schema", "daemon", "start"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "daemon start"
    assert payload["safety"]["mutates"] is True
    assert "cta_envelope" in payload["output"]
    assert any(opt["name"] == "background" for opt in payload["options"])


def test_schema_daemon_stop_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["schema", "daemon", "stop"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "daemon stop"
    assert payload["safety"]["idempotent"] is True
    assert "cta_envelope" in payload["output"]
    assert any(opt["name"] == "soft" for opt in payload["options"])


def test_schema_daemon_restart_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["schema", "daemon", "restart"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "daemon restart"
    assert "cta_envelope" in payload["output"]
    assert any(opt["name"] == "timeout" for opt in payload["options"])


def test_daemon_stop_structured_output(tmp_path, monkeypatch) -> None:
    """daemon stop --json returns durable lifecycle status when no daemon is running."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

    from recall.services.daemon import DaemonStopResult

    monkeypatch.setattr(
        "recall.cli.daemon.stop_daemon_durable",
        lambda _config, *, timeout: DaemonStopResult(
            scheduler="launchd",
            stopped=True,
            pid=None,
            duration_seconds=0.0,
            message="daemon already stopped",
        ),
    )

    runner = CliRunner()
    result = runner.invoke(app, ["daemon", "stop", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["scheduler"] == "launchd"
    assert payload["stopped"] is True
    assert payload["pid"] is None


def test_daemon_start_structured_durable_output(tmp_path, monkeypatch) -> None:
    """daemon start --json returns durable lifecycle status."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

    from recall.services.daemon import DaemonStartResult

    monkeypatch.setattr(
        "recall.cli.daemon.start_daemon_durable",
        lambda _config, *, timeout: DaemonStartResult(
            scheduler="launchd",
            started=True,
            pid=12345,
            duration_seconds=0.1,
            message=None,
        ),
    )

    runner = CliRunner()
    result = runner.invoke(app, ["daemon", "start", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["scheduler"] == "launchd"
    assert payload["started"] is True
    assert payload["pid"] == 12345


def test_daemon_start_structured_output(tmp_path, monkeypatch) -> None:
    """daemon start --background --json returns structured success data."""
    monkeypatch.setenv("HOME", str(tmp_path))
    data_dir = tmp_path / ".local/share/recall"
    monkeypatch.setenv("RECALL_DATA_DIR", str(data_dir))

    from recall.cli.daemon import _BackgroundResult

    def fake_start_background(_config, _verbose):
        # Simulate what the real _start_background does
        log_dir = data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        # Write a PID file so _wait_for_pid_file succeeds
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "recall.pid").write_text("12345")
        return _BackgroundResult(pid=12345, log_path=str(log_dir / "daemon.log"))

    monkeypatch.setattr("recall.cli.daemon._start_background", fake_start_background)

    runner = CliRunner()
    result = runner.invoke(app, ["daemon", "start", "--background", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["started"] is True
    assert payload["scheduler"] == "pid"
    assert payload["pid"] == 12345
    assert "legacy background start" in payload["message"]


def test_daemon_lifecycle_cta_envelope(tmp_path, monkeypatch) -> None:
    """daemon start --background --json --cta wraps success in CTA envelope."""
    monkeypatch.setenv("HOME", str(tmp_path))
    data_dir = tmp_path / ".local/share/recall"
    monkeypatch.setenv("RECALL_DATA_DIR", str(data_dir))

    from recall.cli.daemon import _BackgroundResult

    def fake_start_background(_config, _verbose):
        log_dir = data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "recall.pid").write_text("99999")
        return _BackgroundResult(pid=99999, log_path=str(log_dir / "daemon.log"))

    monkeypatch.setattr("recall.cli.daemon._start_background", fake_start_background)

    runner = CliRunner()
    result = runner.invoke(app, ["daemon", "start", "--background", "--json", "--cta"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "data" in payload
    assert "cta" in payload
    assert payload["data"]["started"] is True
    assert isinstance(payload["cta"], list)
    assert len(payload["cta"]) > 0


def test_daemon_lifecycle_start_then_stop_cta(tmp_path, monkeypatch) -> None:
    """E2E: start --background --json then stop --json --cta returns CTA envelope."""
    monkeypatch.setenv("HOME", str(tmp_path))
    data_dir = tmp_path / ".local/share/recall"
    monkeypatch.setenv("RECALL_DATA_DIR", str(data_dir))

    from recall.cli.daemon import _BackgroundResult

    def fake_start_background(_config, _verbose):
        log_dir = data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "recall.pid").write_text("55555")
        return _BackgroundResult(pid=55555, log_path=str(log_dir / "daemon.log"))

    monkeypatch.setattr("recall.cli.daemon._start_background", fake_start_background)

    # Patch os.kill to simulate a running process that stops on SIGTERM
    kill_calls: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        kill_calls.append((pid, sig))
        if sig == 0 and len([c for c in kill_calls if c[1] == 0]) > 1:
            raise ProcessLookupError  # Process exited after first check

    monkeypatch.setattr("os.kill", fake_kill)

    runner = CliRunner()

    # Start
    start_result = runner.invoke(app, ["daemon", "start", "--background", "--json"])
    assert start_result.exit_code == 0
    start_payload = json.loads(start_result.stdout)
    assert start_payload["started"] is True
    assert start_payload["scheduler"] == "pid"
    assert start_payload["pid"] == 55555

    # Stop with CTA
    stop_result = runner.invoke(app, ["daemon", "stop", "--soft", "--json", "--cta"])
    assert stop_result.exit_code == 0
    stop_payload = json.loads(stop_result.stdout)
    assert "data" in stop_payload
    assert "cta" in stop_payload
    assert stop_payload["data"]["stopped"] is True
    assert stop_payload["data"]["scheduler"] == "sentinel"
    assert stop_payload["data"]["pid"] == 55555
    assert len(stop_payload["cta"]) > 0
    cta_commands = [cta["command"] for cta in stop_payload["cta"]]
    assert "recall daemon start" in cta_commands
    assert not any("--background" in command for command in cta_commands)


def test_daemon_status_falls_back_without_auto_fork(monkeypatch) -> None:
    from recall.cli.contract import CliError, ErrorCode

    def fake_rpc_call(_method, _params, **kwargs):
        assert kwargs["auto_fork"] is False
        raise CliError(code=ErrorCode.RUNTIME, message="daemon not running")

    monkeypatch.setattr("recall.cli.daemon.rpc_call_or_error", fake_rpc_call)
    monkeypatch.setattr(
        "recall.cli.daemon.daemon_status",
        lambda: {
            "configured_scheduler": "auto",
            "scheduler": None,
            "installed": False,
            "command": "/tmp/recall daemon --once",
            "config_path": "/tmp/config.toml",
            "artifact_paths": [],
            "runtime_status": {},
            "mode": "poll",
            "resolved_mode": "poll",
            "daemon_version": None,
            "binary_version": "0.10.4",
            "version_drift": False,
            "embed_phase_enabled": False,
            "embed_model_loaded": False,
            "embed_pending": 0,
        },
    )
    monkeypatch.setattr(
        "recall.cli.daemon.AppConfig.load",
        lambda: type("Config", (), {"daemon": type("Daemon", (), {"interval": 300})()})(),
    )

    runner = CliRunner()
    result = runner.invoke(app, ["daemon", "status", "--format", "text"])

    assert result.exit_code == 0
    assert "Version: binary=0.10.4 daemon=unknown drift=n/a" in result.stdout


def test_daemon_restart_surfaces_failed_stop(tmp_path, monkeypatch) -> None:
    """restart --json surfaces failed durable stop instead of starting."""
    monkeypatch.setenv("HOME", str(tmp_path))
    data_dir = tmp_path / ".local/share/recall"
    data_dir.mkdir(parents=True)
    monkeypatch.setenv("RECALL_DATA_DIR", str(data_dir))

    from recall.services.daemon import DaemonRestartResult

    monkeypatch.setattr(
        "recall.cli.daemon.restart_daemon_durable",
        lambda _config, *, timeout: DaemonRestartResult(
            scheduler="launchd",
            stopped=False,
            started=False,
            pid=12345,
            duration_seconds=0.1,
            message="daemon still alive",
        ),
    )

    runner = CliRunner()
    result = runner.invoke(app, ["daemon", "restart", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["stopped"] is False
    assert payload["started"] is False
    assert payload["message"] == "daemon still alive"


def test_daemon_restart_structured_success(tmp_path, monkeypatch) -> None:
    """restart --json returns durable stop/start status."""
    monkeypatch.setenv("HOME", str(tmp_path))
    data_dir = tmp_path / ".local/share/recall"
    data_dir.mkdir(parents=True)
    monkeypatch.setenv("RECALL_DATA_DIR", str(data_dir))

    from recall.services.daemon import DaemonRestartResult

    monkeypatch.setattr(
        "recall.cli.daemon.restart_daemon_durable",
        lambda _config, *, timeout: DaemonRestartResult(
            scheduler="launchd",
            stopped=True,
            started=True,
            pid=99999,
            duration_seconds=0.2,
            message=None,
        ),
    )

    runner = CliRunner()
    result = runner.invoke(app, ["daemon", "restart", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["stopped"] is True
    assert payload["started"] is True
    assert payload["pid"] == 99999


def test_schema_stats_bash_describes_suggest_variant() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["schema", "stats", "bash"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "stats bash"
    assert payload["output"]["type"] == "oneOf"
    variants = payload["output"]["variants"]
    assert len(variants) == 2
    suggest_variant = next(variant for variant in variants if variant["when"] == {"suggest": True})
    assert suggest_variant["schema"]["type"] == "object"
    assert suggest_variant["schema"]["fields"] == ["suggestions", "skipped"]


def test_structured_validation_errors_are_machine_readable() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["search", "duckdb", "--mode", "nope", "--json"])

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "VALIDATION"
    assert "unsupported search mode" in payload["error"]["message"]


def test_recreate_requires_yes_but_dry_run_is_allowed() -> None:
    runner = CliRunner()

    blocked = runner.invoke(app, ["index", "--recreate", "--json"])
    assert blocked.exit_code == 2
    blocked_payload = json.loads(blocked.stdout)
    assert blocked_payload["error"]["code"] == "CONFIRMATION_REQUIRED"

    dry_run = runner.invoke(app, ["index", "--recreate", "--dry-run", "--json"])
    assert dry_run.exit_code == 0
    dry_run_payload = json.loads(dry_run.stdout)
    assert dry_run_payload["dry_run"] is True
    assert dry_run_payload["command"] == "index"
    assert dry_run_payload["request"]["recreate"] is True


def test_text_mode_dry_runs_do_not_crash() -> None:
    runner = CliRunner()
    commands = [
        ["index", "--dry-run", "--format", "text"],
        ["daemon", "install", "--dry-run", "--format", "text"],
        ["daemon", "uninstall", "--dry-run", "--format", "text"],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, f"command {command} failed: {result.output}"
        assert "dry_run" in result.stdout


def test_plain_daemon_starts_server(monkeypatch) -> None:
    """recall daemon (no subcommand, no --once) should start the RPC server."""
    started = False

    def fake_start(**_kwargs):
        nonlocal started
        started = True

    monkeypatch.setattr("recall.cli.daemon._start_foreground_server", fake_start)

    runner = CliRunner()
    result = runner.invoke(app, ["daemon"])

    assert result.exit_code == 0
    assert started is True


@pytest.mark.parametrize(
    ("argv", "expected_embed"),
    [
        pytest.param([], True, id="default-embeds-on-supported-host"),
        pytest.param(["--no-embed"], False, id="no-embed-overrides-default"),
    ],
)
def test_index_embed_default_follows_backend_availability(
    tmp_path, monkeypatch, argv: list[str], expected_embed: bool
) -> None:
    """When embed is not specified, the RPC server auto-detects backend availability;
    --no-embed overrides that supported-host default."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    from recall.core.config import AppConfig
    from recall.services.rpc_server import RpcServer

    monkeypatch.setattr("recall.core.embeddings.embedding_backend_available", lambda _backend: True)

    captured: dict[str, object] = {}

    async def capture_request(self, **kwargs):
        captured.update(kwargs)
        from recall.services.indexer import IndexSummary

        return IndexSummary(
            total=0,
            indexed=0,
            skipped=0,
            failed=0,
            changed=0,
            fts_rebuilt=False,
            total_seconds=0.0,
        )

    monkeypatch.setattr(
        "recall.services.rpc_server.RpcServer._manual_index_request", capture_request
    )

    config = AppConfig.load()
    server = RpcServer(config=config)
    set_in_process_server(server)

    runner = CliRunner()
    result = runner.invoke(app, ["index", *argv])

    assert result.exit_code == 0, f"index failed: {result.output}"
    scope = captured["scope"]
    assert isinstance(scope, IndexRequestScope)
    assert scope.embed is expected_embed


def test_index_context_flags_flow_to_rpc_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    from recall.core.config import AppConfig
    from recall.services.rpc_server import RpcServer

    captured: dict[str, object] = {}

    async def capture_request(self, **kwargs):
        captured.update(kwargs)
        from recall.services.indexer import IndexSummary

        return IndexSummary(
            total=0,
            indexed=0,
            skipped=0,
            failed=0,
            changed=0,
            fts_rebuilt=False,
            total_seconds=0.0,
            context_messages=0,
            context_mode="template",
        )

    monkeypatch.setattr(
        "recall.services.rpc_server.RpcServer._manual_index_request", capture_request
    )
    server = RpcServer(config=AppConfig.load())
    set_in_process_server(server)

    result = CliRunner().invoke(app, ["index", "--context", "template"])

    assert result.exit_code == 0, f"index failed: {result.output}"
    config_arg = captured["config"]
    assert isinstance(config_arg, AppConfig)
    assert config_arg.embedding.context.mode == "template"


def test_recompute_context_flags_flow_to_rpc(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    from recall.core.config import AppConfig
    from recall.services.rpc_server import RpcServer

    captured: dict[str, object] = {}

    async def capture_recompute(self, **kwargs):
        captured.update(kwargs)
        from recall.services.indexer import IndexSummary

        return IndexSummary(
            total=1,
            indexed=0,
            skipped=0,
            failed=0,
            changed=1,
            fts_rebuilt=True,
            total_seconds=0.0,
            context_messages=1,
            context_mode="template",
        )

    monkeypatch.setattr(
        "recall.services.rpc_server.RpcServer._recompute_context_request", capture_recompute
    )
    server = RpcServer(config=AppConfig.load())
    set_in_process_server(server)

    result = CliRunner().invoke(
        app,
        [
            "index",
            "--recompute-context",
            "--since",
            "30d",
            "--only-mode",
            "off",
            "--context",
            "template",
        ],
    )

    assert result.exit_code == 0, f"index failed: {result.output}"
    config_arg = captured["config"]
    assert isinstance(config_arg, AppConfig)
    assert config_arg.embedding.context.mode == "template"
    assert captured["since"] is not None
    assert captured["only_mode"] == "off"


@_requires_duckdb_lock
def test_stats_context_backend_comes_from_runtime_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    from recall.core.config import AppConfig
    from recall.db import connect
    from recall.services.rpc_server import RpcServer

    config_path = tmp_path / ".config/recall/config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
        [embedding.context]
        mode = "off"
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    config = AppConfig.load()
    conn = connect(config, recreate=True)
    try:
        conn.execute("INSERT OR IGNORE INTO runtime_state (singleton) VALUES (TRUE)")
        conn.execute(
            """
            UPDATE runtime_state
            SET last_index_indexed = 7,
                last_context_messages = 3,
                last_context_mode = 'template',
                last_context_input_tokens = 0,
                last_context_output_tokens = 0,
                last_context_model = NULL
            WHERE singleton = TRUE
            """
        )
    finally:
        conn.close()
    set_in_process_server(RpcServer(config=config))

    result = CliRunner().invoke(app, ["stats", "--format", "text"])

    assert result.exit_code == 0, f"stats failed: {result.output}"
    assert "Context backend:                   template" in result.output
    assert "Context model:                     (none)" in result.output


@_requires_duckdb_lock
def test_search_supports_params_fields_and_limit(tmp_path, monkeypatch) -> None:
    runner = _init_index(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "search",
            "--params",
            '{"query":"git","limit":1}',
            "--fields",
            "session_id,source",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert sorted(payload[0].keys()) == ["session_id", "source"]


@_requires_duckdb_lock
def test_show_supports_message_limit_and_fields(tmp_path, monkeypatch) -> None:
    runner = _init_index(tmp_path, monkeypatch)
    list_result = runner.invoke(app, ["list", "--json", "--limit", "1"])
    session_id = json.loads(list_result.stdout)[0]["id"]

    result = runner.invoke(
        app,
        [
            "show",
            session_id,
            "--fields",
            "id,source,messages",
            "--message-limit",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert sorted(payload.keys()) == ["id", "messages", "source"]
    assert len(payload["messages"]) == 1


@_requires_duckdb_lock
def test_show_message_limit_does_not_leak_later_message_tools(tmp_path, monkeypatch) -> None:
    runner = _init_index(tmp_path, monkeypatch)
    list_result = runner.invoke(app, ["list", "--json", "--limit", "1"])
    session_id = json.loads(list_result.stdout)[0]["id"]

    result = runner.invoke(
        app,
        [
            "show",
            session_id,
            "--tools",
            "--message-limit",
            "1",
            "--fields",
            "messages,orphan_tool_calls",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["tool_calls"] == []
    assert len(payload["orphan_tool_calls"]) == 2
    assert sorted(tool_call["bash_command"] for tool_call in payload["orphan_tool_calls"]) == [
        "ls -la",
        "pwd",
    ]


@_requires_duckdb_lock
def test_show_message_limit_preserves_total_tool_calls(tmp_path, monkeypatch) -> None:
    runner = _init_index(tmp_path, monkeypatch)
    list_result = runner.invoke(app, ["list", "--json", "--limit", "1"])
    session_id = json.loads(list_result.stdout)[0]["id"]

    result = runner.invoke(
        app,
        [
            "show",
            session_id,
            "--tools",
            "--message-limit",
            "1",
            "--fields",
            "tool_count,messages,orphan_tool_calls",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    returned_tool_calls = len(payload["orphan_tool_calls"]) + sum(
        len(message["tool_calls"]) for message in payload["messages"]
    )
    assert returned_tool_calls == payload["tool_count"]


@_requires_duckdb_lock
def test_list_projects_freshness_fields(tmp_path, monkeypatch) -> None:
    """REQ-LIVE-003: freshness is projectable on `list` before `live` exists."""
    runner = _init_index(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "list",
            "--fields",
            "id,file_mtime,file_size,indexed_at,last_activity_at",
            "--limit",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    row = payload[0]
    assert sorted(row.keys()) == [
        "file_mtime",
        "file_size",
        "id",
        "indexed_at",
        "last_activity_at",
    ]
    assert row["file_size"] > 0
    assert row["file_mtime"] > 0
    assert row["indexed_at"] is not None
    assert row["last_activity_at"] is not None


@_requires_duckdb_lock
def test_list_supports_jsonl_format(tmp_path, monkeypatch) -> None:
    runner = _init_index(tmp_path, monkeypatch)

    result = runner.invoke(app, ["list", "--format", "jsonl", "--limit", "2"])

    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert len(rows) == 2
    assert all("id" in row for row in rows)


@_requires_duckdb_lock
def test_stats_tokens_structured_output_uses_object_rows(tmp_path, monkeypatch) -> None:
    runner = _init_index(tmp_path, monkeypatch)

    result = runner.invoke(app, ["stats", "tokens", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload
    assert sorted(payload[0].keys()) == ["input_tokens", "output_tokens", "repo"]
