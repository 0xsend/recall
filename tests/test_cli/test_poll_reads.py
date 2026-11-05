"""Public CLI contracts for owned systemd poll snapshots (REQ-RPC-028)."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import Path

import pytest
from recall.cli.app import app
from recall.core.config import AppConfig
from recall.db import connect
from typer.testing import CliRunner

from tests.scheduler_fakes import SchedulerPlatform


def forbid_detached_daemon(*args, **kwargs):
    raise AssertionError("poll-read tests must not start detached daemons")


@pytest.fixture
def poll_index(socket_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    from recall.cli.rpc import set_in_process_server

    set_in_process_server(None)
    monkeypatch.setattr("recall.core.rpc_client.spawn_detached_recall", forbid_detached_daemon)
    monkeypatch.setenv("RECALL_DATA_DIR", str(socket_dir))
    config_path = socket_dir / "config.toml"
    config_path.write_text(
        "[daemon]\nmode='poll'\nscheduler='systemd'\nembed=false\n[cli]\nstatus_notices=false\n"
    )
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    config = AppConfig.load()
    with connect(config) as conn:
        conn.execute("INSERT INTO sessions VALUES ('fixture', 'codex', '/fixture/session', NULL)")
    directory = Path.home() / ".config/systemd/user"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "recall-daemon.service").write_text(
        "[Service]\nType=oneshot\n"
        f"Environment={shlex.quote('RECALL_DATA_DIR=' + str(socket_dir))}\n"
        "ExecStart=/fixture/recall daemon --once\n"
    )
    return config


def test_inactive_poll_stats_reads_existing_snapshot(
    poll_index: AppConfig, scheduler_records_from_artifacts
) -> None:
    result = CliRunner().invoke(app, ["stats", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["sessions"] == 1
    assert result.stderr.count("Poll mode: reading the last indexed snapshot;") == 1
    assert "last successful index time: unknown" in result.stderr
    assert (
        f"RECALL_DATA_DIR={shlex.quote(str(poll_index.data_dir))} recall daemon --once."
        in result.stderr
    )
    assert not (poll_index.data_dir / "recall.sock").exists()


def cli_process(
    args: list[str],
    *,
    properties: Mapping[str, str | None] | None = None,
    forbid_storage: bool = False,
    platform: SchedulerPlatform | None = None,
):
    import os
    import subprocess
    import sys

    code = """
import json, os, subprocess, sys
from recall.core import rpc_client
from recall.services import service_records
from tests.scheduler_fakes import ArtifactScheduler
from tests.test_cli.test_poll_reads import forbid_detached_daemon
rpc_client.spawn_detached_recall = forbid_detached_daemon
query = ArtifactScheduler().query
changes = json.loads(os.environ['POLL_TEST_PROPERTIES'])
cron = changes.pop('_cron', None)
failed_query = changes.pop('_query_failure', None)
second_state = changes.pop('_second_query_state', None)
queries = 0
def selected(args):
    global queries
    if failed_query:
        raise TimeoutError('scheduler query timed out')
    if args == ['crontab','-l'] and cron is not None:
        return subprocess.CompletedProcess(args, 0, cron, '')
    result = query(args)
    if args[:4] == ['systemctl','--user','show','recall-daemon.service']:
        queries += 1
        values = dict(line.split('=', 1) for line in result.stdout.splitlines())
        for key, value in changes.items():
            if value is None:
                values.pop(key, None)
            else:
                values[key] = value
        if second_state and queries >= 2:
            values['ActiveState'] = second_state
        result.stdout = ''.join(f'{key}={value}\\n' for key,value in values.items())
    return result
service_records.query = selected
if os.environ['POLL_TEST_FORBID_STORAGE'] == '1':
    import recall.cli.daemon as bridge
    def forbidden(*args, **kwargs):
        raise AssertionError('local storage reached before authorization')
    bridge.check_database_schema = forbidden
    bridge.execute_poll_snapshot = forbidden
from recall.cli.app import app
"""
    if platform is not None:
        code += platform.cli_setup()
    code += "app()\n"
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        timeout=40,
        env={
            **os.environ,
            **(platform.environ if platform is not None else {}),
            "POLL_TEST_PROPERTIES": json.dumps(dict(properties or {})),
            "POLL_TEST_FORBID_STORAGE": "1" if forbid_storage else "0",
        },
    )


@pytest.mark.parametrize("state", ["active", "activating", "deactivating", "reloading"])
@pytest.mark.parametrize("endpoint", ["missing", "refused"])
def test_active_poll_refuses_before_storage(
    poll_index: AppConfig, state: str, endpoint: str
) -> None:
    import socket

    if endpoint == "refused":
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
            stale.bind(str(poll_index.data_dir / "recall.sock"))
    result = cli_process(
        ["stats", "--json"], properties={"ActiveState": state}, forbid_storage=True
    )
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout) == {
        "error": {
            "code": "RUNTIME",
            "message": (
                f"installed systemd poll service for data dir {poll_index.data_dir} is {state}; "
                "refusing a local snapshot read. Retry after the tick, "
                "when the service is inactive, "
                "or when its RPC endpoint is reachable."
            ),
        }
    }
    assert not Path(f"{poll_index.db_path}.ownership").exists()
    assert (poll_index.data_dir / "recall.sock").exists() == (endpoint == "refused")


@pytest.mark.parametrize(
    "properties,reason",
    [
        ({"ActiveState": "unknown"}, "state=unknown, MainPID=0"),
        ({"ActiveState": "maintenance"}, "state=maintenance, MainPID=0"),
        ({"MainPID": "10"}, "state=inactive, MainPID=10"),
        ({"MainPID": "invalid"}, "malformed systemd MainPID"),
        ({"MainPID": ""}, "malformed systemd MainPID"),
        ({"MainPID": None}, "missing systemd property MainPID"),
        ({"Job": None}, "missing systemd property Job"),
        ({"Job": "bad"}, "malformed systemd Job"),
        ({"NeedDaemonReload": "bad"}, "malformed systemd NeedDaemonReload"),
    ],
)
def test_uncertain_poll_refuses_before_storage(
    poll_index: AppConfig, properties, reason: str
) -> None:
    result = cli_process(["stats", "--json"], properties=properties, forbid_storage=True)
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout) == {
        "error": {
            "code": "RUNTIME",
            "message": (
                f"cannot prove the installed systemd poll service for data dir "
                f"{poll_index.data_dir} "
                f"is inactive: {reason}; refusing a local snapshot read or auto-fork. "
                "Retry when the systemd user manager is reachable and the service state is known."
            ),
        }
    }
    assert not Path(f"{poll_index.db_path}.ownership").exists()


@pytest.mark.parametrize(
    "properties,detail",
    [({"Job": "123"}, "pending job 123"), ({"NeedDaemonReload": "yes"}, "pending reload")],
)
def test_pending_poll_refuses_before_storage(
    poll_index: AppConfig, properties, detail: str
) -> None:
    result = cli_process(["stats", "--json"], properties=properties, forbid_storage=True)
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["error"]["message"] == (
        f"installed systemd poll service for data dir {poll_index.data_dir} "
        f"is {detail}; "
        "refusing a local snapshot read. Retry after the tick, "
        "when the service is inactive, "
        "or when its RPC endpoint is reachable."
    )
    assert not Path(f"{poll_index.db_path}.ownership").exists()


def test_failed_poll_refusal_is_actionable(poll_index: AppConfig) -> None:
    result = cli_process(
        ["stats", "--json"], properties={"ActiveState": "failed"}, forbid_storage=True
    )
    assert result.returncode == 1
    command = f"RECALL_DATA_DIR={shlex.quote(str(poll_index.data_dir))} recall daemon"
    assert json.loads(result.stdout)["error"]["message"] == (
        f"installed systemd poll service for data dir {poll_index.data_dir} is failed; "
        f"refusing a local snapshot read. Inspect with: {command} status. "
        f"After correcting the failure, update with: {command} --once."
    )


@pytest.mark.parametrize("format_name", ["text", "json", "jsonl", "toon"])
@pytest.mark.parametrize(
    "extra", [[], ["--fields", "sessions"], ["--params", '{"fields":"sessions"}'], ["--cta"]]
)
def test_snapshot_notice_once_in_every_output_mode(
    poll_index: AppConfig, format_name: str, extra: list[str]
) -> None:
    with connect(poll_index) as conn:
        conn.execute("UPDATE runtime_state SET last_successful_at=TIMESTAMP '2026-01-02 03:04:05'")
    result = cli_process(["stats", "--format", format_name, *extra])
    if format_name == "text" and extra[:1] in (["--params"], ["--fields"]):
        assert result.returncode == 2
        assert "Poll mode:" not in result.stderr
        return
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stderr.count("Poll mode: reading the last indexed snapshot;") == 1
    assert "last successful index: 2026-01-02T03:04:05+00:00 (" in result.stderr
    assert "); no refresh ran. Update with: RECALL_DATA_DIR=" in result.stderr
    assert "Poll mode:" not in result.stdout
    if format_name == "json":
        data = json.loads(result.stdout)
        if "--cta" in extra:
            assert set(data) == {"data", "cta"}
        elif extra:
            assert data == {"sessions": 1}
        else:
            assert data["sessions"] == 1
            assert "reconciliation" not in data


@pytest.mark.parametrize(
    "command",
    [
        ["list"],
        ["stats"],
        ["stats", "tools"],
        ["stats", "bash"],
        ["stats", "bash", "--suggest"],
        ["stats", "tokens"],
        ["stats", "usage"],
        ["stats", "skills", "--local"],
    ],
)
def test_each_read_command_keeps_coverage(
    poll_index: AppConfig, monkeypatch, command: list[str]
) -> None:
    monkeypatch.setenv("RECALL_CLI_STATUS_NOTICES", "true")
    root = poll_index.data_dir / "sources"
    root.mkdir()
    with poll_index.config_path.open("a") as config:
        config.write(f'\n[sources.codex]\nroots=["{root}"]\n')
    with connect(poll_index) as conn:
        conn.execute(
            "INSERT INTO source_files(source_key,source,source_path,root_path,ctime_ns,mtime_ns,"
            "size,observed_at,desired_generation,committed_generation,first_pending_at,last_error) "
            "VALUES ('pending-key','codex',?,?,0,0,0,0,2,1,0,'unsupported')",
            [str(root / "fixture.jsonl"), str(root)],
        )
    result = cli_process([*command, "--json"])
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stderr.count("Poll mode: reading the last indexed snapshot;") == 1
    assert result.stderr.count("1 source parked as unsupported parser diagnostics") == 1
    assert result.stderr.count("1 discovered transcripts are not indexed yet") == 1
    assert "daemon appears stale" not in result.stderr


def test_manifest_describes_every_eligible_command_without_shape_changes() -> None:
    result = CliRunner().invoke(app, ["--llms"])
    assert result.exit_code == 0
    manifest = json.loads(result.stdout)["commands"]
    for name in (
        "search",
        "list",
        "show",
        "stats",
        "stats tools",
        "stats bash",
        "stats tokens",
        "stats usage",
        "stats skills",
    ):
        command = manifest[name]
        notes = " ".join(command["notes"])
        assert "systemd poll service proven inactive with MainPID=0" in notes
        assert "mandatory timestamped stderr notice" in notes
        assert "RECALL_DATA_DIR=<data-dir> recall daemon --once" in notes
        assert "Stdout schemas, projection, CTA and idempotence are unchanged" in notes
        assert "show --fresh and --follow never downgrade" in notes
        assert command["safety"]["mutates"] is False
        assert command["safety"]["destructive"] is False
        assert command["safety"]["idempotent"] is True


def test_empty_search_unready_has_one_notice_and_owned_coverage(
    poll_index: AppConfig, monkeypatch
) -> None:
    from recall.db.fts_sidecar import open_sidecar, set_fts_fields_signature, sidecar_path

    monkeypatch.setenv("RECALL_CLI_STATUS_NOTICES", "true")
    root = poll_index.data_dir / "sources"
    root.mkdir()
    with poll_index.config_path.open("a") as config:
        config.write(f'\n[sources.codex]\nroots=["{root}"]\n')
    with connect(poll_index) as conn:
        conn.execute(
            "INSERT INTO source_files(source_key,source,source_path,root_path,ctime_ns,mtime_ns,"
            "size,observed_at,desired_generation,committed_generation,first_pending_at,last_error) "
            "VALUES ('pending-key','codex',?,?,0,0,0,0,2,1,0,'unsupported')",
            [str(root / "fixture.jsonl"), str(root)],
        )
    sidecar = open_sidecar(sidecar_path(poll_index.data_dir))
    set_fts_fields_signature(sidecar, poll_index.fts.fields)
    sidecar.commit()
    sidecar.close()
    result = cli_process(["search", "absent", "--tool", "bash", "--mode", "keyword", "--json"])
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert json.loads(result.stdout) == []
    assert result.stderr.count("Poll mode: reading the last indexed snapshot;") == 1
    assert result.stderr.count("Coverage is incomplete:") == 1
    assert result.stderr.count("1 source parked as unsupported parser diagnostics") == 1
    assert "1 discovered transcripts are not indexed yet" not in result.stderr
    assert result.stderr.count("keyword index is not ready") == 1
    assert "0 results with --tool bash and 0 without it" not in result.stderr
    assert "Widen" not in result.stderr


@pytest.mark.parametrize(
    "arguments",
    [["show", "fixture", "--fresh"], ["show", "fixture", "--follow"], ["index"], ["live"]],
)
def test_noneligible_operations_do_not_downgrade(
    poll_index: AppConfig, arguments: list[str]
) -> None:
    result = cli_process([*arguments, "--json"])
    assert result.returncode != 0, result.stdout
    assert "Poll mode: reading" not in result.stderr
    assert "refusing" in result.stdout
    assert not (poll_index.data_dir / "recall.sock").exists()


def test_installed_mode_controls_poll_admission(poll_index: AppConfig):
    poll_index.config_path.write_text(
        poll_index.config_path.read_text().replace("mode='poll'", "mode='watch'")
    )
    response = cli_process(["stats", "--json"])
    assert response.returncode == 0, response.stdout
    assert json.loads(response.stdout)["sessions"] == 1
    assert response.stderr.count("Poll mode:") == 1
    response = cli_process(["stats", "--json"], properties={"Type": "simple"}, forbid_storage=True)
    assert response.returncode == 1, response.stdout
    assert "installed systemd service for data dir" in response.stdout
    assert "recall daemon start" in response.stdout


@pytest.mark.parametrize(
    "configured_scheduler,scheduler,platform_name",
    [("launchd", "launchd", "darwin"), ("cron", "cron", "linux"), ("auto", "launchd", "darwin")],
    ids=["launchd", "cron", "macos-auto"],
)
def test_other_schedulers_keep_their_refusal(
    poll_index: AppConfig,
    scheduler_platform,
    configured_scheduler: str,
    scheduler: str,
    platform_name,
):
    import plistlib

    platform = scheduler_platform(platform_name)
    poll_index.config_path.write_text(
        poll_index.config_path.read_text().replace(
            "scheduler='systemd'", f"scheduler='{configured_scheduler}'"
        )
    )
    properties = {}
    if scheduler == "launchd":
        path = Path.home() / "Library/LaunchAgents/it.send.recall.daemon.plist"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            plistlib.dumps(
                {
                    "Label": "it.send.recall.daemon",
                    "ProgramArguments": ["/fixture/recall", "daemon", "--once"],
                    "StartInterval": 15,
                    "EnvironmentVariables": {"RECALL_DATA_DIR": str(poll_index.data_dir)},
                }
            )
        )
    else:
        properties["_cron"] = (
            f"# BEGIN recall daemon\n* * * * * "
            f"RECALL_DATA_DIR={shlex.quote(str(poll_index.data_dir))} "
            "/fixture/recall daemon --once\n# END recall daemon\n"
        )
    response = cli_process(
        ["stats", "--json"], properties=properties, forbid_storage=True, platform=platform
    )
    assert response.returncode == 1, (response.stdout, response.stderr)
    assert f"installed {scheduler} service for data dir" in response.stdout
    assert "recall daemon start" in response.stdout
    assert "Poll mode:" not in response.stderr
    assert not Path(f"{poll_index.db_path}.ownership").exists()


@pytest.mark.parametrize(
    "properties,reason",
    [
        ({"_query_failure": "timeout"}, "scheduler query timed out"),
        ({"_second_query_state": "activating"}, "changed during identification"),
    ],
)
def test_indeterminate_discovery_cannot_admit_a_reader(poll_index: AppConfig, properties, reason):
    response = cli_process(["stats", "--json"], properties=properties, forbid_storage=True)
    assert response.returncode == 1, (response.stdout, response.stderr)
    error = json.loads(response.stdout)["error"]
    assert error["code"] == "RUNTIME"
    assert "cannot prove the installed systemd poll service" in error["message"]
    assert reason in error["message"]
    assert not Path(f"{poll_index.db_path}.ownership").exists()
