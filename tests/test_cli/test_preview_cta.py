from __future__ import annotations

import json
import shlex

import pytest
from recall.cli.app import app
from typer.testing import CliRunner

_INDEX_DEFAULTS = {
    "full": False,
    "source": None,
    "recreate": False,
    "yes": False,
    "embed": None,
    "workers": "auto",
    "context": None,
    "project": None,
    "recompute_context": False,
    "since": None,
    "only_mode": None,
    "verbose": False,
    "progress": True,
    "root": None,
    "host": None,
}


@pytest.mark.parametrize("text", [False, True], ids=["json", "text"])
@pytest.mark.parametrize(
    ("command", "options", "expected"),
    [
        (["index"], [], _INDEX_DEFAULTS),
        (["index"], ["--recreate"], {**_INDEX_DEFAULTS, "recreate": True}),
        (
            ["index"],
            ["--params", '{"recreate":true,"yes":true}'],
            {**_INDEX_DEFAULTS, "recreate": True, "yes": True},
        ),
        (
            ["index"],
            [
                "--params",
                '{"full":false,"embed":true,"workers":"bananas"}',
                "--full",
                "--no-embed",
                "--workers",
                "2",
                "--no-progress",
                "--context",
                "template",
                "--source",
                "codex",
                "--verbose",
                "--project",
                "a repo's $(literal); `text`\nnext",
                "--since",
                "30d",
            ],
            {
                **_INDEX_DEFAULTS,
                "full": True,
                "embed": False,
                "workers": "2",
                "progress": False,
                "context": "template",
                "source": "codex",
                "verbose": True,
                "project": "a repo's $(literal); `text`\nnext",
                "since": "30d",
            },
        ),
        (
            ["index"],
            ["--embed", "--recompute-context", "--only-mode", "off", "--project", ""],
            {
                **_INDEX_DEFAULTS,
                "embed": True,
                "recompute_context": True,
                "only_mode": "off",
                "project": "",
            },
        ),
        (["daemon", "install"], [], {"scheduler": None}),
        (["daemon", "install"], ["--scheduler", "cron"], {"scheduler": "cron"}),
        (
            ["daemon", "install"],
            ["--params", '{"scheduler":"systemd"}', "--scheduler", "cron"],
            {"scheduler": "cron"},
        ),
        (["daemon", "uninstall"], [], {"yes": False}),
        (["daemon", "uninstall"], ["--yes"], {"yes": True}),
    ],
)
def test_execute_cta_replays_resolved_preview(
    command: list[str], options: list[str], expected: dict[str, object], text: bool
) -> None:
    runner = CliRunner()
    preview = runner.invoke(
        app, command + options + ["--dry-run", "--format", "text" if text else "json", "--cta"]
    )
    assert preview.exit_code == 0, preview.output
    if text:
        suggestion = preview.stderr.split("Next: ", 1)[1].split(" -- Execute", 1)[0]
        request = json.loads(preview.stdout)["request"]
    else:
        envelope = json.loads(preview.stdout)
        suggestion = envelope["cta"][0]["command"]
        request = envelope["data"]["request"]
    assert request == expected
    args = shlex.split(suggestion)
    assert args[0] == "recall"
    assert "--dry-run" not in args
    replay = runner.invoke(app, [*args[1:], "--dry-run", "--json"])
    assert replay.exit_code == 0, replay.output
    assert json.loads(replay.stdout)["request"] == expected


def test_execute_cta_keeps_inferred_host(tmp_path) -> None:
    root = tmp_path / "a remote's home"
    preview = CliRunner().invoke(
        app, ["index", "--root", str(root), "--dry-run", "--json", "--cta"]
    )
    assert preview.exit_code == 0, preview.output
    suggestion = shlex.split(json.loads(preview.stdout)["cta"][0]["command"])
    replay = CliRunner().invoke(app, [*suggestion[1:], "--dry-run", "--json"])
    assert replay.exit_code == 0, replay.output
    assert json.loads(replay.stdout)["request"] == {
        **_INDEX_DEFAULTS,
        "root": str(root),
        "host": "a remote's home",
    }


@pytest.mark.parametrize("command", [["index", "--recreate"], ["daemon", "uninstall"]])
def test_unconfirmed_execute_cta_requires_confirmation(command: list[str]) -> None:
    runner = CliRunner()
    preview = runner.invoke(app, [*command, "--dry-run", "--json", "--cta"])
    assert preview.exit_code == 0, preview.output
    cta = json.loads(preview.stdout)["cta"][0]
    assert "confirm" in cta["description"].lower()
    assert "--yes" in cta["description"]
    execute = runner.invoke(app, [*shlex.split(cta["command"])[1:], "--json"])
    assert execute.exit_code == 2, execute.output
    assert json.loads(execute.stdout)["error"]["code"] == "CONFIRMATION_REQUIRED"


@pytest.mark.parametrize("cta", [False, True])
@pytest.mark.parametrize("text", [False, True])
def test_preview_cta_output_contract(cta: bool, text: bool) -> None:
    result = CliRunner().invoke(
        app,
        ["index", "--recreate", "--dry-run", "--format", "text" if text else "json"]
        + (["--cta"] if cta else []),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    if text:
        assert payload["request"]["recreate"] is True
        assert "Next: recall index" in result.stderr
    elif cta:
        assert payload["data"]["request"]["recreate"] is True
        assert len(payload["cta"]) == 1
    else:
        assert payload["request"]["recreate"] is True
        assert "cta" not in payload
        assert not result.stderr
