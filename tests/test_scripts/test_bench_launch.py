"""Seam isolation must hold while modules initialize, fail and reload."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

LAUNCH = Path(__file__).resolve().parents[2] / "scripts/recall_bench/launch.py"
SEAM = "recall.services.usage_harvest"
DEFINITION = """
def default_grok_unified_log(*, home=None):
    from pathlib import Path
    Path(__file__).with_name("original-ran").touch()
    return "host-log"
"""


def guarded(tmp_path: Path, source: str, operation: str, *, helper: str = ""):
    package = tmp_path / "recall"
    (package / "services").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "services/__init__.py").write_text("")
    (package / "services/usage_harvest.py").write_text(textwrap.dedent(source))
    (tmp_path / "helper.py").write_text(textwrap.dedent(helper))
    runner = textwrap.dedent(f"""
        import importlib
        import json
        import runpy
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(tmp_path)!r})
        launcher = runpy.run_path({str(LAUNCH)!r})
        launcher['_install_guard'](launcher['Attempts'](Path({str(tmp_path / "audit")!r})))
        install = launcher['_install_stubs']
        attempts = launcher['Attempts'](Path({str(tmp_path / "attempts")!r}))
        usage = Path({str(tmp_path / "injected")!r})
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", runner + textwrap.dedent(operation)],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert not (package / "services/original-ran").exists(), result
    return result


@pytest.mark.parametrize(
    "initializer",
    [
        "assert default_grok_unified_log() == EXPECTED",
        "def indirect():\n    return default_grok_unified_log()\nassert indirect() == EXPECTED",
        "alias = default_grok_unified_log\nassert alias() == EXPECTED",
        "def captured(callback=default_grok_unified_log):\n"
        "    return callback()\nassert captured() == EXPECTED",
        "import helper\nassert helper.callback() == EXPECTED",
        "import threading\nresults = []\n"
        "t = threading.Thread(target=lambda: results.append(default_grok_unified_log()))\n"
        "t.start()\nt.join(5)\nassert results == [EXPECTED]",
    ],
)
def test_initializers_can_only_capture_or_call_stubs(tmp_path: Path, initializer: str):
    source = (
        DEFINITION
        + f"\nfrom pathlib import Path\nEXPECTED = Path({str(tmp_path / 'injected')!r})\n"
        + initializer
    )
    result = guarded(
        tmp_path,
        source,
        f"""
        install(attempts, usage)
        module = importlib.import_module({SEAM!r})
        assert module.default_grok_unified_log() == usage
        print('safe')
    """,
        helper=f"from {SEAM} import default_grok_unified_log as callback",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.strip() == "safe"


@pytest.mark.parametrize(
    "failure", ["initial", "reload", "failed-reload", "signature", "preloaded"]
)
def test_seam_failures_cannot_be_caught_and_continued(tmp_path: Path, failure: str):
    source = DEFINITION
    if failure == "initial":
        source += "\nimport helper\nraise RuntimeError('broken initializer')\n"
    elif failure == "signature":
        source += "\nimport helper\ndef default_grok_unified_log(wrong):\n    return 'wrong'\n"
    operation = ""
    if failure == "preloaded":
        operation += f"importlib.import_module({SEAM!r})\n"
    operation += "install(attempts, usage)\ntry:\n"
    operation += f"    module = importlib.import_module({SEAM!r})\n"
    if failure in ("reload", "failed-reload"):
        if failure == "failed-reload":
            operation += (
                "    Path(module.__file__).write_text(\"raise RuntimeError('broken reload')\\n\")\n"
            )
        operation += "    importlib.reload(module)\n"
    operation += "except BaseException:\n    pass\nprint('continued')\n"
    result = guarded(
        tmp_path,
        source,
        operation,
        helper=f"from {SEAM} import default_grok_unified_log as callback\ncallback()",
    )
    assert result.returncode == 3, result.stderr + result.stdout
    assert "continued" not in result.stdout
    report = json.loads(result.stdout)
    assert report["seams"] == "missing"
    records = [json.loads(line) for line in (tmp_path / "attempts").read_text().splitlines()]
    assert records[-1]["kind"] == "seam_missing"


@pytest.mark.parametrize("mode", ["check", "admission"])
def test_eager_modes_install_interception_before_validation(tmp_path: Path, mode: str):
    from tests.test_scripts.test_bench_daemon import fake_build

    build = fake_build(tmp_path / "build", main="def main(): pass\n")
    site = next((build / ".venv/lib").glob("python*/site-packages"))
    module = site / "recall/services/usage_harvest.py"
    module.write_text(DEFINITION + "\nalias = default_grok_unified_log\nalias()\n")
    result = subprocess.run(
        [
            str(build / ".venv/bin/python"),
            "-I",
            "-B",
            str(LAUNCH),
            "--attempts",
            str(tmp_path / "attempts"),
            "--usage-log",
            str(tmp_path / "injected"),
            mode,
            *(["inspect"] if mode == "admission" else []),
        ],
        input=b"g",
        capture_output=True,
        timeout=30,
    )
    assert not module.with_name("original-ran").exists(), result
    if mode == "check":
        assert result.returncode == 0, result.stderr
    else:
        # This minimal build lacks admission APIs; the seam phase must still be safe.
        assert result.returncode == 3, result.stderr
