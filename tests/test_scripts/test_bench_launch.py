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


def assert_fatal_seam_failure(result: subprocess.CompletedProcess, attempts: Path) -> None:
    assert result.returncode == 3, (result.stdout, result.stderr)
    assert json.loads(result.stdout)["seams"] == "missing"
    records = [json.loads(line) for line in attempts.read_text().splitlines()]
    assert records[-1]["kind"] == "seam_missing"


def test_installation_during_a_seam_import_is_refused(tmp_path: Path):
    result = guarded(
        tmp_path,
        DEFINITION,
        f"""
        class StartupFinder:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == {SEAM!r}:
                    sys.meta_path.remove(self)
                    spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                    install(attempts, usage)
                    return spec
        sys.meta_path.insert(0, StartupFinder())
        importlib.import_module({SEAM!r})
        print('continued')
        """,
    )
    assert_fatal_seam_failure(result, tmp_path / "attempts")


@pytest.mark.parametrize("startup", ["thread", "raw-thread", "finder-thread", "cached-loader"])
def test_startup_imports_cannot_escape_the_guard(tmp_path: Path, startup: str):
    from tests.test_scripts.test_bench_daemon import fake_build

    build = fake_build(tmp_path / "build", main="def main(): pass\n")
    site = next((build / ".venv/lib").glob("python*/site-packages"))
    module = site / "recall/services/usage_harvest.py"
    module.write_text(DEFINITION)
    attempts = tmp_path / "attempts"
    runner = textwrap.dedent(f"""
        import _thread
        import importlib.util
        import runpy
        import sys
        import threading
        from pathlib import Path
        launcher = runpy.run_path({str(LAUNCH)!r})
        ready, release, finished = (threading.Event() for _ in range(3))
        def delayed_import():
            spec = importlib.util.find_spec({SEAM!r})
            ready.set()
            assert release.wait(5)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            finished.set()
        if {startup!r} == 'finder-thread':
            class HoldingFinder:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == {SEAM!r}:
                        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                        ready.set()
                        assert release.wait(5)
                        return spec
            sys.meta_path.insert(0, HoldingFinder())
            def import_in_finder():
                importlib.import_module({SEAM!r})
                finished.set()
            threading.Thread(target=import_in_finder).start()
        elif {startup!r} == 'thread':
            threading.Thread(target=delayed_import).start()
        elif {startup!r} == 'raw-thread':
            _thread.start_new_thread(delayed_import, ())
        else:
            spec = importlib.util.find_spec({SEAM!r})
            ready.set()
        assert ready.wait(5)
        original_install = launcher['_install_stubs']
        def install(*args):
            guard = original_install(*args)
            if {startup!r} == 'cached-loader':
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
            else:
                release.set()
                assert finished.wait(5)
            return guard
        launcher['main'].__globals__['_install_stubs'] = install
        sys.argv = ['launch', '--attempts', {str(attempts)!r},
                    '--usage-log', {str(tmp_path / "injected")!r}, '--ungated', 'check']
        raise SystemExit(launcher['main']())
    """)
    result = subprocess.run(
        [str(build / ".venv/bin/python"), "-I", "-B", "-c", runner],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert_fatal_seam_failure(result, attempts)
    assert not module.with_name("original-ran").exists()


@pytest.mark.parametrize(
    ("mode", "failure"),
    [
        ("check", "parent-import"),
        ("admission", "parent-import"),
        ("check", "entry-point"),
        ("admission", "entry-point"),
        ("admission", "admission-api"),
    ],
)
def test_required_validation_is_fatal_with_a_surviving_thread(
    tmp_path: Path, mode: str, failure: str
):
    from tests.test_scripts.test_bench_daemon import fake_build

    build = fake_build(tmp_path / "build", main="def main(): pass\n")
    site = next((build / ".venv/lib").glob("python*/site-packages"))
    survivor = (
        "import threading\nthreading.Thread(target=threading.Event().wait, daemon=False).start()\n"
    )
    if failure == "parent-import":
        (site / "recall/services/__init__.py").write_text(
            survivor + "raise ImportError('parent initializer failed')\n"
        )
    else:
        main = "" if failure == "entry-point" else "def main(): pass\n"
        (site / "recall/cli/__init__.py").write_text(survivor + main)
    attempts = tmp_path / "attempts"
    result = subprocess.run(
        [
            str(build / ".venv/bin/python"), "-I", "-B", str(LAUNCH),
            "--attempts", str(attempts), "--usage-log", str(tmp_path / "injected"),
            mode, *(["inspect"] if mode == "admission" else []),
        ],
        input=b"g",
        capture_output=True,
        timeout=5,
    )  # fmt: skip
    assert_fatal_seam_failure(result, attempts)
