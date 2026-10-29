"""Tool resolution must survive collection-time and per-test HOME isolation."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def test_home_dependent_shims_are_resolved_before_isolation(tmp_path: Path) -> None:
    home = tmp_path / "caller-home"
    config = home / ".config/mise/config.toml"
    config.parent.mkdir(parents=True)
    config.write_text("# scratch trust record\n")
    (home / ".gitconfig").write_text(
        "[user]\nname = Harness Author\nemail = harness@example.test\n"
        "[committer]\nname = Harness Committer\nemail = committer@example.test\n"
    )
    shims = tmp_path / "mise/shims"
    shims.mkdir(parents=True)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = shutil.which("git")
    bash = shutil.which("bash")
    assert git is not None and bash is not None
    repository = tmp_path / "git"
    subprocess.run(
        [git, "init", "--bare", str(repository)], capture_output=True, check=True, timeout=10
    )
    uv = binaries / "uv"
    uv.write_text("#!/bin/sh\nprintf 'uv scratch-backend\\n'\n")
    uv.chmod(0o755)
    targets = {"bash": bash, "uv": str(uv), "git": git}
    for name, target in targets.items():
        shim = shims / name
        shim.write_text(
            '#!/bin/sh\n[ -f "$HOME/.config/mise/config.toml" ] || exit 97\n'
            f'exec {shlex.quote(target)} "$@"\n'
        )
        shim.chmod(0o755)
    mise = binaries / "mise"
    mise.write_text(
        '#!/bin/sh\n[ -f "$HOME/.config/mise/config.toml" ] || exit 97\n'
        '[ "$1" = which ] || exit 98\ncase "$2" in\n'
        + "".join(
            f"{name}) printf '%s\\n' {shlex.quote(path)};;\n" for name, path in targets.items()
        )
        + "*) exit 99;;\nesac\n"
    )
    mise.chmod(0o755)
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "import subprocess, sys\n"
        "def check_tools():\n"
        "    shell = subprocess.run(['bash', '-c', \"printf shell-ready\"],\n"
        "                           capture_output=True, text=True, check=True)\n"
        "    assert shell.stdout == 'shell-ready'\n"
        "    uv = subprocess.run(['uv', '--version'],\n"
        "                        capture_output=True, text=True, check=True)\n"
        "    assert uv.stdout == 'uv scratch-backend\\n'\n"
        "    for role, expected in [('AUTHOR', 'Harness Author <harness@example.test>'),\n"
        "                           ('COMMITTER', 'Harness Committer <committer@example.test>')]:\n"
        "        git = subprocess.run(['git', 'var', f'GIT_{role}_IDENT'],\n"
        "                             capture_output=True, text=True, check=True)\n"
        "        assert git.stdout.startswith(expected + ' ')\n"
        "check_tools()\n"
        "def test_tools(): check_tools()\n"
        "def test_recursive_pytest():\n"
        "    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'conftest',\n"
        "                             '--noconftest', __file__, '-k', 'test_tools'],\n"
        "                            capture_output=True, text=True, timeout=15)\n"
        "    assert result.returncode == 0, result.stdout + result.stderr\n"
    )
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "PATH": os.pathsep.join((str(shims), str(binaries), os.environ["PATH"])),
        "PYTHONPATH": str(root),
        "GIT_DIR": str(repository),
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    for name in (
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_COMMON_DIR",
        "GIT_WORK_TREE",
    ):
        env.pop(name, None)
    isolated = {**env, "HOME": str(tmp_path / "empty-home")}
    for name in targets:
        result = subprocess.run([name, "--version"], env=isolated, capture_output=True, timeout=5)
        assert result.returncode == 97
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "conftest", "--noconftest", str(probe)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
