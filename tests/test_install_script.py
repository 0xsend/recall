from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not on PATH")

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install.sh"
PACKAGE_PYPROJECT = REPO_ROOT / "packages" / "recall" / "pyproject.toml"


def _package_version() -> str:
    package_data = tomllib.loads(PACKAGE_PYPROJECT.read_text(encoding="utf-8"))
    return package_data["project"]["version"]


def run_install_script(
    *args: str,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    return subprocess.run(
        ["bash", str(INSTALL_SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_exits_zero_and_prints_usage() -> None:
    result = run_install_script("--help")

    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_dry_run_prints_pinned_git_install_command() -> None:
    result = run_install_script(
        "--dry-run",
        "--ref",
        "v0.10.4",
        env_overrides={"RECALL_TEST_UNAME_S": "Linux", "RECALL_TEST_UNAME_M": "x86_64"},
    )

    assert result.returncode == 0
    assert result.stdout == (
        'uv tool install --reinstall --python ">=3.12" "recall @ '
        "git+https://github.com/0xsend/recall.git@v0.10.4"
        '#subdirectory=packages/recall"\n'
    )


def test_dry_run_uses_embedded_pinned_ref() -> None:
    # Without --ref the script must use the embedded RECALL_PINNED_REF, which
    # release-please keeps pinned to v{package version} on each release. Derive
    # the expected ref from pyproject.toml so this assertion never needs a
    # per-release edit (the hardcoded version was itself a source of drift).
    version = _package_version()

    result = run_install_script(
        "--dry-run",
        env_overrides={"RECALL_TEST_UNAME_S": "Linux", "RECALL_TEST_UNAME_M": "x86_64"},
    )

    assert result.returncode == 0
    assert f"@v{version}#subdirectory=packages/recall" in result.stdout


def test_dry_run_selects_mlx_on_darwin_arm64() -> None:
    result = run_install_script(
        "--dry-run",
        "--ref",
        "v0.10.4",
        env_overrides={"RECALL_TEST_UNAME_S": "Darwin", "RECALL_TEST_UNAME_M": "arm64"},
    )

    assert result.returncode == 0
    assert result.stdout == (
        'uv tool install --reinstall --python ">=3.12" "recall[mlx] @ '
        "git+https://github.com/0xsend/recall.git@v0.10.4"
        '#subdirectory=packages/recall"\n'
    )


def test_no_mlx_overrides_darwin_arm64_auto_detection() -> None:
    result = run_install_script(
        "--dry-run",
        "--ref",
        "v0.10.4",
        "--no-mlx",
        env_overrides={"RECALL_TEST_UNAME_S": "Darwin", "RECALL_TEST_UNAME_M": "arm64"},
    )

    assert result.returncode == 0
    assert result.stdout == (
        'uv tool install --reinstall --python ">=3.12" "recall @ '
        "git+https://github.com/0xsend/recall.git@v0.10.4"
        '#subdirectory=packages/recall"\n'
    )
    assert "recall[mlx]" not in result.stdout


def test_missing_uv_detection_exits_two_with_clear_error() -> None:
    result = run_install_script(
        "--ref",
        "v0.10.4",
        env_overrides={"RECALL_TEST_UV_MISSING": "1"},
    )

    assert result.returncode == 2
    assert "uv" in result.stderr


def test_dry_run_does_not_require_uv() -> None:
    result = run_install_script(
        "--dry-run",
        "--ref",
        "v0.10.4",
        env_overrides={
            "RECALL_TEST_UNAME_S": "Linux",
            "RECALL_TEST_UNAME_M": "x86_64",
            "RECALL_TEST_UV_MISSING": "1",
        },
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == (
        'uv tool install --reinstall --python ">=3.12" "recall @ '
        "git+https://github.com/0xsend/recall.git@v0.10.4"
        '#subdirectory=packages/recall"\n'
    )


def test_unknown_flag_exits_nonzero() -> None:
    result = run_install_script("--bogus")

    assert result.returncode != 0
