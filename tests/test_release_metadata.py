from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Fail-closed release-consistency guard. Every place the release version is
# recorded must agree with packages/recall/pyproject.toml (the source of truth):
# the plugin manifests, the CHANGELOG section, and the installer's pinned ref.
# These drift silently because no single hook covers them all (notes once stalled
# at 0.16.2 for 5 releases; install.sh RECALL_PINNED_REF sat at v0.11.1 for
# months). release-please keeps them in lockstep on each release; this guard is
# the defense-in-depth net that catches a manual hotfix or a misconfig.

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PYPROJECT = REPO_ROOT / "packages" / "recall" / "pyproject.toml"
CLAUDE_PLUGIN_MANIFEST = REPO_ROOT / ".claude-plugin" / "plugin.json"
CODEX_PLUGIN_MANIFEST = REPO_ROOT / "plugins" / "recall" / ".codex-plugin" / "plugin.json"
CODEX_MARKETPLACE = REPO_ROOT / ".agents" / "plugins" / "marketplace.json"
KIMI_PLUGIN_MANIFEST = REPO_ROOT / ".kimi-plugin" / "plugin.json"
KIMI_MARKETPLACE = REPO_ROOT / ".kimi-plugin" / "marketplace.json"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install.sh"
RELEASE_PLEASE_CONFIG = REPO_ROOT / "release-please-config.json"
RELEASE_PLEASE_MANIFEST = REPO_ROOT / ".release-please-manifest.json"
SIMPLE_VERSION_FILE = REPO_ROOT / "version.txt"
type _SemanticVersion = tuple[int, int, int]


@dataclass(frozen=True)
class _Commit:
    """Minimal offline representation of the fields release-please uses here."""

    message: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class _ReleaseProposal:
    """The release-please effects that must remain atomic for Recall."""

    package_path: str
    current_version: _SemanticVersion
    proposed_version: _SemanticVersion
    selected_commits: tuple[_Commit, ...]
    update_paths: frozenset[str]


def _release_please_dry_run(
    config: dict[str, Any], manifest: dict[str, str], commits: tuple[_Commit, ...]
) -> _ReleaseProposal | None:
    """Simulate the documented single-package manifest behavior without GitHub.

    release-please's production ``Manifest.buildPullRequests`` requires a GitHub
    SCM implementation even for candidate construction. That makes the actual
    library an unsuitable credential-free CI dependency for this focused guard.
    This deliberately small model tracks the upstream contract that protects
    this repository: ``.`` is the root special case and therefore receives the
    complete commit list; non-root package paths receive only touching commits;
    the manifest supplies the baseline; and ``feat``/``fix`` use the default
    minor/patch semantic-version bump. The corresponding upstream source is
    linked in the test below, so changes to that contract are visible here.
    """
    packages = config.get("packages")
    if not isinstance(packages, dict) or len(packages) != 1:
        raise ValueError("Recall must configure exactly one release-please package")

    package_path, untyped_package_config = next(iter(packages.items()))
    if not isinstance(package_path, str):
        raise ValueError("release-please package path is not a string")
    package_config = _validated_json_object(
        untyped_package_config, context="release-please package configuration"
    )

    current_version_text = manifest.get(package_path)
    if not isinstance(current_version_text, str):
        raise ValueError(f"release manifest has no baseline for {package_path!r}")
    current_version = _parse_stable_semantic_version(current_version_text)

    if package_path == ".":
        selected_commits = commits
    else:
        selected_commits = tuple(
            commit
            for commit in commits
            if any(
                path == package_path or path.startswith(f"{package_path}/") for path in commit.files
            )
        )

    commit_types = tuple(_conventional_commit_type(commit.message) for commit in selected_commits)
    if not any(commit_type in {"feat", "fix"} for commit_type in commit_types):
        return None

    proposed_version = _bump_default_release_please_version(
        current_version, has_feature="feat" in commit_types
    )
    return _ReleaseProposal(
        package_path=package_path,
        current_version=current_version,
        proposed_version=proposed_version,
        selected_commits=selected_commits,
        update_paths=frozenset(
            {
                str(package_config["changelog-path"]).lstrip("/"),
                *(_configured_extra_file_paths(package_config)),
                _simple_strategy_version_file(package_config),
                ".release-please-manifest.json",
            }
        ),
    )


def _conventional_commit_type(message: str) -> str:
    match = re.match(r"^(?P<type>[a-z]+)(?:\([^)]*\))?!?:", message)
    return match.group("type") if match else ""


def _bump_default_release_please_version(
    version: _SemanticVersion, *, has_feature: bool
) -> _SemanticVersion:
    major, minor, patch = version
    if has_feature:
        return (major, minor + 1, 0)
    return (major, minor, patch + 1)


def _parse_stable_semantic_version(version: str) -> _SemanticVersion:
    match = re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version)
    if not match:
        raise ValueError(f"release manifest version is not a stable semantic version: {version!r}")
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch))


def _validated_json_object(value: object, *, context: str) -> dict[str, Any]:
    """Validate untyped JSON before treating it as a string-keyed config object.

    ``json.loads`` intentionally produces an untyped mapping. Validate both
    keys and JSON-compatible values before crossing into the typed test model,
    rather than relying on a cast that could hide malformed release metadata.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{context} is not an object")

    validated: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{context} has a non-string key")
        _validate_json_value(item, context=context)
        validated[key] = item
    return validated


def _validate_json_value(value: object, *, context: str) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, context=context)
        return
    if isinstance(value, dict):
        _validated_json_object(value, context=context)
        return
    raise ValueError(f"{context} contains a non-JSON value")


def _configured_extra_file_paths(package_config: dict[str, Any]) -> tuple[str, ...]:
    extra_files = package_config.get("extra-files")
    if not isinstance(extra_files, list):
        raise ValueError("release-please package must declare its atomic extra files")

    paths: list[str] = []
    for extra_file in extra_files:
        if not isinstance(extra_file, dict) or not isinstance(extra_file.get("path"), str):
            raise ValueError("release-please extra file must have a string path")
        paths.append(extra_file["path"].lstrip("/"))
    return tuple(paths)


def _simple_strategy_version_file(package_config: dict[str, Any]) -> str:
    """Return the release-please Simple strategy's required version-file target.

    ``Simple.buildUpdates`` always writes this target with
    ``createIfMissing: false``. Its documented ``version-file`` option defaults
    to ``version.txt``, so the file must be part of the release surface even
    when the configuration relies on that default:
    https://github.com/googleapis/release-please/blob/main/src/strategies/simple.ts
    https://github.com/googleapis/release-please/blob/main/schemas/config.json
    """
    if package_config.get("release-type") != "simple":
        raise ValueError("Recall must use the simple release-please strategy")

    version_file = package_config.get("version-file", "version.txt")
    if not isinstance(version_file, str) or not version_file:
        raise ValueError("simple release-please version-file must be a non-empty string")
    return version_file.lstrip("/")


def _package_version() -> str:
    package_data = tomllib.loads(PACKAGE_PYPROJECT.read_text(encoding="utf-8"))
    return package_data["project"]["version"]


def test_release_please_root_dry_run_includes_skill_changes_and_preserves_atomic_surface() -> None:
    """Exercise the root-package selection and proposal rules offline.

    Upstream documents ``.`` as the special root package, which releases for
    changes anywhere in the repository, and its manifest implementation passes
    every commit to that path rather than the path-filtered split:
    https://github.com/googleapis/release-please/blob/main/docs/manifest-releaser.md#manifest-releaser
    https://github.com/googleapis/release-please/blob/main/src/manifest.ts#L689-L703
    """
    config = json.loads(RELEASE_PLEASE_CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(RELEASE_PLEASE_MANIFEST.read_text(encoding="utf-8"))
    root_skill_commits = (
        _Commit(
            message="feat(skill): add continuation workflow",
            files=("plugins/recall/skills/recall/SKILL.md",),
        ),
        _Commit(
            message="fix(plugin): keep the root plugin manifest aligned",
            files=(".claude-plugin/plugin.json",),
        ),
    )

    proposal = _release_please_dry_run(config, manifest, root_skill_commits)

    assert proposal is not None, (
        "root-level feat/fix commits must create one Recall release proposal"
    )
    assert proposal.package_path == "."
    package_version = _parse_stable_semantic_version(_package_version())
    expected_feature_version = (package_version[0], package_version[1] + 1, 0)

    assert proposal.current_version == package_version
    assert proposal.proposed_version == expected_feature_version
    assert proposal.proposed_version > proposal.current_version
    assert proposal.selected_commits == root_skill_commits
    assert proposal.update_paths == {
        "CHANGELOG.md",
        "packages/recall/pyproject.toml",
        ".claude-plugin/plugin.json",
        "plugins/recall/.codex-plugin/plugin.json",
        ".kimi-plugin/plugin.json",
        "scripts/install.sh",
        "version.txt",
        ".release-please-manifest.json",
    }


def test_simple_strategy_version_file_exists_and_matches_release_baseline() -> None:
    """Fail before release-please can prepare a PR with a missing update target."""
    config = json.loads(RELEASE_PLEASE_CONFIG.read_text(encoding="utf-8"))
    package_config = config["packages"]["."]

    assert _simple_strategy_version_file(package_config) == "version.txt"
    assert SIMPLE_VERSION_FILE.is_file(), (
        "release-please Simple requires version.txt to exist before it can prepare a release update"
    )
    assert SIMPLE_VERSION_FILE.read_text(encoding="utf-8").strip() == _package_version()


def test_package_and_plugin_versions_are_in_sync() -> None:
    package_version = _package_version()
    claude_plugin_data = json.loads(CLAUDE_PLUGIN_MANIFEST.read_text(encoding="utf-8"))
    codex_plugin_data = json.loads(CODEX_PLUGIN_MANIFEST.read_text(encoding="utf-8"))
    kimi_plugin_data = json.loads(KIMI_PLUGIN_MANIFEST.read_text(encoding="utf-8"))

    assert claude_plugin_data["version"] == package_version
    assert codex_plugin_data["version"] == package_version
    assert kimi_plugin_data["version"] == package_version


def test_codex_marketplace_points_at_canonical_plugin_bundle() -> None:
    marketplace_data = json.loads(CODEX_MARKETPLACE.read_text(encoding="utf-8"))
    plugin_entries = {entry["name"]: entry for entry in marketplace_data["plugins"]}

    recall_entry = plugin_entries["recall"]

    assert marketplace_data["name"] == "0xsend-recall"
    assert recall_entry["source"] == {
        "source": "local",
        "path": "./plugins/recall",
    }
    assert recall_entry["policy"] == {
        "installation": "AVAILABLE",
        "authentication": "ON_INSTALL",
    }


def test_legacy_claude_manifest_uses_canonical_skill_bundle() -> None:
    plugin_data = json.loads(CLAUDE_PLUGIN_MANIFEST.read_text(encoding="utf-8"))

    assert plugin_data["skills"] == "./plugins/recall/skills/"


def test_kimi_marketplace_points_at_canonical_plugin_bundle() -> None:
    marketplace_data = json.loads(KIMI_MARKETPLACE.read_text(encoding="utf-8"))
    plugin_entries = {entry["id"]: entry for entry in marketplace_data["plugins"]}

    recall_entry = plugin_entries["recall"]

    assert marketplace_data["version"] == "2"
    assert recall_entry["source"] == "./"


def test_kimi_manifest_uses_canonical_skill_bundle() -> None:
    plugin_data = json.loads(KIMI_PLUGIN_MANIFEST.read_text(encoding="utf-8"))

    assert plugin_data["name"] == "recall"
    assert plugin_data["skills"] == "./plugins/recall/skills/"


def test_changelog_has_current_version_section() -> None:
    version = _package_version()
    changelog = CHANGELOG.read_text(encoding="utf-8")

    # release-please writes `## [X.Y.Z](url) (date)`; manual/seed entries may be a
    # bare `## X.Y.Z`. Accept either. The trailing \b stops 0.17.3 from matching
    # a longer 0.17.30.
    pattern = re.compile(rf"^##\s+\[?{re.escape(version)}\b", re.MULTILINE)

    assert pattern.search(changelog), (
        f"CHANGELOG.md is missing a release section for {version}. "
        "Every shipped version must have generated notes."
    )


def test_install_script_pinned_ref_matches_version() -> None:
    version = _package_version()
    install_sh = INSTALL_SCRIPT.read_text(encoding="utf-8")

    match = re.search(r'^RECALL_PINNED_REF="([^"]+)"', install_sh, re.MULTILINE)

    assert match, "RECALL_PINNED_REF assignment not found in scripts/install.sh"
    assert match.group(1) == f"v{version}", (
        f"scripts/install.sh RECALL_PINNED_REF is {match.group(1)!r}, "
        f"expected v{version}. curl|bash would install the wrong build."
    )
