from __future__ import annotations

import json
from pathlib import Path

# Fail-closed guard for the pi-package manifest. recall's `package.json` `pi`
# manifest must resolve against the repo tree and point at a skills directory
# that actually contains SKILL.md files — otherwise pi discovers no recall
# skills and the package is silently useless. Pure stdlib so the check has no
# dependency beyond python3 (mirrors test_release_metadata.py).

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_JSON = REPO_ROOT / "package.json"


def _load_pi_manifest() -> dict:
    assert PACKAGE_JSON.is_file(), "missing root package.json"
    pkg = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    assert pkg.get("name") == "recall", "package.json name must be 'recall'"
    assert "pi-package" in pkg.get("keywords", []), (
        "package.json must declare the 'pi-package' keyword"
    )
    pi = pkg.get("pi")
    assert isinstance(pi, dict) and "skills" in pi, (
        "package.json must declare a pi manifest with a skills list"
    )
    return pi


def test_pi_manifest_skills_dir_resolves_and_has_skills() -> None:
    pi = _load_pi_manifest()
    skills = pi.get("skills", [])
    assert skills, "pi manifest must reference at least one skills directory"
    for rel in skills:
        skills_dir = REPO_ROOT / rel.lstrip("./")
        assert skills_dir.is_dir(), f"pi skills dir not found: {rel}"
        skill_mds = list(skills_dir.rglob("SKILL.md"))
        assert skill_mds, f"pi skills dir has no SKILL.md anywhere: {rel}"
        # recall ships the recall + recall-setup skills
        names = {p.relative_to(skills_dir).parts[0] for p in skill_mds}
        assert {"recall", "recall-setup"} <= names, (
            f"pi skills dir must contain recall and recall-setup, got {sorted(names)}"
        )
