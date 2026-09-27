# Release Process

Releases are automated by [release-please](https://github.com/googleapis/release-please)
from Conventional Commits; release notes live in [CHANGELOG.md](CHANGELOG.md).

## How a release happens

1. Land work on `main` using [Conventional Commits](https://www.conventionalcommits.org/)
   (`feat:` → minor, `fix:` → patch, `feat!:`/`BREAKING CHANGE:` → major; `docs`/
   `chore`/`refactor` do not trigger a release).
2. The `release-please` workflow (`.github/workflows/release-please.yml`) runs on
   every push to `main` and maintains a **release PR** titled `chore: release
   X.Y.Z`. It accrues all unreleased commits into the proposed version + notes.
   No release PR appears until a `feat`/`fix` has landed since the last tag.
3. **Merge the release PR** to ship. release-please then, in lockstep:
   - bumps `packages/recall/pyproject.toml` (`[project].version`),
   - bumps `.claude-plugin/plugin.json` (`$.version`),
   - bumps `plugins/recall/.codex-plugin/plugin.json` (`$.version`),
   - bumps `.kimi-plugin/plugin.json` (`$.version`),
   - bumps `scripts/install.sh` `RECALL_PINNED_REF` to `vX.Y.Z` (via the
     `x-release-please-version` annotation on that line),
   - updates the `simple` strategy's root `version.txt`,
   - prepends the version section to `CHANGELOG.md`,
   - tags `vX.Y.Z` and creates the GitHub release.

Config lives in `release-please-config.json` + `.release-please-manifest.json`
(seeded at the last manual release). It has one `.` root package: release-please
therefore considers release-worthy commits anywhere in the repository, including
plugin and skill files. The `simple` strategy keeps the root changelog update;
it also requires the existing root `version.txt` target. Its explicit
TOML/JSON/generic extra files make the Python package metadata, plugin
manifests, installer pin, and strategy version file one atomic release surface. Run
`uv run pytest -q tests/test_release_metadata.py` to exercise that offline
dry-run guard.

The workflow authenticates with the `RELEASE_PLEASE_TOKEN` secret: a
fine-grained PAT scoped to `0xsend/recall` with Contents and Pull requests
read/write. `GITHUB_TOKEN` cannot be used, because pushes and PRs it creates do not
trigger `ci.yml`, and `main` requires the `check` status before a merge. The
token expires (the 0xsend limit is 366 days; the current one lapses 2027-09-28).
When it does, the release-please job fails with bad credentials; regenerate
the PAT with the same scope and replace the secret.

## Consistency guard (defense in depth)

`tests/test_release_metadata.py` fails closed if the recorded version disagrees
anywhere — `pyproject.toml` vs plugin manifests, a missing `CHANGELOG.md`
section for the current version, or `install.sh` `RECALL_PINNED_REF` ≠
`vX.Y.Z`. It runs
in the `release-metadata-sync` pre-commit hook and the CI "Release metadata sync"
step, so a manual hotfix that skips release-please still can't ship inconsistent
metadata. `tests/test_install_script.py` derives the expected ref from
`pyproject.toml`, so it needs no per-release edit.

## Manual fallback

If release-please is unavailable, replicate it by hand, then let the guard verify:
bump the version in `pyproject.toml`, the three plugin manifests, and `install.sh`
`RECALL_PINNED_REF`; update `version.txt`; add a `## X.Y.Z` section to `CHANGELOG.md`; run
`uv run pytest` + `uv run ty check`; commit `chore(release): vX.Y.Z`; tag `vX.Y.Z`;
push to `origin`; create the GitHub release.

---
