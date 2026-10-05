#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/0xsend/recall.git"
SUBDIR_PATH="packages/recall"
# Pins the bootstrap installer to the signed-off release tag instead of
# installing whatever is on main. release-please bumps this automatically on each
# release via the trailing `x-release-please-version` annotation (it rewrites the
# semver and preserves the `v` prefix); keep the annotation on this line. The
# release-consistency guard (tests/test_release_metadata.py) fails closed if this
# ever drifts from the package version.
RECALL_PINNED_REF="v0.36.3" # x-release-please-version

# Internal test hooks -- do not rely on these in user scripts.
# RECALL_TEST_UV_MISSING=1 makes uv discovery fail.
# RECALL_TEST_UNAME_S and RECALL_TEST_UNAME_M override uname results.

usage() {
  cat <<'USAGE'
Usage: install.sh [OPTIONS]

Install recall from a pinned GitHub release tag.

Options:
  --ref VALUE   Git ref or release tag to install (default: embedded pinned ref)
  --mlx         Install the Apple MLX embedding extra
  --no-mlx      Do not install the Apple MLX embedding extra
  --dry-run     Print the uv install command without running it
  --help, -h    Show this help text
USAGE
}

error() {
  printf 'error: %s\n' "$1" >&2
}

die_usage() {
  error "$1"
  printf '\n' >&2
  usage >&2
  exit 2
}

uname_s() {
  if [ -n "${RECALL_TEST_UNAME_S+x}" ]; then
    printf '%s\n' "$RECALL_TEST_UNAME_S"
  else
    uname -s
  fi
}

uname_m() {
  if [ -n "${RECALL_TEST_UNAME_M+x}" ]; then
    printf '%s\n' "$RECALL_TEST_UNAME_M"
  else
    uname -m
  fi
}

uv_is_missing() {
  if [ "${RECALL_TEST_UV_MISSING:-}" = "1" ]; then
    return 0
  fi

  ! command -v uv >/dev/null 2>&1
}

ref="$RECALL_PINNED_REF"
dry_run=false
mlx_override=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --ref)
      if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        die_usage "--ref requires a value"
      fi
      ref="$2"
      shift 2
      ;;
    --mlx)
      mlx_override="true"
      shift
      ;;
    --no-mlx)
      mlx_override="false"
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die_usage "unknown option: $1"
      ;;
  esac
done

mlx_default=false
if [ "$(uname_s)" = "Darwin" ] && [ "$(uname_m)" = "arm64" ]; then
  mlx_default=true
fi

use_mlx="$mlx_default"
if [ "$mlx_override" = "true" ]; then
  use_mlx=true
elif [ "$mlx_override" = "false" ]; then
  use_mlx=false
fi

# PEP 508 direct URL form. We use this for both paths (with and without
# extras) so the script has a single shape. `uv tool install --from <url>
# recall[mlx]` does not work because uv treats the install request and the
# --from URL as conflicting package requirements when extras are present.
from_url="git+${REPO_URL}@${ref}#subdirectory=${SUBDIR_PATH}"
spec_extras=""
if [ "$use_mlx" = "true" ]; then
  spec_extras="[mlx]"
fi
install_spec="recall${spec_extras} @ ${from_url}"

render_install_command() {
  # Pin the interpreter so uv resolves against a Python that satisfies
  # recall's `requires-python = ">=3.12"`. Without this, uv may pick an
  # older interpreter (3.10/3.11) and fail dependency resolution.
  printf 'uv tool install --reinstall --python ">=3.12" "%s"\n' "$install_spec"
}

if [ "$dry_run" = "true" ]; then
  render_install_command
  exit 0
fi

if uv_is_missing; then
  error "uv is required to install recall but was not found on PATH."
  printf 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/\n' >&2
  exit 2
fi

install_cmd=(uv tool install --reinstall --python ">=3.12" "$install_spec")

set +e
"${install_cmd[@]}"
install_status=$?
set -e
if [ "$install_status" -ne 0 ]; then
  exit "$install_status"
fi

if ! command -v recall >/dev/null 2>&1 || ! recall --version >/dev/null 2>&1; then
  error "recall was installed, but recall --version could not run. Check that your uv tool bin directory is on PATH."
  exit 3
fi

cat <<'NEXT_STEPS'

Next steps:
  recall daemon install   # one-time setup for auto-watch mode
  recall daemon status    # verify version drift is cleared
  recall index            # initial index
NEXT_STEPS
