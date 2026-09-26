# Contributing

## Setup

recall is a [uv](https://docs.astral.sh/uv/) workspace and requires Python 3.12+.

```bash
git clone https://github.com/0xsend/recall.git
cd recall
uv sync                    # install dependencies, including dev tools
uv run recall --help       # run the CLI from the workspace
```

## Checks

```bash
uv run pytest -q           # tests
uv run ruff check .        # lint
uv run ruff format --check .
uv run ty check            # type check
```

Git hooks come from `lefthook.yml`: pre-commit runs format, lint, and the release
metadata guard; pre-push runs the type check and the test suite. With
[direnv](https://direnv.net/) the hooks install automatically; otherwise run
`uv run lefthook install`.

Tests use fixtures in `tests/fixtures/` (also reachable as `fixtures/` at the repo
root).

## Development and stable installs

- Inside the checkout, use `uv run recall ...`.
- The stable user binary (commonly `~/.local/bin/recall`) is what the daemon,
  other terminals, and agents use.

To exercise a change against the real daemon, promote the checkout to the stable
tool install:

```bash
uv tool install --force --python 3.12 --editable "packages/recall"
```

Keep the extras the host needs: `packages/recall[mlx]` when the host uses MLX
embeddings or `llm-local` context, and `[anthropic]` for `llm-remote` context.
Without them the configured backend is unavailable and the daemon refuses to
start. `--python 3.12` keeps uv from picking an older interpreter.

Then verify and restart:

```bash
which recall
recall --version
recall daemon install      # rewrites the service to point at the resolved binary
recall daemon restart
```

`recall daemon install` records whichever `recall` is first on `$PATH`. Never run
it from a development environment (for example `uv run recall daemon install`),
or the service will point at the checkout's virtualenv. `uv tool install` honors
`UV_TOOL_BIN_DIR` if you have customized it.

## Pull requests

Use [Conventional Commits](https://www.conventionalcommits.org/); release-please
derives versions and `CHANGELOG.md` from them (see [RELEASE.md](RELEASE.md)).
CI does not run on draft pull requests; run the checks above locally and mark the
PR ready for review when it is complete.

## Where things are documented

| Document | Covers |
|----------|--------|
| [SPEC.md](SPEC.md) | Requirements (`REQ-*`), schema, CLI contract, architecture |
| [packages/recall/src/recall/parsers/SPEC.md](packages/recall/src/recall/parsers/SPEC.md) | Parser field extraction requirements |
| [BRIEF.md](BRIEF.md) | Daemon reconciliation quality floors |
| [RECONCILIATION_QA.md](RECONCILIATION_QA.md) | Reconciliation verification procedures and benchmarks |
| [QA.md](QA.md) | Manual host health checks |
| [RELEASE.md](RELEASE.md) | Release process |
| [docs/](docs/) | User guides: reconciliation operations, contextual retrieval, multiple hosts, storage |

## Reconciliation performance benchmark

```bash
uv run python scripts/benchmark_reconciliation.py --output <new-directory>
uv run python scripts/benchmark_reconciliation.py --output <new-directory> \
  --population <checkpointed-owned-data-directory> --source <Codex-transcript>
```

Use `--baseline <result.json>` for a matched regression comparison. See
[RECONCILIATION_QA.md](RECONCILIATION_QA.md#storage-modernization-benchmarks) for
tolerances and the [DuckDB upgrade procedure](RECONCILIATION_QA.md#evaluating-a-duckdb-upgrade).
