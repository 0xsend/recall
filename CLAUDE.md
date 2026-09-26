# recall

Session recall and analytics for AI agents (Claude Code, Codex, Pi Agent, Grok, Kimi Code).
[README.md](README.md) covers usage; [CONTRIBUTING.md](CONTRIBUTING.md) covers setup and the document map.

## Commands

```bash
uv sync                    # install deps (includes dev tools)
uv run pytest -q           # tests; -k parser for a subset
uv run ruff check .        # lint
uv run ruff format --check .
uv run ty check            # type check
uv run recall --llms       # authoritative command/flag manifest
```

Hooks come from `lefthook.yml` (direnv installs them; otherwise `uv run lefthook install`).
Tests use fixtures in `tests/fixtures/`.

## Development vs stable binary

- In the checkout, use `uv run recall ...`.
- The daemon and other sessions use the stable tool install (commonly `~/.local/bin/recall`).
- Promote with `uv tool install --force --python 3.12 --editable "packages/recall[<extras>]"`,
  keeping the extras the host uses (`mlx` for MLX embeddings or `llm-local`, `anthropic`
  for `llm-remote`); a missing extra makes the daemon refuse to start. Then
  `recall daemon install && recall daemon restart`.
- Never run `recall daemon install` while a development binary is first on `$PATH`;
  it bakes that path into the launchd/systemd service. Details: [CONTRIBUTING.md](CONTRIBUTING.md#development-and-stable-installs).

## Pull requests and releases

- Agent-authored PRs start as drafts. Draft PRs run no hosted CI; use local checks while iterating.
- Mark a PR ready only when the implementation is complete, the diff has been reviewed, and proportionate local checks pass.
- Use Conventional Commits. release-please owns versions and `CHANGELOG.md`; do not hand-edit
  versions. `tests/test_release_metadata.py` guards version parity. See [RELEASE.md](RELEASE.md).

## Layout

```
packages/recall/src/recall/
├── core/       # domain models, config, IDs, bash parsing (no external deps beyond Pydantic)
├── db/         # DuckDB schema, migrations, connection, queries, FTS sidecar
├── parsers/    # per-source JSONL parsers (Claude Code, Codex, Pi Agent, Grok, Kimi Code)
├── services/   # daemon, RPC server, indexing, reconciliation, search, analytics
└── cli/        # Typer commands; most call the daemon over RPC
```

Dependencies flow `cli/ → services/ → parsers/, db/ → core/`.

- [SPEC.md](SPEC.md): requirements (`REQ-*`), schema, CLI contract, architecture. REQ ids are append-only.
- [packages/recall/src/recall/parsers/SPEC.md](packages/recall/src/recall/parsers/SPEC.md): parser requirements.
- [BRIEF.md](BRIEF.md): daemon quality floors; [RECONCILIATION_QA.md](RECONCILIATION_QA.md): how they are verified.
- `packages/recall/src/recall/db/schema.sql`: database schema.

## DuckDB notes

- No `ON DELETE CASCADE` support: deletions are done manually in order.
- FTS via `PRAGMA create_fts_index()` creates the `fts_main_<table>.match_bm25()` macro.
- Naming an indexed column in UPDATE rewrites the row on DuckDB 1.5.5, even
  when its value is unchanged. Preserve exact changed-column masks; the
  rewrite regression in `tests/test_services/test_indexer.py` verifies this.
- Message identity, state and embeddings have separate ownership. Clear orphan
  companions before replacement; `test_orphan_message_state.py` covers repair.
- Embeddings use `FLOAT[384]` columns + `array_cosine_similarity()` for vector search.

## Operating on a real index

- `recall index` runs inside the daemon; only CLI flags travel with the request, not your shell environment.
- On a host with an LLM context mode, a full re-parse must pin `--context template`
  or it summarizes every message without stored context. See
  [docs/reconciliation-operations.md](docs/reconciliation-operations.md#full-re-parse-on-an-llm-context-host).
- Back up `recall.duckdb` (and `recall.fts.sqlite`) before any authorized historical repair.
