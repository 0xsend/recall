# Contextual retrieval

[Contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) improves retrieval by prepending a small context prefix to each chunk before embedding and BM25 indexing. recall applies that prefix to CONTENT and THINKING messages at index time; search queries are left unchanged. Bash commands are deliberately not contextualized so normalized bash cache reuse remains stable. Reranking is out of scope.

## Modes

| Mode | Behavior |
|------|----------|
| `off` | Default. No prefix and no LLM calls. |
| `template` | Static `[{git_repo} {branch}]` prefix. No LLM calls. Cheapest option. |
| `llm-local` | Per-message 50-100 token context through a local MLX model. Defaults to `mlx-community/Llama-3.2-3B-Instruct-4bit` (~2 GB on disk), runs on Apple Silicon Metal GPU, uses no network after the model download, and requires the `[mlx]` extra. Set `model = "mlx-community/Llama-3.2-1B-Instruct-4bit"` under `[embedding.context]` (~700 MB, roughly 3x faster) on tight RAM budgets at the cost of summary quality. |
| `llm-remote` | Per-message context through the Anthropic API with prompt caching. Defaults to `claude-haiku-4-5-20251001`, requires the `[anthropic]` extra and an API key, and uses the paid API. Sends transcript text (the message and a window of its surrounding session) off the host to Anthropic's API. |
| `llm-codex` | Context through the local `codex exec` CLI. Defaults to `gpt-5.4-mini` and requires the `codex` CLI on `PATH`. Sends transcript text (the message and a window of its surrounding session) off the host to OpenAI through the Codex CLI. |

`off`, `template`, and `llm-local` keep all transcript data on the host. Only `llm-remote` and `llm-codex` send transcript text to a third-party model provider.

`template` mostly reorders results without large quality wins: the repeated repo/branch prefix can dilute IDF for common path and branch terms. The LLM modes are where meaningful retrieval gains are expected, because they match Anthropic's per-chunk context technique.

Gains depend on the corpus and the queries. In a small development pilot, per-message LLM prefixes did not improve task completion over keyword retrieval; that result does not establish how other workloads or backends behave. Start with `off` or `template`, and enable an `llm-*` mode only after measuring a benefit on your own workload.

A configured `llm-*` backend that is unavailable on the host (missing extra, credential, or CLI) is fatal at daemon startup, with a message naming what is missing and the `mode = "template"` / `"off"` escape hatch.

## Configuration

Configure contextual retrieval in `~/.config/recall/config.toml`:

```toml
[embedding.context]
mode = "off"           # off | template | llm-local | llm-remote | llm-codex
fallback = "template"  # template | off | error
model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
max_tokens = 120
batch_size = 8
min_chars = 50
```

| Key | Purpose |
|-----|---------|
| `mode` | Context mode to use for new writes. Existing rows keep their stored mode until `--recompute-context` is used. |
| `fallback` | Policy for transient per-message failures of an available `llm-*` backend (timeout, rate limit, network). `template` writes the template prefix and continues; `off` writes no prefix and continues; `error` aborts the run. |
| `model` | LLM model for the `llm-*` modes. The remote and codex backends substitute their own default when this remains at the default local model value. |
| `max_tokens` | Maximum generated prefix length for LLM modes. |
| `batch_size` | LLM context-generation batch size during indexing. |
| `min_chars` | Skip LLM contextualization for tiny messages. Skipped rows make no LLM call, spend no tokens, and are written with no prefix. |

Environment variables override the config file. Indexing runs inside the daemon, so set them in the daemon's environment (its launchd/systemd unit), not only in your shell:

| Variable | Purpose |
|----------|---------|
| `RECALL_CONTEXT_MODE` | Override `mode` |
| `RECALL_CONTEXT_FALLBACK` | Override `fallback`: `template`, `off`, or `error` |
| `RECALL_CONTEXT_MODEL` | Override the context model |

## CLI

`recall index` accepts per-run context controls. `--context` travels with the request to the daemon, so it is the reliable per-run override:

```bash
recall index --context template --recompute-context --since 7d
recall index --context llm-local --recompute-context --since 24h
recall index --recompute-context --only-mode off
```

| Flag | Purpose |
|------|---------|
| `--context MODE` | Override the configured context mode for this run. |
| `--recompute-context` | Rebuild stored context for existing CONTENT and THINKING rows, re-embed those rows, and rebuild FTS. |
| `--since DUR` | Limit context recomputation to recent sessions, such as `7d` or `24h`. |
| `--only-mode MODE` | Limit context recomputation to rows currently stored with the given mode. |

On a host whose configured mode is `llm-local`, `llm-remote`, or `llm-codex`, pin `--context template` for any full re-parse; see [Full re-parse on an LLM-context host](reconciliation-operations.md#full-re-parse-on-an-llm-context-host).

## Enabling llm-local

The bootstrap installer auto-selects the `[mlx]` extra on Darwin arm64. From a clone, install it with:

```bash
uv pip install -e 'packages/recall[mlx]'
```

For an existing stable install, reinstall with the extra from a release tag:

```bash
uv tool install --reinstall "recall[mlx] @ git+https://github.com/0xsend/recall.git@<TAG>#subdirectory=packages/recall"
```

## Enabling llm-remote

Install the `[anthropic]` extra (the installer does not select it):

```bash
uv tool install --reinstall "recall[anthropic] @ git+https://github.com/0xsend/recall.git@<TAG>#subdirectory=packages/recall"
```

Provide the key through `ANTHROPIC_API_KEY` in the daemon's environment, or through `api_key` under `[embedding.context]` in `~/.config/recall/config.toml`. Never pass the key on the command line.

## Visibility

`recall stats` surfaces contextualization counters from the most recent successful run:

| Counter | Meaning |
|---------|---------|
| `last_context_messages` | Messages whose context was written or rewritten. |
| `last_context_mode` | Resolved context mode for that run. |
| `last_context_input_tokens` | LLM input tokens used by that run. |
| `last_context_output_tokens` | LLM output tokens used by that run. |
| `last_context_model` | Context model used by that run, or empty for non-LLM modes. |

These counters describe the most recent successful run, not the current config.
