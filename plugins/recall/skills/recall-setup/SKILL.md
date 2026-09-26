---
name: recall-setup
description: Install, upgrade, bootstrap, and troubleshoot the recall CLI and daemon for users after installing the recall Codex or Claude plugin. Use when recall is missing, recall --version fails, daemon setup or version drift needs repair, first indexing is needed, or the user asks to enable contextual retrieval.
---

# recall setup

## Audience and goal

You are an agent. The user installed the recall plugin or asked to bootstrap
recall on this machine. The plugin only installs agent workflows; the `recall`
binary, daemon, and indexes are local system state and must be installed and
verified explicitly. Follow the path that matches the current installation and
the user's requested release or checkout. Preserve existing backend choices.

After a local plugin update, verify the installed skill contents against the
requested checkout. If a same-version update reuses stale cached contents,
reinstall through the harness's native plugin manager, preserving plugin data,
and check the installed files again. Start a new agent session to load the update.

## Step 0 — Detect

Run these commands first:

```bash
command -v recall
recall --version
uname -sm
recall daemon status
```

Interpretation:

| Output | Meaning | Next step |
|--------|---------|-----------|
| `command -v recall` prints nothing or exits non-zero | The binary is missing | Step 1 |
| `recall --version` fails | The binary is broken or not on `PATH` | Step 1 |
| `uname -sm` prints `Darwin arm64` | Apple Silicon; installer auto-picks `[mlx]` | Continue |
| `recall daemon status` shows `installed: false` | Daemon is not installed | Step 2 |
| `recall daemon status` reports a non-null `runtime_unavailable_reason` or `startup_refusal` | Runtime is unavailable or refused startup | Diagnose the reported reason; Step 2 or Step 5 |
| `recall daemon status` shows `installed: true` and no runtime error or drift | Installation is present | Verify the requested capability and existing index below |

If `recall` is missing, `recall daemon status` cannot run. Treat that as Step 1.

### Existing installation or checkout rollout

Run `recall stats --json` to check whether sessions are already indexed. If they
are, skip the first-index step. An upgrade or a client/daemon mismatch does not
require a full reparse or context recomputation.

For live-session use, check `recall live --help`, `recall show --help`, and
`recall daemon status --json`. Inspect `reconciliation.rpc_ready` and
`live_observation_ready`, then verify a bounded `recall live --limit 1 --json`
read and its coverage. Watch and poll modes both observe activity; backlog drain
and model readiness are separate from RPC startup. Status has no `running` boolean, and installed scheduler metadata alone
does not prove a responsive runtime. A version label alone does not prove that a checkout client
and the running daemon implement the same features. A `RUNTIME` response saying
that the daemon cannot satisfy a bounded/fresh read calls for matching builds;
restarting an older installation does not install the requested checkout.

When the user requests a checkout build, follow that checkout's promotion
instructions. Preserve the configured extras (`mlx` for local MLX context or
embeddings, `anthropic` for remote context) and use Python 3.12. Install and
restart the daemon through the explicit stable tool binary, commonly
`~/.local/bin/recall`, then verify a bounded fresh read. Running
`uv run recall daemon install` can point the service at the development
environment. If the rollout changes the database schema, preserve a consistent
database backup with the prior installation so rollback can restore both.

## Step 1 — Install recall

Use the current bootstrap installer, which selects the release pinned by the
repository's release process. Keep an explicitly requested tag or checkout
instead of replacing it with this default:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/0xsend/recall/main/scripts/install.sh)
```

`uv` is a prerequisite. The installer fails fast if `uv` is missing. On Apple Silicon (`Darwin arm64`), the installer auto-picks the `[mlx]` extra.

If the user refuses curl-pipe-bash, ask them for the release tag they want to pin, then run:

```bash
uv tool install --reinstall --python ">=3.12" "recall @ git+https://github.com/0xsend/recall.git@<TAG>#subdirectory=packages/recall"
```

On Apple Silicon (`Darwin arm64`) where the host uses an MLX backend (`llm-local`
context or MLX embeddings), include the `[mlx]` extra so it is not silently
dropped — the extra travels with the package spec (uv's PEP 508 form), and
`--python ">=3.12"` avoids uv resolving against an older interpreter:

```bash
uv tool install --reinstall --python ">=3.12" "recall[mlx] @ git+https://github.com/0xsend/recall.git@<TAG>#subdirectory=packages/recall"
```

Confirm the install:

```bash
which recall
recall --version
```

Expected: `which recall` points to the user's uv tool bin path, commonly `~/.local/bin/recall`, and `recall --version` prints a version.

## Step 2 — Install and start the daemon

Run:

```bash
recall daemon install
recall daemon start
recall daemon status
```

Look for:

| Field | Expected |
|-------|----------|
| `installed` | `true` |
| `runtime_unavailable_reason` / `startup_refusal` | absent or `null` |
| `reconciliation.runtime_mode` | Actual `watch` or `poll`, followed by a successful live read |
| `reconciliation.rpc_ready` / `live_observation_ready` | `true`; inspect scan coverage separately |
| `version_drift` or `drift` | absent, `false`, or `no` |

If version drift appears, restart the daemon:

```bash
recall daemon restart
recall daemon status
```

## Step 3 — First index

First run may take minutes depending on how many sessions the user has in `~/.claude/projects/`, `~/.codex/sessions/`, `~/.pi/agent/sessions/`, `~/.grok/sessions/`, and `~/.kimi-code/sessions/`.

Run:

```bash
recall index --no-progress
recall stats
```

Expected: `recall stats` shows non-zero `sessions`, `messages`, and `tool_calls` counts. If counts are zero, run `recall index --no-progress` once more and inspect the output for source-path or permission errors.

## Step 4 (optional) — Enable contextual retrieval

Contextual retrieval writes context prefixes for CONTENT and THINKING messages before embedding and BM25 indexing. Do not enable it automatically. Ask the user first:

```text
Do you want to enable contextual retrieval for a recent test run? Recommended first choice: template mode, because it needs no extra install and no API key.
```

Modes:

| Mode | Default model | Notes |
|------|---------------|-------|
| `off` | none | Default; no prefix and no LLM calls. |
| `template` | none | Static `[{git_repo} {branch}]` prefix; cheapest first test. |
| `llm-local` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | Local MLX backend on Apple Silicon Metal GPU; no network after model download. Needs the `[mlx]` extra. |
| `llm-remote` | `claude-haiku-4-5-20251001` | Anthropic API or any Anthropic-compatible proxy (LiteLLM, etc.); paid API usage when hitting `api.anthropic.com`. Needs the `[anthropic]` extra. |

Recommended first run:

```bash
recall index --context template --recompute-context --since 7d
```

For `llm-local`, confirm MLX support before running a real job:

```bash
uv tool list | grep mlx
recall index --context llm-local --dry-run
```

If MLX is missing, reinstall recall with the `[mlx]` extra before continuing.

For `llm-remote`, first confirm the `[anthropic]` extra is installed; without it the configured backend is unavailable and the daemon now **hard-errors at startup** (REQ-CTX-020) with an actionable message instead of silently degrading to template:

```bash
~/.local/share/uv/tools/recall/bin/python -c 'import anthropic; print("anthropic", anthropic.__version__)'
```

If `ModuleNotFoundError`, reinstall recall with the extra. The bootstrap installer in Step 1 does not auto-pick `[anthropic]`, so you need to add it explicitly:

```bash
# Production install (from GitHub):
uv tool install --reinstall --python ">=3.12" "recall[anthropic] @ git+https://github.com/0xsend/recall.git@<TAG>#subdirectory=packages/recall"
# Or, from a local checkout of this repo:
# uv tool install --force --editable --python 3.12 "packages/recall[anthropic]"
recall daemon restart
```

Then confirm a credential is reachable to the daemon. Never ask the user to
paste a secret into chat, never print a secret, and never pass a secret as a CLI
argument. The key can come from one of two places:

1. **Env var** (works for foreground CLI runs and shells): confirm it without printing it.
   ```bash
   test -n "${ANTHROPIC_API_KEY:-}" && printf 'ANTHROPIC_API_KEY=set\n' || printf 'ANTHROPIC_API_KEY=missing\n'
   ```
2. **Config file** (required when the daemon runs under launchd/systemd and does not inherit shell env): write into `~/.config/recall/config.toml` only when the user has authorized storing the key there through an approved secure path. The `api_key` field is read on daemon startup; restart the daemon after editing.
   ```toml
   [embedding.context]
   mode = "llm-remote"
   fallback = "template"
   api_key = "<approved secret value>"
   # Optional: route through a local Anthropic-compatible proxy (LiteLLM, etc.)
   # base_url = "http://localhost:4000"
   # timeout = 60
   # Required when LiteLLM routes upstream to a thinking-mode model (Qwen3,
   # DeepSeek-R1). Recall always strips <think>...</think> blocks from
   # responses, but disabling the preamble keeps generation fast.
   # instruction_prefix = "/no_think\n\n"
   ```
   `api_key` takes precedence over `ANTHROPIC_API_KEY` when both are set. Equivalent env overrides: `RECALL_CONTEXT_API_KEY`, `RECALL_CONTEXT_BASE_URL`, `RECALL_CONTEXT_TIMEOUT`, `RECALL_CONTEXT_INSTRUCTION_PREFIX`.

Each eligible message is one API call, mitigated by Anthropic prompt caching, so ask for user confirmation before running a large window.

## Step 5 — Troubleshooting checklist

| Issue | Action |
|-------|--------|
| `Conflicting lock is held` | Another daemon or process holds `~/.local/share/recall/recall.lock`; check `recall daemon status` and the PID, then kill the process only if it is stale. |
| `version_drift: yes` | Run `recall daemon restart`, then `recall daemon status`. |
| `recall search` returns nothing | Confirm `recall stats` shows non-zero `sessions`; run `recall index --no-progress` again. |
| `connection closed by daemon` | Known transient; retry once. If reproducible, run `recall daemon restart`. |
| Daemon fails to start / crash-loops with `context mode 'llm-remote' is configured but its backend is unavailable` (REQ-CTX-020) | The `[anthropic]` extra is not installed in the recall tool venv. Reinstall with `uv tool install --reinstall "recall[anthropic] @ git+https://github.com/0xsend/recall.git#subdirectory=packages/recall"`, then `recall daemon restart`. The same error on `llm-local` points at the `[mlx]` extra. The error is fatal by design (no silent template fallback); the message names the exact extra and the `mode = template\|off` escape hatch. |
| Same `backend is unavailable` error with `[anthropic]` already installed | Credential is missing — the daemon found neither `ANTHROPIC_API_KEY` in its env nor `api_key` in `[embedding.context]`. Add `api_key` to `~/.config/recall/config.toml` and restart the daemon. |
| `timeout waiting for response to recall.index` on the CLI during llm-remote runs | The RPC client gives up at 300s but the daemon continues processing in the background. Watch `recall daemon status` for the `last_index_summary` to land; do not re-run blindly or you will queue duplicate work behind the existing batch. |

## Hand-off

User is bootstrapped. Subsequent recall requests should use the `recall` skill
for everyday search, listing, session display, and analytics commands.
