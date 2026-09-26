# recall

[![CI](https://github.com/0xsend/recall/actions/workflows/ci.yml/badge.svg)](https://github.com/0xsend/recall/actions/workflows/ci.yml)

Session recall and analytics for AI coding agents (Claude Code, Codex, Pi Agent, Grok, Kimi Code).

Search past sessions, follow what running agents are doing, analyze tool usage, and generate permission suggestions from your local agent history. Everything stays on your machine in a local DuckDB index.

## Quick start

Requires [uv](https://docs.astral.sh/uv/); the installer asks uv for Python 3.12 or newer.

```bash
# Install the latest release
bash <(curl -fsSL https://raw.githubusercontent.com/0xsend/recall/main/scripts/install.sh)

# Run the background daemon (keeps the index current)
recall daemon install

# Index existing sessions, then search them
recall index
recall search "authentication"
```

The installer installs the release tag pinned in the script, auto-selects the MLX extra on Apple Silicon, and prints next steps. Run `bash scripts/install.sh --help` for options (`--ref`, `--mlx`, `--no-mlx`).

## Installation options

To avoid piping a remote script into bash, install a release tag directly. Replace `<TAG>` with the tag from the [latest release](https://github.com/0xsend/recall/releases/latest):

```bash
uv tool install --reinstall "recall @ git+https://github.com/0xsend/recall.git@<TAG>#subdirectory=packages/recall"
# Apple Silicon, with Metal GPU embeddings:
uv tool install --reinstall "recall[mlx] @ git+https://github.com/0xsend/recall.git@<TAG>#subdirectory=packages/recall"
```

Do not run `uv tool install recall` from the bare name: `recall` on PyPI is an unrelated package.

### Agent plugins

The plugins add `recall` and `recall-setup` skills to your agent. They do not install the CLI or daemon; ask the agent to use `recall-setup` for that.

```bash
# Claude Code
claude plugin marketplace add 0xsend/recall
claude plugin install recall@0xsend-recall

# Codex (start a new thread afterwards)
codex plugin marketplace add 0xsend/recall
codex plugin add recall@0xsend-recall

# Pi
pi install git:git@github.com:0xsend/recall.git
```

The `recall` skill triggers on prompts like "search my past sessions", "what are my agents doing", or "suggest permissions to auto-approve".

## Usage

```bash
recall search "git rebase" --tool Bash     # search one tool's calls
recall list --since 7d                     # recent sessions
recall list --project /path/to/repo
recall show <session-id> --tools           # read a session
recall live                                # sessions active right now
recall show <session-id> --tail 20 --fresh # latest messages of a running session
recall stats tools                         # tool usage counts
recall stats bash --suggest                # permission suggestions
recall stats usage --since 30d             # tokens by source x model x host
```

| Command | Description |
|---------|-------------|
| `recall index` | Index sessions from all supported agents |
| `recall search <query>` | Keyword, vector, or hybrid search across sessions |
| `recall list` | List sessions with filters |
| `recall show <id>` | Display a session, or a bounded tail / delta of it |
| `recall live` | Current agent activity, turn state, and index freshness |
| `recall stats` | Analytics: `tools`, `bash`, `tokens`, `usage`, `skills` |
| `recall daemon` | Install, run, inspect, pause, and maintain the daemon |
| `recall fleet status` | Probe hosts in `fleet.toml` |
| `recall db`, `recall snapshots`, `recall compact` | Database maintenance |

Time-bounded commands accept `--since` durations such as `30d` or `12h`. Every command has `--help`.

### Agent-first CLI

Output is human-readable text in a terminal and compact structured TOON when piped. Request another format explicitly:

```bash
recall list --json                          # JSON
recall list --format jsonl                  # NDJSON rows
recall search --params '{"query":"git","limit":5}'
recall show <session-id> --fields id,source,messages --message-limit 5
recall --llms                               # full machine-readable command manifest
recall schema daemon install                # one command's schema
```

Mutating commands support `--dry-run`, and destructive operations require explicit confirmation (`--yes`), for example `recall index --recreate --yes`.

## Daemon

The daemon owns the database, watches agent session directories for changes, and runs indexing, keyword-index, and embedding work in the background. `recall daemon install` registers it with launchd on macOS or a `systemd --user` service on Linux (`--scheduler cron` where systemd is unavailable) so it starts with your user session.

```bash
recall daemon status     # health, mode, backlog, version drift
recall daemon restart
recall daemon pause      # durable; recall daemon resume
```

To upgrade, re-run the installer, then restart the daemon so it loads the new code. If `recall daemon status` shows `drift=yes`, restart the daemon.

On macOS, releases before the launch agent rename installed it as `xyz.metalrodeo.recall.daemon`. Status names such an install, and `recall daemon install` migrates it to `it.send.recall.daemon`. Before downgrading below that release, roll the migration back — `launchctl bootout gui/$(id -u)/it.send.recall.daemon`, wait until `launchctl print gui/$(id -u)/it.send.recall.daemon` fails, then remove `~/Library/LaunchAgents/it.send.recall.daemon.plist` and run the older release's `recall daemon install` — because an older recall does not know the new label and its `daemon install` would load a second daemon.

`recall daemon status` reports the indexing backlog under `reconciliation`; see [docs/reconciliation-operations.md](docs/reconciliation-operations.md) for what it counts and what to do when it stops moving.

## Search and embeddings

`recall search` uses hybrid keyword + vector search when embeddings exist and falls back to keyword search otherwise. The ONNX CPU embedding backend ships by default; the optional MLX backend (`[mlx]` extra) uses the Apple Silicon GPU and is preferred when installed.

Configuration lives in `~/.config/recall/config.toml` (optional). Embedding settings can also come from the daemon's environment:

| Variable | Purpose |
|----------|---------|
| `RECALL_EMBED_BACKEND` | Select `onnx`, `mlx`, or backend auto-detection |
| `RECALL_EMBED_MODEL` | Override the embedding model |
| `RECALL_EMBED_BATCH_SIZE` | Control embedding batch size |

Optional [contextual retrieval](docs/contextual-retrieval.md) adds a per-message context prefix at index time.

## Data locations

| Data | Path |
|------|------|
| Database | `~/.local/share/recall/recall.duckdb` |
| Daemon socket, lock, logs | `~/.local/share/recall/` |
| Config | `~/.config/recall/config.toml` |
| Claude Code sessions | `~/.claude/projects/**/*.jsonl` |
| Codex sessions | `~/.codex/sessions/**/rollout*.jsonl` |
| Pi Agent sessions | `~/.pi/agent/sessions/**/*.jsonl` |
| Grok sessions | `~/.grok/sessions/**/chat_history.jsonl` |
| Grok usage log | `~/.grok/logs/unified.jsonl` (rotates; see [multiple hosts](docs/multi-host.md#grok-log-rotation-cadence)) |
| Kimi Code sessions | `~/.kimi-code/sessions/**/wire.jsonl` |

## Documentation

- [docs/contextual-retrieval.md](docs/contextual-retrieval.md): context modes, configuration, and costs
- [docs/multi-host.md](docs/multi-host.md): live fleet queries over SSH and offline ingest of other machines
- [docs/storage.md](docs/storage.md): DuckDB memory limits and storage format maintenance
- [docs/reconciliation-operations.md](docs/reconciliation-operations.md): reading and unsticking the indexing backlog
- [QA.md](QA.md): manual health checks for an installed host
- [plugins/recall/skills/recall/CLI_REFERENCE.md](plugins/recall/skills/recall/CLI_REFERENCE.md): command and output reference
- [SPEC.md](SPEC.md): requirements and design contracts

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
