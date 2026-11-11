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

`list`, `search`, `show`, `live`, and `stats usage` accept `--fleet` to query
the calling daemon over local RPC plus inventory endpoints. Recognized self
contributes once. List/search/live apply a single global limit, so a selected
host may have no rows in the returned page. Failed selected endpoints, including
the caller, produce `fleet: skip` warnings on stderr; list/search/live/usage
succeed if any selected endpoint succeeds (even with an empty reply), while show
keeps unique-hit/ambiguity/not-found rules. The census requires every endpoint.
Structured consumers must retain stderr for partial-result diagnostics.
Fleet text `list`, `search`, and `live` rows and the `show` session header
append `host=NAME` using the same `host` value as JSON, matching `stats usage`.
Text output without `--fleet` is unchanged.
`--host` on fleet list/live/show selects an endpoint before reads, using its
inventory name, canonical caller name or declared self alias; it does not filter
stored origins. Search and usage have no endpoint selector. Genuine stored
origin hosts survive; missing/empty/`local` labels use the canonical caller or
remote inventory name. IDs and duplicate imported histories are not deduplicated.
`fleet status` tests only inventory SSH routes.

Aggregate fleet reads and default `stats skills` never start the calling daemon
or select/use a poll snapshot, including guidance/notices. With an unavailable
local daemon, default `stats skills` fails with exit 1 and no census rows or
coverage payload, even if all remotes succeed: structured output is the existing
error envelope, text has empty stdout and an error on stderr.
`stats skills --local` retains ordinary local-only startup/snapshot eligibility.

Set `hostname` to the caller's exact short hostname when a self entry uses an
alias, an FQDN such as `aem5.example.com`, or different case such as `AEM5`. For
caller `aem5`, `hostname = "aem5"` folds either entry into local RPC once.
Alternatively rename the entry to `aem5`. Without the declaration or rename,
those names remain separate SSH endpoints and can count the same machine twice.
No automatic DNS, domain removal or case folding occurs. Distinct machines with
the same short hostname require distinct effective identities and display names
for foreign entries; a foreign endpoint must not use the caller's identity.
Automatic identity discovery is outside this contract.

With `--fleet`, `--fleet-inventory-only` selects only inventory endpoints; an
explicitly listed caller still uses local RPC. This restores inventory membership,
including empty-inventory refusal, but not byte-for-byte SSH-to-self failures,
tie order or fallback labels. It requires `--fleet` (otherwise exit 2) and also
accepts the JSON parameter `fleet_inventory_only`. No config/environment toggle
changes this default. The census accepts neither compatibility switch.

```bash
recall list --fleet --fleet-inventory-only --since 7d
```

With no fleet inventory, default `--fleet` queries read the calling daemon
alone. Default `stats skills` still requires a nonempty configured inventory and
complete coverage; `--local` selects a local-only census. Invalid inventory fails
validation (exit 2) before primary or advisory reads. `fleet status` still succeeds
with zero inventory hosts and an empty table.

Time-bounded commands accept `--since` durations such as `30d` or `12h`. Every command has `--help`.

Ordinary non-fleet, local-only commands use the running daemon. If RPC is unavailable,
search, list, ordinary show and eligible stats reads (stats skills only with --local)
may read the existing index between ticks
of a matching installed systemd poll service proven inactive. A mandatory stderr
notice gives the last successful index timestamp and age (or unknown), plus a
shell-quoted data-dir-scoped `recall daemon --once` update command. The result can
lag source files; stdout stays unchanged. Pending/unsupported coverage guidance
comes from that same snapshot.

Active or uncertain services, watch, launchd/cron, `show --fresh`, `--follow` and
other noneligible commands retain their managed-service refusal. The existing
watch/launchd/cron refusal names `recall daemon start`; active systemd poll refuses
with retry-after-tick guidance. For a data directory without its own installed
service, ordinary eligible commands automatically start a background daemon from the build being
invoked. Select a directory with `RECALL_DATA_DIR=/path/to/data` for both commands.

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

For a foreground daemon, an explicit `recall daemon --mode poll|watch|auto`
overrides `[daemon] mode` for that process. Without `--mode`, the configured mode
applies (default `auto`, which resolves to `watch`). Status reports the effective
mode; the override leaves the config file unchanged.

With `[daemon] mode = "poll"`, the systemd timer schedules the first cycle after
the configured `interval`, then repeats relative to service activation. Systemd's
default accuracy window can delay an elapse by up to one minute. A manual service
start is unnecessary. Package upgrades leave installed units unchanged; run the
upgraded installed binary's `recall daemon install` to refresh and reactivate the
timer. Reinstalling or restarting the enabled timer also schedules a first cycle.

On macOS, releases before the launch agent rename installed it as `xyz.metalrodeo.recall.daemon`. Status names such an install, and `recall daemon install` migrates it to `it.send.recall.daemon`. Before downgrading below that release, roll the migration back — `launchctl bootout gui/$(id -u)/it.send.recall.daemon`, wait until `launchctl print gui/$(id -u)/it.send.recall.daemon` fails, then remove `~/Library/LaunchAgents/it.send.recall.daemon.plist` and run the older release's `recall daemon install` — because an older recall does not know the new label and its `daemon install` would load a second daemon.

`recall daemon status` reports the indexing backlog under `reconciliation`; see [docs/reconciliation-operations.md](docs/reconciliation-operations.md) for what it counts and what to do when it stops moving.

## Search and embeddings

`recall search` uses hybrid keyword + vector search when embeddings exist and falls back to keyword search otherwise. The ONNX CPU embedding backend ships by default; the optional MLX backend (`[mlx]` extra) uses the Apple Silicon GPU and is preferred when installed.

Search RPC callers must allow a silence budget greater than six seconds: the
five-second enrichment preference plus a one-second scheduling/transport margin.
The CLI and RpcClient default to 30 seconds; the server’s absolute request deadline
and bounded progress frames are unchanged. See [REQ-RPC-029](SPEC.md).

DuckDB defaults to a `2GB` buffer allowance. Larger explicit `memory_limit`
values are clamped, with a startup warning and requested/applied/clamped values
in `recall daemon status`. To honor a larger override, set
`[duckdb] memory_limit_exceeds_floor = true`; this can exceed the 4 GiB process
memory floor. See [storage configuration](docs/storage.md#large-corpora-and-duckdb-memory).

Configuration lives in `~/.config/recall/config.toml` (optional). Embedding settings can also come from the daemon's environment:

Source roots are configured with `[sources.<source>] roots = ["/path/to/sessions"]`;
`roots = []` disables that source. Directory symlinks below a root delegate their
targets to that parser, including targets outside the root. Only link content the
operator intends Recall to index and send to its configured enrichment backend.
Aliases share one canonical path and identity. Removing the final route removes
catalog eligibility while retaining indexed history. See
[discovery limits and diagnostics](docs/reconciliation-operations.md#linked-directories).


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
- [docs/multi-host.md](docs/multi-host.md): fleet queries over local RPC and SSH, plus offline ingest of other machines
- [docs/storage.md](docs/storage.md): DuckDB memory limits and storage format maintenance
- [docs/reconciliation-operations.md](docs/reconciliation-operations.md): reading and unsticking the indexing backlog
- [QA.md](QA.md): manual health checks for an installed host
- [plugins/recall/skills/recall/CLI_REFERENCE.md](plugins/recall/skills/recall/CLI_REFERENCE.md): command and output reference
- [SPEC.md](SPEC.md): requirements and design contracts

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
