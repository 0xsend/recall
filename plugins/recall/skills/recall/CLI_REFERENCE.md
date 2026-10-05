# CLI Reference

The live source of truth is `recall <command> --help` and the machine-readable
`recall --llms` / `recall schema <command>` — flags and output fields appear there
first. This file is a quick human-oriented reference. Note: non-TTY output defaults
to compact **TOON** (not the human text shown in the examples below); pass `--json`
only when you'll pipe through `jq`.

## recall index

Reconcile sessions from all configured source roots through the daemon.

```bash
recall index [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--full` | Force full reindex of all sessions |
| `--source` | Filter by source: `claude-code`, `codex`, `pi-agent`, `grok`, `kimi-code` |
| `--recreate` | Backup and rebuild database from scratch |
| `--context` | Context mode for this run: `off`, `template`, `llm-local`, `llm-remote`, `llm-codex` (overrides config) |
| `--recompute-context` | Rebuild stored context for existing CONTENT and THINKING rows without reparsing JSONL |
| `--since` | Bound raw source mtime or context recomputation by a window such as `7d`, `24h`, or `2024-01-01` |
| `--project` | Limit reconciliation to an indexed git repo path match |
| `--no-embed` | Omit embedding generation for this request |
| `--only-mode` | Only recompute rows currently marked with this context mode |
| `--root` | Alternate home directory for multi-host ingest, laid out like `$HOME` (`.claude/`, `.codex/`, `.pi/`, `.grok/`, `.kimi-code/`) |
| `--host` | Host label stored on every session indexed in this run; defaults to the machine's short hostname, or the basename of `--root` |
| `-v, --verbose` | Enable verbose logging |
| `--json` | Output results as JSON |

**Multi-host ingest:** sync a remote machine's session trees to a local directory,
then index that directory under the remote host's name. Re-indexing the same tree
is idempotent (session identity is a source+path hash), so host and token totals
update in place.

```bash
recall index --root ~/.local/share/recall/hosts/build-box --host build-box
```

Grok token totals come only from `~/.grok/logs/unified.jsonl`, which rotates
(roughly 22 hours on a busy host). Harvest host-side or sync that log faster than
it rotates, or the usage is lost.

**Example output:**
```
Indexed 15 sessions, skipped 42, failed 0 (total 57).
```

**Contextual JSON output:**
```json
{
  "total": 57,
  "indexed": 15,
  "skipped": 42,
  "failed": 0,
  "changed": 15,
  "fts_rebuilt": true,
  "total_seconds": 12.34,
  "context_messages": 273129,
  "context_mode": "template",
  "context_input_tokens": 0,
  "context_output_tokens": 0,
  "context_model": null
}
```

## recall search

Full-text search across session content and tool calls.

```bash
recall search <query> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--tool` | Restrict results to one tool's calls — accepts any tool name (`Bash`, `Read`, `Edit`, …). **Excludes all message/conversation text;** see caveat below. |
| `--session` | Restrict to a recall session ID (`id` from list/live/show, not the harness `source_session_id`) |
| `--source` | Filter by source: `claude-code`, `codex`, `pi-agent`, `grok`, `kimi-code` |
| `--mode` | `keyword`, `vector`, `hybrid`, `auto` (default `auto` → hybrid when embeddings exist, else keyword) |
| `--limit` | Max results to return (default 20) |
| `--fleet` | Fan-out search to hosts in `fleet.toml` and merge by score |
| `--fleet-config` | Alternate fleet inventory path |
| `--format` | Output format: `auto`, `text`, `json`, `jsonl`, `toon` |
| `--json` | Output results as JSON |

Results include `host` (from `session_state`, or the inventory name under `--fleet`).

**Example output:**
```
[0.85] abc123def456 (claude_code)
  assistant: Implemented OAuth2 authentication flow using...
  source: ~/.claude/projects/my-app/session.jsonl
  time: 2024-01-15T10:30:00
```

**Empty results rarely mean a session is missing.** Most "not indexed" conclusions are actually query problems:

- `--tool Bash` returns only Bash *tool calls* — anything discussed in messages won't match. Drop the filter and re-run.
- Widen with `--mode keyword` and fewer terms; many rare terms together can score everything near zero.
- Confirm indexing with `recall list --project <path> --since 30d` or `recall show <id>` before blaming the daemon. `show` accepts the source UUID or the recall id.
- If `coverage.unsupported` is nonzero, those transcripts are parked on parser diagnostics, not missing. File a recall issue or fix the parser.
- recall only indexes what agents did through tools + conversation. Commands a human typed directly in a terminal are **not** in recall — they're in shell history (`~/.zsh_history`).

**Output fields.** Each result includes `lexical_match`: `true` when it matched the query terms textually (keyword/BM25 leg), `false` for a vector-only semantic neighbor. When **every** row is `lexical_match: false` there were no keyword matches at all — vector search always returns up to `--limit` nearest neighbors, even for gibberish. `recall search` also writes a `note:` to **stderr** when results are empty or semantic-only (e.g. re-probing a zero-result `--tool` query without the filter), so don't discard stderr.

## recall list

List sessions with optional filters.

```bash
recall list [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--source` | Filter by source: `claude-code`, `codex`, `pi-agent`, `grok`, `kimi-code` |
| `--since` | Time window: `7d`, `24h`, `2024-01-01` |
| `--project` | Filter by git repo path |
| `--host` | Exact match on session host label |
| `--fleet` | Fan-out to hosts in `fleet.toml` and merge by recency |
| `--fleet-config` | Alternate fleet inventory path |
| `--json` | Output results as JSON |

Each session includes `host`. Under `--fleet`, host is stamped from the remote
payload or the inventory name.

**Example output:**
```
[2024-01-15 10:30] abc123 (claude_code) /path/to/repo messages=25 tools=12
[2024-01-14 14:22] def456 (codex) /path/to/project messages=8 tools=3
```

## recall daemon status and maintenance

`recall daemon status --json --fields reconciliation` separates RPC readiness,
catalog scan completion, live observation, raw progress, keyword readiness, and
enrichment readiness. Source coverage reports scan errors, counts and oldest
pending age. A nonzero `unsupported` count is a parser gap: read `reconciliation.unsupported_summary`
(source, record detail, file count, sample path), then file a recall issue or fix the parser. Use `--limit` (1..256, default 100) and `--cursor` for source pages.
Absent optional roots are empty scopes; unavailable configured roots are incomplete.
Out-of-scope historical rows remain visible with `eligible: false` and separate
counts. An offline fallback explicitly marks runtime coverage unavailable.

`recall daemon pause` persists across client auto-start and stops mutations while
retaining reads. `recall daemon resume` restores work. Poll mode is ordinary
reconciliation without notifications, not a maintenance pause. An unavailable
embedding or context model degrades enrichment readiness only; it does not mean
history is missing. After a consistent
DB/sidecar backup, authorized historical repair should pin context and embeddings,
for example `recall index --full --no-embed --context template --since 30d`.
Confirmed `--recreate` reports its retained backup path; do not use it merely to
resolve lag or version drift.

## recall live

Inspect current agent activity, turn state, and index freshness. The default
roster includes `active` sessions from recent source observations in watch or
poll mode, unless an observed writer has exited. Indexing success and watcher
subscription capacity do not determine roster membership. Read `turn` to
distinguish working from awaiting input.

Structured output is a version-2 object: `schema_version`, `watching`, `sessions`,
`next_cursor`, and `coverage`. `watching` describes the notification accelerator;
poll mode can still observe writes. `coverage.complete` concerns the scanned
roster and known metadata, while `next_cursor` indicates more candidate rows.
An empty page can have a continuation after dead-process candidates are filtered.
Activity can reorder pages between calls. `--fields` projects fields inside
`sessions`, retaining coverage and continuation. Fleet returns per-host coverage,
watch state and cursors plus explicit global truncation. Legacy array-only hosts
have unknown coverage; continue on each host separately rather than passing a
fleet cursor.

```bash
recall live [OPTIONS]
recall live mark --session <id> --pid <pid>
```

| Option | Description |
|--------|-------------|
| `--all` | Also include recent idle and ended sessions (idle window defaults to 24 h), plus ended rows still watched |
| `--source` | Filter by source: `claude-code`, `codex`, `pi-agent`, `grok`, `kimi-code` |
| `--project` | Filter by git repo path |
| `--host` | Exact match on session host label |
| `--fresh` | Catch up already-indexed rows on this page; does not first-index |
| `--limit` | Maximum live candidates inspected per page (1..256, default 50) |
| `--cursor` | Continue a local page with the prior `next_cursor` and the same filters |
| `--fleet` | Fan-out to hosts in `fleet.toml` and merge by liveness then activity |
| `--fleet-config` | Alternate fleet inventory path |
| `--json` | Output results as JSON |

Each row carries `liveness` (`active`, `idle`, `ended`, `unknown`), `freshness`
(how far the index is behind the file on disk), `turn` (what the agent is doing
this turn, including any running tool), `cursor` for a follow-up
`show --after`, and `writer_pid` when a hook marked the session.

Select an indexed row's `id` or `source_session_id` for a bounded read:

```bash
recall live --json
recall live --project <path> --json
recall live --all --json
recall show <session-id> --tail 20 --fresh --json
```

An `ended` row records observed exit evidence; it is excluded by default and
available through `--all` while still watched or within the recent window.
Use `recall list` for older history. A filtered empty roster can exclude a
session before its first index pass; retry unfiltered before concluding absence.

`--fresh` on `live` catches up already-indexed rows on the selected page whose
index is behind the file, then re-reads. The wait is the lesser of
`live.fresh_timeout` and the 2 s responsive-read bound. It does not first-index
a transcript the coordinator has not committed: those rows stay
`current: false` with limitation `not_yet_indexed`. A catch-up that blows the
budget still comes back, and a `note:` on stderr names how many eligible rows
are still behind. `--fresh` cannot be combined with `--fleet`: the flag is not
forwarded over the SSH hop, so the combination would promise freshness it did
not deliver.

`--fresh` does not force discovery of an unseen path. First index is the fair
coordinator's after resume and inventory. Periodic inventory reconciles old
imports, missed events and subscription overflow. Continuous writes have a
maximum coalescing delay, so a busy source cannot remain unindexed merely by
resetting a quiet-time debounce. A resumed source becomes active when its next
write is observed. Use `show --fresh` to index one already-known session's
pending bytes, bounded by `live.fresh_timeout` (default 10 s). A brand-new or
resumed transcript waits for the discovery loop (default 30 s).

Before indexing, a catalog path has a source but may lack project, host and session
identity. Such unknowns appear in `coverage.unknown_count`
and a sample of at most sixteen `filtered_unknown_paths`; retry unfiltered to
inspect them. Matching a known source does not require the first parse.

**Example output:**
```
[2026-09-08 12:00] active abc123 (claude_code) /path/to/repo turn=working tool=Bash fresh
[2026-09-08 11:40] idle def456 (codex) /path/to/other turn=awaiting_input lag=2048
```

### recall live mark

Records the writer PID observed for a harness session. The
mark supplies evidence for `ended` detection; it does not
send input, stop a process, or resume work. Use the harness's source session ID
(not recall's generated ID), actual writer PID, and source:

```bash
recall live mark --session <source-session-id> --pid <writer-pid> --source <source> --json
```

In a `SessionStart` hook, use the IDs and process relationship that the harness
actually supplies. `$PPID` is suitable only when that hook's parent is the
writer; an observing shell's PID does not identify the observed agent.

| Option | Description |
|--------|-------------|
| `--session` | The harness's own session id (required) |
| `--pid` | Process id of the agent writing this session (required) |
| `--source` | Defaults to `claude-code` |
| `--json` | Output the recorded mark as JSON |

The mark is enrichment, so the command **exits 0 even with no daemon running**
and never forks one — a hook must not fail the agent's session start. It then
reports `"marked": false` with the reason. Inspect `marked`, not only the exit
code. Everything else about `recall live`
works without a mark; what a mark adds is `ended` detection (the pid is probed
with `kill(pid, 0)` on the host that stamped it) and the row's `writer_pid`.

## recall show

Read a current or historical session, with bounded tails, cursor deltas, or a
deadline-limited stream for ongoing work. `recall live` discovers current
activity; `recall list` and `recall search` locate past work.

```bash
recall show <session-id> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--tools` | Include tool calls in output |
| `--thinking` | Include thinking blocks (Claude Code); absent from every message without it |
| `--tail` | Last N messages by index, instead of the whole session |
| `--after` | Only messages after an opaque `cursor` from an earlier read |
| `--fresh` | Have the daemon index pending bytes before answering |
| `--follow` | Stream new messages as NDJSON until a deadline |
| `--timeout` | Seconds `--follow` streams for (default 60) |
| `--message-limit` | First N messages (head-anchored; refused with `--tail`/`--after`) |
| `--fleet` | Resolve the session via fleet hosts |
| `--host` | Fleet host name (required when the id exists on multiple hosts) |
| `--fleet-config` | Alternate fleet inventory path |
| `--json` | Output results as JSON |

Each message carries only what was asked for. Without `--thinking` the
`thinking` field is absent (not null) and `has_thinking` still tells you it
exists; without `--tools` a turn whose payload is tool calls collapses to a
summary row — `id`, `idx`, `role`, `timestamp`, `has_thinking`, `summary` —
naming the flag that reveals the rest. Re-read with `--tools` (or
`--tools --thinking`) rather than treating a summary row as an empty turn.

Every `show` reports `freshness` and a `cursor`, not just a `--fresh` one: a
caller cannot tell a finished transcript from a lagging one by reading its
messages. Feed the `cursor` back as `--after` to read only what arrived since;
an empty delta hands the same cursor back rather than a lower one.

```bash
recall show <session-id> --tail 20 --fresh --json
recall show <session-id> --after '<cursor>' --fresh --json
recall show <session-id> --after '<cursor>' --follow --timeout 60
```

A rewrite invalidates an incompatible cursor: `show` returns `cursor_reset: true`
and a replacement window; follow closes explicitly with `content_rewritten`.
Replace the old window and adopt the new cursor instead of treating replacement
messages as an append.

Copy the opaque cursor from the previous response unchanged. Check
`freshness.current` and stderr after `--fresh`; reaching its refresh budget
still returns the available data, which can be stale.

For `--tail`, `--after`, or `--fresh`, an incompatible daemon response produces
a `RUNTIME` error. Run the client and daemon from the same recall revision.

`--follow` emits NDJSON regardless of `--format` — one JSON object per line,
deltas and then exactly one `closed` line naming why the stream ended
(`timeout`, `daemon_stopped`, or `content_rewritten`) — so a `while read line` consumer terminates.
It cannot be combined with `--fleet` (one SSH per poll is the wrong transport)
or with `--tail` (a follower starts from `--after` or from now, so it has no
tail window), and `--timeout` only applies to it. `--tail`/`--after` are
likewise refused with `--message-limit`, which anchors at the head, and with
`--fleet`.

**Example output:**
```
[2024-01-15 10:30] Session abc123def456 (claude_code)
Project: /path/to/my-app
Duration: 1234s | Messages: 25 | Tools: 12

[10:30:15] user:
Fix the authentication bug

[10:30:45] assistant:
I'll look into the authentication code...
  [Read] src/auth.py
  [Edit] src/auth.py
```

## recall stats

Analytics subcommands for usage patterns.

### recall stats (overview)

```bash
recall stats [--json]
```

**Output:**
```
Overview
  Sessions: 142
  Messages: 3567
  Tool calls: 1234
  Bash calls: 456
```

### recall stats tools

Tool usage frequency.

```bash
recall stats tools [--json]
```

**Output:**
```
Bash: 456
Read: 312
Edit: 234
Glob: 156
Grep: 89
Write: 45
```

### recall stats bash

Bash command breakdown and permission suggestions.

```bash
recall stats bash [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--suggest` | Generate permission suggestions for auto-approval |
| `--json` | Output results as JSON |

**Without --suggest:**
```
git status: 45
npm test: 32
git diff: 28
npm install: 15 (compound)
```

**With --suggest --json:**
```json
{
  "suggestions": [{"pattern": "git status *", "count": 45, "confidence": "high", "reason": "..."}],
  "skipped": [{"pattern": "rm *", "count": 12, "reason": "Destructive command"}]
}
```

`confidence` is `high`, `medium`, or `review`; `review` covers compound commands
and low usage volume and needs a human decision before it becomes an allow rule.
`skipped` lists destructive commands that are never suggested.

### recall stats skills

Fail-closed skill invocation census across Claude Code, Codex, Pi Agent, Grok,
and Kimi Code.

```bash
recall stats skills [--since 30d] [--source codex --source grok] [--local] [--json]
```

| Option | Description |
|--------|-------------|
| `--since` | Time window such as `30d` or `12h`; the all-time control ignores this bound |
| `--source` | Source to include; repeat for multiple sources |
| `--local` | Query only the current daemon instead of the configured fleet |
| `--fleet-config` | Alternate fleet inventory path |
| `--json` | Emit `{rows, coverage}` as JSON |

Without `--local`, the command queries the current daemon and every configured
fleet host. Missing inventory, any remote failure, unsupported remote CLI,
invalid payload, or incomplete source coverage exits nonzero and emits no
census. Each row reports `skill_name`, `source`, `host`, `invocations`, and
distinct `sessions`. Coverage reports expected/successful hosts, covered
sources, windowed population counts, unattributed candidates, and an all-time
`control` population. An empty control is a broken or empty detector, not a
never-fired result.

### recall stats tokens

Token usage by project.

```bash
recall stats tokens [--json]
```

**Output:**
```
/path/to/project-a: 125000 in / 45000 out
/path/to/project-b: 89000 in / 32000 out
```

### recall stats usage

Token totals for the fleet ledger, grouped by source x model x host.

```bash
recall stats usage [--since 30d] [--fleet] [--json]
```

| Option | Description |
|--------|-------------|
| `--since` | Time window like `30d` or `12h`, same grammar as `recall list --since` |
| `--fleet` | Fan-out to hosts in `fleet.toml` and sum rows by source × model × host |
| `--fleet-config` | Alternate fleet inventory path |
| `--json` | Output rows as JSON |

Each row reports `input_tokens`, `cached_input_tokens`, `output_tokens`,
`fresh_input_tokens` (`input - cached`, when both are known), and `session_count`.
Sessions whose input and output tokens are both NULL are excluded, so a Grok
session that has not been harvested yet does not dilute totals. There are no cost
or currency fields.

**JSON row:**
```json
{
  "source": "claude_code",
  "model": "claude-opus-4-8",
  "host": "build-box",
  "input_tokens": 5875495890,
  "cached_input_tokens": 0,
  "output_tokens": 28721331,
  "fresh_input_tokens": null,
  "session_count": 125
}
```

The `host` dimension is populated by `recall index --host` (see `recall index`)
and is always present on list / search / show / stats usage rows.

## recall fleet status

Probe hosts listed in `~/.config/recall/fleet.toml` (or `RECALL_FLEET_PATH` /
`--config`): reachability, binary version, daemon version, version drift.

```toml
# ~/.config/recall/fleet.toml
[[host]]
name = "devbox"
ssh = "devbox.example.ts.net"
```

```bash
recall fleet status
recall fleet status --json
```

Live fleet queries use one-shot SSH (`BatchMode=yes`, `RemoteCommand=none`).
Failed hosts are skipped with a stderr warning; query verbs succeed if any host
returns data. Prefer `--fleet` on list / search / stats usage / show for day-to-day
multi-host views; use multi-host ingest (`index --root`) for offline bulk import.

## JSON Output

All commands support `--json` for machine-readable output. JSON output includes all fields and is suitable for piping to `jq` or programmatic processing.

```bash
recall list --json | jq '.[] | .id'
recall stats tools --json | jq 'max_by(.count)'
```

## Data Locations

| Data | Path |
|------|------|
| Database | `~/.local/share/recall/recall.duckdb` |
| Lock file | `~/.local/share/recall/recall.lock` |
| Claude Code sessions | `~/.claude/projects/**/*.jsonl` |
| Codex sessions | `~/.codex/sessions/**/rollout*.jsonl` |
| Pi Agent sessions | `~/.pi/agent/sessions/**/*.jsonl` |
| Grok sessions | `~/.grok/sessions/**/chat_history.jsonl` |
| Grok usage log | `~/.grok/logs/unified.jsonl` |
| Kimi Code sessions | `~/.kimi-code/sessions/**/wire.jsonl` |
