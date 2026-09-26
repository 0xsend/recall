# QA Runbook

Manual checks to confirm `recall` is healthy on the host. Designed for two cadences:

- **Smoke test** (~2 min) — run weekly, after upgrades, or whenever something feels off.
- **Deeper validation** — run before releases, after schema changes, or when investigating drift.

The CLI abstracts over schedulers, so most checks work the same on macOS and Linux. Where commands diverge, OS-specific blocks are labeled `macOS (launchd)` and `Linux (systemd-user)`.

> Host checks use the stable installed `recall` on `PATH`. Use `uv run` only for
> isolated development checks; never run host `daemon install` from a development
> binary. See [CONTRIBUTING.md](CONTRIBUTING.md#development-and-stable-installs)
> for stable promotion and extras.
> [RECONCILIATION_QA.md](RECONCILIATION_QA.md) defines release-scale reconciliation
> checks; these smoke checks do not establish full freshness or coverage acceptance.

## Platform reference

| Concern | macOS (launchd) | Linux (systemd-user) | Linux (cron) |
|---|---|---|---|
| Scheduler artifact | `~/Library/LaunchAgents/it.send.recall.daemon.plist` | `~/.config/systemd/user/recall-daemon.service` (+ `recall-daemon.timer` in poll mode) | `crontab -l` line + `~/.local/share/recall/logs/daemon.{log,err.log}` |
| Inspect | `launchctl print gui/$(id -u)/it.send.recall.daemon` | `systemctl --user status recall-daemon.service` | `crontab -l` |
| Reload after edit | `launchctl bootout gui/$(id -u)/it.send.recall.daemon && launchctl bootstrap gui/$(id -u) <plist>` | `systemctl --user daemon-reload && systemctl --user restart recall-daemon.service` | next cron tick |
| Stop | `launchctl bootout gui/$(id -u)/it.send.recall.daemon` | `systemctl --user disable --now recall-daemon.service recall-daemon.timer` | edit crontab |

An install made before the label rename runs as `xyz.metalrodeo.recall.daemon`; substitute that label until `recall daemon install` migrates it (`recall daemon status` says when).

The status checks below read `recall daemon status --json` with `jq`.

Cross-platform paths (XDG-based on both):

- DB: `~/.local/share/recall/recall.duckdb`
- Lock / socket / pid: `~/.local/share/recall/recall.{lock,sock,pid}`
- Logs: `~/.local/share/recall/logs/daemon.{log,err.log}`
- Config: `~/.config/recall/config.toml` (optional; defaults applied if absent)

---

## 1. Smoke test

Run these in order. Stop at the first failure and jump to **Troubleshooting**.

### 1.1 Daemon is installed and healthy

```bash
recall daemon status --json | jq '{installed, runtime_unavailable_reason, startup_refusal, scheduler_health_state, last_failure_message: .runtime_status.last_failure_message, failed: .runtime_status.last_index_summary.failed}'
```

Expect:

- `installed: true`
- `runtime_unavailable_reason: null` and `startup_refusal: null` — the daemon RPC answered
- `last_failure_message: null` — the authoritative indexing health signal
- `failed: 0` (or null when no summary exists yet)
- A live socket: `ls ~/.local/share/recall/recall.sock` exists

Notes:

- `runtime_status.last_attempted_at` / `last_successful_at` may show stale timestamps on a freshly-respawned watch-mode daemon — there are no discrete "runs" in watch mode, only event-driven indexing. If `recall list` and `recall search` return current data, the daemon is healthy regardless of those fields.
- `scheduler_health_state` of `ok`, `unknown` (launchctl output could not be parsed), or `null` is fine. `failed` is suspicious but is sometimes a stale launchd/systemd record from a prior install race (see Troubleshooting → Stale scheduler exit code).

### 1.2 Mode matches install (no drift)

```bash
recall daemon status --json | jq '{mode, resolved_mode, installed_mode, mode_mismatch_reason}'
```

Expect:

- `installed_mode` matches `resolved_mode`
- `mode_mismatch_reason: null`

If non-null, the running daemon's mode disagrees with what's declared on disk. See **Troubleshooting → Mode mismatch**.

### 1.3 Plist / unit points at a stable binary

```bash
recall daemon status --json | jq -r .command
```

Expect a system path:

- `~/.local/bin/recall daemon --mode watch` (expanded to your home) or another stable tool path — **not** a project venv path like `.venv/bin/recall`.

A venv path means the install was run via `uv run` from a checkout; the plist/unit will break if that venv is rebuilt or moved. See **Troubleshooting → Stale binary**.

### 1.4 Recent sessions are in the index

```bash
recall list --since 1d --json | jq -r '.[].source' | sort | uniq -c
```

Expect every agent you used in the last 24h to be represented.

### 1.5 Index is keeping up with new files

```bash
recall daemon status --json | jq '{watch_total_indexed, watch_total_failed, watch_last_event_at, watcher_subscription_count, live_session_count, installed_binary_stale}'
```

Expect (watch mode):

- `watcher_subscription_count` > 0 while agent sessions are live
- `watch_last_event_at` recent if you've used an agent in the last few minutes (otherwise null is fine — no events to process)

> `installed_binary_stale` is `true` whenever the service's baked-in binary differs from the `recall` your current shell resolves, including when you run from a checkout. Prefer `.command` from §1.3 as the authoritative check.

For poll mode, prefer the timestamp fields:

```bash
recall daemon status --json | jq '.runtime_status | {last_attempted_at, last_successful_at}'
```

`last_attempted_at` should be within the daemon interval (`daemon.interval`, default 300 seconds).

### 1.6 Search returns hits

```bash
recall search "git" --limit 3 --json | jq 'length'
```

Expect a number ≥ 1.

### 1.7 Stats are populated

```bash
recall stats tools --json | jq 'length'
```

Expect a number > 5 (typical hosts will see dozens of distinct tool names).

---

## 2. Deeper validation

Run before a release, after a schema change, or when a smoke-test signal is suspicious.

### 2.1 Parser correctness — Claude Code

Recall computes its own session ID as `sha256(f"{source}:{path}")[:32]`, so look up the row first, then compare counts against the raw JSONL.

```bash
ROW=$(recall list --since 1d --source claude-code --json --limit 1)
SID=$(echo "$ROW" | jq -r '.[0].id')
INDEXED_MSGS=$(echo "$ROW" | jq -r '.[0].message_count')
SRC=$(recall show "$SID" --json | jq -r '.source_path')
RAW_MSGS=$(grep -c '"type":"user"\|"type":"assistant"' "$SRC")
echo "indexed=$INDEXED_MSGS raw=$RAW_MSGS path=$SRC"
```

Indexed count should be close to raw user/assistant lines. Large divergence (e.g. raw=120, indexed=30) signals parser drift on a new schema field.

### 2.2 Parser correctness — Codex

```bash
ROW=$(recall list --since 1d --source codex --json --limit 1)
SID=$(echo "$ROW" | jq -r '.[0].id')
INDEXED_MSGS=$(echo "$ROW" | jq -r '.[0].message_count')
INDEXED_TOOLS=$(echo "$ROW" | jq -r '.[0].tool_count')
SRC=$(recall show "$SID" --json | jq -r '.source_path')
RAW_LINES=$(wc -l < "$SRC")
echo "indexed_msgs=$INDEXED_MSGS indexed_tools=$INDEXED_TOOLS raw_lines=$RAW_LINES path=$SRC"
```

For Codex, messages + tools combined should track raw line count loosely (one event per line). Wildly low indexed counts vs raw lines means the parser is dropping events.

### 2.3 Embedding backend is loaded and serving queries

```bash
recall daemon status --json | jq '{embed_phase_enabled, embed_model_loaded, embed_pending, embed_deferred_reason, embed_last_error}'
recall search "any reasonable phrase" --mode vector --limit 3 --json | jq 'length'
```

Expect:

- `embed_phase_enabled: true`
- `embed_model_loaded: true` once the daemon has run an embed pass
- Vector search returns ≥ 1 hit

If vector search returns empty but keyword search works, embeddings haven't backfilled yet. Run `recall index` and check `embed_pending` trends down.

### 2.4 Full-text search still works

Pick a string you know exists verbatim in a recent session (e.g. a unique error message, a function name).

```bash
recall search "<known-exact-string>" --mode keyword --json | jq 'length'
```

Expect ≥ 1 hit. Empty here usually means FTS index is stale → see **Troubleshooting → Empty search**.

### 2.5 Daemon recovery (KeepAlive / Restart=always)

```bash
PID=$(cat ~/.local/share/recall/recall.pid)
kill -9 "$PID"
sleep 5
ls -la ~/.local/share/recall/recall.sock
cat ~/.local/share/recall/recall.pid
```

Expect: socket reappears, new PID is different from the killed one. Confirms launchd `KeepAlive` (macOS) or systemd `Restart=always` (Linux watch mode) re-spawned cleanly.

> Skip on cron-backed installs — there's no long-lived process to kill.

### 2.6 Drift detection still works

Inject a known mismatch, confirm the daemon reports it, then restore.

**macOS (launchd):**

```bash
cp ~/Library/LaunchAgents/it.send.recall.daemon.plist /tmp/recall.plist.bak
# Hand-edit the plist to swap --mode watch for --mode poll, then bootstrap:
launchctl bootout gui/$(id -u)/it.send.recall.daemon
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/it.send.recall.daemon.plist
recall daemon status --json | jq .mode_mismatch_reason  # expect non-null
recall daemon install --scheduler launchd          # restore
```

**Linux (systemd-user):**

```bash
cp ~/.config/systemd/user/recall-daemon.service /tmp/recall-daemon.service.bak
sed -i 's/--mode watch/--mode poll/' ~/.config/systemd/user/recall-daemon.service
systemctl --user daemon-reload
systemctl --user restart recall-daemon.service
recall daemon status --json | jq .mode_mismatch_reason  # expect non-null
recall daemon install --scheduler systemd          # restore
```

### 2.7 Backup integrity

A confirmed `recall index --recreate` moves the previous database to `recall.bak-<UTC timestamp>` beside it; `recall snapshots list` shows retained snapshots. When you keep one, confirm it opens cleanly:

```bash
NEWEST_BAK=$(ls -t ~/.local/share/recall/recall.bak-* | head -1)
duckdb "$NEWEST_BAK" -c "SELECT count(*) AS sessions FROM sessions"
```

### 2.8 Watched directories cover all your projects

```bash
recall daemon status --json | jq .watched_dirs
```

Expect the session directory of every agent you use (see the data locations in [README.md](README.md#data-locations)). A missing root means new sessions in that directory won't be picked up by watch mode (but `recall index` would still catch them on a full pass).

---

## 3. Troubleshooting

### Mode mismatch (`mode_mismatch_reason` non-null)

The running daemon's resolved mode disagrees with the mode encoded in the installed plist/unit. Most often caused by re-running `daemon install` from a checkout, or by `watchdog` becoming available after a previous install.

**Fix (re-install with current resolution):**

macOS:

```bash
launchctl bootout gui/$(id -u)/it.send.recall.daemon
recall daemon install --scheduler launchd
```

Linux (systemd-user):

```bash
systemctl --user disable --now recall-daemon.service recall-daemon.timer 2>/dev/null
recall daemon install --scheduler systemd
```

Then confirm:

```bash
recall daemon status --json | jq '{installed_mode, resolved_mode, mode_mismatch_reason}'
```

### DB lock conflict (`Conflicting lock is held in ... PID N`)

A previous daemon is still holding `recall.duckdb`. Identify it and stop cleanly.

```bash
ps -p <PID>
recall daemon stop          # graceful
# if that fails:
launchctl bootout gui/$(id -u)/it.send.recall.daemon  # macOS
systemctl --user stop recall-daemon.service                  # Linux
```

If the daemon was started from a project venv (`uv run recall daemon start`) and you've since closed the shell, the process may be orphaned — `kill <PID>` directly.

### Stale binary — venv path in `command:`

If `recall daemon status --json | jq -r .command` shows a project venv path (e.g. `/Users/<you>/<project>/.venv/bin/recall`), the install was run via `uv run` from a checkout. The plist will break when that venv is rebuilt or moved. Re-run install from the **stable system binary**:

```bash
# Use the stable binary explicitly, not `uv run`:
~/.local/bin/recall daemon install --scheduler launchd   # macOS
~/.local/bin/recall daemon install --scheduler systemd   # Linux
```

> `installed_binary_stale: true` on its own only means your shell resolves a different `recall`; `.command` is the authoritative signal.

### Empty search after upgrade

FTS index or embeddings table can lag a schema change. Rebuild from scratch:

```bash
recall index --recreate --yes
```

This re-parses every session file, rebuilds FTS, and re-embeds. Expect a multi-minute run on a host with thousands of sessions.

### Stale scheduler exit code (`scheduler_health_state: failed` but daemon serves requests)

After an install / reinstall cycle, launchd or systemd can record a non-zero exit from a transient race — typically a respawn that hit "daemon already running (pid N). Use `recall daemon stop` first." The recorded exit then surfaces as `scheduler_health_state: failed` even though the surviving daemon is healthy.

Confirm the daemon is actually healthy:

```bash
recall list --since 1h --limit 1 --json | jq '.[0].id'   # returns an ID
recall search "test" --limit 1 --json | jq 'length'      # returns 1
recall daemon status --json | jq .runtime_status.last_failure_message   # null
```

If those pass, the `failed` reading is stale. To clear it, force a clean respawn:

macOS:

```bash
PID=$(cat ~/.local/share/recall/recall.pid)
kill -9 "$PID"   # launchd KeepAlive will respawn within a few seconds
```

Linux (systemd-user):

```bash
systemctl --user restart recall-daemon.service
```

If `scheduler_health_state` stays `failed` across respawns, check the err log for an actual error.

### Daemon never finds new sessions

Confirm the relevant project root is watched:

```bash
recall daemon status --json | jq .watched_dirs
```

If it isn't, restart the daemon — the discovery loop re-scans on startup. If an agent keeps its sessions somewhere non-default, list the directories under `[sources.<source>] roots` in `~/.config/recall/config.toml`; configured roots replace the built-in location for that source.

### Indexing fails repeatedly

```bash
tail -100 ~/.local/share/recall/logs/daemon.err.log
recall daemon status --json | jq '.runtime_status | {last_failure_message, last_failure_at}'
```

Common causes: disk full (the DB grows with your history and can become large), corrupted JSONL on the source side, schema drift from a new agent version. The error message points at the offending file. For a backlog that is not draining, see [docs/reconciliation-operations.md](docs/reconciliation-operations.md).

---

## Notes

- **Baseline**: capture expected counts (`recall stats` sessions, `stats tools` top entries, `embed_pending`) on a known-good host so future runs have a comparison point.
- **Scope**: this is a manual runbook. Schema migrations are covered by `tests/test_db/test_migrations.py` and `tests/test_core/test_schema_cutover.py`.
