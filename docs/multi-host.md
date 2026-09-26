# Multiple hosts

recall can query several machines live over SSH, or fold synced session trees
from other machines into one local database.

## Fleet query (live)

For a live fleet (Tailscale / SSH), prefer on-demand fan-out to each host's
daemon instead of re-copying session trees. Each machine remains authoritative
for its own DuckDB (including Grok usage harvest).

```toml
# ~/.config/recall/fleet.toml  (or set RECALL_FLEET_PATH)
[[host]]
name = "devbox"
ssh = "devbox.example.ts.net"

[[host]]
name = "build-box"
ssh = "build-box"
```

```bash
recall fleet status
recall stats usage --fleet --since 7d
recall list --fleet --since 7d
recall search "git rebase" --fleet
recall show <session-id> --fleet --host devbox   # --host if ambiguous
```

Transport is one-shot SSH per host (`BatchMode`, `RemoteCommand=none`). Unreachable
hosts are skipped with a stderr warning; the command succeeds if any host returns
data. List / search / show / stats usage rows always include `host`.

## Offline ingest

`recall` discovers sessions under `$HOME`. To fold another machine's sessions into
one database, sync that machine's source trees to a local directory and index the
directory as an alternate home root:

```bash
recall index --root PATH --host NAME
```

`--root PATH` treats `PATH` as a home directory laid out like `$HOME` (`.claude/`,
`.codex/`, `.pi/`, `.grok/`, `.kimi-code/`). `--host NAME` labels every session
indexed in that run and defaults to the basename of `--root`; ordinary local runs
label sessions with the machine's short hostname. Re-indexing the same tree is
idempotent: session identity is a source+path hash, so host and token totals
update in place instead of duplicating rows.

Query the result with `recall stats usage`, which groups by source x model x host.

Ingest is a local operation on synced files. Daemon clustering, remote RPC, and
cross-host write coordination are out of scope.

### rsync recipe

Pull a remote host's session trees, including its Grok usage log, into a local
root, then index that root under the remote host's name:

```bash
REMOTE=user@build-box
ROOT="$HOME/.local/share/recall/hosts/build-box"
mkdir -p "$ROOT"

rsync -az --delete --prune-empty-dirs \
  --include='.claude/'    --include='.claude/**' \
  --include='.codex/'     --include='.codex/**' \
  --include='.pi/'        --include='.pi/**' \
  --include='.grok/'      --include='.grok/**' \
  --include='.kimi-code/' --include='.kimi-code/**' \
  --exclude='*' \
  "$REMOTE:" "$ROOT/"

recall index --root "$ROOT" --host build-box
```

Run it on a schedule (cron, systemd timer, launchd) to keep the view current.

### Grok log rotation cadence

Grok records exact per-inference token usage only in `~/.grok/logs/unified.jsonl`,
and that file rotates; a busy host can roll it within roughly a day. Whatever
rotates away before it is harvested is lost, because Grok token totals are only
ever filled from this log and are never inferred from transcript length.

Either harvest host-side (run `recall` on the remote machine, advancing its cursor
against the live log) or sync the log faster than it rotates. Harvest is
incremental through a byte-offset cursor and is rotation-safe: a file that shrinks
resets the cursor and is re-read from the start, and stable per-event keys keep
re-harvesting from double-counting.
