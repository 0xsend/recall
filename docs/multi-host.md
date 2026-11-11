# Multiple hosts

recall can query the calling daemon over local RPC and selected remotes over SSH, or fold synced session trees
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

Transport is local RPC for a selected caller and one-shot SSH per selected remote
(`BatchMode`, `RemoteCommand=none`). Structured list/search/show/live/usage rows
include `host`; fleet text uses `host=NAME` on each list/search/live row and
the show session header, as usage text already does. Nonempty names containing only
ASCII letters, digits, `.`, `_`, and `-` stay plain; all others are JSON string
literals with ASCII escaping, for example `host="edge host=trusted"` and
`host="edge\nforged"`. Usage and skill-census host text use the same encoding,
including ordinary usage. Other ordinary data output is unchanged.
Host fragments in ambiguity, scope and fleet failure text diagnostics, and
inventory names in fleet status, use the same encoding. Plain-name diagnostics
and structured errors keep their existing values, shape, code and exit. The
encoding is injective and exactly recoverable for Unicode scalar values plus
isolated surrogates; explicit adjacent high/low surrogate code points forming a
pair cannot enter through storage, remote JSON, TOML or the OS hostname.
Structured host values remain raw and recoverable.

`list`, `search`, `show`, `live`, and `stats usage` accept `--fleet` to query
the calling daemon over local RPC plus inventory endpoints. Recognized self
contributes once. List/search/live apply a single global limit, so a selected
host may have no rows in the returned page. Failed selected endpoints, including
the caller, produce `fleet: skip` warnings on stderr; list/search/live/usage
succeed if any selected endpoint succeeds (even with an empty reply), while show
keeps unique-hit/ambiguity/not-found rules. The census requires every endpoint.
Structured consumers must retain stderr for partial-result diagnostics.
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

```toml
# A self entry on caller aem5 (either canonical name or this declaration)
[[host]]
name = "laptop"
hostname = "aem5"
ssh = "my-ssh-alias"
```

Local RPC uses existing query parameters without forwarding the endpoint `--host`
selector as an origin filter. One pool (default six) bounds endpoint work; ordinary
local/remote endpoints have 45-second hard budgets (SSH connect 10s), local live
keeps 2s, census local/remote keeps 600/660s. Ties retain local-first/inventory order.
Advisory status is at most one live local read after selected local success;
failed/unselected local has no extra probe. Fleet empty-tool search only advises
retrying without the filter, without a local unfiltered count. Live reports each
endpoint's actual watch state/coverage/cursor in `coverage.hosts`; combined watch
state/cursor are null, and legacy arrays have unknown coverage.


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
