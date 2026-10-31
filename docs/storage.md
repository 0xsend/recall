# Storage

recall stores its index in DuckDB at `~/.local/share/recall/recall.duckdb`. When
the SQLite FTS sidecar backend is enabled, its keyword index lives beside it in
`recall.fts.sqlite`. The daemon owns both files.

## Large corpora and DuckDB memory

The DuckDB buffer pool defaults to a `2GB` `memory_limit`, independent of host
RAM. Explicit limits above that allowance are clamped unless the configuration
opts out. Smaller limits are honored, and `RECALL_DUCKDB_MEMORY_LIMIT` takes
precedence over the config file while following the same clamp policy.

```toml
# ~/.config/recall/config.toml
[duckdb]
memory_limit = "4GB"
# Explicit opt-out: apply the requested limit even above the default allowance.
memory_limit_exceeds_floor = true
```

Without the opt-out, this example applies the default allowance and logs one
startup warning naming the requested limit, the applied limit and the opt-out
key. `recall daemon status` shows requested, applied and clamped values in text;
JSON exposes `duckdb_memory_limit_requested`, `duckdb_memory_limit_applied` and
`duckdb_memory_limit_clamped`. For a running daemon, requested is null when no override was supplied.
These fields describe the running daemon; all three are null when only
local offline status is available. Restart the daemon after changing these
settings. The opt-out must be a TOML boolean.

For compatibility, explicit memory sizes normalize to binary units (`4GB`
becomes `4GiB`); the unchanged default `2GB` is decimal (2,000,000,000 bytes).
Comparison uses actual bytes, so even an explicit `2GB` is clamped unless opted
out. Invalid sizes fail startup regardless of the opt-out.

The buffer limit is not a total daemon RSS ceiling: models, Python/native
allocations and allocator retention also contribute. Opting out may exceed the
4 GiB whole-process resource floor. Large operations can still fail with
`Out of Memory Error`; measure the complete workload before choosing a larger
limit. See [DuckDB memory management](https://duckdb.org/2024/07/09/memory-management).

## Storage format maintenance

recall pins DuckDB 1.5.5 and creates new databases with fixed storage format
`v1.2.0`. Ordinary opens preserve existing formats. The running daemon can
convert an older file through explicit maintenance with a consistent DuckDB
and keyword-sidecar backup:

```bash
recall daemon migrate-storage --dry-run --json
recall daemon migrate-storage --plan-id <digest-from-preview> --wait --json
recall daemon migrate-storage --dry-run --json  # inspect the same operation
```

Without `--wait`, apply returns acceptance while the daemon continues working.
The wait defaults to five seconds; `--timeout` controls the caller's overall
maintenance wait, and each RPC read remains bounded by five seconds. A wait
timeout does not cancel the operation. Inspect its status before retrying.
Retries retain the original backup and any captured index-migration scope.
Successful completion changes the plan digest. If an exact command retry reports
a stale plan, inspect again and use the current digest; the completed operation
and its original backup remain the same.
Recovery backups are excluded from automatic snapshot-age cleanup.
Storage-only maintenance preserves the pause switch and index version;
completion of a combined migration also requires its captured index scope to
verify. Current or newer formats are never downgraded.

## Runtime compression policy

Owned DuckDB connections use automatic compression with Roaring excluded before
schema or data writes. Existing segments remain readable and writable. The
policy controls future encodings; no codec scan, write fence, journal or
historical conversion runs on open. The existing `migrate-storage` command
continues to handle fixed-header maintenance only. Codec conversion is not an
available maintenance operation.

Compaction uses the same runtime policy. Before replacing the source file it
closes and reopens the candidate, checks table counts and inspects persisted
segments for Roaring. A failed candidate check leaves the source in place.

During explicit fixed-header maintenance, mutating RPC requests return
`APP_LOCKED` with `reason: storage_maintenance`. Retry after inspecting status.
Status serves the captured observation only while the operation is running,
marked `storage_maintenance.snapshot: true`. After success or failure it returns
current database state, `snapshot: false`, the terminal phase and `error`.
A failed operation does not permanently freeze status or write admission.
Text status includes the same phase, snapshot marker, error and write admission.

## Bloat ratio and auto-compaction

The bloat ratio that `recall compact` and `recall daemon status` report is
**used bytes / live bytes**:

- *used bytes* (`before_used_bytes`, "Before used") are the DuckDB blocks that
  hold data, including dead row versions that updates left behind;
- *live bytes* (`before_live_bytes`, "Before live") estimate what the current
  rows would occupy after a compaction;
- the file size (`before_bytes`, "Before size") can be larger than the used
  bytes. Each checkpoint writes the row groups it changed into fresh blocks
  and frees the old ones for the next checkpoint to reuse. That free space is
  headroom, not bloat, and does not count toward the ratio.

Free blocks left by large deletions are excluded the same way. They do not
raise the ratio and never trigger automatic compaction, so the file can stay
well above its used bytes while the ratio reads low. Later writes reuse that
space; run `recall compact --yes` to return it to the filesystem.

A freshly compacted database reads about 1.1. The daemon checks the ratio
every `check_interval_hours` of uptime (default 6) and compacts when it
reaches `bloat_ratio_threshold` (default 1.5) and the file is at least
`min_bytes`; see `[compaction]` in the configuration. Status shows the value
from the last check or daemon start, and warns on stderr when it is at or
above the threshold. `recall compact` defaults to `--threshold 1.0`, so a
manual run compacts unless told otherwise, and also returns free space to the
filesystem.
