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

## Codec profile maintenance

Owned connections use automatic compression with Roaring excluded
(`automatic-without-roaring-v1`). The runtime policy affects future encodings;
it does not rewrite historical segments. On every daemon connection open,
inspection of persisted segments fences ordinary writes when Roaring remains.
Status exposes `write_admission.reason: codec_maintenance_required`, and a
mutating RPC returns `APP_LOCKED` with that typed reason and the maintenance
action. Inspection and reads remain available before explicit conversion.

Use `migrate-storage --dry-run --json` to inspect `desired_codec_profile`,
`codec_eligible`, `codec_phase` and `codec_backup_path`. Complete any existing
format/index operation with its original backup first; at the current fixed
header, inspect a new plan for codec conversion. Applying that plan creates a
consistent preimage backup, verifies its file hashes and builds a separate
DuckDB candidate on the same filesystem. It preserves IDs, logical schema,
indexes/constraints, stored context, NULL markers and vector bits without
reparsing or regenerating data. Exact streamed validation and a fresh reopen
must pass before an atomic replacement of DuckDB. SQLite remains unchanged.
Before the preimage is captured, SQLite settles committed WAL through its own
checkpoint and hot-journal recovery. A busy checkpoint fails the operation;
journals are never deleted to force it through.

For a large database, use an explicit caller deadline, for example
`--wait --timeout 3600`. Copy has a 600-second engine query deadline; exact
validation has a separate 3600-second query deadline. Maintenance uses one
worker, up to 2GB copy buffers and two 768MiB validation pools. Free-space
admission reserves three source-DuckDB sizes plus the sidecar and 2GiB for the
backup, candidate, WAL/spill and margin. These are admission bounds; each host
still needs measured capacity qualification before rollout.

During conversion, maintenance polling reads the recovery journal. Daemon
status returns its captured pre-maintenance observation marked with
`storage_maintenance.snapshot: true` and the current phase; it does not claim a
fresh database observation. Other database reads can wait for activation.
A timeout disconnects the caller without cancelling the work.

The recovery journal is versioned separately from the schema, index migration
and fixed-header versions under `snapshots/storage-codec-state/`. Both it and
`storage-codec-*` backups survive age-based snapshot cleanup. An interrupted
copy retains the original pair and its candidate for diagnosis; explicit retry
uses the original verified backup. Publication recovery identifies the old or
new complete file by hash before writes resume. An unexpected file, missing
file or later durable WAL holds for explicit recovery; a stale journal never
restores an older file over later progress.

Before any rollback, stop admission and snapshot current files and WAL. Restore
and test a backup pair separately; restoring an old preimage over later writes
would discard progress. A same-format header does not establish old-binary
compatibility. The previous release must pass its own read/write/restart
rehearsal before a downgrade can be qualified, and may reintroduce Roaring that
a later forward upgrade detects again.
