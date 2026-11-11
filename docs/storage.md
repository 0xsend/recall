# Storage

recall stores its index in DuckDB at `~/.local/share/recall/recall.duckdb`. When
the SQLite FTS sidecar backend is enabled, its keyword index lives beside it in
`recall.fts.sqlite`. The daemon owns both files.

## Matched rollback after index recovery

Schema 41 adds the durable index-recovery singleton and fences ordinary old
opens. DuckDB remains 1.5.5 with storage compatibility v1.2.0. An old binary alone
is not a rollback artifact. Before upgrade, qualify a consistent pre-upgrade
DuckDB/WAL set, a consistent SQLite sidecar, configuration and source roots,
runtime artifacts, the presence or absence of failure/refusal/spool files,
checksums, and the exact old installation and extras.

For an authorized rollback, stop the exact owner and scheduler. Preserve the
failed upgraded database, WAL, sidecar, config, event spool, refusal, pinned
oracles and acknowledged post-backup work in a separate recovery set. Restore
the matched pre-upgrade set and old installation together into an isolated
validation environment. Verify actual process/build/schema/engine/data paths.
Never lower the live schema version or delete the recovery singleton.

Every acknowledged write after that backup is absent from the restored active
set. Inventory the interval before declaring it authoritative. Reconcile current
supported transcripts under the normal lossless rules; compare raw content,
recoverable enrichment, keyword/vector retrieval and pending retirement exactly.
Usage events/cursors, user live marks, derived inputs, acknowledgements and other
non-transcript writes require a compatible isolated forced-scan export and a
reviewed lossless import/replay procedure. Record every item and exact comparison
in a recovery ledger. A general lossless importer for all such data is not
provided. Any unrecovered or unverified item leaves rollback non-authoritative
and blocked unless the human separately authorizes the named losses. Keep
ordinary writers stopped until the ledger is settled; isolated validation is
not authoritative service. Preserve originals through verification.

A full failure-event queue requires capacity correction followed by this
qualified matched restore. Deleting events or clearing the recovery row is not
the remedy. Restoring a database file alone must not erase a spent attempt or
silently retain mismatched keyword/recovery files.

The released 0.46.1 `recall index --recreate --yes` must never run against an
upgraded original: its confirmed destructive reset bypasses the old ordinary
schema check and destroys recovery state. A reset on a disposable rehearsal
copy demonstrates that exception; it does not authorize resetting an original.
Any later destructive-reset grant must separately name the old installation,
database and associated files, accept the acknowledged losses and possible old
auto-fork, and preserve the complete recovery set outside reset paths. Its own
backup does not qualify as a matched rollback set. An intentionally empty old
database uses a separate empty data directory with no upgraded markers.

Recovery capture, rebuild, copy and reconciliation may take longer than a binary
rollback and do not promise zero downtime. Populated readiness, read latency,
memory, freshness and idle floors remain release gates. Process-kill and injected
I/O checks do not establish whole-device power-loss durability.

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

## Migration backup recovery

Inspection, automatic interrupted-job resume, explicit maintenance and rollback
require the original complete backup inside the selected data directory. A
foreign or unqualified reference is refused before backup reads or migration/
restore effects, even when a local backup has the same basename. Startup records
the failure without completing captured work or advancing its version; status,
stats and indexed reads remain usable. Restarting does not repair the reference.

The selected data directory resolves to its actual existing directory, including
links in that selected root. Relative backup references start at that canonical
root independently of cwd. Absolute references must normalize inside it;
absolute paths through a noncanonical root alias are conservatively unqualified.
New migration backups record canonical absolute references, including through a
relative or linked selected root. Existing records are never rewritten.
No stored path is resolved to infer equivalence or relocated by suffix/basename.
Relative `..` cannot cross above the root even if later components spell a return.
All descendant symlinks are unqualified, including internal, dangling and cyclic
links. Dot components are normalized lexically before traversal. NUL/unencodable
paths and records exceeding 4,096 encoded bytes or 256 slash-separated components
are unqualified; the nonempty string still appears exactly in JSON.

A present directory can still be incomplete. Recovery requires a complete
migration manifest of at most 1 MiB listing regular `recall.duckdb` and optional
`recall.fts.sqlite` inputs. Linked/missing inputs refuse recovery. File reads use
pinned descriptors; replacing a checked pathname cannot redirect restore to a
link target. An already-open directory remains the observation if renamed;
this is copy isolation under a trusted local-account model, not a defense
against hostile local writers moving directories.

For a refused copied job, select the original installation or obtain a separate
recovery authorization to copy the complete matching backup into the selected
installation and deliberately repair its stored reference. Preserve the original
backup and captured scope. There is no automatic import, repair, override flag or
cross-root exception. Presence null with a non-null path means uninspected;
[the status table](reconciliation-operations.md#migration-backup-status) explains
all four JSON pairs. Invalid selected roots and unexpected I/O errors report an
error rather than claiming the backup is missing.

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

Same-schema migration rollback is also an offline, non-authoritative operation. Its caller names a unique failed-set archive outside the data directory. The archive preserves quiesced DB/WAL, SQLite transaction files, config, all selected data files and pinned inputs, with checksums and directory synchronization, before replacement. The active refusal marker is synchronized before copying and remains afterward. Interrupted copying can leave an unreadable active file; the retained failed set and original backup remain the recovery sources, and ordinary startup refuses before a writer open. This helper cannot install a pre-fence logical schema or perform the old-install rollback described above.

Neither a file-copy return nor an index rebuild settles post-backup acknowledged work. The operator completes and verifies the per-item ledger before requesting explicit release. No general lossless importer is provided. The archive operation holds above 32,768 entries, 16,384 files, 16 TiB, available disk capacity, unsupported links or nonregular files; it never proceeds with a partial archive. A separately reviewed matched-restore procedure is required for an unsupported set.
