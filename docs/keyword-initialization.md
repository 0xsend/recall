# Explicit keyword initialization

Unchanged effective message and bash documents retain their keyword statistics
and scores, including after a restart or durable pending replay. Genuine content
changes retain SQLite replacement semantics. Initialization corrects accumulated
historical statistics by constructing an index from canonical DuckDB content.

Fresh sidecars use schema 2 and `sha256-length-prefixed-utf8-v1` fingerprints.
Opening a legacy sidecar for publication refuses with an initialization message;
it never backfills fingerprints or rewrites unknown rows. Keyword reads of the
legacy file remain available. Operators must schedule initialization before
resuming keyword publication with this binary.

The implementation and fixture/copy rehearsals were authorized by the human on
2026-10-23 (PR #141 revision 2, coordinator addendum 1). Execution on a real host
requires a separate approval naming the host, data directory, backup, candidate
build, maintenance window, rollback retention and gaps.

## Operator flow

Stop the owning daemon and all other writers using the deployment's approved
procedure. This operation never stops services. Retain its PID/exit evidence.
Every invocation requires explicit data, database, lock, operation and build
identities, plus the configured keyword field scope. The operation directory's
parent must exist on the sidecar filesystem. A distinct operation directory is
required for another snapshot, field scope or build.

Use the candidate environment's Python. For example, with approved **scratch**
targets (replace all paths and the build identity together):

```sh
/path/to/candidate/.venv/bin/python -m recall.services.fts_sidecar_rebuild inspect \
  --data-dir /scratch/data --db-path /scratch/data/recall.duckdb \
  --lock-path /scratch/data/recall.lock --operation-dir /scratch/keyword-build \
  --build-id APPROVED_COMMIT --fields content,thinking,bash
```

Keep those arguments fixed and invoke the actions in order:

1. `inspect` reports `absent` or the typed manifest without creating a lock,
   directory or database. Check the exact target, field scope and build.
2. `begin` obtains the advisory and exclusive DuckDB locks, checks free space,
   persists `keyword-maintenance.json` and captures a matched consistent backup
   under `data/snapshots/keyword-rebuild-*`. Its manifest binds the full DuckDB
   hash, byte-exact standalone original sidecar hash, fields and build. The
   format-2 manifest also records Python/SQLite/DuckDB versions, captured file
   sizes, schema versions and applied migrations, index-migration version/target/
   phase, sidecar metadata, session counts by source and pending queue count/
   SHA-256. The pending digest covers newline-delimited JSON arrays of kind,
   id, operation and timestamp in captured rowid order. Scope is the entire
   canonical snapshot, independent of configured transcript roots. Format-1
   manifests remain readable/resumable; their missing capture metadata is not
   inferred from the current database.
   durable marker blocks ordinary database opens, daemon starts and client
   auto-starts, including after the maintenance process exits or crashes.
3. `build` constructs a distinct candidate, committing progress with each bounded
   batch. Stderr reports progress; stdout reports the result as JSON. It resumes
   only the captured snapshot/fields/build, verifies sorted identities and
   effective-content fingerprints, FTS and SQLite integrity, membership, and
   live-document token totals. It never parses, contextualizes or embeds.
4. `install` accepts only the verified hash while the canonical hash still
   matches. It checkpoints the old sidecar, rejects attached journals or busy
   readers, and atomically replaces the standalone file. A crash immediately
   after replacement is recovered by comparing the installed hash with the
   manifest. The matched original remains retained separately.
5. `finish` checks the installed hash and unchanged canonical generation before
   clearing maintenance. Restart the approved binary, verify status, stats,
   keyword queries, pending progress and append behavior. Retain those results.

Each mutating action owns both exclusion locks. A lock conflict, changed source,
changed backup, changed candidate, wrong fields/build or insufficient space
fails with exit 2 and a JSON `failed` outcome. Unexpected failures also exit
nonzero; no partial attempt is reported as verified. A failure after `begin`
leaves maintenance held and artifacts retained. Inspect the error and resume the
same operation; do not delete its marker to force a start.

The states are `building`, `verified`, `installed`, `rolled_back`, `released`
and `aborted`. `begin`/`build`/`install` retries converge while held. If the
process dies after the release record but before marker deletion, retry `finish`.
After release the operation cannot roll back or reinstall a sidecar.

## Abandon or replace a held operation

If construction cannot complete, use one of these supported recovery paths;
never remove the maintenance marker by hand:

- `abort` with the original arguments releases maintenance only when the live
  sidecar still has its captured original hash. It works from a partial build
  and after the canonical hash or WAL check fails. It leaves both live files
  in place, retains every artifact and records `aborted`. Retry `abort` if a
  crash occurs after that record but before marker removal. A legacy original
  still needs initialization before publication by the candidate binary.
- `replace` captures the current pair under a fresh operation directory. Pass
  the usual explicit target/build/fields, change `--operation-dir` to the fresh
  directory and add `--previous-operation-dir /scratch/keyword-build` naming
  the held operation. The tool validates that owner and transfers the durable
  marker directly, without releasing writers. Then use `build`, `install` and
  `finish` with the new arguments, omitting `--previous-operation-dir`.
  This works after canonical drift or a prior binary's sidecar deletion; no
  older sidecar is restored. A repeated `replace` with the same two directories
  resumes the captured replacement. An interrupted capture can be retried in
  that same directory. The old operation's actions refuse after transfer.

An ordinary `begin` cannot take another operation's marker. Wrong targets,
wrong predecessor or an existing unrelated candidate fail closed. If a capture
fails before saving its manifest, retry its `begin` (or `replace`) to capture
successfully before using the recovery paths. An abort whose live sidecar no
longer matches the original refuses and names `replace` as the next action.

## Rollback

Before `finish`, invoke `rollback` with the same arguments. It requires the
captured canonical hash and an unchanged original or candidate, restores the
byte-exact retained original using atomic replacement, and leaves maintenance
held. A partial `building` operation can also roll back without completing its
candidate. If canonical validation fails, use `abort` or `replace` above;
rollback never restores a sidecar across canonical generations.
Verify its baseline queries, then `finish` and restart the prior approved
binary. The original statistics drift returns.

Older binaries can read schema 2, but mapping triggers reject their fingerprint-
less publication attempts, even their `INSERT OR IGNORE` for an existing id.
A binary rollback must restore the compatible old sidecar before publication.
After writers resume, sidecar-only rollback is prohibited. Reconstruct from the
current canonical database, or obtain approval for restoring a matched full
backup with an explicit account of subsequent work. No database, backup or
failed candidate is automatically discarded.

The operational design follows the approved [idempotency plan](fts-sidecar-idempotency-plan.md)
and [canonical reconstruction plan](fts-sidecar-repair-plan.md). Author and guest
evidence, including any unexercised rollout scenarios, belongs in the change
record; fixture success does not approve production execution.
