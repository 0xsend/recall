# Checkpoint runtime policy — approved scope amendment

On 2026-10-23T04:0xZ the human selected **Runtime policy only** through the
coordinator question widget (impl-93 addendum 3). This supersedes the prior
historical-rewrite approval and rollback amendment. The historical re-encode,
codec maintenance entry point, journal and recovery machinery are removed by
new commits above reviewed head `aeed39d`; history is retained.

The selected implementation is automatic compression excluding Roaring on every
owned connection. Existing segments stay readable and writable without a codec
admission fence. Default and explicit compaction retain their existing lifecycle
and format target, and must close/reopen and inspect a replacement before
publication. Existing fixed-header maintenance remains available; all RPC DB
writers share its admission gate and failed maintenance status is current with
an explicit error. The removed codec space-admission check has no remaining
caller or filesystem writes. No codec backup, candidate or spill is created.
Existing compaction and fixed-header maintenance retain their own error paths;
this amendment does not claim a new capacity-admission guarantee for them.

Acceptance is focused regressions (red at reviewed head for changed behavior),
a populated-copy author run, fresh guest, three fresh runtime-only threshold
processes under the host bench lock with all load/samples retained, full gates
and both explicit hooks. Specialist re-review is required before merge. No data
migration, schema/key change, production operation, history rewrite or merge is
authorized to this lane. Matched aem5 evidence remains deferred to the human's
next away window. The runtime policy alone does not establish all #93 floors:
the PR uses Refs #93.

## Retained investigation (historical, not implementation authority)

The remainder describes the superseded proposal and its original evidence.
It does not authorize implementing or operating a codec conversion.

## Decision and evidence

Prefer automatic DuckDB compression with Roaring excluded, retaining VARCHAR
IDs, every key/constraint, four workers, the selected base's schema 32, the fixed
v1.2.0 storage target, durable commits, the 64 MiB application WAL high-water, the 256 MiB engine
autocheckpoint and checkpoint-on-close. Re-encode historical column segments
through an explicit, backed-up maintenance operation so future rewrites do not
have to decode old Roaring validity masks. This is a HIGH-RISK data-file
migration even though the logical schema and public ID representation stay the
same. A runtime setting alone leaves old segments in place.

The measurement report and committed `evidence/summary.json` and
`evidence/checkpoints.csv` are the numerical record, including every checkpoint,
complete and censored attempts, host load and qualification gaps. This plan
must be read with that record; unlike load windows do not establish speedup
ratios. No candidate is release-qualified by this measurement-only lane.

Matching across all comparison arms remains incomplete after repeated global
load spikes above 30. The three converted-fixture processes do meet the
declared load rule. The report retains stressed and censored attempts; it does
not relax the rule or turn incomplete trials into floor passes. A reserved host
window is required to finish the missing matched comparisons before treating
the overall performance qualification as complete.

The initial explanations were incomplete. Compact keys reduce matched key-only
checkpoint bytes by about 35%, but full-daemon compact-key checkpoints still
exceed seven seconds. Removing the result-table PK and lowering the WAL
watermark also breach five seconds; the lower watermark substantially increases
total writes. One worker removes the caller's multiworker polling ambiguity
without removing the long checkpoint.

The one-worker native diagnostic places 540 of 561 threshold-caller samples in
column checkpointing, including 412 with Roaring decoding. Its sampler ended
with a post-write error, so it is exploratory evidence, not a clean profiler
pass. The pinned DuckDB 1.5.5
[scan implementation](https://github.com/duckdb/duckdb/blob/d8cdaa33fda8df955cc76ef58a280f68f4cd43fa/src/storage/compression/roaring/scan.cpp#L176)
deserializes and walks container metadata during scan initialization. A
transcript-free synthetic control reproduces a 10.412-second checkpoint with
only 10,231,808 physical bytes written. This demonstrates a CPU-heavy path
independent of gigabyte index rewrites; it does not prove an asymptotic
complexity claim.

The first narrow runtime-only daemon process has five checkpoints, all at most
4.049 seconds. Three fresh daemon processes from the full scratch physical copy
have 15 checkpoints: median 1.610 seconds, worst 2.174 seconds; the median of
the three process maxima is 1.996 seconds. Median/worst interval writes are
971,829,248 / 1,036,787,712 bytes, and median total threshold writes per process
are 3,796,938,752 bytes. Every process completes with matching final counts,
zero generation errors and clean daemon/cleanup exits. Whole-run maximum
one-minute loads are 10.200 / 10.405 / 9.444, satisfying the declared <=12
cohort rule. The largest kernel-recorded owned-child peak is 2,624,507,904 bytes.
The broader uncompressed policy is a
fallback diagnostic, with a first worst checkpoint of 3.045 seconds and 4.33 GiB
of threshold writes; excluding one codec avoids forcing every new column
segment uncompressed.

The scratch physical preparation took 78.778 seconds and peaked at
2,419,474,432 bytes. Its candidate DuckDB file is 9,516,625,920 bytes versus
11,041,779,712 bytes for the source; SQLite remains a byte-identical
1,655,705,600-byte copy. The smaller file includes copy compaction/free-page
reclamation and cannot be attributed to the codec policy alone. Every table's
count/per-column aggregate signatures agree before and after reopen, and no
persisted Roaring segment remains. Aggregate signatures are prototype evidence,
not exact migration qualification.

The physical copy also changes layout. There is no matched fully rebuilt
automatic-codec control, so its advantage over the runtime-only policy cannot
be attributed entirely to the codec. The synthetic controls isolate validity
encoding separately. The actual proposed combination, including the copy, is
what the three daemon processes measure.

The legacy transition control settles whether changing future encodings alone
is sufficient. An automatic-codec seed persists three Roaring validity segments;
after disabling Roaring, its next append checkpoint still takes 11.484 seconds
and writes 25,436,160 bytes. The completed output has Uncompressed validity and
no Roaring, proving the policy took effect while the historical decode cost
remained. A fresh no-Roaring control's append checkpoint takes 0.446 seconds at
the same write volume. These are forced component checkpoints, not normal
daemon qualification, but they expose a concrete >5-second historical-rewrite
case that a runtime-only rollout would retain. The explicit conversion handles
that work inside declared maintenance; ordinary first checkpoints are not
relabeled as maintenance.

## Product changes after approval

1. Have a DuckDB/storage specialist review this plan, the pinned 1.5.5 codec
   semantics, complete measurements and the new maintenance state transitions.
   Obtain a contract review for maintenance CLI/RPC changes. Preserve the
   existing ordinary-open guarantees in SPEC REQ-MIG-001 and the explicit
   storage-maintenance guarantees in REQ-RECON-018. Any needed extension to those
   contracts must be explicit and approved; changing a setting is not a waiver
   for an eager historical rebuild.
2. Apply `disabled_compression_methods='roaring'` in
   `db/connection.py::_apply_runtime_pragmas` for all owned write connections.
   Retain automatic compression otherwise. Verify the actual settings on every
   connection class and the resulting persisted codecs. Unknown/unsupported
   engine settings fail clearly before writes; do not silently select a new
   policy. The documented
   [setting](https://duckdb.org/docs/stable/configuration/pragmas#disabled-compression-methods)
   excludes new encoding choices; it does not make old files unreadable.
3. Extend the existing storage-maintenance inspection, typed plan/digest,
   eligibility, operation identity and recovery journal to include a separately
   versioned codec profile. Header storage version, logical schema version,
   index-maintenance version and codec profile are different facts. Today
   `services/index_migration.py::plan_storage_migration` identifies operations by
   path/storage target, and `perform_storage_transition` attaches with a storage
   target. At the current header that operation is a no-op; it does not
   re-encode historical segments. The proposal must add real physical-copy
   maintenance, not claim the existing command already does it. Do not bump the
   logical schema or index-migration version to trigger an unrelated rebuild.
   Persist the separately versioned codec intent, source/candidate identities,
   sidecar hash and publication phase in typed recovery artifacts under the
   existing backup/recovery owner. The current SQL job has no codec-profile
   field. This proposal retains the selected base's schema 32 and does not
   overload `storage_target` with a codec name or introduce a parallel owner. Its inspect/execute contract
   and recovery-artifact version need specialist review before landing.
4. Ordinary opens may apply the safe runtime policy and report maintenance
   eligibility, but must not start the large copy. Fresh files are created with
   the policy. A legacy file with persisted Roaring opens for inspection/status
   through the existing maintenance owner; normal write/ingestion admission
   remains fenced until explicit conversion and reopen verification complete.
   The existing structured runtime-refusal channel must explain the typed
   `codec_maintenance_required` reason and the inspect/execute action. This is
   a public write-admission contract change requiring human approval and contract
   review, not permission to claim the runtime-only policy meets the floor.
   The old-mask transition control is the reason for that boundary. An explicit planned
   maintenance operation stops admission, freezes producers and drains in-flight
   database work, takes the sole ownership lock, checkpoints/closes both stores
   and creates a verified, consistent
   preimage backup. Inspect persisted codecs for eligibility; a marker alone
   cannot prove the profile, especially after an old binary has written again.
   Do not enable producers based only on an applied marker: the current file's
   profile and all write connections' policy establish the admission invariant.
   Pending enrichment is preserved in the snapshot; maintenance does not wait
   for historical catch-up or authorize a reparse.
   This profile operation requires the current fixed header and no unfinished
   format/index job. An existing interrupted job resumes with its original
   matching backup first; afterward the operator inspects a new profile plan.
   A prior job's backup is not reused as this operation's preimage.
5. Build a separate candidate DuckDB with the same fixed storage target and
   no-Roaring policy, attaching only the quiesced source read-only. Use the
   engine's documented
   [database-copy operation](https://duckdb.org/docs/stable/sql/statements/copy#copy-from-database--to)
   to retain catalog/data/indexes/constraints. Keep the authoritative SQLite
   sidecar in place, quiesced and unchanged; its consistent backup supplies the
   validation preimage. The scratch prototype copied SQLite for independent
   measurement pairs, but the product operation needs no sidecar replacement.
   Build the candidate with the active file's basename in an isolated operation
   directory on the active file's filesystem, matching the prototype's catalog
   naming. Bound resources and
   reject insufficient space before copying. The current fixture's
   copy/check/reopen rehearsal uses one worker,
   2 GB engine buffers and a 600-second copy/aggregate-check budget. That is not
   a measured budget for full exact validation. Rehearsal must establish finite
   exact-validation and source-size/host-based budgets before rollout approval.
   A timeout leaves the
   original pair authoritative and the failed candidate available for diagnosis.
6. Validate the complete candidate: exact streamed rows by logical keys,
   float32 bits (including signed zero and any NaN payloads) and NULL/processed
   markers; every stored context and payload;
   native tool-use mappings, first-result/replay semantics, SQLite state and
   pending repair IDs; constraints, indexes, macros, versions and persisted
   codec profile. Counts/hashes are supporting evidence only. Validation never
   regenerates embeddings/context, normalizes payloads, repairs duplicates or
   changes IDs. Any discrepancy stops activation. Checkpoint, close and reopen
   the candidate using the actual release build, then repeat the invariant and
   durability checks.
7. Activate only under the original ownership lock through the existing
   recovery owner. Both stores remain closed to writers; the source/candidate
   DuckDB WALs and SQLite journals must be settled. Verify the authoritative
   sidecar still matches its captured hash. Fsync the complete candidate, its
   operation directory and the typed publication intent, then atomically replace
   the active DuckDB path with the candidate on the same filesystem and fsync
   both affected parent directories.
   Keep the original verified backup; do not rename the active file away first
   or replace SQLite. The unchanged logical rows/IDs and unchanged sidecar
   make both complete DuckDB versions valid companions by construction.
   Recovery identifies the old or new complete file against the journal before
   resuming writes; any other state holds for explicit recovery. Reopen,
   inspect and prove the profile satisfied, then complete the journal and
   resume ingestion. A failed reopen must not initialize an empty replacement.
   Atomic publication/recovery was not prototyped in this lane and requires
   the crash rehearsal below on each supported filesystem.
   If current files/WAL show later durable progress, recovery must preserve it;
   a stale journal never authorizes replacing it with a preimage or candidate.

## Invariants and rollback

All primary/unique, NOT NULL and CHECK constraints remain enforced. Random
32-character lowercase hexadecimal IDs, their hash inputs, cursors, public
search results and SQLite keys remain unchanged. Vector values, cached/stored
context, all-NULL processed markers and tool-result ordering are preserved
exactly. The migration does not re-enrich data or reinterpret model revisions.
Commit/fsync/WAL recovery remain active throughout normal operation. A normal
checkpoint or embedding backlog is never reclassified as maintenance to evade
its five-second ceiling. Only the explicitly entered, journaled migration
phase uses the existing maintenance exception under REQ-RECON-016.

Before activation, rollback leaves the original pair authoritative and removes
only the attributed failed candidate after evidence collection. Rehearse
restoring the backup pair rather than merely checking that backup files exist.
After activation, quiesce and snapshot current files/WAL before any rollback.
The same logical/header versions do not prove binary compatibility. Rehearse
the actual schema-32 base build at `35bcb24` against both a separately restored
pre-conversion backup pair and the converted files, including reads, search,
a durable append, restart and subsequent forward reopening. Record exact
binaries/engines and persisted codecs. A read-only DuckDB open or a pool
payload comparison alone does not qualify this binary rollback. This is the
human-approved 0.39.2 rollback gate; the v0.39.1 release binary needs the
separate backup taken before the 0.39.2 rollout.

If the old binary can use current files safely, keep all post-activation data;
its old codec policy may reintroduce Roaring, so a later forward upgrade must
inspect actual segments and offer conversion again. If the downgrade fails,
stop rollout and design a preservation-safe reverse export or forward repair.
Restoring the older backup alone would discard later context/enrichment and is
not an acceptable general rollback. Transcript replay is not an equivalent
preimage. No compatibility or rollback guarantee is inferred from this lane's
aggregate checks.

## Rehearsal, capacity and release gates

At the final merging head, drive the real maintenance CLI/RPC plan, stale-plan
rejection, drain, backup, conversion, validation, activation and reopen on the
11 GB fixture plus fresh per-host backups. Record environment, exact head,
phase times, whole-process kernel peak RSS and disk/temporary/WAL use. Inject
termination before/during backup, copy, checkpoint, validation, publication
intent fsync, atomic DuckDB replacement, each parent-directory fsync and first reopen.
Exercise disk exhaustion, an
incompatible candidate, corrupt/incomplete backup and an ownership conflict.
Each must preserve a recoverable complete authoritative pair and name the
remaining recovery action. Repeated operation IDs must be idempotent, while
changed source/profile plans must be rejected and replanned.

Drive ordinary append, duplicate/replay chunks, late results, transcript
rewrite/move, concurrent reads, context/vector preservation and invalidation,
keyword/vector search, pagination, restart and checkpoint contention. Inject a
dirty-WAL shutdown and verify exact recovery. Re-run the populated threshold
workload in at least three fresh processes per arm in a reserved matched-load
window. Every observed normal checkpoint must be <=5 seconds; no outlier is
removed for poor timing. Qualify status/live <=2 seconds, append freshness
p95 <=15/max <=30 seconds and whole-process RSS <=4 GiB. This lane's checkpoint
measurements do not establish the other latency floors. Linux/ONNX evidence
does not qualify macOS, MLX or CUDA.

Measure long-term file/working-set growth, reads/search and repeated historical
row-group rewrites under the narrow policy. Space checks cover the consistent
preimage pair, authoritative pair, candidate DuckDB, WAL, spill and measured
margin at once. The current partly rewritten runtime-only file ratio cannot
predict a fully converted steady state. Approve explicit headroom and finite
copy/validation budgets from the rehearsal rather than guessing from one file.

Human implementation approval and the rollback amendment are recorded above.
Landing requires a storage specialist verdict with no open blocking finding,
the contract decisions, and the author's risk rehearsal at the merging head.
A guest is required for the changed product maintenance flow. Rollout artifacts
and target refs need their own publish grant; this plan does not claim
production authorization for the implementation lane.
