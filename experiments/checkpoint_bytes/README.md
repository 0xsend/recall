# Populated checkpoint investigation

These are measurement-only prototypes for issue #93 at product base
`4637023d35b290ac64d7f40ea37a8d87b4aaf99b`. They do not implement a product
migration. The original measurement report lives in the campaign's `ids-93`
handoff directory. The amended [plan](plan.md) records the human's runtime-policy-only decision
and retains the superseded rewrite proposal as historical evidence. Runtime
evidence lives in the campaign's `impl-93` handoff. The historical rewrite
measurements do not authorize a product conversion.

The scripts require Linux `/proc`, the repository's locked uv environment, the
existing offline BAAI/bge-small-en-v1.5 model cache, a lane-local transcript
pool and copies of the populated rollback fixture. The originals remain
read-only. The DuckDB SHA-256 is
`2849121c0f56043fed5eac2cd7ffac3f1e83b2d64b28b3bdcb0b8806241088b4`;
the SQLite SHA-256 is
`2c18dee9222e9c9fa983d650f7565110bbaa6927e3f6ae19ea40f728e5e81577`.
Prepared copies belong in `<lane>/scratch/fixture/`; the copied writer-55 pool
belongs in `<lane>/scratch/pool/{claude,codex}/`. Set `TMPDIR` to a short,
owned directory under `~/Work` so Unix socket paths fit. Keep `UV_CACHE_DIR`
inside the lane. Every output directory is exclusive: existing runs are never
overwritten. Failed development attempts remain separate from valid samples.

## Workload and variants

The documented io-59 scheduler was not retained. This reconstruction uses its
surviving 24-transcript pool four times, UUID renaming with seed 93, a 70% line
prefix, and 120 seconds containing exactly 1,100 append operations and 24,027
lines. The scheduler cycles over sessions with sufficient remaining lines;
seed 59 chooses chunk sizes, adjusted to the fixed line total. It produces a
303.84 MiB seed and 56,191,718 appended bytes. `pool.json` records the input
hashes and seed boundaries. The first baseline adds 67,076 messages and 25,876
calls, about 0.15% more than the historical io-59 record. This is a close
reconstruction, not an exact schedule replay.

Every process starts from an independent fixture copy. All five source roots
are pinned to scratch, usage harvesting reads an empty scratch log, HOME stays
unchanged, model loading is offline, compaction is disabled, context is
template, and embedding idle/interval are 30/10 seconds. Readiness verifies
the owned PID through RPC and the PID file. Settle and drain require a complete
scan, pending zero, enrichment ready and every seeded session current at its
actual file size. Final counts must match the first baseline. A measurement
is complete only after final status/stats and owned-process cleanup succeed.

| Variant | Scope |
| --- | --- |
| `baseline` | Unchanged product and fixture, ordinary schema handling by the base build. |
| `u128` | Seven PKs/eight message/tool ID columns converted to UHUGEINT; all constraints retained. A scratch-only Python/SQLite decimal-string representation and explicit numeric SQL bindings let the real daemon exercise those keys. Every UHUGEINT fetch result is normalized at a common boundary; other numeric fields retain their types. This representation is not the proposed external ID contract. |
| `drop-results` | Only `tool_results` loses its PK. The sole writer uses bounded first-row-per-ID batches and an anti-join within the caller's transaction, replacing `OR IGNORE`. Other companion constraints stay. See `constraint-audit.md`. |
| `watermark-16` | Non-schema prototype: 16 MiB application WAL high-water, with durable commit semantics, 256 MiB engine autocheckpoint and checkpoint-on-close unchanged. |
| `thread_probe.py` | Additional non-schema prototype: one DuckDB worker for all database operations, using the same baseline fixture and harness. Each opened connection records its actual thread setting. |
| `compression_probe.py` | Additional non-schema prototype: `force_compression='uncompressed'` for newly written column segments, retaining four workers and existing segments. Actual settings are recorded per connection. |
| `no_roaring_probe.py` | Narrower non-schema prototype: disable new Roaring encodings while automatic compression and four workers remain. Existing Roaring segments remain readable. |

The converter recreates tables and indexes, so physical layout differs from
the original fixture. Its counts and per-column aggregate signatures through
reopen are useful prototype checks; they are not exact migration validation.
The numeric source export is generated inside each run's scratch directory.
The checkout's product source files remain unchanged. Its source transformation
normalizes fetch return types and gives internal ID sets and cursor fields
explicit SQL types; native harness and catalog keys retain VARCHAR bindings.

## Running and interpreting results

Run from the repository, using the reference runner for every controlled
measurement. Rehearsals run without the shared benchmark lock:

```bash
uv sync --locked
uv run python experiments/checkpoint_bytes/rehearse.py --prefix rehearsal-new
bash experiments/checkpoint_bytes/with_lock.sh baseline-1 \
  uv run --no-sync python experiments/checkpoint_bytes/measure.py --run baseline-1
bash experiments/checkpoint_bytes/with_lock.sh convert-u128 \
  uv run --no-sync python experiments/checkpoint_bytes/variants.py u128
bash experiments/checkpoint_bytes/with_lock.sh u128-1 \
  uv run --no-sync python experiments/checkpoint_bytes/measure.py \
  --run u128-1 --variant u128
uv run --no-sync python experiments/checkpoint_bytes/batch.py --after u128-1
```

The batch prepares the result-table fixture, completes block 1, then runs two
more blocks of baseline/numeric/result-table/watermark processes, releasing the
lock after every variant. It stops on the first non-complete result and retains
step exit codes. A failed prerequisite also stops the batch. The report names
the actual successful run labels when development attempts required new labels.

`outcome: passed` means the measurement completed. The separate
`five_second_floor_passed` field reports performance acceptance; a reproduced
baseline breach is a successful measurement with a failed performance floor.
`checkpoints.jsonl` records every attempted threshold checkpoint's duration,
WAL before/after, exact surrounding `/proc/<pid>/io` counters and host load.
Physical `write_bytes`, logical `wchar` and WAL growth stay separate. The small
observation wrapper calls the original measured checkpoint function, preserving
the daemon's `checkpoint` telemetry. Raw diagnostics, samples, pool manifest,
status/stats, errors, PID ownership and final predicates stay under
`scratch/evidence/<run>/`. Reports distinguish those retained local inputs
from committed aggregate evidence, which contains no transcript text.

Only recorded, identity-checked owned process groups are signalled. No scratch
command invokes daemon stop/restart/install/uninstall or the stable recall
binary. The numeric prototype and constraint-removal probes are not rollback,
crash-consistency, full search-contract, macOS, MLX or CUDA qualification. The
product migration has human plan approval; specialist review and
real-environment rehearsal remain required before it can land.

## Mechanism controls and native attribution

`key_probe.py --prepare` builds matched key-only VARCHAR/UHUGEINT files from
the seven populated PKs. Run `--arm varchar|u128 --repeat 1|2|3` in six fresh
processes through `with_lock.sh`. Each inserts the same 1,000 new MD5-derived
keys per table and forces one checkpoint with high-water 1 byte. This is not a
daemon threshold workload: it isolates key serialization from payload/table
compaction. Its counts, checkpoint telemetry, physical/logical bytes and host
load are retained separately. `evidence/key-summary.json` preserves all runs.

`native_profile.py` runs py-spy 0.4.2 at 20 Hz over a harness and its scratch
daemon descendants. The profiler launches its Python target, using Linux's
normal ancestor tracing permission; no ptrace/perf system settings change.
A process-local subreaper owns orphaned descendants, and the wrapper stops
and reaps them before releasing the reference lock. `profile_worker.py`
preserves the harness's real exit code independently of the profiler's exit.
Its early-failure/deadline/TERM/normal launch shape is rehearsed with:

```bash
nice -n 10 uv run python experiments/checkpoint_bytes/rehearse.py \
  --prefix native-reh-new --harness experiments/checkpoint_bytes/native_profile.py
bash experiments/checkpoint_bytes/with_lock.sh native-profile \
  uv run --no-sync python experiments/checkpoint_bytes/native_profile.py --run native-profile
```

Profiled timings are explicitly unqualified and excluded from comparison
groups. Raw native stacks and profiler warnings are diagnostic evidence;
sampler lag limits quantitative interpretation. Normal benchmark runs remain
unprofiled. `summarize.py` consumes the product's own diagnostic model,
corroborates observed checkpoint counts/sums/maxima against its stage telemetry,
and emits transcript-free JSON and a CSV containing every checkpoint. It fails
incomplete until all requested fresh-process runs have completed.

The first native diagnostic was deliberately cancelled while host load deferred
embedding. Its partial trace is retained as diagnostic evidence, with harness
exit 143, profiler exit 0 and cleanup exit 0. A scratch file edit attempted to
raise its load threshold, but the running daemon reloads only embedding
context; the edit had no runtime effect. Normal comparison configs are
unchanged. Of 146 samples in the threshold caller, 95 ended in mutex
lock/unlock. Native DuckDB workers without Python state are outside this
profiler's coverage. Calling-thread CPU close to elapsed time therefore does
not identify the worker's serialization or I/O cost.

The version-matched upstream sources show the Python
[pending-query task loop](https://github.com/duckdb/duckdb-python/blob/v1.5.5/src/duckdb_py/pyconnection.cpp#L537)
and the checkpoint
[task executor's yield path](https://github.com/duckdb/duckdb/blob/v1.5.5/src/parallel/task_executor.cpp#L36).
This motivated the one-worker experiment; the source and partial trace do not
establish a latency improvement. Run three fresh unprofiled processes through
the same reference runner, for example:

```bash
bash experiments/checkpoint_bytes/with_lock.sh threads-1 \
  uv run --no-sync python experiments/checkpoint_bytes/thread_probe.py --run threads-1
```

All runs are retained, including stressed samples and incomplete diagnostics.
The report gives checkpoint load and whole-run load separately. Comparisons
between unlike host-load windows are descriptive, not causal speedup claims.
After the initial block exposed load deferral, the follow-up comparison cohort
was defined as sampled one-minute load at or below 12 throughout a process on
this 32-logical-CPU host. This rule is independent of checkpoint duration.
Additional fresh processes attempt to fill arms lacking three such runs;
stressed samples remain in the complete results and their ceiling breaches
still count. Repeated uncontrolled load spikes made that matching effort
nonconvergent. Unstarted optional follow-ups were cancelled, preserving
the three-process minimum through completed or retained required cases.
`evidence/cancelled-queues.json` identifies them. No load criterion or floor
was relaxed; the summary remains incomplete wherever three eligible processes
are unavailable.

The additional `baseline-4` matching trial was stopped after the load limit
failed again. Its partial checkpoints and clean owned-child teardown are
retained as censored evidence; it contributes no completed-run floor verdict.
Three earlier complete baseline processes remain. Completing the missing
cohorts requires a reserved host window; no qualification is inferred from a
planned or incomplete attempt.

The retained comparison has 26 complete processes and 175 threshold checkpoints,
plus three checkpoints from two censored attempts. Every arm has at least three
complete processes. `evidence/summary.json` remains `incomplete` because matched
load coverage is missing; the summary command returns 1 for that reason. The
rebuilt-fixture arm has three eligible processes and a worst checkpoint of
2.174 seconds. The later human implementation approval and rollback amendment
are recorded in [plan.md](plan.md); they do not change these measurement
outcomes or close their qualification gaps.

The compression experiment uses DuckDB's documented
[force-compression setting](https://duckdb.org/docs/stable/configuration/pragmas#force-compression)
to separate column encoding from the key-layout candidates. It does not convert
untouched existing segments. Final file growth and persisted codec metadata
must accompany its timing result. Run it with `compression_probe.py --run
uncompressed-1` through the same reference runner. These settings are diagnostic
prototypes; no product default changes in this lane.

After timed runs have exited, `storage_audit.py --runs baseline-1 uncompressed-1`
records persisted storage versions, file sizes and last-row-group codecs from
read-only connections. It excludes min/max column statistics, which could
contain transcript text. Run that audit under the reference lock as well.

`payload_compare.py --candidate uncompressed-1` streams the completed pool rows
against baseline 1 using natural source/position keys. It compares message
content/context, tool payloads, result/mapping facts, NULL processing markers
and finite vector values packed as exact float32 bytes. It emits counts only.
Derived hash IDs/agent aliases and unrelated history are outside that comparison;
it does not qualify a product migration. Run it under the reference lock after
both daemon processes exit.

The one-worker native harness completed, but py-spy returned 1 after writing its
trace with `No child process`. Startup lag was 1.12 seconds with ten sampler
errors. The trace is exploratory: 540/561 threshold samples pass through column
checkpointing, and 412 include Roaring decoding. The pinned upstream
[scan constructor](https://github.com/duckdb/duckdb/blob/d8cdaa33fda8df955cc76ef58a280f68f4cd43fa/src/storage/compression/roaring/scan.cpp#L176)
deserializes and walks container metadata at initialization. These observations
motivate a narrower policy; they do not establish clean profiler success or an
algorithmic complexity claim.

`no_roaring_probe.py --run no-roaring-1` uses the documented
[disabled compression methods setting](https://duckdb.org/docs/stable/configuration/pragmas#disabled-compression-methods)
to exclude Roaring from new encoding choices. Automatic compression, four
workers, old-segment decoding and logical constraints remain. Three fresh
processes, persisted codec/size audit and exact pool comparison determine
whether it is preferable to forcing every new segment uncompressed.

Changing new encoding choices leaves untouched historical Roaring masks in
place. `rebuild_no_roaring.py` prepares a separate scratch file with
[`COPY FROM DATABASE`](https://duckdb.org/docs/stable/sql/statements/copy#copy-from-database--to),
automatic compression except Roaring, the fixed v1.2.0 target and unchanged
logical schema/constraints. SQLite is copied unchanged. It compares every
table's count and per-column aggregate signatures, closes/reopens, records
persisted codecs, and records kernel peak RSS. These are prototype checks;
exact migration verification, interruption/recovery and activation are not
qualified. The source copy is attached read-only; no candidate is activated.

Only after that preparation passes, `rebuilt_probe.py --run rebuilt-1` selects
the prepared pair for the same daemon harness and no-Roaring runtime policy.
Its small input-copy adapter redirects only the two known fixture-copy calls;
the measured harness remains frozen. `prepared-fixture.json` records the actual
prepared hash. The normal result's original fixture hash remains provenance
for the source of that physical copy, as with the numeric conversion.

`mask_probe.py --arm auto|no-roaring --repeat 1|2|3` is a private-corpus-free
mechanism control: 110,000 rows, three FLOAT[384] columns with repeated NULL
validity runs, one initial checkpoint, a 128-row append, and a second checkpoint.
One worker and 2 GB buffers isolate CPU from query-task polling. Its engine
autocheckpoint is 10 GB and forced application high-water is one byte, so these
are not normal daemon threshold measurements. Codec inspection confirms the
intended representation. The first automatic-compression append checkpoint
took 10.412 seconds for 9.76 MiB of writes; repetitions and the excluded-Roaring
arm provide the paired control. No transcript text or model inference is used.

`mask_transition.py` seeds with automatic compression and disables Roaring only
before the append. Its historical rewrite takes 11.484 seconds for 24.26 MiB,
even though the resulting validity segments are Uncompressed. This distinguishes
excluding future encodings from removing the existing decode path. It is a
forced component control, not a daemon threshold timing. `audit_completed.py`
finishes missing read-only storage and exact pool checks within a 600-second
budget, preserving already completed outputs.
