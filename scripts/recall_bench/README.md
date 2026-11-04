# Bench admission, comparisons and pairs

These development jobs use the same outcome line and path policy as `prepare`
and `daemon-run`. Invoke `scripts/bench.py` with the tool checkout's interpreter,
through the lane entrypoint that pins `BENCH_WRITE_ROOTS`. Every path is absolute.
Inputs are disposable copies, never production recall directories or an original
rollback backup.

## Admission

```text
bench admission --template <prepared-template> --build <checkout> \
  --output <new-evidence-directory> --run-dir <new-short-parent> \
  [--negative-schema <canonical-decimal>]
```

The new run parent holds fixed `inspect`, `positive` and `negative` children;
it is not itself a data directory. All derived socket paths must fit the platform limit
before either copy is created. Output, run parent, template and checkouts must
be disjoint. A retry uses new output and run paths. Originals are never opened
by DuckDB. Every child uses the selected build's interpreter, registration gate,
config/path checks, scheduler/usage replacements, audit guard and owned-group
cleanup. `--ungated admission` is refused. No daemon or RPC server starts.

The job verifies the complete template, checks the build's seams and supported
schema in an empty guarded `inspect` environment. An incompatible negative
schema or missing build seams refuses before any template copy; the small
inspection environment and its evidence remain. The positive copy opens
read-only (with the build's bounded index probe) and through strict write admission, then makes
a separate negative copy from the original template. A supported older control
is normalized through strict write admission first. The control's newer logical
schema is written by the selected build's version writer and retained in a
nonempty WAL, with shutdown checkpointing disabled. The default negative version
is the selected supported schema plus one; an explicit value must be greater
than supported and at most 2147483647. No operation accepts paths, SQL, Python
or arbitrary callables.

Both negative attempts must raise the selected build's `SchemaTooNewError`
with the exact database and supported versions. The parent hashes DB, WAL and
optional SQLite sidecar before and after **each** refusal, after the child group
has stopped, without opening an engine connection. Missing and zero-byte files
are distinct. A wrong error, unexpected open, changed/deleted bytes or absent WAL
fails. Cancellation, child timeout or missing build seams is incomplete. No later
attempt runs after any non-passed step. Partial evidence remains inspectable.

`admission.json` retains requests, actual observations, per-refusal hashes and
safety evidence. `result.json` records the terminal outcome, step exit codes,
selected build and tool HEAD/dirty digests, template manifest, exact config,
environment class, limits and uncovered classes. Per-case `inspect/`, `positive/` and
`negative/` evidence subdirectories contain raw child output and the process
registry. A leaf emits one `recall-bench-admission/1` JSON document; diagnostics
are separate. Completed observations are `inspected`, `opened`, `prepared`,
`refused` or `error`; an observation alone does not establish job success.
Inspect these retained files after disconnection; do not infer completion from
an absent process or rerun an existing destination.

The job has a 7200-second work deadline, including copying and hashing, and a
900-second bound per admission child. At most seven admission children run.
Terminal documents are limited to 1 MiB and migration metadata to 4096 rows.
Returned connections retain the product's memory/thread settings (recorded in
each observation) with a 64 GiB spill bound applied before probe/control work.
Compatibility rejection happens inside product connection code before these
settings can be applied. These bounds do not certify whole-process RSS.

Success covers admission and per-refusal byte preservation. Same-schema opens
also retain schema/migration audit; an actual migration records before/after
without claiming unchanged audit. Probe incompleteness, divergence and
unverifiable samples remain observations, not a healthy-index verdict. Row
preservation needs the separate compare job on freshly prepared owned snapshots.
Migration completeness/runtime/resume, installed-host tasks, model/network paths
and daemon performance floors remain unqualified. This is the existing guarded
Python boundary, not an OS sandbox: live run/build names must never be repurposed.
The inherited name-stability and non-atomic Linux procfs inventory limits remain.

## Compare

```text
bench compare --before <prepared-template> --after <prepared-template> \
  --output <new-directory> [--memory-limit <bytes>]
```

A backup is a flat directory of regular single-link files. Compare and pairs require
`recall.duckdb`; its adjacent WAL is included when present. `prepare` can also
copy other backup files without interpreting them. Both comparison inputs must
verify against a manifest written by `prepare`. Missing database payloads are
refused before output creation. For a completed
run, first prepare a template from an owned flat snapshot of its database, WAL
and sidecar. Current daemon run directories contain no tool comparison manifest;
unmanifested directories are refused. Tampered payloads, aliases and existing
outputs are also refused before creating output. Output cannot be inside either
input.

The job makes verified database/WAL clones under output, then attaches those
clones read-only. It compares every DuckDB base table's ordered column names and
types, then uses `EXCEPT ALL` in both directions to compare complete row
multisets, including duplicate multiplicity and NULLs. Row ordering and physical
storage do not affect equality. Views, constraints, indexes and the SQLite FTS
sidecar are outside this equality claim.

The default engine memory limit is 1 GiB; `--memory-limit` takes bytes between
1 MiB and 4 GiB. Two engine threads and a 64 GiB spill limit bound the comparison.
Spill stays under `output/spill`. The job has a four-hour deadline. TERM/INT
interrupt active queries and preserve partial output. Remove disposable
completed or partial outputs after their evidence is no longer needed.

`comparison.json` contains `equal` and one row per table. Equal inputs exit 0
with `passed`; differences exit 1 with `failed` and error kind `different`.
A table's `before_only`/`after_only` counts include duplicate copies. A schema
mismatch has column descriptions and null row counts because rows with different
shapes were not compared. `result.json` records the outcome, input manifests,
tool identity, settings and scope; stderr carries progress. Database-open or
resource failures retain a failed result, never an equality observation.

## Pairs

```text
bench pairs --lock <shared-queue-directory> \
  --baseline <checkout> --candidate <checkout> --template <prepared-template> \
  --run-parent <existing-short-directory> --output <new-directory> --count 3 \
  -- --profile idle --window 60
```

Each numbered pair runs baseline then candidate, using a new run directory and
output per arm. Each daemon-run is a child process inheriting its locked ticket;
killing the batch owner cannot release admission while that child still runs. The maintained FIFO bench lock is held per arm and released
before the next, so other lanes can interleave. A new queue must be inside a
write root; an existing shared queue passes the path policy and alias checks.
Never repurpose the queue or remove its mutex while any runner might use it.
Admission waits at most three hours and is cancellable.

The arguments after `--` are daemon workload options. They cannot override the
build, template, run directory or output. `--hold` is refused: use a separate
`daemon-run --hold` for a guest environment. The pair count is bounded to 1000.
Runs, evidence and queue must occupy disjoint directory trees. A completed
ticket may remain on disk until the next registration reclaims it; presence
alone does not mean admission is held. Use a dedicated short run parent; actual daemon runs still enforce the Unix
socket path limit.

`pairs.json`/`result.json` record every attempted arm, its outcome and exit code.
Each arm also has a saved outcome document, even if it refused before creating
its own directory. The first non-passed arm stops the batch: failed exits 1,
incomplete exits 2, and cancellation exits 128 plus the signal. No later arm is
started. A successful batch says all arms passed their own profiles; it does
not independently certify timing parity or any missing measurement class.
