# Bench comparisons and pairs

These development jobs use the same outcome line and path policy as `prepare`
and `daemon-run`. Invoke `scripts/bench.py` with the tool checkout's interpreter,
through the lane entrypoint that pins `BENCH_WRITE_ROOTS`. Every path is absolute.
Inputs are disposable copies, never production recall directories or an original
rollback backup. Admission requires a separate guarded-launcher operation and is
not implemented here.

## Compare

```text
bench compare --before <prepared-template> --after <prepared-template> \
  --output <new-directory> [--memory-limit <bytes>]
```

A backup is a flat directory of regular single-link files. Compare requires
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
interrupt active queries and preserve partial output.

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
