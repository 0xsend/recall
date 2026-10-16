# Reconciliation operations

Reconciliation is how the recall daemon keeps its index level with the transcript
files on disk. This guide covers the operator view: what `reconciliation.pending`
counts, how it drains, which fields answer "is it progressing", and what to do
when it stops. [SPEC.md](../SPEC.md) owns the contract (`REQ-RECON-001` through
`REQ-RECON-023`); [RECONCILIATION_QA.md](../RECONCILIATION_QA.md) owns the
verification procedures.

## What pending counts

The daemon keeps a durable catalog of every transcript file it has discovered
under the configured source roots. `reconciliation.pending` is the number of
catalogued **files** — not messages — whose committed index is behind the file
on disk, because any of the following holds:

- the desired generation is ahead of the committed generation (the file changed;
  a new parser build alone does not count),
- committed bytes are behind the file size (the file grew, or was never fully
  indexed), or
- the last attempt on that file recorded an error.

Files the daemon has catalogued but can no longer stat are excluded and counted
as `missing`. Files under a disabled root are excluded from `pending` and
counted as `out_of_scope_pending`.

`pending` is not an embedding count. Embedding work is counted in message and
tool-call rows, read fresh on every status call: `embed_pending_eligible` for
sessions idle at least `embed_idle_session`, which the embed loop may select
now, and `embed_pending_held` for sessions still inside that hold, both measured
at `embed_pending_measured_at`. `embed_pending` and `embed_pending_at` are the
embed loop's last eligible snapshot; it reads 0 while a fresh session waits out
its hold, so it cannot say embedding is done. `reconciliation.enrichment_ready`
is true when embedding is disabled, or when an embed phase is running, both
fresh counts are 0 and no embed error is set (`REQ-ADAPT-022`). With embedding
disabled the flag is true whatever the counts, because that work will not be
embedded. A harness that only needs settled history waits for
`embed_pending_eligible == 0`.

Raw and enrichment readiness are separate flags, so a complete drain is
`raw_indexing_ready` and `enrichment_ready` together. Both describe content the
daemon has observed. In watch mode an append to a live session is seen within
seconds, but a new file, or an append to an idle one, is seen only once a
live-discovery pass (`discovery_interval_seconds`) promotes it. For up to about
that long after such a write, `pending` and the embed counts can read 0 and
both flags true.

Raw indexing progresses independently of model work (`REQ-RECON-008`): a
deferred embed phase does not hold up the reconciliation drain, and an empty
reconciliation backlog does not imply embeddings are done.

## How it drains

Periodic full inventory and filesystem notifications feed the catalog. A bounded
fair scheduler selects catalogued work, preparation (reading and parsing) runs
outside the writer lock, and a single transactional writer commits content with
its checkpoint. Watch mode reacts to filesystem events; poll mode reconciles the
roots on an interval. Both share the same reconciliation path.

`pending` therefore falls as files commit and rises when discovery finds new or
rewritten files, so a backlog that grows briefly during a scan is normal. A
count that only rises across several scans is not.

One measured order of magnitude, not a guarantee: a macOS host went from 27,326
to 25,253 pending files in 21 minutes — about 100 files per minute — while the
same daemon served live watch work. The rate depends on file sizes, the source's
parser, and what else the daemon is doing, so a host's own two readings are a
better baseline than this number.

## Fields to read

All of these live under `reconciliation` in `recall daemon status`:

| Field | Meaning |
|-------|---------|
| `pending` | Catalogued files not yet committed to their current bytes. |
| `paused` | An operator pause is in effect; work is held, not lost. |
| `catalog_scan_complete` | Every configured root finished a scan. While false, totals are still moving. |
| `raw_indexing_ready` | Scan complete and `pending == 0`. |
| `keyword_search_ready` | `raw_indexing_ready` plus an empty keyword sidecar queue. |
| `enrichment_deferred` | Why the embed phase last stood down (for example host load above threshold). Does not block the drain. |
| `error` | The inventory error, when a scan itself failed. |
| `coverage[]` | Per source and root: `pending`, `oldest_pending_age` (seconds), `scan_complete`, `failure_count`, `errors`, `unsupported`, `missing`. |
| `unsupported_summary` | Parked parser gaps an agent can file: source, record detail, file count, and up to three sample paths. Kind-only rows from an older daemon say the detail was omitted. |
| `source_page` / `next_cursor` | A bounded page of per-file catalog records with diagnostics; `next_cursor` is an opaque token to pass back verbatim. |

Text mode's summary block states the headline numbers. For a scripted read, the
existing top-level projection is enough — there is no separate reconcile
command, and none is needed:

```bash
# Headline drain state, with the per-file page held to one row.
recall daemon status --json --limit 1 --fields reconciliation \
  | jq '.reconciliation | {pending, paused, catalog_scan_complete, keyword_search_ready, enrichment_deferred, error}'

# Where the backlog sits, and how old the oldest pending file is.
recall daemon status --json --limit 1 --fields reconciliation \
  | jq '.reconciliation.coverage[] | {source, pending, oldest_pending_age, scan_complete, unsupported, missing}'
```

`--fields` projects top-level keys only, so `--fields reconciliation` returns the
whole object and `jq` (or any JSON reader) selects within it. `--limit` keeps the
per-file page small; raise it with `--limit N` (1..256) and page with `--cursor`
when the per-file diagnostics are what is wanted.

## When pending is not moving

Two readings a minute apart establish whether it is stuck. If the count is flat:

1. **Check `paused`.** A durable pause survives restarts and client auto-start
   (`REQ-RECON-009`). `recall daemon resume` clears it.
2. **Check the daemon is actually serving.** `recall daemon status` reports
   `version_drift` and, when the RPC is unreachable, says so instead of
   reporting runtime fields. A drifted daemon runs old code until restarted.
3. **Check `error` and per-source `failure_count` / `errors`.** A root that
   cannot be scanned (unmounted, permissions) reports incomplete coverage rather
   than silently claiming completeness.
4. **Check `unsupported`.** Those files are parked on parser diagnostics, not
   missing: they stay pending until a parser handles them. Inspect
   `source_page` diagnostics and file a recall issue (`REQ-CLI-023`).
5. **Check the daemon log.** A daemon that crash-loops can keep a healthy-looking
   status while making no progress; `<data dir>/logs/daemon.err.log` is the place
   that shows it.
6. **Check whether the drain is merely slow.** `oldest_pending_age` climbing
   while `pending` falls slowly is a large backlog, not a stall.

## Commands that are safe

- `recall daemon status` — read-only, and the only command needed to diagnose a
  drain.
- `recall daemon pause` / `recall daemon resume` — durable operator intent; the
  pause holds work rather than discarding it.
- `recall index` — asks the running daemon to index new and changed transcripts.
  It does not re-parse history.
- `recall daemon restart` — safe, but it re-runs startup work; on a host with
  LLM context enabled a restart can spend a long stretch on context catch-up
  before embedding resumes.

Not a remedy for a backlog:

- `recall index --recreate` rebuilds the live database and is never the fix for
  pending work or a refused start.
- `recall index --full` on a host with an LLM context mode must pin
  `--context template`; see [below](#full-re-parse-on-an-llm-context-host).
- Deleting or moving the database while the daemon holds it.

## Full re-parse on an LLM-context host

On a host whose `[embedding.context] mode` is `llm-local`, `llm-remote`, or
`llm-codex`, pin the context mode for any full re-parse:

```bash
recall index --full --no-embed --context template
```

A bare `--full` re-parses correctly, but every message with no stored LLM context
is treated as pending and gets summarized. When only a small fraction of a large
corpus carries LLM context, that is millions of model calls and hundreds of hours
of local generation. Reusing stored context (`REQ-INDEX-021`) prevents losing
existing context, not the cost of rows that never had any.

Setting `RECALL_CONTEXT_MODE` in your shell does not help: `recall index` is
executed by the daemon with its own environment, and only `--context` travels in
the request. Check the request before a long run:

```bash
recall index --full --no-embed --context template --dry-run --json
# => "request": { ..., "context": "template", ... }
```

Pinning `template` forecloses nothing: the incremental path only resolves context
for freshly parsed messages, so it never revisits history anyway.
`--recompute-context --only-mode template` remains the upgrade path.

Back up `recall.duckdb` (and `recall.fts.sqlite`, if present) first, and split the
work into bounded runs; `--since` compares the transcript's mtime and `--project`
matches `git_repo` the same way `recall list --project` does. They compose:

```bash
recall index --full --no-embed --context template --since 30d
recall index --full --no-embed --context template --project ~/work/myrepo
```

## Daemon performance diagnostics

Foreground and installed daemons write `<data dir>/diagnostics.jsonl` every ten
seconds and at shutdown, independently of the configured application log level.
The collector performs no database queries. It retains the current file and three
rotated files (`.1` through `.3`); rotation occurs at 5 MiB, with at most one bounded
sample beyond that threshold. Files are created owner-readable/writable. A write
failure is reported in the daemon log and retried at the next sample.

Each JSON line follows `services.diagnostics.Sample`, with `schema_version: 1`.
The `(pid, started_at)` pair identifies a process lifetime. Counters and duration
histograms are cumulative for that lifetime; compare successive samples. Histogram
buckets are disjoint, with the listed inclusive upper boundaries followed by an
overflow bucket. `process_cpu_seconds` is total daemon user+system CPU: divide its
delta by the monotonic time delta and multiply by 100 for percent of one CPU core.
Child processes are not included. `sampling_cpu_seconds` measures the collector
thread's own work through the previous sample, and `output_failures` counts failed
samples. RSS is current resident memory, not a peak. Process I/O bytes are null on
platforms where psutil does not expose them, including macOS.

`scheduling` separates waiting for a CPU from running and blocking. On a loaded
host a writer turn can take far longer than its CPU time because its threads are
runnable but not on a core. `runnable_waiting_seconds` is that cumulative wait in
thread-seconds: on macOS for the whole process (exited threads included), on Linux
summed over the threads alive at the sample, so on Linux it can fall when threads
exit. Because several threads can wait at once, its delta can exceed the wall-clock
window. Compare its delta with the `process_cpu_seconds` delta over the same window
(both are thread-seconds); a wait at or above the CPU means the daemon spends at
least as long queueing for a core as running on one. On macOS, `priority_ceiling` is the daemon's maximum thread
scheduling priority and `qos_cpu_seconds` bills CPU per QoS class. A launchd job
without `ProcessType` runs under a utility QoS clamp: ceiling 20 and nearly all CPU
billed to `utility`. An unclamped process reports 63; its threads mostly bill to
`legacy`, the class of threads that never set a QoS. The other keys are the
remaining macOS QoS classes (`default`, `maintenance`, `background`,
`user_initiated`, `user_interactive`); a few threads set one explicitly. Both are null on Linux;
`scheduling` is null on other platforms, on Linux kernels built without
schedstats, and in samples written before the field existed.

`telemetry.stages` separates writer lock waiting, executor waiting, worker
execution, reads/search, raw preparation/parsing/commit, enrichment selection,
snapshot, context generation, tokenization, inference, validation, publication,
FTS upserts, and checkpoint attempts. Errors and cancellations retain their timing.
A successful span means the function returned; handled generation failures also
increment `generation_errors`. Counters distinguish generated and committed targets
from newly drained targets and rejected sessions. Pending gauges reuse the existing
backlog query and carry a sample timestamp. Selected-session idle age describes the
last selected batch; it is not a global oldest-pending query.

The most recent 256 completed spans per interval carry IDs, parent IDs, native
thread IDs and monotonic start/end times. Up to 256 active spans are retained.
Overflow increments `dropped_spans` or `dropped_active`; aggregates remain complete.
Trace IDs are meaningful only within one process lifetime. Trace records contain
no paths, queries, transcript content or exception messages. Incomplete traces must
not be used to claim a complete critical path.

Durations and calling-thread CPU are **inclusive**: parent and child spans overlap.
Async and queue spans and aggregates have null thread CPU;
`thread_cpu_samples` reports how many completed spans contributed to a CPU sum. Calling-thread CPU excludes DuckDB and
inference worker-pool CPU; the process total includes it. Do not sum nested spans,
attribute whole-process CPU to concurrent operations, or equate elapsed time with
CPU. Amdahl estimates need a representative workload and non-overlapping critical
path segments; the diagnostic timeline identifies where a short native profile is
needed to resolve the unattributed CPU remainder.

`telemetry.backend` identifies the last embedding model loaded, its runtime and
selected device/provider; null means no load has been observed. It does not claim
the model is still resident. Requested backend/model are recorded separately.
ONNX currently selects CPU; these diagnostics do not enable CUDA. MLX inference
spans include the existing evaluation/synchronization, and MLX allocator samples
include their timestamp. Allocator bytes and RSS overlap and must not be added.
GPU utilization and kernel time are not inferred from CPU counters. Kernel profiling
requires a separate bounded capture with the backend's supported profiler.

For a quick inspection (replace the data directory if configured differently):

```bash
tail -n 1 ~/.local/share/recall/diagnostics.jsonl \
  | jq '{pid, started_at, timestamp, process_cpu_seconds, rss_bytes,
         sampling_cpu_seconds, output_failures, scheduling,
         stages: (.telemetry.stages | keys)}'
```

To compare CPU with run-queue waiting over the last minute (six samples apart,
same process lifetime):

```bash
tail -n 7 ~/.local/share/recall/diagnostics.jsonl | jq -s '
  [first, last] as [$a, $b]
  | if length < 2 then "need at least two samples"
    elif [$a.pid, $a.started_at] != [$b.pid, $b.started_at]
    then "daemon restarted inside the window"
    else {
      window_seconds: ($b.monotonic - $a.monotonic),
      cpu_seconds: ($b.process_cpu_seconds - $a.process_cpu_seconds),
      runnable_waiting_seconds:
        (if $a.scheduling and $b.scheduling
         then $b.scheduling.runnable_waiting_seconds - $a.scheduling.runnable_waiting_seconds
         else "not recorded by the daemon that wrote these samples" end),
      priority_ceiling: $b.scheduling.priority_ceiling } end'
```

Keep the diagnostic files with the host, installed build identity and workload
window when comparing CPU and GPU runs. Compare empty idle, active ingestion,
backlog draining and interactive search separately.
