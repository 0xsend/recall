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

Files the daemon has catalogued but can no longer reach through any configured
route are excluded and counted as `missing`, including readable external targets
whose final directory link was removed. Files under a disabled root are excluded from `pending` and
counted as `out_of_scope_pending`.

A configured root that does not exist is an empty scope for readiness after
inventory observes its absence. It stays in `coverage` with
`configuration=configured` and `available=false`; previously indexed history is
retained. A root that appears later is picked up without restart by periodic
inventory, and watch mode attaches its structural watch on a discovery tick.
Permission errors, non-directory paths, and other scan failures still block
readiness and preserve prior catalog membership (`REQ-RECON-030`).

`pending` is not an embedding count. Embedding work is counted in message and
tool-call rows, read fresh on every status call: `embed_pending_eligible` for
sessions idle at least `embed_idle_session`, which the embed loop may select
now, and `embed_pending_held` for sessions still inside that hold, both measured
at `embed_pending_measured_at`. `embed_pending` and `embed_pending_at` are the
embed loop's last eligible snapshot; it reads 0 while a fresh session waits out
its hold, so it cannot say embedding is done. `reconciliation.enrichment_ready`
is true when embedding is disabled, or when an embed phase is running,
`raw_indexing_ready` is true, both fresh counts are 0 and no embed error is set
(`REQ-ADAPT-022`). With embedding
disabled the flag is true whatever the counts, because that work will not be
embedded. A harness that only needs settled history waits for
`embed_pending_eligible == 0`.

With embedding enabled, `enrichment_ready` implies `raw_indexing_ready`: content
not indexed yet has no rows to count as pending embeddings. In watch mode every
transcript write under a watched source root is reported when it happens, and
`noticed_pending` counts files with reported writes awaiting observation. A mark
leaves after observation commits or after three failed captures; give-up lists
the path in `reconciliation.error`. All enabled readiness flags stay false while
the count is nonzero; with embedding disabled, `enrichment_ready` stays true
(`REQ-DAEMON-079`). After a restart raw and keyword readiness stay false until
every root has been scanned since the daemon started, since only such a scan
sees what was written while it was down. Readiness covers only writes the daemon
is told about: in poll mode, or under a root whose watch could not be placed (a
`failed to watch source root` warning), a write is seen at the next inventory
walk, and the flags can read true until then.

Raw indexing progresses independently of model work (`REQ-RECON-008`): a
deferred embed phase does not hold up the reconciliation drain, and an empty
reconciliation backlog does not imply embeddings are done.

## Linked directories

Directory symlinks below a configured root delegate their targets to that source,
including targets outside the root. Relative links, chains and resolvable cycles
use canonical directory membership. Each canonical file is indexed once per source;
file links delegate only the matching encountered filename, never their siblings.
Sidecars are read beside the canonical transcript. The first lexicographically
sorted canonical root reaching a file owns it; coverage counts partition these owners.

Only link content intended for indexing and the configured enrichment backend.
Removing one alias leaves remaining routes eligible. Removing the last alias,
removing a configured root or retargeting a link leaves prior history readable and
searchable while the old target loses catalog backing. `index_only_sessions`
describes that backing, not proof that a file was deleted. A lost link to a readable
target is not a physical move. Proven physical moves continue to retain identity
under REQ-INDEX-027; aliases to an unmoved target also retain IDs and derived rows.

A source scan waits for every root before settling absence. Broken or inaccessible
links, non-directory roots, topology changes during capture and exceeded ceilings
leave it incomplete, preserve old membership and block readiness until a repaired
scan completes. An absent top-level root still completes empty. Root `errors` and
`failure_count` name failures, including source, root and limit; samples are capped
at 32 details of 256 characters each.

Initial source-scan ceilings are 1,024 configured roots, 100,000 canonical directories,
100,000 file-link targets, traversal depth 256, 65,536 entries in one directory,
and 2,000,000 entries examined across scope construction and streaming enumeration.
These are failure ceilings, not truncation with a successful completeness claim.
Watch mode adds at most 1,024 delegated structural watches and removes stale ones.
Failed/excess watches are logged and exposed as runtime limitations; periodic
inventory remains responsible for missed events and old-mtime imports. Content
watch limits and polling fallback retain their existing settings.

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
| `pending` | Catalogued files not yet committed to their current bytes, or with a blocking error. A caught-up unsupported source is current and does not count here. |
| `paused` | An operator pause is in effect; work is held, not lost. |
| `catalog_scan_complete` | Every configured root finished a scan. While false, totals are still moving. |
| `raw_indexing_ready` | Scan complete and `pending == 0`. |
| `keyword_search_ready` | `raw_indexing_ready` plus an empty keyword sidecar queue. |
| `enrichment_deferred` | Why the embed phase last stood down (for example host load above threshold). Does not block the drain. |
| `error` | The inventory error, when a scan itself failed. |
| `coverage[]` | Per source and root: configuration, `available`, `pending`, `oldest_pending_age` (seconds), `scan_complete`, `failure_count`, `errors`, `unsupported`, `missing`. Live RPC adds `scan_current_process`: disabled roots need no scan; enabled roots require a scan started by this daemon. |
| `unsupported_summary` | Parked parser gaps an agent can file: source, record detail, file count, and up to three sample paths. Kind-only rows from an older daemon say the detail was omitted. |
| `source_page` / `next_cursor` | A bounded page of per-file catalog records with diagnostics; `next_cursor` is an opaque token to pass back verbatim. A cursor that is empty, malformed, or names no catalogued file (mangled, truncated, edited) is a validation error (exit 2), never a silently different page. Paging is by key: a file added behind the cursor during a walk appears in the next walk, not this one. |

Text mode's summary block states the headline numbers; its `ready:` line gives
the three readiness flags, or says they are unknown when no daemon answered.
Each `Root blocks readiness: <source> <path>: <reason>` line names a root
with an incomplete scan, an observation inherited from a previous daemon, or
pending files. Scan failures include the recorded filesystem errors. An absent
root whose empty scan completed has no blocking line.
Its `More source coverage: --cursor ...` line prints the cursor shell-quoted,
so it can be pasted into bash or zsh as is. For a scripted read, project the
keys you need; there is no separate reconcile command, and none is needed:

```bash
# The drain signal alone.
recall daemon status --json --fields reconciliation.enrichment_ready

# Headline drain state, with the per-file page held to one row.
recall daemon status --json --limit 1 --fields reconciliation \
  | jq '.reconciliation | {pending, paused, catalog_scan_complete, keyword_search_ready, enrichment_deferred, error}'

# Where the backlog sits, and how old the oldest pending file is.
recall daemon status --json --limit 1 --fields reconciliation \
  | jq '.reconciliation.coverage[] | {source, pending, oldest_pending_age, scan_complete, unsupported, missing}'
```

`--fields reconciliation` returns the whole object, and `--fields
reconciliation.<key>` one key of it. `--limit` keeps the per-file page small;
raise it with `--limit N` (1..256) and page with `--cursor` when the per-file
diagnostics are what is wanted. With no daemon running, `--limit`/`--cursor`
page the same catalog locally, with the same cursors; while a daemon holds the
database without answering, a named page fails rather than answering another.

A caught-up unsupported transcript reports `freshness.current: true` with
`unsupported_records` in `freshness.limitations`. `show --fresh` does not warn
that it is behind. Unsupported notices and `unsupported_summary` still describe
the parser gap; they do not by themselves block raw or keyword readiness.

## When pending is not moving

Two readings a minute apart establish whether it is stuck. If the count is flat:

1. **Check `paused`.** A durable pause survives restarts and client auto-start
   (`REQ-RECON-009`). `recall daemon resume` clears it.
2. **Check the daemon is actually serving.** `recall daemon status` reports
   `version_drift` and, when the RPC is unreachable, says so instead of
   reporting runtime fields. A drifted daemon runs old code until restarted.
3. **Check the `Root blocks readiness` lines, `error`, and per-source
   `failure_count` / `errors`.** An absent root completes empty but remains
   listed unavailable. A root that cannot be scanned because of permissions or
   another filesystem error reports incomplete coverage.
4. **Check `unsupported`.** Those files hold a record type their parser does
   not know. Their supported content keeps indexing. Once the committed generation and
   acknowledged offset match the observed file, they count as current and leave
   pending; the diagnostic remains a content limitation. Unacknowledged bytes
   (including a torn tail) still count as pending. Inspect `unsupported_summary`
   and file a recall issue (`REQ-CLI-023`).
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
billed to `utility`. `recall daemon install` writes `ProcessType = Interactive`, which
lifts the clamp; a plist installed by an earlier release keeps it until install runs
again. An unclamped process reports 63; its threads mostly bill to `legacy`, the
class of threads that never set a QoS. The enrichment thread (model load,
inference and unload) lowers itself to utility, so on an unclamped daemon the
`utility` share is enrichment work, and a failure to lower is logged as `thread QoS
demotion`. The other keys are the remaining macOS QoS classes (`default`,
`maintenance`, `background`, `user_initiated`, `user_interactive`); a few threads set
one explicitly. Both are null on Linux;
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
Because that ring keeps the most recent spans, a burst after a slow operation can
evict it, so each interval also keeps `slowest_spans`: per stage, the four longest
spans completed in that interval, longest first, in the same record shape. Its keys
are the stage names used in `telemetry.stages` (for example `writer_run`,
`raw_commit`, `read`). A span can appear in both lists. Both lists close when the
recorder snapshot is taken, after the sample's `monotonic` is read. A span's `end`
can exceed that `monotonic` by the collection delay; that delay can grow under
load. The stage's `count` delta between successive written samples is the number
of completions the four were chosen from only if no intervening output failed.
An increase in `output_failures` means the delta can include completions from
unwritten snapshots: a failed append discards that snapshot's interval records.
A retained span's parent may be
absent from both lists; its `parent_id` still names it. Samples written before this
field existed omit it and still validate.
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

To list the longest writer turn of each recent interval (samples written before
`slowest_spans` existed print null):

```bash
tail -n 30 ~/.local/share/recall/diagnostics.jsonl | jq -c '{
  timestamp: (.timestamp | todate),
  writer_run: (.telemetry.slowest_spans.writer_run[0]? // null
    | if . then {id, parent_id, seconds: (.end - .start), outcome} else null end)}'
```

Keep the diagnostic files with the host, installed build identity and workload
window when comparing CPU and GPU runs. Compare empty idle, active ingestion,
backlog draining and interactive search separately.

## Historical moved-session census and rehearsal

The report-only form names an owned backup-copy data directory explicitly:

```bash
RECALL_DATA_DIR=/absolute/path/to/owned-backup-copy recall db supersede-moved --json
```

It reports historical move candidates without
creating, migrating or writing the existing database. It requires exclusive
maintenance ownership; stop the database owner only under a separate lifecycle
authorization. Always work first on an owned, consistent copy of DuckDB and its
SQLite sidecar. Transcript files are read for prefix proofs, never modified.
A missing database is an error, not an empty census. Ordinary `show`, `list`
and `search` reads can auto-start a daemon: perform maintenance before those
reads, or arrange a separately authorized shutdown of that exact owner.
Installed-service stop commands can act on production despite a scratch data
directory. `--fields census,pairs` selects the two complete top-level objects;
nested census projections are not supported.

After reviewing that report, fixture rehearsal can apply against the same
disposable copy:

```bash
RECALL_DATA_DIR=/absolute/path/to/owned-backup-copy recall db supersede-moved --apply --json
RECALL_DATA_DIR=/absolute/path/to/owned-backup-copy recall db supersede-moved --json
```

`--apply` refuses unless `RECALL_DATA_DIR` is explicitly set and nonempty,
including when apply is requested through `--params`. A DB or lock path override
alone does not satisfy that requirement. Check any `RECALL_DB_PATH` and
`RECALL_LOCK_PATH` overrides also name the copy. An explicit directory is a
target, not authorization for production repair. The apply response's
`census.phase` is `pre_apply`: its counts, groups and top-level `pairs` describe
the inventory before deletion. `superseded` and `moves` describe the actions.
The following report is the post-image, with `census.phase` equal to `report`.

The version-1 `census` object contains the local session count, timestamp-less
session count, repeated `(host, source, source_session_id, started_at)` key count,
extra row count, candidate groups and overlapping group classification counts.
A candidate group shares source and first timestamp; it is not itself proof of
a duplicate. Rows include exact session ids, declared source ids, hosts, paths,
committed offsets and SHA-256 prefixes. Foreign-host rows are excluded by the
same local-host policy as the move planner. `host_filter.hosts` lists the current
short hostname and the unattributed literal `local`; `host_filter.include_null`
is true. The predicate is `host IS NULL OR host IN (short_hostname, 'local')`.
`total_sessions` counts all session rows, `sessions` counts the included local
rows, and `excluded_sessions` is their difference. The one-million-session cap
uses the total, including foreign hosts. A copy read on another host can
therefore report zero local sessions while excluding a nonempty catalog.
Rows without a timestamp are counted
and repeated declared-id groups without timestamps are retained in the report.
`changed_timestamp_candidates` contains repeated declared ids with different
first timestamps: it cannot distinguish an edited timestamp from a fork or
subagent and never authorizes removal.

| Class | Meaning / disposition |
| --- | --- |
| `proven_move` | Exactly one committed survivor holds the departed prefix; eligible for explicit apply. |
| `missing_timestamp` | Repeated declared-id rows without timestamps; the native id supplies candidates, but committed-prefix proof and a unique departed identity are still required. |
| `candidate_overflow` | More than 16 departed or surviving rows in a mixed group; retain. |
| `missing_catalog_proof` | Missing digest/offset or nonpositive committed offset; retain. |
| `prefix_limit_exceeded` | A committed prefix exceeds 512 MiB; retain. |
| `unreadable_survivor` | The survivor cannot be read through its committed offset, including a shortened file; retain. |
| `lagging_survivor` | Its indexed offset is behind the predecessor; retain. |
| `divergent_content` | Readable prefixes disagree with committed digests; retain. |
| `multiple_proven_survivors` | More than one survivor holds the predecessor; retain. |
| `ambiguous_predecessors` | A survivor holds the prefix but several departed identities can claim it; retain. |
| `surviving_copies` | No departed path in the group; retain. |
| `no_survivor` | No surviving path in the group; retain. |

A group may contain both a proven pair and unrelated unproven rows. Classes count
once per group and may overlap; they must not be summed as a duplicate total.
An oversized predecessor with a lagging survivor records both
`prefix_limit_exceeded` and `lagging_survivor`; changing the byte cap alone would
not establish a complete survivor.
The inventory fails explicitly above one million sessions, rather than returning
a partial result. Existing `moves` retains its path-pair format. The added `pairs`
contains exact predecessor/successor ids, their committed proofs, predecessor
row counts by affected table, and successor message ids eligible for stored
context transfer. `successor_vector_message_ids` and
`successor_vector_tool_call_ids` identify additional absent vector rows eligible
for transfer at matching message/tool positions with identical stored inputs.

Planning and classifying groups read transcript prefixes separately. If a
transcript changes between those reads, `pairs` and the group classifications
can describe different file states. Apply uses the planned `pairs`; the phase
label does not make those reads an atomic file snapshot.
Existing successor embedding rows, including processed NULL markers, remain
unchanged by this additional transfer. The surviving history and its companions
are rekeyed into the departed session's namespace; the departed session id and
positional message/tool ids remain valid, while the separately indexed copy's
ids retire. Usage links and non-null stored usage totals survive, and the
catalog binding records the current path. Context, vectors, tool results and
keyword refresh rows change in that transaction. Offline keyword sidecar
deletion waits for the daemon's ordinary startup reconciliation.

`--apply` remains explicit and replans current candidates; it opens the existing
database without upgrading its schema. Each pair commits atomically. An aborted
pair rolls back, already committed pairs remain committed, and a repeated run
plans only remaining eligible pairs. Rehearse report → apply → report and a
second apply on the copy, including interrupted execution, sidecar reconciliation,
restart retrieval, unrelated-history and context/vector/tool/usage preservation.
Restore DuckDB and SQLite together from the immutable backup to rehearse rollback.

A census is not a production repair grant or a digest-bound selected-pair plan.
Before production repair, obtain a separate per-host approval naming the backup,
database, exact reviewed pairs and executable revision, plus migration specialist
review and a rehearsal on a fresh copy. The current command cannot constrain apply
to an approved manifest; that capability needs its own design before any
restricted-scope production execution. Do not treat all repeated keys as moves.

## Compressed Codex rollouts

Codex `rollout*.jsonl.zst` files are discovered within the configured Codex roots.
The index retains the canonical `.jsonl` path, session IDs and message IDs.
When both representations exist, the plain file is authoritative. Archive roots
are scanned only when explicitly configured. Recall never materializes or
changes source transcripts.

A dangling plain symlink counts as an absent plain representation: its real
compressed sibling uses the canonical `.jsonl` name, not the missing symlink
target. Other plain-path access errors never select the compressed sibling.
Move detection and duplicate supersession stat and hash the selected physical
representation. A present compressed sibling keeps its session present;
renaming or copying then removing it can preserve identity only with the same
unique committed-byte proof used for plain files. Recompression or conversion
to plain changes those bytes and does not by itself prove a move.

A compressed rollout is a complete capture: offsets and digests describe the
physical Zstandard bytes, with no incremental normalization checkpoint. Codex
restores plain JSONL before appending; reconciliation normalizes that restored
file and subsequently uses the ordinary verified append checkpoints. A failed
compressed decode or incomplete JSONL capture leaves indexed history intact
and appears as a source error.

Compressed input and decoded JSONL are each limited to 96 MiB, decoder windows
to 64 MiB and frame counts to 4096. The existing 64 MiB per-record bound still
applies. Preparation reserves for the decoded ceiling rather than the compressed
file size. Files exceeding these bounds remain visibly pending with their prior
indexed history preserved.
