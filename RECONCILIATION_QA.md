# Daemon reconciliation verification guide

This maintained reference maps the [reconciliation contract](SPEC.md#daemon-reconciliation-contract)
to executable checks. [BRIEF.md](BRIEF.md) owns quality floors and dated human
exceptions; this guide owns verification procedures. [QA.md](QA.md) covers
routine host inspection.

## Architecture under verification

One daemon owns DuckDB, the SQLite keyword sidecar, parser adapters, session
notifications and enrichment. Periodic full inventory and filesystem notifications
feed the durable `source_files` catalog. A bounded fair scheduler selects captured
source work; preparation occurs outside the writer, and one transactional writer
commits content and its checkpoint. Watch and poll share reconciliation; watch
adds notifications. Public coverage, activity, currentness and enrichment readiness
are separate observations, not synonyms for successful indexing.

The contract in SPEC is authoritative for generation acknowledgment, complete
record boundaries, prefix verification, retry/unsupported/missing states,
input-equivalent enrichment, rewrite cursor resets, durable pause and maintenance
ownership. Activity policy and its diagnostics have outstanding follow-up work;
the diagram is not a claim that
every latency or observability floor currently passes.

## Regression matrix

| Risk / contract | Required evidence |
| REQ-RECON-031 / REQ-LIVE-015: linked discovery and membership | `test_source_roots.py` covers all adapters, relative/absolute/chained links, canonical aliases and root order. `test_reconciler.py` covers final-link removal with readable target, incomplete preservation, topology mutation and every traversal ceiling. `test_moves.py` retains stable move proof/ID behavior. `test_reconciliation_public.py` and native watcher integration cover public ownership, source routing, retrieval and external updates. Author/guest poll and native Linux watch runs and an isolated populated backup copy qualify assembled behavior; native macOS and hard-floor measurements must be recorded separately. |
| --- | --- |
| REQ-RECON-001/002: incomplete discovery | Real inventory of old imports, missed events, unavailable roots and more than 64 active sessions; subscription capacity never defines catalog existence. Incomplete scans cannot mark omitted paths missing. |
| REQ-RECON-003: fairness and coalescing | Injected-clock scheduler traces with continuous arrivals, appends, retries and saturation; bounded selection, pending-age preservation and 7*N service-slot checks. Compare old-mtime history with freshly copied history explicitly. |
| REQ-RECON-004: unsafe append checkpoints | Parser/DuckDB matrix for append, growth plus prefix edit, equal size/mtime with changed ctime, shrink, replacement, split UTF-8/JSON and concurrent append. Expected final semantic rows are independently declared; incomplete tails never advance the complete boundary. |
| REQ-RECON-005: semantic parser omissions | Current native Codex response messages, reasoning, mirrors, mixed records, compaction and unknown records. Full read-only census checks native completed assistant text bodies, not merely recognized outer record types. |
| REQ-RECON-006: stale derived data/cursors | Rewrites, positional shifts, partial tails and follow reset; equivalent vectors/context survive, changed inputs invalidate derived data, obsolete bash keyword documents disappear, NULL processing markers cannot displace valid vector pairs. |
| REQ-RECON-007: false public claims | CLI/RPC tests for default and fresh reads, indexed/unindexed rows, unknown filters, missing roots, scope and pagination. A partial roster is not an exhausted roster. |
| REQ-RECON-008: model work blocks raw truth | Block model generation while raw writes, roster and reads progress. Generation-bound results cannot overwrite newer content; keyword and enrichment readiness are distinct. |
| REQ-RECON-009: pause lost on restart | Pause, stop the owned process, trigger read-client auto-start, verify pause and held bytes, then resume. |
| REQ-RECON-010..016: historical maintenance | Captured-scope backup, current-content fidelity, interruption/resume, exact completion, retention and recovery on the applicable corpus. |
| REQ-RECON-017..020: storage and performance | Fixed-format compatibility, real owned lifecycle, bounded writes and repeatable small/populated benchmarks below. |
| REQ-RECON-021..023: exceptions | Raw failures stay separate from candidate-specific dated BRIEF Decisions; no exception transfers without source/environment and authority binding. |

## Runtime and write regression coverage

- `tests/test_services/test_reconciliation_lifecycle.py`: aborted writer recovery,
  detached snapshots, model/raw barriers, cancellation ownership, stalled enrichment
  and bounded selection. Assertions concern usable connections and committed rows,
  not removed catch-up helper call sequences.
- `tests/test_services/test_reconciliation_requests.py`: manual/full/source/root/
  host/time/project requests, bounded context recomputation, durable metadata failures,
  embedding configuration, no-op context reuse, consistent recreate backups,
  rollback and stale-inventory invalidation. Non-target rows/vectors remain unchanged.
- `tests/test_services/test_reconciliation_public.py` and
  `tests/test_cli/test_reconciliation_cli.py`: poll pagination, unindexed
  coverage, parser/sidecar invalidation, source-root readiness, bounded envelopes,
  malformed hosts, offline failures and cursor reset.
- `tests/test_db/test_batched_history_writes.py` and
  `tests/test_services/test_indexer.py`: 256-row batches, complete scalar/JSON fields,
  UTC and Europe/Berlin naive/aware/absent timestamps, associations and index shifts,
  final stop facts across batch boundaries, no-op WAL, caller rollback after late
  failure, equivalent/fresh vectors and keyword mapping preservation. Samples include
  highest message/tool cardinality as well as largest byte size.
- `tests/test_services/test_reconciler.py`: unchanged inventory observations preserve
  indexed rows while updating observation/root/membership metadata; real changes,
  retries, duplicate aliases and mixed-batch caller rollback preserve catalog truth.
- `tests/test_services/test_fts_sidecar_reconcile.py`: bounded ordered identity
  comparison and actual mismatch repair, including equal-count drift and rollback.

These checks do not substitute for populated resource or assembled operator gates.

### Bounded read and inference memory

`tests/test_services/test_tail_snapshots.py` checks multi-page ordered payloads,
selected tools, concurrent rewrites between identities/pages/tools, and borrowed
connections retaining uncommitted state and rollback ownership. Existing tail,
show and follow suites cover the public cursor contract.

For physical allocation changes, compare complete tail payloads and tool
associations for windows of 1, 20 and more than 256 on populated owned copies.
Drive public after/follow/rewrite flows concurrently with raw and model work.
Measure the real MLX backend at short, mixed, truncated and 31/32/33/65/128-row
long shapes; check ordered finite normalized vectors, scores and near-tie ranks,
and a failure after one successful microbatch without partial publication.
Retain three fresh-process samples for short memory/CPU/latency comparisons.
Record process RSS separately from Metal active/cache peaks. Qualify the shipped
2GB default and any explicit engine override separately; a default pass does
not cover an override, and prototype peaks do not establish daemon RSS. Preserve
all existing BRIEF floors and historical processed context/vector/NULL markers.

### Bounded ONNX inference

`tests/test_services/test_onnx_embeddings.py` checks the thread share per host
size, run grouping and capacity, per-text vectors independent of batch
neighbours, and the padded-cell bound per inference run.

For changes to ONNX threading or run shaping, measure the real backend on a
Linux CPU host against a populated fixture copy, baseline and candidate
interleaved in three fresh processes each, on three workloads: whole-session
calls at the natural length distribution, inputs at or over the 512-token
limit, and a backlog of several default-size batches. Record texts/s, process
CPU-s per 1k texts, peak cores over 1 s and 10 s windows, peak and retained
RSS, padding ratio, p50/p95 run latency and host load. Declare tolerances
before the candidate runs: per-text cosine against the baseline (minimum and
p01) and unchanged top-10 results for a fixed query set over both the
re-embedded workload and the fixture's stored vectors. Then run a scratch
daemon on the fixture copy through an embed backlog, sampling whole-process
CPU and RSS and `daemon status` latency. CPU ONNX results do not qualify MLX
or CUDA.

## Assembled verification paths

Run objective checks first:

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Evidence records revision, relevant dirty state, verifier, exact environment and
input identities. Required specialist review follows objective checks; selected
independent terminal execution follows on the resulting artifact. Reuse only
matching unaffected evidence. An unavailable check is not green.

### Path A — versioned historical index migration

Use a new owned copy of a paused, populated database whose applied index-migration
version is behind the package's. Historical migration is eligible only when the
durable index version advances. Before mutation, retain a consistent DuckDB/WAL/SQLite
backup and enter explicit maintenance. Capture the source scope once; resume uses the
same scope and original backup. A population with a documented discrepancy record
(see the BRIEF Decisions) accepts only its certified differences; current supported
content, incomplete captures and undocumented differences never inherit that allowance.

Start with a few format samples and a small multi-source matrix, then verify the full
scope. Check per-source canonical content, equivalent enrichment, current keyword
membership, non-target payloads and source stability, not just aggregate counts.
Pin historical context to template/off with embeddings disabled for this isolated
raw-recovery workload. Those fixture settings do not disable installed-host features.
Measure total RSS and the BRIEF migration runtime budget. The normal-operation
five-second writer and concurrent active-freshness floors do not apply in this phase.

Interrupted execution shares one conservative runtime budget: include every
execution segment, setup/shutdown and uncertain gap. Exclude only intervals whose
immutable receipts prove the candidate offline with exact endpoints. Retain original
wall duration separately. Never reset the clock on resume. Source-level semantic
checks remain mandatory alongside aggregate subprocess memory measurements.

### Path B — normal daemon operation

Use the populated post-migration database. Ordinary discovery, full re-parses
after a parser-revision checkpoint mismatch, genuine rewrites and checkpoints are
not versioned migration. Measure all BRIEF normal-operation floors: readiness, discovery, responsive status/live
(including `live --fresh`), at least 20 appends concurrent with history, service-slot
fairness, every exclusive writer/checkpoint turn and total RSS. Large-source and
high-cardinality samples run with default DuckDB allowance and thread settings.

`live --fresh` catches up already-indexed rows only inside its responsive-read
bound; it does not first-index a roster. `show --fresh` owns one-session refresh.
Do not extend RPC deadlines or `live.fresh_timeout` to hide work. Full roster
coverage requires exhausting the continuation cursor under declared stable-membership
conditions; sampled fast pages alone do not prove complete roster acceptance.

### Objective operator charter

The isolated charter runs in order within 30 minutes and records every task and
cleanup result. Exactly ten passing tasks are required for its green verdict;
exceptions stop the run and leave later tasks NOT run.

| # | Task and postcondition |
| --- | --- |
| 1 | Cold paused backlog: RPC/live readiness, complete catalog coverage, bounded source paging and matching resource evidence. |
| 2 | Poll roster: all 80 unindexed sessions paginate exactly once; unknown filters stay incomplete; fresh reads do not falsely mark unindexed rows current. |
| 3 | Four active appends: follow/commit exactly once in order, five final messages. |
| 4 | Old-mtime import becomes searchable within discovery bound; absent configured root completes empty and remains unavailable in coverage, then is discovered when created; permission and non-directory errors block readiness and name the root (REQ-RECON-030). |
| 5 | Split record commits once after completion; malformed complete input retains an error and diagnostics. |
| 6 | Equal-size rewrite becomes current; incompatible follow closes with `content_rewritten`; old show cursor resets, including positional replacement. |
| 7 | Pause survives owned stop/read-client auto-start; held bytes commit only after resume. |
| 8 | Unavailable context backend fails visibly while later raw work commits and RPC remains ready. |
| 9 | Owned confirmed recreate retains a consistent readable backup and restores exact source messages. |
| 10 | Follow closes with `daemon_stopped`; restart preserves content; status/live/show/keyword search succeed. |

The driver-executed objective charter is not an independent experiential judgment.
A selected fresh operator receives tasks, artifact, environment, budget and severity
floor, not the author's diagnosis. Loss/duplication, false current/complete/absence,
uncontrolled writes, undiagnosable pending work and declared-workload starvation
are Major blockers unless a specific human disposition applies.

### Installed host

After applicable clone gates or explicit dated exceptions, freeze the candidate,
verify an immutable executable fallback against a new consistent live backup, and
perform the authorized stable-tool rollout. Follow
[CONTRIBUTING.md](CONTRIBUTING.md#development-and-stable-installs) for Python 3.12
and extras; preserve actual enabled host context/embedding settings.
Never install a service from a development binary or restore an older clone over
newer live data. Keep ownership, disk-reserve and resource monitoring explicit.

Five installed tasks cover readiness/identity and bounded metadata; new discovery
and at least 20 exact-once append/follow updates; pause/equal-size-and-mtime rewrite
and cursor reset; marked-writer active-to-ended transition; and populated historical
read/keyword search plus durable pause/restart/resume. A later forced refresh does
not clear a default-read deadline miss. Partial tasks, omitted timing and sampled
memory remain scoped evidence, not a complete installed-host pass.

## Separate checkpoint maintenance gate

Measure every exclusive checkpoint turn separately from source commits, including
concurrent read start/completion and active queue latency. The application 64MiB
high-water and engine 256MiB fallback do not prove latency bounds; one transaction
may overshoot either. Real tests cover committed WAL, below-mark no-op, maintenance
success/failure, recheck under ownership, crash between commit and checkpoint,
cancellation/shutdown and eventual maintenance from every writer surface. Path A
measures crash-safe progress and total migration time; Path B enforces normal floors.

## Storage modernization baseline

Compare existing runtime/format and proposed fixed targets on owned checkpointed
copies under every supported runtime, including rollback. Record interpreter/engine
paths and versions, format identity, configuration, input hashes and all results.
Verify read/write/index lookup, rollback and ordinary reopen before selecting a
runtime or format. Inventory workarounds by their current reproducible reason;
new engine syntax does not by itself retire bounds or recovery guards.

## Storage modernization lifecycle

The selected new-file target is fixed `v1.2.0` under DuckDB `1.5.5`; existing formats
change only through explicit backed-up maintenance. Test legacy ordinary-open
preservation, durable job/attempt state, persisted header identity, interruption
before/after conversion, same-scope/original-backup resume, separate-copy rollback,
concurrent ownership and cancellation. No diagnostic create/drop-table conversion,
downgrade, hidden logical schema change or historical reparse on ordinary upgrade.

`recall daemon migrate-storage --dry-run` reports current/desired format, index
eligibility and job/backup state. Apply validates a supplied plan digest and returns
an accepted operation; inspection, not acceptance, proves completion. The shared
owner drains users, closes/converts/checkpoints/reopens and verifies persisted
format. Storage-only work may retain operator pause; raw index maintenance respects
it. Recovery backups survive startup retention regardless of age. Validate namespace
inventory against a known-present backup before interpreting an empty query.

## Storage modernization writes

Preserve bounded allocation, atomic commits, row/mapping identity, ordering, NULL
handling, keyword membership, no-op behavior and equivalent enrichment. Refactors
and behavior changes need separately attributable evidence. Captured reparse scope
uses distinct source/path pairs in at most 256-row batches with one pending timestamp;
missing/repeated identities reject before writes, callers own rollback, and resume
never recaptures or advances generations again.

## Storage modernization benchmarks

```bash
uv run python scripts/benchmark_reconciliation.py --output <new-directory>
# Populated: add --population <checkpointed-owned-data-directory> --source <Codex-transcript>
# Comparison: add --baseline <matching-result.json>
```

The command runs three fresh processes, retains logs/results and reports every
sample, median and worst for first index, append, rewrite, no-op, checkpoint,
keyword/vector search, live pages/freshness, RSS, WAL and persistent growth. It
clones only DB/sidecar payloads, verifies input identity and pins isolated historical
template/off context with embeddings disabled. Up to three 50-row live pages use a
one-year activity window; small populations may exhaust sooner. This is a regression
benchmark, not full Path B acceptance.

Predeclared provisional comparison tolerances are 20% plus 10ms for subsecond
latency, 20% plus 50ms for longer latency, and 10% plus 16MiB for RSS/persistent
bytes, for both median and worst on matched inputs. These never relax a hard BRIEF
floor. Normal writer/checkpoint acceptance requires every turn <=5s and seeks <=4s
worst headroom. Missing measurements, absent overlapping reads, failed search,
no-op WAL changes or hard-floor breaches require nonzero exit. Failed queries are
not latency baselines. `tests/test_benchmark_reconciliation.py` checks discrimination.

### Evaluating a DuckDB upgrade

1. Freeze input/verifier identities; retain consistent DB/sidecar backup, previous
   installation, source/dirty state, runtime packages, OS/CPU and transcript hashes.
2. Check official engine release/storage documentation and compare fixed versions,
   never `latest`, under every supported runtime. Engine change is not reparse authority.
3. Run matched small/populated commands with three fresh processes per candidate.
   Inspect all costs and failures. Instrumented diagnosis requires separate
   uninstrumented timing acceptance; no fastest-sample or blind retry acceptance.
4. Reproduce each retained workaround before removal. Check exact canonical/vector/
   keyword/non-target content, no-op behavior, ordinary reopen and used file bytes.
5. Exercise actual backup, interrupted original-job resume, separate-copy rollback,
   shared ownership and persisted format; verify captured scope where applicable.
6. Rebind full suites and affected Path A/B/operator/lifecycle evidence. Installed
   promotion still needs its own authorization, matching backup and faithful config.

### Retained storage and write safeguards

Production compression and row-group defaults remain unchanged. Alternative
compression/row-group timing experiments do not authorize adoption. Any proposal
must prove ordinary root/cursor/second-connection ownership, reopen/rollback,
complete existing vector hashes and counts, search, persistent used bytes and all
hard/comparison floors; no connection proxy or schema change hides in a benchmark.

| Mechanism | Removal evidence |
| --- | --- |
| Exact indexed-column masks | `test_rewrite_updates_only_changed_columns_and_preserves_related_rows` and fixed-format capability matrix pass without row rewrites. |
| Orphan companion cleanup | Both `test_orphan_message_state.py` repair cases retain an explicit owner. |
| Embedding change detection | `test_tool_call_embedding_churn.py` remains flat across repeated unchanged writes. |
| Materialized vector scores/ranks | `test_vector_search_memory.py` preserves filtered IDs/scores/payloads for 8192 wide candidates and 32768-vector joins under64MiB; populated write-then-search also passes. |
| Legacy DuckDB-FTS checkpoint | `test_create_fts_indexes_checkpoints_unreplayable_shadow_schema_drop` replays after abrupt exit with an independently checkpointed baseline shadow schema. |
| 256-row writes / 500-ID lookups | Batching/rollback and populated benchmarks retain ordering, no-op behavior, bounded memory and latency. |
| Ordered keyword deletion/reinsertion | The `test_fts_sidecar_*` suites cover repeated IDs, interleaved absent/present NULLs, maximum-rowid reuse, batch boundaries and rollback; only proven absent deletions may disappear. |
| Durable sidecar repair | Cross-store failure tests repair touched identities after either independent commit boundary fails. |
| WAL thresholds | Connection/maintenance crash, no-op, failure and ownership checks plus separately measured Path B checkpoint latency. |
| Shared maintenance owner | Real conversion/compaction and populated lifecycle checks drain users and recover cancellation/failure to a usable handle. |

Source-qualified identity (REQ-MIG-012, REQ-RECON-032) is covered by
`tests/test_db/test_source_path_migration.py` and
`tests/test_services/test_shared_source_path.py`: migration preserves rows,
rolls back failed DDL and replays idempotently; both insertion orders and concurrent
fresh reads retain independent source histories. Populated migration, interruption,
backup restore, native-watch and backlog rehearsals remain release evidence.

## Responsive cohort qualification

`REQ-RPC-024..026` and `REQ-LIVE-014` require matching client/server builds.
Measure complete actual CLI calls, including startup and output, against an
owned prepared copy of the populated rollback backup. Run three fresh
baseline/candidate pairs under the [shared bench lock](#bench-lock), releasing it
between pairs. Record build/artifact identity, fixture hashes, host load, model
residency, every call and final exact session/message/tool deltas. Retain failed
and incomplete attempts. Instrumented spans attribute inclusive work and queue
waits; a subsequent uninstrumented run establishes wall-time acceptance.

For each loaded phase, run 20 live and 20 status processes while 12 appenders
make 40 writes at 250 ms. Cover default fresh pages, 252-row maximum pages and
full 256-row pages, then quiet controls and complete drain (pending=0, no retries,
failures or generation errors). Every successful loaded call must complete in
<=2.000 s. Errors, refusals, local fallback and skipped callers fail this floor.
A shorter catch-up wait alone is insufficient: coherent final snapshots and
removal of repeated query work must be observed. Record CLI startup/output,
transport, server queue/worker and catch-up spans separately without adding
nested intervals. Remote fleet transport does not qualify the local floor.

Real-DuckDB regression scenarios cover pre-transaction admission, late callers,
method/page/filter/phase keys, caller-local output, exact fresh status counts and
same-transaction coverage, time-only cutoff changes, and NULL processing markers.
Exercise single/all queued and executing departures, shutdown, deadlines and
epoch replacement without leaking slots or returning old-epoch results. Drive
fresh/non-fresh and failure flows through actual CLI/RPC, including torn-tail
repair, rewrite/reset, exact identity and unrelated-history preservation.

At the merging head also rehearse >=25k eligible historical sources with active
appends, largest supported sources/checkpoints, real vector/hybrid search and
enrichment. Report per-append p95/max, every normal writer/checkpoint duration,
whole-process RSS and completed drain against BRIEF floors. Small fixtures and
model-free timing do not qualify populated pressure, MLX or CUDA. Record missing
classes as gaps; the PR stays draft pending specialist review and those gates.

## Bench lock

Controlled measurements use the maintained POSIX runner
[`scripts/with_bench_lock.py`](scripts/with_bench_lock.py). All lanes on one
host must agree on the same absolute queue directory; relative `--lock` paths
are refused before creating a queue or running the command. Gates and functional
rehearsals run without the shared measurement lock, under `nice -n 10`.

```bash
export WITH_LOCK_PATH="$HOME/Work/recall-performance-$(hostname)-bench.lock"
uv run python scripts/with_bench_lock.py --lock "$WITH_LOCK_PATH" baseline-1 \
  uv run --no-sync python scripts/benchmark_reconciliation.py --output <new-directory>
```

Waiters register FIFO under a short kernel mutex and wait on all earlier live
tickets, including when an intervening waiter is killed. Each ticket records
the supervisor PID, process start time (Unix seconds), label and queue number.
Kernel `flock` ownership, rather than PID liveness or marker-file presence,
decides admission, so PID reuse cannot preserve an orphan. Queue wait defaults
to three hours; `--timeout <seconds>` bounds admission only. The queue admits
at most 1,024 live entries. JSON events on stderr report `queued`, `acquired`,
`released`, `timed_out` and `failed`; stdout belongs to the command. Exit status
is the command's status, 75 on wait timeout, or 128 plus the cancellation signal.
The `released` event ends the supervisor/ticket lifecycle and carries an
`acquired` boolean: `true` if that invocation was admitted, `false` if it timed
out, was cancelled, or failed before admission. For example:

```json
{"lock": "released", "label": "baseline-1", "child_status": 75, "acquired": false}
```

Invalid arguments, including relative `--lock` paths, exit 2 with an argparse
diagnostic before any JSON event is emitted.

TERM/INT stop the owned command process group before releasing the ticket. The
command inherits the locked descriptor: after supervisor SIGKILL, admission
remains blocked until that command exits, preventing overlapping measurements.
Inner harnesses must contain and reap descendants they launch in separate
process groups. An escaped descendant can retain the inherited lock descriptor
after the supervisor emits `released`; that event does not prove the shared
queue is free. Such descendants are outside the runner's containment contract.
Queue metadata can remain after exit and is reclaimed on the
next registration; its presence does not mean the lock is held. Never remove
the queue directory or its `queue` mutex file while runners may use it.

The former experiment-local mkdir lock is superseded by this runner. A host
must drain users of that runner before switching; a stale legacy directory
does not carry kernel ownership. No production database or daemon lock uses
this queue.
