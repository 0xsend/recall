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

Use the populated post-migration database. Ordinary discovery, parser-revision
recapture, genuine rewrites and checkpoints are not versioned migration. Measure
all BRIEF normal-operation floors: readiness, discovery, responsive status/live
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
| 4 | Old-mtime import becomes searchable within discovery bound; unavailable configured root reports incomplete coverage. |
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
