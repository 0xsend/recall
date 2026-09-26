Law doc for daemon reconciliation, present-tense, no narrated history — git is the changelog. The Boundary and ratified Decisions amend only with human confirmation; provisional Decisions are dated.

## Bar

The daemon is shippable when versioned historical index migration and normal reconciliation each meet their own floors: truthful and lossless in both; bounded and observable in migration; responsive, fair, and bounded in normal operation.

## Dimensions

Correctness, readiness, latency, fairness, bounded resources, observability, and control semantics.

## Floors

- Readiness: RPC/status ready <=30s regardless of backlog/model, including during versioned index-migration maintenance.
- Responsive reads: status/live <=2s after readiness under workload. `live --fresh` is a responsive read: it may catch up already-indexed pending bytes within that bound and must not first-index or wait on coordinator backlog.
- Active freshness: during normal operation, >=20 appends while history runs, p95 <=15s and max <=30s. This floor does not apply inside explicit versioned index-migration maintenance.
- Discovery: newly discovered path observable <=45s at default scan under normal operation.
- Daemon idle CPU: on the integrated installed candidate, after no raw commit for 120s and with no active agent session, process CPU averages <=5% of one core over a 10-minute trace and no periodic operation costs >1 CPU-second per occurrence. The retained trace records per-second process CPU and periodic-operation attribution; the discovery floor remains in force.
- Write bound: during normal operation, including ordinary historical discovery, genuine rewrites, and checkpoints against a populated database, the largest corrected 86.4MB sample holds the exclusive writer <=5s under the synthetic backlog and against that populated database; otherwise staged/chunked. This ceiling does not apply inside explicit versioned index-migration maintenance.
- Fairness: during normal operation, 4 active : 2 recent : 1 oldest scheduler satisfies the 7*N service-slot bound. Ordinary pending work is not relabeled as migration to evade this floor.
- Resources: <=2 preparation slots, <=256 ready rows, <=4096 buffered notifications with overflow root scan, RSS <=4GiB under the largest-session plus >=25k backlog and under versioned index migration, and bounded capture/read buffers and futures.
- Correctness/control: zero loss/duplicates; truthful coverage/freshness/cursors; pause honored; coalescing <=10s; equivalent enrichment preserved; current keyword membership; transcripts unmodified by index migration.
- Index migration: eligible only when durable `INDEX_MIGRATION_VERSION` advances; consistent backup before mutation; explicit maintenance state; observable progress; crash-safe resume of the captured scope; verified completion before the version is applied; safe rollback; subsequent transcript changes caught up by normal operation; overall runtime <=4 hours for the reference corpus. A captured source that remains pending only as stable unsupported, with prior indexed history preserved, settles its captured key and stays non-current. The five-second writer ceiling and concurrent active-freshness gate do not apply in this phase.

- Storage maintenance: new files use the selected fixed format; existing files convert only through explicit backed-up maintenance. Real-DuckDB interruption/resume/rollback and concurrent connection-ownership checks prove persisted identity and preservation. Format and index-migration eligibility remain distinct.
- Performance regression: at least three fresh-process samples for short comparisons, with every result retained and median/worst reported. Additional timing tolerances are declared before comparison in QA. Small representative cases provide feedback; populated-corpus gates establish scale. Normal writer/checkpoint acceptance uses every measured turn <=5s and seeks measured headroom.
- Incremental raw append: equal 200-line appends to small and at least 33 MiB histories use verified committed normalization checkpoints. Across three fresh-process samples, the large-history median is <=1.0 CPU-second, at least 5x below the retained 5.9 CPU-second baseline, and <=`3 * small_median + 0.25` CPU-second; every supported adapter passes append equivalence and fallback coverage.

## Oracle

Objective unit, schema, concurrency, CLI/RPC, and resource harnesses run first on an isolated daemon. Versioned historical index migration and normal daemon operation are separate acceptance paths; a green result on one does not satisfy the other. Retained evidence binds to the tested revision, relevant dirty state, artifact, inputs, and environment. The local host gate runs separately with a new consistent live backup only after both clone paths pass on the same candidate. Major or blocking findings fail the gate.

## Never

Never silently lose or duplicate content, claim readiness falsely, exceed hard resource bounds, starve finite work, lengthen a timeout to make a floor pass, modify transcript files during index migration, mark a migration version complete before its captured scope is verified, or relabel ordinary backlog as migration to evade normal-operation floors.

## Decisions

- 2026-09-10 — Performance is a core Recall feature and a standing product pillar. DuckDB performance work includes evaluating current engine/storage capabilities and simplifying or refactoring affected code so obsolete workarounds can be retired. The user supports a broader budget than the previous eight-iteration proposal; the expanded execution scope and numeric cap are to be made concrete before resuming. Existing correctness, recovery, and resource floors remain in force. **ratified (human)**

- 2026-09-08 — Thresholds above are provisional driver choices awaiting measurement. **provisional**

- 2026-09-09 — The default DuckDB buffer allowance is 2GB, independent of host RAM; explicit overrides remain supported. Populated-corpus recovery measures total RSS because buffer limits exclude other process allocations. **provisional**

- 2026-09-09 — Current supported transcript files are authoritative during recovery. Preserve the prior index snapshot and the recorded discrepancy record; only those documented historical differences may be accepted by the recovery verifier. Omissions of current supported content, incomplete captures, and missing sources remain failures or unresolved states, and equivalent derived inputs remain preserved. **ratified (human)**

- 2026-09-09 — Versioned historical index migration is separate from normal daemon reconciliation. Migration updates the derived index from current transcript files and does not modify those files. Eligibility is a durable index/parser migration version. The five-second writer ceiling and concurrent active-freshness gate do not apply inside that explicit maintenance phase; correctness, equivalent enrichment, current keyword membership, truthful coverage, and the 4GiB RSS ceiling remain. Normal-operation floors stay in force outside that phase. This supersedes the combined recovery/performance gate and the storage-format-first proposal; that hypothesis remains evidence, not a prerequisite. **ratified (human)**

- 2026-09-09 — Overall index-migration runtime budget is 4 hours (14400s) for the reference corpus. **provisional**

- 2026-09-09 — `live --fresh` catches up already-indexed behind rows only and stays inside the 2s responsive-read floor. First index stays on the fair coordinator. `show --fresh` remains the one-session write bounded by `live.fresh_timeout`. Do not lengthen a timeout to pass a floor. **ratified (human)**

- 2026-09-19 — On a quiet host (no raw commit for 120 s, no active agent session), the daemon averages <=5% of one core over 10 minutes, and no periodic operation costs more than 1 CPU-s per occurrence. The per-occurrence bound defines no periodic burst: it excludes main's harvest tick (about 8 CPU-s every 38 s) and admits the walk (about 0.5 CPU-s every 31 s). **ratified (human)**
- 2026-09-19 — The predeclared 33 MiB append floor is <=1.0 CPU-second median, >=5x below the retained 5.9 CPU-second baseline, <=3 * small median + 0.25 CPU-second, and <=5 seconds for every writer turn. **ratified (human)**

## Boundary

A rollout proceeds only after every non-excepted gate passes against the same candidate and a new consistent backup exists. Live secrets remain human boundary material.
