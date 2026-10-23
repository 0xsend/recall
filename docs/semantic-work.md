# Semantic work initialization and repair

Schema 33 adds `semantic_work`, grouped `semantic_work_counts`, a singleton
`semantic_work_state` and the `semantic_pending` view. The payload tables remain
authoritative. Ordinary enrichment repairs nonempty content, thinking and bash
fields whose vectors are NULL, including existing processed companions. Empty
fields need a companion marker once. Valid sibling vectors and stored context
survive conditional recovery.

The lifecycle owns one shared writer handle. Raw insert, append, rewrite, move,
delete, supplied vectors and detached enrichment publish their final work masks
and grouped counters in the same transaction as their payloads. A stale detached
input is rejected; an already published valid vector wins. Read snapshots require
verified work metadata. Status and eligibility read counters and narrow session
metadata, without scanning historical text or vectors.

## Inspect

Run `recall daemon status --json` against the selected index. The additive
`reconciliation.semantic_work` object exposes `definition_version`, `phase`,
`build_id`, `completed_targets`, `captured_messages`, `captured_tools`,
`started_at`, `verified_at` and `error`. Timestamps use epoch seconds or null.
`semantic_coverage` and fresh eligible/held counts are unknown until verified.
The initialization phases are backup, building, verifying, auditing, complete
and failed. Schema installation alone does not establish completeness.

An approved daemon owner initializes an existing schema-32 index automatically.
Before DDL it fences ordinary writes, settles the DuckDB/SQLite pair, retains and
verifies a consistent `snapshots/recreate-semantic-work-*` preimage, and records
a durable intent. It publishes at most 8,192 target identities per initialization
chunk with counters and cursor together. Restart resumes the same build and scope.
Bidirectional key/mask comparison and grouped counter parity precede admission;
each ownership incarnation audits before trusting completed metadata. A fresh
empty index creates a verified empty relation without a fabricated backup.

During a hold, status remains reachable and ordinary writers receive APP_LOCKED
with `semantic_work_initializing` or `semantic_work_invalid`, the build id and
an action to inspect status. Pause persists through initialization. Enabling
embeddings requires full semantic coverage for enrichment readiness; disabling
embeddings preserves the operational convention while reporting actual gaps.
Local status inspection neither migrates an index nor infers zero from an
unverified or unreadable relation.

## Repair

A drifted or invalid schema-33 relation waits for explicit metadata repair:

```bash
recall daemon repair-semantic-work --dry-run --json
recall daemon repair-semantic-work --plan-id <digest> --json
recall daemon status --json --fields reconciliation.semantic_work
```

The first form is the default. The command requires a running approved owner;
it neither auto-starts a daemon nor opens a local write handle. `--dry-run` and
`--plan-id` are mutually exclusive. The preview identifies the target, schema,
definition, owner generation, failed and proposed builds, mutation sequence,
retained preimages, disk reserve, metadata to replace and available rollback.
A changed owner generation or target conflicts without mutation. Repeated apply
of an accepted digest returns its existing build, including after restart.
Acceptance is not evidence of completion: inspect the reported phase and error.

Apply first retains a verified preimage of the current payload and links the
predecessor intent. It replaces only derived semantic objects. It never restores
older raw history, removes vectors to improve counts, or manufactures a missing
schema-32 backup. Interrupted repair resumes its own scope/cursor/preimage.
Canonical corruption, unsupported newer schema, active keyword/source/storage
maintenance and exceeded capacity refuse toward the owning operation or human
recovery. Space/cancellation failures remain unknown with their diagnostic;
restart does not automatically repair demonstrated drift.

The limits are 100 million pending targets, one million pending sessions, 256
identities per normal metadata publication chunk, and 32 retained preimages per
index, including superseded unpublished captures. Full-pair, work/WAL/spill and
repair reserve must fit on each relevant filesystem. Capacity refuses visibly;
it never drops work. Retirement of retained data requires a human decision.

## Rollback

The schema-32 fallback is exact build
`5fa253be197fbd57b7c1299f2e90fac7fa26d434`, paired with the prepared keyword-schema-2
preimage. Both DuckDB and SQLite restore together to a separate staged copy.
There is no supported binary-only downgrade on schema 33 or in-place version
decrement. A schema-33 repair preimage supports the current schema-33 binary;
it cannot replace a lost pre-0033 rollback. Restoring an old pair over later raw
or embedding progress requires an explicit restore/loss decision.

Before DDL, a stale or missing unpublished preimage is retained and linked to a
fresh capture of the current schema-32 pair. An unchanged restart reuses its
original capture. If backup or reserve fails at that point, the unchanged-schema
current pair remains available to the pinned fallback, subject to its keyword
initialization state. Production initialization and repair need a separate go
per host. Schema initialization, audit and repair do not waive any daemon
responsiveness, freshness, writer, fairness or whole-process memory floor.
