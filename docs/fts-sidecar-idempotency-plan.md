# FTS no-op suppression plan — revision 2, approved and implemented

Issue: [#79](https://github.com/0xsend/recall/issues/79). Base: `4637023d35b290ac64d7f40ea37a8d87b4aaf99b`.

## Outcome and approval boundary

An unchanged effective keyword document causes no FTS write, including after process restart or replay of a durable pending upsert. A changed document publishes exactly once while retaining its mapping rowid. Fingerprint and FTS publication succeed or roll back together.

A durable fingerprint requires a sidecar schema change. Existing contentless rows cannot supply their original text for fingerprinting; treating current DuckDB text as the published content is unsafe when a pending update has not reached SQLite. Deleting and reinserting unknown rows also grows totals on the tested runtime. The complete design therefore requires canonical initialization of existing sidecars as well as schema evolution. The human approved revision 2 implementation and fixture/copied-backup rehearsals on 2026-10-23 (PR #141, coordinator addendum 1). The implemented command is `python -m recall.services.fts_sidecar_rebuild`; its [operator guide](keyword-initialization.md) describes initialization and recovery. Production execution still requires a separate per-host grant.

## Production behavior and observations

Message and tool batch upserts in `recall/db/fts_sidecar.py` use contentless FTS5 `INSERT OR REPLACE`. Bootstrap and field rescoping also contain independent direct FTS writes in `services/fts_sidecar_bootstrap.py` and `services/fts_sidecar_reconcile.py`. Pending repair delegates to the ordinary upsert functions. The durable identity must cover every writer.

At the base with Python 3.12.3 / SQLite 3.45.1, 20 identical retries kept 20 live documents but changed the averages record from `143B00` to `286300` and the needle score from 2.9541297834827644 to 3.5495166208042006. A populated fixture daemon full reparse did not drift on the exercised path, because existing change detection suppressed publication. Startup repair of two seeded durable pending upserts did drift for unchanged message and bash documents, and RPC searches exposed the changed scores.

A separate real SQLite probe made 20 explicit delete/insert cycles on the same rowid. It produced the same totals and score drift as replacement. Explicit deletion is not a verified legacy transition or statistics-preserving update mechanism. This constrains initialization rather than weakening the unchanged-write requirement.

## Proposed construction after approval

1. Fresh sidecars store `content_fingerprint TEXT NOT NULL` in each mapping table (`message_fts_rowid`, `tool_calls_fts_rowid`) and a fingerprint-format/schema version. There is no unknown-fingerprint writer state. Existing schema inspection identifies the required explicit initialization; an ordinary open does not silently recreate or backfill a sidecar.
2. Initialize an existing sidecar through the separately approved [canonical reconstruction procedure](fts-sidecar-repair-plan.md): freeze canonical DuckDB, retain a consistent matched backup, create a new sidecar with the fingerprint schema, publish every effective document once, verify it, atomically install it and retain rollback. The implementation/entry point for this maintenance must be approved before it is built. No old-schema fallback writer or first-touch replacement substitutes for initialization.
3. One transaction-owning publication operation scopes content, computes the fingerprint, acquires SQLite writer ownership, resolves the mapping and decides whether to publish. On a fingerprint match, it skips all FTS/fingerprint writes. On a change, it performs the one required document replacement and publishes its fingerprint in the same transaction. A new mapping is created with its fingerprint in that transaction. Genuine content changes retain replacement semantics; this plan promises suppression of unchanged retries, not repair of arbitrary past edit-history statistics.
4. Fingerprint exactly the ordered effective FTS columns after field scoping. Use SHA-256 over a version tag and length-prefixed UTF-8 columns, distinguishing empty columns and column boundaries. Excluded raw fields, unrelated metadata and field-list ordering do not change it.
5. All single/batch upserts, bootstrap, reconciliation, pending replay and field rescoping use that transaction owner. Keep current batch/progress bounds and atomic boundaries; independently committing nested helpers must not split progress from publication. Deletion removes mapping/fingerprint and FTS membership atomically. Preserve ordered duplicates, NULL tool deletion, disabled bash fields, delete/reinsert and batch boundaries.
6. A SQLite failure cannot leave a fingerprint claiming publication succeeded. A successful SQLite publication followed by a failed DuckDB acknowledgement replays as a no-op. Initialization, schema identity and fingerprint format are checked before normal publication begins. Binary rollback restores an approved compatible sidecar state: an older writer must not silently invalidate fingerprints that a newer writer later trusts.

## Alternatives ruled out

- Caller-side DuckDB comparison cannot identify an already-committed sidecar publication whose acknowledgement was lost. In-memory fingerprints lose that knowledge at restart.
- Per-document keys in `fts_sidecar_meta` still create a persistent per-document storage contract and initialization problem. Avoiding DDL text does not avoid the approval boundary.
- Seeding legacy fingerprints from current DuckDB without publishing/verifying that content can skip an outstanding update and preserve stale keyword membership.
- Replacing unknown legacy rows once still changes an unchanged corpus's totals; explicit delete/insert was experimentally disconfirmed as a workaround. Neither satisfies the universal unchanged-write requirement.
- Periodic optimize is disconfirmed by the issue evidence. Rebuild is unavailable for contentless tables. Direct shadow-table edits are excluded.

## Verifiers and risk scenarios

The smallest regression uses the real sidecar API: seed a declared corpus; compare totals/BM25 before and after identical single/batch writes, duplicates, field rescoping and close/reopen. Cover message content/thinking, bash, excluded fields, empty strings and deletion/reinsert. A real changed row must update membership and then stop writing on identical retries. Compare a retried run with a single-publication run of the same declared change history; do not assume different genuine edit histories have canonical statistics on this SQLite runtime.

Kill a real SQLite subprocess before commit and after publication/before DuckDB acknowledgement, then resume durable pending work and compare with the clean single pass. Rehearse schema/canonical initialization, partial-build resume, atomic install, rollback and prior-binary compatibility on a fresh populated fixture copy. Record head, runtime, manifest hashes, every scenario and gaps. No production database is involved.

Run affected regressions, the full pytest/ruff/format/ty gates and both explicit lefthook hooks. Author-drive the candidate scratch daemon through keyword searches, restart, unchanged full reparse and killed/resumed indexing; compare identities and scores. Then invite a fresh guest at that build. The schema/data change additionally needs the approved plan, matching specialist review and author risk rehearsal before landing.

## Approval question

1. Approve implementation of fingerprint publication plus explicit canonical initialization for legacy sidecars, including lifting the plan-only restriction for that implementation and its fixture rehearsal (recommended), or keep repair plan-only and defer implementation? Production rollout and production repair retain separate host/artifact-specific authority. The unchanged-write requirement is not waived. Until this decision, the work is blocked; a partial legacy guarantee is not an accepted alternative.

## Reference

[SQLite FTS5 documentation](https://sqlite.org/fts5.html#contentless_delete_tables) documents contentless-delete operations. The [averages record](https://sqlite.org/fts5.html#averages_record_format) contains document/token totals, and [rebuild](https://sqlite.org/fts5.html#the_rebuild_command) is unavailable for contentless tables. The replacement and delete/insert drift statements above come from this lane's experiments, not from those documentation claims.
