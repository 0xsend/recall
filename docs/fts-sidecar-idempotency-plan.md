# FTS no-op suppression plan — revision 1, awaiting approval

Issue: [#79](https://github.com/0xsend/recall/issues/79). Base: `4637023d35b290ac64d7f40ea37a8d87b4aaf99b`.

## Outcome and boundary

An unchanged effective keyword document causes no FTS write, including after a process restart or replay of a durable pending upsert. A changed document replaces the previous document once while retaining its mapping rowid. Fingerprint and FTS publication succeed or roll back together. This proposal changes the SQLite sidecar schema; implementation requires approval under common-brief.md, Boundaries. It does not change DuckDB schema, CLI/RPC contracts, host configuration, versions or existing corpus totals. Historical repair remains separately approved, plan-only work in [the repair plan](fts-sidecar-repair-plan.md).

## Production behavior and evidence

The message and tool batch upserts in `recall/db/fts_sidecar.py` use contentless FTS5 `INSERT OR REPLACE`. Bootstrap and field rescoping also contain independent direct FTS writes in `services/fts_sidecar_bootstrap.py` and `services/fts_sidecar_reconcile.py`. Pending repair delegates to the ordinary upsert functions. New fingerprints must cover all those paths; fixing only the ordinary upserts would leave an untracked writer.

A real sidecar at the base, using Python 3.12.3 / SQLite 3.45.1, retained 20 live documents after 20 identical retries while its averages record changed from `143B00` to `286300` and the needle score changed from 2.9541297834827644 to 3.5495166208042006. A populated fixture daemon full reparse did not reproduce drift: its existing change detection suppressed unchanged publication. Full-reparse success alone therefore cannot establish retry safety.

## Proposed construction

1. Add a nullable `content_fingerprint` column to each mapping table (`message_fts_rowid`, `tool_calls_fts_rowid`) through an explicitly versioned, transactional sidecar schema upgrade. Fresh sidecars create these columns directly. The proposed entry point is `open_sidecar`: atomically add the two nullable columns and set `PRAGMA user_version` to the fingerprint schema version before allowing publication. This additive upgrade leaves every existing FTS row untouched. The deployment owner retains a consistent backup before starting this version on an existing sidecar. No ordinary open discards, rebuilds or backfills an existing sidecar. Approval of this revision must explicitly include this automatic additive upgrade; it does not authorize the separate historical repair.
2. Give one transaction-owning publication operation responsibility for mapping allocation, field scoping, content comparison, FTS modification and fingerprint publication. Public single-row and batch functions, bootstrap, reconciliation, pending replay and field rescoping all use that operation. Batch/progress commits remain bounded and retain their current atomic boundaries; nesting an independently committing helper inside bootstrap is prohibited.
3. Fingerprint exactly the effective, ordered FTS columns after field scoping. Use SHA-256 over a format/version tag plus length-prefixed UTF-8 columns, distinguishing empty columns and column boundaries. Do not include raw excluded fields, unrelated message metadata or field-list ordering. Record the hash format with the schema version.
4. Acquire SQLite writer ownership before comparing durable fingerprints. On a match, skip both the FTS and fingerprint update. On a mismatch, explicitly delete the old FTS row and insert its replacement inside the same transaction, then publish the new fingerprint. Avoid `INSERT OR REPLACE`: it also affects changed-row statistics, and relying only on no-op suppression leaves that path exposed. Keep the mapping rowid stable.
5. Existing rows with NULL fingerprints are unknown, not equal to the current DuckDB text. A pending sidecar may legitimately differ from DuckDB. On first touch, explicitly delete/insert that document and fingerprint it atomically; this preserves the current document contribution instead of adding another replacement contribution. Prove this transition with SQLite experiments and include it in the migration rehearsal. This transition cannot repair already-drifted totals. If approval requires literally zero physical writes for unchanged legacy rows, canonical reconstruction plus fingerprint publication is required instead, and must use the separate approved repair procedure.
6. Deletion removes mapping/fingerprint and FTS membership atomically. Preserve ordered duplicates, NULL tool deletion, disabled bash scoping, delete/reinsert and batch-boundary semantics. A failed SQLite transaction must never leave a fingerprint claiming an FTS write committed. A successful SQLite publication followed by a failed DuckDB acknowledgement must replay as a no-op.

## Alternatives

- Caller-side DuckDB comparison alone is insufficient: after SQLite commits but DuckDB acknowledgement fails, the caller cannot tell whether publication succeeded. Durable proof must live in the same transactional store as the FTS document.
- An in-memory fingerprint cache loses that proof at restart.
- Putting per-document fingerprints into `fts_sidecar_meta` avoids DDL text but still creates a persistent per-document storage contract and migration problem. It is not a reason to bypass schema approval.
- Repeated replacement plus periodic optimize is disconfirmed by the issue evidence. SQLite documents that contentless tables do not support rebuild. Shadow-table writes are excluded.
- Deleting and reinserting on every request can preserve statistics but fails the required unchanged-write no-op outcome and repeats indexing work.

## Verification after approval

The smallest regression uses real SQLite and the existing sidecar API: seed a declared corpus, take totals and BM25, replay identical single/batch writes, close/reopen, then assert exact totals/scores. Test message content/thinking and tool bash, excluded fields, field rescoping, duplicates in/across batches, empty strings and deletion/reinsert. Compare a real changed row with a clean index of the final declared corpus, including statistics, membership and scores.

Use a real SQLite subprocess killed before commit and after publication/before DuckDB acknowledgement; resume from durable pending rows and compare to a clean single pass. Exercise legacy NULL fingerprints, migration failure/rollback and reopening under the prior binary. On binary rollback, restore the retained sidecar against the same DuckDB generation before permitting the old writer to run; otherwise an old writer can make fingerprints stale. After further writes, use an approved compatible reconstruction or matched full-store restoration rather than silently reusing fingerprints. Rehearse all schema risk scenarios on a fresh copy of the populated fixture, recording environment, candidate head, backup hashes and gaps. No production database is involved.

Run affected regressions, `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`, and both explicit lefthook hooks. Author-drive the scratch daemon on the populated fixture through keyword searches, restart, unchanged full reparse and killed/resumed indexing; compare both result identities and scores. Invite a fresh guest only at the resulting build, with the required process and service boundaries. A matching specialist review and approval/rehearsal evidence are required before the schema change can be offered for landing.

## Approval question

1. Approve implementing and rehearsing this sidecar schema change (recommended), or choose canonical reconstruction as the required legacy transition? This question authorizes neither production mutation nor historical totals repair. The approval must explicitly cover the additive `open_sidecar` upgrade, backup prerequisite and legacy first-touch behavior. Until then, implementation remains blocked.

## Reference

[SQLite FTS5 documentation](https://sqlite.org/fts5.html#contentless_delete_tables) documents DELETE support for contentless-delete tables. Its [averages record description](https://sqlite.org/fts5.html#averages_record_format) identifies the row and token totals. Its [rebuild documentation](https://sqlite.org/fts5.html#the_rebuild_command) excludes contentless tables. The observed replacement drift is established by this lane's experiment, rather than inferred from those documents.
