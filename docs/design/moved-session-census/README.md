# Moved-session census

People and agents inventory historical duplicates on an owned backup copy using
`db supersede-moved`; explicit apply rehearses the proven pairs on that copy.
Production repair requires separate per-host authorization and reviewed scope.
The approved procedure is recorded in SPEC REQ-INDEX-027 and reconciliation
operations documentation.

## States and decisions

| Caller action | Result |
| --- | --- |
| Report an existing copy | Read-only census, pairs, host predicate and excluded count; phase `report`. |
| Report a missing database | Validation error; no database creation. |
| Apply without nonempty `RECALL_DATA_DIR` | Validation error before config or database access, including raw JSON params. |
| Apply an explicitly targeted copy | Action counts plus a census labeled `pre_apply`; report again for the post-image. |
| Repeat apply | Zero repairs; unchanged database files. |
| Inspect a mixed lagging/oversize group | Both retain reasons, zero eligible pairs; no prefix read beyond the cap. |
| Read a foreign-host copy | Total and excluded counts explain local zero results. |
| Another owner holds the database | Ownership error; separately authorize shutdown of that exact owner. |

Baseline at 7d7bdd3 uses unlabeled pre-apply results, one cap/lag reason and
implicit default targeting. Captures and before/after CLI receipts are retained
in the task record's `scratch/evidence/review-r2/` directory. The existing census
agreement is `design-census.md` in that record.

The coordinator's 2026-10-23 addendum 1 selects explicit-target refusal, safe
copy-only examples, overlapping cap/lag reasons and an unambiguous apply result.
Labeling the pre-image avoids repeating potentially racing proof reads after
mutation. The planner and classification already read prefixes separately;
these are two observations and can differ if transcripts change. Pair plans
drive deletion. This limitation remains disclosed, without changing proof or
transaction semantics. Conservative manifest safety flags cover optional apply;
the description identifies the read-only default. No new safety schema is added.
