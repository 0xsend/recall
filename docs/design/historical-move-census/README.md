# Historical move census

Operators and agents need to inventory historical duplicate candidates before
requesting any production repair. The existing `db supersede-moved` maintenance
flow supplies report, explicit apply and subsequent inspection. The selected
2026-10-22 human direction was a report-only census on backup copies and fixture
rehearsal first; every production host requires separate authority.

The prior default reported only proven path pairs, opened read-write and could
migrate an older schema. The current default opens an existing database read-only
without migration. The original `moves`, `applied` and `superseded` fields remain;
`census` adds candidate classes and `pairs` adds proofs and affected rows. Text
keeps the original path arrows and adds overlapping classification counts.

| State | Operator observation and next action |
| --- | --- |
| No candidate | Zero pairs; retained history is unchanged. |
| Unique proven survivor | Exact ids, paths, committed proofs and affected rows; rehearse explicit apply on the copy. |
| Unproven or ambiguous | Named classification, rows retained; collect evidence or request a separate decision. |
| Missing database / unsupported schema | Error exit; select an existing compatible copy. No implicit creation or upgrade. |
| Owner conflict | Error exit; arrange separately authorized maintenance ownership. |
| Interrupted apply | Current pair rolls back; committed pairs persist; repeat report and resume on the copy. |
| Repeated apply | No remaining proven pairs means no changes. |

Candidate grouping deliberately stays separate from identity proof: timestamps
and repeated declared ids can collide across subagents and forks. Existing move
limits and committed-prefix rules remain authoritative. A new heuristic repair,
a wider candidate limit and stable logical ids were rejected as outside the
approved repair procedure. A separate command or flag adds no value to this
existing flow, so census information accompanies its ordinary report.

[Reconciliation operations](../../reconciliation-operations.md#historical-moved-session-census-and-rehearsal)
owns the field and classification definitions. A production grant must name the
host, executable revision, backup and pairs; the current operation does not bind
apply to a selected-pair manifest. That capability and its approval remain a
separate prerequisite for restricted-scope production repair.
