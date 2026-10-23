# Semantic coverage

People and agents use daemon status to decide whether indexed history supports
semantic retrieval. Ordinary enrichment recovers every required content,
thinking and bash field; status exposes a fresh whole-history census separately
from discovery scope and operational embedding enablement.

## Flow and states

The caller reads `daemon status --json` and inspects
`reconciliation.semantic_coverage`. Complete coverage allows semantic retrieval
once raw/marker readiness also holds. Incomplete coverage keeps enabled enrichment
unready: eligible or held work waits for recovery/idle release; cooled work waits
for bounded retry; generation/capacity errors identify the affected targets;
blocked targets require raw-state repair through its owning path. Unreadable or
locally owner-blocked status returns unknown counts and an explicit error.
Disabled embeddings retain operational readiness while showing actual coverage.
Empty databases have complete semantic coverage, subject to ordinary readiness.

Text presents the same state beside embedding diagnostics:

```
semantic: complete content=0 thinking=0 bash=0 blocked=0
semantic: incomplete content=84 thinking=1118 bash=0 blocked=0
semantic: unknown content=? thinking=? bash=? blocked=?
```

These are illustrative states, not measured observations. Structured consumers
use the typed object rather than parsing text. The prior surface exposed only
eligible/held pending row counts and could declare readiness with processed NULL
fields; the retained lane baseline captures that behavior.

## Decisions

- 2026-10-22: Human approved plan revision 1 ("#117 re-embed skipped rows") via
  the coordinator question widget. Approval includes additive coverage, expanded
  pending semantics and stricter enabled readiness. SPEC REQ-ADAPT-003/019/022
  owns the standing contract.
- Automatic bounded field recovery was selected over diagnostic-only reporting
  plus explicit maintenance. No schema change, repair command, model change,
  transcript reparse or production repair is authorized by this change.
- Existing companion context stays fixed; explicit inference/publication masks
  preserve valid sibling vectors. Explicit context recomputation retains its
  replacement policy. Unknown/unselectable history never reports completeness.

## Transactional work metadata

On 2026-10-23 the human approved revision 2 and its rollback/repair amendments
through the coordinator question widget. This supersedes the revision-1
no-schema/no-repair decision above. [Semantic work](../../semantic-work.md)
owns the initialization, status, capacity, repair and rollback contract.

The caller still reads the existing semantic coverage object. Unknown now also
covers backup, build, verification, restart audit and held invalid metadata.
These states expose a bounded semantic-work diagnostic line and the additive
structured operation fields. Ordinary writes are refused during the fence;
status remains available. A complete relation with vector gaps reports exact
incomplete coverage and allows normal enrichment. A verified empty relation
reports complete coverage. Disabled embedding reports actual coverage beside
its existing operational readiness. Invalid metadata directs the operator to
inspect a dry-run repair plan, accept its digest, then observe completion.

Durable narrow work and session counters replace the full-history read-time
census. They were selected because the populated read-time census missed the
status floor even in the earlier storage integration experiment. A TTL or cached
zero would make readiness stale; neither is part of this design. Payload and
work/counters have one transaction owner. The exact schema and supported
rollback/repair boundary are approved, while production actions remain per-host
decisions. The examples above remain illustrative, not qualification evidence.
