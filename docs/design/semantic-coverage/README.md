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
