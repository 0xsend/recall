# Session identifiers

People and agents copy a session identifier from a harness or recall output,
then use it to inspect history, search within it, filter a roster, follow
updates, or mark its writer. The identifier must name the same indexed session
in each operation.

## States and flow

The caller supplies an exact recall ID, an exact source session ID, or an
existing supported lowercase hex recall ID prefix (6–31 characters). Exact
recall identity takes precedence, followed by source identity, then prefix.
Source and host filters narrow identity lookup before ambiguity is decided.
A unique match supplies the canonical ID to read filters. Multiple matches
produce an error naming the candidates' recall IDs, sources, and hosts; the
caller retries with a canonical ID. An unknown indexed identifier produces an
explicit error saying unindexed sources were not searched. Resolution does
not discover transcripts or load messages.

A live mark can precede indexing. An indexed match supplies its stored source
ID and source; an unknown ID continues to mean a source ID, under the supplied
source or the existing Claude Code default, on the local host. Ambiguity is an
error and never writes a mark. A session without a source ID cannot be marked.
Read filters do not widen to other sessions when no rows satisfy time,
project, source, host, or liveness restrictions. Fleet session filters retain
the local-only restriction already used by search.

## As-built baseline and decision

At 35bcb24, show and follow accept both IDs and prefixes; search only compares
to the recall ID, live mark only accepts the source ID, and list/live roster
have no session selector. On a copied populated v0.39.1 index, show with the
source ID succeeds while the same source-ID search returns no hits and its
canonical-ID equivalent does return hits. Captures live in the task evidence,
outside the repository.

2026-10-22: The human chose “Accept both IDs (Recommended)” in the coordinator's
question widget. A shared indexed resolver and additive list/live selectors
implement that contract. Requiring canonical IDs everywhere was rejected by
that decision. Pre-index marks retain their existing behavior because harness
SessionStart hooks can run before transcript creation. Output envelopes,
cursors, and liveness rules stay with their existing contracts.
