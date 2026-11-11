# Offline recovery

The operator repairs an index after an interrupted or refused automatic attempt. People and automation use the existing `db rebuild-indexes` command and the confirmed `compact` command. Structured results keep their existing fields. The accepted recovery specification governs these commands; a stored phase alone never grants execution or release.

The prior rebuild command recreated discoverable indexes and cleared failure markers without an independent content comparison. An interrupted DROP could hide a catalog-only target from that command. A successful return did not prove that recovery state was released or acknowledged content survived. The implementation replaces that path with explicit offline ownership, a saved complete target manifest, a pre-attempt exact reference, and a capability issued only after the new running attempt commits.

## States and flow

The command first owns the offline database and advisory lock, then reads the complete bounded event namespace and refusal state. Unreadable, malformed, conflicting or full event storage holds without index mutation. An active daemon denies ownership. An unavailable exact reference requires a matching operator-supplied reference or matched restore.

Required and running work retains its saved reference. An explicit new manual selection may capture a fresh settled reference after a completed attempt. Manual arm starts only from required or running with a usable counter and consistent budget. It commits running and spent flags before any index DDL or database replacement. Another process cannot resume that execution capability.

After rebuilding the complete saved target catalog, the owner closes and freshly reopens the database. Exact canonical content and target checks precede keyword publication. Publication uses bounded native work and actual keyword verification; only the acknowledged intent membership can retire. The final transaction records verified manual completion, its matching publication certificate and success flags together. It never fabricates a successful daemon run.

The release removes only matching acknowledged events, then removes refusal last with directory synchronization. A later event prevents release. Failed retirement retains or re-establishes refusal and reports failure. A verified release retry validates the certificate and current keyword generation rather than comparing correctly retired intents with their earlier contents.

## Decisions

The existing command names and result shapes remain. A failed check reports its operation and the next supported action; it never silently falls back to recreation. Manual rebuild is the named index remedy. Confirmed manual compaction shares the same verification and release authority; the automatic compaction primitive preserves recovery state and has no release authority.

The post-publication reference contains the permitted retirement and actual keyword generation within the pinned pre-attempt oracle directory. It does not consume a second complete-oracle pin, so the next complete capture has a replacement slot. Attempt-scoped working and backup paths preserve prior failed recovery sets.

Native keyword publication and manual compaction build an attempt-scoped database candidate. The source commits running before clone or COPY; both source and candidate remain running through installation. Fresh candidate checks use the independent saved canonical reference, unchanged duplicate intent membership and complete actual target catalog. A prepared publication binding records the exact permitted native keyword generation without granting retirement or release. Checkpoint, close and fresh verification precede a single atomic installation under the same offline owner. A fresh installed parent then captures that incarnation's intent rowids and commits retirement, verified history, certificate and success flags together. Failed construction, replacement or pre-commit interruption retains running and refusal. Retained original backups and failed candidates are never silently reused or removed. Exact event membership and owner admission are checked again at installation; files and refusal retire only after fresh installed-file checks.

A new manual attempt validates any prepared native generation against its saved exact publication streams while canonical, companion and unretired intent streams still match the original independent oracle. Before the new arm, copy that prior proof durably into the next recorded workspace. Commit its archived binding with the running instruction before reclaiming an unreferenced oracle-child publication. A candidate records its new prepared binding before installation. This preserves verifiable retry across the renamed-file window without recapturing damaged canonical content, excluding arbitrary metadata or inferring release from running. Matching external references may restore a missing publication child by its recorded hash/identity. Original matched sets include every bound proof.

The native publication reference covers the two managed schemas' known table shapes, exact rows and generated macro definitions, with backend and field scope. Other tables remain canonical. Custom objects in those namespaces prevent replacement. Canonical document IDs, field scope, token multiplicity, document frequencies, lengths and corpus statistics verify native membership; a generation digest alone is insufficient.

Read-only WAL diagnosis may be unavailable. Ordinary startup then holds. Explicit manual ownership may diagnose a copied database/WAL pair; that is distinct from granting ordinary startup permission to replay the held original read-write.

Baseline and changed command captures, process-crash evidence, final CLI rehearsals and usability review belong in the change evidence. Small functional checks establish behavior, not populated latency, memory, device durability or production recovery.
