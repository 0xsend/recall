# Canonical FTS sidecar repair — approved implementation contract

Issue: [#79](https://github.com/0xsend/recall/issues/79). The human approved implementation and fixture/copied-backup rehearsals on 2026-10-23 (PR #141 revision 2, coordinator addendum 1). The explicit command is `python -m recall.services.fts_sidecar_rebuild`; its [operator guide](keyword-initialization.md) describes initialization and recovery. Production execution still requires a separate per-host grant.

## Goal and prerequisites

The repair produces a fresh sidecar from one frozen canonical DuckDB snapshot. Every eligible message and bash document is inserted exactly once, with the configured field scope, so its totals reflect that corpus rather than its retry history. DuckDB conversation, context and vector data are preserved. Rowids may be allocated in deterministic entity-id order; public entity ids, search membership, payloads and tie ordering remain stable. Corrected BM25 scores may legitimately change rankings relative to the drifted file.

Execution depends on an approved, reviewed no-op implementation at a named head. Otherwise retries can immediately reintroduce drift. Before production execution, the operator must approve the exact host, data directory, backup artifact, candidate head, maintenance window, rollback retention and rehearsal gaps. The implementation and fixture/copied-backup rehearsal grant does not authorize repairing a real host.

## Construction and validation

1. The existing connection owner enters maintenance, drains active reads/writes and pending keyword publications, and stops all index/enrichment/repair writers. It records the owned daemon PID and confirms its exit. A durable maintenance marker prevents an unattended read client from auto-starting a writer. Installation/service operations are excluded from a scratch rehearsal.
2. Use the repository's consistent backup mechanism to retain a matched DuckDB and SQLite sidecar snapshot, including any WAL contents. Record hashes, bytes, SQLite/DuckDB/Python versions, schema and index-migration identity, FTS field signature, source scope and pending queue. Open neither a rollback backup nor a live database from this lane. Work only on a new copy. Require enough disk for the matched backup, old sidecar, candidate and temporary SQLite journal; insufficient space fails before construction.
3. Hold DuckDB at the captured generation throughout export. Create a distinct candidate SQLite file on the same filesystem as the eventual sidecar. Do not reuse an old candidate, its bootstrap markers or its mappings. Read messages in bounded entity-id order from `message_state`: `message_id`, `COALESCE(fts_content, '')`, `COALESCE(fts_thinking, '')`. Read tools in bounded id order from `tool_calls` where `bash_command IS NOT NULL`. Apply the captured configured fields before publication. Exclude tools when bash is disabled; preserve message rows even when effective columns are empty. No new contextualization, parsing, embedding or host usage harvesting occurs.
4. Use the approved transaction-owning sidecar writer for every candidate row, so its content fingerprint and FTS entry commit together. Fresh candidate construction inserts each unique entity exactly once. Persist progress only with its batch commit. On interruption, resume only the same snapshot, field signature and build; a mismatched identity fails rather than mixing generations. A fresh attempt gets a fresh candidate path. Do not modify FTS shadow tables.
5. Verify the complete sorted identity sets against the captured DuckDB selection, including empty columns, NULL tools, disabled fields and duplicate rejection. Validate effective content using the candidate's fingerprints and a declared hash format; membership alone cannot establish correct content. Check both FTS integrity commands and SQLite integrity. Compare the FTS averages row/token totals to the sum of live per-document sizes read for verification, and to an independently declared small-corpus oracle in the rehearsal. Reading shadow data for evidence is allowed; writing it is excluded.
6. Open a fresh read connection and exercise fixed message/content/thinking/bash, phrase, prefix and filtered queries. Compare result identities/payloads to canonical membership expectations. Build a second clean reference from the same snapshot and compare totals and scores. Replay all unchanged rows, including a deliberately interrupted pending acknowledgement, and require exact unchanged totals and BM25. Restart and re-read. Historical drifted scores are recorded for diagnosis, not used as the corrected-score oracle.

## Swap and crash behavior

The daemon remains stopped during installation. Close all candidate SQLite connections, checkpoint any WAL and ensure the standalone candidate file contains all committed rows. Close old sidecar readers and retain the complete prior file as a hashed rollback artifact. Never replace a main SQLite file while an old WAL or shared-memory file can attach to it. Persist the candidate and a manifest binding it to the captured DuckDB snapshot, then atomically replace the current sidecar path with the candidate and sync the containing directory. The retained backup is separate; installation must not create a two-rename window with no current index.

A crash before replacement leaves the original active file and resumable candidate. A crash after replacement leaves the complete candidate and original rollback artifact. A manifest and observed file hash decide which state exists; an incomplete candidate is never installed. Startup verifies the expected installed hash and generation before clearing maintenance.

Restart the approved binary with writes still held, verify stats and representative keyword identities/scores through RPC, then resume normal work and verify append/currentness plus keyword progress. Catch up transcripts changed after capture through ordinary reconciliation. Capture final pending=0, generation errors=0, both stores' integrity and unchanged DuckDB enrichment identities. A status socket alone is not success.

## Rollback

Before writers resume, rollback restores the retained original sidecar atomically while the daemon remains stopped and DuckDB still matches its captured generation. Verify its hash, reopen with the prior approved binary and check its known baseline queries. Its original statistics drift returns; rollback restores availability and prior behavior rather than claiming repaired scores.

After writers resume, an old sidecar no longer necessarily matches current DuckDB. Restoring just that file is prohibited. The operator chooses an approved compatible reconstruction from current DuckDB, or a matched full-store backup restoration with an explicit account of post-backup derived data and source replay. A binary rollback must not reuse fingerprints that an older binary can silently make stale. Never discard current DuckDB, pending publications or newly generated embeddings implicitly. Backups and failed candidates remain retained until explicit data-deletion authority exists.

An operation may be abandoned with `abort` whenever the live sidecar still has its captured original hash, including during construction or after canonical hash/WAL validation fails. This leaves that file in place and releases maintenance; it restores nothing. `rollback` also accepts a partial build while the captured canonical generation still matches. If the canonical generation or installed sidecar has changed, `replace --previous-operation-dir` transfers maintenance directly to a fresh operation directory naming the current owner, captures the current pair, and reconstructs from current DuckDB. It never releases writers between operations or restores an older sidecar over a newer canonical generation. Previous manifests, backups and candidates remain retained.

## Fixture rehearsal required before production approval

Before daemon startup/status, fixture preparation remaps inherited absolute backup references to copied scratch artifacts or an explicitly missing scratch target; status must never stat a production path from copied metadata. The rehearsal uses a fresh copy of the rollout fixture, all five source roots pinned to scratch, explicit scratch config/data dir, the tested build first on PATH, and an empty injected Grok usage log. It never uses scratch `recall daemon stop`, `restart`, `install` or `uninstall`; only owned recorded PIDs/groups are signalled. Every daemon call validates the expected PID and calls stats to confirm isolation.

Record candidate head, fixture/manifest hashes, runtime/config, disk requirements and every scenario below. The reference measurement runner is required for reported CPU/RSS/latency comparisons; functional-only runs make no performance claim.

| Scenario | Required observation |
| --- | --- |
| Clean populated construction | Complete canonical identity sets, content fingerprints, live-document totals and reference scores agree. |
| Declared small corpus | Exact independently specified row/token counts and keyword membership match, including empty/NULL/excluded-field cases. |
| Drifted input | The original is unchanged; the new index agrees with the clean canonical oracle, with before/after scores retained. |
| Kill during a candidate batch | Only whole batches persist; resume at the captured snapshot yields exactly the clean reference totals/scores. |
| Kill after sidecar publication, before pending acknowledgement | Replay leaves totals/scores unchanged with pending drained. |
| Failure before swap (including disk exhaustion) | Original file stays readable and its hash is unchanged; no completion claim is emitted. |
| Kill immediately before/after atomic replacement | Hash/manifest identify exactly the old or complete candidate file; no partial index is installed. |
| Rollback before resume | Exact original sidecar is restored against the same DuckDB snapshot and serves its baseline queries. |
| Restart, full reparse and append | Stable identities/scores for unchanged rows, correct changed membership and completed pending work. |
| Unrelated history/enrichment | Sorted canonical content, context and vector identities are unchanged. |

The approval record must include specialist review and an author rehearsal at the candidate head. Missing scenarios remain gaps; a guest does not substitute for author rehearsal. Fixture and copied-backup observations are recorded in the change record; they do not establish production qualification.

## Disallowed shortcuts

`optimize` is not a canonical reconstruction. [SQLite's rebuild documentation](https://sqlite.org/fts5.html#the_rebuild_command) excludes contentless tables. Resetting totals, editing shadow records, deleting the current sidecar before successful construction, or treating a count match as proof of correct membership/content is prohibited.
