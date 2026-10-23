# Companion constraint ownership at 4637023

The measured population has 1,369,078 messages/state rows, 1,368,340 message
embedding rows, 1,281,988 calls, 1,281,728 call embedding rows, 732,686 results
and 733,226 harness-ID mappings. The seven large primary keys are separate
from the `messages(session_id, idx)` unique constraint and the nonunique
`tool_use_ids(session_id, tool_use_id)` lookup index. The latter must not be
described as a UNIQUE constraint.

| Table / key | Readers | Writers and invariant | Removal consequence |
| --- | --- | --- | --- |
| `messages.id` | session snapshots, show/live, sidecar reconciliation, embedding joins | raw inserts; positional `(session, idx)` also unique | Identity authority; retain uniqueness. |
| `message_state.message_id` | bounded ID history in `enrichment_inputs.py`, context updates in `embed_phase.py`, indexer, live/search joins, sidecar paging | raw state inserts/updates, context publication | Hot point access exists; retain uniqueness/index. |
| `message_embeddings.message_id` | target anti-joins and publication validation in `enrichment_inputs.py` and `embed_phase.py`; vector search in `search.py`; stored-vector/history reads | `queries.insert_message_embeddings` uses `INSERT OR IGNORE`; invalidation updates a selected ID or deletes a bound set | Removal needs explicit transaction-owned deduplicated publication, replacement of conflict semantics, bounded read-plan evidence, and preservation of all-NULL processed markers. |
| `tool_calls.id` | live/open calls, snapshots, search, rewrite validation | raw insert/update/replacement | Identity authority; retain uniqueness. |
| `tool_call_embeddings.tool_call_id` | pending anti-joins, vector search, bounded existing-vector selection inside `insert_tool_call_embeddings` | `INSERT OR REPLACE` guarded by vector difference; raw invalidation/deletion | Point/set access exists; removing the PK alone breaks writer conflict semantics. |
| `tool_results.tool_call_id` | live open-call LEFT JOIN, tail/history snapshots | `insert_tool_results` is insert-only `OR IGNORE`; deletion before mapping retirement | A scan/join candidate, but replay must remain idempotent and obsolete results must never attach to a reused positional ID. |
| `tool_use_ids.tool_call_id` | `fetch_tool_use_ids` by bounded call IDs; inverse `resolve_tool_call_ids` by session/harness ID; live joins | insert-only `OR IGNORE`; retire result then mapping before repointing a positional call | Both lookup directions are used. Retain the inverse nonunique index unless independently measured. |

Source anchors: `db/queries.py` functions `insert_tool_use_ids`,
`insert_tool_results`, `resolve_tool_call_ids`, `fetch_tool_use_ids`,
`delete_tool_call_tail_facts`, `insert_message_embeddings`,
`insert_tool_call_embeddings`; `services/indexer.py` `_update_message_states`,
`_invalidate_changed_message_embeddings`, `_write_tail_facts`;
`services/live.py` `_OPEN_TOOL_CALLS_SELECT`;
`services/embed_phase.py` `commit_prepared_embed_cycle`;
`services/search.py` vector-search SQL.

Single-writer serialization alone does not construct uniqueness. A replacement
must first deduplicate a bounded batch by parent identity, decide insert/replace
under that same writer transaction, and make that transaction the only mutation
capability for the relation. Raw reconciliation, model publication, moves,
repair, maintenance and direct database helpers are all callers to close over.
Repeated input, duplicate identities within one batch, interrupted publication,
reused positions and stale model work are required rehearsal scenarios. This is
a design requirement, not evidence that index removal is already safe.

## Observed tool-use mapping plans

An author functional probe ran the actual 256-ID Arrow-bound query shapes for
`fetch_tool_use_ids` and `resolve_tool_call_ids` on the populated VARCHAR copy.
`EXPLAIN ANALYZE` reported a sequential table scan in both. The retained plans
are `scratch/evidence/tool-use-read-plans/{by-call,by-native}.txt`; their elapsed
times are not benchmark claims. This distinguishes an identity-set reader from
an observed ART point scan. It makes the tool-call-key PK an additional removal
candidate; the session/native lookup index is a separate index and remains.
Other cardinalities, live joins and write conflict semantics still require
their own evidence. The initial `drop-results` daemon variant removes only the
result PK and must not be presented as measuring both removals.
