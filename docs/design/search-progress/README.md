# Search wait visibility

People and agents running local semantic search need to tell embedding preference from a stuck query, and need a timeout to identify where its budget went. The existing flow submits one search and renders a final result or error; it gives no intermediate wait signal and reports a generic timeout. Status and final relevance guidance are separate stderr consumers.

The approved flow emits plain, flushed stderr lines during a real embedding preference wait, then renders the existing final payload. The notification/error schemas and normative copy live in SPEC REQ-RPC-022/023 and REQ-CLI-025. The five-second preference and 30-second request deadline retain their meaning. This command has no terminal styling or cursor movement, so redirected stderr and narrow terminals preserve complete lines; wrapped lines remain ordinary text.

| State | Existing surface | Approved surface |
| --- | --- | --- |
| Idle, keyword, tool filter or automatic keyword fallback | Final result only | Final result only |
| Semantic search waits on enrichment | Silent | `search: waiting for embedding drain (preference up to 5s; request deadline 30s)` |
| Preference ends at cap | Silent until final result | `search: embedding preference ended after 5.00s (limit reached); joining scan admission` |
| Drain becomes idle | Silent until final result | Release line uses actual duration and `(drain idle)` |
| Recovery retry | Silent | Same lines prefixed `search (attempt 2):` |
| Request times out | Generic runtime error | Active and largest phase suffix; structured error.details carries exclusive durations |
| Peer leaves or wait is cancelled | No result for departed peer | No released frame; request cleanup owns cancellation |
| JSON, JSONL, TOON | Final structured stdout | Same stdout; progress only on stderr |
| Older daemon | Final result only | Same silent compatibility behavior |

The bounded two-frame notification uses the existing JSON-RPC channel and local client callback. Heartbeats would add traffic without identifying progress; logs would leave the waiting caller blind. An in-place spinner would make redirected and agent output harder to consume. Fleet progress transport, new options, model policy and durable operations are outside this change.

Decision: on 2026-10-22 the human selected “#118 embedding wait notice” and approved plan revision 1, including the five-second preference. This is the final design intent; before/after runtime captures and guest/design verdicts belong to the lane evidence record.

## Keyword-index lock error

A person who encounters another connection's SQLite lock needs the affected index and a next action. An agent needs a stable conflict class and retry information. The daemon records failed-read readiness before returning the error; the CLI renders it through the existing error path, with no empty or partial success and no automatic retry. Pure vector search without a tool has no keyword dependency. Inactive poll snapshots retain RUNTIME.

| State / output | Before (#331) | Approved surface |
| --- | --- | --- |
| Reader-excluding keyword-index lock, text / TTY auto | `error: search failed: keyword index read failed: OperationalError: database is locked` on stderr; empty stdout; exit 1 | `error: search failed: keyword index read failed: OperationalError: database is locked; index: /scratch/data/recall.fts.sqlite; retry after the lock is released` on stderr; empty stdout; exit 1 |
| JSON / JSONL / ndjson / `--json` | `{"error":{"code":"RUNTIME","message":"search failed: keyword index read failed: OperationalError: database is locked"}}` | `{"error":{"code":"LOCKED","message":"search failed: keyword index read failed: OperationalError: database is locked; index: /scratch/data/recall.fts.sqlite; retry after the lock is released","details":{"kind":"keyword_index_locked","retryable":true}}}`; exit 1 |
| TOON / non-TTY auto | TOON encoding of the RUNTIME object | TOON encoding of the LOCKED object, including boolean true; exit 1 |
| Missing TOON support | Existing JSON fallback and one-time stderr warning | Same fallback and warning, preserving the LOCKED object |
| Release and retry | Successful result after existing recovery | Same daemon and successful result after existing recovery; no automatic retry |
| Older daemon | RUNTIME without retry details | Same RUNTIME answer; a newer CLI does not infer details from prose |

The absolute path is displayed as a filesystem path, including spaces or URI-sensitive characters. The example path above stands for the actual configured sidecar. Progress remains on stderr; status-notice opt-out, field projection and CTA options cannot suppress or reshape the error. Existing waiting, success, timeout and older-daemon states remain as specified above. Callers supporting a staggered upgrade accept both RUNTIME and LOCKED; only `details.kind=keyword_index_locked` identifies this retryable keyword lock. A new invocation after release owns its bounded backoff and may still fail for another reason.

Decision: on 2026-11-09 the human selected “LOCKED, retryable” and confirmed the revision-2 plan (SHA-256 `0a803c03c2312367395dcc039539f49a1baa9019cde115b3a48199439dd96446`). Generic RUNTIME loses the stable retry signal; empty or vector-only success can misstate keyword coverage; client message parsing cannot classify older failures reliably. Native before/after captures and design verdict belong to the implementation evidence record.
