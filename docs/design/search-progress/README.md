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
