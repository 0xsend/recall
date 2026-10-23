# Runtime compression and maintenance status

People and automation need to keep writing existing databases, inspect explicit
fixed-header maintenance, and distinguish an active operation from a failed one.
The human selected runtime policy only on 2026-10-23 (impl-93 addendum 3).
The codec conversion surface and its recovery artifacts are removed.

The existing command remains preview → bind plan → apply → inspect/wait. The
version-1 fixed-header response retains its fields; it has no codec target,
codec eligibility, codec phase or codec backup. Automatic compression excludes
Roaring on owned connections. Historical segments alone do not block writers.

| State | Status observation | Caller action |
| --- | --- | --- |
| No explicit operation | Current database; `storage_maintenance: null`, admission allowed | Continue ordinary reads/writes |
| Accepted/running | Marked captured observation; `phase: running`, `snapshot: true`; admission refused | Inspect/wait; reconnect without cancelling |
| Overlapping apply | Same running operation; `accepted: false`, `error: null` | Wait for its terminal observation |
| Success | Current database; `phase: succeeded`, `snapshot: false`, `error: null`; admission allowed | Continue ordinary writes |
| Failure | Current database; `phase: failed`, `snapshot: false`, error text; admission allowed | Inspect failed operation and retry with a fresh plan |
| Conflicting plan | Existing invalid-params response, no mutation | Preview again |
| Caller wait deadline | Existing timed-out response; work continues | Inspect operation before retry |
| Daemon unavailable | Existing connection failure | Inspect startup status using existing tools |

All mutating RPC database handlers use the shared writer admission, including
installed scheduler and live marks. During maintenance the denial is
`APP_LOCKED`, `reason: storage_maintenance`, `action: recall.migrate_storage`.
Text status displays the same phase, snapshot marker, terminal error and write
admission in its summary. A writer already executing before acceptance drains normally. The maintenance
owner has the sole explicit admission exemption. Terminal success and failure
wake reconciliation so an earlier admission refusal is refreshed immediately. Shutdown and fatal recovery
retain their lifecycle ownership; neither is a caller-admitted RPC write.
The operation reserves its task before capturing status. That task alone owns
snapshot capture, maintenance, terminal error and shutdown cancellation; a
duplicate cannot replace it or report its own failure as the operation's error.

Compaction retains its command/automatic trigger and fixed-header policy. It
writes without Roaring, closes/reopens the candidate, verifies counts and checks
persisted segments before atomic replacement. Candidate failure leaves the
original live file authoritative. No codec journal can hold a later open.

The reviewed version froze status after failure and omitted the error; the
terminal observation above replaces it. Retaining that frozen snapshot would
misrepresent failure as ongoing work. Keeping the historical write fence would
require the discarded conversion, so it was rejected by the human's narrowing.
The behavioral test receipts and author/guest status captures live in the lane's
round-3 evidence record; this is an agent-facing surface with no visual UI.
