# Keyword maintenance recovery

People and agents maintaining an offline keyword index need a supported exit
when construction or generation validation fails. The module command emits one
JSON result with the operation directory and manifest; errors exit 2, progress
goes to stderr, and every mutation owns advisory and DuckDB exclusion locks.
The [operator guide](../../keyword-initialization.md) is the command reference.

## States and decisions

The existing flow is inspect, begin, build, install, finish, with rollback before
release. At reviewed head `30c0013`, rollback from `building` reports “rollback
requires verified or installed state before release”. Canonical drift rejects
every continuation, and beginning elsewhere rejects the held marker. These
states leave callers blocked; the regression record captures those outcomes.

Coordinator addendum 2, 2026-10-23, directs supported abandonment, rollback and
replacement without manually removing the marker or restoring mixed generations.
The chosen recovery keeps the existing command and explicit identity arguments:

| Starting condition | Action and observation |
| --- | --- |
| No operation | `inspect` reports `absent`; `begin` captures and reports `building`. |
| Partial construction, canonical unchanged | `rollback` reports `rolled_back`; `finish` releases. |
| Original sidecar intact, any canonical generation | `abort` reports `aborted`, leaves files intact and releases. |
| Installed or changed sidecar, canonical drift or failed verification | `replace` names the prior owner and a fresh directory, captures current data and reports `building`; construction/install/release follow. |
| Wrong predecessor/target, busy store, insufficient space | Failure leaves exclusion and retained data intact; correct the input or retry the same capture. |
| Interrupted abort/release | Retry the same action; terminal state is saved before marker removal. |
| Interrupted ownership transfer/capture | Retry replacement at the same fresh directory; normal writers remain excluded. |
| Terminal or replaced operation | Reinstallation/rollback refuses; an old operation cannot remove its successor's marker. |

Automatic marker deletion loses the binding between recovery and its owner.
Unconditional rollback can pair old SQLite with current DuckDB. Requiring a new
complete build before any exit recreates the original deadlock. Explicit abort
and replacement avoid these effects by leaving live files alone until a
captured-generation install or rollback passes its checks.

Format-2 manifests carry capture metadata; format-1 manifests retain their
existing readable/resumable contract without invented historical metadata.
Rescope logs count actual committed publications rather than visited rows.
CLI observations and fresh guest verdicts belong in the change record; there is
no visual UI or new production authority in this flow.
