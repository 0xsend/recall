# Fleet text provenance

People reading aggregated sessions need the same origin information available to
structured consumers. They run list/search/live to locate work, then show to read
a session; a colliding ID requires an explicit endpoint selection.

The established text surface has one primary line per list/live row, a header
plus continuation lines per search hit, and a session header plus content in
show. Empty replies print the command's existing empty message. Partial failures
are diagnosed on stderr; invalid scope and ambiguous show retain their errors.
Ordinary list/search/show/live output and all structured formats retain their existing rendering.

## Decision and format

The human approved fleet host labels on 2026-11-11: “Add the host label in this PR
(Recommended)”. The existing stats usage convention supplies `host=NAME`.
The coordinator's provisional 2026-11-11 encoding decision keeps a nonempty name
plain only for ASCII letters, digits, `.`, `_`, and `-`. Every other value is a
JSON string literal with ASCII escaping: `host="edge host=trusted"`,
`host="edge\nforged"`, `host="\u00e9quipe"`. This preserves one physical line and
exact recovery, including quotes, backslashes and terminal controls. One shared
value formatter also serves ordinary/fleet usage and skill-census host text.
Raw interpolation allows forged rows; rejecting names would change stored data
semantics. JSON quoting preserves both data and provenance without either cost.

A shared formatter appends it to each fleet primary line or show header. Search
continuation lines and show messages belong to that labeled header. The label
reads the merged JSON field: stored origins survive; unattributed rows use the
canonical caller or remote inventory name. A selected self alias therefore does
not replace the canonical caller name. No separate endpoint label is inferred.

Examples use illustrative identities and omit unchanged continuation lines:

```text
list before:   [2026-11-10 12:00] session-id (codex) /project messages=1 tools=0
list after:    [2026-11-10 12:00] session-id (codex) /project messages=1 tools=0 host=CALLER
search before: [2.00] session-id (codex)
search after:  [2.00] session-id (codex) host=CALLER
show before:   [2026-11-10 12:00] Session session-id (codex)
show after:    [2026-11-10 12:00] Session session-id (codex) host=CALLER
live before:   [2026-11-10 12:00] active session-id (codex) /project turn=unknown fresh
live after:    [2026-11-10 12:00] active session-id (codex) /project turn=unknown fresh host=CALLER
```

The suffix keeps existing row information and follows the neighboring usage
surface. A separate host column would require a wider layout change; replacing
IDs or adding an envelope would alter unrelated contracts. Neither is needed.

## States and verification

The same labels apply to caller-plus-remote, declared self aliases, inventory-only
membership and caller-only empty inventories. Collision selections show the host
from the selected row, including imported origins. Empty output has no row to
label. Failure diagnostics and exits remain unchanged. Exact ordinary-output
controls and public CLI fleet tests cover rendering; the author rehearsal covers
the assembled real CLI with two scratch daemons. The coordinator owns delta
review, guest qualification and macOS rehearsal before landing.
