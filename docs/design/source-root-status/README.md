# Source root readiness in daemon status

People and agents use `recall daemon status` to decide whether indexing has
caught up and to locate a root that prevents it. The existing status surface
has a readiness headline, a reconciliation summary, and structured per-root
coverage. Root diagnostics follow the reconciliation flags in text mode; JSON
retains the coverage records and adds `scan_current_process`.

## Baseline and decision

At 35bcb24, a nonexistent configured root reports `available=false`,
`scan_complete=false` and an ENOENT error in JSON. The text view reports false
readiness without naming the root. Default absent roots already complete empty.
The populated scratch-daemon captures are retained with the #138 task evidence.

The human selected “Missing = empty (Recommended)” on 2026-10-22: “a configured
folder that doesn't exist counts as empty for readiness but stays listed as
unavailable, and status names any folder that blocks readiness.” REQ-RECON-030
owns this contract. Treating configured absence as an error was rejected by
that decision. Treating every unavailable root as empty would hide permission
and non-directory errors, so only an observed FileNotFoundError qualifies.

## States and output

| Root state | Coverage and readiness | Text diagnostic |
| --- | --- | --- |
| No observation yet | Incomplete; blocks | `Root blocks readiness: codex /path: not scanned` |
| Inventory running | Incomplete; blocks | `Root blocks readiness: codex /path: scan in progress` |
| Previous process's completed scan | `scan_current_process=false`; blocks | `Root blocks readiness: codex /path: not scanned since daemon start` |
| Absent, observed in this process | Unavailable, complete empty; permits readiness | No blocking line |
| Root appears later | Periodic inventory discovers and indexes its files | Pending count until caught up |
| Permission or non-directory error | Incomplete; blocks; recorded errors retained | Root source, path and filesystem error |
| Complete scan with backlog | Pending files block | `Root blocks readiness: codex /path: 2 pending files` |
| Disabled | No observation required | No blocking line |
| RPC unavailable | Existing unknown readiness and coverage diagnostic | No fabricated root diagnostics |

Every blocking root gets a line, independent of source-file pagination. Multiple
reasons for a root are separated by semicolons. Existing status formatting,
projection, exit codes, and stderr health notices remain the surface conventions.
The author checks text and JSON for absent and permission-denied roots against a
populated scratch daemon; outcome regressions cover the additional scan states.
