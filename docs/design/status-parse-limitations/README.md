# Per-file parse limitations in text status

People reading `recall daemon status --format text` need to find the file and
diagnostic behind pending work. The existing summary gives readiness and backlog;
root lines describe scan failures or pending counts. Per-file parser diagnostics
are already available through the structured source page. REQ-CLI-012 and
REQ-CLI-018..023 establish stderr advisories with a concrete next action.
The neighboring [source-root status record](../source-root-status/README.md)
owns the existing readiness and root states.

## Baseline and decision

At 9d3ff3b, a newline-terminated malformed Codex record yields pending=1,
raw/keyword readiness false, null reconciliation error and empty root errors.
The source page reports `last_error="incomplete capture"` and `malformed_record`
with its detail and byte offset. Text status has no pointer to those diagnostics.
Baseline and candidate CLI captures live with the #258/#259 lane evidence.

The text summary emits a stderr advisory:

```text
Warning: 1 file on this source page has parse limitations; inspect with: recall daemon status --json --limit 10 --fields reconciliation.source_page
```

The count includes only eligible, nonmissing files on the selected source page
with a blocking error and nonempty parser diagnostic records. Unsupported parks
retain their existing notice. The command repeats the requested limit and, when
present, the input cursor in bash/zsh-safe quoting. Its JSON lists the affected
paths and diagnostics. Existing `More source coverage` navigation continues to
identify the next page.

## States and boundaries

| State | Advisory |
| --- | --- |
| One blocking file on the page | `1 file ... has parse limitations` and inspection command |
| Several blocking files on the page | `N files ... have parse limitations` and inspection command |
| No parser records, healthy files, or no source page | Silent; existing text unchanged |
| Missing or disabled-scope file | Silent; does not explain eligible pending work |
| Unsupported park | Existing unsupported coverage advisory |
| Continued page | Same notice, command retains the input cursor |
| Structured output | No added notice or data; byte-identical for identical daemon readings |
| Limitation outside the selected page | Not counted; inspect later pages |

A global count was rejected because a bounded page cannot establish one and
adding a status field or extra RPC reads exceeds this presentation change.
Printing parser payloads in the summary was rejected because the existing JSON
listing already supplies the details. No colors, terminal controls or width-
dependent layout are added; the advisory uses the existing plain stderr style.
