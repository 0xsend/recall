# Storage

recall stores its index in DuckDB at `~/.local/share/recall/recall.duckdb`. When
the SQLite FTS sidecar backend is enabled, its keyword index lives beside it in
`recall.fts.sqlite`. The daemon owns both files.

## Large corpora and DuckDB memory

The DuckDB buffer pool defaults to a `2GB` `memory_limit`. On a large index (a
multi-GB `recall.duckdb` with millions of messages and tool calls) some
operations, such as the embed-backlog snapshot, an FTS rebuild, or a fresh index,
can exceed that limit and fail with `Out of Memory Error`, which can escalate to
a `database has been invalidated` fatal. Raise the limit only when your index
needs it:

```toml
# ~/.config/recall/config.toml
[duckdb]
memory_limit = "16GB"
```

or set `RECALL_DUCKDB_MEMORY_LIMIT` in the daemon's environment. Size it to your
host and index.

## Storage format maintenance

recall pins DuckDB 1.5.5 and creates new databases with fixed storage format
`v1.2.0`. Ordinary opens preserve existing formats. The running daemon can
convert an older file through explicit maintenance with a consistent DuckDB
and keyword-sidecar backup:

```bash
recall daemon migrate-storage --dry-run --json
recall daemon migrate-storage --plan-id <digest-from-preview> --wait --json
recall daemon migrate-storage --dry-run --json  # inspect the same operation
```

Without `--wait`, apply returns acceptance while the daemon continues working.
The wait defaults to five seconds; `--timeout` controls the caller's overall
maintenance wait, and each RPC read remains bounded by five seconds. A wait
timeout does not cancel the operation. Inspect its status before retrying.
Retries retain the original backup and any captured index-migration scope.
Recovery backups are excluded from automatic snapshot-age cleanup.
Storage-only maintenance preserves the pause switch and index version;
completion of a combined migration also requires its captured index scope to
verify. Current or newer formats are never downgraded.
