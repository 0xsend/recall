# Reconciliation operations

Reconciliation is how the recall daemon keeps its index level with the transcript
files on disk. This guide covers the operator view: what `reconciliation.pending`
counts, how it drains, which fields answer "is it progressing", and what to do
when it stops. [SPEC.md](../SPEC.md) owns the contract (`REQ-RECON-001` through
`REQ-RECON-023`); [RECONCILIATION_QA.md](../RECONCILIATION_QA.md) owns the
verification procedures.

## What pending counts

The daemon keeps a durable catalog of every transcript file it has discovered
under the configured source roots. `reconciliation.pending` is the number of
catalogued **files** — not messages — whose committed index is behind the file
on disk, because any of the following holds:

- the desired generation is ahead of the committed generation (the file changed,
  or its parser revision did),
- committed bytes are behind the file size (the file grew, or was never fully
  indexed), or
- the last attempt on that file recorded an error.

Files the daemon has catalogued but can no longer stat are excluded and counted
as `missing`. Files under a disabled root are excluded from `pending` and
counted as `out_of_scope_pending`.

`pending` is not `embed_pending`, which counts message and tool-call rows
awaiting embeddings. Raw indexing progresses independently of model work
(`REQ-RECON-008`): a deferred embed phase does not hold up the reconciliation
drain, and an empty reconciliation backlog does not imply embeddings are done.

## How it drains

Periodic full inventory and filesystem notifications feed the catalog. A bounded
fair scheduler selects catalogued work, preparation (reading and parsing) runs
outside the writer lock, and a single transactional writer commits content with
its checkpoint. Watch mode reacts to filesystem events; poll mode reconciles the
roots on an interval. Both share the same reconciliation path.

`pending` therefore falls as files commit and rises when discovery finds new or
rewritten files, so a backlog that grows briefly during a scan is normal. A
count that only rises across several scans is not.

One measured order of magnitude, not a guarantee: a macOS host went from 27,326
to 25,253 pending files in 21 minutes — about 100 files per minute — while the
same daemon served live watch work. The rate depends on file sizes, the source's
parser, and what else the daemon is doing, so a host's own two readings are a
better baseline than this number.

## Fields to read

All of these live under `reconciliation` in `recall daemon status`:

| Field | Meaning |
|-------|---------|
| `pending` | Catalogued files not yet committed to their current bytes. |
| `paused` | An operator pause is in effect; work is held, not lost. |
| `catalog_scan_complete` | Every configured root finished a scan. While false, totals are still moving. |
| `raw_indexing_ready` | Scan complete and `pending == 0`. |
| `keyword_search_ready` | `raw_indexing_ready` plus an empty keyword sidecar queue. |
| `enrichment_deferred` | Why the embed phase last stood down (for example host load above threshold). Does not block the drain. |
| `error` | The inventory error, when a scan itself failed. |
| `coverage[]` | Per source and root: `pending`, `oldest_pending_age` (seconds), `scan_complete`, `failure_count`, `errors`, `unsupported`, `missing`. |
| `unsupported_summary` | Parked parser gaps an agent can file: source, record detail, file count, and up to three sample paths. Kind-only rows from an older daemon say the detail was omitted. |
| `source_page` / `next_cursor` | A bounded page of per-file catalog records with diagnostics; `next_cursor` is an opaque token to pass back verbatim. |

Text mode's summary block states the headline numbers. For a scripted read, the
existing top-level projection is enough — there is no separate reconcile
command, and none is needed:

```bash
# Headline drain state, with the per-file page held to one row.
recall daemon status --json --limit 1 --fields reconciliation \
  | jq '.reconciliation | {pending, paused, catalog_scan_complete, keyword_search_ready, enrichment_deferred, error}'

# Where the backlog sits, and how old the oldest pending file is.
recall daemon status --json --limit 1 --fields reconciliation \
  | jq '.reconciliation.coverage[] | {source, pending, oldest_pending_age, scan_complete, unsupported, missing}'
```

`--fields` projects top-level keys only, so `--fields reconciliation` returns the
whole object and `jq` (or any JSON reader) selects within it. `--limit` keeps the
per-file page small; raise it with `--limit N` (1..256) and page with `--cursor`
when the per-file diagnostics are what is wanted.

## When pending is not moving

Two readings a minute apart establish whether it is stuck. If the count is flat:

1. **Check `paused`.** A durable pause survives restarts and client auto-start
   (`REQ-RECON-009`). `recall daemon resume` clears it.
2. **Check the daemon is actually serving.** `recall daemon status` reports
   `version_drift` and, when the RPC is unreachable, says so instead of
   reporting runtime fields. A drifted daemon runs old code until restarted.
3. **Check `error` and per-source `failure_count` / `errors`.** A root that
   cannot be scanned (unmounted, permissions) reports incomplete coverage rather
   than silently claiming completeness.
4. **Check `unsupported`.** Those files are parked on parser diagnostics, not
   missing: they stay pending until a parser handles them. Inspect
   `source_page` diagnostics and file a recall issue (`REQ-CLI-023`).
5. **Check the daemon log.** A daemon that crash-loops can keep a healthy-looking
   status while making no progress; `<data dir>/logs/daemon.err.log` is the place
   that shows it.
6. **Check whether the drain is merely slow.** `oldest_pending_age` climbing
   while `pending` falls slowly is a large backlog, not a stall.

## Commands that are safe

- `recall daemon status` — read-only, and the only command needed to diagnose a
  drain.
- `recall daemon pause` / `recall daemon resume` — durable operator intent; the
  pause holds work rather than discarding it.
- `recall index` — asks the running daemon to index new and changed transcripts.
  It does not re-parse history.
- `recall daemon restart` — safe, but it re-runs startup work; on a host with
  LLM context enabled a restart can spend a long stretch on context catch-up
  before embedding resumes.

Not a remedy for a backlog:

- `recall index --recreate` rebuilds the live database and is never the fix for
  pending work or a refused start.
- `recall index --full` on a host with an LLM context mode must pin
  `--context template`; see [below](#full-re-parse-on-an-llm-context-host).
- Deleting or moving the database while the daemon holds it.

## Full re-parse on an LLM-context host

On a host whose `[embedding.context] mode` is `llm-local`, `llm-remote`, or
`llm-codex`, pin the context mode for any full re-parse:

```bash
recall index --full --no-embed --context template
```

A bare `--full` re-parses correctly, but every message with no stored LLM context
is treated as pending and gets summarized. When only a small fraction of a large
corpus carries LLM context, that is millions of model calls and hundreds of hours
of local generation. Reusing stored context (`REQ-INDEX-021`) prevents losing
existing context, not the cost of rows that never had any.

Setting `RECALL_CONTEXT_MODE` in your shell does not help: `recall index` is
executed by the daemon with its own environment, and only `--context` travels in
the request. Check the request before a long run:

```bash
recall index --full --no-embed --context template --dry-run --json
# => "request": { ..., "context": "template", ... }
```

Pinning `template` forecloses nothing: the incremental path only resolves context
for freshly parsed messages, so it never revisits history anyway.
`--recompute-context --only-mode template` remains the upgrade path.

Back up `recall.duckdb` (and `recall.fts.sqlite`, if present) first, and split the
work into bounded runs; `--since` compares the transcript's mtime and `--project`
matches `git_repo` the same way `recall list --project` does. They compose:

```bash
recall index --full --no-embed --context template --since 30d
recall index --full --no-embed --context template --project ~/work/myrepo
```
