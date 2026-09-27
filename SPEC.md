# recall Specification

Session recall and analytics for AI agents (Claude Code, Codex, Pi Agent, Grok, Kimi Code).

## Overview

`recall` indexes AI agent session files into a unified DuckDB database, enabling search, analytics, and permission suggestions based on actual usage patterns.

**Primary use case:** Analyze tool usage across Claude Code, Codex, and Pi Agent sessions to identify safe auto-approve permissions and understand development patterns.

## Design Decisions

### Error Handling

| Scenario | Behavior |
|----------|----------|
| Malformed JSONL line | Skip line, log warning, continue |
| Corrupted session file | Skip file, log error, continue indexing |
| Mid-write session | Parse available lines, mark as incomplete |
| Database corruption | Error on run, `--recreate` flag to backup and rebuild |
| Schema mismatch | Error with clear message, suggest `--recreate`. Automatic migration is attempted for supported upgrades (v13→v14 CHECK relaxation; see Migration Policy); only unsupported/very old versions require `--recreate`. |

### Session Identity

**Session ID generation:** `SHA256(source:absolute_path)[:16]`
- Deterministic - same file always gets same ID
- Collision-resistant across sources
- Reproducible for debugging

**Message ID generation:** `SHA256(session_id:idx)[:16]`
- Unique within session via message index
- Stable across reindexing (same session + idx = same ID)

**ToolCall ID generation:**
- With message: `SHA256(message_id:idx)[:16]` where idx is position within message
- Orphan calls (message_id is null): `SHA256(session_id:orphan:global_idx)[:16]` where global_idx is the tool call's sequential position within the session's orphan calls
- Stable across reindexing (deterministic from source data)

**Orphan tool calls:** Codex `function_call` events that appear outside message blocks are stored with `message_id=NULL`. Their `idx` field represents position among orphan calls in the session (0-indexed).

### Parser Protocol

**SessionParser protocol:**
- `source: Source` — which agent source this parser handles
- `parse(path, offset=0) -> ParseResult` — parse a session file from a byte offset
- `watch_root() -> Path | None` — root directory to watch for file events
- `file_pattern: str` — glob pattern for matching session files (e.g., `*.jsonl`, `rollout*.jsonl`)

**ParseResult:**
- `session: Session` — full session object when `offset=0`; partial session (new messages + merged metadata) when `offset > 0`
- `next_byte_offset: int` — byte position after the last successfully parsed line; stored in `session_state.last_byte_offset` for resume
- `is_full_parse: bool` — whether this was a full parse (offset=0) or incremental

When `offset > 0`, the `Session` object contains only newly parsed messages and tool calls. Session-level metadata (`ended_at`, token counts) reflects values from the new lines only — the caller merges these into the existing DB row. Token merge semantics are source-aware: delta-token sources are added, while cumulative-token sources are merged as absolute totals.

### Incremental Indexing

**Staleness detection:** `file_mtime + file_size`
- Reindex if either changes
- Stored in `session_state.file_mtime` and `session_state.file_size`
- Skip unchanged files for fast incremental runs

**Byte-offset resume:** Session JSONL files are append-only. The indexer tracks `last_byte_offset` per session in `session_state`. On re-index of a changed file:
- If `last_byte_offset > 0` and `file_size >= last_byte_offset`: seek to offset, parse only new lines
- If `file_size < last_byte_offset` (file truncated/rewritten): fall back to full parse from offset 0
- New sessions (no stored offset): full parse from offset 0

**Incremental reindex workflow (when file changed, offset > 0):**
1. Seek to `last_byte_offset`
2. Parse new JSONL lines only
3. Merge session metadata incrementally (update `ended_at`, merge token counts according to source semantics, preserve `started_at`)
4. Insert new `messages` / `tool_calls` and their embeddings
5. Update `session_state` with new `file_mtime`, `file_size`, and `last_byte_offset`
6. Commit transaction

**Full reindex workflow (new session or offset 0):**
1. Begin transaction
2. Parse file from byte 0
3. Upsert mutable `session_state` and `message_state` rows in place
4. Delete removed `tool_calls` / `messages` rows and their embedding table rows
5. Insert newly discovered `messages` / `tool_calls`
6. Commit transaction

Deletion detection is only performed during full parse (offset 0). Append-only JSONL files do not remove lines, so incremental parsing does not need to detect deletions.

The `is_complete` flag reflects the latest parse state — TRUE if all parsed lines decoded successfully.

**Discovery ordering:** File discovery is sorted by `file_mtime` descending (newest first). This ensures recent sessions are indexed and searchable before older history during full reindexes and catch-up scans.

**Note:** No foreign key constraints are used (see `REQ-SCHEMA-003`). Deletions can be performed in any order but embedding tables should be cleaned up alongside their content tables.

### Concurrency

**Advisory file lock** at `~/.local/share/recall/recall.lock`
- Fail fast if another `recall index` is running
- Lock released on process exit (normal or crash)
- Database writes remain serialized under this lock even when any future parse/preparation concurrency is enabled.

**Read/write isolation:**
- `REQ-CONC-001`: Read-only CLI commands (`search`, `list`, `show`, `stats`) must use read-only database connections that do not block or conflict with concurrent write operations from the daemon or other index processes.
- `REQ-CONC-002`: Write operations must checkpoint and release the database file lock between discrete units of work (per-session index, FTS rebuild, runtime state update). Long-lived daemon processes must not hold the database lock during idle periods.
- `REQ-CONC-003`: If a read-only connection fails because the database does not yet exist, the command must fall back to creating the database via a read-write connection.

### Agent-First CLI Contract

#### Requirements

- `REQ-CLI-001`: `recall` must expose a machine-readable command manifest via `recall --llms` and a targeted schema view via `recall schema <command...>`. The manifest is the source of truth for command arguments, options, environment variables, output shape, examples, and safety metadata.
- `REQ-CLI-002`: Every command must support structured output formats. Human-readable text is allowed for TTY use, but non-TTY stdout must default to structured output without requiring `--json`.
- `REQ-CLI-003`: `--json` is shorthand for `--format json`. Supported structured formats must include `json`, `jsonl`, and `toon`.
- `REQ-CLI-004`: Structured output must be deterministic and compact. Output emitted for agent use must not include surrounding prose, ANSI escape sequences, or progress text on stdout.
- `REQ-CLI-005`: Errors must have stable machine-readable codes and details in structured modes. Commands must not leak raw Typer tracebacks for expected operational failures such as validation errors, schema mismatch, or database lock contention.
- `REQ-CLI-006`: Mutating commands must support `--dry-run`, returning the resolved request and declared operation metadata without executing side effects.
- `REQ-CLI-007`: Destructive commands must require explicit confirmation in non-interactive mode. `recall index --recreate` must fail fast unless `--yes` is provided or `--dry-run` is used.
- `REQ-CLI-008`: Command manifests must classify operations with `mutates`, `destructive`, and `idempotent` metadata so agents can reason about safety without scraping prose help.
- `REQ-CLI-009`: Commands must accept a raw JSON request payload via `--params <json>`. Payload fields map onto the same validated request schema as flags and arguments; explicit CLI arguments override `--params`.
- `REQ-CLI-010`: High-volume commands must expose context-window controls. At minimum, `recall list` and `recall search` must support `--limit`; `recall show` must support output filtering and message limiting.
- `REQ-CLI-011`: Structured commands that return objects or arrays must support `--fields` to project top-level fields in structured output. Unknown fields must fail validation.
- `REQ-CLI-012`: Human-only notices and progress updates belong on stderr. Structured stdout must remain reserved for command data.
- `REQ-CLI-013`: Non-TTY stdout must default to TOON format for token-efficient agent communication. `--json` forces JSON. `--format` overrides the default. Supported formats: `auto`, `text`, `json`, `jsonl`, `toon`. Introspection commands (`--llms`, `schema`) always emit JSON regardless of format since their output serves schema discovery where JSON is the standard interchange format.
- `REQ-CLI-014`: When `toon-format` is unavailable at runtime, the CLI must fall back to JSON with a one-time stderr warning per process. This enables graceful degradation in minimal installations.
- `REQ-CLI-015`: Commands may emit call-to-action suggestions (CTAs). In TEXT mode, CTAs render on stderr as `Next:` hints. In structured modes, CTAs appear only when `--cta` is passed, wrapping the response in `{"data": <original>, "cta": [...]}`. Without `--cta`, the output shape is unchanged — backward compatibility is preserved.
- `REQ-CLI-016`: CTA suggestions must be contextual — using real IDs and values from the command's output data to pre-fill suggested commands. Generic placeholders are acceptable only when no concrete values are available.
- `REQ-CLI-017`: Every `recall search` result must carry a boolean `lexical_match` field indicating whether the result matched the lexical (BM25/FTS) leg of the query. Keyword-mode results are always `true`; pure vector results are always `false`; hybrid results are `true` only when the row also matched the keyword leg. This lets callers distinguish a keyword hit from a semantic-only neighbor in a single field that survives `2>/dev/null`.
- `REQ-CLI-018`: When `recall search` returns results but none have `lexical_match=true` and the requested mode is not `vector`, the CLI must emit a stderr advisory that the results are semantic-only and may be unrelated. The advisory is human guidance and must not appear on structured stdout (per `REQ-CLI-012`).
- `REQ-CLI-019`: When `recall search` returns zero results, the CLI must emit actionable stderr guidance (widen the query, confirm indexing via `recall show`/`recall list`, that parked `coverage.unsupported` sources are a recall parser gap to file or fix, and that recall does not index commands run directly in a terminal). The guidance must never name the mode the query already ran in, and must name another mode only where that mode can return more: `--mode keyword` is told to drop the flag so `auto` adds the semantic leg when the host has embeddings, `--mode vector` is told to try `--mode keyword` for exact matches, and `hybrid` or `auto` get no mode hint at all because both legs already ran. When the zero-result query used `--tool`, the CLI must re-probe once without the tool filter and report both the unfiltered total and its `lexical_match` count, so callers can distinguish an over-restrictive filter from a genuinely unmatched query. Every zero-result branch must additionally carry the pending-coverage clause when `reconciliation.pending` is nonzero (see `REQ-CLI-023`): naming the backlog — and an unfinished catalog scan when one is reported — is what separates "your query did not match" from "recall has not indexed it yet", which is the one condition under which an empty answer is not evidence of absence. The coverage read costs one extra `daemon_status` call and is made only on an empty result; an unreachable daemon drops the clause rather than failing the search.
- `REQ-CLI-020`: `recall show` must resolve a bare lowercase-hex session id of 6–31 characters as a prefix of the internal 32-hex session id via an indexed table lookup. A unique match resolves to that session; multiple matches fail fast with an error listing the candidate ids; zero matches fail fast with `session not found`. A prefix-shaped miss must never trigger the unindexed-session filesystem scan — that scan reads every Codex rollout file and exceeds the 30s RPC read timeout, which previously turned every prefix lookup into a `RUNTIME` timeout error.
- `REQ-CLI-021`: The CLI status notice (`render_status_notice`, emitted on `recall search`, `recall list`, `recall stats` and each of its subcommands — `tools`, `bash`, `tokens`, `usage`, `skills` — and `recall daemon status`) must warn when the daemon reports `version_drift` — i.e. the running daemon's package version differs from the resolved binary version (including when binary metadata is missing after a reinstall). The warning must name both versions and direct the operator to `recall daemon restart`. This surfaces the recurring failure where an upgraded `recall` keeps serving stale daemon code (old parsers, un-applied fixes) until restarted. The notice is stderr-only per `REQ-CLI-012`. When a command renders structured (non-text) output, the notice must still reach stderr but drop the human-oriented informational line (index freshness) via `include_informational=False`, so agents using `--json`/`--toon` receive actionable health warnings without a status line on every call.
- `REQ-CLI-022`: The daemon must cache its most recent `estimate_bloat_ratio` (computed once at startup before the shared connection opens, and refreshed on every auto-compact check — including updating the cache to the post-compaction ratio after a successful compact, so the notice stops warning about bloat already reclaimed) and expose it via `daemon_status` as `bloat_ratio`, alongside `bloat_ratio_threshold` and `bloat_auto_trigger`. All three fields must be listed in the `daemon status` manifest so they are projectable via `--fields`. The CLI status notice must warn when `bloat_ratio >= bloat_ratio_threshold`, naming the ratio and either that the daemon will auto-compact (when `auto_trigger`) or that the operator should run `recall compact --yes`. Database bloat is otherwise invisible until it degrades queries or trips an out-of-memory failure; the estimate must never be recomputed synchronously on the status path (it scans storage metadata and is too slow per call). The notice is stderr-only per `REQ-CLI-012`.
- `REQ-CLI-023`: The CLI status notice (same surface as REQ-CLI-021, subcommands included) must warn when reconciliation coverage reports any `unsupported` sources. The warning names the count, that those transcripts are parked on parser diagnostics rather than missing, and that the caller must file a recall issue or fix the parser. When `reconciliation.unsupported_summary` is present it names up to three groups (source, record detail or that an older payload omitted the detail, and file count) and points at that field rather than at a source page. When the field is absent, the notice points at `reconciliation.source_page` diagnostics, which an older daemon still returns. It is stderr-only per `REQ-CLI-012` and still emits in structured modes (`include_informational=False`). A zero or absent unsupported count is silent.

  The same notice must also warn when `reconciliation.pending` is nonzero **and** the backlog has aged past a noise floor — transcripts recall discovered and has not committed. `recall live` already refuses to let a caller infer absence from an incomplete page; `search`, `list` and `stats` are where that inference is most costly and were silent about it, so they name the pending count, add that the catalog scan has not finished when `catalog_scan_complete` is false, and point at `recall daemon status --json --fields reconciliation`. `catalog_scan_complete` is only detail: a host that indexes without ever reconciling reports an unfinished scan permanently, and an always-on warning is one nobody reads. A zero or absent pending count is silent.

  The noise floor is the largest `reconciliation.coverage[].oldest_pending_age` among roots whose `configuration` is not `disabled` — the seconds the longest-waiting discovered transcript has gone uncommitted. It must exceed 300s for the notice to fire. Without it, a watch host warns permanently: every append puts its own live session back in `pending` for the seconds before the daemon commits it, so `pending > 0` is the steady state rather than a backlog, and `--json` callers receive the warning on every call. When the payload reports no age for any eligible root the notice fires on the count alone, because under-warning about coverage is the costlier error. The zero-result `search` guidance of `REQ-CLI-019` applies the same floor: a seconds-old append never explains why a query matched nothing.

  A command must read `daemon_status` at most once per invocation and render each coverage fact at most once on stderr. `recall search` reads it with `limit=1` — the notice consumes one count, not the 100-row default source page — and reuses that reading for both the health notice and the `REQ-CLI-019` zero-result clause; when the clause states the backlog, the notice omits its pending-coverage part so the fact does not reach stderr twice.

  Every advisory read, the zero-result clause included, is gated on `[cli] status_notices` and is made with `auto_fork=False`. Neither a notice nor guidance may start a daemon: forking one to decorate stderr both defeats the opt-out and makes an advisory line the reason a background process exists.
- `REQ-CLI-024`: `recall show` must project each message onto what the caller asked to see, identically in text, structured, and `--follow` output. `--thinking` gates the `thinking` and `thinking_embedding` fields: without it they are absent, not null, while `has_thinking` remains so a caller still learns that thinking exists. `--tools` continues to gate tool calls. A message left with no `content`, no visible `thinking`, and no tool calls must collapse to a summary row — `id`, `idx`, `role`, `timestamp`, `has_thinking`, and a `summary` naming what was withheld and the flag that reveals it — instead of a record whose every payload field is null; text mode renders it as one line with no trailing blank. The summary carries `tool_call_count` only when `--tools` was passed, because a read that never requested tool calls cannot honestly report zero. Before this, `--thinking` changed only the text renderer, so the structured output agents actually read was byte-identical with and without it, and a 185-message session returned 169 blank records whose payload appeared only under `--tools`.
- `REQ-RECALL-0145-H1`: `recall snapshots gc [--days N] [--dry-run] [--yes]` must prune top-level entries in `<data_dir>/snapshots/` whose top-level mtime is older than `--days` (default 7). Output reports `removed_paths`, `kept_paths`, `failed_paths`, `total_bytes_freed`, and `dry_run`. `removed_paths` contains only paths verified absent after deletion, while `failed_paths` contains stale entries that could not be statted, removed, or verified absent. It operates only inside `<data_dir>/snapshots/`, refuses paths that resolve outside that directory, and treats a missing snapshots directory as a no-op.
- `REQ-RECALL-0145-H4`: `recall snapshots list` must report the entries in `<data_dir>/snapshots/` read-only, as `snapshots_dir`, `snapshots_dir_missing`, `entries`, `entry_count`, `total_bytes` and `unreadable_paths`. Each entry carries its `paths`, `total_bytes`, `age_seconds`, and the two reasons gc leaves an entry alone: `retained` for a recovery backup and `paths` outside the directory, `partial` for a snapshot missing one half of the DuckDB/FTS-sidecar pair. Entries are ordered oldest first. It must mutate nothing, must be registered in the manifest as non-mutating, and must derive its grouping from the same scan `gc` uses, so a listing never describes a grouping the pruner does not apply. Without it, the only way to see what is on disk was to ask a destructive command what it would destroy — and `gc --dry-run` never shows the retained recovery backups a rollback depends on.

#### Invariants

- Non-TTY `recall list`, `recall search`, `recall show`, `recall stats`, `recall index`, and `recall daemon status` emit TOON by default for token-efficient agent communication, with JSON fallback when `toon-format` is unavailable.
- `--dry-run` never mutates the database, scheduler artifacts, or config files.
- `recall --llms` and `recall schema <command...>` return valid JSON without requiring hidden environment flags.
- `--cta` is the only mechanism that changes the output envelope shape; without it, structured output is always the bare data payload.
- `recall snapshots gc` never follows a symlink that resolves outside `<data_dir>/snapshots/`.

#### Non-goals

- YAML/CSV/Markdown output formats.
- MCP server (`--mcp`) — deferred to a follow-up; the RPC server architecture provides the foundation.
- Replacing Typer's built-in human `--help` renderer.
- Arbitrary nested field selection beyond top-level `--fields` projection in the first implementation.

#### Acceptance Criteria

- [ ] `recall --llms` returns a manifest describing all commands and safety metadata.
- [ ] `recall schema search` returns the targeted command schema without dumping the full manifest.
- [ ] Non-TTY `recall list` returns structured output by default without `--json`.
- [ ] Validation failures in structured mode return a JSON error object with `code`, `message`, and optional `details`.
- [ ] `recall index --recreate` refuses to run in non-interactive mode without `--yes`, but `--dry-run` succeeds.
- [ ] `recall search --params '{"query":"git","limit":1}'` resolves the request through the same validation path as flags.
- [ ] `recall list --fields id,source --limit 2` returns only those fields for at most two items in structured output.
- [ ] Non-TTY `recall list` returns TOON by default (REQ-CLI-013).
- [ ] `recall list --json` returns JSON despite TOON being the default (REQ-CLI-013).
- [ ] `recall list --format json` returns JSON (REQ-CLI-013).
- [ ] When `toon-format` is unavailable, non-TTY output falls back to JSON with a stderr warning (REQ-CLI-014).
- [ ] `recall list --cta` wraps output in `{"data": ..., "cta": [...]}` (REQ-CLI-015).
- [ ] `recall search "git" --cta` includes a contextual `recall show <session-id>` in CTAs (REQ-CLI-016).
- [ ] TEXT mode `recall list` shows CTA `Next:` hints on stderr (REQ-CLI-015).
- [ ] `recall --llms` and `recall schema <command>` always return JSON regardless of TTY state (REQ-CLI-013).
- [ ] `recall search` structured output includes a `lexical_match` boolean on every result (REQ-CLI-017).
- [ ] A query with no lexical matches (e.g. gibberish) emits a "semantic-only" advisory on stderr while stdout stays pure data; `--mode vector` suppresses the advisory (REQ-CLI-018).
- [ ] `recall search <q> --tool Bash` with zero hits emits a stderr re-probe reporting the unfiltered count and keyword-match count, leaving stdout as an empty array (REQ-CLI-019).
- [ ] Zero-result `recall search` guidance names parked `coverage.unsupported` as a recall parser gap to file or fix (REQ-CLI-019).
- [ ] Zero-result `recall search --mode keyword` guidance never suggests `--mode keyword`; `--mode hybrid` and `--mode auto` guidance names no mode at all (REQ-CLI-019).
- [ ] Status notice on search/list/stats/daemon status warns when coverage.unsupported is nonzero, including in structured modes on stderr, and names `reconciliation.unsupported_summary` groups when that field is present (REQ-CLI-023, REQ-RECON-029).
- [ ] `recall stats tools|bash|tokens|usage|skills` render the same stderr notices as bare `recall stats`, in text and structured modes, and never on stdout (REQ-CLI-021/023).
- [ ] Status notice on search/list/stats warns when `reconciliation.pending` is nonzero and the oldest pending transcript is older than the 300s noise floor, on stderr in both text and structured modes, and is silent once the backlog clears (REQ-CLI-023).
- [ ] A watch host whose only pending rows are live sessions appended seconds ago emits no coverage warning (REQ-CLI-023).
- [ ] A zero-result `recall search` issues one `daemon_status` read with `limit=1` and names the backlog once on stderr, not once per notice (REQ-CLI-019/023).
- [ ] Zero-result `recall search` guidance names the pending backlog when one exists and omits the clause when coverage has caught up (REQ-CLI-019).
- [ ] `recall show <id> --json` omits `thinking` on every message; `--thinking` restores it, and `has_thinking` is present either way (REQ-CLI-024).
- [ ] A tool-only turn in `recall show <id> --json` is a summary row with `summary`, not a record of nulls; `--tools` returns the full record and adds `tool_call_count` to any turn that still has nothing (REQ-CLI-024).
- [ ] `recall snapshots gc --dry-run --json` reports stale snapshot entries without deleting them (REQ-RECALL-0145-H1).
- [ ] `recall snapshots gc --days 7 --yes --json` removes only stale entries inside `<data_dir>/snapshots/` (REQ-RECALL-0145-H1).
- [ ] `recall snapshots list --json` reports every entry with size and age, marks recovery backups `retained` and half-pairs `partial`, and leaves every path on disk (REQ-RECALL-0145-H4).

#### snapshots list

`recall snapshots list` answers what is in `<data_dir>/snapshots/` without
running the pruner: each entry's paths, size, age, and whether `gc` would leave
it alone because it is a retained recovery backup or a partial pair. Entries
come back oldest first. It removes nothing.

#### snapshots gc

`recall snapshots gc [--days N] [--dry-run] [--yes]` prunes entries in
`<data_dir>/snapshots/` whose top-level mtime is older than `--days`
(default 7). Output reports `removed_paths`, `kept_paths`, `failed_paths`, and
`total_bytes_freed`; `failed_paths` lists stale entries that could not be
statted, removed, or verified absent. It operates only inside
`<data_dir>/snapshots/` and refuses symlinks or other paths that resolve outside
that directory. A missing snapshots directory is a no-op.

`recall compact` invokes the same GC with the default threshold after a
successful rebuild; the recall daemon invokes it once on startup. Both in-band
invocations are silent when nothing is removed.

### Indexing UX and Performance

#### Requirements

- `REQ-INDEX-001`: Human-readable `recall index` runs may emit compact progress updates to `stderr` showing processed, indexed, skipped, and failed counts. Progress output must not contaminate `--json`.
- `REQ-INDEX-002`: Incremental indexing must preload persisted session staleness metadata in bulk and compare candidate files in memory. The unchanged-file skip path must not execute one database lookup per candidate file.
- `REQ-INDEX-003`: Candidate file discovery for incremental indexing must capture `resolved_path`, `file_mtime`, and `file_size` once per file and reuse that metadata for staleness checks instead of restatting the same file in the hot path.
- `REQ-INDEX-004`: `recall index` may support worker concurrency for changed-file preparation. Concurrent workers operate only on parse and pre-write preparation; DuckDB writes, delete/replace semantics, and FTS rebuild remain serialized.
- `REQ-INDEX-005`: When worker concurrency is enabled, result ordering and persisted IDs must remain deterministic. Parallel preparation must not change session/message/tool-call identity or final row contents.
- `REQ-INDEX-006`: Worker concurrency is a power-user optimization, not a new runtime mode. The default CLI behavior is `workers=auto`, which resolves to a bounded worker count based on changed-file count and host CPU availability.
- `REQ-INDEX-007`: Progress reporting must remain low overhead. Non-interactive runs may disable live rendering. Structured runs keep stdout reserved for the final payload and may emit bounded progress only on stderr.
- `REQ-INDEX-008`: Reindexing an existing session must update mutable state in place and delete only removed rows. Existing-session rewrites must not depend on a foreign-key-sensitive full-session delete/reinsert retry path.
- `REQ-INDEX-009`: Rebuilding FTS indexes must be conditional on actual searchable-content changes. Metadata-only session rewrites must not trigger an FTS rebuild.
- `REQ-INDEX-010`: The parser protocol must support byte-offset incremental parsing via a single `parse(path, offset=0)` method that returns a `ParseResult` containing new messages, updated session metadata, and the next byte offset. Full parse is the `offset=0` case.
- `REQ-INDEX-011`: Session JSONL files are treated as append-only. Incremental parsing (offset > 0) must only append new messages and merge metadata — it must not detect or remove previously indexed messages.
- `REQ-INDEX-012`: If a session file's size is smaller than the stored `last_byte_offset`, the indexer must fall back to a full parse from offset 0 (file was truncated or rewritten).
- `REQ-INDEX-013`: File discovery for all indexing paths (batch `recall index`, daemon poll, watch catch-up) must sort candidates by `file_mtime` descending so that recently active sessions are indexed first.
- `REQ-INDEX-014`: Incremental indexing of a changed session must embed only newly parsed messages inline during the index write. Already-embedded messages from previous offsets must not be re-embedded. During the same incremental merge, token counts must be source-aware: cumulative/absolute-token sources (`ABSOLUTE_TOKEN_SOURCES`, currently Codex) merge via `GREATEST`/last-wins so re-seeing duplicate cumulative totals is idempotent; delta-token sources (Claude Code, Pi Agent) merge additively.
- `REQ-INDEX-015`: The `session_state` table must store `last_byte_offset` (BIGINT) to track parse resume position per session. A NULL or 0 value means the next index must perform a full parse.
- `REQ-INDEX-016`: Session metadata fields (`ended_at`, `duration_seconds`, `input_tokens`, `output_tokens`, `message_count`, `tool_count`) must be merged incrementally during offset-based parsing — accumulating delta counts, applying source-aware cumulative token semantics from `REQ-INDEX-014`, and extending time ranges without re-scanning earlier lines.
- `REQ-INDEX-017`: `insert_tool_call_embeddings` must be idempotent for unchanged rows: re-inserting a tool call whose stored `bash_embedding` is byte-identical must be a no-op (no row rewrite), while an absent or changed vector must still be written. A plain `INSERT OR REPLACE` rewrites every row on every call (REPLACE is DELETE+INSERT in DuckDB), so a session's carried-forward embeddings churned on each re-index and left a dead row-version whenever they were re-presented; because DuckDB reclaims a row group's blocks only once all its rows are dead, `tool_call_embeddings` accumulated ~115 physical versions per live row (mostly-dead 140 GiB) while `message_embeddings` stayed dense via `INSERT OR IGNORE`. Both the pyarrow and the executemany-fallback paths must apply the same guard, writing only rows where the target is absent or `bash_embedding IS DISTINCT FROM` the incoming vector.

- `REQ-INDEX-018`: The index change signal must include every sidecar file a parser enriches session metadata from. `SessionParser.sidecar_paths(path)` declares them (empty for sources with none), discovery records the newest sidecar mtime as `session_state.sidecar_mtime`, and `_is_unchanged` compares it alongside `file_mtime`/`file_size`. A NULL `sidecar_mtime` marks a row written before this column and must be treated as changed whenever the parser declares sidecars, so the row re-indexes exactly once and backfills; sources declaring none fingerprint to 0.0 and never churn. Because a sidecar can move `started_at` while the append-only stream cannot, the incremental merge writes `started_at = COALESCE(?, started_at)` as the one exception to REQ-INDEX-016. Without this, metadata sourced from a sidecar could never refresh: the session file stops changing when the session ends, so the row is skipped forever -- which left every Grok session indexed before sidecar support with NULL `started_at`/`ended_at` and no error anywhere.
- `REQ-INDEX-019`: An index run must count sessions that parsed without a `started_at` while the parser declared at least one sidecar, expose the per-source counts on `IndexSummary.metadata_drift`, and log them at WARNING. The warning is aggregated (one line per source per run, carrying the count and the total) and rate-limited per source, because a long-lived daemon re-indexes continuously and a per-session warning would bury the log. A present, parseable sidecar that yields no timestamp is the signature of an upstream format change; it is invisible at the row level and only detectable in aggregate.
- `REQ-INDEX-023`: A plain no-op `recall index` over the populated corpus on a reference host completes in under one minute. Catalog comparison is performed in bounded batches or bulk reads, and writer work scales with changed observations or bounded batches rather than the total candidate-file count. Scoped requests do not make one writer call for each unchanged file.
- `REQ-INDEX-024`: `recall index --json` reserves stdout for one valid final JSON payload while bounded progress remains observable on stderr. Progress is coalesced to at most one update per captured inventory batch plus phase transitions and the terminal update; it is never emitted once per unchanged file.
- `REQ-INDEX-025`: Raw reconciliation may resume normalization only from a checkpoint committed atomically with the catalog acknowledgement. The checkpoint envelope records its version, parser revision, captured file device/inode identity, next message and orphan-call indices, and adapter-owned normalization state. Its durable JSON representation is bounded to 64 KiB; adapter state that needs it is compressed from at most 4 MiB before that bound is applied, while an adapter state that still does not fit declines with an operator-visible warning. A legacy or malformed checkpoint, malformed compressed state, decompression overflow, adapter state beyond the 64-level nesting bound, use of the reserved compression key by an adapter, a negative captured file identity, parser-revision mismatch, missing committed prefix digest, captured file-identity mismatch, shrink below the committed offset, unsupported adapter state, or failed prefix verification falls back to a full parse without publishing speculative suffix rows. Full and resumed parses acknowledge only complete records, and a failed or stale generation leaves the prior checkpoint and indexed history intact. A suffix that diagnoses anything is discarded for the full reference path rather than published, and a resumed append whose session rows are missing is discarded the same way. A full parse whose only diagnostic is an unterminated tail still declares a checkpoint: the capture never yielded that record, so its committed offset, digest and adapter state all describe the last complete boundary. Because a diagnostic never acknowledges past itself, a resumed append reports `is_complete` for the bytes it re-read instead of latching the previous turn's verdict. The envelope is stored in a nullable `source_files.normalization_checkpoint` column added additively at schema version 31; NULL is the only value meaning "no resume proof", every pre-existing row reads NULL, and the acknowledgement statement is the only one that may write or advance it -- including clearing it when the adapter declined to resume, so a stale proof can never sit beside a newer committed offset. The explicit-reparse statement may also clear it, and only clear it: an explicit reparse raises the same desired generation an ordinary append raises, so the intent has to be durable or a turn that no longer holds the request answers the rebuild with a suffix. That statement leaves the committed offset and prefix digest untouched, so no proof is ever created or advanced outside an acknowledgement. A suffix acknowledgement may replace a proof only while the catalog still carries one; a full parse may always stamp the complete boundary it established.
- `REQ-INDEX-026`: Every supported source adapter declares whether its captured boundary is resumable and returns the state needed to normalize the next suffix equivalently to a full parse. Claude Code and Pi Agent have stateless boundaries; Codex carries turn-in-progress state; Grok falls back when reasoning or backend-tool state remains open, and also when the captured boundary's last message is an assistant message, because a following `backend_tool_call` amends that message's tool calls in place and a suffix result has no way to express that amendment; Kimi Code falls back while an assistant step remains open. Verified appends preserve deterministic identities, source-aware token merging, tool-result/stop-marker pairing, sidecar metadata refresh, content epoch, and idempotent retry. Rewrite, truncation, replacement, parser revision, malformed-tail, and unsupported-checkpoint tests exercise the full-parse fallback for every adapter. A fallback commits exactly the rows a from-scratch reconciliation of the same bytes commits -- including removing stop markers whose message indices the shortened transcript no longer has.
- `REQ-INDEX-027`: A transcript indexed again after its project directory moved supersedes the row left at its old path. Session ids hash the absolute path, so the moved file is a new session; the old row carries the same history. When a new session is first written, a local row of the same source whose path below `projects/<dir>/` or `sessions/<dir>/` is equal is superseded only when the move is proven: its path no longer exists, the successor's rows were built from at least that row's `committed_offset` bytes, and the new file's first `committed_offset` bytes hash to that row's catalog `committed_prefix_sha256` (at most 512 MiB is read). A successor whose rows lag the file proves nothing, however far the file has grown. The predecessor's removal, the re-linking of its `usage_events` to the successor, the clearing of its catalog `session_id`, the queueing of its keyword-search rows for deletion from the SQLite sidecar, and the successor's insert commit in one transaction; no sidecar row is deleted before that commit. Stored `llm-*` message context carries over by position when the old conversation is exactly a prefix of the new one, the same context an in-place append keeps; embeddings then carry over by text. A superseded session's id and message ids no longer resolve. More than 16 vanished rows sharing the key are not a move history and none is superseded. A copy indexed while its original still existed (a copy-then-delete move: cross-volume `mv`, `rsync` then `rm`) is proven when reconciliation's complete walk observes the original missing: before a page of the catalog is marked missing, each vanished path's row is checked against the one surviving local row of its key, whose committed catalog prefix its file must still begin with and must extend the vanished row's `committed_prefix_sha256`, the same proof `db supersede-moved` applies. One walk examines at most 64 such proofs and 16 MiB of survivor prefix (the proof that crosses the byte budget still runs, so an oversized transcript progresses); vanished paths it did not reach stay present so that a later walk can examine them, unless that later walk trips the guard below, which marks them missing without examining them; no walk revisits a path once it is marked missing. A proven predecessor's path is marked missing before its supersession is attempted, and each supersession commits in its own transaction: it carries stored `llm-*` context and matching embeddings as `db supersede-moved` does, and queues its keyword-search deletes and carried upserts set-based in that transaction. A supersession that fails is logged with the predecessor and successor ids and path, is never retried by a walk (not even after a process abort), keeps both rows, and does not stop the rest of the page or the walk. A walk that finds no files under a root, or that finds fewer than half of the root's present catalog rows, supersedes nothing (a copy-then-delete inside one root never loses more than half of it); its absences are marked as before. A pair left behind by a failed or aborted supersession, a guard-skipped walk, or a deferral that a guard-skipped walk then marked missing is never collapsed automatically; `recall db supersede-moved` is the recovery for all of them. A delete observed before the copy is indexed is proven by the copy's insert. A row whose only survivor had committed less than it when the delete was observed is not revisited; `db supersede-moved` collapses it. A copy whose original remains, a divergent or shorter file, an unrecorded prefix, and rows of other hosts keep every row. `recall db supersede-moved` applies the same proof to rows written before this rule, against the survivor's committed catalog prefix (which its file must still begin with) rather than its current file, and carries the predecessor's stored `llm-*` context and matching embeddings onto the survivor's message at the same position with the same text: it reports by default, removes with `--apply`, and refuses while another process holds the database. It queues no sidecar deletes; the daemon's startup reconciliation removes keyword-search rows whose ids are gone. Unknown `--fields` are rejected before anything is removed.
#### Invariants

- The incremental skip path is `O(n)` in discovered files with one bulk session-state load and in-memory comparisons.
- Parallel preparation never writes directly to the live database from worker threads/processes.
- `recall index --json` emits only the JSON payload on `stdout`.

#### Non-goals

- Parallel DuckDB writes from multiple workers.
- Real-time per-message or per-tool-call progress bars.
- Automatic concurrency tuning based on host CPU, GPU, or memory heuristics in the first implementation.

#### CLI Integration

```bash
recall index                             # Incremental index with human-readable progress
recall index --json                      # JSON summary on stdout, bounded progress on stderr
recall index --no-progress               # Suppress live progress updates
recall index --workers auto              # Default bounded worker selection
recall index --workers 8                 # Explicit concurrent changed-file preparation
recall index --embed --workers 4         # Concurrent parse/prep, serialized writes
```

#### Acceptance Criteria

- [ ] Incremental indexing of unchanged files uses one bulk session-state read rather than one DB lookup per candidate file.
- [ ] Human-readable `recall index` can display processed/indexed/skipped/failed progress without affecting the final summary line.
- [ ] `recall index --json` emits one valid JSON payload on stdout and bounded inventory progress on stderr (`REQ-INDEX-024`).
- [ ] Raw reconciliation resumes from an atomically committed, parser-compatible normalization checkpoint for every supported adapter; every invalidation condition falls back to the full reference path without changing observable rows (`REQ-INDEX-025`, `REQ-INDEX-026`).
- [ ] A moved-and-extended transcript collapses to one row with its usage re-linked; a move among many same-named transcripts is found; a successor indexed short of the old history, and a survivor whose rows lag its file, keep the old row; stored summaries carry over; superseded rows are queued for sidecar deletion; a copy indexed before its original is deleted is superseded when a complete walk observes the delete, with its summaries and usage carried and its sidecar deletes queued; a failed supersession is given up while the walk finishes; a walk supersedes a bounded number (by proofs and by bytes) and the next continues; a walk that finds nothing or loses most of its root supersedes nothing; a lagging copy or two surviving copies keep every row; a delete observed before the copy is indexed supersedes on insert; a remaining original, a divergent file, and a failed successor insert keep the old row; `db supersede-moved` reports, applies, and is then a no-op (`REQ-INDEX-027`, `tests/test_services/test_moves.py`).
- [ ] Three fresh-process equal-append samples compare small and at least 33 MiB histories: the large-history candidate median is at most 1.0 CPU-second, at least 5x below the retained 5.9 CPU-second baseline, and no more than `3 * small_median + 0.25` CPU-second; all raw-commit turns remain below the standing five-second ceiling.
- [ ] `recall index --workers auto` resolves to a bounded positive worker count and preserves deterministic IDs and row contents relative to a single-worker run.
- [ ] Worker concurrency does not bypass the advisory lock or introduce concurrent DuckDB writes.
- [ ] Existing-session rewrites do not rely on the foreign-key-sensitive transactional delete/reinsert retry path.
- [ ] Metadata-only session rewrites do not rebuild FTS indexes.
- [ ] Byte-offset incremental parse of a changed session reads only new bytes, not the entire file.
- [ ] A session with 500 existing messages and 1 new append embeds only 1 message, not 501.
- [ ] File discovery sorts by mtime descending; the most recently modified session is indexed first.
- [ ] Read-only CLI commands (`search`, `list`, `show`, `stats`) succeed while the daemon is idle between writes.
- [ ] Daemon write operations release the database file lock between per-session index operations.
- [ ] `session_state.last_byte_offset` is persisted and used on subsequent incremental index of the same session.
- [ ] A truncated session file (size < stored offset) triggers a full re-parse from byte 0.

### Bash Command Parsing

**Strategy:** First command with subcommand extraction

```
Input: "git commit -m 'msg' && git push"
→ bash_base: "git"
→ bash_sub: "commit"
→ is_compound: true

Input: "kubectl get pods -n default"
→ bash_base: "kubectl"
→ bash_sub: "get"
→ is_compound: false

Input: "cat file.txt | grep error"
→ bash_base: "cat"
→ bash_sub: null
→ is_compound: true
```

**Recognized subcommand tools:** git, kubectl, docker, npm, yarn, pnpm, cargo, go, uv, pip, brew, apt, systemctl

### Thinking Content

**Storage:** Separate `thinking` column in messages table
- Full thinking content preserved for search
- `has_thinking` boolean for quick filtering
- Enables search across reasoning when needed

### Full-Text Search

DuckDB's FTS extension provides keyword-based search using inverted indexes and Okapi BM25 scoring.

**Indexed fields (configurable, all enabled by default):**
- Message content (`message_state.content`)
- Thinking content (`message_state.thinking`)
- Bash commands (`tool_calls.bash_command`)

**Configuration:**

Environment variable: `RECALL_FTS_FIELDS`
- Comma-separated list of fields to index
- Values: `content`, `thinking`, `bash`
- Default: `content,thinking,bash` (all enabled)
- Example: `RECALL_FTS_FIELDS=content,bash` (skip thinking)

Config file (`~/.config/recall/config.toml`):
```toml
[fts]
fields = ["content", "thinking", "bash"]
```

**Known limitation (sqlite_sidecar backend):** changing `fts.fields` re-scopes the
existing SQLite sidecar index on the next **daemon restart** (the daemon detects
a field-signature change at startup and re-scopes already-indexed rows). It does
NOT auto-re-scope on an ad-hoc `recall search` / `recall index` against a still-
running daemon's data. Disabling a field takes effect immediately for search
*results* (query-time field scoping restricts matches), but the excluded content
remains stored in the sidecar until a re-scope runs; enabling a previously
excluded field returns no hits for pre-existing rows until re-scope. To force a
re-scope without a daemon restart, run `recall index --full`.

**Known limitation (sidecar consistency after an indexing failure):** per
`REQ-FTS-SIDECAR-006`/`-007` the sidecar is best-effort and eventually consistent,
not transactionally atomic with DuckDB. When an indexing operation rolls back a
session's DuckDB write after its sidecar rows were already mutated (a per-session
failure inside `index_sessions`/`index_single_session`, or a daemon cycle failure
before the DB swap), the affected entity ids are enqueued to `fts_sidecar_pending`
and the live sidecar is healed by the next reconciliation pass (daemon cycle/
startup, or `run_cycle`'s post-swap heal when a cycle had failures) — NOT
instantaneously. A search issued in the brief window between the failure and the
next reconcile may see stale sidecar content; the DuckDB store remains
authoritative and the drift is always healed (no permanent drift).

### DuckDB Runtime

Recall-managed DuckDB connections, including compaction estimate and rebuild
handles, apply a `memory_limit` on open for both read-write and read-only handles.
By default, recall uses a `2GB` buffer allowance, leaving process memory for
source captures, result vectors and native allocations. The buffer allowance
is not a total-RSS limit; the reconciliation resource gate measures total RSS.
Explicit environment/config values override this default.

Environment variable: `RECALL_DUCKDB_MEMORY_LIMIT`
- Positive DuckDB-style memory size such as `32GB`, `500MB`, or `1TB`
- Overrides the config file for one-shot commands and daemon launches
- Invalid values fail connection startup instead of falling back silently

Config file (`~/.config/recall/config.toml`):
```toml
[duckdb]
memory_limit = "32GB"
temp_directory = "/path/to/big/disk/duckdb_spill"
```

- `REQ-RECALL-DUCKDB-TEMP-DIR-001`: All recall-managed DuckDB connections MUST
  apply a `temp_directory` pragma immediately after `memory_limit`, so
  out-of-memory operations can spill to disk instead of failing the query.
- `REQ-RECALL-DUCKDB-TEMP-DIR-002`: The default spill path MUST be
  `<data_dir>/duckdb_spill`, and recall MUST create it with
  `mkdir(parents=True, exist_ok=True)` before applying the pragma.
- `REQ-RECALL-DUCKDB-TEMP-DIR-003`: Spill path override precedence MUST be
  `RECALL_DUCKDB_TEMP_DIR` environment variable, then `[duckdb] temp_directory`
  config, then the default path. Empty or whitespace-only override values MUST
  fail resolution with a visible error instead of falling back silently.
- `REQ-RECALL-DUCKDB-TEMP-DIR-004`: `db/connection.py` MUST provide a shared
  `_apply_runtime_pragmas` helper that applies both `memory_limit` and
  `temp_directory`, and all four DuckDB connect callsites (`connect`,
  `connect_readonly`, compaction bloat estimation, and compaction rebuild) MUST
  route through it.

**FTS rebuild OOM backoff:**

The DuckDB `memory_limit` knob is the primary mitigation for FTS rebuild memory
pressure. If a rebuild still raises an out-of-memory error in watch mode, recall
uses backoff as a last-resort guardrail so the daemon remains usable and the
failure is visible in status output.

- Initial backoff is `60s`.
- Each consecutive FTS rebuild OOM doubles the retry window.
- Backoff is capped at `3600s`.
- A successful FTS rebuild resets the active backoff counter to `0`.

`recall daemon status` reports the latest watch-mode FTS rebuild OOM state:

- `last_fts_rebuild_failure_at: datetime | null` — wall-clock timestamp of the
  latest FTS rebuild OOM, or `null` when none has occurred.
- `last_fts_rebuild_failure_reason: string | null` — latest OOM reason, or
  `null` when none has occurred.
- `fts_rebuild_consecutive_failures: int` — consecutive FTS rebuild OOM count.
- `fts_rebuild_next_retry_at: datetime | null` — wall-clock retry time while
  backoff is active, or `null` outside an active backoff window.

**Index creation:**

The FTS extension is autoloaded when the PRAGMA is called. Indexes are created per-table:

```sql
-- Messages index (content + thinking in one index)
PRAGMA create_fts_index(
    message_state,      -- table
    message_id,         -- document identifier column
    content, thinking,  -- columns to index
    stemmer = 'porter',
    stopwords = 'english',
    overwrite = 1
);

-- Tool calls index (bash commands only)
PRAGMA create_fts_index(
    tool_calls,
    id,
    bash_command,
    stemmer = 'porter',
    stopwords = 'english',
    overwrite = 1
);
```

**FTS parameters:**
- `stemmer`: Word stemming algorithm (`porter` default, `none` to disable)
- `stopwords`: Common words to ignore (`english` default)
- `ignore`: Regex for characters to skip (default: `(\\.|[^a-z])+`)
- `strip_accents`: Convert accented chars (default: 1)
- `lower`: Convert to lowercase (default: 1)
- `overwrite`: Replace existing index (default: 0)

**Search queries:**

Creating an index generates a `match_bm25` macro in schema `fts_main_<table>`:

```sql
-- Search messages
SELECT m.*, ms.*, fts_main_message_state.match_bm25(
    ms.message_id,
    'search query here',
    fields := 'content'  -- or 'content,thinking', or NULL for all
) AS score
FROM messages m
JOIN message_state ms ON ms.message_id = m.id
WHERE score IS NOT NULL
ORDER BY score DESC
LIMIT 10;

-- Search bash commands
SELECT tc.*, fts_main_tool_calls.match_bm25(
    tc.id,
    'git commit',
    fields := 'bash_command'
) AS score
FROM tool_calls tc
WHERE score IS NOT NULL
ORDER BY score DESC;
```

**BM25 parameters:**
- `k` (default 1.2): Term frequency saturation
- `b` (default 0.75): Document length normalization
- `conjunctive` (default 0): Set to 1 to require all keywords match

**Tool input extraction:**
- Bash commands are extracted to `tool_calls.bash_command` for FTS
- Other tool inputs remain in JSON (`tool_input`) and are not FTS-indexed
- Query tool inputs via JSON functions: `tool_input->>'file_path'`

#### Incremental FTS column refresh

`refresh_message_fts_columns()` in `db/queries.py` currently runs an
unconditional full-table UPDATE that rewrites every row's `fts_content`
and `fts_thinking` columns from `content`, `thinking`, and
`context_text` on every FTS rebuild trigger. With 1M+ messages the
UPDATE is internally a DELETE+INSERT inside DuckDB, doubles the row
footprint mid-operation, and dominates the memory pressure that pushes
`PRAGMA create_fts_index` over the `memory_limit` ceiling on large DBs.
The derived FTS columns are pure functions of source columns that only
change on insert / sync, so a row-incremental refresh is sound.

- `REQ-FTS-INC-001`: `message_state.fts_content` and
  `message_state.fts_thinking` MUST be populated at row write time
  (`_insert_messages`, `_upsert_messages`) as
  `COALESCE(context_text, '') || COALESCE(content, '')` and
  `COALESCE(context_text, '') || COALESCE(thinking, '')` respectively,
  so the derived columns are always in sync with their source columns
  without a separate refresh pass.
- `REQ-FTS-INC-002`: `refresh_message_fts_columns()` MUST become an
  idempotent backfill that only updates rows where `fts_content IS NULL`
  OR `fts_thinking IS NULL` OR `fts_content` does not match the
  derivation rule above (the existing unconditional UPDATE form
  pre-`REQ-FTS-INC-001` is disallowed). On a fully-populated DB the
  backfill MUST touch zero rows and report a count so callers can
  observe convergence.
- `REQ-FTS-INC-003`: First daemon run after upgrading from a pre-`REQ-FTS-INC-001`
  DB MUST schedule one backfill pass via
  `refresh_message_fts_columns()` so any pre-existing rows missing the
  derived columns are populated. The backfill MUST run inside the
  existing FTS rebuild backoff (`fts_rebuild_consecutive_failures`) so
  an OOM during backfill does not poison the daemon.
- `REQ-FTS-INC-004`: PRAGMA `create_fts_index(... overwrite=1)` over
  `message_state` and `tool_calls` is still monolithic per DuckDB
  extension limitations and MUST NOT be claimed as incremental. The
  scope of `REQ-FTS-INC-*` is the per-row column refresh only; reducing
  the inverted-index rebuild cost itself is tracked separately as a
  recency-partitioned FTS design (out of scope for this requirement
  family).

Non-goals for this requirement family:
- Changing the DuckDB FTS extension's drop-and-rebuild behaviour.
- Adding a hash column or content fingerprint — write-site population
  is sufficient because `_insert_messages` and `_upsert_messages` are
  the only paths that mutate the source columns.
- Partitioning the index by recency or source. Tracked as future work
  once `REQ-FTS-INC-*` lands and the next OOM root cause surfaces.

#### SQLite FTS5 Sidecar (target architecture)

The DuckDB FTS extension is structurally rebuild-only: `PRAGMA create_fts_index(... overwrite=1)` drops and rebuilds the entire inverted index on every refresh, and the extension's term-position materialize+sort pipeline does not reliably spill to `temp_directory`. On the 30+ GB recall DB at 1M+ messages this consistently exhausts the DuckDB `memory_limit` ceiling; the rebuild-OOM backoff above is a last-resort guardrail, not a fix. As of DuckDB 1.5.3 (May 2026) the FTS extension is out-of-tree at `duckdb/duckdb-fts` with no tagged releases and no public roadmap for incremental indexing.

SQLite FTS5 supports proper incremental insert/update/delete and offers contentless-delete tables (`content=''` plus `contentless_delete=1`) that avoid duplicating row text. recall keeps DuckDB as the source-of-truth columnar store and moves only the inverted index into a sidecar SQLite database (`recall.fts.sqlite`) alongside `recall.duckdb`. Search queries hit the sidecar for FTS scores and join back to DuckDB by `message_id` / `tool_call_id`.

**Risk tags**: schema migration (new sidecar file + `schema_version` bump); search-behavior change (FTS5 BM25 ranking differs from DuckDB FTS); transitional dual-write path; coupling to `sqlite_scanner` for the in-DuckDB join path. Requires PLAN-gate approval before implementation.

- `REQ-FTS-SIDECAR-001`: The inverted index for `message_state` (over `fts_content` and `fts_thinking`) and `tool_calls` (over `bash_command`) MUST be served by a SQLite FTS5 sidecar database at `<data_dir>/recall.fts.sqlite` when `[fts] backend = "sqlite_sidecar"` is active. During the transition release defined by `REQ-FTS-SIDECAR-015`, the legacy DuckDB FTS callsites, rebuild path, and `fts_main_*` schemas MUST remain available so `backend = "duckdb"` can preserve current behavior. Only in the later sidecar-only release that removes the fallback flag may the DuckDB-side `PRAGMA create_fts_index(...)` callsites and `fts_main_*` schemas be removed; after that removal no code path may rebuild a DuckDB FTS schema.

- `REQ-FTS-SIDECAR-002`: The sidecar MUST use FTS5 contentless-delete mode (`content=''`, `contentless_delete=1`). Source text is not duplicated into the sidecar; only the inverted index is stored. Before creating or opening the sidecar schema, recall MUST run a runtime capability probe against the active stdlib `sqlite3` module that verifies SQLite is at least 3.43.0 and can create an FTS5 table with `contentless_delete=1`. If the probe fails while the transition fallback still exists, `backend = "sqlite_sidecar"` MUST fail loudly with a clear message that names the missing SQLite capability and points users to `backend = "duckdb"` or a supported Python/SQLite runtime; after the fallback is retired, unsupported runtimes are outside recall's supported platform matrix. Tokenization MUST use the `porter unicode61` tokenizer chain so stemming and accent-stripping match the DuckDB FTS configuration. SQLite FTS5 has no built-in stopword filtering equivalent to DuckDB FTS's `stopwords='english'`; BM25 normalization is relied upon to deprioritize common terms, and the resulting match-set and ranking drift is the documented compatibility exception in `REQ-FTS-SIDECAR-010`.

- `REQ-FTS-SIDECAR-003`: The sidecar MUST maintain a stable `rowid → id` mapping table for each indexed entity:
  - `message_fts_rowid(rowid INTEGER PRIMARY KEY, message_id TEXT NOT NULL UNIQUE)`
  - `tool_calls_fts_rowid(rowid INTEGER PRIMARY KEY, tool_call_id TEXT NOT NULL UNIQUE)`

  The rowid is the FTS5 docid. It MUST be stable across the row's lifetime so updates rewrite the same FTS5 row instead of creating duplicates. Rowids are assigned monotonically on first insert and never reused.

- `REQ-FTS-SIDECAR-004`: Every write site that today populates `message_state.fts_content` / `message_state.fts_thinking` (`REQ-FTS-INC-001`: `_insert_messages`, `_upsert_messages`) MUST also write the corresponding FTS5 rows in the sidecar within the same logical batch. Because SQLite UPSERT is not implemented for virtual tables, updates MUST use SQLite-supported FTS5 contentless-delete operations: either `UPDATE message_fts SET fts_content = ?, fts_thinking = ? WHERE rowid = ?` with all indexed columns supplied, followed by `INSERT INTO message_fts(rowid, fts_content, fts_thinking) VALUES (...)` when no row exists, or `INSERT OR REPLACE INTO message_fts(rowid, fts_content, fts_thinking) VALUES (...)`. The same rule applies to `_insert_tool_calls` and `tool_calls_fts`.

- `REQ-FTS-SIDECAR-005`: Deletes of `message_state` / `tool_calls` rows MUST issue the corresponding `DELETE FROM message_fts WHERE rowid = ?` (or `tool_calls_fts`) in the sidecar, including cascading deletes from `delete_session()`. Failures to delete a sidecar row are recorded for reconciliation (`REQ-FTS-SIDECAR-007`) and MUST NOT block the DuckDB delete.

- `REQ-FTS-SIDECAR-006`: Sidecar writes MUST be best-effort with respect to DuckDB commits. The DuckDB transaction is authoritative; if a sidecar write fails after the DuckDB commit succeeds, the affected entity ids MUST be appended to a `fts_sidecar_pending(kind TEXT, id TEXT, op TEXT, queued_at TIMESTAMP)` table inside the DuckDB store, the failure logged at WARN, and indexing MUST continue. Cross-store two-phase commit is explicitly out of scope.

- `REQ-FTS-SIDECAR-007`: On daemon start and on the FTS refresh trigger, recall MUST run a bounded reconciliation pass that:

  (a) drains the `fts_sidecar_pending` queue, retrying the corresponding sidecar inserts / updates / deletes;
  (b) detects orphans by anti-join between `message_state.message_id` and `message_fts_rowid.message_id` (and `tool_calls.id` ↔ `tool_calls_fts_rowid.tool_call_id`), backfilling any missing entries in bounded-memory batches;
  (c) detects ghosts (sidecar rows without a matching DuckDB row) and deletes them.

  Reconciliation MUST be incremental — it MUST NOT rebuild the entire FTS5 index unless the sidecar file is missing or corrupt. Per-pass row counts MUST be reported through `recall daemon status`.

- `REQ-FTS-SIDECAR-008`: First daemon run after upgrade from a pre-sidecar DB MUST bootstrap the sidecar by streaming `(message_id, fts_content, fts_thinking)` and `(tool_call_id, bash_command)` rows from DuckDB into FTS5 in bounded-memory batches (≤10k rows per transaction). Bootstrap MUST be idempotent — interrupting and re-running picks up from the last successfully committed batch, tracked in `runtime_state`. Bootstrap progress MUST be exposed through `recall daemon status` so users can observe completion on large DBs.

- `REQ-FTS-SIDECAR-009`: `services/search.py` MUST query the sidecar for FTS hits and JOIN back to DuckDB by `message_id` / `tool_call_id`. Two integration paths are sanctioned:

  (a) Default: `search.py` MUST perform a two-step query — read hits + scores from the sidecar via a direct `sqlite3` connection, then `WHERE id IN (...)` against DuckDB. This path is the compatibility baseline because the locked DuckDB/sqlite_scanner stack does not currently prove support for FTS5 `contentless_delete=1` introspection or `MATCH` predicates over attached SQLite FTS5 virtual tables.
  (b) Optional optimization: `ATTACH '<data_dir>/recall.fts.sqlite' AS fts (TYPE sqlite)` via the `sqlite_scanner` extension MAY be used only after a concrete DuckDB/sqlite_scanner version and SQL query shape have been checked into implementation tests and verified to support the configured FTS5 schema and ranking query.

  If the optional DuckDB/sqlite_scanner path is enabled, it MUST yield identical result sets and ordering to the direct-`sqlite3` path for the same query.

- `REQ-FTS-SIDECAR-010`: Hybrid search (`REQ-SEM-001`, `REQ-SEM-011`) MUST consume FTS scores from the sidecar through the same RRF fusion and tie-breaking semantics used for DuckDB `match_bm25` scores. The CLI shape, flags, filters, JSON schema, vector-only behavior, and deterministic RRF ordering contract from `REQ-SEM-011` MUST be preserved. Keyword and hybrid result sets MAY differ from the legacy DuckDB FTS path when the difference is caused by FTS5 tokenizer, stopword, or BM25 behavior described in `REQ-FTS-SIDECAR-002`; compatibility is judged by the top-K overlap acceptance criterion, not by exact match-set, ordering, or score parity.

- `REQ-FTS-SIDECAR-011`: `recall snapshots` MUST snapshot `recall.duckdb` and `recall.fts.sqlite` together as a consistent pair. Restoring a snapshot MUST restore both files atomically; a snapshot containing only one of the two files is invalid and MUST be refused with a clear error.

- `REQ-FTS-SIDECAR-012`: `recall compact` MUST issue SQLite FTS5 `INSERT INTO message_fts(message_fts) VALUES('optimize')` (and the equivalent for `tool_calls_fts`) as part of its sweep so the sidecar's internal b-trees stay merged. The DuckDB `CHECKPOINT` step is unchanged.

- `REQ-FTS-SIDECAR-013`: During the transition release defined by `REQ-FTS-SIDECAR-015`, successful sidecar bootstrap (`REQ-FTS-SIDECAR-008`) MUST NOT drop the legacy `fts_main_*` schemas because the `backend = "duckdb"` fallback still depends on them. In the later sidecar-only release that removes the fallback flag, the daemon MUST drop the legacy `fts_main_*` schemas from `recall.duckdb` after sidecar bootstrap is complete to reclaim the storage occupied by the old inverted index. The drop MUST be reported in `recall daemon status` as a one-time event with the bytes reclaimed.

- `REQ-FTS-SIDECAR-014`: The FTS rebuild OOM backoff fields (`last_fts_rebuild_failure_at`, `last_fts_rebuild_failure_reason`, `fts_rebuild_consecutive_failures`, `fts_rebuild_next_retry_at`) become inert under the sidecar architecture because SQLite FTS5 writes are bounded-memory by construction. recall MUST continue to expose these fields with `null` / `0` defaults for status-output backward compatibility for at least one minor release, then MAY remove them.

- `REQ-FTS-SIDECAR-015`: A `[fts] backend = "duckdb" | "sqlite_sidecar"` config knob and corresponding `RECALL_FTS_BACKEND` env var MUST be honored for one transition release so the sidecar can be A/B tested against the DuckDB FTS path on production data. Default is `sqlite_sidecar`. Both paths share the same write-site population rules from `REQ-FTS-INC-001`. While the flag exists, selecting `backend = "duckdb"` MUST use the existing DuckDB FTS rebuild/query path and MUST NOT require the sidecar file or its bootstrap state. The flag MUST be removed in the same later sidecar-only release that retires the DuckDB FTS path and performs the `REQ-FTS-SIDECAR-013` schema drop.
- `REQ-FTS-SIDECAR-016`: Every sidecar query path MUST escape the user query into FTS5 terms/phrases before it reaches `MATCH`, never passing the raw string through. This applies to the default (no `--field`) and `content`+`thinking` message paths and the tool-call path (`search_tool_calls_fts`), not only single-field message queries. An ordinary identifier containing FTS5 operator characters (`-`, `:`, `(`, `)`) — e.g. `REQ-BRIDGE`, `foo:bar`, `foo(bar)` — MUST be searched as terms (`"REQ" "BRIDGE"`) and MUST NOT leak a raw SQLite `no such column` / `fts5: syntax error`. Escaping MUST preserve the prefix (`foo*`) and quoted-phrase (`"..."`) operators. A query that escapes to the empty string returns no results (no error). The DuckDB FTS path tokenizes its bound query argument and is unaffected.

**Invariants:**

- `message_fts_rowid.message_id` is a subset of `message_state.message_id` after reconciliation.
- `tool_calls_fts_rowid.tool_call_id` is a subset of `tool_calls.id` after reconciliation.
- The sidecar contains no source row text; FTS5 always runs in `content=''` mode.
- Hybrid search results are deterministic for a fixed sidecar index and query. Exact top-K equivalence with DuckDB FTS is not invariant because FTS5 tokenizer, stopword, and BM25 differences may change both candidate sets and ordering.
- The `fts_sidecar_pending` queue is empty after a successful reconciliation pass.

**Non-goals:**

- Cross-store two-phase commit. Best-effort dual write plus reconciliation is sufficient given DuckDB owns the truth.
- Exact match-set, ordering, or BM25 score parity with DuckDB FTS. Top-K overlap is the compatibility bar; identical results are not achievable across implementations, tokenizer chains, and stopword behavior.
- Exposing the sidecar as a public CLI surface. The sidecar file is an implementation detail.
- Recency- or source-partitioned FTS5 indexes. SQLite FTS5's incremental insert/update makes partitioning unnecessary at recall's projected scale; if a future ceiling appears, partitioning would be a separate REQ family.
- Supporting runtimes whose stdlib `sqlite3` module lacks FTS5 or SQLite 3.43.0+ contentless-delete support. Python version alone is not the compatibility check; the runtime probe in `REQ-FTS-SIDECAR-002` is authoritative.

**Acceptance criteria:**

- [ ] `recall.fts.sqlite` is created at first daemon start under the sidecar backend, containing `message_fts`, `tool_calls_fts`, `message_fts_rowid`, and `tool_calls_fts_rowid`.
- [ ] Sidecar startup probes the active `sqlite3` runtime and refuses `backend = "sqlite_sidecar"` with a clear error before schema creation when FTS5 or `contentless_delete=1` support is unavailable.
- [ ] Inserting a new message via `_insert_messages` writes one row to `message_state` and one to `message_fts` + `message_fts_rowid` in the same logical batch.
- [ ] Upserting an existing message via `_upsert_messages` updates the FTS5 row in place — no rowid churn, no duplicate FTS docs.
- [ ] `delete_session()` removes all matching rows from both stores; the sidecar has no ghost rows after a session delete.
- [ ] A simulated sidecar-write failure logs the affected ids into `fts_sidecar_pending` and the next reconciliation pass drains the queue.
- [ ] Bootstrap from a 30+ GB DuckDB DB completes within a single daemon process without OOM; peak memory per batch stays under 1 GB.
- [ ] `recall search --mode keyword` and `recall search --mode hybrid` each return top-10 overlap ≥ 8/10 with the legacy DuckDB FTS path for 95% of queries on a representative 100-query suite; exact match-set, ordering, and score parity are not required.
- [ ] `recall search --mode hybrid` produces deterministic results per `REQ-SEM-011`.
- [ ] `recall snapshots gc` and `recall compact` operate on `recall.duckdb` and `recall.fts.sqlite` together; partial snapshots are refused.
- [ ] `recall daemon status` exposes bootstrap progress, reconciliation counts, and the inert FTS rebuild OOM fields per `REQ-FTS-SIDECAR-014`.
- [ ] Setting `RECALL_FTS_BACKEND=duckdb` for the transitional release falls back to the legacy DuckDB FTS path with current behavior preserved, including after the default sidecar backend has bootstrapped.
- [ ] During the transitional release, sidecar bootstrap does not drop `fts_main_*`; after the later `REQ-FTS-SIDECAR-013` sidecar-only drop, `SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name LIKE 'fts_main_%'` returns 0.

### Semantic Search

Embedding-based vector search enables semantic similarity matching beyond keyword overlap. Combined with FTS via Reciprocal Rank Fusion (RRF), this provides hybrid search that handles both exact keyword matches and semantic intent.

#### Requirements

- `REQ-SEM-001`: `recall search --mode auto` must select `hybrid` when any searchable embeddings exist (`message_embeddings.content_embedding`, `message_embeddings.thinking_embedding`, or `tool_call_embeddings.bash_embedding`); otherwise it must fall back to `keyword`.
- `REQ-SEM-002`: Message vector search must consider both `content_embedding` and `thinking_embedding`. When both are present for the same message, the message score is the maximum cosine similarity of the two fields for the query.
- `REQ-SEM-003`: The configured embedding model must be validated against storage and runtime constraints before embeddings are written. Embedding dimensions are configurable and stored in `runtime_state.embedding_dimensions`. Dimension changes require `recreate_embedding_tables()` (see `REQ-SCHEMA-005`). The MLX backend supports only BERT-family encoder models that expose HuggingFace safetensors weights. The ONNX backend supports only BERT-family encoder models that expose an ONNX model file (`onnx/model.onnx` or `model.onnx`) on HuggingFace Hub.
- `REQ-SEM-004`: `embed_session()` must behave atomically with respect to the in-memory `Session` object. If embedding generation fails for any field type, the session must not be left partially mutated with a subset of embedding fields populated.
- `REQ-SEM-005`: Embedding writes use dedicated tables (`message_embeddings`, `tool_call_embeddings`) via `INSERT ... ON CONFLICT DO UPDATE`. Since embedding tables are separate from content tables and have no FK constraints, embedding failures do not risk corrupting content data.
- `REQ-SEM-006`: Embedding generation must reuse persisted vectors for repeated text instead of recomputing them on every index pass. Reuse keys are field-specific: message `content` and `thinking` reuse exact text; bash command reuse is based on normalized command text. The `embedding_cache` table provides cross-session dedup; incremental index writes only embed messages parsed from new byte offsets.
- `REQ-SEM-007`: Bash command normalization must remove high-cardinality ephemeral tokens that do not materially change command intent, including pod/container IDs, Git SHAs, timestamps, temp paths, and numeric identifiers. Normalization must be deterministic and versionable.
- `REQ-SEM-008`: Bash embedding must operate on distinct normalized command texts and fan each resulting vector out to all matching `tool_calls` within a session.
- `REQ-SEM-009`: _Removed — embedding is inline during indexing; no separate backfill command._
- `REQ-SEM-010`: The MLX backend must prefer an already-cached local HuggingFace snapshot and avoid network metadata checks on steady-state vector or hybrid search when the model is already present locally.
- `REQ-SEM-011`: Hybrid search ranking must be deterministic when fused RRF scores tie. Tie-breaking must prefer higher raw component scores before falling back to a stable document key.
- `REQ-SEM-012`: Search ranking must downweight boilerplate command-wrapper and system-prompt messages relative to direct conversational content or concrete tool calls for the same query.
- `REQ-SEM-013`: On hosts that support the configured embedding backend, `recall index` must default to embedding inline during indexing. Embeddings are generated per-session as part of the parse → embed → write pipeline. No separate backfill pass exists; `recall index --full` re-embeds all sessions.
- `REQ-SEM-014`: Default embedding enablement must be based on a cheap local support probe for the configured backend. The probe must validate platform/runtime prerequisites and required local Python dependencies without loading model weights or performing network I/O.

#### Invariants

- All persisted embedding vectors match the configured `embedding_dimensions` stored in `runtime_state`.
- Message vector search returns at most one result row per message ID.
- Embedding writes are idempotent via `INSERT ... ON CONFLICT DO UPDATE` on dedicated embedding tables.
- Cached embedding keys are stable for a given normalization version.
- Inline bash embedding never embeds the same normalized command text more than once per session.
- On unsupported hosts, default indexing behavior remains keyword-only unless the user explicitly requests embeddings.

#### Non-goals

- Supporting embedding dimension changes without `recreate_embedding_tables()` (a lightweight rebuild of embedding tables only).
- Supporting non-BERT model families in the MLX backend.
- Weighting `content` and `thinking` differently during message vector scoring.
- Cross-model embedding reuse across incompatible embedding backends or dimensions.

#### Embedding Backend Protocol

Embedding generation is pluggable via a backend Protocol. Backends handle model loading, tokenization, and inference for a specific runtime.

```python
class EmbeddingBackend(Protocol):
    @property
    def dimensions(self) -> int: ...

    @property
    def model_id(self) -> str: ...

    @property
    def query_prefix(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

#### Backends

| Backend | Platform | Acceleration | Dependencies |
|---------|----------|-------------|--------------|
| `mlx` (default on macOS, priority 10) | macOS (Apple Silicon) | Metal GPU | `mlx`, `tokenizers`, `huggingface-hub` |
| `onnx` (cross-platform fallback, priority 20) | Any | CPU | `onnxruntime`, `tokenizers`, `huggingface-hub` |
| `openai` (planned) | Any (requires API key) | Remote API | `httpx` |

Default backend: `auto` selects the highest-priority available backend. MLX is preferred on Apple Silicon (priority 10); ONNX serves as a cross-platform CPU fallback (priority 20). On hosts with no available backend, default indexing remains keyword-only.

#### Default Model

`BAAI/bge-small-en-v1.5` (384 dimensions, BERT-family):
- Default embedding dimension; configurable via model selection (see `REQ-SCHEMA-005`)
- CLS token pooling + L2 normalization
- Query prefix: `"Represent this sentence for searching relevant passages: "`
- Document embedding: no prefix
- Weights loaded from HuggingFace Hub in safetensors format

#### Configuration

Environment variables:
- `RECALL_EMBED_BACKEND` — backend selection (`mlx`, `onnx`, `openai`)
- `RECALL_EMBED_MODEL` — model identifier (default: `BAAI/bge-small-en-v1.5`). The model must be a public HuggingFace BERT-family encoder. Embedding dimensions are determined by the model's `hidden_size` and stored in `runtime_state`; dimension changes trigger `recreate_embedding_tables()` automatically. MLX requires `model.safetensors` weights; ONNX requires an `onnx/model.onnx` or `model.onnx` file.
- `RECALL_EMBED_BATCH_SIZE` — batch size for embedding generation (default: `64`)

Config file (`~/.config/recall/config.toml`):
```toml
[embedding]
backend = "mlx"
model = "BAAI/bge-small-en-v1.5"
batch_size = 64
```

Environment variables override config file values.

#### Embedding Pipeline

Embedding is always inline during indexing — there is no separate backfill command. The unified pipeline is: discover → parse (byte-offset) → embed new messages → write → FTS.

**During indexing (`recall index`, with embeddings enabled by default on supported hosts or via `--embed`):**
1. Parse session (full or incremental via byte offset)
2. For each message with content: generate `content_embedding`
3. For each message with thinking: generate `thinking_embedding`
4. For each tool_call with bash_command: generate `bash_embedding`
5. Insert session with embeddings populated

For incremental parsing (offset > 0), only newly parsed messages are embedded — previously embedded content is untouched.

**Re-embedding all (`recall index --full`):**
Re-parses and re-embeds all sessions. Use after changing the embedding model. `--no-embed` opts out of embedding for keyword-only indexing.

#### Hybrid Search (RRF)

Reciprocal Rank Fusion merges rankings from multiple retrieval methods without requiring score normalization:

```
RRF_score(d) = Σ_{r∈R} 1/(k + rank_r(d))
```

- `k = 60` (standard constant)
- `R = {bm25, vector}` — the set of rankers
- `rank_r(d)` is 1-indexed position of document `d` in ranker `r`'s output
- Documents appearing in only one ranker still receive a score from that ranker
- No parameter tuning required (unlike weighted linear combination)

**Search modes (`recall search --mode`):**

| Mode | Behavior |
|------|----------|
| `auto` (default) | Use `hybrid` when embeddings exist, fall back to `keyword` |
| `keyword` | BM25 full-text search only |
| `vector` | Cosine similarity only (requires embeddings) |
| `hybrid` | RRF fusion of BM25 + vector rankings |

**Ranking rules:**
- RRF remains the primary fusion score.
- When two results have the same fused score, ranking falls back to the highest available raw BM25/vector score and then a stable result key.
- Boilerplate command-wrapper content is downweighted before final ordering.

**Vector similarity query** (N = configured embedding dimensions):
```sql
SELECT
    m.id, ms.role, ms.content, ms.thinking, ms.timestamp,
    CASE
        WHEN me.content_embedding IS NOT NULL AND me.thinking_embedding IS NOT NULL THEN GREATEST(
            array_cosine_similarity(me.content_embedding, ?::FLOAT[N]),
            array_cosine_similarity(me.thinking_embedding, ?::FLOAT[N])
        )
        WHEN me.content_embedding IS NOT NULL THEN array_cosine_similarity(me.content_embedding, ?::FLOAT[N])
        ELSE array_cosine_similarity(me.thinking_embedding, ?::FLOAT[N])
    END AS score
FROM messages m
JOIN message_state ms ON ms.message_id = m.id
LEFT JOIN message_embeddings me ON me.message_id = m.id
WHERE me.content_embedding IS NOT NULL OR me.thinking_embedding IS NOT NULL
ORDER BY score DESC
LIMIT ?;
```

**Auto-mode detection:** Search auto-mode checks whether any searchable embeddings exist in `message_embeddings` or `tool_call_embeddings` before selecting `hybrid`.

#### MLX Backend Implementation

The MLX backend implements a BERT-family encoder in Apple's MLX framework for Metal GPU acceleration:

- Model architecture: BertEmbeddings → BertEncoder (N × BertLayer) → CLS pooling → L2 normalize
- Weight loading: HuggingFace safetensors → name mapping → `model.load_weights()`
- Weight name mapping: `bert.` prefix removed, `encoder.layer.` → `encoder.layers.`, `attention.self.` → `attention.self_attn.`, `LayerNorm` → `layer_norm`
- Tokenization: HuggingFace `tokenizers` library (Rust-based, fast)
- Lazy loading: model loaded on first `embed()` call, cached for subsequent calls
- Local-first snapshot resolution: use cached model files when available and only fall back to network download when the snapshot is missing locally
- Optional dependency: `recall[mlx]` extra
- Unsupported models fail fast with a descriptive error before indexing or search writes embeddings

#### ONNX Backend Implementation

The ONNX backend uses ONNX Runtime for cross-platform CPU inference, enabling semantic search on any host without GPU acceleration:

- Model format: Pre-exported ONNX model from HuggingFace Hub (`onnx/model.onnx` or `model.onnx`)
- Inference: `onnxruntime.InferenceSession` with `CPUExecutionProvider` only
- No custom model code: ONNX Runtime handles the entire forward pass (no weight mapping needed)
- Tokenization: HuggingFace `tokenizers` library (same as MLX backend)
- Pooling: CLS token (`[:, 0, :]`) + L2 normalization (same as MLX backend)
- Model validation: `config.json` must have `model_type == "bert"` and `hidden_size == 384`
- Lazy loading: model loaded on first `embed()` call, cached for subsequent calls
- Local-first snapshot resolution: use cached model files when available and only fall back to network download when the snapshot is missing locally
- Platform probe: checks `onnxruntime`, `tokenizers`, and `huggingface_hub` via `importlib.util.find_spec` (no platform restriction, no actual imports)
- Optional dependency: `recall[onnx]` extra
- Priority 20: MLX preferred on Apple Silicon (priority 10), ONNX as cross-platform fallback
- numpy for array construction: transitive dependency of onnxruntime, no extra install

#### CLI Integration

```bash
# Index with embeddings (default on supported hosts)
recall index                            # Parse + embed inline, default on supported hosts
recall index --no-embed                 # Force keyword-only indexing for this run
recall index --full                     # Full reindex with embeddings (re-embeds all)

# Search modes
recall search "query"                   # Auto-select mode
recall search "query" --mode hybrid     # Explicit hybrid
recall search "query" --mode vector     # Vector only
recall search "query" --mode keyword    # Keyword only

# Background daemon (embeds inline during each cycle)
recall daemon --interval 300            # Periodically index with inline embedding
recall daemon --once                    # Run one daemon cycle and exit
recall daemon install                   # Install platform scheduler integration
recall daemon uninstall                 # Remove platform scheduler integration
recall daemon status                    # Show platform scheduler status
```

#### Acceptance Criteria

- [ ] Invalid embedding backend names and non-positive batch sizes fail during config/backend setup with descriptive errors.
- [ ] Unsupported MLX model families fail before embeddings are written. Dimension changes are handled by `recreate_embedding_tables()` (see `REQ-SCHEMA-005`).
- [ ] Message vector search can return a hit based on `thinking_embedding` when `content_embedding` is absent or less relevant.
- [ ] A failure during `embed_session()` does not leave a `Session` object with partially populated embeddings.
- [ ] Inline embedding writes to dedicated embedding tables via `INSERT ... ON CONFLICT DO UPDATE` without affecting content tables.
- [ ] Repeated message text and repeated normalized bash commands reuse cached embeddings instead of calling the backend again.
- [ ] Cached local MLX models do not trigger HuggingFace metadata lookups on steady-state vector search.
- [ ] Hybrid ranking is deterministic on ties and downweights wrapper/system boilerplate relative to direct matches.
- [ ] On a host that supports the configured backend, `recall index` and default daemon cycles enable embeddings without requiring explicit `--embed`.
- [ ] On a host that does not support the configured backend, default indexing does not attempt to load the embedding backend unless the user explicitly opts in.
- [ ] ONNX backend is registered with priority 20 and auto-selects when MLX is unavailable.
- [ ] ONNX backend validates model config (`model_type == "bert"`, `hidden_size == 384`) before inference.

### Contextual Retrieval

Status: implemented for index-time context generation and storage; reranking remains deferred.

Background: Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) technique prepends per-chunk context to each chunk before embedding and BM25 indexing, reducing top-20 retrieval failures by 35% with contextual embeddings alone, 49% with contextual embeddings + contextual BM25, or 67% with a reranking step on top. recall implements index-time context generation with static template prefixes and per-message LLM prefixes. Reranking is an orthogonal future pass over hybrid search candidates.

Anthropic explicitly noted that "adding generic document summaries to chunks" yielded "very limited gains" relative to per-chunk LLM-generated context. The static template prefix is closer to a generic summary than to true contextual retrieval; its retrieval-accuracy gain is expected to be modest. Per-message LLM context provides the material retrieval-accuracy path while using the same storage, embedding, and BM25 pipeline.

#### Requirements

- `REQ-CTX-001`: For each newly indexed message eligible for contextualization (CONTENT and THINKING kinds), the rendered context prefix must be persisted in `message_state.context_text` at index time. Context must not be deferred to query time, recomputed on read, or stored only in memory.
- `REQ-CTX-002`: Bash command embeddings (`tool_call_embeddings.bash_embedding`) and bash FTS indexing must not be contextualized. Bash command normalization, cross-session bash cache reuse (REQ-SEM-006), and `tool_call_embeddings` shape must remain unchanged across all context modes.
- `REQ-CTX-003`: Each row's write-time context mode must be recorded in `message_state.context_mode` and constrained to the set `{off, template, llm-local, llm-remote, llm-codex}`. A mode of `off` denotes that an empty `context_text` was stored and no enrichment was applied.
- `REQ-CTX-004`: The configured context mode is the write-time mode. Existing rows retain the mode under which they were originally written; changing the configured mode does not retroactively re-render existing rows. Re-rendering requires an explicit `recall index --recompute-context` invocation.
- `REQ-CTX-005`: For `mode = template`, if at least one of `git_repo`, `cwd`, or `git_branch` is present on the session, the rendered prefix must equal `f"[{repo_label} {git_branch_or_head}] "` where `repo_label = git_repo OR basename(cwd) OR "local"` and `git_branch_or_head = git_branch OR "HEAD"`. If `git_repo`, `cwd`, and `git_branch` are all NULL, the prefix must be the empty string and `context_mode = off` for that row's writes.
- `REQ-CTX-006`: When `context_mode != off` for a row, both the vector embedding input and the BM25 indexed text for CONTENT and THINKING must include the same prefix. The embedded text must be `context_text || content` for content embeddings (analogously for thinking). The FTS index field must be the same concatenation. Vector and BM25 contextualization must move together; partial contextualization is not permitted.
- `REQ-CTX-007`: The embedding cache key must change when the embedded text changes due to context enrichment. CONTENT and THINKING cache lookups must use the effective embedded text (`context_text || raw_text`) plus a content/thinking-specific cache-version component. This must not be implemented as a global `NORMALIZATION_VERSION` bump while the bash cache key still includes `NORMALIZATION_VERSION`, because that would invalidate unchanged BASH entries. Existing BASH cache keys must remain byte-identical for the same backend namespace and normalized bash command across contextual retrieval rollout. `embedding_cache.context_version` defaults to `0`; LLM modes populate non-zero `context_version` values for CONTENT and THINKING keys so swapping prefix generators does not invalidate prior LLM-generated cache entries unnecessarily. Existing rows in `embedding_cache` are not deleted on version changes; they simply become unreachable for new lookups.
- `REQ-CTX-008`: Search queries must not be modified at query time. The query string is embedded and BM25-scored verbatim regardless of `context_mode`. The asymmetry between document-side enrichment and query-side rawness is intentional and load-bearing for the technique.
- `REQ-CTX-009`: A configured LLM context backend that fails *per-message at runtime* (timeout, rate limit, network error, model-specific incompatibility) while the backend was otherwise available must fall back according to `[embedding.context] fallback`. Per-message fallback must be local: the run must not abort. A run abort on a per-message failure is permitted only when `fallback = "error"`. Backend *unavailability* detected by the local-support probe (`is_available()` returns False — e.g. `llm-local` without the MLX extra, `llm-remote` without the Anthropic extra or credential, `llm-codex` without the CLI on PATH) is NOT a fallback-governed condition: it is always fatal per REQ-CTX-020, independent of `fallback`. (Superseded for the unavailability case by REQ-CTX-020; the original "local-support probe failure falls back per `fallback`" behavior is intentionally removed.)
- `REQ-CTX-010`: `recall index --recompute-context` must rebuild `context_text` for the selected rows using the currently configured context mode (or an explicit `--context` override), re-embed the corresponding CONTENT and THINKING fields, and rebuild the FTS index. Bash embeddings, bash FTS, and tool_call rows must not be touched.
- `REQ-CTX-011`: `mode = llm-local`, `mode = llm-remote`, and `mode = llm-codex` must be accepted in config and CLI without parse errors and must generate per-message context at index time when their backend is available. Watch-mode catch-up and live indexing must use the same per-message context resolution path as `recall index`.
- `REQ-CTX-012`: The background daemon must read the configured context mode each cycle, not only at daemon start. Editing `[embedding.context] mode` in `~/.config/recall/config.toml` and waiting for one daemon cycle must be sufficient for newly indexed sessions to be written under the new mode without restarting the daemon.
- `REQ-CTX-013`: Per-message context generation in `mode = llm-*` must skip generation when message content length is below `[embedding.context] min_chars`. A length-based skip is not an LLM backend failure and must not consult `[embedding.context] fallback`; it writes `context_text = ''` and `context_mode = 'off'` for that row, continues the index run, and does not increment LLM token counters. The skip threshold applies to the raw content character length only; it does not apply to template mode.
- `REQ-CTX-019`: Incremental-delta writes in `mode = llm-local` and `mode = llm-remote` must contextualize only the newly parsed messages, but the LLM backend must receive the full persisted session history plus the new delta. Full-parse writes and non-LLM context modes continue to contextualize exactly the parsed session messages for the write.
- `REQ-CTX-021`: The `llm-local` (MLX) backend MUST load each model at most once per process, caching the loaded `(model, tokenizer)` by model name and reusing it across `MlxLocalBackend` instances. Watch-mode indexing builds a fresh backend per session (`_prepare_context_run`); without caching each session re-ran `mlx_lm.load()`, re-reading weights and issuing a huggingface.co revision request per session (~0.73s + one network round-trip each). The cache is process-lifetime and keyed by the configured model name; switching the configured model loads the new model and retains the prior entry. The REQ-CTX-020 startup probe naturally warms this cache.
- `REQ-CTX-020`: A configured `llm-*` context mode whose backend is unavailable on this host (its `is_available()` probe returns False — missing extra, missing credential, model load failure, or an unresolvable CLI) MUST be fatal, independent of `[embedding.context] fallback`. The daemon MUST fail fast at startup — at the shared watch-runtime construction point so both the synchronous (`run_watch_daemon`) and async (`RpcServer._start_watch_mode`) entrypoints enforce it — and `recall index` MUST abort at run start, raising an actionable error that names the configured mode, the underlying cause, the remediation (matching extra/credential/CLI install), and the `mode = template|off` escape hatch. The daemon MUST NOT silently degrade to template when the configured LLM backend is unsupported. Modes `off` and `template` require no backend and MUST NOT trigger this check. `fallback` is thereby narrowed to govern only the transient per-message generation failures of REQ-CTX-009. Rationale: a configured-but-unsupported LLM backend is a misconfiguration, and silent template degradation hid it for thousands of daemon cycles.
- `REQ-CTX-014`: `runtime_state` must track contextualization activity over the most recent successful index run via new columns: `last_context_messages` (count of rows whose `context_text` was written or rewritten), `last_context_mode` (resolved context mode for the run after CLI/config selection and run-level fallback), `last_context_input_tokens`, `last_context_output_tokens`, and `last_context_model`. Template/off runs populate token counters with zero and model with NULL. LLM runs populate all five fields, including daemon once/watch runtime paths. `recall stats` must read the persisted `last_context_mode`; it must not infer the previous run's backend from current config.
- `REQ-CTX-015`: `mode = llm-remote` must accept optional `base_url` and `timeout` settings (and matching env overrides `RECALL_CONTEXT_BASE_URL`, `RECALL_CONTEXT_TIMEOUT`) and forward them to the Anthropic SDK constructor so the backend can target any Anthropic-protocol-compatible endpoint (e.g., LiteLLM proxy in front of a local model). Unset values must leave the SDK at its built-in defaults — they must not be passed as `None`. `base_url`, when set, participates in `config_fingerprint` so changing the endpoint forces re-contextualization; `timeout` does not.
- `REQ-CTX-016`: `mode = llm-codex` must drive context generation by shelling out to the local OpenAI Codex CLI in non-interactive mode (`codex exec`). The backend must invoke the CLI with `--json --ephemeral --skip-git-repo-check --sandbox read-only --ignore-user-config --ignore-rules -c web_search="disabled"`, pipe the per-chunk prompt via stdin, capture the final agent text via `--output-last-message`, and parse `turn.completed.usage` from stdout JSONL for token accounting. The CLI binary path is configurable via `[embedding.context] executable` (default `"codex"`) and `RECALL_CONTEXT_EXECUTABLE`. When `is_available()` cannot resolve the binary on PATH, the backend is unavailable and the daemon must fail fast per REQ-CTX-020 (this is not a fallback-governed condition); the integration must never attempt to install or update the Codex CLI itself. The Codex CLI exposes no `max_tokens` equivalent for output capping, so `[embedding.context] max_tokens` is unused by this backend; the prompt's `"short succinct context"` instruction is the only output-length hint and may be exceeded by the model.
- `REQ-CTX-018`: When `mode = llm-codex` is selected and `[embedding.context] model` is left at the shared `DEFAULT_CONTEXT_MODEL` (the MLX model name used as the global default), the backend must substitute `gpt-5.4-mini` at initialization so users opting into Codex CLI without setting an explicit model do not hand the MLX model name to `codex exec --model`. This mirrors the substitution the `llm-remote` backend performs for `claude-haiku-4-5-20251001`. The substituted value is what is reported in `ContextResult.model` and `runtime_state.last_context_model`. The default was chosen empirically over `gpt-5.3-codex-spark`: mini produces ~3× more concise prefixes with ~42% fewer input tokens at the same latency, and is available to API-key users in addition to ChatGPT Pro.
- `REQ-CTX-017`: `mode = llm-codex` must accept an optional `reasoning_effort` setting (and matching env override `RECALL_CONTEXT_REASONING_EFFORT`) drawn from `{minimal, low, medium, high, xhigh, max, ultra}`. When set, it is forwarded to Codex CLI as `-c model_reasoning_effort="<value>"`. When unset, recall must not pass the flag so the CLI uses its built-in default. An `ultra` request must retain Codex's `multi_agent` feature because automatic task delegation is part of that effort's contract; all other efforts keep Recall's normal disabled-feature set. Model-specific incompatibility (for example, Spark rejects `minimal` and models without multi-agent support reject `ultra`) is surfaced as a per-message failure that obeys `[embedding.context] fallback`; recall must not pre-validate effort against the configured model name. Both `reasoning_effort` and `executable` participate in `config_fingerprint` so changing either invalidates cached llm-codex prefixes.

#### Invariants

- For any persisted message with `context_mode != 'off'`, the value of `context_text` is the exact string that was prepended to its CONTENT and THINKING embeddings and to the FTS-indexed text for that row. No transformation, normalization, or whitespace change occurs between persisting `context_text` and the embedding/FTS inputs.
- For BASH-kind rows, `tool_calls.bash_command` and `tool_call_embeddings.bash_embedding` are produced from normalized bash command text with no session-context prefix. Bash cache reuse semantics (REQ-SEM-006, REQ-SEM-008) are unchanged.
- `context_mode` for a given row is mutated only by an indexing pass that selects that row (initial write or `recall index --recompute-context`). Other commands must not modify it.
- Search code (vector, BM25, hybrid) does not branch on `context_mode`. The contextualization is invisible to retrieval beyond the choice of FTS field name and the embedded text already stored in `message_embeddings`.
- `context_text` is the empty string (`''`) when `context_mode = 'off'`. A NULL `context_text` indicates a pre-migration row or an in-flight write that has not yet completed; queries must tolerate NULL by treating it as the empty string.
- The `embedding_cache` table is append-only with respect to version changes. Old CONTENT/THINKING cache entries with prior normalization or context-version values remain on disk; they are excluded by key mismatch, not deleted. BASH entries are reused when their normalized command text and backend namespace are unchanged.

#### Non-goals

- Computing context at query time. All context is index-time.
- Contextualizing bash commands.
- Cross-session context. A message's context never references content outside its own session.
- Generating LLM context using the full session as the prompt for each message. LLM modes must use prompt caching (Anthropic remote) or a windowed prompt (MLX local) to keep per-message cost bounded; the SPEC does not prescribe the exact windowing.
- Backward-compatible automatic re-rendering when `mode` changes. Explicit `recall index --recompute-context` is required to migrate existing rows.
- Query-side prefix synthesis. A `recall search --repo foo` flag that prepends a synthesized prefix to the query string is tracked as a possible follow-up, not part of this feature.
- Selecting a default context mode of `template`. Contextual retrieval ships with `mode = off` as the default and opts users in via config edit.
- Replacing or modifying `tool_call_embeddings` behavior in any phase.

#### Risk Tags

- **[RISK-HIGH] Schema change**: Contextual retrieval adds `message_state.context_text`, `message_state.context_mode`, `embedding_cache.context_version`, and `runtime_state` context counters. The PLAN gate must explicitly approve the schema expansion and its `--recreate` impact before implementation.
- **[RISK-HIGH] Public config and CLI surface**: `[embedding.context]`, `RECALL_CONTEXT_*`, `recall index --context`, and `recall index --recompute-context` create user-facing contracts that future phases must preserve.
- **[RISK-MEDIUM] Retrieval ranking change**: Context-enriched CONTENT and THINKING text changes BM25 and vector inputs for opted-in rows, so acceptance tests must compare `mode = off` against the v0.12 baseline and isolate ranking deltas to contextualized modes.
- **[RISK-MEDIUM] External LLM dependency**: LLM modes introduce local and remote LLM calls during indexing. Bounded fallback, timeout, and token-accounting behavior must be approved before those backends are enabled.

#### Phase Staging

Contextual retrieval has three independently shippable surfaces. Each surface preserves the schema additions and search code from prior work.

| Phase | Prefix source | Approx tokens | Per-message generation cost | New schema | Search code changes |
|---|---|---|---|---|---|
| 1 | Static template `[{repo} {branch}]` | ~6 | None | `message_state.context_text`, `message_state.context_mode`; `embedding_cache.context_version` defaulting to `0`; new `runtime_state` context counters/mode; FTS rebuilt over `context_text || content` and `context_text || thinking` | FTS field selector only |
| 2 | LLM-generated (Anthropic-style 50–100 token contextualizer) | 50–100 | 1 LLM call per eligible message; remote uses prompt caching | None; populates non-zero `embedding_cache.context_version` values for LLM prefix-generator revisions | None |
| 3 | Reranker over top-N hybrid candidates | N/A (rerank is post-retrieval) | 1 rerank call per query | None | New rerank stage between hybrid retrieval and final ordering |

Template and LLM context generation share the same storage, embedding, FTS, and runtime-state pipeline. Phase 3 reranking is architecturally independent and tracked separately.

#### Prefix Template

For `mode = template`, the prefix is resolved at index time using session-level metadata:

```
if session_state.git_repo IS NULL
   AND session_state.cwd IS NULL
   AND session_state.git_branch IS NULL:
    context_text = ""
    context_mode = "off"
else:
    repo_label       = session_state.git_repo
                    OR basename(session_state.cwd)
                    OR "local"
    git_branch_or_head = session_state.git_branch OR "HEAD"

    context_text     = f"[{repo_label} {git_branch_or_head}] "
```

Coverage of constituent fields across the reference local corpus (15,983 indexed sessions):

| Field | Coverage | Role |
|---|---|---|
| `git_repo` | ~38% | Primary `repo_label` when set |
| `git_branch` | ~97% | Used verbatim; defaults to literal `HEAD` if NULL |
| `cwd` (basename) | ~99.6% | Fallback for `repo_label` |

Combined coverage of the resolved prefix is ≥99% of new sessions; the remainder are sessions with no `git_repo`, no `git_branch`, and no `cwd`, for which `context_mode = off` and `context_text = ""`.

Square brackets are chosen as delimiters because they do not appear in the vast majority of message content and give BM25 a stable, indexable boundary.

#### Configuration

Configuration lives under `[embedding.context]` in `~/.config/recall/config.toml`:

```toml
[embedding.context]
# off:        no prefix; existing v0.x behavior. Default mode.
# template:   render prefix from session metadata.
# llm-local:  MLX-based local LLM.
# llm-remote: Anthropic API.
# llm-codex:  OpenAI Codex CLI in non-interactive mode (`codex exec`).
mode = "off"

# Policy ONLY for transient per-message generation errors of an otherwise-available
# LLM backend (timeout, rate limit, network). A configured llm-* backend that is
# unsupported on this host (missing extra/credential/CLI) is ALWAYS fatal at daemon
# startup regardless of this setting — see REQ-CTX-020.
#   template: render the template prefix in place and continue (default).
#   off:      store empty context_text and continue (row is not contextualized).
#   error:    abort the index run with a descriptive failure.
fallback = "template"

# Only used when mode is llm-local or llm-remote.
model        = "mlx-community/Llama-3.2-3B-Instruct-4bit"
max_tokens   = 120        # cap on generated prefix length in tokens
batch_size   = 8          # LLM calls batched per index pass
min_chars    = 50         # skip LLM context generation for messages shorter than this

# Only used when mode is llm-remote. Overrides the Anthropic SDK's default endpoint
# so the same code path can target Anthropic-compatible proxies (LiteLLM,
# llama.cpp Anthropic mode, MLX-LM behind a shim) without touching code. Unset by
# default; the SDK then resolves ANTHROPIC_BASE_URL or falls back to api.anthropic.com.
# base_url = "http://litellm.local:4000"

# Request timeout (seconds) forwarded to the Anthropic SDK. Unset by default
# (SDK's built-in 600s applies). Useful to shorten when talking to a local endpoint.
# timeout  = 30

# Only used when mode is llm-codex.
# When `model` is left at the shared default it is substituted with
# `gpt-5.4-mini` at backend init (parallels the llm-remote backend swap).
# model           = "gpt-5.4-mini"          # any model the local codex CLI accepts; substituted from the shared default
#                                            # GPT-5.6 alternatives: Luna for repeatable/high-volume work,
#                                            # Terra for balanced work, or Sol for maximum capability.
# executable      = "codex"                  # binary name/path; resolved via PATH if relative
# reasoning_effort = "low"                   # minimal|low|medium|high|xhigh|max|ultra
#                                            # model support varies; Codex reports incompatibilities
```

Environment overrides:
- `RECALL_CONTEXT_MODE`
- `RECALL_CONTEXT_FALLBACK`
- `RECALL_CONTEXT_MODEL`
- `RECALL_CONTEXT_BASE_URL`
- `RECALL_CONTEXT_TIMEOUT`
- `RECALL_CONTEXT_REASONING_EFFORT`
- `RECALL_CONTEXT_EXECUTABLE`

Environment variables take precedence over config file values.

#### CLI Integration

```bash
# Per-run overrides
recall index --context off              # disable contextualization for this run
recall index --context template         # explicit template mode
recall index --context llm-local        # local MLX per-message context
recall index --context llm-remote       # Anthropic per-message context
recall index --context llm-codex        # OpenAI Codex CLI per-message context (non-interactive)

# Re-render context_text for existing rows without re-parsing source JSONL.
# Re-embeds CONTENT and THINKING for the affected rows and rebuilds FTS.
# Does not touch bash embeddings or bash FTS.
recall index --recompute-context                          # full rebuild using current mode
recall index --recompute-context --since 30d              # rebuild rows from sessions started in the last 30 days
recall index --recompute-context --only-mode off          # rebuild rows currently marked off (e.g., after switching mode = template)
recall index --recompute-context --context template       # rebuild and force template mode for the run

# Daemon picks up configuration changes on each cycle (REQ-CTX-012); no restart required.
```

#### Cost and State Visibility

`recall stats` surfaces context-generation activity from `runtime_state` after the most recent successful index run. `Context backend` is `runtime_state.last_context_mode`, not the current `[embedding.context] mode`, so stats remain accurate after config changes:

```
Last index run:
  Messages indexed:                 12,847
  Messages contextualized:           4,182
  Context backend:                   template
  Context tokens (input):            0
  Context tokens (output):           0
  Context model:                     (none)
```

For LLM modes, context token counts reflect the backend-reported usage for that run. For the remote backend, the majority of input tokens are expected to be served from Anthropic prompt-caching reads; the SPEC does not separate cached vs uncached input tokens.

#### Acceptance Criteria

- [ ] `recall index` writes `message_state.context_text` and `message_state.context_mode` for every newly indexed CONTENT/THINKING-bearing message under the configured mode.
- [ ] When `mode = template`, `context_text` equals `[{repo_label} {branch_or_head}] ` for every session in the test fixture set with at least one of `git_repo`, `cwd`, or `git_branch`; fixture sessions with all three fields NULL are written with `context_mode = 'off'` and empty `context_text`.
- [ ] When `mode = off`, `context_text` is the empty string and `context_mode = 'off'`. FTS hits and vector cosine scores for fixture sessions match the v0.12 baseline byte-for-byte.
- [ ] BASH embeddings and bash FTS results across all context modes are byte-equivalent to v0.12 on a synthetic re-index of fixtures (no prefix applied, no cache key change).
- [ ] CONTENT and THINKING cache-versioning invalidates entries whose embedded text changed due to context enrichment (cache miss on lookup) without deleting rows from `embedding_cache`; the BASH cache key for an unchanged normalized command remains byte-identical before and after contextual retrieval rollout.
- [ ] `mode = llm-local`, `mode = llm-remote`, and `mode = llm-codex` parse from config and CLI without error and write backend-generated per-message prefixes when the selected backend is available.
- [ ] `mode = llm-codex` shells out to `codex exec` with the locking flag set (`--ephemeral --skip-git-repo-check --sandbox read-only --ignore-user-config --ignore-rules -c web_search="disabled"`) and never with `--sandbox workspace-write` or `danger-full-access`. When the configured `executable` cannot be resolved on PATH, the run falls back per `[embedding.context] fallback` and never raises an unhandled exception.
- [ ] With `fallback = "template"`, a simulated per-message LLM error in `mode = llm-*` writes the template prefix for that row and continues; the index run completes successfully.
- [ ] With `fallback = "error"`, a simulated per-message LLM error in `mode = llm-*` aborts the index run with a descriptive error.
- [ ] With `fallback = "error"`, a message shorter than `[embedding.context] min_chars` does not call the LLM backend, writes `context_text = ''` and `context_mode = 'off'`, and the index run completes successfully.
- [ ] `recall index --recompute-context` updates `context_text`, `context_mode`, and re-embeds CONTENT and THINKING for selected rows; rebuilds the FTS index; leaves bash embeddings, bash FTS, and tool_calls rows unchanged (verified via row hashes).
- [ ] Editing `[embedding.context] mode` from `off` to `template` and waiting for one daemon cycle is sufficient for newly discovered sessions to be written with `context_mode = template` without restarting the daemon.
- [ ] `recall stats` surfaces the new `runtime_state` context counters and persisted `last_context_mode` after an index run; changing `[embedding.context] mode` after the run does not change the reported last-run backend until another successful index run updates `runtime_state`.
- [ ] Daemon once/watch runs persist `last_context_input_tokens`, `last_context_output_tokens`, and `last_context_model` from their `IndexSummary` when LLM context generation is used.
- [ ] For two synthetic sessions with identical message content but different `git_repo` values, a BM25 search whose query contains one repo name ranks the matching-repo session strictly higher than the other under `mode = template`; ranking is identical under `mode = off`.
- [ ] `recall search "foo bar"` issues an embedding and BM25 query for the string `"foo bar"` verbatim regardless of `context_mode` (verified via a backend mock that captures inputs).
- [ ] Search code paths (`services/search.py`) contain no conditional branching on `context_mode`.

### Batched & Concurrent llm-codex Context Generation

Status: planned for v0.18.4. Extends `REQ-CTX-016`; supersedes the *blocking ordering* clause of `REQ-DAEMON-049`.

**Problem.** With `mode = llm-codex`, context generation is fully serial and re-pays the document on every chunk. `index_single_session` resolves context **per chunk** (`resolve_message_context`), and `CodexCliBackend._build_prompt` embeds the entire (windowed) session document into *each* per-chunk prompt; `_run_codex` then spawns one blocking `codex exec` subprocess at a time. A session with N contextualized chunks therefore costs N subprocess invocations and re-sends the document N times. On startup the watch-mode catch-up scan (`_catch_up_scan`, `REQ-DAEMON-049`) drives this serial path over the **entire stale-session backlog before the discovery loop starts**, and `RpcServer._start_watch_mode` holds `self._write_lock` for catch-up's **full duration**. On a host with a large backlog the daemon reports its version but never reaches the live state (no `discovery_last_run_at`, no fsevent subscriptions, no live indexing) until catch-up fully drains — tens of minutes to hours on a multi-thousand-session corpus.

**Solution.** (1) Batch multiple same-session chunks into one `codex exec` call using the CLI's `--output-schema` structured-output flag, so the document is sent once and N contexts return in one call; (2) generate context for distinct sessions/batches concurrently under a bounded pool while keeping DuckDB writes serialized; (3) launch discovery concurrently with catch-up so the daemon goes live immediately and back-fills context behind the scenes.

#### Requirements

- `REQ-CTX-022`: With `mode = llm-codex`, the backend MAY situate multiple chunks of the **same session** in a single `codex exec` invocation. The windowed, `max_document_chars`-bounded session document is included once; the prompt carries an ordered list of chunks; and the call passes `--output-schema <FILE>` whose schema constrains the model's final message to a JSON **object** carrying a `contexts` array of objects, each element carrying the chunk's request index and its situating context string. (codex `--output-schema` relays the schema as the model's `response_format`, which requires an object root; a top-level array is rejected server-side with 400 `invalid_json_schema` — see Drift Notes.) The per-row contract is unchanged: each contextualized chunk's `context_text` is the model's returned context for that chunk, persisted exactly as in the single-chunk path (`REQ-CTX-001`, `REQ-CTX-006`). Batching is an internal generation optimization, not a change to what is stored. Extends `REQ-CTX-016`; the structured-output path is the machine-checkable response shape that `REQ-CTX-016` lacked (it remains *not* a token cap — see Drift Notes).
- `REQ-CTX-023`: Batch size MUST be bounded so the document plus all chunk payloads in one call fit within `[embedding.context] max_document_chars`, and additionally by a maximum chunk count `[embedding.context] batch_size` (env `RECALL_CONTEXT_BATCH_SIZE`, default proposed `8`, finalized in PLAN). Only chunks that resolve to the **same** sliding document window (the existing `_smart_truncate` window, consistent with `REQ-CTX-019`) are eligible to batch together; a chunk requiring a different window is generated in a separate call. A batch of size 1 MUST be byte-identical in behavior to the pre-batching single-chunk path.
- `REQ-CTX-024`: Batched results MUST be mapped back to their source chunks by the schema's request-index field. A response whose item count or indices do not match the requested chunks, a non-zero `codex exec` exit, a timeout (`REQ-CTX-016` timeout policy), or a schema-invalid/empty body is a **batch failure**. On batch failure the backend MUST resolve every chunk in that batch through the per-message fallback policy of `REQ-CTX-009` (`template`/`off`, or abort only when `fallback = "error"`); a batch failure MUST NOT abort an otherwise-healthy run except under `fallback = "error"`, and MUST NOT silently drop chunks. `turn.completed.usage` for the batched call is attributed to the batch and summed into the `runtime_state` context counters of `REQ-CTX-014` with no double counting — the shared-document input tokens are counted once, not once per chunk.
- `REQ-CTX-025`: Context generation for distinct sessions and batches MAY run concurrently up to `[embedding.context] concurrency` (env `RECALL_CONTEXT_CONCURRENCY`, default proposed `4`, minimum `1`, finalized in PLAN). Concurrency governs only LLM/subprocess context generation; all DuckDB writes remain serialized through the daemon `_write_lock` (or the single owned connection in `recall index`), so the single-writer rule (`REQ-CONC-*`, DuckDB Runtime) is preserved. `concurrency = 1` reproduces fully serial behavior. In-flight subprocesses MUST remain registered in `CodexCliBackend._active_processes` so watch-mode shutdown can terminate them (`REQ-CTX-016`, watch-shutdown shielding). `concurrency` does NOT participate in `config_fingerprint` (it does not change generated text). Initial scope is the `llm-codex` backend; `llm-remote`/`llm-local` MAY adopt the same seam later without a spec change.
- `REQ-DAEMON-068`: Watch-mode startup MUST launch the discovery loop (`REQ-DAEMON-045`) and live-session machinery WITHOUT waiting for the catch-up scan (`REQ-DAEMON-049`) to finish. The daemon MUST reach the live state — non-null `WatcherLiveSnapshot.discovery_last_run_at`, active observer subscriptions, responsive live indexing — within one `live_discovery_interval` of process start, regardless of catch-up backlog size. Catch-up runs as a background task that back-fills stale sessions concurrently. This supersedes the ordering clause of `REQ-DAEMON-049` ("run … once before the discovery loop starts"); catch-up's staleness algorithm (`REQ-DAEMON-021`), byte-offset incremental parsing (`REQ-DAEMON-030`), and newest-first ordering (`REQ-INDEX-013`) are retained.
- `REQ-DAEMON-069`: The catch-up scan MUST NOT hold `self._write_lock` for its entire duration. It MUST acquire the write lock per session (or per bounded batch) and release it between units, so live indexing, idle FTS rebuilds, the embed phase, and RPC reads are not starved while a large backlog drains. A message indexed by catch-up and one indexed by live indexing MUST converge to identical persisted state (idempotent staleness check per `REQ-DAEMON-021`); concurrent catch-up and live indexing of the same session MUST NOT corrupt `last_byte_offset` or double-write rows.
- `REQ-DAEMON-070`: `recall daemon status` MUST expose catch-up progress as additive `DaemonSchedulerStatus` fields so operators can distinguish "live and back-filling" from "idle": `catchup_in_progress: bool`, `catchup_total: int` (stale sessions enqueued at scan start), and `catchup_done: int` (units completed). Fields default to a terminal/zero state when no catch-up is running.
- `REQ-DAEMON-071`: `recall daemon install` MUST treat `launchctl bootout` as asynchronous. Because a service still tearing down continues to answer `launchctl print`, a probe cannot distinguish a dying service from a live one, and `launchctl bootstrap` refuses (observed: `Bootstrap failed: 5: Input/output error`) for the duration of the teardown. The install MUST wait, under an explicit bound, for the unit to leave the loaded state before bootstrapping, and MUST retry the bootstrap a bounded number of times, evicting the unit between attempts so a `KeepAlive` reload of the old job cannot deadlock the install.
- `REQ-DAEMON-072`: `recall daemon install` MUST verify that the unit is actually loaded after `launchctl bootstrap` returns, and MUST fail closed when it is not. A zero exit from `bootstrap` is not evidence the job took. A failed verification MUST roll back the artifacts the install wrote, matching `REQ-DAEMON-034`, so the host is never left reporting a successful install with no daemon running.
- `REQ-DAEMON-074`: DuckDB's file lock is exclusive to a read-write holder across processes, so while a live daemon holds the database even a read-only open from the CLI fails (`is_lock_conflict` in `recall.db` names that error; `db rebuild-indexes` and `compact` use the same predicate). The local `daemon_status()` path must therefore never open the database under a live daemon: when the pid file names a live process (written before the daemon opens the database) — or, for a lost race, when the open raises a lock conflict — it returns `daemon_pid` and `runtime_unavailable_reason` with default runtime fields and `embed_pending = 0` instead of raising; a dead pid in the file does not suppress the read. A live PID is deliberately treated as the daemon because portable PID ownership proof is unavailable; after a crash and PID reuse, status can therefore remain in the unread state until an operator verifies that the PID is not recall and removes the stale `recall.pid`. A lock conflict with no live pid file names *another process* (carrying the holder pid from DuckDB's message — a foreground `recall index`, or a daemon that has not written its pid file yet) and leaves `daemon_pid` null; it never claims a daemon. While `runtime_unavailable_reason` is set the runtime fields are defaults, not readings, so the status notices derived from them (the freshness/"has not completed a successful run" warning and the informational freshness line) are suppressed; artifact-, scheduler- and marker-derived notices still show. A caller that passes its own connection is the daemon and reads directly. The pid-file attribution governs the local fallback only: when the daemon's RPC answers (the CLI's first choice), the served status carries `daemon_pid` from the serving process itself, so a healthy status names the daemon the CLI is talking to and text status prints `Daemon: pid N`. `recall daemon status` then exits 0 and prints `Daemon: pid N alive, RPC not answering (starting or busy); runtime fields unavailable …` when the holder is the daemon (the RPC is the source of those fields; the CLI tries it first) or `Database: …` with the other-process reason, and both fields are `daemon status` manifest entries. `install_scheduler` tolerates the same state: loading the unit relaunches the daemon, which takes the database before the installed-scheduler record and the closing status read run, so the record is skipped with an info log (status derives the kind from the installed unit files) and the install reports success for the unit it wrote. `uninstall_scheduler` waits up to 30 seconds for asynchronous teardown to release the database before clearing the persisted scheduler kind; if another holder outlives that bound, scheduler artifacts remain authoritative and rerunning uninstall after the holder exits clears the record. Only recognized lock conflicts are retried; any other DuckDB I/O error propagates immediately. Not covered: the stale record case where the scheduler *kind* changed on a host with a running daemon — re-run `recall daemon install` with the daemon stopped.
- `REQ-DAEMON-073`: When `recall daemon stop` cannot stop a launchd daemon because the unit is not loaded but a live daemon process remains (an orphan, typically hand-started outside launchd), the result message MUST name that condition and the offending pid rather than returning launchd's own `Boot-out failed: 3: No such process`, which describes the unit and points nowhere. The remediation MUST name stopping the process directly and re-running `recall daemon install`.
- `REQ-DAEMON-075`: A default `recall daemon status` MUST be readable without paging tools. The source-coverage page defaults to 10 rows (a 100-row default rendered 2,501 of 2,719 output lines and 99 KB of a 110 KB JSON payload); `--limit` (1..256) and `--cursor` still page the catalog and every response still carries `next_cursor`. The default page is not an operator page request, so an unreachable daemon still falls back to the local read of `REQ-DAEMON-074`, while an explicit `--limit`/`--cursor` still fails rather than answering a different page (`REQ-LIVE-009`). Text output MUST open with a summary block — `Daemon: version=… mode=… scheduler=…` plus `pid=` when known; an `embed:` line naming pending, last loop outcome, last batch, any `enrichment_deferred` reason and any last error; a `reconciliation:` line naming pending with the `catalog_scan_complete` and `keyword_search_ready` readiness flags, or naming that coverage is unavailable and why, never a fabricated zero; and a `health:` line naming version drift, the bloat ratio against its threshold when known, and any startup refusal — followed by a blank line and the existing detail. The summary states facts only; the human advisory itself stays on stderr per `REQ-CLI-012`. Every epoch-seconds field the text renderer prints MUST read as local ISO-8601 with a relative age (`2026-09-17T22:08:04 (3m ago)`) rather than a bare float, matching its ISO-8601 sibling fields; `embed_loop_stage_at` remains reported as `stage_age` in seconds because stall detection reads that span directly. Structured output field names and values are unchanged.
- `REQ-DAEMON-076`: On macOS the launch agent label is `it.send.recall.daemon`; releases before it used `xyz.metalrodeo.recall.daemon`. `recall daemon install` is the only command that retires the legacy label, and it never runs `launchctl disable` on it (the override persists and would block a rollback). Until install runs, every lifecycle command acts on the legacy install as before: one resolver picks the new label when its plist exists, else the legacy label when its plist exists or its job is loaded, else the new label, and stop/start/restart, compaction's stop and restart, pid discovery, health, and status (installed, installed mode, binary path, drift) all use it, so no command splits between two jobs. Status reports a legacy install as installed, sets `launchd_legacy_label`, and warns that `recall daemon install` migrates it. When the new plist exists but a legacy plist or loaded legacy job remains (an interrupted migration), status sets `launchd_legacy_leftover` and warns that `recall daemon install` retires it. Status only reads launchd (`print`/`list`) and never changes it. In that interrupted state `recall daemon stop` also boots out the loaded legacy job and waits for both labels to unload, so stop stays durable. Install snapshots both plists and runs every step under one restore path: boot out the legacy job and wait for it to unload, failing before any bootstrap if it is still loaded at the deadline; unlink the legacy plist (a failure fails the install); `launchctl enable` the new label; bootstrap with bounded retries; verify the job loaded; run the smoke test. On any failure, including exhausted bootstrap retries, it boots out the new label and waits for it to unload. If it unloads, both plists are restored and the job that was loaded before install is re-bootstrapped: the new label's restored plist when it was loaded (so a failed re-install on an already-migrated host leaves the previous daemon running, not stopped), else the legacy plist when the legacy job was loaded; a legacy install that was stopped stays stopped. If the new label does not unload, only the legacy plist is restored, the new plist is kept so every command resolves to the job still loaded, and nothing is bootstrapped, so both labels are never loaded at once. Uninstall boots out and removes both labels. Downgrade: an older recall does not know the new label, and its `daemon install` would load a second daemon beside it; before downgrading, boot out `it.send.recall.daemon`, wait until `launchctl print gui/<uid>/it.send.recall.daemon` fails, and remove its plist.

#### Invariants

- Batching and concurrency are generation-time optimizations only: for every contextualized row the persisted `context_text` remains the exact string prepended to its CONTENT/THINKING embeddings and FTS text (the `REQ-CTX-006` invariant is unaffected).
- At most one DuckDB write transaction is in flight at any instant; `concurrency > 1` parallelizes context generation but never DB writes.
- The daemon's live-indexing correctness is independent of catch-up state: a given message is contextualized and persisted at most once for a given source byte range, whether the writer is catch-up or live indexing.

#### Non-goals

- Cross-session batching. A single `codex exec` call situates chunks from one session against one document; mixing chunks from different sessions into one prompt is out of scope and would violate the "situate within its own document" semantics.
- Unbounded concurrency or removal of the per-call timeout (`REQ-CTX-016`). The pool is bounded and each subprocess keeps its timeout.
- Changing the stored contextual-retrieval contract, the embedding/FTS pipeline, or search code (`REQ-CTX-006`, `REQ-CTX-008`).
- Automatically re-contextualizing a backlog with codex after a `template` fast-drain. Operators restore quality via explicit `recall index --recompute-context` (`REQ-CTX-010`).
- Batching/concurrency for `llm-local`/`llm-remote` in this work (codex-first; the seam is reusable).

#### Acceptance criteria

- [ ] A session with M contextualized chunks sharing one document window is contextualized in ⌈M / `batch_size`⌉ `codex exec` invocations (verified via a backend spy), each passing `--output-schema`, versus M invocations before.
- [ ] Each batched call sends the session document payload exactly once; total input tokens for a batched session are strictly less than the serial per-chunk path on the same fixture (token counters compared).
- [ ] `context_text` written per row under batched generation is non-empty and situates the chunk on a fixture session; a batch of size 1 is byte-identical to the pre-batching path.
- [ ] A simulated batch failure (non-zero exit / timeout / schema-invalid body / index mismatch) with `fallback = "template"` writes the template prefix for every chunk in the batch and the run completes; with `fallback = "error"` the run aborts with a descriptive error.
- [ ] Token accounting for a batched call attributes `turn.completed.usage` once into `runtime_state` counters (no per-chunk multiplication of the shared-document input tokens).
- [ ] With `RECALL_CONTEXT_CONCURRENCY = K > 1`, up to K `codex exec` subprocesses are live simultaneously during catch-up (observed via `_active_processes` / process sampling) while DuckDB shows a single writer; `K = 1` reproduces serial behavior.
- [ ] A watch-mode daemon started against a fixture with a non-empty stale backlog reports non-null `discovery_last_run_at` and ≥1 observer subscription within one (test-shortened) `live_discovery_interval`, BEFORE catch-up finishes (`catchup_in_progress = true`, `catchup_done < catchup_total`).
- [ ] During a long catch-up drain, an RPC read (`recall search` / `daemon status`) and a live-session index both complete without blocking for the full catch-up duration (write lock released between units).
- [ ] `recall daemon status` reports `catchup_in_progress`, `catchup_total`, `catchup_done`, transitioning to `catchup_in_progress = false` when the backlog drains.
- [ ] In-flight catch-up `codex exec` subprocesses are terminated on watch-mode shutdown (no orphaned `codex` processes after `recall daemon stop`).

#### Phase Staging

Three independently shippable surfaces; PLAN may land them separately. Recommended order:

| Phase | Requirements | Win |
|---|---|---|
| A | `REQ-DAEMON-068`, `REQ-DAEMON-069`, `REQ-DAEMON-070` | Daemon goes live immediately on restart; biggest UX fix, no LLM-contract change |
| B | `REQ-CTX-022`, `REQ-CTX-023`, `REQ-CTX-024` | Batched `--output-schema` generation: fewer calls, document tokens paid once |
| C | `REQ-CTX-025` | Bounded concurrency across sessions/batches; composes with A and B |

#### Risk Tags

- **[RISK-MEDIUM] Public config surface**: `[embedding.context] concurrency` / `RECALL_CONTEXT_CONCURRENCY` and `[embedding.context] batch_size` / `RECALL_CONTEXT_BATCH_SIZE` are new runtime knobs future versions must preserve. Defaults must be conservative (small concurrency) to avoid LLM-provider rate-limit storms.
- **[RISK-MEDIUM] Structured-output reliance**: batched generation depends on the model honoring `--output-schema`. Malformed / partial / over-length responses must degrade per `REQ-CTX-024` and never corrupt or drop rows.
- **[RISK-MEDIUM] Retrieval quality**: situating several chunks in one prompt may yield less-focused per-chunk context than the single-chunk prompt. Acceptance must compare batched vs single-chunk context quality on a fixture before defaulting `batch_size > 1`.
- **[RISK-MEDIUM] Concurrency correctness**: parallel context generation with serialized writes must not corrupt `last_byte_offset` or double-write when catch-up and live indexing touch the same session (`REQ-DAEMON-069`).
- **[RISK-LOW] Daemon lifecycle change**: non-blocking catch-up changes the observable startup sequence; status fields are additive and the staleness/ordering guarantees of `REQ-DAEMON-021/030/049` (minus the blocking clause) are retained.

#### Drift Notes (surfaced per spec mutation policy)

- `REQ-CTX-016` states the codex backend "pipes the per-chunk prompt via stdin" and that the CLI "exposes no `max_tokens` equivalent … so `[embedding.context] max_tokens` is unused." Batching (`REQ-CTX-022`) keeps stdin piping but sends multiple chunks; `--output-schema` is a response-*shape* mechanism, still **not** a token cap, so the `max_tokens`-unused statement stands. `REQ-CTX-016`'s flag set and the availability/fail-fast rules (`REQ-CTX-020`) are otherwise retained.
- `REQ-CTX-022`'s `--output-schema` body wraps the per-chunk array under a `contexts` object property (`_batch_output_schema`), because codex forwards the schema as the model's `response_format`, which requires an object root; a top-level array is rejected. The parser (`_parse_batch_contexts`) also tolerates a bare array. The stored per-row `context_text` contract (`REQ-CTX-006`) is unchanged.

### Background Indexing Daemon

#### Requirements

- `REQ-DAEMON-001`: `recall daemon` must support a single-run mode (`--once`) and a periodic mode driven by a validated interval in seconds.
- `REQ-DAEMON-002`: Each daemon cycle must discover new or changed session files using the same staleness algorithm as `recall index` (`source_path`, `file_mtime`, `file_size`) and skip unchanged files.
- `REQ-DAEMON-003`: _Superseded by `REQ-ADAPT-001` and `REQ-ADAPT-014`._ When the RPC daemon server owns all DB connections (the normal production path), `daemon_run` writes directly to the live database on the shared connection; the daemon is the single DB owner and DuckDB MVCC ensures read consistency during writes. The work-copy swap pattern is retained only for standalone `run_cycle` execution (tests, no server running).
- `REQ-DAEMON-004`: Failed daemon cycles must not corrupt the live database. When writing via the RPC shared connection, per-session transactions ensure atomicity — committed sessions persist and failed sessions roll back. Partial progress is valid and retried on the next cycle. In standalone mode (work-copy swap), a failed cycle leaves the previous live database untouched.
- `REQ-DAEMON-005`: Daemon configuration must be available via env vars and config file, including interval, whether inline embedding is enabled during indexing, and optional source filtering.
- `REQ-DAEMON-006`: Daemon lock contention must fail the current cycle cleanly and retry on the next scheduled interval instead of corrupting state.
- `REQ-DAEMON-007`: `recall daemon install` must install a user-level scheduler integration appropriate to the host platform.
- `REQ-DAEMON-008`: On macOS, the installed scheduler integration must be a `launchd` LaunchAgent that runs under the current user.
- `REQ-DAEMON-009`: On Linux, the preferred installed scheduler integration must be a `systemd --user` service plus timer when `systemctl --user` is available.
- `REQ-DAEMON-010`: On Linux hosts without usable `systemd --user`, `recall daemon install --scheduler cron` may install a user crontab entry that runs `recall daemon --once`.
- `REQ-DAEMON-011`: Installed scheduler definitions must be generated from the active recall configuration and must invoke the same installed `recall` CLI entrypoint the user would run manually.
- `REQ-DAEMON-012`: `recall daemon uninstall` must remove only scheduler artifacts created by recall and must be idempotent.
- `REQ-DAEMON-013`: `recall daemon status` must report the configured scheduler type, whether it appears installed, and the effective command/config path being used.
- `REQ-DAEMON-014`: Each index or daemon cycle must persist enough status metadata to report the last attempted run, last successful run, summary counts, and the last failure message if any.
- `REQ-DAEMON-015`: Human-readable CLI commands may surface a one-line freshness/status notice derived from persisted daemon/index metadata, including the age of the live index and the last successful daemon run when available.
- `REQ-DAEMON-016`: `recall daemon status` must include last-attempted and last-successful run metadata, including indexed/skipped/failed counts.
- `REQ-DAEMON-017`: Status notices must be informational only and must never prevent the primary command from succeeding.
- `REQ-DAEMON-018`: _Auto-resolution clause superseded by `REQ-DAEMON-041`._ `recall daemon` must accept a `--mode watch|poll|auto` option (default `auto`). `auto` resolves to watch mode unconditionally; `--mode poll` remains available for CI and debug.
- `REQ-DAEMON-019`: Watch mode must use platform-native filesystem events (FSEvents on macOS, inotify on Linux) via the `watchdog` library.
- `REQ-RECALL-0145-H2`: Daemon startup must invoke snapshots GC once with the default 7-day threshold before watch-mode loops begin. It must be silent when nothing is removed and log a concise summary when entries are removed.
- `REQ-DAEMON-020`: Watch mode must debounce per-file events (default 5 seconds, configurable via `daemon.debounce` config key or `RECALL_DAEMON_DEBOUNCE` env var).
- `REQ-DAEMON-021`: Watch mode must perform a catch-up scan on startup using the same staleness algorithm as poll mode (`source_path`, `file_mtime`, `file_size`), sorted newest-first per `REQ-INDEX-013`.
- `REQ-DAEMON-022`: Watch mode must write single-session updates directly to the live database (no work-copy-swap). An advisory lock must be acquired briefly per write.
- `REQ-DAEMON-023`: FTS index rebuilds in watch mode must be debounced separately (default 10 seconds, configurable via `daemon.fts_debounce` config key or `RECALL_DAEMON_FTS_DEBOUNCE` env var).
- `REQ-DAEMON-024`: _Superseded by `REQ-ADAPT-009`._
- `REQ-DAEMON-025`: _Superseded by `REQ-DAEMON-040`._ `watchdog` is now a required runtime dependency; `recall[watch]` is retained as an empty alias for backward compatibility.
- `REQ-DAEMON-026`: `recall daemon install` on macOS must use `KeepAlive` and `RunAtLoad` for watch mode instead of `StartInterval`, and must omit `--once`.
- `REQ-DAEMON-027`: `recall daemon status` must report the active daemon mode, watched directories, and debounce intervals.
- `REQ-DAEMON-028`: Watch mode must shut down gracefully on SIGTERM/SIGINT — flushing pending files, stopping the watcher, and closing the database connection.
- `REQ-DAEMON-029`: Watch mode indexing of changed sessions must use byte-offset incremental parsing per `REQ-INDEX-010`. Only new JSONL lines since the last indexed offset are parsed and embedded. This avoids re-parsing and re-embedding entire sessions when a single message is appended.
- `REQ-DAEMON-030`: Watch mode catch-up scan must use byte-offset incremental parsing. Sessions with a stored `last_byte_offset` must seek past already-indexed content. Embeddings for new messages are generated inline per `REQ-INDEX-014`.
- `REQ-DAEMON-031`: `recall daemon status` must detect the installed daemon mode on all supported platforms. On macOS, the launchd plist must be parsed for `KeepAlive` (watch mode) vs `StartInterval` (poll mode). On Linux, the existing systemd service file parsing applies. The detected mode must populate `resolved_mode` and `watched_dirs` in the status output.
- `REQ-DAEMON-032`: `recall daemon status` must emit a status notice when the installed scheduler mode differs from the mode that `auto` would resolve to given current dependencies (e.g., `watchdog` is installed but the scheduler runs in poll mode). The notice must suggest `recall daemon install` to upgrade.
- `REQ-DAEMON-033`: `recall daemon install` must verify that the resolved `recall` binary exists on disk before writing any scheduler artifact. When `resolve_recall_binary()` raises or returns a path that does not exist, installation must fail with a descriptive error and must not leave a stale plist, systemd unit, or crontab entry on disk. This applies to all supported schedulers (`launchd`, `systemd`, `cron`).
- `REQ-DAEMON-034`: After writing scheduler artifacts and activating them, `recall daemon install` must perform a post-install smoke test by executing the freshly installed program path with a cheap flag that is guaranteed to exit zero on a healthy CLI (e.g., `--help`, which Typer short-circuits before RPC/DB init). The install must fail when the probe exits non-zero. Failed installs must roll back the artifacts they just wrote so the user is never left in a broken-but-looks-installed state.
- `REQ-DAEMON-035`: `recall daemon status` must detect binary-path drift between the installed scheduler artifact and the currently resolved `recall` binary. The status response must expose `installed_binary_path` (the program path parsed from the launchd `ProgramArguments`, systemd `ExecStart`, or crontab line) and `installed_binary_stale` (true when that baked-in path does not exist on disk or differs from the currently resolved `recall` binary). Drift detection is mandatory on launchd and systemd; on cron it applies when the crontab line embeds an absolute path.
- `REQ-DAEMON-036`: `recall daemon status` must query scheduler-level health and surface the most recent exit status of the installed job when the platform exposes one. On macOS, parse `launchctl list <label>` for `LastExitStatus` (or `launchctl print gui/$UID/<label>` as a fallback). Asked about one label, launchctl answers with a plist-style dictionary (`"LastExitStatus" = 0;`), not the tabular PID/Status/Label listing the bare `launchctl list` prints, so the dictionary entry is what the parse must read; the `print` fallback cannot stand in for it, because a job that is still running reports `last exit code = (never exited)` and would leave a healthy daemon reading `"unknown"` forever, which also silences the `REQ-DAEMON-039` warning. On Linux systemd, read `systemctl --user show recall-daemon.service --property=Result,ExecMainStatus` (and the timer's `Result` where applicable). On cron, the field is unavailable and must be reported as null. The dataclass must expose `scheduler_last_exit_status: int | None` and `scheduler_health_state: str | None` (e.g., `"ok"`, `"failed"`, `"unknown"`).
- `REQ-DAEMON-037`: Status notices must emit a loud freshness warning when the daemon appears installed but has not successfully run recently. The warning must fire when (a) `installed=True` and `last_successful_at is None`, or (b) `installed=True` and `now - last_successful_at > max(2 * daemon.interval, 600s)` in poll mode, or (c) `installed=True` in watch mode and `now - last_successful_at > 3600s`. The warning must suggest `recall daemon install` as remediation. This supersedes the current behavior where `render_status_notice` returns `None` when `last_successful_at is None`.
- `REQ-DAEMON-038`: When `installed_binary_stale` is true, status notices must emit a high-priority warning naming both the baked-in program path and the currently resolved `recall` binary, and must recommend re-running `recall daemon install`. This drift notice takes precedence over the freshness notice in `REQ-DAEMON-037` and the mode-mismatch notice in `REQ-DAEMON-032` when both apply.
- `REQ-DAEMON-039`: When `scheduler_last_exit_status` is non-zero and non-null, status notices must emit a warning containing the exit status and a platform-appropriate remediation hint. On macOS, the notice must map exit code `78` (EX_CONFIG) to a human-readable "launchd could not execute the configured program" message and must recommend re-running `recall daemon install`. On Linux systemd, the notice must include the `Result` string when available.

##### Active-Session Watcher

The existing watch mode recursively subscribes to each parser `watch_root` (e.g., `~/.claude/projects/`), which forces the OS to stream fsevents for thousands of idle session files. Active-session targeting narrows the subscription surface to only the sessions currently being appended to, detaching automatically once they go idle. This cuts steady-state fsevent traffic to zero when no session is live and removes the recursive root subscription as a bottleneck on large history directories.

- `REQ-DAEMON-040`: `watchdog>=4.0` must be a required runtime dependency of `recall`, not an optional extras group. Supersedes `REQ-DAEMON-025`. The `[project.optional-dependencies].watch` group is retained as an empty alias so existing `recall[watch]` install commands continue to resolve without error.
- `REQ-DAEMON-041`: `resolve_daemon_mode(AUTO)` must return `WATCH` unconditionally now that `watchdog` is always present. Supersedes the fallback-to-poll clause of `REQ-DAEMON-018`. `--mode poll` remains supported as an explicit opt-in for CI and debug; it must not be selected implicitly by `auto`.
- `REQ-DAEMON-042`: The `SessionParser` protocol in `packages/recall/src/recall/parsers/protocol.py` must expose `live_candidates(*, now: datetime, idle_threshold: float) -> list[Path]` returning the set of jsonl paths under the parser's `watch_root` whose filesystem `mtime >= now - idle_threshold`. Concrete parsers may share a default implementation that `rglob`s the `watch_root` and filters by `stat().st_mtime`.
- `REQ-DAEMON-043`: Watch mode must maintain an in-memory **live session set** — the set of jsonl paths currently considered active. On startup the live set must be seeded by calling `live_candidates` across every registered parser.
- `REQ-DAEMON-044`: The watchdog `Observer` must subscribe narrowly based on the live set. On Linux (inotify backend), the observer schedules exactly one watch per live file. On macOS (FSEvents backend), the observer schedules one watch per parent directory of each live file, because FSEvents only coalesces at directory granularity. When the live set is non-empty, the watcher must not schedule a recursive watch against any parser `watch_root`.
- `REQ-DAEMON-045`: A periodic discovery loop (default interval `30s`) must re-run `live_candidates` per parser and promote any newly-active files into the live set. Promotion adds the corresponding schedule (file on Linux, dir on macOS) to the observer. Discovery is the sole source of promotion other than fsevents fired against an already-subscribed ancestor.
- `REQ-DAEMON-046`: The watcher must demote a live-set member when its `mtime` has been unchanged for `live_idle_threshold` seconds (default `300`) **and** no fsevent has fired against that path within the same window. On demotion the observer schedule is released; on macOS the parent-directory schedule is released only when no other live-set member still shares that parent directory.
- `REQ-DAEMON-047`: When the live set is empty, the watcher must hold zero fsevent subscriptions. Only the discovery loop runs at its configured interval. This is the steady-state resting cost of the daemon.
- `REQ-DAEMON-048`: Watch mode must continue to run as a single long-lived process managed by launchd with `KeepAlive` (conditioned on the refusal marker being absent, per `REQ-RESIL-024`) and `RunAtLoad=true` (per `REQ-DAEMON-026`). Active-session targeting must not change scheduler integration semantics on either macOS or Linux systemd.
- `REQ-DAEMON-049`: On startup, the watcher must run the existing `_catch_up_scan` (`services/watcher.py:497`) once before the discovery loop starts, so sessions updated while the daemon was down are indexed. The catch-up path must continue to use byte-offset incremental parsing per `REQ-DAEMON-030` and honor newest-first ordering per `REQ-INDEX-013`.
- `REQ-DAEMON-050`: `recall daemon status` must report the following additive fields on `DaemonSchedulerStatus`: `live_session_count: int`, `live_session_paths: list[str]` (capped to the top `N=10` entries by `mtime` descending), `discovery_interval_seconds: float`, `discovery_last_run_at: datetime | None`, `discovery_last_promoted: int`, `discovery_last_demoted: int`, `watcher_subscription_count: int`. The existing `watched_dirs` field is retained but its semantics change in watch mode to reflect the dynamically-scheduled directories rather than the parser roots.
- `REQ-DAEMON-051`: The `[daemon]` config section must accept three new fields with matching env-var equivalents:
    - `live_idle_threshold` (seconds, default `300`, env `RECALL_DAEMON_LIVE_IDLE_THRESHOLD`)
    - `live_discovery_interval` (seconds, default `30`, env `RECALL_DAEMON_LIVE_DISCOVERY_INTERVAL`)
    - `live_max_subscriptions` (count, default `64`, env `RECALL_DAEMON_LIVE_MAX_SUBSCRIPTIONS`)
- `REQ-DAEMON-052`: Overflow handling. When a promotion would push the live set beyond `live_max_subscriptions`, the watcher must evict the live-set member with the oldest `mtime` in favor of the new candidate. If every current member is strictly newer than the candidate, the promotion must be rejected and the event logged at `WARNING` level. This bounds memory and subscription count on abnormal workloads.
- `REQ-DAEMON-053`: Per-file event debouncing (`REQ-DAEMON-020`) and FTS rebuild debouncing (`REQ-DAEMON-023`) are unchanged. Active-session targeting changes **what** is subscribed, not **how** events propagate through the debounce queue or the adaptive index/embed split in `REQ-ADAPT-001..015`.
- `REQ-DAEMON-054`: `live_candidates` must rely on `mtime` only. Implementations must not call `lsof`, read `/proc/*/fd`, or perform any process-fd introspection. Claude Code and Codex both append-on-write to their session files, so `mtime` is a sufficient activity signal, and the non-portable alternatives are explicitly out of scope.
- `REQ-DAEMON-055`: Graceful degradation. If scheduling a new observer subscription fails (e.g., `PermissionError`, inotify `ENOSPC`, or a `watchdog` backend exception), the watcher must log a warning, exclude the affected path from the observer, and fall back to polling that path at `live_discovery_interval` granularity via the discovery loop. The failure must not crash the daemon or prevent other subscriptions from being scheduled.
- `REQ-DAEMON-056`: Test hooks. `live_idle_threshold`, `live_discovery_interval`, and `live_max_subscriptions` must be overridable at runtime via the existing config plumbing. `run_watch_daemon` must expose a seam that allows unit tests to inject a synthetic clock and synthetic fsevents without instantiating a real `watchdog.Observer`. Tests must be able to drive promotion, demotion, eviction, and graceful-degradation code paths deterministically.
- `REQ-DAEMON-057`: Single watch-mode entry point. The async RPC daemon (`recall.services.rpc_server.RpcServer._start_watch_mode`) is the sole production entry point for watch mode — it is invoked by both `recall daemon start` (foreground and background) and by `recall daemon --mode watch` via `_start_foreground_server`. That entry point must instantiate `LiveSessionSet`, seed it per `REQ-DAEMON-043`, schedule observer subscriptions per `REQ-DAEMON-044`, run the discovery loop per `REQ-DAEMON-045`, and emit snapshot updates per `REQ-DAEMON-050`. The synchronous `run_watch_daemon` helper in `services/watcher.py` must either be removed or reduced to a thin wrapper that delegates to the same shared core used by the RPC path; it must not hold an independent copy of the seeding, reconciliation, or discovery logic. Supersedes the implicit dual-path assumption in the original active-session watcher phases.
- `REQ-DAEMON-058`: Shared core helpers. The seeding, reconciliation, discovery-tick, and snapshot-update routines (`_seed_live_session_set`, `_reconcile_observer_subscriptions`, `_run_discovery_tick`, `_update_live_snapshot`, and `_catch_up_scan` as it applies to watch mode) must live in `services/watcher.py` as reusable building blocks. The async RPC watch path must call them via `loop.run_in_executor` for any blocking filesystem/DB work and must not duplicate their logic inline. Any future change to seeding, promotion, demotion, reconciliation, or snapshot semantics must apply automatically to both entry points.
- `REQ-DAEMON-059`: Live snapshot lifecycle. `WatcherLiveSnapshot` module state must be updated exactly once per state-changing event: after the initial seed, after each observer subscription reconcile, after each discovery tick (whether or not any promotion/demotion occurred), and after each fsevent-driven promote that changes live-set membership or `last_event_at`. On watch-mode shutdown the snapshot must be reset to zero so stale data does not leak into subsequent daemon_status calls. The async RPC path must schedule snapshot updates from the same task that performs the corresponding mutation, so snapshot reads from other coroutines observe a monotonically consistent view.
- `REQ-DAEMON-060`: Runtime mode detection in `daemon_status`. When a watch-mode daemon is actively running on the host, `daemon_status` must report `resolved_mode = WATCH` and populate the live_session_* block, even when the installed scheduler unit (launchd plist, systemd service, or cron entry) runs `recall daemon --once` in poll mode. The existing `installed_mode` signal (derived from the scheduler unit file) must remain available on the status payload as a separate field so status notices can still raise mismatch warnings. Runtime detection must rely on a live signal from the running daemon (e.g., liveness of the `recall.sock` + `recall.pid` contract, or a non-null `WatcherLiveSnapshot.discovery_last_run_at`), never on static config inspection alone. When both signals disagree (installed=poll, running=watch, or vice versa), the running daemon's actual mode wins for `resolved_mode` and the mismatch is surfaced via `mode_mismatch_reason` so CLI notices can explain the state.
- `REQ-DAEMON-061`: End-to-end integration test. An integration test in `tests/test_services/` must start the RPC server in watch mode against a temp directory with seeded fake parser roots, append bytes to a jsonl file under a parser `watch_root`, and assert that within one `live_discovery_interval` (test-overridden to <=1s) `get_live_snapshot()` reports `live_session_count >= 1`, `watcher_subscription_count >= 1`, and a non-null `discovery_last_run_at`. The test must exercise the same `_start_watch_mode` code path the production daemon uses — it must not call `run_watch_daemon` or hand-roll a parallel `LiveSessionSet` wiring. This test guards against the regression class where the active-session machinery exists and has passing unit tests but is never invoked by the production entry point.
- `REQ-DAEMON-062`: CLI status parity. `recall daemon status` text output must render the live-session block (`Live sessions: N (subs=K, discovery=...)`) whenever the status payload reports `live_session_count > 0` or `discovery_last_run_at is not None`, independent of whether `resolved_mode` is `watch` or `poll`. This ensures a running watch daemon behind a poll-mode launchd plist still surfaces its live-watch state in human output.
- `REQ-DAEMON-063`: Orphan-wiring lint. A repository test (can live under `tests/test_services/test_watcher_wiring.py`) must statically assert that `LiveSessionSet(` is constructed inside a shared helper reachable from `rpc_server._start_watch_mode`, and that no production module other than that shared helper constructs a `LiveSessionSet` instance. Tests may freely construct their own instances. This lint catches accidental re-introduction of a parallel dead-code path during future refactors.
- `REQ-DAEMON-064`: Watch-mode FTS rebuilds that fail with `FtsRebuildOutOfMemoryError` must enter an exponential retry backoff: `60s` initial, doubling per consecutive OOM, capped at `3600s`. During the active backoff window the FTS rebuild debouncer must report not-ready.
- `REQ-DAEMON-065`: A successful watch-mode FTS rebuild must reset the active OOM backoff counter and next-retry timestamp while preserving the latest failure timestamp and reason for status visibility.
- `REQ-DAEMON-066`: `recall daemon status` must expose `last_fts_rebuild_failure_at: datetime | null`, `last_fts_rebuild_failure_reason: string | null`, `fts_rebuild_consecutive_failures: int`, and `fts_rebuild_next_retry_at: datetime | null`.
- `REQ-DAEMON-067`: Watch-mode FTS OOM backoff warnings must be logged only when a rebuild attempt fails and changes the backoff state, not on later loop ticks that merely observe the debouncer as not-ready.

#### Invariants

- The live database remains readable by search commands during writes. When the RPC daemon owns the connection, DuckDB MVCC provides read consistency. In standalone mode, file-level isolation via the work-copy swap serves this role.
- Database replacement is atomic at the file level (standalone mode only; the RPC path writes in-place).
- A daemon `--once` cycle is behaviorally equivalent to one scheduled cycle.
- Installed scheduler integrations are user-scoped and do not require root privileges.
- Timer-driven integrations invoke `recall daemon --once`; they do not spawn nested long-lived daemons.
- Status metadata is updated only after successful indexing. In standalone mode, this occurs after the work-copy swap; via the RPC daemon, it occurs after writes commit on the shared connection.
- Watch mode writes are serialized on the main thread; the observer thread only updates a debounce queue.
- Watch mode advisory locks are held briefly per-session write, not for the watcher's lifetime.
- The watch-mode live session set is always a subset of the parser-known session set; promotion cannot invent paths that no parser would index.
- On macOS, at most one observer schedule exists per parent directory of any live-set member. On Linux, at most one observer schedule exists per live-set member path.
- The discovery loop and the fsevent handler must not race on the live set; access is serialized by a single lock so promotion, demotion, and eviction appear atomic to both producers.
- Idle demotion for a file must complete — schedule released and the path removed from the live set — before the next discovery cycle examines that same file, so a stale entry cannot shadow a new one.
- When the live set is empty, the observer holds zero subscriptions and the daemon's only periodic work is the discovery loop.

#### Non-goals

- Multi-process distributed scheduling.
- Live progress/state APIs beyond CLI logs.
- Installing or managing system-wide root services.
- Supporting every init system on Linux.
- Rich desktop/mobile push notifications.
- Process-level introspection of which sessions are "open" in an editor or agent — no `lsof`, no `/proc/*/fd` scanning, no platform-specific fd enumeration. `mtime` is the sole liveness signal.
- Cross-host session tracking. The live set is always local to the machine running the daemon.
- Ranking or prioritization within the live set. All live sessions are equal; there is no "hot" vs "warm" distinction among active members.
- Removal of `--mode poll`. Poll mode must remain available for CI and debugging; only the `auto` resolver changes.

#### Algorithm

**Periodic daemon cycle:**
1. Acquire daemon/index advisory lock
2. Copy the current live database to a work database path
3. Run incremental indexing (without embedding — see `REQ-ADAPT-001`) against the work database
4. Rebuild FTS indexes on the work database when indexing changed rows
5. Close the work database
6. Atomically replace the live database with the work database
7. Sleep until the next interval and repeat

If any step fails after the work copy is created but before swap, discard the work copy and keep the live database unchanged.

**Watch mode daemon cycle (`recall daemon --mode watch`):**

The sole entry point is `recall.services.rpc_server.RpcServer._start_watch_mode` (see `REQ-DAEMON-057`). Both `recall daemon start` (via `_start_foreground_server`) and any future scheduler unit configured for watch mode reach the active-session machinery through that async routine.

1. Perform a catch-up scan using the staleness algorithm (same as poll mode, per `REQ-DAEMON-049`)
2. Seed the live session set by calling `live_candidates(now, live_idle_threshold)` across every registered parser (`REQ-DAEMON-043`)
3. Start a `watchdog` Observer with narrow subscriptions per `REQ-DAEMON-044` — one schedule per live file on Linux, one schedule per parent directory on macOS. Do not schedule a recursive watch on any parser `watch_root` when the live set is non-empty.
4. Start the discovery loop on its own interval (default `30s`, `REQ-DAEMON-045`)
5. On file events, update a per-file debounce queue (thread-safe) and refresh the live-set member's "last-event" stamp
6. Main loop polls the debounce queue every 1 second:
   a. Acquire advisory lock briefly
   b. Parse the session file (incremental per `REQ-DAEMON-029`)
   c. Upsert the session directly to the live database
   d. Release advisory lock
   e. Mark FTS rebuild as dirty
7. Every `live_discovery_interval` tick (`REQ-DAEMON-045`/`REQ-DAEMON-046`):
   a. Call `live_candidates` per parser and promote new entries into the live set (subject to `live_max_subscriptions` and eviction rules in `REQ-DAEMON-052`)
   b. Demote any live-set member whose `mtime` has been unchanged and for which no fsevent has fired in the last `live_idle_threshold` seconds
   c. Reconcile observer subscriptions — schedule new entries, release demoted entries (file on Linux, parent dir on macOS only when no other live member still shares it)
   d. Update the discovery counters exposed on `DaemonSchedulerStatus` (`REQ-DAEMON-050`)
8. When the FTS debounce window elapses with no new dirty marks, rebuild FTS indexes
9. On SIGTERM/SIGINT: flush all pending files, stop the observer, stop the discovery loop, close the database

#### Scheduler Integration

**macOS (`launchd`):**
- Install path: `~/Library/LaunchAgents/it.send.recall.daemon.plist` (label `it.send.recall.daemon`; earlier releases used `xyz.metalrodeo.recall.daemon`, which install retires per `REQ-DAEMON-076`)
- Command: invoke the installed `recall daemon --once` entrypoint
- Schedule: `StartInterval = <daemon.interval>`
- Standard output/error: written to recall-managed log files under the user data or log directory
- Activation: `launchctl enable gui/$UID/<label>`, then `launchctl bootstrap gui/$UID <plist>`
- Watch mode: uses `KeepAlive` and `RunAtLoad` instead of `StartInterval`; command omits `--once` and adds `--mode watch`

**Linux preferred (`systemd --user`):**
- Unit paths:
  - `~/.config/systemd/user/recall-daemon.service`
  - `~/.config/systemd/user/recall-daemon.timer`
- Service command: invoke the installed `recall daemon --once` entrypoint
- Timer schedule: `OnUnitActiveSec=<daemon.interval>`
- Activation: `systemctl --user daemon-reload`, `enable --now recall-daemon.timer`
- Logs: standard `journald` user-unit logs

**Linux fallback (`cron`):**
- Installed only when explicitly requested or when `systemd --user` is unavailable and the user accepts the fallback
- Crontab entry invokes `recall daemon --once`
- Interval mapping:
  - supported directly for minute-aligned intervals (`>= 60s`)
  - sub-minute intervals are not supported by cron and must fail with a descriptive error
- Cron fallback is best-effort and provides weaker status/logging than `systemd --user`

**Scheduler selection:**
- `auto`:
  - macOS → `launchd`
  - Linux with working `systemd --user` → `systemd`
  - Linux without working `systemd --user` → error with suggestion to use `--scheduler cron`
- explicit:
  - `--scheduler launchd|systemd|cron`
  - invalid platform/scheduler combinations fail fast

#### Status Metadata

The daemon and direct indexing flows persist operational metadata for CLI visibility.

**Persisted fields:**
- last attempted cycle timestamp
- last successful cycle timestamp
- last cycle kind (`index`, `daemon-once`, `daemon-scheduled`, `daemon-watch`)
- last indexed session counts (`indexed`, `skipped`, `failed`, `total`)
- last failure message and timestamp
- installed scheduler kind, when known

**Freshness semantics:**
- “Last indexed” means the last successful update to the live database.
- Work-copy progress is not reported as live freshness until the swap completes.
- Failed cycles update failure metadata but do not advance “last indexed”.

#### CLI Integration

```bash
# Runtime (embedding is inline during indexing, enabled by default on supported hosts)
recall daemon --interval 300
recall daemon --once
recall daemon --mode watch
recall daemon --mode poll

# Scheduler management
recall daemon install
recall daemon install --scheduler launchd
recall daemon install --scheduler systemd
recall daemon install --scheduler cron
recall daemon uninstall
recall daemon status

# Optional status visibility
recall search "query"                  # may print one-line index/daemon freshness notice
recall list                            # may print one-line index/daemon freshness notice
```

`recall daemon install` writes scheduler artifacts. The daemon indexes without embedding; embedding is handled by a separate adaptive phase (see [Adaptive Daemon](#adaptive-daemon-two-phase-indexembed-pipeline)). In poll mode, installed schedulers execute `recall daemon --once` on each tick. In watch mode, the installed scheduler runs `recall daemon --mode watch` as a long-lived process with `KeepAlive`.

**Status notice rules:**
- Human-readable commands may print a compact notice before normal output, for example:
  - `Index: last updated 12m ago via daemon (indexed 3, skipped 7842)`
  - `Daemon: installed via launchd, last success 5m ago`
- JSON output must not include human-readable notices outside the JSON payload.
- A future `--no-status` flag may suppress notices for scripting-oriented human-readable usage.

#### Configuration

Environment variables:
- `RECALL_DAEMON_INTERVAL` — periodic interval in seconds
- `RECALL_DAEMON_EMBED` — whether daemon cycles embed inline during indexing (`true` / `false` / `auto`, default `auto`)
- `RECALL_DAEMON_SOURCE` — optional single-source filter (`claude-code`, `codex`, `pi-agent`, `grok`, `kimi-code`)
- `RECALL_DAEMON_SCHEDULER` — optional scheduler preference (`auto`, `launchd`, `systemd`, `cron`)
- `RECALL_DAEMON_MODE` — daemon mode (`auto`, `watch`, `poll`)
- `RECALL_DAEMON_DEBOUNCE` — per-file debounce interval in seconds for watch mode (default `5`)
- `RECALL_DAEMON_FTS_DEBOUNCE` — FTS rebuild debounce interval in seconds for watch mode (default `10`)
- `RECALL_DAEMON_LIVE_IDLE_THRESHOLD` — seconds a session must be unmodified before it is demoted from the active-session live set (default `300`)
- `RECALL_DAEMON_LIVE_DISCOVERY_INTERVAL` — seconds between `live_candidates` discovery sweeps (default `30`)
- `RECALL_DAEMON_LIVE_MAX_SUBSCRIPTIONS` — hard upper bound on the live session set size (default `64`)
- `RECALL_CLI_STATUS_NOTICES` — whether human-readable commands print daemon/index freshness notices (`true` / `false`)

Config file (`~/.config/recall/config.toml`):
```toml
[daemon]
interval = 300
embed = "auto"
source = "codex"
scheduler = "auto"
mode = "auto"
debounce = 5
fts_debounce = 10
live_idle_threshold = 300
live_discovery_interval = 30
live_max_subscriptions = 64

[cli]
status_notices = true
```

#### Acceptance Criteria

- [ ] `recall daemon --once` indexes new content and exits without sleeping.
- [ ] Two daemon cycles against unchanged input skip reprocessing on the second cycle.
- [ ] A changed file is reindexed on the next cycle.
- [ ] A failed work-copy cycle leaves the previous live database intact.
- [ ] Search remains usable against the live database while the daemon cycle writes the work copy.
- [ ] `recall daemon install` chooses `launchd` on macOS and `systemd --user` on Linux by default when supported.
- [ ] `recall daemon install --scheduler cron` installs a user crontab entry that runs `recall daemon --once`.
- [ ] Unsupported scheduler/platform combinations fail with descriptive errors.
- [ ] `recall daemon uninstall` removes recall-managed scheduler artifacts and can be run repeatedly without error.
- [ ] `recall daemon status` reports the installed scheduler type and effective command path.
- [ ] Successful index and daemon runs persist last-run metadata that can be queried later.
- [ ] Human-readable commands can show last indexed / daemon freshness without affecting JSON output.
- [ ] Failed daemon cycles preserve the last successful live-index timestamp while updating failure metadata.
- [ ] `recall daemon install` refuses to write scheduler artifacts when the resolved `recall` binary does not exist, and leaves no partial artifacts behind on failure (REQ-DAEMON-033).
- [ ] `recall daemon install` runs a `--version` smoke test against the installed program path before returning success, and rolls the install back on non-zero exit (REQ-DAEMON-034).
- [ ] `recall daemon install` waits out an asynchronous `launchctl bootout` under an explicit bound and retries the bootstrap a bounded number of times instead of treating a mid-teardown `launchctl print` as proof the service is live (REQ-DAEMON-071).
- [ ] `recall daemon install` verifies the unit is loaded after `bootstrap` returns and rolls back rather than reporting a successful install with no daemon running (REQ-DAEMON-072).
- [ ] `recall daemon stop` names the orphaned pid when a live daemon is running that launchd does not manage, instead of returning `Boot-out failed: 3: No such process` (REQ-DAEMON-073).
- [ ] A default `recall daemon status` requests a 10-row source page, still falls back to the local read when the daemon is down, opens its text output with the `Daemon:`/`embed:`/`reconciliation:`/`health:` summary block, and prints no bare epoch float (REQ-DAEMON-075).
- [ ] With only the legacy launchd plist installed, `recall daemon start`/`restart`, `recall compact` and `recall daemon status` act on the legacy label and status warns it needs migrating without calling a mutating `launchctl` verb; `recall daemon install` then leaves only `it.send.recall.daemon` loaded and the legacy plist gone, aborts before any bootstrap when the legacy job does not unload, restores and re-bootstraps the legacy job when bootstrap retries run out, and `recall daemon uninstall` removes both labels; with both labels loaded, status warns of the leftover legacy job and `recall daemon stop` unloads both; a rollback whose new job will not unload never bootstraps the legacy job and keeps the new plist, and a failed re-install on a migrated host re-bootstraps the previous plist (REQ-DAEMON-076).
- [ ] With another process holding the database read-write and the pid file naming it, `daemon_status(config)` returns `daemon_pid` and `runtime_unavailable_reason` with default runtime fields instead of raising; with a dead pid in the file the runtime fields are read; if the pid appears between the initial check and the lock conflict the holder is attributed to that daemon; `install_scheduler` under the same holder writes the unit and returns that status; `uninstall_scheduler` retries a transient teardown lock and clears the persisted scheduler kind, stops at the 30-second deadline when the conflict persists, and propagates a non-lock DuckDB I/O error immediately; `recall daemon status` with the RPC down exits 0 and prints the `Daemon: pid N alive, RPC not answering` line in text and both fields in JSON, with no "has not completed a successful run" notice on stderr even when installed; the real CLI-to-daemon integration path treats that lock error as a failure rather than a skip; a holder with no pid file yields `daemon_pid = null` and a reason naming another process and its pid, on the status and install paths alike; a status served over RPC carries the serving daemon's own pid in `daemon_pid`; the manifest lists them (REQ-DAEMON-074).
- [ ] `recall daemon status` surfaces `installed_binary_path` and `installed_binary_stale` when the baked-in path is missing or diverges from the currently resolved `recall` binary on both launchd and systemd hosts (REQ-DAEMON-035).
- [ ] `recall daemon status` surfaces `scheduler_last_exit_status` and `scheduler_health_state` from `launchctl list`/`systemctl --user show` on the respective platforms, and returns `null` on cron; the launchd parse reads `"LastExitStatus" = N;` out of the plist-style dictionary, so a loaded job reads `0`/`"ok"` rather than `"unknown"` (REQ-DAEMON-036).
- [ ] Human-readable status notices emit a freshness warning when the daemon is installed but has never succeeded, or when the last success is older than `max(2 * interval, 600s)` in poll mode / `3600s` in watch mode (REQ-DAEMON-037).
- [ ] Human-readable status notices emit a drift warning naming the baked-in binary and the currently resolved binary when `installed_binary_stale` is true, taking precedence over freshness and mode-mismatch notices (REQ-DAEMON-038).
- [ ] Human-readable status notices emit a non-zero-exit warning when the scheduler reports a recent failure, and map launchd exit `78` to a "launchd could not execute the configured program" message (REQ-DAEMON-039).
- [ ] `watchdog>=4.0` resolves as a base dependency of a fresh `uv sync` without selecting any optional extras (REQ-DAEMON-040).
- [ ] `resolve_daemon_mode(DaemonMode.AUTO)` returns `DaemonMode.WATCH` on every supported platform regardless of prior extras-group history (REQ-DAEMON-041).
- [ ] `SessionParser.live_candidates(now=…, idle_threshold=…)` returns only session paths whose `mtime >= now - idle_threshold`, for every concrete parser (REQ-DAEMON-042).
- [ ] Starting `recall daemon --mode watch` seeds the live set from `live_candidates` across all parsers before the observer begins processing events (REQ-DAEMON-043).
- [ ] On Linux, the observer holds exactly one schedule per live-set member; on macOS, exactly one schedule per distinct parent directory of live-set members. No recursive `watch_root` schedule is created when the live set is non-empty (REQ-DAEMON-044).
- [ ] The discovery loop promotes a jsonl file into the live set within `live_discovery_interval` seconds of its `mtime` entering the idle window (REQ-DAEMON-045).
- [ ] A live-set member whose `mtime` has been unchanged for `live_idle_threshold` seconds and has received no fsevents in that window is demoted and its observer schedule released (REQ-DAEMON-046).
- [ ] With no active sessions on the host, `watcher_subscription_count` in `recall daemon status` is `0` and only the discovery loop is running (REQ-DAEMON-047).
- [ ] The installed macOS launchd plist for watch mode still uses `KeepAlive` (the `PathState` dict of REQ-RESIL-024, kept alive while the refusal marker is absent) + `RunAtLoad=true` and the daemon runs as a single long-lived process (REQ-DAEMON-048).
- [ ] A session file updated while the daemon was down is indexed on the next startup via the catch-up scan before the discovery loop begins (REQ-DAEMON-049).
- [ ] `recall daemon status` (JSON) exposes `live_session_count`, `live_session_paths` (top 10 by mtime), `discovery_interval_seconds`, `discovery_last_run_at`, `discovery_last_promoted`, `discovery_last_demoted`, and `watcher_subscription_count` (REQ-DAEMON-050).
- [ ] `RECALL_DAEMON_LIVE_IDLE_THRESHOLD`, `RECALL_DAEMON_LIVE_DISCOVERY_INTERVAL`, and `RECALL_DAEMON_LIVE_MAX_SUBSCRIPTIONS` override the config defaults and are visible in `recall daemon status` (REQ-DAEMON-051).
- [ ] When a promotion would exceed `live_max_subscriptions`, the oldest-mtime live-set member is evicted; when all current members are strictly newer than the candidate, the promotion is rejected and a warning is logged (REQ-DAEMON-052).
- [ ] Per-file debouncing and FTS rebuild debouncing still apply to events that flow through the active-session watcher unchanged (REQ-DAEMON-053).
- [ ] `live_candidates` implementations rely solely on `mtime`; no call into `lsof`, `/proc`, or platform-specific fd enumeration exists in the codebase (REQ-DAEMON-054).
- [ ] When observer scheduling fails (simulated `PermissionError` / `OSError`), the daemon logs a warning, excludes the path from the observer, and still re-indexes the path via the discovery loop (REQ-DAEMON-055).
- [ ] Unit tests drive promotion, demotion, eviction, and graceful-degradation paths deterministically using a synthetic clock and synthetic fsevents without instantiating a real `watchdog.Observer` (REQ-DAEMON-056).
- [ ] `recall daemon start` / `recall daemon --mode watch` routes through `RpcServer._start_watch_mode`, which instantiates `LiveSessionSet`, seeds it, schedules observer subscriptions, runs the discovery loop, and updates the live snapshot — with no parallel Observer setup elsewhere in the production codebase (REQ-DAEMON-057).
- [ ] Seeding, reconciliation, discovery-tick, and snapshot-update routines are shared helpers in `services/watcher.py`; both synchronous and async callers invoke the same functions and any change propagates to both (REQ-DAEMON-058).
- [ ] `WatcherLiveSnapshot` is updated after seed, after every reconcile, after every discovery tick, and after every fsevent-driven promote; and it is reset on watch-mode shutdown (REQ-DAEMON-059).
- [ ] With an installed launchd plist running `recall daemon --once` (poll mode) and a watch-mode RPC daemon running in the background, `recall daemon status` reports `resolved_mode: watch`, populates live_session_* fields, and still exposes `installed_mode: poll` separately so mode-mismatch notices can fire (REQ-DAEMON-060).
- [ ] An integration test starts `RpcServer._start_watch_mode`, writes bytes to a jsonl file under a parser `watch_root`, and asserts `get_live_snapshot()` reports non-zero `live_session_count`, non-zero `watcher_subscription_count`, and a non-null `discovery_last_run_at` within a test-overridden discovery interval (REQ-DAEMON-061).
- [ ] `recall daemon status` text output renders the live-session block whenever `live_session_count > 0` or `discovery_last_run_at is not None`, regardless of `resolved_mode` (REQ-DAEMON-062).
- [ ] A wiring-lint test statically verifies that `LiveSessionSet(` is constructed from a single shared helper reachable from `RpcServer._start_watch_mode`, with no other production construction sites (REQ-DAEMON-063).
- [ ] FTS rebuild OOMs in watch mode back off for `60s`, double on consecutive failures, cap at `3600s`, and suppress retry readiness until the backoff window expires (REQ-DAEMON-064).
- [ ] A successful FTS rebuild clears the active OOM backoff counter and next-retry timestamp while preserving the latest failure metadata (REQ-DAEMON-065).
- [ ] `recall daemon status` exposes `last_fts_rebuild_failure_at`, `last_fts_rebuild_failure_reason`, `fts_rebuild_consecutive_failures`, and `fts_rebuild_next_retry_at` with `null` values when no OOM/backoff state exists (REQ-DAEMON-066).
- [ ] Repeated watch-mode loop ticks during an active FTS OOM backoff do not emit additional backoff warning logs (REQ-DAEMON-067).

### Adaptive Daemon: Two-Phase Index/Embed Pipeline

#### Problem

The daemon currently embeds inline during indexing — every file-change event triggers MLX model inference (300-400% CPU for several seconds) plus 3GB resident memory for the warm model. With multiple concurrent Claude Code sessions, the daemon is almost continuously embedding, consuming significant CPU and memory on the user's laptop. Real-time embedding is wasteful because users rarely vector-search active sessions; keyword/FTS search is sufficient until sessions settle.

#### Solution

Split the daemon into two independent phases:

1. **Index phase** — real-time, event-driven. Parse + DB write + FTS rebuild only. No embedding. Fast (~0.3s per batch).
2. **Embed phase** — batched, adaptive. Periodically checks preconditions (session idle, system load, power state), then batch-embeds all un-embedded content. Loads MLX model on demand, unloads after idle timeout.

The DB serves as the work queue: un-embedded content is discovered via `LEFT JOIN ... WHERE embedding IS NULL`. No separate queue data structure or state management needed. Keyword search works immediately; vector search catches up when conditions allow.

#### Requirements

- `REQ-ADAPT-001`: The daemon index phase must not invoke embedding. Parsing, DB writes, and FTS rebuilds are the only work performed on file-change events.
- `REQ-ADAPT-002`: The daemon index phase in watch mode must batch all debounce-ready files into a single DB transaction per main-loop tick, with one FTS rebuild check at the end of the batch. Advisory lock held once per batch, not per file.
- `REQ-ADAPT-003`: The daemon must run a separate embed phase on a configurable timer interval (default 120 seconds). The embed phase discovers un-embedded content by querying for messages and tool calls that lack corresponding rows in `message_embeddings` and `tool_call_embeddings`.
- `REQ-ADAPT-004`: The embed phase must skip content belonging to sessions whose `file_mtime` is newer than a configurable idle threshold (default 600 seconds). Only idle sessions are eligible for embedding.
- `REQ-ADAPT-005`: The embed phase must check system load before starting a batch. On all platforms, `os.getloadavg()[0]` must be below a configurable fraction of `os.cpu_count()` (default 0.7). When the threshold is exceeded, the batch is skipped.
- `REQ-ADAPT-006`: On macOS, the embed phase must check power state via `IOPSCopyPowerSourcesInfo`. When on battery, the load threshold is reduced to a separate configurable fraction (default 0.3). On non-macOS platforms, power state is not checked. A load deferral names which ceiling applied and the power state that selected it — `load 41.0 > battery threshold 5.4 (on battery: Battery Power)` against `load 41.0 > load threshold 72.0 (AC power)` — because the two fractions differ by an order of magnitude and a reason naming a bare "threshold" leaves an operator whose `pmset` says AC unable to tell a misread power state from a stale reason. The reason carries the time it was evaluated per `REQ-ADAPT-012`.
- `REQ-ADAPT-007`: When embed phase preconditions fail, the next check must be delayed by a configurable backoff interval (default 300 seconds) instead of the normal embed interval.
- `REQ-ADAPT-008`: After completing an embed batch, the embed phase must recheck for work after a short interval (30 seconds) to pick up sessions that became idle during the previous batch.
- `REQ-ADAPT-009`: The MLX embedding model must be loaded on demand when the embed phase has work to do. If no embed batch runs for a configurable timeout period (default 600 seconds), the model must be unloaded — backend reference deleted, `gc.collect()` called — to reclaim memory (~3GB).
- `REQ-ADAPT-010`: The existing `daemon.embed` config field must change semantics from "embed inline during indexing" to "enable the embed phase." Default remains `auto` (enabled when MLX backend is available). Setting `false` disables the embed phase entirely (keyword-only mode).
- `REQ-ADAPT-011`: The `recall index` CLI must retain `--embed` / `--no-embed` flags for manual one-shot embedding, independent of the daemon's embed phase.
- `REQ-ADAPT-012`: `recall daemon status` must report embed phase state: whether enabled, whether the MLX model is currently loaded, count of pending un-embedded items with the time that count was measured, timestamp and size of the last embed batch, loop liveness — iteration count, last iteration time, the trigger that drove the last cycle, the stage the loop last entered with the time it entered it, the last iteration outcome, the reason the phase last deferred a cycle with the time that reason was evaluated (both cleared once a timer cycle proceeds past its preconditions), the interval the timer will wait next, the number of sessions currently deprioritized for cooldown with the wall-clock deadline the last cooldown expires, and the last embed error when one is set. Every embed cycle records its trigger, outcome and batch through the shared phase state, whichever path drives it: the phase timer (`trigger=timer`) and requested enrichment from indexing or context recomputation (`trigger=requested`). The pacing fields — iteration count, stage, stage time and next interval — belong to the timer alone. A requested cycle runs while the timer sleeps in a stage of its own, so it reports its count, its stage and that stage's time in separate `embed_requested_*` fields and leaves the timer's untouched: an index request must not age a stage the daemon already left, and enriching a thousand sessions is not a thousand timer iterations. A requested cycle always closes with a terminal outcome, including one that fails before it commits (`outcome=failed`), so no status ever describes requested work that already stopped. The deferral reason is reported in the embed block; `reconciliation.enrichment_deferred` keeps carrying the same value for existing readers. A stalled loop must be distinguishable from an idle one without process introspection, skipped work must be distinguishable from slow work, and a pending count frozen because the backlog is idle must be distinguishable from one frozen because no cycle has measured it since. In text output the last-batch and last-iteration timestamps read as local ISO-8601 with a relative age per `REQ-DAEMON-075`; the structured fields stay epoch seconds.
- `REQ-ADAPT-013`: Existing `REQ-DAEMON-024` (keep MLX model warm in watch mode) is superseded by `REQ-ADAPT-009` (load on demand, unload after idle timeout).
- `REQ-ADAPT-014`: Existing `REQ-DAEMON-003` inline embedding during poll cycles is superseded by `REQ-ADAPT-001`. Poll mode cycles index without embedding; the embed phase handles embedding separately.
- `REQ-ADAPT-015`: `REQ-SEM-013` (default inline embedding on supported hosts) applies only to `recall index` CLI, not to daemon operation. The daemon delegates embedding to the embed phase.
- `REQ-ADAPT-016`: A session that cannot be committed must not stop the embed drain. The commit-time staleness check must compare the snapshot against the database keyed by message/tool-call identity, never by row order (`tool_calls.idx` is not unique within a session, so `ORDER BY idx` is not a total order). Rejections are counted per session, and only where a commit was attempted: a session whose generated result that check discards counts one rejection, and a session that commits forgets its count. A cycle short-circuited by the no-progress backoff is pure pacing — it attempts no commit, so it must neither count a rejection nor extend the backoff, or a threshold of N would fire on one real rejection plus N-1 idle timer ticks (`embed_interval` is shorter than `embed_backoff`, so those ticks always arrive). Once a session reaches a bounded number of rejections the embed phase must deprioritize it for a bounded cooldown derived from `daemon.embed_backoff`, drop its count with it, keep embedding the other pending sessions, and log the skip at WARNING once per cooldown. Every roster the phase compares — the cycle's snapshot and the roster whose signature paces the backoff — must exclude the deprioritized sessions: a cooled session keeps its place in the raw oldest-first order, so an unfiltered signature could never match the next cycle's snapshot and the pacing would be bypassed. A cycle that finds no eligible session drops the stall pacing and every rejection count, whether or not a pause ends that cycle; a backend or model change drops the cooldowns with it. Both the cooldown map and the rejection counts must be bounded (a fixed maximum number of entries) and must stay readable from the status thread while a cycle mutates them on the writer executor. Cooldown state is reported through `daemon status` per `REQ-ADAPT-012`.
- `REQ-ADAPT-017`: The background embed phase is opportunistic and must yield to an operator index request. While one or more `recall.index`, `recall.daemon_run`, or context recompute requests are in flight, the embed loop must not start a new cycle; it must wait for them, report the wait through `daemon status` per `REQ-ADAPT-012` (a distinct stage and deferral reason), and resume after a bounded wait derived from `daemon.embed_backoff` so a request that never completes cannot disable enrichment. An index request therefore waits at most one already-running embed cycle in total, never one cycle per session it enriches. Mutual exclusion between the drain and requested enrichment is unchanged — this governs which of the two goes next, not whether they may overlap.
- `REQ-ADAPT-018`: A refusal caused by a paused reconciliation and a refusal caused by daemon shutdown are distinct conditions and must carry distinct messages. The pause message names how to resume; the shutdown message tells the operator to retry after the restart. No refusal may name both conditions at once.

#### Configuration

New config fields under `[daemon]`:

| Field | Env Var | Default | Description |
|-------|---------|---------|-------------|
| `embed_interval` | `RECALL_DAEMON_EMBED_INTERVAL` | `120` | Seconds between embed phase checks |
| `embed_backoff` | `RECALL_DAEMON_EMBED_BACKOFF` | `300` | Seconds to wait after preconditions fail |
| `embed_idle_session` | `RECALL_DAEMON_EMBED_IDLE_SESSION` | `600` | Seconds a session file must be unmodified before embedding |
| `embed_model_timeout` | `RECALL_DAEMON_EMBED_MODEL_TIMEOUT` | `600` | Seconds of no embed work before unloading MLX model |
| `load_threshold` | `RECALL_DAEMON_LOAD_THRESHOLD` | `0.7` | Max system load as a multiple of CPU count; values above 1.0 opt into embedding under oversubscription |
| `battery_threshold` | `RECALL_DAEMON_BATTERY_THRESHOLD` | `0.3` | Max load fraction allowed when on battery |

Existing fields with changed semantics:

| Field | Change |
|-------|--------|
| `daemon.embed` | "embed inline during indexing" → "enable embed phase" (values/default unchanged) |
| `daemon.debounce` | Now applies to index phase batching only |
| `daemon.fts_debounce` | Now applies to index phase FTS rebuild only |

#### Algorithm

**Index phase (watch mode):**
1. File event → update debounce queue (existing)
2. Main loop tick (1 second):
   a. Drain all debounce-ready files from queue
   b. If any ready: acquire advisory lock once
   c. Parse all ready files (incremental byte-offset)
   d. Write all sessions in a single DB transaction
   e. Release advisory lock
   f. Check FTS rebuild (existing 10s debounce)

**Embed phase (runs in watch mode; in poll mode, runs between cycles against the live DB):**

In poll mode, the embed phase writes directly to the live database (not the work copy). Poll cycle swaps do not affect embedding tables because the work copy is created from the live DB (which already contains prior embeddings) and the embed phase only writes to `message_embeddings` / `tool_call_embeddings` via idempotent upserts.

1. Timer fires (every `embed_interval` seconds)
2. Query for un-embedded content:
   ```sql
   SELECT m.id FROM messages m
   JOIN session_state ss ON ss.session_id = m.session_id
   LEFT JOIN message_embeddings me ON me.message_id = m.id
   WHERE me.message_id IS NULL
     AND ss.file_mtime < epoch(now()) - :idle_threshold
   ```
   (Same pattern for tool_calls/tool_call_embeddings)
3. If no work: check model idle timeout, unload if exceeded, sleep `embed_interval`
4. Check preconditions:
   - System load: `os.getloadavg()[0] < load_threshold * os.cpu_count()`
   - Power state (macOS): if on battery, use `battery_threshold` instead
5. If preconditions fail: log reason, sleep `embed_backoff`
6. If preconditions pass:
   a. Load MLX model if not loaded
   b. Batch-embed all eligible content (existing `_batch_embed` logic)
   c. Write embeddings to DB
   d. Update `last_used_at` timestamp
   e. Sleep 30 seconds, then recheck for more work

**Model lifecycle:**
1. Model loaded on first embed batch (existing lazy-load path)
2. `last_used_at` updated after each batch
3. On each embed timer tick with no work: if `now - last_used_at > embed_model_timeout`, set backend to `None`, call `gc.collect()`
4. Next batch with work triggers reload (~2-3s)

#### Invariants

- The index phase never loads the embedding model or generates embeddings.
- Keyword and FTS search work immediately after indexing, regardless of embed phase state.
- The embed phase never modifies content tables (`messages`, `message_state`, `tool_calls`). It only writes to `message_embeddings` and `tool_call_embeddings`.
- Un-embedded content is always discoverable via DB query. No separate queue state can become inconsistent.
- The embed phase is idempotent: running it multiple times on the same content produces the same result.
- Model unload reclaims memory; model reload is transparent to the embed pipeline.
- Power state and load checks are advisory — they skip the batch, they do not error.
- In poll mode, work-copy-swap preserves embeddings because the work copy is created from the live DB (which contains prior embed-phase writes) and embed-phase upserts are idempotent.

#### Non-goals

- Cloud embedding backend (future work; the embed phase architecture supports swapping backends, but this spec covers local MLX only).
- Per-session or per-message embedding priority.
- GPU memory pressure sensing (load average is a sufficient proxy).
- Real-time embedding for any use case (users needing immediate vector search use `recall index --embed`).

#### Observability

Index phase logging (INFO level):
- `index batch: {n} sessions ({x.xs})` — one line per batch

Embed phase logging (INFO level):
- `embed check: {n} pending across {m} sessions ({k} idle)` — on each timer tick
- `embed skipped: {reason}` — when preconditions fail (e.g., `load 8.2 > threshold 7.0`, `on battery`, `no idle sessions`)
- `embed batch: {n} items across {m} sessions ({x.xs})` — on completion
- `embed model loaded ({x.xs})` / `embed model unloaded (idle {x}m)` — lifecycle events

Structured JSON (daemon.log):
- Existing `index_summary` unchanged
- New `embed_summary`: `{pending, embedded, skipped_reason, model_loaded}`

`recall daemon status` JSON additions:
- `embed_phase_enabled`: bool
- `embed_model_loaded`: bool
- `embed_pending`: int
- `embed_pending_at`: timestamp the pending count was measured
- `embed_last_batch_at`: timestamp
- `embed_last_batch_size`: int
- `embed_loop_iterations`, `embed_loop_last_iteration_at`, `embed_loop_last_trigger`,
  `embed_loop_stage`, `embed_loop_stage_at`, `embed_loop_last_outcome`,
  `embed_loop_next_interval`: timer-loop liveness (REQ-ADAPT-012)
- `embed_requested_cycles`, `embed_requested_at`, `embed_requested_stage`,
  `embed_requested_stage_at`: requested-enrichment liveness (REQ-ADAPT-012)
- `embed_deferred_reason`: why the last cycle deferred, or null
- `embed_deferred_at`: when that reason was evaluated, or null

#### Migration

- No schema changes. The DB-as-queue pattern uses existing tables.
- Embed phase operational state (`embed_last_batch_at`, `embed_last_batch_size`, `embed_pending`, `embed_pending_at`, `embed_model_loaded`, the `embed_loop_*` and `embed_requested_*` liveness fields, `embed_deferred_reason` and `embed_deferred_at`) is volatile (daemon memory only). These fields reset on daemon restart. This is acceptable because the embed phase rediscovers work from the DB on every tick.
- Existing `daemon.embed = true` users get the new behavior automatically — embedding still happens, just deferred.
- Existing `daemon.embed = false` users are unaffected — embed phase stays disabled.
- First run after upgrade may see a larger initial embed batch (accumulated un-embedded content). Load/battery preconditions throttle this naturally.
- `REQ-DAEMON-024` is superseded but not deleted from the spec; marked as superseded by `REQ-ADAPT-009`.

#### Acceptance Criteria

- [ ] Watch mode file events trigger parse + DB write only, no embedding model loaded.
- [ ] Watch mode batches multiple debounce-ready files into a single DB transaction.
- [ ] Idle daemon with no active sessions and MLX model unloaded shows near-zero CPU usage.
- [ ] `recall daemon status` reports embed phase state including pending count.
- [ ] Sessions actively being written to are not embedded until idle for `embed_idle_session` seconds.
- [ ] Embed phase skips batches when system load exceeds `load_threshold * cpu_count`.
- [ ] Embed phase uses stricter threshold on battery (macOS).
- [ ] MLX model unloads after `embed_model_timeout` seconds of no embed work.
- [ ] MLX model reloads transparently when next embed batch is needed.
- [ ] `recall index --embed` still works for manual one-shot embedding.
- [ ] Keyword/FTS search returns results immediately after indexing, before embedding runs.
- [ ] Vector/hybrid search returns results for sessions that have been embedded by the embed phase.
- [ ] Embed phase is idempotent — repeated runs on the same content produce no errors or duplicate rows.

### Daemon RPC

#### Problem

The CLI accesses DuckDB directly for reads (search, list, show, stats) and writes (index). The watch-mode daemon also holds the database for indexing. When both run concurrently, DuckDB file-level locking causes `IO Error: Could not set lock` failures — particularly during vector/hybrid search, which takes longer due to model loading. The current architecture has two independent DB access paths that cannot be safely multiplexed.

#### Solution

Move all database and embedding model access behind a single daemon process that serves CLI commands over a Unix domain socket using JSON-RPC 2.0. The CLI becomes a thin RPC client. The daemon process is the single owner of all DuckDB connections, the MLX embedding model, filesystem watchers, and FTS indexes. No CLI process opens database connections. This eliminates cross-process lock contention, keeps the model warm across queries, and provides a clean foundation for a future remote MCP server (HTTP transport added later).

**Threading model:** The daemon uses asyncio for connection management and request dispatch. Ordinary reads use a dedicated bounded executor with one per-task cursor and admission token; cancellation interrupts and drains only that cursor before its token is released. Query-model work uses a separate single-worker executor and owns no database cursor. Write operations and bounded source/enrichment preparation use the four-worker writer executor; statements on the shared write connection remain serialized by an asyncio lock. Connection lifecycle and cancellation ownership follow REQ-RESIL-021 through REQ-RESIL-023. The event loop thread handles only dispatch, progress streaming, and socket I/O — it never blocks on DuckDB queries, model calls, or long-running index operations. The embedding model is lazy-loaded and stays warm across requests (MLX/numpy operations are thread-safe).

#### Requirements

- `REQ-RPC-001`: The daemon must listen on a Unix domain socket at `{data_dir}/recall.sock` (default `~/.local/share/recall/recall.sock`). The socket file must be created on daemon startup and removed on clean shutdown. The socket is owner-only (mode 0600, set before it listens). recall creates a missing data directory, and its `logs/` subdirectory, owner-only (0700); an existing directory keeps its mode.
- `REQ-RPC-002`: The wire protocol must be JSON-RPC 2.0 over newline-delimited messages on the Unix socket. Each request and response is a single JSON object terminated by `\n`.
- `REQ-RPC-003`: RPC method names must use a flat `recall.` prefix namespace: `recall.index`, `recall.search`, `recall.list`, `recall.show`, `recall.stats`, `recall.stats_tools`, `recall.stats_bash`, `recall.stats_tokens`, `recall.stats_usage`, `recall.stats_skills`, `recall.daemon_status`. Method parameters must match the existing CLI parameter schemas defined in `manifest.py`.
- `REQ-RPC-004`: All CLI commands (`index`, `search`, `list`, `show`, `stats`, `daemon status`) must be implemented as RPC calls to the daemon. The CLI layer must not import or call service/DB modules directly.
- `REQ-RPC-005`: If no daemon is running when a CLI command is invoked, the CLI must auto-fork a daemon process in the background before sending the RPC request. The forked daemon stays alive after the originating command completes.
- `REQ-RPC-006`: Daemon management commands (`daemon install`, `daemon uninstall`, `daemon start`, `daemon stop`, `daemon restart`) must work without an active RPC connection — they operate on scheduler artifacts and process signals directly.
- `REQ-RPC-007`: In watch mode, the daemon must stay alive indefinitely (until `recall daemon stop`, SIGTERM, or system reboot). In poll or on-demand mode, the daemon must exit after an idle timeout (default 30 minutes, configurable via `daemon.idle_timeout`).
- `REQ-RPC-008`: Long-running operations (`recall.index` with `--full` or `--recreate`) must stream progress to the client via JSON-RPC notifications (messages with `id: null`) on the same socket connection. The client renders progress notifications to stderr. The client's read timeout is an idle timeout refreshed by each frame, so the daemon must emit a frame for every inventory batch it captures, not only for files that survive a `--since` or `--project` filter: a scoped request that matches nothing must still keep the connection alive to its summary.
- `REQ-RPC-009`: The embedding model must be lazy-loaded on the first `recall.search` request that requires vector or hybrid mode. The model is subject to the adaptive unload policy in `REQ-ADAPT-009` — it may be unloaded after the configured idle timeout and reloaded on the next request.
- `REQ-RPC-010`: Embedding model configuration changes (model swap, dimension change) must require a daemon restart. The daemon does not hot-reload models. The CLI should detect config staleness and suggest `recall daemon restart`.
- `REQ-RPC-011`: Destructive operations (`recall.index` with `recreate: true`) must include a `confirmed: true` parameter. The CLI handles interactive `--yes` confirmation locally before sending the RPC request.
- `REQ-RPC-012`: The daemon must handle multiple concurrent RPC connections. Ordinary reads (search, list, show, stats) must be admitted before submission to a dedicated bounded executor and served through per-call cursors from the daemon-owned connection. A cancelled read must interrupt only its cursor and retain admission and lifecycle ownership until the executor task closes it. Query embedding must run without a database cursor in a separate bounded executor. Shared-connection writes remain serialized by an asyncio lock and retain that lock until executor completion. Concurrent reads during writes must remain available; saturated reads or model work must not consume the writer executor's last worker.
- `REQ-RPC-013`: RPC errors must use JSON-RPC 2.0 error objects with `code`, `message`, and optional `data` fields. Error codes must map to the existing CLI error codes (`VALIDATION`, `RUNTIME`, `LOCKED`, `CONFIRMATION_REQUIRED`).
- `REQ-RPC-014`: A soft daemon stop must send SIGTERM to the daemon process, which triggers graceful shutdown: flush pending index writes, close the database, remove the socket file, and exit. The CLI exposes this legacy PID-file path as `recall daemon stop --soft`.
- `REQ-RPC-015`: The daemon must write its PID to `{data_dir}/recall.pid` on startup. The CLI uses this to detect a running daemon and send signals for `daemon stop`/`daemon restart`.
- `REQ-RPC-016`: The daemon must log to `{data_dir}/logs/daemon.log` (stdout) and `{data_dir}/logs/daemon.err.log` (stderr) when running as a background process.
- `REQ-RPC-017`: RPC request ids must be unique per request (never derived from a clock alone), and the client must close its socket on any exchange that leaves the byte stream unusable — read timeout, connection closed mid-response, undecodable response frame, or send failure — surfacing each as `RpcConnectionError`. Rationale: cancellation is the daemon's own decision and covers only the methods of `REQ-RPC-012` and `REQ-RPC-018`, so an abandoned request's late reply can stay queued on the connection; a reused client reading it with a colliding timestamp id once returned a different request's session payload as the answer.
- `REQ-RPC-018`: A long-running write request (`recall.index`, `recall.daemon_run`) owns no deadline of its own, but a write whose remainder reconciliation owns must end when the client that asked for it is gone. The connection loop is parked inside the handler while it runs, so nothing else on that connection observes EOF; the daemon must therefore watch for peer departure and shutdown alongside the handler and cancel it on either. The departure answers nobody — there is no frame to send and no error to report — so the cancelled run records its own outcome in runtime state rather than leaving an attempt that never ended. No work is lost by the cancellation, because indexing commits per session and reconciliation owns whatever the cancelled run had not reached, and no executor thread is abandoned, because a cancelled shared-connection await still drains under `REQ-RESIL-021`. Work watch mode cannot resume is exempt and runs to completion for nobody, logged at INFO when it starts and recorded on completion: `--recreate` (a cancellation after its reset leaves an emptied database with no run recorded), `--full` (a cancellation mid-inventory leaves batches it never captured un-observed) and `--recompute-context`. A request that keeps its client must not lose it to silence either: one source may be served for longer than the client's idle timeout, so the pass emits a keepalive frame for the source in flight. Rationale: an abandoned `recall index` held its index turn indefinitely, and every later index request queued behind it while the operator saw only a client-side idle timeout.

#### Domain Model

```
┌─────────────┐       Unix socket        ┌──────────────────────┐
│   CLI        │  ─── JSON-RPC 2.0 ───▶  │     Daemon           │
│  (client)    │  ◀── responses/notifs ── │                      │
└─────────────┘                           │  ┌────────────────┐  │
                                          │  │ RPC Server     │  │
                                          │  │  (asyncio)     │  │
                                          │  └───────┬────────┘  │
                                          │          │           │
                                          │  ┌───────▼────────┐  │
                                          │  │ Services       │  │
                                          │  │  index, search │  │
                                          │  │  list, show    │  │
                                          │  │  stats, embed  │  │
                                          │  └───────┬────────┘  │
                                          │          │           │
                                          │  ┌───────▼────────┐  │
                                          │  │ DuckDB + MLX   │  │
                                          │  │ (daemon-owned) │  │
                                          │  └────────────────┘  │
                                          │                      │
                                          │  ┌────────────────┐  │
                                          │  │ File Watcher   │  │
                                          │  │ (watchdog)     │  │
                                          │  └────────────────┘  │
                                          └──────────────────────┘
```

**RPC Request lifecycle:**
1. CLI resolves output format, validates flags, confirms destructive ops locally
2. CLI checks for `recall.sock` — if missing, auto-forks daemon and waits for socket
3. CLI connects to Unix socket, sends JSON-RPC request
4. Daemon dispatches to service layer, streams progress notifications if applicable
5. Daemon sends final JSON-RPC response (result or error)
6. CLI renders response to stdout (JSON or text) and exits

**Auto-fork sequence:**
1. CLI detects no socket at `{data_dir}/recall.sock`
2. CLI forks `recall daemon --background` (detached, stdio redirected to log files)
3. CLI polls for socket existence with backoff (max ~5 seconds)
4. If socket appears, CLI proceeds with RPC
5. If timeout, CLI exits with error suggesting `recall daemon start`

**Daemon lifecycle:**

- `REQ-RECALL-0145-J1`: `recall daemon stop` is durable by default. On macOS it runs `launchctl bootout gui/<uid>/<label>`, where `<label>` is `it.send.recall.daemon`, or the legacy `xyz.metalrodeo.recall.daemon` while an install made under it has not been migrated (`REQ-DAEMON-076`); on Linux it runs `systemctl --user stop` against the installed recall daemon unit(s). The command then polls boundedly (default 10 seconds) until the scheduler is unloaded/inactive and the PID file no longer points at a live daemon. If the daemon is still alive or the scheduler is still loaded after the timeout, the command reports `stopped=false` with a clear message.
- `REQ-RECALL-0145-J2`: `recall daemon start` starts the installed durable scheduler. On macOS it runs `launchctl bootstrap gui/<uid> <plist>` for the plist of the label `REQ-RECALL-0145-J1` resolves; on Linux it runs `systemctl --user start <unit>`. The command polls boundedly (default 10 seconds) until the daemon is responsive or a live PID file is present. If the daemon is already running, the command succeeds and reports the existing PID.
- `REQ-RECALL-0145-J3`: `recall daemon stop --soft` preserves the legacy PID-file stop path. It signals the daemon directly and does not unload launchd/systemd, so a keep-alive scheduler may immediately respawn it.
- `REQ-RECALL-0145-J4`: `recall daemon restart` is durable stop followed by durable start. If durable stop fails, restart reports the stop failure and does not attempt start.
- `REQ-RECALL-0145-X2`: Lifecycle command structured output includes `scheduler`, a success boolean (`stopped`, `started`, or both for restart), `pid`, `duration_seconds`, and `message`.

#### Invariants

- The daemon is the single process that opens DuckDB connections (read or write). No CLI process opens any database connection. Read handlers use per-call cursors within their executor threads and close those cursors before returning; the daemon retains ownership of the shared write connection.
- All CLI commands produce identical output whether the daemon was already running or was auto-started.
- JSON-RPC request parameters are a strict superset of CLI `--params` payloads — the same validation applies.
- The socket file exists if and only if the daemon is running and accepting connections.
- Progress notifications are only sent for operations that emit progress to stderr in the current CLI (i.e., `recall.index` with progress enabled).
- The PID file is advisory for RPC readiness, but lifecycle commands also use it to wait for bounded process exit/startup when operating outside RPC.

#### Non-goals

- HTTP/SSE transport (future work for remote MCP).
- Authentication or authorization on the Unix socket (file permissions suffice for single-user).
- Multiplexed batch requests (JSON-RPC batch is not required in the first implementation).
- Hot-reloading embedding models or configuration without restart.
- Daemon clustering or multi-writer multi-host coordination. Multi-host **ingest**
  (`recall index --root` + rsync) is REQ-MULTIHOST-*. Read-only **fleet query
  fan-out** over SSH to existing per-host daemons is REQ-FLEET-* — it does not
  cluster writers or share one DuckDB across hosts.
- Backward compatibility with the direct-DB CLI access pattern — this is a hard cutover.

#### Risk Tags

- **[RISK-HIGH] Public API change**: CLI no longer works without a daemon process (mitigated by auto-fork).
- **[RISK-HIGH] Architecture change**: All service calls move behind RPC boundary — serialization bugs, latency regressions.
- **[RISK-MEDIUM] Process lifecycle**: Auto-fork, PID file management, orphan daemon cleanup.

#### Acceptance Criteria

- [ ] `recall search "query"` works identically whether the daemon was pre-started or auto-forked.
- [ ] `recall index --full --yes` streams progress notifications to stderr via JSON-RPC and returns the same summary as before.
- [ ] `recall search "query" --json` with daemon running returns results with no lock contention errors.
- [ ] `recall daemon stop` unloads the launchd/systemd scheduler by default and waits for the daemon to exit.
- [ ] `recall daemon stop --soft` sends SIGTERM and the daemon exits cleanly, removing the socket and PID files.
- [ ] `recall daemon start` loads/starts the installed launchd/systemd scheduler and waits for daemon readiness.
- [ ] `recall daemon restart` performs durable stop followed by durable start and short-circuits if stop fails.
- [ ] `recall daemon status` works without an active RPC connection (reads PID file and scheduler artifacts directly).
- [ ] Two concurrent `recall search` commands return correct results without errors.
- [ ] A concurrent `recall search` during `recall index` does not block or fail.
- [ ] `recall index --recreate` without `--yes` fails at the CLI layer before any RPC is sent.
- [ ] No CLI module imports `recall.services.*`, `recall.db.*`, or `duckdb` (enforced by import lint or test).
- [ ] The daemon exits after 30 minutes idle in on-demand/poll mode.
- [ ] The daemon stays alive indefinitely in watch mode.
- [ ] Auto-forked daemon logs to `{data_dir}/logs/daemon.log`.
- [ ] `recall --llms` manifest reflects the RPC methods and their parameter schemas.

### Project Tracking

**Both git root and cwd tracked:**
- `session_state.cwd` - working directory from session
- `session_state.git_repo` - detected git root (if in repo)
- `session_state.git_branch` - branch at session start (if available)

### Schema Management

- `REQ-SCHEMA-001`: `schema.sql` is the single source of truth for table structure. No incremental migrations exist.
- `REQ-SCHEMA-002`: On schema version mismatch, the system must error with instructions to run `recall index --recreate --yes`. No automatic migration is attempted.
- `REQ-SCHEMA-003`: No foreign key constraints. Referential integrity is guaranteed by construction (deterministic SHA256 IDs, atomic session writes). JOINs use natural key relationships without enforcement.
- `REQ-SCHEMA-004`: Embedding vectors are stored in dedicated tables (`message_embeddings`, `tool_call_embeddings`) separate from content tables (`message_state`, `tool_calls`).
- `REQ-SCHEMA-005`: Embedding tables can be dropped and recreated independently via `recreate_embedding_tables()` without rebuilding content. Dimension changes use this path.
- `REQ-SCHEMA-006`: `--recreate` backs up the existing database file before rebuilding.

### Migration Policy

The original policy (REQ-SCHEMA-001/002) required `--recreate` for any schema version mismatch because `schema.sql` was the sole source of truth and no incremental path existed. The addition of new agent sources (Grok) exposed that tight `CHECK` constraints on open enums (`sessions.source`, `message_state.role`) forced costly rebuilds on large histories just to add support for one more agent. DuckDB 1.x does not support `ALTER TABLE ... DROP CONSTRAINT` for CHECKs, making table-recreate the only reliable relaxation mechanism.

A limited automatic migration framework was introduced for cheap, safe, idempotent upgrades on small tables. `schema.sql` remains the source of truth for base structure; migrations handle deltas (constraint relaxation, future additive small-table changes) and are recorded in a new `schema_migrations` table. `schema_version` (MAX) continues as the logical gate. Migrations run automatically inside `ensure_schema` / `ensure_schema_lenient` (and thus on every `index`, `list`, `search`, `show`, `stats`, `daemon` start, `compact`, etc.). Very old versions (<13) still require `--recreate` because structural changes between schema.sql revisions may not be replayable.

- `REQ-MIG-001`: Migrations are automatic (no separate `recall db migrate` command), idempotent, and safe on production-sized databases. Only `sessions` and `message_state` (small even on large histories) may use the "CREATE new table + INSERT SELECT + DROP + RENAME" pattern to relax CHECK constraints. Other tables must use additive or metadata-only changes, except that lossless in-place widening of token-counter columns to `BIGINT` is permitted on `session_state`, `usage_events`, and `runtime_state` when dependent indexes are captured and dropped in a compensatable committed phase, the type changes, index recreation, and version advancement occur in one transaction, and any failure restores the original indexes and leaves the schema version unchanged. A migration MAY additionally mutate row *data* (as opposed to schema shape) only when it records a pre-image first per REQ-MIG-008 and performs the record and the mutation in one transaction.
- `REQ-MIG-002`: The framework is implemented in `packages/recall/src/recall/db/migrations/` (versioned `00NN_name.py` modules discovered via pkgutil + registry) with runner logic in `db/schema.py`. Python migrations only (for 0.12.0); each exposes a `Migration` with `id`, `target_version`, and `upgrade(conn)` callable. Framework ensures `schema_migrations` table, skips applied, runs pending where current_version < target, records after success, logs via "recall.schema".
- `REQ-MIG-003`: `schema_migrations (migration_id TEXT PRIMARY KEY, applied_at TIMESTAMP)` is created in `schema.sql` (fresh DBs) and via `CREATE TABLE IF NOT EXISTS` during the first migration run (v13 DBs). The 0014 migration (and future ones) use `INSERT OR IGNORE` for idempotency.
- `REQ-MIG-004`: `ensure_schema` (strict) and `ensure_schema_lenient` (daemon) both invoke the pending-migration runner after fetching current version and before the final mismatch check. On success the version may be bumped (selectively inside the migration for conditional cases like v13-only relax). Failures are non-fatal (log + continue) to protect daemon startup.
- `REQ-MIG-005`: Supported automatic upgrade span is documented: versions 13 through the current `SCHEMA_VERSION` are auto-applied through the ordered migration chain (`db/migrations/0014_*` onward), including 13→14 open-enum CHECK relaxation for Grok, 20→21 lossless token-counter widening, and 21→22 session-host attribution backfill. Versions <13 still raise "schema version mismatch" + `--recreate` suggestion. `--recreate` remains the supported escape hatch for corrupted, very old, or unsupported schema states and always performs a full backup first.
- `REQ-MIG-006`: Future agent support ("add NewAgentX") requires: (1) parser + `Source` enum value + registry entry, (2) optional migration (only if new CHECK or small-table change needed), (3) bump `SCHEMA_VERSION` if a new target is introduced. No user ever runs `--recreate` for this class of additive/relaxing change.
- `REQ-MIG-007`: All test paths (`:memory:`, simulated v13/v11 CHECK tables, fresh, compaction fresh-attach, RPC recreate) exercise the framework. `tests/test_core/test_schema_cutover.py` and `tests/test_db/test_migrations.py` cover migration behavior and framework invariants, including the v20→21 token-counter migration.
- `REQ-MIG-008`: A migration that mutates row **data** MUST record the prior value of every row it changes into `schema_migration_undo (migration_id, table_name, row_key, column_name, old_value, recorded_at)` before mutating, in the same transaction as the mutation. The pre-image makes the change reversible by a single `UPDATE ... FROM` without copying the database aside; the reverse statement is documented in the migration's module docstring. Rationale: migrations are already transaction-wrapped, so a *crashed* migration rolls back on its own — the failure mode a pre-image protects against is the migration that **succeeds and writes semantically wrong data**, which is exactly how 0019 backfilled every pre-existing row to the unattributed-host sentinel. A whole-database copy is deliberately **not** the mechanism: recall databases reach 100+ GiB, so a full copy would need that much free space and minutes under the advisory lock to protect a single-column update, and any "copy only if space allows" rule would withhold protection precisely from the largest and most at-risk databases. Whole-database backup stays reserved for migrations that rewrite tables wholesale; none exist today. Automated *application* of the undo data is out of scope — downgrades remain a non-goal (the recorded pre-image plus the documented statement is the recovery path).
- `REQ-MIG-009`: `schema_migration_undo` is additive audit/recovery state. It is created in `schema.sql` for fresh DBs and via `CREATE TABLE IF NOT EXISTS` by the first data-mutating migration, is part of the schema shape at its introducing version regardless of whether that host recorded any rows, and is never read by query paths. Its size is bounded by the rows each migration actually touches.
- `REQ-MIG-010`: Schema version 30 removes `idx_source_files_pending` and `source_files.inventory_generation` in one separately reviewable migration. Fresh databases omit both objects. Migration from version 29 preserves every logical row and reconciliation invariant, advances the version only after successful DDL, and is idempotent when replayed. The migration changes schema shape only; it never authorizes deletion of `source_files` rows or companion state.

**Invariants (in addition to REQ-SCHEMA-003..006):**
- `schema_version` table and `MAX(version)` gate remain; `schema_migrations` is additive audit trail only, as is `schema_migration_undo` (REQ-MIG-009).
- Migrations never touch FTS shadow schemas, embedding tables, or large content tables (`messages`, `tool_calls`); the explicitly permitted token-counter widening may touch only the named columns in `session_state`, `usage_events`, and `runtime_state`.
- Every migration is wrapped such that partial failure leaves the original DB consistent (transaction + ROLLBACK + non-fatal logging); migration 0021 uses one transaction for all three tables after its required compensatable index-drop phase, so a failed type change cannot be recorded as complete and captured indexes are restored.
- `advisory_lock` + daemon connection release discipline (from compaction) continues to protect long-running writers.

**Non-goals:**
- Downgrades or bidirectional migrations.
- Online / zero-downtime migrations while daemon serves traffic.
- SQL-only migration files in 0.12.0 (Python required for introspection + conditional DDL).
- New `[migrations]` config section or CLI surface (`recall db` subcommand); automatic trigger via existing `ensure_*` is sufficient and user-friendly.
- Full replay of all historical schema.sql revisions for very old DBs (recreate remains the policy).

**Risk tags:**
- **[RISK-HIGH] Data mutation on user DBs**: table recreate + DDL under advisory lock; mitigated by backup in `--recreate`, tx/rollback in migs, non-fatal error handling, and exhaustive :memory: + simulated v13 tests.
- **[RISK-HIGH] Token-counter DDL on existing DBs**: migration 0021 drops and recreates dependent indexes around lossless type widening; mitigated by compensatable index handling, one transaction for schema advancement, metadata-derived index recreation, value/type/index assertions, and copied-live-DB verification.
- **[RISK-HIGH] Catalog column removal**: migration 0030 removes an index and column from a production-sized table. Clone rehearsal and specialist review must cover data loss, migration idempotency, compatibility, rollback, and insufficient-disk failure before live application.
- **[RISK-MEDIUM] Daemon stability**: `ensure_schema_lenient` must tolerate migration errors (warnings only) so the RPC socket can still accept `--recreate` commands.
- **[RISK-LOW] Future contributor burden**: migration authoring must be simple (drop `00NN_*.py`, implement `upgrade(conn)` with lazy helpers, keep idempotent); documented in this section.

**Test traceability (added post-TDD):**
- `tests/test_core/test_schema_cutover.py:183` (v13 relax), new framework tests for fresh/v13/v11/idempotency paths, `schema_migrations` table presence, and Grok source acceptance post-mig.

**Migration authoring guide (for future agents):**
1. Add parser + `Source` value + registry (if new agent).
2. If CHECK relaxation or small-table change needed: create `db/migrations/00NN_descriptive.py`.
3. In the module: `from dataclasses import dataclass; ... register(Migration(id="00NN_...", target_version=NN, upgrade=upgrade))`.
4. `def upgrade(conn):` must be idempotent, use `CREATE TABLE IF NOT EXISTS schema_migrations` (or rely on runner), lazy-import `get_schema_version`/`set_schema_version` from `..schema`, perform minimal work, log, and call `set_schema_version` only when the DB structure actually advanced.
5. Bump `SCHEMA_VERSION` in `schema.py`.
6. Add/update tests in `tests/test_core/test_schema_cutover.py` and `tests/test_db/test_migrations.py` (simulate old CHECK table, assert ensure succeeds, data preserved, new values accepted, version bumped, `schema_migrations` row present; for 0021 also assert BIGINT types, values beyond INT32, idempotency, rollback, and index preservation).
7. Update this SPEC (append REQ-MIG-NNN); release notes come from the Conventional Commit message.
8. Verify with isolated `RECALL_DB_PATH` fresh + v13 sim; never drop real user DB.

### Database Maintenance

DuckDB stores updated rows by writing new row groups while preserving prior versions for MVCC, and `CHECKPOINT` does not aggressively repack historical row groups. In a long-running daemon workload that updates `session_state` (and equivalent rows) on every incremental ingest, the file accumulates orphaned blocks — observed in production at ~270× the live working set after several months. Reclamation requires a full rewrite, not a `CHECKPOINT`.

#### Requirements

- `REQ-COMPACT-001`: `recall compact` must rebuild the database into a fresh sibling file containing only live rows, then replace the original with one same-directory atomic `rename`/`os.replace` operation. The live database path must never be absent during replacement. No row data may be lost.
- `REQ-COMPACT-002`: `recall compact` must verify per-table row counts match between the source and the rebuilt file before performing replacement. Any mismatch must abort the operation with the original file untouched.
- `REQ-COMPACT-003`: `recall compact` must rebuild FTS indexes on the new file using the same configuration (`stemmer`, `stopwords`, fields) as initial schema bootstrap. FTS shadow tables are not copied from the source.
- `REQ-COMPACT-004`: `recall compact` must coordinate with any running daemon before opening or replacing the database. It must use the socket/PID contract from `REQ-RPC-014`, `REQ-RPC-015`, and the RPC daemon invariants to detect and stop scheduler-managed daemons and auto-forked daemons, wait for the socket to disappear and database handles to close, perform the rebuild, and restart the daemon afterward using the original lifecycle unless `--no-restart` is set. If no daemon is running, no daemon coordination is performed.
- `REQ-COMPACT-004a`: For the purposes of `REQ-COMPACT-004`, "running daemon" means a process whose PID appears in `<data_dir>/recall.pid` AND is alive (verified via `os.kill(pid, 0)` returning success). The presence of `<data_dir>/recall.sock` is a strong corroborating signal but is NOT required — a daemon mid-startup or mid-shutdown may have a live PID with no socket, and compaction must still stop that process before opening the database. PID-reuse risk is bounded because the recall daemon rewrites its own pidfile on start (`REQ-RPC-014`); a stale pidfile pointing at an unrelated process is detectable when subsequent stop/restart calls fail to reach a recall daemon. When in doubt, compaction prefers the safety of stopping a possibly-recall PID over leaving an active writer in the lifecycle window.
- `REQ-COMPACT-005`: `recall compact` must report a bloat ratio (`file_size / live_data_bytes`) before and after the operation, and must support a `--dry-run` mode that reports the current ratio without rebuilding.
- `REQ-COMPACT-006`: `recall compact` must use a temporary `.compact` sibling file during the rebuild and a `.pre-compact` backup of the original before replacement. On any failure before replacement, the original file remains the live database. If atomic replacement fails, the original path remains live and the `.pre-compact` backup is retained for diagnosis.
- `REQ-COMPACT-007`: A `recall.services.compaction.estimate_bloat_ratio(db_path, config)` helper must compute live-data bytes as the compacted footprint the database would collapse to, discounting dead row-versions. For each `main` schema table (excluding FTS shadow schemas, which rebuild fresh) it must scale the bytes of the distinct block IDs referenced by that table's active segments by the table's live/physical row ratio: live rows are `count(*)` (which excludes dead versions); physical row-versions are the minimum across columns of each column's summed data-segment row counts (`VALIDITY` segments excluded so a pristine table does not read as 2x; the minimum discards fixed-size ARRAY columns whose per-element counts inflate the row count). Dead row-versions left by `INSERT OR REPLACE` / `UPDATE` churn stay *referenced* until their entire row group dies, so counting referenced blocks alone reported a ~95%-dead database as ~1.0x (no bloat, so auto-compaction never fired); the live/physical scaling makes the ratio track the true compacted size. A dense table (live == physical) scales by 1.0 and still reads ~1.0x. The helper must apply the resolved DuckDB `memory_limit` to its connection, honoring the `[duckdb] memory_limit` config knob and `RECALL_DUCKDB_MEMORY_LIMIT` env var.
- `REQ-COMPACT-008`: The compaction service must use DuckDB's `COPY FROM DATABASE old TO new` primitive to copy table data, not a per-table `INSERT INTO ... SELECT * FROM old.<t>` loop. This is the [DuckDB-recommended path](https://duckdb.org/docs/current/operations_manual/footprint_of_duckdb/reclaiming_space.html) for whole-database compaction. Per-table row-count verification (`REQ-COMPACT-002`) must run after the COPY but before atomic replacement. FTS shadow schemas (`fts_main_*`) and singleton seed rows must be excluded from the COPY (singletons are restored from `old` via DELETE-then-INSERT after the COPY) since `COPY FROM DATABASE` would attempt to copy the seed rows from the schema-bootstrap step into the destination, conflicting with the COPY of `old.runtime_state` / `old.schema_version`.
- `REQ-COMPACT-009`: The recall daemon must support automatic compaction triggered by a bloat-ratio threshold. On a configurable interval (default every 6 hours of daemon uptime, minimum 1 hour), the daemon must call `estimate_bloat_ratio()` with its resolved config, and if the ratio exceeds the configured threshold, run `compact()` against itself: release its DuckDB connection, run the compaction (which acquires the advisory lock and creates the compact sentinel), then reopen its connection. The sentinel and lock guarantee no other writer / auto-fork can race the rebuild during the window.
- `REQ-COMPACT-010`: A new `[compaction]` config section in `~/.config/recall/config.toml` (and matching env vars `RECALL_COMPACTION_AUTO`, `RECALL_COMPACTION_THRESHOLD`, `RECALL_COMPACTION_INTERVAL_HOURS`) must control the auto-trigger: `auto_trigger` (bool, default `true`), `bloat_ratio_threshold` (float, default `2.0`), `check_interval_hours` (int, default `6`, minimum `1`). When `auto_trigger=false`, the daemon never invokes compaction; users must run `recall compact` manually.
- `REQ-RECALL-0145-H2`: `recall compact` must invoke snapshots GC with the default 7-day threshold after successful database replacement and before reporting completion. It must be silent when nothing is removed and log a concise summary when entries are removed.
- `REQ-RECALL-0145-H3`: Snapshots GC must refuse to touch any path outside `<data_dir>/snapshots/`. A missing snapshots directory is a no-op, not an error.

#### Non-goals

- Online compaction (rebuild while daemon serves traffic).
- Per-table compaction. The rebuild is whole-database.

#### Risk Tags

- **[RISK-HIGH] Database rewrite**: `recall compact` rewrites and replaces the live DuckDB file, so row-count verification and backup retention are required before any replacement.
- **[RISK-HIGH] Process lifecycle**: Compaction stops and restarts daemon processes, including auto-forked daemons and launchd/systemd-managed daemons, before touching the database file.
- **[RISK-MEDIUM] Operational availability**: CLI requests cannot be served while compaction owns the daemon lifecycle and database replacement window.
- **[RISK-MEDIUM] Auto-trigger surprise**: Auto-trigger (`REQ-COMPACT-009`) means the daemon may take itself down briefly (typically 20-30 seconds per compact) without explicit user invocation. With the default 6-hour check interval and 2.0 ratio threshold, active workstations should see auto-compact fire roughly once per workday; mostly-idle hosts will see it weekly or less. Users who want zero unannounced downtime can set `auto_trigger=false` in `[compaction]` config and run `recall compact --yes` manually.

#### CLI Surface

- `recall compact` is a top-level command (not nested under `daemon`) because it stops the daemon as a precondition.
- Options:
  - `--dry-run`: report current `file_size`, estimated `live_bytes`, and ratio; exit without modifying anything.
  - `--no-restart`: skip restarting the daemon after a successful rebuild. Use when the caller will manage the daemon lifecycle (e.g. shutdown for backup).
  - `--threshold <float>`: skip the rebuild when the bloat ratio is below this value. Default `1.0` (always compact). Useful when wired into a scheduled job: `recall compact --threshold 5`.
  - `--json` / `--format`: standard recall output controls per `REQ-CLI-002`/`REQ-CLI-003`.
- Human output reports (in order): before-size, before-ratio, per-table copy progress on stderr, after-size, after-ratio, elapsed seconds, daemon-restart status.
- JSON payload fields: `before_bytes`, `before_live_bytes`, `before_ratio`, `after_bytes`, `after_live_bytes`, `after_ratio`, `elapsed_seconds`, `tables_copied: {<name>: <rows>}`, `daemon_was_running: bool`, `daemon_restarted: bool`, `skipped_reason: string | null`.
- Exit codes: `0` success or no-op below threshold, `1` row-count mismatch (rebuild aborted, original intact), `2` daemon coordination failed (could not stop or restart), `3` atomic replacement failed (original remains live and `.pre-compact` is retained).

#### Algorithm

1. Resolve daemon lifecycle using the existing socket/PID contract: if `{data_dir}/recall.sock` exists or the PID file identifies a live daemon, stop that process with the same SIGTERM path as `recall daemon stop` and wait for socket removal. If a launchd/systemd user unit is registered and active, stop the unit first to prevent scheduler restart during compaction. Record whether the daemon was auto-forked or scheduler-managed for restart.
2. Open a fresh DuckDB file at `<db>.compact`. Apply `ensure_schema` with the configured embedding dimensions.
3. `ATTACH '<db>' AS old (READ_ONLY)`.
4. Truncate the seeded singleton rows (`runtime_state`, `schema_version`) so the subsequent COPY does not collide with them.
5. `COPY FROM DATABASE old TO <new_alias>` (per `REQ-COMPACT-008`), copying every table except FTS shadow schemas. Then per-table verify `COUNT(*)` matches between source and dest. Abort on mismatch.
6. `DETACH old`. Rebuild FTS indexes via `create_fts_indexes(conn, config.fts)`.
7. `CHECKPOINT` and close.
8. Create `<db>.pre-compact` as a backup of the original while `<db>` remains in place.
9. Atomic replacement: `os.replace("<db>.compact", "<db>")` (or equivalent same-directory rename-over-existing). If replacement fails, `<db>` still points at the original and `<db>.pre-compact` is retained.
10. Run snapshots GC with the default 7-day threshold.
11. Remove `<db>.pre-compact` after successful replacement.
12. If daemon was running and `--no-restart` not set: restart it through its original lifecycle (scheduler unit for launchd/systemd-managed daemons, `recall daemon --background` for auto-forked daemons).

#### Acceptance Criteria

- [ ] `recall compact --dry-run` reports the bloat ratio and exits without modifying the database.
- [ ] `recall compact` against a database with `>2×` bloat reduces file size and preserves all per-table row counts.
- [ ] `recall compact --threshold 100` against a fresh database reports `skipped_reason="below_threshold"` and does not write a `.compact` file.
- [ ] Failure to copy any table aborts the operation with the original DB intact, the `.compact` temp file removed, and exit code `1`.
- [ ] FTS-backed `recall search` returns equivalent results before and after compaction on a synthetic bloated fixture.
- [ ] If a scheduler-managed or auto-forked daemon is running before compact, it is stopped, the rebuild succeeds, and the daemon is running again afterward through the same lifecycle (unless `--no-restart`).
- [ ] Successful `recall compact` removes stale entries from `<data_dir>/snapshots/` using the default threshold.

### Daemon Resilience

The watch-mode daemon holds a single long-lived `duckdb.DuckDBPyConnection` (`RpcServer._conn`) for write paths (FTS rebuild, embed-phase writes, indexer commits) and `.cursor()` duplicates for read paths (search, list, show, stats). When a query on the shared write connection raises mid-transaction, DuckDB marks the transaction aborted; every subsequent statement on that connection returns `TransactionContext Error: Current transaction is aborted (please ROLLBACK)` until either `ROLLBACK` is issued or the connection is closed. A May 2026 incident exposed this: an OOM in `create_fts_indexes` left the connection wedged, after which `recall search` returned the abort error indefinitely and the embed-phase loop spammed the same error on every cycle. The connection was only recovered by `recall daemon restart`.

The same incident exposed two adjacent fragilities: the FTS rebuild path can exhaust the configured DuckDB `memory_limit` (16 GiB ceiling on a 128 GiB host) and crash the connection, and the daemon's `daemon.err.log` had grown to 450 MiB — ~99% of which was `Fetching 11 files: 100%|…` progress bars from `huggingface_hub.snapshot_download` printed on every embed cycle — which both buried the actual failure and would have eventually filled the disk.

This section covers three resilience requirements derived from that incident: (1) the daemon must recover from a wedged shared connection automatically rather than requiring a manual restart, (2) the FTS rebuild path must not be able to OOM the connection in the common case, and (3) daemon log output must not flood from third-party progress bars and must be bounded in size.

An August 2026 disk-full incident exposed the next gap. A DuckDB checkpoint failed on `fsync` with `No space left on device`; the file stayed consistent at the table level but not at the index level (rows present in `session_state` were absent from its ART indexes). Every later `UPDATE session_state … COMMIT` on such a row died with `Failed to delete all rows from index. Only deleted 0 out of 1 rows`, `is_fatal_db_invalidation` fired, the daemon exited cleanly, and launchd `KeepAlive` restarted it — 1,679 identical runs over ~8 hours, `last_failure_message: null` throughout because the invalidated instance rejects every statement, including the one that would have recorded the failure. The repair that worked was a one-second `DROP INDEX` / `CREATE INDEX` of every entry in `duckdb_indexes()` plus `CHECKPOINT`. A second host lost one index outright (`idx_sessions_source` absent from the live file) after its own ENOSPC on a WAL write. `REQ-RESIL-014` through `REQ-RESIL-020` make the daemon remember the failure, attempt that repair itself, refuse to hand the scheduler an identical failure twice, and make the condition visible from `recall daemon status` instead of a 300k-line log.

The same sweep found a second, quieter class: `tuple index out of range` from the usage harvest on one host and 63 times from the embed phase on another, interleaved with `No open result set` and `Connection already closed!` around a restart. All three are the signature of two threads on one `DuckDBPyConnection`: `execute()` stores the result on the connection object, so a `fetchone()` after another thread's `execute()` returns that thread's row (shorter, hence the index error) or nothing (`No open result set`), and a `close()` from a third thread yields the last message. The write lock was supposed to make the shared connection single-user, but a cancelled `await` released it while the executor thread was still mid-statement, and `stop()` closed the handle without taking it at all. `REQ-RESIL-021` through `REQ-RESIL-023` close that gap and make any residual occurrence attributable.

#### Requirements

- `REQ-RESIL-001`: Any code path in `RpcServer` that calls a DuckDB statement on the shared write connection (`self._conn` returned by `_get_conn()`/`_open_conn_unlocked()`) must, on exception, run a recovery step before the connection is reused for the next operation. The recovery step is: (a) attempt `conn.execute("ROLLBACK")` swallowing only `duckdb.TransactionException` arising from "no transaction is active"; if that itself raises any other exception, (b) close the connection (`conn.close()`) and clear `self._conn` so the next `_get_conn()` opens a fresh handle. Recovery must run for failures in `_init_fts`, the watch-mode drain loop's FTS rebuild blocks (`_do_fts_idle`, `_do_fts`), the embed-phase loop (`_embed_loop`), the catch-up scan (`_run_catch_up`), and the indexer write paths invoked by `_handle_index`/`daemon_run`. The recovery must hold `self._conn_lifecycle_gate` write-side when it closes the connection so concurrent readers cannot observe a half-closed handle.
- `REQ-RESIL-002`: The FTS-missing retry in `_handle_search` (`rpc_server.py:709-717`) must treat `TransactionContext Error: Current transaction is aborted` raised by the shared connection as recoverable: it must trigger the same connection-recovery step as `REQ-RESIL-001` before retrying `_init_fts`, so a single past OOM cannot wedge search across all subsequent requests.
- `REQ-RESIL-003`: The embed-phase loop's existing top-level `except Exception` (`rpc_server.py:1374-1376`) must call the connection-recovery step from `REQ-RESIL-001` before the next sleep/retry cycle, so a transient OOM in one cycle does not turn into a permanent embed stall.
- `REQ-RESIL-004`: The connection-recovery step must be unit-testable in isolation: it lives in `RpcServer` as a private method (recommended name `_recover_shared_conn(err)`) callable from any executor thread, takes the originating exception for logging, and emits one `logger.warning` line that includes the originating error class and message plus whether recovery rolled back the txn or reopened the connection.
- `REQ-RESIL-005`: `create_fts_indexes` in `db/queries.py` must, on entry, save the current values of `threads` and `preserve_insertion_order`, then apply DuckDB session settings that bound memory and thread pressure for the rebuild: `SET preserve_insertion_order=false`, and `SET threads = max(2, host_threads/2)` (clamped to `[2, 8]`). After the rebuild completes (successfully or via raised exception), the function must restore the saved values in a `finally` block. Per cycle-13 challenger review, DuckDB's `SET threads` and `SET preserve_insertion_order` are database-wide (not connection-scoped), so leaving them changed would affect the embed phase and concurrent reads on the same database — the save/restore is mandatory, not optional. The daemon's startup `connect()` already sets `memory_limit`, which is sufficient as the upper bound. The settings must be applied before the `UPDATE message_state SET fts_content/fts_thinking = …` and before the `PRAGMA create_fts_index` calls. The function must log at INFO level the values it applied (with the prior values for context) so OOM regressions are diagnosable from the log.
- `REQ-RESIL-006`: `create_fts_indexes` must catch `duckdb.OutOfMemoryException` from the `UPDATE message_state` step or the `PRAGMA create_fts_index` step and re-raise it as `recall.db.queries.FtsRebuildOutOfMemoryError` (new exception type, subclass of `RuntimeError`) wrapping the original; the caller in `RpcServer` is responsible for invoking the `REQ-RESIL-001` recovery step. This requirement does not promise that the rebuild itself completes under arbitrary RAM pressure — it promises only that the failure surfaces with a distinguishable type so recovery is unambiguous and so subsequent rebuilds can be retried after recovery clears the wedged connection.
- `REQ-RESIL-007`: The daemon process must disable `huggingface_hub` and `tqdm` progress bars before the first import of `huggingface_hub`, `transformers`, `sentence_transformers`, `tokenizers`, or any embedding-backend module that may transitively import them. The mechanism is: `_start_foreground_server` in `cli/daemon.py` must set `os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")` and `os.environ.setdefault("TQDM_DISABLE", "1")` before any other recall import that loads embedding code. Programmatic disabling (`huggingface_hub.utils.logging.disable_progress_bars()`) is acceptable as a belt-and-suspenders addition but must not replace the env-var path, since some HF code paths read the env var directly at import time.
- `REQ-RESIL-008`: When the daemon writes to `{data_dir}/logs/daemon.log` and `{data_dir}/logs/daemon.err.log` (`REQ-RPC-016`), it must rotate those files so neither grows unbounded. Rotation is implemented in the daemon process — not as a launchd/systemd-side concern — because the existing launchd plist and systemd unit redirect stdout/stderr with simple file paths that don't support rotation. The daemon must, on every startup, check the size of `daemon.log` and `daemon.err.log` and if either exceeds `daemon.log_max_bytes` (config, default 50 MiB), perform an atomic rotate-and-reopen: (1) `os.replace(path, path + ".1")` — overwrites any prior `.1` — (2) open a fresh file at the original path, (3) flush `sys.stdout` / `sys.stderr` and use `os.dup2(new_fd, 1)` for `daemon.log` and `os.dup2(new_fd, 2)` for `daemon.err.log` so subsequent writes from the running daemon process land in the new file (not the renamed `.1` inode that the inherited launchd/systemd fd still points at). Retention is one rotated generation per stream (current + `.1`). Any failure of the `os.dup2` step must fall back to leaving the inherited fd as-is and emit a `logger.warning` so a partial rotation does not silently swallow subsequent log output. Rotation runs exactly once per `_start_foreground_server` invocation (`INV-RESIL-002`); mid-process rotation is out of scope.
- `REQ-RESIL-009`: A new `[daemon]` config field `log_max_bytes` (int, default `52428800` == 50 MiB) and matching env var `RECALL_DAEMON_LOG_MAX_BYTES` must be added. The field must be loaded via the existing `AppConfig.load()` path in `core/config.py`. Setting `log_max_bytes = 0` disables rotation.
- `REQ-RESIL-025`: The daemon process must configure logging once, at startup, before it can emit a record. `_start_foreground_server` in `cli/daemon.py` installs a stderr handler on the root logger whose format carries a timestamp, level, and logger name (`%(asctime)s %(levelname)s %(name)s: %(message)s`, `datefmt` `%Y-%m-%dT%H:%M:%S%z` — local wall-clock time with an explicit UTC offset), and sets the `recall` logger to the daemon's own level. That level is a new `[daemon]` config field `log_level` (string, default `info`, one of `debug|info|warning|error|critical`) with matching env var `RECALL_DAEMON_LOG_LEVEL`, loaded through `AppConfig.load()` and rejected at load time when it names no known level; `recall daemon -v` overrides it with `debug`. Only `recall` loggers are re-levelled, so third-party loggers keep their default effective level while still emitting through the timestamped handler. No RPC parameter may change the daemon's handlers or level: the `verbose` parameter of `recall.daemon_run` and `recall.index` is accepted and ignored by the server, and the library entry points in `services/indexer.py` bootstrap logging only when the root logger has no handlers (a bare script or test), never inside the daemon. Without this the daemon ran on Python's lastResort handler — bare `WARNING`+ text, no timestamp — until whichever client request first reached `logging.basicConfig` fixed the level for the rest of the process lifetime, so the `REQ-ADAPT-016` skip warning and the `embed check` / `embed skipped` / `embed batch` INFO lines were unobservable on a production host.
- `REQ-RESIL-010`: After `duckdb.OutOfMemoryException`, `FtsRebuildOutOfMemoryError`, or any DuckDB error whose message starts with `Out of Memory Error`, `RpcServer._recover_shared_conn` must close and clear the shared connection instead of reusing it, because DuckDB can retain pinned buffer-manager memory after OOM and reusing the handle propagates OOM failures to subsequent RPC queries.
- `REQ-RESIL-011`: When any RPC handler fails with a fatal DuckDB invalidation (`duckdb.FatalException`, or an error whose message contains `database has been invalidated` or `previous fatal error`), the daemon must send the client its error response and then set the shutdown event so the scheduler restarts the process (mirroring the watch-task loud-fail). In-process recovery is impossible: within one process, `duckdb.connect()` to the same path returns the same cached — still invalidated — instance, so close+reopen cannot heal it; without this rule the daemon serves `INTERNAL_ERROR` for every query until a manual restart (the observed wedged-daemon state). Ordinary handler exceptions must not trigger the shutdown path.

- `REQ-RESIL-012`: The fatal-invalidation rule of `REQ-RESIL-011` applies to *every* surface that touches the shared connection, not only RPC handlers. Each background loop that otherwise logs-and-continues -- usage harvest (watch tick, discovery tick, post-catch-up, index run), catch-up scan, and the embed loop -- must classify its exception first and treat a fatal invalidation as terminal: synchronous callers re-raise, daemon-resident callers set the shutdown event and stop. Rationale: a swallowed fatal spins the loop on a dead instance indefinitely while `daemon status` still reports healthy (observed on a live host: 4,891 identical errors and a 46 MB error log across a single day, with `last_failure_message: null` throughout). The classifier is shared (`recall.db.fatal.is_fatal_db_invalidation`) so `services/watcher.py` need not import `services/rpc_server.py`.
- `REQ-RESIL-013`: `_recover_shared_conn` must not attempt recovery for a fatal invalidation. Because DuckDB caches the database instance per path per process, the reopened handle reattaches to the same dead instance, so the close-and-reopen path reports a recovery that did not occur. It must escalate to shutdown instead.
- `REQ-RESIL-014`: When the daemon classifies an exception as a fatal DuckDB invalidation (`is_fatal_db_invalidation`), it must persist a failure record *outside* DuckDB before it sets the shutdown event — the invalidated instance rejects every statement, which is why `last_failure_message` stayed `null` for 1,679 restarts. The record is written atomically (temp file + `os.replace`) to `<data_dir>/daemon-failure.json` and carries: the site label, the exception class, the first line of the message (bounded to 500 characters), a normalized signature `site:ExceptionClass:message` in which hex runs of 6+ characters, decimal numbers, and double-quoted strings are replaced by placeholders so two runs that die on different session ids compare equal, a failure class (`index-divergence` when the message contains `Failed to delete all rows from index`; `disk-full` when it contains `No space left on device` or is an `OSError` with `errno.ENOSPC`; otherwise `other`), and a UTC timestamp. On the next daemon start, before any DuckDB write, the startup self-repair step folds the record into `runtime_state`: `last_failure_message` / `last_failure_at` (the human-readable message), `last_fatal_signature`, `fatal_repeat_count` (incremented when the signature equals the stored one and no successful run landed after the stored failure; reset to 1 otherwise), and `last_fatal_at` (the record's timestamp, written whenever the signature is set or its count repeats), then consumes the marker. The timestamp is not optional decoration: an undated signature reads as a live failure however old it is — a live host showed `last_fatal_signature` from a fault whose log had rotated hours earlier — so `daemon status` carries `last_fatal_at` in JSON beside `fatal_repeat_count` and the text renderer prints it on the `Last fatal:` line (`at an unrecorded time` for a row written before migration 0029). Clearing the fatal memory clears it with the rest. Marker writes are best-effort: when the disk is full the write itself fails, which is logged and never prevents shutdown; the failure the *next* run records (the index fatal) is the one that drives repair.
- `REQ-RESIL-015`: When the consumed record's class is `index-divergence`, or the startup probe of `REQ-RESIL-018` (which runs on every start, and is how a `disk-full` record or a set `needs_index_verification` flag is checked) reports at least one diverged key, the startup step runs `rebuild_indexes` (`REQ-RESIL-017`) before FTS init, sidecar sync, catch-up, or any other write; records `last_index_repair_at` and `last_index_repair_signature`; logs the counts; and continues serving. The socket is bound before the step, so a request may arrive mid-repair: the step runs on the daemon's shared connection while holding exclusive lifecycle access, and a request that reaches the connection waits for the repair instead of opening a second connection whose schema bootstrap would race the repair's. The bloat estimator's read-only handle is taken before the step, because the same file cannot be opened read-only once the shared connection exists. Any failure inside the step other than the refusal of `REQ-RESIL-016` is logged and does not stop the daemon from listening — except a fatal invalidation (`is_fatal_db_invalidation`), which is not a step failure to listen past: it routes through the fatal funnel of `REQ-RESIL-020` (marker written with site `startup_self_repair`, shutdown event set), the socket and pid file are released, and the exception propagates so the process exits non-zero and the scheduler relaunches it against the spooled record (`INV-RESIL-004`).
- `REQ-RESIL-016`: If the consumed record's signature equals `last_index_repair_signature` and no successful run has completed since `last_index_repair_at`, the repair already failed to hold: the daemon must not repair again or serve. It writes `last_failure_message` naming the condition and the manual fix (`recall daemon stop`, then `recall db rebuild-indexes`; if that does not hold, `recall compact --yes` — never `recall index --recreate`, which is served by the daemon whose auto-forked start refuses again while the memory persists), leaves the marker in place so subsequent starts refuse identically and cheaply, releases the socket and pid file, and exits with status 3 (`DaemonStartupRefused`) instead of exiting 0 into an identical restart. `recall db rebuild-indexes` and a successful `recall compact` clear the marker and the fatal memory (`last_fatal_signature`, `fatal_repeat_count`, `last_index_repair_signature`) so a repaired database starts normally. The repair attempt is *armed* before the rebuild runs: `last_index_repair_at` / `last_index_repair_signature` are written first, so a rebuild that raises counts as a repair that did not hold and the next start on the same signature refuses instead of retrying the rebuild on every start. If the arming write itself fails (`duckdb.Error`), the rebuild is not run: an unrecorded attempt would be retried on every start, so the start refuses (exit 3, refusal marker written) with a message naming the write failure and the manual fix. A `runtime_state` that lacks the memory columns (`last_index_repair_at`, `last_index_repair_signature` — a schema that predates migration 0023, which the lenient open admits so the operator can recreate) is the one exception: it cannot be armed at all and the fix needs the daemon listening, so the start logs a warning naming `recall index --recreate --yes`, runs the rebuild unarmed, and serves; a recurrence rebuilds again on every start until the schema is recreated. A `duckdb.Error` that is a fatal invalidation is never converted into a refusal — neither here nor by the tolerant `runtime_state` writes (`_persist`) — it propagates to the funnel of `REQ-RESIL-015`, because a refusal holds the scheduler down while a relaunch could still repair the database.
- `REQ-RESIL-017`: `recall.db.maintenance.rebuild_indexes(conn)` drops and recreates every index listed by `duckdb_indexes()` in schema `main` (PRIMARY KEY / UNIQUE constraint indexes are not listed there and are never touched), then executes every `CREATE INDEX IF NOT EXISTS` statement in `schema.sql` so an index missing from the live file is healed, then `CHECKPOINT`s. It returns `IndexRebuildResult(dropped, created, healed, elapsed_seconds)` where `healed` names the indexes that existed only in `schema.sql`. It touches no table data, so every table's row count is unchanged. It is exposed as `recall db rebuild-indexes`, which takes the advisory lock and a read-write connection and refuses with a message that names `recall daemon stop` when the daemon holds the database (advisory lock held, or DuckDB `Conflicting lock` / `Could not set lock`); it never stops or restarts the daemon itself.
- `REQ-RESIL-018`: `recall.db.maintenance.probe_index_divergence(conn, sample=2, budget_seconds=10.0)` is read-only. It derives its targets from `duckdb_indexes()` joined with `duckdb_columns()` — every non-constraint `main` index whose columns are all `VARCHAR` — rather than a hard-coded list. For each target it samples the `sample` most recently appended distinct keys (`max(rowid)` descending; the incident's diverged rows were exactly the ones written after the failed checkpoint) and compares `count(*) WHERE col = ?` (index-eligible) with `count(*) WHERE (col || '') = ?` (forced sequential scan). A sample counts as *checked* only when its full-scan count is within DuckDB's index-scan bound, `max(index_scan_max_count, index_scan_percentage × table rows)`; above it the planner scans both arms sequentially and cannot see divergence, so such samples are reported as `unverifiable`, never as healthy. The probe issues no new query once `budget_seconds` has elapsed and marks the report incomplete. It returns `IndexDivergenceReport(checked_at, indexes_probed, samples_checked, samples_unverifiable, diverged, complete, elapsed_seconds)` with `diverged` a tuple of `DivergedKey(table, column, key, index_count, full_count)`. Surfaces: the daemon runs the probe once at startup and caches the report; `recall daemon status` exposes it as `index_divergence` (a `daemon status` manifest field) and the text renderer prints one line; the `recall.check_indexes` RPC runs it live on a read cursor; `recall db check-indexes` calls that RPC when the daemon is up, falls back to a read-only local connection otherwise, and exits 1 when any key diverged. The probe detects the divergence class that was observed; it is not a proof of index integrity — unsampled keys and non-`VARCHAR` indexes are not covered.
- `REQ-RESIL-019`: Any exception on the shared connection whose message contains `No space left on device` (or an `OSError` with `errno.ENOSPC`) — whether fatal (checkpoint fsync) or not (WAL write during usage harvest, observed as `TransactionContext Error: Failed to commit: Could not write file ".../recall.duckdb.wal": No space left on device`) — sets `needs_index_verification` in the marker (best-effort) and, while the instance is still alive, in `runtime_state`. On the next open the startup step runs the probe and, when it reports diverged keys, the rebuild; the flag is then cleared in both places. A `disk-full` fatal record routes through the same path.
- `REQ-RESIL-020`: The fatal funnel (`RpcServer._stop_on_fatal_db`) must set the shutdown event even when persisting the marker or emitting the log record raises — the incident's `daemon.err.log` shows `--- Logging error ---` blocks from a handler failing under ENOSPC. Persistence and logging each run under their own `except Exception`, so neither can preempt the terminal signal. `_process_request`'s fatal branch and `_fail_on_watch_task_exit` route through the same funnel so every fatal leaves a record.
- `REQ-RESIL-021`: The shared write connection is single-user: every executor job that executes a statement on it runs under `_write_lock`, and the awaiting task holds that lock until the job has *finished* — a cancelled `await` waits for the executor thread (`RpcServer._await_shared_conn_work`, the shielded wait the watch drain loop already used) rather than releasing the lock and abandoning the thread. `stop()` acquires `_write_lock` before it shuts the executor down and closes the connection, so no thread can be mid-statement when the handle closes. This covers the index and recreate handlers, `daemon_run`, FTS init, the startup self-repair step (which additionally holds the lifecycle gate so `_get_conn()` callers wait instead of opening a second connection), the embed loop, the watch attempt record, and every watch-mode job. Connection *recovery* is the same kind of work: an event-loop handler (catch-up, discovery, embed loop, search) sees its job's failure only after that job released the lock, so it recovers through `_recover_shared_conn_locked` — lock, executor, awaited to completion — never by calling `_recover_shared_conn` inline on the loop thread. Read paths are exempt because `_ReadExecution` opens, interrupts, and closes a per-task cursor under `_run_readonly`; these are distinct DuckDB connection objects and remain covered by the connection lifecycle gate until close.
- `REQ-RESIL-022`: `run_embed_cycle` accepts `should_stop: Callable[[], bool]` and polls it before each session; when it returns true the cycle stops at that session boundary, writes the embeddings it has already produced, and returns as usual. The RPC embed loop passes `self._shutdown_event.is_set`. With `REQ-RESIL-021` a shutdown must wait for in-flight work, so this bounds that wait to one session (one model call) instead of one 50-session batch — the interpreter joins executor threads at exit regardless, so an unbounded cycle would have stalled the process, not just `stop()`.
- `REQ-RESIL-023`: A swallowed failure on a daemon-resident loop is logged with its traceback (`exc_info=True`): the per-session `embed failed for session` error in `run_embed_cycle`, the `usage harvest failed` warning in `watcher._maybe_harvest_usage` and `RpcServer._harvest_usage_on_shared_conn`, and the index run's `usage harvest failed after index run` in `index_sessions` (worded to name its site, since the three loops otherwise log the same text). `str(err)` alone (`tuple index out of range`) left the failing statement unknowable on two fleet hosts; a loop that keeps going after an error owes the reader the line it failed on.
- `REQ-RESIL-024`: A refused start (`REQ-RESIL-016`) must not be relaunched by the scheduler until the memory is cleared. The refusal writes `<data_dir>/daemon-refused` (best-effort, never raises) before it raises; `clear_failure_marker` removes it together with the failure marker, so `recall db rebuild-indexes`, a successful `recall compact`, and a normal start that consumes the marker all release it. The installed watch-mode units key on it: the systemd service carries `RestartPreventExitStatus=3` alongside `Restart=always` (exit 0 after a fatal is still relaunched); launchd has no exit-status filter, so the plist's `KeepAlive` is `{PathState: {<refusal marker path>: false}}` — the job is kept alive only while the marker is absent, and launchd relaunches on its own when the marker is removed. `RunAtLoad` stays true and `_detect_installed_mode` still reads the dict form as watch mode. The policy takes effect on the next `recall daemon install`; an existing plist keeps the unconditional `KeepAlive` until then. Because the marker can outlive the failure marker it belongs to (the unarmed refusal is raised after the fold consumed the failure marker), every path that ends a refusal releases it: a start that completes the self-repair step (it is serving, so the refusal is over), `recall db rebuild-indexes`, and a successful `recall compact`. `recall index --recreate` is not one: it is served by the daemon, and the start that `recall index` auto-forks refuses again while the memory persists, so the refusal never names it. While it exists, `recall daemon status` exposes its message as `startup_refusal` (a `daemon status` manifest field, read from the file so it is available with the daemon down), the text renderer prints `Startup refused: …`, and structured mode emits a warning notice on stderr — a refused daemon is exactly the state in which nothing else reports.
- `REQ-RESIL-026`: A `runtime_state` failure stamp always carries a reason. `record_run_failure` normalizes a blank or whitespace-only message to `unknown failure during the <run kind> run`, so `last_failure_at` is never set beside an empty `last_failure_message`. The messages themselves must be real: the RPC error types (`RpcError`, `RpcCallError`, `RpcConnectionError`) are frozen dataclasses, which never populate `BaseException.args`, so the inherited `__str__` rendered as `""` — a restart that interrupted an index request on a live host recorded exactly that. Each defines `__str__` as its `message`, so every caller that logs or records `str(err)` reports the reason.
- `REQ-INDEX-020`: A session rewrite triggered by an identity change runs inside a single transaction and is undone by `ROLLBACK`. It must not hand-roll compensation by deleting the session and re-inserting saved rows: that re-insert enumerates `session_state` columns explicitly and so silently drops any column added later (`sidecar_mtime` was being lost), and the compensating delete issues a second delete against an index the failed attempt already disturbed -- a fault DuckDB answers by invalidating the whole instance for the life of the process.
- `REQ-INDEX-021`: A full re-parse (`index --full`) must carry stored `llm-*` context forward onto the messages it re-parses, and summarize only the messages that have none. Message ids are deterministic (`make_message_id(session_id, idx)`) and transcripts are append-only, so a stored summary is still valid for an unchanged message. Without this, `--full` is not a re-parse but a re-summarization: on an `llm-local` host that measured ~0.76 messages/s against ~3.1M messages, roughly 47 days. Worse, the obvious way to avoid that cost destroyed data — parsed messages arrive with empty `context_text`, so resolving them under `off` wrote `''` over every stored summary and reported success. Re-summarization stays reachable through `--recompute-context` (with `--only-mode` / `--since`), which is the surface built for it; re-reading a file and re-running a model are separate acts. Template context is excluded from reuse deliberately: it is derived from session metadata, free to rebuild, and must refresh when that metadata changes. An index run must report `context_reused` alongside `context_messages`, so a run about to spend days summarizing says so.
- `REQ-INDEX-022`: `index` must accept `--since` and `--project` to scope which transcripts a run considers, so a `--full` backfill can be taken in chunks instead of one all-or-nothing pass. A parser change only reaches already-indexed sessions through `--full`, and on a large corpus that is a single 73-157 minute run; a user who cannot bound it will not run it. `--since` compares the transcript's mtime rather than the session's recorded time, because it is the file being considered for re-parse and using it needs no database, so a never-indexed transcript is scoped the same way as an indexed one. `--project` matches `session_state.git_repo ILIKE '%value%'`, identical to `list --project`, so the flag means one thing across the CLI; because it must resolve through the database, a project scope with no connection is an error rather than a silent no-op — ignoring it would re-parse the whole corpus when the caller asked for one repo. The scopes compose, and both apply to incremental runs as well as `--full`.

#### Invariants

- `INV-RESIL-001`: After `_recover_shared_conn` returns, `self._conn` is either a connection on which `conn.execute("SELECT 1")` succeeds, or `None` (so the next `_get_conn()` opens fresh). It is never a connection with an aborted transaction.
- `INV-RESIL-002`: The `daemon.log` / `daemon.err.log` rotation step from `REQ-RESIL-008` runs exactly once per `_start_foreground_server` invocation, before the executor or RPC server starts. It does not run periodically during the daemon's lifetime; restart-triggered rotation is sufficient because the daemon is the only writer and the size bound is generous.
- `INV-RESIL-003`: The settings applied by `create_fts_indexes` (`REQ-RESIL-005`) are restored to their prior values before the function returns. (Rationale: DuckDB `SET threads` and `SET preserve_insertion_order` are database-wide, not per-connection, so they would propagate to read cursors and the embed phase. The implementation must save the pre-call values, apply FTS-friendly values, run the rebuild in a `try`, and restore the prior values in a `finally`. Tests assert that on both successful and exception-raising paths, the post-call settings match the pre-call settings.)
- `INV-RESIL-004`: No `except` block that handles errors from the shared DuckDB connection may swallow a fatal invalidation — including the tolerant `runtime_state` writes of the self-repair step. Equivalently: after any such handler runs, either the exception has propagated or the daemon's shutdown event is set.
- `INV-RESIL-005`: `rebuild_indexes` leaves every table's row count unchanged and leaves the set of `main` indexes equal to the union of `schema.sql`'s `CREATE INDEX` names and the names that were already in the live file. A second run is a no-op in effect: it reports `healed == ()` and the same index set.
- `INV-RESIL-006`: After `_stop_on_fatal_db` returns `True`, the shutdown event is set, regardless of whether the marker write or the log emission succeeded.
- `INV-RESIL-007`: A reader of `<data_dir>/daemon-failure.json` observes either the previous complete record or the new complete record, never a partial write.
- `INV-RESIL-008`: At most one thread executes statements on the shared write connection at any instant, and the connection is never closed while such a statement is in flight. Equivalently: `_write_lock` is held for the whole lifetime of every executor job that uses the connection, including after the awaiting task is cancelled, and `stop()` closes the handle only while holding it.

#### Non-goals

- Health monitoring or automatic daemon process restart (launchd/systemd already restart on crash; `REQ-RESIL-001` covers in-process recovery).
- Eliminating OOM in `create_fts_indexes` for every possible input. Hosts with extreme `message_state` row counts can still hit `memory_limit`; the requirement is that the failure does not wedge the daemon, not that it never fails.
- Hot-swapping log files mid-process. Rotation runs only at startup (`INV-RESIL-002`).
- Configurable retention beyond one rotated generation (`.1`). If users want more, they can wire an external `logrotate` rule; recall's built-in rotation is a floor, not a feature.
- Streaming embed-cycle progress to a structured log target separate from stderr.
- Rebuilding PRIMARY KEY / UNIQUE constraint indexes; DuckDB neither lists them in `duckdb_indexes()` nor allows `DROP INDEX` on them. A diverged constraint index needs `recall compact --yes` or `recall index --recreate --yes`.
- Recovering the sibling corruption a partial checkpoint can leave — reopen aborting the process with `INTERNAL Error: Failed to append to PRIMARY_…: duplicate key` when WAL replay re-appends rows the failed checkpoint already persisted (reproduced on an 8 MiB APFS image). That state needs a backup or `--recreate`; the index rebuild never runs because `connect()` does not return.
- Automatically stopping the daemon from `recall db rebuild-indexes`. The command refuses while the daemon holds the database; `recall compact` remains the command that owns daemon lifecycle.

#### Risk tags

- **[RISK-MEDIUM] Connection lifecycle race**: `_recover_shared_conn` mutates `self._conn` from an executor thread. The existing `_conn_lifecycle_gate` (acquire_read for reads, acquire_write for close/reopen) is the right primitive but is currently only used by `close_db_connection`/`reopen_db_connection`. Misuse (taking the read gate while closing) will deadlock under concurrent search load. The implementation must take the write gate around any path that may set `self._conn = None`.
- **[RISK-LOW] FTS settings persistence**: `REQ-RESIL-005`'s `SET threads`/`SET preserve_insertion_order` persist on the shared connection across subsequent unrelated writes. Embed-phase batch writes and indexer commits will run under the reduced thread count. Measured impact is acceptable on the daemon hot path (writes are small and serialized), but worth flagging.
- **[RISK-LOW] Env-var ordering**: `REQ-RESIL-007` requires setting `HF_HUB_DISABLE_PROGRESS_BARS` before HF imports. The contract is enforced in `_start_foreground_server`; any future code path that constructs an embedding backend before `_start_foreground_server` runs (e.g. inline `recall daemon --once` against an installed daemon) bypasses the env-var path. A wiring-lint test should fail on any new import of `huggingface_hub` or `sentence_transformers` outside of `services/onnx_embeddings.py` / `services/mlx_embeddings.py`.

#### Acceptance criteria

- [ ] A unit test injects a `duckdb.TransactionException` into a fake `_init_fts` call, asserts `_recover_shared_conn` runs, then asserts a subsequent `recall.search` RPC returns search results rather than the abort error (`REQ-RESIL-001`, `REQ-RESIL-002`).
- [ ] A unit test injects an exception into one embed-phase cycle and asserts the next cycle does not raise the abort error and does make forward progress (`REQ-RESIL-003`).
- [ ] A unit test verifies `_recover_shared_conn` emits the warning log line described in `REQ-RESIL-004` and that it covers both the rollback path and the close-and-reopen path.
- [ ] A unit test on `create_fts_indexes` against an in-memory DuckDB connection inspects `conn.execute("SELECT current_setting('threads')").fetchone()` and `… 'preserve_insertion_order'` after the call and asserts the values set by `REQ-RESIL-005`.
- [ ] A unit test patches `conn.execute` to raise `duckdb.OutOfMemoryException` on the `PRAGMA create_fts_index` call and asserts `create_fts_indexes` raises `FtsRebuildOutOfMemoryError` with the original exception chained as `__cause__` (`REQ-RESIL-006`).
- [ ] A test on `_start_foreground_server` asserts that after invocation, `os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"` and `os.environ["TQDM_DISABLE"] == "1"` and that `recall.services.onnx_embeddings` / `recall.services.mlx_embeddings` are not present in `sys.modules` until first embed-backend construction (`REQ-RESIL-007`).
- [ ] A test creates `<data_dir>/logs/daemon.log` and `<data_dir>/logs/daemon.err.log` each at 100 MiB, runs the rotation step from `REQ-RESIL-008` with default `log_max_bytes = 50 MiB`, and asserts both originals are truncated and `daemon.log.1`/`daemon.err.log.1` exist with the prior contents.
- [ ] A test with `log_max_bytes = 0` confirms no rotation occurs even when logs exceed the default threshold (`REQ-RESIL-009`).
- [ ] A test asserts `_start_foreground_server` leaves a root handler whose formatter emits a timestamp and logger name, that the `recall` logger sits at the configured `[daemon] log_level` (`debug` under `-v`), and that servicing a `recall.daemon_run` request with `verbose=True` changes neither the root handlers nor the `recall` logger level (`REQ-RESIL-025`).
- [ ] The unit test `tests/test_services/test_rpc_server.py::TestSharedConnRecovery::test_search_recovers_after_fts_abort` constructs a real `RpcServer` against an in-memory DuckDB, forces `_init_fts` to fail once (by patching `create_fts_indexes` to raise a synthetic `TransactionContext` error), then calls `_handle_search` and asserts results are returned and `_recover_shared_conn` was invoked — no daemon restart between steps. This unit-level acceptance replaces the previously-proposed in-production fault-injection RPC, which was rejected on risk grounds (a daemon must not expose a method that deliberately corrupts its own connection, even gated by env var, because the gate is a runtime check rather than a build-time exclusion).
- [ ] A fatal raised through `_stop_on_fatal_db` writes `<data_dir>/daemon-failure.json` with a normalized signature that is identical for two messages differing only in session id and row counts, and the shutdown event is set even when the marker write and the log handler both raise (`REQ-RESIL-014`, `REQ-RESIL-020`, `INV-RESIL-006`).
- [ ] Startup against a marker of class `index-divergence` rebuilds the indexes, records `last_index_repair_at`, populates `last_failure_message`, and consumes the marker; a second start with the same signature and no success since exits 3 with a `last_failure_message` that names `recall db rebuild-indexes` (`REQ-RESIL-015`, `REQ-RESIL-016`).
- [ ] `rebuild_indexes` on a database missing `idx_sessions_source` recreates every listed index, reports the missing one in `healed`, preserves row counts, and a second run reports `healed == ()` (`REQ-RESIL-017`, `INV-RESIL-005`).
- [ ] `recall db rebuild-indexes` exits non-zero with a message naming `recall daemon stop` while another process holds the advisory lock (`REQ-RESIL-017`).
- [ ] `probe_index_divergence` derives its targets from the catalog (a new VARCHAR index is probed without code changes, a TIMESTAMP index is skipped), reports a key whose index-arm and full-arm counts disagree, and classifies a key above the index-scan bound as unverifiable (`REQ-RESIL-018`). True divergence cannot be manufactured through DuckDB's public API — the RLIMIT_FSIZE and APFS-image reproductions both left the file consistent or unopenable — so the disagreement is injected through a connection wrapper that answers the two query shapes differently.
- [ ] An ENOSPC error on the harvest path sets `needs_index_verification`; the next startup runs the probe and clears the flag (`REQ-RESIL-019`).
- [ ] `recall daemon status --json` carries `index_divergence` and the manifest lists it (`REQ-RESIL-018`).
- [ ] Cancelling the embed task while `run_embed_cycle` blocks in the executor leaves the task pending and `_write_lock` held until the cycle returns; `stop()` called while another task holds `_write_lock` does not close the shared connection until that task releases it (`REQ-RESIL-021`, `INV-RESIL-008`).
- [ ] `run_embed_cycle` with two pending sessions and a `should_stop` that turns true after the first embeds exactly one session, writes it, and leaves the other pending (`REQ-RESIL-022`).
- [ ] An `IndexError` raised inside `load_session` during the embed cycle, inside `harvest_grok_unified_log` during `_maybe_harvest_usage`, and inside the index run's harvest each produce a log record whose `exc_info` names `IndexError` (`REQ-RESIL-023`).
- [ ] A shared-connection failure caught by the catch-up handler, by the discovery handler, and the startup self-repair step each run their shared-connection work with `_write_lock` held and off the event-loop thread (`REQ-RESIL-021`).
- [ ] A rebuild that raises leaves `last_index_repair_signature` set; a second marker with the same signature then refuses without calling the rebuild again, and the refusal leaves `<data_dir>/daemon-refused`; `clear_failure_marker` removes both markers (`REQ-RESIL-016`, `REQ-RESIL-024`).
- [ ] When `record_index_repair` raises, `rebuild_indexes` is not called and the start refuses with exit 3 and the refusal marker; when it raises a fatal invalidation, the fatal propagates, `rebuild_indexes` is not called, and no refusal marker is written; a fatal from `record_fatal_failure` in the fold propagates and leaves the failure marker; when `runtime_state` lacks the memory columns, `rebuild_indexes` runs unarmed, the start serves without a refusal marker, and the warning names `recall index --recreate --yes` (`REQ-RESIL-016`, `INV-RESIL-004`).
- [ ] A start on a clean database with an orphaned refusal marker completes with `action=none` and removes the marker; `daemon_status(config).startup_refusal` carries the marker's message and `recall daemon status` prints `Startup refused:` in text, `startup_refusal` in JSON, and a stderr warning in structured mode (`REQ-RESIL-024`).
- [ ] A fatal invalidation raised inside the startup self-repair step sets the shutdown event, writes a marker with site `startup_self_repair`, releases the socket and pid file, and propagates out of `start()` (`REQ-RESIL-015`).
- [ ] The installed systemd watch service contains `RestartPreventExitStatus=3`; the installed launchd watch plist parses to `KeepAlive == {"PathState": {<refusal marker path>: False}}` and is still detected as watch mode (`REQ-RESIL-024`).
- [ ] Folding a marker sets `last_fatal_at` to the record's timestamp, clearing the fatal memory clears it, a database migrated from a pre-0029 schema reads it as null while keeping the other fatal columns, and `recall daemon status` text dates the `Last fatal:` line (`REQ-RESIL-014`).
- [ ] `record_run_failure` with an empty message stores `unknown failure during the <run kind> run`; an index request that fails with an `RpcError` records that error's message, not `""` (`REQ-RESIL-025`).

## Data Model

### Schema

```sql
-- Schema version tracking
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Singleton runtime metadata row.
CREATE TABLE runtime_state (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    last_attempted_at TIMESTAMP,
    last_successful_at TIMESTAMP,
    last_run_kind TEXT CHECK (
        last_run_kind IS NULL
        OR last_run_kind IN ('index', 'embed', 'daemon-once', 'daemon-scheduled', 'daemon-watch')
    ),
    last_index_total INTEGER,
    last_index_indexed INTEGER,
    last_index_skipped INTEGER,
    last_index_failed INTEGER,
    last_index_changed INTEGER,
    last_index_total_seconds DOUBLE,
    last_embed_messages INTEGER,
    last_embed_thinking INTEGER,
    last_embed_bash INTEGER,
    last_failure_message TEXT,
    last_failure_at TIMESTAMP,
    installed_scheduler TEXT CHECK (
        installed_scheduler IS NULL
        OR installed_scheduler IN ('launchd', 'systemd', 'cron')
    ),
    embedding_dimensions INTEGER,
    last_context_messages INTEGER NOT NULL DEFAULT 0,
    last_context_mode TEXT CHECK (
        last_context_mode IN ('off', 'template', 'llm-local', 'llm-remote', 'llm-codex')
    ),
    last_context_input_tokens INTEGER NOT NULL DEFAULT 0,
    last_context_output_tokens INTEGER NOT NULL DEFAULT 0,
    last_context_model TEXT,
    -- Failure-signature memory and index self-repair (REQ-RESIL-014..019).
    -- Nullable with defaults: DuckDB's ADD COLUMN cannot carry NOT NULL, so
    -- migrations 0023/0029 and a fresh schema.sql yield the same shape.
    last_fatal_signature TEXT,
    fatal_repeat_count INTEGER DEFAULT 0,
    last_fatal_at TIMESTAMP,
    last_index_repair_at TIMESTAMP,
    last_index_repair_signature TEXT,
    needs_index_verification BOOLEAN DEFAULT FALSE
);

INSERT OR IGNORE INTO runtime_state (singleton) VALUES (TRUE);

-- Session identity table (no FK constraints; see REQ-SCHEMA-003)
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,              -- SHA256(source:path)[:16]
    source TEXT NOT NULL CHECK (source IN ('claude_code', 'codex', 'pi_agent')),
    source_path TEXT UNIQUE NOT NULL,
    source_session_id TEXT            -- Original session ID from source
);

-- Mutable session metadata and file state
CREATE TABLE session_state (
    session_id TEXT PRIMARY KEY,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    duration_seconds INTEGER,

    model TEXT,
    cwd TEXT,
    git_repo TEXT,                    -- Detected git root
    git_branch TEXT,

    message_count INTEGER DEFAULT 0,
    tool_count INTEGER DEFAULT 0,
    input_tokens INTEGER,             -- NULL if not available
    output_tokens INTEGER,            -- NULL if not available

    is_complete BOOLEAN DEFAULT TRUE, -- FALSE if parsing was partial
    file_mtime DOUBLE NOT NULL,
    file_size BIGINT NOT NULL,
    last_byte_offset BIGINT DEFAULT 0, -- Resume position for incremental parse
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    agent_id TEXT,                   -- Subagent identity; NULL for the main thread
    UNIQUE(session_id, idx)
);

-- Mutable message content state (no embedding columns; see REQ-SCHEMA-004)
CREATE TABLE message_state (
    message_id TEXT PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT,
    thinking TEXT,                    -- Separate thinking content
    timestamp TIMESTAMP,
    has_thinking BOOLEAN DEFAULT FALSE,

    -- Contextual Retrieval (see Contextual Retrieval section).
    -- Rendered prefix prepended to content/thinking before embedding and FTS.
    -- Empty string when context_mode = 'off'; NULL only for pre-migration rows.
    context_text TEXT DEFAULT '',
    context_mode TEXT DEFAULT 'off' CHECK (
        context_mode IN ('off', 'template', 'llm-local', 'llm-remote', 'llm-codex')
    )
);

CREATE TABLE tool_calls (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_id TEXT,
    idx INTEGER NOT NULL,

    tool_name TEXT NOT NULL,
    tool_input JSON,                  -- Full tool input preserved
    agent_id TEXT,
    subagent_type TEXT,
    subagent_description TEXT,
    subagent_model TEXT,
    skill_name TEXT,

    -- Denormalized Bash fields for fast queries
    bash_command TEXT,                -- Full command string
    bash_base TEXT,                   -- First command (git, kubectl, etc.)
    bash_sub TEXT,                    -- Subcommand (commit, get, etc.)
    is_compound BOOLEAN DEFAULT FALSE -- Has pipes, &&, ||, etc.
);

-- Embedding tables (independently droppable/recreatable; see REQ-SCHEMA-005)
-- N = configured embedding dimensions (default 384, stored in runtime_state)
CREATE TABLE message_embeddings (
    message_id TEXT PRIMARY KEY,
    content_embedding FLOAT[N],       -- For semantic search of message content
    thinking_embedding FLOAT[N]       -- For semantic search of thinking blocks
);

CREATE TABLE tool_call_embeddings (
    tool_call_id TEXT PRIMARY KEY,
    bash_embedding FLOAT[N]           -- For semantic search of bash commands
);

CREATE TABLE embedding_cache (
    cache_key TEXT PRIMARY KEY,       -- Stable field-specific reuse key
    kind TEXT NOT NULL CHECK (kind IN ('content', 'thinking', 'bash')),
    raw_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    embedding FLOAT[N] NOT NULL,
    normalization_version INTEGER NOT NULL DEFAULT 1,
    -- Identifies the prefix-generator version (template vs LLM model+revision).
    -- Template/off entries leave this at 0; LLM modes use monotonically increasing values.
    context_version INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_sessions_source ON sessions(source);
CREATE INDEX idx_session_state_cwd ON session_state(cwd);
CREATE INDEX idx_session_state_git_repo ON session_state(git_repo);
CREATE INDEX idx_session_state_started ON session_state(started_at DESC);
CREATE INDEX idx_messages_session ON messages(session_id);
CREATE INDEX idx_messages_agent ON messages(agent_id);
CREATE INDEX idx_tool_calls_agent ON tool_calls(agent_id);
CREATE INDEX idx_tool_calls_subagent_type ON tool_calls(subagent_type);
CREATE INDEX idx_message_state_has_thinking ON message_state(has_thinking);
CREATE INDEX idx_tool_calls_session ON tool_calls(session_id);
CREATE INDEX idx_tool_calls_name ON tool_calls(tool_name);
CREATE INDEX idx_tool_calls_bash_base ON tool_calls(bash_base);
CREATE INDEX idx_tool_calls_bash_sub ON tool_calls(bash_sub);
CREATE INDEX idx_embedding_cache_kind ON embedding_cache(kind);

-- FTS indexes (created after data load, see Full-Text Search section).
-- Under Contextual Retrieval (REQ-CTX-006), CONTENT/THINKING FTS is built over the
-- concatenation (context_text || content) and (context_text || thinking) so BM25
-- scoring sees the prefix tokens. Implementations may store the concatenation in
-- derived columns or use DuckDB FTS over computed strings; the SPEC does not
-- prescribe a single mechanism, only the resulting indexed text.
-- PRAGMA create_fts_index(message_state, message_id, content, thinking, overwrite=1);
-- PRAGMA create_fts_index(tool_calls, id, bash_command, overwrite=1);
```

### Pydantic Models

These models are **domain objects** that map directly to database rows. They include all DB fields for insert/query operations. Nested relationships (Session.messages, Message.tool_calls) are populated when loading full session data.

```python
from datetime import datetime
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field

class Source(StrEnum):
    CLAUDE_CODE = "claude_code"
    CODEX = "codex"
    PI_AGENT = "pi_agent"
    GROK = "grok"
    KIMI_CODE = "kimi_code"

class ContextMode(StrEnum):
    OFF = "off"
    TEMPLATE = "template"
    LLM_LOCAL = "llm-local"
    LLM_REMOTE = "llm-remote"

class ToolCall(BaseModel):
    id: str
    session_id: str              # FK to sessions.id
    message_id: str | None       # FK to messages.id (nullable for orphan calls)
    idx: int                     # Position within message
    tool_name: str
    tool_input: dict[str, Any] | None = None

    # Bash-specific (denormalized)
    bash_command: str | None = None
    bash_base: str | None = None
    bash_sub: str | None = None
    is_compound: bool = False

    # v2: Embedding (NULL in v1)
    bash_embedding: list[float] | None = None

class Message(BaseModel):
    id: str
    session_id: str              # FK to sessions.id
    idx: int                     # Position within session
    role: str                    # user, assistant, system
    content: str | None = None
    thinking: str | None = None
    timestamp: datetime | None = None
    has_thinking: bool = False
    context_text: str | None = ""   # NULL only for pre-migration rows
    context_mode: ContextMode | None = ContextMode.OFF
    tool_calls: list[ToolCall] = Field(default_factory=list)  # Populated on load

    # v2: Embeddings (NULL in v1)
    content_embedding: list[float] | None = None
    thinking_embedding: list[float] | None = None

class Session(BaseModel):
    id: str
    source: Source
    source_path: str
    source_session_id: str | None = None  # Original ID from source file

    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: int | None = None

    model: str | None = None
    cwd: str | None = None
    git_repo: str | None = None
    git_branch: str | None = None

    message_count: int = 0
    tool_count: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None

    is_complete: bool = True
    file_mtime: float              # File modification time (for staleness)
    file_size: int                 # File size in bytes (for staleness)
    indexed_at: datetime | None = None  # When this session was indexed

    messages: list[Message] = Field(default_factory=list)  # Populated on load
    orphan_tool_calls: list[ToolCall] = Field(default_factory=list)  # Codex orphan calls
```

## CLI Interface

### Commands

```bash
# Indexing
recall index                        # Incremental index all sources
recall index --full                 # Force full reindex
recall index --source claude-code   # Index specific source
recall index --recreate             # Backup old DB and rebuild

# Search
recall search "auth"                # FTS across all content
recall search "kubectl" --tool Bash # Filter by tool
recall search --source codex "err"  # Filter by source
recall search --json                # JSON output

# List sessions
recall list                         # Recent sessions
recall list --source claude-code    # Filter by source
recall list --since 7d              # Time filter
recall list --project /path/to/repo # Filter by git repo
recall list --json                  # JSON output

# Analytics
recall stats                        # Overview dashboard
recall stats tools                  # Tool usage frequency
recall stats bash                   # Bash command breakdown
recall stats bash --suggest         # Generate permission rules
recall stats tokens                 # Token usage by project (git_repo)
recall stats usage [--since 30d]    # Token usage by source × model × host (REQ-USAGE-020)
recall stats --json                 # JSON output

# Indexing multi-host (REQ-MULTIHOST-001)
recall index --root ~/fleet/buildbox --host buildbox

# Session details
recall show <id>                    # Conversation transcript; tool-only turns collapse
recall show <id> --tools            # Include tool calls
recall show <id> --thinking         # Include thinking blocks (absent without it)
recall show <id> --json             # JSON output

# Database maintenance
recall compact                      # Rebuild the database file to reclaim space
recall db check-indexes             # Probe ART index/table divergence (REQ-RESIL-018)
recall db rebuild-indexes           # Drop and recreate every index; daemon must be stopped
```

### Output Formats

**TTY default:** Human-readable, conversation-style
```
[2024-01-15 10:30] Session abc123 (claude_code)
Project: /Users/dev/myproject
Duration: 45m | Messages: 23 | Tools: 47

user: Help me fix the auth bug
assistant: I'll investigate the authentication...
  [Bash] git status
  [Read] src/auth/handler.ts
```

**Non-TTY default (TOON):** Token-efficient structured output (~60% fewer tokens than JSON)
```
id, abc123
source, claude_code
started_at, 2024-01-15T10:30:00Z
messages, [...]
```

**JSON (--json or --format json):** Machine-readable for scripting and backward compatibility
```json
{
  "id": "abc123",
  "source": "claude_code",
  "started_at": "2024-01-15T10:30:00Z",
  "messages": [...]
}
```

**Format resolution order:**
1. `--json` flag → JSON
2. `--format <name>` → specified format
3. TTY → text, non-TTY → toon (with JSON fallback if `toon-format` unavailable)

**Call-to-Actions (--cta flag):**
```json
{
  "data": [{"id": "abc123", "source": "claude_code"}],
  "cta": [
    {"command": "recall show abc123", "description": "View session details"}
  ]
}
```
In TEXT mode, CTAs always render on stderr:
```
  Next: recall show abc123 -- View session details
```

### Verbosity

- Default: Progress bars, summary stats
- `-v/--verbose`: Detailed file-by-file logging

### Flag Normalization

**`--source` flag:**
- CLI accepts: `claude-code`, `codex`, `pi-agent`, `grok`, `kimi-code` (kebab-case for CLI ergonomics)
- Internally normalized to: `claude_code`, `codex`, `pi_agent`, `grok`, `kimi_code` (schema values)
- Mapping: `claude-code` → `claude_code`, `pi-agent` / `pi` / `pi_agent` → `pi_agent`, `kimi` / `kimi-code` / `kimi_code` → `kimi_code`

**`--since` flag:**
- Accepts relative durations: `7d`, `24h`, `30m`, `1w`
- Accepts absolute timestamps: `2024-01-15`, `2024-01-15T10:30:00`
- Parsed to UTC datetime for query
- Filters on "last active" time: `COALESCE(ss.ended_at, ss.started_at, ss.indexed_at)`. A session that started 3 hours ago but ended 5 minutes ago appears with `--since 1h`.

**`--project` flag:**
- Filters by `session_state.git_repo` (git root path)
- Accepts partial paths: `--project myrepo` matches `/Users/dev/myrepo`

**Tool name normalization:**
- Tool names stored as-is from source (case-sensitive)
- `--tool` filter is case-insensitive match

### Permission Suggestions

`recall stats bash --suggest` generates permission rules for Claude Code auto-approve:

**Output format (human-readable):**
```
Suggested Bash Permissions
==========================

High confidence (>=50 uses, no dangerous patterns):
  - git *           (523 uses)
  - npm test        (89 uses)
  - ruff check *    (67 uses)

Medium confidence (>=10 uses):
  - pytest *        (45 uses)
  - uv sync         (32 uses)

Review carefully (contains arguments/pipes):
  - docker build *  (12 uses)

Not suggested (contains rm, sudo, or writes):
  - rm -rf *        (3 uses) [SKIPPED]
```

**Confidence thresholds:**
- High: >=50 uses AND no dangerous patterns (rm, sudo, chmod, etc.)
- Medium: >=10 uses AND no dangerous patterns
- Low/Review: <10 uses OR contains pipes/complex arguments

**Output format (JSON with --json):**
```json
{
  "suggestions": [
    {
      "pattern": "git *",
      "count": 523,
      "confidence": "high",
      "reason": "No dangerous patterns detected"
    },
    {
      "pattern": "npm test",
      "count": 89,
      "confidence": "high",
      "reason": "Read-only test command"
    }
  ],
  "skipped": [
    {
      "pattern": "rm -rf *",
      "count": 3,
      "reason": "Destructive command"
    }
  ]
}
```

**Permission rule generation:**
- Groups by `bash_base` + `bash_sub`
- Applies safety heuristics (skip rm, sudo, chmod, etc.)
- Wildcards added for common argument patterns

## Session Sources

### Claude Code

**Location:** `~/.claude/projects/<encoded-path>/<session-id>.jsonl`

**Format:** JSONL with message objects containing:
- `type`: message type
- `message`: content object with role, content blocks
- `costUSD`, `inputTokens`, `outputTokens`: usage data (when present)

Subagent `progress` entries with `data.type = agent_progress` contribute nested
`data.message.message` content to the parent session, interleaved in file order
with sequential message indices. Messages and their tool calls carry
`data.agentId`; main-thread rows have NULL `agent_id`. The enclosing entry supplies
the timestamp, and nested usage contributes to session totals. This flat model
preserves the parent conversation rather than inventing separate sessions.
`Agent` dispatch calls retain `subagent_type`, description and model from their
input. `Skill` calls retain `input.skill`; other tool calls may derive `skill_name`
under the [Cross-Harness Skill Census](#cross-harness-skill-census) rules.
The Claude Code subagent fixture and parser tests cover this representation.

### Codex

**Location:**
- `~/.codex/sessions/<session_id>/rollout.jsonl` - Full session data
- `~/.codex/history.jsonl` - Session index (not parsed, use rollout files)

**Session ID source:** `payload.id` from `session_meta` entry in rollout.jsonl

**Format:** JSONL with entry types:

```jsonl
{"type": "session_meta", "payload": {"id": "abc123", "timestamp": "...", "cwd": "/path", "cli_version": "1.0", "git": {"branch": "main", "commit_hash": "..."}}}
{"type": "event_msg", "timestamp": "...", "payload": {"type": "user_message", "message": "Help me..."}}
{"type": "event_msg", "timestamp": "...", "payload": {"type": "agent_message", "message": "I'll help..."}}
{"type": "event_msg", "timestamp": "...", "payload": {"type": "function_call", "name": "shell", "parameters": {...}}}
{"type": "message", "timestamp": "...", "payload": {"role": "assistant", "content": [{"type": "text", "text": "..."}, {"type": "tool_use", "name": "shell", ...}]}}
```

**Entry type mapping:**
| Entry Type | Payload Type | Maps To |
|------------|--------------|---------|
| `session_meta` | - | Session metadata (id, cwd, git) |
| `event_msg` | `user_message` | Message(role="user") |
| `event_msg` | `agent_message` | Message(role="assistant") |
| `event_msg` | `function_call` | ToolCall |
| `message` | - | Legacy format: Message + embedded ToolCalls |

**Notes:**
- Both `event_msg` and `message` formats may appear in same file (legacy support)
- Tool calls in `message` format are embedded in `content` array as `tool_use` blocks
- `history.jsonl` contains session summaries but lacks full message data

### Pi Agent

**Location:** `~/.pi/agent/sessions/**/*.jsonl`

**Schema version:** Added in schema v2.

**Format:** JSONL with entry types:

```jsonl
{"type": "session", "id": "pi-session-123", "timestamp": "...", "cwd": "/path"}
{"type": "model_change", "timestamp": "...", "modelId": "gpt-5.4"}
{"type": "message", "timestamp": "...", "message": {"role": "user", "content": [{"type": "text", "text": "..."}]}}
{"type": "message", "timestamp": "...", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}, {"type": "toolCall", "name": "bash", "arguments": {"command": "ls"}}], "usage": {"input": 100, "output": 20}}}
{"type": "message", "timestamp": "...", "message": {"role": "toolResult", "content": [{"type": "text", "text": "output"}]}}
```

**Entry type mapping:**
| Entry Type | Maps To |
|------------|---------|
| `session` | Session metadata (id, cwd) |
| `model_change` | `session.model` (from `modelId` field) |
| `message` (role=user) | Message(role="user") |
| `message` (role=assistant) | Message(role="assistant") + embedded ToolCalls |
| `message` (role=toolResult) | Message(role="system") |

**Content block types:**
| Block Type | Maps To |
|------------|---------|
| `text`, `input_text`, `output_text` | Message content |
| `thinking` | Message thinking |
| `toolCall` | ToolCall (name, arguments) |

**Bash tool detection:** Tool names `bash`, `shell`, `exec_command`, `shell_command` trigger bash command extraction from `command` or `cmd` fields in the tool arguments.

### Kimi Code

**Location:** `$KIMI_CODE_HOME/sessions/<workDirKey>/<sessionId>/agents/<agent>/wire.jsonl` (default `$KIMI_CODE_HOME` = `~/.kimi-code`). Sub-agents get their own `agents/agent-N/wire.jsonl`; every wire file is indexed as its own session.

**Session ID source:** the `<sessionId>` directory name; cwd comes from the sibling `state.json` `workDir` field (the `workDirKey` path component is a non-reversible slug+hash).

**Format:** JSONL wire events, each with a millisecond epoch `time`:

```jsonl
{"type": "metadata", "protocol_version": "1.4", "created_at": 1784300000000}
{"type": "context.append_message", "message": {"role": "user", "content": [{"type": "text", "text": "..."}]}, "time": ...}
{"type": "context.append_loop_event", "event": {"type": "content.part", "part": {"type": "think", "think": "..."}}, "time": ...}
{"type": "context.append_loop_event", "event": {"type": "tool.call", "name": "Bash", "args": {"command": "ls"}, "toolCallId": "..."}, "time": ...}
{"type": "context.append_loop_event", "event": {"type": "tool.result", "toolCallId": "...", "result": {"output": "..."}}, "time": ...}
{"type": "usage.record", "model": "kimi-code/k3", "usage": {"inputOther": 100, "output": 20, "inputCacheRead": 50, "inputCacheCreation": 10}, "time": ...}
```

**Entry type mapping:**
| Entry Type | Maps To |
|------------|---------|
| `metadata` | `started_at` (via `created_at`) |
| `context.append_message` | Message (role from payload; carries user inputs) |
| loop event `step.begin` / `step.end` | assistant Message boundaries (one Message per LLM step) |
| loop event `content.part` (`think` / `text`) | Message thinking / content |
| loop event `tool.call` | ToolCall (name, args) |
| loop event `tool.result` | Message(role="system"), `[tool_call_id: ...]` prefix |
| `llm.request` | `session.model` (first `modelAlias`) |
| `usage.record` | additive token deltas (input includes cache counters) |
| `context.apply_compaction` | Message(role="system") with the compaction summary |

**Notes:**
- `turn.prompt` / `turn.steer` duplicate `context.append_message` and are skipped.
- Wire files under `agents/agent-N/` mark all their Messages/ToolCalls with `agent_id`.
- Per-session token totals are populated from `usage.record` (REQ-PARSE-013); the Fleet Token/Usage Ledger consumes them as-is and does not re-implement the Kimi parser.

### Grok (xAI Grok Build)

**Location:**
- Session transcript: `~/.grok/sessions/<encoded-cwd>/<sid>/chat_history.jsonl`
- Sidecars (same session dir): `summary.json`, `signals.json`
- Usage log (rotating): `~/.grok/logs/unified.jsonl`

**Session ID source:** directory name `<sid>` (= `source_session_id`); joins to usage events via `ctx`/top-level `sid` on `shell.turn.inference_done` log lines.

**Transcript notes:** `chat_history.jsonl` has **no** token usage fields. The session parser must leave `input_tokens` / `output_tokens` as `None` when only the transcript is available (REQ-PARSE-012). Exact tokens come only from the usage harvester (REQ-USAGE-010+).

**Sidecars (REQ-PARSE-014):** see parsers SPEC and the Fleet Token/Usage Ledger section below.

### Pi Package Manifest

recall installs as a pi package so pi agents get the `recall` and
`recall-setup` skills.

- **Skills-only:** recall ships no hooks (`hooks.json`), so the pi package
exposes `pi.skills: ["./plugins/recall/skills"]` with no hooks extension. **ratified (human 2026-08-12)**
- **Manifest:** root `package.json` with the `pi-package` keyword and a `pi`
key; the conventional top-level `skills/` layout is not used so the
claude/codex plugin tree stays put. **ratified (human 2026-08-12)**
- **Additive and non-invasive:** no change to the claude/codex plugin layout,
the Python packaging, or the release-please flow; the manifest must not
disturb the release-metadata guard (`tests/test_release_metadata.py`).
**ratified (human 2026-08-12)**
- **Validation floor:** `tests/test_pi_manifest.py` fails closed if the
manifest loses `pi-package`, its skills dir, or the `recall` + `recall-setup`
skills. **ratified (human 2026-08-12)**

Install from the repo: `pi install git:git@github.com:0xsend/recall.git`.

## Fleet Token / Usage Ledger

**Problem.** Fleet-wide token reporting is a load-bearing operator signal: when plans or routing shift work across providers, per-provider token consumption decides where capacity goes. As of v0.24.0:

| Source | Per-session tokens | Notes |
|--------|-------------------|--------|
| Claude Code | Yes | Cache counters folded into `input_tokens` |
| Codex | Yes | Absolute cumulative totals (`ABSOLUTE_TOKEN_SOURCES`) |
| Pi Agent | Yes | `usage.input` / `usage.output` |
| Kimi Code | Yes | `usage.record` deltas; cache folded into `input_tokens` (REQ-PARSE-013) |
| Grok | **No** | Transcript has no usage; exact usage lives in rotating `unified.jsonl` (~1 day retention under heavy use) |

Sessions on other hosts are invisible: discovery is home-dir only. `recall stats tokens` groups by `git_repo` only — not source × model × host × day, and has no cache split.

**Solution.** (1) Harvest Grok `unified.jsonl` into `usage_events` and roll up onto Grok sessions. (2) Enrich Grok session metadata from sidecars. (3) Multi-host ingest via alternate home root + `host` dimension. (4) `recall stats usage` rollup. Cost/pricing is explicitly deferred.

### Domain model additions

```sql
-- Per-inference (or per-harvest) usage events. Grok is the first writer;
-- other sources may remain session-level only.
CREATE TABLE usage_events (
    id TEXT PRIMARY KEY,                 -- stable hash (see REQ-USAGE-012)
    source TEXT NOT NULL,                -- e.g. 'grok'
    source_session_id TEXT NOT NULL,     -- joins sessions.source_session_id
    session_id TEXT,                     -- recall sessions.id when known; NULL if orphan
    ts TIMESTAMP,
    prompt_tokens INTEGER,
    cached_prompt_tokens INTEGER,
    completion_tokens INTEGER,
    reasoning_tokens INTEGER,
    host TEXT,
    harvested_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_usage_events_source_sid ON usage_events(source, source_session_id);
CREATE INDEX idx_usage_events_session ON usage_events(session_id);
CREATE INDEX idx_usage_events_ts ON usage_events(ts);

-- Cursor for rotating usage logs (byte-offset incremental harvest).
CREATE TABLE usage_log_cursors (
    path TEXT PRIMARY KEY,
    byte_offset BIGINT NOT NULL DEFAULT 0,
    file_size BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMP NOT NULL
);

-- session_state additions (migration):
--   cached_input_tokens INTEGER  -- NULL if unknown; not a subset re-add for Codex
--   host TEXT NOT NULL DEFAULT 'local'
```

**Rollup semantics (Grok):** after harvest (and on re-harvest), for each Grok session with matching `source_session_id = sid`:

- `input_tokens` = `SUM(prompt_tokens)` over that sid's events
- `cached_input_tokens` = `SUM(cached_prompt_tokens)`
- `output_tokens` = `SUM(completion_tokens)`
- `reasoning_tokens` stay on events only for v1 of this feature (no session column required); stats may expose them later

`prompt_tokens` and `cached_prompt_tokens` are stored as reported by the log. Fresh input for display = `input_tokens - cached_input_tokens` when both are non-null and cached ≤ input; otherwise fresh is null/omitted rather than inventing a number.

**Host:** every session row carries `session_state.host`. Default for ordinary local index is the short hostname of the machine (fallback `"local"` if unobtainable). `recall index --host NAME` overrides. Multi-host roots always set host (see REQ-MULTIHOST-002).

### Requirements

#### Grok sidecars — REQ-GROK-TS family

- `REQ-GROK-TS-001`: On full or incremental parse of a Grok `chat_history.jsonl`, when sibling `summary.json` exists and is valid JSON, the parser MUST set `started_at` from `created_at` and `ended_at` from `last_active_at` (ISO-8601 / parseable timestamps). Missing fields stay `None`.
- `REQ-GROK-TS-002`: From `summary.json` when present: `cwd` from `info.cwd` (prefer over path-decoded workspace), `git_repo` from `git_root_dir`, `git_branch` from `head_branch`, `model` from `current_model_id` when the transcript has not already set a model (or prefer summary when both exist — prefer summary when non-empty).
- `REQ-GROK-TS-003`: From sibling `signals.json` when present: `duration_seconds` from `sessionDurationSeconds` when that field is a non-negative number. Do **not** write `contextTokensUsed` into `input_tokens` or `output_tokens`.
- `REQ-GROK-TS-004`: Sidecar read failures (missing file, JSON error, unexpected shape) MUST NOT fail the session parse; fall back to pre-sidecar behavior for the affected fields.

#### Grok usage harvest — REQ-USAGE / REQ-GROK-USAGE

- `REQ-USAGE-010`: recall MUST harvest `~/.grok/logs/unified.jsonl` (or `$GROK_HOME/logs/unified.jsonl` if a home override is configured) using a byte-offset cursor in `usage_log_cursors`. Harvest runs during daemon scheduled/watch ticks and during `recall index` (same process topology as indexing — no separate long-lived harvester process).
- `REQ-USAGE-011`: Harvest MUST only persist events where `msg == "shell.turn.inference_done"`. Events missing `sid` or with non-object `ctx` are skipped (fail soft). Missing numeric fields in `ctx` default to null/0 per field without aborting the run.
- `REQ-USAGE-012`: Each persisted event has a stable primary key `id` derived from content that makes re-harvest idempotent (e.g. hash of `source|sid|ts|loop_index|prompt_tokens|completion_tokens` or log path + byte offset of the line). Re-inserting the same event is a no-op or upsert of identical values — never double-count in rollups.
- `REQ-USAGE-013`: When the log file size is **smaller** than the stored cursor `file_size` (or size < `byte_offset`), the harvester MUST treat the file as rotated/truncated, reset the cursor to 0, and re-read from the start **without** deleting already-stored `usage_events` (idempotent keys prevent double-count).
- `REQ-USAGE-014`: After ingesting new events (and periodically on index), roll up per `(source='grok', source_session_id)` onto matching `session_state` token columns per Domain model. Sessions not yet indexed remain events-only (`session_id` null) until a later index joins them. Every harvest rolls up every sid, so a session row rewritten without tokens heals on the next tick even when the log has nothing new; the rollup writes only rows whose totals or link differ, so a harvest with nothing new writes nothing. A sid can map to several sessions (a moved workspace leaves the old path indexed beside the new one): each such session carries the sid's totals, and the sid's events link to one owner, the smallest session id, so the indexed link column never alternates between them. The harvest cursor is written only when its offset or size moves.
- `REQ-USAGE-015`: Grok `input_tokens` / `output_tokens` / `cached_input_tokens` MUST be filled only from harvest rollup, never from transcript proxies, message lengths, or `signals.json.contextTokensUsed`.
- `REQ-USAGE-016` **[RISK-HIGH — schema]**: Schema migration adds `usage_events`, `usage_log_cursors`, `session_state.cached_input_tokens`, and `session_state.host` (and bumps `SCHEMA_VERSION`). Migration is automatic, idempotent, and follows REQ-MIG-* authoring rules. Fresh `schema.sql` matches post-migration shape.

#### Multi-host ingest — REQ-MULTIHOST

- `REQ-MULTIHOST-001` **[RISK-HIGH — public CLI]**: `recall index` accepts `--root PATH` where `PATH` is treated as an alternate home directory containing source trees (`.grok/`, `.claude/`, `.codex/`, `.pi/`, `.kimi-code/`, …) in the same layout as `$HOME`. Discovery for that run uses `PATH` instead of the real home for all registered parsers.
- `REQ-MULTIHOST-002`: When `--root` is set, `--host NAME` is required (or SPEC-equivalent: if omitted, host defaults to the final path component of `--root`). All sessions indexed in that run get `session_state.host = NAME`. Re-index of the same `source_path` is idempotent (same session id from source+path hash); host and tokens update in place.
- `REQ-MULTIHOST-003`: Multi-host **ingest** (rsync/copy of session dirs + local index with `--root`) remains in scope as an offline / air-gapped path. For a live Tailscale/SSH fleet, preferred product path is read-only **fleet query fan-out** (REQ-FLEET-*) so each edge daemon remains authoritative and Grok harvest is not re-derived on a hub. Daemon clustering, remote multi-writer RPC, and cross-host locks remain out of scope (see daemon non-goals). Documentation MUST still include an rsync recipe and a Grok rotation cadence note for the ingest path: busy Grok hosts can rotate `unified.jsonl` within ~22h — harvest host-side or sync logs faster than rotation.
- `REQ-MULTIHOST-004`: Default local indexing (no `--root`) continues to work unchanged aside from populating `host` with the local default.

#### Stats usage rollup — REQ-USAGE-ROLLUP

- `REQ-USAGE-020` **[RISK-HIGH — public CLI / RPC]**: Add `recall stats usage` and RPC `recall.stats_usage` reporting token totals grouped by **source × model × host** (and by **day** when `--group day` or default day breakdown is specified — minimum bar: source × model × host with optional `--since`).
- `REQ-USAGE-021`: Each group reports `input_tokens`, `cached_input_tokens` (0 or null when unknown), `output_tokens`, and derived `fresh_input_tokens` when computable. No currency / cost fields.
- `REQ-USAGE-022`: `recall stats usage` and other stats subcommands that accept time bounds support `--since` with the same duration grammar as `recall list --since` (e.g. `30d`, `12h`). Existing `recall stats tokens` (by git_repo) remains and is not removed.
- `REQ-USAGE-023`: Rollup includes all sources that have session token data (Claude, Codex, Pi, Kimi, Grok after harvest). Sources with null tokens do not contribute invented zeros to group totals (SUM of non-null / COALESCE policy: use `SUM(COALESCE(x,0))` only when documenting that unknown sessions count as zero — prefer excluding sessions where both input and output are null so Grok-before-harvest does not pollute totals). **Decision:** exclude sessions where `input_tokens IS NULL AND output_tokens IS NULL` from usage totals.

### Invariants

- Never invent token usage for a provider/event stream that did not record it.
- Grok transcript parse and Grok usage harvest are separate paths; harvest may update sessions the parser left token-null.
- Re-harvest and re-index are idempotent with respect to token totals (no double-count under rotation).
- `host` is a dimension for analytics, not part of `sessions.id` identity (identity remains source+path hash).
- Cost/pricing is not computed in core.

### Non-goals

- Cost calculation, list prices, billing-dollar authority, or user pricing config (follow-up).
- Wire interception / MITM to observe tokens.
- Re-implementing Kimi Code, Claude, Codex, or Pi parsers.
- Changing Kimi sub-agent = separate session design.
- Daemon multi-host clustering or cross-host locks.
- Promoting Claude/Kimi cache counters into separate `cached_input_tokens` columns (optional follow-up). Grok **must** populate `cached_input_tokens` from harvest, and Pi Agent populates it from transcript `cacheRead` (parser `REQ-PARSE-029`). Other sources may leave it NULL.

### Decisions

- 2026-09-09 — Current supported transcript files are authoritative during recovery. Preserve the prior index snapshot and the recorded discrepancy certificate; only those documented historical differences may be accepted by the recovery verifier. The certificate binds source hashes and missing semantic keys. Omissions of current supported content, incomplete captures, and missing sources remain failures or unresolved states, and equivalent derived inputs remain preserved. Older content absent from current files remains recoverable from the saved snapshot rather than ordinary search. **ratified (human)**

1. Cost/pricing is deferred (human-ratified).
2. Kimi Code is already shipped; the ledger consumes its session tokens in rollups only.
3. Usage harvest reuses the daemon/index process; no new long-lived binary.
4. `stats usage` excludes sessions with both token fields NULL.
5. Fresh input is displayed as `input - cached` when both are known and `cached ≤ input`.

### Risk tags

- **[RISK-HIGH]** Schema migration (`usage_events`, cursors, columns) — REQ-USAGE-016.
- **[RISK-HIGH]** Public CLI/RPC (`index --root/--host`, `stats usage`, `--since`) — REQ-MULTIHOST-001, REQ-USAGE-020–022.
- **[RISK-MEDIUM]** Rotation-sensitive log harvest — wrong cursor logic loses or double-counts fleet signal.

### Acceptance criteria

- [ ] Grok fixture with `summary.json` / `signals.json` gets real `started_at` / `ended_at` / git / model / cwd (REQ-GROK-TS-*).
- [ ] Grok fixture harvest: session tokens match hand-sum of `inference_done` events; mid-fixture rotation neither drops nor double-counts (REQ-USAGE-010–015).
- [ ] Re-harvest is idempotent (event counts and session totals stable).
- [ ] `recall index --root <alt> --host <name>` indexes under that host; second run does not duplicate sessions.
- [ ] `recall stats usage --since 30d` returns per-source (× model × host) totals including Grok after harvest and Kimi/Claude/Codex/Pi when present; no cost fields.
- [ ] Kimi / Claude / Codex / Pi existing parser tests remain green.
- [ ] Migration applies on open; `SCHEMA_VERSION` and `schema.sql` agree.
- [ ] Full `uv run pytest -q`, `uv run ruff check .`, typecheck green.

### Test traceability

Filled during TDD (paths provisional):

| REQ | Tests (planned) |
|-----|-----------------|
| REQ-GROK-TS-* | `tests/test_parsers/test_grok.py` |
| REQ-USAGE-010–015 | `tests/test_services/test_usage_harvest.py` (name TBD) |
| REQ-MULTIHOST-* | `tests/test_cli/test_index_root.py` or indexer tests |
| REQ-USAGE-020–023 | `tests/test_cli/test_stats_usage.py` / analytics tests |

## Fleet Query Fan-out (daemon-centric)

**Problem.** Operators and agents need one view of sessions, search hits, and token
usage across a small personal fleet (laptop, build box, phone-adjacent hosts). Each
machine already runs a recall daemon that watches sessions, indexes into a local
DuckDB, and harvests Grok usage. Re-syncing home trees to a hub and re-indexing
duplicates that work and re-introduces Grok log-rotation risk on the hub. Without
a stable `host` field on agent-facing list/search/show rows, merged fleet results
are ambiguous (same git remote / similar cwd on multiple machines).

**Solution.** (1) Expose `session_state.host` on local list, show, and search
outputs so every session-bearing row is host-labeled. (2) Fleet config names
reachable hosts. (3) On-demand CLI fan-out: one-shot SSH per host runs the same
structured `recall … --json` (or equivalent RPC) against that host’s daemon, then
the client merges results. Edge daemons remain the write authority; the fleet
client is read-only aggregation. No fleet worker inside the daemon for v1.

**Relationship to multi-host ingest.** REQ-MULTIHOST-* file ingest stays available
for offline / bulk import. Live fleet UX is REQ-FLEET-* fan-out. Grok token
correctness on a host is that host’s harvest path (REQ-USAGE-010+); fleet does not
re-harvest remote `unified.jsonl` for normal queries.

### Domain model

```toml
# ~/.config/recall/fleet.toml  (path may also be set in config.toml)
[[host]]
name = "devbox"                          # stable label; preferred stamp on rows
ssh = "devbox.example.ts.net"         # OpenSSH target (user@host or Host alias)

[[host]]
name = "buildbox"
ssh = "buildbox"
```

```text
FleetHost { name: str, ssh: str }
FleetTransport.exec(host, argv|rpc, params, timeout) -> JSON
  v1: ssh -o BatchMode=yes -o RemoteCommand=none -o ConnectTimeout=…
      -- <target> -- recall <cmd> … --json
  later: ControlMaster with recall-private ControlPath + short ControlPersist;
         or multi-method one-shot remote; or native RPC over Tailscale
FleetMerge: list | search | stats_usage | show-locate
```

**Host field (storage already present):** `session_state.host` and
`usage_events.host` (Fleet Token / Usage Ledger). Local index stamps short
hostname by default; multi-host ingest uses `--host` / root basename.

**Host field (API — gap today, required by this section):** list, show, and search
structured outputs MUST include `host` so agents and fleet merge never invent
machine identity from path heuristics alone.

### Requirements

#### Host on agent-facing surfaces — REQ-HOST-API

- `REQ-HOST-API-001` **[RISK-HIGH — public CLI/RPC]**: `recall list` and RPC
  `recall.list` MUST include `host` on each session summary (from
  `session_state.host`). CLI manifests / `--fields` allowlists MUST allow `host`.
- `REQ-HOST-API-002` **[RISK-HIGH — public CLI/RPC]**: `recall show` and RPC
  `recall.show` MUST include `host` on the session object.
- `REQ-HOST-API-003` **[RISK-HIGH — public CLI/RPC]**: `recall search` and RPC
  `recall.search` MUST include `host` on each hit (join/lookup from
  `session_state` for the hit’s `session_id`).
- `REQ-HOST-API-004`: When `session_state.host` is missing (pre-migration row edge
  case), surfaces MUST emit a non-empty fallback consistent with local default
  labeling (short hostname or `"local"`) rather than omitting the key in
  structured output. `"local"` is a sentinel meaning "unattributed", not a machine
  identity; it is correct on a local surface and MUST NOT be treated as an
  authoritative label by any cross-machine merge (REQ-FLEET-MERGE-002).
- `REQ-HOST-API-005` (optional filter, same ship train preferred): `list` and
  `search` accept `--host NAME` to restrict to that `session_state.host` (exact
  match). Useful for multi-host ingest DBs and for local debugging.

#### Fleet inventory — REQ-FLEET-CFG

- `REQ-FLEET-CFG-001`: Fleet inventory is a dedicated file (default
  `~/.config/recall/fleet.toml`) listing zero or more hosts. Each host has a
  non-empty `name` (stable label) and an `ssh` target string. Duplicate `name`
  values are invalid and MUST fail load with a clear error. An `ssh` target
  starting with `-` is invalid and MUST fail load, so a target can never be
  parsed as an ssh option.
- `REQ-FLEET-CFG-002`: Empty inventory or missing file means fleet commands report
  “no hosts configured” and exit non-zero for fan-out verbs (or no-op with a clear
  message for `fleet status` — **Decision:** `fleet status` succeeds with zero
  hosts and empty table; fan-out query verbs fail closed with guidance to add
  hosts).
- `REQ-FLEET-CFG-003`: Inventory is the operator address book only. It does not
  grant write access to remote DBs and does not replace SSH auth (keys / agent).

#### Transport — REQ-FLEET-SSH

- `REQ-FLEET-SSH-001`: v1 transport is **on-demand**, **one SSH session per host
  per fleet CLI invocation** (not a long-lived fleet worker in the daemon).
- `REQ-FLEET-SSH-002`: SSH MUST pass at least: `BatchMode=yes`,
  `RemoteCommand=none`, `RequestTTY=no`, `ConnectTimeout` (bounded, configurable
  with a default such as 10s), and MUST NOT inherit a user ControlMaster path
  that may carry an interactive `RemoteCommand` (deploy-hosts class of failure).
  Prefer `ControlPath=none` in v1 unless an explicit recall-private mux is
  enabled later. ssh has no remote argv: it joins its trailing arguments with
  spaces and hands one string to the remote login shell, which re-splits it.
  Every remote argv element MUST therefore be shell-quoted so that a single
  round of shell word-splitting reproduces the argv exactly and each element
  reaches the remote `recall` CLI as one argument — multi-word search queries
  and paths containing spaces included.
- `REQ-FLEET-SSH-003`: Per-host work has a hard timeout budget; a hung host MUST
  not block the fleet command forever. Concurrent fan-out is allowed with a
  bounded concurrency cap (default small, e.g. 4–8).
- `REQ-FLEET-SSH-004`: Remote command is the installed `recall` CLI with structured
  output (`--json` or equivalent non-TTY default). Remote daemon auto-fork /
  existing RPC path is used as today — fleet does not open the remote DuckDB file
  over SSH.
- `REQ-FLEET-SSH-005`: Unreachable, auth-failed, or timed-out hosts are **skipped
  with an explicit per-host error** in the command summary/stderr; other hosts
  still contribute. Fan-out succeeds if at least one host returns data unless
  zero hosts succeeded (**Decision:** exit non-zero if every host failed; exit
  zero if any host succeeded, with warnings for skips). The per-host error is a
  single bounded line and MUST NOT report an informational SSH client notice
  (pseudo-terminal, known-hosts) as the failure when the remote command's own
  output carries the real message.

#### Fan-out commands — REQ-FLEET-CMD

- `REQ-FLEET-CMD-001` **[RISK-HIGH — public CLI]**: `recall fleet status` probes
  each inventory host (e.g. `recall --version` and/or `recall daemon status`) and
  reports name, reachability, binary version, daemon version / drift when
  available, and error if any.
- `REQ-FLEET-CMD-002` **[RISK-HIGH — public CLI]**: Fleet-scoped query verbs cover
  at least: `stats usage`, `list`, and `search` (via `--fleet` on existing
  commands and/or `recall fleet <subcommand>` — **Decision:** prefer `--fleet` on
  existing verbs for agent discoverability, plus `fleet status` as a dedicated
  subcommand). Filters (`--since`, `--source`, `--project`, query string, etc.)
  pass through to each remote invocation where applicable.
- `REQ-FLEET-CMD-003`: `show` under fleet either accepts an explicit host
  (`--host` / `--fleet-host`) or probes hosts until the session id resolves
  (**Decision:** require `--host` when using fleet show if ambiguous; allow
  omit-when-unique after single-host hit). Documented in CLI help.
- `REQ-FLEET-CMD-004`: Local-only behavior (no `--fleet`) is unchanged except for
  REQ-HOST-API-* field additions.

#### Merge and host stamping — REQ-FLEET-MERGE

- `REQ-FLEET-MERGE-001`: Every object in a fleet-merged structured result MUST
  include a non-empty `host` field.
- `REQ-FLEET-MERGE-002`: Host precedence for each row: (1) `host` from the remote
  payload when present, non-empty, and not the unattributed-host sentinel; (2)
  else the inventory `name` for that SSH hop. Never drop `host` on fleet output.
  The sentinel is the literal `local` that REQ-HOST-API-004 has surfaces emit for
  a session they cannot attribute to a named machine — it states "unattributed in
  this database's own frame of reference" and carries no identity for the control
  host, so over an SSH hop it MUST lose to the inventory name exactly as a missing
  label does. Matching is exact after stripping, so real hostnames that merely
  contain `local` (`localhost`, `local-dev`) stay attributed. This applies to
  every fan-out surface — `list`, `search`, `show`, and `stats usage` — which
  share one stamping implementation.
- `REQ-FLEET-MERGE-003`: `stats usage` fleet merge groups by
  `(source, model, host)` (and day if requested) and **sums** numeric token fields
  and session counts; does not invent cost fields.
- `REQ-FLEET-MERGE-004`: `list` fleet merge concatenates remote lists, stamps host,
  then applies global sort by recency and a single global `--limit` (default same
  as local list). Document that per-host limits may be used internally to bound
  payload size (e.g. request `limit` per host then re-limit globally).
- `REQ-FLEET-MERGE-005`: `search` fleet merge concatenates hits, stamps host, sorts
  by score descending, applies global `--limit`. Scores are not re-normalized
  across hosts in v1 (per-host ranking only).
- `REQ-FLEET-MERGE-006`: Session `id` remains the recall session id from the
  remote DB. Fleet does not rewrite ids. Callers that need a globally unique
  agent handle use `(host, id)` or pass `--host` on show.

### Invariants

- Edge daemon + local DuckDB is the write authority for that host’s sessions and
  Grok usage; fleet fan-out is read-only.
- Structured fleet results never omit `host`.
- No multi-writer access to a remote DuckDB file over SSH.
- Fail soft per host; fail closed only when the inventory is invalid or every host
  fails for a query verb.
- SSH options neutralize interactive `RemoteCommand` hijacks (`RemoteCommand=none`).

### Non-goals

- Rsync / home-tree pull as the primary fleet path (remains REQ-MULTIHOST ingest).
- Re-harvesting remote Grok logs on a hub for normal fleet queries.
- Daemon-resident fleet worker, permanent ControlMaster, or always-on mesh.
- Pair codes, public internet pairing, or new auth beyond SSH.
- Cross-host global search re-ranking / shared embedding space.
- Multi-master DB merge, snapshot exchange, or FTS index federation (optional
  later export/import is out of this section).
- Exposing remote daemon Unix sockets on the network without SSH.

### Decisions

1. Live fleet = SSH fan-out to edge daemons; not hub re-index (ratified 2026-08-07).
2. On-demand transport; one-shot SSH per host per invocation in v1 (ratified).
3. `ControlPath=none` (or no user mux) in v1; recall-private short ControlPersist
   is a later optimization (provisional until measured need).
4. Prefer `--fleet` on existing verbs + `fleet status` (ratified for SPEC; CLI
   shape may use `fleet list` aliases if discoverability suffers — record if so).
5. Exit zero if any host succeeded for query verbs; non-zero if all failed or
   inventory invalid (ratified).
6. `show --fleet` requires `--host` when not uniquely resolved (ratified).
7. Host API exposure (list/show/search) ships with or before fleet fan-out so
   agents are not trained on host-blind rows (ratified).

### Risk tags

- **[RISK-HIGH]** Public CLI/RPC shape: `host` on list/show/search — REQ-HOST-API-001–003.
- **[RISK-HIGH]** Public CLI: `--fleet` / `fleet status` — REQ-FLEET-CMD-*.
- **[RISK-MEDIUM]** SSH option interaction with operator `~/.ssh/config` (mitigate
  with explicit `RemoteCommand=none`, BatchMode, timeouts).
- **[RISK-MEDIUM]** Partial fleet results look complete to agents — mitigate with
  stderr/summary listing skipped hosts and structured optional `fleet_errors`.

### Acceptance criteria

- [ ] `recall list --json` includes `host` on each row; `--fields` can select `host`.
- [ ] `recall show <id> --json` includes `host`.
- [ ] `recall search "…" --json` includes `host` on each hit.
- [ ] `~/.config/recall/fleet.toml` with two hosts: `recall fleet status` shows both;
      one forced-down host is skipped with error, the other still reports.
- [ ] `recall stats usage --fleet --since 7d --json` returns rows with `host` and
      totals consistent with summing per-host `stats usage` (within skip set).
- [ ] `recall list --fleet --since 7d --json` every row has non-empty `host`; global
      limit honored.
- [ ] `recall search "…" --fleet --json` every hit has non-empty `host`.
- [ ] SSH invocation includes `RemoteCommand=none` and `BatchMode=yes` (unit or
      integration assertion on constructed argv).
- [ ] Without `--fleet`, list/search/show remain single-host; only additive `host`
      field appears.
- [ ] Full `uv run pytest -q`, lint, typecheck green for the ship train.

### Test traceability

Filled during TDD (paths provisional):

| REQ | Tests (planned) |
|-----|-----------------|
| REQ-HOST-API-* | `tests/test_services/test_sessions.py`, search/show CLI tests |
| REQ-MIG-008/009 | `tests/test_db/test_migrations.py` (0022 backfill, pre-image, idempotency) |
| REQ-FLEET-CFG-* | `tests/test_core/test_fleet_config.py` |
| REQ-FLEET-SSH-* | `tests/test_services/test_fleet_transport.py` (argv/mocks) |
| REQ-FLEET-CMD-* / MERGE-* | `tests/test_cli/test_fleet.py` |

## Live Agent Sessions

Status: implemented. Extends `REQ-DAEMON-043..050` (the daemon's in-memory live set),
`REQ-FLEET-CMD-*` (fan-out), and the `recall show` contract. One additive
migration (`REQ-LIVE-006`, schema version 24).

**Problem.** An operator driving several agents (a "lane driver" starting and
steering harness sessions in terminal multiplexer shells across three hosts)
has no reliable read of what those agents are doing. recall models the past;
it has no notion of a session that is *happening*. Observed while a driver
session monitored four lanes:

1. **No liveness.** `is_complete` means "parsed without JSON errors", not
   "ended". `ended_at` is the last message timestamp. Nothing says whether an
   agent is still running, waiting for input, or gone. The driver derived it
   from `ps` output, the session multiplexer's session list, and hand-written Python tailing the raw jsonl in
   a `wc -l` poll loop, which re-implements per-harness transcript knowledge
   outside recall.
2. **No freshness contract.** The driver "trusts the daemon is up-to-date".
   At one observation the index trailed the on-disk transcript by 117 KB and
   74 s (`file_size` 10,896,104 indexed vs 11,013,475 on disk; watch debounce
   is 5 s, so the lag came from a queued write behind other live sessions).
   Nothing on `list`/`show` exposes that lag, and `--fields` rejects
   `file_mtime` and `indexed_at`.
3. **No tail or delta read.** `recall show` returns the whole session (1,591
   messages, 10 MB for the driver session) and `--message-limit N` is the
   *first* N. There is no cursor, so a monitor loop re-reads everything.
4. **No join to the steering surface.** The driver steers with
   the session multiplexer's send command, addressed by pane handle, but recall
   knows nothing of handles. The mapping was recovered by comparing process
   start times to the multiplexer's session reservation timestamps. Claude Code hook payloads carry
   `session_id`, `transcript_path`, and `cwd`, and the session multiplexer's attention hook
   receives them and discards them.
5. **No turn state.** recall discards `tool_result` records and
   `stop_reason`, so "mid-tool", "turn finished, awaiting input", and
   "running subagents" cannot be derived from the index at all.
6. **Fleet latency.** `recall list --fleet` is one SSH per host per call
   (11 s for four hosts vs 1–3 s local). Fine for a sitrep, wrong inside a
   monitor loop.

**Why recall, not the session multiplexer.** The session multiplexer already
has the *surface* half: attention states (`blocked`/`done`/`working`), a
planned PTY read path (history/tail, and a viewport with cursor as the
input-readiness signal), and steering (its send command). Those answer "is the
terminal ready for input" and are deliberately harness-agnostic (the
multiplexer forbids special-casing coding agents, and a cross-host inventory
service is outside its scope). They cannot answer "what did the agent just
decide, which tool is it in, is it blocked on a question, how many subagents
are running" — that is transcript knowledge, alt-screen redraws never commit
to scrollback, and recall already owns five harness parsers, a watch daemon
with a live-session set on every fleet host, host-labeled rows, and fleet
fan-out. The split is therefore:

| Question | Owner | Source |
|---|---|---|
| Is the agent process alive; is the PTY accepting input; is it `blocked` on a permission prompt | session multiplexer | attention envelope, viewport cursor |
| What is the agent doing, what did it last say, is the turn over, which tool, which subagents, what changed since I last looked | recall | transcript, indexed live |
| Steer it | session multiplexer | its send command |

recall never claims `blocked`: permission prompts are not in any transcript.
The session multiplexer never parses transcripts. The join key is the harness session id.

### Domain model

```text
LiveSession                       # one per roster row; --all includes ended sessions
  id, source, source_session_id, host, cwd, git_repo, git_branch, model
  liveness        : enum { active, idle, ended, unknown }      # REQ-LIVE-001
  last_activity_at: datetime                                   # max(last message ts, file_mtime)
  freshness       : Freshness                                  # REQ-LIVE-003
  turn            : TurnState                                  # REQ-LIVE-005
  cursor          : str                                        # opaque; REQ-LIVE-004
  writer_pid      : int?                                       # enrichment only; REQ-LIVE-008

Freshness
  file_mtime, file_size            # on-disk now (stat at query time)
  indexed_mtime, indexed_size      # session_state row
  lag_seconds, lag_bytes           # file minus indexed; 0 when current
  current         : bool           # lag_bytes == 0

TurnState
  state           : enum { working, awaiting_input, subagents_running, ended, unknown }
  last_user_at, last_assistant_at
  last_user_text, last_assistant_text        # truncated (default 400 chars)
  running_tool    : { name, summary, started_at }?   # open tool_use with no tool_result
  subagents_active: int
  stop_reason     : str?                     # harness-native, passthrough
```

### Requirements

- `REQ-LIVE-001`: **Liveness is derived, never declared.** `active` iff the
  session's transcript path is a member of the daemon live set (`REQ-DAEMON-043`,
  demotion after `live_idle_threshold`, default 300 s). `ended` iff the parser
  observed a harness end marker for the session or a pid stamped on the
  session (`REQ-LIVE-008`) no longer exists. `idle` iff neither, and
  `last_activity_at` is within `live.idle_window` (default 24 h). Otherwise
  `unknown`. The existing `is_complete` field keeps its meaning ("parsed
  clean") and is not renamed; it is a public list field.
- `REQ-LIVE-002`: `recall live` **[public CLI]** lists sessions with liveness
  `active` by default; `--all` also includes ended rows still in the watched
  set and recent indexed `idle`/`ended` sessions within `live.idle_window`
  (default 24 h); `--source`, `--project`, `--host`,
  `--fleet` compose as on `list`. Rows are `LiveSession` objects sorted by
  `last_activity_at` descending. Structured output obeys `REQ-CLI-004/012`.
- `REQ-LIVE-003`: **Freshness is always reported and never assumed.** Every
  `LiveSession` and every `recall show` on a non-ended session carries
  `freshness`. `current` is true only after validated catalog progress through
  the complete captured source boundary. A source whose `committed_generation`
  is 0, or that has no committed catalog row, is `current: false` with
  limitation `not_yet_indexed`.

  `--fresh` on `live` is a bounded catch-up of already-indexed rows on the
  selected page (`committed_generation > 0` and not current). It does not
  first-index never-indexed rows; those stay on the fair coordinator after
  resume and inventory. The wait is bounded by the lesser of
  `live.fresh_timeout` and the responsive-read floor (2 s). On timeout the
  listing still answers, with `freshness.current = false` and a `note:` on
  stderr for rows that were eligible to catch up.

  `--fresh` on `show` asks the daemon to index that one already-known
  session's pending bytes synchronously before answering, bounded by
  `live.fresh_timeout` (default 10 s); on timeout the command still answers,
  with `freshness.current = false` and a `note:` on stderr.

  Without a daemon, `--fresh` fails closed with a stable error code rather
  than reading stale rows silently.
- `REQ-LIVE-004`: **Tail and cursor reads.** `recall show <id> --tail N`
  returns the last N messages (existing `--message-limit` stays head-anchored;
  the two are mutually exclusive). `recall show <id> --after <cursor>` returns
  only messages appended after the cursor and a new cursor; an empty delta is
  a successful, empty response. Cursors are opaque strings encoding
  `(session id, last message idx)`; a cursor for another session is a
  validation error. `--follow` (in v1, decided 2026-09-07) streams deltas as newline-delimited
  JSON, one object per line regardless of `--format`, until `--timeout`
  (default 60 s) or the session reaches `ended`. Each delta is produced by the
  daemon's session-indexed event (`REQ-LIVE-011`), not by client polling; a
  quiet session yields no lines, and the final line is a terminal object
  naming why the stream closed (`timeout` | `ended` | `daemon_stopped`).
- `REQ-LIVE-005`: **Turn state comes from the transcript tail.** Parsers
  must surface the records needed to derive it: tool results paired to their
  tool_use id, harness stop/end-of-turn markers, and session-end markers.
  A named harness stop field is recorded verbatim. A harness that has no
  named stop field still surfaces a marker when its own protocol
  distinguishes mid-turn from end-of-turn. Derivation is generic over
  `Message`/`ToolCall`: an assistant tool_use with no paired result →
  `working` with `running_tool`; a last stop with `ends_turn=False` and no
  later user record → `working` (the harness left the turn open, even after
  the tool result landed); a subagent message stream (`agent_id`) with no
  terminal marker → `subagents_running`; a final assistant message with an
  end-of-turn stop reason and no later user record → `awaiting_input`; a
  session-end marker → `ended`. A harness that emits none of these resolves
  to `unknown`, never to a guess. Codex `task_started` keeps subsequent
  main-thread assistant records mid-turn until `task_complete` or `turn_aborted`.
  A later lifecycle event at the same message position replaces that marker,
  including repeated positions across input batches;
  re-presenting unchanged marker facts remains a no-op. Turn state is computed
  at read time from the tail; it is not stored.
- `REQ-LIVE-006`: **Tool results are indexed in their own table.** A new
  `tool_results(tool_call_id PRIMARY KEY, result_summary TEXT, is_error BOOLEAN,
  completed_at TIMESTAMP)` table holds the paired result (summary truncated,
  default 1 KB). It is insert-only: a result arriving in a later incremental
  chunk than its tool_use inserts a row, it never updates `tool_calls`.
  Rationale: DuckDB UPDATE is DELETE+INSERT, and `tool_calls` is the row set
  whose churn produced the 140 GiB `tool_call_embeddings` bloat
  (`REQ-INDEX-017`); widening it with mutable columns would reopen that path.
  This is the one schema addition (additive migration); a tool_call with no
  `tool_results` row derives `unknown`, not `working`, when the session is
  not `active`.
- `REQ-LIVE-007`: **Fleet live view.** `recall live --fleet` fans out per
  `REQ-FLEET-CMD-*`; each row carries `host`. A single fan-out is the unit;
  `--follow --fleet` is rejected in v1 (one SSH per poll is the wrong
  transport; ControlMaster is the "later" already named in the fleet section).
- `REQ-LIVE-008`: **The writer pid is enrichment, supplied by the harness
  plugin, never inferred from process trees.** recall accepts an optional
  sidecar record per session `(source, source_session_id, host, pid)` written
  by a hook (`recall live mark --session <id> --pid $PPID` from the harness
  `SessionStart` hook). When present it feeds `ended` detection
  (`kill(pid,0)` on the same host) and appears on the row as `writer_pid`.
  A terminal's own session keys are not recorded: a join to a particular
  terminal belongs to that terminal, keyed on `(host, source, source_session_id)`.
  When absent everything else in this section still works.
- `REQ-LIVE-009`: **Read-only and lock-safe.** `live`, `show --tail/--after`,
  and `--follow` use read-only connections per `REQ-CONC-001`. `--fresh` is
  the only write path and goes through the daemon RPC, never a CLI-held
  write lock.
- `REQ-LIVE-010`: `recall daemon status` exposes `live_session_count` (already
  `REQ-DAEMON-050`) and additionally `live_fresh_requests`, `live_fresh_timeouts`
  so a driver can see whether `--fresh` is being served.

- `REQ-LIVE-011`: **One daemon primitive serves `--fresh` and `--follow`.**
  The daemon exposes `index_session_now(path)` which bypasses the debounce
  queue for that one path, runs the existing `index_single_session` under the
  write lock for exactly one incremental parse, and then fires a per-session
  "indexed" event carrying the new high-water `idx`. The drain loop fires the
  same event after every ordinary watch write. `--fresh` awaits one firing
  with a timeout; `--follow` subscribes until its deadline. A follow
  subscription holds no DB connection and no write lock while waiting; client
  disconnect cancels the subscription (the single-user shared-connection rule
  of REQ-RESIL-021 applies).
- `REQ-LIVE-012`: **Discovery roots are per-source config.** Claude Code's
  root is hard-coded to `~/.claude/projects`; agents launched with
  `CLAUDE_CONFIG_DIR` pointing elsewhere are invisible unless the operator
  symlinks `projects` (the arrangement on the reference host). `[sources.claude_code]
  roots = [...]` (and the equivalent per source) feeds both `discover` and
  `live_candidates`, so a live session is watched wherever the harness writes.

### Refactors this depends on (prefactoring)

1. Expose the daemon `LiveSessionSet` over RPC (`live_sessions`: path, mtime,
   last_event_at) and join paths to `session_state.source_path`; today it is
   reachable only as a capped path list in status.
2. Add `file_mtime`, `file_size`, `indexed_at` to the `list`/`show` field
   manifests so freshness is projectable before `live` exists.
3. Parser protocol: `ParseResult` gains `tail_facts` (tool results keyed by
   tool_use id, stop reasons, session-end markers); each of the five parsers
   declares which markers it emits, `unknown` otherwise. Pairing across
   incremental chunks happens at write time against stored `tool_calls`.
4. `services/sessions.py`: split head (`--message-limit`) from tail/cursor
   reads; add a tail query (last K by `idx`, plus open tool_calls with no
   `tool_results` row) so turn state never scans the session.
5. RPC client (`core/rpc_client.py`): generalize the `progress`-only
   notification hook to `on_notification(method, params)` and make the socket
   timeout an idle timeout for streaming methods; today `READ_TIMEOUT` (30 s)
   would kill any `--follow` longer than one quiet period.
6. RPC server: a streaming-handler pattern (handler that emits notifications
   and returns a final result) with cancellation on client disconnect, built
   on the existing `ClientConnection.send_notification`.
7. Drain loop (`rpc_server._drain_loop`): after each session write, fire the
   per-session indexed event of `REQ-LIVE-011`; add `queue.flush(path)` to
   `DebouncedIndexQueue` for the `--fresh` bypass.
8. `list_sessions` ordering: add `last_activity_at` (max of `ended_at` and
   `file_mtime`) so `live` and `list` can sort by it without a second query.
9. Manifest and contract: register `live` and the new `show` flags in
   `cli/manifest.py` so `recall --llms` / `recall schema` and
   `tests/test_cli/test_agent_contract.py` cover them; add `fleet_live` beside
   `fleet_list`/`fleet_search`/`fleet_show`.
10. Fixtures: three cut Claude Code transcripts (mid-tool, post-`end_turn`,
    subagent stream) plus one per other harness where the marker exists;
    `tests/test_parsers_live_candidates.py` is the seed for live-set tests.

### Non-goals

- Attention states (`blocked`, `done`, `stale`) — the session multiplexer owns them.
- Steering, PTY input, scrollback, viewport — the session multiplexer owns them.
- Summarizing "what the agent is doing" with an LLM. `TurnState` is
  structural; a driver reads `last_assistant_text` itself.
- A push channel from recall to the session multiplexer or its command center.
- Terminal-specific surface keys on rows, marks, or filters. A terminal joins on
  `(host, source, source_session_id)` itself.
- Sub-second freshness. The floor is the watch debounce (5 s) plus queueing;
  `--fresh` bounds it, it does not eliminate it.

### Acceptance

- `recall live --json` on a host with one running Claude Code session and one
  finished session returns exactly one `active` row whose `freshness.current`
  is true within one debounce window after the agent writes. A session's
  *first* appearance waits for the daemon's discovery loop
  (`live_discovery_interval`, 30 s) rather than the 5 s debounce, because
  nothing subscribes to a transcript before it is discovered; the debounce
  bounds updates to a session already in the live set.
- `recall show <id> --tail 3 --json` returns the last three messages by `idx`;
  `--after <cursor>` twice in a row returns a delta then an empty delta.
- A session whose last record is an unpaired tool_use reports
  `turn.state = working` with `running_tool.name`; after the result lands it
  reports `awaiting_input` when the assistant ended the turn.
- `--fresh` against a stopped daemon exits nonzero with a stable code.
- Fixture-driven: `tests/fixtures/` gains a live Claude Code transcript cut
  mid-tool, one cut after end_turn, and one with subagent records.

### Test traceability

| REQ | Tests |
|-----|-------|
| REQ-LIVE-001 | `tests/test_services/test_live_derivations.py`, `tests/test_services/test_live_sessions.py` |
| REQ-LIVE-002 | `tests/test_services/test_live_view.py`, `tests/test_cli/test_live.py` |
| REQ-LIVE-003 | `tests/test_services/test_fresh_reads.py`, `tests/test_cli/test_show_fresh.py`, `tests/test_core/test_config.py` (`fresh_timeout`) |
| REQ-LIVE-004 | `tests/test_services/test_show_tail.py`, `tests/test_cli/test_show_tail.py`, `tests/test_services/test_show_follow.py`, `tests/test_cli/test_show_follow.py` |
| REQ-LIVE-005 | `tests/test_services/test_live_derivations.py` (turn state), `tests/test_parsers/test_tail_facts.py` |
| REQ-LIVE-006 | `tests/test_db/test_migrations.py` (0024), `tests/test_services/test_indexer.py` (tool results) |
| REQ-LIVE-007 | `tests/test_services/test_fleet_query.py` (`merge_live_rows`, `fleet_live`), `tests/test_cli/test_live.py` (`--fleet`) |
| REQ-LIVE-008 | `tests/test_services/test_live_marks.py`, `tests/test_services/test_live_mark_rpc.py`, `tests/test_cli/test_live_mark.py` |
| REQ-LIVE-009 | `tests/test_cli/test_cli_import_lint.py`, `tests/test_services/test_live_view.py` (read-only connection) |
| REQ-LIVE-010 | `tests/test_services/test_fresh_reads.py` (counters), `tests/test_cli/test_daemon_cli.py` |
| REQ-LIVE-011 | `tests/test_services/test_index_session_now.py`, `tests/test_services/test_live_events.py` |
| REQ-LIVE-012 | `tests/test_parsers/test_source_roots.py`, `tests/test_services/test_catch_up_scope.py`, `tests/test_services/test_index_scoping.py` |

### Risks

- **[RISK-HIGH — public CLI]** `recall live`, `show --tail/--after/--follow`
  are new agent-facing contracts; field names above are the proposal.
- **[RISK-MEDIUM — schema]** three new tables (`REQ-LIVE-006`,
  `REQ-LIVE-008`); the migration is additive and `tool_calls` is untouched,
  because widening it would reopen the `REQ-INDEX-017` churn path.
- **[RISK-LOW]** `--fresh` adds a synchronous path on the daemon; it must not
  take the write lock for longer than one session's incremental parse.

## Cross-Harness Skill Census

### Problem and solution

Skill archival decisions need a fleet-wide control population, but typed skill
calls exist only in some harnesses and historical rows may predate parser-side
attribution. A zero from one transcript format is therefore not evidence of
non-use. Recall owns the normalized session/tool data and exposes one census
that attributes every supported harness, derives historical NULL rows at query
time, and refuses to produce an authoritative result when fleet coverage is
incomplete.

### Domain model

- A census row is `(skill_name, source, host, invocations, sessions)`. `host`
  identifies the queried endpoint that observed the invocation, not an imported
  session's stored origin host; fleet merge replaces a remote endpoint's local
  name with its inventory name.
- A coverage object names its `scope`, `expected_hosts`, `successful_hosts`,
  `covered_sources`, windowed `considered_sessions`, `attributed_invocations`,
  and `unattributed_candidates`, plus the same population totals under
  `control` without the requested time bound.
- An unattributed candidate is a typed `Skill` call or a tool call whose stored
  command/input mentions `SKILL.md` but does not satisfy parser attribution.
  Candidates are diagnostic coverage debt, never invocations.

The local query reads candidate tool rows only; it does not scan transcript
files. Its work is linear in stored candidate rows plus one grouped session
count, bounded by the caller's source/time filters. Because an all-time census
can legitimately inspect tens of thousands of sessions, the command has a
10-minute local RPC allowance and fleet fan-out has an 11-minute per-host
allowance. Other RPC and fleet commands retain their shorter generic bounds.

### Requirements

- `REQ-SKILL-001`: `recall stats skills` MUST return rows grouped by
  `(skill_name, source, host)` with invocation count and distinct-session count.
  Supported sources are Claude Code, Codex, Pi Agent, Grok, and Kimi Code.
- `REQ-SKILL-002`: Attribution MUST apply parser SPEC `REQ-PARSE-017` to every
  stored candidate call at query time, with the session cwd, so a corrected rule
  reaches history without a reparse. The persisted `skill_name` holds one name
  from whichever rules indexed the row and is not the census source. A call that
  loaded several skills counts one invocation per skill. Shipping this command
  MUST require neither a data migration nor a full reindex.
- `REQ-SKILL-003`: Structured output MUST be an object with `rows` and
  `coverage`. Coverage MUST contain scope, expected and successful hosts,
  covered sources, considered sessions, attributed invocations, unattributed
  candidates, and an all-time `control` population unaffected by `--since`.
- `REQ-SKILL-004` **[RISK-HIGH — public CLI/RPC]**: The command accepts the
  standard output controls plus `--since`, repeatable `--source`, `--local`, and
  `--fleet-config`. `--source` uses the normal source aliases and filters both
  rows and coverage. `--local` explicitly limits the query to the current
  daemon. Without `--local`, scope is the current daemon plus every host in the
  configured fleet inventory.
- `REQ-SKILL-005`: The default local-plus-fleet census MUST fail closed before
  emitting data when inventory is missing/empty, any configured host is
  unreachable or times out, a remote CLI lacks the command, a remote query
  fails, its payload is invalid, or its covered sources do not equal the
  request. This command deliberately overrides the partial-success rule in
  `REQ-FLEET-SSH-005`: a partial census cannot support a never-fired claim.
- `REQ-SKILL-006` **[RISK-HIGH — public RPC/agent contract]**: The local query is
  exposed as `recall.stats_skills`; the CLI manifest, parameter validation,
  output projection, Recall skill documentation, and CLI reference MUST expose
  the same options and object shape.
- `REQ-SKILL-007`: Fleet merge MUST stamp remote rows with the inventory host,
  merge identical row keys by summing invocation/session counts, preserve the
  local endpoint host, and report the exact expected/successful endpoint set.
- `REQ-SKILL-008`: Filters use the same duration grammar and session timestamp
  fallback as other stats surfaces. The all-time control population keeps the
  source filter but ignores only `--since`, so a recent empty window is
  distinguishable from a broken or empty detector.
- `REQ-SKILL-009`: The skill census MUST remain bounded while allowing a
  fleet-scale historical query to complete: local RPC waits up to 600 seconds
  and each remote endpoint waits up to 660 seconds. These allowances are
  specific to `stats skills` and MUST NOT lengthen the default bounds of other
  RPC or fleet commands.
- `REQ-SKILL-010`: The census MUST count each transcript once. A transcript
  whose project or cwd directory was renamed is indexed again under its new
  path; rows of one source and host whose path below the project or cwd
  directory (below `projects/<dir>/` or `sessions/<dir>/`) is equal are one
  transcript. The row whose catalog file is present wins, then the latest to
  run. Considered sessions and invocations both use the surviving rows. A
  Codex code-mode program counts once per session, however many of its inner
  calls carry it, and each skill once per program, however many of its
  commands (including every path a statically resolved loop template expands
  to) load it. A command literal equal to a command an inner call resolved
  counts only as that call's own load.

### Invariants

- Incomplete host or source coverage never emits census rows that a caller can
  mistake for an authoritative never-fired result.
- Persisted and query-time attribution share one pure implementation.
- Discovery, editing, and quoted pseudo-calls never increment an invocation
  count. A plain read of a repository's own project skill is a load.
- The command is read-only and does not mutate the derived database.

### Non-goals

- Declaring, renaming, or archiving skills.
- Inferring skill use from prose, descriptions, or assistant claims.
- Migrating historical rows solely to populate `skill_name`.
- Making other fleet query verbs fail on partial coverage.

### Decisions

1. Skill census defaults to local plus all configured remotes; `--local` is the
   explicit opt-out (ratified 2026-08-30).
2. Partial fleet results are invalid for archival decisions even though they
   remain useful for other fleet views (ratified 2026-08-30).
3. Historical fallback is computed at query time; future parses persist it
   without requiring a fleet-wide reindex (ratified 2026-08-30).
4. A read of a repository's own `.agents|.claude|.codex/skills` project skill
   counts as a load even inside the session cwd, and Codex code-mode programs
   are attributed from read-shaped command literals (ratified 2026-09-24),
   including templates that resolve statically over a const path or a loop
   over a literal path array; anything needing a runtime stays unattributed (provisional 2026-09-25).
5. The census counts each transcript once and re-derives attribution for every
   candidate at query time (ratified 2026-09-24).

### Risk tags

- **[RISK-HIGH]** New public CLI, RPC, manifest, and default fleet fan-out —
  REQ-SKILL-004/006.
- **[RISK-MEDIUM]** False positive attribution could archive a live skill;
  negative authoring and pseudo-call cases are release gates.
- **[RISK-MEDIUM]** False complete coverage could turn missing telemetry into a
  never-fired claim; all host and source failures are fail-closed.

### Acceptance criteria

- [ ] All five harnesses produce attributed rows through their real invocation shapes.
- [ ] Historical NULL rows produce the same result without migration or reindex.
- [ ] Source and time filters affect rows/window coverage while control ignores only time.
- [ ] Discovery, editing, quoted pseudo-calls, and Grok working-copy reads stay unattributed.
- [ ] Missing inventory and every remote failure class exit nonzero without data.
- [ ] Local-plus-fleet output names every expected/successful host and all covered sources.
- [ ] CLI schema and Recall's agent docs expose the complete contract.
- [ ] Fleet-scale historical queries use census-specific bounded timeouts without changing other command defaults.

### Test traceability

| REQ | Tests |
|-----|-------|
| REQ-SKILL-001/002/003/008 | `tests/test_services/test_skill_stats.py` |
| REQ-SKILL-004/006 | `tests/test_cli/test_skill_stats.py`, `tests/test_cli/test_agent_contract.py`, `tests/test_services/test_rpc_server.py` |
| REQ-SKILL-005/007 | `tests/test_services/test_fleet_skills.py` |
| REQ-SKILL-009 | `tests/test_cli/test_skill_stats.py`, `tests/test_services/test_fleet_skills.py` |
| REQ-PARSE-017 | `tests/test_parsers/test_common.py`, `tests/test_parsers/test_grok.py`, `tests/test_parsers/test_kimi_code.py` |

## File Locations

| File | Location |
|------|----------|
| Database | `~/.local/share/recall/recall.duckdb` |
| Lock file | `~/.local/share/recall/recall.lock` |
| Config | `~/.config/recall/config.toml` (optional) |
| Fleet inventory | `~/.config/recall/fleet.toml` (optional; REQ-FLEET-CFG) |
| Daemon logs | `~/.local/share/recall/logs/daemon.{log,err.log}` |

## Implementation Notes

Learnings captured during v1 implementation:

### DuckDB Constraints

1. **No FK constraints used**: The schema does not use foreign key constraints (see `REQ-SCHEMA-003`). Referential integrity is guaranteed by construction via deterministic SHA256 IDs and atomic session writes.

2. **FTS extension autoloads**: The FTS extension is automatically loaded when `PRAGMA create_fts_index` is called. No need for explicit `INSTALL fts; LOAD fts;` in most cases.

3. **FTS creates schema**: Creating an FTS index generates a `fts_main_<table>` schema with the `match_bm25()` macro for queries.

### uv Workspace Setup

For monorepo with `uv` workspaces:
- Dev dependencies go in root `pyproject.toml` under `[dependency-groups]`
- Workspace packages need `[tool.uv.sources]` mapping: `recall = { workspace = true }`
- Set `default-groups = ["dev"]` in `[tool.uv]` for automatic dev dep installation
- Package `readme` paths cannot reference files outside the package directory

### Testing

- Fixtures should be accessible from test files via relative paths
- Use `Path(__file__).resolve().parents[N]` carefully - verify the correct ancestor level
- DuckDB in-memory databases work well for isolated test runs

## Decisions

- 2026-09-08 — The watch debounce having no max-wait cap — a session written to steadily is never indexed while it stays busy, measured at 96 s over 26 turns — is recorded for a follow-up campaign, not fixed here. It is index lag, not liveness: round 5 confirmed the row reads `active` and reports `freshness.current: false` honestly throughout, and `--fresh` (REQ-LIVE-010/011) exists for exactly this. Capping the debounce changes the daemon write cadence on the hot path, which is the overcorrection three rounds were spent ruling out; doing it at 22/24 with no margin for its own bug bash is the expensive option. CLI_REFERENCE now states the real duration instead of `a few seconds`. **provisional (driver)**
- 2026-09-08 — LiveMember.last_event_at is float | None, where None means no fsevent has ever fired for that member — it is NOT `an event just now`. A new member promoted from an mtime alone carries no event evidence, so demote_idle lets its mtime decide alone; only a real fsevent stamps the monotonic clock. seed() therefore promotes with bump_event=False: seeding is an inventory of what is on disk, not an observation of activity. Rejected deriving the initial stamp from the file age (monotonic_now - (wall_now - mtime)): it converts between the two clocks REQ-DAEMON-046 keeps independent, and needs wall_now threaded into promote/seed. last_event_at is internal — no CLI or RPC reader — so the Optional does not reach the output shape. **provisional (driver)**
- 2026-09-08 — `live --source` dropping a live-but-unindexed row is documented, not fixed: everything past path and mtime comes from the index, and giving LiveMember a source means threading the discovering parser through promote/seed/the event handler. The window lasts until the row's first index pass, which for a steadily-written session is as long as it stays busy (measured at 96 s); an unfiltered `recall live` always shows the row. **provisional (driver)**
- 2026-09-08 — The discovery sweep enumerates with max(live_idle_threshold, 2 * live_discovery_interval) and promotes with live_idle_threshold. The two windows had been one, so a config that set the threshold below the discovery interval — which config.py accepts, validating only <= 0 — silently and permanently dropped any transcript that quiesced inside the gap. Rejected cross-field config validation: refusing the configuration removes a legitimate operator choice and still leaves the invariant stated in prose rather than held by the sweep. At the shipped defaults (300 s over 30 s) the max collapses to the threshold and nothing changes. **provisional (driver)**
- 2026-09-08 — A resumed session reading `idle` for up to one discovery interval is documented, not fixed. Liveness comes from live-set membership; making it mtime-derived for already-indexed rows is a change to REQ-LIVE-001's derivation, and the observed defect is Minor, bounded by one tick, and self-healing. **provisional (driver)**
- 2026-09-07 — Tool results live in an insert-only tool_results table, never as columns on tool_calls (DuckDB UPDATE is DELETE+INSERT; REQ-INDEX-017 churn). **ratified (human)**
- 2026-09-07 — One daemon primitive index_session_now + a per-session indexed event serves both --fresh and --follow. **ratified (human)**
- 2026-09-07 — live_marks ships in migration 0024 with tool_results so there is one schema bump. **ratified (human)**
- 2026-09-07 — --follow emits NDJSON regardless of --format and is rejected with --fleet. **ratified (human)**
- 2026-09-07 — --follow is in v1. **ratified (human)**
- 2026-09-07 — The harness tool_use id lives in an insert-only side table tool_use_ids(tool_call_id PK, session_id, tool_use_id) in migration 0024 — NOT as a column on tool_calls, which would make a full re-parse UPDATE every existing row (NULL vs parsed id) and reopen the REQ-INDEX-017 churn path. **provisional (driver)**
- 2026-09-07 — Claude Code end-of-turn stop reasons are end_turn and stop_sequence; tool_use means mid-turn. **provisional (driver)**
- 2026-09-08 — live_path_key is str(path.expanduser().resolve()) — byte-identical to what parsers write into sessions.source_path — and is the one spelling for both the event channel key and the index lookup. **provisional (driver)**
- 2026-09-08 — Configured `[sources.<source>] roots` REPLACE the parser's built-in location rather than extend it: an operator who moved a harness's home stops paying to scan the old one. A configured root that does not exist is skipped, not raised on, so one config can list the lanes a fleet host *may* have. **provisional (driver)**
- 2026-09-08 — Configured `[sources.<source>] roots = []` means the host scans nothing for that source; an ABSENT section means the built-in root. `SourceConfig.roots` and the parser field are `tuple | None` so the two are distinguishable. Found by dogfooding: without it a scratch config swept the operator's whole corpus for every source it did not name. **provisional (driver)**
- 2026-09-08 — Harness stop markers persist in `session_stop_markers(session_id, message_idx, reason, ends_turn)`, added to the unshipped migration 0024 rather than a new 0025. `ends_turn` is the PARSER's reading of its own vocabulary, so the read-time derivation stays generic; `reason` passes through unnormalized as the SPEC requires. **provisional (driver)**
- 2026-09-08 — `subagents_running` is derived from OPEN TOOL CALLS carrying an `agent_id`, not from messages carrying one: a finished subagent's messages stay in the transcript forever, so their presence cannot distinguish running from finished. The parent's own open `Agent` call alone is `working`. **provisional (driver)**
- 2026-09-08 — `session_ended` is NOT persisted in U11: no parser on any of the five harnesses emits a session-end marker (censused in iteration 2), so a column for it would have no writer. `derive_liveness`/`derive_turn_state` already take it as a parameter, and `ended` comes from the REQ-LIVE-008 pid stamp that U16 lands. **provisional (driver)**
- 2026-09-08 — `recall.live` returns `{watching, sessions}` and the CLI emits the array plus a stderr note when `watching` is false. The array keeps `list`'s output contract while the caveat lands where warnings already go. **provisional (driver)**
- 2026-09-08 — A `--fresh` index that runs out of budget is SHIELDED, not cancelled. `_await_shared_conn_work` only honours cancellation once the executor finishes (REQ-RESIL-021), so `wait_for` without a shield would extend the wait instead of bounding it; abandoning the write lock mid-statement is the interleaving REQ-RESIL-021 rules out. The answer returns on time and the index completes on its own. **provisional (driver)**
- 2026-09-08 — `recall show` reports `freshness` on EVERY read, not only `--fresh` ones (REQ-LIVE-002): a caller cannot tell a finished transcript from a lagging one by reading its messages. `resolve_session_path` shares `load_session`'s identifier resolution so the refresh and the read can never disagree about which session was asked for. **provisional (driver)**
- 2026-09-08 — `show` returns a `cursor` on EVERY read, and an empty `--after` delta hands back the cursor it was given rather than a lower one — nothing arrived, so the caller has still seen everything it had. Cursor idx defaults to -1 for a session with no messages, matching what `live_view` already encodes. **provisional (driver)**
- 2026-09-08 — `--tail`/`--after` are REFUSED alongside `--message-limit` (head-anchored vs tail-anchored) and alongside `--fleet` (`run_fleet_show` cannot carry a window). Silently dropping a bound is how a monitor loop that asked for 20 messages gets 10 MB. **provisional (driver)**
- 2026-09-08 — `default_watch_roots` and `default_discover` RESOLVE every path they return. The watcher hands these to the filesystem observer and the kernel reports events under the real path, so a root reached through a symlink was watched under a spelling the kernel never uses — discovered, promoted into the live set, then never indexed again while `daemon status` reported it live. Now the same canonical spelling the parsers write into `sessions.source_path` and `live_path_key` reads back. **provisional (driver)**
- 2026-09-08 — The close reasons shipped are `timeout` and `daemon_stopped` only. `ended` waits for U16: no parser emits a session-end marker, so REQ-LIVE-008's pid stamp is its only writer, and an enum value no code can produce is the same defect as a permanently-null column. **provisional (driver)**
- 2026-09-08 — `merge_live_rows` orders by liveness rank (active, idle, unknown, ended) and then by most recent activity, comparing EPOCH SECONDS rather than datetimes. Rows arrive from several hosts and one may stamp an aware value while another stamps a naive one; comparing those raises. A row with no stamp at all is still live, so it sorts last within its rank instead of being dropped. **provisional (driver)**
- 2026-09-08 — `live --fresh --fleet` is REFUSED. `--fresh` is not forwarded over the hop for the same reason `--host` is not (an edge on an older build fails the whole host on an unknown flag), so answering would hand back rows that are stale by construction under a flag promising the opposite. **provisional (driver)**
- 2026-09-08 — `live --fleet` prints NO `not running a live watcher` note. Only the row array crosses the SSH hop -- each remote writes its own caveat to its own stderr -- so no host's watch state is knowable on the control host, and claiming the local daemon's would name the wrong machine. **provisional (driver)**
- 2026-09-08 — `recall live mark` FAILS OPEN: a daemon it cannot reach yields `{marked: false, reason}` on stdout, the reason on stderr, and exit 0. It also passes `auto_fork=False`. The command runs inside a harness SessionStart hook, so a non-zero exit or a cold daemon fork would put recall's failure -- or an index pass -- inside the agent's own session start. A refusal the caller can fix (missing session, non-positive pid) is still exit 2; only the daemon's absence is survived. **provisional (driver)**
- 2026-09-08 — `live_view` takes `local_host` as a REQUIRED keyword and probes a marked pid only when the row's own host matches it. A pid number means nothing off the machine that stamped it, so probing a foreign mark would report a running agent as `ended` -- the exact false answer REQ-LIVE-001 forbids. The SQL join already pairs `live_marks.host` with `session_state.host`; the parameter is the second half, for a database copied between machines. **provisional (driver)**
- 2026-09-25 — Surface keys are removed before the first public release: `live --surface`, `live mark --surface`, the row's `surface` object and the mark's `surface_key` field. A key only one terminal understands does not belong in a harness-agnostic roster. The row carries `writer_pid` instead. The nullable `live_marks.surface_key` column is left in place, unwritten and unread, so no migration ships for it; the next schema migration drops it. **ratified (human)**
- 2026-09-08 — `live mark --source` is forwarded ONLY when given; the daemon owns the `claude-code` default. One default in one place, and the manifest documents it. Both `--session` and `--pid` are required. **provisional (driver)**
- 2026-09-08 — A pid the kernel recycled onto an unrelated process reads as alive, and nothing cheap distinguishes it. Accepted: the failure direction is safe -- the session stays `idle` rather than being wrongly declared `ended`. `kill(pid, 0)` treats EPERM as alive (a harness running as another user can still own the pid) and refuses non-positive pids, which would otherwise signal the caller's own process group. **provisional (driver)**
- 2026-09-08 — `tail_facts` is a REQUIRED keyword on `_write_session`, `_incremental_write_session` and the three helpers they delegate to. Its `= _NO_TAIL_FACTS` default let all four WATCH call sites omit it silently, so the daemon -- the only writer a live session ever goes through -- persisted no `tool_results` and no `session_stop_markers`: every finished turn read `working` on a tool that had already returned, and `awaiting_input` was unreachable. Same shape as the `_discover_paths(source, *, sources)` fix, and the same remedy: make the next omission a type error. Note ty does NOT catch a `**kwargs` forwarder (one test helper still failed at runtime after ty was clean). **provisional (driver)**
- 2026-09-08 — `derive_liveness` now ranks a marked pid probed DEAD above live-set membership; the previous order is reversed. `kill(pid, 0)` is re-observed on every read, while membership is an inference from a write up to `live_idle_threshold` (300 s) old -- and the bug bash measured a SIGKILLed agent reading `active` for 5.5 minutes. The stale-mark case the old order protected is covered by the hook re-marking at SessionStart before the resumed harness writes. `watched` still outranks a harness end MARKER, which is a record rather than a fresh observation. **provisional (driver)**
- 2026-09-08 — `live --fresh` READS FIRST, then refreshes eligible rows of that answer, then re-reads. The previous scope (the watched set only, on the reasoning that an idle row `has nothing pending by definition`) was wrong: a session being written to right now but not yet promoted -- discovery runs every 30 s -- is unwatched and stale, which is exactly the row a `--fresh` caller is chasing. Reading first makes the scope exact rather than assumed, and a fully current listing now spends nothing at all. Eligible catch-up is later narrowed to already-indexed rows (2026-09-09). **provisional (driver)**
- 2026-09-08 — `--fresh` bounds INDEX lag, not DISCOVERY lag, and both SPEC's acceptance line and CLI_REFERENCE now say so. A transcript's first appearance in `recall live` waits for the daemon's 30 s discovery loop, not the 5 s watch debounce; the debounce governs updates to a session already in the live set. The bug bash measured 14-20 s to first appearance and reasonably read the old wording as a missed contract. **provisional (driver)**
- 2026-09-09 — `live --fresh` is a roster catch-up, not a first-index. It refreshes only already-indexed behind rows (`committed_generation > 0`) and is bounded by the responsive-read floor (2 s). Never-indexed rows stay `current: false` with limitation `not_yet_indexed`; first index remains the fair coordinator's after resume and inventory. `show --fresh` remains the one already-known session write, bounded by `live.fresh_timeout` (10 s). Do not lengthen the operator RPC deadline or `live.fresh_timeout` to hide first-index cost. This supersedes treating `live --fresh` as indexing every behind row on the page. **ratified (human)**
- 2026-09-11 — Pi Agent `stopReason` (`toolUse` mid-turn; `stop`/`aborted`/`error` end the turn) is recorded as stop markers; a text-only assistant without the field is `stop`. Grok has no named stop field: the parser reads assistant `tool_calls` as mid-turn and a text-only assistant as the end of the turn. A last stop with `ends_turn=False` and no later user record is `working` even after the tool result has landed — otherwise a Pi/Grok tool loop reads `unknown` for almost the entire turn. This does not guess in the absence of markers. **provisional (driver)**
- 2026-09-11 — Codex `event_msg` `turn_aborted` is an end-of-turn stop (`reason` passed through). Completed activity items `Extension` and `CollabAgentToolCall` are skipped like `FileChange`: they are UI/collab mirrors; `response_item` remains the tool record. A user-only synthetic transcript with no agent turn still resolves `unknown`. **provisional (driver)**
- 2026-09-08 — Discovery promotion ENQUEUES the promoted path for indexing, but only on the transition into the live set. A file written entirely between two discovery ticks has no observer subscription while it is being written, so no event ever fires for it; once it quiesces there is no future write to ride on, and its bytes were never indexed at all. Promotion is the one moment that observes the file exists and is behind. Marking on every re-sweep instead would re-index the entire live set every 30 s, which is why the transition -- not membership -- is the trigger. **provisional (driver)**
- 2026-09-15 — Architecture and isolated local implementation are approved by the human; prior release-specific exceptions do not apply. **ratified (human)**
- 2026-09-15 — First attempt preserves logical schema and engine and moves preparation outside critical sections; a missed floor requiring staged storage needs a separate human proposal. **ratified (human)**
- 2026-09-19 — Rollup: a guarded set-based rollup of every sid on every tick replaces the seed's skip-plus-touched-sids plan. Measured 0.03 CPU-s per no-op tick; it heals cleared token columns with no new state. **provisional (driver)**
- 2026-09-19 — Walk cadence: min(daemon.interval, 30 s), measured from the end of the walk. Moving to daemon.interval (300 s) would push old-mtime import discovery past the BRIEF 45 s floor. Keep the 30 s safety walk unless another design proves the 45 s floor. **ratified (human)**
- 2026-09-19 — Walk filter: each root's present catalog rows are read once per walk (27k Codex rows = 17 MiB transient, 54 ms CPU) instead of a per-batch lookup (2-3 CPU-s/walk; parameterized IN is a per-row OR chain). Unchanged sources never reach the writer. **provisional (driver)**
- 2026-09-19 — Threads: unchanged. On a large append commit, SET threads=4 cut CPU 22% (4.21 -> 3.29 CPU-s) but raised wall time 23% (1.45 -> 1.78 s), and threads is engine-wide, so vector/keyword search would pay too. **provisional (driver)**
- 2026-09-19 — Raw commits: no change here. They dominate on an active host, but the cost is structural: prepare_raw_sources re-parses each source from offset 0 and the commit re-reads and diffs the whole persisted session, so cost scales with session size. An incremental append path needs per-adapter resumable normalization under the REQ-INDEX-010..016 prefix contracts. Proposed as a follow-up campaign, not a simple removal of work. **provisional (driver)**
- 2026-09-19 — DuckDB 1.5.5 probes `import pandas` twice per bound non-NULL parameter and never caches the failure (pandas is absent from the daemon env): 25,310 failed imports, ~1 CPU-s, in one 2.9k-message commit. A sys.modules None marker is unsafe (DuckDB reads sys.modules['pandas'].__version__ and a write transaction failed); a bound Arrow ID table is 60x slower on these PK lookups; validated literal IDs save ~0.3 CPU-s per large commit. Not shipped; follow-up with options. **provisional (driver)**
- 2026-09-19 — Ship validated literal ID sets only on the measured raw-commit primary-key paths: IDs matching `[A-Za-z0-9_.:+@-]{1,128}` render as SQL literals and any other value keeps parameter binding. Three fresh-process 2.9k-message samples reduced failed pandas probes from 35,264 to 90 and median total CPU from 1.0151s to 0.4055s; injection-shaped and historical IDs retain exact behavior. Retire the workaround when the pinned DuckDB caches unavailable optional imports. **provisional (driver)**
- 2026-09-19 — reconciliation_roots scan-record upserts stay: they are two small real state changes per root per walk. **provisional (driver)**
- 2026-09-19 — Idle raw drain (U6): after a turn that finds nothing eligible, the poll loop waits for an edge (source activity, a catalogued source, a released claim, a request left to the drain, a resume, lifted write pressure) or a 5 s rescan, a constant rather than a config key. The rescan bounds the delay for work that becomes eligible with time alone (retry backoff) and keeps the keyword repair on its debounce. Worked turns are unchanged. **provisional (driver)**
- 2026-09-19 — Schema removal is one separate version-30 migration; it does not authorize row deletion. **ratified (human)**
- 2026-09-19 — The exact orphan manifest is a destructive boundary: produce and review it, then stop for explicit approval before deleting any row. **ratified (human)**
- 2026-09-19 — Migration 0030 drops every secondary source_files index before DROP COLUMN because DuckDB rejects the DDL otherwise, then recreates the retained source/path index in the same transaction. Injected failure proves rollback restores the version-29 shape. **ratified (human)**
- 2026-09-19 — Compaction is storage-format neutral: it retains the persisted source compatibility tag, while only the separately backed-up migrate-storage workflow may advance that tag. **ratified (human)**
- 2026-09-19 — A resumable checkpoint is trusted only when it was committed atomically with the catalog acknowledgement, carries the matching parser revision, and its complete-prefix digest verifies; every failed precondition takes the full reference path. **ratified (human)**
- 2026-09-19 — An adapter may decline resume at an open normalization boundary; this is a correctness fallback, not permission to skip that adapter from append and fallback coverage. **ratified (human)**

## Daemon reconciliation contract

The daemon must reconcile filesystem discovery, live session state, indexing, and public status views while historical work, enrichment, and continuous appends proceed concurrently. The domain consists of discovered transcript paths, bounded capture/index work, session rows and cursors, live notifications, and RPC/CLI status projections.

### Requirements

- **REQ-RECON-001:** Every discovered configured source has a durable accounted state and complete scans report scope and unresolved work.
- **REQ-RECON-002:** Missed events, old-mtime imports, retries, and subscription overflow converge through periodic reconciliation.
- **REQ-RECON-003:** Active/newer work receives preference without starving pending sessions. This fairness and writer-occupancy contract applies to normal operation, including ordinary historical discovery, genuine rewrites, and checkpoints against a populated database. It is not the versioned historical index-migration contract (REQ-RECON-010..016).
- **REQ-RECON-004:** Checkpoints acknowledge complete committed records and validated prefixes; rewrites cannot masquerade as appends.
- **REQ-RECON-005:** Parser incompleteness and unsupported content remain visible; supported current Codex records are indexed.
- **REQ-RECON-006:** Rewrites preserve equivalent derived inputs, invalidate different ones, and reset incompatible tail cursors.
- **REQ-RECON-007:** Activity, freshness, coverage, and runtime readiness are independently inspectable. Versioned index-migration phase, captured-scope progress, and applied migration version are part of this inspectable surface.
- **REQ-RECON-008:** Raw indexing progresses independently of model work; derived artifacts are generation-bound and repairable.
- **REQ-RECON-009:** Maintenance pause persists across client auto-start; poll mode reconciles all roots. Operator pause is distinct from versioned index-migration maintenance (REQ-RECON-013).
- **REQ-RECON-010:** Versioned historical index migration is eligible only when the package-declared durable `INDEX_MIGRATION_VERSION` is greater than the database's applied index-migration version. A package upgrade, schema-only upgrade, daemon restart, or parser-source hash change does not by itself start this migration. Fresh databases record the current version as applied and do not enter maintenance.
- **REQ-RECON-011:** Index migration updates Recall's derived index from the current transcript files. It does not modify those files. Current supported transcript contents are authoritative; the prior index snapshot and the recorded discrepancy record remain preserved, and only those documented historical differences may be accepted.
- **REQ-RECON-012:** Before mutating derived-index state for an eligible versioned migration, the daemon takes a consistent DuckDB and FTS-sidecar backup with an explicit complete manifest. An incomplete backup never authorizes mutation. Safe rollback restores that backup and leaves the applied index-migration version unchanged.
- **REQ-RECON-013:** An eligible versioned migration runs in an explicit inspectable maintenance state, distinct from operator pause and from normal reconciliation. Status reports phase, target and applied versions, captured-scope counts, completed and remaining work, backup identity, timestamps, and errors. RPC/status remain usable for reads during this phase. Backup identity is reported as both the recorded `backup_path` and a `backup_path_present` reading of whether that path still resolves: the job record outlives the directory, and a path an operator reads as a live rollback when nothing is there is worse than no path at all. `backup_path_present` is null when no backup was taken.
- **REQ-RECON-014:** Versioned migration is crash-safe: an interrupted run resumes the same captured scope, does not redo already-verified captured sources, and does not treat transcript changes observed after capture as in-scope until the captured scope is verified. A captured source whose parse is a stable unsupported classification, with prior indexed history left intact, settles its captured key. That settlement does not acknowledge the source as current, erase diagnostics, or replace history. Outside a running migration, captured-scope settlement is a no-op. Transient failures, malformed input, and other incomplete captures are not settled by this rule.
- **REQ-RECON-015:** The applied index-migration version advances only after the captured scope is verified for current-file fidelity, equivalent enrichment preservation, current keyword membership, and truthful coverage. Every captured key must have been written successfully or received the history-preserving unsupported classification in REQ-RECON-014. Transcript changes after capture are then caught up by normal operation after maintenance exits. A failed or unverified run does not mark the version complete. Settling an unsupported captured key does not make that source current or restore future indexing for it.
- **REQ-RECON-016:** Versioned migration has a separately declared overall runtime budget. The five-second exclusive-writer ceiling and the concurrent active-freshness gate do not apply inside this explicit maintenance phase. Correctness, equivalent enrichment preservation, current keyword membership, truthful coverage, and the 4GiB RSS ceiling remain in force. Relabeling ordinary backlog, discovery, rewrites, or checkpoints as migration to evade normal-operation floors is forbidden.

- **REQ-RECON-017:** Production pins an exact tested DuckDB engine and fixed storage target, never the moving `latest` alias. New databases use the selected format. Ordinary opens preserve existing format. Selection compares legacy, fixed v1.2.0, and relevant newer stable capabilities on owned populated copies, and demonstrates read/write/reopen/rollback compatibility for every retained supported runtime, including the rollback installation. Engine upgrades alone never trigger transcript reparse.
- **REQ-RECON-018:** Existing storage conversion is explicit inspectable maintenance with a complete consistent DuckDB/sidecar backup before mutation. The existing connection owner drains database work, safely closes the shared handle, attaches with the selected format, commits a legitimate durable maintenance-state transition, checkpoints, reopens and verifies persisted format identity. Diagnostic create/drop-table tricks are forbidden. Existing backup, migration, coordinator and recovery mechanisms own this path; resume retains captured scope and the original matching backup. Completion requires persisted format identity and any applicable index verification. Format identity remains distinct from `INDEX_MIGRATION_VERSION`. Recovery backup directories, including interrupted/incomplete artifacts, are retained independently of query-snapshot age-based garbage collection; retiring them requires explicit operator removal.
- **REQ-RECON-019:** Storage/write simplification preserves keyword visibility, row identity, ordering, no-op behavior, equivalent enrichment and recovery. Duplicated batching/change detection/checkpoint policy is consolidated where measured evidence supports it. Retained engine workarounds have a current reproducer and removal condition near authoritative code/tests. Behavior-preserving refactoring remains separately verifiable from behavior changes; no broader logical-schema or public-contract redesign is implied.
- **REQ-RECON-020:** The repository harness provides reusable small and populated performance checks for first indexing, append, rewrite, no-op reconciliation, checkpoint, keyword/vector search, freshness, RSS, WAL and database growth. Evidence records source/relevant dirty state, environment and input identities, all samples and predeclared tolerances. Short comparisons use at least three fresh processes and report median and worst sample, never acceptance by fastest sample. Future DuckDB engine/storage upgrades repeat the documented comparison and applicable correctness/durability gates. Existing hard floors remain mandatory.
- **REQ-RECON-021:** A dated human performance exception may authorize one identified local rollout candidate despite named measured latency failures. The exception records the exact measurements and preserves their raw red outcomes; it does not create a new threshold, excuse unmeasured regression, change timeouts, or waive correctness, durability, resource, readiness/responsiveness, freshness, fairness, corpus fidelity, discrepancy-record retention, or recovery gates.
- **REQ-RECON-022:** A dated human resource exception may authorize one identified candidate, local rollout, PR, and release despite named measured RSS failures. It preserves the observations as raw failures, keeps memory measurement and reporting mandatory, and creates no replacement threshold. Every unexcepted correctness, durability, backup/rollback, readiness, responsiveness, freshness, fairness, discovery, corpus fidelity, discrepancy-record, recovery, installed-host, CI, and publication gate remains required.
- **REQ-RECON-023:** A dated human migration-read exception may authorize one identified candidate, local rollout, PR, and release despite commands hanging or timing out only while installed historical reconciliation is running. Raw timeouts remain reported. The exception changes no runtime/client timeout and creates no numerical replacement ceiling. Migration completion within its declared overall budget, integrity, current-content fidelity, recovery, all post-migration host tasks, normal-operation responsiveness/freshness, CI, and publication verification remain required.
- **REQ-RECON-026:** When no transcript observation or derived total has changed, periodic reconciliation performs no per-source writer update. Inventory walks may read the durable catalog once per root and persist bounded scan state; usage rollups update only differing totals; an idle raw scheduler waits for a relevant edge with a bounded eligibility rescan. Periodic work must not grow with tick count or create writer work proportional to the total unchanged corpus.
- **REQ-RECON-027:** Repeated scalar binding in a raw commit must not repeatedly probe an unavailable optional pandas installation. The fix must not place a `None` sentinel in `sys.modules` and must not add pandas solely as a workaround without measurements showing that dependency to be the simplest viable design. A red-then-green import-probe reproducer and three fresh-process large-commit samples prove the failed imports are gone without a CPU regression.
- **REQ-RECON-029:** A parked unsupported file stores a bounded diagnostic payload (`kind`, `detail`, `byte_offset`; at most 32 records, detail truncated at 160 characters). `reconciliation.unsupported_summary` aggregates those parks by source and detail, with a file count and at most three sample paths per group, at most eight groups, and an explicit truncated flag when the catalog page or group list is cut. A kind-only payload from an older daemon is reported as detail omitted, not invented. Daemon startup reopens those opaque parks once, through the writer, so the next raw commit rewrites the payload; a payload that already carries `detail` stays parked, and raw scheduling itself performs no reopen write (REQ-RECON-026). A reopened file re-parks with the same committed offset: it is not acknowledged as current and its unsupported record is not skipped.
- **REQ-RECON-028:** Reconciliation status reports `index_only_sessions`, a per-source count of this host's indexed sessions whose transcript no longer exists anywhere the catalog can see. A session counts as backed when any indexed row with the same source session identity maps to a present, non-missing catalog file, so a transcript that was only moved and indexed again under its new path is not counted as lost. Rows attributed to another host are excluded. The text summary appends `index_only_sessions=<total>` to its `reconciliation:` line when the total is nonzero. The count describes history that recall now holds as its only copy. It is not an actionable per-command warning, so no status notice renders it.

### Invariants and non-goals

The daemon never reports readiness before its RPC/status contract is usable, never lets historical work starve recent or finite work, and never trades correctness for a latency result. Resource bounds named as ceilings are hard except for an exact candidate covered by a dated REQ-RECON-022 human decision. Schema migration, versioned index migration, and concurrency behavior are high-risk and require real database/crash-barrier evidence. Public CLI/RPC contracts require contract evidence. Parser behavior is specified by the ingestion worker's parser SPEC.

The domain is readable, supported transcripts under configured roots, including old imports, resumed sessions and supported subagent paths. Missing or unreadable roots, unsupported records and disappeared sources remain accounted limitations. Missing sources never authorize deletion of indexed history. A finite stable source converges; active sources advance through finite complete prefixes.

Versioned historical index migration and normal reconciliation are separate acceptance paths. Migration mutates only derived index state. Transcript files are never inputs for in-place edit. Normal-operation responsiveness, freshness, fairness, writer occupancy, and checkpoint bounds remain required against a populated database, including ordinary historical discovery and genuine rewrites.

Non-goals are another storage engine, broker, distributed scheduler, new harness-control features, rebuilding unchanged embeddings, remote deployment, converting DuckDB storage format as a prerequisite for either path, and treating every package upgrade or restart as a historical reparse.

### Amendments to existing contracts

The following clauses replace the conflicting behavior in earlier sections and their acceptance items; unrelated clauses remain in force.

- `REQ-INDEX-010/011/012/015/016`, `REQ-DAEMON-030`: append-only parsing is an optimization whose precondition is a verified complete-prefix digest and compatible parser normalization checkpoint. Legacy byte offsets alone confer no trust. Arbitrary rewrites, shrink, absent trusted checkpoints and per-source parser-revision mismatch require full reconciliation of that source under the path that owns the work. Adapters without proven resumable normalization use full parsing. `next_byte_offset` acknowledges complete committed records only. Prefix verification may read all prior bytes; the earlier acceptance requiring only new bytes to be read is replaced by this verified-prefix contract. Source-aware token merging remains required when resumable normalization is used. A parser-source hash change does not by itself start versioned historical index migration or suspend normal-operation floors; only `INDEX_MIGRATION_VERSION` (REQ-RECON-010) does that.
- `REQ-INDEX-002/003/018`: discovery compares durable source observations including file identity, nanosecond ctime/mtime, size, declared sidecars and parser revision. Matching size/mtime alone cannot establish currentness. Enumeration and hashing occur outside the writer lock. Parser revision remains observational currentness evidence; corpus-wide historical reparse eligibility is REQ-RECON-010.
- `REQ-RECON-024`: Raw source admission bounds concurrent preparation, and no source may be refused admission forever. `FairScheduler.select` stops the whole selection at a refused head so a lane position survives transient back-pressure; a source whose own reservation exceeds the preparation budget has no release to wait for, so it must be admitted when nothing else is reserved. Such a claim already exceeds the budget on its own, so it is not also charged against it: every other source is admitted as if it were absent, which keeps the active-protected slice available to live sources for the length of its preparation. Rationale: four codex transcripts of 143-301 MiB reserved 574-1204 MiB against a 512 MiB budget, refused every turn, and wedged raw reconciliation for the life of the daemon -- 1,376 scheduling calls with zero commits and 9,607 sources pending; charging one of them to the shared budget instead refuses every `live`/`show --fresh` refresh until it commits.
- `REQ-RECON-025`: A requested path waits on the shared scheduler, so its wait is bounded by scheduler *progress* rather than by a fixed duration: any turn the scheduler serves renews the budget -- this caller's or another worker's -- as does a preparation claim in flight, and a scheduler that neither serves nor prepares anything within the bound must fail the request with a message naming the path, the elapsed bound, and where to inspect reconciliation state. A requested index must never wait indefinitely on work the scheduler will never do, and never on work that is not its own: a plain incremental `recall index` waits only for the sources whose pending state its own observation created or advanced, hands every source that was already pending when it observed it to the shared drain, and returns a summary carrying the backlog still pending and the sources served in the last minute. A request whose per-path options must ride with each source -- `--full`, `--recreate`, `--since`, `--project`, `--root/--host`, `--context` -- keeps waiting for every source it observes, because those options exist only while it is in flight. Rationale: a plain `recall index` against a 6,515-source backlog blocked for the whole drain to report work the daemon was already doing, and a background worker's 300 MiB parse made a truthful wait read as a wedged scheduler.
- `REQ-INDEX-013`, `REQ-DAEMON-049`: asynchronous complete inventory replaces catch-up-before-discovery ordering. All pending paths share the fair coordinator; newest-first preference does not override reserved historical service. Subscription capacity never bounds source coverage. Versioned index migration (REQ-RECON-010..016) is a distinct maintenance phase that uses the same coordinator and catalog rather than a second work queue; it must not be used as a label for ordinary pending work.
- `REQ-RECON-001/002/019`: periodic reconciliation is a complete inventory walk of every configured root, repeated `min(daemon.interval, 30 s)` after the previous walk ends; the 30 s cap keeps old-mtime imports, which only the walk discovers, inside the discovery floor. A walk reads each root's present catalog rows once and hands the writer only the sources whose observation would change them: new, changed file identity or parser revision, marked missing, or catalogued under another root. A walk over an unchanged root therefore writes only its scan record. After a complete walk, equal present and discovered counts with no symlinked discovery prove nothing vanished; otherwise each present source under the root is stat-checked, and only a source whose stat reports it absent is marked missing. The shared raw drain idles the same way: after a turn that finds nothing eligible, it waits for source activity, a newly catalogued source, a released claim, a request left to the drain, a resume, or lifted write pressure, and rescans at most 5 s later for work that becomes eligible with time alone, such as an expiring retry backoff. An idle daemon therefore makes one scheduling query per rescan, not two per second.
- `REQ-INDEX-014`, `REQ-INDEX-021`, `REQ-CTX-004/006/020`: raw supported conversation/tool content and keyword progress are independent of model preparation. Equivalent effective derived inputs retain enrichment, including across positional shifts. Changed conversations with legacy context lacking dependency provenance cannot reuse summaries solely by ID. Derived writes bind to content/input generation; required backend failures are visible without disabling historical reads or status.
- `REQ-LIVE-001/002`: observed writes and available process marks determine activity independently of indexing success and subscription capacity. The roster exposes pagination/truncation and incomplete filtered coverage where an unindexed path has unknown metadata; absent metadata cannot prove filtered absence.
- `REQ-LIVE-002/007/009`: structured `live` output is a version-2 object with `schema_version`, `watching`, `sessions`, `next_cursor`, and `coverage`. `--fields` projects session fields inside this envelope; coverage and continuation survive projection. Local `--cursor` resumes a bounded page; activity changes between calls may reorder the roster. Fleet returns per-host observation, coverage and continuation plus explicit global truncation; legacy array-only hosts have unknown coverage. A combined fleet cursor is not supported; continuation is per host. `daemon status --limit/--cursor` pages durable source coverage (default 10 rows per `REQ-DAEMON-075`). Default offline status explicitly marks runtime coverage unavailable; an RPC pagination error cannot silently fall back to a different local page.
- `REQ-LIVE-003`: currentness requires validated catalog progress through the complete captured source boundary. Partial tails, semantic diagnostics, mismatched generations, and never-indexed sources (`committed_generation = 0` or no committed catalog row) remain non-current, the last with limitation `not_yet_indexed`. `live --fresh` catches up already-indexed behind rows on the page and is bounded by the responsive-read floor (2 s); it does not first-index. `show --fresh` remains the one-session write, bounded by `live.fresh_timeout`. Fresh writes are coalesced through the fair coordinator.
- `REQ-LIVE-004/011`: cursors include a content epoch. Verified appends preserve it; semantic rewrites return a structured reset requirement and close follow explicitly. Fresh/follow use coordinator commit notifications; explicit fresh calls cannot bypass fairness or duplicate jobs. The earlier direct `index_single_session` debounce-bypass requirement is replaced.
- The earlier Decisions deferring a maximum debounce, relying on live membership for coverage, or accepting silent filtered absence are superseded within this reconciliation contract. Quantitative acceptance targets are the named floors in `BRIEF.md`.

### Acceptance

- [ ] REQ-RECON-001 through REQ-RECON-027 satisfy the mapped QA gates in `RECONCILIATION_QA.md`; any REQ-RECON-021/022/023 exception is recorded separately from its raw verdict.
- [ ] Complete inventory accounts for old imports, missed events, overflow, retries, unsupported sources and missing roots; continuing arrivals and appends satisfy the declared fairness bound under normal operation.
- [ ] Real parser/database rewrite, partial-record, crash and concurrent-generation regressions preserve exact supported semantic content and valid derived inputs.
- [ ] Public status, bounded live/tail/follow, pagination, cursor reset, persistent pause, and index-migration phase/progress tasks pass on the assembled daemon.
- [ ] Schema 24 through current schema applies on a retained host clone. Versioned historical index migration and normal daemon operation pass as separate paths: migration verifies the captured source scope, backup/resume/rollback, and overall runtime budget without applying the five-second writer or concurrent active-freshness floors; normal operation then verifies responsiveness, freshness, fairness, ordinary historical discovery, genuine rewrites, and checkpoints against the populated database.
- [ ] Full local checks, the driver-executed ten-task objective operator charter defined in `RECONCILIATION_QA.md` (<=30 minutes), both split clone paths, and the installed-host gate pass on their recorded artifact identities.

### Risk tags

- **[RISK-HIGH] Versioned historical index migration**: additive durable applied-version and in-flight job state, consistent backup before mutation, crash-safe resume, verified completion, and rollback on a populated database. Requires real database/crash-barrier and full captured-scope evidence; not a schema-only DDL change and not a DuckDB storage-format conversion.

### Decisions

- 2026-09-08 — The reconciliation requirements and execution-based QA are authorized. **ratified (human)**
- 2026-09-08 — The prior unlimited-debounce decision is superseded: continuous appends have a 10-second maximum coalesce window and finite captures yield. **provisional (driver)**
- 2026-09-09 — Current supported transcript files are authoritative. Preserve the prior index snapshot and the recorded discrepancy record; only those documented historical differences may be accepted. **ratified (human)**
- 2026-09-09 — Versioned historical index migration during daemon upgrades is separate from normal daemon reconciliation. Migration updates the derived index from current transcript files and does not modify those files. Eligibility is a durable `INDEX_MIGRATION_VERSION`, not every package upgrade or restart. This split supersedes the combined recovery/performance gate and the storage-format-first proposal; that hypothesis remains evidence, not a prerequisite or presumed fix. Prefer existing migration and coordinator mechanisms. **ratified (human)**
- 2026-09-09 — Overall index-migration runtime budget is 4 hours (14400s) for the reference corpus. The five-second writer ceiling and concurrent active-freshness gate do not apply during that explicit maintenance phase. **provisional (driver)**
- 2026-09-11 — A stable unsupported parse with preserved indexed history settles its captured migration key so versioned migration can finish. The catalog row stays pending with `last_error=unsupported` until the source or parser changes. This does not index unsupported content, replace history, or waive current-content/freshness floors for supported sources. **provisional (driver)**
- 2026-09-09 — `live --fresh` is a roster catch-up, not a first-index. It refreshes only already-indexed behind rows (`committed_generation > 0`) and is bounded by the responsive-read floor (2 s). Never-indexed rows stay `current: false` with limitation `not_yet_indexed`; first index remains the fair coordinator's after resume and inventory. `show --fresh` remains the one already-known session write, bounded by `live.fresh_timeout` (10 s). Do not lengthen the operator RPC deadline or `live.fresh_timeout` to hide first-index cost. This supersedes treating `live --fresh` as indexing every behind row on the page. **ratified (human)**

- 2026-09-10 — The user authorizes explicit backed-up storage-format modernization, scoped storage/write simplification and reusable performance checks, superseding the earlier storage-conversion exclusion. Exact engine and fixed format are selected from current capabilities and owned populated evidence; v1.2.0 is a comparison point. Real-DuckDB backup, interruption, resume, rollback and connection-ownership evidence remain required. No broader logical-schema/public-contract redesign is authorized. **ratified (human)**

- 2026-09-10 — Select exact DuckDB1.5.5 and fixed storage `v1.2.0`, based on the owned-populated legacy/1.2/1.4/1.5 comparison and real interoperability with checkout1.5.2 and staged/rollback1.5.5. Newer formats expose DICT_FSST but show no measured workload benefit;1.5 has first/rewrite samples above5s. The fixed target retains the modern engine independently. All final floors still require evidence bound to the selected candidate. **provisional (driver)**
