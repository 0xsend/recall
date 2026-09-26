---
name: recall
description: Use when the user wants to inspect or follow current AI agent sessions and what they are doing; resume or continue past work; search or analyze session, tool, or bash history; or derive permission suggestions across Claude Code, Codex, Pi Agent, Grok, and Kimi Code.
---

# recall

Inspect current AI agent activity and recover past work across Claude Code, Codex, Pi Agent, Grok, and Kimi Code.

## When invoked

For "what is running / what are the agents doing / follow this session," start with [current sessions](#current-sessions). For a historical lookup, use the [past-session workflow](#past-session-workflow). The user may give a topic, timeframe, project, source, or just `/recall:recall` with the surrounding turn as context.

For any request to resume, continue, pick up, or reconstruct where work left off, read [CONTINUATION.md](CONTINUATION.md) and follow it before acting.

Treat short forms such as `/recall:recall continue <session-id>` and `/recall:recall resume last` as complete continuation requests. Do not ask the user to restate that you should recover the intent, larger arc, current state, and next step.

## Current sessions

Start with the roster, select a session, then read a bounded tail:

```bash
recall live --json                         # active sessions observed in watch or poll mode
recall live --project <path> --json         # scope an indexed roster to the project
recall live --all --json                    # also recent idle and ended sessions
recall show <session-id> --tail 20 --fresh --json
```

Read the version-2 envelope's `sessions`; preserve `coverage` and `next_cursor`.
Continue local pages with `recall live --cursor '<next_cursor>' --json` using the
same filters. `--fields` projects session fields and retains the envelope. A page,
an incomplete scan, or unknown filtered metadata cannot establish absence.

Use a row's `id` or `source_session_id` for `show`; `search --session` needs
the recall `id`, while `live mark --session` needs `source_session_id`.
If an unindexed row has
`id: null`, it is listed with `freshness.current: false` and
`not_yet_indexed`; `live --fresh` does not first-index it. Re-read the roster
after it gains an `id`, then `recall show <id> --fresh` for a one-session
catch-up. Read `turn` for the latest working/awaiting-input state and running
tool; liveness alone does not say what the agent is doing.

- **Read coverage and liveness together.** `active` means a recent observed
  write, unless an observed writer process has ended; it does not prove
  continuous work. Poll mode also observes activity. `--all` adds recent `idle`
  and `ended` rows. Inspect `coverage.complete`, `unknown_count` and continuation
  before concluding absence; project/host metadata may be unknown before
  indexing even when the source is known. Fleet coverage retains each host's
  cursor and limitations; query hosts individually when the merge is truncated.
- **Check `freshness.current` on every row and `show`.** `false` means the
  index is behind, not yet indexed, or freshness could not be established.
  `live --fresh` catches up already-indexed rows on the page within 2 s; it
  does not first-index. `show --fresh` indexes one already-known session
  (default 10 s). A timeout still returns stale data with a stderr note.
  Neither flag forces discovery of a new or resumed session: that waits for
  the discovery loop (default 30 s).

Keep the opaque `cursor` from the bounded `show` and pass it back unchanged:

```bash
recall show <session-id> --after '<cursor>' --fresh --json
recall show <session-id> --after '<cursor>' --follow --timeout 60
```

An empty delta is success and preserves the cursor. A semantic rewrite returns
`cursor_reset: true`; replace the prior window and use the new cursor instead of
appending it as a delta. Follow closes with `reason: content_rewritten` on such a
reset. Follow emits NDJSON deltas
and one `closed` event; without `--after` it starts from now. Use a deadline,
and inspect the closing reason and stderr. Tail/after cannot combine with
`--message-limit` or `--fleet`; follow cannot combine with `--tail` or `--fleet`.

When recording a harness observation, use its actual source session ID,
writer PID, and source:

```bash
recall live mark --session <source-session-id> --pid <writer-pid> --source <source> --json
```

A mark supplies process-exit evidence; it does not control
or resume an agent. Do not substitute the observing shell's PID. Inspect
`marked`: an unavailable daemon returns `marked: false` even with exit 0.
See [CLI_REFERENCE.md](CLI_REFERENCE.md#recall-live) for discovery, watcher,
fleet, and mark details.

## Past-session workflow

1. **Find** the candidate session(s):
   ```bash
   recall search "<topic or phrase>" --json        # hybrid semantic + keyword
   recall list --since 24h --json                  # by recency
   recall list --project <path> --since 7d --json  # by repo + window
   ```
   Combine flags as needed. Add `--source claude-code|codex|pi-agent|grok|kimi-code` when the user names one. Default mode is `auto` (hybrid when embeddings exist, else keyword); widen a stubborn search with `--mode keyword`.

   `--tool <Name>` (e.g. `--tool Bash`) **restricts results to that tool's calls only and drops all conversation/message text.** It is the wrong flag for "where was X discussed or decided" — use it only when you specifically want the command/tool invocations themselves, and always run the unfiltered query first.

2. **Read** the matching session:
   ```bash
   recall show <session-id> --json                       # messages only
   recall show <session-id> --tools --json               # + tool calls
   recall show <session-id> --tools --thinking --json    # + thinking blocks
   ```

3. **Use** the result:
   - For a lookup or analysis request, summarize only the relevant findings.
   - For a continuation request, follow [CONTINUATION.md](CONTINUATION.md) to recover the larger arc, reconcile it with current reality, and take the next safe in-scope action.

Do not paste raw `--json` or full transcripts into the reply. Compress past context into what is usable now.

When piped or captured (non-TTY), recall already emits compact structured **TOON** by default — you only need `--json` if you'll pipe through `jq`. Either way work from the structured output, not the verbose TTY text. When `recall search` returns dozens of hits, only the top few matter, and the `lexical_match` field tells you which are real keyword hits vs semantic guesses (see below).

## Empty results ≠ missing session

A search returning nothing — or only junk — almost never means the daemon missed a session. It usually means the query didn't match or was over-filtered. Two signals make this explicit, so **don't `2>/dev/null` recall** — read its stderr:

- `recall search` prints a `note:` on stderr when results are empty or semantic-only, often with the exact next step (e.g. "0 results with --tool Bash; 12 without it").
- Every result carries `lexical_match`. `true` = matched your query terms textually; `false` = a semantic neighbor that may be unrelated. **All rows `false` means there were no keyword matches at all** — vector search always returns ~`limit` nearest neighbors even for gibberish, so a wall of `lexical_match: false` is the "nothing really matched" signal.

Work down this list before concluding anything about indexing:

1. **Drop `--tool`.** Filtering to one tool's calls hides all message text — the single most common cause of a false empty. Re-run the plain query.
2. **Widen the query.** Use fewer terms and `--mode keyword` to bypass semantic ranking; many rare terms together can score everything near zero.
3. **Confirm it's indexed, not just unmatched:** `recall list --project <path> --since 30d` or `recall show <session-id>` (accepts the original source UUID or the recall id). If `show` returns the session, it was indexed — the search query is the problem.
4. **Check it's in recall's scope at all.** recall indexes what agents did through their tools and the conversation; it does **not** index commands a human ran directly in a terminal. For those, look in shell history (e.g. `~/.zsh_history`), not recall.
5. **Parked unsupported is a recall gap, not absence.** A stderr warning about `unsupported` parser diagnostics, or `reconciliation.coverage[].unsupported > 0` on `recall daemon status --json`, means the transcript exists but recall could not commit records. Read `reconciliation.unsupported_summary` for the source, record detail, count, and a sample path. File a recall issue, or fix the parser when you are in the recall repo. Do not conclude the session never happened.
6. **A pending backlog means "not indexed yet", not "never happened".** `search`, `list` and `stats` print `coverage is incomplete — N discovered transcripts are not indexed yet` on stderr whenever `reconciliation.pending` is nonzero. Read `recall daemon status --json --fields reconciliation` and re-run once the backlog drains before concluding anything about absence.

Heads-up — observer effect: a `recall search "<terms>"` is itself indexed within seconds, so re-running the same search can surface your *own* just-run diagnostic commands as hits. Don't mistake them for the session you were looking for.

## Other commands

```bash
recall stats              # overview
recall stats tools        # tool usage counts
recall stats bash         # bash command breakdown
recall stats bash --suggest    # permission suggestions
recall stats tokens       # token usage by project
recall stats usage        # tokens by source x model x host (fleet ledger)
recall stats usage --since 30d # same, limited to a time window
recall stats skills --json     # fail-closed local + configured-fleet skill census
recall stats skills --local    # explicitly query only this host
recall fleet status       # probe hosts in ~/.config/recall/fleet.toml
recall list --fleet --since 7d
recall search "topic" --fleet --json
recall stats usage --fleet --since 7d
```

`recall index` runs automatically via the daemon — rarely needed mid-session.
Inspect `recall daemon status --json --fields reconciliation` for separate RPC,
observation, raw, keyword and enrichment readiness. `daemon pause` persists
across client auto-start; `daemon resume` restores work. Model unavailability
must not be mistaken for unavailable history. Before an authorized historical
repair, retain a consistent DB/sidecar backup and use bounded scopes with
`index --full --no-embed --context template --since 30d`; a bare full reparse
can trigger expensive context generation for previously unsummarized history.

**Fleet (live):** with `~/.config/recall/fleet.toml` (`[[host]] name` + `ssh`),
`--fleet` fans out over SSH to each host's daemon and merges results (every row
carries `host`). Prefer this over re-indexing remote trees for day-to-day fleet
views. Offline bulk import remains `recall index --root PATH --host NAME` after
syncing home-shaped trees; see [CLI_REFERENCE.md](CLI_REFERENCE.md).

`recall stats skills` is deliberately stricter: it defaults to this host plus
every configured fleet host and exits nonzero without census data if inventory,
host, CLI, query, payload, or source coverage is incomplete. Use `--local` only
when the requested scope is explicitly one host. A zero-control population is
not evidence that no skill fired.

## Further reading (load on demand)

- [CLI_REFERENCE.md](CLI_REFERENCE.md) — every command, flag, and JSON shape.
- [CONTINUATION.md](CONTINUATION.md) — target resolution, multi-session arc recovery, current-state reconciliation, and safe continuation.
- [EXAMPLES.md](EXAMPLES.md) — search, browse, deep-dive, and scripting recipes.

If `recall` is missing, broken, or not indexed yet, use the `recall-setup` skill
instead of improvising install or daemon commands from this search workflow.
