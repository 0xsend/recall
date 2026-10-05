---
name: recall
description: Use when the user asks what coding-agent sessions are doing now or did before - following a live session, resuming or continuing earlier work, finding where something was discussed, decided, or run, analyzing tool, command, token, or skill usage, or deriving permission suggestions across Claude Code, Codex, Pi Agent, Grok, and Kimi Code transcripts.
---

# recall

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

Use a row's `id` or `source_session_id` for `show`; `search --session` needs
the recall `id`. Read `turn` for what the agent is doing; `active` liveness only
means a recent write.

Two signals decide whether the answer is complete. `coverage.complete: false`,
`unknown_count`, or a `next_cursor` means the roster is partial, so a missing row
is not evidence of absence. `freshness.current: false` on a row or a `show` means
the index is behind the file; `--fresh` catches up an already-indexed session,
but a brand-new session (`id: null`, `not_yet_indexed`) appears only after the
discovery loop indexes it.

To read only what arrived since, pass the opaque `cursor` back unchanged:

```bash
recall show <session-id> --after '<cursor>' --fresh --json
recall show <session-id> --after '<cursor>' --follow --timeout 60
```

An empty delta is success. `cursor_reset: true` (or a follow closing with
`content_rewritten`) means the transcript was rewritten: replace the prior
window instead of appending. Paging, fleet coverage, flag combinations, and
`live mark` are in [CLI_REFERENCE.md](CLI_REFERENCE.md#recall-live).

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
   A `user` row is not necessarily human-authored: tool results (summary rows with no `content`) and harness-injected text share the role. For long sessions, use the projections in [CONTINUATION.md](CONTINUATION.md#long-sessions) instead of dumping raw user rows or a raw tail.

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

The daemon indexes continuously; do not run `recall index` to refresh a read.
Daemon state, maintenance, historical repair, fleet inventory, and the
fail-closed `stats skills` census are in [CLI_REFERENCE.md](CLI_REFERENCE.md).

## Further reading (load on demand)

- [CLI_REFERENCE.md](CLI_REFERENCE.md) — every command, flag, and JSON shape.
- [CONTINUATION.md](CONTINUATION.md) — target resolution, multi-session arc recovery, current-state reconciliation, and safe continuation.
- [EXAMPLES.md](EXAMPLES.md) — search, browse, deep-dive, and scripting recipes.

If `recall` is missing, broken, or not indexed yet, use the `recall-setup` skill
instead of improvising install or daemon commands from this search workflow.
