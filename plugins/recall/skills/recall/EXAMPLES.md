# Examples

Common usage patterns and workflows for recall.

## Finding Past Work

### Search for a Topic

Find sessions where you worked on authentication:

```bash
recall search "authentication"
recall search "oauth login"
recall search "JWT token"
```

### Search Within Bash Commands

Find bash commands related to git:

```bash
recall search "git" --tool Bash
recall search "rebase" --tool Bash
```

The `--tool` flag filters to Bash tool calls only, searching within bash command text.

### Filter by Source

Search only Claude Code sessions (input accepts `claude-code` or `claude_code`):

```bash
recall search "refactor" --source claude-code
```

Search only Codex sessions:

```bash
recall search "fix bug" --source codex
```

Note: Output displays `claude_code` (underscore) in results.

## Browsing Sessions

### Recent Activity

See what you worked on today:

```bash
recall list --since 24h
```

This week:

```bash
recall list --since 7d
```

Since a specific date:

```bash
recall list --since 2024-01-01
```

### By Project

Find sessions in a specific repository:

```bash
recall list --project /path/to/my-app
recall list --project ~/code/api-server
```

Combine with time filter:

```bash
recall list --project /path/to/repo --since 7d
```

## Session Deep Dive

### View Session Details

Get full conversation with tool calls:

```bash
recall show abc123def456 --tools
```

Include thinking blocks (Claude Code):

```bash
recall show abc123def456 --tools --thinking
```

### Export for Analysis

Get JSON for processing:

```bash
recall show abc123def456 --json > session.json
```

Extract specific fields with jq:

```bash
recall show abc123 --json | jq '.messages[] | select(.role == "assistant") | .content'
```

## Analytics

### Tool Usage Patterns

See which tools you use most:

```bash
recall stats tools
```

Output:
```
Bash: 456
Read: 312
Edit: 234
Glob: 156
Grep: 89
Write: 45
```

### Bash Command Analysis

Break down bash commands by type:

```bash
recall stats bash
```

Output:
```
git status: 45
npm test: 32
git diff: 28
uv run: 22
```

### Permission Suggestions

Generate patterns for auto-approval:

```bash
recall stats bash --suggest
```

Output:
```
Suggested Bash Permissions
==========================
high: run tests (45 uses)
high: git operations (73 uses)
medium: npm/yarn commands (47 uses)
```

Use these suggestions to configure Claude Code permissions:

```json
{
  "permissions": {
    "allow": [
      {"tool": "Bash", "prompt": "run tests"},
      {"tool": "Bash", "prompt": "git operations"}
    ]
  }
}
```

### Token Usage

See token consumption by project:

```bash
recall stats tokens
```

Output:
```
/path/to/project-a: 125000 in / 45000 out
/path/to/project-b: 89000 in / 32000 out
```

## Contextual retrieval

### One-shot template run

Add static repo and branch context to recent CONTENT and THINKING messages:

```bash
recall index --context template --recompute-context --since 7d
```

`--since 7d` keeps the first try fast while you confirm the ranking impact. Bash commands are not contextualized.

### Persist template mode

Set the mode under `[embedding.context]` in `~/.config/recall/config.toml`:

```toml
[embedding.context]
mode = "template"
```

The daemon picks up the change on its next cycle. Indexing runs inside the daemon, so a
`RECALL_CONTEXT_MODE` exported in your shell does not reach it; use `--context` for a
one-off run.

### Local LLM (MLX)

Install the MLX extra from a clone, or use the bootstrap installer with `--mlx`:

```bash
uv pip install -e 'packages/recall[mlx]'
```

Then contextualize a small recent window:

```bash
recall index --context llm-local --recompute-context --since 24h
```

The first run downloads the default `mlx-community/Llama-3.2-3B-Instruct-4bit` model, about 2 GB, into the MLX cache. The backend runs locally on Apple Silicon Metal GPU and does not use the network after model download. For tight RAM budgets, set the smaller model in `~/.config/recall/config.toml` (about 700 MB, roughly 3x faster, lower summary quality):

```toml
[embedding.context]
model = "mlx-community/Llama-3.2-1B-Instruct-4bit"
```

The daemon runs `recall index`, so an environment variable prefixed to the CLI command never reaches it. The daemon picks up `[embedding.context]` edits without a restart.

### Remote (Anthropic)

Install the `[anthropic]` extra and provide `ANTHROPIC_API_KEY` in the daemon's environment (or `api_key` under `[embedding.context]`). Never pass the key as a CLI argument.

```bash
recall index --context llm-remote --recompute-context --since 24h
```

Each eligible message is one Anthropic API call. Prompt caching mitigates repeated session-context input, but this is still paid API usage. Check `recall stats` afterward for `last_context_input_tokens` and `last_context_output_tokens`.

### Inspect counters

```bash
recall stats
```

Look for `last_context_mode`, `last_context_messages`, `last_context_input_tokens`, and `last_context_output_tokens`. They describe the most recent successful run, not the current config.

## Maintenance

### Initial Setup

Index all existing sessions:

```bash
recall index
```

This scans:
- `~/.claude/projects/**/*.jsonl` (Claude Code)
- `~/.codex/sessions/**/rollout*.jsonl` (Codex)
- `~/.pi/agent/sessions/**/*.jsonl` (Pi Agent)
- `~/.grok/sessions/**/chat_history.jsonl` (Grok)
- `~/.kimi-code/sessions/**/wire.jsonl` (Kimi Code)

### Periodic Updates

Run index to pick up new sessions:

```bash
recall index
```

Already-indexed sessions are skipped automatically.

### Force Reindex

Inspect freshness and reconciliation coverage before deciding to reparse history.
After a consistent DB/sidecar backup, bound an authorized repair and avoid model
work on previously unsummarized history:

```bash
recall daemon status --json --fields reconciliation
recall index --full --no-embed --context template --since 30d
```

### Rebuild Database

Start fresh (creates backup first):

```bash
recall index --recreate
```

### Index Specific Source

Only index Claude Code sessions:

```bash
recall index --source claude-code
```

Only index Codex sessions:

```bash
recall index --source codex
```

## JSON Output for Scripts

All commands support `--json` for scripting:

```bash
# Get session IDs from last week
recall list --since 7d --json | jq -r '.[].id'

# Find most-used tool
recall stats tools --json | jq 'to_entries | max_by(.value) | .key'

# Count sessions by source
recall list --json | jq 'group_by(.source) | map({source: .[0].source, count: length})'
```

## Workflow: Review Past Implementation

1. Search for the topic:
   ```bash
   recall search "rate limiting"
   ```

2. List sessions in that project:
   ```bash
   recall list --project /path/to/api --since 30d
   ```

3. Deep dive into relevant session:
   ```bash
   recall show abc123 --tools
   ```

## Workflow: Optimize Permissions

1. Index recent sessions:
   ```bash
   recall index
   ```

2. Get permission suggestions:
   ```bash
   recall stats bash --suggest
   ```

3. Review and add trusted patterns to Claude Code config.

4. Periodically re-run to discover new patterns:
   ```bash
   recall index && recall stats bash --suggest
   ```
