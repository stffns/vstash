# Integrating vstash with Claude

vstash can provide document memory to Claude in two ways: as a **Claude Code hook** (automatic) or as a **Claude Desktop MCP server** (tool-based). Both are independent — you can use one, the other, or both.

---

## Option A: Claude Code — Auto-Context Hook

Every time you send a message, vstash automatically searches your document memory and injects relevant context. No manual action needed.

### How it works

```
You type a question
       ↓
Hook runs `vstash search` with your prompt
       ↓
Relevant chunks injected as context
       ↓
Claude responds with document knowledge
```

### Setup

**1. Create the hook script**

```bash
mkdir -p .claude/hooks
```

Create `.claude/hooks/vstash-context.sh`:

```bash
#!/bin/bash
# Auto-search vstash memory before Claude responds.
# Skips action prompts (commit, merge, push) — only injects on knowledge questions.

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt')

# Skip short prompts and slash commands
if [ ${#PROMPT} -lt 20 ] || [[ "$PROMPT" == /* ]]; then
  exit 0
fi

# Skip action prompts that don't benefit from document context
if echo "$PROMPT" | grep -iEq '(^(haz|crea|sube|publica|borra|ejecuta|corre|run|push|merge|commit|delete|fix|refactor|deploy|tag|build|test|lint) |commit|merge|PR|pull request|pypi|tests? |please$|por favor$|github|git )'; then
  exit 0
fi

RESULTS=$(vstash search "$PROMPT" --top-k 3 --json 2>/dev/null)

if [ -z "$RESULTS" ] || [ "$RESULTS" = "[]" ]; then
  exit 0
fi

CONTEXT=$(echo "$RESULTS" | jq -r '
  [.[] | "[\(.title)] \(.text[:200] | gsub("\n"; " "))..."]
  | join("\n")
')

jq -n --arg ctx "$CONTEXT" '{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": ("vstash context:\n" + $ctx)
  }
}'

exit 0
```

```bash
chmod +x .claude/hooks/vstash-context.sh
```

**2. Register the hook in `.claude/settings.json`**

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/vstash-context.sh",
            "timeout": 15,
            "statusMessage": "Searching vstash memory..."
          }
        ]
      }
    ]
  }
}
```

**3. Add documents to memory**

```bash
vstash add docs/ --collection my-project
vstash add https://some-spec.com/api-docs
```

**4. Use normally** — ask questions and vstash context appears automatically.

### Optional: Manual skills

Create `.claude/commands/memory.md` for explicit searches:

```markdown
---
argument-hint: <query>
description: Search your vstash document memory for relevant context
---

Run `vstash search "$ARGUMENTS" --top-k 5 --json` and present a summary of the results.
```

Usage: `/memory how does authentication work`

### Scope

- **Project-level**: Place files in `.claude/` inside your repo (shareable via git)
- **Global**: Place files in `~/.claude/` (applies to all projects, requires vstash installed globally)

---

## Option B: Claude Desktop — MCP Server

Claude Desktop connects to vstash via the Model Context Protocol. This gives Claude tools to search, add, and ask your document memory.

### Setup

**1. Install vstash with MCP support**

```bash
pip install vstash[mcp]
```

**2. Configure Claude Desktop**

Open Claude Desktop settings and edit `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vstash": {
      "command": "vstash-mcp",
      "env": {
        "VSTASH_CONFIG": "~/.vstash/vstash.toml"
      }
    }
  }
}
```

**3. Restart Claude Desktop**

After restart, Claude will have these tools available:

| Tool | Description |
|------|-------------|
| `vstash_search` | Semantic search across all documents |
| `vstash_ask` | Search + LLM answer in one step |
| `vstash_add` | Ingest files, directories, or URLs |
| `vstash_list` | List all documents in memory |
| `vstash_stats` | Memory statistics |
| `vstash_find` | Find a document by partial name |
| `vstash_forget` | Remove a document |

**4. Add documents**

Either via CLI:
```bash
vstash add report.pdf notes.md
```

Or ask Claude Desktop directly:
> "Add this URL to my memory: https://example.com/api-docs"

**5. Use naturally**

> "What does the API spec say about authentication?"
> "Summarize the key findings from the report I added yesterday"

Claude Desktop will call `vstash_search` or `vstash_ask` automatically.

---

## Comparison

| | Claude Code (Hook) | Claude Desktop (MCP) |
|---|---|---|
| Context injection | Automatic on every prompt | On-demand (Claude decides) |
| Requires | `vstash` CLI + `jq` | `vstash[mcp]` |
| Config | `.claude/settings.json` | `claude_desktop_config.json` |
| Tools available | Search only (via hook) | Search, add, ask, list, forget |
| Best for | Coding with document context | Research, Q&A, document management |
| Interference | None between them | None between them |

Both use the same `~/.vstash/memory.db` by default, so documents added via CLI are available in both Claude Code and Claude Desktop.
