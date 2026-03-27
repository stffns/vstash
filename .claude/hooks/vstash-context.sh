#!/bin/bash
# Hook: auto-search vstash memory before responding to user prompts.
# Injects relevant document chunks as additional context.
#
# Filtering strategy: pattern-based (not score-based).
# Score thresholds don't work on small corpora — everything scores ~0.83+.
# Instead, skip prompts that are clearly actions/commands, not questions.

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt')

# Skip short prompts and slash commands
if [ ${#PROMPT} -lt 20 ] || [[ "$PROMPT" == /* ]]; then
  exit 0
fi

# Skip action prompts — commands, git ops, deploys, confirmations.
# These don't benefit from document context.
if echo "$PROMPT" | grep -iEq '(^(haz|crea|sube|publica|borra|ejecuta|corre|run|push|merge|commit|delete|fix|refactor|deploy|tag|build|test|lint) |commit|merge|PR|pull request|pypi|tests? |please$|por favor$|github|git )'; then
  exit 0
fi

# Run vstash search (JSON output, top 3 chunks, stderr suppressed)
RESULTS=$(vstash search "$PROMPT" --top-k 3 --json 2>/dev/null)

# Skip if no results or empty array
if [ -z "$RESULTS" ] || [ "$RESULTS" = "[]" ]; then
  exit 0
fi

# Format: title + truncated text (200 chars max per chunk)
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
