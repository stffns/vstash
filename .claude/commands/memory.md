---
argument-hint: <query>
description: Search your vstash document memory for relevant context
---

# vstash Memory Search

Search your local document memory using vstash semantic search. Use this before answering questions that might benefit from project docs, specs, notes, or past decisions.

## Instructions

1. Run `vstash search` with the user's query using the Bash tool:

```bash
vstash search "$ARGUMENTS" --top-k 5 --json
```

2. Parse the JSON results and present a concise summary of the most relevant chunks found.

3. If no results are found, tell the user their memory is empty for that query and suggest adding documents with `vstash add`.

4. If the query has filters (collection, project), pass them:
```bash
vstash search "$ARGUMENTS" --top-k 5 --json --collection <name>
vstash search "$ARGUMENTS" --top-k 5 --json --project <name>
```

## Output Format

Present results as a brief list with source and relevance, then use the context to help answer the user's underlying question. Do not just dump raw chunks — synthesize them.
