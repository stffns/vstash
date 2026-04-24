# MCP Server — Claude Desktop Integration

vstash includes a built-in [MCP](https://modelcontextprotocol.io/) server that gives Claude Desktop persistent document memory across sessions. Add files once, and Claude can search and answer questions from them in any conversation.

---

## Setup

### 1. Install vstash

```bash
pip install vstash
```

### 2. Add to Claude Desktop config

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vstash": {
      "command": "vstash-mcp"
    }
  }
}
```

> **pyenv users:** Use the full path to the binary:
> ```json
> "command": "/path/to/.pyenv/versions/3.x.x/bin/vstash-mcp"
> ```

### 3. Restart Claude Desktop

The vstash tools should appear in Claude's tool list.

---

## Available Tools

| Tool | Description |
|------|-------------|
| `vstash_add(path)` | Ingest a file, directory, or URL into memory |
| `vstash_remember(text, title)` | Ingest text directly without a file |
| `vstash_ask(query, ...)` | Semantic search + LLM answer with temporal filters |
| `vstash_search(query, ...)` | Hybrid search with recency boost, temporal filters, and context expansion |
| `vstash_get_chunk(chunk_id)` | Retrieve a single chunk by database row ID |
| `vstash_get_document_chunks(path, collection?)` | Get all chunk texts for a document by path |
| `vstash_list()` | List all ingested documents |
| `vstash_stats()` | Database statistics (doc count, chunks, size) |
| `vstash_forget(source)` | Remove a document from memory |
| `vstash_collections()` | List all available collections |
| `vstash_export(...)` | Export chunks with metadata for training data curation |
| `vstash_job(job_id)` | Check status of background directory ingestion |
| `vstash_journal_save(text)` | Save a journal entry (cross-session memory) |
| `vstash_journal_recall(query)` | Search journal entries semantically |
| `vstash_journal_log()` | List recent journal entries chronologically |
| `vstash_journal_prune(older_than)` | Remove old journal entries |

### Search Response Fields

`vstash_search` returns a JSON object with:

| Field | Description |
|-------|-------------|
| `chunks` | Array of search results with expanded context (±1 adjacent chunks) |
| `relevance` | Confidence tier: `"high"`, `"medium"`, `"low"`, or `"none"` |
| `hint` | Human-readable relevance explanation |
| `best_distance` | Cosine distance of the best vector match (lower = more relevant) |

The relevance signal uses the best vector distance to estimate confidence:

| Distance | Tier | Meaning |
|----------|------|---------|
| ≤ 0.95 | **high** | Results are relevant |
| 0.95–0.98 | **medium** | Results may be tangential |
| > 0.98 | **low** | Results may not be relevant |

This works from the first search — no usage history or warm-up required.

### Per-call RRF controls *(added in v0.27.0, issue #159; `retrieval_mode` added in v0.33.0, issue #275)*

Both `vstash_search` and `vstash_ask` expose four parameters for overriding the default hybrid-retrieval behavior on a per-call basis. All four default to `None` / `False` — if you omit them, adaptive RRF (the default pipeline) runs unchanged.

| Parameter | Type | Meaning |
|---|---|---|
| `vec_weight` | `float \| None` | Pin the RRF vector weight for this query (valid range `[0.0, 1.0]`). Overrides adaptive RRF. May be passed alone — the store derives `fts_weight = 1.0 - vec_weight` when only one is provided. |
| `fts_weight` | `float \| None` | Pin the RRF FTS weight. Same range and same alone-or-together behavior as `vec_weight`. |
| `retrieval_mode` | `str \| None` | Which search branches to run. One of `"hybrid"` (default), `"vec_only"`, or `"fts_only"`. When set to a non-hybrid mode, `vec_weight` and `fts_weight` are ignored (the mode is a stronger statement of intent). The legacy `fts_only=true` bool was deprecated in v0.33.0 and removed in v0.35.0 (#281); passing it now hits a `TypeError` from the argument binder. |

**When to use them:**

- **`retrieval_mode="fts_only"`** — for debugging ranking (answers "is this a vector problem or an FTS problem?") and for queries containing literal terms that must match exactly (drug names, diagnostic codes, error strings, SKUs). Particularly useful with the clinical-domain weakness documented in `docs/embedding-models.md` for `paraphrase-multilingual-MiniLM-L12-v2`.
- **`retrieval_mode="vec_only"`** — for corpora where the keyword signal is noise (tabular data, code where identifiers are not semantically informative, cross-lingual corpora where the query and documents tokenize differently) or for ranking debug (answers "what does pure semantic search think?"). Symmetric to `fts_only`.
- **`vec_weight=0.1, fts_weight=0.9`** — bias a specific query toward keyword matching without disabling the vector path entirely. Adaptive RRF resumes on the next query with defaults.
- **`vec_weight=0.9, fts_weight=0.1`** — inverse: bias toward semantic paraphrase when you know the exact term may not be in the corpus.

**Type coercion note.** The MCP server accepts string values for these parameters (`"0.5"`, `"hybrid"`, `"vec_only"`, `"fts_only"`) and coerces them internally, so clients that serialize JSON numbers or enum strings get a consistent parse at the tool boundary instead of a 422. An unparseable value surfaces as a structured `{"error": "..."}` response naming the offending field. If a caller supplies a non-hybrid `retrieval_mode` together with explicit weights, the weights are silently ignored.

Example MCP call:

```json
{
  "tool": "vstash_search",
  "arguments": {
    "query": "morphine contraindications",
    "top_k": 5,
    "retrieval_mode": "fts_only"
  }
}
```

---

## API Key Configuration

MCP servers don't inherit your shell environment variables. If you use Cerebras or OpenAI for inference, add your API key to `~/.vstash/vstash.toml`:

```toml
[cerebras]
api_key = "your-key-here"
```

Or use a fully local backend (Ollama) which needs no API key:

```toml
[inference]
backend = "ollama"

[ollama]
host = "http://localhost:11434"
model = "llama3.2"
```

---

## Troubleshooting

- **Tools don't appear:** Make sure `vstash-mcp` is on your PATH. Run `which vstash-mcp` in terminal to verify.
- **"No module named vstash":** The MCP server is using a different Python than where vstash is installed. Use the full path in the config.
- **Search works but ask fails:** Check that your inference backend is configured and the API key is in `vstash.toml` (not just in your shell env).
