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
| `vstash_ask(query, top_k)` | Semantic search + LLM-generated answer with sources |
| `vstash_search(query, top_k)` | Hybrid search with context expansion and relevance signal |
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
