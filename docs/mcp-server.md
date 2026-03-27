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
| `vstash_ask(query, top_k)` | Semantic search + LLM-generated answer with sources |
| `vstash_search(query, top_k)` | Raw retrieval without LLM — returns chunks with scores |
| `vstash_list()` | List all ingested documents |
| `vstash_stats()` | Database statistics (doc count, chunks, size) |
| `vstash_forget(source)` | Remove a document from memory |

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
