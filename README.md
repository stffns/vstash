# vstash

[![PyPI](https://img.shields.io/pypi/v/vstash)](https://pypi.org/project/vstash/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10+-blue)]()
[![tests](https://img.shields.io/badge/tests-~750_passing-brightgreen)]()

**Local document memory with hybrid retrieval that beats ColBERTv2 on 3/5 BEIR datasets.** Single SQLite file. Zero cloud dependencies. 20.9 ms at 50K chunks.

A 33M parameter embedding model, fine-tuned with zero human labels using vstash's own hybrid retrieval disagreement signal, surpasses ColBERTv2 (110M params) on SciFact, NFCorpus, and SciDocs. The model is published as [`Stffens/bge-small-rrf-v2`](https://huggingface.co/Stffens/bge-small-rrf-v2).

```bash
pip install vstash
vstash add paper.pdf notes.md https://example.com/article
vstash search "what's the main argument?"
```

---

## Retrieval Quality

| Dataset | Docs | vstash (tuned) | ColBERTv2 | BM25 | vs ColBERTv2 |
|---------|:----:|:---:|:---:|:---:|:---:|
| SciFact | 5K | **0.695** | 0.693 | 0.665 | **+0.2%** |
| NFCorpus | 3.6K | **0.395** | 0.344 | 0.325 | **+14.8%** |
| SciDocs | 25K | **0.188** | 0.154 | 0.158 | **+21.8%** |
| FiQA | 57K | 0.328 | **0.356** | 0.236 | -7.8% |
| ArguAna | 8.7K | 0.424 | **0.463** | 0.315 | -8.4% |

*NDCG@10 on [BEIR](https://github.com/beir-cellar/beir). Tuned model: `Stffens/bge-small-rrf-v2` (33M params, 384d). Reproducible via `python -m experiments.beir_benchmark`.*

---

## How It Works

```
Query --> Embed --+--> Vector ANN (sqlite-vec) --+
                  |                               +--> Adaptive RRF --> MMR Dedup --> Results
                  +--> FTS5 BM25 ----------------+
```

1. **Hybrid search**: vector similarity + keyword matching, fused via Reciprocal Rank Fusion
2. **Adaptive RRF**: IDF-based per-query weights. Rare terms boost keywords, common terms boost vectors. +21.4% on ArguAna
3. **MMR dedup**: diverse sections from long documents surface instead of redundant chunks
4. **Self-tuned embedding**: `vstash retrain` fine-tunes your embedding model using disagreements between vector and keyword search. Zero labels needed

---

## Install

```bash
pip install vstash                    # SDK + search
pip install 'vstash[ingest]'          # + PDF, DOCX, PPTX parsing
pip install 'vstash[serve]'           # + web UI (vstash serve)
pip install 'vstash[all]'             # everything
```

---

## Quick Start

```bash
# Search (free, no API key)
vstash add report.pdf ~/notes/ https://arxiv.org/abs/2310.06825
vstash search "what is the proposed method?"

# Ask (needs a local LLM -- auto-detects Ollama, LM Studio)
vstash ask "summarize the key findings"
vstash chat                           # interactive session

# Fine-tune on your own data
vstash retrain                        # generates training data from your corpus, trains locally
vstash reindex --model ~/.vstash/models/retrained
```

---

## Python SDK

```python
from vstash import Memory

mem = Memory(project="my_agent")
mem.add("docs/spec.pdf")
mem.remember("OAuth uses PKCE for public clients", title="auth-notes")

results = mem.search("deployment strategy", top_k=5)
for r in results:
    print(r.text, r.score, r.collection, r.tags, r.added_at)

answer = mem.ask("What are the system requirements?")
```

---

## Commands

```
vstash add <file/dir/url>    Add documents to memory
vstash remember "<text>"     Ingest text directly
vstash search "<query>"      Semantic search (free, local)
vstash ask "<question>"      Answer from your documents (needs LLM)
vstash chat                  Interactive Q&A
vstash list                  Show all documents
vstash stats                 Memory statistics
vstash forget <file>         Remove a document
vstash retrain               Fine-tune embeddings on your data
vstash reindex               Re-embed with a new model
vstash watch <dir>           Auto-ingest on file changes
vstash serve                 Web UI on localhost
vstash check [--repair]      Integrity check and repair
vstash config                Show configuration
vstash profile <cmd>         Manage named profiles
vstash journal <cmd>         Cross-session agent memory
```

---

## MCP Server

16 tools for Claude Desktop, Claude Code, Cursor, or any MCP client:

```bash
vstash-mcp                            # start MCP server
```

```json
{
  "mcpServers": {
    "vstash": {
      "command": "vstash-mcp"
    }
  }
}
```

---

## Self-Supervised Embedding Refinement

vstash can improve its own embedding model by exploiting disagreements between vector and keyword search:

```bash
vstash retrain                        # 1. Generate training pairs from your corpus
                                      # 2. Fine-tune with MNRL (needs sentence-transformers)
vstash reindex --model ~/.vstash/models/retrained  # 3. Apply the improved model
```

82% of queries produce disagreement between vector and FTS search. These disagreements are free training signal. The published model (`Stffens/bge-small-rrf-v2`) was trained this way: 76K triples, zero human labels, 30 min on a T4 GPU.

**Results**: +7.4% NDCG on SciFact, +19.5% on NFCorpus, +5.5% on SciDocs. The 33M model surpasses an untrained 110M model on 3/5 datasets.

---

## Privacy

| Component | Data leaves machine? |
|---|---|
| Embeddings (FastEmbed) | Never |
| Search (sqlite-vec + FTS5) | Never |
| Inference (Ollama/LM Studio) | Never |
| Inference (Cerebras/OpenAI) | Yes (query + context sent to API) |

Search is always private. Use a local LLM for fully private answers.

---

## Paper

[vstash: Local-First Hybrid Retrieval with Adaptive Fusion for LLM Agents](paper/vstash-paper.md)

Four contributions: adaptive RRF, self-supervised embedding refinement, negative result on post-RRF scoring, production substrate. LaTeX version at `paper/arxiv/vstash.tex`.

---

## Documentation

| Guide | Description |
|---|---|
| [How It Works](docs/how-it-works.md) | Search pipeline, chunking, RRF |
| [Configuration](docs/configuration.md) | Full TOML reference |
| [Embedding Models](docs/embedding-models.md) | Model comparison, `vstash retrain` |
| [MCP Server](docs/mcp-server.md) | 16 tools for LLM agents |
| [Experiments](docs/experiments.md) | BEIR benchmarks, ablations |

---

## Experiments

| Experiment | Key Result | Command |
|---|---|---|
| [BEIR Benchmark](experiments/beir_benchmark.py) | Beats ColBERTv2 on 3/5 datasets | `python -m experiments.beir_benchmark` |
| [Embedding Fine-tune](experiments/finetune_rrf.py) | +7.4% NDCG, zero labels | `python -m experiments.finetune_rrf` |
| [Scale Benchmark](experiments/scale_benchmark.py) | 20.9ms at 50K chunks | `python -m experiments.scale_benchmark` |
| [Relevance Signal](experiments/relevance_signal_beir.py) | F1=0.996 cross-domain | `python -m experiments.relevance_signal_beir` |

---

## What's New in v0.28

- **`vstash retrain`**: fine-tune embeddings on your own data using hybrid retrieval disagreement
- **`Stffens/bge-small-rrf-v2`**: published embedding model (+7.4% SciFact, +19.5% NFCorpus)
- **`SearchResult.added_at/collection/tags/layer`**: full metadata on search hits
- **`add_documents_batch()`**: bulk ingest in single transaction
- **Embedder provenance**: `embedding_model` stamped on fresh stores
- **Search 32% faster**: MMR cache, batch expand_context, norm precompute

See [CHANGELOG](CHANGELOG.md) for full version history.
