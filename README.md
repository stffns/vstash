# vstash

[![PyPI](https://img.shields.io/pypi/v/vstash)](https://pypi.org/project/vstash/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10+-blue)]()
[![tests](https://img.shields.io/badge/tests-900+_passing-brightgreen)]()

**Local document memory with hybrid retrieval.** Single SQLite file. Zero cloud dependencies for search. Beats ColBERTv2 on SciFact, NFCorpus, SciDocs, and FiQA ([BEIR](https://github.com/beir-cellar/beir), 4 of 5 datasets). Under 60 ms p50 at 50K chunks.

```bash
pip install vstash
vstash add paper.pdf notes.md https://example.com/article
vstash search "what's the main argument?"
```

---

## Retrieval Quality

| Dataset | Docs | vstash (tuned) | ColBERTv2 | BM25 | vs ColBERTv2 |
|---------|:----:|:--------------:|:---------:|:----:|:------------:|
| SciFact | 5.2K | **0.744** | 0.693 | 0.665 | **+7.3%** |
| NFCorpus | 3.6K | **0.409** | 0.344 | 0.325 | **+18.9%** |
| SciDocs | 25.7K | **0.197** | 0.154 | 0.158 | **+27.9%** |
| FiQA | 57.6K | **0.377** | 0.356 | 0.236 | **+5.8%** |
| ArguAna | 8.7K | 0.440 | **0.463** | 0.315 | -5.0% |

*NDCG@10 on [BEIR](https://github.com/beir-cellar/beir). Tuned model: `Stffens/bge-small-rrf-v2` (33M params, 384d). [`Stffens/bge-small-rrf-v3`](https://huggingface.co/Stffens/bge-small-rrf-v3) (2026-04-19 update) retrains with 2x volume and lifts macro NDCG@10 another +5.35%. Reproducible via `python -m experiments.beir_benchmark`.*

---

## How It Works

```
Query --> Embed --+--> Vector ANN (sqlite-vec) --+
                  |                               +--> Adaptive RRF --> MMR Dedup --> Results
                  +--> FTS5 BM25 ----------------+
```

- **Hybrid search**: vector + keyword, fused via Reciprocal Rank Fusion.
- **Adaptive RRF**: IDF-based per-query weights. Rare terms boost keywords, common terms boost vectors.
- **MMR dedup**: diverse sections from long documents, not redundant chunks from one.
- **Self-tuned, gated**: `vstash retrain` fine-tunes embeddings from your own disagreement signal; the eval gate refuses regressions.

---

## Install

```bash
pip install vstash                    # SDK + search
pip install 'vstash[ingest]'          # + PDF, DOCX, PPTX parsing
pip install 'vstash[serve]'           # + web UI (vstash serve)
pip install 'vstash[all]'             # everything
```

---

## Usage

```bash
# Ingest: files, folders, URLs
vstash add report.pdf ~/notes/ https://arxiv.org/abs/2310.06825

# Search: local, no API key
vstash search "what is the proposed method?"

# Ask: needs a local LLM, auto-detects Ollama / LM Studio
vstash ask "summarize the key findings"
vstash chat                           # interactive

# Fine-tune on your own corpus (eval-gated, refuses regressions)
vstash retrain
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

vstash can tune its own embedding model to your corpus, without any human labels.

```bash
vstash retrain                        # generate training pairs + fine-tune
vstash reindex --model ~/.vstash/models/retrained
```

**How it works, in one paragraph.** When you search your corpus, the vector and keyword halves of the pipeline sometimes rank different documents at the top. Those disagreements are a free signal: the document each half picked is probably relevant, the one only one half picked might not be. vstash turns this into training pairs and fine-tunes the embedding model on them. The run is eval-gated: it evaluates the candidate against the base model on a held-out slice of your corpus and refuses to save a model that performs worse.

**Published results.** [`Stffens/bge-small-rrf-v2`](https://huggingface.co/Stffens/bge-small-rrf-v2) was trained this way from 76K pairs across three BEIR datasets in 30 min on a T4 GPU. [`Stffens/bge-small-rrf-v3`](https://huggingface.co/Stffens/bge-small-rrf-v3) (2026-04-19) retrains with the [H-R9](experiments/retrain_roadmap.md) winning config (`temperature=0.5, total_triples=60000`) for a cleaner +5.35% macro NDCG@10 gain. See the [Retrieval Quality](#retrieval-quality) table and [docs/retrain.md](docs/retrain.md) for the full recipe.

Requires `sentence-transformers`, `torch`, and `accelerate`:

```bash
pip install 'sentence-transformers>=3' torch 'accelerate>=1.1.0'
```

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

Adaptive RRF, self-supervised embedding refinement, a negative result on post-RRF scoring, and the production substrate all in one place. PDF build at `paper/arxiv/vstash.pdf`.

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
| [BEIR Benchmark](experiments/beir_benchmark.py) | Beats ColBERTv2 on 4/5 BEIR datasets (SciFact, NFCorpus, SciDocs, FiQA) | `python -m experiments.beir_benchmark --no-chroma` |
| [Retrain (eval-gated)](docs/retrain.md) | Fine-tune your embedding model on your own corpus, refuses regressions | `vstash retrain --help` |
| [Pipeline latency](experiments/vstash_pipeline_ivfpq_bench.py) | Under 60 ms p50 @ 50K, 0.80x with snapvec-ivfpq @ 100K | `python -m experiments.vstash_pipeline_ivfpq_bench --n 100000` |
| [Relevance Signal](experiments/relevance_signal_beir.py) | F1=0.996 cross-domain | `python -m experiments.relevance_signal_beir` |

---

## What's New in v0.32

- **Persistent embedder daemon** (v0.32) — `vstash serve` pre-loads the embedding model and exposes `/api/embed` on `localhost:8585`. CLI and SDK clients auto-detect and delegate; cold start drops from ~2 s to ~5 ms.
- **Query LRU cache** (v0.31) — opt-in repeated-query cache via `[cache] query_cache_size`. Roughly 700x on cache hits, automatically invalidated on writes.
- **Batched directory ingest** (v0.31) — single-transaction writes with deferred FTS. 5x faster at 500 docs versus per-file ingest.
- **`snapvec-ivfpq` vector backend** (v0.30) — IVFPQ with fp16 rerank. Pareto-dominant over sqlite-vec at N >= 50K: 0.80x mean latency at 100K, NDCG within noise.

See [CHANGELOG](CHANGELOG.md) for full version history.
