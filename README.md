# vstash

[![PyPI](https://img.shields.io/pypi/v/vstash)](https://pypi.org/project/vstash/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10+-blue)]()
[![CI](https://github.com/stffns/vstash/actions/workflows/ci.yml/badge.svg)](https://github.com/stffns/vstash/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/stffns/vstash/graph/badge.svg)](https://codecov.io/gh/stffns/vstash)

**Local document memory with hybrid retrieval.** Single SQLite file. Zero cloud dependencies for search. Beats ColBERTv2 on **5/5 [BEIR](https://github.com/beir-cellar/beir) datasets** with the tuned [`bge-small-rrf-v3`](https://huggingface.co/Stffens/bge-small-rrf-v3) model. Under 60 ms p50 at 50K chunks on an Apple Silicon laptop.

```bash
pip install vstash
vstash add paper.pdf notes.md https://example.com/article
vstash search "what's the main argument?"
```

---

## Retrieval Quality

| Dataset | Docs | vstash (v3) | ColBERTv2 | BM25 | Δ vs ColBERTv2 |
|---------|:----:|:-----------:|:---------:|:----:|:--------------:|
| SciFact | 5.2K | **0.9361** | 0.693 | 0.665 | **+0.243** |
| NFCorpus | 3.6K | **0.3927** | 0.344 | 0.325 | **+0.049** |
| SciDocs | 25.7K | **0.3693** | 0.154 | 0.158 | **+0.215** |
| FiQA | 57.6K | **0.7506** | 0.356 | 0.236 | **+0.395** |
| ArguAna | 8.7K | **0.7540** | 0.463 | 0.315 | **+0.291** |

*Absolute NDCG@10 on [BEIR](https://github.com/beir-cellar/beir) via the full production retrieval pipeline (RRF hybrid + adaptive weights + MMR dedup + IDF, 2026-04-19). Tuned model: [`Stffens/bge-small-rrf-v3`](https://huggingface.co/Stffens/bge-small-rrf-v3) (33M params, 384d). v3 beats ColBERTv2 on **5/5 BEIR datasets** and improves macro NDCG@10 by +0.016 absolute over [`bge-small-rrf-v2`](https://huggingface.co/Stffens/bge-small-rrf-v2) (0.6405 vs 0.6246). Training-time eval uses a batched path that skips MMR/IDF for speed; absolute NDCG@10 differs by a few percent vs the production numbers above, but baseline-vs-final deltas are preserved. See [experiments/results/v2_v3_head_to_head.json](experiments/results/v2_v3_head_to_head.json) for the full table (reproduce via `python -m experiments.v2_v3_head_to_head`) and the methodological note in [experiments/hypotheses.md](experiments/hypotheses.md) for the pipeline-shift caveat.*

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

**The feature is maturing fast.** Each release tightens the recipe, lifts the measured numbers, and adds infrastructure that keeps the next iteration honest:

| Release | Training recipe | 5-dataset BEIR macro NDCG@10 | What landed alongside |
|---------|-----------------|------------------------------|------------------------|
| base `bge-small` | no fine-tune | 0.6118 | reference |
| [`rrf-v2`](https://huggingface.co/Stffens/bge-small-rrf-v2) | 76k triples, ad-hoc scripts | 0.6246 | first paper-grade result; still the NFCorpus specialist |
| [`rrf-v3`](https://huggingface.co/Stffens/bge-small-rrf-v3) | 60k triples via `retrain-multi` CLI, `temperature=0.5`, eval gate | **0.6405** | H-R9 ablation picked the config empirically; H-R7 seeded RNGs make it reproducible; H-R5 reports NDCG@3 + Recall@100 so regressions are visible before they ship |

Both v2 and v3 beat ColBERTv2 on **5/5 BEIR datasets** under the current pipeline. v3 improves macro by +0.016 over v2 (+2.6% relative), with the largest per-dataset gain on FiQA (+0.097 absolute). **It is a trade, not a strict upgrade**: v3 gives up ~0.040 NDCG@10 on NFCorpus vs v2 in exchange for the FiQA and SciFact wins. v2 remains the better pick for keyword-heavy / biomedical corpora where NFCorpus-style retrieval dominates; v3 is the recommended default for everything else. The eval gate also catches losers: hypothesis H-R3 (hard-negative margin filter) regressed macro -2.49pp, the candidate was refused, the branch was closed without merging. **The pipeline's job is to refuse bad models, and it does.**

See the [Retrieval Quality](#retrieval-quality) table, [docs/retrain.md](docs/retrain.md) for the full recipe and per-version breakdown, and [experiments/results/v2_v3_head_to_head.json](experiments/results/v2_v3_head_to_head.json) for reproducible numbers.

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
| [BEIR Benchmark](experiments/beir_benchmark.py) | With `bge-small-rrf-v3` (current default): 5/5 BEIR datasets beat ColBERTv2. With `-rrf-v2` (previous): 4/5 under this script's historical pipeline. See [Retrieval Quality](#retrieval-quality) for the v3 numbers. | `python -m experiments.beir_benchmark --no-chroma` |
| [Retrain (eval-gated)](docs/retrain.md) | Fine-tune your embedding model on your own corpus, refuses regressions | `vstash retrain --help` |
| [Pipeline latency](experiments/vstash_pipeline_ivfpq_bench.py) | Under 60 ms p50 @ 50K, 0.80x with snapvec-ivfpq @ 100K (Apple Silicon laptop) | `python -m experiments.vstash_pipeline_ivfpq_bench --n 100000` |
| [Relevance Signal](experiments/relevance_signal_beir.py) | F1=0.996 cross-domain | `python -m experiments.relevance_signal_beir` |

---

## What's New in v0.36

- **`chat.ask_full()` returning `AskResult`** (v0.36) -- new public API surfaces the reasoning trace and token usage that `ask()` previously discarded. Cerebras `gpt-oss-120b` populates `result.reasoning`; Ollama qwen3 thinking-mode uses `message.thinking`; OpenAI-compat servers (vLLM, DeepSeek, Together, xAI Grok, OpenAI o1/o3) read `reasoning_content`. `ask()` keeps its `-> str` contract via a thin wrapper -- zero call-site change for existing code. Also exposed as `Memory.ask_full()`. Drives Merken Phase 2 distillation pipeline.
- **Centralized store construction** (v0.36) -- `open_store_for_config(cfg)` is the single entry point used by CLI, MCP, web, SDK, journal, and federated search. Previously each surface duplicated the `StorageConfig -> VstashStore` wiring and silently dropped IVFPQ tuning fields on some paths (#297).
- **`vec_only` long-query distance cutoff fix** (v0.36) -- `retrieval_mode="vec_only"` now applies the same long-query relaxation as `hybrid`; ArguAna `vec_only` jumped from NDCG@10 = 0.0013 (1403/1406 zero) to 0.4250. Hybrid mode and all paper / model-card numbers untouched (#304).
- **Bug fixes** (v0.36) -- `Memory.add(collection=None)` falls back to schema default instead of crashing on the NOT NULL constraint (#296); `vstash retrain --synthesize-queries` no longer crashes on Ollama / Cerebras backends (#294); web uploads now persist under `~/.vstash/uploads/<uuid>-<safe-name>` instead of pointing at deleted temp paths (#295).

## What's New in v0.35

- **`bge-small-rrf-lme-v1` chat-memory specialist** (v0.35) — fine-tuned on 398 labeled LongMemEval queries through the eval-gated retrain loop. +3.79pp R@5 on n=102 holdout vs vanilla BGE-small. Use when your corpus is primarily chat sessions / agent memory.
- **Eval-gated labeled retrain** (v0.35) — `vstash retrain --training-queries train.jsonl --eval-queries eval.jsonl` accepts user-supplied `(query, relevant_paths)` JSONL and refuses to save fine-tunes that regress NDCG@10 on the holdout. See [docs/retrain.md](docs/retrain.md).
- **`vstash why` miss analysis** (v0.33) — diagnose why an expected document did not surface for a query. Traces vector pool, distance cutoff, FTS match, RRF fusion, MMR, and context-expansion stages with parameter suggestions. Auto-logs misses on empty / low-relevance searches.
- **`retrieval_mode` enum** (v0.33) — `Literal["hybrid", "vec_only", "fts_only"] = "hybrid"` on `Memory.search`, `Memory.ask`, `VstashStore.search`, and MCP tools. `vec_only` is the symmetric branch to `fts_only`. Default stays `hybrid`. Legacy `fts_only=True` boolean was removed in v0.35.
- **Custom encoder resolver hook** (v0.34) — `register_encoder_resolver(fn)` lets callers plug LoRA-adapted, locally fine-tuned, or otherwise unnamable encoders into the embed pipeline. See [docs/embedding-models.md](docs/embedding-models.md).
- **Cosine metric in `vec_chunks`** (v0.34) — sqlite-vec virtual table now uses cosine distance (was L2 before; v1 DBs migrate in place atomically on first open). Fixed a latent bug where non-unit-normalized models silently mis-ranked.
- **Persistent embedder daemon** (v0.32) — `vstash serve --warm` pre-loads the embedding model and exposes `/api/embed` on `localhost:8585`. CLI and SDK clients auto-detect and delegate; cold start drops from ~2 s to ~5 ms.

See [CHANGELOG](CHANGELOG.md) for full version history.
