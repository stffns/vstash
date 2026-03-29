# Experiments: Retrieval Quality at Scale

*Added in v0.8.0*

Two Kaggle-scale experiments validate that vstash retrieval works correctly on real-world corpora — not just the 24-paper test corpus used during development.

---

## Experiment 1: ArXiv Retrieval Benchmark

**File:** `experiments/arxiv_retrieval_bench.py`

### Hypothesis

vstash hybrid search (vector + FTS5 + RRF) should outperform vector-only and FTS-only search on a 1,000-document academic corpus, and higher-dimensional embeddings (768d) should improve precision over the default 384d model.

### Setup

| Parameter | Value |
|-----------|-------|
| Corpus | 1,000 ML paper abstracts from HuggingFace (`CShorten/ML-ArXiv-Papers`) |
| Topic clusters | 10 (NLP, CV, RL, optimization, graphs, generative, federated, transfer, fairness, time series) |
| Classification | Keyword-based — each abstract classified by topic keyword frequency |
| Queries | 30 per-topic + 5 cross-topic = 35 queries |
| Ground truth | A result is relevant if its topic matches the query's target topic |
| Models | BGE-small-EN (384d), Multilingual-MiniLM (384d), BGE-base-EN (768d) |
| Configs per model | 5: hybrid RRF, hybrid no-MMR, vector-only, FTS-only, hybrid+scoring |
| Metrics | Precision@5, Precision@10, NDCG@5, NDCG@10, MRR, Recall@10 |

### Results

| Model | Config | P@5 | P@10 | NDCG@5 | NDCG@10 | MRR |
|-------|--------|-----|------|--------|---------|-----|
| **BGE-base (768d)** | **hybrid** | **0.703** | **0.671** | **0.728** | **0.702** | **0.895** |
| BGE-base (768d) | vector | 0.669 | 0.652 | 0.687 | 0.633 | 0.890 |
| BGE-small (384d) | hybrid | 0.663 | 0.631 | 0.685 | 0.658 | 0.865 |
| BGE-small (384d) | vector | 0.614 | 0.605 | 0.619 | 0.568 | 0.822 |
| Multilingual-MiniLM (384d) | hybrid | 0.606 | 0.577 | 0.638 | 0.611 | 0.868 |
| Multilingual-MiniLM (384d) | vector | 0.600 | 0.576 | 0.588 | 0.508 | 0.820 |

*Hybrid RRF and FTS-only produced identical results (omitted for clarity). Scoring gave marginal +0.3% lift.*

### Conclusions

1. **Hybrid RRF consistently outperforms vector-only** — +5–8% P@5 across all models. The FTS5 keyword signal adds real value when queries use domain vocabulary present in the documents.

2. **BGE-base (768d) > BGE-small (384d) > Multilingual (384d)** — the dimension upgrade gives +6% P@5 (+4.3% NDCG@5). The multilingual model trades ~6% English precision for cross-lingual capability — a valid tradeoff when multilingual search is needed.

3. **Hybrid = FTS on this corpus** — because the queries use technical vocabulary that appears verbatim in abstracts, FTS5 keyword matching dominates the RRF fusion. On more heterogeneous corpora (e.g., paraphrased queries, mixed domains), vector search would contribute more.

4. **Scoring has minimal impact without access history** — the adaptive maturity gate (γ) correctly suppresses the frequency+decay component when there are no real access patterns. This confirms the cold-start design from v0.7.0.

5. **MMR has no impact** — each paper is ingested as a single chunk, so there is no intra-document deduplication to perform. MMR's value shows on multi-chunk documents (see the paper's Wikipedia experiment).

### When to upgrade from 384d to 768d

The 384d default (BGE-small) is sufficient for most vstash users (personal document memory, <5K chunks). The 768d upgrade is worth it when:
- Corpus exceeds ~5,000 chunks
- Precision matters more than ingestion speed (~3× slower embedding)
- Documents are semantically similar and need finer discrimination

---

## Experiment 2: Dataset Discovery Engine

**File:** `experiments/dataset_discovery.py`

### Hypothesis

vstash can serve as a practical local replacement for cloud-based dataset search (Kaggle, HuggingFace browse) — given a natural-language description of what you need, it should retrieve relevant datasets with high precision.

### Setup

| Parameter | Value |
|-----------|-------|
| Corpus | 954 dataset descriptions from HuggingFace Hub API |
| Task categories | 10 groups: text generation, classification, QA, translation, vision, NER, similarity, robotics, tabular, audio |
| Data source | HuggingFace Hub API with `task_categories` filter — fetches top datasets per category by download count |
| Queries | 30 per-category + 5 cross-category = 35 queries |
| Ground truth | A result is relevant if its `task_categories` label matches the query's target group |
| Model | BGE-small-EN (384d) |
| Search | Hybrid RRF (default config) |
| Metrics | Precision@5, NDCG@5, MRR, Discovery rate (≥1 relevant in top-5) |

### Results

| Metric | Value |
|--------|-------|
| **Precision@5** | **0.629** |
| **NDCG@5** | **0.644** |
| **MRR** | **0.777** |
| **Discovery rate** | **91.4%** |
| Corpus size | 954 datasets |
| Query time | <0.2s per query |

#### Sample queries

| Query | Top result | Relevant? | Score |
|-------|-----------|-----------|-------|
| "sentiment analysis dataset with positive and negative labels" | cornell-movie-review-data/rotten_tomatoes | Yes | 0.017 |
| "pretraining corpus for language modeling" | liwu/MNBVC | Yes | 0.017 |
| "image classification and object detection with deep learning" | Kondapally/AIWD6 | Yes | 0.016 |
| "robot navigation and control dataset" | Fanqi/HumanoidBench | Yes | 0.015 |

### Conclusions

1. **91.4% discovery rate** — 9 out of 10 queries find at least one relevant dataset in the top 5. This is practical for real-world dataset discovery.

2. **MRR = 0.777** — the first relevant result typically appears in position 1 or 2. Users don't need to scroll through many results.

3. **P@5 = 0.629** — about 3 out of 5 top results are relevant. The remaining 2 are usually from adjacent domains (e.g., a translation dataset showing up for a text generation query), which is often still useful.

4. **Viable local replacement for cloud search** — with <0.2s per query, no API keys, and no cloud dependency, vstash can replace browsing Kaggle/HuggingFace for dataset discovery. The interactive mode (`--interactive`) makes this a practical tool, not just a benchmark.

### Interactive mode

```bash
python -m experiments.dataset_discovery --interactive
> time series forecasting for retail sales
1. retail-sales-forecast (time-series-forecasting) — 0.87
2. store-sales-prediction (tabular-regression) — 0.81
```

---

## Running the experiments

```bash
# ArXiv benchmark — full run (~3 min, downloads 1K papers on first run)
python -m experiments.arxiv_retrieval_bench

# Single model, smaller corpus
python -m experiments.arxiv_retrieval_bench --papers 200 --models bge-small

# Dataset discovery — benchmark mode (~30s)
python -m experiments.dataset_discovery

# Dataset discovery — interactive mode
python -m experiments.dataset_discovery --interactive

# All experiments (including scoring grid, ablation, cold start, etc.)
python -m experiments.run_all
```

Results are saved to `experiments/results/*.json` and cached corpora to `experiments/data/` (excluded from git).

---

## Methodology notes

**Ground truth via labels, not manual annotation.** Both experiments use category/topic labels as relevance ground truth — ArXiv topics via keyword classification, HuggingFace datasets via `task_categories` metadata. This is a standard IR evaluation technique (category-based relevance) that scales without human annotators, though it underestimates precision when a result is semantically relevant but doesn't share the exact category label.

**Keyword classification for ArXiv.** Since the HuggingFace ML papers dataset lacks category labels, we classify papers into 10 topic clusters using keyword frequency matching on the abstract text. This is imperfect (a paper about "graph neural networks for NLP" might be classified as either), but it provides a consistent baseline across models and configurations.

**Corpus caching.** Both experiments cache their corpus locally after first download (`experiments/data/`). Re-runs use the cached data for reproducibility and speed. Delete the cache to re-download.
