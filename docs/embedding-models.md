# Embedding Models

vstash uses [FastEmbed](https://github.com/qdrant/fastembed) for local embeddings — no API calls, no server, fully in-process.

---

## Supported Models

### English-only

| Model | Dimensions | Speed | Quality | Notes |
|-------|-----------|-------|---------|-------|
| `BAAI/bge-small-en-v1.5` (default) | 384 | ~700 ch/s | Great | Best speed/quality ratio |
| `Stffens/bge-small-rrf-v2` | 384 | ~700 ch/s | **Best** | Self-tuned, +7-19% NDCG vs base, beats ColBERTv2 on 3/5 BEIR |
| `BAAI/bge-base-en-v1.5` | 768 | ~300 ch/s | Excellent | |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | ~300 ch/s | Excellent | |

> **New in v0.28:** `Stffens/bge-small-rrf-v2` is a BGE-small model fine-tuned with vstash's own hybrid retrieval disagreement signal. Same dimensions, same speed, better quality. Use `vstash reindex --model Stffens/bge-small-rrf-v2` to switch, or run `vstash retrain` to fine-tune on your own data.

### Multilingual

| Model | Dimensions | Languages | Speed | Quality |
|-------|-----------|-----------|-------|---------|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 50+ | ~600 chunks/s | Great |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | 50+ | ~250 chunks/s | Excellent |
| `intfloat/multilingual-e5-large` | 1024 | 100+ | ~100 chunks/s | Best |

**Recommendation:** For **general-purpose** multilingual corpora (conversational text, news, encyclopedic content, cross-lingual search over everyday topics), use `paraphrase-multilingual-MiniLM-L12-v2`. Same 384 dimensions as the default model = zero latency impact on search. Cross-lingual query similarity improves ~40% over English-only models.

---

## Known weakness: `paraphrase-multilingual-MiniLM` on clinical / specialized domains

Observed in production use on a medical-document corpus (msdlocal). This model has a **specific, reproducible weakness** that is not captured by standard MTEB benchmarks:

**Symptom.** Vector distances for clinical-domain queries are diffuse — cosine distances collapse into a narrow range around 0.85–1.00 even for genuinely relevant documents. The default distance cutoff (1.15× best distance) then either eliminates relevant results or passes almost everything, producing either empty result sets or noise.

**Why it happens.** The model was trained on general-purpose paraphrase pairs; specialized vocabularies (clinical, legal, heavily-jargoned technical domains) push query and document embeddings into a crowded neighborhood where the model cannot discriminate. Cross-lingual recall (e.g. Spanish query against English document) is notably worse than the model's aggregate benchmarks suggest.

**Diagnostic signal.** If `miss_analysis()` (v0.21) consistently reports `vector_search: not_found` or `distance_cutoff: failed` on queries where the target document is provably FTS-reachable, that is the fingerprint of this weakness. Run the same query with `fts_only=True` (v0.26, see `Memory.search`) — if it surfaces the document, the vector side is the problem, not your corpus or your query.

**Mitigations** (in order of effort):

1. **Use `fts_only=True` for the query.** If the term you care about is literal (drug name, diagnosis code, SKU), keyword search alone is often sufficient and the fastest fix. See the `fts_only` parameter on `Memory.search()` and `Memory.ask()`.
2. **Relax the distance cutoff.** Pass a larger `distance_cutoff` (e.g. 1.5 or 2.0) on specific queries to let more vector candidates reach RRF fusion. FTS5 + RRF will usually re-rank the noise down.
3. **Pin the RRF weights toward FTS.** Use `vec_weight=0.2, fts_weight=0.8` on `Memory.search()` to trust keyword matching over the diffuse vector signal for this query only — adaptive RRF will still kick in for the next query with default weights.
4. **Switch models.** `paraphrase-multilingual-mpnet-base-v2` (768 dims) handles specialized vocabularies noticeably better at ~2× the embedding cost. `intfloat/multilingual-e5-large` is the strongest multilingual option we have tested.
5. **Run `vstash reindex --model <better-model>`** to re-embed the whole corpus with a stronger model. One-time cost, permanent fix.

**What does not reliably help.** Increasing `top_k` expands the candidate pool (it scales as `top_k × 10`), which can marginally help when distances are *borderline* — a few more chunks survive the cutoff and compete in RRF. But in the pathological diffuse-embedding case (cosine ~0.85–1.00), every candidate exceeds the cutoff ratio and a larger `top_k` changes nothing. Boosting recency (`recency_boost`) is strictly post-RRF — it multiplies a score that was already too low to rank, so it cannot rescue chunks the vector path eliminated.

**Current state.** Since v0.26, vstash **does** auto-detect this failure mode at runtime: when the vector pool is empty after the distance cutoff and FTS5 has results, the pipeline transparently collapses to FTS-only scoring with `vec_weight=0.0, fts_weight=1.0` and increments the `adaptive_rrf_vector_empty_fallback_total` metric. See `docs/observability.md` for the alerting recipe and [#156](https://github.com/stffns/vstash/issues/156) for the design discussion.

---

## Changing the Model

In `vstash.toml`:

```toml
[embeddings]
model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

Then re-embed your existing chunks:

```bash
vstash reindex
```

The `reindex` command drops and recreates the vector index with the new model's dimensions, re-embeds all chunks in batches, and shows a progress bar. Text, metadata, and FTS index are preserved.

Options:

```bash
vstash reindex                          # uses model from vstash.toml
vstash reindex --model BAAI/bge-base-en-v1.5  # override model
vstash reindex --batch-size 128         # smaller batches (less RAM)
vstash reindex --yes                    # skip confirmation prompt
```

---

## When to upgrade dimensions

| Corpus size | Recommended dims | Model |
|-------------|:---:|-------|
| < 5,000 chunks | 384 | `bge-small-en-v1.5` or `multilingual-MiniLM` |
| 5,000 - 50,000 | 768 | `bge-base-en-v1.5` or `multilingual-mpnet` |
| 50,000+ | 1024 | `multilingual-e5-large` |

Higher dimensions give better discrimination between semantically similar chunks, but the improvement is negligible on small corpora. Search latency scales linearly with dimensions (~0.3ms at 384, ~0.7ms at 1024).

---

## Backend Selection

On Apple Silicon Macs, vstash can use MLX for GPU-accelerated embeddings:

```toml
[embeddings]
backend = "mlx"   # "onnx" (default, portable) | "mlx" (Apple Silicon) | "auto"
```

`auto` (the default) uses MLX when available, falling back to ONNX.
