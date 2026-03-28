# Paper Roadmap

Status tracker for addressing limitations before submission.

## Priority 1 — Required for submission

### 1. FTS keyword-OR in RRF hybrid search
- **What:** Replace exact-phrase FTS5 matching with per-word quoted keyword-OR in `store.py` RRF search
- **Why:** Ablation shows FTS keyword-OR (0.817 NDCG@5) outperforms current RRF (0.521). The phrase-match safety tradeoff costs ~36% retrieval quality. Per-word quoting preserves injection safety.
- **Impact:** Product improvement + paper results improve
- **Effort:** 1-2h code + 30min re-run ablation
- **Status:** [x] Done (2026-03-27) — RRF NDCG@5 improved from 0.521 → 0.688 (+26.22% vs vector-only)
- **Files:** `vstash/store.py` (search method FTS5 query construction), `experiments/ablation_rrf.py`

### 2. Diverse corpus evaluation
- **What:** Add 1-2 non-technical corpora to the ablation (Wikipedia articles, news, or mixed-domain docs)
- **Why:** Current corpus (24 LLM memory papers) is thematically homogeneous. FTS keyword dominance may not hold on diverse corpora where semantic paraphrases matter more. Reviewers will question generalizability.
- **Impact:** Validates (or nuances) the ablation findings across domains
- **Effort:** 2-3h (curate corpus, ingest, run ablation)
- **Status:** [x] Done (2026-03-28) — 17 Wikipedia articles (2,602 chunks). RRF wins on both corpora (0.814 / 0.758 NDCG@5).
- **Files:** `experiments/diverse_corpus.py`, `experiments/results/diverse_corpus.json`

### 3. Complete relevance labels (pooling)
- **What:** Expand from 5 graded judgments per query to full pooled labels. Take union of top-10 results from all 3 methods (vector, FTS, RRF), manually annotate each for relevance.
- **Why:** Partial labels cause NDCG>1.0 at deeper cutoffs and weaken metric validity. Pooling is standard IR practice.
- **Impact:** All tables become unassailable. Can report NDCG@10 confidently.
- **Effort:** 3-4h (mostly manual annotation)
- **Status:** [x] Done (2026-03-28) — LLM-as-judge (84 labels) + document-level deduplication in ndcg_at_k. NDCG now bounded ≤1.0.
- **Files:** `experiments/scoring_grid.py` (EVAL_QUERIES, ndcg_at_k), `experiments/llm_judge.py`

### 4. Cold start curve
- **What:** New experiment measuring how many queries (access events) are needed before scoring overtakes baseline RRF
- **Why:** Scoring depends on access history. A new corpus has zero history → β contributes nothing. Reviewers will ask "how long until this helps?"
- **Impact:** Quantifies the warm-up period, sets user expectations
- **Effort:** 1h (new experiment script)
- **Status:** [x] Done (2026-03-27) — With α=0.5/β=0.5, scoring hasn't crossed over after 3700 accesses (20 rounds). Gap narrows from -17% → -10%. Validates default scoring=disabled.
- **Files:** `experiments/cold_start.py`, `experiments/results/cold_start.json`

## Priority 2 — Nice to have

### 5. Multiple embedding models
- **What:** Re-run ablation with bge-base-en-v1.5 (768-dim) and all-MiniLM-L6-v2 (384-dim)
- **Why:** Shows whether findings generalize across embedding quality levels
- **Effort:** 2h (re-ingest per model, re-run)
- **Status:** [ ] Not started

### 6. Larger chunking evaluation
- **What:** Evaluate code-aware vs naive on 50+ real files from popular GitHub repos
- **Why:** Current eval (2 files, 8 queries) is too small for strong quantitative claims
- **Effort:** 3-4h
- **Status:** [ ] Not started

## Priority 3 — Future work (mention in paper only)

### 7. Scale benchmark (10K-1M chunks)
- Not urgent — vstash targets personal/local use. Mention SQLite WAL + batching as mitigations.

### 8. Learned re-ranking
- Bayesian optimization or multi-armed bandit for α/β/λ per-user adaptation. Research-grade effort.

### 9. Multi-modal embeddings
- CLIP for images, table-aware chunking for spreadsheets. Different scope entirely.

## Execution order

```
1. FTS keyword-OR fix     ──► re-run ablation  ✓ DONE
2. Cold start curve       ──► new experiment    ✓ DONE
3. Diverse corpus         ──► re-run ablation (now with keyword-OR)  ✓ DONE
4. Complete labels        ──► re-run scoring_grid + ablation         ✓ DONE
5. Update paper           ──► new tables, revised §8, updated §9    ✓ DONE
```

## Done

### 1. FTS keyword-OR in RRF hybrid search ✓
- Completed 2026-03-27
- Changed `store.py` FTS5 from exact-phrase to per-word keyword-OR
- Re-ran ablation: RRF NDCG@5 0.521 → 0.688, now +26.22% vs vector-only
- All 280 tests pass

### 4. Cold start curve ✓
- Completed 2026-03-27
- 20 rounds of non-uniform usage simulation (Zipf-weighted queries)
- Key finding: with α=0.5/β=0.5, scoring needs significant history before helping
- Gap narrows from -17.1% (round 1) to -10.1% (round 20) over 3700 accesses
- Validates default scoring=disabled and validates the scoring_grid's pre-set scenarios

### 3. Complete relevance labels (pooling) ✓
- Completed 2026-03-28
- LLM-as-judge (Qwen 3.5:9B) annotated 84 document-query pairs on 0–3 scale
- Document-level deduplication in ndcg_at_k fixes NDCG > 1.0 issue
- EVAL_QUERIES expanded from 5 to 3–9 labels per query

### 2. Diverse corpus evaluation ✓
- Completed 2026-03-28
- 17 Wikipedia articles across 6 domains (history, science, sports, tech, arts, geography)
- 2,602 chunks, 10 cross-domain queries
- RRF wins on both corpora: 0.814 (papers) and 0.758 (Wikipedia) NDCG@5

### 5. Paper updated ✓
- Tables 2a/2b with pooled labels and dual-corpus results
- Limitations section revised (corpus homogeneity → corpus breadth)
- Abstract updated with two-corpus evaluation
- Cold start §8.6 added
