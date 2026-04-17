# Retrain module roadmap

Living design doc for extending `vstash/retrain.py`. Tiered by ROI / effort.
Owner: Jay + Claude. Not shipped (under `experiments/`, excluded from sdist).

## Current state (v0.32)

- 247 LOC self-supervised fine-tuning via MNRL.
- Signal: first 200 chars of each chunk becomes a pseudo-query, the chunk itself
  is the positive, first vec/FTS top-5 disagreement provides 1 hard negative.
- Validated on 5 BEIR datasets: +4.5% NDCG avg, +18.3% NFCorpus, +5% SciFact.
  Released as `stffens/bge-small-rrf-v1` and `v2` on HuggingFace.
- CLI: `vstash retrain` with `--quick`, `--max-queries`, `--epochs`, `--lr`,
  `--batch-size`, `--base-model`, `--output`.
- Tests: `tests/test_retrain.py` covers triple generation and training wrapper
  with mocked `sentence_transformers`.

## Limitations driving this roadmap

1. Queries do not look like real queries (chunk prefixes, not short human text).
2. Only 1 triplet emitted per pseudo-query (top-5 disagreement has more signal).
3. Positives lack diversity (always the source chunk itself).
4. Hard negatives are not quality-filtered (first disagreement wins).
5. Training is blind: no hold-out, no NDCG delta, no regression guard.
6. Only MNRL loss. No `CachedMNRL`, no `GISTEmbedLoss`, no MarginMSE.
7. No reranker training path (stack is bi-encoder only).
8. No continual / incremental retrain on delta corpus.
9. Lineage gap: `training_meta.json` records hyperparameters but not corpus
   fingerprint, disagreement rate, or eval delta.
10. No HF Hub push helper, no integration with embedder daemon hot-swap.

---

## Tier 1 (high ROI, ~1-2 days each)

### T1.1 Eval-gated training

**Goal:** never save a model that is worse than the base on the user's corpus.

**Design:**
- New function `split_corpus_for_eval(store, eval_fraction=0.15, seed=42)`
  returns `(train_chunk_ids, held_out_rows)` where each held-out row is
  `{query: str, relevant_path: str}`.
- New function `evaluate_model(model_name, store, held_out, top_k=10)` loads
  the model, embeds each held-out query, runs `store.search(top_k)`, computes
  binary NDCG@10 against the single relevant path. Returns
  `{"ndcg@10": float, "mrr": float, "hit@10": float, "n": int}`.
- `train_mnrl` gains a `baseline_metrics` and `final_metrics` block in
  `training_meta.json`.
- Top-level `retrain()` helper composes: split -> baseline eval -> generate
  -> train -> final eval -> gate on `min_gain`.
- CLI flags: `--min-gain 0.0`, `--no-eval`, `--eval-fraction 0.15`.
- If `final - baseline < min_gain`: print delta table, do NOT save to
  `output_path`, exit 1. `--min-gain -1` acts as "always save".

**Files:**
- `vstash/retrain.py`: add `split_corpus_for_eval`, `evaluate_model`, wrap
  composition into a new `retrain()` entry function (used by CLI).
- `vstash/cli.py`: wire flags, print delta table.
- `tests/test_retrain.py`: tests for split determinism, ndcg computation,
  gate pass/fail, `--no-eval` path.

**Risk:** reindexing temporarily with a new model to evaluate is expensive on
large stores. Mitigation: evaluate by re-embedding held-out queries only and
comparing search results against the existing vector index. That is not
strictly fair (base vs fine-tuned produce different query embeddings), but
captures the delta at the decision point: "will this model improve retrieval
as a drop-in query encoder?".

**Exit criteria:**
- `vstash retrain` on empty / trivial store exits cleanly with "not enough
  data for eval".
- On a BEIR SciFact-sized corpus, runs end-to-end, prints delta, saves model
  only when delta >= `--min-gain`.
- Tests cover split determinism, NDCG@10 math, gate behavior.

---

### T1.2 Multi-triplet emission

**Goal:** extract 3-5x more training signal from the same disagreement data.

**Design:**
- `generate_triples` gains `triplets_per_query: int = 1` (default preserves
  current behavior).
- When > 1: iterate over top-K disagreements; each disagreeing path produces
  one triplet (query, positive, hard_neg_i). Skip duplicates on negative text.
- Log disagreement histogram so users see "your corpus produces 2.3 triplets
  per query on average".

**Files:**
- `vstash/retrain.py`: refactor the hard-negative loop to yield multiple.
- `vstash/cli.py`: add `--triplets-per-query` (default 5 in retrain command).
- `tests/test_retrain.py`: existing tests stay green with default=1; new test
  asserts K triplets emitted when K disagreements exist.

**Risk:** larger training sets increase epoch time. Benchmark and document.

**Exit criteria:**
- Existing tests untouched.
- New test verifying 5 triplets emitted from a crafted 5-way disagreement.
- Run on SciFact shows > 1.5 avg triplets/query.

---

### T1.3 LLM query synthesis

**Goal:** replace chunk-prefix pseudo-queries with short, realistic queries
produced by a local LLM. Closes the query-chunk distribution gap that limits
the current approach.

**Design:**
- New module boundary: `vstash/retrain_synth.py` (keeps the loss-free query
  generation separate from training).
- Function `synthesize_queries(chunks, backend="local", n_per_chunk=2,
  cache_path=None)` -> list of `{chunk_id, query}` dicts. Uses the existing
  `chat.py` local LLM backend (Ollama / LM Studio / OpenAI-compatible).
- Prompt (InPars-style, short):
  > "Given the following passage, write 2 short queries (5 to 15 words each)
  > that a user would type to find it. Output JSON: [\"q1\", \"q2\"]."
- JSONL cache keyed by `(chunk_id, prompt_hash, model)` so re-runs are free.
- `generate_triples` gains `query_source="prefix" | "synthesized" | "both"`.
  When "synthesized": replace pseudo-query with LLM output; retain the
  disagreement mining logic unchanged.
- CLI: `--synthesize-queries`, `--synth-model <name>`, `--synth-n 2`.

**Files:**
- `vstash/retrain_synth.py` (new).
- `vstash/retrain.py`: thread `query_source` parameter.
- `vstash/cli.py`: wire flags.
- `tests/test_retrain_synth.py` (new): mocked LLM backend; cache hit/miss;
  malformed JSON resilience.

**Risk:** LLM failures / rate limits on large corpora. Mitigations: JSONL
cache, graceful skip on parse failure, timeout per call.

**Exit criteria:**
- Generates realistic short queries for a sample chunk (manual sanity check).
- Cached runs re-use cache.
- End-to-end `vstash retrain --synthesize-queries` works on SciFact and
  produces a model with NDCG@10 >= baseline.

---

## Tier 2 (high impact, ~1 week each)

### T2.4 Cross-encoder reranker training

`vstash retrain-reranker` using the same disagreement data. Small cross
(~10 MB) on top of the bi-encoder typically adds 3-8 NDCG points. Requires
a new `rerank()` call in the search pipeline (opt-in) and a separate model
file under `~/.vstash/models/reranker/`.

### T2.5 GISTEmbedLoss option

Teacher-guided hard negative filtering; 2024 SOTA for small embedding
models. Add `--loss {mnrl, gist}` flag, teacher model defaults to
`sentence-transformers/all-mpnet-base-v2`. Drop-in replacement for MNRL in
`train_mnrl`.

### T2.6 Continual retrain

`--continue-from <model_path>` plus "retrain on delta corpus since last run".
Turns retrain into a workflow rather than a one-off. Needs a timestamp
watermark in `training_meta.json` and a corpus diff query.

---

## Tier 3 (strategic)

### T3.7 Model lineage / registry

New `models` table in the store: `(id, path, base_model, corpus_fingerprint,
disagreement_rate, ndcg_baseline, ndcg_final, trained_at)`. CLI:
`vstash models list`, `vstash models use <id>`. Answers "which model is best
for profile X?".

### T3.8 HF Hub push helper

`vstash retrain --push-to-hub stffens/bge-small-rrf-v3` plus auto model card
(training pairs, base model, evals, corpus description). Formalizes what is
already happening manually.

### T3.9 Distillation mode

Train small model to match a big model's top-K rankings on the user's
corpus. Different goal (speed under constant recall, not calibrated gains).
Loss: MarginMSE from cross-encoder scores.

---

## Execution order

1. **T1.1 first.** Without eval, every other change is blind. Foundation.
2. **T1.2 second.** Cheap, independent, multiplies data for every later run.
3. **T1.3 third.** Biggest quality lever, but only meaningful once T1.1 can
   prove gains.
4. Tier 2 gated on Tier 1 completion + a clean baseline eval on SciFact.
5. Tier 3 reopens after Tier 2 lands.

## Tracking

Tier 1 items tracked as in-session tasks. Check off in this doc as each
lands on `develop`.
