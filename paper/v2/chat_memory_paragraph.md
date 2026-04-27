# Chat-memory section -- canonical paragraph for paper v2

Drop-in paragraph for the chat-memory ablation section. Numbers measured
2026-04-27 on Mac local; full per-question JSON outputs live under
`experiments/results/lme_full_500_*.json`.

---

> Domain-specific fine-tuning lifts retrieval ranking at top positions
> on a 102-query stratified holdout disjoint from training: R@1 +3.4pp,
> R@3 +3.0pp, R@5 +3.8pp over vanilla BGE-small. R@10 saturates at 0.96
> across all four arms -- ranking, not coverage, is the lever. On the
> same holdout, gains concentrate per-type at R@5: multi-session
> +6.9pp, temporal-reasoning +5.6pp. The BEIR-tuned baseline (v3)
> regresses 2.8pp on temporal-reasoning, evidence that BEIR
> specialization actively damages cross-temporal chat retrieval.
> Macro R@1 (0.55, 74% of structural ceiling) remains the next
> frontier; we propose a cross-encoder reranker over top-10 as
> future work.

---

## Supporting numbers (auditable)

### Table A. R@K macro on the 102-question stratified holdout (clean)

|              | base BGE | v3 (BEIR-tuned) | lme-v1 (chat) | lme-v1-from-v3 |
|--------------|---------:|----------------:|--------------:|---------------:|
| R@1          |   0.5209 |          0.5214 |        0.5552 |         0.5454 |
| R@3          |   0.8418 |          0.8505 |        0.8716 |         0.8765 |
| R@5          |   0.8905 |          0.9025 |        0.9284 |         0.9304 |
| R@10         |   0.9634 |          0.9516 |        0.9658 |         0.9585 |
| R@20         |   0.9779 |          0.9833 |        0.9804 |         0.9853 |
| R@50         |   1.0000 |          1.0000 |        1.0000 |         1.0000 |

### Table B. Per-type R@5 on the same holdout (where the gain concentrates)

| Type                       | n  | base   | v3                | lme-v1            | lme-v1-from-v3    |
|----------------------------|----|--------|-------------------|-------------------|-------------------|
| single-session-user        | 14 | 0.929  | 1.000 (+7.1pp)    | 0.929 (0.0pp)     | 1.000 (+7.1pp)    |
| single-session-assistant   | 12 | 1.000  | 1.000             | 1.000             | 1.000             |
| single-session-preference  |  6 | 1.000  | 1.000             | 1.000             | 1.000             |
| multi-session              | 27 | 0.869  | 0.886 (+1.7pp)    | **0.938 (+6.9pp)**| 0.927 (+5.8pp)    |
| knowledge-update           | 16 | 0.969  | 1.000 (+3.1pp)    | 1.000 (+3.1pp)    | 1.000 (+3.1pp)    |
| temporal-reasoning         | 27 | 0.774  | 0.746 (-2.8pp)    | **0.829 (+5.6pp)**| 0.811 (+3.7pp)    |

### Methodology notes (anticipating reviewer scrutiny)

- **Split**: 80/20 stratified by `question_type`, deterministic seed 42.
  398 train, 102 holdout, **disjoint by construction**. Code in
  `experiments/lme_prepare_retrain.py:stratified_split`.
- **Train data**: real `(question, answer_session_ids)` labels from
  LongMemEval-s -- no synthetic queries, no LLM-generated pseudo-
  queries, no human re-labeling. Routes through
  `vstash.retrain.generate_labeled_triples_batched` (the v5 recipe).
- **Eval gate**: refuse-to-save if NDCG@10 on the holdout regresses;
  the `lme-v1` model passed the gate at NDCG@10 = 0.6878 vs base
  0.6143 (delta +0.0735, two seeded runs +-0.0006 apart).
- **Structural R@1 ceiling = 0.75**: 3 of 6 question types have 2-5
  gold sessions structurally (multi-session always has 2-5;
  knowledge-update always has 2; temporal-reasoning is 85% multi-
  gold). Macro R@1 cannot exceed (3 * 1.0 + 3 * 0.5) / 6 = 0.75 by
  construction, regardless of retriever quality.
- **What the N=500 number does NOT prove**: the full-dataset numbers
  (e.g., lme-v1 R@10 = 0.9815) include the 398 training questions and
  are **partially memorisation**. We disclose them in the appendix
  but the headline claims rest only on the 102-query holdout.

### Reproducibility

```bash
python -m experiments.lme_prepare_retrain \
    --output-db    experiments/lme_retrain_full.db \
    --output-train experiments/results/lme_train.jsonl \
    --output-eval  experiments/results/lme_eval.jsonl \
    --output-meta  experiments/results/lme_retrain_meta.json

# Train on Colab T4 (~12 min):
vstash retrain \
    --training-queries experiments/results/lme_train.jsonl \
    --eval-queries     experiments/results/lme_eval.jsonl \
    --base-model       BAAI/bge-small-en-v1.5 \
    --output           ~/.vstash/models/bge-small-rrf-lme-v1 \
    --bulk-mine --bulk-mine-device cuda

# Evaluate on full LongMemEval-s (Mac local, ~9 min):
python -m experiments.longmemeval_retrieval --all \
    --model ~/.vstash/models/bge-small-rrf-lme-v1 \
    --output experiments/results/lme_full_500_lme-v1.json
```
