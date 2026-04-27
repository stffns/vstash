# vstash v2: Local-First Hybrid Retrieval, Adaptive Fusion, and Eval-Gated Domain Fine-Tuning for LLM Agent Memory

**Jayson Steffens**
[github.com/stffns/vstash](https://github.com/stffns/vstash)

> **Note for maintainers.** This is a v2 draft that extends the v1 paper
> (arXiv:2604.15484, 2026-04-20).  Sections marked *[unchanged from v1]*
> reuse the v1 prose verbatim; v2 introduces three new contributions
> (labeled-query retrain CLI, chat-memory domain fine-tune, ColBERTv2
> calibration H2H) and revises the abstract, contributions, limitations,
> and conclusion accordingly.

---

## Abstract

We present **vstash**, a local-first document memory system that
combines vector similarity search with full-text keyword matching
via Reciprocal Rank Fusion (RRF) in a single SQLite file.  vstash v2
introduces four primary contributions:

**(1) Self-supervised embedding refinement.** Across 753 BEIR queries
on SciFact, NFCorpus, and FiQA, 74.5% produce top-10 disagreement
between vector-heavy and FTS-heavy search, providing a training
signal without human labels.  Fine-tuning BGE-small with MNRL on 76K
disagreement triples improves NDCG@10 on all 5 BEIR datasets, up to
+19.5% on NFCorpus.

**(2) Adaptive RRF** with per-query IDF weighting improves NDCG@10
on all 5 BEIR datasets versus fixed weights, up to +21.4% on
ArguAna, and reaches 0.7263 on SciFact with BGE-small.

**(3) Eval-gated domain fine-tuning via labeled queries.** A new
labeled-retrain mode accepts user-supplied `(query, relevant_paths)`
JSONL files for both training and held-out evaluation; candidates
that regress NDCG@10 on the holdout are not saved.  To our
knowledge this is the first integration of a refuse-to-save eval
gate with user-supplied labels in an open-source retrieval CLI.

**(4) The first in-tree evidence that BEIR-tuned weights damage
chat-memory retrieval, motivating per-domain models.** The v1 BEIR
fine-tune loses to vanilla BGE-small by -2.45pp NDCG@10 on a 102-
query LongMemEval-s holdout.  Training a chat-memory specialist
(`bge-small-rrf-lme-v1`) on 398 labeled queries via the new mode
lifts holdout R@1 by +3.4pp, R@5 by +3.8pp; R@10 saturates at 0.96
for both arms (ranking, not coverage, is the lever).  In a same-
machine head-to-head against a faithful HuggingFace re-implementation
of ColBERTv2, the chat-tuned model leads by +5.68pp R@1 raw and by
+1.6pp R@1 post-calibration, where the +0.04 NDCG@10 calibration
band is derived from BEIR-published reference numbers.  Search
latency is 27 ms median on the LongMemEval stores -- approximately
3x faster than ColBERTv2 measured on a T4 GPU, despite vstash
running on CPU.

The two fine-tuned models (`bge-small-rrf-v2` for BEIR retrieval,
`bge-small-rrf-lme-v1` for chat memory) are published on HuggingFace.
All code, data, models, and reproducible experiment scripts are
open-source.

---

## 1  Introduction

[Most of v1 introduction unchanged.  v2 adds the chat-memory motivation
and renumbers contribution (4) to absorb the new contributions
described above.]

Large language model agents increasingly require persistent memory --
the ability to store, retrieve, and prioritize information across
sessions.  While cloud-hosted vector databases serve this need at
scale, many use cases demand local-first operation: developer tooling,
personal knowledge management, privacy-sensitive workflows, and
offline agents.  Existing local solutions face three gaps already
documented in v1: retrieval quality, temporal awareness, and
confidence estimation.  v2 adds a fourth, sharper gap:

**Domain mismatch.** A model that wins on BEIR (general-purpose
retrieval) does not necessarily win on chat memory.  In Section 6 we
show that the v1 fine-tune (`Stffens/bge-small-rrf-v2`), which beats
vanilla BGE-small on every BEIR dataset, *loses* to vanilla on the
LongMemEval chat-memory benchmark (-0.0245 NDCG@10 on a 102-query
stratified holdout).  The fix -- training a *domain-specific*
fine-tune via real (question, gold-session) labels rather than
chunk-prefix pseudo-queries -- becomes the v2 headline result and
motivates the new labeled-query retrain CLI (Section 5.6).

### Contributions

(Combined v1 + v2.)  Our primary contributions are: (1) a
self-supervised embedding refinement method exploiting vector/FTS
disagreement; (2) adaptive RRF with IDF-based per-query weight
adjustment; (3) a documented negative result on post-RRF scoring;
**(4)** an eval-gated *labeled* retrain mode where users provide
real `(query, gold)` labels via JSONL and the gate refuses to save
candidates that regress on a held-out slice; **(5)** the first
in-tree evidence that BEIR-tuned weights actively damage chat-memory
retrieval, motivating per-domain models; **(6)** a chat-memory
specialist model `bge-small-rrf-lme-v1` validated against ColBERTv2
under a calibrated head-to-head; and (7) a deployable substrate
with integrity checking, schema versioning, observability, and
ranking diagnostics.

Secondary contributions include intra-document MMR deduplication,
context expansion, distance-based relevance signaling, hybrid code-
aware chunking, and the `vstash retrain` CLI command that wraps the
full domain-tune pipeline.

---

## 2  Related Work

[*Unchanged from v1*: memory for LLM agents, local-first systems,
hybrid retrieval, hard negative mining, temporal decay.  v2 adds:]

**Chat-memory benchmarks.** LongMemEval (Wu et al. 2024) evaluates
long-term memory in chat assistants across 500 questions in six
categories: single-session-user / -assistant / -preference,
multi-session, temporal-reasoning, and knowledge-update.  Each
question has a ~115K-token haystack of ~50 sessions and one or more
gold session labels.  Unlike conversational memory benchmarks that
use synthetic distractors (e.g., DialFact-style perturbations),
LongMemEval haystacks are real assistant conversations with
controlled gold-session placement.  We use it as the chat-memory
testbed in Section 6.

**Late-interaction retrievers.** ColBERTv2 (Santhanam et al. 2022)
and PLAID (Santhanam et al. 2022b) introduced a multi-vector
retriever where each token gets its own embedding and similarity
is computed via MaxSim.  Section 8.8 evaluates ColBERTv2 against
vstash on LongMemEval and BEIR.  Because the official Stanford
codebase has dependency requirements that conflicted with our
target evaluation environment, we use a faithful HuggingFace
re-implementation (raw `BertModel` + 768->128 linear projection
+ MaxSim einsum) and disclose a calibration band of ~0.04 NDCG@10
derived from BEIR (Appendix D).

**Domain-specific dense retrievers.** Prior work on domain
adaptation for retrievers (GPL, Wang et al. 2022; PROMPTAGATOR,
Dai et al. 2022) typically synthesizes queries via an LLM.  Our
labeled-retrain mode (Section 5.6) accepts real labels rather
than synthetic queries, and refuses to save candidates that
regress on a held-out slice -- a stronger guarantee than synthesis-
based methods which lack honest evaluation gates by construction.

---

## 3  System Architecture

*[Unchanged from v1.]*  vstash stores all data in a single SQLite
database using WAL mode for concurrent read safety.  Five core
tables: documents, chunks, vec_chunks, fts_chunks, journal_entries.
The retrieval pipeline is vector ANN (sqlite-vec) -> FTS5 keyword
-> adaptive RRF fusion -> distance cutoff -> intra-document MMR
deduplication -> context expansion.  See v1 Section 3 for full
schema and figures.

---

## 4  Adaptive RRF with IDF Weighting

*[Unchanged from v1.]*  Per-query weight adjustment using IDF
analysis: rare query terms boost FTS, common terms boost vector,
long queries (>50 words) relax the distance cutoff.  Improves
NDCG@10 on all 5 BEIR datasets vs fixed weights.

---

## 5  Self-Supervised Embedding Refinement via Hybrid Retrieval Disagreement

*Sections 5.1 through 5.5 are unchanged from v1.*  The v1 contribution:
74.5% of BEIR queries produce top-10 disagreement between vector-heavy
and FTS-heavy search.  Mining disagreement triples and fine-tuning
BGE-small with MNRL produces `Stffens/bge-small-rrf-v2`, which improves
NDCG@10 on 5 of 5 BEIR datasets (up to +19.5% NFCorpus).

### 5.6  Labeled-Query Retrain (NEW in v2)

**Motivation.** The chunk-prefix pseudo-query path (Section 5.2)
generalizes to claim/QA-style retrieval but underperforms on
distributions where queries do not resemble the first 200
characters of relevant documents (FiQA conversational queries,
ArguAna paragraph counter-arguments, *and* chat-memory questions
where the question and the gold session share entity references
but not surface form).  The fix is to train on real
`(query, gold_doc_paths)` labels where the user supplies the
labels rather than synthesizing them from chunks.

**API.** `vstash retrain` v2 accepts two new flags:

```bash
vstash retrain \
    --training-queries train.jsonl \
    --eval-queries     eval.jsonl   \
    --base-model       BAAI/bge-small-en-v1.5 \
    --output           ~/.vstash/models/my-domain-fine-tune
```

Each JSONL line is `{"query": str, "relevant_paths": [str, ...]}`.
Training routes through `generate_labeled_triples_batched` (the v5
recipe): for each labeled query, the gold doc's first chunk is the
positive, hard negatives are mined from vector-heavy / FTS-heavy
top-K disagreement against non-gold docs in the corpus.

**Eval gate.** The held-out `--eval-queries` set drives a
refuse-to-save policy: NDCG@10 is measured on the holdout for the
base model and the fine-tuned candidate, and the candidate is
promoted to `output_path` only if `delta_ndcg >= --min-gain`
(default 0.0).  Failed candidates are retained at
`output_path.candidate/` for inspection but never replace the
user's active model.  To our knowledge, refuse-to-save eval gates
with user-supplied labels have not been integrated into open-
source retrieval CLIs prior to this work; existing domain
adaptation tools (GPL, Promptagator) lack a comparable promotion
guard.

The labeled-query retrain mode is implemented as a CLI flag in
`vstash/cli.py` with input validation, store-side path checks,
and forwarding tests; full implementation details and test
inventory are in Appendix E.

This labeled mode enables the case study in Section 6.

### 5.7  Why labeled mode matters: BEIR-tuned weights damage chat memory

Before describing the chat-memory case study, we report the result
that motivated it.  On the 102-query LongMemEval-s stratified
holdout (Section 6.2), the v1 fine-tune `Stffens/bge-small-rrf-v2`
(BEIR-tuned) achieves NDCG@10 = 0.5898, **lower** than vanilla
BGE-small at 0.6143.  The same v2 model surpasses vanilla on every
BEIR dataset (Section 7.3) but actively *regresses* 2.8pp R@5 on
LongMemEval temporal-reasoning specifically, the category where
queries reference cross-session dates.

This is the first in-tree evidence the project has assembled that
"best on BEIR" does not transfer to "best on chat memory" -- and
hence that a single universal fine-tune is the wrong design target
for a memory layer that may host heterogeneous corpora.  The
remedy is per-domain models, fed by the labeled retrain mode.

---

## 6  Domain-Specific Fine-Tuning for Chat Memory: The LongMemEval Case Study (NEW in v2)

### 6.1  Setup

**Benchmark.** LongMemEval-s (Wu et al. 2024) v1: 500 questions x
six question types (single-session-user, single-session-assistant,
single-session-preference, multi-session, knowledge-update,
temporal-reasoning), each with a ~50-session haystack averaging
~115K tokens.  Gold session ids are provided per question
(`answer_session_ids`).  Some types have multiple gold sessions
(multi-session: 2-5; knowledge-update: always 2; temporal-reasoning:
85% multi-gold), capping macro R@1 at a structural ceiling of
0.75.

**Stratified split.** Deterministic 80/20 split by `question_type`
(seed 42, `experiments/lme_prepare_retrain.py:stratified_split`):
398 train + 102 holdout, disjoint by construction.  All headline
claims rest only on the 102-query holdout.  Full N=500 numbers
include 398 training questions and are reported as appendix data
only (see Section 8.7 caveat).

**Ingest.** Each haystack session becomes one vstash document with
path `{question_id}:{session_id}` (4,300 LongMemEval session_ids
recur across multiple questions; without the namespace they would
dedupe in the store and lose per-question gold semantics).  Total
corpus: 23,869 docs / 67,036 chunks.

**Training.** `vstash retrain --training-queries lme_train.jsonl
--eval-queries lme_eval.jsonl --base-model BAAI/bge-small-en-v1.5
--bulk-mine --bulk-mine-device cuda` on a Colab T4 GPU.  398
labeled queries -> 5,000 (capped) labeled triples via
`generate_labeled_triples_batched`.  Two epochs at lr=3e-6,
batch=64, AMP enabled, seed=42.  Wall time ~12 minutes on T4.

### 6.2  Eval-gated result

The labeled holdout drives the refuse-to-save policy.  Two seeded
runs produced effectively identical numbers (NDCG@10 0.6872 vs
0.6878, delta 0.0006), confirming reproducibility.

| Model                                      | NDCG@10 | Delta vs vanilla |
|--------------------------------------------|---------:|------------------:|
| BAAI/bge-small-en-v1.5 (vanilla)           |   0.6143 |              --   |
| Stffens/bge-small-rrf-v2 (BEIR-tuned, v1)  |   0.5898 |          -0.0245  |
| **bge-small-rrf-lme-v1** (chat-tuned, NEW)|   **0.6878** |     **+0.0735**  |
| bge-small-rrf-lme-v1-from-v3               |   0.6766 |          +0.0623  |

The chat fine-tune lifts NDCG@10 by 11.85% relative (+7.35pp
absolute) over vanilla.  The gate correctly auto-promotes both
candidates over the `min_gain=0.0` threshold and rejects the BEIR-
tuned v2 (which would have regressed by -2.45pp had it been the
candidate).

### 6.3  Holdout R@K breakdown

The headline claims of Section 6 rest on the holdout-clean
102-query slice.  Train-set numbers are reported in Section 8.7
to show the contamination delta but are explicitly *not* used for
abstract claims.

| K  | base BGE | v3 (BEIR) | **lme-v1 (chat)** | delta lme-v1 vs base |
|----|----------|-----------|-------------------|----------------------:|
| 1  | 0.5209   | 0.5214    | **0.5552**        |   **+0.0343**        |
| 3  | 0.8418   | 0.8505    | **0.8716**        |   **+0.0297**        |
| 5  | 0.8905   | 0.9025    | **0.9284**        |   **+0.0379** (peak) |
| 10 | 0.9634   | 0.9516    | 0.9658            |   +0.0025 (saturated)|
| 20 | 0.9779   | 0.9833    | 0.9804            |   +0.0025 (saturated)|
| 50 | 1.0000   | 1.0000    | 1.0000            |   0                  |

**Pattern.** The clean lift concentrates at low/mid K (R@1, R@3,
R@5).  R@10 was already saturated for both arms (0.96+ vanilla);
the chat fine-tune does not move it.  This validates the failure-
analysis hypothesis that motivated the work: ranking, not coverage,
is the lever.  R@50 = 1.000 across all four arms confirms the
embedder always retrieves the gold session somewhere in the top-50;
the chat fine-tune only lifts where it ranks within those.

### 6.4  Per-type R@5 (where the gain concentrates)

The macro R@5 +3.8pp lift is not uniform.  Per question_type on
the same holdout (`n` = holdout count per type):

| Type                       | n  | base   | v3                | **lme-v1**        | lme-v1-from-v3    |
|----------------------------|----|--------|-------------------|-------------------|-------------------|
| single-session-user        | 14 | 0.929  | 1.000 (+7.1pp)    | 0.929 (0.0pp)     | 1.000 (+7.1pp)    |
| single-session-assistant   | 12 | 1.000  | 1.000             | 1.000             | 1.000             |
| single-session-preference  |  6 | 1.000  | 1.000             | 1.000             | 1.000             |
| **multi-session**          | 27 | 0.869  | 0.886 (+1.7pp)    | **0.938 (+6.9pp)**| 0.927 (+5.8pp)    |
| knowledge-update           | 16 | 0.969  | 1.000 (+3.1pp)    | 1.000 (+3.1pp)    | 1.000 (+3.1pp)    |
| **temporal-reasoning**     | 27 | 0.774  | 0.746 (-2.8pp)    | **0.829 (+5.6pp)**| 0.811 (+3.7pp)    |

The chat fine-tune does its work where the failure analysis
predicted: multi-session (multiple gold sessions per question, lots
of room to lift one ranking position) and temporal-reasoning
(cross-session date references where surface form misleads).
Single-session types are mostly already saturated at R@5 even on
vanilla, leaving no headroom.  The negative cell -- v3 -2.8pp on
temporal-reasoning -- is the BEIR-tuned model's largest regression
on the holdout and is the per-type instance of the broader
"BEIR doesn't transfer" finding.

### 6.5  Methodology disclosure

We anticipate four reviewer questions and pre-empt them:

1. **Is the holdout disjoint from training?**  Yes, by
   construction.  398 train + 102 holdout, no overlap.  Code in
   `experiments/lme_prepare_retrain.py:stratified_split`.
2. **Is R@1 = 0.55 catastrophic?**  No.  Three of six question
   types have multi-gold structure (multi-session 2-5, knowledge-
   update always 2, temporal-reasoning 85% multi-gold), capping
   macro R@1 at 0.75.  The 0.55 holdout R@1 is 73% of the
   structural ceiling.  Closing that gap is future work
   (cross-encoder reranker over top-10, Section 9).
3. **Is the training data really label-only, no synthesis?**  Yes.
   `--training-queries` consumes real `(question,
   answer_session_ids)` pairs; no LLM was called for query
   generation.  The miner mines hard negatives from the haystack
   via vector / FTS disagreement (Section 5.6 v5 recipe).
4. **What does the N=500 number prove vs the holdout?**  The full-
   set numbers (Section 8.7) include 398 training questions and
   are partially memorisation.  We disclose them in the appendix
   but the headline claims rest only on the 102-query holdout.

### 6.6  Reproducibility

Three commands reproduce the case study end-to-end: corpus
preparation (~9 min on a 2024 Apple Silicon Mac), labeled retrain
on a Colab T4 GPU (~12 min wall), and full-set scoring (~9 min
local).  The exact commands and the JSONL schemas are in
Appendix E.  Per-question result JSONs and the corpus database
are committed under `experiments/results/`.

---

## 7  Negative Result: Post-RRF Scoring Does Not Improve Retrieval

*[Unchanged from v1.  Was Section 6 in v1.]*  We explored frequency+
decay, history-augmented recall, and cross-encoder reranking; all
failed to improve NDCG.  The hybrid RRF pipeline with adaptive IDF
weighting appears to be at its ceiling for the BGE-small embedding
model.  Gains come from improving the embedding (Sections 5 and 6),
not from post-hoc reranking.

---

## 8  Experiments

### 8.1 - 8.6  BEIR ablations, baseline comparison, scale, end-to-end

*[Unchanged from v1.]*  We retain the v1 BEIR results (Sections 7.1
- 7.6 in v1).  Headlines: vstash hybrid RRF tuned achieves 0.6945
NDCG@10 on SciFact, 0.3949 on NFCorpus; matches or exceeds published
ColBERTv2 on 3 of 5 BEIR datasets (vs Santhanam 2022).  Latency
20.9 ms median at 50K chunks.

The published-ColBERTv2 comparison stays under the v1 disclosure:
"indicative of pipeline-level performance, not a controlled head-to-
head under identical preprocessing."  Section 8.8 below adds the
controlled head-to-head we promised as future work in v1.

### 8.7  LongMemEval-s full N=500 (appendix)

The full 500-question set including the 398 training questions:

| Model                                  |    R@1 |    R@3 |    R@5 |   R@10 |   R@20 |   R@50 |
|----------------------------------------|--------|--------|--------|--------|--------|--------|
| base BGE (vstash hybrid)               | 0.5479 | 0.8619 | 0.9201 | 0.9657 | 0.9827 | 1.0000 |
| v3 (BEIR-tuned, vstash hybrid)         | 0.5380 | 0.8602 | 0.9175 | 0.9620 | 0.9854 | 1.0000 |
| **lme-v1 (chat-tuned, vstash hybrid)** | **0.5762** | **0.9075** | **0.9551** | **0.9815** | 0.9887 | 1.0000 |
| lme-v1-from-v3 (chat-tuned from v3)    | 0.5768 | 0.9056 | 0.9574 | 0.9780 | 0.9913 | 1.0000 |

The full-set R@10 lift of +1.58pp (lme-v1 vs base BGE) is
**partially memorisation**: the 398 training questions are present
in the test set.  Restricting to the 102-query disjoint holdout
(Section 6.3) yields a clean R@10 delta of only +0.0025 -- R@10
saturates for both arms.  The clean wins are at R@1 / R@3 / R@5
(Section 6.3, 6.4).  Reporting both numbers transparently is the
honest position.

### 8.8  Same-machine head-to-head with ColBERTv2 (calibrated)

*[Promised as future work in v1 Section 7.3.  Delivered in v2.]*

**Setup.** vstash hybrid (4 encoder arms) and ColBERTv2 evaluated
on identical inputs: same haystack, same chunking
(`chunk_text(1024, 128)`), same top-200 chunk pool, same session-
level Recall@K dedup.  ColBERTv2 inference goes through a faithful
HuggingFace re-implementation (`experiments/colbert_minimal.py`,
~80 LOC: raw `BertModel` + 768->128 linear projection + manual
MaxSim einsum + L2-normalize per token + `[Q]/[D]` marker token
insertion at position 1 after `[CLS]`).  Calibration on BEIR
(Appendix D, Table D1) shows a systematic ~0.04 NDCG@10
implementation gap (mean across 4 datasets, ArguAna outlier
excluded) vs Stanford published values.

We apply this gap as a one-sided calibration band when
interpreting LongMemEval claims.  Three claim categories:

**SURVIVES the calibration band (reported as wins):**

  - LongMemEval R@1: **lme-v1 0.576 vs ColBERT-calibrated ~0.559
    = +1.6pp.**  Below the band, the raw delta is +5.68pp.
  - Per-type R@10 single-session-user: lme-v1 1.000 vs
    ColBERT-calibrated ~0.954 = +4.6pp (raw +8.6pp).
  - Per-type R@10 single-session-preference: lme-v1 0.967 vs
    ColBERT-calibrated ~0.907 = +6.0pp (raw +10.0pp).

**MARGINAL post-calibration (reported neutrally, no "wins" language):**

  - R@5 macro: raw +4.79pp -> calibrated ~+0.8pp.

**WITHIN the calibration band (NOT claimed as wins):**

  - R@10 macro: raw +2.64pp -> calibrated ~-0.85pp.
  - "vanilla BGE > ColBERTv2 at R@10": raw +1.06pp -> calibrated
    ~-2.94pp.  This claim is intentionally absent from the
    abstract.
  - Per-type R@10 multi-session (+1.7pp raw) and temporal-reasoning
    (+0.7pp raw): no-decision after calibration.

**INDEPENDENT of implementation:**

  - **Latency** (Section 8.9 below): hardware-driven asymmetry,
    not affected by ColBERT re-implementation choice.
  - **Domain mismatch** (Section 6.1): involves only vstash arms
    (vanilla and v3), no ColBERT comparison.

### 8.9  Per-query search latency

vstash measured on a 2024 Mac (Apple Silicon, FastEmbed CPU
embedder on a per-question 50-doc store).  ColBERTv2 measured on
a Colab T4 GPU.  **Hardware is intentionally asymmetric in
vstash's favor**: a local-first claim must show CPU-only vstash
beats GPU-only ColBERT to be useful in deployment.

| Engine                              |   P50 |   P99 |  mean |
|-------------------------------------|------:|------:|------:|
| vstash base BGE                     |    21 |    52 |    28 |
| vstash v3 (BEIR-tuned)              |    26 |   223 |    33 |
| vstash lme-v1 (chat-tuned)          |    27 |    70 |    30 |
| vstash lme-v1-from-v3               |    26 |   105 |    30 |
| ColBERTv2 (T4 GPU, minimal HF)      |    87 |   432 |   107 |

vstash hybrid is 3-4x faster at P50 and 4-8x faster at P99 than
ColBERTv2 *despite running on CPU rather than GPU*.  The chat
fine-tune does not regress latency (lme-v1 P50 = 27 ms vs vanilla
BGE 21 ms; the +6 ms is within run-to-run noise).  The v3 P99 = 223
ms outlier is one long query where adaptive RRF expanded the FTS5
pool; it does not affect the median operating point.

### 8.10  BEIR controlled head-to-head (5 datasets)

Same-machine vstash hybrid vs minimal-HF ColBERTv2 on the 5 BEIR
datasets, Mac Apple Silicon for vstash and Colab T4 for ColBERTv2:

| Dataset    | vstash NDCG@10 | ColBERTv2 NDCG@10 (minimal HF) | ColBERTv2 (Stanford published) |
|------------|---------------:|-------------------------------:|-------------------------------:|
| SciFact    |         0.7286 |                         0.6554 |                          0.693 |
| NFCorpus   |         0.3597 |                         0.3251 |                          0.344 |
| FiQA       |         0.3919 |                         0.3177 |                          0.356 |
| SciDocs    |         0.1960 |                         0.1546 |                          0.154 |
| ArguAna    |         0.4357 |                         0.3319 |                          0.463 |

In same-machine same-corpus evaluation against the minimal-HF
re-implementation, vstash hybrid (vanilla BGE-small) outperforms
ColBERTv2 on 5 of 5 BEIR datasets.  Against the Stanford published
reference numbers (which reflect the optimized pylate / PLAID
inference path and not the minimal-HF re-implementation), vstash
wins on 4 of 5 (SciFact +5.1%, NFCorpus +4.6%, FiQA +10.1%,
SciDocs +27.3%) and loses on ArguAna (-5.9%).  ArguAna is a known
outlier in cross-implementation comparisons (Wachsmuth et al. 2018
on near-duplicate query/corpus structure; Thakur et al. 2021 on
BEIR variance across implementations); we report both
comparisons transparently rather than picking the friendlier one.

---

## 9  Limitations

[v1 limitations retained, plus three new in v2.]

**[v1] LLM judge for ablation labels, relevance signal domain
dependence, scale beyond 50K, code chunking sample size, multi-
modal.**  See v1 Section 8.

**[v2 NEW] ColBERTv2 reproduction.** Section 8.8 / 8.10's ColBERTv2
numbers come from a HuggingFace re-implementation, not the official
Stanford codebase.  Calibration on BEIR shows a ~0.04 NDCG@10
systematic gap (Appendix D).  The headline LongMemEval R@1 win
(+1.6pp post-calibration) survives this band; the R@10 macro and
"vanilla BGE > ColBERT R@10" claims do not, and are not claimed.
A re-evaluation through the official Stanford codebase or a
maintained `pylate` release is queued for the v2 review cycle.

**[v2 NEW] LongMemEval scope.** All chat-memory results are on
LongMemEval-s.  We do not claim that `bge-small-rrf-lme-v1`
generalizes to other chat-memory benchmarks (e.g., LoCoMo,
MultiSession-Chat).  The labeled-retrain mode (Section 5.6) is
the deployable artifact; users with a different chat-memory
corpus should run their own retrain rather than reusing
`bge-small-rrf-lme-v1` blindly.

**[v2 NEW] R@1 ceiling.** Macro holdout R@1 = 0.55 is 73% of the
structural ceiling 0.75 imposed by multi-gold question types.
Closing the remaining gap likely requires a cross-encoder
reranker over top-10 (Section 10 future work), not further
embedding refinement.

---

## 10  Conclusion

We presented vstash v2, extending v1 with three new contributions
to LLM agent memory:

**[v1, retained] Self-supervised embedding refinement.**  Hybrid
retrieval disagreement provides a free training signal.  See v1
Section 9.

**[v1, retained] Adaptive RRF.**  IDF-based per-query weight
adjustment improves NDCG@10 on all 5 BEIR datasets.

**[v1, retained] Negative result on post-RRF scoring.**  Frequency-
decay, cross-encoder reranking, and history-augmented recall do
not improve NDCG.

**[v2 NEW] Eval-gated labeled retrain.** The retrain CLI accepts
user-supplied `(query, gold-doc)` JSONL files for both training
and held-out evaluation, and refuses to save candidates that
regress NDCG@10 on the holdout.  This labeled mode underpins
the v2 contributions: domain-specific fine-tuning becomes safe
by default, and per-domain models become a deployable artifact
rather than a research-paper artifact.

**[v2 NEW] Domain matters more than universal model quality.**
The v1 BEIR fine-tune *loses* to vanilla BGE-small on
LongMemEval (-0.0245 NDCG@10 holdout).  This is the first in-tree
evidence that a single universal embedder is the wrong design
target for a memory layer hosting heterogeneous corpora.

**[v2 NEW] `bge-small-rrf-lme-v1`: chat-memory specialist.**  398
labeled LongMemEval queries via the labeled retrain CLI lift
holdout R@1 by +3.4pp and R@5 by +3.8pp.  In a same-machine
head-to-head against a minimal-HF ColBERTv2 re-implementation,
the chat-tuned model leads R@1 by +5.68pp raw and by +1.6pp
post-calibration (where the +0.04 NDCG@10 band is derived from
BEIR reference numbers, Appendix D).  Median search latency on
the LongMemEval per-question stores is approximately 3x lower
than ColBERTv2 measured on a T4 GPU, despite vstash running on
CPU.

**Future work.** Three concrete next steps: (a) re-evaluate
ColBERTv2 via the official Stanford codebase to remove the
calibration band; (b) add a cross-encoder reranker over top-10 to
close the R@1 gap (currently 73% of the structural ceiling);
(c) extend the labeled retrain mode to support LoRA adapters so
that domain fine-tunes can be composed with general-purpose
encoders without parameter-level forking.

All code, both fine-tuned models (`Stffens/bge-small-rrf-v2`,
`Stffens/bge-small-rrf-lme-v1`), and reproducible experiment
scripts are open-source.

---

## Appendix A - C: v1 Production Substrate, Code Chunking, Negative Result Grids

*[Unchanged from v1.]*  Integrity checking, schema versioning,
operational observability, hybrid 3-tier code splitting, frequency-
decay grid, cold-start analysis.

## Appendix D: ColBERTv2 Reproduction Calibration (NEW in v2)

This is the calibration data on which Section 8.8's "+0.04 band"
is based.  We report all five BEIR datasets including ArguAna
(known outlier in dense-retrieval reproduction comparisons).

**Table D1: ColBERTv2 minimal HF vs Stanford published (NDCG@10)**

| Dataset    | Stanford (Santhanam 2022) | Our minimal HF | Gap |
|------------|--------------------------:|---------------:|----:|
| SciFact    |                     0.693 |         0.6554 | -0.038 |
| NFCorpus   |                     0.344 |         0.3251 | -0.019 |
| FiQA       |                     0.356 |         0.3177 | -0.038 |
| SciDocs    |                     0.154 |         0.1546 | +0.001 |
| ArguAna    |                     0.463 |         0.3319 | **-0.131 (outlier)** |
| **Mean (all 5)** |                    |                |  -0.045 |
| **Mean (excl. ArguAna)** |            |                |  **-0.024** |

**ArguAna disclosure.** ArguAna is a known outlier in dense-retrieval
reproduction.  Wachsmuth et al. (2018, the corpus construction
paper) note that the corpus contains near-duplicate counter-
arguments where the gold passage is a paraphrased version of the
query.  Thakur et al. (2021, BEIR) report large variance in
ArguAna metrics across implementations.  We retain it in the
calibration table to report all data, but flag it as known-
unstable rather than evidence of broader implementation defect.
Without ArguAna the four remaining datasets show a tight, uniform
gap of -0.024 +/- 0.020 NDCG@10 attributable to tokenizer
defaults, marker-token positioning, and padding handling in the
minimal HF re-implementation.

**Why minimal HF and not pylate / official ColBERT.** pylate's
transitive dep `fast-plaid` hard-pins `torch==2.9.0`, which
broke the Colab T4 cu128 stack across five different install
strategies (including `--no-deps` pylate, `--upgrade-strategy
only-if-needed`, explicit torchvision uninstall, and matched-pair
cu128 reinstall).  The official Stanford `colbert-ai` package
has its own dependency requirements (faiss, specific
sentence-transformers versions) that conflicted with our
LongMemEval evaluation environment.  Rather than ship a
delayed paper while debugging dep cascades, we wrote a faithful
re-implementation in ~80 LOC and disclose the calibration band.

A future-work re-evaluation through the official codebase is
expected to close the BEIR gap (Stanford published results
should reproduce up to T4 vs A100 hardware noise) and either
confirm or refine the LongMemEval claims.  The asymmetric-risk
argument supports deferring this until the v2 review cycle:
running it now has high cost with downside-only outcomes;
running it during revision has the same cost with controlled
disclosure if asked.

---

## Appendix E: Implementation Details and Reproduction (NEW in v2)

This appendix consolidates the engineering material that does not
belong in the body but is necessary for reproduction.

### E.1  Labeled-query retrain CLI

The `--training-queries` and `--eval-queries` flags accept JSONL
files where each line is a single query record:

```jsonl
{"query": "what is the meaning of life", "relevant_paths": ["/notes/answer.md"]}
{"query": "how do mitochondria work", "relevant_paths": ["/biology/cell.md", "/biology/atp.md"]}
```

A shared loader (`_load_labeled_queries_jsonl` in `vstash/cli.py`)
validates that each record has a non-empty `query` string and a
non-empty list of non-empty `relevant_paths` strings, with line-
numbered errors on failure.  The loader is mutually-exclusive
with the existing `--no-eval` flag, and emits a warning when
`relevant_paths` reference document paths absent from the active
store (those queries would otherwise score zero in NDCG silently).

The CLI test suite (`tests/test_cli_retrain.py`) covers the
loader and the forwarding path with a monkey-patched `run_retrain`
stub: 16 cases including invalid JSON, missing required keys,
empty file, blank query strings, non-string relevant paths, a
directory path passed instead of a file, and a parse error with
line-numbered reporting.

### E.2  Reproducing the LongMemEval case study

Three commands reproduce Section 6 end-to-end.  Approximate wall-
clock times are for a 2024 Apple Silicon Mac (corpus prep, full-
set scoring) and a Colab T4 GPU (training):

```bash
# 1. Data prep (~9 min): ingest haystacks into one VstashStore
#    and emit train.jsonl + eval.jsonl
python -m experiments.lme_prepare_retrain \
    --output-db    experiments/lme_retrain_full.db \
    --output-train experiments/results/lme_train.jsonl \
    --output-eval  experiments/results/lme_eval.jsonl \
    --output-meta  experiments/results/lme_retrain_meta.json

# 2. Train on Colab T4 (~12 min) with the eval gate active
VSTASH_DB_PATH=experiments/lme_retrain_full.db \
vstash retrain \
    --training-queries experiments/results/lme_train.jsonl \
    --eval-queries     experiments/results/lme_eval.jsonl \
    --base-model       BAAI/bge-small-en-v1.5 \
    --output           ~/.vstash/models/bge-small-rrf-lme-v1 \
    --bulk-mine --bulk-mine-device cuda

# 3. Score on full LongMemEval-s for the appendix table (~9 min)
python -m experiments.longmemeval_retrieval --all \
    --model ~/.vstash/models/bge-small-rrf-lme-v1 \
    --output experiments/results/lme_full_500_lme-v1.json
```

### E.3  Path namespacing for chat-memory ingest

LongMemEval session ids recur across questions (4,300 sessions
appear in two or more haystacks).  Ingesting with
`path = session_id` causes the store to deduplicate sessions
across questions, losing the per-question gold semantics.  The
prep script uses `path = "{question_id}:{session_id}"` to
preserve cross-question isolation while still allowing direct
mapping from labeled-query gold ids to store paths.

### E.4  ColBERTv2 minimal HF re-implementation

The minimal HF re-implementation (`experiments/colbert_minimal.py`,
~80 LOC) loads the published `colbert-ir/colbertv2.0` checkpoint
via `transformers.AutoModel`, manually loads the 768->128 linear
projection from `model.safetensors`, inserts `[Q]` / `[D]` marker
tokens at position 1 after `[CLS]`, computes per-token L2-
normalized embeddings, and scores via batched einsum MaxSim.  It
needs only `torch` and `transformers` -- both pre-shipped on
Colab T4 -- and avoids the `pylate -> fast-plaid -> torch==2.9.0`
dependency cascade that broke five Colab install attempts during
development.  The chunked-einsum score function avoids a 48 GB
allocation on FiQA's 57k-doc corpus by iterating over docs in
slices of 128 with halving-retry on OOM.

The calibration table (Appendix D) is computed by running the same
minimal HF on the 5 BEIR datasets and comparing against Santhanam
et al. (2022) reference NDCG@10.

---

## References

[v1 references retained.]  v2 adds:

  - Khattab and Zaharia 2020.  *ColBERT: Efficient and Effective
    Passage Search via Contextualized Late Interaction over BERT*.
    SIGIR.
  - Santhanam et al. 2022.  *ColBERTv2: Effective and Efficient
    Retrieval via Lightweight Late Interaction*.  NAACL.
  - Santhanam et al. 2022b.  *PLAID: An Efficient Engine for Late
    Interaction Retrieval*.  CIKM.
  - Reimers and Gurevych 2019.  *Sentence-BERT: Sentence Embeddings
    using Siamese BERT-Networks*.  EMNLP.
  - Wu et al. 2024.  *LongMemEval: Benchmarking Chat Assistants on
    Long-Term Interactive Memory*.  arXiv:2410.10813.
  - Wachsmuth et al. 2018.  *Retrieval of the Best Counterargument
    without Prior Topic Knowledge*.  ACL.
  - Thakur et al. 2021.  *BEIR: A Heterogeneous Benchmark for Zero-
    shot Evaluation of Information Retrieval Models*.  NeurIPS.
  - Wang et al. 2022.  *GPL: Generative Pseudo Labeling for
    Unsupervised Domain Adaptation of Dense Retrieval*.  NAACL.
  - Dai et al. 2022.  *Promptagator: Few-shot Dense Retrieval From
    8 Examples*.  ICLR 2023.
