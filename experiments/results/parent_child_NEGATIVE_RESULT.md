# Parent-Child Chunking — Negative Result

**Date**: 2026-04-07
**Branch**: `feat/parent-child-chunking` (abandoned, not merged)
**Status**: Feature **not shipped**. Code preserved in branch history.

## TL;DR

Parent-child chunking — embedding small "child" chunks for precise
matching while returning large "parent" chunks for rich LLM context —
**did not improve answer quality** in any tested scenario when measured
rigorously with paired statistical analysis.

We tested three corpus types (short BEIR docs, medium BEIR docs, and a
full novel), three chunking ratios (1:1 baseline, ~2 children/parent,
~5 children/parent), and used a real LLM judge (llama3.1-8b via
Cerebras) on N=100 paired queries per BEIR dataset and N=30 paired
queries on a 750KB book.

In every paired comparison the 95% Wilson confidence interval for the
win rate **included 50/50**, with paired t-test p-values between 0.58
and 0.89.  The feature was abandoned.

## Why we tried it

The idea is well-known in the RAG literature (LangChain
ParentDocumentRetriever, Dify, etc.) and has an intuitive appeal:

- Small embeddings are more focused → better semantic matching
- Large parent chunks give the LLM more context to reason over
- Best of both worlds, supposedly

The vstash hypothesis was that this would be especially valuable for
agentic memory use cases where the LLM needs surrounding context to
synthesize good answers, even if NDCG@10 stayed flat.

## What we measured

Three experiments, all in this directory:

### 1. NDCG ablation on BEIR (`parent_child_ablation.json`)

Standard NDCG@10 retrieval metric on full BEIR test sets (300 queries
each).

| Dataset | baseline | pc_512 | pc_256 | pc_128 |
|---------|---------:|-------:|-------:|-------:|
| SciFact | 0.7263 | -0.06% | **+0.77%** | **+0.94%** |
| NFCorpus | 0.3589 | -0.70% | +0.03% | **-2.56%** |

**Reading**: SciFact shows tiny gains, NFCorpus shows tiny losses.
Domain-dependent, not consistent.

### 2. Answer quality with mean scores (n=20-30, exploratory)

`parent_child_answer_quality_*.json` — initial run with small N. Showed
+2.78% on SciFact and +4.76% on NFCorpus for pc_128. **This turned
out to be sample-size noise** when re-tested with paired analysis.

### 3. Paired answer quality with N=100 per dataset

`parent_child_paired_*.json` — the rigorous test.  For each query we
compared the two configs side by side and counted wins/losses/ties.

**SciFact (N=100)**:

| Variant | wins | losses | ties | win rate (excl ties) | 95% CI | p-value |
|---------|-----:|-------:|-----:|---------------------:|-------:|--------:|
| pc_256 | 8 | 6 | 86 | 57.1% | [32.6%, 78.6%] | 0.58 |
| pc_128 | 9 | 8 | 83 | 52.9% | [31.0%, 73.8%] | 0.77 |

**NFCorpus (N=100)**:

| Variant | wins | losses | ties | win rate (excl ties) | 95% CI | p-value |
|---------|-----:|-------:|-----:|---------------------:|-------:|--------:|
| pc_256 | 15 | 14 | 71 | 51.7% | [34.4%, 68.6%] | 0.89 |
| pc_128 | 16 | 15 | 69 | 51.6% | [34.8%, 68.0%] | 0.69 |

**Reading**: The vast majority of queries (69-86%) produce the **exact
same score** under both configs.  Of the queries where there's a
difference, it's almost a perfect coin flip.  No detectable signal.

### 4. The "long document" final test (`parent_child_long_doc.json`)

The fairest possible test: a single 750KB book (Pride and Prejudice,
~150K tokens) ingested as one document, with 30 narrative queries that
should benefit from broader context.

| Variant | wins | losses | ties | win rate | 95% CI | p-value |
|---------|-----:|-------:|-----:|---------:|-------:|--------:|
| pc_256 | 6 | 4 | 20 | 60.0% | [31.3%, 83.2%] | 0.85 |

**Mean scores**: baseline 2.433 vs pc_256 2.400 — pc_256 was actually
*lower* on average.  20/30 queries were ties.

## Why we think it didn't work

Three plausible mechanisms, none verified:

1. **BGE-small is robust to chunk size variation.**  The embedding
   model was trained on chunks of varying lengths, so a 1024-token
   chunk's embedding is not significantly noisier than a 256-token
   chunk's embedding.  The "precision" gain we expected from smaller
   children doesn't materialize.

2. **The LLM extracts what it needs regardless of context size.**  A
   2.5B-parameter judge looking at 1024 tokens vs 4×256 tokens of the
   same content produces the same answer score.  The "richer context"
   benefit is real for humans but invisible to a small LLM that's good
   enough at extraction.

3. **The matched-child highlighting doesn't change LLM behavior.**  We
   designed parent-child so the LLM gets the parent text with the
   matched child highlighted.  In practice, the LLM treats it as one
   blob of text — the highlighting metadata is information for *human*
   readers, not models.

We did not falsify these explanations rigorously.  They are guesses
consistent with the data.

## What we kept

- The **`_thread_local._stem_conn` close fix** in `VstashStore.close()`
  (a real production bug found while chasing test ResourceWarnings).
- The **`test_frontmatter.py` fixture leak fix**.
- The **`test_mcp.py` singleton cleanup fix**.
- This document and the JSON results, as a record of what was tried
  and what was learned.

## What we deleted

The full parent-child chunking branch (`feat/parent-child-chunking`):

- `parent_chunks` table and the `chunks.parent_id` column
- `add_document_with_parents()` API on `VstashStore`
- `split_into_children()` in `vstash/ingest.py`
- `child_size`/`child_overlap` in `ChunkingConfig`
- The aggregation pass in `VstashStore.search()`
- `matched_child_id`/`matched_child_text` on `SearchResult`
- The CLI/web/MCP plumbing for matched_child highlighting
- The `parent_child_*.py` experiment scripts (preserved in branch
  history, not in develop)

The branch is preserved on the remote for archaeology but will not be
merged.

## What this confirms

This is the second negative result we've shipped honestly:

1. **v0.18.0** — frequency+decay scoring was removed after BEIR showed
   it didn't improve NDCG (-1.6% on SciFact).  The hypothesis: usage
   history correlates with relevance.  It doesn't.

2. **v0.21 (this)** — parent-child chunking was abandoned after paired
   answer-quality analysis showed it was indistinguishable from baseline.
   The hypothesis: small-chunk precision + large-chunk context > either
   alone.  It isn't, at least not measurably with current models.

Both ideas were intuitively appealing.  Both failed empirical validation.
The pattern is worth remembering: **embedding-based retrieval is a
narrow and tightly-coupled system, and most "obvious" architectural
improvements turn out to be neutral once you control for noise.**

## How to reproduce

The experiment scripts live in the abandoned branch
`feat/parent-child-chunking`.  To replay:

```bash
git checkout feat/parent-child-chunking
python -m experiments.parent_child_paired --dataset scifact --queries 100
python -m experiments.parent_child_paired --dataset nfcorpus --queries 100
python -m experiments.parent_child_long_doc --queries 30
```

You will need a Cerebras API key in `CEREBRAS_API_KEY` and the BEIR
caches in `experiments/data/`.
