# Canonical data sources for paper tables

This file is the single source of truth for which JSON in `experiments/results/`
produces which table, figure, or inline number in `paper/vstash-paper.md`.
Add an entry every time a paper claim is backed by data, and update the
relevant row whenever an experiment is re-run.

The motivation is concrete: PR #221 uncovered that one release commit
accidentally overwrote `arxiv_bench.json` from a 1,000-paper run to a
100-paper run, breaking the reproducibility of Table 7. With this file,
a reviewer or future maintainer can regenerate every paper number in
one hop.

## Paper table map

| Paper section | Source JSON | Producing script | Notes |
|---|---|---|---|
| Abstract: 74.5% disagreement | `rrf_training_pairs.stats.json` | `experiments/rrf_training_pairs.py` (`TOP_K=10`) | Per-dataset rates 63.4/73.4/86.7 on SciFact/NFCorpus/FiQA |
| Abstract: +21.4% on ArguAna | see Table 3 | `experiments/beir_benchmark.py` | |
| Abstract: 0.7263 on SciFact | see Table 3 | `experiments/beir_benchmark.py` | |
| Abstract: +19.5% on NFCorpus | see Table 6 | `experiments/beir_benchmark.py` + finetuned run | (0.3949-0.3304)/0.3304 |
| Abstract: 50,425 queries | see Table 2 | sum of two rightmost columns of Table 2 | |
| Section 3.4: 2.64x expansion, +0.12 ms | `e2e_improvements.json` | `experiments/e2e_improvements.py` | Keys: `expand_context_text_increase.detail`, `expand_overhead_ms` |
| Section 3.6 Table 1 (pilot, N=20) | `relevance_signal_analysis.json` | `experiments/relevance_signal.py` | Legacy 10+10 pilot study |
| Section 3.6 Table 2 (BEIR relevance signal) | `relevance_signal_beir.json` | `experiments/relevance_signal_beir.py` | Per-dataset F1 + bootstrap CI |
| Section 4.2 Table 3 (adaptive vs fixed RRF) | `beir_multi.json` + `beir_adaptive_baseline.json` | `experiments/beir_benchmark.py` | Fixed run is the baseline JSON |
| Section 5.2 disagreement table | `rrf_training_pairs.stats.json` | `experiments/rrf_training_pairs.py` | Same source as abstract |
| Section 5.4 Table 4 (TripletLoss progression) | historical, see commit messages | `experiments/finetune_rrf.py` early runs | SciFact-only; TripletLoss delta is `-91.5%` = `(0.0550-0.6464)/0.6464` |
| Section 6.1 (-1.6% / -9.0% scoring deltas) | `scoring_lifecycle_scifact.json` | `experiments/scoring_lifecycle.py` | Deltas computed from `baseline.all` vs `summary.final_adaptive.all` and `summary.final_fixed.all`; the JSON stores the raw NDCG, not the pre-computed deltas |
| Section 7.2 Table 5a (LLM memory corpus ablation) | `ablation.json` | `experiments/ablation_rrf.py` | 24 docs, 10 queries, LLM judge |
| Section 7.2 Table 5b (Wikipedia ablation) | `diverse_corpus.json` | `experiments/diverse_corpus.py` | 17 articles, 10 queries |
| Section 7.3 Table 6 (BEIR vs baselines) | `beir_multi.json` + finetuned run + published numbers | `experiments/beir_benchmark.py` with `--finetuned-model` | BGE-small base RRF row is the baseline JSON; tuned row uses `Stffens/bge-small-rrf-v2` |
| Section 7.4 Table 7 (1,000 ArXiv papers) | `arxiv_bench.json` (corpus_size=1000) | `experiments/arxiv_retrieval_bench.py` | See "historical note" below on this file |
| Section 7.5 Table 8 (latency across corpus sizes) | `scale_benchmark.json` for 5K/10K/50K rows; smaller rows from pipeline-latency one-off measurements | `experiments/scale_benchmark.py` | Rightmost column is P99, not max. 786- and 1,087-chunk rows come from one-off pipeline measurements consistent with Appendix E.3 |
| Section 7.5 Table 9 (NDCG stability at scale) | `scale_benchmark.json` | `experiments/scale_benchmark.py` | 1K/5K/10K/50K points; same JSON as Table 8 at-scale rows |
| Section 7.6 Table 10 (end-to-end answer quality) | `answer_relevance_scifact.json`, `answer_relevance_nfcorpus.json` | `experiments/answer_relevance.py` | LLM judge Qwen 3.5 9B, N=30 per dataset |
| Appendix C.1 Table C1 (scoring grid) | output of `experiments/scoring_grid.py` | `experiments/scoring_grid.py` | 16 configs x 5 access scenarios. "Best Scenario" peak is `benchmark_focused` for all top rows |
| Appendix C.2 (cold start) | `adaptive_cold_start.json`, `adaptive_cold_start_wikipedia.json` | `experiments/adaptive_cold_start.py` | 120-article Wikipedia corpus, Zipf-weighted |
| Appendix E.1 Table E1 (EmbeddingGemma) | `embedding_gemma_eval.json` | `experiments/embedding_gemma_eval.py` | 5 BEIR datasets |
| Appendix E.2 Table E2 (MMR effect) | `ablation_pipeline.json` | `experiments/ablation_pipeline.py` | 24 docs, compares hard dedup / MMR / no dedup |
| Appendix E.3 Table E3 (latency breakdown) | `ablation_pipeline.json` | `experiments/ablation_pipeline.py` | Per-stage latency on 786 chunks |
| Appendix E.4 Table E4 (relevance strategies) | `combined_signal_analysis.json` | `experiments/combined_signal_analysis.py` | distance vs spread vs combined |

## Historical note on `arxiv_bench.json`

The original file landed in commit `a4c90c7` (2026-03-29,
"feat(experiments): add ArXiv retrieval benchmark") with
`corpus_size: 1000` and the NDCG/P@5/MRR numbers that Table 7
reports. Release `b1ef37c` ("Release v0.17.0") then overwrote the
same JSON with a 100-paper debug re-run, breaking the reproducibility
of Table 7 for several months. Commit `084afab` (PR #221) restored
the 1,000-paper data.

Rule of thumb for future releases: if a commit diff shows a results
JSON going from a larger `corpus_size` / `n_queries` to a smaller one
without a corresponding paper edit, treat it as an accident and
revert.

## How to add a new entry

1. Re-run the experiment from its script, committing the output JSON
   under `experiments/results/` (one JSON per experiment).
2. Add or update the row in the table above with:
   - Paper section / table number.
   - Path to the JSON.
   - Path to the producing script.
   - Any non-obvious derivation (which keys, which math).
3. If the paper number is derived from multiple JSONs, list each one.
4. If the JSON does not store the paper-ready number directly
   (like the scoring-lifecycle deltas, which require a baseline
   subtraction), spell out the computation in the Notes column so a
   reviewer can reproduce it in one step.

## Integrity check

To verify every paper number still matches its JSON, run:

```bash
python -m experiments.run_all
```

This is the single command a maintainer should run before every
arxiv version bump. A release PR whose diff touches any file listed
above but does not touch both the JSON and the corresponding paper
table should be treated as a red flag.
