# Pre-commitment: chat-to-BEIR generalization asymmetry decision rule

**Date written:** 2026-04-28
**Author:** Jayson Steffens
**Status:** Written BEFORE the validation runs are executed. This file is committed to git so the decision threshold is verifiable as having been set in advance, not post-hoc.

## Background

The paper v2 draft headline is "BEIR-tuned weights regress on chat memory" (S5.7, S6, lme-v1 case study). The casual symmetric expectation was "chat-tuned weights regress on BEIR." A first single-seed run (2026-04-28 morning) showed the opposite: lme-v1 (chat-tuned) **lifts** BEIR macro NDCG@10 by +0.0387 absolute / +10.2% relative vs vanilla BGE-small base, on the 5-dataset BEIR slice (SciFact, NFCorpus, FiQA, SciDocs, ArguAna).

Before updating the paper to claim a one-way asymmetry ("BEIR-tune hurts chat; chat-tune helps BEIR"), we run a validation pass to (a) eliminate the code-version drift confound between the original Table 4 base-BGE numbers (run weeks ago on pre-v0.34 code) and the lme-v1 numbers (today, v0.35 code), and (b) compute paired bootstrap CIs over per-query NDCG@10 to quantify the uncertainty.

Both validation runs use:

- Embedding backend: sentence-transformers via `--device cpu` resolver (registers `_STEncoder` in `experiments/beir_benchmark.py`, identical to the lme-v1 morning run).
- vstash code: current `develop` branch at v0.35.0.
- BEIR datasets: SciFact, NFCorpus, FiQA, SciDocs, ArguAna (BEIR test split with `qrels/test.tsv`, same files cached in `experiments/data/beir_*`).
- Per-query NDCG@10 logged via a new sidecar JSON for paired bootstrap.

## Threshold (committed)

Let `delta_macro = macro_NDCG@10_lme_v1 - macro_NDCG@10_base_bge` and `CI_lower = 2.5th percentile of macro_delta bootstrap distribution`.

| Outcome | Decision |
|---------|----------|
| `CI_lower > 0` **AND** `delta_macro > 0.04` | **ROBUST** -- update paper thesis to "training-signal quality > target alignment", add Section 5.8 (or new appendix) with the asymmetry table including per-dataset CIs. |
| `CI_lower > 0` **AND** `0 < delta_macro <= 0.04` | **DIRECTIONAL** -- update paper but with caveat "within the backend calibration band of +0.02 to +0.08 NDCG@10 disclosed in S8.10; leave the asymmetry magnitude as preliminary." |
| `CI_lower <= 0` | **ARTIFACT** -- retire the asymmetry claim. Keep the original v2 thesis ("BEIR-tune regresses on chat") which is still backed by S5.7 / S6 evidence. The reverse direction is not supportable. |

`B = 1000` paired bootstrap resamples, seed=42, paired by qid within each dataset, macro-averaged across the 5 datasets per bootstrap iteration (not over the 5 datapoints directly).

## What this rule prevents

- Post-hoc rationalization: even if the lift turns out to be +0.039 (just under threshold), the rule forces the "directional" caveat rather than letting the reported framing follow the data.
- Backend-noise inflation: 0.04 absolute is at the upper edge of the FastEmbed-vs-ST gap disclosed in the paper's existing S8.10 backend note. Setting that as the threshold for "robust" guarantees the claim survives even in the conservative interpretation that the backend noise is the worst-case observed gap (FiQA +0.064, ArguAna +0.017 minimum).
- Over-claiming on a single seed: a CI that crosses zero would mean the +10.2% point estimate was within paired-query variance and therefore not strong enough to update the paper's thesis on a single training seed.

## What this rule does NOT cover

- Training-stochastic robustness (different lme-v1 training seeds). If the bootstrap CI on this run is tight and excludes zero by a large margin, we accept the result without multi-seed retraining. If the CI is borderline, multi-seed becomes a follow-up.
- Out-of-domain BEIR datasets (TREC-COVID, CQADupStack, HotpotQA). Adding them is a separate generalization claim, not part of this asymmetry decision.

## Files committed in advance

- `experiments/beir_benchmark.py` -- patched to log per-query NDCG@10 to a sidecar JSON.
- `experiments/paired_bootstrap_beir.py` -- analysis script implementing the bootstrap rule above.

The git history will show this decision file and the analysis script were both committed before the result-aware runs were executed.
