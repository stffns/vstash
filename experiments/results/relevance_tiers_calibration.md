# Relevance tiers per embedding model (2026-11-10)

**Question.** `relevance_tier` labels a search high / medium / low from the
best match's cosine distance, with cutoffs set for BGE-small (0.4513 /
0.4802). KakesuI, running multilingual MiniLM on a Spanish memory, saw correct
answers at 0.49-0.56, all "low". What should the cutoffs be per model, and
what do the tiers actually detect?

Script: `experiments/calibrate_relevance_tiers.py`. Raw numbers:
`relevance_tiers_calibration.json`.

## Finding: the tiers detect off-topic queries, not unanswerable ones

In-domain hold-out on SciFact: half the queries held out, **all** their
relevant documents removed from the corpus. Their best distance is barely
distinguishable from that of answerable queries:

| model | answerable p10 / p50 / p90 | unanswerable (hold-out) p10 / p50 / p90 |
|---|---|---|
| BGE-small | 0.119 / 0.181 / 0.264 | 0.154 / 0.208 / 0.265 |
| MiniLM | 0.234 / 0.343 / 0.492 | 0.260 / 0.367 / 0.479 |
| Jina v5 nano | 0.201 / 0.310 / 0.465 | 0.289 / 0.405 / 0.501 |
| EmbeddingGemma | 0.317 / 0.423 / 0.539 | 0.388 / 0.492 / 0.576 |

Under BGE's constants every held-out question is "high". Even FiQA (finance)
questions against SciFact (science) are 98% under 0.4802. So "low" means
"about as far as an unrelated domain", and `min_relevance` gates off-topic
questions; an on-topic question about something never stored passes. (Jina
separates the two best, but the distributions still overlap heavily.)

## Criterion: transfer BGE's relative position

Per model and corpus, `m_ans` = median best distance of the corpus's own
queries, `m_off` = median for off-domain queries. BGE's constants sit at
`p = (t - m_ans) / (m_off - m_ans)` = **1.228** (high) and **1.357**
(medium) on SciFact with FiQA as off-domain; each model gets
`t = m_ans + p (m_off - m_ans)` on its own scale. BGE reproduces 0.4513 /
0.4802 exactly. `p > 1`: the tiers only fire beyond the typical off-domain
query, which is what BGE's constants already do (1.7% of FiQA-on-SciFact
queries are "low").

| model | corpus | m_ans | m_off | high_max | medium_max | off-domain "low" |
|---|---|---|---|---|---|---|
| BGE-small | SciFact (EN) | 0.177 | 0.400 | 0.4513 | 0.4802 | 1.7% |
| BGE-small | S2P (ES) | 0.202 | 0.276 | 0.293 | 0.303 | 25.7% |
| MiniLM | SciFact (EN) | 0.335 | 0.674 | 0.751 | 0.795 | 1.0% |
| MiniLM | S2P (ES) | 0.271 | 0.603 | **0.679** | **0.722** | 9.0% |
| Jina v5 nano | SciFact (EN) | 0.319 | 0.751 | 0.849 | 0.905 | 0% |
| Jina v5 nano | S2P (ES) | 0.290 | 0.708 | **0.803** | **0.857** | 0% |
| EmbeddingGemma | SciFact (EN) | 0.418 | 0.713 | 0.780 | 0.818 | 0% |
| EmbeddingGemma | S2P (ES) | 0.373 | 0.724 | **0.804** | **0.849** | 0% |

Off-domain sets: 300 FiQA questions (English), 300 Spanish news snippets from
`SpanishNewsClassification` (Spanish, against a health corpus). S2P corpus
subsampled to its 1 011 relevant passages + random fill to 3 000.

The registry (`vstash/relevance.py`) takes the **Spanish** values for the
multilingual models (bold): they are the lower of the two languages, and
those models are chosen for non-English memories. BGE-small keeps its
English constants; on Spanish its whole distance scale is compressed (good
and off-domain matches both around 0.2-0.3), one more reason not to run an
English-only model on a Spanish memory.

## Sanity check against KakesuI

KakesuI's correct MiniLM answers at 0.49-0.56 land in "high" with the new
MiniLM thresholds (they were all "low" before). KakesuI's own cut at
`best_distance` 0.58 is stricter than BGE's operating point transferred to
MiniLM (0.72); it is a valid local choice, not a sign the calibration is off.
Set both bounds (a lone `medium_max = 0.58` is rejected at load because
MiniLM's default `high_max` 0.6786 would exceed it):

```toml
[relevance]
high_max = 0.50
medium_max = 0.58
```

## Caveats

- MiniLM truncates at 128 tokens and S2P passages are long, which likely
  inflates its corpus-side distances relative to short notes.
- Thresholds depend on corpus size and passage length (more documents ->
  smaller best distances). Short-note memories of a few hundred entries sit
  differently from 3-10k passages; `[relevance]` exists for that.
- Jina / EmbeddingGemma numbers were produced with the backends from #522
  (Jina is not in `develop` yet; Gemma does not load on `develop` under
  onnxruntime 1.31).
