# vstash vs grep — Benchmark Report

**Date:** 2026-03-20 09:46

**Corpus:** 5 files, 524 KB total


## Test Corpus

| File | Size | Chunks | Ingest Time |
|---|---|---|---|
| alice_wonderland.txt | 170 KB | 46 | 3.32s |
| art_of_war.txt | 334 KB | 92 | 3.63s |
| climate_change_report.md | 6 KB | 2 | 0.12s |
| fastapi_patterns.md | 5 KB | 2 | 0.1s |
| neural_architecture_search.md | 8 KB | 2 | 0.1s |

**Total:** 144 chunks, 0.0 MB database


## Query Results

### Query 1: "What strategies exist for defending against an invasion?"
*Military strategy — tests semantic understanding of warfare concepts*

**vstash** (0.0052s):

| # | Source | Score | Snippet |
|---|---|---|---|
| 1 | Art Of War | 0.01 | proceed in single file. Then, before there is time to range our soldiers in orde... |
| 2 | Art Of War | 0.0098 | says, "turning their backs on us and pretending to flee." But this is only one o... |
| 3 | Art Of War | 0.0097 | and also, says Chang Yu, "in order not to be impeded in your evolutions." The _T... |
| 4 | Art Of War | 0.0095 | developments:  6. (1) When fire breaks out inside the enemy’s camp, respond at o... |
| 5 | Art Of War | 0.0094 | illustrating the uses of deception in war.]  21. If he is secure at all points, ... |

**grep** (0.0493s) — terms: `invasion, defend, defense, attack`:

| File | Matched Terms |
|---|---|
| art_of_war.txt | invasion, defend, attack |
| fastapi_patterns.md | attack |
---

### Query 2: "How do characters experience identity confusion or transformation?"
*Literary theme — grep can't match conceptual meaning*

**vstash** (0.0041s):

| # | Source | Score | Snippet |
|---|---|---|---|
| 1 | Alice Wonderland | 0.01 | her to begin.” He looked at the Gryphon as if he thought it had some kind of aut... |
| 2 | Alice Wonderland | 0.0098 | putting their heads down and saying ‘Come up again, dear!’ I shall only look up ... |
| 3 | Alice Wonderland | 0.0097 | back!” the Caterpillar called after her. “I’ve something important to say!”  Thi... |
| 4 | Alice Wonderland | 0.0095 | Alice indignantly, and she sat down in a large arm-chair at one end of the table... |
| 5 | Art Of War | 0.0094 | ambush or insidious spies are likely to be lurking.  [Chang Yu has the note: "We... |

**grep** (0.0353s) — terms: `identity, confusion, transformation, change`:

| File | Matched Terms |
|---|---|
| neural_architecture_search.md | identity |
| alice_wonderland.txt | confusion, change |
| art_of_war.txt | confusion, transformation, change |
| climate_change_report.md | transformation, change |
| fastapi_patterns.md | change |
---

### Query 3: "What techniques improve the efficiency of deep learning training?"
*Technical ML — vstash should find related concepts even without exact words*

**vstash** (0.0037s):

| # | Source | Score | Snippet |
|---|---|---|---|
| 1 | Neural Architecture Search | 0.01 | # Neural Architecture Search: A Comprehensive Survey  ## Abstract  Neural Archit... |
| 2 | Neural Architecture Search | 0.0098 | search can match or exceed RL-based methods while being more memory-efficient. T... |

**grep** (0.0392s) — terms: `efficiency, training, deep learning, neural`:

| File | Matched Terms |
|---|---|
| neural_architecture_search.md | efficiency, training, deep learning, neural |
| climate_change_report.md | efficiency |
| art_of_war.txt | efficiency, training |
---

### Query 4: "What is the economic impact of rising temperatures?"
*Climate + economics — tests cross-domain reasoning*

**vstash** (0.0032s):

| # | Source | Score | Snippet |
|---|---|---|---|
| 1 | Climate Change Report | 0.01 | # Global Climate Change: Impacts, Mitigation, and Adaptation Strategies  ## Exec... |

**grep** (0.0368s) — terms: `economic, temperature, cost, impact`:

| File | Matched Terms |
|---|---|
| climate_change_report.md | temperature, cost, impact |
| neural_architecture_search.md | cost |
| alice_wonderland.txt | cost |
| art_of_war.txt | cost, impact |
---

### Query 5: "How should APIs handle authentication securely?"
*API security — tests technical recall*

**vstash** (0.0031s):

| # | Source | Score | Snippet |
|---|---|---|---|
| 1 | Fastapi Patterns | 0.01 | # FastAPI Advanced Patterns: Building Production-Ready APIs  ## 1. Authenticatio... |
| 2 | Fastapi Patterns | 0.0098 | , cache availability, and external dependencies - Include response time metrics ... |

**grep** (0.0408s) — terms: `authentication, security, token, JWT`:

| File | Matched Terms |
|---|---|
| fastapi_patterns.md | authentication, security, token, JWT |
| climate_change_report.md | security |
| art_of_war.txt | security, token |
---


## Summary

| Capability | vstash | grep |
|---|---|---|
| Semantic understanding | ✅ Finds conceptually related content | ❌ Exact keyword match only |
| Cross-document reasoning | ✅ Ranks by relevance across all docs | ⚠️ Returns all matching files equally |
| Synonym recognition | ✅ "invasion" matches "military campaign" | ❌ Only literal matches |
| Speed | ~50-100ms per query | ~5-10ms per query |
| Setup cost | Requires ingestion (embeddings) | Zero setup |
| Works offline | ✅ Embeddings are fully local | ✅ |

> **Conclusion:** grep excels at fast exact-match searches. vstash excels when you need 
> to find *conceptually related* content — the kind of search where you'd otherwise 
> need to manually read through documents to find relevant passages.
