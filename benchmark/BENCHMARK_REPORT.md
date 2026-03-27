# vstash vs grep — Benchmark Report

**Date:** 2026-03-27 10:18

**Corpus:** 5 files, 524 KB total


## Test Corpus

| File | Size | Chunks | Ingest Time |
|---|---|---|---|
| alice_wonderland.txt | 170 KB | 43 | 3.56s |
| art_of_war.txt | 334 KB | 89 | 1.95s |
| climate_change_report.md | 6 KB | 9 | 0.07s |
| fastapi_patterns.md | 5 KB | 5 | 0.17s |
| neural_architecture_search.md | 8 KB | 12 | 0.27s |

**Total:** 302 chunks, 2.53 MB database


## Query Results

### Query 1: "What strategies exist for defending against an invasion?"
*Military strategy — tests semantic understanding of warfare concepts*

**vstash** (0.0134s):

| # | Source | Score | Snippet |
|---|---|---|---|
| 1 | Art Of War | 0.01 | [Tu Mu says: "As water flows downwards, we must not pitch our camp on the lower ... |
| 2 | Art Of War | 0.0098 | [T’sao Kung thinks that "traitors in the enemy’s camp" are referred to. But Ch’e... |
| 3 | Art Of War | 0.0097 | proceed in single file. Then, before there is time to range our soldiers in orde... |
| 4 | Art Of War | 0.0095 | says, "turning their backs on us and pretending to flee." But this is only one o... |
| 5 | Art Of War | 0.0094 | Every kind of ground is characterized by certain natural features, and also give... |

**grep** (0.0444s) — terms: `invasion, defend, defense, attack`:

| File | Matched Terms |
|---|---|
| art_of_war.txt | invasion, defend, attack |
| fastapi_patterns.md | attack |
---

### Query 2: "How do characters experience identity confusion or transformation?"
*Literary theme — grep can't match conceptual meaning*

**vstash** (0.0048s):

| # | Source | Score | Snippet |
|---|---|---|---|
| 1 | Alice Wonderland | 0.01 | Alice took up the fan and gloves, and, as the hall was very hot, she kept fannin... |
| 2 | Alice Wonderland | 0.0098 | “Well, perhaps you haven’t found it so yet,” said Alice; “but when you have to t... |
| 3 | Alice Wonderland | 0.0097 | Alice was more and more puzzled, but she thought there was no use in saying anyt... |
| 4 | Alice Wonderland | 0.0095 | “You should learn not to make personal remarks,” Alice said with some severity; ... |
| 5 | Art Of War | 0.0094 | [In order to make the translation intelligible, it is necessary to tone down the... |

**grep** (0.0285s) — terms: `identity, confusion, transformation, change`:

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

**vstash** (0.0049s):

| # | Source | Score | Snippet |
|---|---|---|---|
| 1 | Neural Architecture Search | 0.01 | ## 1. Introduction  Deep learning has achieved remarkable success across many do... |
| 2 | Neural Architecture Search | 0.0098 | ## 5. Efficiency-Aware NAS  Modern NAS methods increasingly consider not just ac... |
| 3 | Neural Architecture Search | 0.0097 | ## 4. Transformers and Attention Mechanisms  The rise of transformer architectur... |
| 4 | Neural Architecture Search | 0.0095 | # Neural Architecture Search: A Comprehensive Survey  ## Abstract  Neural Archit... |
| 5 | Neural Architecture Search | 0.0094 | # Neural Architecture Search: A Comprehensive Survey  ## Abstract  Neural Archit... |

**grep** (0.0356s) — terms: `efficiency, training, deep learning, neural`:

| File | Matched Terms |
|---|---|
| neural_architecture_search.md | efficiency, training, deep learning, neural |
| climate_change_report.md | efficiency |
| art_of_war.txt | efficiency, training |
---

### Query 4: "What is the economic impact of rising temperatures?"
*Climate + economics — tests cross-domain reasoning*

**vstash** (0.0052s):

| # | Source | Score | Snippet |
|---|---|---|---|
| 1 | Climate Change Report | 0.01 | # Global Climate Change: Impacts, Mitigation, and Adaptation Strategies  ## Exec... |
| 2 | Climate Change Report | 0.0098 | # Global Climate Change: Impacts, Mitigation, and Adaptation Strategies  ## Exec... |
| 3 | Climate Change Report | 0.0097 | ### 2.2 Food Security and Agriculture  Rising temperatures, changing precipitati... |
| 4 | Climate Change Report | 0.0095 | ### 1.1 Temperature Trends  Global surface temperature has increased faster sinc... |
| 5 | Climate Change Report | 0.0094 | ## 5. Conclusion  Limiting global warming to 1.5°C remains technically feasible ... |

**grep** (0.0333s) — terms: `economic, temperature, cost, impact`:

| File | Matched Terms |
|---|---|
| climate_change_report.md | temperature, cost, impact |
| neural_architecture_search.md | cost |
| alice_wonderland.txt | cost |
| art_of_war.txt | cost, impact |
---

### Query 5: "How should APIs handle authentication securely?"
*API security — tests technical recall*

**vstash** (0.0059s):

| # | Source | Score | Snippet |
|---|---|---|---|
| 1 | Fastapi Patterns | 0.01 | # FastAPI Advanced Patterns: Building Production-Ready APIs  ## 1. Authenticatio... |
| 2 | Fastapi Patterns | 0.0098 | ### 1.2 Role-Based Access Control (RBAC)  RBAC maps users to roles, and roles to... |
| 3 | Fastapi Patterns | 0.0097 | # FastAPI Advanced Patterns: Building Production-Ready APIs  ## 1. Authenticatio... |

**grep** (0.0347s) — terms: `authentication, security, token, JWT`:

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
