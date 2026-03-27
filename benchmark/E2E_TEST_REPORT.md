# vstash End-to-End Test Report

**Date:** 2026-03-27 10:18
**Backend:** cerebras / gpt-oss-120b
**Embedding:** BAAI/bge-small-en-v1.5

**Corpus:** 47 documents, 401 chunks, 3.86 MB

---

## Query 1: "Explain the difference between DARTS and evolutionary NAS methods, including their computational costs"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 7.9ms |
| Vector + FTS search | 6.1ms |
| **Total retrieval** | **14.0ms** |
| LLM inference | 2.1s |
| **End-to-end** | **2.11s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | E2E Test Report (chunk 1) | 0.01 |
| 2 | Neural Architecture Search (chunk 7) | 0.009836 |
| 3 | Benchmark (chunk 8) | 0.009677 |
| 4 | E2E Test Report (chunk 0) | 0.006667 |
| 5 | E2E Test (chunk 0) | 0.006557 |

### 💬 Answer (98 words)

**DARTS (Differentiable Architecture Search)**  
- Treats the discrete NAS search space as a continuous one, so the architecture parameters can be optimized with gradient descent.  
- This approach “dramatically reduces search cost from thousands of GPU‑days to just a few GPU‑days”【3†L1-L4】.  

**Evolutionary NAS methods**  
- The provided excerpts contain no description of evolutionary NAS, nor any information about their computational cost.  

**Summary**  
Based on the available context, DARTS is a gradient‑based, continuous‑relaxation method that cuts search time to a few GPU‑days. The context does not provide details on how evolutionary NAS works or how its computational cost compares.

---

## Query 2: "What are the main arguments for renewable energy, and how do costs compare to fossil fuels?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 11.4ms |
| Vector + FTS search | 2.9ms |
| **Total retrieval** | **14.3ms** |
| LLM inference | 0.83s |
| **End-to-end** | **0.84s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | E2E Test Report (chunk 1) | 0.016344 |
| 2 | Climate Change Report (chunk 6) | 0.01 |
| 3 | Benchmark (chunk 19) | 0.009836 |
| 4 | E2E Test (chunk 0) | 0.006557 |

### 💬 Answer (113 words)

**Main arguments for renewable energy (from the provided context)**  
- The cost of renewable energy has fallen dramatically over the past decade.  
  - Solar photovoltaic (PV) costs are down ≈ 89 % since 2010.  
  - On‑shore wind costs are down ≈ 70 % since 2010.【Context 2†L3-L7】  
- Renewable sources now provide the majority of new power‑capacity additions worldwide.【Context 2†L7-L9】  
- These trends make renewables increasingly competitive and attractive for future power systems.  

**Cost comparison to fossil fuels**  
The excerpts mention the steep decline in renewable‑energy costs, but they do **not** provide any specific numbers or direct comparison of renewable costs versus fossil‑fuel costs. Therefore, a cost‑comparison cannot be derived from the given context.

---

## Query 3: "Describe Alice's emotional journey and key character interactions in Wonderland"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 8.4ms |
| Vector + FTS search | 3.1ms |
| **Total retrieval** | **11.5ms** |
| LLM inference | 1.16s |
| **End-to-end** | **1.17s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Benchmark Report (chunk 2) | 0.01 |
| 2 | Alice Wonderland (chunk 13) | 0.009836 |
| 3 | Alice Wonderland (chunk 4) | 0.009677 |
| 4 | Alice Wonderland (chunk 38) | 0.009524 |
| 5 | Alice Wonderland (chunk 10) | 0.009375 |

### 💬 Answer (434 words)

**Alice’s emotional arc in Wonderland**

1. **Confusion and irritation with the Caterpillar** –  
   Alice becomes annoyed by the Caterpillar’s short, contemptuous remarks and the endless questioning of “Who are you?” She feels “a little irritated” and later “melancholy” when she cannot remember things as she used to and when her speech comes out “different” (Context 2).  

2. **Self‑doubt and identity crisis** –  
   While holding a fan she wonders whether she has been “changed in the night” and asks herself “Who in the world am I?” She tries to match herself to other children (Ada, Mabel) and feels “puzzling” and “very tired of being all alone” (Context 3).  

3. **Fear and relief after shrinking** –  
   When the fan makes her shrink, Alice is “frightened” and then relieved that she “still [is] in existence,” calling the near‑miss a “narrow escape” (Context 3).  

4. **Nostalgia and yearning for home** –  
   After the chaotic adventures she longs for the stability of home, thinking it would be “much pleasanter … when one wasn’t always growing larger and smaller” (Context 5). She worries she may “never get any older” and laments the lack of “room … for any lesson‑books.”  

5. **Comfort in the sister’s dream** –  
   When she recounts her dream to her sister, the sister’s own dreaming creates a gentle, reflective mood. The sister imagines Alice’s future as a grown woman who keeps “the simple and loving heart of her childhood,” offering a hopeful, soothing closure (Context 4).  

**Key character interactions**

| Character | Interaction & Effect on Alice |
|-----------|------------------------------|
| **Caterpillar** | Questions her identity (“Who are you?”), calls her “changed,” and critiques her speech, provoking irritation, confusion, and a sense of being “old” or “different.” (Context 2) |
| **Rabbit** | Appears as a frantic, large figure; Alice’s fear of the Rabbit’s size leads to frantic attempts to reach the window, producing anxiety and a feeling of helplessness. (Context 5) |
| **Sister** | Listens to Alice’s story, kisses her, and later dreams of Alice, providing a comforting, nostalgic frame that ends the narrative on a tender note. (Context 4) |
| **Other Wonderland figures** (e.g., Gryphon, Mock Turtle, Queen) | Mentioned in the sister’s dream, they reinforce the chaotic, ever‑changing world that Alice must navigate, underscoring her emotional turbulence. (Context 4) |

**Overall journey** – Alice moves from irritation and confusion (Caterpillar) → deep identity doubt (fan, “who am I?”) → fear of physical change (shrinking) → yearning for normalcy (home) → gentle reassurance through her sister’s dream. This progression is driven by repeated encounters with bizarre characters who constantly challenge her sense of self.

---

## Query 4: "What does Sun Tzu say about the importance of knowing your enemy and terrain?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 11.9ms |
| Vector + FTS search | 3.4ms |
| **Total retrieval** | **15.3ms** |
| LLM inference | 2.09s |
| **End-to-end** | **2.11s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Art Of War (chunk 31) | 0.01 |
| 2 | Art Of War (chunk 25) | 0.009836 |
| 3 | Art Of War (chunk 73) | 0.009677 |
| 4 | Art Of War (chunk 52) | 0.009524 |
| 5 | Art Of War (chunk 83) | 0.009375 |

### 💬 Answer (121 words)

Sun Tzu stresses that success in war depends first on knowledge.  He writes that **“If you know the enemy and know yourself, you need not fear the result of a hundred battles… If you know neither the enemy nor yourself, you will succumb in every battle.”**【Context 1】.  

He also treats the ground on which a force fights as a vital source of information.  In the passages on terrain he advises that any **hilly country, ponds, hollow basins, woods, or narrow passes near the camp must be carefully searched and cleared**, because such places are where ambushes or spies may be hidden【Context 4】.  Thus, for Sun Tzu, both the enemy’s intentions and the character of the terrain are essential to secure victory.

---

## Query 5: "How should a production API implement rate limiting and caching with Redis?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 12.1ms |
| Vector + FTS search | 3.3ms |
| **Total retrieval** | **15.5ms** |
| LLM inference | 0.92s |
| **End-to-end** | **0.93s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Benchmark (chunk 26) | 0.01 |
| 2 | Fastapi Patterns (chunk 4) | 0.009836 |
| 3 | E2E Test Report (chunk 4) | 0.006667 |
| 4 | E2E Test (chunk 0) | 0.006557 |

### 💬 Answer (176 words)

A production FastAPI service can use Redis to enforce rate limits by storing a counter for each client/key and expiring it after a fixed window. The pattern shown in the context is:

```python
async def rate_limit(key: str, limit: int, window: int) -> bool:
    # Increment the request count for the given key
    current = await redis.incr(key)
    # On the first request, set the TTL for the window
    if current == 1:
        await redis.expire(key, window)
    # Allow the request only while the count stays ≤ limit
    return current <= limit
```

This function:

1. Calls `INCR` on a Redis key (e.g., a user‑ID or IP address) to count requests.  
2. Sets an expiration (`EXPIRE`) for the key on the first increment, defining the time window (seconds).  
3. Returns `True` while the count is within the allowed limit, otherwise `False`.

You would call `rate_limit` at the start of each endpoint (or via a dependency) and reject requests that exceed the limit.

**Caching** is not described in the provided contexts, so no specific Redis‑based caching pattern is available there.

---

## Query 6: "What is vstash's technology stack and how does it achieve fast local search?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 10.0ms |
| Vector + FTS search | 2.8ms |
| **Total retrieval** | **12.7ms** |
| LLM inference | 1.0s |
| **End-to-end** | **1.01s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | E2E Test Report (chunk 5) | 0.01619 |
| 2 | E2E Test (chunk 0) | 0.015648 |
| 3 | Vstash Test Frontmatter (chunk 0) | 0.01 |
| 4 | E2E Test Report (chunk 6) | 0.009836 |
| 5 | Readme (chunk 0) | 0.009677 |

### 💬 Answer (351 words)

**vstash’s technology stack**

| Layer | Technology | Reason it’s used |
|-------|------------|------------------|
| Vector store | **sqlite‑vec** | Stores embeddings in a single SQLite file and performs cosine‑similarity search directly inside SQLite. |
| Keyword search | **FTS5 (SQLite)** | Built‑in full‑text index with exact‑match and Porter‑stemming; runs locally in the same DB. |
| Hybrid ranking | **Reciprocal Rank Fusion (RRF)** | Merges the semantic scores from sqlite‑vec with the keyword scores from FTS5 for better relevance without extra training. |
| Embeddings | **FastEmbed (ONNX Runtime)** | Generates embeddings locally on CPU (~700 chunks / s) using small models. |
| Embedding model | **BAAI/bge‑small‑en‑v1.5** (384‑dim) | Fast quality‑/‑speed trade‑off; dimensions match the vector store. |
| Inference back‑ends | **Cerebras API**, **OpenAI API**, **Ollama (local)** | Configurable; only the top‑k retrieved chunks are sent to a remote API, or everything stays local with Ollama. |
| Document parsing | **markitdown** | Handles PDF, DOCX, PPTX, HTML, Markdown, code files, and URLs locally. |
| Storage | **SQLite (single .db file)** | All metadata, chunk text, embeddings, and FTS5 indexes live in one file that can be copied or backed up. |
| Configuration | **Pydantic v2** | Type‑safe config handling. |
| CLI | **Typer + Rich** | Clean, interactive terminal UI. |
| Language | **Python 3.10+** | Easy ecosystem and rapid development. |【Context 4】

**How fast local search is achieved**

1. **All data in one SQLite database** – no external server, so there is no network latency.  
2. **Vector similarity** is computed by `sqlite‑vec` inside SQLite, giving sub‑second semantic retrieval.  
3. **Keyword matching** uses SQLite’s highly optimized `FTS5` index for rapid exact‑match and stemming searches.  
4. **Hybrid ranking with RRF** efficiently merges the two result sets, improving relevance without extra computation.  
5. **Embeddings are generated locally** with FastEmbed (ONNX), avoiding remote calls and allowing ~700 chunks per second.  
6. **Chunking** creates overlapping token windows (default 1024‑token size, 128‑token overlap) that keep retrieval both fast and accurate.  

These components together let vstash return semantic answers in under a second, entirely on the user’s machine【Context 4】【Context 3】.

---

## Query 7: "How does Retrieval-Augmented Generation reduce hallucinations in language models?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 9.5ms |
| Vector + FTS search | 2.5ms |
| **Total retrieval** | **12.0ms** |
| LLM inference | 0.67s |
| **End-to-end** | **0.68s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | E2E Test Report (chunk 7) | 0.01 |
| 2 | E2E Test Report (chunk 6) | 0.006667 |
| 3 | E2E Test (chunk 0) | 0.006557 |

### 💬 Answer (28 words)

The provided excerpts do not contain an explanation of how Retrieval‑Augmented Generation reduces hallucinations in language models. (See Context 1, which notes that the excerpts lack this information.)

---

## Summary

| Metric | Value |
|---|---|
| Queries tested | 7 |
| Total retrieval time | 95ms (14ms avg) |
| Total LLM time | 8.8s (1.25s avg) |
| Avg end-to-end | 1.27s |
| Corpus size | 401 chunks |