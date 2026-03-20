# vstash End-to-End Test Report

**Date:** 2026-03-20 10:26
**Backend:** cerebras / gpt-oss-120b
**Embedding:** BAAI/bge-small-en-v1.5

**Corpus:** 8 documents, 171 chunks, 2.93 MB

---

## Query 1: "Explain the difference between DARTS and evolutionary NAS methods, including their computational costs"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 2.2ms |
| Vector + FTS search | 6.2ms |
| **Total retrieval** | **8.4ms** |
| LLM inference | 1.97s |
| **End-to-end** | **1.98s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Neural Architecture Search (chunk 1) | 0.01 |

### 💬 Answer (191 words)

**DARTS (gradient‑based NAS)**  
- Treats the discrete architecture space as continuous and optimizes architecture parameters with gradient descent.  
- Because the search is performed by back‑propagation on a single “supernet,” the search cost drops dramatically—from **thousands of GPU‑days** (typical of many earlier methods) to **only a few GPU‑days**【Context 1†3.3】.  

**Evolutionary NAS (e.g., regularized evolution)**  
- Maintains a population of candidate architectures and iteratively selects parents (often using tournament selection) to generate new offspring.  
- The method is noted for being **more memory‑efficient** than RL‑based approaches, but the text does not give a precise GPU‑day figure; it implies the cost is higher than DARTS because the search still requires evaluating many separate architectures across generations【Context 1†3.2】.  

**Key differences**  
1. **Search mechanism** – DARTS uses differentiable, gradient‑based optimization; evolutionary NAS uses population‑based mutation and selection.  
2. **Computational cost** – DARTS reduces search time to a few GPU‑days, whereas evolutionary NAS, while memory‑efficient, generally incurs a larger computational burden (no exact number given, but it is higher than DARTS).  

Thus, DARTS is far cheaper in terms of GPU time, while evolutionary methods trade some extra compute for memory efficiency and a different search paradigm.

---

## Query 2: "What are the main arguments for renewable energy, and how do costs compare to fossil fuels?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 6.7ms |
| Vector + FTS search | 2.6ms |
| **Total retrieval** | **9.3ms** |
| LLM inference | 0.73s |
| **End-to-end** | **0.74s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Climate Change Report (chunk 1) | 0.01 |
| 2 | Climate Change Report (chunk 0) | 0.009836 |
| 3 | Art Of War (chunk 28) | 0.009677 |

### 💬 Answer (128 words)

**Main arguments for renewable energy (from the report)**  

- **Rapid cost declines:** Solar photovoltaic costs have fallen ≈ 89 % since 2010, and on‑shore wind costs have dropped ≈ 70 % over the same period【Context 2, §3.1 Renewable Energy Transition】.  
- **Dominance in new capacity:** Renewable energy now provides the majority of new power‑capacity additions worldwide【Context 2, §3.1 Renewable Energy Transition】.  
- **Broader benefits:** While not detailed in the excerpt, the transition is presented as a key mitigation strategy to reduce greenhouse‑gas emissions.

**Cost comparison with fossil fuels**

The provided excerpts give the magnitude of renewable‑energy cost reductions but do **not** include explicit cost figures for fossil‑fuel generation, nor a direct side‑by‑side comparison. Therefore, the report does not specify how current renewable‑energy costs compare numerically to fossil‑fuel costs.

---

## Query 3: "Describe Alice's emotional journey and key character interactions in Wonderland"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 8.4ms |
| Vector + FTS search | 1.8ms |
| **Total retrieval** | **10.1ms** |
| LLM inference | 1.13s |
| **End-to-end** | **1.14s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Alice Wonderland (chunk 21) | 0.01 |
| 2 | Alice Wonderland (chunk 5) | 0.009836 |
| 3 | Alice Wonderland (chunk 20) | 0.009677 |
| 4 | Alice Wonderland (chunk 4) | 0.009524 |
| 5 | Alice Wonderland (chunk 24) | 0.009375 |

### 💬 Answer (521 words)

**Alice’s emotional journey**

1. **Indignation and anger** – At the tea‑party Alice sits “indignantly” in a large arm‑chair and rebukes the March Hare for offering wine that isn’t there and for not inviting her (Context 1). She also scolds the Hatter for a personal remark about her hair (Context 1).

2. **Puzzlement and frustration** – When the Hatter asks “What day of the month is it?” Alice guesses “the fourth,” only to be told she is two days wrong. She then becomes “dreadfully puzzled” by the Hatter’s cryptic comments about watches and years (Context 1).

3. **Exhaustion and despair** – Later, after shrinking to about two feet, Alice worries about being “very tired of being all alone here,” cries, and swims in a “pool of tears” she herself has created (Context 2). She feels “frightened” at the narrow escape from further shrinking (Context 2).

4. **Confusion and curiosity** – While speaking to a mouse she wonders whether the mouse understands English or French, showing her bewildered curiosity (Context 2).

5. **Madness and bewilderment** – The Cheshire Cat tells her “we’re all mad here,” and Alice debates the cat’s logic about being mad (Context 3). She is “not much surprised” by the cat’s vanishing acts, yet feels “giddy” (Context 3).

6. **Hope and determination** – After leaving the chaotic tea‑party, Alice decides she will “never go there again” but then discovers a door in a tree, enters a hall, and finally reaches the beautiful garden, showing renewed resolve (Context 5).

**Key character interactions**

| Character | Interaction with Alice | Notable exchange |
|-----------|------------------------|------------------|
| **March Hare** | Offers non‑existent wine; argues about civility; checks the date on his watch. | “Have some wine… There isn’t any.” (Context 1) |
| **Hatter** | Asks riddles (“Why is a raven like a writing‑desk?”), questions the watch, and debates the riddle’s answer. | “What day of the month is it?” … “Two days wrong!” (Context 1) |
| **Dormouse** | Sleeps on the table, is used as a cushion, later wakes with a shriek when pinched. | “The Dormouse fell asleep instantly…” (Context 5) |
| **White Rabbit** | Returns with gloves and fan, drops them in panic when Alice asks for help. | “The Rabbit started violently, dropped the white kid gloves and the fan…” (Context 4) |
| **Cheshire Cat** | Declares everyone is mad, engages in a logical‑paradox discussion, vanishes repeatedly. | “You’re mad… I growl when I’m pleased, and wag my tail when I’m angry.” (Context 3) |
| **Mouse** | Listens (silently) to Alice’s pleas for directions; reacts fearfully when she speaks French. | “The Mouse gave a sudden leap out of the water… ‘Not like cats!’” (Context 2) |
| **Garden gardeners (Five, Seven, Two)** | Paint white roses red while Alice watches; they explain the danger of the Queen’s wrath. | “Why the fact is… this here ought to have been a red rose‑tree.” (Context 5) |

Overall, Alice moves from **anger and confusion** through **sadness and loneliness**, to **curiosity**, **frustration with madness**, and finally to **determination and hope** as she navigates the bizarre characters of Wonderland.

---

## Query 4: "What does Sun Tzu say about the importance of knowing your enemy and terrain?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 4.3ms |
| Vector + FTS search | 2.6ms |
| **Total retrieval** | **6.9ms** |
| LLM inference | 2.68s |
| **End-to-end** | **2.68s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Art Of War (chunk 32) | 0.01 |
| 2 | Art Of War (chunk 54) | 0.009836 |
| 3 | Art Of War (chunk 51) | 0.009677 |
| 4 | Art Of War (chunk 58) | 0.009524 |
| 5 | Art Of War (chunk 6) | 0.009375 |

### 💬 Answer (165 words)

Sun Tzu stresses that success in war begins with knowledge.  He writes that a commander who “knows the enemy and knows himself… need not fear the result of a hundred battles,” while one who knows neither “will succumb in every battle” (Context 1).  Knowing the enemy lets a leader take the offensive; knowing himself lets him defend (Context 1).

He also makes terrain a central factor in deciding how to act.  He classifies six kinds of ground—accessible, entangling, temporising, narrow passes, precipitous heights, and distant positions—and advises that on *accessible* ground the commander should “be before the enemy in occupying the raised and sunny spots, and carefully guard your line of supplies” (Context 4).  For *temporising* ground he counsels restraint and retreat to lure the enemy, and for narrow passes he recommends occupying and garrisoning them first (Context 4).  In short, Sun Tzu says that understanding both the enemy and the terrain is essential to choosing the right offensive or defensive moves and securing victory.

---

## Query 5: "How should a production API implement rate limiting and caching with Redis?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 10.5ms |
| Vector + FTS search | 2.9ms |
| **Total retrieval** | **13.5ms** |
| LLM inference | 0.97s |
| **End-to-end** | **0.99s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Fastapi Patterns (chunk 1) | 0.01 |

### 💬 Answer (181 words)

**Rate limiting**

In production you can use Redis to enforce request limits. A common pattern is to store a counter per client (or per key) and set an expiration that defines the time window. The context provides a concise async implementation:

```python
async def rate_limit(key: str, limit: int, window: int) -> bool:
    # Increment the counter for the given key
    current = await redis.incr(key)

    # If this is the first request, set the TTL for the window
    if current == 1:
        await redis.expire(key, window)

    # Allow the request only while the count is within the limit
    return current <= limit
```

- `key` identifies the client or endpoint (e.g., `"user:123:api"`).  
- `limit` is the maximum number of requests allowed in the window.  
- `window` is the duration of the window in seconds.  

This approach works for the **fixed‑window** algorithm; you can adapt it for sliding‑window, token‑bucket, or leaky‑bucket strategies if needed.

**Caching**

The provided context does not include a specific Redis caching pattern or code example. Therefore, a production API’s caching implementation with Redis cannot be described from the given material.

---

## Query 6: "What is vstash's technology stack and how does it achieve fast local search?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 10.6ms |
| Vector + FTS search | 2.1ms |
| **Total retrieval** | **12.7ms** |
| LLM inference | 0.91s |
| **End-to-end** | **0.93s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Readme (chunk 0) | 0.01 |
| 2 | Vstash Constitution (chunk 0) | 0.009836 |

### 💬 Answer (328 words)

**vstash’s technology stack**

| Layer | Technology | Why it’s used |
|-------|------------|---------------|
| **Embeddings** | FastEmbed (ONNX Runtime) | Generates embeddings locally at ~700 chunks / s, no server needed |
| **Embedding model** | `BAAI/bge-small-en-v1.5` (384‑dim) | Fastest quality‑speed ratio |
| **Vector store** | `sqlite-vec` | Single‑file SQLite DB, cosine similarity, zero external dependencies |
| **Keyword search** | SQLite FTS5 | Built‑in full‑text search with porter stemming, exact‑match speed |
| **Hybrid ranking** | Reciprocal Rank Fusion | Combines semantic and keyword results without training |
| **Document parsing** | `markitdown` | Handles PDF, DOCX, PPTX, HTML, Markdown, URLs, etc. locally |
| **Inference back‑ends** | Cerebras API / OpenAI API / Ollama | Configurable; Ollama runs fully locally, Cerebras/OpenAI are remote |
| **CLI / UI** | Typer + Rich | Clean terminal interface with progress feedback |
| **Configuration** | Pydantic v2 | Type‑safe config validation |
| **Language** | Python 3.10+ | Broad ecosystem and rapid development |

**How fast local search is achieved**

1. **All heavy‑weight work stays on the machine** – parsing, chunking, embedding generation, vector storage, and keyword indexing are performed locally using FastEmbed (ONNX) and SQLite‑vec/FTS5, eliminating network latency.  
2. **Efficient vector store** – `sqlite-vec` stores embeddings in a single `.db` file and performs cosine‑similarity look‑ups directly in SQLite, which is extremely fast for millions of vectors.  
3. **Hybrid ranking** – Reciprocal Rank Fusion merges semantic (vector) and keyword (FTS5) results instantly, avoiding costly re‑ranking models.  
4. **Optimized chunking** – 1 024‑token chunks with 128‑token overlap keep the number of vectors manageable while preserving context.  
5. **No server overhead** – The tool runs as a simple CLI; there’s no Docker, Postgres, or external service required, so queries complete in sub‑second times.  

These design choices together give vstash “instant semantic search” with sub‑second responses, as described in the README and constitution.  

*Sources: Context 1 (technology table and layer descriptions) and Context 2 (technology stack table and speed rationale).*

---

## Query 7: "How does Retrieval-Augmented Generation reduce hallucinations in language models?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 9.4ms |
| Vector + FTS search | 2.3ms |
| **Total retrieval** | **11.7ms** |
| LLM inference | 0.81s |
| **End-to-end** | **0.82s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 10) | 0.01 |
| 2 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 1) | 0.009836 |
| 3 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 15) | 0.009677 |
| 4 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 3) | 0.009524 |
| 5 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 8) | 0.009375 |

### 💬 Answer (131 words)

Retrieval‑augmented generation (RAG) lowers the rate of “hallucinations” by grounding a model’s output in real, external documents instead of relying only on the knowledge stored in its parameters.  

* Before answering, the LLM first **retrieves relevant text** from a database, uploaded files, or the web 【Context 4】.  
* The retrieved passages supply **up‑to‑date, domain‑specific facts** that the model can copy or paraphrase, so it is less likely to invent information that isn’t present in the source material 【Context 4】.  
* Because RAG can **include citations or source references** in its response, users can verify the content, further discouraging the model from fabricating unsupported statements 【Context 4】.  

In short, by coupling information‑retrieval with generation, RAG provides factual grounding and source transparency, which together help suppress the hallucinations that pure‑parameter LLMs often produce.

---

## Summary

| Metric | Value |
|---|---|
| Queries tested | 7 |
| Total retrieval time | 73ms (10ms avg) |
| Total LLM time | 9.2s (1.32s avg) |
| Avg end-to-end | 1.33s |
| Corpus size | 171 chunks |