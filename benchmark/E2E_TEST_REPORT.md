# vstash End-to-End Test Report

**Date:** 2026-03-20 10:19
**Backend:** cerebras / llama3.1-8b
**Embedding:** BAAI/bge-small-en-v1.5

**Corpus:** 8 documents, 171 chunks, 2.93 MB

---

## Query 1: "Explain the difference between DARTS and evolutionary NAS methods, including their computational costs"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 432.7ms |
| Vector + FTS search | 6.1ms |
| **Total retrieval** | **438.9ms** |
| LLM inference | 0.78s |
| **End-to-end** | **1.22s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Neural Architecture Search (chunk 1) | 0.01 |

### 💬 Answer (231 words)

According to the provided context:

**DARTS (Differentiable Architecture Search)**:
- Relaxes the discrete search space to be continuous, allowing architecture parameters to be optimized using gradient descent.
- Dramatically reduces search cost from thousands of GPU-days to just a few GPU-days.
- However, can suffer from performance collapse, where the search converges to degenerate architectures dominated by skip connections.

**Evolutionary NAS methods (not explicitly mentioned as a single method, but implied by the tournament selection strategy mentioned in the introduction)**:
- Use a tournament selection strategy, where a subset of the population is randomly sampled and the best architecture in the subset becomes the parent for the next generation.
- Can match or exceed RL-based methods while being more memory-efficient.

The key differences between DARTS and evolutionary NAS methods are:

1. **Search space relaxation**: DARTS relaxes the discrete search space to be continuous, while evolutionary NAS methods do not explicitly mention relaxation of the search space.
2. **Computational cost**: DARTS reduces search cost from thousands of GPU-days to just a few GPU-days, while evolutionary NAS methods are not explicitly mentioned in terms of computational cost.
3. **Risk of performance collapse**: DARTS can suffer from performance collapse, while evolutionary NAS methods are not mentioned as having this issue.

Note that the context does not provide a direct comparison between DARTS and evolutionary NAS methods, but rather provides information about each method separately.

---

## Query 2: "What are the main arguments for renewable energy, and how do costs compare to fossil fuels?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 4.0ms |
| Vector + FTS search | 2.6ms |
| **Total retrieval** | **6.6ms** |
| LLM inference | 1.18s |
| **End-to-end** | **1.19s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Climate Change Report (chunk 1) | 0.01 |
| 2 | Climate Change Report (chunk 0) | 0.009836 |
| 3 | Art Of War (chunk 28) | 0.009677 |

### 💬 Answer (90 words)

Based on the provided context, the main arguments for renewable energy are:

1. The cost of renewable energy has plummeted over the past decade. 
2. Solar photovoltaic costs have decreased by 89% since 2010, while onshore wind costs have decreased by 70%. 
3. Renewable energy now accounts for the majority of new power capacity additions globally.

However, challenges remain in grid integration, energy storage, and ensuring a just transition for fossil fuel-dependent communities.

There is no direct comparison of costs between renewable energy and fossil fuels in the provided context.

---

## Query 3: "Describe Alice's emotional journey and key character interactions in Wonderland"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 9.9ms |
| Vector + FTS search | 2.7ms |
| **Total retrieval** | **12.5ms** |
| LLM inference | 1.17s |
| **End-to-end** | **1.18s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Alice Wonderland (chunk 21) | 0.01 |
| 2 | Alice Wonderland (chunk 5) | 0.009836 |
| 3 | Alice Wonderland (chunk 20) | 0.009677 |
| 4 | Alice Wonderland (chunk 4) | 0.009524 |
| 5 | Alice Wonderland (chunk 24) | 0.009375 |

### 💬 Answer (321 words)

Based on the provided context, Alice's emotional journey in Wonderland can be described as follows:

1. **Initial Curiosity and Confusion**: Alice's journey begins with her curiosity and confusion as she navigates the strange and illogical world of Wonderland. She is initially excited to explore this new world, but soon becomes frustrated and disoriented by its absurdity.

2. **Growing Frustration and Anger**: As Alice encounters more and more bizarre creatures and situations, her frustration and anger grow. She becomes increasingly exasperated by the March Hare's and Hatter's illogical behavior and the absurdity of their tea party.

3. **Desperation and Loneliness**: Alice's emotional state reaches a low point when she is left alone in the garden, feeling desperate and lonely. She begins to question her own identity and sanity, wondering if she has been changed in some way.

4. **Determination and Resilience**: Despite her emotional turmoil, Alice shows determination and resilience as she continues to navigate Wonderland. She tries to make sense of the world around her and seeks to find a way out.

Key character interactions that contribute to Alice's emotional journey include:

1. **The March Hare and Hatter**: Their absurd and illogical behavior frustrates Alice and adds to her sense of confusion and disorientation.

2. **The Cheshire Cat**: The Cat's enigmatic and sometimes cruel behavior tests Alice's patience and understanding. However, the Cat also provides some guidance and insight, helping Alice to see the absurdity of the world around her.

3. **The White Rabbit**: The Rabbit's frantic and anxious behavior creates a sense of urgency and chaos, contributing to Alice's growing sense of desperation and confusion.

4. **The Queen of Hearts**: The Queen's tyrannical and violent behavior creates a sense of fear and danger, highlighting the absurdity and illogic of Wonderland's social hierarchy.

Overall, Alice's emotional journey in Wonderland is marked by a growing sense of frustration, desperation, and confusion, as she navigates a world that is increasingly absurd and illogical.

---

## Query 4: "What does Sun Tzu say about the importance of knowing your enemy and terrain?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 4.8ms |
| Vector + FTS search | 1.4ms |
| **Total retrieval** | **6.2ms** |
| LLM inference | 1.16s |
| **End-to-end** | **1.17s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Art Of War (chunk 32) | 0.01 |
| 2 | Art Of War (chunk 54) | 0.009836 |
| 3 | Art Of War (chunk 51) | 0.009677 |
| 4 | Art Of War (chunk 58) | 0.009524 |
| 5 | Art Of War (chunk 6) | 0.009375 |

### 💬 Answer (307 words)

According to the provided context, Sun Tzu emphasizes the importance of knowing the enemy and terrain in the following passages:

1. "If you know the enemy and know yourself, you need not fear the result of a hundred battles." (Chapter IV, Tactical Dispositions)

This suggests that having knowledge of the enemy's strengths, weaknesses, and tactics is crucial for success in battle.

2. "To secure ourselves against defeat lies in our own hands, but the opportunity of defeating the enemy is provided by the enemy himself." (Chapter IV, Tactical Dispositions)

This implies that the enemy's mistakes and weaknesses can be exploited to gain an advantage.

3. "We may distinguish six kinds of terrain, to wit: (1) Accessible ground; (2) entangling ground; (3) temporising ground; (4) narrow passes; (5) precipitous heights; (6) positions at a great distance from the enemy." (Chapter X, Terrain)

This classification highlights the importance of understanding the terrain and its characteristics to gain an advantage.

4. "Ground which can be freely traversed by both sides is called accessible. With regard to ground of this nature, be before the enemy in occupying the raised and sunny spots, and carefully guard your line of supplies." (Chapter X, Terrain)

This passage emphasizes the importance of securing favorable terrain and protecting one's supply lines.

5. "Ground which can be abandoned but is hard to re-occupy is called entangling. From a position of this sort, if the enemy is unprepared, you may sally forth and defeat him. But if the enemy is prepared for your coming, and you fail to defeat him, then, return being impossible, disaster will ensue." (Chapter X, Terrain)

This passage highlights the risks of occupying entangling ground and the importance of being prepared for the enemy's response.

Overall, Sun Tzu's teachings emphasize the importance of knowing the enemy and terrain to gain an advantage in battle.

---

## Query 5: "How should a production API implement rate limiting and caching with Redis?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 12.2ms |
| Vector + FTS search | 3.0ms |
| **Total retrieval** | **15.2ms** |
| LLM inference | 0.91s |
| **End-to-end** | **0.93s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Fastapi Patterns (chunk 1) | 0.01 |

### 💬 Answer (212 words)

Based on the provided context, a production API should implement rate limiting and caching with Redis as follows:

1. **Rate Limiting:**
   Use the `rate_limit` function provided in the context, which implements rate limiting using a token bucket algorithm. This function takes a `key`, `limit`, and `window` as parameters and returns a boolean indicating whether the request should be allowed.

   ```python
async def rate_limit(key: str, limit: int, window: int) -> bool:
    current = await redis.incr(key)
    if current == 1:
        await redis.expire(key, window)
    return current <= limit
```

2. **Caching:**
   The context does not provide a specific caching function, but it mentions cache availability and external dependencies. To implement caching with Redis, you can use the `redis` library to store and retrieve cached data.

   Here's an example of a simple caching function:

   ```python
async def cache_data(key: str, data: str, expire: int = 60) -> bool:
    await redis.set(key, data)
    await redis.expire(key, expire)
    return True
```

   To retrieve cached data, you can use the following function:

   ```python
async def get_cached_data(key: str) -> str:
    return await redis.get(key)
```

Note that these are simplified examples and you may need to adapt them to your specific use case. Additionally, you should consider implementing rate limiting and caching in a way that is scalable and fault-tolerant in a production environment.

---

## Query 6: "What is vstash's technology stack and how does it achieve fast local search?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 3.2ms |
| Vector + FTS search | 1.0ms |
| **Total retrieval** | **4.2ms** |
| LLM inference | 0.79s |
| **End-to-end** | **0.8s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Readme (chunk 0) | 0.01 |
| 2 | Vstash Constitution (chunk 0) | 0.009836 |

### 💬 Answer (309 words)

**Technology Stack:**

| Layer | Technology | Why |
|-------|-----------|-----|
| **Vector store** | `sqlite-vec` | Single file, no server, fast enough for millions of vectors |
| **Keyword search** | `FTS5` (SQLite) | Exact matches, porter stemming, built into SQLite |
| **Hybrid ranking** | Reciprocal Rank Fusion | Best of both: semantic + keyword, no training needed |
| **Embeddings** | `FastEmbed` (Qdrant ONNX) | ONNX runtime, ~700 chunks/s on CPU, no server, `pip install` only |
| **Embedding model** | `BAAI/bge-small-en-v1.5` | 384 dims, fastest quality/speed ratio |
| **Inference (fast)** | Cerebras API | ~2,000 tokens/second — the fastest available API |
| **Inference (flexible)** | OpenAI API | Compatible with any OpenAI-compatible endpoint |
| **Inference (private)** | Ollama (any model) | 100% local fallback, no data leaves the machine |
| **Document parsing** | `markitdown` | Universal: PDF, DOCX, PPTX, HTML, code, URLs |
| **Configuration** | Pydantic v2 | Type-safe config with validation and defaults |
| **CLI** | `Typer` + `Rich` | Clean, beautiful terminal interface with progress feedback |
| **Language** | Python 3.10+ | Ecosystem, speed of development, accessibility |

**Achieving Fast Local Search:**

vstash achieves fast local search by leveraging the following technologies:

1. **FastEmbed**: ONNX runtime for generating embeddings at ~700 chunks/s on CPU.
2. **sqlite-vec**: A vector store that allows for fast storage and retrieval of embeddings in a single `.db` file.
3. **FTS5**: A full-text search engine built into SQLite for exact matches and keyword search.
4. **Reciprocal Rank Fusion**: A hybrid ranking algorithm that combines semantic and keyword search results without requiring training.
5. **Cerebras API**: A fast inference API that can process ~2,000 tokens/second, providing a fast and flexible inference option.

By combining these technologies, vstash is able to provide fast and accurate local search results without requiring a server or cloud dependencies.

---

## Query 7: "How does Retrieval-Augmented Generation reduce hallucinations in language models?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 7.5ms |
| Vector + FTS search | 2.2ms |
| **Total retrieval** | **9.7ms** |
| LLM inference | 0.99s |
| **End-to-end** | **1.0s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 10) | 0.01 |
| 2 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 1) | 0.009836 |
| 3 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 15) | 0.009677 |
| 4 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 3) | 0.009524 |
| 5 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 8) | 0.009375 |

### 💬 Answer (48 words)

According to the Wikipedia article on Retrieval-Augmented Generation, RAG reduces hallucinations in language models by incorporating information retrieval before generating responses. This allows LLMs to use domain-specific and/or updated information that is not available in the training data, thereby reducing the likelihood of generating incorrect or misleading information.

---

## Summary

| Metric | Value |
|---|---|
| Queries tested | 7 |
| Total retrieval time | 493ms (70ms avg) |
| Total LLM time | 7.0s (1.0s avg) |
| Avg end-to-end | 1.07s |
| Corpus size | 171 chunks |