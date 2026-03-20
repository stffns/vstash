# vstash End-to-End Test Report

**Date:** 2026-03-20 10:36
**Backend:** cerebras / gpt-oss-120b
**Embedding:** BAAI/bge-small-en-v1.5

**Corpus:** 8 documents, 171 chunks, 3.08 MB

---

## Query 1: "Explain the difference between DARTS and evolutionary NAS methods, including their computational costs"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 7.6ms |
| Vector + FTS search | 1.3ms |
| **Total retrieval** | **8.9ms** |
| LLM inference | 0.94s |
| **End-to-end** | **0.95s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Neural Architecture Search (chunk 1) | 0.01 |

### 💬 Answer (250 words)

**DARTS (gradient‑based NAS)**  
- Treats the architecture search space as continuous and optimises the architecture parameters with gradient descent.  
- Because it avoids training a separate model for every candidate, the search cost drops from *thousands of GPU‑days* (typical of earlier RL‑based or evolutionary searches) to only *a few GPU‑days*【Context 1†3.3】.  

**Evolutionary NAS**  
- Works with a population of discrete architectures that are mutated and recombined over generations.  
- Uses strategies such as tournament selection, where a random subset of the population is sampled and the best individual becomes the parent for the next generation【Context 1†3.2】.  
- Although it can “match or exceed RL‑based methods while being more memory‑efficient,” the approach still requires evaluating many architectures, so its computational cost is generally higher than DARTS (it does not enjoy the same reduction to a few GPU‑days).  

**Key differences**  
| Aspect | DARTS | Evolutionary NAS |
|--------|-------|-------------------|
| Search paradigm | Gradient‑based optimisation of continuous architecture parameters | Population‑based mutation/selection of discrete architectures |
| Typical computational cost | Few GPU‑days (dramatically reduced) | Higher (requires evaluating many candidates), though more memory‑efficient than RL‑based evolution【Context 1†3.2】 |
| Main advantage | Very fast search, low GPU time | Can explore a broader discrete space and often yields competitive or better results, with good memory usage |

Thus, DARTS achieves a much lower GPU‑time cost by turning the search into a differentiable problem, whereas evolutionary NAS relies on evaluating many architectures across generations, leading to higher computational expense despite being memory‑efficient.

---

## Query 2: "What are the main arguments for renewable energy, and how do costs compare to fossil fuels?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 12.1ms |
| Vector + FTS search | 2.0ms |
| **Total retrieval** | **14.1ms** |
| LLM inference | 0.76s |
| **End-to-end** | **0.77s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Climate Change Report (chunk 1) | 0.01 |
| 2 | Climate Change Report (chunk 0) | 0.009836 |

### 💬 Answer (149 words)

**Main arguments for renewable energy (from the report):**

1. **Rapid cost declines** – Solar photovoltaic costs have fallen ≈ 89 % since 2010, and on‑shore wind costs have dropped ≈ 70 % over the same period【Context 2, §3.1 Renewable Energy Transition】.  
2. **Dominance in new capacity** – Renewable energy now provides the majority of new power‑capacity additions worldwide【Context 2, §3.1 Renewable Energy Transition】.  
3. **Climate‑mitigation benefit** – Shifting to renewables reduces greenhouse‑gas emissions, helping limit warming (the report’s broader focus on mitigation)【Context 2, Executive Summary & §3.1】.  
4. **Challenges to address** – Grid integration, storage, and a just transition for fossil‑fuel‑dependent communities must be managed【Context 2, §3.1】.

**Cost comparison to fossil fuels:**  
The provided excerpts give detailed information on how renewable‑energy costs have fallen, but they do **not** include any specific cost figures for fossil‑fuel generation or a direct cost‑comparison statement. Therefore, the report does not supply that comparison.

---

## Query 3: "Describe Alice's emotional journey and key character interactions in Wonderland"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 13.1ms |
| Vector + FTS search | 1.9ms |
| **Total retrieval** | **14.9ms** |
| LLM inference | 1.0s |
| **End-to-end** | **1.01s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Alice Wonderland (chunk 21) | 0.01 |
| 2 | Alice Wonderland (chunk 5) | 0.009836 |
| 3 | Alice Wonderland (chunk 4) | 0.009677 |
| 4 | Alice Wonderland (chunk 2) | 0.009524 |
| 5 | Alice Wonderland (chunk 20) | 0.009375 |

### 💬 Answer (459 words)

**Alice’s emotional journey**

- **Initial irritation and anger** – When she sits at the tea‑party she is “indignantly” angry about the lack of wine and the March Hare’s rudeness, and she rebukes both the Hare and the Hatter for “personal remarks” and “civil” behaviour【Context 5†L1-L9】.  
- **Confusion and puzzlement** – After the Hatter asks riddles (“Why is a raven like a writing‑desk?”) Alice tries to solve them, feels “dreadfully puzzled” by the Hatter’s nonsensical comments about watches and years【Context 1†L71-L78】.  
- **Frustration and fatigue** – Later, while shrinking and swimming in her own tears, she laments being “very tired of being all alone here” and cries repeatedly【Context 2†L1-L9】【Context 3†L1-L7】.  
- **Self‑doubt and identity crisis** – She wonders “Who in the world am I?” and compares herself to other girls (Ada, Mabel), trying arithmetic and geography only to get wrong answers, showing deep uncertainty about her identity【Context 3†L31-L55】.  
- **Hope and determination** – Finding the tiny golden key, the “DRINK ME” bottle and shrinking to a size that fits the little door, she feels a “curious feeling” and decides to go into the garden, showing renewed resolve【Context 4†L45-L61】.  
- **Relief mixed with lingering anxiety** – After drinking, she is “now only ten inches high” and ready to leave, yet worries that she might shrink further or disappear “like a candle”【Context 4†L62-L71】.

**Key character interactions**

| Character | Interaction with Alice | Effect / Notable Quote |
|-----------|------------------------|------------------------|
| **March Hare** | Offers wine that isn’t there, argues about civility, asks riddles, critiques her answer about the date, and repeatedly says “It was the best butter.” | “Then it wasn’t very civil of you to offer it,” Alice angrily replies【Context 5†L3-L7】. |
| **Mad Hatter** | Asks the famous riddle, debates logical equivalences (“I see what I eat” vs. “I eat what I see”), questions the watch, and later asks “What day of the month is it?” | “Why is a raven like a writing‑desk?”【Context 1†L45-L48】. |
| **Dormouse** | Sleeps on the table, is used as a cushion, and later speaks in a “shrill, passionate voice” about cats【Context 5†L13-L20】. |
| **White Rabbit** | Returns with gloves and a fan, drops them in panic, prompting Alice to take them and continue fanning herself【Context 3†L9-L18】. |
| **Cheshire Cat** | Appears and vanishes repeatedly, discusses madness, and asks “Did you say pig, or fig?” showing a playful, teasing tone【Context 5†L31-L44】. |
| **Other figures (Ada, Mabel, etc.)** | Serve as reference points for Alice’s identity crisis; she rules herself out of being them based on physical and knowledge differences【Context 3†L31-L45】. |

Overall, Alice moves from **anger and confusion** through **fatigue, self‑doubt, and puzzlement**, finally reaching **hopeful determination** as she overcomes the bizarre obstacles and interacts with the eccentric inhabitants of Wonderland.

---

## Query 4: "What does Sun Tzu say about the importance of knowing your enemy and terrain?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 8.9ms |
| Vector + FTS search | 0.9ms |
| **Total retrieval** | **9.7ms** |
| LLM inference | 0.89s |
| **End-to-end** | **0.9s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Art Of War (chunk 32) | 0.01 |
| 2 | Art Of War (chunk 54) | 0.009836 |
| 3 | Art Of War (chunk 51) | 0.009677 |
| 4 | Art Of War (chunk 74) | 0.009524 |
| 5 | Art Of War (chunk 58) | 0.009375 |

### 💬 Answer (212 words)

Sun Tzu makes two linked points:

1. **Knowing the enemy (and yourself) is the foundation of victory.**  
   - He states: “If you know the enemy and know yourself, you need not fear the result of a hundred battles… If you know neither the enemy nor yourself, you will succumb in every battle.” (§ Context 1)  
   - He adds that “knowing the enemy enables you to take the offensive, knowing yourself enables you to stand on the defensive.” (§ Context 1)

2. **Knowing the terrain is equally crucial for choosing the right tactics.**  
   - Sun Tzu classifies six kinds of ground (accessible, entangling, temporising, narrow passes, precipitous heights, distant positions) and advises that on **accessible ground** you should “be before the enemy in occupying the raised and sunny spots, and carefully guard your line of supplies.” (§ Context 5)  
   - On **temporising ground** (where neither side gains by moving first) he recommends “not to stir forth, but rather to retreat, thus enticing the enemy… then attack with advantage.” (§ Context 5)  
   - For **narrow passes** he says to occupy them first, garrison them strongly, and await the enemy. (§ Context 5)

Together, Sun Tzu teaches that success depends on a clear understanding of both the opponent’s intentions and the characteristics of the battlefield.

---

## Query 5: "How should a production API implement rate limiting and caching with Redis?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 10.1ms |
| Vector + FTS search | 1.2ms |
| **Total retrieval** | **11.4ms** |
| LLM inference | 0.74s |
| **End-to-end** | **0.75s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Fastapi Patterns (chunk 1) | 0.01 |

### 💬 Answer (183 words)

A production API can use Redis both to enforce rate limits and to store cached data.  

**Rate‑limiting**  
1. Choose an algorithm that fits the traffic pattern (e.g., fixed‑window, sliding‑window, token‑bucket, or leaky‑bucket).  
2. Implement the limit check with a Redis key that tracks the request count for a given identifier (IP, user‑id, etc.). The example from the context uses a simple fixed‑window counter:

```python
async def rate_limit(key: str, limit: int, window: int) -> bool:
    current = await redis.incr(key)          # increment request count
    if current == 1:
        await redis.expire(key, window)      # set TTL for the window
    return current <= limit                  # allow request if under limit
```

- `key` identifies the client (e.g., `"rate:user:123"`).  
- `limit` is the maximum requests allowed in the `window` (seconds).  
- Redis’ atomic `INCR` and `EXPIRE` guarantee correct counting even under high concurrency.

**Caching**  
The provided context does not include details on how to implement caching with Redis, so a specific caching pattern cannot be derived from it. You would need additional guidance or documentation to design the caching layer.  

*Source: “Rate Limiting and Throttling” section, FastAPI Patterns document.*

---

## Query 6: "What is vstash's technology stack and how does it achieve fast local search?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 13.1ms |
| Vector + FTS search | 1.8ms |
| **Total retrieval** | **14.8ms** |
| LLM inference | 0.86s |
| **End-to-end** | **0.88s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Readme (chunk 0) | 0.01 |
| 2 | Vstash Constitution (chunk 0) | 0.009836 |

### 💬 Answer (314 words)

**vstash’s technology stack**

| Layer | Technology | Why it’s used |
|-------|------------|---------------|
| **Embeddings** | **FastEmbed (ONNX Runtime)** – `BAAI/bge-small-en-v1.5` (384‑dim) | Generates embeddings locally at ~700 chunks / s, no server needed【1†L19-L23】 |
| **Vector store** | **sqlite‑vec** (SQLite extension) | Stores embeddings in a single `.db` file, supports cosine similarity, zero external dependencies【1†L25-L27】 |
| **Keyword search** | **FTS5 (SQLite)** | Built‑in full‑text search with exact matches and Porter stemming【1†L29-L31】 |
| **Hybrid ranking** | **Reciprocal Rank Fusion** | Combines semantic (vector) and keyword results without any training【1†L33-L35】 |
| **Inference** | Cerebras API / Ollama / OpenAI (configurable) | Remote fast API (≈2 000 tok/s) or fully local model via Ollama【1†L37-L41】 |
| **Parsing** | **markitdown** | Handles PDF, DOCX, PPTX, HTML, Markdown, URLs, etc. locally【1†L43-L45】 |
| **Configuration** | **Pydantic v2** | Type‑safe config validation【2†L71-L73】 |
| **CLI** | **Typer + Rich** | Clean terminal UI with progress feedback【2†L77-L78】 |
| **Language** | **Python 3.10+** | Easy ecosystem, fast development【2†L79-L80】 |

**How fast local search is achieved**

1. **All heavy‑lifting stays on the machine** – parsing, chunking, embedding generation, vector storage, and keyword indexing run locally, avoiding network latency【2†L1-L4】.  
2. **Fast embedding model** – FastEmbed on ONNX runs at ~700 chunks per second, producing vectors instantly【1†L19-L23】.  
3. **Single‑file vector store** – `sqlite‑vec` keeps embeddings in a compact SQLite DB, enabling rapid cosine‑similarity lookups without a separate server【1†L25-L27】.  
4. **Built‑in full‑text search** – SQLite’s FTS5 provides instant exact‑match and stemmed keyword retrieval【1†L29-L31】.  
5. **Hybrid ranking** – Reciprocal Rank Fusion merges semantic and keyword results efficiently, giving the best of both worlds without extra model training【1†L33-L35】.  
6. **No server requirement** – The entire pipeline (except optional remote inference) runs in‑process, so queries complete in sub‑second times, meeting the “speed as a feature” principle【2†L4-L5】.  

Together, these components let vstash retrieve relevant document chunks locally in under a second, delivering instant semantic search.

---

## Query 7: "How does Retrieval-Augmented Generation reduce hallucinations in language models?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 9.2ms |
| Vector + FTS search | 1.6ms |
| **Total retrieval** | **10.8ms** |
| LLM inference | 0.73s |
| **End-to-end** | **0.74s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 10) | 0.01 |
| 2 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 8) | 0.009836 |
| 3 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 1) | 0.009677 |
| 4 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 15) | 0.009524 |
| 5 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 5) | 0.009375 |

### 💬 Answer (139 words)

Retrieval‑augmented generation (RAG) lowers the chance that a language model “hallucinates” by **grounding its output in external documents that are fetched at inference time**.  
- After a query is encoded, a retriever pulls relevant passages from a vector store; the generator then conditions its response on both the original prompt **and the retrieved texts**【5†L1-L3】.  
- Because the model can copy or paraphrase factual information that it has actually seen, it is less likely to fabricate details.  
- Additional RAG‑specific improvements (e.g., re‑ranking of retrieved documents, context‑selection modules, and fine‑tuning of the generator on retrieved‑grounded data) further tighten the link between the answer and the evidence, further curbing hallucinations【5†L4-L9】.  

Thus, by integrating a retrieval step that supplies concrete evidence and by refining that step through encoder, retriever‑centric, and language‑model enhancements, RAG helps language models produce more accurate, less hallucinated outputs.

---

## Summary

| Metric | Value |
|---|---|
| Queries tested | 7 |
| Total retrieval time | 85ms (12ms avg) |
| Total LLM time | 5.9s (0.85s avg) |
| Avg end-to-end | 0.86s |
| Corpus size | 171 chunks |