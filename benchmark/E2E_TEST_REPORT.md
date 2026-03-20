# vstash End-to-End Test Report

**Date:** 2026-03-20 10:34
**Backend:** cerebras / gpt-oss-120b
**Embedding:** BAAI/bge-small-en-v1.5

**Corpus:** 8 documents, 171 chunks, 3.08 MB

---

## Query 1: "Explain the difference between DARTS and evolutionary NAS methods, including their computational costs"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 127.5ms |
| Vector + FTS search | 4.0ms |
| **Total retrieval** | **131.5ms** |
| LLM inference | 0.78s |
| **End-to-end** | **0.91s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Neural Architecture Search (chunk 1) | 0.01 |

### 💬 Answer (198 words)

**DARTS (gradient‑based NAS)**  
- Treats the discrete architecture space as a continuous one, introducing architecture parameters that can be optimized with gradient descent.  
- Because the search is performed by back‑propagation rather than by training many separate models, the cost drops from “thousands of GPU‑days” (typical of early RL‑based or evolutionary searches) to only “a few GPU‑days”【Context 1†3.3】.  

**Evolutionary NAS (population‑based search)**  
- Maintains a population of candidate architectures and iteratively creates new candidates by mutating or recombining existing ones.  
- The tournament selection strategy samples a random subset of the population and promotes the best member as the parent for the next generation, which is noted as “particularly effective” and “memory‑efficient”【Context 1†3.2】.  
- Although more memory‑efficient than some RL approaches, evolutionary methods still require evaluating many architectures, so their computational cost is higher than DARTS (typically measured in many GPU‑days rather than a few).  

**Key differences**  
1. **Search mechanism** – DARTS uses differentiable, gradient‑based optimization; evolutionary NAS uses stochastic population evolution with selection, mutation, and crossover.  
2. **Computational cost** – DARTS reduces search time dramatically to a few GPU‑days, whereas evolutionary NAS, while memory‑efficient, still incurs a larger cost because many architectures must be evaluated across generations.

---

## Query 2: "What are the main arguments for renewable energy, and how do costs compare to fossil fuels?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 9.9ms |
| Vector + FTS search | 1.3ms |
| **Total retrieval** | **11.1ms** |
| LLM inference | 0.72s |
| **End-to-end** | **0.73s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Climate Change Report (chunk 1) | 0.01 |
| 2 | Climate Change Report (chunk 0) | 0.009836 |

### 💬 Answer (165 words)

**Main arguments for renewable energy (from the report)**  

- **Rapid cost declines:** Solar photovoltaic prices have fallen ≈ 89 % since 2010, and on‑shore wind costs have dropped ≈ 70 % in the same period【Context 2, 3.1 Renewable Energy Transition】.  
- **Dominance in new capacity:** Renewable technologies now provide the majority of new power‑capacity additions worldwide【Context 2, 3.1 Renewable Energy Transition】.  
- **Decarbonisation benefit:** Shifting to renewables is a key part of limiting global warming, as they emit little or no CO₂ compared with fossil‑fuel power plants (implied by the overall mitigation focus of the report).  

**Cost comparison to fossil fuels**

The provided excerpts give the magnitude of renewable‑energy cost reductions but do **not** include explicit cost figures for fossil‑fuel generation, so a direct numeric comparison cannot be drawn from the context. The report only states that renewable costs have “plummeted” and that they now dominate new capacity additions, suggesting they are becoming cost‑competitive with fossil fuels, but no specific fossil‑fuel cost numbers are given.

---

## Query 3: "Describe Alice's emotional journey and key character interactions in Wonderland"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 9.1ms |
| Vector + FTS search | 1.6ms |
| **Total retrieval** | **10.6ms** |
| LLM inference | 1.41s |
| **End-to-end** | **1.42s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Alice Wonderland (chunk 21) | 0.01 |
| 2 | Alice Wonderland (chunk 5) | 0.009836 |
| 3 | Alice Wonderland (chunk 4) | 0.009677 |
| 4 | Alice Wonderland (chunk 2) | 0.009524 |
| 5 | Alice Wonderland (chunk 20) | 0.009375 |

### 💬 Answer (494 words)

**Alice’s emotional arc**

1. **Indignation & anger** – When she first sits at the tea‑party she feels “indignantly” and later “angrily” at the March Hare for offering wine that isn’t there and for being invited without permission【Context 5†L1-L9】.  
2. **Confusion & puzzlement** – After the Hatter’s riddles and the strange “watch that tells the day but not the hour,” Alice is “dreadfully puzzled” and admits she “doesn’t quite understand” him【Context 1†L84-L90】.  
3. **Frustration & fatigue** – While shrinking and swimming in her own tears she repeatedly bursts into tears, declares she is “very tired of being all alone here,” and complains about the endless crying【Context 2†L1-L9】【Context 3†L1-L7】.  
4. **Fear & relief** – She is “a good deal frightened” when she almost shrinks away, then sighs with relief after escaping the narrow‑escape with the fan【Context 2†L13-L18】.  
5. **Self‑doubt & identity crisis** – In the hall she wonders “Who in the world am I?” and tries to match herself to other girls (Ada, Mabel), repeatedly questioning her own identity【Context 3†L31-L53】.  
6. **Hope & determination** – Finding the tiny golden key, the “DRINK ME” bottle and shrinking to a size that fits the little door, she feels hopeful about entering the garden, though later worries about shrinking further【Context 4†L57-L78】.  
7. **Resignation & curiosity** – After the Cat’s disappearing act she comments that “the most curious thing” is a grin without a cat, showing a mix of resignation and fascination【Context 5†L71-L78】.

**Key character interactions**

| Character | Interaction | Emotional impact on Alice |
|-----------|-------------|---------------------------|
| **March Hare** | Offers wine that isn’t there, argues about civility, asks riddles, checks the watch, repeats “It was the best butter.” | Triggers Alice’s anger, confusion, and later amusement at his absurd logic. |
| **Hatter** | Asks the famous riddle, comments on the watch, questions the meaning of his own words, repeatedly says “Why is a raven like a writing‑desk?” | Causes puzzlement and a sense of being out‑of‑sync with English logic. |
| **Dormouse** | Sleeps on the table, is used as a cushion, speaks in a “shrill, passionate voice” about cats, disappears and reappears. | Adds to Alice’s feeling of absurdity and curiosity. |
| **White Rabbit** | Returns with gloves and fan, drops them in panic, prompting Alice to take them and fan herself. | Heightens Alice’s desperation and need for comfort. |
| **The Cat** | Declares “we’re all mad here,” explains his own madness, vanishes repeatedly, asks about the baby‑turned‑pig. | Reinforces the theme of madness, makes Alice feel both unsettled and intrigued. |
| **Other figures (Ada, Mabel, etc.)** | Serve as reference points for Alice’s identity test. | Deepen her self‑doubt and the “who am I?” crisis. |

Overall, Alice moves from **indignant anger** at the tea‑party’s absurdities, through **confusion, fear, and self‑doubt**, to **hopeful determination** when she finally shrinks enough to pass through the little door, all while engaging with a cast of bizarre characters whose illogical behavior fuels her emotional turbulence.

---

## Query 4: "What does Sun Tzu say about the importance of knowing your enemy and terrain?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 13.1ms |
| Vector + FTS search | 2.9ms |
| **Total retrieval** | **16.0ms** |
| LLM inference | 0.95s |
| **End-to-end** | **0.96s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Art Of War (chunk 32) | 0.01 |
| 2 | Art Of War (chunk 54) | 0.009836 |
| 3 | Art Of War (chunk 51) | 0.009677 |
| 4 | Art Of War (chunk 74) | 0.009524 |
| 5 | Art Of War (chunk 58) | 0.009375 |

### 💬 Answer (187 words)

Sun Tzu makes two complementary points:

1. **Knowing the enemy (and yourself) is the foundation of victory.**  
   - He writes that *“If you know the enemy and know yourself, you need not fear the result of a hundred battles… If you know neither the enemy nor yourself, you will succumb in every battle.”*【Context 1†L1-L4】.  
   - Conversely, ignorance of the enemy (or of yourself) guarantees defeat【Context 1†L5-L7】.

2. **Understanding and exploiting terrain is equally vital.**  
   - He classifies six kinds of ground and advises that on *accessible* terrain you should “be before the enemy in occupying the raised and sunny spots, and carefully guard your line of supplies”【Context 5†L1-L4】.  
   - On *temporising* ground (where neither side gains by moving first) he recommends “not to stir forth, but rather to retreat, thus enticing the enemy… then… deliver our attack with advantage”【Context 5†L6-L8】.  
   - Throughout the chapters on signs and marching, he stresses reading the enemy’s movements and the land (e.g., dust, birds, screens) to anticipate attacks【Context 2†L18‑L23】.

Together, Sun Tzu teaches that success depends on *knowing the opponent* and *knowing the ground* on which the conflict will be fought.

---

## Query 5: "How should a production API implement rate limiting and caching with Redis?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 11.8ms |
| Vector + FTS search | 1.5ms |
| **Total retrieval** | **13.3ms** |
| LLM inference | 0.68s |
| **End-to-end** | **0.69s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Fastapi Patterns (chunk 1) | 0.01 |

### 💬 Answer (158 words)

A production API can use Redis both to enforce rate limits and to store cached data.

**Rate‑limiting** – the context shows a simple Redis‑backed implementation:

```python
async def rate_limit(key: str, limit: int, window: int) -> bool:
    current = await redis.incr(key)          # increment request count
    if current == 1:
        await redis.expire(key, window)      # set TTL for the window
    return current <= limit                  # allow request if under limit
```

* Create a unique `key` per client (e.g., IP address or API token).  
* Choose a `limit` (max requests) and a `window` (seconds).  
* The function increments the counter atomically, sets an expiration on the first hit, and returns whether the request is within the allowed quota.

**Caching** – the provided context does not include a specific Redis caching pattern or code example. It only mentions “cache availability” in the introductory bullet points, but no implementation details are given. Therefore, the exact caching strategy would need to be defined elsewhere.

---

## Query 6: "What is vstash's technology stack and how does it achieve fast local search?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 13.2ms |
| Vector + FTS search | 1.1ms |
| **Total retrieval** | **14.3ms** |
| LLM inference | 1.02s |
| **End-to-end** | **1.03s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | Readme (chunk 0) | 0.01 |
| 2 | Vstash Constitution (chunk 0) | 0.009836 |

### 💬 Answer (313 words)

**vstash’s technology stack**

| Layer | Technology (as listed in the docs) | Why it’s used |
|-------|------------------------------------|---------------|
| **Embeddings** | **FastEmbed** (ONNX Runtime) | Generates embeddings locally at ~700 chunks / s, no server needed【Context 1†L23-L26】 |
| **Vector store** | **sqlite‑vec** | Single‑file `.db`, cosine‑similarity search, zero external dependencies【Context 1†L27-L30】 |
| **Keyword search** | **FTS5** (SQLite) | Built‑in exact‑match + porter‑stemming index, runs entirely in SQLite【Context 1†L31-L33】 |
| **Hybrid ranking** | **Reciprocal Rank Fusion** | Combines semantic (vector) and keyword results without any training【Context 1†L34-L36】 |
| **Inference** | **Cerebras API**, **Ollama**, **OpenAI** (configurable) | Fast remote generation (≈2 000 tok/s via Cerebras) or fully‑local generation via Ollama【Context 1†L38-L44】 |
| **Parsing** | **markitdown** | Handles PDF, DOCX, PPTX, HTML, Markdown, URLs, etc. locally【Context 1†L45-L48】 |
| **Configuration** | **Pydantic v2** | Type‑safe config validation【Context 2†L53-L55】 |
| **CLI** | **Typer** + **Rich** | Clean, interactive terminal UI【Context 2†L57-L58】 |
| **Language** | **Python 3.10+** | Broad ecosystem, rapid development【Context 2†L59-L60】 |

**How fast local search is achieved**

1. **In‑process embedding generation** – FastEmbed runs on the CPU via ONNX, producing embeddings at ~700 chunks per second, eliminating network latency【Context 1†L23-L26】.  
2. **Single‑file vector store** – `sqlite‑vec` stores embeddings in a local SQLite file and performs cosine‑similarity look‑ups directly, avoiding external databases【Context 1†L27-L30】.  
3. **Built‑in keyword index** – SQLite’s FTS5 provides instant exact‑match and stemmed searches without a separate search engine【Context 1†L31-L33】.  
4. **Hybrid ranking with Reciprocal Rank Fusion** – Merges semantic and keyword results locally, giving high‑quality rankings without costly model training【Context 1†L34-L36】.  
5. **No server requirement** – All of the above run on the user’s machine; only the optional inference step may contact a remote API, keeping the search path entirely local and sub‑second【Context 2†L3-L5】【Context 2†L13-L15】.  

Together, these components let vstash retrieve relevant document chunks in under a second, delivering “instant” semantic search while staying fully local.

---

## Query 7: "How does Retrieval-Augmented Generation reduce hallucinations in language models?"

### ⏱️ Timing

| Phase | Time |
|---|---|
| Embed query | 8.2ms |
| Vector + FTS search | 0.8ms |
| **Total retrieval** | **9.0ms** |
| LLM inference | 1.68s |
| **End-to-end** | **1.69s** |

### 📚 Retrieved Sources

| # | Source | Score |
|---|---|---|
| 1 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 10) | 0.01 |
| 2 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 8) | 0.009836 |
| 3 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 1) | 0.009677 |
| 4 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 15) | 0.009524 |
| 5 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation (chunk 5) | 0.009375 |

### 💬 Answer (141 words)

Retrieval‑augmented generation (RAG) lowers the chance that a language model “hallucinates” by forcing the model to base its output on actual documents it pulls from an external knowledge store instead of relying only on its internal parameters.

* After a query is encoded, a **retriever** fetches the most relevant passages from a vector database.  
* The generator then conditions its response on both the original prompt **and the retrieved texts** (“the LLM can generate output based on both the query and the retrieved documents”【Context 5】).  
* Improvements such as better encoder vectors, late‑interaction scoring, hybrid sparse‑dense representations, and retriever‑centric training (e.g., Inverse Cloze Task pre‑training, supervised optimization, reranking) make the retrieved material more accurate and relevant, further limiting the model’s tendency to fabricate information【Context 5】.  

By grounding generation in verified external sources, RAG helps keep the model from inventing unsupported facts.

---

## Summary

| Metric | Value |
|---|---|
| Queries tested | 7 |
| Total retrieval time | 206ms (29ms avg) |
| Total LLM time | 7.2s (1.03s avg) |
| Avg end-to-end | 1.06s |
| Corpus size | 171 chunks |