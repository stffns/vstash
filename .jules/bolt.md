## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2025-02-13 - Lazy Evaluate L2 Norms and Bypass Penalty Updates in MMR
**Learning:** Eagerly evaluating values that may not be used, combined with running iterations when there is no work to do, negatively affects performance. In MMR deduplication, `chunk_norms` (L2 norms) were precomputed for all chunks, but only a fraction are evaluated during similarity updates. Furthermore, the similarity penalty updates are meaningless if a chosen document has no other remaining sibling chunks.
**Action:** Initialize values lazily (`[None] * len(...)`), evaluate them only when accessed. Introduce an early exit before nested loops: bypass similarity penalty updates (like `max_sims`) entirely if the selected chunk is the only one from its document (`len(doc_indices) <= 1`).
