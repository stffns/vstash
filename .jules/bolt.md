## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2024-06-25 - [Lazy Evaluation for High-Dimensional Vector Norms]
**Learning:** In purely Python-based algorithms like Maximal Marginal Relevance (MMR) deduplication, eagerly evaluating expensive operations (like `math.hypot` for L2 norms) on all candidate vectors can be heavily wasteful if many are never actually compared (e.g. because they are the only hit from their document). Fast-path bypasses combined with lazy initialization arrays perform much faster.
**Action:** When working on greedy diversity algorithms (MMR) or sibling penalty loops, delay vector norm calculations until the exact moment of comparison and skip penalty updates entirely if a document only contributes a single chunk to the candidate pool. Use a `[None] * len(items)` array for cached lazy evaluation.
