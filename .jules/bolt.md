## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2026-05-29 - [Lazy L2 Norm Evaluation in MMR Dedup]
**Learning:** Eagerly evaluating L2 norms using `math.hypot()` for all items in the candidate pool during MMR (`_mmr_dedup`) incurs an $O(N)$ penalty upfront. Since MMR only penalizes chunks belonging to the same document, many chunks are never compared against siblings if they are unique to their document in the candidate list.
**Action:** Lazily compute the L2 norm only when a chunk is actually compared against a sibling. Furthermore, explicitly skip updating the diversity tracking (`max_sims`) for documents that only have a single chunk in the candidate pool, avoiding both unnecessary array iterations and norm calculations. This dropped MMR execution time for typical sizes significantly in the critical path without altering ranking accuracy.
