## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2025-02-23 - Lazy L2 Norm Evaluation and In-Place List Sorting in Hot Paths
**Learning:** In MMR deduplication, precomputing L2 norms (e.g., `math.hypot`) for all candidates is inefficient when most chunks are never compared against a selected sibling. Additionally, `list.sort()` is significantly faster than `sorted()` for existing lists, while `sorted(d.values())` remains faster than `list(d.values()).sort()` for dictionaries.
**Action:** Lazily evaluate L2 norms only when a chunk is actually compared against a sibling. Bypass similarity penalty updates entirely if the selected chunk is the only one from its document. Use in-place `.sort()` on existing lists in hot loops like search ranking (`_apply_recency_boost`) to avoid the overhead of allocating new lists.
