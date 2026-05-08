## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2024-05-19 - Pre-group items to avoid linear scans in similarity updates
**Learning:** In MMR deduplication, updating the redundancy penalty for chunks of the same document previously required a linear scan over all remaining items to find siblings. This made the update process $O(K \times N)$ operations, where K is top_k and N is the number of remaining items.
**Action:** Pre-group items by document into a `doc_to_indices` dictionary, and iterate only over those sibling indices. Use a boolean mask `in_remaining` to efficiently check if an item is still active in $O(1)$ time. This drops the update complexity from $O(K \times N)$ to $O(N + K \times S)$, where S is the average number of chunks per document.
