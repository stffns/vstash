## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2024-05-20 - Grouping Siblings and Swap-with-last for MMR Deduplication
**Learning:** During MMR deduplication (`_mmr_dedup`), tracking candidates and computing diversity penalties across a `remaining` list causes severe bottlenecks: `list.remove()` takes O(N), and updating similarity arrays for just one document's sibling chunks requires scanning all remaining indices (another O(N) cost per selection).
**Action:** Replace `list.remove()` with an O(1) swap-with-last (`remaining[idx] = remaining[-1]; remaining.pop()`) guarded by an `in_remaining` boolean array. To avoid O(N) linear scans when updating penalties, pre-group chunk indices by their document key (`doc_keys`) into a dictionary (`doc_to_indices`) and only iterate through the pre-grouped siblings. This reduces the redundancy penalty update complexity from O(K * N) to O(N + K * S).
