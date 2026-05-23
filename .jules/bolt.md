## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2024-05-23 - Optimize O(N) array removals and nested linear scans in MMR
**Learning:** During MMR deduplication, maintaining a shrinking `remaining` list via `list.remove()` causes `O(N)` shifting costs for every selected item. Furthermore, scanning the entire remaining list to find sibling chunks from the same document incurs `O(K * N)` complexity, severely bottlenecking large document corpus search results.
**Action:** Replace `list.remove()` with an `O(1)` swap-with-last removal (`remaining[idx] = remaining[-1]; remaining.pop()`) guarded by an `in_remaining` boolean mask for `O(1)` membership checks. Pre-group chunk indices by document into a dictionary (`doc_to_indices`) to drop redundancy scan complexity from `O(K * N)` to `O(N + K * S)` where `S` is the average chunks per document.
