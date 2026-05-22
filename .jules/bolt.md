## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2024-05-22 - O(1) swap-with-last removal in hot loops
**Learning:** In MMR deduplication (`_mmr_dedup`), using `list.remove()` inside the top-k selection loop introduces an O(N) penalty because it requires scanning the list and shifting elements. Combined with an O(N) array scan to update siblings, this degrades performance.
**Action:** Replace `list.remove()` with an O(1) swap-with-last removal (`remaining[idx] = remaining[-1]; remaining.pop()`). Use a boolean mask array (`in_remaining`) for O(1) membership checks, and pre-group sibling indices into a dictionary (`doc_to_indices`) to avoid O(N) array scans when updating similarity penalties.
