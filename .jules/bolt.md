## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2024-05-20 - Avoid N+1 queries in batch lookups when data is already available
**Learning:** Batched lookups (using `WHERE id IN (...)` and `_SQLITE_PARAM_BATCH`) are better than N+1 queries, but fetching data that is already populated on objects retrieved upstream is even faster and avoids the database entirely. In `_apply_recency_boost`, a batched lookup was used to fetch `created_at` from chunks, but this field is logically identical to `added_at` on documents which was already fetched and included in the `ranked` dictionary payload during the initial candidate search.
**Action:** When performing post-processing steps (like recency boosts), check if the required fields have already been fetched and populated into the candidate dictionaries in the initial SQL query instead of doing redundant batched SQL queries.
