## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2024-07-27 - Pre-fetch properties to avoid N+1 queries in batch post-processing
**Learning:** In pipelines where a large number of database rows are fetched, ranked, and then subsequently processed (e.g., in a recency boost stage), attempting to lazily fetch missing properties with batched `IN` queries can still lead to an N+1 performance bottleneck over the candidate pool.
**Action:** When querying the initial candidate chunks (e.g., `vec_chunks`, `fts_chunks`), eagerly select necessary properties (like `created_at`) and pass them through the fusion pipeline dictionaries, allowing post-processing loops to run entirely in Python memory without additional database roundtrips.
