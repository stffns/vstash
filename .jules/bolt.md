## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2024-05-20 - Lazy Evaluation of L2 Norms in MMR Deduplication
**Learning:** In MMR deduplication, computing the L2 norm for every candidate chunk upfront via `math.hypot` scales linearly with the number of remaining candidates ($O(N)$), which is wasteful if many chunks are never compared against the selected one. Additionally, updating redundancy penalties (`max_sims`) can be completely bypassed if the newly selected chunk is the only remaining chunk from its document.
**Action:** Implement lazy evaluation by initializing a list of `None` for chunk norms and calculating `math.hypot` only when a candidate is explicitly compared during the similarity calculation. Furthermore, introduce a fast path (`if len(doc_to_indices[new_doc_key]) <= 1: continue`) to bypass penalty updates entirely for documents that have no siblings left in the remaining pool.
