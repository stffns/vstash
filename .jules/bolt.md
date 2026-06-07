## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2024-06-07 - Lazy Evaluation of Sequence Operations in Greedy Selection
**Learning:** Eagerly evaluating properties (like L2 norms for cosine similarity) on all candidates prior to entering a greedy selection loop can result in unnecessary overhead. In the case of `_mmr_dedup`, many chunks are never compared to other siblings from the same document (especially if they are the only chunk for that document in the top-k list). Eager calculation computes values that may never be read.
**Action:** Lazily initialize such structures with `None` and calculate values just-in-time only for items undergoing active comparison. In addition, completely skip iterations if properties like sibling-count dictate that calculations are not necessary.
