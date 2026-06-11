## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2024-06-11 - Lazy Evaluation of L2 Norms in MMR Deduplication
**Learning:** In MMR deduplication, computing L2 norms (`math.hypot`) upfront for all remaining chunks is a significant O(N) performance bottleneck when searching over large contexts. If a chunk is the only chunk from its document or if its siblings have already been processed/selected, the penalty calculation isn't even triggered, meaning the eager computation was totally wasted.
**Action:** Always use lazy evaluation for L2 norms (and other heavy invariants) during deduplication loops. Initialize an array of `None` and calculate the norm on the fly only when the chunk is actively compared against a sibling. Also, ensure the similarity penalty loop checks `len(siblings) > 1` early to bypass processing entirely if there are no sibling chunks.
