## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2025-05-20 - Lazily evaluate expensive computations in nested loops when they can be bypassed
**Learning:** In MMR deduplication, L2 norms (via `math.hypot`) were eagerly precomputed for all candidate chunks, but they are only ever used when a chunk is actually compared against a selected sibling from the *same* document. If a document only has one chunk, or a chunk is never compared against siblings, its L2 norm calculation is completely wasted.
**Action:** Initialize arrays with `None` and lazily compute values (like L2 norms) only when they are first accessed. Additionally, wrap blocks containing these computations with fast-path bypasses (like checking if the document has siblings) to skip them entirely when possible.
