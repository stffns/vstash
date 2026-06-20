## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2024-05-20 - In-place Sorting Avoids Reallocating Lists
**Learning:** In hot paths (like search rankings or combining items), `sorted(lst, key=...)` allocates a new list, creating memory overhead and adding unnecessary object creations.
**Action:** When working with existing lists or generating lists from other structures like dictionaries (e.g. `list(d.values())`), use the in-place `.sort(key=...)` method to avoid the allocation overhead while maintaining fast O(N log N) sorting.

## 2024-05-20 - Lazily Evaluate Distances in MMR
**Learning:** In MMR deduplication, computing L2 norms (e.g., using `math.hypot`) for *all* candidates upfront results in unnecessary computations because many candidates might never actually be evaluated against siblings. Moreover, updating similarity penalties via `_cosine_sim` is entirely unneeded if the selected chunk is the only one from its document.
**Action:** Lazily evaluate L2 norms only when a chunk is actually compared against a selected sibling, and bypass similarity penalty updates entirely if the selected chunk is the only one from its document.
