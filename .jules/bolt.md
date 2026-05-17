## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## 2026-05-17 - O(1) Swap-with-last Removal and Pre-grouping in Hot Loops
**Learning:** In tight ranking/deduplication hot loops (like MMR), iterating repeatedly over remaining candidate chunks with O(N) `list.remove()` operations and O(N) membership/property checks (e.g. "does this chunk have the same `doc_key`?") scales very poorly, dropping from O(K * N) to slower empirical speeds due to array shifts and linear scanning.
**Action:** In loops tracking 'remaining' candidates:
1. Replace `list.remove()` with an O(1) swap-with-last approach (`remaining[pos] = remaining[-1]; remaining.pop()`), tracking index positions.
2. Introduce an `in_remaining` boolean array accessed directly via candidate index for O(1) membership checks.
3. Pre-group indices by keys (like `doc_to_indices`) to directly access matching elements without full iteration scans.
