## 2024-05-18 - Optimize SQLite compound key fetches with CTE VALUES
**Learning:** When retrieving a sparse set of rows matched on compound keys (like `doc_id` and `seq`), using a range query like `BETWEEN` retrieves all intermediate rows unnecessarily, which can be massively inefficient. Iterative fetches are also slow due to N+1 queries.
**Action:** Chunk the input keys to stay within parameter limits (`_SQLITE_PARAM_BATCH`), and use a Common Table Expression (CTE) with a `VALUES` clause (`WITH targets(doc_id, seq) AS (VALUES ...)`) joined to the target tables to fetch exactly the rows needed in batches.

## 2024-05-19 - Precompute Loop Invariants in Greedy Selection Algorithms
**Learning:** In greedy selection algorithms like MMR deduplication, recalculating invariant values within nested loops (such as the term `mmr_lambda * norm_score`) drastically increases computational overhead. Because `norm_score` depends only on the candidate's static score and `mmr_lambda` is a constant, recalculating this product inside the inner loop wastes $O(K \times N)$ operations.
**Action:** Always hoist per-item invariant calculations (like `mmr_lambda * norm_score` or `1 - mmr_lambda`) out of nested loops and precompute them in arrays or variables before the loop begins to reduce complexity and improve runtime performance.

## YYYY-MM-DD - [Optimize Recency Boost]
**Learning:** The pipeline was fetching `created_at` again via batched SQL for recency boosting, even though `added_at` (which populates it) was already available in the pre-fetched `ranked` dictionary payload.
**Action:** When a property is already attached to chunk dictionaries from early candidate generation stages, use `.get()` to retrieve it directly rather than performing redundant database round-trips in post-processing steps.
## YYYY-MM-DD - [Safe Property Usage]
**Learning:** Attempting to eliminate an $O(N)$ database query by blindly relying on dictionary `.get()` properties from earlier stages introduces severe regression risks if that data is ever excluded from the dictionary payload.
**Action:** When migrating from database lookups to in-memory payload fields (like `added_at`), always preserve a database fallback path for chunks where the expected property is `None` to guarantee backward compatibility and pipeline safety.
