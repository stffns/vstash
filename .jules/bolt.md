## 2025-02-28 - Optimizing `expand_context`
**Learning:** The `expand_context` function was executing an N+1 query pattern by resolving each `doc_id` individually inside a loop via a complex JOIN and text match.
**Action:** Always look for opportunities to batch database lookups. Replacing the individual queries with a single batch `IN` query to fetch `doc_id`s improved local performance by ~1.25x for large result sets. Make sure to clean up scratch/benchmark files before submitting PRs.
