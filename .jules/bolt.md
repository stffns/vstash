## 2025-03-08 - Bolt: Batched Document Deletion Optimization
**Learning:** We replaced single-row N+1 document deletions inside loops with batched deletions (`DELETE FROM ... WHERE id IN (...)`). SQLite parameter limits were respected by splitting the batches (e.g. 100 docs per chunk, and 900 chunks).
**Action:** When performing loop-based single operations on an array of elements in SQLite, check if `IN` queries can be batched safely within `sqlite3` limits to gain significant performance speedups.
