## 2024-05-18 - Resolve N+1 SQLite queries for Document Deletions
**Learning:** Document deletion loops previously iterated path-by-path, triggering individual transactions and queries.
**Action:** Created `delete_documents` to resolve N+1 queries by leveraging `WITH targets(path) AS (VALUES ...)` logic and processing paths under `_SQLITE_PARAM_BATCH`.
