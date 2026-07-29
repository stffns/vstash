1. **Apply lazy L2 norm evaluation and bypass similarity penalty updates in MMR**
   - Modify `vstash/store/_search.py`.
   - Instead of computing `chunk_norms = [math.hypot(*emb) if emb is not None else 0.0 for emb in chunk_embs]` upfront, initialize a lazy array `chunk_norms = [None] * len(ranked)`.
   - Update the inner loop to skip the similarity penalty updates entirely if `len(doc_to_indices[new_doc_key]) <= 1`.
   - Lazily compute the L2 norm for the newly selected chunk and for its siblings only when a similarity calculation is required.

2. **Verify changes**
   - Run tests excluding `requires_sqlite_vec` and `snapvec`.
   - Format and lint with `ruff`.

3. **Complete pre-commit steps**
   - Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.

4. **Submit PR**
   - Create a PR with title "⚡ Bolt: [performance improvement]" and required description.
