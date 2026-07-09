import math
import time
import random

def _cosine_sim(a, b, norm_a, norm_b):
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return sum(x*y for x, y in zip(a, b)) / (norm_a * norm_b)

def run_eager(ranked, embeddings, top_k, mmr_lambda=0.5):
    scores = [float(r["rrf"]) for r in ranked]
    s_min, s_max = min(scores), max(scores)
    s_range = s_max - s_min if s_max > s_min else 1.0
    norm_scores = [(s - s_min) / s_range for s in scores]
    relevance_terms = [mmr_lambda * ns for ns in norm_scores]
    penalty_multiplier = 1.0 - mmr_lambda
    doc_keys = [str(r["path"]) for r in ranked]
    chunk_embs = [embeddings.get(int(r["id"])) for r in ranked]
    chunk_norms = [math.hypot(*emb) if emb is not None else 0.0 for emb in chunk_embs]
    doc_to_indices = {}
    for i, doc_key in enumerate(doc_keys):
        doc_to_indices.setdefault(doc_key, []).append(i)
    max_sims = [0.0] * len(ranked)
    selected = []
    remaining = list(range(len(ranked)))
    in_remaining = [True] * len(ranked)

    for _ in range(min(top_k, len(ranked))):
        best_idx = -1
        best_mmr = -float("inf")
        best_rem_idx = -1
        for rem_idx, idx in enumerate(remaining):
            mmr_score = relevance_terms[idx] - penalty_multiplier * max_sims[idx]
            if mmr_score > best_mmr or (mmr_score == best_mmr and idx < best_idx):
                best_mmr = mmr_score
                best_idx = idx
                best_rem_idx = rem_idx
        if best_idx < 0: break
        selected.append(ranked[best_idx])
        remaining[best_rem_idx] = remaining[-1]
        remaining.pop()
        in_remaining[best_idx] = False

        new_doc_key = doc_keys[best_idx]
        new_emb = chunk_embs[best_idx]
        new_norm = chunk_norms[best_idx]
        if new_emb is not None:
            for idx in doc_to_indices[new_doc_key]:
                if in_remaining[idx]:
                    idx_emb = chunk_embs[idx]
                    if idx_emb is not None:
                        sim = _cosine_sim(idx_emb, new_emb, chunk_norms[idx], new_norm)
                        if sim > max_sims[idx]: max_sims[idx] = sim
    return selected

def run_lazy(ranked, embeddings, top_k, mmr_lambda=0.5):
    scores = [float(r["rrf"]) for r in ranked]
    s_min, s_max = min(scores), max(scores)
    s_range = s_max - s_min if s_max > s_min else 1.0
    norm_scores = [(s - s_min) / s_range for s in scores]
    relevance_terms = [mmr_lambda * ns for ns in norm_scores]
    penalty_multiplier = 1.0 - mmr_lambda
    doc_keys = [str(r["path"]) for r in ranked]
    chunk_embs = [embeddings.get(int(r["id"])) for r in ranked]

    # LAZY INITIALIZATION
    chunk_norms = [None] * len(ranked)

    doc_to_indices = {}
    for i, doc_key in enumerate(doc_keys):
        doc_to_indices.setdefault(doc_key, []).append(i)
    max_sims = [0.0] * len(ranked)
    selected = []
    remaining = list(range(len(ranked)))
    in_remaining = [True] * len(ranked)

    for _ in range(min(top_k, len(ranked))):
        best_idx = -1
        best_mmr = -float("inf")
        best_rem_idx = -1
        for rem_idx, idx in enumerate(remaining):
            mmr_score = relevance_terms[idx] - penalty_multiplier * max_sims[idx]
            if mmr_score > best_mmr or (mmr_score == best_mmr and idx < best_idx):
                best_mmr = mmr_score
                best_idx = idx
                best_rem_idx = rem_idx
        if best_idx < 0: break
        selected.append(ranked[best_idx])
        remaining[best_rem_idx] = remaining[-1]
        remaining.pop()
        in_remaining[best_idx] = False

        new_doc_key = doc_keys[best_idx]
        siblings = doc_to_indices[new_doc_key]

        # BYPASS PENALTY UPDATES IF NO SIBLINGS
        if len(siblings) <= 1:
            continue

        new_emb = chunk_embs[best_idx]
        if new_emb is not None:
            if chunk_norms[best_idx] is None:
                chunk_norms[best_idx] = math.hypot(*new_emb)
            new_norm = chunk_norms[best_idx]

            for idx in siblings:
                if in_remaining[idx]:
                    idx_emb = chunk_embs[idx]
                    if idx_emb is not None:
                        if chunk_norms[idx] is None:
                            chunk_norms[idx] = math.hypot(*idx_emb)
                        sim = _cosine_sim(idx_emb, new_emb, chunk_norms[idx], new_norm)
                        if sim > max_sims[idx]: max_sims[idx] = sim
    return selected

# Generate dummy data
N = 1000
D = 384
K = 20
ranked = []
embeddings = {}
for i in range(N):
    doc = f"doc_{i % (N//2)}"  # 2 chunks per doc average
    ranked.append({"id": i, "path": doc, "rrf": random.random()})
    embeddings[i] = [random.random() for _ in range(D)]

t0 = time.time()
for _ in range(50):
    run_eager(ranked, embeddings, K)
t_eager = time.time() - t0

t0 = time.time()
for _ in range(50):
    run_lazy(ranked, embeddings, K)
t_lazy = time.time() - t0

print(f"Eager: {t_eager:.4f}s")
print(f"Lazy:  {t_lazy:.4f}s")
print(f"Speedup: {t_eager/t_lazy:.2f}x")
