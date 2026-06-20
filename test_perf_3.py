import time
import random
import math

def test_norm_precalc():
    chunk_embs = [[random.random() for _ in range(384)] for _ in range(1000)]
    start = time.perf_counter()
    chunk_norms = [math.hypot(*emb) if emb is not None else 0.0 for emb in chunk_embs]
    end = time.perf_counter()
    return end - start

def test_norm_lazy():
    chunk_embs = [[random.random() for _ in range(384)] for _ in range(1000)]
    start = time.perf_counter()
    chunk_norms = [None] * len(chunk_embs)
    for i, emb in enumerate(chunk_embs):
        if chunk_norms[i] is None:
            chunk_norms[i] = math.hypot(*emb) if emb is not None else 0.0
    end = time.perf_counter()
    return end - start

print(f"precalc: {test_norm_precalc():.5f}")
print(f"lazy: {test_norm_lazy():.5f}")
