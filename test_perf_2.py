import time
import random
import math

def test_norm():
    chunk_embs = [[random.random() for _ in range(384)] for _ in range(1000)]
    start = time.perf_counter()
    chunk_norms = [math.hypot(*emb) if emb is not None else 0.0 for emb in chunk_embs]
    end = time.perf_counter()
    return end - start

print(f"norm time: {test_norm():.5f}")
