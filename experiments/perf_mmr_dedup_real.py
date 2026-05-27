"""End-to-end MMR dedup benchmark with real embeddings and real search.

Companion to ``experiments/perf_mmr_dedup.py``: where that script
isolates the _mmr_dedup function in pure Python, this one drives the
real ``store.search()`` pipeline (vector + FTS5 + RRF + MMR + context
expansion) so the optimization is measured under the same conditions
vstash actually runs in production. If the synthetic and the real
benchmark disagree, the real one wins -- this is the configuration
users see.

Methodology:

1. Build a tmp vstash store of N docs, each with M chunks. Embeddings
   come from BGE-small-en-v1.5 over real text (vstash's default model).
2. Run Q queries through ``store.search(top_k=K)`` end-to-end.
3. Monkey-patch ``VstashStore._mmr_dedup`` with the two implementations
   ("naive" mirrors pre-rewrite, "optimized" mirrors the swap-pop +
   pre-grouped doc_to_indices rewrite proposed by PR #351).
4. Time each over R rounds, report median.

The MMR dedup runs on the RRF-fused candidate pool, which by default
is ``min(top_k * 10, max(top_k * 3, total_chunks // 3))`` -- so the
relevant ``len(ranked)`` is at most ~100 for top_k=10 and a corpus of
a few hundred docs. If the optimization does not register at that
scale, it never will under default usage.

Run: ``python -m experiments.perf_mmr_dedup_real [--docs 200] [--chunks 5]``
"""

from __future__ import annotations

import argparse
import math
import statistics
import tempfile
import time

from vstash.embed import embed_texts, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"


# ----- the two implementations under test -------------------------------- #


def _cosine_sim_local(a: list[float], b: list[float], norm_a: float, norm_b: float) -> float:
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def _make_naive_dedup(original):
    """Wrap the real method body with the pre-rewrite penalty-loop shape."""

    def _dedup(self, ranked, top_k, mmr_lambda, _explain=False):
        # Reuse the fast path from the real implementation -- only the
        # greedy loop differs between naive and optimized.
        if not ranked:
            return []
        from collections import Counter

        doc_counts = Counter(str(r["path"]) for r in ranked)
        if mmr_lambda >= 1.0 or not any(c > 1 for c in doc_counts.values()):
            return original(self, ranked, top_k, mmr_lambda, _explain=_explain)
        # Fetch embeddings via the real path so we exercise the same
        # SQLite work both implementations would do.
        dup_doc_paths = {p for p, c in doc_counts.items() if c > 1}
        dup_ids = [int(r["id"]) for r in ranked if str(r["path"]) in dup_doc_paths]
        embeddings: dict[int, list[float]] = {}
        if dup_ids:
            from vstash.store import _deserialize

            placeholders = ",".join("?" * len(dup_ids))
            rows = self._conn.execute(
                f"SELECT rowid, embedding FROM vec_chunks WHERE rowid IN ({placeholders})",
                dup_ids,
            ).fetchall()
            for row in rows:
                embeddings[row["rowid"]] = _deserialize(row["embedding"])

        scores = [float(r["rrf"]) for r in ranked]
        s_min, s_max = min(scores), max(scores)
        s_range = s_max - s_min if s_max > s_min else 1.0
        norm_scores = [(s - s_min) / s_range for s in scores]
        relevance_terms = [mmr_lambda * ns for ns in norm_scores]
        penalty_multiplier = 1.0 - mmr_lambda
        doc_keys = [str(r["path"]) for r in ranked]
        chunk_embs = [embeddings.get(int(r["id"])) for r in ranked]
        chunk_norms = [math.hypot(*emb) if emb is not None else 0.0 for emb in chunk_embs]
        max_sims = [0.0] * len(ranked)
        selected: list[dict] = []
        remaining = list(range(len(ranked)))
        for _ in range(min(top_k, len(ranked))):
            best_idx = -1
            best_mmr = -float("inf")
            for idx in remaining:
                mmr_score = relevance_terms[idx] - penalty_multiplier * max_sims[idx]
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
            if best_idx < 0 or best_mmr < 0:
                break
            chosen = ranked[best_idx]
            if _explain:
                chosen["_mmr_penalty"] = (1 - mmr_lambda) * max_sims[best_idx]
            selected.append(chosen)
            remaining.remove(best_idx)
            new_doc_key = doc_keys[best_idx]
            new_emb = chunk_embs[best_idx]
            new_norm = chunk_norms[best_idx]
            if new_emb is not None:
                for idx in remaining:
                    if doc_keys[idx] == new_doc_key:
                        idx_emb = chunk_embs[idx]
                        if idx_emb is not None:
                            sim = _cosine_sim_local(
                                idx_emb, new_emb, norm_a=chunk_norms[idx], norm_b=new_norm
                            )
                            if sim > max_sims[idx]:
                                max_sims[idx] = sim
        return selected

    return _dedup


# ----- corpus generation ------------------------------------------------- #


_TOPICS = [
    "machine learning",
    "databases",
    "networking",
    "kubernetes",
    "rust",
    "python async",
    "graph algorithms",
    "compilers",
    "linear algebra",
    "transformer attention",
    "vector search",
    "sqlite",
    "redis caching",
    "kernel scheduling",
    "memory allocators",
]


def _build_corpus(docs: int, chunks_per_doc: int) -> list[dict]:
    """Realistic-ish corpus where each doc focuses on one topic but its
    chunks paraphrase the same idea -- which is the case MMR dedup is
    designed for (near-duplicate intra-document chunks)."""
    corpus = []
    for d in range(docs):
        topic = _TOPICS[d % len(_TOPICS)]
        chunks = [
            f"{topic} chunk {c}: this passage discusses {topic} from angle {c} "
            f"with examples and pseudocode. doc {d}."
            for c in range(chunks_per_doc)
        ]
        corpus.append(
            {
                "path": f"/probe/doc_{d}.md",
                "title": f"Doc {d}: {topic}",
                "chunks": chunks,
            }
        )
    return corpus


def _queries() -> list[str]:
    return [
        "how does machine learning work",
        "what is a database index",
        "redis caching strategies",
        "python asyncio event loop",
        "compilers and code generation",
        "kubernetes pod scheduling",
        "vector search at scale",
        "rust borrow checker",
        "transformer attention mechanism",
        "linear algebra for ML",
    ]


# ----- benchmark driver -------------------------------------------------- #


def _populate_store(store: VstashStore, corpus: list[dict]) -> None:
    all_chunks: list[str] = []
    doc_slices: list[tuple[int, int]] = []
    cursor = 0
    for doc in corpus:
        n = len(doc["chunks"])
        doc_slices.append((cursor, cursor + n))
        all_chunks.extend(doc["chunks"])
        cursor += n
    print(f"  embedding {len(all_chunks)} chunks with {MODEL} ...", flush=True)
    embeddings = embed_texts(all_chunks, model_name=MODEL)
    for doc, (start, end) in zip(corpus, doc_slices):
        store.add_document(
            path=doc["path"],
            title=doc["title"],
            chunks=doc["chunks"],
            embeddings=embeddings[start:end],
        )


def _time_searches(store: VstashStore, queries: list[str], top_k: int, rounds: int) -> float:
    """Median total wall time (s) to run all queries once, over rounds."""
    # Build query embeddings ONCE so embedding cost doesn't pollute the
    # search-side timing.
    q_embs = embed_texts(queries, model_name=MODEL)
    latencies = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for q_text, q_emb in zip(queries, q_embs):
            store.search(q_emb, q_text, top_k=top_k)
        latencies.append(time.perf_counter() - t0)
    return statistics.median(latencies)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=int, default=200)
    p.add_argument("--chunks", type=int, default=5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--rounds", type=int, default=5)
    args = p.parse_args()

    print(
        f"# perf_mmr_dedup_real -- docs={args.docs} chunks_per_doc={args.chunks} "
        f"top_k={args.top_k} rounds={args.rounds} model={MODEL}\n"
    )

    corpus = _build_corpus(args.docs, args.chunks)
    queries = _queries()
    dim = get_embedding_dim(MODEL)

    with tempfile.TemporaryDirectory() as tmp:
        store = VstashStore(f"{tmp}/probe.db", embedding_dim=dim)
        with store:
            _populate_store(store, corpus)

            # --- baseline: real (optimized) implementation as currently in store.py
            print("  timing optimized (current store.py implementation) ...", flush=True)
            t_opt = _time_searches(store, queries, args.top_k, args.rounds)

            # --- swap in the naive implementation, time again
            original = VstashStore._mmr_dedup
            VstashStore._mmr_dedup = _make_naive_dedup(original)
            try:
                print("  timing naive (pre-rewrite implementation) ...", flush=True)
                t_naive = _time_searches(store, queries, args.top_k, args.rounds)
            finally:
                VstashStore._mmr_dedup = original

    speedup = t_naive / t_opt if t_opt > 0 else float("inf")
    print()
    print(f"{'implementation':>18} {'total search time (s)':>25}")
    print(f"{'-' * 18} {'-' * 25}")
    print(f"{'naive (pre-#351)':>18} {t_naive:>25.4f}")
    print(f"{'optimized (#351)':>18} {t_opt:>25.4f}")
    print(f"{'speedup':>18} {speedup:>24.2f}x")


if __name__ == "__main__":
    main()
