"""Probe: MMR dedup loop shape (supersedes #351).

Goal: measure the rewrite of ``VstashStore._mmr_dedup`` from
``remaining.remove(best_idx)`` + ``for idx in remaining: if doc_keys[idx]
== new_doc_key`` (O(K * N) penalty loop per outer iteration) to
swap-with-last + pre-grouped ``doc_to_indices`` (O(K * S_avg)) and
verify the two implementations select identical candidates.

Both implementations are reproduced inline as pure functions so the
probe is self-contained and the witness numbers can be regenerated on
any commit. The reference (``_mmr_dedup_naive``) mirrors the
implementation present in ``vstash/store.py`` immediately prior to this
PR. The optimized version (``_mmr_dedup_o1``) mirrors the new
implementation. Inputs and outputs are identical for any fixed seed,
which the script asserts before timing -- if the two diverge, the
rewrite is not a pure perf change and the comparison is invalid.

Empirical finding: this synthetic probe shows essentially **no
speedup** (0.95x - 1.02x) across N from 500 to 50000, top_k from 50 to
1000, siblings from 5 to 20, dim 32/384. The pre-grouping save in the
penalty loop is eaten by CPython overhead (``enumerate`` tuple
allocation, Python-level swap-pop vs C-level ``list.remove``) and the
fact that the selection scan (also O(K * N), unchanged by this
rewrite) dominates total cost at these sizes. The optimization only
materialises end-to-end in ``experiments.perf_mmr_dedup_real`` once
real BGE embeddings and the SQL vec fetch are in the mix, where it
shows a steady ~1.15x - 1.19x on ``store.search``.

The takeaway is methodological: a pure-Python micro-benchmark of an
algorithm whose hot path is dominated by surrounding pipeline cost
will under-report the gain. Both probes are kept (this one and
``_real``) so future perf claims for ``_mmr_dedup`` have to defend
themselves against both.

Run: ``python -m experiments.perf_mmr_dedup [--n-sweep 200,1000,5000]``
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time

DEFAULT_N_SWEEP = (200, 1000, 5000, 10000)
DEFAULT_SIBLINGS = 10
DEFAULT_TOP_K = 50
DEFAULT_ROUNDS = 5


def _cosine_sim(a: list[float], b: list[float], norm_a: float, norm_b: float) -> float:
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (norm_a * norm_b)


def _mmr_dedup_naive(
    ranked: list[dict],
    embeddings: dict[int, list[float]],
    top_k: int,
    mmr_lambda: float,
) -> list[dict]:
    """Reference implementation (pre-rewrite). O(K * N) penalty loop."""
    if not ranked:
        return []
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
        selected.append(ranked[best_idx])
        remaining.remove(best_idx)  # O(N)
        new_doc_key = doc_keys[best_idx]
        new_emb = chunk_embs[best_idx]
        new_norm = chunk_norms[best_idx]
        if new_emb is not None:
            for idx in remaining:  # O(N) scan
                if doc_keys[idx] == new_doc_key:
                    idx_emb = chunk_embs[idx]
                    if idx_emb is not None:
                        sim = _cosine_sim(
                            idx_emb, new_emb, norm_a=chunk_norms[idx], norm_b=new_norm
                        )
                        if sim > max_sims[idx]:
                            max_sims[idx] = sim
    return selected


def _mmr_dedup_o1(
    ranked: list[dict],
    embeddings: dict[int, list[float]],
    top_k: int,
    mmr_lambda: float,
) -> list[dict]:
    """Optimized implementation (post-rewrite). O(K * S_avg) penalty loop."""
    if not ranked:
        return []
    scores = [float(r["rrf"]) for r in ranked]
    s_min, s_max = min(scores), max(scores)
    s_range = s_max - s_min if s_max > s_min else 1.0
    norm_scores = [(s - s_min) / s_range for s in scores]
    relevance_terms = [mmr_lambda * ns for ns in norm_scores]
    penalty_multiplier = 1.0 - mmr_lambda
    doc_keys = [str(r["path"]) for r in ranked]
    chunk_embs = [embeddings.get(int(r["id"])) for r in ranked]
    doc_to_indices: dict[str, list[int]] = {}
    for i, doc_key in enumerate(doc_keys):
        doc_to_indices.setdefault(doc_key, []).append(i)

    chunk_norms = [None] * len(ranked)
    max_sims = [0.0] * len(ranked)
    selected: list[dict] = []
    remaining = list(range(len(ranked)))
    in_remaining = [True] * len(ranked)

    for _ in range(min(top_k, len(ranked))):
        best_idx = -1
        best_mmr = -float("inf")
        best_rem_idx = -1
        for rem_idx, idx in enumerate(remaining):
            mmr_score = relevance_terms[idx] - penalty_multiplier * max_sims[idx]
            # Tie-break on smaller original ``idx`` to match the naive
            # (pre-rewrite) ordering, which always iterated a sorted
            # ``remaining`` and let strict ``>`` keep the first-seen.
            if mmr_score > best_mmr or (mmr_score == best_mmr and idx < best_idx):
                best_mmr = mmr_score
                best_idx = idx
                best_rem_idx = rem_idx
        if best_idx < 0 or best_mmr < 0:
            break
        selected.append(ranked[best_idx])
        # O(1) swap-with-last
        remaining[best_rem_idx] = remaining[-1]
        remaining.pop()
        in_remaining[best_idx] = False
        new_doc_key = doc_keys[best_idx]

        doc_indices = doc_to_indices[new_doc_key]
        if len(doc_indices) > 1:
            new_emb = chunk_embs[best_idx]
            if new_emb is not None:
                new_norm = chunk_norms[best_idx]
                if new_norm is None:
                    new_norm = math.hypot(*new_emb)
                    chunk_norms[best_idx] = new_norm
                for idx in doc_indices:  # O(S)
                    if in_remaining[idx]:
                        idx_emb = chunk_embs[idx]
                        if idx_emb is not None:
                            idx_norm = chunk_norms[idx]
                            if idx_norm is None:
                                idx_norm = math.hypot(*idx_emb)
                                chunk_norms[idx] = idx_norm
                            sim = _cosine_sim(idx_emb, new_emb, norm_a=idx_norm, norm_b=new_norm)
                            if sim > max_sims[idx]:
                                max_sims[idx] = sim
    return selected


def _make_fixture(
    n: int, siblings_per_doc: int, dim: int, seed: int
) -> tuple[list[dict], dict[int, list[float]]]:
    """Build a deterministic ranked list with same-doc duplicates."""
    rng = random.Random(seed)
    ranked: list[dict] = []
    embeddings: dict[int, list[float]] = {}
    for chunk_id in range(n):
        doc_id = chunk_id // siblings_per_doc
        # Embedding: fully random unit-ish vectors so within-doc cosines
        # are moderate (not ~1.0), which keeps MMR scores positive across
        # the full ``top_k`` loop rather than breaking early after the
        # first sibling triggers a near-1.0 penalty.
        emb = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        ranked.append(
            {
                "id": chunk_id,
                "path": f"/probe/doc_{doc_id}.md",
                # Random RRF so MMR's relevance term varies meaningfully
                # across candidates, instead of decaying monotonically.
                "rrf": rng.uniform(0.001, 0.033),
            }
        )
        embeddings[chunk_id] = emb
    return ranked, embeddings


def _equiv(a: list[dict], b: list[dict]) -> bool:
    return [r["id"] for r in a] == [r["id"] for r in b]


def _time(fn, ranked, embeddings, top_k, mmr_lambda, rounds: int) -> float:
    """Median wall time over ``rounds`` invocations, in seconds."""
    latencies = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn(ranked, embeddings, top_k, mmr_lambda)
        latencies.append(time.perf_counter() - t0)
    return statistics.median(latencies)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--n-sweep",
        default=",".join(str(n) for n in DEFAULT_N_SWEEP),
        help="comma-separated corpus sizes (number of ranked chunks)",
    )
    p.add_argument("--siblings", type=int, default=DEFAULT_SIBLINGS)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--mmr-lambda", type=float, default=0.5)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Fail fast on invalid CLI input so the script does not crash deep
    # in fixture construction (``siblings_per_doc`` is the divisor for
    # ``doc_id``; ``rounds`` is the floor for ``statistics.median``).
    if args.siblings < 1 or args.top_k < 1 or args.rounds < 1 or args.dim < 1:
        raise SystemExit("--siblings, --top-k, --rounds, and --dim must all be >= 1")
    sizes = [int(s) for s in args.n_sweep.split(",") if s.strip()]
    if not sizes or any(n < 1 for n in sizes):
        raise SystemExit("--n-sweep must contain one or more positive integers")

    print(
        f"# perf_mmr_dedup -- siblings={args.siblings} top_k={args.top_k} "
        f"mmr_lambda={args.mmr_lambda} dim={args.dim} rounds={args.rounds}\n"
    )
    print(f"{'N':>8} {'naive (s)':>12} {'optimized (s)':>14} {'speedup':>10} {'equiv?':>7}")
    print(f"{'-' * 8} {'-' * 12} {'-' * 14} {'-' * 10} {'-' * 7}")
    for n in sizes:
        ranked, embeddings = _make_fixture(n, args.siblings, args.dim, args.seed)
        # Correctness gate: the two implementations must select the same
        # chunk_ids in the same order, otherwise the timing comparison is
        # meaningless.
        out_naive = _mmr_dedup_naive(ranked, embeddings, args.top_k, args.mmr_lambda)
        out_o1 = _mmr_dedup_o1(ranked, embeddings, args.top_k, args.mmr_lambda)
        equiv = _equiv(out_naive, out_o1)
        if not equiv:
            print(
                f"{n:>8} -- ABORT: outputs differ between naive and optimized "
                "implementations; the rewrite is not a pure perf change."
            )
            raise SystemExit(2)

        t_naive = _time(
            _mmr_dedup_naive, ranked, embeddings, args.top_k, args.mmr_lambda, args.rounds
        )
        t_o1 = _time(_mmr_dedup_o1, ranked, embeddings, args.top_k, args.mmr_lambda, args.rounds)
        speedup = t_naive / t_o1 if t_o1 > 0 else float("inf")
        print(
            f"{n:>8} {t_naive:>12.4f} {t_o1:>14.4f} {speedup:>9.2f}x {'yes' if equiv else 'NO':>7}"
        )


if __name__ == "__main__":
    main()
