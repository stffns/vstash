"""ColBERTv2 head-to-head retrieval on LongMemEval-s.

Mirrors `experiments/longmemeval_retrieval.py` exactly so the resulting
JSONs are directly comparable, but swaps the retrieval engine: vstash
(single-vector + FTS5 + adaptive RRF + MMR) becomes ColBERTv2
(multi-vector MaxSim).  Same per-question fresh-index pattern, same
chunking (vstash's default 1024/128), same session-level Recall@K
metric against `answer_session_ids`, same `--stratify` and `--n` knobs.

The comparison is "iguales condiciones": both systems get the same
haystack, the same chunks, and aggregate to unique sessions in
top-K.  Truncation note: ColBERTv2 caps documents at ~220 tokens
internally; vstash's BGE-small caps at 512.  Long chunks are
slightly more penalised on the ColBERT side -- documented in the
result JSON's `caveats` block.

Usage:
    python -m experiments.longmemeval_colbert --all \
        --output experiments/results/lme_full_500_colbertv2.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from experiments.longmemeval_retrieval import (
    DATA_PATH,
    DEFAULT_KS,
    _format_session,
    _recall_at_k,
    _stratified_subset,
)
from vstash.config import VstashConfig
from vstash.ingest import chunk_text


def _load_pylate(model_name: str, device: str):
    """Lazy import + load pylate ColBERT.

    pylate is an optional dep installed only on Colab; we fail with a
    clear hint so a user invoking this locally without it knows what
    to do instead of staring at a generic ImportError.
    """
    try:
        from pylate import models  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "pylate is required for the ColBERT h2h.  Install with:\n"
            "    pip install pylate\n"
            "On Colab T4 just include it in the setup cell of the "
            "notebook -- see experiments/retrain_longmemeval_colab.ipynb "
            "for the pattern."
        ) from exc
    return models.ColBERT(model_name_or_path=model_name, device=device)


def _load_minimal(model_name: str, device: str):
    """Load the raw-HF MinimalColBERT fallback.

    Used when ``--engine minimal`` is passed (Colab pylate dependency
    storms).  Wraps the colbert_minimal API so callers can stay
    engine-agnostic.
    """
    from experiments.colbert_minimal import MinimalColBERT

    impl = MinimalColBERT(model_name=model_name, device=device)

    class _MinimalEngine:
        engine_name = "minimal"

        def encode_query(self, text: str):
            return impl.encode_queries([text])[0]

        def encode_docs(self, texts, batch_size: int = 32):
            doc_embs, _ = impl.encode_documents(texts, batch_size=batch_size)
            return doc_embs

        def score(self, query_emb, doc_embs):
            return impl.maxsim_scores(query_emb, doc_embs)

    return _MinimalEngine()


def _load_pylate_engine(model_name: str, device: str):
    """Wrap pylate ColBERT in the same engine-agnostic surface."""
    pylate_model = _load_pylate(model_name, device)

    class _PylateEngine:
        engine_name = "pylate"

        def encode_query(self, text: str):
            from pylate import models  # noqa: F401  (kept for clarity)

            embs = pylate_model.encode([text], is_query=True, show_progress_bar=False)
            return embs[0] if hasattr(embs, "__getitem__") else embs

        def encode_docs(self, texts, batch_size: int = 32):
            return pylate_model.encode(
                texts,
                is_query=False,
                batch_size=batch_size,
                show_progress_bar=False,
            )

        def score(self, query_emb, doc_embs):
            from pylate import scores as colbert_scores  # type: ignore

            sim = colbert_scores.colbert_scores(
                query_emb.unsqueeze(0) if query_emb.dim() == 2 else query_emb,
                doc_embs,
            )
            return sim[0] if sim.dim() == 2 else sim

    return _PylateEngine()


def _load_engine(engine: str, model_name: str, device: str):
    if engine == "pylate":
        return _load_pylate_engine(model_name, device)
    if engine == "minimal":
        return _load_minimal(model_name, device)
    raise SystemExit(f"Unknown --engine: {engine!r}; expected 'pylate' or 'minimal'.")


def _encode_haystack(
    engine,
    cfg: VstashConfig,
    session_ids: list[str],
    sessions: list[list[dict]],
    encode_batch_size: int,
):
    """Chunk + encode every session of one question's haystack.

    Returns parallel lists ``(chunk_session_ids, chunk_embeddings)`` so
    we can map MaxSim scores back to source sessions afterwards.

    First-wins-dedupes duplicate session_ids inside one haystack (15
    questions in longmemeval_s ship duplicates) so the resulting
    chunk_session_ids has every distinct session contributing the
    chunks it actually owns -- avoiding cross-document leakage from a
    later occurrence overwriting an earlier one.
    """
    seen: set[str] = set()
    flat_chunks: list[str] = []
    flat_session_ids: list[str] = []
    for sid, turns in zip(session_ids, sessions):
        if sid in seen:
            continue
        seen.add(sid)
        text = _format_session(turns)
        if not text.strip():
            continue
        chunks = chunk_text(text, cfg.chunking.size, cfg.chunking.overlap)
        if not chunks:
            continue
        flat_chunks.extend(chunks)
        flat_session_ids.extend([sid] * len(chunks))

    if not flat_chunks:
        return flat_session_ids, None

    embeddings = engine.encode_docs(flat_chunks, batch_size=encode_batch_size)
    return flat_session_ids, embeddings


def _retrieve_top_chunks(
    engine,
    query_text: str,
    doc_embeddings,
    top_k_chunks: int,
):
    """Return the indices of the top-k chunks by ColBERT MaxSim."""
    import numpy as np

    q_emb = engine.encode_query(query_text)
    sim_row = engine.score(q_emb, doc_embeddings)
    if hasattr(sim_row, "cpu"):
        sim_row = sim_row.cpu().numpy()
    n = len(sim_row)
    k = min(top_k_chunks, n)
    # argpartition + sort the top-k slice; cheaper than full argsort
    # when n is large (a long haystack hits ~hundreds of chunks).
    if k >= n:
        idx = np.argsort(-sim_row)
    else:
        part = np.argpartition(-sim_row, k - 1)[:k]
        idx = part[np.argsort(-sim_row[part])]
    return idx.tolist()


def _unique_sessions_in_order(ranked_session_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for sid in ranked_session_ids:
        if sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def run(
    n: int | None,
    top_k_chunks: int,
    ks: tuple[int, ...],
    output: Path | None,
    seed_offset: int,
    stratify_per_type: int | None,
    model_name: str,
    device: str,
    encode_batch_size: int,
    engine: str = "pylate",
) -> dict:
    cfg = VstashConfig()
    print(f"Loading {model_name} on {device} (engine={engine})...", flush=True)
    model = _load_engine(engine, model_name, device)
    print("Loaded.", flush=True)

    data = json.loads(DATA_PATH.read_text())
    if stratify_per_type is not None:
        data = _stratified_subset(data, stratify_per_type)
    elif n is not None:
        data = data[seed_offset : seed_offset + n]

    per_question: list[dict] = []
    by_type: dict[str, list[dict]] = defaultdict(list)
    overall_start = time.time()

    for i, q in enumerate(data):
        qid = q["question_id"]
        qtype = q["question_type"]
        question = q["question"]
        gold = set(q["answer_session_ids"])
        session_ids = q["haystack_session_ids"]
        sessions = q["haystack_sessions"]

        t0 = time.time()
        chunk_sids, doc_emb = _encode_haystack(model, cfg, session_ids, sessions, encode_batch_size)
        encode_s = time.time() - t0
        if doc_emb is None:
            per_question.append(
                {
                    "question_id": qid,
                    "question_type": qtype,
                    "n_sessions": len(session_ids),
                    "n_chunks_retrieved": 0,
                    "n_unique_sessions_retrieved": 0,
                    "gold_session_ids": sorted(gold),
                    "first_gold_rank": None,
                    "encode_s": round(encode_s, 2),
                    "search_s": 0.0,
                    **{f"recall@{k}": float("nan") for k in ks},
                }
            )
            continue

        t0 = time.time()
        idxs = _retrieve_top_chunks(model, question, doc_emb, top_k_chunks)
        ranked_sids = [chunk_sids[j] for j in idxs]
        retrieved_sessions = _unique_sessions_in_order(ranked_sids)
        search_s = time.time() - t0

        recalls = {f"recall@{k}": _recall_at_k(retrieved_sessions, gold, k) for k in ks}
        gold_rank = next(
            (idx + 1 for idx, sid in enumerate(retrieved_sessions) if sid in gold),
            None,
        )
        row = {
            "question_id": qid,
            "question_type": qtype,
            "n_sessions": len(session_ids),
            "n_chunks_retrieved": len(idxs),
            "n_unique_sessions_retrieved": len(retrieved_sessions),
            "gold_session_ids": sorted(gold),
            "first_gold_rank": gold_rank,
            "encode_s": round(encode_s, 2),
            "search_s": round(search_s, 3),
            **recalls,
        }
        per_question.append(row)
        by_type[qtype].append(row)
        elapsed = time.time() - overall_start
        print(
            f"[{i + 1}/{len(data)}] {qtype:25s} R@5={recalls['recall@5']:.2f} "
            f"R@10={recalls['recall@10']:.2f} encode={encode_s:.1f}s "
            f"first_gold={gold_rank} elapsed={elapsed:.0f}s",
            flush=True,
        )

    def _agg(rows: list[dict], k: int) -> float:
        vals = [r[f"recall@{k}"] for r in rows if r[f"recall@{k}"] == r[f"recall@{k}"]]
        return statistics.mean(vals) if vals else float("nan")

    summary = {
        "n_questions": len(per_question),
        "top_k_chunks": top_k_chunks,
        "retrieval_engine": "colbertv2",
        "embedding_model": model_name,
        "device": device,
        "chunking": {"size": cfg.chunking.size, "overlap": cfg.chunking.overlap},
        "macro": {f"recall@{k}": _agg(per_question, k) for k in ks},
        "by_question_type": {
            t: {"n": len(rows), **{f"recall@{k}": _agg(rows, k) for k in ks}}
            for t, rows in by_type.items()
        },
        "total_elapsed_s": round(time.time() - overall_start, 1),
        "caveats": (
            "ColBERTv2 truncates documents at ~220 tokens internally; "
            "vstash's BGE-small caps at 512.  Identical chunking is "
            "applied on both sides (vstash chunk_text 1024 char / 128 "
            "overlap), but long chunks are slightly more penalised on "
            "the ColBERT side."
        ),
    }

    print()
    print("=" * 72)
    print(f"Macro over {summary['n_questions']} questions  (model={model_name}, device={device}):")
    for k in ks:
        print(f"  Recall@{k:<3d} = {summary['macro'][f'recall@{k}']:.4f}")
    print()
    print("By question_type:")
    for t, agg in summary["by_question_type"].items():
        line = f"  {t:30s} n={agg['n']:3d}  "
        line += "  ".join(f"R@{k}={agg[f'recall@{k}']:.3f}" for k in ks)
        print(line)
    print(f"Total wall time: {summary['total_elapsed_s']}s")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"summary": summary, "per_question": per_question}, indent=2))
        print(f"\nWrote {output}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--all", action="store_true")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--stratify", type=int, default=None)
    p.add_argument("--top-k-chunks", type=int, default=200)
    p.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--model", type=str, default="colbert-ir/colbertv2.0")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--encode-batch-size", type=int, default=32)
    p.add_argument(
        "--engine",
        choices=["pylate", "minimal"],
        default="pylate",
        help="ColBERT inference backend.  'pylate' is the default (full-featured, "
        "depends on sentence-transformers).  'minimal' is the raw-HF fallback "
        "(experiments/colbert_minimal.py) -- no pylate, no sentence-transformers, "
        "needs only torch + transformers.  Use it on Colab when the pylate / "
        "sentence-transformers / torchcodec dependency cascade gets unstable.",
    )
    args = p.parse_args()
    n = None if args.all else args.n
    run(
        n=n,
        top_k_chunks=args.top_k_chunks,
        ks=tuple(args.ks),
        output=args.output,
        seed_offset=args.offset,
        stratify_per_type=args.stratify,
        engine=args.engine,
        model_name=args.model,
        device=args.device,
        encode_batch_size=args.encode_batch_size,
    )


if __name__ == "__main__":
    main()
