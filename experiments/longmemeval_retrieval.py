"""LongMemEval retrieval-only benchmark for vstash.

Per question, ingest the haystack sessions (one document per session) into a
fresh per-question vstash store, run a single search with the question text,
and measure session-level Recall@K against `answer_session_ids`.

No LLM, no API calls. Just retrieval quality on multi-session chat haystacks
that average ~115k tokens per question (the longmemeval_s variant).

Usage:
    python -m experiments.longmemeval_retrieval --n 10
    python -m experiments.longmemeval_retrieval --n 500 --output results/lme_full.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from vstash.config import VstashConfig
from vstash.embed import (
    embed_query,
    embed_texts,
    get_embedding_dim,
    register_encoder_resolver,
    unregister_encoder_resolver,
)
from vstash.ingest import chunk_text
from vstash.store import VstashStore

DATA_PATH = Path(__file__).parent / "data" / "longmemeval" / "longmemeval_s"
DEFAULT_KS = (1, 3, 5, 10, 20, 50)

# Models we want to load via SentenceTransformer (safetensors) instead of the
# MLX or FastEmbed paths, because their HF repo either lacks an MLX-compatible
# layout (Stffens/bge-small-rrf-v3) or ships an ONNX stub without external
# data.  The fix in vstash core is to extend _HF_ONNX_MODELS, but we use the
# v0.34 register_encoder_resolver hook here so the experiment is self-contained.
_ST_FALLBACK_MODELS = {"Stffens/bge-small-rrf-v3"}


class _STEncoder:
    """Minimal SentenceTransformer adapter to the Encoder protocol.

    ``device`` is honoured so callers on Colab T4 can pin the model to
    ``cuda``; FastEmbed has no CUDA wheel and pegs the Colab CPU at
    ~1 ingest per minute on LongMemEval-scale haystacks (~50x slower
    than even Apple-Silicon CPU).
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
        self.embedding_dim = self._m.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]):
        return self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False)


_st_cache: dict[str, _STEncoder] = {}
# Process-global override for the device every _STEncoder picks up.
# Set by run() when --device cuda is on; falls back to ST default
# (auto-detect) when None.
_st_device_override: str | None = None
# When True the resolver claims every model_name -- used to route the
# default cfg.embeddings.model through ST on CUDA instead of FastEmbed
# on CPU.
_st_force_for_all = False


def _is_local_st_model(model_name: str) -> bool:
    """Detect a local SentenceTransformer model directory.

    A local path is anything that resolves to an existing directory and
    has the SentenceTransformer markers we expect (``modules.json`` is
    the cheapest discriminator -- HF model names never contain it as
    a path component because they go through the HF cache, not raw
    filesystem).
    """
    p = Path(model_name).expanduser()
    return p.is_dir() and (p / "modules.json").is_file()


def _st_resolver(model_name: str):
    if (
        not _st_force_for_all
        and model_name not in _ST_FALLBACK_MODELS
        and not _is_local_st_model(model_name)
    ):
        return None
    if model_name not in _st_cache:
        _st_cache[model_name] = _STEncoder(model_name, device=_st_device_override)
    return _st_cache[model_name]


def _format_session(turns: list[dict]) -> str:
    """Concatenate one session's turns as a single text blob.

    LongMemEval turns are {role, content}. We tag each turn with its role so
    the chunker can keep user/assistant boundaries semantically visible.
    """
    parts: list[str] = []
    for t in turns:
        role = t.get("role", "user").upper()
        content = t.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _ingest_question(
    store: VstashStore,
    cfg: VstashConfig,
    session_ids: list[str],
    sessions: list[list[dict]],
) -> int:
    """Ingest all haystack sessions for one question in a single batch.

    Each session becomes one vstash document with path=session_id so we can
    map retrieved chunks back to their source session.

    Returns the number of duplicate session_ids skipped (15 questions in
    longmemeval_s ship duplicate ids inside the same haystack; defer_fts
    mode forbids re-ingesting a path within one batch).
    """
    documents: list[dict] = []
    seen: set[str] = set()
    dups = 0
    for sid, turns in zip(session_ids, sessions):
        if sid in seen:
            dups += 1
            continue
        seen.add(sid)
        text = _format_session(turns)
        if not text.strip():
            continue
        chunks = chunk_text(text, cfg.chunking.size, cfg.chunking.overlap)
        if not chunks:
            continue
        embeddings = embed_texts(chunks, cfg.embeddings.model)
        documents.append(
            {
                "path": sid,
                "title": sid,
                "chunks": chunks,
                "embeddings": embeddings,
                "source_type": "chat",
                "collection": "lme",
            }
        )
    if documents:
        with store.batch_mode(defer_fts=True):
            store.add_documents_batch(documents)
    return dups


def _unique_sessions_in_order(results: list) -> list[str]:
    """Walk ranked search results and return unique session_ids in order."""
    seen: set[str] = set()
    out: list[str] = []
    for r in results:
        sid = r.path
        if sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def _recall_at_k(retrieved_sessions: list[str], gold: set[str], k: int) -> float:
    """Session-level Recall@K. K refers to unique session_ids, not chunks."""
    if not gold:
        return float("nan")
    top_k = set(retrieved_sessions[:k])
    return len(top_k & gold) / len(gold)


def _stratified_subset(data: list[dict], per_type: int) -> list[dict]:
    """Take the first `per_type` questions from each question_type.

    longmemeval_s is grouped by type, so this is a deterministic, fast
    sample that touches all 6 categories without random seeds.
    """
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in data:
        by_type[q["question_type"]].append(q)
    out: list[dict] = []
    for t in sorted(by_type.keys()):
        out.extend(by_type[t][:per_type])
    return out


def run(
    n: int | None,
    top_k_chunks: int,
    ks: tuple[int, ...],
    output: Path | None,
    retrieval_mode: str,
    seed_offset: int = 0,
    stratify_per_type: int | None = None,
    model: str | None = None,
    distance_cutoff: float = 1.3225,
    device: str | None = None,
) -> dict:
    global _st_device_override, _st_force_for_all
    cfg = VstashConfig()
    if model is not None:
        cfg = cfg.model_copy(update={"embeddings": cfg.embeddings.model_copy(update={"model": model})})

    # When --device cuda is set, route EVERY model through the
    # SentenceTransformer-CUDA path -- otherwise FastEmbed (no CUDA
    # wheel) stays on CPU and an ingest of LongMemEval's ~150 chunks
    # per question takes ~55s on Colab vs ~1s on Apple Silicon CPU.
    resolver_registered = False
    if device == "cuda":
        _st_device_override = "cuda"
        _st_force_for_all = True
        register_encoder_resolver(_st_resolver)
        resolver_registered = True
    elif cfg.embeddings.model in _ST_FALLBACK_MODELS or _is_local_st_model(cfg.embeddings.model):
        register_encoder_resolver(_st_resolver)
        resolver_registered = True
    try:
        return _run_locked(
            cfg=cfg,
            n=n,
            top_k_chunks=top_k_chunks,
            ks=ks,
            output=output,
            retrieval_mode=retrieval_mode,
            seed_offset=seed_offset,
            stratify_per_type=stratify_per_type,
            distance_cutoff=distance_cutoff,
        )
    finally:
        if resolver_registered:
            unregister_encoder_resolver(_st_resolver)
        _st_force_for_all = False
        _st_device_override = None


def _run_locked(
    *,
    cfg: VstashConfig,
    n: int | None,
    top_k_chunks: int,
    ks: tuple[int, ...],
    output: Path | None,
    retrieval_mode: str,
    seed_offset: int,
    stratify_per_type: int | None,
    distance_cutoff: float,
) -> dict:
    dim = get_embedding_dim(cfg.embeddings.model)

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

        with tempfile.TemporaryDirectory(prefix="lme_") as tmp:
            db_path = str(Path(tmp) / "lme.db")
            store = VstashStore(
                db_path,
                embedding_dim=dim,
                vector_backend=cfg.storage.vector_backend,
                snapvec_bits=cfg.storage.snapvec_bits,
                observability=cfg.observability,
                limits=cfg.limits,
                cache=cfg.cache,
            )
            try:
                t0 = time.time()
                dup_skips = _ingest_question(store, cfg, session_ids, sessions)
                ingest_s = time.time() - t0

                t0 = time.time()
                q_emb = embed_query(question, cfg.embeddings.model)
                results = store.search(
                    query_embedding=q_emb,
                    query_text=question,
                    top_k=top_k_chunks,
                    collection="lme",
                    retrieval_mode=retrieval_mode,
                    distance_cutoff=distance_cutoff,
                )
                search_s = time.time() - t0
            finally:
                store.close()

        retrieved_sessions = _unique_sessions_in_order(results)
        recalls = {f"recall@{k}": _recall_at_k(retrieved_sessions, gold, k) for k in ks}
        gold_rank = next(
            (idx + 1 for idx, sid in enumerate(retrieved_sessions) if sid in gold),
            None,
        )

        row = {
            "question_id": qid,
            "question_type": qtype,
            "n_sessions": len(session_ids),
            "n_dup_session_ids": dup_skips,
            "n_chunks_retrieved": len(results),
            "n_unique_sessions_retrieved": len(retrieved_sessions),
            "gold_session_ids": sorted(gold),
            "first_gold_rank": gold_rank,
            "ingest_s": round(ingest_s, 2),
            "search_s": round(search_s, 3),
            **recalls,
        }
        per_question.append(row)
        by_type[qtype].append(row)

        elapsed = time.time() - overall_start
        print(
            f"[{i + 1}/{len(data)}] {qtype:25s} R@5={recalls['recall@5']:.2f} "
            f"R@10={recalls['recall@10']:.2f} ingest={ingest_s:.1f}s "
            f"first_gold={gold_rank} elapsed={elapsed:.0f}s"
        )

    def _agg(rows: list[dict], k: int) -> float:
        vals = [r[f"recall@{k}"] for r in rows if r[f"recall@{k}"] == r[f"recall@{k}"]]
        return statistics.mean(vals) if vals else float("nan")

    summary = {
        "n_questions": len(per_question),
        "top_k_chunks": top_k_chunks,
        "retrieval_mode": retrieval_mode,
        "embedding_model": cfg.embeddings.model,
        "chunking": {"size": cfg.chunking.size, "overlap": cfg.chunking.overlap},
        "macro": {f"recall@{k}": _agg(per_question, k) for k in ks},
        "by_question_type": {
            t: {"n": len(rows), **{f"recall@{k}": _agg(rows, k) for k in ks}}
            for t, rows in by_type.items()
        },
        "total_elapsed_s": round(time.time() - overall_start, 1),
    }

    print()
    print("=" * 72)
    print(
        f"Macro over {summary['n_questions']} questions  "
        f"(model={summary['embedding_model']}, mode={retrieval_mode}):"
    )
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
        output.write_text(
            json.dumps({"summary": summary, "per_question": per_question}, indent=2)
        )
        print(f"\nWrote {output}")

    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10, help="Number of questions to run (None = all 500)")
    p.add_argument("--all", action="store_true", help="Run all 500 questions (overrides --n)")
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index into the 500-question list (for sharded runs / debugging)",
    )
    p.add_argument(
        "--top-k-chunks",
        type=int,
        default=200,
        help="Chunks to retrieve before deduping to unique sessions",
    )
    p.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "vec_only", "fts_only"],
        default="hybrid",
    )
    p.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help="Session-level K values to report Recall at",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--stratify",
        type=int,
        default=None,
        help="Take K questions from each of the 6 question_types (overrides --n / --offset)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the embedding model (e.g. Stffens/bge-small-rrf-v3). "
        "Default: whatever vstash.toml resolves to.",
    )
    p.add_argument(
        "--distance-cutoff",
        type=float,
        default=1.3225,
        help="distance_cutoff multiplier passed to VstashStore.search "
        "(default 1.3225 matches v0.34+ cosine semantics). Pass a large "
        "value (e.g. 100.0) to effectively disable the cutoff.",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Pin the embedder to a device. 'cuda' on Colab T4 is ~50x "
        "faster than the FastEmbed CPU default (FastEmbed has no CUDA "
        "wheel).  Defaults to FastEmbed's auto on the local Mac.",
    )
    args = p.parse_args()

    n = None if args.all else args.n
    run(
        n=n,
        top_k_chunks=args.top_k_chunks,
        ks=tuple(args.ks),
        output=args.output,
        retrieval_mode=args.retrieval_mode,
        seed_offset=args.offset,
        stratify_per_type=args.stratify,
        model=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
